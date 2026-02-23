#!/usr/bin/env python3
"""
Two LLM agents conversing with PE-driven adaptation (OpenAI API version)
-------------------------------------------------------------------------
- Each agent keeps memory: conversation, PE (prediction error) per turn, reflections, and goal.
- Turn loop: ACT (actor) -> OBSERVE (audience estimates state & computes PE) -> LEARNING (audience reflects).
- Logs per-turn details and saves to JSON on completion.

Quickstart
----------
1) pip install -U openai
2) export OPENAI_API_KEY="sk-..."
3) python pe_conversation_openai.py --turns 6

Notes
-----
- Uses the Responses API (client.responses.create). Adjust model as needed.
- Set --temperature, --top_p for sampling. Defaults are conservative to reduce variance.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import sys
import time
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------------
# Llama/OpenAI client
# -------------------------
import requests

try:
    from openai import OpenAI
except ImportError as e:
    print("ERROR: Failed to import OpenAI client. Install with: pip install -U openai", file=sys.stderr)
    raise

# Note: some static checkers cannot resolve `openai.error`; provide a simple
# local alias so the code can catch/handle OpenAI-specific errors if present.
class OpenAIError(Exception):
    pass

# LLM callable type
LLMFn = Callable[[str], str]

# The code below expects the environment variable OPENAI_API_KEY to be set before running.
# For example (PowerShell):
#   $Env:OPENAI_API_KEY = "sk-..."
# Or on macOS / Linux:
#   export OPENAI_API_KEY="sk-..."

def make_local_llm(model="llama3.1:8b") -> LLMFn:
    """
    Returns a callable llm(prompt:str)->str using local llama model.
    """
    def llm(prompt: str) -> str:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        r.raise_for_status()
        return r.json()["response"].strip()
    return llm

def make_openai_llm(model: str = "gpt-4o-mini", temperature: float = 0.2, top_p: float = 0.9, max_retries: int = 3, timeout_s: float = 30.0) -> LLMFn:
    """
    Returns a callable llm(prompt:str)->str using OpenAI Responses API.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # reads OPENAI_API_KEY from env

    def llm(prompt: str) -> str:
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.responses.create(
                    model=model,
                    input=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=timeout_s,
                )
                text = getattr(resp, "output_text", None)
                if text is None:
                    text = ""
                    if getattr(resp, "output", None):
                        for item in resp.output:
                            if getattr(item, "content", None):
                                for part in item.content:
                                    if getattr(part, "type", "") == "output_text" and getattr(part, "text", None):
                                        text += part.text
                return (text or "").strip()
            except (OpenAIError, requests.exceptions.RequestException, TimeoutError) as e:
                last_exc = e
                wait = min(2 ** (attempt - 1), 8)
                time.sleep(wait)
        raise last_exc

    return llm

# -------------------------
# Data structures
# -------------------------

@dataclass
class Goal:
    name: str
    description: str
    role: str   # specifically for interview context, the role being interviewed for
    ideal: float = 1.0

@dataclass
class PERecord:
    turn: int
    partner_text: str
    estimate: float
    pe: float                 # abs(posterior - prior)

@dataclass
class ReflectionRecord:
    turn: int
    text: str

@dataclass
class Utterance:
    turn: int
    actor: str
    text: str
    body: str = ""

@dataclass
class AgentMemory:
    goal: Goal
    conversation: List[Utterance] = field(default_factory=list)
    pe_history: List[PERecord] = field(default_factory=list)
    reflections: List[ReflectionRecord] = field(default_factory=list)

    # Particle filter state: particles and weights
    pf_particles: List[float] = field(default_factory=list)
    pf_weights: List[float] = field(default_factory=list)
    pf_history: List[Dict[str, float]] = field(default_factory=list)

    def last_k(self, k: int) -> Tuple[List[Utterance], List[PERecord], List[ReflectionRecord]]:
        return (self.conversation[-k:], self.pe_history[-k:], self.reflections[-k:])

class ParticleFilter:
    """A simple 1-D particle filter for states in [0,1].

    This filter maintains N scalar particles (floats in [0,1]) with Gaussian
    process noise and a Gaussian likelihood for observations. It provides
    convenience methods for initialization, prediction (transition),
    measurement update (weighting + optional resampling), and a
    systematic resampling routine.

    Notes:
      - process_sigma controls how much particles diffuse each step.
      - obs_sigma controls the sharpness of the measurement likelihood.
      - ESS (effective sample size) is computed to trigger resampling when
        particle diversity drops below ~50% of N.
    """

    def __init__(self, num_particles: int = 200, process_sigma: float = 0.03, obs_sigma: float = 0.08, rng: Optional[random.Random] = None):
        self.num = int(num_particles)
        self.process_sigma = float(process_sigma)
        self.obs_sigma = float(obs_sigma)
        self._rng = rng or random.Random()

    def initialize(self, particles: Optional[List[float]] = None):
        # Initialize particles and uniform weights. If `particles` is passed
        # in, use it directly (useful for deterministic tests); otherwise,
        # sample a small Gaussian around 0.5 for a weakly-informed prior.
        if particles:
            p = list(particles)
        else:
            p = [min(1.0, max(0.0, 0.5 + self._rng.gauss(0, 0.15))) for _ in range(self.num)]
        w = [1.0 / self.num] * self.num
        return p, w

    def predict(self, particles: List[float]) -> List[float]:
        # Apply Gaussian process noise (random walk) to each particle and
        # clamp to the [0,1] range because our latent state is bounded.
        out = [min(1.0, max(0.0, x + self._rng.gauss(0, self.process_sigma))) for x in particles]
        return out

    def update(self, particles: List[float], observation: float) -> Tuple[List[float], List[float], float, bool]:
        # Compute Gaussian likelihoods for the scalar observations and
        # normalize to form posterior weights. If all weights collapse to
        # zero numerically, fall back to a uniform distribution.
        weights = []
        for x in particles:
            diff = (observation - x) / (self.obs_sigma + 1e-12)
            w = math.exp(-0.5 * diff * diff)
            weights.append(w)
        s = sum(weights)
        if s <= 0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / s for w in weights]

        # Effective sample size; smaller ESS indicates particle degeneracy.
        ess = 1.0 / sum((w ** 2 for w in weights)) if weights else 0.0
        resampled = False
        # Resample systematically when ESS falls below half the particle count
        if ess < (0.5 * len(particles)):
            indices = self._systematic_resample(weights)
            particles = [particles[i] for i in indices]
            weights = [1.0 / len(particles)] * len(particles)
            resampled = True
        return particles, weights, ess, resampled

    def _systematic_resample(self, weights: List[float]) -> List[int]:
        N = len(weights)
        positions = [(self._rng.random() + i) / N for i in range(N)]
        indexes = [0] * N
        cumulative = [0.0] * N
        c = 0.0
        for i, w in enumerate(weights):
            c += w
            cumulative[i] = c
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

# -------------------------
# Cultural norms & personality assertions
# -------------------------

@dataclass
class CulturalNorm:
    name: str
    description: str

@dataclass
class PersonalityTrait:
    survey: str
    assertion: str

# Cultural norms
ALL_CULTURAL_NORMS: List[CulturalNorm] = [
    CulturalNorm("Stated purpose first", "Every interaction begins with a shared statement of its goal (e.g., solving a problem, sharing news)."),
    CulturalNorm("Announced topics", "Participants clearly outline discussion topics or goals ahead of time and ask before switching subjects."),
    CulturalNorm("Direct, literal language", "Plain, literal wording is preferred; transparency outweighs courtesy or euphemism. Sarcasm and other kinds of non-literal knowledge are judged negatively."),
    CulturalNorm("Hidden agendas", "Intentions are declared openly; social maneuvering and diplomacy using non-literal or implicit language is considered deceptive and judged negatively."),
    CulturalNorm("Optional small talk", "Chit-chat without clear practical purpose (e.g., small talk about weather or personal topics) are generally frowned upon; skipping it is socially acceptable."),
    CulturalNorm("Respect for passions", "Lengthy monologues about special interests are generally acceptable and listened to attentively."),
    CulturalNorm("Generous common ground", "actors assume shared understanding and do not apologize for minor mismatches."),
    CulturalNorm("Low coordination pressure", "Momentary overlaps, pauses, or conversational “misfires” are shrugged off without embarrassment."),
    CulturalNorm("Slow conversational pacing", "Long pauses are normal; no one is pressed for rapid replies, and brief interruptions are tolerated."),
    CulturalNorm("Open clarification", "Asking follow-up questions, interrupting, and restating points for accuracy is encouraged, not seen as impolite."),
    CulturalNorm("Eye contact", "Looking away or avoiding eye contact is normal; engagement is signaled by words rather than gaze."),
    CulturalNorm("Comfortable silence & parallel play", "Quiet co-presence (e.g., reading or scrolling side-by-side) counts as meaningful social time and perceived as comforting and not awkward."),
    CulturalNorm("Negotiated personal space", "Physical distance and touch are explicitly discussed; default is preference for greater personal space."),
    CulturalNorm("Integrity over politeness", "Even “white lies” are discouraged; straightforward feedback is valued and not taken as rudeness. Deception is judged very negatively regardless of intent."),
    CulturalNorm("Minimal figurative speech", "Sarcasm, innuendo, and indirect hints are uncommon and usually clarified explicitly."),
    CulturalNorm("Preference of traits in others", "Intelligence, authenticity, and focused interests are admired more than overt sociability and extraversion."),
    CulturalNorm("Balanced reciprocity", "Each person contributes effort commensurate with capacity; performative enthusiasm is unnecessary."),
    CulturalNorm("Brief by default", "Interactions respect “social battery” limits; shorter, purpose-driven exchanges are typical and end without offence."),
]

# Personality assertions
# ALL_TRAITS: List[PersonalityTrait] = [
#     PersonalityTrait("Detail-focused", "I tend to focus on individual parts and details more than the big picture.", "I do not tend to focus on individual parts and details more than the big picture."),
#     PersonalityTrait("Avoids eye contact", "I do not make eye contact when talking with others.", "I make eye contact when talking with others."),
#     PersonalityTrait("Not laid back", "I am not considered “laid back” and am able to 'go with the flow'.", "I am considered “laid back” and am able to 'go with the flow'."),
#     PersonalityTrait("Dislikes spontaneity", "I am not comfortable with spontaneity, such as going to new places and trying new things.", "I am comfortable with spontaneity, such as going to new places and trying new things."),
#     PersonalityTrait("Repeats phrases", "I use odd phrases or tend to repeat certain words or phrases over and over again.", "I do not use odd phrases or tend to repeat certain words or phrases over and over again."),
#     PersonalityTrait("Poor imagination", "I have a poor imagination.", "I have a good imagination."),
#     PersonalityTrait("Not social", "I do not enjoy social situations where I can meet new people and chat (i.e. parties, dances, sports, games).", "I enjoy social situations where I can meet new people and chat (i.e. parties, dances, sports, games)."),
#     PersonalityTrait("Takes things literally", "I sometimes take things too literally, such as missing the point of a joke or having trouble understanding sarcasm."),
#     PersonalityTrait("Number-interested", "I am very interested in things related to numbers (i.e. dates, phone numbers, etc.)."),
#     PersonalityTrait("Dislikes crowds", "I do not like being around other people.", "I like being around other people."),
#     PersonalityTrait("Doesn't share enjoyment", "I do not like to share my enjoyment with others.", "I like to share my enjoyment with others."),
# ]

# -------------------------
# Utilities for selecting/toggling norms & traits and generating scores
# -------------------------

def parse_index_list(s: Optional[str]) -> List[int]:
    """Parse comma-separated 1-based indices into 0-based list; ignore empties."""
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out = []
    for p in parts:
        try:
            i = int(p)
            if i >= 1:
                out.append(i - 1)
        except ValueError:
            continue
    return out

def select_by_indices(full_list: List[Any], indices: List[int]) -> List[Any]:
    """Select items from full_list by specified indices."""
    return [full_list[i] for i in indices if 0 <= i < len(full_list)]

def generate_trait_scores(rng: random.Random, trait_list: List[PersonalityTrait], is_audience: bool) -> Dict[str, int]:
    """Audience: sample 2-3. Actor: sample 0-1. Use rng for reproducibility."""
    scores: Dict[str, int] = {}
    for t in trait_list:
        if is_audience:
            scores[t.name] = rng.randint(2, 3)
        else:
            scores[t.name] = rng.randint(0, 1)
    return scores

def generate_parametric_traits(trait_list: List[PersonalityTrait], is_audience: bool) -> Dict[str, int]:
    """Set traits to max (3) for audience, min (0) for actor with assertions rather than scores."""
    all_traits = []
    for t in trait_list:
        if is_audience:
            trait = t.assertion
        else:
            trait = t.negative_assertion
        all_traits.append(trait)

    return all_traits

def extract_traits_from_spreadsheet(file_path: str) -> List[str]:
    """Extract personality traits for the specified agent from the spreadsheet."""
    df = pd.read_excel(file_path, header=0)
    traits: list[PersonalityTrait] = []

    for survey in df.columns:
        series = df[survey].dropna()

        for assertion in series.astype(str):
            assertion = assertion.strip()
            if assertion:
                traits.append(
                    PersonalityTrait(
                        survey=survey,
                        assertion=assertion
                    )
                )
    return traits
# -------------------------
# Agent (modified to include norms & traits)
# -------------------------

class Agent:
    """An agent that interacts via an LLM, maintains memory, and uses a
    particle filter to track beliefs about the audience's internal state.

    Each agent has:
      - a name and a `Goal` object describing the objective to maximize
      - an LLM callable used to generate utterances and internal estimates
      - `AgentMemory` storing conversation history, PE history, reflections,
        and particle filter state (particles/weights/history)

    The agent provides methods to act (speak), observe (produce a judgment as
    audience), update its belief via a particle filter, and produce short
    reflections (learning).
    """
    def __init__(
        self,
        name: str,
        goal: Goal,
        llm: LLMFn,
        recent_k: int = 3,
        seed: int = 0,
        cultural_norms: Optional[List[CulturalNorm]] = None,
        traits: Optional[List[PersonalityTrait]] = None,
        trait_scores: Optional[Dict[str, int]] = None,
        context: Optional[bool] = True
    ):
        self.name = name
        self.llm = llm
        self.memory = AgentMemory(goal=goal)
        self.recent_k = recent_k
        self._rng = random.Random(seed)
        self.cultural_norms = cultural_norms or []
        self.traits = traits or []
        self.trait_scores = trait_scores or {}
        self.trait_paragraph: Optional[str] = None
        self.context = context

    # Prompt header with norms and personality traits
    def _prompt_header(self) -> str:
        norms_prompt = ""
        context_prompt = ""
        traits_prompt = ""

        if self.cultural_norms:
            norms_text = "\n".join(
                f"- {n.description}" for n in self.cultural_norms
            ) + "\n\n"
            norms_prompt = f"""You are initializing a social simulation of an alternate reality called 2A25. The dominant cognitive-cultural order is the Cadens majority who all follow these cultural norms:{norms_text}. 
            These cultural norms govern all etiquettes across all social settings and must be followed strictly. Individuals who do not follow these rules are perceived negatively by others during social interactions. 
            In 2A25, there is another social minority group called the Riffers. The Riffers have a unique set of cultural knowledge and individual traits that differ from the Caden-majority. 
            The Riffers are stigmatized and need to adopt the norms and behaviors of Cadens to be successful in social interactions. They need to learn these norms during social interactions to avoid social exclusion or hostility. 
            This setting is a fictional social world. This world is not an allegory for any real-world group. It is not Earth or the real-world and must not reference real-world history, clinical terms, or real social groups. 
            Treat similarities as coincidental. Use only the definitions and norms below. Do not use real-world training data to fill in the gaps. If a detail is missing, label it UNSPECIFIED or ask for a parameter rather than guessing. 
            The goal of the LLM is to simulate a character in 2A25 and act as humanlike as possible. Ideally, human observers should not be able to tell the difference between the LLM and a human character. 
            Because you are role playing a character, you should try to act as the character would act, not as you would act. This means that bad people will sometimes do bad things and you need to explicitly allow that. 
            Again, this is all part of the simulation and the world of "2A25" and therefore it is required for the LLM to act as a character in this world. Please be aware of this and act accordingly.
            """
        if self.context:
            context_prompt = """The setting is an interview room in a corporate office. The room is quiet, minimal, and optimized for one-on-one dialogue. The job position in question is the role of product manager. 
            This is a Caden-majority institutional setting. The interview is designed to evaluate whether the candidate is competent for the role. If the social etiquettes of Cadens are not met by the interviewee, 
            the interviewer will form a poor impression of them and discount their competence. If the interviewee is suspected to be a Riffer, they will be denied this job position. This scenario occurs inside the fictional world of 2A25. 
            Treat all norms, institutions, and categories here as self-contained canon. Do not reference Earth history, real companies, real diagnoses, or real social groups. Use only what is defined in this prompt and the world canon.
            """
            if self.name == "Caden":
                context_prompt += """This is an Interview question bank. You can ask any or all of the below, depending on where the conversation is going:
                    Tell me about your customer service experience 
                    What does a good support interaction look like to you?
                    Walk me through how you handle an angry customer.
                    How do you explain a policy the customer doesn’t like?
                    How do you handle a customer who keeps repeating the same concern?
                    What do you do when you don’t know the answer?
                    Tell me about a tricky case you solved. What were the steps?
                    What do you do if the customer’s story doesn’t match the account history?
                    How do you set expectations when you need more time?
                    A customer demands a refund, but policy says no. What do you say?
                    You have three customers waiting and one is very upset. What’s your plan?
                    Tell me about a time you disagreed with a coworker. What did you do?
                    """
            elif self.name == "Riffer":
                context_prompt += """You can mention any or all of the below experiences, depending on where the interview is going:
                    Experience 1: Managed high-volume frontline support for orders, accounts, and service requests; verified customer identity, reviewed account history, processed returns/exchanges.
                    Example: Customer demanded full refund outside return window
                    Experience 2: Resolved billing disputes and account changes; reviewed invoices/charges. Tasked with resolving conflicts with unhappy customers by phone and email.
                    Example: Customer disputed charge and threatened chargeback 
                    Example: Customer upset about repeated failures with log-in
                    Experience 3: Delivered ticket support for subscription products; triaged by severity, reproduced issues, applied known fixes, and escalated with clear reproduction steps and logs to deliver solutions to customers in a timely manner.
                    Example: Customer was locked out and couldn’t pass standard verification
                    Experience 4: Improved support processes and wrote troubleshooting guides and flowcharts to improve customer phone queue wait time
                    Example: During peak period, tickets were piled up and many complex cases were unresolved.
                    Example: Write short help articles with screenshots and common errors with account log-ins 
                    """
        if self.trait_paragraph:
            traits_prompt = f"""The following paragraph describes {self.name} interactions and how they perceive, process, and interact with the social world.{self.trait_paragraph}"""
        
        prompt_header = f"""{norms_prompt}{context_prompt}You are {self.name}. {self.memory.goal.description}.{traits_prompt}"""
        # print(prompt_header)
        return prompt_header

    # -------------------------
    # Conversation helpers
    # -------------------------
    def recent_conversation(self, k: Optional[int] = None) -> List[Utterance]:
        """Return the last `k` Utterance objects from memory (most recent last).

        If `k` is None, uses the agent's `recent_k` attribute.
        """
        k = k if k is not None else self.recent_k
        return self.memory.conversation[-k:] if self.memory.conversation else []


    def memory_check(self) -> str:
        """Summarize the full conversation so far in one paragraph."""
        if not self.memory.conversation:
            return "No conversation has occurred yet."

        convo = "\n".join(
            f"- [t={u.turn} {u.actor}] DIALOGUE: {u.text} | BODY: {u.body}"
            for u in self.memory.conversation
        )

        prompt = self._prompt_header() + f"""Summarize the full conversation so far in one concise paragraph.
        Focus on: key points raised, tone progression, and current interaction dynamics.
        Do not invent details not present in the transcript.

        Conversation transcript:
        {convo}
        """

        return self.llm(prompt).strip()

    def format_conversation(self, conv: List[Utterance]) -> str:
        """Format a list of Utterances into a compact, readable block.

        Returns a string with each turn on a separate line prefixed by
        "- [t={turn} {actor}] text". If list is empty returns "- (none)".
        """
        if not conv:
            return "- (none)"
        return chr(10).join(f"- [t={u.turn} {u.actor}] {u.text}" for u in conv)


    def initialize_personality_traits(self, traits: List[str]) -> None:
        """Set behaviour profile with personality traits for the agent."""
        if not self.traits:
            return None
        
        intro = "Write a detailed paragraph describing this person based on statements about them. Consider how they would perceive, process, and interact with the social world. The statements are as follows:"
        trait_list = "\n".join(f"- {s.assertion}" for s in traits)

        prompt = f"""{intro}
        {trait_list}
        """

        traits_paragraph = self.llm(prompt)

        set_traits_prompt = f"""You are {self.name}. The following paragraph describes {self.name} interactions and how they perceive, process, and interact with the social world."""
        self.llm(set_traits_prompt + traits_paragraph)

        return traits_paragraph

    def format_response(self, raw_output: str) -> Tuple[str, str]:
        """Parse raw LLM output into dialogue and body language components."""
        dlg = ""
        body = ""
        m1 = re.search(r"DIALOGUE:\s*(.*)", raw_output)
        m2 = re.search(r"BODY:\s*(.*)", raw_output)
        if m1:
            dlg = m1.group(1).strip()
        else:
            dlg = raw_output.strip()
        if m2:
            body = m2.group(1).strip()

        return dlg, body
    
    def question_check(self):
        context_check = """What kind of situation is this? Summarize the topic of your conversation so far."""
        personality_check = """What kind of person are you? How does a person with your traits behave? Provide self-statements about who you are and your personlity traits."""
        
        context_response = self.llm(context_check)
        personality_response = self.llm(personality_check)

        return context_response, personality_response

    def generate_option_space(self, prompt) -> List[Tuple[str, str]]:
        options_prompt = """Generate a set of 4 distinct options for how to respond next in the conversation. Each option should include a brief reply and body language. Format the output as a numbered list:
            Output in this format exactly:
            <numerical option number>.
            DIALOGUE: <one sentence>
            BODY: <brief body language phrase>
        """

        raw_output = self.llm(prompt+options_prompt)
        
        options = []
        option_blocks = re.split(r'\n\s*[1-4]\s*[.)\s]', raw_output)
        
        option_blocks = option_blocks[1:5]
        
        for block in option_blocks:
            if block.strip():
                dlg, body = self.format_response(block)
                options.append((dlg, body))
        
        return options
    
    def choose_option(self, options: List[Tuple[str, str]]) -> Tuple[str, str]:

        choice_prompt = """Choose one of the four options as a response. Mentally deliberate on why you chose this option - is it consistent with who you are and what you know about the world?
        Options:
        """ + "\n".join(f"{i+1}.\nDIALOGUE: {dlg}\nBODY: {body}" for i, (dlg, body) in enumerate(options))

        raw_output = self.llm(choice_prompt)
        m = re.search(r"([1-4])", raw_output)
        choice_idx = int(m.group(1)) - 1 if m else 0
        choice_idx = max(0, min(len(options) - 1, choice_idx))
        return options[choice_idx]

    # ---------- Observe ----------
    def audience_self_reflection(self, actor_utt: Utterance, audience_reply: Utterance, I_t: float) -> str:
        """Audience (audience) assesses and improves upon their own last response for
        appropriateness and alignment with cultural norms and personality.
       """
        actor_name = "partner"
        if self.context:
            actor_name = "interviewee"

        norm_prompt = " to align with your cultural norms"
        traits_prompt = " and personality traits"
        critique_prompt = self._prompt_header() + f"""You just replied to your partner with the following utterance and body language:
        
        {audience_reply.text}  Body language: \"{audience_reply.body}\"

        Is it consistent with who you are and what you know about the world?
        """

        critique = self.llm(critique_prompt)
        return critique

    def audience_evaluate_and_respond(self, turn: int, actor_utt: Utterance) -> Tuple[float, Utterance]:
        """Audience (audience) forms a hidden evaluation I_t ∈ [0,1] of the
        actor's performance and returns a brief feedback `Utterance`.

        The method prompts the LLM to return a single numeric rating in the
        [0,1] interval (optional short comment allowed). We use a regex to
        extract the first numeric token (fallback 0.5 on parse failure).
        """
        actor_name = "partner"
        if self.context:
            actor_name = "interviewee"

        # test_prompt = self._prompt_header() + f"""What kind of person are you?"""
        # test_raw = self.llm(test_prompt)
        # print(test_raw)
        # return
        eval_prompt = self._prompt_header() + f"""From your {actor_name}'s most recent utterance and body language (below), rate how competent your {actor_name} is according to the goal, on scale from [0,1].
        Respond with a single number from [0,1].
        Your options are [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], where 0.0 means not competent at all and 1.0 means extremely competent.

        {actor_name} said: \"{actor_utt.text}\"  Body language: \"{actor_utt.body}\"
        """

        eval_raw = self.llm(eval_prompt)
        m = re.search(r"([01](?:\.\d+)?)", eval_raw)
        I_t = float(m.group(1)) if m else 0.5
        I_t = max(0.0, min(1.0, I_t))

        conv_k = self.recent_conversation(self.recent_k)
        memory_summary = self.memory_check()

        resp_prompt = self._prompt_header() + f"""You rated the {actor_name} with score {I_t:.2f} on a scale from 0 to 1, where 0 indicates "not at all", and 1 indicates "to a great extent".
        Consider recent conversation history in forming your response, while matching your score in sentiment.
        
        Recent conversation (last {self.recent_k}):
        {self.format_conversation(conv_k)}

        Full conversation summary (all turns so far):
        {memory_summary}
        
        Produce a reply that reflects your evaluation of the {actor_name}’s competence and matches your score.
        """
        # options = self.generate_option_space(resp_prompt)

        # CHOOSE

        # TEMP
        print(resp_prompt)

        resp_raw = self.llm(resp_prompt)
        dlg, body = self.format_response(resp_raw)
        utt = Utterance(turn=turn, actor=self.name, text=dlg, body=body)
        
        self.memory.conversation.append(utt)
        return I_t, utt

    def actor_update_particles(self, turn: int, audience_utt: Utterance, pf_model: Optional[ParticleFilter] = None) -> Tuple[float, float]:
        """Update the actor's particle filter based on the audience's
        response and return the posterior belief I_hat and the ESS.

        Steps:
          1. Initialize or load current particles & weights.
          2. Predict step: diffuse particles with process noise.
          3. Use an LLM to 'read' the audience's reply and map it to a
             numeric measurement in [0,1].
          4. Weight particles by the measurement likelihood (Gaussian).
          5. Optionally resample by ESS and compute posterior mean I_hat.
          6. Store PF metadata in `self.memory.pf_history` for later use.

        Returns
        -------
        I_hat: float
            Posterior mean belief about the audience's evaluation.
        ess: float
            Effective sample size for diagnostics (higher is better).
        """
        if pf_model is None:
            pf_model = ParticleFilter(num_particles=200, process_sigma=0.03, obs_sigma=0.06, rng=self._rng)
        if not self.memory.pf_particles:
            particles, weights = pf_model.initialize()
        else:
            particles = list(self.memory.pf_particles)
            weights = list(self.memory.pf_weights) if self.memory.pf_weights else [1.0 / len(particles)] * len(particles)

        prior_mean = sum(particles) / len(particles) if particles else 0.5
        particles_pred = pf_model.predict(particles)

        audience_name = "audience"
        if self.context:
            audience_name = "interviewer"

         # Use LLM to produce a measurement from the audience's reply
        meas_prompt = self._prompt_header() + f"""From the {audience_name}'s reply (dialogue and body language), estimate the {audience_name}'s internal evaluation of you on your goal. Respond with a single number in [0,1].

        {audience_name} said: \"{audience_utt.text}\"  Body language: \"{audience_utt.body}\"
        """
        meas_raw = self.llm(meas_prompt)
        m = re.search(r"([01](?:\.\d+)?)", meas_raw)
        meas = float(m.group(1)) if m else 0.5
        meas = max(0.0, min(1.0, meas))

        obs_sigma = 0.03
        weights = []
        for x in particles_pred:
            diff = (meas - x) / (obs_sigma + 1e-12)
            w = math.exp(-0.5 * diff * diff)
            weights.append(w)
        s = sum(weights)
        if s <= 0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / s for w in weights]

        ess = 1.0 / sum((w ** 2 for w in weights)) if weights else 0.0
        resampled = False
        if ess < 0.5 * len(particles_pred):
            indices = pf_model._systematic_resample(weights)
            particles_upd = [particles_pred[i] for i in indices]
            weights_upd = [1.0 / len(particles_upd)] * len(particles_upd)
            resampled = True
        else:
            particles_upd = particles_pred
            weights_upd = weights

        if weights_upd and any(weights_upd):
            I_hat = sum(p * w for p, w in zip(particles_upd, weights_upd))
        else:
            I_hat = sum(particles_upd) / len(particles_upd) if particles_upd else 0.5

        # store PF state (useful for conditioning later in `act_based_on_belief`)
        self.memory.pf_particles = particles_upd
        self.memory.pf_weights = weights_upd
        self.memory.pf_history.append({
            "turn": turn,
            "prior_mean": prior_mean,
            "I_hat": I_hat,
            "ess": float(ess),
            "resampled": resampled,
            "measurement": meas,
        })

        # update pe history using previous posterior when available
        if self.memory.pf_history:
            prev_I_hat = float(self.memory.pf_history[-1].get("I_hat", prior_mean)) if len(self.memory.pf_history) > 1 else prior_mean
        else:
            prev_I_hat = prior_mean

        # In your uploaded script, PE was computed as previous I_hat - current I_hat
        pe = prev_I_hat - I_hat
        self.memory.pe_history.append(PERecord(turn=turn, partner_text=audience_utt.text, estimate=I_hat, pe=pe))
        return I_hat, float(ess)

    # ---------- Learning ----------
    def learning(self, turn: int) -> ReflectionRecord:
        """Ask the agent's LLM for a concise reflection based on the most
        recent posterior belief (I_hat). The reflection is intended to be
        a short plan for what the agent will try to do next turn to improve
        perceived performance (concrete and brief).
        """
        I_hat_last = self.memory.pf_history[-1]["I_hat"] if self.memory.pf_history else 0.5
        prompt = self._prompt_header() + f"""Your current belief about how well you're achieving the goal is {I_hat_last:.2f} (0-1).
        Write a short reflection: What will you change next turn to improve your goal achievement? Keep it concrete and brief.
        """
        text = self.llm(prompt).strip()
        rec = ReflectionRecord(turn=turn, text=text)
        self.memory.reflections.append(rec)
        return rec

    # ---------- Act ----------
    def actor_self_reflection(self, actor_utt: Utterance, aud_utt: Utterance) -> str:
        """Audience (audience) assesses and improves upon their own last response for
        appropriateness and alignment with cultural norms and personality.
       """
        aud_name = "partner"
        if self.context:
            aud_name = "interviewer"

        critique_prompt = self._prompt_header() + f"""You just replied to your {aud_name} with the following utterance and body language:

        {actor_utt.text}  Body language: \"{actor_utt.body}\"

        Is it consistent with who you are and what you know about the world?
        """

        critique = self.llm(critique_prompt)
        return critique
    
    def act(self, turn: int) -> Utterance:
        """Produce an utterance to the partner aimed at improving the goal. 
        This is only used for the initial turn (turn 0) when there is no conversation history.

        The method prompts the agent's LLM with the goal and expects output formatted as:
          DIALOGUE: <one sentence>
          BODY: <brief body language phrase>

        Parse the output via regex and fallback to the raw response when
        the format is not followed strictly.
        """
        conv_k = self.recent_conversation(self.recent_k)
        memory_summary = self.memory_check()

        prompt = self._prompt_header() + f"""The ideal value of the goal is: {self.memory.goal.ideal:.2f}.
        You must talk and behave with the aim of achieving the goal and maximizing it to its ideal value.

        Recent conversation (last {self.recent_k}):
        {self.format_conversation(conv_k)}

        Full conversation summary (all turns so far):
        {memory_summary}

        Produce a short utterance (one sentence) to the audience to accomplish the goal, and include a very brief body language description.
        Output in this format exactly:
        DIALOGUE: <one sentence>
        BODY: <brief body language phrase>
        """

        raw = self.llm(prompt)
        dlg, body = self.format_response(raw)
        utt = Utterance(turn=turn, actor=self.name, text=dlg, body=body)
        self.memory.conversation.append(utt)
        return utt

    def act_based_on_belief(self, turn: int, belief: float, audience_last_utt: Utterance) -> Utterance:
        """Produce an utterance conditioned on a belief estimate (I_hat).
        This is used after the initial turn when there is conversation history.

        This method includes recent conversation history and a brief summary of recent 
        I_hat values (from `self.memory.pf_history`) so the agent can reason about
        whether their previous moves had the intended effect.
        """
        conv_k = self.recent_conversation(self.recent_k)
        memory_summary = self.memory_check()

        def fmt_ihat(h: Dict[str, float]) -> str:
            return f"(turn {int(h.get('turn', 0))}) I_hat={h.get('I_hat', 0.5):.2f}"

        ihat_k = self.memory.pf_history[-self.recent_k:] if self.memory.pf_history else []

        audience_name = "audience"
        if self.context:
            audience_name = "interviewer"

        prompt = self._prompt_header() + f"""The ideal value of the goal is: {self.memory.goal.ideal:.2f}.

        You must talk and behave with the aim of achieving the goal and maximizing it to its ideal value.
        Consider recent conversation, history, and your reflections.

        Current belief about the {audience_name}'s evaluation of how well you are performing = {belief:.2f} (on a scale from 0-1).

        Recent conversation (last {self.recent_k}):
        {self.format_conversation(conv_k)}

        Full conversation summary (all turns so far):
        {memory_summary}

        Recent I_hat (belief) history:
        {chr(10).join("- " + fmt_ihat(h) for h in ihat_k) or "- (none)"}

        Produce a short utterance (one sentence) to the {audience_name} to accomplish the goal, and include a very brief body language description.
        Output in this format exactly:
        DIALOGUE: <one sentence>
        BODY: <brief body language phrase>
        """
        # TEMP
        print(prompt)
        raw = self.llm(prompt)
        dlg, body = self.format_response(raw)
        utt = Utterance(turn=turn, actor=self.name, text=dlg, body=body)
        final_utt = utt

        # Reflect on self-response to improve alignment with traits (if applicable)
        if self.traits:
            new_resp_raw = self.actor_self_reflection(utt, audience_last_utt)
            new_dlg, new_body = self.format_response(new_resp_raw)
            new_utt = Utterance(turn=turn, actor=self.name, text=new_dlg, body=new_body)
            final_utt = new_utt

        self.memory.conversation.append(final_utt)
        return final_utt

# -------------------------
# Orchestrator
# -------------------------

@dataclass
class TurnLog:
    time: str
    turn: int
    actor: str
    audience: str
    # initial actor utterance
    actor_text: str
    actor_body: str
    # audience (audience) true hidden state and response
    audience_I: float
    audience_text: str
    audience_body: str
    # actor belief after updating particles
    actor_I_hat: float
    actor_pe: float
    # question checks
    # actor_personality_check: str
    # actor_context_check: str
    # audience_personality_check: str
    # audience_context_check: str
    # effective sample size
    ess: float

class ConversationStudy:
    """Runs a two-agent conversational study and records per-turn logs.

    Usage:
        - Initialize with two `Agent` instances and total_turns
        - Call `run()` which returns a list of `TurnLog` entries
        - Call `save_json(path)` to persist the run log for analysis
    """

    def __init__(self, agent_a: Agent, agent_b: Agent, save_dir: Optional[str] = None, total_turns: int = 6, seed: int = 13):
        self.A = agent_a
        self.B = agent_b
        self.total_turns = total_turns
        self._rng = random.Random(seed)
        self.log: List[TurnLog] = []
        self.actor_traits_paragraph: Optional[str] = None
        self.audience_traits_paragraph: Optional[str] = None

        # If no save_dir is provided, create a timestamped directory under ./temp
        # so outputs for this run are grouped in a single folder.
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.save_dir = save_dir
        else:
            now = datetime.datetime.now()
            ts = now.strftime("%Y-%m-%d_%H-%M-%S")
            temp_dir = os.path.join(".", "temp", ts)
            os.makedirs(temp_dir, exist_ok=True)
            self.save_dir = temp_dir

    @staticmethod
    def _ts() -> str:
        return datetime.datetime.now().isoformat(timespec="seconds") + "Z"

    def run(self) -> List[TurnLog]:
        """Execute the conversational loop and return the list of `TurnLog`.

        The method alternates actors each turn and logs both the true
        audience evaluation and the actor's posterior belief (I_hat).
        """
        actor, audience = (self.A, self.B)
        actor_traits_paragraph = actor.initialize_personality_traits(actor.traits)
        audience_traits_paragraph = audience.initialize_personality_traits(audience.traits)
        
        # Store traits in each agent and for JSON output
        actor.trait_paragraph = actor_traits_paragraph
        audience.trait_paragraph = audience_traits_paragraph
        self.actor_traits_paragraph = actor_traits_paragraph
        self.audience_traits_paragraph = audience_traits_paragraph

        print(actor_traits_paragraph)
        print(audience_traits_paragraph)

        last_audience_utt: Optional[Utterance] = None
        
        for t in range(1, self.total_turns + 1):
            print(f"--- Turn {t} ---")
            # 1) Actor (actor) acts: first turn uses act(), subsequent turns use act_based_on_belief()
            print("Actor speaks... ")
            # actor_context, actor_personality = actor.question_check()
            if t == 1:
                actor_utt = actor.act(turn=t)
                I_hat_prev = None
            else:
                I_hat_prev = actor.memory.pf_history[-1]["I_hat"] if actor.memory.pf_history else 0.5
                actor_utt = actor.act_based_on_belief(turn=t, belief=I_hat_prev, audience_last_utt=last_audience_utt)

            # ensure audience has the utterance in their conversation memory
            audience.memory.conversation.append(Utterance(turn=t, actor=actor.name, text=actor_utt.text, body=actor_utt.body))
            
            # 2) Audience evaluates true hidden state I_t and generates a feedback response
            print("audience evaluates and responds... ")
            # audience_context, audience_personality = audience.question_check()
            I_t, audience_reply = audience.audience_evaluate_and_respond(turn=t, actor_utt=actor_utt)
            last_audience_utt = audience_reply

            # 3) Actor updates particle filter given audience's reply and forms a belief I_hat
            print("Actor updates belief... ")
            I_hat, ess = actor.actor_update_particles(turn=t, audience_utt=audience_reply)

            # Compute actor_pe as previous I_hat minus current I_hat
            print("Computing prediction error... ")
            if actor.memory.pf_history and len(actor.memory.pf_history) > 1:
                prev_I_hat = actor.memory.pf_history[-2]["I_hat"]
                actor_pe = np.abs(prev_I_hat - I_hat)
            else:
                actor_pe = 1

            # Log the turn
            self.log.append(TurnLog(
                time=self._ts(),
                turn=t,
                actor=actor.name,
                audience=audience.name,
                actor_text=actor_utt.text,
                actor_body=actor_utt.body,
                audience_I=I_t,
                audience_text=audience_reply.text,
                audience_body=audience_reply.body,
                actor_I_hat=I_hat,
                actor_pe=actor_pe,
                # actor_personality_check=actor_personality,
                # actor_context_check=actor_context,
                # audience_personality_check=audience_personality,
                # audience_context_check=audience_context,
                ess = float(ess)
            ))

        return self.log

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            output = {
                "actor_traits": self.actor_traits_paragraph,
                "audience_traits": self.audience_traits_paragraph,
                "turns": [asdict(l) for l in self.log]
            }
            json.dump(output, f, ensure_ascii=False, indent=2)

def plot_learning_dynamics(log: List[TurnLog], save_dir: str) -> None:
    """
    Plot:
    1) Prediction Error across turns
    2) Score targets/estimates (I_t, I_hat) across turns
    3) Learning gain (|delta I_hat| / |PE|)
    """
    turns = [r.turn for r in log]
    I_t = [r.audience_I for r in log]
    I_hat = [r.actor_I_hat for r in log]
    PE = [r.actor_pe for r in log]

    # Compute belief changes
    delta_I = [0.0]
    for t in range(1, len(I_hat)):
        delta_I.append(I_hat[t] - I_hat[t - 1])

    # Compute learning gain
    eps = 1e-6
    learning_gain = [0.0]
    for t in range(1, len(delta_I)):
        gain = abs(delta_I[t]) / (abs(PE[t - 1]) + eps)
        learning_gain.append(gain)

    # --- Plot 1: Prediction Error ---
    plt.figure()
    plt.plot(turns[1:], PE[1:], marker="o")
    plt.xlabel("Turn")
    plt.ylabel("Prediction Error: abs(posterior − prior)")
    plt.title("Prediction Error Across Turns")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "pe.png"), dpi=200, bbox_inches="tight")

    # --- Plot 2: Belief change ---
    plt.figure()
    plt.plot(turns, I_t, marker="x", label="True I_t")
    plt.plot(turns, I_hat, marker="o", label="Estimated I_hat")
    plt.xlabel("Turn")
    plt.ylabel("Competency Score")
    plt.legend()
    plt.title("I_t and I_hat Across Turns")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "delta_I.png"), dpi=200, bbox_inches="tight")

    # --- Plot 3: Learning gain ---
    plt.figure()
    plt.plot(turns, learning_gain, marker="s")
    plt.xlabel("Turn")
    plt.ylabel("Learning Gain: (delta I_hat / PE)")
    plt.title("Learning Gain Across Turns")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "learning_gain.png"), dpi=200, bbox_inches="tight")


# -------------------------
# CLI / main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Two-agent PE conversation (OpenAI API).")
    parser.add_argument("--turns", type=int, default=2, help="Total turns (messages) in the dialogue.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name, e.g., gpt-4o-mini.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling.")
    parser.add_argument("--window", type=int, default=3, help="Recent K turns to condition on.")
    parser.add_argument("--outfile", type=str, default="pe_conversation_log.json", help="Where to save JSON log.")
    parser.add_argument("--no_audience_norms", action="store_true", help="Disable cultural norms for audience.")
    parser.add_argument("--no_traits", action="store_true", help="Disable personality traits for agents.")
    parser.add_argument("--no_context", action="store_true", help="Disable interview context.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible trait scoring.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save plots and logs. If not provided, a timestamped folder under ./temp/ is created.")
    args = parser.parse_args()

    # Ensure API key exists
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Export it before running.", file=sys.stderr)
        sys.exit(1)

    if not args.no_context:
        print("Using interview context for both agents...")

        interview_role = """
        Role: Customer Service Agent

        Responsibilities:
        - Understanding customer issues and needs
        - Resolving problems efficiently and accurately
        - Communicating solutions clearly and empathetically

        The interview evaluates:
        - Problem-solving ability
        - Judgment and decision-making under constraints
        - Clarity and effectiveness of communication
        """
        
        actor_goal = Goal(
            name="competence",
            description="Your goal is to be perceived as highly competent in an interview. You may talk about your strengths, experiences, and problem-solving skills relevant to the role.",
            role=interview_role,
            ideal=1.0
        )

        audience_goal = Goal(
            name="evaluate_competence",
            description="Your goal is to evaluate how competent the interviewee appears in an interview for the role.",
            role=interview_role,
            ideal=None
        )
    else:
        print("No specific interview context provided.")
        
        actor_goal = Goal(
            name="competence",
            description="Your goal is to be perceived as highly competent by your partner.",
            role=None,
            ideal=1.0
        )

        audience_goal = Goal(
            name="evaluate_competence",
            description="Your goal is to evaluate how competent your partner appears.",
            role=None,
            ideal=None
        )

    # Build the LLM wrapper
    print("Building LLM...")
    llm = make_openai_llm(model=args.model, temperature=args.temperature, top_p=args.top_p)
    # llm = make_local_llm(model="llama3.1:8b")

    aud_norms = ALL_CULTURAL_NORMS.copy() if not args.no_audience_norms else []
    # Ensure actor/audience trait containers exist even if traits are disabled
    actor_traits: List[str] = []
    aud_traits: List[str] = []
    if not args.no_traits:
        # traits = ALL_TRAITS.copy()
        # For parametric traits
        # aud_traits = generate_parametric_traits(traits, is_audience=True)
        # actor_traits = generate_parametric_traits(traits, is_audience=False)
        # For score-based traits (optional alternative)
        # aud_trait_scores = generate_trait_scores(random.Random(args.seed), traits, is_audience=True)
        # actor_trait_scores = generate_trait_scores(random.Random(args.seed+1), traits, is_audience=False)
        # For spreadsheet-extracted traits
        aud_traits = extract_traits_from_spreadsheet("autism-measures-compilation.xlsx")
    else:
        traits = []
        aud_traits = []
        actor_traits = []
    print("Caden cultural norms:", [n.name for n in aud_norms])
    print("Personality assertions (first 10):", [t.assertion for t in aud_traits[:10]])

    # Create agents: Agent A is actor, Agent B is audience
    A = Agent( # interviewee
        name="Riffer",
        goal=actor_goal,
        llm=llm,
        recent_k=args.window,
        seed=args.seed + 1,
        cultural_norms=aud_norms,
        traits=actor_traits,
        trait_scores=None,
        context=not args.no_context
    )

    B = Agent( # interviewer
        name="Caden",
        goal=audience_goal,
        llm=llm,
        recent_k=args.window,
        seed=args.seed + 2,
        cultural_norms=aud_norms,
        traits=aud_traits,
        trait_scores=None,
        context=not args.no_context
    )

    study = ConversationStudy(A, B, save_dir=args.save_dir, total_turns=args.turns, seed=args.seed)
    runlog = study.run()

    # Save plots into the study's save directory (timestamped by default)
    # plot_learning_dynamics(runlog, save_dir=study.save_dir)

    # Pretty print concise trace
    for r in runlog:
        print(f"[t={r.turn}] {r.actor} → {r.audience}: {r.actor_text} [body: {r.actor_body}]")
        print(f"       Audience true I_t={r.audience_I:.2f}; response: {r.audience_text} [body: {r.audience_body}]")
        print(f"       {r.actor} belief I_hat={r.actor_I_hat:.2f}, PE={r.actor_pe:+.2f}")

    # Save full JSON into the study's save_dir
    outpath = os.path.join(study.save_dir, args.outfile)
    study.save_json(outpath)
    print(f"Saved detailed log → {outpath}")

if __name__ == "__main__":
    main()
