#!/usr/bin/env python3
"""
Two LLM agents conversing with PE-driven adaptation (OpenAI API version)
-------------------------------------------------------------------------
- Each agent keeps memory: conversation, PE (prediction error) per turn, reflections, and goal.
- Turn loop: ACT (speaker) -> OBSERVE (listener estimates state & computes PE) -> LEARNING (listener reflects).
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

# -------------------------
# Llama/OpenAI client
# -------------------------
import requests

try:
    from openai import OpenAI
except Exception as e:
    print("ERROR: Failed to import OpenAI client. Install with: pip install -U openai", file=sys.stderr)
    raise

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
            except Exception as e:
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
    speaker: str
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
    name: str
    assertion: str

# Cultural norms
ALL_CULTURAL_NORMS: List[CulturalNorm] = [
    CulturalNorm("Stated purpose first", "Every interaction begins with a shared statement of its goal (e.g., solving a problem, sharing news)."),
    CulturalNorm("Announced topics", "Participants clearly outline discussion topics or goals ahead of time and ask before switching subjects."),
    CulturalNorm("Direct, literal language", "Plain, literal wording is preferred; transparency outweighs courtesy or euphemism. Sarcasm and other kinds of non-literal knowledge are judged negatively."),
    CulturalNorm("Hidden agendas", "Intentions are declared openly; social maneuvering and diplomacy using non-literal or implicit language is considered deceptive and judged negatively."),
    CulturalNorm("Optional small talk", "Chit-chat without clear practical purpose (e.g., small talk about weather or personal topics) are generally frowned upon; skipping it is socially acceptable."),
    CulturalNorm("Respect for passions", "Lengthy monologues about special interests are generally acceptable and listened to attentively."),
    CulturalNorm("Generous common ground", "Speakers assume shared understanding and do not apologize for minor mismatches."),
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
ALL_TRAITS: List[PersonalityTrait] = [
    PersonalityTrait("Detail-focused", "I tend to focus on individual parts and details more than the big picture."),
    PersonalityTrait("Avoids eye contact", "I do not make eye contact when talking with others."),
    PersonalityTrait("Not laid back", "I am not considered “laid back” and am able to 'go with the flow'."),
    PersonalityTrait("Dislikes spontaneity", "I am not comfortable with spontaneity, such as going to new places and trying new things."),
    PersonalityTrait("Repeats phrases", "I use odd phrases or tend to repeat certain words or phrases over and over again."),
    PersonalityTrait("Poor imagination", "I have a poor imagination."),
    PersonalityTrait("Not social", "I do not enjoy social situations where I can meet new people and chat (i.e. parties, dances, sports, games)."),
    PersonalityTrait("Takes things literally", "I sometimes take things too literally, such as missing the point of a joke or having trouble understanding sarcasm."),
    PersonalityTrait("Number-interested", "I am very interested in things related to numbers (i.e. dates, phone numbers, etc.)."),
    PersonalityTrait("Dislikes crowds", "I do not like being around other people."),
    PersonalityTrait("Doesn't share enjoyment", "I do not like to share my enjoyment with others."),
]

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
        except Exception:
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

# -------------------------
# Agent (modified to include norms & traits)
# -------------------------

class Agent:
    """An agent that interacts via an LLM, maintains memory, and uses a
    particle filter to track beliefs about the listener's internal state.

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
        self.context = context

    # Prompt header with norms and personality traits
    def _prompt_header(self) -> str:
        norms_text = ""
        if self.cultural_norms:
            norms_text = "CULTURAL NORMS YOU FOLLOW:\n" + "\n".join(
                f"- {n.name}: {n.description}" for n in self.cultural_norms
            ) + "\n\n"
        traits_text = ""
        if self.traits:
            traits_text = "YOUR PERSONALITY TRAITS, scored from 0 to 3. Each score has the following meaning: {0: False, not at all, 1: Slightly true, 2: Mainly true, 3: Very true.}:\n" + "\n".join(
                f"- {t.name} ({self.trait_scores.get(t.name, 'NA')} / 3): {t.assertion}"
                for t in self.traits
            ) + "\n\n"
        return norms_text + traits_text

    # -------------------------
    # Conversation helpers
    # -------------------------
    def recent_conversation(self, k: Optional[int] = None) -> List[Utterance]:
        """Return the last `k` Utterance objects from memory (most recent last).

        If `k` is None, uses the agent's `recent_k` attribute.
        """
        k = k if k is not None else self.recent_k
        return self.memory.conversation[-k:] if self.memory.conversation else []

    def format_conversation(self, conv: List[Utterance]) -> str:
        """Format a list of Utterances into a compact, readable block.

        Returns a string with each turn on a separate line prefixed by
        "- [t={turn} {speaker}] text". If list is empty returns "- (none)".
        """
        if not conv:
            return "- (none)"
        return chr(10).join(f"- [t={u.turn} {u.speaker}] {u.text}" for u in conv)


    def initialize_cultural_norms(self, norms: List[CulturalNorm]) -> None:
        """Set cultural norms for the agent."""
        if not self.cultural_norms:
            return
        prompt = f"""You are {self.name}. You are in an alternative world in the year 3025 where there is a new set of cultural norms. In all your interactions, you must follow these cultural norms:{norms.description}.
        If you fail to do so, you will be unsuccessful in your interactions and perceived negatively by others. Always follow these norms strictly."""
        self.llm(prompt)

    # ---------- Observe ----------

    def audience_evaluate_and_respond(self, turn: int, actor_utt: Utterance) -> Tuple[float, Utterance]:
        """Audience (listener) forms a hidden evaluation I_t ∈ [0,1] of the
        actor's performance and returns a brief feedback `Utterance`.

        The method prompts the LLM to return a single numeric rating in the
        [0,1] interval (optional short comment allowed). We use a regex to
        extract the first numeric token (fallback 0.5 on parse failure).
        """
        actor_name = "partner"
        if self.context:
            actor_name = "interviewee"

        context_prompt = f"""You are interviewing for a candidate for the following role: {self.memory.goal.role}."""
        eval_prompt = self._prompt_header() + f"""You are {self.name}. You want to {self.memory.goal.name}. {self.memory.goal.description}. {context_prompt if self.context else ""}
        From your {actor_name}'s most recent utterance and body language (below), rate how competent your {actor_name} is according to the goal, on scale from [0,1].
        Respond with a single number from [0,1].
        Your options are [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], where 0.0 means not competent at all and 1.0 means extremely competent.

        {actor_name} said: \"{actor_utt.text}\"  Body language: \"{actor_utt.body}\"
        """

        eval_raw = self.llm(eval_prompt)
        m = re.search(r"([01](?:\.\d+)?)", eval_raw)
        I_t = float(m.group(1)) if m else 0.5
        I_t = max(0.0, min(1.0, I_t))

        conv_k = self.recent_conversation(self.recent_k)

        resp_prompt = self._prompt_header() + f"""You are {self.name}. You want to {self.memory.goal.name}. {self.memory.goal.description}. {context_prompt if self.context else ""}
        You rated the {actor_name} with score {I_t:.2f} on a scale from 0 to 1, where 0 indicates "not at all", and 1 indicates "to a great extent".
        Produce a short reply that reflects your evaluation of the {actor_name}’s competence and matches your score, and include a very brief body language description.
        
        Consider recent conversation history in forming your response, while matching your score in sentiment.

        Recent conversation (last {self.recent_k}):
        {self.format_conversation(conv_k)}

        Output in this format exactly:
        DIALOGUE: <one sentence>
        BODY: <brief body language phrase>"
"""
        resp_raw = self.llm(resp_prompt)
        dlg = ""
        body = ""
        m1 = re.search(r"DIALOGUE:\s*(.*)", resp_raw)
        m2 = re.search(r"BODY:\s*(.*)", resp_raw)
        if m1:
            dlg = m1.group(1).strip()
        else:
            dlg = resp_raw.strip()
        if m2:
            body = m2.group(1).strip()

        utt = Utterance(turn=turn, speaker=self.name, text=dlg, body=body)
        self.memory.conversation.append(utt)
        return I_t, utt

    def actor_update_particles(self, turn: int, listener_utt: Utterance, goal_description: str, pf_model: Optional[ParticleFilter] = None) -> Tuple[float, float]:
        """Update the actor's particle filter based on the listener's
        response and return the posterior belief I_hat and the ESS.

        Steps:
          1. Initialize or load current particles & weights.
          2. Predict step: diffuse particles with process noise.
          3. Use an LLM to 'read' the listener's reply and map it to a
             numeric measurement in [0,1].
          4. Weight particles by the measurement likelihood (Gaussian).
          5. Optionally resample by ESS and compute posterior mean I_hat.
          6. Store PF metadata in `self.memory.pf_history` for later use.

        Returns
        -------
        I_hat: float
            Posterior mean belief about the listener's evaluation.
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

        audience_name = "listener"
        if self.context:
            audience_name = "interviewer"

         # Use LLM to produce a measurement from the audience's reply
        meas_prompt = self._prompt_header() + f"""You are {self.name}. {goal_description}. From the {audience_name}'s reply (dialogue and body language), estimate the {audience_name}'s internal evaluation of you on your goal. Respond with a single number in [0,1].

        {audience_name} said: \"{listener_utt.text}\"  Body language: \"{listener_utt.body}\"
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
        self.memory.pe_history.append(PERecord(turn=turn, partner_text=listener_utt.text, estimate=I_hat, pe=pe))
        return I_hat, float(ess)

    # ---------- Learning ----------
    def learning(self, turn: int) -> ReflectionRecord:
        """Ask the agent's LLM for a concise reflection based on the most
        recent posterior belief (I_hat). The reflection is intended to be
        a short plan for what the agent will try to do next turn to improve
        perceived performance (concrete and brief).
        """
        I_hat_last = self.memory.pf_history[-1]["I_hat"] if self.memory.pf_history else 0.5
        context_prompt = f"""You are interviewing for the following role: {self.memory.goal.role}."""
        prompt = self._prompt_header() + f"""You are {self.name}. You want to achieve: {self.memory.goal.name}. {self.memory.goal.description}. {context_prompt if self.context else ""}
        Your current belief about how well you're achieving the goal is {I_hat_last:.2f} (0-1).
        Write a short reflection: What will you change next turn to improve your goal achievement? Keep it concrete and brief.
        """
        text = self.llm(prompt).strip()
        rec = ReflectionRecord(turn=turn, text=text)
        self.memory.reflections.append(rec)
        return rec

    # ---------- Act ----------
    def act(self, turn: int) -> Utterance:
        """Produce an utterance to the partner aimed at improving the goal. 
        This is only used for the initial turn (turn 0) when there is no conversation history.

        The method prompts the agent's LLM with the goal and expects output formatted as:
          DIALOGUE: <one sentence>
          BODY: <brief body language phrase>

        Parse the output via regex and fallback to the raw response when
        the format is not followed strictly.
        """
        context_prompt = f"""You are interviewing for the following role: {self.memory.goal.role}."""
        prompt = self._prompt_header() + f"""You are {self.name}. You want to achieve: {self.memory.goal.name}.
        Definition: {self.memory.goal.description}. {context_prompt if self.context else ""}
        Ideal value: {self.memory.goal.ideal:.2f}

        You must talk and behave with the aim of achieving the goal and maximizing it to its ideal value.

        Produce a short utterance (one sentence) to the listener to accomplish the goal, and include a very brief body language description.
        Output in this format exactly:
        DIALOGUE: <one sentence>
        BODY: <brief body language phrase>
        """
        raw = self.llm(prompt).strip()
        m1 = re.search(r"DIALOGUE:\s*(.*)", raw)
        m2 = re.search(r"BODY:\s*(.*)", raw)
        text = m1.group(1).strip() if m1 else raw
        body = m2.group(1).strip() if m2 else ""
        utt = Utterance(turn=turn, speaker=self.name, text=text, body=body)
        self.memory.conversation.append(utt)
        return utt

    def act_based_on_belief(self, turn: int, belief: float) -> Utterance:
        """Produce an utterance conditioned on a belief estimate (I_hat).
        This is used after the initial turn when there is conversation history.

        This method includes recent conversation history and a brief summary of recent 
        I_hat values (from `self.memory.pf_history`) so the agent can reason about
        whether their previous moves had the intended effect.
        """
        conv_k = self.recent_conversation(self.recent_k)

        def fmt_ihat(h: Dict[str, float]) -> str:
            return f"(turn {int(h.get('turn', 0))}) I_hat={h.get('I_hat', 0.5):.2f}"

        ihat_k = self.memory.pf_history[-self.recent_k:] if self.memory.pf_history else []

        audience_name = "listener"
        if self.context:
            audience_name = "interviewer"

        context_prompt = f"""You are interviewing for the following role: {self.memory.goal.role}."""
        prompt = self._prompt_header() + f"""You are {self.name}. You want to achieve: {self.memory.goal.name}.
        Definition: {self.memory.goal.description}. {context_prompt if self.context else ""}
        Ideal value: {self.memory.goal.ideal:.2f}

        You must talk and behave with the aim of achieving the goal and maximizing it to its ideal value.
        Consider recent conversation, history, and your reflections.

        Current belief about the {audience_name}'s evaluation of how well you are performing = {belief:.2f} (on a scale from 0-1).

        Recent conversation (last {self.recent_k}):
        {self.format_conversation(conv_k)}

        Recent I_hat (belief) history:
        {chr(10).join("- " + fmt_ihat(h) for h in ihat_k) or "- (none)"}

        Produce a short utterance (one sentence) to the {audience_name} to accomplish the goal, and include a very brief body language description.
        Output in this format exactly:
        DIALOGUE: <one sentence>
        BODY: <brief body language phrase>
        """
        raw = self.llm(prompt).strip()
        m1 = re.search(r"DIALOGUE:\s*(.*)", raw)
        m2 = re.search(r"BODY:\s*(.*)", raw)
        text = m1.group(1).strip() if m1 else raw
        body = m2.group(1).strip() if m2 else ""
        utt = Utterance(turn=turn, speaker=self.name, text=text, body=body)
        self.memory.conversation.append(utt)
        return utt

# -------------------------
# Orchestrator
# -------------------------

@dataclass
class TurnLog:
    time: str
    turn: int
    speaker: str
    listener: str
    # initial speaker utterance
    speaker_text: str
    speaker_body: str
    # audience (listener) true hidden state and response
    audience_I: float
    audience_text: str
    audience_body: str
    # actor belief after updating particles
    actor_I_hat: float
    actor_pe: float
    # actor reflction
    reflection_text: str
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

        The method alternates speakers each turn and logs both the true
        audience evaluation and the actor's posterior belief (I_hat).
        """
        speaker, listener = (self.A, self.B)
        speaker.initialize_cultural_norms(speaker.cultural_norms)
        
        for t in range(1, self.total_turns + 1):
            print(f"--- Turn {t} ---")
            # 1) Actor (speaker) acts: first turn uses act(), subsequent turns use act_based_on_belief()
            print("Actor speaks... ")
            if t == 1:
                speaker_utt = speaker.act(turn=t)
                I_hat_prev = None
            else:
                I_hat_prev = speaker.memory.pf_history[-1]["I_hat"] if speaker.memory.pf_history else 0.5
                speaker_utt = speaker.act_based_on_belief(turn=t, belief=I_hat_prev)

            # ensure listener has the utterance in their conversation memory
            listener.memory.conversation.append(Utterance(turn=t, speaker=speaker.name, text=speaker_utt.text, body=speaker_utt.body))

            # 2) Audience evaluates true hidden state I_t and generates a feedback response
            print("Listener evaluates and responds... ")
            I_t, listener_reply = listener.audience_evaluate_and_respond(turn=t, actor_utt=speaker_utt)

            # 3) Actor updates particle filter given listener's reply and forms a belief I_hat
            print("Actor updates belief... ")
            I_hat, ess = speaker.actor_update_particles(turn=t, listener_utt=listener_reply, goal_description=speaker.memory.goal.description)

            # 4) Optional learning/reflection by listener (based on PE of actor's belief)
            refl = speaker.learning(turn=t)

            # Compute actor_pe as previous I_hat minus current I_hat
            print("Computing prediction error... ")
            if speaker.memory.pf_history and len(speaker.memory.pf_history) > 1:
                prev_I_hat = speaker.memory.pf_history[-2]["I_hat"]
                actor_pe = np.abs(prev_I_hat - I_hat)
            else:
                actor_pe = 1

            # Log the turn
            self.log.append(TurnLog(
                time=self._ts(),
                turn=t,
                speaker=speaker.name,
                listener=listener.name,
                speaker_text=speaker_utt.text,
                speaker_body=speaker_utt.body,
                audience_I=I_t,
                audience_text=listener_reply.text,
                audience_body=listener_reply.body,
                actor_I_hat=I_hat,
                actor_pe=actor_pe,
                reflection_text=refl.text,
                ess = float(ess)
            ))

        return self.log

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(l) for l in self.log], f, ensure_ascii=False, indent=2)

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
        Role: Product Manager

        Responsibilities:
        - Defining product requirements
        - Making tradeoff decisions
        - Communicating priorities clearly

        The interview evaluates:
        - Structured thinking
        - Decision rationale
        - Clarity of communication
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

    rng = random.Random(args.seed)
    aud_norms = ALL_CULTURAL_NORMS.copy() if not args.no_audience_norms else []
    traits = ALL_TRAITS.copy() if not args.no_traits else []
    print("Audience cultural norms:", [n.name for n in aud_norms])
    print("Personality traits:", [t.name for t in traits])

    aud_trait_scores = generate_trait_scores(rng, traits, is_audience=True)
    actor_trait_scores = generate_trait_scores(rng, traits, is_audience=False)


    # Create agents: Agent A is actor, Agent B is audience
    A = Agent(
        name="John",
        goal=actor_goal,
        llm=llm,
        recent_k=args.window,
        seed=args.seed + 1,
        cultural_norms=[],
        traits=traits,
        trait_scores=actor_trait_scores,
        context=not args.no_context
    )

    B = Agent(
        name="Jane",
        goal=audience_goal,
        llm=llm,
        recent_k=args.window,
        seed=args.seed + 2,
        cultural_norms=aud_norms,
        traits=traits,
        trait_scores=aud_trait_scores,
        context=not args.no_context
    )

    study = ConversationStudy(A, B, save_dir=args.save_dir, total_turns=args.turns, seed=args.seed)
    runlog = study.run()

    # Save plots into the study's save directory (timestamped by default)
    plot_learning_dynamics(runlog, save_dir=study.save_dir)

    # Pretty print concise trace
    for r in runlog:
        print(f"[t={r.turn}] {r.speaker} → {r.listener}: {r.speaker_text} [body: {r.speaker_body}]")
        print(f"       Audience true I_t={r.audience_I:.2f}; response: {r.audience_text} [body: {r.audience_body}]")
        print(f"       {r.speaker} belief I_hat={r.actor_I_hat:.2f}, PE={r.actor_pe:+.2f}; reflection: {r.reflection_text}\n")

    # Save full JSON into the study's save_dir
    outpath = os.path.join(study.save_dir, args.outfile)
    study.save_json(outpath)
    print(f"Saved detailed log → {outpath}")

if __name__ == "__main__":
    main()
