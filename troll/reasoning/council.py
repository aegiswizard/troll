"""
troll/reasoning/council.py

The Reasoning Council — Troll's brain.

Five internal roles, one underlying model (v1).
Each role gets a focused system prompt and a compact state pack.

  Observer          — what changed, what matters
  HypothesisBuilder — candidate world rules
  Planner           — what to do next
  Verifier          — did last action match expectation?
  Compressor        — shrink context between long runs
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from troll.memory.store import EpisodicMemory, HypothesisMemory, ProceduralMemory
from troll.perception.frame_parser import StatePack
from troll.providers.router import ProviderRouter


# ---------------------------------------------------------------------------
# Council output shapes
# ---------------------------------------------------------------------------

@dataclass
class Observation_:
    what_changed: str
    what_matters: str
    last_action_helped: bool
    notes: str = ""


@dataclass
class HypothesisUpdate:
    new_hypotheses: List[str]
    updated_ids: List[str]
    deltas: List[float]
    reasoning: str


@dataclass
class Plan:
    chosen_action: Any
    action_params: Dict[str, Any]
    reasoning: str
    confidence: float
    alternatives: List[Any] = field(default_factory=list)


@dataclass
class Verification:
    matched_expectation: bool
    surprise: str
    suggested_hypothesis_updates: List[str]


@dataclass
class CouncilOutput:
    observation: Observation_
    hypothesis_update: HypothesisUpdate
    plan: Plan
    verification: Optional[Verification]
    compressed_summary: Optional[str]


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_OBSERVER = """You are the Observer role in an ARC-AGI-3 agent.
You read the current StatePack and identify:
1. What visibly changed from the last step
2. What likely matters for making progress
3. Whether the last action helped (reward increased or visible progress)

Be factual, brief, and specific. Reference grid coordinates and colors when relevant.
Respond in JSON: {"what_changed": "...", "what_matters": "...", "last_action_helped": true/false, "notes": "..."}"""

SYSTEM_HYPOTHESIS = """You are the Hypothesis Builder in an ARC-AGI-3 agent.
Maintain a running model of the environment's rules.
Based on the current state and observation, either:
- Propose NEW hypotheses (e.g. "Action 2 moves the agent left")
- Update confidence in EXISTING hypotheses based on evidence

Respond in JSON:
{
  "new_hypotheses": ["<description>", ...],
  "updated_ids": ["h001", ...],
  "deltas": [+0.2, ...],
  "reasoning": "..."
}
Only include updates if there is evidence. Do not hallucinate."""

SYSTEM_PLANNER = """You are the Planner in an ARC-AGI-3 agent.
Given the current state, hypotheses, and memory — choose the BEST next action.

Rules:
- Prefer actions that maximise information gain if unsure
- Prefer goal-directed actions if the goal is known
- Avoid recently repeated actions unless they improved reward
- If stuck in a loop, try a novel action

Respond in JSON:
{
  "chosen_action": <int or dict>,
  "action_params": {},
  "reasoning": "...",
  "confidence": 0.0-1.0,
  "alternatives": [<action>, ...]
}"""

SYSTEM_VERIFIER = """You are the Verifier in an ARC-AGI-3 agent.
After an action was taken, compare what was EXPECTED vs what ACTUALLY happened.
Identify surprises and suggest hypothesis updates.

Respond in JSON:
{
  "matched_expectation": true/false,
  "surprise": "...",
  "suggested_hypothesis_updates": ["..."]
}"""

SYSTEM_COMPRESSOR = """You are the Compressor in an ARC-AGI-3 agent.
The episode history is getting long. Summarise it into a compact paragraph
that preserves what has been learned about:
- The environment rules
- What actions do
- What the goal appears to be
- What has and hasn't worked

Max 150 words. Return only the summary text."""


# ---------------------------------------------------------------------------
# Council
# ---------------------------------------------------------------------------

class ReasoningCouncil:
    """
    Runs all five reasoning roles in sequence per step.
    v1: single model handles all roles.
    v2: different models for cheap vs expensive roles.
    """

    def __init__(
        self,
        router: ProviderRouter,
        episodic: EpisodicMemory,
        hypotheses: HypothesisMemory,
        procedural: ProceduralMemory,
        compress_every: int = 20,
    ) -> None:
        self._router = router
        self._episodic = episodic
        self._hypotheses = hypotheses
        self._procedural = procedural
        self._compress_every = compress_every
        self._compressed_context: Optional[str] = None
        self._prev_plan: Optional[Plan] = None

    def run(self, pack: StatePack, step: int) -> CouncilOutput:
        context = self._build_context(pack)

        # 1. Observe
        obs = self._observe(context, step)

        # 2. Update hypotheses
        hyp_update = self._update_hypotheses(context, obs, step)
        self._apply_hypothesis_updates(hyp_update, step)

        # 3. Plan
        plan = self._plan(context, obs, step)

        # 4. Verify previous action (if we have a prior plan)
        verification = None
        if self._prev_plan is not None:
            verification = self._verify(context, self._prev_plan, obs, step)

        # 5. Compress occasionally
        compressed = None
        if step > 0 and step % self._compress_every == 0:
            compressed = self._compress(step)
            self._compressed_context = compressed

        self._prev_plan = plan

        return CouncilOutput(
            observation=obs,
            hypothesis_update=hyp_update,
            plan=plan,
            verification=verification,
            compressed_summary=compressed,
        )

    # -----------------------------------------------------------------------
    # Role implementations
    # -----------------------------------------------------------------------

    def _observe(self, context: str, step: int) -> Observation_:
        result = self._router.complete(
            messages=[{"role": "user", "content": context}],
            system=SYSTEM_OBSERVER,
            max_tokens=512,
            temperature=0.1,
            role="observer",
            step=step,
        )
        data = _parse_json(result.text, {
            "what_changed": "Unknown",
            "what_matters": "Unknown",
            "last_action_helped": False,
            "notes": "",
        })
        return Observation_(
            what_changed=data.get("what_changed", ""),
            what_matters=data.get("what_matters", ""),
            last_action_helped=bool(data.get("last_action_helped", False)),
            notes=data.get("notes", ""),
        )

    def _update_hypotheses(self, context: str, obs: Observation_, step: int) -> HypothesisUpdate:
        hyp_context = (
            f"{context}\n\n"
            f"[Observer says]\n"
            f"Changed: {obs.what_changed}\n"
            f"Matters: {obs.what_matters}\n\n"
            f"[Current hypotheses]\n{self._hypotheses.recap()}"
        )
        result = self._router.complete(
            messages=[{"role": "user", "content": hyp_context}],
            system=SYSTEM_HYPOTHESIS,
            max_tokens=512,
            temperature=0.2,
            role="hypothesis",
            step=step,
        )
        data = _parse_json(result.text, {
            "new_hypotheses": [],
            "updated_ids": [],
            "deltas": [],
            "reasoning": "",
        })
        return HypothesisUpdate(
            new_hypotheses=data.get("new_hypotheses", []),
            updated_ids=data.get("updated_ids", []),
            deltas=data.get("deltas", []),
            reasoning=data.get("reasoning", ""),
        )

    def _plan(self, context: str, obs: Observation_, step: int) -> Plan:
        plan_context = (
            f"{context}\n\n"
            f"[Observer]\nChanged: {obs.what_changed}\nMatters: {obs.what_matters}\n\n"
            f"[Hypotheses]\n{self._hypotheses.recap()}\n\n"
            f"[Recent memory]\n{self._episodic.recap(5)}\n\n"
            f"[Tactics available]\n{self._procedural.recap()}"
        )
        if self._compressed_context:
            plan_context = f"[Compressed history]\n{self._compressed_context}\n\n" + plan_context

        result = self._router.complete(
            messages=[{"role": "user", "content": plan_context}],
            system=SYSTEM_PLANNER,
            max_tokens=768,
            temperature=0.3,
            thinking=(step % 5 == 0),  # use thinking mode periodically
            role="planner",
            step=step,
        )
        data = _parse_json(result.text, {
            "chosen_action": 0,
            "action_params": {},
            "reasoning": "Fallback: action 0",
            "confidence": 0.5,
            "alternatives": [],
        })
        return Plan(
            chosen_action=data.get("chosen_action", 0),
            action_params=data.get("action_params", {}),
            reasoning=data.get("reasoning", ""),
            confidence=float(data.get("confidence", 0.5)),
            alternatives=data.get("alternatives", []),
        )

    def _verify(self, context: str, prev_plan: Plan, obs: Observation_, step: int) -> Verification:
        verify_context = (
            f"Previous plan reasoning: {prev_plan.reasoning}\n"
            f"Previous action: {prev_plan.chosen_action}\n\n"
            f"What actually happened:\n{obs.what_changed}\n{obs.what_matters}"
        )
        result = self._router.complete(
            messages=[{"role": "user", "content": verify_context}],
            system=SYSTEM_VERIFIER,
            max_tokens=384,
            temperature=0.1,
            role="verifier",
            step=step,
        )
        data = _parse_json(result.text, {
            "matched_expectation": True,
            "surprise": "",
            "suggested_hypothesis_updates": [],
        })
        return Verification(
            matched_expectation=bool(data.get("matched_expectation", True)),
            surprise=data.get("surprise", ""),
            suggested_hypothesis_updates=data.get("suggested_hypothesis_updates", []),
        )

    def _compress(self, step: int) -> str:
        history_text = self._episodic.recap(self._compress_every)
        result = self._router.complete(
            messages=[{"role": "user", "content": f"Episode history so far:\n{history_text}"}],
            system=SYSTEM_COMPRESSOR,
            max_tokens=256,
            temperature=0.1,
            role="compressor",
            step=step,
        )
        return result.text.strip()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _build_context(self, pack: StatePack) -> str:
        return pack.to_prompt_block()

    def _apply_hypothesis_updates(self, update: HypothesisUpdate, step: int) -> None:
        for desc in update.new_hypotheses:
            self._hypotheses.add(desc, confidence=0.5, step=step)
        for hid, delta in zip(update.updated_ids, update.deltas):
            self._hypotheses.update(hid, delta, "model update", step)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _parse_json(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Robustly parse JSON from LLM output, stripping markdown fences."""
    text = text.strip()
    # Strip ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    # Find first {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return default
