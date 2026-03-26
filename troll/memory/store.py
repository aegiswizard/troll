"""
troll/memory/store.py

Four memory stores in one module:
  EpisodicMemory    — step-by-step run history
  HypothesisMemory  — candidate world-model rules
  ProceduralMemory  — reusable tactics
  RunStore          — experiment metadata
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# ===========================================================================
# Episodic Memory
# ===========================================================================

@dataclass
class EpisodicEntry:
    step: int
    action: Any
    action_params: Dict[str, Any]
    state_summary: str
    delta_summary: str
    reward: float
    progress_improved: bool
    reasoning_note: str
    cost_usd: float
    latency_s: float
    timestamp: float = field(default_factory=time.time)


class EpisodicMemory:
    """Rolling step-by-step run history."""

    def __init__(self, max_entries: int = 200) -> None:
        self._entries: List[EpisodicEntry] = []
        self._max = max_entries

    def record(self, entry: EpisodicEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max:
            self._entries.pop(0)

    def recent(self, n: int = 10) -> List[EpisodicEntry]:
        return self._entries[-n:]

    def recap(self, n: int = 5) -> str:
        """Compact narrative of last N steps for injection into prompts."""
        recent = self.recent(n)
        if not recent:
            return "No steps taken yet."
        lines = []
        for e in recent:
            improved = "✓" if e.progress_improved else "✗"
            lines.append(
                f"Step {e.step}: action={e.action} {improved} reward={e.reward:.2f} | {e.reasoning_note[:60]}"
            )
        return "\n".join(lines)

    def loop_detected(self, window: int = 5) -> bool:
        """True if the last `window` actions are identical (stuck loop)."""
        recent = self.recent(window)
        if len(recent) < window:
            return False
        actions = [str(e.action) for e in recent]
        return len(set(actions)) == 1

    def all(self) -> List[EpisodicEntry]:
        return list(self._entries)

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [vars(e) for e in self._entries]


# ===========================================================================
# Hypothesis Memory
# ===========================================================================

HypothesisStatus = Literal["candidate", "confirmed", "refuted", "unknown"]


@dataclass
class Hypothesis:
    id: str
    description: str                     # "Action 1 = move up"
    confidence: float                    # 0.0 – 1.0
    status: HypothesisStatus = "candidate"
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    last_tested_step: Optional[int] = None
    created_step: int = 0
    timestamp: float = field(default_factory=time.time)


class HypothesisMemory:
    """Tracks and updates candidate world-model hypotheses."""

    def __init__(self) -> None:
        self._hypotheses: Dict[str, Hypothesis] = {}
        self._id_counter = 0

    def add(self, description: str, confidence: float = 0.5, step: int = 0) -> Hypothesis:
        hid = f"h{self._id_counter:03d}"
        self._id_counter += 1
        h = Hypothesis(id=hid, description=description, confidence=confidence, created_step=step)
        self._hypotheses[hid] = h
        return h

    def update(self, hid: str, delta_confidence: float, evidence: str, step: int) -> None:
        if hid not in self._hypotheses:
            return
        h = self._hypotheses[hid]
        h.confidence = max(0.0, min(1.0, h.confidence + delta_confidence))
        h.last_tested_step = step
        if delta_confidence > 0:
            h.evidence_for.append(f"step {step}: {evidence}")
            if h.confidence > 0.85:
                h.status = "confirmed"
        else:
            h.evidence_against.append(f"step {step}: {evidence}")
            if h.confidence < 0.15:
                h.status = "refuted"

    def active(self) -> List[Hypothesis]:
        return [h for h in self._hypotheses.values() if h.status not in ("refuted",)]

    def top(self, n: int = 5) -> List[Hypothesis]:
        return sorted(self.active(), key=lambda h: h.confidence, reverse=True)[:n]

    def recap(self) -> str:
        top = self.top(5)
        if not top:
            return "No hypotheses yet."
        return "; ".join(f"{h.description} ({h.confidence:.0%})" for h in top)

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [vars(h) for h in self._hypotheses.values()]


# ===========================================================================
# Procedural Memory
# ===========================================================================

@dataclass
class Tactic:
    name: str
    description: str
    when_to_use: str
    action_template: Optional[str] = None   # pseudocode
    success_count: int = 0
    failure_count: int = 0


BUILT_IN_TACTICS: List[Tactic] = [
    Tactic(
        name="edge_probe",
        description="Probe all four edges of the grid systematically.",
        when_to_use="When the goal / reward source is unknown and the grid boundary matters.",
        action_template="Iterate: top row, bottom row, left col, right col",
    ),
    Tactic(
        name="open_space_explore",
        description="Walk through empty regions to discover objects or triggers.",
        when_to_use="When the grid is sparse and object positions are unknown.",
        action_template="Move agent toward largest empty region",
    ),
    Tactic(
        name="cheap_info_first",
        description="Choose the action that maximises information gain cheaply.",
        when_to_use="When no hypothesis has > 60 % confidence.",
        action_template="Try each novel action once before repeating any",
    ),
    Tactic(
        name="object_follow",
        description="Track a moving or blinking object.",
        when_to_use="When a changing object was detected in delta.",
    ),
    Tactic(
        name="hazard_avoidance",
        description="Avoid cells that caused reward drops.",
        when_to_use="When reward turned negative after moving to certain cell.",
    ),
    Tactic(
        name="hypothesis_test",
        description="Deliberately trigger a predicted state to verify a hypothesis.",
        when_to_use="When one hypothesis has 40-80 % confidence and hasn't been tested.",
    ),
]


class ProceduralMemory:
    """Library of reusable tactics."""

    def __init__(self) -> None:
        self._tactics: Dict[str, Tactic] = {t.name: t for t in BUILT_IN_TACTICS}

    def add(self, tactic: Tactic) -> None:
        self._tactics[tactic.name] = tactic

    def record_outcome(self, name: str, success: bool) -> None:
        if name in self._tactics:
            if success:
                self._tactics[name].success_count += 1
            else:
                self._tactics[name].failure_count += 1

    def recommend(self, context: str = "") -> List[Tactic]:
        """Return tactics sorted by success rate (placeholder logic)."""
        return sorted(
            self._tactics.values(),
            key=lambda t: t.success_count - t.failure_count,
            reverse=True,
        )

    def recap(self) -> str:
        return "; ".join(
            f"{t.name}(✓{t.success_count}/✗{t.failure_count})"
            for t in self._tactics.values()
        )


# ===========================================================================
# Run Store — experiment metadata
# ===========================================================================

@dataclass
class RunMetadata:
    run_id: str
    env_id: str
    provider: str
    model_id: str
    mode: str
    max_budget_usd: float
    max_steps: int
    search_mode: str
    scorecard_id: Optional[str] = None
    checkpoint_path: Optional[str] = None
    seed: Optional[int] = None
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def finish(self) -> None:
        self.finished_at = time.time()

    @property
    def elapsed_s(self) -> float:
        end = self.finished_at or time.time()
        return round(end - self.started_at, 2)
