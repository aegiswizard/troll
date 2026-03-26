"""
troll/search/engine.py

Search engine for Troll 🧌.
Manages action candidates, branch scoring, novelty, and loop breaking.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from troll.memory.store import EpisodicMemory


# ---------------------------------------------------------------------------
# Branch
# ---------------------------------------------------------------------------

@dataclass
class Branch:
    action: Any
    novelty_score: float
    progress_score: float
    combined_score: float
    source: str  # "planner" | "random" | "probe"


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------

class SearchEngine:
    """
    Generates and ranks action candidates per step.

    Modes
    -----
    greedy      — always pick the planner's top choice
    balanced    — blend planner + novelty (default)
    deep        — keep multiple branches alive
    competition — greedy + strict loop breaking
    """

    def __init__(
        self,
        mode: str = "balanced",
        max_branches: int = 3,
        novelty_weight: float = 0.4,
        progress_weight: float = 0.6,
        loop_detection_window: int = 5,
    ) -> None:
        self._mode = mode
        self._max_branches = max_branches
        self._nw = novelty_weight
        self._pw = progress_weight
        self._loop_window = loop_detection_window

        self._action_counts: Dict[str, int] = {}
        self._visited_states: Set[str] = set()
        self._total_steps = 0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def select_action(
        self,
        planner_action: Any,
        planner_alternatives: List[Any],
        available_actions: List[Any],
        episodic: EpisodicMemory,
        reward: float,
    ) -> Any:
        """Choose the final action to execute."""
        self._total_steps += 1

        if self._mode == "greedy":
            return self._greedy(planner_action, episodic)

        if self._mode == "competition":
            return self._competition(planner_action, episodic)

        # balanced / deep
        branches = self._generate_branches(
            planner_action, planner_alternatives, available_actions, episodic
        )
        if not branches:
            return planner_action
        best = max(branches, key=lambda b: b.combined_score)
        return best.action

    def record_step(self, action: Any, state_hash: str) -> None:
        key = str(action)
        self._action_counts[key] = self._action_counts.get(key, 0) + 1
        self._visited_states.add(state_hash)

    def is_novel(self, action: Any) -> bool:
        return self._action_counts.get(str(action), 0) == 0

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _greedy(self, planner_action: Any, episodic: EpisodicMemory) -> Any:
        if episodic.loop_detected(self._loop_window):
            # break the loop with a random novel action if possible
            novel = self._find_novel_action()
            if novel is not None:
                return novel
        return planner_action

    def _competition(self, planner_action: Any, episodic: EpisodicMemory) -> Any:
        """Strict: planner first, break loops, but never random."""
        if episodic.loop_detected(self._loop_window):
            least_tried = self._least_tried_action()
            if least_tried is not None:
                return least_tried
        return planner_action

    def _generate_branches(
        self,
        planner_action: Any,
        alternatives: List[Any],
        available: List[Any],
        episodic: EpisodicMemory,
    ) -> List[Branch]:
        candidates = [planner_action] + alternatives[:self._max_branches - 1]

        # add a probe action if we're stuck
        if episodic.loop_detected(self._loop_window):
            probe = self._find_novel_action(available)
            if probe is not None:
                candidates.append(probe)

        branches = []
        for i, action in enumerate(candidates):
            novelty = self._novelty_score(action)
            progress = 1.0 - (i * 0.2)  # planner is top, alternatives diminish
            combined = self._nw * novelty + self._pw * progress
            source = "planner" if i == 0 else ("probe" if action == candidates[-1] else "alternative")
            branches.append(Branch(
                action=action,
                novelty_score=novelty,
                progress_score=progress,
                combined_score=combined,
                source=source,
            ))
        return branches

    def _novelty_score(self, action: Any) -> float:
        count = self._action_counts.get(str(action), 0)
        if count == 0:
            return 1.0
        return 1.0 / (1.0 + count)

    def _find_novel_action(self, available: Optional[List[Any]] = None) -> Optional[Any]:
        if available:
            novel = [a for a in available if self._action_counts.get(str(a), 0) == 0]
            if novel:
                return random.choice(novel)
        return None

    def _least_tried_action(self) -> Optional[Any]:
        if not self._action_counts:
            return None
        return min(self._action_counts, key=lambda k: self._action_counts[k])

    def stats(self) -> Dict[str, Any]:
        return {
            "mode": self._mode,
            "total_steps": self._total_steps,
            "unique_actions_tried": len(self._action_counts),
            "unique_states_visited": len(self._visited_states),
            "action_counts": dict(self._action_counts),
        }
