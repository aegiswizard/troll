"""
troll/core/budgets.py
Real-time budget tracking. Stops the run before we overspend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class StepCost:
    step: int
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    role: str  # "observer" | "planner" | "verifier" | "compressor" | "hypothesis"
    timestamp: float = field(default_factory=time.time)


@dataclass
class BudgetTracker:
    max_budget_usd: float
    max_per_step_usd: float
    max_steps: int

    _total_cost: float = field(default=0.0, init=False)
    _total_input_tokens: int = field(default=0, init=False)
    _total_output_tokens: int = field(default=0, init=False)
    _steps_completed: int = field(default=0, init=False)
    _history: List[StepCost] = field(default_factory=list, init=False)
    _run_start: float = field(default_factory=time.time, init=False)

    # -----------------------------------------------------------------------
    # Recording
    # -----------------------------------------------------------------------

    def record(self, cost: StepCost) -> None:
        self._total_cost += cost.cost_usd
        self._total_input_tokens += cost.input_tokens
        self._total_output_tokens += cost.output_tokens
        self._history.append(cost)

    def mark_step_complete(self) -> None:
        self._steps_completed += 1

    # -----------------------------------------------------------------------
    # Budget checks
    # -----------------------------------------------------------------------

    def over_total(self) -> bool:
        return self._total_cost >= self.max_budget_usd

    def over_step(self, estimated: float) -> bool:
        return estimated > self.max_per_step_usd

    def over_steps(self) -> bool:
        return self._steps_completed >= self.max_steps

    def budget_remaining(self) -> float:
        return max(0.0, self.max_budget_usd - self._total_cost)

    def pct_budget_used(self) -> float:
        if self.max_budget_usd == 0:
            return 100.0
        return (self._total_cost / self.max_budget_usd) * 100

    def steps_remaining(self) -> int:
        return max(0, self.max_steps - self._steps_completed)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def summary(self) -> dict:
        elapsed = time.time() - self._run_start
        return {
            "total_cost_usd": round(self._total_cost, 6),
            "budget_remaining_usd": round(self.budget_remaining(), 6),
            "pct_budget_used": round(self.pct_budget_used(), 2),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "steps_completed": self._steps_completed,
            "steps_remaining": self.steps_remaining(),
            "elapsed_seconds": round(elapsed, 2),
            "avg_cost_per_step": round(
                self._total_cost / max(1, self._steps_completed), 6
            ),
            "calls_by_role": self._calls_by_role(),
        }

    def _calls_by_role(self) -> dict:
        out: dict = {}
        for c in self._history:
            out.setdefault(c.role, {"calls": 0, "cost_usd": 0.0})
            out[c.role]["calls"] += 1
            out[c.role]["cost_usd"] = round(out[c.role]["cost_usd"] + c.cost_usd, 6)
        return out

    @property
    def history(self) -> List[StepCost]:
        return list(self._history)
