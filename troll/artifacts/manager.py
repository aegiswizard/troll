"""
troll/artifacts/manager.py

Scorecard, checkpoint, and report generation.
Every run emits a full audit trail:
  - manifest.json
  - scorecard.json
  - action_timeline.jsonl
  - cost_summary.json
  - report.md
  - checkpoint/  (resumable)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from troll.core.budgets import BudgetTracker
from troll.memory.store import EpisodicMemory, HypothesisMemory, RunMetadata


# ---------------------------------------------------------------------------
# Paths helper
# ---------------------------------------------------------------------------

def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

class Scorecard:
    """Tracks per-run score and final outcome, ARC-compatible."""

    def __init__(self, run_id: str, env_id: str, artifacts_dir: Path) -> None:
        self.run_id = run_id
        self.env_id = env_id
        self._dir = _ensure(artifacts_dir / "scorecards")
        self._path = self._dir / f"{run_id}.json"

        self.data: Dict[str, Any] = {
            "run_id": run_id,
            "env_id": env_id,
            "created_at": time.time(),
            "steps": [],
            "final_score": None,
            "success": False,
            "tags": [],
            "source_url": "https://github.com/aegiswizard/troll",
            "version": "1.0.0",
            "meta": {},
        }
        self._save()

    def record_step(self, step: int, reward: float, done: bool, action: Any, notes: str = "") -> None:
        self.data["steps"].append({
            "step": step,
            "reward": reward,
            "done": done,
            "action": action,
            "notes": notes,
            "ts": time.time(),
        })

    def finalise(self, score: float, success: bool, meta: Optional[Dict] = None) -> None:
        self.data["final_score"] = score
        self.data["success"] = success
        self.data["finished_at"] = time.time()
        if meta:
            self.data["meta"].update(meta)
        self._save()

    def add_tag(self, tag: str) -> None:
        self.data["tags"].append(tag)

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self.data, f, indent=2)

    @property
    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Save / load resumable run state."""

    def __init__(self, run_id: str, checkpoints_dir: Path) -> None:
        self._dir = _ensure(checkpoints_dir / run_id)
        self._latest = self._dir / "latest.json"

    def save(
        self,
        step: int,
        episodic: EpisodicMemory,
        hypotheses: HypothesisMemory,
        budget: BudgetTracker,
        extra: Optional[Dict] = None,
    ) -> Path:
        data = {
            "step": step,
            "saved_at": time.time(),
            "episodic": episodic.to_dict_list(),
            "hypotheses": hypotheses.to_dict_list(),
            "budget": budget.summary(),
            "extra": extra or {},
        }
        path = self._dir / f"step_{step:05d}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        # Also write latest pointer
        with open(self._latest, "w") as f:
            json.dump({"step": step, "path": str(path)}, f)
        return path

    def load_latest(self) -> Optional[Dict]:
        if not self._latest.exists():
            return None
        with open(self._latest) as f:
            pointer = json.load(f)
        cp_path = Path(pointer["path"])
        if not cp_path.exists():
            return None
        with open(cp_path) as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class Reporter:
    """Generates human-readable + machine-readable run reports."""

    def __init__(self, run_id: str, reports_dir: Path) -> None:
        self._run_id = run_id
        self._dir = _ensure(reports_dir)

    def write(
        self,
        run_meta: RunMetadata,
        scorecard: Scorecard,
        budget: BudgetTracker,
        episodic: EpisodicMemory,
        hypotheses: HypothesisMemory,
        search_stats: Dict[str, Any],
    ) -> Path:
        md = self._build_markdown(run_meta, scorecard, budget, episodic, hypotheses, search_stats)
        path = self._dir / f"{self._run_id}_report.md"
        with open(path, "w") as f:
            f.write(md)

        # also write JSON summary
        summary = {
            "run_id": self._run_id,
            "env_id": run_meta.env_id,
            "provider": run_meta.provider,
            "model": run_meta.model_id,
            "mode": run_meta.mode,
            "score": scorecard.data.get("final_score"),
            "success": scorecard.data.get("success"),
            "budget": budget.summary(),
            "search": search_stats,
            "elapsed_s": run_meta.elapsed_s,
            "tags": run_meta.tags,
        }
        json_path = self._dir / f"{self._run_id}_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        return path

    def _build_markdown(
        self,
        run_meta: RunMetadata,
        scorecard: Scorecard,
        budget: BudgetTracker,
        episodic: EpisodicMemory,
        hypotheses: HypothesisMemory,
        search_stats: Dict[str, Any],
    ) -> str:
        b = budget.summary()
        score = scorecard.data.get("final_score", "N/A")
        success = "✅ SUCCESS" if scorecard.data.get("success") else "❌ DID NOT COMPLETE"

        lines = [
            f"# Troll 🧌 Run Report",
            f"",
            f"> Made with ❤️ by Aegis Wizard 🧙‍♂️",
            f"",
            f"## Result: {success}",
            f"",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Run ID | `{run_meta.run_id}` |",
            f"| Environment | `{run_meta.env_id}` |",
            f"| Provider | {run_meta.provider} |",
            f"| Model | `{run_meta.model_id}` |",
            f"| Mode | {run_meta.mode} |",
            f"| Final Score | **{score}** |",
            f"| Steps | {b['steps_completed']} / {run_meta.max_steps} |",
            f"| Total Cost | ${b['total_cost_usd']:.4f} |",
            f"| Elapsed | {run_meta.elapsed_s}s |",
            f"",
            f"## Budget Breakdown",
            f"",
            f"| Role | Calls | Cost USD |",
            f"|------|-------|----------|",
        ]
        for role, info in b.get("calls_by_role", {}).items():
            lines.append(f"| {role} | {info['calls']} | ${info['cost_usd']:.4f} |")

        lines += [
            f"",
            f"## Search Stats",
            f"",
            f"- Mode: {search_stats.get('mode')}",
            f"- Unique actions tried: {search_stats.get('unique_actions_tried')}",
            f"- Unique states visited: {search_stats.get('unique_states_visited')}",
            f"",
            f"## Top Hypotheses",
            f"",
            hypotheses.recap(),
            f"",
            f"## Action Timeline (last 20 steps)",
            f"",
            f"```",
            episodic.recap(20),
            f"```",
            f"",
            f"## Reproducibility",
            f"",
            f"To reproduce this run exactly:",
            f"```bash",
            f"troll run --env {run_meta.env_id} --provider {run_meta.provider} \\",
            f"    --model-id {run_meta.model_id} --mode {run_meta.mode} \\",
            f"    --seed {run_meta.seed or 'None'}",
            f"```",
            f"",
            f"Scorecard: `{scorecard.path}`",
            f"",
            f"---",
            f"*Generated by Troll v{run_meta.version} — github.com/aegiswizard/troll*",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(run_id: str, artifacts_dir: Path, meta: Dict[str, Any]) -> Path:
    path = _ensure(artifacts_dir) / f"{run_id}_manifest.json"
    meta["run_id"] = run_id
    meta["generated_at"] = time.time()
    meta["troll_version"] = "1.0.0"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path
