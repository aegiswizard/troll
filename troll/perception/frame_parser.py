"""
troll/perception/frame_parser.py

Converts raw ARC observations into structured, LLM-digestible state packs.
The agent never reads raw history — it reads distilled state packs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from troll.env.arc_adapter import Observation


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GridObject:
    id: str
    color: int
    positions: List[Tuple[int, int]]   # (row, col)
    bounding_box: Tuple[int, int, int, int]  # r0, c0, r1, c1
    area: int


@dataclass
class FrameDelta:
    cells_changed: int
    appeared: List[Tuple[int, int]]   # (row, col)
    disappeared: List[Tuple[int, int]]
    color_changes: List[Tuple[int, int, int, int]]  # row, col, old_color, new_color


@dataclass
class StatePack:
    """
    Everything the reasoning council needs for one step.
    This is the primary cognitive unit in Troll.
    """
    step: int
    grid: Optional[List[List[int]]]          # current grid (if ARC grid-type)
    grid_shape: Optional[Tuple[int, int]]
    objects: List[GridObject]
    delta: Optional[FrameDelta]              # vs previous step
    action_space: List[Any]
    reward: float
    done: bool
    info: Dict[str, Any]

    # Narrative summaries (filled by Summarizer)
    current_summary: str = ""
    delta_summary: str = ""
    hypothesis_hints: List[str] = field(default_factory=list)
    progress_estimate: float = 0.0
    memory_recap: str = ""

    def to_prompt_block(self) -> str:
        """Render state pack as a compact LLM-readable block."""
        lines = [
            f"=== STATE PACK (step {self.step}) ===",
            f"Grid shape: {self.grid_shape}",
            f"Reward: {self.reward}  Done: {self.done}",
            f"Actions available: {len(self.action_space)}",
            "",
            f"[Current] {self.current_summary}",
            f"[Delta]   {self.delta_summary}",
        ]
        if self.hypothesis_hints:
            lines.append(f"[Hints]   {'; '.join(self.hypothesis_hints)}")
        if self.memory_recap:
            lines.append(f"[Memory]  {self.memory_recap}")
        lines.append(f"[Progress estimate] {self.progress_estimate:.0%}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Frame parser
# ---------------------------------------------------------------------------

class FrameParser:
    """
    Parses raw ARC observations into StatePack objects.
    Handles both grid-based and non-grid observations gracefully.
    """

    def __init__(self) -> None:
        self._prev_grid: Optional[np.ndarray] = None

    def parse(self, obs: Observation) -> StatePack:
        grid, shape = self._extract_grid(obs.raw)
        objects = self._extract_objects(grid) if grid is not None else []
        delta = self._compute_delta(grid) if grid is not None else None

        if grid is not None:
            self._prev_grid = grid

        return StatePack(
            step=obs.frame_id,
            grid=grid.tolist() if grid is not None else None,
            grid_shape=shape,
            objects=objects,
            delta=delta,
            action_space=obs.action_space,
            reward=obs.reward,
            done=obs.done,
            info=obs.info,
        )

    def reset(self) -> None:
        self._prev_grid = None

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _extract_grid(self, raw: Any) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Try to extract a 2-D integer grid from various observation formats."""
        if raw is None:
            return None, None

        # Already a numpy array
        if isinstance(raw, np.ndarray):
            if raw.ndim == 2:
                return raw.astype(int), (raw.shape[0], raw.shape[1])
            if raw.ndim == 3:          # H×W×C → take channel 0 or argmax
                g = raw[:, :, 0].astype(int)
                return g, (g.shape[0], g.shape[1])

        # List of lists
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            try:
                g = np.array(raw, dtype=int)
                return g, (g.shape[0], g.shape[1])
            except Exception:
                pass

        # Dict with "grid" key (common ARC format)
        if isinstance(raw, dict):
            for key in ("grid", "observation", "state", "board"):
                if key in raw:
                    return self._extract_grid(raw[key])

        # JSON string
        if isinstance(raw, str):
            try:
                return self._extract_grid(json.loads(raw))
            except Exception:
                pass

        return None, None

    def _extract_objects(self, grid: np.ndarray) -> List[GridObject]:
        """Simple connected-component object finder by color."""
        objects: List[GridObject] = []
        seen: set = set()
        obj_id = 0

        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                color = int(grid[r, c])
                if color == 0 or (r, c) in seen:
                    continue

                # BFS flood fill
                positions: List[Tuple[int, int]] = []
                queue = [(r, c)]
                while queue:
                    cr, cc = queue.pop()
                    if (cr, cc) in seen:
                        continue
                    if cr < 0 or cr >= grid.shape[0] or cc < 0 or cc >= grid.shape[1]:
                        continue
                    if int(grid[cr, cc]) != color:
                        continue
                    seen.add((cr, cc))
                    positions.append((cr, cc))
                    queue.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])

                if positions:
                    rows = [p[0] for p in positions]
                    cols = [p[1] for p in positions]
                    objects.append(GridObject(
                        id=f"obj_{obj_id}_{color}",
                        color=color,
                        positions=positions,
                        bounding_box=(min(rows), min(cols), max(rows), max(cols)),
                        area=len(positions),
                    ))
                    obj_id += 1

        return objects

    def _compute_delta(self, current: np.ndarray) -> Optional[FrameDelta]:
        if self._prev_grid is None:
            return None
        if self._prev_grid.shape != current.shape:
            return None

        diff = current != self._prev_grid
        changed_positions = list(zip(*np.where(diff)))

        appeared = [(int(r), int(c)) for r, c in changed_positions if self._prev_grid[r, c] == 0]
        disappeared = [(int(r), int(c)) for r, c in changed_positions if current[r, c] == 0]
        color_changes = [
            (int(r), int(c), int(self._prev_grid[r, c]), int(current[r, c]))
            for r, c in changed_positions
            if self._prev_grid[r, c] != 0 and current[r, c] != 0
        ]

        return FrameDelta(
            cells_changed=int(diff.sum()),
            appeared=appeared,
            disappeared=disappeared,
            color_changes=color_changes,
        )


# ---------------------------------------------------------------------------
# Summarizer  (text descriptions → fed to LLM council)
# ---------------------------------------------------------------------------

class Summarizer:
    """Produces human-readable summaries of StatePacks."""

    def summarize(self, pack: StatePack, prev_pack: Optional[StatePack] = None) -> StatePack:
        pack.current_summary = self._describe_current(pack)
        pack.delta_summary = self._describe_delta(pack)
        pack.progress_estimate = self._estimate_progress(pack)
        return pack

    def _describe_current(self, pack: StatePack) -> str:
        if pack.grid_shape is None:
            return f"Non-grid observation. Info keys: {list(pack.info.keys())}"

        h, w = pack.grid_shape
        n_colors = len({obj.color for obj in pack.objects})
        n_objects = len(pack.objects)
        return (
            f"{h}×{w} grid. {n_objects} objects across {n_colors} colors. "
            f"Action space size: {len(pack.action_space)}."
        )

    def _describe_delta(self, pack: StatePack) -> str:
        if pack.delta is None:
            return "No previous frame (first step)."
        d = pack.delta
        if d.cells_changed == 0:
            return "No visible change from last action."
        parts = [f"{d.cells_changed} cells changed."]
        if d.appeared:
            parts.append(f"{len(d.appeared)} new cells appeared.")
        if d.disappeared:
            parts.append(f"{len(d.disappeared)} cells disappeared.")
        if d.color_changes:
            parts.append(f"{len(d.color_changes)} cells changed color.")
        return " ".join(parts)

    def _estimate_progress(self, pack: StatePack) -> float:
        # Heuristic: reward + done signal
        if pack.done and pack.reward > 0:
            return 1.0
        if pack.reward > 0:
            return min(0.9, pack.reward)
        return 0.0
