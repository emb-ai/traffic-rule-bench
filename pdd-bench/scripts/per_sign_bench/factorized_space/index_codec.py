"""
Bijective flat-index ↔ scene-spec codec for the factorized benchmark space.

The space has variable-size cells because sign_accident_combos differ per
(template_id, lane_num). Uses a two-level index:
  Level 1: binary search over cell cumulative offsets
  Level 2: uniform mixed-radix decomposition within the cell

spawn_velocity is NOT part of flat_index anymore — each base scene expands
into N_SPAWN_VELOCITY_SAMPLES rows at materialization with velocities drawn
from the nuPlan initial_speed distribution.
"""

from __future__ import annotations

import bisect
from typing import List, Optional

from .space_definition import (
    TOPOLOGY_TEMPLATES,
    LANE_NUM_GRID,
    LANE_WIDTH_GRID,
    CellInfo,
    build_cells,
    effective_spawn_lanes,
    resolve_spawn_lane,
)


class FactorizedSpaceCodec:
    """Bijective map between flat index in [0, N) and base-scene spec dict.

    Agent profile is NOT an axis — sampled per-scene from nuPlan using
    seed = 100_000 + flat_index in the runner, then applied to NPC only
    (ego uses DEFAULT_EGO_PARAMS).

    spawn_velocity is NOT an axis — sampled N_SPAWN_VELOCITY_SAMPLES times
    per base-scene at materialization.
    """

    def __init__(self):
        self.cells = build_cells()

        self._cell_sizes = [c.cell_size() for c in self.cells]
        self._cum_offsets = []
        running = 0
        for s in self._cell_sizes:
            self._cum_offsets.append(running)
            running += s
        self._total = running

        self._cell_spawn_lanes = []
        for c in self.cells:
            self._cell_spawn_lanes.append(effective_spawn_lanes(c.lane_num))

    @property
    def total_size(self) -> int:
        return self._total

    def decode(self, flat_index: int) -> dict:
        """Decode flat_index in [0, total_size) to a base-scene spec dict.

        Does NOT include spawn_velocity_ms — that is added per materialization
        row by the runner after drawing from nuPlan.
        """
        if flat_index < 0 or flat_index >= self._total:
            raise IndexError(f"flat_index {flat_index} out of range [0, {self._total})")

        cell_idx = bisect.bisect_right(self._cum_offsets, flat_index) - 1
        remainder = flat_index - self._cum_offsets[cell_idx]

        cell = self.cells[cell_idx]
        tmpl = TOPOLOGY_TEMPLATES[cell.template_id]
        spawn_lanes = self._cell_spawn_lanes[cell_idx]
        combos = cell.sign_accident_combos

        # Mixed-radix order: sign_accident, lane_width, spawn_lane
        n_sa = len(combos)
        n_lw = len(LANE_WIDTH_GRID)
        n_sl = len(spawn_lanes)

        sl_idx = remainder % n_sl
        remainder //= n_sl
        lw_idx = remainder % n_lw
        remainder //= n_lw
        sa_idx = remainder  # should be < n_sa

        sign_key, accident_prob = combos[sa_idx]
        lane_width = LANE_WIDTH_GRID[lw_idx]
        spawn_lane_semantic = spawn_lanes[sl_idx]
        spawn_lane_actual = resolve_spawn_lane(spawn_lane_semantic, cell.lane_num)

        seed = 100_000 + flat_index

        return {
            "flat_index": flat_index,
            "cell_index": cell_idx,
            "template_id": cell.template_id,
            "block_id": tmpl.block_id,
            "route_intent": tmpl.route_intent,
            "block_sequence": tmpl.map_string,
            "socket_index": tmpl.socket_index,
            "block_name": tmpl.block_name,
            "lane_num": cell.lane_num,
            "lane_width": lane_width,
            "spawn_lane_semantic": spawn_lane_semantic,
            "spawn_lane_index": spawn_lane_actual,
            "accident_prob": accident_prob,
            "sign_type": sign_key,
            "seed": seed,
        }

    def encode(self, spec: dict) -> int:
        """Encode a base-scene spec back to flat_index. Inverse of decode.

        Ignores spawn_velocity_ms (not part of the index anymore).
        """
        tmpl_id = spec["template_id"]
        lane_num = spec["lane_num"]

        cell_idx = None
        for ci, c in enumerate(self.cells):
            if c.template_id == tmpl_id and c.lane_num == lane_num:
                cell_idx = ci
                break
        if cell_idx is None:
            raise ValueError(f"No cell for template_id={tmpl_id}, lane_num={lane_num}")

        cell = self.cells[cell_idx]
        spawn_lanes = self._cell_spawn_lanes[cell_idx]
        combos = cell.sign_accident_combos

        sa_key = (spec["sign_type"], spec["accident_prob"])
        sa_idx = combos.index(sa_key)
        lw_idx = LANE_WIDTH_GRID.index(spec["lane_width"])
        sl_idx = spawn_lanes.index(spec["spawn_lane_semantic"])

        n_sl = len(spawn_lanes)
        n_lw = len(LANE_WIDTH_GRID)

        within_cell = (sa_idx * n_lw + lw_idx) * n_sl + sl_idx
        return self._cum_offsets[cell_idx] + within_cell

    def decode_batch(self, indices: list[int]) -> list[dict]:
        return [self.decode(i) for i in indices]

    def cell_range(self, cell_idx: int) -> tuple[int, int]:
        """Return (start, end) flat indices for a cell."""
        start = self._cum_offsets[cell_idx]
        end = start + self._cell_sizes[cell_idx]
        return start, end

    def cell_count(self) -> int:
        return len(self.cells)

    # ------------------------------------------------------------------
    # Targeted filtering by sign key (§5)
    # ------------------------------------------------------------------

    def indices_for_sign(self, sign_key: str) -> List[int]:
        """Return all flat_index values that map to a spec with sign_type=sign_key.

        Computed directly from the mixed-radix layout (O(n_cells), no full
        decode pass). Handy for building per-sign manifests.
        """
        result: List[int] = []
        for ci, cell in enumerate(self.cells):
            start = self._cum_offsets[ci]
            combos = cell.sign_accident_combos
            n_sa = len(combos)
            n_lw = len(LANE_WIDTH_GRID)
            n_sl = len(self._cell_spawn_lanes[ci])
            per_sa = n_lw * n_sl

            for sa_idx, (sk, _ap) in enumerate(combos):
                if sk != sign_key:
                    continue
                sa_start = start + sa_idx * per_sa
                sa_end = sa_start + per_sa
                result.extend(range(sa_start, sa_end))
        return result
