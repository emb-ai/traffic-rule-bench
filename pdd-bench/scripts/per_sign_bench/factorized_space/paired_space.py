"""
Factorized space for "paired" sign scenes — one scene with BOTH a zone-start
sign and a zone-end sign, separated by a `zone_length` along a linear segment
(§12 of the plan).

Topologies restricted to S (Straight) and C (Curve) since paired signs need a
contiguous segment with a measurable zone length. Each paired scene counts
toward BOTH PDD codes' per-sign manifests.

Axes inside a paired cell:
  sign_bundle (pair_id)                   — from PAIRED_SIGN_BUNDLES
  lane_num ∈ [2, 3, 4, 5]                 — bundle.min_lane_num constrains
  spawn_lane_semantic (dedup per lane_num)
  zone_length ∈ ZONE_LENGTH_BINS
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import List, Tuple

from .space_definition import (
    LANE_NUM_GRID,
    LANE_WIDTH_GRID,
    ZONE_LENGTH_BINS,
    PAIRED_SIGN_BUNDLES,
    PairedSignBundle,
    SPAWN_LANE_SEMANTIC,
    effective_spawn_lanes,
    resolve_spawn_lane,
)

# Paired scenes live on linear topologies only.
PAIRED_TOPOLOGIES: List[Tuple[str, str, str, int, str]] = [
    # (block_id, route_intent, map_string, socket_index, block_name)
    ("S", "through", "SS", 0, "Straight"),
    ("C", "through", "SC", 0, "Curve"),
]


@dataclass
class PairedCell:
    """One cell = (topology_idx, lane_num). Contains compatible bundle indices."""
    topology_idx: int
    lane_num: int
    bundle_ids: List[int]          # indices into PAIRED_SIGN_BUNDLES that fit this cell
    n_spawn_lanes: int

    def cell_size(self) -> int:
        # bundles × lane_widths × spawn_lanes × zone_lengths
        return (
            len(self.bundle_ids)
            * len(LANE_WIDTH_GRID)
            * self.n_spawn_lanes
            * len(ZONE_LENGTH_BINS)
        )


def build_paired_cells() -> List[PairedCell]:
    cells = []
    for ti, (block_id, _route_intent, _map_str, _sock, _name) in enumerate(PAIRED_TOPOLOGIES):
        for ln in LANE_NUM_GRID:
            eligible = []
            for bi, bundle in enumerate(PAIRED_SIGN_BUNDLES):
                if bundle.min_lane_num > ln:
                    continue
                eligible.append(bi)
            if not eligible:
                continue
            n_sl = len(effective_spawn_lanes(ln))
            cells.append(PairedCell(
                topology_idx=ti,
                lane_num=ln,
                bundle_ids=eligible,
                n_spawn_lanes=n_sl,
            ))
    return cells


class PairedSpaceCodec:
    """Bijective codec for paired-scene flat indices.

    decode(flat_index) → spec dict with both `sign_type_start` and
    `sign_type_end`, plus `pdd_code_start`, `pdd_code_end` for bookkeeping.
    """

    def __init__(self):
        self.cells = build_paired_cells()
        self._cell_sizes = [c.cell_size() for c in self.cells]
        self._cum_offsets = []
        running = 0
        for s in self._cell_sizes:
            self._cum_offsets.append(running)
            running += s
        self._total = running
        self._cell_spawn_lanes = [effective_spawn_lanes(c.lane_num) for c in self.cells]

    @property
    def total_size(self) -> int:
        return self._total

    def cell_count(self) -> int:
        return len(self.cells)

    def cell_range(self, cell_idx: int) -> Tuple[int, int]:
        start = self._cum_offsets[cell_idx]
        return start, start + self._cell_sizes[cell_idx]

    def decode(self, flat_index: int) -> dict:
        if flat_index < 0 or flat_index >= self._total:
            raise IndexError(f"flat_index {flat_index} out of range [0, {self._total})")

        cell_idx = bisect.bisect_right(self._cum_offsets, flat_index) - 1
        remainder = flat_index - self._cum_offsets[cell_idx]

        cell = self.cells[cell_idx]
        topology = PAIRED_TOPOLOGIES[cell.topology_idx]
        block_id, route_intent, map_str, sock, block_name = topology
        spawn_lanes = self._cell_spawn_lanes[cell_idx]

        n_b = len(cell.bundle_ids)
        n_lw = len(LANE_WIDTH_GRID)
        n_sl = len(spawn_lanes)
        n_zl = len(ZONE_LENGTH_BINS)

        # Mixed-radix order: bundle, lane_width, spawn_lane, zone_length
        zl_idx = remainder % n_zl
        remainder //= n_zl
        sl_idx = remainder % n_sl
        remainder //= n_sl
        lw_idx = remainder % n_lw
        remainder //= n_lw
        b_idx = remainder  # < n_b

        bundle_id = cell.bundle_ids[b_idx]
        bundle: PairedSignBundle = PAIRED_SIGN_BUNDLES[bundle_id]
        lane_width = LANE_WIDTH_GRID[lw_idx]
        spawn_lane_semantic = spawn_lanes[sl_idx]
        spawn_lane_actual = resolve_spawn_lane(spawn_lane_semantic, cell.lane_num)
        zone_length = ZONE_LENGTH_BINS[zl_idx]

        # Deterministic seed distinct from single-sign codec (offset +200_000
        # avoids collision with FactorizedSpaceCodec's 100_000 + flat_index).
        seed = 200_000 + flat_index

        return {
            "paired_flat_index": flat_index,
            "paired_cell_index": cell_idx,
            "bundle_id": bundle_id,
            "pdd_code_start": bundle.pdd_a,
            "pdd_code_end": bundle.pdd_b,
            "sign_type_start": bundle.sign_key_a,
            "sign_type_end": bundle.sign_key_b,
            "zone_length_m": zone_length,
            "block_id": block_id,
            "route_intent": route_intent,
            "block_sequence": map_str,
            "socket_index": sock,
            "block_name": block_name,
            "lane_num": cell.lane_num,
            "lane_width": lane_width,
            "spawn_lane_semantic": spawn_lane_semantic,
            "spawn_lane_index": spawn_lane_actual,
            "seed": seed,
        }

    def indices_for_pdd(self, pdd_code: str) -> List[int]:
        """Return flat_index values where either pdd_code_start or pdd_code_end equals pdd_code."""
        result = []
        for ci, cell in enumerate(self.cells):
            start = self._cum_offsets[ci]
            n_lw = len(LANE_WIDTH_GRID)
            n_sl = self._cell_spawn_lanes[ci].__len__()
            n_zl = len(ZONE_LENGTH_BINS)
            per_bundle = n_lw * n_sl * n_zl
            for b_pos, bundle_id in enumerate(cell.bundle_ids):
                bundle = PAIRED_SIGN_BUNDLES[bundle_id]
                if bundle.pdd_a != pdd_code and bundle.pdd_b != pdd_code:
                    continue
                b_start = start + b_pos * per_bundle
                b_end = b_start + per_bundle
                result.extend(range(b_start, b_end))
        return result


if __name__ == "__main__":
    codec = PairedSpaceCodec()
    print(f"Paired space total: {codec.total_size:,} base scenes across {codec.cell_count()} cells")
    for pdd in ("2.1", "2.2", "3.24", "3.25", "5.31", "5.32", "5.14.1", "5.14.3"):
        idxs = codec.indices_for_pdd(pdd)
        print(f"  PDD {pdd}: {len(idxs)} paired base scenes")
    sample = codec.decode(0)
    print(f"\nFirst paired scene:")
    for k, v in sample.items():
        print(f"  {k}: {v}")
