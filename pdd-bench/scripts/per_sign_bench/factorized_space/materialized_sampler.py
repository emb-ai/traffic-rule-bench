"""
Materialized subset sampler.

Samples a practical subset from the factorized benchmark space for
validation and behavioral evaluation.
"""

from __future__ import annotations

import random
from typing import List

from .index_codec import FactorizedSpaceCodec


def stratified_sample(
    codec: FactorizedSpaceCodec,
    n: int = 2000,
    seed: int = 42,
    min_per_cell: int = 2,
) -> List[int]:
    """Stratified sampling across cells, proportional to cell size.

    Ensures at least `min_per_cell` scenes per non-empty cell,
    then distributes remaining budget proportionally.
    """
    rng = random.Random(seed)
    n_cells = codec.cell_count()

    # Allocate budget
    cell_sizes = [codec._cell_sizes[i] for i in range(n_cells)]
    total_space = sum(cell_sizes)

    # Floor allocation
    allocation = [min(min_per_cell, cell_sizes[i]) for i in range(n_cells)]
    remaining = n - sum(allocation)

    if remaining > 0:
        # Proportional allocation of remaining budget
        weights = [cell_sizes[i] for i in range(n_cells)]
        total_w = sum(weights)
        for i in range(n_cells):
            extra = int(remaining * weights[i] / total_w) if total_w > 0 else 0
            extra = min(extra, cell_sizes[i] - allocation[i])
            allocation[i] += max(0, extra)

    # Sample from each cell
    indices = []
    for ci in range(n_cells):
        start, end = codec.cell_range(ci)
        cell_n = allocation[ci]
        if cell_n <= 0 or end <= start:
            continue
        if cell_n >= (end - start):
            indices.extend(range(start, end))
        else:
            indices.extend(rng.sample(range(start, end), cell_n))

    rng.shuffle(indices)
    return indices[:n]


def uniform_sample(
    codec: FactorizedSpaceCodec,
    n: int = 1000,
    seed: int = 42,
) -> List[int]:
    """Uniform random sample from the full space."""
    rng = random.Random(seed)
    total = codec.total_size
    return rng.sample(range(total), min(n, total))


def targeted_sign_sample(
    codec: FactorizedSpaceCodec,
    signs_per_type: int = 40,
    seed: int = 42,
) -> List[int]:
    """Sample ensuring coverage of every sign type on every compatible topology."""
    rng = random.Random(seed)
    indices = []

    for ci in range(codec.cell_count()):
        cell = codec.cells[ci]
        start, end = codec.cell_range(ci)
        if end <= start:
            continue

        n_sa = len(cell.sign_accident_combos)
        from .space_definition import LANE_WIDTH_GRID
        n_lw = len(LANE_WIDTH_GRID)
        n_sl = cell.n_spawn_lanes
        per_sign_accident = n_lw * n_sl

        for sa_idx in range(n_sa):
            sa_start = start + sa_idx * per_sign_accident
            sa_end = sa_start + per_sign_accident
            k = min(2, sa_end - sa_start)
            indices.extend(rng.sample(range(sa_start, sa_end), k))

    rng.shuffle(indices)
    return indices


def sample_for_sign(
    codec: FactorizedSpaceCodec,
    sign_key: str,
    n: int = 250,
    seed: int = 42,
) -> List[int]:
    """Return up to `n` flat_index values that materialize to sign_type=sign_key.

    Uses codec.indices_for_sign to get the pre-computed ranges, then samples
    uniformly without replacement. If fewer than `n` are available, returns
    all of them. Quality > quantity (§17): never duplicates or hacks.
    """
    rng = random.Random(seed)
    candidates = codec.indices_for_sign(sign_key)
    if n >= len(candidates):
        rng.shuffle(candidates)
        return candidates
    return rng.sample(candidates, n)
