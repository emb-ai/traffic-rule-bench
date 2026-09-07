"""Fixed ego spawn-velocity probes from nuPlan ``routes.initial_speed``.

Default levels are empirical p25/p50/p75 of moving-track initial speeds.
Speed-sign families that need task-conditioned approach (brake / accel relative
to the plate) skip this axis and keep their own ``braking_v0_mps`` /
``accel_v0_mps`` logic.
"""
from __future__ import annotations

from typing import List, Sequence

# routes.csv.gz initial_speed percentiles (m/s), rounded for manifests.
DEFAULT_SPAWN_VELOCITY_LEVELS_MS: tuple[float, ...] = (3.61, 7.75, 11.05)


def list_spawn_velocity_levels_ms(
    levels: Sequence[float] | None = None,
) -> List[float]:
    raw = levels if levels is not None else DEFAULT_SPAWN_VELOCITY_LEVELS_MS
    out = sorted({round(float(x), 4) for x in raw if float(x) >= 0.0})
    return out if out else list(DEFAULT_SPAWN_VELOCITY_LEVELS_MS)


def resolve_spawn_velocity_levels_ms(sim: object | None = None) -> List[float]:
    raw = getattr(sim, "spawn_velocity_levels_ms", None) if sim is not None else None
    return list_spawn_velocity_levels_ms(raw)
