"""Speed-scene constants and formulas (from sumo_catalog.py).

Ego spawns at the start of the edge; the start sign sits at least one
braking (or accel) approach beyond that. Limits are assigned by a
deterministic round-robin, not at random.
"""
from __future__ import annotations

from typing import Optional

BRAKING_SPAWN_CODES = {"3.24", "5.21", "5.31"}
ACCEL_SPAWN_CODES = {"4.6"}

ACCEL_DEFICIT_KMH = 15.0
ACCEL_V0_FLOOR_KMH = 5.0
ACCEL_APPROACH_M = 20.0
MIN_SPEED_FLOOR_KMH = 35.0

ALLOWED_LIMITS_KMH = (20, 30, 40)
SPEED_LIMIT_TARGETS_KMH = (20, 30, 40)
MIN_SPEED_TARGETS_KMH = (40, 50, 60)

BRAKE_DECEL_MPS2_DEFAULT = 3.5
BRAKE_DELAY_S_DEFAULT = 0.5
BRAKE_MARGIN_M_DEFAULT = 3.0
V0_MIN_EXCESS_MPS_DEFAULT = 2.0
V0_MAX_EXCESS_KMH = 30.0
BRAKE_DIST_FACTOR = 1.0
EGO_MAX_SPAWN_MPS = 60.0 / 3.6

PAIRED_END_CODES = {
    "3.24": "3.25",
    "5.31": "5.32",
    "5.21": "5.22",
}

ZONE_TAIL_M = 8.0
ZONE_MIN_M = 20.0
RESIDENTIAL_LIMIT_KMH = 20.0


def assign_limit_kmh(pdd_code: str, scene_index: int) -> float:
    """Deterministic per-scene limit. Not random.

    3.24 / 5.31: round-robin {20, 30, 40} (exact 1/3 split).
    5.21: always 20.
    4.6: round-robin {40, 50, 60}.
    """
    if pdd_code == "5.21":
        return RESIDENTIAL_LIMIT_KMH
    if pdd_code == "4.6":
        targets = MIN_SPEED_TARGETS_KMH
    else:
        targets = SPEED_LIMIT_TARGETS_KMH
    return float(targets[int(scene_index) % len(targets)])


def spawn_mode_for(pdd_code: str) -> str:
    if pdd_code in ACCEL_SPAWN_CODES:
        return "accel"
    return "brake"


def accel_v0_mps(v_target_kmh: float) -> float:
    v0_kmh = max(ACCEL_V0_FLOOR_KMH, float(v_target_kmh) - ACCEL_DEFICIT_KMH)
    return v0_kmh / 3.6


def braking_v0_mps(seed: int, v_target_kmh: float) -> float:
    """v0 strictly above the limit, capped like the colleague catalog."""
    v_target_mps = float(v_target_kmh) / 3.6
    v0_cap_mps = min(
        EGO_MAX_SPAWN_MPS,
        (float(v_target_kmh) + V0_MAX_EXCESS_KMH) / 3.6,
    )
    try:
        from core.profiles.agent_profile_bank import (
            sample_spawn_velocity_above_limit,
        )
        v0 = float(
            sample_spawn_velocity_above_limit(
                int(seed),
                v_target_mps,
                min_excess=V0_MIN_EXCESS_MPS_DEFAULT,
                max_v=v0_cap_mps,
            )
        )
    except Exception:
        v0 = min(v0_cap_mps, v_target_mps + V0_MIN_EXCESS_MPS_DEFAULT)
    if v0 <= v_target_mps + 1e-6:
        v0 = min(v0_cap_mps, v_target_mps + V0_MIN_EXCESS_MPS_DEFAULT)
    return float(v0)


def braking_d_required_m(v0_mps: float, v_target_kmh: float) -> float:
    v_target_mps = float(v_target_kmh) / 3.6
    try:
        from core.profiles.agent_profile_bank import braking_required_distance
        d_req = braking_required_distance(
            float(v0_mps),
            v_target_mps,
            BRAKE_DECEL_MPS2_DEFAULT,
            BRAKE_DELAY_S_DEFAULT,
            BRAKE_MARGIN_M_DEFAULT,
        )
    except Exception:
        a = BRAKE_DECEL_MPS2_DEFAULT
        d_brake = max(0.0, (v0_mps * v0_mps - v_target_mps * v_target_mps) / (2.0 * a))
        d_req = d_brake + float(v0_mps) * BRAKE_DELAY_S_DEFAULT + BRAKE_MARGIN_M_DEFAULT
    return BRAKE_DIST_FACTOR * float(d_req)


def approach_m(pdd_code: str, v0_mps: float, v_target_kmh: float) -> float:
    if pdd_code in ACCEL_SPAWN_CODES:
        return float(ACCEL_APPROACH_M)
    return braking_d_required_m(v0_mps, v_target_kmh)


def paired_end_code(pdd_code: str) -> Optional[str]:
    return PAIRED_END_CODES.get(str(pdd_code))
