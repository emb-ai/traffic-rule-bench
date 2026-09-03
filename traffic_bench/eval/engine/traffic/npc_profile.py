"""NPC profile embedding + aux density credit for manifest rows."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from traffic_bench.eval.engine.traffic.traffic_density_levels import (
    META_DENSITY_CAP,
    META_DENSITY_SCALE,
)


def aux_traffic_vehicle_credit(entry: dict) -> int:
    """Vehicles spawned as auxiliary convoy (count toward scene density budget)."""
    if not entry.get("auxiliary_agent"):
        return 0
    convoy = int(entry.get("aux_convoy_size") or 1)
    lanes = int(entry.get("aux_lanes_occupied") or 1)
    occupied = entry.get("aux_occupied_lane_keys") or []
    if occupied:
        lanes = max(lanes, len(occupied))
    return convoy * lanes


def _profile_nuplan_count(profile: Dict[str, Any]) -> float:
    """Vehicles/frame sampled from nuPlan (prefer raw count over MetaDrive density)."""
    raw = profile.get("nuplan_vehicles_per_frame")
    if raw is not None:
        return float(raw)
    return float(profile.get("traffic_density", 0.0) or 0.0) * META_DENSITY_SCALE


def embed_npc_profile(
    entry: Dict[str, Any],
    profile: Dict[str, Any],
    *,
    apply_aux_credit: bool = False,
    density_cap: float = META_DENSITY_CAP,
) -> Dict[str, Any]:
    """Stamp NPC fields with a single contract for every sign.

    Always written (aux signs and non-aux alike):
      - ``nuplan_vehicles_per_frame`` / ``profile_*`` — raw nuPlan sample (stats)
      - ``aux_traffic_credit`` — reserved aux vehicles (0 if no aux / credit off)
      - ``background_npc_count`` — max(0, N − aux)
      - ``traffic_density`` — background only (what SumoTrafficManager spawns)

    Stats: total planned NPC ≈ nuplan (or aux + background).
    Spawn/env: always read ``traffic_density`` (never ``profile_traffic_density``).
    Signs without aux keep credit=0 so remaining == N — no special cases later.
    """
    out = dict(entry)
    for key, value in profile.items():
        # Density / count are set explicitly below (raw vs spawn-adjusted).
        if key in {"traffic_density", "nuplan_vehicles_per_frame"}:
            continue
        out[f"profile_{key}"] = value

    nuplan_target = _profile_nuplan_count(profile)
    # The profile already carries a density drawn from the curve measured on
    # these scenes. Re-deriving it from nuplan_target / SCALE would discard that
    # and hand every scene back to a divisor no measurement supports, so take
    # what the profile sampled and only fall back to the scale when a profile
    # was built without one.
    raw_density = profile.get("traffic_density")
    if raw_density is None:
        raw_density = nuplan_target / META_DENSITY_SCALE
    sampled_density = round(float(np.clip(float(raw_density), 0.0, density_cap)), 4)

    aux_credit = aux_traffic_vehicle_credit(out) if apply_aux_credit else 0
    remaining = max(0.0, nuplan_target - float(aux_credit))
    # The convoy is reserved in vehicles while the spawn knob is a density, so
    # the reservation is applied as the share of the target it consumes rather
    # than converted through the scale a second time.
    share = (remaining / nuplan_target) if nuplan_target > 0.0 else 1.0
    background_density = round(
        float(np.clip(sampled_density * share, 0.0, density_cap)), 4
    )

    # Raw sample — immutable for stats / nuPlan matching across all signs.
    out["nuplan_vehicles_per_frame"] = round(nuplan_target, 4)
    out["profile_nuplan_vehicles_per_frame"] = round(nuplan_target, 4)
    out["profile_traffic_density"] = sampled_density
    # Decomposition (always present; credit is 0 when aux unused).
    out["aux_traffic_credit"] = int(aux_credit)
    out["background_npc_count"] = round(remaining, 4)
    # Spawn control only — remaining background after aux reservation.
    out["traffic_density"] = background_density

    horizon = profile.get("horizon_steps")
    if horizon is not None:
        out["horizon"] = int(horizon)
        out["horizon_steps"] = int(horizon)
        out["profile_horizon_steps"] = int(horizon)
    return out
