"""Speed-scene constants and formulas (from sumo_catalog.py).

Ego spawns at the start of the edge; the start sign sits at least one
braking (or accel) approach beyond that. Plate values are assigned by a
deterministic round-robin, never snapped to the road's own speed: the
nearest-snap this replaced put 85% of ceiling scenes on 40 km/h, which is
the base policies' cruise, so compliance came for free. The road speed is
still read for 4.6, but only to decide which minima the road can deliver.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional

BRAKING_SPAWN_CODES = {"3.24", "5.21", "5.31"}
ACCEL_SPAWN_CODES = {"4.6"}

ACCEL_DEFICIT_KMH = 15.0
# The deficit is sampled per seed inside this band (mean stays at the 15 above),
# so the profile axis varies the approach for 4.6 as it does for the braking
# families. The band is deliberately narrow: ACCEL_APPROACH_M is a fixed 20 m
# and the zone starts 10 m past the sign, so a deeper deficit would change how
# hard the scene is, not just how varied.
ACCEL_DEFICIT_MIN_KMH = 12.0
ACCEL_DEFICIT_MAX_KMH = 18.0
ACCEL_V0_FLOOR_KMH = 5.0
# Run-up from the spawn to the 4.6 plate; the zone opens 10 m past the plate.
# The ego starts 12-18 km/h under the minimum and must be at it when the zone
# opens. Over the 30 m the old 20 m gave, that needs 1.3 m/s^2 for a 40 plate
# and 2.0 m/s^2 for a 60 plate -- above the comfortable IDM acceleration, so
# the rule expert only made it by flooring the throttle regardless of the car
# ahead, and once that was stopped every episode opened the zone in breach,
# traffic or no traffic. 50 m (60 m to the zone) brings the requirement down
# to 0.6-1.0 m/s^2, which a safe approach can deliver.
ACCEL_APPROACH_M = 50.0
MIN_SPEED_FLOOR_KMH = 35.0

# Share of the background cars that honour the plate, drawn per sampled
# variant from this range (the nominal variant carries no traffic). With every
# car obeying, a sign-blind policy passes the zone by trailing the car ahead;
# a few cars ignoring the plate take that shortcut away while the scene stays
# coherent. Speed families only -- detour traffic must still leave the closed
# lane, that is physics rather than a rule.
NPC_COMPLIANCE_RANGE = (0.5, 1.0)

SPEED_LIMIT_TARGETS_KMH = (20, 30, 40)
MIN_SPEED_TARGETS_KMH = (40, 50, 60)
# 3.24/5.31 geometry does not suit a 20-40 plate above this road speed.
MOTORWAY_DROP_KMH = 80.0
# A 4.6 minimum must sit this far below the road's own speed to be reachable
# before the zone starts.
MIN_SPEED_ROAD_MARGIN_KMH = 10.0

BRAKE_DECEL_MPS2_DEFAULT = 3.5
BRAKE_DELAY_S_DEFAULT = 0.5
BRAKE_MARGIN_M_DEFAULT = 3.0
V0_MIN_EXCESS_MPS_DEFAULT = 2.0
V0_MAX_EXCESS_KMH = 30.0
BRAKE_DIST_FACTOR = 1.0

PAIRED_END_CODES = {
    "3.24": "3.25",
    "5.31": "5.32",
    "5.21": "5.22",
}

ZONE_TAIL_M = 8.0
ZONE_MIN_M = 20.0
RESIDENTIAL_LIMIT_KMH = 20.0


def edge_speed_mps(net_path: str, road_id: str) -> float:
    """Max lane ``speed`` (m/s) of the named edge in a SUMO ``*.net.xml``, or 0.0.

    This is the value SpeedLimitSign reads from lane.speed; x3.6 gives km/h.
    """
    try:
        root = ET.parse(net_path).getroot()
    except (ET.ParseError, OSError):
        return 0.0
    for edge in root.findall("edge"):
        if edge.get("id") != road_id:
            continue
        speeds = [float(l.get("speed")) for l in edge.findall("lane") if l.get("speed")]
        if speeds:
            return max(speeds)
        s = edge.get("speed")
        return float(s) if s else 0.0
    return 0.0


def new_limit_state() -> Dict[str, Any]:
    """Cross-scene bookkeeping for `assign_limit_kmh`.

    The ceiling round-robin advances only on scenes that survive, so dropping a
    motorway does not punch a hole in the 1/3 split. Pass one state per family.
    """
    return {"rr": 0, "min_counts": {float(t): 0 for t in MIN_SPEED_TARGETS_KMH}}


def assign_limit_kmh(
    pdd_code: str,
    road_speed_kmh: Optional[float] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Deterministic per-scene plate value. Not random. ``None`` = drop the scene.

    5.21 -- always 20, the value the sign carries by regulation.

    3.24 / 5.31 -- round-robin over {20, 30, 40}, an exact 1/3 split. These are
    deliberately NOT filtered against the road's nominal speed: a plate at or
    above the edge speed looks vacuous, but the nominal edge speed is not the
    speed a policy drives and a zone can span faster edges downstream, so the
    comparison does not test discriminability. Motorways are still dropped.

    4.6 -- the least-filled of the buckets the road can actually deliver
    (plate <= road - MIN_SPEED_ROAD_MARGIN_KMH). A blind round-robin skews the
    split, because 50 and 60 are reachable only on roads >= 60 and >= 70 while
    fast roads get handed 40; spending fast roads on the deficit buckets makes
    the split as uniform as the road pool allows. A scene whose best realistic
    minimum falls below MIN_SPEED_FLOOR_KMH is dropped -- a minimum at or under
    the base policies' cruise is satisfied without reading the sign.
    """
    if pdd_code == "5.21":
        return RESIDENTIAL_LIMIT_KMH

    if state is None:
        state = new_limit_state()
    road = float(road_speed_kmh or 0.0)

    if pdd_code == "4.6":
        if road <= 0.0:
            return None
        counts = state.setdefault(
            "min_counts", {float(t): 0 for t in MIN_SPEED_TARGETS_KMH}
        )
        reachable = [t for t in MIN_SPEED_TARGETS_KMH
                     if t <= road - MIN_SPEED_ROAD_MARGIN_KMH]
        if reachable:
            target = float(min(reachable, key=lambda t: (counts[float(t)], t)))
        else:
            target = road - MIN_SPEED_ROAD_MARGIN_KMH
        if target < MIN_SPEED_FLOOR_KMH:
            return None
        if reachable:
            counts[target] += 1
        return float(target)

    if road > MOTORWAY_DROP_KMH:
        return None
    idx = int(state.get("rr", 0))
    state["rr"] = idx + 1
    return float(SPEED_LIMIT_TARGETS_KMH[idx % len(SPEED_LIMIT_TARGETS_KMH)])



def sample_npc_compliance_rate(seed: int) -> float:
    """Per-row share of plate-abiding NPCs. Drawn from a stream of its own so
    the other per-row draws (density, v0, plate offset) keep their values."""
    import random

    lo, hi = NPC_COMPLIANCE_RANGE
    rng = random.Random(int(seed) ^ 0x6E7063)
    return round(float(rng.uniform(float(lo), float(hi))), 3)

def spawn_mode_for(pdd_code: str) -> str:
    if pdd_code in ACCEL_SPAWN_CODES:
        return "accel"
    return "brake"


def accel_v0_mps(v_target_kmh: float, seed: Optional[int] = None) -> float:
    """v0 below the minimum the 4.6 plate demands, so the ego has to accelerate.

    Without a seed the deficit is the fixed ACCEL_DEFICIT_KMH, which is what the
    call sites used before the profile axis existed. With one it is drawn from
    the nuPlan spread like the braking exceedance, then clamped to the band, so
    the variants of one cell differ in approach and not only in traffic.
    """
    deficit = ACCEL_DEFICIT_KMH
    if seed is not None:
        try:
            from traffic_bench.eval.engine.traffic.agent_profile_bank import (
                sample_spawn_velocity,
            )
            deficit = float(sample_spawn_velocity(int(seed))) * 3.6
        except Exception:
            deficit = ACCEL_DEFICIT_KMH
        deficit = min(ACCEL_DEFICIT_MAX_KMH, max(ACCEL_DEFICIT_MIN_KMH, deficit))
    v0_kmh = max(ACCEL_V0_FLOOR_KMH, float(v_target_kmh) - deficit)
    return v0_kmh / 3.6


def braking_v0_mps(seed: int, v_target_kmh: float) -> float:
    """v0 strictly above the limit, exceeding it by at most V0_MAX_EXCESS_KMH.

    There is no absolute ceiling on top of that. The 60 km/h one this replaced
    bound a single plate -- only 40 km/h reaches limit+30 = 70 -- and it clipped
    exactly the approach speeds that make a ceiling plate demanding.
    """
    v_target_mps = float(v_target_kmh) / 3.6
    v0_cap_mps = (float(v_target_kmh) + V0_MAX_EXCESS_KMH) / 3.6
    try:
        from traffic_bench.eval.engine.traffic.agent_profile_bank import (
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


def nominal_braking_v0_mps(v_target_kmh: float) -> float:
    """The reference approach for a ceiling plate: the limit plus the minimum
    exceedance, with no draw. This is the value braking_v0_mps falls back to
    when the sampler is unavailable, so the nominal row of a scene is the same
    approach the sampled rows are spread around."""
    v_target_mps = float(v_target_kmh) / 3.6
    v0_cap_mps = (float(v_target_kmh) + V0_MAX_EXCESS_KMH) / 3.6
    return float(min(v0_cap_mps, v_target_mps + V0_MIN_EXCESS_MPS_DEFAULT))


def braking_d_required_m(v0_mps: float, v_target_kmh: float) -> float:
    v_target_mps = float(v_target_kmh) / 3.6
    try:
        from traffic_bench.eval.engine.traffic.agent_profile_bank import braking_required_distance
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
