"""
Scene instantiation and validation for the factorized benchmark space.

Accepts decoded specs from the index codec and creates actual MetaDrive
environments with sign placement and route-intent handling.
"""

from __future__ import annotations

import hashlib
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_DIR = SCRIPT_PATH.parent.parent
SCRIPTS_DIR = BENCHMARK_DIR.parent
PDD_BENCH_DIR = SCRIPTS_DIR.parent
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"

for _p in (PDD_BENCH_DIR, METADRIVE_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from envs.traffic_sign_env import TrafficSignEnv
from traffic_signs.stop_sign import StopSign
from traffic_signs.speed_limit_sign import (
    SpeedLimitSign20, SpeedLimitSign30, SpeedLimitSign40, SpeedLimitSign60,
)
from traffic_signs.zone_signs import (
    ZoneSpeedLimitSign20, ZoneSpeedLimitSign30,
    ZoneSpeedLimitSign40, ZoneSpeedLimitSign60,
)
from traffic_signs.min_speed_limit_sign import (
    MinimumSpeedLimit30, MinimumSpeedLimit40, MinimumSpeedLimit50,
)
from traffic_signs.detour_sign import DetourRightSign, DetourLeftSign, DetourEitherSign
from traffic_signs.detour_obstacle import spawn_detour_obstacle
from traffic_signs.restricted_lane_sign import (
    BusLaneRoadSign, BikeLaneRoadSign,
    EndBusLaneRoadSign, EndBikeLaneRoadSign,
    BusLaneSign, BikeLaneSign, EndBusLaneSign, EndBikeLaneSign,
)
from traffic_signs.no_entry_sign import NoEntrySign
from traffic_signs.no_traffic_sign import NoTrafficSign
from traffic_signs.no_stopping_allowed_sign import NoStoppingAllowedSign
from traffic_signs.no_turn_allowed import NoRightTurnSign, NoLeftTurnSign, NoUTurnSign
from traffic_signs.bus_station_sign import BusStationSign
from traffic_signs.only_auto_sign import OnlyAutoSign
from traffic_signs.no_overtaking_sign import NoOvertakingSign
from traffic_signs.right_turn_rule import RightTurnRule
from traffic_signs.pg_direction_sign import PGDirectionSign
from traffic_signs.priority_signs import (
    YieldSign, MainRoadSign, EndMainRoadSign, SecondaryRoadSign,
    SecondaryRoadLeftSign, SecondaryRoadRightSign,
)
from traffic_signs.end_of_zone_signs import (
    EndOfSpeedLimitSign20, EndOfSpeedLimitSign30,
    EndOfSpeedLimitSign40, EndOfSpeedLimitSign60,
    EndOfZoneSpeedLimitSign20, EndOfZoneSpeedLimitSign30,
    EndOfZoneSpeedLimitSign40, EndOfZoneSpeedLimitSign60,
    EndOfAllRestrictionsSign,
)
# 5.13.x — exit-to-restricted-lane signs
from traffic_signs.restricted_lane_sign import (
    ExitToBusLaneSign, ExitToBusLaneSignLeft,
    ExitToBikeLaneSign, ExitToBikeLaneSignLeft,
)

from .space_definition import N_SPAWN_VELOCITY_SAMPLES, BIKE_RELATED_SIGNS
from .agent_profile_bank import sample_one_profile, sample_spawn_velocity
from .ego_defaults import apply_ego_defaults
from .sign_placement import compute_placement, PlacementContext

# Sign key → class mapping
SIGN_CLASS_MAP = {
    "stop": StopSign,
    "speed20": SpeedLimitSign20, "speed30": SpeedLimitSign30,
    "speed40": SpeedLimitSign40, "speed60": SpeedLimitSign60,
    "zone_speed20": ZoneSpeedLimitSign20, "zone_speed30": ZoneSpeedLimitSign30,
    "zone_speed40": ZoneSpeedLimitSign40, "zone_speed60": ZoneSpeedLimitSign60,
    "minspeed30": MinimumSpeedLimit30, "minspeed40": MinimumSpeedLimit40,
    "minspeed50": MinimumSpeedLimit50,
    "no_stop": NoStoppingAllowedSign,
    "bus_station": BusStationSign, "only_auto": OnlyAutoSign,
    "no_entry": NoEntrySign, "no_traffic": NoTrafficSign,
    "bus_lane_road": BusLaneRoadSign, "bike_lane_road": BikeLaneRoadSign,
    "bus_lane": BusLaneSign, "bike_lane": BikeLaneSign,
    "detour_right": DetourRightSign, "detour_left": DetourLeftSign,
    "detour_either": DetourEitherSign,
    "no_right_turn": NoRightTurnSign, "no_left_turn": NoLeftTurnSign,
    "no_uturn": NoUTurnSign,
    "no_overtaking": NoOvertakingSign,
    "right_turn_rule": RightTurnRule,
    "pg_direction": PGDirectionSign,
    "yield": YieldSign,
    "main_road": MainRoadSign,
    "secondary_road": SecondaryRoadSign,
    "secondary_road_left": SecondaryRoadLeftSign,
    "secondary_road_right": SecondaryRoadRightSign,
    "end_main_road": EndMainRoadSign,
    # End-of-zone informational
    "end_speed20": EndOfSpeedLimitSign20,
    "end_speed30": EndOfSpeedLimitSign30,
    "end_speed40": EndOfSpeedLimitSign40,
    "end_speed60": EndOfSpeedLimitSign60,
    "end_zone_speed20": EndOfZoneSpeedLimitSign20,
    "end_zone_speed30": EndOfZoneSpeedLimitSign30,
    "end_zone_speed40": EndOfZoneSpeedLimitSign40,
    "end_zone_speed60": EndOfZoneSpeedLimitSign60,
    "end_all": EndOfAllRestrictionsSign,
    # End-of-restricted-lane standalone (also used in BEGIN_TO_END for paired)
    "end_bus_lane_road": EndBusLaneRoadSign,
    "end_bike_lane_road": EndBikeLaneRoadSign,
    "end_bus_lane": EndBusLaneSign,
    "end_bike_lane": EndBikeLaneSign,
    # Exit-to-restricted-lane (5.13.1-4): intersection approach, right lane
    "exit_to_bus_lane": ExitToBusLaneSign,
    "exit_to_bus_lane_left": ExitToBusLaneSignLeft,
    "exit_to_bike_lane": ExitToBikeLaneSign,
    "exit_to_bike_lane_left": ExitToBikeLaneSignLeft,
}


def _stable_seed(*parts) -> int:
    """Deterministic 32-bit seed from arbitrary parts (SHA-256)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest()[:4], "big")


def _spawn_cyclists_on_lane(env, lane, scene_seed: int, n: int = 3,
                            speed_range=(3.0, 7.0)) -> int:
    """Spawn `n` cyclists spaced along `lane` (used for bike-related scenes).

    Returns how many were successfully spawned. Deterministic given scene_seed.
    Uses MetaDrive's built-in Cyclist class — no custom manager needed.
    """
    try:
        from metadrive.component.traffic_participants.cyclist import Cyclist
    except ImportError:
        return 0
    rng = random.Random(scene_seed ^ 0xB1CE)
    spawned = 0
    lane_len = float(getattr(lane, "length", 0.0))
    if lane_len < 10.0:
        return 0
    for i in range(n):
        long = (i + 1) * lane_len / (n + 1) + rng.uniform(-2.0, 2.0)
        long = max(2.0, min(lane_len - 2.0, long))
        # Cyclist sits on the right edge of the lane
        lateral = float(lane.width_at(long)) / 2 - 0.3
        try:
            pos = lane.position(long, lateral)
            heading = float(lane.heading_theta_at(long))
            speed = rng.uniform(*speed_range)
            c = env.engine.spawn_object(
                Cyclist, position=pos, heading_theta=heading,
                random_seed=scene_seed + i,
            )
            try:
                c.set_velocity([speed, 0.0], in_local_frame=True)
            except Exception:
                pass
            spawned += 1
        except Exception:
            continue
    return spawned

PRIORITY_KEYS = {"main_road", "secondary_road", "secondary_road_left", "secondary_road_right", "yield"}
DETOUR_KEYS = {"detour_right", "detour_left", "detour_either"}
RESTRICTED_BEGIN_KEYS = {"bus_lane_road", "bike_lane_road", "bus_lane", "bike_lane"}
LANE_CHANGE_KEYS = {"no_entry", "no_traffic"}
BEGIN_TO_END = {
    "bus_lane_road": EndBusLaneRoadSign,
    "bike_lane_road": EndBikeLaneRoadSign,
    "bus_lane": EndBusLaneSign,
    "bike_lane": EndBikeLaneSign,
}


def place_pgmap_sign_from_row(env, row: dict, *, seed: int | None = None) -> tuple[bool, str | None]:
    """Place the PG-map sign(s) described by `row` into env.engine.traffic_sign_manager.

    Handles both manifest schemas used in this repo:
      * single-sign rows with `sign_type=<key>`
      * paired rows with `sign_type=None`, `sign_type_start=<begin>`,
        `sign_type_end=<end>`, `zone_length_m=<float>` — places a begin/end
        pair on one lane segment, mirroring _place_pgmap_sign in
        run_benchmark_mini.py.

    Designed to be called from inside TrafficSignEnv.reset() so the manager has
    signs at step 0 — verifier (`check_all_violations`) and reward_function are
    active from t=0, and TopDownRenderer (lazily constructed on first
    env.render()) caches sign icons for GIF rendering.

    Returns (success, failure_reason).
    """
    # ----- paired schema --------------------------------------------------
    if (row.get("sign_type") is None
            and row.get("sign_type_start") and row.get("sign_type_end")):
        if not _override_route_intent(env, row):
            return False, "route_intent_override_failed"
        route_lanes = _get_route_lanes(env)
        if not route_lanes:
            return False, "no_route_lanes"

        sign_start_cls = SIGN_CLASS_MAP.get(row["sign_type_start"])
        sign_end_cls = SIGN_CLASS_MAP.get(row["sign_type_end"])
        if sign_start_cls is None or sign_end_cls is None:
            return False, (f"paired sign class missing: "
                           f"start={row['sign_type_start']!r} "
                           f"end={row['sign_type_end']!r}")

        zone_length = float(row.get("zone_length_m", 20.0))
        min_lane = 8.0 + zone_length + 4.0
        lane = _pick_route_lane(route_lanes, min_length=min_lane,
                                 road_network=env.current_map.road_network)
        if lane is None:
            return False, "no_suitable_lane"

        sign_mgr = env.engine.traffic_sign_manager
        half_w = lane.width_at(0) / 2 + 0.8
        start_long = 5.0
        end_long = start_long + zone_length
        # MetaDrive expects longitudinal offset from lane end (negative).
        md_start = start_long - lane.length
        md_end = end_long - lane.length
        try:
            sign_mgr.add_sign(sign_start_cls, lane=lane,
                               longitudinal_offset=md_start,
                               lateral_offset=half_w,
                               use_random_lane=False)
            sign_mgr.add_sign(sign_end_cls, lane=lane,
                               longitudinal_offset=md_end,
                               lateral_offset=half_w,
                               use_random_lane=False)
        except Exception as exc:
            return False, f"add_sign(paired): {type(exc).__name__}: {exc}"
        return True, None

    # ----- single-sign schema --------------------------------------------
    sign_key = row.get("sign_type")
    if sign_key is None or sign_key not in SIGN_CLASS_MAP:
        return False, f"sign_type {sign_key!r} not in SIGN_CLASS_MAP"

    if not _override_route_intent(env, row):
        return False, "route_intent_override_failed"

    route_lanes = _get_route_lanes(env)
    if not route_lanes:
        return False, "no_route_lanes"

    sign_cls = SIGN_CLASS_MAP[sign_key]
    sign_mgr = env.engine.traffic_sign_manager
    rn = env.current_map.road_network
    veh = env.vehicle

    if sign_key in DETOUR_KEYS:
        lane = _pick_detour_lane(route_lanes, sign_cls, min_length=30.0)
    elif sign_key in LANE_CHANGE_KEYS:
        vidx = getattr(veh.lane, "index", None)
        vnum = vidx[2] if (vidx and len(vidx) >= 3) else 0
        lane = _pick_lane_for_lane_change(route_lanes, vnum, road_network=rn)
    elif sign_key in RESTRICTED_BEGIN_KEYS:
        lane = _pick_rightmost_lane(route_lanes, min_length=30.0)
    else:
        lane = _pick_route_lane(route_lanes, min_length=15.0, road_network=rn)

    if lane is None:
        return False, "no_suitable_lane"

    try:
        sign = sign_mgr.add_sign(sign_cls, lane=lane, use_random_lane=False)
    except Exception as exc:
        return False, f"add_sign: {type(exc).__name__}: {exc}"

    if sign_key in BEGIN_TO_END and sign is not None:
        try:
            sign_mgr.add_sign(BEGIN_TO_END[sign_key], lane=sign.lane,
                               use_random_lane=False)
        except Exception as exc:
            return False, f"add_sign(end): {type(exc).__name__}: {exc}"

    if sign_key in DETOUR_KEYS and sign is not None:
        spawn_detour_obstacle(env.engine, sign.lane, sign)
        if not getattr(sign, "detour_feasible", True):
            return False, "detour_infeasible"

    if sign_key in BIKE_RELATED_SIGNS and sign is not None:
        if seed is None:
            seed = int(row.get("seed") or row.get("deterministic_seed") or 0)
        _spawn_cyclists_on_lane(env, sign.lane, seed, n=3)

    return True, None

# ============================================================================
# Lane-picking helpers (from original generator)
# ============================================================================

def _get_route_lanes(env):
    vehicle = env.vehicle
    nav = getattr(vehicle, "navigation", None)
    if nav is None:
        return []
    checkpoints = getattr(nav, "checkpoints", None)
    if not checkpoints or len(checkpoints) < 2:
        return []
    road_network = env.current_map.road_network
    route_lanes = []
    for ckpt_start, ckpt_end in zip(checkpoints[:-1], checkpoints[1:]):
        try:
            lanes = road_network.graph[ckpt_start][ckpt_end]
            route_lanes.extend(lanes)
        except KeyError:
            continue
    return route_lanes


def _lane_has_continuation(lane, road_network):
    idx = getattr(lane, "index", None)
    if idx is None or not (isinstance(idx, tuple) and len(idx) >= 2):
        return True
    graph = getattr(road_network, "graph", None)
    if graph is None:
        return True
    end_node = idx[1]
    outgoing = graph.get(end_node)
    if outgoing is None or (isinstance(outgoing, dict) and len(outgoing) == 0):
        return False
    return True


def _pick_route_lane(route_lanes, min_length=10.0, road_network=None):
    candidates = [l for l in route_lanes if l.length >= min_length]
    if road_network is not None:
        candidates = [l for l in candidates if _lane_has_continuation(l, road_network)]
    return random.choice(candidates) if candidates else None


def _pick_rightmost_lane(route_lanes, min_length=30.0):
    segments = {}
    for lane in route_lanes:
        idx = getattr(lane, "index", None)
        if not (isinstance(idx, tuple) and len(idx) >= 3):
            continue
        key = (idx[0], idx[1])
        segments.setdefault(key, {})[idx[2]] = lane
    candidates = []
    for lanes_by_num in segments.values():
        if not lanes_by_num:
            continue
        if max(lanes_by_num.keys()) + 1 < 2:
            continue
        rightmost = lanes_by_num[max(lanes_by_num.keys())]
        if rightmost.length >= min_length:
            candidates.append(rightmost)
    return random.choice(candidates) if candidates else None


def _pick_detour_lane(route_lanes, sign_class, min_length=30.0):
    directions = getattr(sign_class, "allowed_directions", set())
    segments = {}
    for lane in route_lanes:
        idx = getattr(lane, "index", None)
        if not (isinstance(idx, tuple) and len(idx) >= 3):
            continue
        key = (idx[0], idx[1])
        segments.setdefault(key, {})[idx[2]] = lane
    candidates = []
    for lanes_by_num in segments.values():
        num_lanes = max(lanes_by_num.keys()) + 1 if lanes_by_num else 0
        for lane_num, lane in lanes_by_num.items():
            if lane.length < min_length or lane_num <= 0:
                continue
            ok = True
            if "right" in directions and lane_num >= num_lanes - 1:
                ok = False
            if "left" in directions and lane_num <= 0:
                ok = False
            if ok:
                candidates.append(lane)
    return random.choice(candidates) if candidates else None

def _pick_priority_lane(route_lanes, min_length=15.0):
    for lane in route_lanes:
        if lane.length >= min_length:
            return lane
    return None

def _pick_lane_for_lane_change(route_lanes, vehicle_lane_num=0,
                               min_length=15.0, road_network=None):
    segments = {}
    for lane in route_lanes:
        idx = getattr(lane, "index", None)
        if not (isinstance(idx, tuple) and len(idx) >= 3):
            continue
        key = (idx[0], idx[1])
        segments.setdefault(key, {})[idx[2]] = lane
    candidates = []
    for lanes_by_num in segments.values():
        if len(lanes_by_num) < 2:
            continue
        if vehicle_lane_num not in lanes_by_num:
            continue
        lane = lanes_by_num[vehicle_lane_num]
        if lane.length < min_length:
            continue
        if road_network is not None and not _lane_has_continuation(lane, road_network):
            continue
        candidates.append(lane)
    return random.choice(candidates) if candidates else None


# ============================================================================
# Route-intent implementation
# ============================================================================

def _override_route_intent(env, spec):
    """Override navigation destination to match the route_intent.

    Cases:
    - "through" (S, C): no-op, default routing
    - "merge" (r InRamp): agent already spawned on ramp, default destination is fine
    - "exit" (R OutRamp): re-route destination to a node on the exit ramp dead-end
    - exit_X / right / straight / left (T, X, O): pick socket by socket_index
    """
    route_intent = spec.get("route_intent", "through")
    block_id = spec.get("block_id", "")

    if route_intent in ("through", "merge"):
        return True  # default routing is fine

    if route_intent == "exit" and block_id == "R":
        # OutRamp: navigate to the ramp dead-end (block_idx=2 in "SR" sequence).
        # Ramp chain: 2R0_0_ -> 2R1_1_ -> 2R1_2_ -> 2R1_3_ -> 2R1_4_ (dead-end).
        try:
            vehicle = env.vehicle
            nav = vehicle.navigation
            current_lane_index = vehicle.lane.index
            nav.set_route(current_lane_index, "2R1_4_")
            return True
        except Exception:
            return False

    socket_index = spec.get("socket_index", 0)
    try:
        blocks = env.current_map.blocks
        target_block = blocks[-1]  # last block is the sign-bearing block
        sockets = target_block.get_socket_list()
        if socket_index >= len(sockets):
            return False  # invalid socket

        target_socket = sockets[socket_index]
        target_node = target_socket.positive_road.end_node

        # Re-route navigation
        vehicle = env.vehicle
        nav = vehicle.navigation
        current_lane_index = vehicle.lane.index
        nav.set_route(current_lane_index, target_node)
        return True
    except Exception:
        return False


# ============================================================================
# Scene generation
# ============================================================================

def _apply_npc_profile(spec: dict) -> None:
    """Push profile-derived IDM parameters into IDMPolicy class attrs for NPCs.

    These class attrs are read by NPC IDMPolicy instances. Ego is isolated
    separately via apply_ego_defaults() — see §11.1 of the plan.

    Speeds in the profile are m/s; IDMPolicy.NORMAL_SPEED is in km/h, convert.
    """
    from metadrive.policy.idm_policy import IDMPolicy
    if "profile_NORMAL_SPEED" in spec:
        IDMPolicy.NORMAL_SPEED = float(spec["profile_NORMAL_SPEED"]) * 3.6
    if "profile_MAX_SPEED" in spec:
        IDMPolicy.MAX_SPEED = float(spec["profile_MAX_SPEED"]) * 3.6
    if "profile_ACC_FACTOR" in spec:
        IDMPolicy.ACC_FACTOR = float(spec["profile_ACC_FACTOR"])
    if "profile_DEACC_FACTOR" in spec:
        IDMPolicy.DEACC_FACTOR = -abs(float(spec["profile_DEACC_FACTOR"]))
    # DISTANCE_WANTED / TIME_WANTED intentionally NOT applied — keep MetaDrive
    # defaults (10 m / 1.5 s) so NPC IDM doesn't tail-gate ego.
    if "profile_LANE_CHANGE_FREQ" in spec:
        rate_per_km = float(spec["profile_LANE_CHANGE_FREQ"])
        IDMPolicy.LANE_CHANGE_FREQ = int(max(50, 1250.0 / max(rate_per_km, 1.0)))


# Keep backward-compatible alias for code that still calls the old name.
_apply_idm_profile = _apply_npc_profile


# ----------------------------------------------------------------------------
# Per-NPC vehicle type: nuPlan-derived size distribution.
# The default PGTrafficManager.random_vehicle_type uses a hardcoded
# probability vector [0.2, 0.3, 0.3, 0.2, 0.0]. We replace it with a
# nuPlan-derived vector so each NPC spawn samples from the real-traffic
# distribution. Applied at module load (idempotent).
# ----------------------------------------------------------------------------

_NPC_VEHICLE_TYPE_HOOK_INSTALLED = False


def install_npc_vehicle_type_hook() -> None:
    """Replace PGTrafficManager.random_vehicle_type with a nuPlan-weighted variant.

    Idempotent. Derives weights from NuPlanSampler.size_dist so every NPC
    spawned by the default traffic manager picks a vehicle class according to
    the real-traffic distribution.
    """
    global _NPC_VEHICLE_TYPE_HOOK_INSTALLED
    if _NPC_VEHICLE_TYPE_HOOK_INSTALLED:
        return
    try:
        from metadrive.manager.traffic_manager import PGTrafficManager
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type
        from nuplan_sampler import NuPlanSampler
    except ImportError:
        return  # env without MetaDrive — skip silently

    sampler = NuPlanSampler(
        stats_dir=str(SDC_ROOT.parent / "nuplan_statistics")
    ) if (SDC_ROOT.parent / "nuplan_statistics").exists() else None
    if sampler is None or not getattr(sampler, "size_dist", None):
        return

    size_dist = sampler.size_dist  # dict like {'m': 0.7, 's': 0.1, ...}
    # random_vehicle_type expects order (s, m, l, xl, default)
    weights = [
        float(size_dist.get("s", 0.0)),
        float(size_dist.get("m", 0.0)),
        float(size_dist.get("l", 0.0)),
        float(size_dist.get("xl", 0.0)),
        0.0,
    ]
    total = sum(weights)
    if total <= 0:
        return
    weights = [w / total for w in weights]

    def _nuplan_random_vehicle_type(self):
        return random_vehicle_type(self.np_random, weights)

    PGTrafficManager.random_vehicle_type = _nuplan_random_vehicle_type
    _NPC_VEHICLE_TYPE_HOOK_INSTALLED = True


def generate_scene(spec: dict, v_idx: int = 0, max_validation_steps: int = 50) -> dict:
    """Generate and validate a single scene from a decoded base spec.

    Args:
        spec: base-scene spec from FactorizedSpaceCodec.decode (no
            spawn_velocity_ms — that's added here from nuPlan per v_idx).
        v_idx: velocity-sample index in [0, N_SPAWN_VELOCITY_SAMPLES). Used as
            part of the seed for sampling spawn_velocity and for the result's
            scene_id so each (flat_index, v_idx) is reproducible.
    """
    base_seed = spec["seed"]
    seed = _stable_seed(base_seed, "v", v_idx)  # per-materialization seed
    result = dict(spec)
    result["v_idx"] = v_idx
    result["scene_id"] = f"pgmap_{spec['flat_index']}_{v_idx}"

    # NPC-only profile (applied to IDMPolicy class attrs); ego uses defaults.
    profile = sample_one_profile(base_seed)
    for k, v in profile.items():
        result[f"profile_{k}"] = v
    spec = result

    # Deterministic spawn velocity from nuPlan initial_speed distribution.
    spawn_vel = sample_spawn_velocity(_stable_seed(base_seed, "spawn_vel", v_idx))
    spec["spawn_velocity_ms"] = round(spawn_vel, 4)

    np.random.seed(seed)
    random.seed(seed)

    # Install NPC-vehicle-type hook once (idempotent).
    install_npc_vehicle_type_hook()

    # Push NPC profile params into IDMPolicy class attrs (NPC-only).
    _apply_npc_profile(spec)

    from metadrive.component.pgblock.first_block import FirstPGBlock

    vehicle_config = {"show_lidar": False}
    if spawn_vel > 0:
        vehicle_config["spawn_velocity"] = [spawn_vel, 0]
        vehicle_config["spawn_velocity_car_frame"] = True

    spawn_lane = spec["spawn_lane_index"]

    # Special spawn for InRamp merge: agent starts on the ramp's external entry node.
    # The ramp entry chain is 2r1_0_ -> 2r1_1_ -> ... -> 2r0_0_ (joins main road).
    # For OutRamp exit: spawn on rightmost lane (the one that connects to the exit ramp).
    is_ramp_merge = spec.get("block_id") == "r" and spec.get("route_intent") == "merge"
    is_ramp_exit = spec.get("block_id") == "R" and spec.get("route_intent") == "exit"
    if is_ramp_merge:
        spawn_lane_index_tuple = ("2r1_0_", "2r1_1_", 0)  # ramp has only 1 lane
    elif is_ramp_exit:
        # Force rightmost lane so the agent can actually take the exit ramp
        spawn_lane_index_tuple = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, spec["lane_num"] - 1)
    else:
        spawn_lane_index_tuple = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, spawn_lane)

    config = dict(
        start_seed=seed,
        use_render=False,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.CRITICAL,
        traffic_density=spec.get("profile_traffic_density", 0.1),
        horizon=spec.get("profile_horizon_steps", 500),
        vehicle_config=vehicle_config,
        map_config={
            "type": "block_sequence",
            "config": spec["block_sequence"],
            "lane_num": spec["lane_num"],
            "lane_width": spec["lane_width"],
        },
        random_spawn_lane_index=False,
        agent_configs={
            "default_agent": dict(
                use_special_color=True,
                spawn_lane_index=spawn_lane_index_tuple,
            ),
        },
    )

    env = None
    try:
        env = TrafficSignEnv(config)
        obs, info = env.reset(seed=seed)
    except Exception as exc:
        if env:
            try: env.close()
            except: pass
        result["valid"] = False
        result["failure_reason"] = f"env_creation: {exc}"
        return result

    # Ego isolation: override ego-policy IDM params on the instance so the
    # NPC-flavour profile in IDMPolicy class attrs doesn't bleed into ego.
    # Safe no-op if ego policy isn't IDM-based (e.g. direct action control).
    try:
        ego_policy = env.engine.get_policy(env.vehicle.id) if hasattr(env, "engine") else None
        if ego_policy is not None:
            apply_ego_defaults(ego_policy)
    except Exception:
        pass  # never let ego-policy lookup fail scene generation

    try:
        # Override route intent for multi-exit blocks
        if not _override_route_intent(env, spec):
            result["valid"] = False
            result["failure_reason"] = "route_intent_override_failed"
            return result

        route_lanes = _get_route_lanes(env)
        if not route_lanes:
            result["valid"] = False
            result["failure_reason"] = "no_route_lanes"
            return result

        # Place sign
        sign_key = spec["sign_type"]
        sign_class = SIGN_CLASS_MAP[sign_key]
        sign_mgr = env.engine.traffic_sign_manager
        road_network = env.current_map.road_network
        vehicle = env.vehicle

        is_detour = sign_key in DETOUR_KEYS
        is_restricted = sign_key in RESTRICTED_BEGIN_KEYS
        is_lane_change = sign_key in LANE_CHANGE_KEYS

        lane = None
        sign = None
        try:
            if is_detour:
                lane = _pick_detour_lane(route_lanes, sign_class, min_length=30.0)
            elif is_lane_change:
                veh_idx = getattr(vehicle.lane, "index", None)
                veh_lane_num = veh_idx[2] if (veh_idx and len(veh_idx) >= 3) else 0
                lane = _pick_lane_for_lane_change(
                    route_lanes, veh_lane_num, min_length=15.0, road_network=road_network)
            elif is_restricted:
                lane = _pick_rightmost_lane(route_lanes, min_length=30.0)
            else:
                lane = _pick_route_lane(route_lanes, min_length=15.0,
                                        road_network=road_network)

            if lane is None:
                result["valid"] = False
                result["failure_reason"] = "no_suitable_lane"
                return result

            if is_restricted:
                sign = sign_mgr.add_sign(sign_class, lane=lane, use_random_lane=False)
                if sign_key in BEGIN_TO_END:
                    sign_mgr.add_sign(BEGIN_TO_END[sign_key], lane=sign.lane,
                                      use_random_lane=False)
            else:
                sign = sign_mgr.add_sign(sign_class, lane=lane, use_random_lane=False)

            if is_detour and sign is not None:
                spawn_detour_obstacle(env.engine, sign.lane, sign)
                if not getattr(sign, "detour_feasible", True):
                    result["valid"] = False
                    result["failure_reason"] = "detour_infeasible"
                    return result

            # Bike-related sign → spawn a few cyclists on the sign's lane so
            # the scene is visually/semantically coherent (§14).
            if sign_key in BIKE_RELATED_SIGNS and sign is not None:
                n_cyc = _spawn_cyclists_on_lane(env, sign.lane, seed, n=3)
                result["n_cyclists_spawned"] = n_cyc

        except Exception as exc:
            result["valid"] = False
            result["failure_reason"] = f"sign_placement: {exc}"
            return result

        # Quick validation steps
        try:
            for _ in range(max_validation_steps):
                obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
                if terminated or truncated:
                    break
        except Exception as exc:
            result["valid"] = False
            result["failure_reason"] = f"validation_step: {exc}"
            return result

        result["valid"] = True
        result["failure_reason"] = None
        return result

    except Exception as exc:
        result["valid"] = False
        result["failure_reason"] = f"unexpected: {exc}"
        return result
    finally:
        if env:
            try: env.close()
            except: pass


def validate_and_materialize(
    specs: list[dict],
    n_velocity_samples: int = N_SPAWN_VELOCITY_SAMPLES,
    max_validation_steps: int = 50,
    verbose: bool = True,
) -> list[dict]:
    """Validate a batch of base-scene specs, expanding each to n_velocity_samples rows.

    Each base spec (from FactorizedSpaceCodec.decode) produces
    `n_velocity_samples` materialized scene rows, each with a spawn_velocity
    drawn from nuPlan initial_speed (deterministic seed per (spec, v_idx)).
    """
    results = []
    valid = 0
    invalid = 0
    failures = Counter()
    t0 = time.time()

    total = len(specs) * n_velocity_samples
    done = 0

    for i, spec in enumerate(specs):
        for v_idx in range(n_velocity_samples):
            done += 1
            if verbose:
                print(f"  → [{done}/{total}] flat_index={spec.get('flat_index')} "
                      f"v_idx={v_idx} sign={spec.get('sign_type')} "
                      f"block={spec.get('block_id')} lane_num={spec.get('lane_num')}",
                      flush=True)
            result = generate_scene(spec, v_idx=v_idx,
                                    max_validation_steps=max_validation_steps)
            results.append(result)
            if result.get("valid"):
                valid += 1
            else:
                invalid += 1
                reason = (result.get("failure_reason") or "unknown").split(":")[0]
                failures[reason] += 1

            if verbose and done % 20 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{total}] valid={valid} invalid={invalid} "
                      f"({rate:.1f} scenes/s)")

    if verbose:
        elapsed = time.time() - t0
        print(f"Materialization: {valid} valid, {invalid} invalid ({elapsed:.1f}s)")
        if failures:
            print(f"Failures: {dict(failures)}")

    return results


# ============================================================================
# Paired scenes (zone-start + zone-end) — §12
# ============================================================================

def generate_paired_scene(spec: dict, v_idx: int = 0,
                          max_validation_steps: int = 50) -> dict:
    """Generate a scene with TWO signs (zone start + zone end) on a linear segment.

    spec comes from PairedSpaceCodec.decode; includes sign_type_start,
    sign_type_end, zone_length_m. One scene contributes to both PDD codes'
    per-sign manifests.
    """
    base_seed = spec["seed"]
    seed = _stable_seed(base_seed, "v", v_idx)
    result = dict(spec)
    result["v_idx"] = v_idx
    result["scene_id"] = (f"pgmap_paired_{spec['paired_flat_index']}_{v_idx}")
    result["pair_bundle_id"] = spec["bundle_id"]

    profile = sample_one_profile(base_seed)
    for k, v in profile.items():
        result[f"profile_{k}"] = v
    spec = result

    spawn_vel = sample_spawn_velocity(_stable_seed(base_seed, "spawn_vel", v_idx))
    spec["spawn_velocity_ms"] = round(spawn_vel, 4)

    np.random.seed(seed)
    random.seed(seed)

    install_npc_vehicle_type_hook()
    _apply_npc_profile(spec)

    from metadrive.component.pgblock.first_block import FirstPGBlock

    vehicle_config = {"show_lidar": False}
    if spawn_vel > 0:
        vehicle_config["spawn_velocity"] = [spawn_vel, 0]
        vehicle_config["spawn_velocity_car_frame"] = True

    config = dict(
        start_seed=seed,
        use_render=False,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.CRITICAL,
        traffic_density=spec.get("profile_traffic_density", 0.1),
        horizon=spec.get("profile_horizon_steps", 500),
        vehicle_config=vehicle_config,
        map_config={
            "type": "block_sequence",
            "config": spec["block_sequence"],
            "lane_num": spec["lane_num"],
            "lane_width": spec["lane_width"],
        },
        random_spawn_lane_index=False,
        agent_configs={
            "default_agent": dict(
                use_special_color=True,
                spawn_lane_index=(
                    FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, spec["spawn_lane_index"]
                ),
            ),
        },
    )

    env = None
    try:
        env = TrafficSignEnv(config)
        env.reset(seed=seed)
    except Exception as exc:
        if env:
            try: env.close()
            except: pass
        result["valid"] = False
        result["failure_reason"] = f"env_creation: {exc}"
        return result

    try:
        ego_policy = env.engine.get_policy(env.vehicle.id) if hasattr(env, "engine") else None
        if ego_policy is not None:
            apply_ego_defaults(ego_policy)
    except Exception:
        pass

    try:
        route_lanes = _get_route_lanes(env)
        if not route_lanes:
            result["valid"] = False
            result["failure_reason"] = "no_route_lanes"
            return result

        zone_length = float(spec["zone_length_m"])
        min_lane = 8.0 + zone_length + 4.0       # 5 start_off + zone + 5 tail buffer
        lane = _pick_route_lane(route_lanes, min_length=min_lane,
                                 road_network=env.current_map.road_network)
        if lane is None:
            result["valid"] = False
            result["failure_reason"] = (
                f"zone_too_long: no lane >= {min_lane:.1f}m for zone={zone_length}m"
            )
            return result

        sign_start_cls = SIGN_CLASS_MAP[spec["sign_type_start"]]
        sign_end_cls = SIGN_CLASS_MAP[spec["sign_type_end"]]
        sign_mgr = env.engine.traffic_sign_manager

        half_w = lane.width_at(0) / 2 + 0.8
        start_long = 5.0
        end_long = start_long + zone_length
        # MetaDrive convention: longitudinal_offset counted from lane end.
        md_start = start_long - lane.length
        md_end = end_long - lane.length

        try:
            sign_mgr.add_sign(sign_start_cls, lane=lane,
                              longitudinal_offset=md_start,
                              lateral_offset=half_w,
                              use_random_lane=False)
            sign_mgr.add_sign(sign_end_cls, lane=lane,
                              longitudinal_offset=md_end,
                              lateral_offset=half_w,
                              use_random_lane=False)
        except Exception as exc:
            result["valid"] = False
            result["failure_reason"] = f"sign_placement: {exc}"
            return result

        try:
            for _ in range(max_validation_steps):
                _obs, _r, terminated, truncated, _info = env.step([0.0, 0.0])
                if terminated or truncated:
                    break
        except Exception as exc:
            result["valid"] = False
            result["failure_reason"] = f"validation_step: {exc}"
            return result

        result["valid"] = True
        result["failure_reason"] = None
        return result

    except Exception as exc:
        result["valid"] = False
        result["failure_reason"] = f"unexpected: {exc}"
        return result
    finally:
        if env:
            try: env.close()
            except: pass


def validate_and_materialize_paired(
    paired_specs: list[dict],
    n_velocity_samples: int = N_SPAWN_VELOCITY_SAMPLES,
    max_validation_steps: int = 50,
    verbose: bool = True,
) -> list[dict]:
    """Batch materialize paired-scene base specs. One spec → n_velocity_samples rows."""
    results = []
    valid = 0
    invalid = 0
    failures = Counter()
    t0 = time.time()
    total = len(paired_specs) * n_velocity_samples
    done = 0

    for spec in paired_specs:
        for v_idx in range(n_velocity_samples):
            done += 1
            if verbose:
                print(f"  → [{done}/{total}] paired flat_index={spec.get('paired_flat_index')} "
                      f"v_idx={v_idx} {spec.get('sign_type_start')} → {spec.get('sign_type_end')} "
                      f"zone={spec.get('zone_length_m')}m", flush=True)
            result = generate_paired_scene(spec, v_idx=v_idx,
                                           max_validation_steps=max_validation_steps)
            results.append(result)
            if result.get("valid"):
                valid += 1
            else:
                invalid += 1
                reason = (result.get("failure_reason") or "unknown").split(":")[0]
                failures[reason] += 1

    if verbose:
        elapsed = time.time() - t0
        print(f"Paired materialization: {valid} valid, {invalid} invalid ({elapsed:.1f}s)")
        if failures:
            print(f"Failures: {dict(failures)}")

    return results
