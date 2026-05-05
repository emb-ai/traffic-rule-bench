#!/usr/bin/env python3
"""
Validation script for RuleCompliantExpertPolicy on TrafficSignEnv.

Places traffic signs on the agent's route on 3 diverse map types, runs
episodes with:
  - RuleCompliantExpertPolicy  (rule-aware PPO expert)
  - plain ExpertPolicy          (baseline PPO, no sign awareness)
  - no agent                    (stationary vehicle)

Saves per-episode GIF recordings and compliance statistics (CSV + JSON).

Usage examples:
    python check_rule_compliant_expert.py
    python check_rule_compliant_expert.py --render-gif
    python check_rule_compliant_expert.py --agents rule_expert --maps SSSSS --sign-types stop,speed40
"""

import argparse
import csv
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------

def _find_pdd_bench_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for parent in (current, *current.parents):
        if (parent / "envs").is_dir() and (parent / "traffic_signs").is_dir():
            return parent
    raise RuntimeError("Could not locate pdd-bench root")


SCRIPT_PATH = Path(__file__).resolve()
PDD_BENCH_DIR = _find_pdd_bench_root(SCRIPT_PATH)
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"

for _p in (PDD_BENCH_DIR, METADRIVE_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from metadrive.policy.expert_policy import ExpertPolicy

from envs.traffic_sign_env import TrafficSignEnv
from agents.policies.rule_compliant_expert import RuleCompliantExpertPolicy
from agents.policies.comprehensive_rule_expert import ComprehensiveRuleExpertPolicy

from traffic_signs.stop_sign import StopSign
from traffic_signs.speed_limit_sign import (
    SpeedLimitSign20, SpeedLimitSign30,
    SpeedLimitSign40, SpeedLimitSign60,
)
from traffic_signs.zone_signs import (
    ZoneSpeedLimitSign20, ZoneSpeedLimitSign30,
    ZoneSpeedLimitSign40, ZoneSpeedLimitSign60,
)
from traffic_signs.end_of_zone_signs import (
    EndOfSpeedLimitSign20, EndOfSpeedLimitSign30,
    EndOfSpeedLimitSign40, EndOfSpeedLimitSign60,
    EndOfZoneSpeedLimitSign20, EndOfZoneSpeedLimitSign30,
    EndOfZoneSpeedLimitSign40, EndOfZoneSpeedLimitSign60,
    EndOfAllRestrictionsSign,
)
from traffic_signs.min_speed_limit_sign import (
    MinimumSpeedLimit30, MinimumSpeedLimit40,
    MinimumSpeedLimit50, MinimumSpeedLimit60,
)
from traffic_signs.no_entry_sign import NoEntrySign
from traffic_signs.no_traffic_sign import NoTrafficSign
from traffic_signs.no_stopping_allowed_sign import NoStoppingAllowedSign
from traffic_signs.detour_sign import (
    DetourSign, DetourRightSign, DetourLeftSign, DetourEitherSign,
)
from traffic_signs.detour_obstacle import spawn_detour_obstacle
from traffic_signs.restricted_lane_sign import (
    BusLaneRoadSign, BikeLaneRoadSign,
    EndBusLaneRoadSign, EndBikeLaneRoadSign,
    ExitToBusLaneSign, ExitToBusLaneSignLeft,
    ExitToBikeLaneSign, ExitToBikeLaneSignLeft,
    BusLaneSign, BikeLaneSign,
    EndBusLaneSign, EndBikeLaneSign,
)
from traffic_signs.bus_station_sign import BusStationSign
from traffic_signs.only_auto_sign import OnlyAutoSign
from traffic_signs.right_turn_rule import RightTurnRule
from traffic_signs.traffic_light_sign import TrafficLightSign
from traffic_signs.direction_sign import DirectionSign
from traffic_signs.pg_direction_sign import PGDirectionSign
from traffic_signs.no_turn_allowed import NoRightTurnSign, NoLeftTurnSign, NoUTurnSign
from traffic_signs.one_way_entry_sign import OneWayEntrySignR, OneWayEntrySignL, OneWayEntrySignS
from traffic_signs.no_overtaking_sign import NoOvertakingSign


# ---------------------------------------------------------------------------
# Catalogues
# ---------------------------------------------------------------------------

SIGN_CATALOGUE = {
    # --- Speed / stop ---
    "stop":             StopSign,
    "speed20":          SpeedLimitSign20,
    "speed30":          SpeedLimitSign30,
    "speed40":          SpeedLimitSign40,
    "speed60":          SpeedLimitSign60,
    "zone_speed20":     ZoneSpeedLimitSign20,
    "zone_speed30":     ZoneSpeedLimitSign30,
    "zone_speed40":     ZoneSpeedLimitSign40,
    "zone_speed60":     ZoneSpeedLimitSign60,
    "minspeed30":       MinimumSpeedLimit30,
    "minspeed40":       MinimumSpeedLimit40,
    "minspeed50":       MinimumSpeedLimit50,
    "minspeed60":       MinimumSpeedLimit60,
    # --- End-of-zone (informational, terminate previous zones) ---
    "end_speed20":      EndOfSpeedLimitSign20,
    "end_speed30":      EndOfSpeedLimitSign30,
    "end_speed40":      EndOfSpeedLimitSign40,
    "end_speed60":      EndOfSpeedLimitSign60,
    "end_zone_speed20": EndOfZoneSpeedLimitSign20,
    "end_zone_speed30": EndOfZoneSpeedLimitSign30,
    "end_zone_speed40": EndOfZoneSpeedLimitSign40,
    "end_zone_speed60": EndOfZoneSpeedLimitSign60,
    "end_all":          EndOfAllRestrictionsSign,
    # --- Prohibitory ---
    "no_entry":         NoEntrySign,
    "no_traffic":       NoTrafficSign,
    "no_stop":          NoStoppingAllowedSign,
    # --- Detour (4.2.x) ---
    "detour_right":     DetourRightSign,
    "detour_left":      DetourLeftSign,
    "detour_either":    DetourEitherSign,
    # --- Restricted lane (5.11.x / 5.14.x) ---
    "bus_lane_road":    BusLaneRoadSign,
    "bike_lane_road":   BikeLaneRoadSign,
    "bus_lane":         BusLaneSign,
    "bike_lane":        BikeLaneSign,
    # --- End of restricted lane (5.12.x / 5.14.3-4, informational) ---
    "end_bus_lane_road":  EndBusLaneRoadSign,
    "end_bike_lane_road": EndBikeLaneRoadSign,
    "end_bus_lane":       EndBusLaneSign,
    "end_bike_lane":      EndBikeLaneSign,
    # --- Intersection restricted (5.13.x) ---
    "exit_bus_right":   ExitToBusLaneSign,
    "exit_bus_left":    ExitToBusLaneSignLeft,
    "exit_bike_right":  ExitToBikeLaneSign,
    "exit_bike_left":   ExitToBikeLaneSignLeft,
    # --- Other ---
    "bus_station":      BusStationSign,
    "only_auto":        OnlyAutoSign,
    "right_turn_rule":  RightTurnRule,
    "traffic_light":    TrafficLightSign,
    "direction":        DirectionSign,
    "pg_direction":     PGDirectionSign,
    # --- Turn prohibition (3.18.x / 3.19) ---
    "no_right_turn":    NoRightTurnSign,
    "no_left_turn":     NoLeftTurnSign,
    "no_uturn":         NoUTurnSign,
    # --- One-way (5.5 / 5.7) ---
    "oneway_r":         OneWayEntrySignR,
    "oneway_l":         OneWayEntrySignL,
    "oneway_s":         OneWayEntrySignS,
    # --- No overtaking (3.20) ---
    "no_overtaking":    NoOvertakingSign,
}

DEFAULT_SIGN_TYPES = [
    "stop", "speed40", "zone_speed40", "minspeed40",
    "no_entry", "no_traffic", "no_stop",
    "detour_right", "detour_left",
    "bus_lane_road", "bus_lane",
    "bus_station", "only_auto",
]

AGENT_CATALOGUE = {
    "rule_expert":          RuleCompliantExpertPolicy,
    "comprehensive_expert": ComprehensiveRuleExpertPolicy,
    "expert":               ExpertPolicy,
}

DEFAULT_AGENTS = ["rule_expert", "comprehensive_expert", "expert"]

MAP_CODES = {
    # "SSSSS": "SSSSS",
    # "STSTS": "STSTS",
    # "SXSXS": "SXSXS",
    "X":"X",
    "T":"T",
    "S":"S",
    "O":"O",
}

DEFAULT_MAPS = ["X", "T", "S", "O"]

# ---------------------------------------------------------------------------
# Sign-type placement groups
# ---------------------------------------------------------------------------

# Restricted-lane begin signs — place on rightmost lane, pair with end sign.
_RESTRICTED_BEGIN_KEYS = {"bus_lane_road", "bike_lane_road", "bus_lane", "bike_lane"}
# Restricted-lane end signs — cannot be used alone, auto-prepend begin sign.
_RESTRICTED_END_KEYS = {"end_bus_lane_road", "end_bike_lane_road",
                        "end_bus_lane", "end_bike_lane"}

_BEGIN_TO_END = {
    "bus_lane_road":  ("end_bus_lane_road",  EndBusLaneRoadSign),
    "bike_lane_road": ("end_bike_lane_road", EndBikeLaneRoadSign),
    "bus_lane":       ("end_bus_lane",       EndBusLaneSign),
    "bike_lane":      ("end_bike_lane",      EndBikeLaneSign),
}
_END_TO_BEGIN = {
    "end_bus_lane_road":  ("bus_lane_road",  BusLaneRoadSign),
    "end_bike_lane_road": ("bike_lane_road", BikeLaneRoadSign),
    "end_bus_lane":       ("bus_lane",       BusLaneSign),
    "end_bike_lane":      ("bike_lane",      BikeLaneSign),
}

_DETOUR_KEYS = {"detour_right", "detour_left", "detour_either"}
# Signs that require the vehicle to change lanes — must be placed on the
# vehicle's actual lane on a multi-lane road segment.
_LANE_CHANGE_KEYS = {"no_entry", "no_traffic"}


# ---------------------------------------------------------------------------
# Route-aware sign placement
# ---------------------------------------------------------------------------

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
    """Return False if lane ends at a dead-end outer socket (no outgoing edges)."""
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
    if not candidates:
        return None
    return random.choice(candidates)


def _pick_rightmost_lane(route_lanes, min_length=30.0):
    """Pick the rightmost lane from a multi-lane road segment on the route."""
    segments = {}
    for lane in route_lanes:
        idx = getattr(lane, "index", None)
        if not (isinstance(idx, tuple) and len(idx) >= 3):
            continue
        key = (idx[0], idx[1])
        segments.setdefault(key, {})[idx[2]] = lane

    candidates = []
    for _key, lanes_by_num in segments.items():
        if not lanes_by_num:
            continue
        num_lanes = max(lanes_by_num.keys()) + 1
        if num_lanes < 2:
            continue  # need at least 2 lanes for restricted to make sense
        rightmost_num = max(lanes_by_num.keys())
        lane = lanes_by_num[rightmost_num]
        if lane.length >= min_length:
            candidates.append(lane)

    if not candidates:
        return None
    return random.choice(candidates)


def _pick_detour_lane(route_lanes, sign_class, min_length=30.0):
    """Pick a middle lane for detour signs — not edge lanes.

    For DetourRightSign the sign lane must have a neighbour to the right
    (i.e. sign is NOT on the rightmost lane).
    For DetourLeftSign — NOT on the leftmost lane.
    For DetourEitherSign — neither leftmost nor rightmost.

    In PG maps lane 0 = leftmost, lane N-1 = rightmost.
    """
    directions = getattr(sign_class, "allowed_directions", set())

    # Group route lanes by road segment (from_node, to_node).
    segments = {}  # (from, to) -> {lane_num: lane}
    for lane in route_lanes:
        idx = getattr(lane, "index", None)
        if not (isinstance(idx, tuple) and len(idx) >= 3):
            continue
        key = (idx[0], idx[1])
        segments.setdefault(key, {})[idx[2]] = lane

    candidates = []
    for _key, lanes_by_num in segments.items():
        if not lanes_by_num:
            continue
        num_lanes = max(lanes_by_num.keys()) + 1
        for lane_num, lane in lanes_by_num.items():
            if lane.length < min_length:
                continue
            # Always skip leftmost (lane_num 0) — keeps sign away from
            # oncoming-traffic edge and ensures a left neighbour exists.
            if lane_num <= 0:
                continue
            ok = True
            if "right" in directions and lane_num >= num_lanes - 1:
                ok = False  # no right neighbour
            if "left" in directions and lane_num <= 0:
                ok = False  # no left neighbour
            if ok:
                candidates.append(lane)

    if not candidates:
        return None
    return random.choice(candidates)


def _pick_lane_for_lane_change(route_lanes, vehicle_lane_num=0,
                               min_length=15.0, road_network=None):
    """Pick a lane matching the vehicle's lane number on a multi-lane segment.

    Ensures the sign is placed on the vehicle's actual path AND there is at
    least one adjacent lane to escape to.
    """
    segments = {}
    for lane in route_lanes:
        idx = getattr(lane, "index", None)
        if not (isinstance(idx, tuple) and len(idx) >= 3):
            continue
        key = (idx[0], idx[1])
        segments.setdefault(key, {})[idx[2]] = lane

    candidates = []
    for _key, lanes_by_num in segments.items():
        if len(lanes_by_num) < 2:
            continue  # need multi-lane for lane change
        if vehicle_lane_num not in lanes_by_num:
            continue
        lane = lanes_by_num[vehicle_lane_num]
        if lane.length < min_length:
            continue
        if road_network is not None and not _lane_has_continuation(lane, road_network):
            continue
        candidates.append(lane)

    if not candidates:
        return None
    return random.choice(candidates)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _parse_speed_spec(spec_str):
    """Parse a speed specification string.

    Accepts:
      "30"       → fixed 30 km/h
      "20-40"    → uniform sample in [20, 40] km/h
      None / ""  → None (use default)

    Returns (min_kmh, max_kmh) or None.
    """
    if spec_str is None or str(spec_str).strip() == "":
        return None
    s = str(spec_str).strip()
    if "-" in s:
        parts = s.split("-", 1)
        lo, hi = float(parts[0]), float(parts[1])
        if lo < 0 or hi < 0:
            raise ValueError(f"Speed must be non-negative, got '{spec_str}'")
        if lo > hi:
            raise ValueError(f"Speed range min > max: '{spec_str}'")
        return (lo, hi)
    v = float(s)
    if v < 0:
        raise ValueError(f"Speed must be non-negative, got '{spec_str}'")
    return (v, v)


def _sample_speed(speed_range, rng):
    """Sample a speed in km/h from a (min, max) range, return m/s."""
    if speed_range is None:
        return None
    lo, hi = speed_range
    kmh = rng.uniform(lo, hi) if lo != hi else lo
    return kmh / 3.6  # convert to m/s


def _parse_lane_spec(spec_str):
    """Parse an ego/traffic lane specification string.

    Accepts:
      None / ""    → None (use default / random)
      "0"          → lane number 0
      "0,1,2"      → list of lane numbers to sample from
      "random"     → None (explicit random)

    Returns list of ints, or None.
    """
    if spec_str is None or str(spec_str).strip() == "" or str(spec_str).strip().lower() == "random":
        return None
    parts = [p.strip() for p in str(spec_str).split(",") if p.strip()]
    lanes = []
    for p in parts:
        v = int(p)
        if v < 0:
            raise ValueError(f"Lane index must be non-negative, got {v}")
        lanes.append(v)
    return lanes


def run_episode(
    sign_type_key,
    agent_key,
    seed,
    max_steps,
    map_code="SSSSS",
    render_gif=False,
    gif_path=None,
    gif_duration=40,
    traffic_density=0.0,
    ego_lane_spec=None,
    traffic_lane_spec=None,
    ego_speed_range=None,
    traffic_speed_range=None,
    debug_spawn=False,
):
    """Run one episode with configurable spawn lanes, density, and speeds.

    Parameters
    ----------
    ego_lane_spec : list[int] | None
        Candidate lane numbers for ego. If None, use env default (random).
        If list, one is sampled under the episode seed.
    traffic_lane_spec : list[int] | None
        Not directly supported by MetaDrive traffic manager (NPC lanes are
        chosen automatically).  Logged for documentation; reserved for
        future SUMO-based filtering.
    ego_speed_range : (float, float) | None
        (min_kmh, max_kmh). Sampled uniformly → converted to m/s →
        set as ``spawn_velocity`` in vehicle config.
    traffic_speed_range : (float, float) | None
        (min_kmh, max_kmh). Sampled → set as ``spawn_velocity`` in
        ``traffic_vehicle_config``.  All NPC vehicles get the same
        initial speed (IDM will then adapt).
    debug_spawn : bool
        Print spawn parameters for this episode.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    random.seed(seed)

    sign_class = SIGN_CATALOGUE.get(sign_type_key)
    if sign_class is None:
        return _error_result(sign_type_key, agent_key, map_code, seed,
                             f"Unknown sign type: {sign_type_key}")

    if agent_key not in AGENT_CATALOGUE:
        return _error_result(sign_type_key, agent_key, map_code, seed,
                             f"Unknown agent: {agent_key}")

    agent_policy = AGENT_CATALOGUE[agent_key]

    # --- Build vehicle config ---
    vehicle_config = {"show_lidar": False}

    # Ego initial speed
    ego_vel_ms = _sample_speed(ego_speed_range, rng)
    if ego_vel_ms is not None:
        vehicle_config["spawn_velocity"] = [ego_vel_ms, 0]
        vehicle_config["spawn_velocity_car_frame"] = True

    # --- Build env config ---
    config = dict(
        map=map_code,
        start_seed=seed,
        use_render=False,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.CRITICAL,
        traffic_density=traffic_density,
        horizon=max_steps,
        vehicle_config=vehicle_config,
    )

    # Ego spawn lane
    chosen_ego_lane = None
    if ego_lane_spec is not None and len(ego_lane_spec) > 0:
        chosen_ego_lane = rng.choice(ego_lane_spec)
        # Override the default spawn lane index — keep road nodes, change lane num.
        # MetaDriveEnv default is (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0).
        from metadrive.component.pgblock.first_block import FirstPGBlock
        config["random_spawn_lane_index"] = False
        config["agent_configs"] = {
            "default_agent": dict(
                use_special_color=True,
                spawn_lane_index=(
                    FirstPGBlock.NODE_1,
                    FirstPGBlock.NODE_2,
                    chosen_ego_lane,
                ),
            )
        }

    # Traffic initial speed
    traffic_vel_ms = _sample_speed(traffic_speed_range, rng)
    if traffic_vel_ms is not None:
        config["traffic_vehicle_config"] = {
            "spawn_velocity": [traffic_vel_ms, 0],
            "spawn_velocity_car_frame": True,
        }

    if agent_policy is not None:
        config["agent_policy"] = agent_policy

    # --- Debug output ---
    if debug_spawn:
        ego_spd_str = (
            f"{ego_vel_ms * 3.6:.1f} km/h" if ego_vel_ms is not None
            else "default (stationary)"
        )
        traffic_spd_str = (
            f"{traffic_vel_ms * 3.6:.1f} km/h" if traffic_vel_ms is not None
            else "default (IDM)"
        )
        print(
            f"  [SPAWN] ego_lane={chosen_ego_lane or 'random'} "
            f"ego_speed={ego_spd_str} "
            f"traffic_density={traffic_density} "
            f"traffic_lanes={traffic_lane_spec or 'auto'} "
            f"traffic_speed={traffic_spd_str}"
        )

    env = TrafficSignEnv(config)
    try:
        obs, info = env.reset()

        sign_mgr = env.engine.traffic_sign_manager
        route_lanes = _get_route_lanes(env)
        road_network = env.current_map.road_network

        # -- Type-specific sign spawning --
        is_detour = sign_type_key in _DETOUR_KEYS
        is_restricted_begin = sign_type_key in _RESTRICTED_BEGIN_KEYS
        is_restricted_end = sign_type_key in _RESTRICTED_END_KEYS
        is_lane_change = sign_type_key in _LANE_CHANGE_KEYS

        try:
            lane = None
            if route_lanes:
                if is_detour:
                    lane = _pick_detour_lane(route_lanes, sign_class, min_length=30.0)
                elif is_lane_change:
                    # Place on vehicle's lane on a multi-lane segment so the
                    # expert actually needs to perform a lane change.
                    veh_idx = getattr(env.vehicle.lane, "index", None)
                    veh_lane_num = veh_idx[2] if (veh_idx and len(veh_idx) >= 3) else 0
                    lane = _pick_lane_for_lane_change(
                        route_lanes, veh_lane_num,
                        min_length=15.0, road_network=road_network,
                    )
                elif is_restricted_begin or is_restricted_end:
                    lane = _pick_rightmost_lane(route_lanes, min_length=30.0)
                else:
                    lane = _pick_route_lane(route_lanes, min_length=15.0,
                                            road_network=road_network)

            # --- Restricted-end: auto-prepend begin sign (5.12 needs 5.11) ---
            if is_restricted_end:
                _bkey, begin_cls = _END_TO_BEGIN[sign_type_key]
                if lane is not None:
                    sign_mgr.add_sign(begin_cls, lane=lane, use_random_lane=False)
                    sign = sign_mgr.add_sign(sign_class, lane=lane,
                                             use_random_lane=False)
                else:
                    sign_mgr.add_sign(begin_cls, use_random_lane=True)
                    sign = sign_mgr.add_sign(sign_class, use_random_lane=True)

            # --- Restricted-begin: auto-append end sign ---
            elif is_restricted_begin:
                if lane is not None:
                    sign = sign_mgr.add_sign(sign_class, lane=lane,
                                             use_random_lane=False)
                else:
                    sign = sign_mgr.add_sign(sign_class, use_random_lane=True)
                if sign_type_key in _BEGIN_TO_END:
                    _ekey, end_cls = _BEGIN_TO_END[sign_type_key]
                    sign_mgr.add_sign(end_cls, lane=sign.lane,
                                      use_random_lane=False)

            # --- Default ---
            else:
                if lane is not None:
                    sign = sign_mgr.add_sign(sign_class, lane=lane,
                                             use_random_lane=False)
                else:
                    sign = sign_mgr.add_sign(sign_class, use_random_lane=True)

            # Spawn traffic cones at the obstacle point for detour signs.
            if is_detour and sign is not None:
                spawn_detour_obstacle(env.engine, sign.lane, sign)

        except Exception as exc:
            return _error_result(sign_type_key, agent_key, map_code, seed,
                                 f"add_sign failed: {exc}")

        total_violations = 0
        violations_by_class = defaultdict(int)
        reached_dest = False
        crashed = False
        out_of_road = False
        final_step = 0
        max_speed_kmh = 0.0
        avg_speed_acc = 0.0

        # --- Trajectory collection ---
        traj_positions = []
        traj_headings = []
        traj_speeds = []
        traj_velocities = []
        traj_actions = []
        traj_steerings = []
        traj_throttles = []
        traj_lane_indices = []
        traj_violations = []

        for step in range(max_steps):
            action = [0.0, 0.0]
            obs, reward, terminated, truncated, info = env.step(action)
            final_step = step + 1

            vehicle = env.vehicle
            speed_kmh = vehicle.speed_km_h
            max_speed_kmh = max(max_speed_kmh, speed_kmh)
            avg_speed_acc += speed_kmh

            sign_violations = sign_mgr.check_all_violations(vehicle)
            step_violations = 0
            for sign, violated in sign_violations:
                if violated:
                    total_violations += 1
                    step_violations += 1
                    violations_by_class[type(sign).__name__] += 1

            # --- Record trajectory step ---
            traj_positions.append(np.array(vehicle.position, dtype=np.float32))
            traj_headings.append(float(vehicle.heading_theta))
            traj_speeds.append(float(speed_kmh))
            vel = getattr(vehicle, "velocity", [0.0, 0.0])
            traj_velocities.append(np.array(vel, dtype=np.float32))
            act = getattr(vehicle, "current_action", action)
            traj_actions.append(np.array(act, dtype=np.float32))
            traj_steerings.append(float(getattr(vehicle, "steering", 0.0)))
            traj_throttles.append(float(getattr(vehicle, "throttle_brake", 0.0)))
            lane_idx = getattr(vehicle.lane, "index", None)
            traj_lane_indices.append(str(lane_idx) if lane_idx is not None else "")
            traj_violations.append(step_violations)

            if render_gif:
                text_dict = {
                    "Step": step,
                    "Agent": agent_key,
                    "Map": map_code,
                    "Sign": sign_type_key,
                    "Speed": f"{speed_kmh:.1f} km/h",
                    "Violations": total_violations,
                    "Lane": str(getattr(vehicle.lane, "index", "?")),
                }
                env.render(
                    mode="top_down",
                    text=text_dict,
                    scaling=40.0,
                    screen_size=(600, 600),
                    semantic_map=False,
                    draw_target_vehicle_trajectory=True,
                    target_agent_heading_up=True,
                    screen_record=True,
                )

            done = terminated or truncated
            if done:
                reached_dest = info.get("arrive_dest", False)
                crashed = (
                    info.get("crash", False)
                    or info.get("crash_vehicle", False)
                    or getattr(vehicle, "crash_vehicle", False)
                )
                out_of_road = info.get("out_of_road", False)
                break

        compliant = total_violations == 0 and not crashed and not out_of_road
        avg_speed = avg_speed_acc / max(final_step, 1)

        # --- Build trajectory dict ---
        trajectory = {
            "positions": np.array(traj_positions, dtype=np.float32),      # (T, 2)
            "headings": np.array(traj_headings, dtype=np.float32),        # (T,)
            "speeds_kmh": np.array(traj_speeds, dtype=np.float32),        # (T,)
            "velocities": np.array(traj_velocities, dtype=np.float32),    # (T, 2)
            "actions": np.array(traj_actions, dtype=np.float32),          # (T, 2)
            "steerings": np.array(traj_steerings, dtype=np.float32),      # (T,)
            "throttles": np.array(traj_throttles, dtype=np.float32),      # (T,)
            "lane_indices": np.array(traj_lane_indices, dtype=str),       # (T,)
            "violations_per_step": np.array(traj_violations, dtype=np.int32),  # (T,)
        }

        # Sign metadata
        sign_meta = {}
        if sign is not None:
            sign_meta["sign_position"] = np.array(
                sign.position, dtype=np.float32
            ) if hasattr(sign, "position") else np.array([0., 0.], dtype=np.float32)
            sign_meta["sign_lane_index"] = str(getattr(getattr(sign, "lane", None), "index", ""))
            sign_meta["placement_long"] = float(getattr(sign, "placement_long", 0.0))
            zone_start = getattr(sign, "zone_start", None)
            zone_end = getattr(sign, "zone_end", None)
            if zone_start is not None:
                sign_meta["zone_start"] = float(zone_start)
            if zone_end is not None:
                sign_meta["zone_end"] = float(zone_end)

        saved_gif = None
        if render_gif and gif_path:
            renderer = getattr(env, "top_down_renderer", None)
            if renderer is not None:
                frames = getattr(renderer, "_screen_frames", [])
                if frames:
                    gif_out = Path(gif_path)
                    gif_out.parent.mkdir(parents=True, exist_ok=True)
                    renderer.generate_gif(
                        gif_name=str(gif_out), duration=int(gif_duration)
                    )
                    saved_gif = str(gif_out)

        return dict(
            sign_type=sign_type_key,
            agent=agent_key,
            map_code=map_code,
            seed=seed,
            total_violations=total_violations,
            violations_by_sign_class=dict(violations_by_class),
            reached_destination=reached_dest,
            crashed=crashed,
            out_of_road=out_of_road,
            max_step=final_step,
            compliant_episode=compliant,
            max_speed_kmh=round(max_speed_kmh, 2),
            avg_speed_kmh=round(avg_speed, 2),
            gif_path=saved_gif,
            trajectory=trajectory,
            sign_meta=sign_meta,
            error=None,
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return _error_result(sign_type_key, agent_key, map_code, seed, str(exc))
    finally:
        env.close()


def _error_result(sign_type_key, agent_key, map_code, seed, msg):
    return dict(
        sign_type=sign_type_key,
        agent=agent_key,
        map_code=map_code,
        seed=seed,
        total_violations=-1,
        violations_by_sign_class={},
        reached_destination=False,
        crashed=False,
        out_of_road=False,
        max_step=0,
        compliant_episode=False,
        max_speed_kmh=0.0,
        avg_speed_kmh=0.0,
        gif_path=None,
        trajectory=None,
        sign_meta={},
        error=msg,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "sign_type", "agent", "map_code", "seed", "total_violations",
    "violations_by_sign_class", "reached_destination", "crashed",
    "out_of_road", "max_step", "compliant_episode",
    "max_speed_kmh", "avg_speed_kmh", "gif_path", "error",
]


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            row = {k: v for k, v in r.items() if k in CSV_COLUMNS}
            row["violations_by_sign_class"] = json.dumps(
                r.get("violations_by_sign_class", {}), ensure_ascii=False
            )
            writer.writerow(row)


def build_summary(rows):
    by_group = defaultdict(list)
    for r in rows:
        key = f"{r['agent']}|{r['sign_type']}|{r['map_code']}"
        by_group[key].append(r)

    summary = {}
    for key, episodes in sorted(by_group.items()):
        valid = [e for e in episodes if e.get("error") is None]
        n = len(valid)
        if n == 0:
            summary[key] = dict(
                episodes=0, compliant=0, compliance_rate=0.0,
                avg_violations=0.0, crash_rate=0.0, dest_rate=0.0,
                avg_speed_kmh=0.0, max_speed_kmh=0.0,
            )
            continue
        n_compliant = sum(1 for e in valid if e["compliant_episode"])
        summary[key] = dict(
            episodes=n,
            compliant=n_compliant,
            compliance_rate=round(n_compliant / n, 4),
            avg_violations=round(float(np.mean([e["total_violations"] for e in valid])), 2),
            crash_rate=round(float(np.mean([e["crashed"] for e in valid])), 4),
            dest_rate=round(float(np.mean([e["reached_destination"] for e in valid])), 4),
            avg_speed_kmh=round(float(np.mean([e["avg_speed_kmh"] for e in valid])), 2),
            max_speed_kmh=round(float(np.max([e["max_speed_kmh"] for e in valid])), 2),
        )

    by_agent = defaultdict(list)
    for r in rows:
        by_agent[r["agent"]].append(r)

    agent_overall = {}
    for agent_key, episodes in sorted(by_agent.items()):
        valid = [e for e in episodes if e.get("error") is None]
        n = len(valid)
        if n == 0:
            agent_overall[agent_key] = dict(
                episodes=0, compliant=0, compliance_rate=0.0,
                avg_violations=0.0, crash_rate=0.0, dest_rate=0.0,
                avg_speed_kmh=0.0,
            )
            continue
        n_comp = sum(1 for e in valid if e["compliant_episode"])
        agent_overall[agent_key] = dict(
            episodes=n,
            compliant=n_comp,
            compliance_rate=round(n_comp / n, 4),
            avg_violations=round(float(np.mean([e["total_violations"] for e in valid])), 2),
            crash_rate=round(float(np.mean([e["crashed"] for e in valid])), 4),
            dest_rate=round(float(np.mean([e["reached_destination"] for e in valid])), 4),
            avg_speed_kmh=round(float(np.mean([e["avg_speed_kmh"] for e in valid])), 2),
        )

    all_valid = [r for r in rows if r.get("error") is None]
    n_all = len(all_valid)
    if n_all:
        n_comp = sum(1 for e in all_valid if e["compliant_episode"])
        overall = dict(
            episodes=n_all,
            compliant=n_comp,
            compliance_rate=round(n_comp / n_all, 4),
            avg_violations=round(float(np.mean([e["total_violations"] for e in all_valid])), 2),
            crash_rate=round(float(np.mean([e["crashed"] for e in all_valid])), 4),
            dest_rate=round(float(np.mean([e["reached_destination"] for e in all_valid])), 4),
            avg_speed_kmh=round(float(np.mean([e["avg_speed_kmh"] for e in all_valid])), 2),
        )
    else:
        overall = dict(
            episodes=0, compliant=0, compliance_rate=0.0,
            avg_violations=0.0, crash_rate=0.0, dest_rate=0.0,
            avg_speed_kmh=0.0,
        )

    return summary, agent_overall, overall


def print_summary_table(summary, agent_overall, overall):
    hdr = (
        f"{'Agent':<14} {'Sign':<12} {'Map':<7} "
        f"{'Eps':>4} {'Cmpl':>5} {'Rate':>7} "
        f"{'AvgViol':>8} {'Crash%':>7} {'Dest%':>7} {'AvgSpd':>7}"
    )
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for key, m in sorted(summary.items()):
        agent, sign, map_code = key.split("|")
        print(
            f"{agent:<14} {sign:<12} {map_code:<7} "
            f"{m['episodes']:>4} {m['compliant']:>5} "
            f"{m['compliance_rate']:>6.1%} "
            f"{m['avg_violations']:>8.2f} "
            f"{m['crash_rate']:>6.1%} "
            f"{m['dest_rate']:>6.1%} "
            f"{m['avg_speed_kmh']:>6.1f}"
        )
    print(sep)

    print(f"\n{'--- Per-Agent Totals ---':^{len(hdr)}}")
    for agent_key, m in sorted(agent_overall.items()):
        print(
            f"{agent_key:<14} {'ALL':<12} {'ALL':<7} "
            f"{m['episodes']:>4} {m['compliant']:>5} "
            f"{m['compliance_rate']:>6.1%} "
            f"{m['avg_violations']:>8.2f} "
            f"{m['crash_rate']:>6.1%} "
            f"{m['dest_rate']:>6.1%} "
            f"{m['avg_speed_kmh']:>6.1f}"
        )
    print(sep)

    o = overall
    print(
        f"{'OVERALL':<14} {'':12} {'':7} "
        f"{o['episodes']:>4} {o['compliant']:>5} "
        f"{o['compliance_rate']:>6.1%} "
        f"{o['avg_violations']:>8.2f} "
        f"{o['crash_rate']:>6.1%} "
        f"{o['dest_rate']:>6.1%} "
        f"{o['avg_speed_kmh']:>6.1f}"
    )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_sign_keys = sorted(SIGN_CATALOGUE.keys())
    all_agent_keys = sorted(AGENT_CATALOGUE.keys())
    all_map_keys = sorted(MAP_CODES.keys())

    parser = argparse.ArgumentParser(
        description="Validate RuleCompliantExpertPolicy vs baselines on TrafficSignEnv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Sign types : {', '.join(all_sign_keys)}\n"
            f"Agent modes: {', '.join(all_agent_keys)}\n"
            f"Map codes  : {', '.join(all_map_keys)}"
        ),
    )
    parser.add_argument(
        "--sign-types", type=str, default=None,
        help=f"Comma-separated sign type keys (default: {','.join(DEFAULT_SIGN_TYPES)}).",
    )
    parser.add_argument(
        "--agents", type=str, default=None,
        help=f"Comma-separated agent keys (default: {','.join(DEFAULT_AGENTS)}).",
    )
    parser.add_argument(
        "--maps", type=str, default=None,
        help=f"Comma-separated map codes (default: {','.join(DEFAULT_MAPS)}).",
    )
    parser.add_argument(
        "--seeds", type=str, default="0,1,2",
        help="Comma-separated random seeds per combination.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000,
        help="Maximum simulation steps per episode (default: 2000).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="rule_expert_output",
        help="Directory for CSV, JSON, and GIF outputs.",
    )
    parser.add_argument(
        "--render-gif", action="store_true", default=False,
        help="Record top-down frames and save a GIF for every episode.",
    )
    parser.add_argument(
        "--gif-duration", type=int, default=40,
        help="GIF frame duration in ms (default: 40 ~ 25 fps).",
    )
    parser.add_argument(
        "--traffic-density", type=float, default=0.0,
        help="NPC traffic density 0.0-1.0 (default: 0.0 = no traffic).",
    )
    parser.add_argument(
        "--ego-lane", type=str, default=None,
        help=(
            "Ego spawn lane number(s).  Examples: '0' (fixed lane 0), "
            "'0,1,2' (sample from these), 'random' or omit (env default)."
        ),
    )
    parser.add_argument(
        "--traffic-lanes", type=str, default=None,
        help=(
            "Traffic spawn lane hint (reserved for SUMO-based filtering). "
            "Currently logged but not enforced for PG maps."
        ),
    )
    parser.add_argument(
        "--ego-speed", type=str, default=None,
        help=(
            "Ego initial speed.  Examples: '30' (fixed 30 km/h), "
            "'20-40' (uniform sample in range), omit = stationary."
        ),
    )
    parser.add_argument(
        "--traffic-speed", type=str, default=None,
        help=(
            "Traffic initial speed.  Examples: '30' (fixed 30 km/h), "
            "'20-40' (sample range), omit = IDM default (start from 0)."
        ),
    )
    parser.add_argument(
        "--debug-spawn", action="store_true", default=False,
        help="Print spawn parameters (ego lane, speeds, density) per episode.",
    )
    parser.add_argument(
        "--fail-on-rate-below", type=float, default=None,
        help="Exit code 1 if rule_expert overall compliance < threshold.",
    )
    args = parser.parse_args()

    # --- Validate spawn specs ---
    try:
        ego_lane_spec = _parse_lane_spec(args.ego_lane)
    except ValueError as e:
        parser.error(f"--ego-lane: {e}")
    try:
        traffic_lane_spec = _parse_lane_spec(args.traffic_lanes)
    except ValueError as e:
        parser.error(f"--traffic-lanes: {e}")
    try:
        ego_speed_range = _parse_speed_spec(args.ego_speed)
    except ValueError as e:
        parser.error(f"--ego-speed: {e}")
    try:
        traffic_speed_range = _parse_speed_spec(args.traffic_speed)
    except ValueError as e:
        parser.error(f"--traffic-speed: {e}")
    if args.traffic_density < 0.0 or args.traffic_density > 1.0:
        parser.error("--traffic-density must be between 0.0 and 1.0")

    sign_types = (
        [s.strip() for s in args.sign_types.split(",") if s.strip()]
        if args.sign_types else DEFAULT_SIGN_TYPES
    )
    agents = (
        [s.strip() for s in args.agents.split(",") if s.strip()]
        if args.agents else DEFAULT_AGENTS
    )
    maps = (
        [s.strip() for s in args.maps.split(",") if s.strip()]
        if args.maps else DEFAULT_MAPS
    )
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gif_dir = out_dir / "gifs"
    if args.render_gif:
        gif_dir.mkdir(parents=True, exist_ok=True)
        print(f"GIF recording enabled -> {gif_dir}")

    total_episodes = len(agents) * len(sign_types) * len(maps) * len(seeds)
    print(f"Agents    : {agents}")
    print(f"Sign types: {sign_types}")
    print(f"Maps      : {maps}")
    print(f"Seeds     : {seeds}")
    print(f"Max steps : {args.max_steps}")
    print(f"Total episodes: {total_episodes}")

    if total_episodes == 0:
        print("Nothing to run.")
        return

    all_rows = []
    done_count = 0
    t0 = time.time()

    for agent_key in agents:
        for map_code in maps:
            for sign_key in sign_types:
                for seed in seeds:
                    done_count += 1
                    tag = (
                        f"[{done_count}/{total_episodes}] "
                        f"agent={agent_key} map={map_code} sign={sign_key} seed={seed}"
                    )
                    print(f"{tag} ...", end=" ", flush=True)

                    ep_gif_path = None
                    if args.render_gif:
                        ep_gif_path = str(
                            gif_dir / f"{agent_key}_{map_code}_{sign_key}_s{seed}.gif"
                        )

                    result = run_episode(
                        sign_type_key=sign_key,
                        agent_key=agent_key,
                        seed=seed,
                        max_steps=args.max_steps,
                        map_code=map_code,
                        render_gif=args.render_gif,
                        gif_path=ep_gif_path,
                        gif_duration=args.gif_duration,
                        traffic_density=args.traffic_density,
                        ego_lane_spec=ego_lane_spec,
                        traffic_lane_spec=traffic_lane_spec,
                        ego_speed_range=ego_speed_range,
                        traffic_speed_range=traffic_speed_range,
                        debug_spawn=args.debug_spawn,
                    )

                    status = "OK" if result["compliant_episode"] else "FAIL"
                    if result.get("error"):
                        status = "ERR"
                    viol = result["total_violations"]
                    spd = result["avg_speed_kmh"]
                    gif_note = ""
                    if result.get("gif_path"):
                        gif_note = f"  gif={Path(result['gif_path']).name}"
                    print(
                        f"{status}  viol={viol}  steps={result['max_step']}  "
                        f"avg_spd={spd:.1f}{gif_note}"
                    )

                    all_rows.append(result)

    elapsed = time.time() - t0
    print(f"\nCompleted {len(all_rows)} episodes in {elapsed:.1f}s")

    csv_path = out_dir / "compliance_episodes.csv"
    write_csv(all_rows, csv_path)
    print(f"CSV  -> {csv_path}")

    summary, agent_overall, overall = build_summary(all_rows)
    json_payload = {
        "per_group": summary,
        "per_agent": agent_overall,
        "overall": overall,
    }
    json_path = out_dir / "compliance_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)
    print(f"JSON -> {json_path}")

    print_summary_table(summary, agent_overall, overall)

    if args.fail_on_rate_below is not None:
        rc_rows = [r for r in all_rows if r["agent"] == "rule_expert"]
        rc_valid = [r for r in rc_rows if r.get("error") is None]
        if rc_valid:
            n_comp = sum(1 for e in rc_valid if e["compliant_episode"])
            rate = n_comp / len(rc_valid)
        else:
            rate = 0.0
        threshold = args.fail_on_rate_below
        if rate < threshold:
            print(
                f"\nFAIL: rule_expert compliance {rate:.1%} "
                f"< threshold {threshold:.1%}"
            )
            sys.exit(1)
        else:
            print(
                f"\nPASS: rule_expert compliance {rate:.1%} "
                f">= threshold {threshold:.1%}"
            )


if __name__ == "__main__":
    main()
