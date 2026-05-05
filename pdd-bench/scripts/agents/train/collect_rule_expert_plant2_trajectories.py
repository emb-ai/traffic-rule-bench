#!/usr/bin/env python3
"""
Collect Plant2 training trajectories using RuleCompliantExpertPolicy on TrafficSignEnv.

Environment and sign-spawning logic mirror
pdd-bench/scripts/validation/check_rule_compliant_expert.py exactly
(same path bootstrap, same sign catalogue, same route-aware placement helpers,
same _RESTRICTED_END / _BEGIN_TO_END handling).

Output format is the same as collect_metadrive_carl_plant2_trajectories.py so
both datasets can be combined for training:
  plant2_batch, target_speed,
  ego_pos_world_before/after, ego_heading_before/after,
  ego_pos_world_future_4, ego_pos_world_future_4_s3,
  action_env, reward, terminated, truncated, step_idx

Output files
  <output_dir>/rule_expert_plant2_traj_ep<NNN>.pt
  <output_dir>/gifs/rule_expert_plant2_traj_ep<NNN>.gif  (optional)

Usage
  python collect_rule_expert_plant2_trajectories.py
      --episodes 50
      --output-dir outputs/rule_expert_plant2_trajectories
      --maps X,T,S,O
      --sign-types stop,speed40,no_entry
"""

import logging
import os
import sys
import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Path setup — mirrors check_rule_compliant_expert.py exactly
# ---------------------------------------------------------------------------

def _find_pdd_bench_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for parent in (current, *current.parents):
        if (parent / "envs").is_dir() and (parent / "traffic_signs").is_dir():
            return parent
    raise RuntimeError("Could not locate pdd-bench root")


SCRIPT_PATH   = Path(__file__).resolve()
PDD_BENCH_DIR = _find_pdd_bench_root(SCRIPT_PATH)
SDC_ROOT      = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"
PLANT2_DIR    = SDC_ROOT / "plant2"

for _p in (PDD_BENCH_DIR, METADRIVE_DIR, PLANT2_DIR / "PlanT", PLANT2_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

if os.environ.get("SDL_VIDEODRIVER") is None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


# ---------------------------------------------------------------------------
# Project imports — identical to check_rule_compliant_expert.py
# ---------------------------------------------------------------------------

from envs.traffic_sign_env import TrafficSignEnv                              # type: ignore
from agents.policies.rule_compliant_expert import RuleCompliantExpertPolicy   # type: ignore

from metadrive.policy.metadrive_obs_to_plant2 import metadrive_obs_to_plant2_batch  # type: ignore

from traffic_signs.stop_sign import StopSign                                  # type: ignore
from traffic_signs.speed_limit_sign import (                                  # type: ignore
    SpeedLimitSign20, SpeedLimitSign30,
    SpeedLimitSign40, SpeedLimitSign60,
)
from traffic_signs.zone_signs import (                                        # type: ignore
    ZoneSpeedLimitSign20, ZoneSpeedLimitSign30,
    ZoneSpeedLimitSign40, ZoneSpeedLimitSign60,
)
from traffic_signs.end_of_zone_signs import (                                 # type: ignore
    EndOfSpeedLimitSign20, EndOfSpeedLimitSign30,
    EndOfSpeedLimitSign40, EndOfSpeedLimitSign60,
    EndOfZoneSpeedLimitSign20, EndOfZoneSpeedLimitSign30,
    EndOfZoneSpeedLimitSign40, EndOfZoneSpeedLimitSign60,
    EndOfAllRestrictionsSign,
)
from traffic_signs.min_speed_limit_sign import (                              # type: ignore
    MinimumSpeedLimit30, MinimumSpeedLimit40,
    MinimumSpeedLimit50, MinimumSpeedLimit60,
)
from traffic_signs.no_entry_sign import NoEntrySign                           # type: ignore
from traffic_signs.no_traffic_sign import NoTrafficSign                       # type: ignore
from traffic_signs.no_stopping_allowed_sign import NoStoppingAllowedSign      # type: ignore
from traffic_signs.detour_sign import (                                       # type: ignore
    DetourSign, DetourRightSign, DetourLeftSign, DetourEitherSign,
)
from traffic_signs.detour_obstacle import spawn_detour_obstacle               # type: ignore
from traffic_signs.restricted_lane_sign import (                              # type: ignore
    BusLaneRoadSign, BikeLaneRoadSign,
    EndBusLaneRoadSign, EndBikeLaneRoadSign,
    ExitToBusLaneSign, ExitToBusLaneSignLeft,
    ExitToBikeLaneSign, ExitToBikeLaneSignLeft,
    BusLaneSign, BikeLaneSign,
    EndBusLaneSign, EndBikeLaneSign,
)
from traffic_signs.bus_station_sign import BusStationSign                     # type: ignore
from traffic_signs.only_auto_sign import OnlyAutoSign                         # type: ignore
from traffic_signs.right_turn_rule import RightTurnRule                       # type: ignore
from traffic_signs.traffic_light_sign import TrafficLightSign                 # type: ignore
from traffic_signs.direction_sign import DirectionSign                        # type: ignore
from traffic_signs.pg_direction_sign import PGDirectionSign                   # type: ignore
from traffic_signs.no_turn_allowed import NoRightTurnSign, NoLeftTurnSign, NoUTurnSign  # type: ignore
from traffic_signs.one_way_entry_sign import OneWayEntrySignR, OneWayEntrySignL, OneWayEntrySignS  # type: ignore
# from traffic_signs.no_overtaking_sign import NoOvertakingSign                 # type: ignore


# ---------------------------------------------------------------------------
# Sign catalogue — identical to check_rule_compliant_expert.py
# ---------------------------------------------------------------------------

SIGN_CATALOGUE: Dict[str, Any] = {
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
    # --- End-of-zone (informational) ---
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
    # --- End of restricted lane (informational) ---
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
    # "no_overtaking":    NoOvertakingSign,
}

DEFAULT_SIGN_TYPES = [
    "stop", "speed40", "zone_speed40", "minspeed40",
    "no_entry", "no_traffic", "no_stop",
    "detour_right", "detour_left",
    "bus_lane_road", "bus_lane",
    "bus_station", "only_auto",
]

DEFAULT_MAPS = ["X", "T", "S", "O"]

# Sign placement groups — identical to check_rule_compliant_expert.py
_RESTRICTED_BEGIN_KEYS = {"bus_lane_road", "bike_lane_road", "bus_lane", "bike_lane"}
_RESTRICTED_END_KEYS   = {"end_bus_lane_road", "end_bike_lane_road",
                           "end_bus_lane", "end_bike_lane"}

_BEGIN_TO_END: Dict[str, Any] = {
    "bus_lane_road":  ("end_bus_lane_road",  EndBusLaneRoadSign),
    "bike_lane_road": ("end_bike_lane_road", EndBikeLaneRoadSign),
    "bus_lane":       ("end_bus_lane",       EndBusLaneSign),
    "bike_lane":      ("end_bike_lane",      EndBikeLaneSign),
}
_END_TO_BEGIN: Dict[str, Any] = {
    "end_bus_lane_road":  ("bus_lane_road",  BusLaneRoadSign),
    "end_bike_lane_road": ("bike_lane_road", BikeLaneRoadSign),
    "end_bus_lane":       ("bus_lane",       BusLaneSign),
    "end_bike_lane":      ("bike_lane",      BikeLaneSign),
}

_DETOUR_KEYS      = {"detour_right", "detour_left", "detour_either"}
_LANE_CHANGE_KEYS = {"no_entry", "no_traffic"}


# ---------------------------------------------------------------------------
# Route-aware sign placement helpers — identical to check_rule_compliant_expert.py
# ---------------------------------------------------------------------------

def _get_route_lanes(env) -> List:
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
            route_lanes.extend(road_network.graph[ckpt_start][ckpt_end])
        except KeyError:
            continue
    return route_lanes


def _lane_has_continuation(lane, road_network) -> bool:
    idx = getattr(lane, "index", None)
    if idx is None or not (isinstance(idx, tuple) and len(idx) >= 2):
        return True
    graph    = getattr(road_network, "graph", None)
    outgoing = graph.get(idx[1]) if graph is not None else None
    if outgoing is None or (isinstance(outgoing, dict) and len(outgoing) == 0):
        return False
    return True


def _pick_route_lane(route_lanes, min_length=10.0, road_network=None):
    candidates = [l for l in route_lanes if l.length >= min_length]
    if road_network is not None:
        candidates = [l for l in candidates if _lane_has_continuation(l, road_network)]
    return random.choice(candidates) if candidates else None


def _pick_rightmost_lane(route_lanes, min_length=30.0):
    segments: Dict = {}
    for lane in route_lanes:
        idx = getattr(lane, "index", None)
        if not (isinstance(idx, tuple) and len(idx) >= 3):
            continue
        segments.setdefault((idx[0], idx[1]), {})[idx[2]] = lane
    candidates = []
    for lanes_by_num in segments.values():
        num_lanes = max(lanes_by_num) + 1
        if num_lanes < 2:
            continue
        lane = lanes_by_num[max(lanes_by_num)]
        if lane.length >= min_length:
            candidates.append(lane)
    return random.choice(candidates) if candidates else None


def _pick_detour_lane(route_lanes, sign_class, min_length=30.0):
    directions = getattr(sign_class, "allowed_directions", set())
    segments: Dict = {}
    for lane in route_lanes:
        idx = getattr(lane, "index", None)
        if not (isinstance(idx, tuple) and len(idx) >= 3):
            continue
        segments.setdefault((idx[0], idx[1]), {})[idx[2]] = lane
    candidates = []
    for lanes_by_num in segments.values():
        num_lanes = max(lanes_by_num) + 1
        for lane_num, lane in lanes_by_num.items():
            if lane.length < min_length or lane_num <= 0:
                continue
            if "right" in directions and lane_num >= num_lanes - 1:
                continue
            if "left" in directions and lane_num <= 0:
                continue
            candidates.append(lane)
    return random.choice(candidates) if candidates else None


def _pick_lane_for_lane_change(route_lanes, vehicle_lane_num=0,
                                min_length=15.0, road_network=None):
    segments: Dict = {}
    for lane in route_lanes:
        idx = getattr(lane, "index", None)
        if not (isinstance(idx, tuple) and len(idx) >= 3):
            continue
        segments.setdefault((idx[0], idx[1]), {})[idx[2]] = lane
    candidates = []
    for lanes_by_num in segments.values():
        if len(lanes_by_num) < 2 or vehicle_lane_num not in lanes_by_num:
            continue
        lane = lanes_by_num[vehicle_lane_num]
        if lane.length < min_length:
            continue
        if road_network is not None and not _lane_has_continuation(lane, road_network):
            continue
        candidates.append(lane)
    return random.choice(candidates) if candidates else None


def spawn_sign_on_route(env, sign_key: str) -> Optional[Any]:
    """Spawn one sign on the agent's route.

    Logic mirrors check_rule_compliant_expert.py::run_episode sign-spawning
    block, including restricted-end / restricted-begin paired placement and
    detour-obstacle spawning.
    """
    sign_class = SIGN_CATALOGUE.get(sign_key)
    if sign_class is None:
        print(f"[WARN] Unknown sign type: {sign_key}")
        return None

    sign_mgr     = env.engine.traffic_sign_manager
    road_network = env.current_map.road_network
    route_lanes  = _get_route_lanes(env)

    is_detour           = sign_key in _DETOUR_KEYS
    is_restricted_begin = sign_key in _RESTRICTED_BEGIN_KEYS
    is_restricted_end   = sign_key in _RESTRICTED_END_KEYS
    is_lane_change      = sign_key in _LANE_CHANGE_KEYS

    try:
        lane = None
        if route_lanes:
            if is_detour:
                lane = _pick_detour_lane(route_lanes, sign_class, min_length=30.0)
            elif is_lane_change:
                veh_idx      = getattr(env.vehicle.lane, "index", None)
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
            _bkey, begin_cls = _END_TO_BEGIN[sign_key]
            if lane is not None:
                sign_mgr.add_sign(begin_cls, lane=lane, use_random_lane=False)
                sign = sign_mgr.add_sign(sign_class, lane=lane, use_random_lane=False)
            else:
                sign_mgr.add_sign(begin_cls, use_random_lane=True)
                sign = sign_mgr.add_sign(sign_class, use_random_lane=True)

        # --- Restricted-begin: auto-append end sign ---
        elif is_restricted_begin:
            sign = (sign_mgr.add_sign(sign_class, lane=lane, use_random_lane=False)
                    if lane else sign_mgr.add_sign(sign_class, use_random_lane=True))
            if sign_key in _BEGIN_TO_END:
                _ekey, end_cls = _BEGIN_TO_END[sign_key]
                sign_mgr.add_sign(end_cls, lane=sign.lane, use_random_lane=False)

        # --- Default ---
        else:
            sign = (sign_mgr.add_sign(sign_class, lane=lane, use_random_lane=False)
                    if lane else sign_mgr.add_sign(sign_class, use_random_lane=True))

        # Spawn traffic cones at the obstacle point for detour signs.
        if is_detour and sign is not None:
            spawn_detour_obstacle(env.engine, sign.lane, sign)

        return sign

    except Exception as exc:
        print(f"[WARN] sign spawning failed ({sign_key}): {exc}")
        return None


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def create_env(map_code: str, seed: int, traffic_density: float,
               horizon: int) -> TrafficSignEnv:
    return TrafficSignEnv(dict(
        map=map_code,
        start_seed=seed,
        agent_policy=RuleCompliantExpertPolicy,
        use_render=False,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.CRITICAL,
        traffic_density=traffic_density,
        horizon=horizon,
        random_lane_width=True,
        random_lane_num=True,
        vehicle_config={"show_lidar": False},
        # Reward shaping consistent with CaRL collect script
        use_lateral_reward=True,
        success_reward=20.0,
        out_of_road_penalty=15.0,
        crash_vehicle_penalty=25.0,
        crash_object_penalty=20.0,
        crash_sidewalk_penalty=5.0,
        driving_reward=0.5,
        speed_reward=0.05,
    ))


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_trajectories(
    output_dir: str,
    num_episodes: int = 50,
    max_steps: int = 500,
    seed: int = 42,
    maps: Optional[List[str]] = None,
    sign_types: Optional[List[str]] = None,
    traffic_density: float = 0.0,
    save_gifs: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    gif_dir = os.path.join(output_dir, "gifs")
    if save_gifs:
        os.makedirs(gif_dir, exist_ok=True)

    maps       = maps       or DEFAULT_MAPS
    sign_types = sign_types or DEFAULT_SIGN_TYPES

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for ep in range(num_episodes):
        ep_seed  = seed + ep
        map_code = maps[ep % len(maps)]
        sign_key = sign_types[ep % len(sign_types)]

        random.seed(ep_seed)
        np.random.seed(ep_seed)

        print(f"\n{'='*60}")
        print(f"Episode {ep+1}/{num_episodes}  map={map_code}  sign={sign_key}  seed={ep_seed}")
        print(f"{'='*60}")

        env = create_env(map_code, ep_seed, traffic_density, max_steps)

        try:
            obs, _info = env.reset()

            vehicle      = env.vehicle
            engine       = env.engine
            road_network = env.current_map.road_network
            sign_mgr     = engine.traffic_sign_manager

            # Spawn sign on the agent's route
            spawn_sign_on_route(env, sign_key)

            # Collect all spawned sign world positions for episode metadata
            sign_world_positions: List[np.ndarray] = []
            for s in sign_mgr.signs:
                if hasattr(s, "position"):
                    sign_world_positions.append(
                        np.array(s.position, dtype=np.float32))

            # Clear renderer frames for GIF recording
            if save_gifs:
                renderer = getattr(env, "top_down_renderer", None)
                if renderer and hasattr(renderer, "_screen_frames"):
                    renderer._screen_frames.clear()

            ep_data: Dict[str, Any] = {
                "episode_index":        ep,
                "reset_seed":           ep_seed,
                "base_seed":            seed,
                "map_code":             map_code,
                "sign_type":            sign_key,
                "steps":                [],
                "road_network":         road_network,
                "sign_world_positions": sign_world_positions,
            }
            ep_reward = 0.0

            for step_i in range(max_steps):
                if vehicle is None:
                    break

                # ---- Build plant2 batch BEFORE the step ----
                plant2_batch = metadrive_obs_to_plant2_batch(
                    engine,
                    vehicle,
                    route_ego_20x2=None,
                    speed_limit_kmh=None,
                    max_objects=30,
                    max_distance=75.0,
                    range_factor_front=16.0,
                    input_bev=True,
                    input_ego_speed=True,
                    bev_resolution=128,
                    bev_size_meters=64.0,
                    device="cpu",
                )

                # Serialize tensors → numpy for storage
                plant2_batch_save: Dict[str, Any] = {
                    k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                    for k, v in plant2_batch.items()
                }
                # Ego speed supervision target — current speed after all prior
                # sign-compliance actions by the expert policy.
                plant2_batch_save["target_speed"] = np.array(
                    [[float(getattr(vehicle, "speed", 0.0))]], dtype=np.float32)

                ego_pos_before     = np.asarray(vehicle.position[:2], dtype=np.float32)
                ego_heading_before = float(getattr(vehicle, "heading_theta", 0.0))

                # ---- Step — RuleCompliantExpertPolicy acts internally ----
                obs, reward, terminated, truncated, info = env.step(
                    np.zeros(2, dtype=np.float32))
                ep_reward += float(reward)

                # Actual action taken by the policy this step
                action_env = np.asarray(
                    getattr(vehicle, "current_action", [0.0, 0.0]),
                    dtype=np.float32)

                ego_pos_after     = np.asarray(vehicle.position[:2], dtype=np.float32)
                ego_heading_after = float(getattr(vehicle, "heading_theta", 0.0))

                # ---- Render for GIF ----
                if save_gifs:
                    try:
                        env.render(mode="top_down", screen_record=True,
                                   window=False, screen_size=(640, 640))
                    except Exception:
                        pass

                ep_data["steps"].append({
                    "plant2_batch":         plant2_batch_save,
                    "ego_pos_world_before": ego_pos_before,
                    "ego_heading_before":   ego_heading_before,
                    "ego_pos_world_after":  ego_pos_after,
                    "ego_heading_after":    ego_heading_after,
                    "action_env":           action_env,
                    "reward":               float(reward),
                    "terminated":           bool(terminated),
                    "truncated":            bool(truncated),
                    "step_idx":             np.array([float(step_i)], dtype=np.float32),
                })

                if terminated or truncated:
                    print(f"  Ended at step {step_i+1}"
                          f"  terminated={terminated}  truncated={truncated}")
                    break

            # ---- Compute future-position supervision targets ----
            # ego_pos_world_future_4    — 4 consecutive steps (reference)
            # ego_pos_world_future_4_s3 — stride-3 steps (used for wps loss)
            steps_list = ep_data["steps"]
            n = len(steps_list)
            if n > 0:
                pos_after = [
                    np.asarray(s["ego_pos_world_after"], dtype=np.float32)
                    for s in steps_list
                ]
                for i in range(n):
                    steps_list[i]["ego_pos_world_future_4"] = np.stack(
                        [pos_after[min(i + k, n - 1)] for k in range(4)]
                    ).astype(np.float32)
                    steps_list[i]["ego_pos_world_future_4_s3"] = np.stack(
                        [pos_after[min(i + k * 3, n - 1)] for k in range(1, 5)]
                    ).astype(np.float32)

            ep_data["return"]    = ep_reward
            ep_data["num_steps"] = n

            # ---- Save trajectory ----
            out_path = os.path.join(
                output_dir, f"rule_expert_plant2_traj_ep{ep:03d}.pt")
            torch.save(ep_data, out_path)
            print(f"  Saved  {out_path}  (steps={n}, return={ep_reward:.2f})")

            # ---- Save GIF ----
            if save_gifs:
                gif_path = os.path.join(
                    gif_dir,
                    f"rule_expert_plant2_traj_ep{ep:03d}_{map_code}_{sign_key}.gif")
                try:
                    renderer = getattr(env, "top_down_renderer", None)
                    if renderer is not None:
                        renderer.generate_gif(gif_path, duration=10)
                        print(f"  GIF    {gif_path}")
                except Exception as exc:
                    print(f"  [WARN] GIF failed: {exc}")

        except Exception as exc:
            import traceback
            print(f"  [ERROR] Episode {ep} failed: {exc}")
            traceback.print_exc()

        finally:
            env.close()

    print("\nDone. All trajectories saved to:", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    all_sign_keys = sorted(SIGN_CATALOGUE.keys())

    parser = argparse.ArgumentParser(
        description=(
            "Collect Plant2 training trajectories with RuleCompliantExpertPolicy "
            "on TrafficSignEnv (env mirrors check_rule_compliant_expert.py)."
        ),
        epilog=f"Available sign types: {', '.join(all_sign_keys)}",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PDD_BENCH_DIR / "outputs" / "rule_expert_plant2_trajectories"),
    )
    parser.add_argument("--episodes",        type=int,   default=50)
    parser.add_argument("--max-steps",       type=int,   default=500)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument(
        "--maps", type=str, default=",".join(DEFAULT_MAPS),
        help=f"Comma-separated map codes to cycle through (default: {','.join(DEFAULT_MAPS)})",
    )
    parser.add_argument(
        "--sign-types", type=str, default=",".join(DEFAULT_SIGN_TYPES),
        help="Comma-separated sign type keys to cycle through",
    )
    parser.add_argument("--traffic-density", type=float, default=0.0)
    parser.add_argument("--no-gifs",         action="store_true")
    args = parser.parse_args()

    maps       = [m.strip() for m in args.maps.split(",")       if m.strip()]
    sign_types = [s.strip() for s in args.sign_types.split(",") if s.strip()]

    bad_maps  = [m for m in maps       if m not in MAP_CODES]
    bad_signs = [s for s in sign_types if s not in SIGN_CATALOGUE]
    if bad_maps:
        parser.error(f"Unknown map codes: {bad_maps}. Valid: {sorted(MAP_CODES)}")
    if bad_signs:
        parser.error(f"Unknown sign types: {bad_signs}. Valid: {all_sign_keys}")

    print("=" * 70)
    print("RuleCompliantExpert Plant2 trajectory collection")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Episodes       : {args.episodes}")
    print(f"  Max steps/ep   : {args.max_steps}")
    print(f"  Base seed      : {args.seed}")
    print(f"  Maps           : {maps}")
    print(f"  Sign types     : {sign_types}")
    print(f"  Traffic density: {args.traffic_density}")
    print(f"  Save GIFs      : {not args.no_gifs}")
    print("=" * 70)

    collect_trajectories(
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        maps=maps,
        sign_types=sign_types,
        traffic_density=args.traffic_density,
        save_gifs=not args.no_gifs,
    )


# Keep MAP_CODES consistent with check_rule_compliant_expert.py
MAP_CODES = {"X": "X", "T": "T", "S": "S", "O": "O"}

if __name__ == "__main__":
    main()
