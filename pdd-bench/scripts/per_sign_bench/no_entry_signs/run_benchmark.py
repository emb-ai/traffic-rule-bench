from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from envs.sumo_env import TrafficSignSumoEnv
from envs.sumo_traffic_manager import SumoTrafficManager
from agents.policies.comprehensive_rule_expert import ComprehensiveRuleExpertPolicy
from agents.policies.modified_idm_sign_compliant import ModifiedIDMSignCompliantPolicy
from agents.policies.rule_compliant_expert import RuleCompliantExpertPolicy
from metadrive.policy.idm_policy import IDMPolicy, ModifiedIDMPolicy
from metadrive.policy.expert_policy import ExpertPolicy
from scripts.per_sign_bench.factorized_space.ego_defaults import (
    apply_ego_defaults,
    apply_ego_sampled,
    sample_ego_params,
)
from traffic_signs.no_entry_sign import NoEntrySign
from traffic_signs.no_traffic_sign import NoTrafficSign
from lib.lane_keys import make_lane_key
from lib.no_entry_sign_spec import (
    DEFAULT_PDD_CODE,
    get_no_entry_sign_spec,
    resolve_sign_class,
)
from lib.junction_sign_placement import (
    lateral_offset_beside_lane,
    resolve_sign_lane_for_edge,
    sign_longitudinal_offset_from_start,
    sign_placement_long_from_start,
)
from lib.manifest_config import (
    enrich_manifest_row,
    load_manifest_config,
)

# Defaults for compliant-stop early success (overridable via manifest / config).
DEFAULT_COMPLIANT_STOP_SUCCESS_SECONDS = 3.0
DEFAULT_COMPLIANT_STOP_MAX_DIST_M = 12.0
DEFAULT_COMPLIANT_STOP_SPEED_MPS = 0.5
DEFAULT_DESTINATION_PAST_SIGN_M = 8.0
DEFAULT_SIGN_DISTANCE_FROM_START_M = 10.0

BENCH_DIR = Path(__file__).resolve().parent
PER_SIGN_BENCH_DIR = BENCH_DIR.parent
PDD_BENCH_DIR = PER_SIGN_BENCH_DIR.parent.parent
SDC_ROOT = PDD_BENCH_DIR.parent

_SUMO_SIGN_DISTANCE_CACHE: dict[Path, float] = {}
_PROFILE_KEYS = (
    "NORMAL_SPEED",
    "MAX_SPEED",
    "CREEP_SPEED",
    "ACC_FACTOR",
    "DEACC_FACTOR",
    "DISTANCE_WANTED",
    "TIME_WANTED",
    "LANE_CHANGE_FREQ",
    "traffic_density",
    "horizon_steps",
)


def _slug_to_code(slug: str) -> str:
    return slug.replace("_", ".")


def _manifest_profile(row: dict) -> dict:
    profile: dict = {}
    for key in _PROFILE_KEYS:
        if f"profile_{key}" in row:
            profile[key] = row[f"profile_{key}"]
        elif key in row:
            profile[key] = row[key]
    return profile


def _manifest_traffic_density(row: dict, default: float) -> float:
    profile = _manifest_profile(row)
    val = profile.get("traffic_density", default)
    return float(val)


def _manifest_horizon(row: dict, fallback: int) -> int:
    profile = _manifest_profile(row)
    val = profile.get("horizon_steps", fallback)
    return int(val)


def _apply_manifest_profile_to_npcs(row: dict) -> None:
    profile = _manifest_profile(row)
    if not profile:
        return
    from scripts.per_sign_bench.factorized_space.agent_profile_bank import apply_profile_to_idm_class

    apply_profile_to_idm_class(profile)


def _load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _load_enriched_manifest_rows(path: Path) -> list[dict]:
    config = load_manifest_config(path)
    return [enrich_manifest_row(row, config) for row in _load_jsonl_rows(path)]


def _choose_manifest(code_dir: Path) -> Path | None:
    """Find real_manifest.jsonl for SUMO scenes."""
    p1 = code_dir / "sumo" / "sumo_manifest.jsonl"
    p2 = code_dir / "real_manifest.jsonl"
    if p1.exists() and p1.stat().st_size > 0:
        return p1
    if p2.exists() and p2.stat().st_size > 0:
        return p2
    return None


def collect_rows(
    benchmark_output_dir: Path,
    only_codes: set[str],
    max_scenes_per_sign: int | None,
    unique_scene_id: bool = False,
) -> list[dict]:
    """Iterate manifests and collect rows for evaluation (SUMO/real maps only)."""
    rows: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    seen_scene_ids: set[str] = set()

    sign_dirs = sorted([d for d in benchmark_output_dir.iterdir() if d.is_dir() and d.name[:1].isdigit()])
    for sign_dir in sign_dirs:
        sign_code = _slug_to_code(sign_dir.name)
        if only_codes and sign_code not in only_codes:
            continue

        manifest = _choose_manifest(sign_dir)
        if manifest is None:
            continue

        config = load_manifest_config(manifest)
        for row in _load_jsonl_rows(manifest):
            row = enrich_manifest_row(row, config)
            if "valid" in row and not row["valid"]:
                continue
            if unique_scene_id:
                sid_key = str(row.get("scene_id") or "")
                if sid_key in seen_scene_ids:
                    continue
                seen_scene_ids.add(sid_key)
            if max_scenes_per_sign is not None and counts[sign_code] >= max_scenes_per_sign:
                continue
            row["_backend"] = "sumo"
            row["_sign_code"] = sign_code
            rows.append(row)
            counts[sign_code] += 1

    return rows


def _build_sumo_env(row: dict, scenes_root: Path, max_steps: int) -> TrafficSignSumoEnv:
    # Keep background traffic off the ego approach so spawn/T-bone pileups
    # don't start the episode already blocked.
    SumoTrafficManager.EGO_SAFE_RADIUS = 30
    _apply_manifest_profile_to_npcs(row)
    traffic_density = _manifest_traffic_density(row, default=0.0)
    horizon = _manifest_horizon(row, fallback=max_steps)
    net_path = str(scenes_root / row["net_path"]) if not str(row["net_path"]).startswith("/") else str(row["net_path"])
    sign_spawn_distance = _resolve_sign_spawn_distance(row, scenes_root)

    vehicle_config: dict = {"show_lidar": False}
    spawn_vel = float(row.get("spawn_velocity_ms", 0.0) or 0.0)
    if spawn_vel > 0:
        vehicle_config["spawn_velocity"] = [spawn_vel, 0.0]
        vehicle_config["spawn_velocity_car_frame"] = True

    config = dict(
        use_render=False,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.CRITICAL,
        map_name=net_path,
        sign_type=row.get("sign_code") or row.get("sign_type"),
        traffic_density=traffic_density,
        tl_speed_factor=float(row.get("tl_speed_factor", 20.0)),
        sign_spawn_distance=sign_spawn_distance,
        min_route_hops_after_spawn=int(row.get("min_route_hops_after_spawn", 10)),
        max_route_hops_after_spawn=int(row.get("max_route_hops_after_spawn", 10)),
        horizon=horizon,
        num_scenarios=100000,
        vehicle_config=vehicle_config,
        debug_one_way_sign_selection=bool(row.get("debug_one_way_sign_selection", False)),
        show_lane_arrows=row.get("show_lane_arrows", False),
        show_traffic_lights=row.get("show_traffic_lights", False),
        show_npc_vehicles=row.get("show_npc_vehicles", False),
        skip_auto_signs=True,
        use_pedestrian_manager=False,
        use_pedestrian_yield_rule=False,
        # NPCs brake for ego in a 15 m front hemisphere so T-bones don't
        # fail the 4.1.x skill check for the planner under test.
        npc_ego_yield_radius=15.0,
    )
    if row.get("road_id"):
        config["vehicle_config"]["spawn_lane_index"] = row["road_id"]
    if "spawn_lane_num" in row:
        config["spawn_lane_num"] = int(row["spawn_lane_num"])
    if row.get("destination_lane_id"):
        config["vehicle_config"]["destination"] = row["destination_lane_id"]

    class _RealMapEnv(TrafficSignSumoEnv):
        @classmethod
        def default_config(cls):
            cfg = super().default_config()
            cfg["traffic_density"] = 0.0
            cfg["show_lane_arrows"] = True
            cfg["show_traffic_lights"] = True
            cfg["show_npc_vehicles"] = True
            cfg["skip_auto_signs"] = False
            return cfg

        def setup_engine(self):
            super().setup_engine()
            # Only add SumoTrafficManager if traffic_density > 0
            # Otherwise keep the default SimpleTrafficManager (no NPC spawning)
            if self.config.get("traffic_density", 0.0) > 0:
                self.engine.update_manager("traffic_manager", SumoTrafficManager())

        def reset(self, *, seed=None):
            # Skip TrafficSignSumoEnv.reset() sign creation by calling grandparent directly
            if self.config.get("skip_auto_signs", False):
                # Call BaseEnv.reset() directly, skipping TrafficSignSumoEnv.reset()
                from metadrive.envs import BaseEnv
                obs, info = BaseEnv.reset(self, seed=seed)
                return obs, info
            else:
                return super().reset(seed=seed)

    return _RealMapEnv(config)


def _resolve_sign_spawn_distance(row: dict, scenes_root: Path) -> float:
    """Soft env hint for MetaDrive map config (placement uses before_end offset)."""
    direct = row.get("sign_spawn_distance")
    if direct is not None:
        return max(float(direct), 30.0)

    direct = row.get("distance_from_start")
    if direct is not None:
        return max(float(direct), 30.0)

    net_path = row.get("net_path")
    if not net_path:
        return 30.0

    net_file = Path(str(net_path))
    scene_dir = (scenes_root / net_file).parent if not net_file.is_absolute() else net_file.parent
    meta_path = scene_dir / "meta.json"

    if meta_path in _SUMO_SIGN_DISTANCE_CACHE:
        return max(_SUMO_SIGN_DISTANCE_CACHE[meta_path], 30.0)

    distance = 0.0
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("sign_spawn_distance") is not None:
                distance = float(meta["sign_spawn_distance"])
            elif meta.get("distance_from_start") is not None:
                distance = float(meta["distance_from_start"])
        except Exception:
            distance = 0.0

    _SUMO_SIGN_DISTANCE_CACHE[meta_path] = distance
    return max(distance, 30.0)


def _wrap_for_policy(env, policy_type: str):
    return env


def _format_violation(sign, vehicle):
    sign_name = type(sign).__name__
    lane = getattr(sign, "lane", None)
    lane_idx = getattr(lane, "index", None)
    intersection = getattr(sign, "intersection_name", None)
    parts = [f"{sign_name}"]
    if intersection:
        parts.append(f"J:{intersection}")
    if lane_idx is not None:
        parts.append(f"L:{lane_idx}")
    try:
        if lane is not None:
            veh_long = float(lane.local_coordinates(vehicle.position)[0])
            dist = float(lane.length - veh_long)
            parts.append(f"d={dist:.1f}m")
    except Exception:
        pass
    return " | ".join(parts)


def _violation_bucket(sign_obj) -> str:
    name = type(sign_obj).__name__.lower()
    if "trafficlight" in name or "traffic_light" in name or "light" in name:
        return "traffic_light"
    if "crosswalk" in name or "pedestrian" in name or "zebra" in name:
        return "crosswalk"
    return "sign"


def _on_same_road(lane_a, lane_b) -> bool:
    idx_a = getattr(lane_a, "index", None)
    idx_b = getattr(lane_b, "index", None)
    if idx_a is None or idx_b is None:
        return False
    if isinstance(idx_a, str) and isinstance(idx_b, str):
        return idx_a.rsplit("_", 1)[0] == idx_b.rsplit("_", 1)[0]
    try:
        return idx_a[0] == idx_b[0] and idx_a[1] == idx_b[1]
    except (IndexError, TypeError):
        return False


_IN_ZONE_LOOKAHEAD_M = 50.0


def _ego_in_sign_zone(sign, vehicle) -> bool:
    lane = getattr(sign, "lane", None)
    if lane is None:
        return False
    veh_lane = getattr(vehicle, "lane", None)
    if veh_lane is None:
        return False
    if not _on_same_road(veh_lane, lane):
        return False
    try:
        veh_long = float(lane.local_coordinates(vehicle.position)[0])
    except Exception:
        return False

    zone_start = getattr(sign, "zone_start", None)
    zone_end = getattr(sign, "zone_end", None)
    if zone_start is not None and zone_end is not None:
        if float(zone_start) <= veh_long <= float(zone_end):
            return True
        if veh_long < float(zone_start) and (float(zone_start) - veh_long) < _IN_ZONE_LOOKAHEAD_M:
            return True
        return False

    anchor = (getattr(sign, "stop_line_position", None)
              or getattr(sign, "placement_long", None))
    if anchor is not None:
        anchor = float(anchor)
        dist = anchor - veh_long
        if 0 <= dist < _IN_ZONE_LOOKAHEAD_M:
            return True
        if -5.0 < dist < 0:
            return True
        return False

    return False


def _unwrap_base_env(env):
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    return base_env


def _extract_sign_info(env) -> list[dict]:
    signs = []
    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return signs
    for s in sign_mgr.signs:
        lane = getattr(s, "lane", None)
        lane_index = list(getattr(lane, "index", ())) if lane is not None else None
        pos = None
        try:
            pos = [float(s.position[0]), float(s.position[1])]
        except Exception:
            pass
        signs.append({
            "sign_class": type(s).__name__,
            "lane_index": lane_index,
            "longitudinal_offset": float(getattr(s, "longitudinal_offset", 0.0)),
            "lateral_offset": float(getattr(s, "lateral_offset", 0.0)),
            "position_world": pos,
        })
    return signs


def _ego_at_fault_for_crash(ego, engine, contact_dist: float = 4.0) -> bool:
    import math as _m
    try:
        ego_pos = ego.position
        ego_heading = ego.heading_theta
        ego_speed_kmh = float(getattr(ego, "speed_km_h", 0.0))
    except Exception:
        return True
    cos_h, sin_h = _m.cos(ego_heading), _m.sin(ego_heading)
    try:
        objs = engine.get_objects(lambda o: o is not ego).values()
    except Exception:
        return True
    from metadrive.component.vehicle.base_vehicle import BaseVehicle
    for obj in objs:
        if not isinstance(obj, BaseVehicle):
            continue
        try:
            dx = obj.position[0] - ego_pos[0]
            dy = obj.position[1] - ego_pos[1]
            dist = _m.hypot(dx, dy)
        except Exception:
            continue
        if dist > contact_dist:
            continue
        rel_x = cos_h * dx + sin_h * dy
        if rel_x > 0.0 and ego_speed_kmh > 0.5:
            return True
    return False


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _route_completion_percent(info: dict, reached_dest: bool) -> float:
    candidates = (
        "route_completion",
        "route_completion_rate",
        "route_completion_ratio",
        "route_completion_percentage",
    )
    for k in candidates:
        if k in info:
            v = _safe_float(info.get(k), 0.0)
            if v <= 1.0:
                v *= 100.0
            return max(0.0, min(100.0, v))
    return 100.0 if reached_dest else 0.0


def _route_length_m(info: dict) -> float | None:
    candidates = (
        "route_length_m",
        "route_length",
        "route_total_length",
        "route_distance_m",
        "episode_route_length",
    )
    for k in candidates:
        if k in info:
            try:
                v = float(info.get(k))
            except Exception:
                continue
            if math.isfinite(v) and v >= 0.0:
                return v
    return None


def _infraction_penalty(crashed: bool, out_of_road: bool, violations: int) -> float:
    p = 1.0
    if crashed:
        p *= 0.5
    if out_of_road:
        p *= 0.7
    if violations > 0:
        p *= (0.9 ** int(violations))
    return max(0.0, min(1.0, p))


def _nearby_speed_percentage(vehicle) -> float | None:
    try:
        nearby = vehicle.lidar.get_surrounding_objects(vehicle)
    except Exception:
        return None

    speeds = []
    for obj in nearby:
        if obj is vehicle:
            continue
        s = None
        if hasattr(obj, "speed_km_h"):
            s = _safe_float(getattr(obj, "speed_km_h"), 0.0)
        elif hasattr(obj, "speed"):
            s = _safe_float(getattr(obj, "speed"), 0.0) * 3.6
        if s is not None and s > 0.5:
            speeds.append(s)

    if not speeds:
        return None
    avg = float(np.mean(speeds))
    if avg <= 1e-3:
        return None

    ego = _safe_float(getattr(vehicle, "speed_km_h", 0.0), 0.0)
    pct = 100.0 * ego / avg
    if pct > 1000.0:
        return None
    return float(pct)


def _min_ttc_seconds(vehicle) -> float | None:
    try:
        nearby = vehicle.lidar.get_surrounding_objects(vehicle)
        ego_pos = np.asarray(vehicle.position, dtype=np.float64)
        ego_speed = _safe_float(getattr(vehicle, "speed", 0.0), 0.0)
        ego_heading = _safe_float(getattr(vehicle, "heading_theta", 0.0), 0.0)
    except Exception:
        return None

    ego_dir = np.array([math.cos(ego_heading), math.sin(ego_heading)], dtype=np.float64)
    ego_vel = ego_dir * ego_speed

    best = None
    for obj in nearby:
        if obj is vehicle:
            continue
        try:
            rel = np.asarray(obj.position, dtype=np.float64) - ego_pos
        except Exception:
            continue
        dist = float(np.linalg.norm(rel))
        if dist < 1e-3 or dist > 60.0:
            continue

        rel_along = float(np.dot(rel, ego_dir))
        if rel_along <= 0.0:
            continue

        obj_speed = _safe_float(getattr(obj, "speed", 0.0), 0.0)
        obj_heading = _safe_float(getattr(obj, "heading_theta", 0.0), ego_heading)
        obj_vel = np.array([math.cos(obj_heading), math.sin(obj_heading)], dtype=np.float64) * obj_speed
        rel_vel = ego_vel - obj_vel
        closing = float(np.dot(rel_vel, ego_dir))
        if closing <= 1e-3:
            continue
        ttc = rel_along / closing
        if ttc < 0.0:
            continue
        if best is None or ttc < best:
            best = ttc
    return best


def _compute_smoothness(step_vars: list[dict], segment_len: int = 20) -> dict:
    if not step_vars:
        return {
            "smoothness_ratio": 0.0,
            "smooth_segments": 0,
            "total_segments": 0,
            "frame_smooth_ratio": 0.0,
        }

    def _frame_ok(v: dict) -> bool:
        return (
            -4.05 <= v["long_acc"] <= 2.40
            and abs(v["lat_acc"]) <= 4.89
            and abs(v["yaw_rate"]) <= 0.95
            and abs(v["yaw_acc"]) <= 1.93
            and abs(v["long_jerk"]) <= 4.13
            and abs(v["jerk_mag"]) <= 8.37
        )

    frame_flags = [_frame_ok(v) for v in step_vars]
    frame_smooth_ratio = float(np.mean(frame_flags)) if frame_flags else 0.0

    total_segments = len(step_vars) // segment_len
    if total_segments <= 0:
        return {
            "smoothness_ratio": frame_smooth_ratio,
            "smooth_segments": int(sum(frame_flags)),
            "total_segments": len(frame_flags),
            "frame_smooth_ratio": frame_smooth_ratio,
        }

    smooth_segments = 0
    for i in range(total_segments):
        seg = frame_flags[i * segment_len : (i + 1) * segment_len]
        if seg and all(seg):
            smooth_segments += 1

    return {
        "smoothness_ratio": float(smooth_segments / total_segments),
        "smooth_segments": int(smooth_segments),
        "total_segments": int(total_segments),
        "frame_smooth_ratio": frame_smooth_ratio,
    }


def _load_policy_models(policy: str, model_path: str | None, plant2_action_mode: str = "pid"):
    policy_cls = None

    if policy == "carl":
        if not model_path:
            raise ValueError("--model-path is required for --policy carl")
        from agents.policies.plain_carl_policy import PlainCarlPolicy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        PlainCarlPolicy.set_checkpoint(model_path, device=device)
        policy_cls = PlainCarlPolicy
    elif policy == "plant2":
        if not model_path:
            raise ValueError("--model-path is required for --policy plant2")
        PLANT2_PATH = SDC_ROOT / "plant2"
        from agents.policies.plain_plant2_policy import PlainPlanT2Policy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        PlainPlanT2Policy.set_checkpoint(
            model_path, PLANT2_PATH, device=device, action_mode=plant2_action_mode,
        )
        policy_cls = PlainPlanT2Policy
    elif policy == "carl_rule":
        if not model_path:
            raise ValueError("--model-path is required for --policy carl_rule")
        from agents.policies.carl_sign_compliant import CarlSignCompliantPolicy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        CarlSignCompliantPolicy.set_checkpoint(model_path, device=device)
        policy_cls = CarlSignCompliantPolicy
    elif policy == "plant2_rule":
        if not model_path:
            raise ValueError("--model-path is required for --policy plant2_rule")
        PLANT2_PATH = SDC_ROOT / "plant2"
        from agents.policies.plant2_sign_compliant import PlanT2SignCompliantPolicy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        PlanT2SignCompliantPolicy.set_checkpoint(
            model_path, PLANT2_PATH, device=device, action_mode=plant2_action_mode,
        )
        policy_cls = PlanT2SignCompliantPolicy

    return {
        "policy_cls": policy_cls,
    }


def _format_lane_pos(pos) -> str:
    """Format lane position for logging (handles numpy arrays)."""
    if pos is None:
        return "N/A"
    return f"({float(pos[0]):.1f}, {float(pos[1]):.1f})"


def _analyze_junction_lanes(env) -> dict:
    """Analyze and print incoming/outgoing lanes of the junction.
    
    Incoming lanes: lanes that feed INTO the junction (have exit_lanes)
    Outgoing lanes: lanes that exit FROM the junction (have entry_lanes but no exit_lanes)
    
    Args:
        env: The environment instance.
        
    Returns:
        Dict with 'incoming' and 'outgoing' lane lists.
    """
    result = {"incoming": [], "outgoing": [], "junction_id": None}
    
    road_network = env.engine.current_map.road_network
    graph = road_network.graph
    
    incoming_lanes = []
    outgoing_lanes = []
    junction_id = None
    
    for lane_name, lane_info in graph.items():
        # Find the junction polygon
        if lane_name.startswith("junction"):
            junction_id = lane_name
            continue
        
        # Skip non-lane entries
        if not lane_name.startswith("lane_"):
            continue
        
        exit_lanes = getattr(lane_info, "exit_lanes", None) or []
        entry_lanes = getattr(lane_info, "entry_lanes", None) or []
        
        # Extract edge ID from lane name (e.g., "lane_46710990#1_0" -> "46710990#1")
        raw_name = lane_name[5:] if lane_name.startswith("lane_") else lane_name
        edge_id = raw_name.rsplit("_", 1)[0] if "_" in raw_name else raw_name
        
        lane_obj = None
        lane_length = 0.0
        start_pos = None
        end_pos = None
        
        try:
            lane_obj = road_network.get_lane(lane_name)
            lane_length = lane_obj.length
            # Get start and end positions of the lane
            start_pos = lane_obj.position(0.0, 0.0)  # Beginning of lane
            end_pos = lane_obj.position(lane_length, 0.0)  # End of lane
        except Exception:
            pass
        
        lane_data = {
            "lane_name": lane_name,
            "edge_id": edge_id,
            "length": lane_length,
            "exit_lanes": exit_lanes,
            "entry_lanes": entry_lanes,
            "start_pos": start_pos,
            "end_pos": end_pos,
        }
        
        # Incoming: has exit_lanes (feeds INTO junction)
        # Outgoing: has entry_lanes but no exit_lanes (exits FROM junction)
        if exit_lanes:
            incoming_lanes.append(lane_data)
        elif entry_lanes and not exit_lanes:
            outgoing_lanes.append(lane_data)
    
    result["incoming"] = incoming_lanes
    result["outgoing"] = outgoing_lanes
    result["junction_id"] = junction_id

    return result.get("incoming", []), result.get("outgoing", [])


def _apply_manifest_ego_spawn_lane(env, row: dict) -> bool:
    """Teleport ego onto the manifest parallel lane (needed when skip_auto_signs=True)."""
    road_id = row.get("road_id")
    if not road_id:
        return False
    lane_num = int(row.get("spawn_lane_num", 0) or 0)
    target_key = make_lane_key(str(road_id), lane_num)
    try:
        vehicle = env.agent
        if vehicle is None:
            return False
        road_network = env.engine.current_map.road_network
        target_lane = road_network.get_lane(target_key)
        start_long = min(1.0, target_lane.length - 0.1)
        pos = target_lane.position(start_long, 0.0)
        heading = target_lane.heading_theta_at(start_long)
        vehicle.set_position(pos)
        vehicle.set_heading_theta(heading)
        try:
            vehicle.spawn_place = pos.copy()
        except Exception:
            pass
        if hasattr(env, "_refresh_navigation_after_spawn"):
            env._refresh_navigation_after_spawn(target_lane)
        else:
            vehicle.reset_navigation(target_lane)
        return True
    except Exception as exc:
        print(f"[EgoSpawn] Could not teleport to {target_key}: {exc}")
        return False


def _reposition_ego_before_lane_end(env, distance_before_end: float) -> bool:
    """Reposition the ego vehicle to a specific distance before the lane end.
    
    Args:
        env: The environment instance.
        distance_before_end: Distance in meters before lane end to place the vehicle.
        
    Returns True if repositioning succeeded, False otherwise.
    """
    try:
        vehicle = env.agent
        if vehicle is None:
            return False
        
        lane = vehicle.lane
        if lane is None:
            return False
        
        lane_length = lane.length
        # Longitudinal from lane START: lane_length - distance_before_end
        spawn_long = max(1.0, min(lane_length - distance_before_end, lane_length - 0.1))
        
        pos = lane.position(spawn_long, 0.0)
        heading = lane.heading_theta_at(spawn_long)
        
        vehicle.set_position(pos)
        vehicle.set_heading_theta(heading)
        
        # Update spawn_place so navigation uses the new position
        try:
            vehicle.spawn_place = pos.copy()
        except Exception:
            pass
        
        # Rebuild navigation from new position
        if hasattr(env, "_refresh_navigation_after_spawn"):
            env._refresh_navigation_after_spawn(lane)
        else:
            try:
                vehicle.reset_navigation(lane)
            except Exception:
                pass
        
        return True
    except Exception as e:
        print(f"[EgoReposition] Failed to reposition ego: {e}")
        return False


def _clear_sign_manager(sign_mgr) -> None:
    sign_mgr.signs.clear()
    sign_mgr.rules.clear()


def _resolve_row_pdd_code(row: dict) -> str:
    code = (
        row.get("_sign_code")
        or row.get("sign_code")
        or row.get("pdd_code")
        or row.get("sign_type")
        or DEFAULT_PDD_CODE
    )
    code = str(code).strip()
    try:
        return get_no_entry_sign_spec(code).pdd_code
    except ValueError:
        return DEFAULT_PDD_CODE


def _place_no_entry_sign(
    env,
    row: dict,
    scenes_root: Path,
    show_model: bool = True,
) -> bool:
    """Place NoEntrySign / NoTrafficSign at the start of the forbidden lane.

    The forbidden lane is the destination edge (the road ego must not enter).
    The sign sits a few metres after that lane's start so baselines that enter
    it accumulate a violation, while experts stop before entering.
    """
    del scenes_root  # kept for call-site parity with other benches
    try:
        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False

        sign_road_id = row.get("sign_road_id") or row.get("destination_edge_id")
        if not sign_road_id and row.get("destination_lane_id"):
            from lib.lane_keys import lane_edge_id

            sign_road_id = lane_edge_id(str(row["destination_lane_id"]))
        if not sign_road_id:
            print("[NoEntrySign] Missing sign_road_id / destination edge")
            return False

        lane_keys: list = []
        layout = row.get("junction_layout") or {}
        for arm in layout.get("arms", []) or []:
            if arm.get("edge_id") == sign_road_id:
                lane_keys = list(arm.get("lane_keys") or [])
                break

        lane = resolve_sign_lane_for_edge(env, str(sign_road_id), lane_keys)
        if lane is None:
            print(f"[NoEntrySign] Lane not found for forbidden edge {sign_road_id}")
            return False

        distance_from_start = float(
            row.get("sign_distance_from_start", DEFAULT_SIGN_DISTANCE_FROM_START_M)
            or DEFAULT_SIGN_DISTANCE_FROM_START_M
        )
        past_sign_m = float(
            row.get("destination_past_sign_m", DEFAULT_DESTINATION_PAST_SIGN_M)
            or DEFAULT_DESTINATION_PAST_SIGN_M
        )
        # Sign + short dest past it must both fit on the forbidden lane;
        # otherwise dest coincides with the sign and violations never occur.
        need_len = distance_from_start + past_sign_m
        if float(lane.length) <= need_len:
            print(
                f"[NoEntrySign] Forbidden lane too short "
                f"({float(lane.length):.2f}m <= sign+past {need_len:.2f}m) "
                f"on edge {sign_road_id}; skip placement"
            )
            return False
        placement_long = sign_placement_long_from_start(lane, distance_from_start)
        longitudinal_offset = sign_longitudinal_offset_from_start(lane, distance_from_start)
        lateral = lateral_offset_beside_lane(lane, placement_long)

        pdd_code = _resolve_row_pdd_code(row)
        sign_cls = resolve_sign_class(pdd_code)

        _clear_sign_manager(sign_mgr)
        sign = sign_mgr.add_sign(
            sign_cls,
            lane=lane,
            longitudinal_offset=longitudinal_offset,
            lateral_offset=lateral,
            show_model=show_model,
            use_random_lane=False,
        )
        spec = get_no_entry_sign_spec(pdd_code)
        print(
            f"[NoEntrySign] Placed {pdd_code} ({spec.title}/{spec.class_name}) on "
            f"forbidden edge {sign_road_id} at {distance_from_start:.2f}m from lane start "
            f"(long_offset={longitudinal_offset:.2f})"
        )
        return sign is not None
    except Exception as e:
        print(f"[NoEntrySign] Failed to place: {e}")
        return False


def _ego_compliant_stop_before_no_entry(
    env,
    vehicle,
    *,
    max_dist_before_sign_m: float = DEFAULT_COMPLIANT_STOP_MAX_DIST_M,
    speed_max_mps: float = DEFAULT_COMPLIANT_STOP_SPEED_MPS,
) -> bool:
    """True when ego is nearly stopped just before a 3.1/3.2 sign line.

    Used ONLY for the compliant-stop early success path (sign-compliant agents
    that cannot reach the literal destination past the forbidden road).
    """
    if vehicle is None:
        return False
    try:
        speed = float(getattr(vehicle, "speed", 0.0) or 0.0)
    except Exception:
        return False
    if speed > float(speed_max_mps):
        return False

    sign_mgr = getattr(getattr(env, "engine", None), "traffic_sign_manager", None)
    if sign_mgr is None:
        return False

    max_dist = float(max_dist_before_sign_m)
    for sign in list(getattr(sign_mgr, "signs", None) or []):
        if not isinstance(sign, (NoEntrySign, NoTrafficSign)):
            continue
        sign_lane = getattr(sign, "lane", None)
        if sign_lane is None:
            continue
        sign_long = float(
            getattr(sign, "sign_line_position", getattr(sign, "placement_long", 0.0))
            or 0.0
        )
        try:
            veh_long = float(sign_lane.local_coordinates(vehicle.position)[0])
        except Exception:
            continue
        dist_to_line = sign_long - veh_long
        # Still before the violation threshold past the line, and close enough
        # that the stop is at the sign (not stuck back in the junction).
        if -0.25 < dist_to_line <= max_dist:
            return True
    return False


def _ego_past_sign_route_end(
    env,
    vehicle,
    *,
    past_sign_m: float = DEFAULT_DESTINATION_PAST_SIGN_M,
) -> bool:
    """True when ego has driven past the no-entry sign by ``past_sign_m``.

    This is the short route endpoint on the forbidden lane (so non-compliant
    policies do not keep driving to the far end of a long edge). Independent
    of the compliant-stop arrive override.
    """
    if vehicle is None:
        return False
    sign_mgr = getattr(getattr(env, "engine", None), "traffic_sign_manager", None)
    if sign_mgr is None:
        return False
    past = float(past_sign_m)
    for sign in list(getattr(sign_mgr, "signs", None) or []):
        if not isinstance(sign, (NoEntrySign, NoTrafficSign)):
            continue
        sign_lane = getattr(sign, "lane", None)
        if sign_lane is None:
            continue
        sign_long = float(
            getattr(sign, "sign_line_position", getattr(sign, "placement_long", 0.0))
            or 0.0
        )
        try:
            veh_long = float(sign_lane.local_coordinates(vehicle.position)[0])
        except Exception:
            continue
        if veh_long >= sign_long + past:
            return True
    return False


def run_one_episode(
    row: dict,
    policy_type: str,
    models: dict,
    scenes_root: Path,
    max_steps: int,
    ego_variant: str,
    ego_sample_seed_base: int,
    replay_root: Path | None = None,
    save_gif: Path | None = None,
    hide_signs: bool = False,
) -> dict:
    seed = int(row.get("seed") or row.get("deterministic_seed") or 0)
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    env = _build_sumo_env(row, scenes_root=scenes_root, max_steps=max_steps)

    raw_env = env
    env = _wrap_for_policy(env, policy_type)

    policy_cls = None
    if policy_type == "idm":
        policy_cls = ModifiedIDMPolicy  # Good driving, no sign compliance
    elif policy_type == "modified_idm":
        policy_cls = ModifiedIDMSignCompliantPolicy
    elif policy_type == "comprehensive_rule_expert":
        policy_cls = ComprehensiveRuleExpertPolicy
    elif policy_type == "rule_compliant":
        policy_cls = RuleCompliantExpertPolicy
    elif policy_type == "ppo_lidar":
        policy_cls = ExpertPolicy
    elif policy_type in ("carl", "plant2", "carl_rule", "plant2_rule"):
        policy_cls = models.get("policy_cls")
        if policy_cls is None:
            raise RuntimeError(f"policy_cls for --policy {policy_type} not loaded; "
                               "check _load_policy_models")

    try:
        env_seed = (int(row.get("sign_id", 0)) + int(row.get("var_idx", 0))) % 100000
        obs, info = env.reset(seed=env_seed)
        base_env = _unwrap_base_env(env)
        try:
            if hasattr(base_env, "engine") and hasattr(base_env.engine, "np_random"):
                base_env.engine.np_random = np.random.RandomState(seed)
        except Exception:
            pass

        # Manifest spawn lane + position before lane end (approach to junction)
        _apply_manifest_ego_spawn_lane(base_env, row)
        spawn_distance = float(row.get("spawn_distance_before_end", 0) or 0)
        if spawn_distance > 0:
            _reposition_ego_before_lane_end(base_env, spawn_distance)

        # Validate route: check that destination is different from spawn
        nav = getattr(base_env.vehicle, "navigation", None)
        if nav is not None:
            checkpoints = getattr(nav, "checkpoints", [])
            spawn_lane_idx = getattr(base_env.vehicle.lane, "index", None)
            if checkpoints and spawn_lane_idx:
                if len(checkpoints) <= 1 or checkpoints[-1] == spawn_lane_idx or checkpoints[0] == checkpoints[-1]:
                    scene_id = row.get("scene_id", "unknown")
                    dest = row.get("destination_lane_id", "unknown")
                    print(f"[RouteValidation] INVALID: {scene_id} - route loops back to spawn. "
                          f"spawn={spawn_lane_idx}, dest={dest}, checkpoints={checkpoints[:3]}...")
                    return {
                        "ok": False,
                        "error": f"Invalid route: spawn and destination are the same or unreachable",
                        "scene_id": scene_id,
                    }

        # Place NoEntrySign / NoTrafficSign near approach lane end
        _place_no_entry_sign(
            base_env,
            row,
            scenes_root=scenes_root,
            show_model=not hide_signs,
        )

        # Analyze and print junction lanes (for debugging/info only)
        incoming_lanes, outgoing_lanes = _analyze_junction_lanes(base_env)

        policy_obj = None
        sampled_ego_params = None
        if policy_cls is not None:
            policy_obj = policy_cls(base_env.vehicle, seed)
            if policy_type in ("idm", "modified_idm", "comprehensive_rule_expert"):
                if ego_variant == "default":
                    apply_ego_defaults(policy_obj)
                elif ego_variant.startswith("s") and ego_variant[1:].isdigit():
                    k = int(ego_variant[1:])
                    sample_seed = int(ego_sample_seed_base) + int(seed) + k * 1000003
                    sampled_ego_params = sample_ego_params(sample_seed)
                    apply_ego_sampled(policy_obj, sampled_ego_params)
                else:
                    apply_ego_defaults(policy_obj)

        total_reward = 0.0
        violations = 0
        sign_violations = 0
        traffic_light_violations = 0
        crosswalk_violations = 0
        violations_by_class_step: dict[str, int] = {}
        in_zone_total_steps = 0
        in_zone_by_class_step: dict[str, int] = {}
        violations_event_count = 0
        violations_by_class_event: dict[str, int] = {}
        violations_timeline: list[dict] = []
        prev_violated_class_names: set = set()
        steps = 0
        crashed = False
        reached_dest = False
        out_of_road = False
        last_violation_texts = []
        violation_text_ttl = 0
        speed_pct_samples: list[float] = []
        min_ttc: float | None = None
        abs_lane_offsets: list[float] = []
        steer_delta_abs: list[float] = []
        hard_brake_count = 0
        hard_accel_count = 0
        distance_travelled_m = 0.0
        visited_lane_lengths: dict[str, float] = {}

        dt = 0.1
        prev_speed_mps: float | None = None
        prev_heading: float | None = None
        prev_long_acc: float | None = None
        prev_lat_acc: float | None = None
        prev_yaw_rate: float | None = None
        prev_action_steer: float | None = None
        smoothness_step_vars: list[dict] = []
        expert_actions: list[list[float]] = []
        sign_info_snapshot = _extract_sign_info(base_env)
        last_info: dict = {}

        stop_success_s = float(
            row.get(
                "compliant_stop_success_seconds",
                DEFAULT_COMPLIANT_STOP_SUCCESS_SECONDS,
            )
            or DEFAULT_COMPLIANT_STOP_SUCCESS_SECONDS
        )
        stop_max_dist_m = float(
            row.get("compliant_stop_max_dist_m", DEFAULT_COMPLIANT_STOP_MAX_DIST_M)
            or DEFAULT_COMPLIANT_STOP_MAX_DIST_M
        )
        stop_speed_max = float(
            row.get("compliant_stop_speed_mps", DEFAULT_COMPLIANT_STOP_SPEED_MPS)
            or DEFAULT_COMPLIANT_STOP_SPEED_MPS
        )
        past_sign_m = float(
            row.get("destination_past_sign_m", DEFAULT_DESTINATION_PAST_SIGN_M)
            or DEFAULT_DESTINATION_PAST_SIGN_M
        )
        stop_success_steps = max(1, int(round(stop_success_s / dt)))
        compliant_stop_steps = 0
        compliant_stop_success = False
        past_sign_arrive = False

        for step in range(max_steps):
            if policy_obj is not None:
                action = policy_obj.act(base_env.vehicle.name)
            else:
                action = [0.0, 0.0]
            try:
                expert_actions.append([float(action[0]), float(action[1])])
            except Exception:
                expert_actions.append([0.0, 0.0])

            obs, reward, terminated, truncated, info = env.step(action)

            last_info = info
            total_reward += float(reward)
            steps += 1

            vehicle = base_env.agent
            sign_mgr = getattr(base_env.engine, "traffic_sign_manager", None)
            current_violation_texts = []
            current_violated_class_names: set = set()
            current_violations = []
            if sign_mgr is not None and vehicle is not None:
                step_in_any_zone = False
                for _s in sign_mgr.signs:
                    if _ego_in_sign_zone(_s, vehicle):
                        step_in_any_zone = True
                        cls = type(_s).__name__
                        in_zone_by_class_step[cls] = in_zone_by_class_step.get(cls, 0) + 1
                if step_in_any_zone:
                    in_zone_total_steps += 1

                for sign in sign_mgr.signs:
                    if sign._is_violating(vehicle):
                        sign_violations += 1
                        violations += 1

                current_violations = sign_mgr.check_all_violations(vehicle)
                for _sign, violated in current_violations:
                    if violated:
                        violations += 1
                        bucket = _violation_bucket(_sign)
                        if bucket == "traffic_light":
                            traffic_light_violations += 1
                        elif bucket == "crosswalk":
                            crosswalk_violations += 1
                        else:
                            sign_violations += 1
                        current_violation_texts.append(_format_violation(_sign, vehicle))
                        cls_name = type(_sign).__name__
                        violations_by_class_step[cls_name] = (
                            violations_by_class_step.get(cls_name, 0) + 1)
                        current_violated_class_names.add(cls_name)
                        if cls_name not in prev_violated_class_names:
                            violations_event_count += 1
                            violations_by_class_event[cls_name] = (
                                violations_by_class_event.get(cls_name, 0) + 1)
                            try:
                                rule = _sign.get_rule_description() or ""
                            except Exception:
                                rule = ""
                            violations_timeline.append({
                                "step": int(step),
                                "sign_class": cls_name,
                                "rule": rule,
                            })
                prev_violated_class_names = current_violated_class_names
                if current_violation_texts:
                    last_violation_texts = current_violation_texts[:3]
                    violation_text_ttl = 40
                elif violation_text_ttl > 0:
                    violation_text_ttl -= 1
                else:
                    last_violation_texts = []

            if vehicle is not None:
                sp = _nearby_speed_percentage(vehicle)
                if sp is not None:
                    speed_pct_samples.append(float(sp))

            if vehicle is not None:
                speed_mps = _safe_float(getattr(vehicle, "speed", 0.0), 0.0)
                distance_travelled_m += max(0.0, speed_mps) * dt
                heading = _safe_float(getattr(vehicle, "heading_theta", 0.0), 0.0)

                lane = getattr(vehicle, "lane", None)
                if lane is not None:
                    try:
                        _long, lat = lane.local_coordinates(vehicle.position)
                        abs_lane_offsets.append(abs(float(lat)))
                    except Exception:
                        pass
                    try:
                        lane_idx = getattr(lane, "index", None)
                        lane_key = repr(lane_idx) if lane_idx is not None else f"lane_obj_{id(lane)}"
                        lane_len = float(getattr(lane, "length", 0.0) or 0.0)
                        if lane_len > 0.0 and lane_key not in visited_lane_lengths:
                            visited_lane_lengths[lane_key] = lane_len
                    except Exception:
                        pass

                step_ttc = _min_ttc_seconds(vehicle)
                if step_ttc is not None and step_ttc > 0.0:
                    min_ttc = step_ttc if min_ttc is None else min(min_ttc, step_ttc)

                cur_steer = _safe_float(action[0], 0.0)
                if prev_action_steer is not None:
                    steer_delta_abs.append(abs(cur_steer - prev_action_steer))
                prev_action_steer = cur_steer

                if prev_speed_mps is not None and prev_heading is not None:
                    long_acc = (speed_mps - prev_speed_mps) / dt
                    yaw_delta = math.atan2(math.sin(heading - prev_heading), math.cos(heading - prev_heading))
                    yaw_rate = yaw_delta / dt
                    lat_acc = speed_mps * yaw_rate

                    if long_acc < -3.0:
                        hard_brake_count += 1
                    if long_acc > 2.5:
                        hard_accel_count += 1

                    if prev_long_acc is not None and prev_lat_acc is not None and prev_yaw_rate is not None:
                        long_jerk = (long_acc - prev_long_acc) / dt
                        lat_jerk = (lat_acc - prev_lat_acc) / dt
                        yaw_acc = (yaw_rate - prev_yaw_rate) / dt
                        jerk_mag = float(math.sqrt(long_jerk * long_jerk + lat_jerk * lat_jerk))
                        smoothness_step_vars.append(
                            {
                                "long_acc": float(long_acc),
                                "lat_acc": float(lat_acc),
                                "yaw_rate": float(yaw_rate),
                                "yaw_acc": float(yaw_acc),
                                "long_jerk": float(long_jerk),
                                "jerk_mag": jerk_mag,
                            }
                        )

                    prev_long_acc = long_acc
                    prev_lat_acc = lat_acc
                    prev_yaw_rate = yaw_rate

                prev_speed_mps = speed_mps
                prev_heading = heading

            if terminated or truncated:
                # Natural MetaDrive terminate only — do NOT invent arrive_dest here.
                reached_dest = bool(info.get("arrive_dest", False))
                out_of_road = bool(info.get("out_of_road", False))
                crashed = bool(info.get("crash", False) or out_of_road)
                break

            # (1) ONLY override of arrive_dest without reaching the route end:
            #     sign-compliant stop before the line (0 violations, nearly stopped
            #     within compliant_stop_max_dist_m). Timeout / crash / other stops
            #     do not get this override.
            if sign_violations == 0 and _ego_compliant_stop_before_no_entry(
                base_env,
                vehicle,
                max_dist_before_sign_m=stop_max_dist_m,
                speed_max_mps=stop_speed_max,
            ):
                compliant_stop_steps += 1
                if compliant_stop_steps >= stop_success_steps:
                    reached_dest = True
                    compliant_stop_success = True
                    print(
                        f"[NoEntrySign] Compliant stop for {stop_success_s:.1f}s "
                        f"before sign → arrive_dest (step={steps})"
                    )
                    break
            else:
                compliant_stop_steps = 0

            # (2) Short route endpoint just past the sign (same for all policies).
            #     Independent of (1); lets non-compliant agents finish quickly.
            if _ego_past_sign_route_end(
                base_env,
                vehicle,
                past_sign_m=past_sign_m,
            ):
                reached_dest = True
                past_sign_arrive = True
                print(
                    f"[NoEntrySign] Past sign by {past_sign_m:.1f}m "
                    f"→ destination (step={steps})"
                )
                break

            text_dict: dict = {}
            if save_gif:
                text_dict = {
                    "Step": step,
                    "Speed": f"{vehicle.speed_km_h:.2f} km/h",
                    "Vehicle lane: ": vehicle.lane.index,
                    "Current lane width: ": vehicle.lane.width,
                    "Violations: ": sign_violations,
                }
                
                no_entry_signs = [
                    sign for sign in sign_mgr.signs
                    if isinstance(sign, (NoEntrySign, NoTrafficSign))
                ] if sign_mgr is not None else []
                text_dict["No-entry signs"] = len(no_entry_signs)
                if no_entry_signs:
                    text_dict["Sign class"] = type(no_entry_signs[0]).__name__

            if current_violation_texts:
                text_dict["Violation"] = current_violation_texts[0]
                if len(current_violation_texts) > 1:
                    text_dict["Violation +"] = f"+{len(current_violation_texts) - 1} more"
            elif last_violation_texts:
                text_dict["Last violation"] = last_violation_texts[0]
                if len(last_violation_texts) > 1:
                    text_dict["Last violation +"] = f"+{len(last_violation_texts) - 1} more"

            if save_gif:
                try:
                    base_env.render(
                        mode="top_down",
                        film_size=(2400, 2400), scaling=12.0,
                        screen_size=(800, 800),
                        semantic_map=True,
                        semantic_broken_line=True,
                        draw_target_vehicle_trajectory=True,
                        target_agent_heading_up=True,
                        screen_record=True, window=False,
                        text=text_dict,
                    )
                except Exception:
                    pass

        route_completion_pct = _route_completion_percent(last_info, reached_dest)
        infraction_penalty = _infraction_penalty(crashed=crashed, out_of_road=out_of_road, violations=violations)
        driving_score = route_completion_pct * infraction_penalty
        driving_efficiency = float(np.mean(speed_pct_samples)) if speed_pct_samples else 0.0
        smoothness = _compute_smoothness(smoothness_step_vars, segment_len=20)
        route_length_m = _route_length_m(last_info)
        route_length_source = "info"
        if route_length_m is None:
            approx = float(sum(visited_lane_lengths.values()))
            if approx > 0.0:
                route_length_m = approx
                route_length_source = "visited_lanes"
            else:
                route_length_source = "none"

        crashed_flag_raw = bool(last_info.get("crash", False)) if last_info else False
        crash_attribution = None
        if crashed_flag_raw or bool(getattr(base_env.agent, "crash_vehicle", False)):
            try:
                crash_attribution = "ego" if _ego_at_fault_for_crash(base_env.agent, base_env.engine) else "npc"
            except Exception:
                crash_attribution = None

        if replay_root is not None:
            try:
                _sign_for_path = (row.get("_sign_code") or row.get("sign_code")
                                  or row.get("pdd_code") or row.get("sign_type") or "")
                sign_slug = str(_sign_for_path).replace(".", "_")
                scene_id_for_uid = row.get("scene_id") or f"scene_{seed}"
                lane_for_uid = int(row.get("spawn_lane_num", 0) or 0)
                var_for_uid = int(row.get("var_idx", 0) or 0)
                scene_uid = f"{scene_id_for_uid}_lane{lane_for_uid}_seed{seed}_v{var_for_uid}"
                expert_subdir = f"{policy_type}_{ego_variant}" if ego_variant else policy_type

                out_replay = (Path(replay_root) / sign_slug / "by_sign" / sign_slug
                              / "by_scene" / scene_uid / expert_subdir)
                out_replay.mkdir(parents=True, exist_ok=True)
                sidecar_path = out_replay / "replay.json"

                sidecar_metrics = {
                    "arrived_dest": bool(reached_dest),
                    "compliant_stop_success": bool(compliant_stop_success),
                    "past_sign_arrive": bool(past_sign_arrive),
                    "crashed": crashed_flag_raw,
                    "crash_attribution": crash_attribution,
                    "crashed_ego_fault": bool(crashed_flag_raw and crash_attribution == "ego"),
                    "crashed_npc_fault": bool(crashed_flag_raw and crash_attribution == "npc"),
                    "out_of_road": bool(out_of_road),
                    "final_step": int(steps),
                    "total_violations": int(violations),
                    "violations_by_class": {
                        "sign": int(sign_violations),
                        "traffic_light": int(traffic_light_violations),
                        "crosswalk": int(crosswalk_violations),
                    },
                    "violations_by_class_step": dict(violations_by_class_step),
                    "in_zone_total_steps": int(in_zone_total_steps),
                    "in_zone_by_class_step": dict(in_zone_by_class_step),
                    "violations_event_count": int(violations_event_count),
                    "violations_by_class_event": dict(violations_by_class_event),
                    "violations_timeline": list(violations_timeline),
                    "route_completion": (float(route_completion_pct) / 100.0
                                          if route_completion_pct else 0.0),
                    "total_reward": round(float(total_reward), 4),
                    "smoothness_ratio": smoothness["smoothness_ratio"],
                    "frame_smooth_ratio": smoothness["frame_smooth_ratio"],
                    "smooth_segments": smoothness["smooth_segments"],
                    "total_segments": smoothness["total_segments"],
                    "driving_score": float(driving_score),
                    "driving_efficiency": float(driving_efficiency),
                    "infraction_penalty": float(infraction_penalty),
                    "min_ttc_sec": float(min_ttc) if min_ttc is not None else None,
                    "mean_abs_lane_offset": (float(np.mean(abs_lane_offsets))
                                              if abs_lane_offsets else None),
                    "mean_abs_steer_delta": (float(np.mean(steer_delta_abs))
                                              if steer_delta_abs else None),
                    "hard_brake_count": int(hard_brake_count),
                    "hard_accel_count": int(hard_accel_count),
                    "route_length_m": (float(route_length_m)
                                        if route_length_m is not None else None),
                    "distance_travelled_m": float(distance_travelled_m),
                    "success": bool(reached_dest and not crashed_flag_raw and not out_of_road),
                }
                sidecar = {
                    "scene_id": scene_id_for_uid,
                    "scene_uid": scene_uid,
                    "backend": "sumo",
                    "pdd_code": (row.get("pdd_code") or row.get("sign_code")
                                 or row.get("sign_type")),
                    "sign_key": row.get("sign_type"),
                    "sign_slug": sign_slug,
                    "policy": policy_type,
                    "variant": ego_variant,
                    "source_row": row,
                    "env_config_summary": {
                        "map_name": row.get("net_path"),
                        "road_id": row.get("road_id"),
                        "spawn_lane_num": row.get("spawn_lane_num"),
                        "horizon": max_steps,
                        "seed": seed,
                    },
                    "signs": sign_info_snapshot,
                    "expert_actions": expert_actions,
                    "smoothness_step_vars": smoothness_step_vars,
                    "metrics": sidecar_metrics,
                    "ego_idm_params": (sampled_ego_params if sampled_ego_params is not None
                                        else "DEFAULT_EGO_PARAMS"),
                    "pkl_path": None,
                    "sidecar_path": str(sidecar_path),
                    "valid": True,
                }
                with open(sidecar_path, "w", encoding="utf-8") as _sf:
                    json.dump(sidecar, _sf, default=str)
            except Exception:
                pass

        return {
            "ok": True,
            "backend": "sumo",
            "scene_id": row.get("scene_id"),
            "sign_type": row.get("_sign_code") or row.get("sign_code") or row.get("pdd_code") or row.get("sign_type"),
            "seed": seed,
            "total_reward": total_reward,
            "steps": steps,
            "violations": violations,
            "sign_violations": int(sign_violations),
            "traffic_light_violations": int(traffic_light_violations),
            "crosswalk_violations": int(crosswalk_violations),
            "crashed": crashed,
            "out_of_road": out_of_road,
            "reached_dest": reached_dest,
            "compliant_stop_success": bool(compliant_stop_success),
            "past_sign_arrive": bool(past_sign_arrive),
            "success": reached_dest and not crashed,
            "route_completion_pct": route_completion_pct,
            "infraction_penalty": infraction_penalty,
            "driving_score": driving_score,
            "driving_efficiency": driving_efficiency,
            "smoothness": smoothness["smoothness_ratio"],
            "smoothness_frame_ratio": smoothness["frame_smooth_ratio"],
            "smooth_segments": smoothness["smooth_segments"],
            "smooth_total_segments": smoothness["total_segments"],
            "hard_brake_count": int(hard_brake_count),
            "hard_accel_count": int(hard_accel_count),
            "mean_abs_lane_offset": float(np.mean(abs_lane_offsets)) if abs_lane_offsets else None,
            "mean_abs_steer_delta": float(np.mean(steer_delta_abs)) if steer_delta_abs else None,
            "min_ttc_sec": float(min_ttc) if min_ttc is not None else None,
            "route_length_m": float(route_length_m) if route_length_m is not None else None,
            "route_length_source": route_length_source,
            "distance_travelled_m": float(distance_travelled_m),
            "variant": ego_variant,
            "ego_params": sampled_ego_params,
            "violations_by_class_step": dict(violations_by_class_step),
            "violations_event_count": int(violations_event_count),
            "violations_by_class_event": dict(violations_by_class_event),
            "violations_timeline": list(violations_timeline),
            "in_zone_total_steps": int(in_zone_total_steps),
            "in_zone_by_class_step": dict(in_zone_by_class_step),
        }
    finally:
        if save_gif is not None:
            try:
                save_gif.parent.mkdir(parents=True, exist_ok=True)
                renderer = getattr(_unwrap_base_env(env), "top_down_renderer", None)
                if renderer is not None:
                    renderer.generate_gif(str(save_gif), duration=40)
            except Exception:
                pass
        try:
            env.close()
        except Exception:
            pass
        if raw_env is not env:
            try:
                raw_env.close()
            except Exception:
                pass


def aggregate_results(results: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if not r.get("ok"):
            continue
        key = str(r.get("sign_type"))
        grouped[key].append(r)

    summary: dict[str, dict] = {}
    for sign, runs in sorted(grouped.items()):
        success_rate = float(np.mean([x["success"] for x in runs])) if runs else 0.0
        crash_rate = float(np.mean([x["crashed"] for x in runs])) if runs else 0.0
        avg_violations = float(np.mean([x["violations"] for x in runs])) if runs else 0.0
        avg_sign_viol = float(np.mean([x.get("sign_violations", 0) for x in runs])) if runs else 0.0
        avg_tl_viol = float(np.mean([x.get("traffic_light_violations", 0) for x in runs])) if runs else 0.0
        avg_cw_viol = float(np.mean([x.get("crosswalk_violations", 0) for x in runs])) if runs else 0.0
        avg_violations_event = float(np.mean([x.get("violations_event_count", 0) for x in runs])) if runs else 0.0
        violations_by_class_event_total: dict[str, int] = {}
        for x in runs:
            for cls, cnt in (x.get("violations_by_class_event") or {}).items():
                violations_by_class_event_total[cls] = (
                    violations_by_class_event_total.get(cls, 0) + int(cnt))
        violations_by_class_step_total: dict[str, int] = {}
        for x in runs:
            for cls, cnt in (x.get("violations_by_class_step") or {}).items():
                violations_by_class_step_total[cls] = (
                    violations_by_class_step_total.get(cls, 0) + int(cnt))
        avg_in_zone_steps = float(np.mean([x.get("in_zone_total_steps", 0) for x in runs])) if runs else 0.0
        in_zone_by_class_step_total: dict[str, int] = {}
        for x in runs:
            for cls, cnt in (x.get("in_zone_by_class_step") or {}).items():
                in_zone_by_class_step_total[cls] = (
                    in_zone_by_class_step_total.get(cls, 0) + int(cnt))
        avg_reward = float(np.mean([x["total_reward"] for x in runs])) if runs else 0.0
        avg_ds = float(np.mean([x.get("driving_score", 0.0) for x in runs])) if runs else 0.0
        avg_route = float(np.mean([x.get("route_completion_pct", 0.0) for x in runs])) if runs else 0.0
        avg_eff = float(np.mean([x.get("driving_efficiency", 0.0) for x in runs])) if runs else 0.0
        avg_smooth = float(np.mean([x.get("smoothness", 0.0) for x in runs])) if runs else 0.0
        avg_frame_smooth = float(np.mean([x.get("smoothness_frame_ratio", 0.0) for x in runs])) if runs else 0.0
        avg_hb = float(np.mean([x.get("hard_brake_count", 0) for x in runs])) if runs else 0.0
        avg_ha = float(np.mean([x.get("hard_accel_count", 0) for x in runs])) if runs else 0.0
        lane_offsets = [x.get("mean_abs_lane_offset") for x in runs if x.get("mean_abs_lane_offset") is not None]
        steer_deltas = [x.get("mean_abs_steer_delta") for x in runs if x.get("mean_abs_steer_delta") is not None]
        min_ttc_vals = [x.get("min_ttc_sec") for x in runs if x.get("min_ttc_sec") is not None]
        route_len_vals = [x.get("route_length_m") for x in runs if x.get("route_length_m") is not None]
        dist_vals = [x.get("distance_travelled_m") for x in runs if x.get("distance_travelled_m") is not None]
        summary[sign] = {
            "backend": "sumo",
            "sign_type": sign,
            "total_runs": len(runs),
            "success_rate": success_rate,
            "crash_rate": crash_rate,
            "average_violations": avg_violations,
            "average_sign_violations": avg_sign_viol,
            "average_traffic_light_violations": avg_tl_viol,
            "average_crosswalk_violations": avg_cw_viol,
            "average_violations_event_count": avg_violations_event,
            "violations_by_class_event_total": violations_by_class_event_total,
            "violations_by_class_step_total": violations_by_class_step_total,
            "average_in_zone_steps": avg_in_zone_steps,
            "in_zone_by_class_step_total": in_zone_by_class_step_total,
            "average_reward": avg_reward,
            "average_route_completion_pct": avg_route,
            "average_infraction_penalty": float(np.mean([x.get("infraction_penalty", 1.0) for x in runs])) if runs else 1.0,
            "average_driving_score": avg_ds,
            "average_driving_efficiency": avg_eff,
            "average_smoothness": avg_smooth,
            "average_smoothness_frame_ratio": avg_frame_smooth,
            "average_hard_brake_count": avg_hb,
            "average_hard_accel_count": avg_ha,
            "average_mean_abs_lane_offset": float(np.mean(lane_offsets)) if lane_offsets else None,
            "average_mean_abs_steer_delta": float(np.mean(steer_deltas)) if steer_deltas else None,
            "average_min_ttc_sec": float(np.mean(min_ttc_vals)) if min_ttc_vals else None,
            "average_route_length_m": float(np.mean(route_len_vals)) if route_len_vals else None,
            "average_distance_travelled_m": float(np.mean(dist_vals)) if dist_vals else None,
        }
    return summary


def _episode_key_from_row(row: dict) -> tuple[str, str, int]:
    return (
        str(row.get("scene_id", "")),
        str(row.get("_sign_code") or row.get("sign_code") or row.get("pdd_code") or row.get("sign_type") or ""),
        int(row.get("seed") or row.get("deterministic_seed") or -1),
    )


def _episode_key_from_result(r: dict) -> tuple[str, str, int]:
    return (
        str(r.get("scene_id", "")),
        str(r.get("sign_type", "")),
        int(r.get("seed") or -1),
    )


def _load_existing_results(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


import time


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run policies on junction crops (no-entry 3.1 / 3.2); "
            "signs placed near approach lane end"
        )
    )
    parser.add_argument("--policy", required=True,
                        choices=["idm", "modified_idm", "comprehensive_rule_expert",
                                 "rule_compliant", "ppo_lidar",
                                 "carl", "carl_rule",
                                 "plant2", "plant2_rule"])
    parser.add_argument("--model-path", type=str, default=None,
                        help="Required for carl/plant2")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--preset", type=str, default="full", choices=["full", "full_last"])
    parser.add_argument("--benchmark-output", type=str, default="benchmark_output",
                        help="Base dir that contains <preset>/")
    parser.add_argument("--scenes-root", type=str, default=str(BENCH_DIR / "scenes"))
    parser.add_argument("--sign-type", type=str, default=None,
                        help="Single sign code, e.g. 3.1")
    parser.add_argument("--sign-types", type=str, default="",
                        help="Comma-separated sign codes")
    parser.add_argument("--max-scenes-per-sign", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--ego-variant", type=str, default="default",
                        help="Ego IDM variant label: default or s1/s2/s3/s4")
    parser.add_argument("--ego-sample-seed-base", type=int, default=42,
                        help="Base seed for sampled IDM ego variants")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="Recompute scenes with existing failed records (ok=false)")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Ignore existing results and rerun all scenes")
    parser.add_argument("--skip-error-episodes", action="store_true",
                        help="When used with --rerun-failed, keep previously errored episodes skipped")
    parser.add_argument("--debug-one-way-sign-selection", action="store_true",
                        help="Enable verbose lane-selection debug logs")
    parser.add_argument("--emit-replay-sidecar", action="store_true",
                        help="Also emit per-(scene_uid, variant) replay.json sidecar")
    parser.add_argument("--replay-root", type=str, default=None,
                        help="Output dir for sidecar files")
    parser.add_argument("--unique-scene-id", action="store_true",
                        help="Dedup manifest rows by scene_id")
    parser.add_argument("--scene-id", type=str, default=None,
                        help="Run only the scene with this scene_id")
    parser.add_argument("--scene-uid", type=str, default=None,
                        help="Run only the scene matching this exact UID <scene_id>:<sign_type>:<seed>")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to a custom *.jsonl manifest")
    parser.add_argument("--save-gifs", action="store_true",
                        help="Record top-down GIF per episode")
    parser.add_argument("--gif-dir", type=str, default=None,
                        help="Directory for GIFs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Write episodes/summary/gifs here directly "
                             "(skips <benchmark-output>/<preset>/policy_eval/<run-name>)")
    parser.add_argument("--plant2-action-mode", type=str, default="pid",
                        choices=["pid", "wps_pure_pursuit"],
                        help="How PlanT2 converts pred_plan -> action")
    parser.add_argument("--hide-signs", action="store_true",
                        help="Hide traffic sign visual models (signs still affect behavior)")

    args = parser.parse_args()

    if args.scene_id and args.scene_uid:
        raise ValueError("--scene-id and --scene-uid are mutually exclusive")

    assert args.ego_variant in ("default", "s1", "s2", "s3", "s4"), \
        f"--ego-variant must be one of default/s1/s2/s3/s4, got {args.ego_variant!r}"

    if args.ego_variant != "default":
        _t = args.ego_sample_seed_base + 12345
        _p1 = sample_ego_params(_t)
        _p2 = sample_ego_params(_t)
        for _k in _p1:
            assert math.isclose(float(_p1[_k]), float(_p2[_k]), abs_tol=1e-9), (
                f"sample_ego_params nondeterministic on key {_k!r}: "
                f"{_p1[_k]} vs {_p2[_k]}")
        print(f"[determinism check OK] sample_ego_params({_t}) reproducible.")

    logging.getLogger().setLevel(getattr(logging, "CRITICAL"))

    benchmark_output_dir = (BENCH_DIR / args.benchmark_output / args.preset).resolve()
    if not args.manifest and not benchmark_output_dir.exists():
        raise ValueError(f"Benchmark output not found: {benchmark_output_dir}")

    scenes_root = Path(args.scenes_root).resolve()

    only_codes: set[str] = set()
    if args.sign_type:
        only_codes.add(args.sign_type)
    if args.sign_types.strip():
        only_codes.update([c.strip() for c in args.sign_types.split(",") if c.strip()])

    print(f"Policy: {args.policy}")
    print(f"Preset: {args.preset}")
    print(f"Backend: sumo (real maps only)")
    print(f"Input: {benchmark_output_dir}")

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"--manifest not found: {manifest_path}")
        rows: list[dict] = []
        for row in _load_enriched_manifest_rows(manifest_path):
            if "valid" in row and not row["valid"]:
                continue
            row["_backend"] = "sumo"
            if not row.get("_sign_code"):
                row["_sign_code"] = (row.get("sign_code") or row.get("pdd_code")
                                      or row.get("sign_type") or "")
            rows.append(row)
    else:
        rows = collect_rows(
            benchmark_output_dir=benchmark_output_dir,
            only_codes=only_codes,
            max_scenes_per_sign=args.max_scenes_per_sign,
            unique_scene_id=args.unique_scene_id,
        )

    if args.scene_id:
        rows = [r for r in rows if str(r.get("scene_id")) == args.scene_id]
    if args.scene_uid:
        print(f"[DEBUG] Looking for scene_uid: {args.scene_uid}")
        print(f"[DEBUG] Available scene keys (first 5):")
        for i, r in enumerate(rows[:5]):
            key = ":".join(str(x) for x in _episode_key_from_row(r))
            print(f"  [{i}] {key}")
        rows = [r for r in rows
                if ":".join(str(x) for x in _episode_key_from_row(r)) == args.scene_uid]
        print(f"[DEBUG] Matched {len(rows)} rows")

    if not rows:
        raise RuntimeError(
            "No scenes selected. Check --preset/--sign-type/"
            "--scene-id/--scene-uid/--manifest")

    print(f"Selected scenes: {len(rows)}")
    models = _load_policy_models(
        args.policy, args.model_path, plant2_action_mode=args.plant2_action_mode,
    )

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = benchmark_output_dir / "policy_eval" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / f"episodes_{args.policy}.jsonl"

    replay_root: Path | None = None
    if args.emit_replay_sidecar:
        replay_root = Path(args.replay_root) if args.replay_root else (out_dir / "replays")
        replay_root.mkdir(parents=True, exist_ok=True)
        print(f"Sidecars: {replay_root}")

    gifs_dir: Path | None = None
    if args.save_gifs:
        gifs_dir = Path(args.gif_dir) if args.gif_dir else (out_dir / "gifs")
        gifs_dir.mkdir(parents=True, exist_ok=True)
        print(f"GIFs: {gifs_dir}")

    existing_results = _load_existing_results(episodes_path)
    existing_by_key: dict[tuple[str, str, int], dict] = {}
    for r in existing_results:
        existing_by_key[_episode_key_from_result(r)] = r

    rows_to_run: list[dict] = []
    skipped = 0
    for row in rows:
        key = _episode_key_from_row(row)
        old = existing_by_key.get(key)
        if args.force_rerun:
            rows_to_run.append(row)
            continue
        if old is None:
            rows_to_run.append(row)
            continue
        if args.skip_error_episodes and not bool(old.get("ok", False)):
            skipped += 1
            continue
        if args.rerun_failed and not bool(old.get("ok", False)):
            rows_to_run.append(row)
            continue
        skipped += 1

    print(f"Resume: loaded {len(existing_results)} existing episodes, skip {skipped}, run {len(rows_to_run)}"
          + (" (--force-rerun: ignoring existing)" if args.force_rerun else ""))

    results_by_key: dict[tuple[str, str, int], dict] = dict(existing_by_key)
    write_mode = "a" if episodes_path.exists() else "w"
    with open(episodes_path, write_mode, encoding="utf-8") as f:
        for idx, row in enumerate(rows_to_run, start=1):
            scene_id = row.get("scene_id")
            sign_code = row.get("_sign_code")
            print(f"[{idx}/{len(rows_to_run)}] sign={sign_code} scene={scene_id}")
            
            if args.debug_one_way_sign_selection:
                row["debug_one_way_sign_selection"] = True
            gif_path = None
            if gifs_dir is not None:
                seed_val = int(row.get("seed") or row.get("deterministic_seed") or 0)
                var_idx = int(row.get("var_idx", 0) or 0)
                uid = f"{scene_id or 'scene'}_v{var_idx}_s{seed_val}"
                gif_path = gifs_dir / f"{uid}_{args.policy}_{args.ego_variant}.gif"
            episode_t0 = time.time()

            r = run_one_episode(
                row=row,
                policy_type=args.policy,
                models=models,
                scenes_root=scenes_root,
                max_steps=args.max_steps,
                ego_variant=args.ego_variant,
                ego_sample_seed_base=args.ego_sample_seed_base,
                replay_root=replay_root,
                save_gif=gif_path,
                hide_signs=args.hide_signs,
            )
            episode_dt = time.time() - episode_t0
            print(f"{args.policy}  elapsed_s={episode_dt:.3f}")

            key = _episode_key_from_row(row)
            results_by_key[key] = r
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()

    results: list[dict] = list(results_by_key.values())
    summary = aggregate_results(results)
    summary_path = out_dir / f"summary_{args.policy}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    ok_runs = sum(1 for r in results if r.get("ok"))
    print("\n=== Done ===")
    print(f"Episodes OK: {ok_runs}/{len(results)}")
    print(f"Episodes: {episodes_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
