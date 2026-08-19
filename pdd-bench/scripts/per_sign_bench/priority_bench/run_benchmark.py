from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

# priority_bench/ (this package) + per_sign_bench/ on path
_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))
_PER_SIGN_DIR = _PACKAGE_DIR.parent
if str(_PER_SIGN_DIR) not in sys.path:
    sys.path.insert(0, str(_PER_SIGN_DIR))
_PDD_BENCH = _PER_SIGN_DIR.parent.parent
if str(_PDD_BENCH) not in sys.path:
    sys.path.insert(0, str(_PDD_BENCH))

# Keep metadrive package unmodified: strip split-via shortcuts at graph build.
from core.runtime.metadrive_sumo_patch import apply_metadrive_sumo_via_patch  # noqa: E402
from core.runtime.checkpoints import (  # noqa: E402
    CHECKPOINTS_DIR,
    DEFAULT_MODEL_PATHS,
    NN_NEED_CHECKPOINT,
    PLAIN_PLANT2_POLICIES,
    PLANT2_POLICIES,
    resolve_nn_checkpoint,
)
from bench.top_down_text_patch import apply_top_down_violations_text_patch  # noqa: E402
from bench.top_down_path_conflict_patch import (  # noqa: E402
    apply_top_down_path_conflict_overlay_patch,
    is_path_conflict_overlay_enabled,
    set_path_conflict_overlay_enabled,
)

apply_metadrive_sumo_via_patch()
apply_top_down_violations_text_patch()
apply_top_down_path_conflict_overlay_patch()

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
    numpy_legacy_seed,
    sample_ego_params,
)
from traffic_signs.priority_signs import (
    MainRoadSign,
    RightHandYieldSign,
    RoundaboutSign,
    RoundaboutYieldSign,
    SecondaryRoadLeftSign,
    SecondaryRoadRightSign,
    SecondaryRoadSign,
    StopSign,
    YieldSign,
)
from traffic_signs.no_traffic_sign import NoTrafficSign
from traffic_signs.pedestrian_crossing_sign import PedestrianCrossingSign
from traffic_signs.pedestrian_yield_rule import PedestrianYieldRule
from traffic_signs.detour_sign import DetourRightSign, DetourLeftSign, DetourEitherSign
from traffic_signs.speed_limit_sign import SpeedLimitSign
from traffic_signs.min_speed_limit_sign import MinimumSpeedLimitSign
from traffic_signs.zone_signs import ZoneSpeedLimitSign
from traffic_signs.residential_zone_signs import ResidentialZoneSign, EndOfResidentialZoneSign
from traffic_signs.end_of_zone_signs import EndOfSpeedLimitSign, EndOfZoneSpeedLimitSign
from core.sumo.lane_keys import clamp_lane_key_to_graph, lane_edge_id, make_lane_key
from core.runtime.one_way_support import (
    OneWaySumoTrafficManager,
    install_one_way_compliant_nav_route,
    resolve_row_background_excluded_edges,
)
from core.scenarios.one_way_bridge import (
    get_one_way_sign_spec,
    resolve_sign_class as resolve_one_way_sign_class,
)
from core.scenarios.direction_bridge import (
    get_direction_sign_spec,
    resolve_sign_class as resolve_direction_sign_class,
)
from core.scenarios.no_turn_bridge import (
    get_no_turn_sign_spec,
    resolve_sign_class as resolve_no_turn_sign_class,
)
from core.scenarios.no_entry_bridge import (
    get_no_entry_sign_spec,
    resolve_sign_class as resolve_no_entry_sign_class,
)
from core.layout.junction_priority_layout import right_arm_edge_id, straight_arm_edge_id
from core.scenarios.auxiliary_agent import (
    DEFAULT_CONVOY_GAP_M,
    DEFAULT_CONVOY_SIZE,
    DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END,
    DEFAULT_SPAWN_VELOCITY_MS,
    add_auxiliary_agents,
    resolve_aux_spawn_plan,
)
from core.manifest.manifest_config import (
    DEFAULT_AUX_LANES_OCCUPIED_MAX,
    DEFAULT_COMPLIANT_STOP_MAX_DIST_M,
    DEFAULT_COMPLIANT_STOP_SPEED_MPS,
    DEFAULT_COMPLIANT_STOP_SUCCESS_SECONDS,
    DEFAULT_DESTINATION_MAX_ALONG_M,
    DEFAULT_SIGN_DISTANCE_FROM_START,
)
from core.layout.junction_sign_placement import (
    SIGN_SHOULDER_OFFSET_M,
    arms_for_road_class,
    collect_lanes_for_keys,
    lateral_offset_beside_lane,
    resolve_sign_lane_for_edge,
    sign_longitudinal_offset,
    sign_longitudinal_offset_from_start,
    sign_placement_long,
    sign_placement_long_from_start,
)
from core.layout.junction_priority_layout import (
    JunctionLayoutError,
    build_junction_priority_layout,
    secondary_side_from_main_arm,
)
from core.layout.roundabout_topology import build_roundabout_layout
from core.layout.roundabout_yield_zone import (
    all_entry_conflict_ring_edges,
    collect_all_entry_conflict_lanes,
    collect_entry_conflict_lanes,
    collect_lanes_for_edge_ids,
    conflict_aux_ring_edge_ids,
    entry_conflict_ring_edges,
)
from core.scenarios.scene_augmentation import _roundabout_meta_ring_kwargs
from core.manifest.manifest_config import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_STOP_WAIT_STEPS,
    enrich_manifest_row,
    load_manifest_config,
)

BENCH_DIR = Path(__file__).resolve().parent
PER_SIGN_BENCH_DIR = BENCH_DIR.parent
PDD_BENCH_DIR = PER_SIGN_BENCH_DIR.parent.parent
SDC_ROOT = PDD_BENCH_DIR.parent


def resolve_model_path(policy: str, model_path: str | None) -> str | None:
    """Use ``--model-path`` when set; else fall back to repo defaults."""
    return resolve_nn_checkpoint(policy, model_path)

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
    val = profile.get("horizon_steps")
    if val is None:
        val = row.get("horizon", fallback)
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
    SumoTrafficManager.EGO_SAFE_RADIUS = 15
    _apply_manifest_profile_to_npcs(row)
    traffic_density = _manifest_traffic_density(row, default=0.0)
    horizon = _manifest_horizon(row, fallback=max_steps)
    net_path = str(scenes_root / row["net_path"]) if not str(row["net_path"]).startswith("/") else str(row["net_path"])
    sign_spawn_distance = _resolve_sign_spawn_distance(row, scenes_root)
    background_excluded_edges = (
        resolve_row_background_excluded_edges(row, net_path)
        if _row_is_one_way(row)
        else []
    )

    vehicle_config: dict = {"show_lidar": False}
    spawn_vel = float(row.get("spawn_velocity_ms", 0.0) or 0.0)
    if spawn_vel > 0:
        vehicle_config["spawn_velocity"] = [spawn_vel, 0.0]
        vehicle_config["spawn_velocity_car_frame"] = True

    use_ped = bool(row.get("use_pedestrian_manager", False))
    use_yield = bool(row.get("use_pedestrian_yield_rule", False))
    ped_cfg = dict(row.get("pedestrian_manager") or {})

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
        background_excluded_edges=list(background_excluded_edges),
        skip_auto_signs=True,
        use_pedestrian_manager=use_ped,
        use_pedestrian_yield_rule=use_yield,
        enforce_pedestrian_yield_for_traffic=False,
    )
    if ped_cfg:
        config["pedestrian_manager"] = ped_cfg
    if row.get("road_id"):
        config["vehicle_config"]["spawn_lane_index"] = row["road_id"]
    if "spawn_lane_num" in row:
        config["spawn_lane_num"] = int(row["spawn_lane_num"])
    if row.get("destination_lane_id"):
        config["vehicle_config"]["destination"] = row["destination_lane_id"]

    is_one_way = _row_is_one_way(row)

    class _RealMapEnv(TrafficSignSumoEnv):
        @classmethod
        def default_config(cls):
            cfg = super().default_config()
            cfg["traffic_density"] = 0.0
            cfg["show_lane_arrows"] = True
            cfg["show_traffic_lights"] = True
            cfg["show_npc_vehicles"] = True
            cfg["skip_auto_signs"] = False
            cfg["background_excluded_edges"] = []
            return cfg

        def setup_engine(self):
            super().setup_engine()
            # Only add SumoTrafficManager if traffic_density > 0
            # Otherwise keep the default SimpleTrafficManager (no NPC spawning)
            if self.config.get("traffic_density", 0.0) > 0:
                mgr = OneWaySumoTrafficManager() if is_one_way else SumoTrafficManager()
                self.engine.update_manager("traffic_manager", mgr)

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
    direct = row.get("sign_spawn_distance")
    if direct is not None:
        return max(float(direct), 30.0)

    direct = row.get("distance_from_start")
    if direct is not None:
        return max(float(direct), 30.0)

    net_path = row.get("net_path")
    if not net_path:
        return 0.0

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
            distance = float(meta.get("distance_from_start", 0.0) or 0.0)
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


def _is_ego_in_yield_zone(sign_mgr, vehicle) -> bool:
    """True when ego is in a YieldSign / RightHandYieldSign approach zone."""
    if sign_mgr is None or vehicle is None:
        return False
    for sign in getattr(sign_mgr, "signs", []) or []:
        if not isinstance(sign, YieldSign):
            continue
        if isinstance(sign, MainRoadSign):
            continue
        if _ego_in_sign_zone(sign, vehicle):
            return True
    return False


def _is_aux_in_main_zone(sign_mgr, aux_vehicles, ego_vehicle=None) -> bool:
    """True when any aux is in the main conflict zone (GIF / debug).

    Uses geometric main-zone presence so gated (not-yet-released) aux still
    count — matching what the camera shows. Yield decisions continue to ignore
    gated aux via ``_is_waiting_gated_aux``.
    """
    if sign_mgr is None or not aux_vehicles:
        return False
    yield_signs = [
        sign
        for sign in (getattr(sign_mgr, "signs", []) or [])
        if isinstance(sign, YieldSign) and not isinstance(sign, MainRoadSign)
    ]
    if not yield_signs:
        return False
    for aux in aux_vehicles:
        if aux is None:
            continue
        for sign in yield_signs:
            try:
                if sign.is_vehicle_on_main_road(aux):
                    return True
            except Exception:
                continue
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
    elif policy in PLAIN_PLANT2_POLICIES:
        if not model_path:
            raise ValueError(f"--model-path is required for --policy {policy}")
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
        clamped_spawn = clamp_lane_key_to_graph(target_key, road_network.graph)
        if clamped_spawn and clamped_spawn != target_key:
            print(f"[EgoSpawn] Clamped spawn {target_key} -> {clamped_spawn}")
            target_key = clamped_spawn
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


def _apply_manifest_ego_spawn_velocity(env, row: dict) -> None:
    """Re-apply spawn speed after skip_auto_signs teleport."""
    v = float(row.get("spawn_velocity_ms") or 0.0)
    if v <= 0:
        return
    vehicle = getattr(env, "agent", None) or getattr(env, "vehicle", None)
    if vehicle is None:
        return
    try:
        vehicle.set_velocity([v, 0.0], in_local_frame=True)
    except TypeError:
        try:
            vehicle.set_velocity([v, 0.0])
        except Exception:
            pass
    except Exception as exc:
        print(f"[EgoSpawn] Could not set spawn velocity {v:.2f} m/s: {exc}")


def _apply_manifest_ego_destination(env, row: dict) -> Optional[str]:
    """Clamp ego destination to a real graph lane and re-bind navigation."""
    dest = row.get("destination_lane_id")
    if not dest:
        return None
    try:
        vehicle = env.agent
        if vehicle is None:
            return None
        road_network = env.engine.current_map.road_network
        clamped = clamp_lane_key_to_graph(str(dest), road_network.graph)
        if not clamped:
            return None
        if clamped != str(dest):
            print(f"[EgoDest] Clamped destination {dest} -> {clamped}")
        spawn_key = getattr(vehicle.lane, "index", None)
        if spawn_key and vehicle.navigation is not None:
            vehicle.navigation.set_route(spawn_key, clamped)
        _apply_destination_along_cap(env, row)
        return clamped
    except Exception as exc:
        print(f"[EgoDest] Could not apply destination {dest}: {exc}")
        return None


def _apply_destination_along_cap(env, row: dict) -> None:
    """Cap ego finish to ``min(cap, final_lane.length-5)`` on the final lane.

    Used by roundabout (4.3) and blocked_road (3.2): sets
    ``_priority_bench_dest_along_m`` for GIF/top-down and moves MetaDrive
    ``_dest_node_path``. Arrive for both signs uses the same cap (compliant-stop
    remains an alternate success path for 3.2). Violation on 3.2 is driving
    past the no-entry sign (sign manager), not a separate past-sign distance.
    """
    raw = row.get("destination_max_along_m")
    if raw is None and not (
        _row_is_roundabout(row)
        or _row_is_blocked_road(row)
        or _row_uses_dual_path_nav(row)
        or _row_is_crosswalk(row)
        or _row_is_detour(row)
        or _row_is_speed(row)
    ):
        return
    if raw is None and _row_uses_dual_path_nav(row):
        return
    if raw is None and _row_is_crosswalk(row):
        raw = 40.0
    try:
        cap = float(DEFAULT_DESTINATION_MAX_ALONG_M if raw is None else raw)
    except (TypeError, ValueError):
        return
    if cap <= 0.0:
        return

    vehicle = getattr(env, "agent", None) or getattr(env, "vehicle", None)
    if vehicle is None:
        return
    nav = getattr(vehicle, "navigation", None)
    final = getattr(nav, "final_lane", None) if nav is not None else None
    if final is None:
        return
    try:
        target = min(cap, max(0.5, float(final.length) - 5.0))
    except Exception:
        return

    try:
        vehicle._priority_bench_dest_along_m = float(target)
    except Exception:
        pass
    try:
        if nav is not None:
            nav._priority_bench_dest_along_m = float(target)
    except Exception:
        pass

    try:
        from metadrive.utils.coordinates_shift import panda_vector

        dest_path = getattr(nav, "_dest_node_path", None)
        if dest_path is not None:
            check_point = final.position(target, 0.0)
            height = float(getattr(nav, "MARK_HEIGHT", 1.0) or 1.0)
            dest_path.setPos(
                panda_vector(float(check_point[0]), float(check_point[1]), height)
            )
    except Exception:
        pass

    try:
        if _row_is_blocked_road(row):
            label = "Blocked-road"
        elif _row_is_one_way(row):
            label = "One-way"
        elif _row_is_direction(row):
            label = "Direction"
        elif _row_is_crosswalk(row):
            label = "Crosswalk"
        elif _row_is_detour(row):
            label = "Detour"
        elif _row_is_speed(row):
            label = "Speed"
        else:
            label = "Roundabout"
        print(
            f"[EgoDest] {label} destination cap at {target:.1f}m "
            f"on final lane (len={float(final.length):.1f}m)"
        )
    except Exception:
        pass


# Back-compat alias (older call sites / notebooks).
_apply_roundabout_destination_cap = _apply_destination_along_cap


def _lane_index_road_key(lane_index) -> tuple | str | None:
    """Comparable road identity for a MetaDrive / SUMO lane index."""
    if lane_index is None:
        return None
    if isinstance(lane_index, str):
        try:
            return lane_edge_id(lane_index) or lane_index
        except Exception:
            return lane_index
    try:
        if len(lane_index) >= 2:
            return (lane_index[0], lane_index[1])
    except Exception:
        pass
    return lane_index


def _same_road_lane_index(a, b) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    return _lane_index_road_key(a) == _lane_index_road_key(b)


def _ego_reached_capped_destination(
    vehicle,
    *,
    max_along_m: float,
    arrive_tol_m: float = 2.0,
    allow_same_lane: bool = False,
) -> bool:
    """True when ego is on the route's final exit lane at/after the cap.

    Destination point = ``min(max_along_m, final_lane.length - 5)`` along the
    navigation final lane (MetaDrive end criterion, capped when the exit is
    longer). Works with ``EdgeNetworkNavigation`` (SUMO: checkpoints are lane
    indices) and node-network checkpoints.

    ``max_along_m <= 0`` disables the cap.
    """
    if vehicle is None:
        return False
    try:
        cap = float(max_along_m)
    except (TypeError, ValueError):
        return False
    if cap <= 0.0:
        return False

    nav = getattr(vehicle, "navigation", None)
    final = getattr(nav, "final_lane", None) if nav else None
    lane = getattr(vehicle, "lane", None)
    if nav is None or final is None or lane is None:
        return False

    lane_idx = getattr(lane, "index", None)
    final_idx = getattr(final, "index", None)
    if lane_idx is None or final_idx is None:
        return False

    checkpoints = list(getattr(nav, "checkpoints", None) or [])
    if checkpoints:
        first_cp = checkpoints[0]
        last_cp = checkpoints[-1]
        # Degenerate route (spawn lane == dest lane) — never early-arrive,
        # except detour: finish is a along-cap on the same obstacle edge.
        if (
            not allow_same_lane
            and _same_road_lane_index(first_cp, last_cp)
            and len(checkpoints) <= 2
        ):
            return False
        # Must be on the final checkpoint / final_lane road — not the approach.
        on_final = _same_road_lane_index(lane_idx, last_cp) or _same_road_lane_index(
            lane_idx, final_idx
        )
        if not on_final:
            return False
        if not _same_road_lane_index(first_cp, last_cp) and _same_road_lane_index(
            lane_idx, first_cp
        ):
            return False
    elif not _same_road_lane_index(lane_idx, final_idx):
        return False

    try:
        lane_len = float(final.length)
        # Use the vehicle's current lane coords (same road as final).
        long, _lat = lane.local_coordinates(vehicle.position)
    except Exception:
        return False

    target = min(cap, max(0.5, lane_len - 5.0))
    return float(long) >= (target - float(arrive_tol_m))


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
        # spawn_longitude is from lane START, so: lane_length - distance_before_end
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


def _row_is_yield(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "")
    sign_type = str(row.get("sign_type") or "")
    return code in {"2.4", "2_4"} or sign_type == "yield"


def _row_is_stop(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "")
    sign_type = str(row.get("sign_type") or "")
    return code in {"2.5", "2_5"} or sign_type in {"stop", "stop_sign"}


def _row_is_secondary_road(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "")
    sign_type = str(row.get("sign_type") or "")
    return code in {
        "2.3",
        "2_3",
        "2.3.1",
        "2.3.2",
        "2.3.3",
    } or sign_type in {"secondary", "secondary_road"}


def _row_is_roundabout(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "")
    sign_type = str(row.get("sign_type") or "")
    return code in {"4.3", "4_3"} or sign_type == "roundabout"


def _row_is_blocked_road(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "")
    sign_type = str(row.get("sign_type") or "")
    return code in {"3.2", "3_2"} or sign_type == "blocked_road"


def _row_is_one_way(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "")
    sign_type = str(row.get("sign_type") or row.get("sign_family") or "")
    return code in {"5.7.1", "5_7_1", "5.7.2", "5_7_2"} or sign_type in {
        "one_way",
        "one_way_right",
        "one_way_left",
    }


_DIRECTION_CODES = {
    "4.1.1",
    "4_1_1",
    "4.1.2",
    "4_1_2",
    "4.1.3",
    "4_1_3",
    "4.1.4",
    "4_1_4",
    "4.1.5",
    "4_1_5",
    "4.1.6",
    "4_1_6",
}
_DIRECTION_TYPES = {
    "direction",
    "direction_straight",
    "direction_right",
    "direction_left",
    "direction_straight_right",
    "direction_straight_left",
    "direction_left_right",
}


def _row_is_direction(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "")
    sign_type = str(row.get("sign_type") or row.get("sign_family") or "")
    return code in _DIRECTION_CODES or sign_type in _DIRECTION_TYPES


_NO_TURN_CODES = {"3.18.1", "3_18_1", "3.18.2", "3_18_2"}
_NO_TURN_TYPES = {"no_turn", "no_turn_right", "no_turn_left"}


def _row_is_no_turn(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "")
    sign_type = str(row.get("sign_type") or row.get("sign_family") or "")
    return code in _NO_TURN_CODES or sign_type in _NO_TURN_TYPES


def _row_is_no_entry(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "")
    sign_type = str(row.get("sign_type") or row.get("sign_family") or "")
    return code in {"3.1", "3_1"} or sign_type == "no_entry"


def _row_is_crosswalk(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "")
    sign_type = str(row.get("sign_type") or row.get("sign_family") or "")
    if bool(row.get("place_crosswalk_sign")):
        return True
    return code in {"5.19", "5_19", "5.19.1", "5.19.2"} or sign_type == "crosswalk"


def _row_is_detour(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "")
    sign_type = str(row.get("sign_type") or row.get("sign_family") or "")
    if bool(row.get("place_detour_sign")):
        return True
    return code in {"4.2.1", "4.2.2", "4.2.3", "4_2_1", "4_2_2", "4_2_3"} or sign_type == "detour"


def _row_is_speed(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "")
    sign_type = str(row.get("sign_type") or row.get("sign_family") or "")
    if bool(row.get("place_speed_sign")):
        return True
    return code in {"3.24", "4.6", "5.21", "5.31", "3_24", "4_6", "5_21", "5_31"} or sign_type == "speed"


def _row_uses_dual_path_nav(row: dict) -> bool:
    """5.7 / 4.1 / 3.18 / 3.1: truncated dests; only rule-compliant policies replan."""
    return (
        _row_is_one_way(row)
        or _row_is_direction(row)
        or _row_is_no_turn(row)
        or _row_is_no_entry(row)
    )


_ONE_WAY_COMPLIANT_NAV_POLICIES = frozenset({
    "modified_idm",
    "comprehensive_rule_expert",
    "rule_compliant",
    "carl_rule",
    "plant2_rule",
})


def _resolve_dual_path_row_for_policy(row: dict, policy_type: str) -> dict:
    """Pick baseline vs compliant dest (and along-cap) for the active policy."""
    if not _row_uses_dual_path_nav(row):
        return row
    out = dict(row)
    use_compliant = policy_type in _ONE_WAY_COMPLIANT_NAV_POLICIES
    if use_compliant:
        dest = row.get("compliant_destination_lane_id") or row.get("destination_lane_id")
        along = row.get("compliant_destination_max_along_m")
        if along is None:
            along = row.get("destination_max_along_m")
    else:
        dest = row.get("baseline_destination_lane_id") or row.get("destination_lane_id")
        along = row.get("baseline_destination_max_along_m")
        if along is None:
            along = row.get("destination_max_along_m")
    if dest:
        out["destination_lane_id"] = dest
        edge = lane_edge_id(str(dest))
        if edge:
            out["destination_edge_id"] = edge
    if along is not None:
        out["destination_max_along_m"] = float(along)
    elif out.get("destination_max_along_m") is None:
        out["destination_max_along_m"] = 1e9
    return out


def _resolve_one_way_row_for_policy(row: dict, policy_type: str) -> dict:
    return _resolve_dual_path_row_for_policy(row, policy_type)


def _resolve_one_way_pdd_code(row: dict) -> str:
    code = (
        row.get("_sign_code")
        or row.get("sign_code")
        or row.get("pdd_code")
        or row.get("sign_type")
        or "5.7.1"
    )
    try:
        return get_one_way_sign_spec(str(code).strip()).pdd_code
    except ValueError:
        return "5.7.1"


def _resolve_direction_pdd_code(row: dict) -> str:
    code = (
        row.get("_sign_code")
        or row.get("sign_code")
        or row.get("pdd_code")
        or row.get("sign_type")
        or "4.1.1"
    )
    try:
        return get_direction_sign_spec(str(code).strip()).pdd_code
    except ValueError:
        return "4.1.1"


def _resolve_no_turn_pdd_code(row: dict) -> str:
    code = (
        row.get("_sign_code")
        or row.get("sign_code")
        or row.get("pdd_code")
        or row.get("sign_type")
        or "3.18.1"
    )
    try:
        return get_no_turn_sign_spec(str(code).strip()).pdd_code
    except ValueError:
        return "3.18.1"


def _resolve_no_entry_pdd_code(row: dict) -> str:
    code = (
        row.get("_sign_code")
        or row.get("sign_code")
        or row.get("pdd_code")
        or row.get("sign_type")
        or "3.1"
    )
    try:
        return get_no_entry_sign_spec(str(code).strip()).pdd_code
    except ValueError:
        return "3.1"


def _row_is_main_secondary(row: dict) -> bool:
    """Yield / stop / 2.3 / roundabout share secondary-ego road_class."""
    return (
        _row_is_yield(row)
        or _row_is_stop(row)
        or _row_is_secondary_road(row)
        or _row_is_roundabout(row)
    )


def _get_junction_layout(row: dict, scenes_root: Path) -> dict | None:
    """Load junction layout from manifest row or build from scene net.xml."""
    if row.get("junction_layout"):
        layout = row["junction_layout"]
        if _row_is_roundabout(row):
            if layout.get("shape") == "O" and layout.get("mode") == "roundabout":
                return layout
        else:
            return layout

    net_path = row.get("net_path")
    if not net_path:
        return None

    net_file = Path(str(net_path))
    full_path = net_file if net_file.is_absolute() else scenes_root / net_file

    if _row_is_roundabout(row):
        scene_meta: dict | None = None
        meta_path = full_path.parent / "meta.json"
        if meta_path.is_file():
            try:
                scene_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                scene_meta = None
        ego_edge = row.get("road_id")
        if scene_meta:
            ego_edge = (
                scene_meta.get("catalog_sign_road_id")
                or scene_meta.get("road_id")
                or ego_edge
            )
        try:
            layout_obj = build_roundabout_layout(
                full_path,
                sign_edge_id=ego_edge,
                **_roundabout_meta_ring_kwargs(scene_meta),
            )
        except JunctionLayoutError as exc:
            print(f"[JunctionLayout] Failed to build roundabout layout: {exc}")
            return None
        if layout_obj.shape != "O" or layout_obj.mode != "roundabout":
            print(
                f"[JunctionLayout] Rejecting non-roundabout layout "
                f"(shape={layout_obj.shape}, mode={layout_obj.mode})"
            )
            return None
        return layout_obj.to_dict()

    mode = "main_secondary" if _row_is_main_secondary(row) else "main_main"
    try:
        layout = build_junction_priority_layout(full_path, mode=mode)
    except JunctionLayoutError as exc:
        print(f"[JunctionLayout] Failed to build layout: {exc}")
        return None
    return layout.to_dict()


def _ensure_secondary_ego_spawn(row: dict, scenes_root: Path) -> None:
    """Ensure manifest spawn edge is on a secondary / spoke junction arm."""
    if not _row_is_main_secondary(row):
        return
    layout = _get_junction_layout(row, scenes_root)
    if not layout:
        return

    secondary_ids = set(layout.get("secondary_edge_ids") or [])
    road_id = row.get("road_id")
    if road_id and road_id in secondary_ids:
        return

    seed = int(row.get("seed") or row.get("deterministic_seed") or 0)
    pool: list[tuple[str, int]] = []
    for arm in layout.get("arms", []):
        if arm.get("road_class") != "secondary":
            continue
        edge_id = arm["edge_id"]
        lane_keys = arm.get("lane_keys") or []
        if not lane_keys:
            pool.append((edge_id, int(row.get("spawn_lane_num", 0) or 0)))
            continue
        for lane_key in lane_keys:
            try:
                lane_num = int(str(lane_key).rsplit("_", 1)[1])
            except (ValueError, IndexError):
                lane_num = 0
            pool.append((edge_id, lane_num))

    if not pool:
        label = "spoke" if _row_is_roundabout(row) else "secondary"
        print(f"[EgoSpawn] No {label} arms in layout; keeping original spawn")
        return

    edge_id, lane_num = pool[seed % len(pool)]
    if road_id and road_id != edge_id:
        print(
            f"[EgoSpawn] Repicked spawn from main/non-secondary {road_id!r} "
            f"-> secondary {edge_id!r} lane {lane_num}"
        )
    row["road_id"] = edge_id
    row["spawn_lane_num"] = lane_num


def _clear_sign_manager(sign_mgr) -> None:
    sign_mgr.signs.clear()
    # Keep logical rules (PedestrianYieldRule). They are registered once in
    # setup_engine and are not re-created with plates. Clearing them here made
    # 5.19 yield / violations a no-op for every policy.


def _ensure_pedestrian_yield_rule(env) -> None:
    """Re-register PedestrianYieldRule if sign placement left the manager empty."""
    engine = getattr(env, "engine", None)
    sign_mgr = getattr(engine, "traffic_sign_manager", None) if engine is not None else None
    if sign_mgr is None:
        return
    if any(type(rule).__name__ == "PedestrianYieldRule" for rule in getattr(sign_mgr, "rules", []) or []):
        return
    ped_cfg = engine.global_config.get("pedestrian_manager", {})
    if hasattr(ped_cfg, "get_dict"):
        ped_cfg = ped_cfg.get_dict()
    ped_cfg = dict(ped_cfg or {})
    sign_mgr.add_rule(
        PedestrianYieldRule(
            yield_distance=float(ped_cfg.get("yield_distance", 12.0)),
            yield_speed_kmh=float(ped_cfg.get("yield_speed_kmh", 8.0)),
            no_stop_before_m=float(ped_cfg.get("no_stop_before_crosswalk_m", 0.0)),
            no_stop_speed_kmh=float(ped_cfg.get("no_stop_speed_kmh", 1.0)),
            no_stop_min_duration_s=float(ped_cfg.get("no_stop_min_duration_s", 1.0)),
        )
    )
    print("[PedestrianYieldRule] re-registered after 5.19 placement")


def _place_main_sign_on_spawn_lane(
    env, distance_before_end: float = 20.0, show_model: bool = True
) -> bool:
    """Fallback: place a single MainRoadSign beside the ego approach road."""
    try:
        vehicle = env.agent
        if vehicle is None or vehicle.lane is None:
            return False

        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False

        _clear_sign_manager(sign_mgr)
        lane = vehicle.lane
        placement_long = sign_placement_long(lane, distance_before_end)
        sign = sign_mgr.add_sign(
            MainRoadSign,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=show_model,
            use_random_lane=False,
        )
        if sign is not None:
            sign.is_priority_sign = False
        return sign is not None
    except Exception as e:
        print(f"[MainRoadSign] Failed to place sign: {e}")
        return False


def _place_equal_priority_main_signs(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Place MainRoadSign (2.1) on every incoming arm — equal priority intersection."""
    layout = _get_junction_layout(row, scenes_root)
    if layout is None:
        print("[JunctionSigns] No layout available, falling back to ego-only main sign")
        return _place_main_sign_on_spawn_lane(
            env, distance_before_end=distance_before_end, show_model=show_model
        )

    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return False

    _clear_sign_manager(sign_mgr)

    all_arms = layout.get("arms", [])
    if not all_arms:
        print("[JunctionSigns] No arms in layout, falling back to ego-only main sign")
        return _place_main_sign_on_spawn_lane(
            env, distance_before_end=distance_before_end, show_model=show_model
        )

    junction_id = layout.get("junction_id", "")
    placed_main = 0

    for arm in all_arms:
        edge_id = arm.get("edge_id", "")
        lane = resolve_sign_lane_for_edge(env, edge_id, arm.get("lane_keys", []))
        if lane is None:
            print(f"[JunctionSigns] Skipping main sign, lane not found for edge: {edge_id}")
            continue
        placement_long = sign_placement_long(lane, distance_before_end)
        try:
            sign = sign_mgr.add_sign(
                MainRoadSign,
                lane=lane,
                longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
                lateral_offset=lateral_offset_beside_lane(lane, placement_long),
                show_model=show_model,
                use_random_lane=False,
                intersection_name=junction_id,
            )
            if sign is not None:
                sign.is_priority_sign = False
            placed_main += 1
        except Exception as exc:
            print(f"[JunctionSigns] Failed MainRoadSign on edge {edge_id}: {exc}")

    print(
        f"[JunctionSigns] Placed {placed_main} MainRoadSign(s) "
        f"at junction {junction_id} ({layout.get('shape')}), "
        f"shoulder offset={SIGN_SHOULDER_OFFSET_M}m"
    )
    return placed_main > 0


def _place_right_hand_yield_tracker(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
) -> bool:
    """Invisible RightHandYieldSign on ego lane for violation metrics."""
    layout = _get_junction_layout(row, scenes_root)
    if layout is None:
        return False

    ego_edge = row.get("road_id")
    if not ego_edge:
        return False

    right_edge = right_arm_edge_id(layout, str(ego_edge))
    if not right_edge:
        print(f"[RightHandRule] No right arm for ego edge {ego_edge!r}")
        return False

    right_lane_keys = []
    for arm in layout.get("arms", []):
        if arm.get("edge_id") == right_edge:
            right_lane_keys = list(arm.get("lane_keys", []))
            break

    right_lanes = collect_lanes_for_keys(env, right_lane_keys)
    if not right_lanes:
        print(f"[RightHandRule] Could not resolve lanes for right arm {right_edge}")
        return False

    vehicle = env.agent
    if vehicle is None or vehicle.lane is None:
        return False

    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return False

    lane = vehicle.lane
    placement_long = sign_placement_long(lane, distance_before_end)
    try:
        sign = sign_mgr.add_sign(
            RightHandYieldSign,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=False,
            use_random_lane=False,
            intersection_name=layout.get("junction_id", ""),
            right_road_lanes=right_lanes,
        )
        if sign is not None:
            sign.is_priority_sign = False
        print(
            f"[RightHandRule] Tracker on ego {ego_edge}, "
            f"yield-to-right arm {right_edge} ({len(right_lanes)} lane(s))"
        )
        return sign is not None
    except Exception as exc:
        print(f"[RightHandRule] Failed to place tracker: {exc}")
        return False


def _place_secondary_sign_on_spawn_lane(
    env,
    secondary_sign_cls,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Fallback: place a single secondary-arm plate beside the ego approach."""
    label = secondary_sign_cls.__name__
    try:
        vehicle = env.agent
        if vehicle is None or vehicle.lane is None:
            return False

        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False

        _clear_sign_manager(sign_mgr)
        lane = vehicle.lane
        placement_long = sign_placement_long(lane, distance_before_end)
        sign = sign_mgr.add_sign(
            secondary_sign_cls,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=show_model,
            use_random_lane=False,
            auto_detect_main_roads=False,
        )
        if sign is not None:
            sign.is_priority_sign = False
        return sign is not None
    except Exception as e:
        print(f"[{label}] Failed to place sign: {e}")
        return False


def _place_main_secondary_junction_signs(
    env,
    row: dict,
    scenes_root: Path,
    secondary_sign_cls,
    distance_before_end: float = 20.0,
    show_model: bool = True,
    secondary_sign_for_arm=None,
) -> bool:
    """Place MainRoadSign on main arms and secondary plates on secondary arms.

    Shared by yield (2.4 / YieldSign) and stop (2.5 / StopSign). Keeps priority_bench
    outgoing_edge_ids exclusions that the standalone stop_sign bench lacked.

    ``secondary_sign_for_arm(arm) -> sign_cls`` optionally picks a different plate
    per secondary arm (stop X: ego arm StopSign, opposite YieldSign).
    """
    label = secondary_sign_cls.__name__
    layout = _get_junction_layout(row, scenes_root)
    if layout is None:
        print(f"[JunctionSigns] No layout available, falling back to ego-only {label}")
        return _place_secondary_sign_on_spawn_lane(
            env,
            secondary_sign_cls,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )

    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return False

    _clear_sign_manager(sign_mgr)

    main_arms = arms_for_road_class(layout, "main")
    secondary_arms = arms_for_road_class(layout, "secondary")
    if not main_arms or not secondary_arms:
        print(
            f"[JunctionSigns] Missing main/secondary arms, falling back to ego-only {label}"
        )
        return _place_secondary_sign_on_spawn_lane(
            env,
            secondary_sign_cls,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )

    main_lanes = []
    outgoing_edge_ids: set[str] = set()
    for arm in main_arms:
        main_lanes.extend(collect_lanes_for_keys(env, arm.get("lane_keys", [])))
        for out_edge in arm.get("outgoing_to") or arm.get("straight_to") or []:
            outgoing_edge_ids.add(str(out_edge))
    for arm in secondary_arms:
        for out_edge in arm.get("outgoing_to") or arm.get("straight_to") or []:
            outgoing_edge_ids.add(str(out_edge))
    # Never treat a monitored main approach as outgoing.
    main_approach_edges = {str(arm.get("edge_id")) for arm in main_arms if arm.get("edge_id")}
    outgoing_edge_ids -= main_approach_edges

    if not main_lanes:
        print(
            f"[JunctionSigns] Could not resolve main lanes, falling back to ego-only {label}"
        )
        return _place_secondary_sign_on_spawn_lane(
            env,
            secondary_sign_cls,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )

    junction_id = layout.get("junction_id", "")
    placed_main = 0
    placed_by_cls: dict[str, int] = {}

    for arm in main_arms:
        edge_id = arm.get("edge_id", "")
        lane = resolve_sign_lane_for_edge(env, edge_id, arm.get("lane_keys", []))
        if lane is None:
            print(f"[JunctionSigns] Skipping main sign, lane not found for edge: {edge_id}")
            continue
        placement_long = sign_placement_long(lane, distance_before_end)
        try:
            sign = sign_mgr.add_sign(
                MainRoadSign,
                lane=lane,
                longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
                lateral_offset=lateral_offset_beside_lane(lane, placement_long),
                show_model=show_model,
                use_random_lane=False,
                intersection_name=junction_id,
            )
            if sign is not None:
                sign.is_priority_sign = False
            placed_main += 1
        except Exception as exc:
            print(f"[JunctionSigns] Failed MainRoadSign on edge {edge_id}: {exc}")

    for arm in secondary_arms:
        edge_id = arm.get("edge_id", "")
        lane = resolve_sign_lane_for_edge(env, edge_id, arm.get("lane_keys", []))
        if lane is None:
            print(f"[JunctionSigns] Skipping secondary sign, lane not found for edge: {edge_id}")
            continue
        arm_cls = secondary_sign_cls
        if secondary_sign_for_arm is not None:
            arm_cls = secondary_sign_for_arm(arm) or secondary_sign_cls
        arm_label = arm_cls.__name__
        placement_long = sign_placement_long(lane, distance_before_end)
        try:
            sign = sign_mgr.add_sign(
                arm_cls,
                lane=lane,
                longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
                lateral_offset=lateral_offset_beside_lane(lane, placement_long),
                show_model=show_model,
                use_random_lane=False,
                intersection_name=junction_id,
                main_road_lanes=main_lanes,
                outgoing_edge_ids=sorted(outgoing_edge_ids),
                auto_detect_main_roads=False,
            )
            if sign is not None:
                sign.is_priority_sign = False
            placed_by_cls[arm_label] = placed_by_cls.get(arm_label, 0) + 1
        except Exception as exc:
            print(f"[JunctionSigns] Failed {arm_label} on edge {edge_id}: {exc}")

    placed_secondary = sum(placed_by_cls.values())
    secondary_summary = ", ".join(
        f"{n} {name}(s)" for name, n in sorted(placed_by_cls.items())
    ) or f"0 {label}(s)"
    print(
        f"[JunctionSigns] Placed {placed_main} MainRoadSign(s) and "
        f"{secondary_summary} "
        f"at junction {junction_id} ({layout.get('shape')}), "
        f"shoulder offset={SIGN_SHOULDER_OFFSET_M}m"
    )
    return placed_main > 0 or placed_secondary > 0


def _place_yield_sign_on_spawn_lane(
    env, distance_before_end: float = 20.0, show_model: bool = True
) -> bool:
    return _place_secondary_sign_on_spawn_lane(
        env, YieldSign, distance_before_end=distance_before_end, show_model=show_model
    )


def _place_yield_junction_signs(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Place MainRoadSign on main arms and YieldSign on secondary arms."""
    return _place_main_secondary_junction_signs(
        env,
        row,
        scenes_root,
        YieldSign,
        distance_before_end=distance_before_end,
        show_model=show_model,
    )


def _place_stop_sign_on_spawn_lane(
    env, distance_before_end: float = 20.0, show_model: bool = True
) -> bool:
    return _place_secondary_sign_on_spawn_lane(
        env, StopSign, distance_before_end=distance_before_end, show_model=show_model
    )


def _stop_secondary_sign_for_arm(layout: dict, row: dict, arm: dict):
    """On X stop scenes: ego secondary → StopSign (2.5), opposite → YieldSign (2.4).

    T / single-secondary keep StopSign on every secondary arm. Priority logic is
    unchanged (main vs secondary); only the opposite plate differs visually.
    """
    if str(layout.get("shape") or "") != "X":
        return StopSign
    ego_edge = str(row.get("road_id") or "")
    edge_id = str(arm.get("edge_id") or "")
    if not ego_edge or not edge_id:
        return StopSign
    if edge_id == ego_edge:
        return StopSign
    opposite = straight_arm_edge_id(layout, ego_edge)
    if opposite is not None and edge_id == str(opposite):
        return YieldSign
    # Other non-ego secondary on X (should be the opposite in the usual layout).
    secondary_ids = {str(e) for e in (layout.get("secondary_edge_ids") or [])}
    if edge_id in secondary_ids and edge_id != ego_edge:
        return YieldSign
    return StopSign


def _place_stop_junction_signs(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Place MainRoadSign on main; StopSign on ego secondary; YieldSign opposite on X."""
    layout = _get_junction_layout(row, scenes_root)

    def _for_arm(arm: dict):
        if layout is None:
            return StopSign
        return _stop_secondary_sign_for_arm(layout, row, arm)

    return _place_main_secondary_junction_signs(
        env,
        row,
        scenes_root,
        StopSign,
        distance_before_end=distance_before_end,
        show_model=show_model,
        secondary_sign_for_arm=_for_arm,
    )


def _secondary_road_sign_cls_for_main_arm(layout: dict, main_edge_id: str):
    """X: 2.3.1 on all main arms. T: 2.3.2 (stem on right) / 2.3.3 (stem on left)."""
    if str(layout.get("shape") or "") == "X":
        return SecondaryRoadSign

    secondary_ids = list(layout.get("secondary_edge_ids") or [])
    if len(secondary_ids) != 1:
        return SecondaryRoadSign

    side = secondary_side_from_main_arm(layout, main_edge_id, secondary_ids[0])
    if side == "right":
        return SecondaryRoadRightSign  # 2.3.2
    if side == "left":
        return SecondaryRoadLeftSign  # 2.3.3
    return SecondaryRoadSign


def _secondary_road_plate_label(sign_cls) -> str:
    if sign_cls is SecondaryRoadRightSign:
        return "2.3.2"
    if sign_cls is SecondaryRoadLeftSign:
        return "2.3.3"
    return "2.3.1"


def _place_secondary_road_junction_signs(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Place 2.3.x on main arms and YieldSign on secondary arms.

    X → SecondaryRoadSign (2.3.1) on every main approach.
    T → SecondaryRoadRightSign (2.3.2) and SecondaryRoadLeftSign (2.3.3) on the
    two main approaches (stem on right / left of that approach).
    Secondary approaches always get YieldSign (2.4), with priority_bench
    outgoing_edge_ids exclusions.
    """
    layout = _get_junction_layout(row, scenes_root)
    if layout is None:
        print("[JunctionSigns] No layout available, falling back to ego-only YieldSign")
        return _place_yield_sign_on_spawn_lane(
            env, distance_before_end=distance_before_end, show_model=show_model
        )

    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return False

    _clear_sign_manager(sign_mgr)

    main_arms = arms_for_road_class(layout, "main")
    secondary_arms = arms_for_road_class(layout, "secondary")
    if not main_arms or not secondary_arms:
        print(
            "[JunctionSigns] Missing main/secondary arms, falling back to ego-only YieldSign"
        )
        return _place_yield_sign_on_spawn_lane(
            env, distance_before_end=distance_before_end, show_model=show_model
        )

    main_lanes = []
    outgoing_edge_ids: set[str] = set()
    for arm in main_arms:
        main_lanes.extend(collect_lanes_for_keys(env, arm.get("lane_keys", [])))
        for out_edge in arm.get("outgoing_to") or arm.get("straight_to") or []:
            outgoing_edge_ids.add(str(out_edge))
    for arm in secondary_arms:
        for out_edge in arm.get("outgoing_to") or arm.get("straight_to") or []:
            outgoing_edge_ids.add(str(out_edge))
    main_approach_edges = {
        str(arm.get("edge_id")) for arm in main_arms if arm.get("edge_id")
    }
    outgoing_edge_ids -= main_approach_edges

    if not main_lanes:
        print(
            "[JunctionSigns] Could not resolve main lanes, falling back to ego-only YieldSign"
        )
        return _place_yield_sign_on_spawn_lane(
            env, distance_before_end=distance_before_end, show_model=show_model
        )

    junction_id = layout.get("junction_id", "")
    placed_by_plate: dict[str, int] = {}
    placed_yield = 0

    for arm in main_arms:
        edge_id = arm.get("edge_id", "")
        lane = resolve_sign_lane_for_edge(env, edge_id, arm.get("lane_keys", []))
        if lane is None:
            print(f"[JunctionSigns] Skipping 2.3 plate, lane not found for edge: {edge_id}")
            continue
        sign_cls = _secondary_road_sign_cls_for_main_arm(layout, edge_id)
        plate = _secondary_road_plate_label(sign_cls)
        placement_long = sign_placement_long(lane, distance_before_end)
        try:
            sign = sign_mgr.add_sign(
                sign_cls,
                lane=lane,
                longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
                lateral_offset=lateral_offset_beside_lane(lane, placement_long),
                show_model=show_model,
                use_random_lane=False,
                intersection_name=junction_id,
            )
            if sign is not None:
                sign.is_priority_sign = False
            placed_by_plate[plate] = placed_by_plate.get(plate, 0) + 1
            print(f"[JunctionSigns] Placed {plate} on main edge {edge_id}")
        except Exception as exc:
            print(f"[JunctionSigns] Failed {plate} on edge {edge_id}: {exc}")

    for arm in secondary_arms:
        edge_id = arm.get("edge_id", "")
        lane = resolve_sign_lane_for_edge(env, edge_id, arm.get("lane_keys", []))
        if lane is None:
            print(f"[JunctionSigns] Skipping yield sign, lane not found for edge: {edge_id}")
            continue
        placement_long = sign_placement_long(lane, distance_before_end)
        try:
            sign = sign_mgr.add_sign(
                YieldSign,
                lane=lane,
                longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
                lateral_offset=lateral_offset_beside_lane(lane, placement_long),
                show_model=show_model,
                use_random_lane=False,
                intersection_name=junction_id,
                main_road_lanes=main_lanes,
                outgoing_edge_ids=sorted(outgoing_edge_ids),
                auto_detect_main_roads=False,
            )
            if sign is not None:
                sign.is_priority_sign = False
            placed_yield += 1
        except Exception as exc:
            print(f"[JunctionSigns] Failed YieldSign on edge {edge_id}: {exc}")

    plate_summary = ", ".join(
        f"{n}×{code}" for code, n in sorted(placed_by_plate.items())
    ) or "0×2.3"
    print(
        f"[JunctionSigns] Placed {plate_summary} on main and {placed_yield} YieldSign(s) "
        f"at junction {junction_id} ({layout.get('shape')}), "
        f"shoulder offset={SIGN_SHOULDER_OFFSET_M}m"
    )
    return bool(placed_by_plate) or placed_yield > 0


def _place_roundabout_sign_on_spawn_lane(
    env, distance_before_end: float = 20.0, show_model: bool = True
) -> bool:
    """Fallback: place a single RoundaboutSign beside the ego approach road."""
    try:
        vehicle = env.agent
        if vehicle is None or vehicle.lane is None:
            return False

        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False

        _clear_sign_manager(sign_mgr)
        lane = vehicle.lane
        placement_long = sign_placement_long(lane, distance_before_end)
        sign = sign_mgr.add_sign(
            RoundaboutSign,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=show_model,
            use_random_lane=False,
            auto_detect_main_roads=False,
        )
        if sign is not None:
            sign.is_priority_sign = False
        return sign is not None
    except Exception as e:
        print(f"[RoundaboutSign] Failed to place sign: {e}")
        return False


def _place_roundabout_signs(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Place RoundaboutSign (4.3) on ego spoke + invisible RoundaboutYieldSign tracker."""
    layout = _get_junction_layout(row, scenes_root)
    if layout is None:
        print("[RoundaboutSigns] No layout available, falling back to ego-only sign")
        return _place_roundabout_sign_on_spawn_lane(
            env, distance_before_end=distance_before_end, show_model=show_model
        )

    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return False

    _clear_sign_manager(sign_mgr)

    main_arms = arms_for_road_class(layout, "main")
    secondary_arms = arms_for_road_class(layout, "secondary")
    if not main_arms:
        print("[RoundaboutSigns] No ring arms in layout, falling back to ego-only sign")
        return _place_roundabout_sign_on_spawn_lane(
            env, distance_before_end=distance_before_end, show_model=show_model
        )

    ring_lanes = []
    for arm in main_arms:
        ring_lanes.extend(collect_lanes_for_keys(env, arm.get("lane_keys", [])))

    if not ring_lanes:
        print("[RoundaboutSigns] Could not resolve ring lanes, falling back to ego-only sign")
        return _place_roundabout_sign_on_spawn_lane(
            env, distance_before_end=distance_before_end, show_model=show_model
        )

    ego_edge = str(row.get("road_id") or "")
    ego_arm = next((a for a in secondary_arms if a.get("edge_id") == ego_edge), None)
    entry_junction = (ego_arm or {}).get("to_node") or row.get("roundabout_entry_junction")
    incoming_edges = entry_conflict_ring_edges(
        layout,
        ego_edge,
        entry_junction_id=entry_junction,
    )
    entry_incoming_lanes = collect_entry_conflict_lanes(
        env,
        layout,
        ego_edge,
        entry_junction_id=entry_junction,
    )
    ego_main_traffic_edges = conflict_aux_ring_edge_ids(
        layout,
        ego_edge,
        entry_junction_id=entry_junction,
    )
    ego_main_traffic_lanes = collect_lanes_for_edge_ids(
        env,
        layout,
        ego_main_traffic_edges,
    )
    all_incoming_edges = all_entry_conflict_ring_edges(layout)
    all_entry_incoming_lanes = collect_all_entry_conflict_lanes(env, layout)
    print(
        f"[RoundaboutSigns] Yield conflict zone: "
        f"ego_entry={len(incoming_edges)} edge(s) "
        f"({', '.join(incoming_edges) or 'none'}), "
        f"ego_main_traffic={len(ego_main_traffic_edges)} edge(s) "
        f"({', '.join(ego_main_traffic_edges) or 'none'}), "
        f"{len(ego_main_traffic_lanes)} lane(s), "
        f"all_entries={len(all_incoming_edges)} edge(s), "
        f"{len(all_entry_incoming_lanes)} lane(s)"
    )

    junction_id = layout.get("junction_id", "")
    placed_plate = 0

    plate_arms = secondary_arms
    if ego_edge:
        if ego_arm is not None:
            plate_arms = [ego_arm]
        elif ego_edge not in layout.get("main_edge_ids", []):
            plate_arms = [{"edge_id": ego_edge, "lane_keys": []}]

    for arm in plate_arms:
        edge_id = arm.get("edge_id", "")
        lane = resolve_sign_lane_for_edge(env, edge_id, arm.get("lane_keys", []))
        if lane is None:
            print(f"[RoundaboutSigns] Skipping plate, lane not found for edge: {edge_id}")
            continue
        placement_long = sign_placement_long(lane, distance_before_end)
        try:
            sign = sign_mgr.add_sign(
                RoundaboutSign,
                lane=lane,
                longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
                lateral_offset=lateral_offset_beside_lane(lane, placement_long),
                show_model=show_model,
                use_random_lane=False,
                intersection_name=junction_id,
            )
            if sign is not None:
                sign.is_priority_sign = False
            placed_plate += 1
        except Exception as exc:
            print(f"[RoundaboutSigns] Failed RoundaboutSign on edge {edge_id}: {exc}")

    ego_lane = getattr(env.agent, "lane", None)
    if ego_lane is not None:
        placement_long = sign_placement_long(ego_lane, distance_before_end)
        try:
            tracker = sign_mgr.add_sign(
                RoundaboutYieldSign,
                lane=ego_lane,
                longitudinal_offset=sign_longitudinal_offset(ego_lane, distance_before_end),
                lateral_offset=lateral_offset_beside_lane(ego_lane, placement_long),
                show_model=False,
                use_random_lane=False,
                intersection_name=junction_id,
                ring_road_lanes=ring_lanes,
                entry_incoming_lanes=ego_main_traffic_lanes or entry_incoming_lanes,
                entry_junction_xy=None,
            )
            if tracker is not None:
                tracker.is_priority_sign = False
        except Exception as exc:
            print(f"[RoundaboutSigns] Failed yield tracker on ego lane: {exc}")

    print(
        f"[RoundaboutSigns] Placed {placed_plate} RoundaboutSign(s) + yield tracker "
        f"at entry {junction_id} (shape={layout.get('shape')}), "
        f"ring_lanes={len(ring_lanes)}, shoulder offset={SIGN_SHOULDER_OFFSET_M}m"
    )
    return placed_plate > 0


def _place_blocked_road_sign(
    env,
    row: dict,
    scenes_root: Path,
    show_model: bool = True,
) -> bool:
    """Place NoTrafficSign (3.2) at the start of the forbidden lane."""
    del scenes_root
    try:
        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False

        sign_road_id = row.get("sign_road_id") or row.get("destination_edge_id")
        if not sign_road_id and row.get("destination_lane_id"):
            sign_road_id = lane_edge_id(str(row["destination_lane_id"]))
        if not sign_road_id:
            print("[NoTrafficSign] Missing sign_road_id / destination edge")
            return False

        lane_keys: list = []
        layout = row.get("junction_layout") or {}
        for arm in layout.get("arms", []) or []:
            if arm.get("edge_id") == sign_road_id:
                lane_keys = list(arm.get("lane_keys") or [])
                break

        lane = resolve_sign_lane_for_edge(env, str(sign_road_id), lane_keys)
        if lane is None:
            print(f"[NoTrafficSign] Lane not found for forbidden edge {sign_road_id}")
            return False

        distance_from_start = float(
            row.get("sign_distance_from_start", DEFAULT_SIGN_DISTANCE_FROM_START)
            or DEFAULT_SIGN_DISTANCE_FROM_START
        )
        raw_cap = row.get("destination_max_along_m")
        try:
            dest_cap = float(
                DEFAULT_DESTINATION_MAX_ALONG_M if raw_cap is None else raw_cap
            )
        except (TypeError, ValueError):
            dest_cap = float(DEFAULT_DESTINATION_MAX_ALONG_M)
        needed = max(distance_from_start + 1.0, dest_cap + 5.0)
        lane_len = float(getattr(lane, "length", 0.0) or 0.0)
        if lane_len <= needed:
            print(
                f"[NoTrafficSign] Forbidden lane too short on {sign_road_id}: "
                f"{lane_len:.2f}m <= needed {needed:.2f}m (sign/dest cap)"
            )
            return False

        placement_long = sign_placement_long_from_start(lane, distance_from_start)
        longitudinal_offset = sign_longitudinal_offset_from_start(lane, distance_from_start)
        lateral = lateral_offset_beside_lane(lane, placement_long)

        _clear_sign_manager(sign_mgr)
        sign = sign_mgr.add_sign(
            NoTrafficSign,
            lane=lane,
            longitudinal_offset=longitudinal_offset,
            lateral_offset=lateral,
            show_model=show_model,
            use_random_lane=False,
        )
        print(
            f"[NoTrafficSign] Placed 3.2 on forbidden edge {sign_road_id} at "
            f"{distance_from_start:.2f}m from lane start "
            f"(long_offset={longitudinal_offset:.2f})"
        )
        return sign is not None
    except Exception as e:
        print(f"[NoTrafficSign] Failed to place: {e}")
        return False


def _place_one_way_sign_on_spawn_lane(
    env,
    pdd_code: str,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Fallback: place OneWayEntrySign beside the ego approach lane."""
    try:
        vehicle = env.agent
        if vehicle is None or vehicle.lane is None:
            return False
        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False
        _clear_sign_manager(sign_mgr)
        lane = vehicle.lane
        sign_cls = resolve_one_way_sign_class(pdd_code)
        placement_long = sign_placement_long(lane, distance_before_end)
        sign = sign_mgr.add_sign(
            sign_cls,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=show_model,
            use_random_lane=False,
        )
        return sign is not None
    except Exception as e:
        print(f"[OneWaySign] Failed to place {pdd_code}: {e}")
        return False


def _place_one_way_signs(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Place OneWayEntrySignR/L on the ego approach; attach wrong-way edges."""
    pdd_code = _resolve_one_way_pdd_code(row)
    layout = _get_junction_layout(row, scenes_root)
    if layout is None:
        print(f"[OneWaySign] No layout; ego-only {pdd_code}")
        return _place_one_way_sign_on_spawn_lane(
            env,
            pdd_code,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )

    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return False
    _clear_sign_manager(sign_mgr)

    ego_edge = row.get("road_id")
    if not ego_edge:
        return _place_one_way_sign_on_spawn_lane(
            env,
            pdd_code,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )

    arm = next(
        (a for a in layout.get("arms", []) if a.get("edge_id") == ego_edge),
        None,
    )
    lane = resolve_sign_lane_for_edge(
        env, str(ego_edge), (arm or {}).get("lane_keys", [])
    )
    if lane is None:
        print(f"[OneWaySign] Lane not found for ego edge {ego_edge}; ego-only fallback")
        return _place_one_way_sign_on_spawn_lane(
            env,
            pdd_code,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )

    sign_cls = resolve_one_way_sign_class(pdd_code)
    placement_long = sign_placement_long(lane, distance_before_end)
    try:
        sign = sign_mgr.add_sign(
            sign_cls,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=show_model,
            use_random_lane=False,
            intersection_name=str(
                row.get("junction_id") or layout.get("junction_id") or ""
            ),
        )
        spec = get_one_way_sign_spec(pdd_code)
        net_path = env.config.get("map_name") or ""
        forbidden_edges = resolve_row_background_excluded_edges(row, net_path)
        if sign is not None and forbidden_edges:
            try:
                sign.one_way_forbidden_edges = frozenset(str(e) for e in forbidden_edges)
            except Exception:
                pass
        print(
            f"[OneWaySign] Placed {pdd_code} ({spec.title}) on {ego_edge}, "
            f"junction={row.get('junction_id') or layout.get('junction_id')}, "
            f"forbidden={spec.forbidden_dir}, allowed={sorted(spec.allowed_dirs)}, "
            f"shoulder offset={SIGN_SHOULDER_OFFSET_M}m"
        )
        return sign is not None
    except Exception as exc:
        print(f"[OneWaySign] Failed {pdd_code} on {ego_edge}: {exc}")
        return False


def _place_direction_sign_on_spawn_lane(
    env,
    pdd_code: str,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Fallback: place LaneAllowedDirectionSign beside the ego approach lane."""
    try:
        vehicle = env.agent
        if vehicle is None or vehicle.lane is None:
            return False
        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False
        _clear_sign_manager(sign_mgr)
        lane = vehicle.lane
        sign_cls = resolve_direction_sign_class(pdd_code)
        placement_long = sign_placement_long(lane, distance_before_end)
        sign = sign_mgr.add_sign(
            sign_cls,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=show_model,
            use_random_lane=False,
        )
        return sign is not None
    except Exception as e:
        print(f"[DirectionSign] Failed to place {pdd_code}: {e}")
        return False


def _place_direction_signs(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Place LaneAllowedDirectionSign 4.1.x on the ego approach."""
    pdd_code = _resolve_direction_pdd_code(row)
    layout = _get_junction_layout(row, scenes_root)
    if layout is None:
        print(f"[DirectionSign] No layout; ego-only {pdd_code}")
        return _place_direction_sign_on_spawn_lane(
            env,
            pdd_code,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )

    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return False
    _clear_sign_manager(sign_mgr)

    ego_edge = row.get("road_id")
    if not ego_edge:
        return _place_direction_sign_on_spawn_lane(
            env,
            pdd_code,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )

    arm = next(
        (a for a in layout.get("arms", []) if a.get("edge_id") == ego_edge),
        None,
    )
    lane = resolve_sign_lane_for_edge(
        env, str(ego_edge), (arm or {}).get("lane_keys", [])
    )
    if lane is None:
        print(f"[DirectionSign] Lane not found for ego edge {ego_edge}; ego-only fallback")
        return _place_direction_sign_on_spawn_lane(
            env,
            pdd_code,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )

    sign_cls = resolve_direction_sign_class(pdd_code)
    placement_long = sign_placement_long(lane, distance_before_end)
    try:
        sign = sign_mgr.add_sign(
            sign_cls,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=show_model,
            use_random_lane=False,
            intersection_name=str(
                row.get("junction_id") or layout.get("junction_id") or ""
            ),
        )
        spec = get_direction_sign_spec(pdd_code)
        print(
            f"[DirectionSign] Placed {pdd_code} ({spec.title}) on {ego_edge}, "
            f"junction={row.get('junction_id') or layout.get('junction_id')}, "
            f"allowed={sorted(spec.allowed_dirs)}, "
            f"shoulder offset={SIGN_SHOULDER_OFFSET_M}m"
        )
        return sign is not None
    except Exception as exc:
        print(f"[DirectionSign] Failed {pdd_code} on {ego_edge}: {exc}")
        return False


def _place_no_turn_sign_on_spawn_lane(
    env,
    pdd_code: str,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Fallback: place NoLeft/RightTurnSign beside the ego approach lane."""
    try:
        vehicle = env.agent
        if vehicle is None or vehicle.lane is None:
            return False
        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False
        _clear_sign_manager(sign_mgr)
        lane = vehicle.lane
        sign_cls = resolve_no_turn_sign_class(pdd_code)
        placement_long = sign_placement_long(lane, distance_before_end)
        sign = sign_mgr.add_sign(
            sign_cls,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=show_model,
            use_random_lane=False,
        )
        return sign is not None
    except Exception as e:
        print(f"[NoTurnSign] Failed to place {pdd_code}: {e}")
        return False


def _place_no_turn_signs(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Place 3.18.x NoTurn sign on the ego approach (same as 4.1 / 5.7)."""
    pdd_code = _resolve_no_turn_pdd_code(row)
    layout = _get_junction_layout(row, scenes_root)
    if layout is None:
        print(f"[NoTurnSign] No layout; ego-only {pdd_code}")
        return _place_no_turn_sign_on_spawn_lane(
            env,
            pdd_code,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )

    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return False
    _clear_sign_manager(sign_mgr)

    ego_edge = row.get("road_id")
    if not ego_edge:
        return _place_no_turn_sign_on_spawn_lane(
            env,
            pdd_code,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )

    arm = next(
        (a for a in layout.get("arms", []) if a.get("edge_id") == ego_edge),
        None,
    )
    lane = resolve_sign_lane_for_edge(
        env, str(ego_edge), (arm or {}).get("lane_keys", [])
    )
    if lane is None:
        print(f"[NoTurnSign] Lane not found for ego edge {ego_edge}; ego-only fallback")
        return _place_no_turn_sign_on_spawn_lane(
            env,
            pdd_code,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )

    sign_cls = resolve_no_turn_sign_class(pdd_code)
    placement_long = sign_placement_long(lane, distance_before_end)
    try:
        sign = sign_mgr.add_sign(
            sign_cls,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=show_model,
            use_random_lane=False,
            intersection_name=str(
                row.get("junction_id") or layout.get("junction_id") or ""
            ),
        )
        spec = get_no_turn_sign_spec(pdd_code)
        print(
            f"[NoTurnSign] Placed {pdd_code} ({spec.title}) on {ego_edge}, "
            f"junction={row.get('junction_id') or layout.get('junction_id')}, "
            f"forbidden={spec.forbidden_dir}, allowed={sorted(spec.allowed_dirs)}, "
            f"shoulder offset={SIGN_SHOULDER_OFFSET_M}m"
        )
        return sign is not None
    except Exception as exc:
        print(f"[NoTurnSign] Failed {pdd_code} on {ego_edge}: {exc}")
        return False


def _place_no_entry_sign(
    env,
    row: dict,
    scenes_root: Path,
    show_model: bool = True,
) -> bool:
    """Place NoEntrySign (3.1) at the start of the short forbidden exit."""
    del scenes_root
    pdd_code = _resolve_no_entry_pdd_code(row)
    try:
        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False

        sign_road_id = (
            row.get("sign_road_id")
            or row.get("baseline_first_exit")
            or row.get("destination_edge_id")
        )
        if not sign_road_id and row.get("destination_lane_id"):
            sign_road_id = lane_edge_id(str(row["destination_lane_id"]))
        if not sign_road_id:
            print("[NoEntrySign] Missing sign_road_id / baseline_first_exit")
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
            row.get("sign_distance_from_start", DEFAULT_SIGN_DISTANCE_FROM_START)
            or DEFAULT_SIGN_DISTANCE_FROM_START
        )
        raw_cap = row.get("destination_max_along_m")
        try:
            dest_cap = float(
                DEFAULT_DESTINATION_MAX_ALONG_M if raw_cap is None else raw_cap
            )
        except (TypeError, ValueError):
            dest_cap = float(DEFAULT_DESTINATION_MAX_ALONG_M)
        needed = max(distance_from_start + 1.0, dest_cap + 5.0)
        lane_len = float(getattr(lane, "length", 0.0) or 0.0)
        if lane_len <= needed:
            print(
                f"[NoEntrySign] Forbidden lane too short on {sign_road_id}: "
                f"{lane_len:.2f}m <= needed {needed:.2f}m (sign/dest cap)"
            )
            return False

        _clear_sign_manager(sign_mgr)
        sign_cls = resolve_no_entry_sign_class(pdd_code)
        placement_long = sign_placement_long_from_start(lane, distance_from_start)
        longitudinal_offset = sign_longitudinal_offset_from_start(lane, distance_from_start)
        lateral = lateral_offset_beside_lane(lane, placement_long)

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
            f"[NoEntrySign] Placed {pdd_code} ({spec.title}) on forbidden edge "
            f"{sign_road_id} at {distance_from_start:.2f}m from lane start "
            f"(long_offset={longitudinal_offset:.2f})"
        )
        return sign is not None
    except Exception as e:
        print(f"[NoEntrySign] Failed to place: {e}")
        return False


def _ego_compliant_stop_before_blocked_road(
    env,
    vehicle,
    *,
    max_dist_before_sign_m: float = DEFAULT_COMPLIANT_STOP_MAX_DIST_M,
    speed_max_mps: float = DEFAULT_COMPLIANT_STOP_SPEED_MPS,
) -> bool:
    """True when ego is nearly stopped just before a 3.2 sign line."""
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
        if not isinstance(sign, NoTrafficSign):
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
        if -0.25 < dist_to_line <= max_dist:
            return True
    return False


def _iter_sumo_graph_lanes(graph) -> list:
    """Yield (lane_key, lane_obj) for EdgeRoadNetwork-style graphs."""
    if not isinstance(graph, dict):
        return []
    out = []
    for key, val in graph.items():
        lane = val
        if isinstance(val, dict):
            lane = val.get("lane", val)
        if hasattr(lane, "position") and hasattr(lane, "length"):
            out.append((str(key), lane))
    return out


def _install_segment_crosswalk_geometry(env, row: dict) -> bool:
    """Build an OSM-style zebra from driving lanes at the injected split.

    netconvert often collapses the SUMO crossing to ~0.1 m. Reconstruct a
    rectangle that spans every carriageway at the junction (walk axis = across
    the road, 4 m along the road) and pass >4 vertices so TopDownRenderer uses
    the same PCA stripe path as real ``crosswalk_sign`` maps.
    """
    if not _row_is_crosswalk(row):
        return False
    current_map = getattr(getattr(env, "engine", None), "current_map", None)
    if current_map is None:
        return False
    graph = getattr(getattr(current_map, "road_network", None), "graph", {}) or {}
    road_network = getattr(current_map, "road_network", None)

    edge_id = str(row.get("road_id") or "")
    lane_num = int(row.get("spawn_lane_num", 0) or 0)
    approach_key = make_lane_key(edge_id, lane_num) if edge_id else ""
    approach_key = clamp_lane_key_to_graph(approach_key, graph) if approach_key else None
    approach = None
    if approach_key and road_network is not None and hasattr(road_network, "get_lane"):
        try:
            approach = road_network.get_lane(approach_key)
        except Exception:
            approach = None
    if approach is None and approach_key:
        approach = graph.get(approach_key)
    if approach is None:
        print(f"[CrosswalkGeom] approach lane missing: {approach_key}")
        return False

    try:
        lane_len = float(getattr(approach, "length", 0.0) or 0.0)
        sample_s = max(0.5, lane_len - 0.5)
        heading = float(approach.heading_theta_at(sample_s))
        center = np.asarray(approach.position(sample_s, 0.0), dtype=np.float64)[:2]
    except Exception as exc:
        print(f"[CrosswalkGeom] Could not sample approach lane: {exc}")
        return False

    forward = np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)
    lateral = np.array([-forward[1], forward[0]], dtype=np.float64)

    lat_hits: list[float] = []
    for key, lane in _iter_sumo_graph_lanes(graph):
        raw = key[5:] if key.startswith("lane_") else key
        if raw.startswith(":"):
            continue
        try:
            length = float(lane.length)
            s_end = max(0.5, length - 0.5)
            s_start = min(0.5, max(0.1, length * 0.05))
            p_end = np.asarray(lane.position(s_end, 0.0), dtype=np.float64)[:2]
            p_start = np.asarray(lane.position(s_start, 0.0), dtype=np.float64)[:2]
            d_end = float(np.linalg.norm(p_end - center))
            d_start = float(np.linalg.norm(p_start - center))
            if d_end <= 14.0:
                s, pt = s_end, p_end
            elif d_start <= 14.0:
                s, pt = s_start, p_start
            else:
                continue
            width = float(lane.width_at(s))
        except Exception:
            continue
        lat0 = float(np.dot(pt - center, lateral))
        lat_hits.append(lat0 - width / 2.0)
        lat_hits.append(lat0 + width / 2.0)

    if lat_hits:
        min_lat = min(lat_hits) - 0.6
        max_lat = max(lat_hits) + 0.6
    else:
        min_lat, max_lat = -5.0, 5.0

    half_thick = max(1.75, float(row.get("crosswalk_width_m") or 4.0) / 2.0)
    corners = [
        center - forward * half_thick + lateral * min_lat,
        center - forward * half_thick + lateral * max_lat,
        center + forward * half_thick + lateral * max_lat,
        center + forward * half_thick + lateral * min_lat,
    ]
    # Midpoints → 8 vertices so the renderer PCA path matches OSM crossings.
    polygon_pts = []
    for i, corner in enumerate(corners):
        nxt = corners[(i + 1) % 4]
        polygon_pts.append(corner)
        polygon_pts.append(0.5 * (corner + nxt))
    polygon = np.asarray(polygon_pts, dtype=np.float64)

    existing = dict(getattr(current_map, "crosswalks", {}) or {})
    cleaned = {}
    for key, feat in existing.items():
        poly = np.asarray((feat or {}).get("polygon", []), dtype=np.float64)
        if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] < 2:
            continue
        span = float(np.linalg.norm(poly.max(axis=0)[:2] - poly.min(axis=0)[:2]))
        if span >= 2.0:
            cleaned[key] = feat
    cleaned["segment_cw_5_19"] = {
        "type": "CROSSWALK",
        "polygon": polygon,
        # Across the road — without this, PCA picks the short (along-road) axis.
        "walk_direction": lateral.tolist(),
    }
    current_map.crosswalks = cleaned

    ped_mgr = getattr(env.engine, "pedestrian_manager", None)
    n_specs = 0
    if ped_mgr is not None and hasattr(ped_mgr, "_collect_crosswalk_specs"):
        ped_mgr._crosswalks = ped_mgr._collect_crosswalk_specs()
        n_specs = len(getattr(ped_mgr, "_crosswalks", {}) or {})
        # ego_proximity waits until ego is close; put the first pedestrian on
        # the zebra now so yield is testable before ego reaches the marking.
        if (
            n_specs > 0
            and str(getattr(ped_mgr, "spawn_mode", "") or "").lower() == "ego_proximity"
            and int(getattr(ped_mgr, "_ego_spawns_scheduled", 0) or 0) == 0
            and hasattr(ped_mgr, "_schedule_track")
        ):
            preferred = [k for k in ped_mgr._crosswalks if k == "segment_cw_5_19"]
            rest = [k for k in ped_mgr._crosswalks if k != "segment_cw_5_19"]
            for cw_id in preferred + rest:
                if not ped_mgr._schedule_track(cw_id, on_crosswalk=True, immediate=True):
                    continue
                ped_mgr._ego_spawns_scheduled = 1
                ped_mgr._ego_trigger_crosswalk_id = cw_id
                if hasattr(ped_mgr, "_spawn_due_tracks"):
                    ped_mgr._spawn_due_tracks()
                print(f"[CrosswalkGeom] primed pedestrian on {cw_id}")
                break
    span_m = float(max_lat - min_lat)
    print(
        f"[CrosswalkGeom] zebra span={span_m:.1f}m thick={half_thick * 2:.1f}m "
        f"ped_specs={n_specs}"
    )
    return True


def _place_crosswalk_sign_on_spawn_lane(
    env,
    row: dict,
    distance_before_end: float = 15.0,
    show_model: bool = True,
) -> bool:
    """Place PedestrianCrossingSign (5.19 icon) beside the ego approach lane."""
    try:
        vehicle = env.agent
        if vehicle is None or vehicle.lane is None:
            return False

        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False

        _clear_sign_manager(sign_mgr)
        _ensure_pedestrian_yield_rule(env)

        edge_id = row.get("road_id") or row.get("depart_edge_id")
        lane = None
        if edge_id:
            lane = resolve_sign_lane_for_edge(env, str(edge_id), [])
        if lane is None:
            lane = vehicle.lane

        placement_long = sign_placement_long(lane, distance_before_end)
        sign = sign_mgr.add_sign(
            PedestrianCrossingSign,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=show_model,
            use_random_lane=False,
        )
        if sign is not None:
            sign.is_priority_sign = False
            print(
                f"[PedestrianCrossingSign] Placed 5.19 on edge "
                f"{getattr(lane, 'index', edge_id)} "
                f"({distance_before_end:.1f}m before end), "
                f"yield_rules={sum(type(r).__name__ == 'PedestrianYieldRule' for r in sign_mgr.rules)}"
            )
        return sign is not None
    except Exception as e:
        print(f"[PedestrianCrossingSign] Failed to place sign: {e}")
        return False


def _place_detour_signs(
    env,
    row: dict,
    show_model: bool = True,
) -> bool:
    """Place a DetourSign (4.2.x) on the obstacle lane at sign_s from manifest meta.

    The sign is placed on the lane identified by ``sign_lane_index`` at the
    longitudinal position ``sign_s`` stored in the scene's meta.json.
    Violation detection uses a zone: [sign_s - 30m, sign_s + 15m].
    """
    from core.layout.junction_sign_placement import resolve_layout_lane

    try:
        vehicle = env.agent
        if vehicle is None or vehicle.lane is None:
            return False

        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False

        _clear_sign_manager(sign_mgr)

        pdd_code = str(row.get("pdd_code") or row.get("detour_code") or "4.2.1")
        sign_cls_map = {
            "4.2.1": DetourRightSign,
            "4.2.2": DetourLeftSign,
            "4.2.3": DetourEitherSign,
        }
        sign_cls = sign_cls_map.get(pdd_code, DetourRightSign)

        road_id = str(row.get("road_id") or "")
        sign_lane_index = int(row.get("sign_lane_index", 0))
        sign_s = float(row.get("sign_s", 60.0))

        lane = None
        if road_id:
            lane_key = f"{road_id}_{sign_lane_index}"
            lane = resolve_layout_lane(env, lane_key)

        if lane is None:
            lane = vehicle.lane

        placement_long = max(0.0, min(sign_s, lane.length - 1.0))
        longitudinal_offset = placement_long - lane.length

        sign = sign_mgr.add_sign(
            sign_cls,
            lane=lane,
            longitudinal_offset=longitudinal_offset,
            lateral_offset=0,
            show_model=show_model,
            use_random_lane=False,
        )
        if sign is not None:
            sign.is_priority_sign = False
            print(
                f"[DetourSign] Placed {pdd_code} on lane "
                f"{getattr(lane, 'index', lane_key)} "
                f"at s={sign_s:.1f}m (zone [{sign.zone_start:.1f}, {sign.zone_end:.1f}])"
            )
        return sign is not None
    except Exception as e:
        print(f"[DetourSign] Failed to place sign: {e}")
        import traceback
        traceback.print_exc()
        return False


def _place_speed_signs(
    env,
    row: dict,
    show_model: bool = True,
) -> bool:
    """Place start (and paired end) speed signs at sign_s / s_end from the manifest."""
    from core.layout.junction_sign_placement import resolve_layout_lane

    try:
        vehicle = env.agent
        if vehicle is None or vehicle.lane is None:
            return False
        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False
        _clear_sign_manager(sign_mgr)

        pdd_code = str(row.get("pdd_code") or row.get("sign_code") or "3.24")
        start_cls_map = {
            "3.24": SpeedLimitSign,
            "4.6": MinimumSpeedLimitSign,
            "5.21": ResidentialZoneSign,
            "5.31": ZoneSpeedLimitSign,
        }
        end_cls_map = {
            "3.25": EndOfSpeedLimitSign,
            "5.22": EndOfResidentialZoneSign,
            "5.32": EndOfZoneSpeedLimitSign,
        }
        start_cls = start_cls_map.get(pdd_code, SpeedLimitSign)
        v_target = float(row.get("v_target_kmh") or 0.0)
        road_id = str(row.get("road_id") or "")
        lane_num = int(row.get("sign_lane_index", row.get("spawn_lane_num", 0)) or 0)
        sign_s = float(row.get("sign_s", 60.0))

        lane = None
        if road_id:
            lane = resolve_layout_lane(env, f"{road_id}_{lane_num}")
        if lane is None:
            lane = vehicle.lane

        placement_long = max(0.1, min(sign_s, float(lane.length) - 1.0))
        start_kwargs = dict(
            lane=lane,
            longitudinal_offset=placement_long,
            lateral_offset=0,
            show_model=show_model,
            use_random_lane=False,
        )
        if start_cls is SpeedLimitSign or start_cls is ZoneSpeedLimitSign:
            if v_target > 0:
                start_kwargs["speed_limit_override"] = v_target
        elif start_cls is MinimumSpeedLimitSign:
            if v_target > 0:
                start_kwargs["min_speed_override"] = v_target

        start_sign = sign_mgr.add_sign(start_cls, **start_kwargs)
        if start_sign is None:
            print(f"[SpeedSign] Failed to place start {pdd_code}")
            return False
        start_sign.is_priority_sign = False

        end_code = str(row.get("sign_type_end") or "")
        s_end = row.get("s_end")
        if end_code and s_end is not None:
            end_cls = end_cls_map.get(end_code)
            if end_cls is not None:
                end_long = max(placement_long + 1.0, min(float(s_end), float(lane.length) - 0.5))
                end_kwargs = dict(
                    lane=lane,
                    longitudinal_offset=end_long,
                    lateral_offset=0,
                    show_model=show_model,
                    use_random_lane=False,
                )
                if end_cls is EndOfSpeedLimitSign or end_cls is EndOfZoneSpeedLimitSign:
                    if v_target > 0:
                        end_kwargs["speed_limit"] = v_target
                end_sign = sign_mgr.add_sign(end_cls, **end_kwargs)
                if end_sign is not None:
                    end_sign.is_priority_sign = False
            try:
                sign_mgr.build_zones()
            except Exception as exc:
                print(f"[SpeedSign] build_zones failed: {exc}")

        print(
            f"[SpeedSign] Placed {pdd_code}@{placement_long:.1f}m "
            f"v_target={v_target:.0f} end={end_code or '-'} "
            f"s_end={float(s_end) if s_end is not None else float('nan'):.1f}"
        )
        return True
    except Exception as e:
        print(f"[SpeedSign] Failed to place sign: {e}")
        import traceback
        traceback.print_exc()
        return False


def _place_junction_priority_signs(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Dispatch sign placement by row pdd_code / sign_type."""
    if _row_is_detour(row):
        return _place_detour_signs(
            env,
            row,
            show_model=show_model,
        )
    if _row_is_speed(row):
        return _place_speed_signs(
            env,
            row,
            show_model=show_model,
        )
    if _row_is_crosswalk(row):
        return _place_crosswalk_sign_on_spawn_lane(
            env,
            row,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )
    if _row_is_no_entry(row):
        return _place_no_entry_sign(
            env,
            row,
            scenes_root=scenes_root,
            show_model=show_model,
        )
    if _row_is_no_turn(row):
        return _place_no_turn_signs(
            env,
            row,
            scenes_root=scenes_root,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )
    if _row_is_direction(row):
        return _place_direction_signs(
            env,
            row,
            scenes_root=scenes_root,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )
    if _row_is_one_way(row):
        return _place_one_way_signs(
            env,
            row,
            scenes_root=scenes_root,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )
    if _row_is_blocked_road(row):
        return _place_blocked_road_sign(
            env,
            row,
            scenes_root=scenes_root,
            show_model=show_model,
        )
    if _row_is_roundabout(row):
        return _place_roundabout_signs(
            env,
            row,
            scenes_root=scenes_root,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )
    if _row_is_stop(row):
        return _place_stop_junction_signs(
            env,
            row,
            scenes_root=scenes_root,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )
    if _row_is_secondary_road(row):
        return _place_secondary_road_junction_signs(
            env,
            row,
            scenes_root=scenes_root,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )
    if _row_is_yield(row):
        return _place_yield_junction_signs(
            env,
            row,
            scenes_root=scenes_root,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )
    return _place_equal_priority_main_signs(
        env,
        row,
        scenes_root=scenes_root,
        distance_before_end=distance_before_end,
        show_model=show_model,
    )


def _topdown_gif_film_and_scaling(
    env,
    *,
    screen_size: tuple[int, int] = (800, 800),
    window_m: float = 80.0,
) -> tuple[tuple[int, int], float]:
    """Choose film_size + MetaDrive scaling for a fixed visible window (meters).

    Dual-path crops are often >1 km; a fixed ``film_size=(4800,4800)`` would
    clamp zoom. Grow film with map bbox so ``window_m`` is honored without
    editing MetaDrive.
    """
    screen = int(max(screen_size[0], screen_size[1]))
    win = float(window_m) if window_m and float(window_m) > 0.0 else 80.0
    scaling_req = float(screen) / win

    max_len = 400.0
    try:
        b_box = env.engine.current_map.road_network.get_bounding_box()
        max_len = max(
            float(b_box[1] - b_box[0]),
            float(b_box[3] - b_box[2]),
            1.0,
        )
    except Exception:
        pass

    # MetaDrive: scaling = min(requested, film/max_len - 0.1)
    need = int((scaling_req + 0.1) * max_len + 64)
    film = max(4800, need)
    film = min(film, 24000)  # soft cap for RAM
    return (film, film), float(scaling_req)


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
    gif_window_m: float = 80.0,
    hide_signs: bool = False,
    draw_path_conflict: bool = False,
    auxiliary_agent: bool = False,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    aux_policy: str = "idm",
    aux_spawn_velocity_ms: float = DEFAULT_SPAWN_VELOCITY_MS,
    aux_release_when_ego_within_m: float = DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END,
    aux_convoy_size: int = DEFAULT_CONVOY_SIZE,
    aux_convoy_gap_m: float = DEFAULT_CONVOY_GAP_M,
    aux_lanes_occupied: int = DEFAULT_AUX_LANES_OCCUPIED_MAX,
    record_episode: bool = False,
) -> dict:
    seed = int(row.get("seed") or row.get("deterministic_seed") or 0)
    set_path_conflict_overlay_enabled(bool(draw_path_conflict) and save_gif is not None)
    np.random.seed(numpy_legacy_seed(seed))
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

    _ensure_secondary_ego_spawn(row, scenes_root)
    env = _build_sumo_env(row, scenes_root=scenes_root, max_steps=max_steps)

    raw_env = env
    env = _wrap_for_policy(env, policy_type)

    if record_episode:
        # Shared patch: tolerate post-reset sign/aux spawn before first step.
        from bench.record_manager_patch import patch_record_manager_once
        patch_record_manager_once()
        # RecordManager reads this off global_config on reset.
        try:
            raw_env.config["record_episode"] = True
        except Exception:
            pass
        try:
            env.config["record_episode"] = True
        except Exception:
            pass

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
    elif policy_type in ("carl", "carl_rule") or policy_type in PLANT2_POLICIES:
        policy_cls = models.get("policy_cls")
        if policy_cls is None:
            raise RuntimeError(f"policy_cls for --policy {policy_type} not loaded; "
                               "check _load_policy_models")

    try:
        if record_episode:
            from bench.record_manager_patch import patch_record_manager_once
            patch_record_manager_once()

        env_seed = (int(row.get("sign_id", 0)) + int(row.get("var_idx", 0))) % 100000
        obs, info = env.reset(seed=env_seed)
        base_env = _unwrap_base_env(env)
        try:
            if hasattr(base_env, "engine") and hasattr(base_env.engine, "np_random"):
                base_env.engine.np_random = np.random.RandomState(seed)
        except Exception:
            pass

        # Manifest spawn lane + distance before intersection
        if _row_uses_dual_path_nav(row):
            row = _resolve_dual_path_row_for_policy(row, policy_type)
        _apply_manifest_ego_spawn_lane(base_env, row)
        spawn_distance = float(row.get("spawn_distance_before_end", 0) or 0)
        if spawn_distance > 0:
            _reposition_ego_before_lane_end(base_env, spawn_distance)
        if _row_is_speed(row):
            _apply_manifest_ego_spawn_velocity(base_env, row)
        _apply_manifest_ego_destination(base_env, row)
        if _row_is_detour(row) or _row_is_speed(row):
            _apply_destination_along_cap(base_env, row)
        _install_segment_crosswalk_geometry(base_env, row)

        # Dual-path dest is already policy-resolved (truncated finish).
        # Plain baselines: MetaDrive unrestricted set_route — short violating
        # path is a property of the map, we do not pin or block anything.
        # Rule-compliant: rebuild with the forbidden branch blocked.
        if _row_uses_dual_path_nav(row):
            if policy_type in _ONE_WAY_COMPLIANT_NAV_POLICIES:
                install_one_way_compliant_nav_route(base_env, row)
            else:
                nav = getattr(base_env.vehicle, "navigation", None)
                n_ck = len(getattr(nav, "checkpoints", None) or [])
                print(
                    f"[DualPathNav] kept MetaDrive default route "
                    f"({n_ck} checkpoints) for {policy_type}"
                )
            _apply_destination_along_cap(base_env, row)

        # Validate route: check that destination is different from spawn.
        # Detour finishes on the same obstacle edge (along-cap), so skip this.
        nav = getattr(base_env.vehicle, "navigation", None)
        if nav is not None and not _row_is_detour(row) and not _row_is_speed(row):
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

        # Place MainRoadSign (+ RH yield tracker) under RecordManager guard.
        sign_distance = float(row.get("sign_distance_before_end", 20.0))
        _rm = getattr(base_env.engine, "record_manager", None) if record_episode else None
        _rm_original_add_spawn = None
        _signs_pre = set(base_env.engine._spawned_objects.keys())
        if _rm is not None:
            _rm_original_add_spawn = _rm.add_spawn_info
            _rm.add_spawn_info = lambda *a, **kw: None
        try:
            _place_junction_priority_signs(
                base_env,
                row,
                scenes_root=scenes_root,
                distance_before_end=sign_distance,
                show_model=not hide_signs,
            )
            if (
                not _row_is_main_secondary(row)
                and not _row_is_blocked_road(row)
                and not _row_uses_dual_path_nav(row)
                and not _row_is_crosswalk(row)
                and not _row_is_detour(row)
                and not _row_is_speed(row)
            ):
                _place_right_hand_yield_tracker(
                    base_env,
                    row,
                    scenes_root=scenes_root,
                    distance_before_end=sign_distance,
                )
        finally:
            if record_episode:
                _signs_post = set(base_env.engine._spawned_objects.keys())
                for _sid in _signs_post - _signs_pre:
                    obj = base_env.engine._spawned_objects.get(_sid)
                    body = getattr(obj, "_body", None) if obj is not None else None
                    if obj is not None and body is None:
                        base_env.engine._spawned_objects.pop(_sid, None)
                if _rm is not None and _rm_original_add_spawn is not None:
                    _rm.add_spawn_info = _rm_original_add_spawn

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
            if hasattr(policy_obj, "STOP_WAIT_STEPS"):
                policy_obj.STOP_WAIT_STEPS = int(
                    row.get("stop_wait_steps", DEFAULT_STOP_WAIT_STEPS)
                )

        # Add auxiliary agents on every incoming lane (except ego's road)
        # aux_agent_mgr = None
        # if auxiliary_agent:
        #     ego_lane_index = getattr(base_env.vehicle.lane, "index", "")
        #     aux_spawn_lanes = [
        #         lane["lane_name"]
        #         for lane in incoming_lanes
        #         # if lane["edge_id"] not in ego_lane_index
        #     ]
        #     if aux_spawn_lanes:
        #         aux_agent_mgr = add_auxiliary_agents(
        #             base_env,
        #             spawn_lane_indices=aux_spawn_lanes,
        #             distance_from_intersection=aux_distance_from_intersection,
        #         )
        #         print(f"[AuxAgent] Spawned on {len(aux_spawn_lanes)} incoming lane(s)")
        #     else:
        #         print("[AuxAgent] No incoming lanes available for auxiliary agents")

        aux_agent_mgr = None
        aux_spawn_lanes: list[str] = []
        if auxiliary_agent and not _row_is_blocked_road(row) and not _row_uses_dual_path_nav(row):
            aux_distance_from_intersection = float(
                row.get("aux_distance_from_intersection", aux_distance_from_intersection)
            )
            aux_spawn_velocity_ms = float(
                row.get("aux_spawn_velocity_ms", aux_spawn_velocity_ms)
            )
            aux_convoy_size = int(row.get("aux_convoy_size", aux_convoy_size))
            aux_convoy_gap_m = float(row.get("aux_convoy_gap_m", aux_convoy_gap_m))
            aux_lanes_occupied = int(row.get("aux_lanes_occupied", aux_lanes_occupied))
            ego_lane_index = (
                make_lane_key(str(row.get("road_id")), int(row.get("spawn_lane_num", 0) or 0))
                if row.get("road_id")
                else str(getattr(base_env.vehicle.lane, "index", ""))
            )

            aux_spawn_lanes, aux_destination_lanes, alternate_spawn_dest_map, aux_spawn_longs, ring_circulate = (
                resolve_aux_spawn_plan(
                    row,
                    ego_lane_index=str(ego_lane_index),
                    incoming_lanes=incoming_lanes,
                    aux_lanes_occupied=aux_lanes_occupied,
                    aux_distance_from_intersection=aux_distance_from_intersection,
                    scenes_root=scenes_root,
                )
            )
            aux_destination_lanes = [
                dest or None for dest in aux_destination_lanes
            ]

            # Keep release distance >= ego spawn offset so aux is not held while a
            # yielding ego freezes outside the release radius.
            ego_spawn_before_end = float(row.get("spawn_distance_before_end", 0) or 0)
            release_before_end = float(aux_release_when_ego_within_m)
            if release_before_end > 0 and ego_spawn_before_end > 0:
                release_before_end = max(release_before_end, ego_spawn_before_end)

            if aux_spawn_lanes:
                aux_agent_mgr = add_auxiliary_agents(
                    base_env,
                    spawn_lane_indices=aux_spawn_lanes,
                    outgoing_lanes=outgoing_lanes,
                    distance_from_intersection=aux_distance_from_intersection,
                    policy=aux_policy,
                    spawn_velocity_ms=aux_spawn_velocity_ms,
                    destination_lanes=aux_destination_lanes,
                    ego_vehicle=base_env.vehicle,
                    ego_spawn_lane_index=ego_lane_index,
                    ego_release_distance_before_end=release_before_end,
                    convoy_size=aux_convoy_size,
                    convoy_gap_m=aux_convoy_gap_m,
                    alternate_spawn_dest_map=alternate_spawn_dest_map,
                    spawn_longitudinal_by_lane=aux_spawn_longs,
                    ring_circulate_by_lane=ring_circulate,
                    junction_layout=row.get("junction_layout"),
                )
                if aux_agent_mgr is not None:
                    print(
                        f"[AuxAgent] lanes={len(aux_spawn_lanes)}, "
                        f"convoy_size={aux_convoy_size}, gap={aux_convoy_gap_m}m, "
                        f"spawned={aux_agent_mgr.get_status().get('count', 0)}"
                    )

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

        is_blocked_road_row = _row_is_blocked_road(row)
        compliant_stop_steps = 0
        compliant_stop_success = False
        stop_success_s = float(
            row.get(
                "compliant_stop_success_seconds",
                DEFAULT_COMPLIANT_STOP_SUCCESS_SECONDS,
            )
            or DEFAULT_COMPLIANT_STOP_SUCCESS_SECONDS
        )
        stop_success_steps = max(1, int(round(stop_success_s / dt)))
        stop_max_dist_m = float(
            row.get("compliant_stop_max_dist_m", DEFAULT_COMPLIANT_STOP_MAX_DIST_M)
            or DEFAULT_COMPLIANT_STOP_MAX_DIST_M
        )
        stop_speed_max = float(
            row.get("compliant_stop_speed_mps", DEFAULT_COMPLIANT_STOP_SPEED_MPS)
            or DEFAULT_COMPLIANT_STOP_SPEED_MPS
        )

        # Re-apply after spawn/signs/policy — nav may have been rebuilt.
        _apply_destination_along_cap(base_env, row)

        episode_horizon = _manifest_horizon(row, max_steps)
        for step in range(episode_horizon):
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
                for rule in getattr(sign_mgr, "rules", []):
                    if type(rule).__name__ != "PedestrianYieldRule":
                        continue
                    try:
                        ped_status = rule.get_status(vehicle)
                    except Exception:
                        continue
                    if (
                        ped_status.get("in_yield_zone")
                        or ped_status.get("in_crosswalk")
                        or ped_status.get("in_no_stop_zone")
                    ):
                        step_in_any_zone = True
                        in_zone_by_class_step["PedestrianYieldRule"] = (
                            in_zone_by_class_step.get("PedestrianYieldRule", 0) + 1
                        )
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

            # Finish: compliant stop (3.2 success) and/or capped dest (3.2 / 4.3).
            # Violation on 3.2 is recorded by NoTrafficSign when ego passes the sign.
            dest_cap_m = 0.0
            capped_arrive = False
            if is_blocked_road_row:
                if sign_violations == 0 and _ego_compliant_stop_before_blocked_road(
                    base_env,
                    vehicle,
                    max_dist_before_sign_m=stop_max_dist_m,
                    speed_max_mps=stop_speed_max,
                ):
                    compliant_stop_steps += 1
                    if compliant_stop_steps >= stop_success_steps:
                        capped_arrive = True
                        compliant_stop_success = True
                        print(
                            f"[NoTrafficSign] Compliant stop for {stop_success_s:.1f}s "
                            f"before sign → arrive_dest (step={steps})"
                        )
                else:
                    compliant_stop_steps = 0

            if (
                is_blocked_road_row
                or _row_is_roundabout(row)
                or _row_uses_dual_path_nav(row)
                or _row_is_crosswalk(row)
                or _row_is_detour(row)
                or _row_is_speed(row)
            ):
                raw_cap = row.get("destination_max_along_m")
                if raw_cap is None:
                    dest_cap_m = float(DEFAULT_DESTINATION_MAX_ALONG_M)
                else:
                    dest_cap_m = float(raw_cap or 0.0)
                stored = getattr(vehicle, "_priority_bench_dest_along_m", None)
                if stored is not None:
                    try:
                        dest_cap_m = float(stored)
                    except (TypeError, ValueError):
                        pass
            capped_arrive = capped_arrive or _ego_reached_capped_destination(
                vehicle,
                max_along_m=dest_cap_m,
                allow_same_lane=_row_is_detour(row) or _row_is_speed(row),
            )
            natural_done = bool(terminated or truncated)

            text_dict: dict = {}
            if save_gif:
                aux_vehicles = []
                if aux_agent_mgr is not None:
                    try:
                        aux_vehicles = list(aux_agent_mgr.auxiliary_vehicles)
                    except Exception:
                        aux_vehicles = []
                text_dict = {
                    "Step": step,
                    "Speed": f"{vehicle.speed_km_h:.2f} km/h" if vehicle else "n/a",
                    "Violations": sign_violations + crosswalk_violations,
                    "is_aux_in_main_zone": _is_aux_in_main_zone(
                        sign_mgr, aux_vehicles, ego_vehicle=vehicle
                    ),
                    "is_ego_in_yield_zone": _is_ego_in_yield_zone(sign_mgr, vehicle),
                }
                if draw_path_conflict or is_path_conflict_overlay_enabled():
                    text_dict["paths"] = "cyan=ego magenta=auxX yellow=X amber=zone"

            # Render before breaking so arrive/terminate frames are in the GIF.
            if save_gif:
                try:
                    screen_size = (800, 800)
                    film_size, scaling = _topdown_gif_film_and_scaling(
                        base_env,
                        screen_size=screen_size,
                        window_m=gif_window_m,
                    )
                    base_env.render(
                        mode="top_down",
                        film_size=film_size,
                        scaling=float(scaling),
                        screen_size=screen_size,
                        semantic_map=True,
                        semantic_broken_line=True,
                        draw_target_vehicle_trajectory=True,
                        target_agent_heading_up=True,
                        screen_record=True, window=False,
                        text=text_dict,
                    )
                except Exception:
                    pass

            if capped_arrive:
                reached_dest = True
                info["arrive_dest"] = True
                last_info = info
                break

            if natural_done:
                reached_dest = bool(info.get("arrive_dest", False))
                out_of_road = bool(info.get("out_of_road", False))
                crashed = bool(info.get("crash", False) or out_of_road)
                break

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

        pkl_path_str: str | None = None
        dump_error: str | None = None

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
                output_pkl = out_replay / "replay.pkl"

                if record_episode:
                    scenario_desc = None
                    try:
                        from metadrive.scenario.utils import (
                            convert_recorded_scenario_exported,
                        )
                        raw_frames = base_env.engine.record_manager.episode_info
                        scenario_desc = convert_recorded_scenario_exported(
                            raw_frames, to_dict=True
                        )
                    except Exception:
                        scenario_desc = None
                    if scenario_desc is not None:
                        with open(output_pkl, "wb") as f:
                            pickle.dump(scenario_desc, f)
                        pkl_path_str = str(output_pkl)
                    else:
                        try:
                            base_env.engine.dump_episode(str(output_pkl))
                            if output_pkl.is_file() and output_pkl.stat().st_size > 0:
                                pkl_path_str = str(output_pkl)
                            else:
                                dump_error = "dump_episode wrote empty file"
                        except Exception as exc:
                            dump_error = f"dump_episode: {type(exc).__name__}: {exc}"
                            pkl_path_str = None

                sidecar_metrics = {
                    "arrived_dest": bool(reached_dest),
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
                    "success": (
                        bool(compliant_stop_success and not crashed_flag_raw and not out_of_road)
                        if is_blocked_road_row
                        else bool(reached_dest and not crashed_flag_raw and not out_of_road)
                    ),
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
                        "horizon": episode_horizon,
                        "seed": seed,
                    },
                    "signs": sign_info_snapshot,
                    "expert_actions": expert_actions,
                    "smoothness_step_vars": smoothness_step_vars,
                    "metrics": sidecar_metrics,
                    "ego_idm_params": (sampled_ego_params if sampled_ego_params is not None
                                        else "DEFAULT_EGO_PARAMS"),
                    "pkl_path": pkl_path_str,
                    "sidecar_path": str(sidecar_path),
                    "valid": True,
                }
                if dump_error:
                    sidecar["dump_error"] = dump_error
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
            "success": (
                bool(compliant_stop_success)
                if is_blocked_road_row
                else bool(reached_dest and not crashed)
            ),
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
            "pkl_path": pkl_path_str,
            "dump_error": dump_error,
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
    parser = argparse.ArgumentParser(description="Run policies on real SUMO maps (main sign / 2.1 benchmark)")
    parser.add_argument("--policy", required=True,
                        choices=["idm", "modified_idm", "comprehensive_rule_expert",
                                 "rule_compliant", "ppo_lidar",
                                 "carl", "carl_rule",
                                 "plant2", "plant2_rule", "plant2_ft"])
    parser.add_argument("--model-path", type=str, default=None,
                        help="Checkpoint for carl/plant2 (defaults under pdd-bench/checkpoints/; "
                             "plant2_ft → checkpoints/plant2_finetuned)")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--preset", type=str, default="full", choices=["full", "full_last"])
    parser.add_argument("--benchmark-output", type=str, default="benchmark_output",
                        help="Base dir that contains <preset>/")
    parser.add_argument("--scenes-root", type=str, default=str(BENCH_DIR / "scenes"))
    parser.add_argument("--sign-type", type=str, default=None,
                        help="Single sign code, e.g. 2.1")
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
    parser.add_argument(
        "--gif-window-m",
        type=float,
        default=80.0,
        help="Visible top-down GIF window in meters (same across signs; "
             "film_size auto-grows so MetaDrive does not clamp zoom).",
    )
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Write episodes/summary/gifs here directly "
                             "(skips <benchmark-output>/<preset>/policy_eval/<run-name>)")
    parser.add_argument("--plant2-action-mode", type=str, default="pid",
                        choices=["pid", "wps_pure_pursuit"],
                        help="How PlanT2 converts pred_plan -> action")
    parser.add_argument("--hide-signs", action="store_true",
                        help="Hide traffic sign visual models (signs still affect behavior)")
    parser.add_argument(
        "--draw-path-conflict",
        action="store_true",
        help="Overlay ego/aux route polylines + conflict point on top-down GIFs",
    )

    # Auxiliary agent options
    parser.add_argument("--auxiliary-agent", action="store_true", default=True,
                        help="Spawn an auxiliary agent on an incoming lane near intersection")
    parser.add_argument(
        "--aux-distance-from-intersection",
        type=float,
        default=DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
        help=f"Fallback aux spawn distance from intersection (meters); "
             f"manifest row aux_distance_from_intersection takes precedence "
             f"(default: {DEFAULT_AUX_DISTANCE_FROM_INTERSECTION})",
    )
    parser.add_argument("--aux-policy", type=str, default="idm", choices=["idm", "stationary"],
                        help="Auxiliary agent behavior: idm drives to outgoing lane, stationary stays put")
    parser.add_argument(
        "--aux-spawn-velocity-ms",
        type=float,
        default=DEFAULT_SPAWN_VELOCITY_MS,
        help=f"Aux IDM cruise/release speed in m/s (default: {DEFAULT_SPAWN_VELOCITY_MS})",
    )
    parser.add_argument(
        "--aux-release-when-ego-within-m",
        type=float,
        default=DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END,
        help=(
            "Release gated IDM aux when ego is within this distance of spawn lane end (m); "
            f"0 = immediate (default: {DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END}). "
            "Clamped up to spawn_distance_before_end so aux is not held while ego yields."
        ),
    )
    parser.add_argument(
        "--aux-convoy-size",
        type=int,
        default=DEFAULT_CONVOY_SIZE,
        help=f"Max convoy size at manifest generation; spawns rows for sizes 1..N "
             f"(default: {DEFAULT_CONVOY_SIZE}). Per-row size is stored as aux_convoy_size.",
    )
    parser.add_argument(
        "--aux-convoy-gap-m",
        type=float,
        default=DEFAULT_CONVOY_GAP_M,
        help=f"Longitudinal spacing between convoy vehicles in meters (default: {DEFAULT_CONVOY_GAP_M})",
    )
    parser.add_argument(
        "--aux-lanes-occupied",
        type=int,
        default=DEFAULT_AUX_LANES_OCCUPIED_MAX,
        help=f"Fallback max main-road lanes to occupy when manifest row omits aux_lanes_occupied "
             f"(default: {DEFAULT_AUX_LANES_OCCUPIED_MAX})",
    )
    parser.add_argument(
        "--stop-wait-steps",
        type=int,
        default=None,
        help=f"Override expert stop-line dwell in sim steps (default from manifest / "
             f"{DEFAULT_STOP_WAIT_STEPS} ≈ 1.5 s at 0.1 s/step)",
    )

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
    if args.auxiliary_agent:
        print(f"Auxiliary agent: ENABLED ({args.aux_policy}, near intersection)")
        print(f"  - Distance from intersection: {args.aux_distance_from_intersection}m")
        if args.aux_policy == "idm":
            print(f"  - Release when ego within: {args.aux_release_when_ego_within_m}m of spawn lane end")
            print(f"  - Speed after release: {args.aux_spawn_velocity_ms} m/s")
            print(f"  - Convoy size: from manifest row aux_convoy_size (CLI default {args.aux_convoy_size})")
            print(f"  - Lanes occupied: from manifest row aux_lanes_occupied (CLI default {args.aux_lanes_occupied})")
            print(f"  - Convoy gap: {args.aux_convoy_gap_m}m")
            print(f"  - Route: incoming lane -> reachable outgoing lane")

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
    model_path = resolve_model_path(args.policy, args.model_path)
    if args.policy in NN_NEED_CHECKPOINT and not model_path:
        default = DEFAULT_MODEL_PATHS.get(args.policy)
        raise ValueError(
            f"--model-path is required for --policy {args.policy}"
            + (f" (default missing: {default})" if default else "")
        )
    models = _load_policy_models(
        args.policy, model_path, plant2_action_mode=args.plant2_action_mode,
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
            if args.stop_wait_steps is not None:
                row["stop_wait_steps"] = int(args.stop_wait_steps)
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
                gif_window_m=args.gif_window_m,
                hide_signs=args.hide_signs,
                draw_path_conflict=bool(args.draw_path_conflict),
                auxiliary_agent=args.auxiliary_agent,
                aux_distance_from_intersection=args.aux_distance_from_intersection,
                aux_policy=args.aux_policy,
                aux_spawn_velocity_ms=args.aux_spawn_velocity_ms,
                aux_release_when_ego_within_m=args.aux_release_when_ego_within_m,
                aux_convoy_size=args.aux_convoy_size,
                aux_convoy_gap_m=args.aux_convoy_gap_m,
                aux_lanes_occupied=args.aux_lanes_occupied,
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
