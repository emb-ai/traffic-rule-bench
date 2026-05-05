from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def _find_pdd_bench_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for parent in (current, *current.parents):
        if (parent / "envs").is_dir() and (parent / "traffic_signs").is_dir():
            return parent
    raise RuntimeError("Could not locate pdd-bench root")


SCRIPT_PATH = Path(__file__).resolve()
BENCH_DIR = SCRIPT_PATH.parent
SCRIPTS_DIR = BENCH_DIR.parent
PDD_BENCH_DIR = _find_pdd_bench_root(SCRIPT_PATH)
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"

for p in (PDD_BENCH_DIR, METADRIVE_DIR, BENCH_DIR):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

CARL_NUPLAN_DIR = SDC_ROOT / "CaRL" / "nuPlan"
if CARL_NUPLAN_DIR.exists():
    carl_path = str(CARL_NUPLAN_DIR)
    if carl_path not in sys.path:
        sys.path.insert(0, carl_path)

from envs.sumo_env import TrafficSignSumoEnv
from envs.sumo_traffic_manager import SumoTrafficManager
from envs.traffic_sign_env import TrafficSignEnv
from metadrive.policy.idm_policy import IDMPolicy, ModifiedIDMPolicy
from metadrive.policy.expert_policy import ExpertPolicy
from stable_baselines3 import PPO
from agents.policies.comprehensive_rule_expert import ComprehensiveRuleExpertPolicy
from agents.policies.rule_compliant_expert import RuleCompliantExpertPolicy

from metadrive_core.bev_cnn import CustomBEVCNN as CustomBEVCNN_5ch
from metadrive_core.observation_wrappers import AddStateObservationWrapper as AddStateObservationWrapper_5ch
from metadrive_core.ppo_w_o_stop_sign.wrappers import EnsureSuccessInfoWrapper
from factorized_space.ego_defaults import apply_ego_defaults, apply_ego_sampled, sample_ego_params


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
    from factorized_space.agent_profile_bank import apply_profile_to_idm_class

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


def _choose_manifest(code_dir: Path, backend: str) -> Path | None:
    if backend == "pgmap":
        p = code_dir / "pgmap_materialized.jsonl"
        return p if p.exists() and p.stat().st_size > 0 else None
    if backend == "paired":
        p = code_dir / "paired_materialized.jsonl"
        return p if p.exists() and p.stat().st_size > 0 else None
    if backend == "citymap":
        p = code_dir / "citymap_materialized.jsonl"
        return p if p.exists() and p.stat().st_size > 0 else None
    if backend == "sumo":
        p1 = code_dir / "sumo" / "sumo_manifest.jsonl"
        p2 = code_dir / "real_manifest.jsonl"
        if p1.exists() and p1.stat().st_size > 0:
            return p1
        if p2.exists() and p2.stat().st_size > 0:
            return p2
        return None
    return None


def collect_rows(
    benchmark_output_dir: Path,
    backends: list[str],
    only_codes: set[str],
    max_scenes_per_sign: int | None,
) -> list[dict]:
    rows: list[dict] = []
    counts: dict[tuple[str, str], int] = defaultdict(int)

    sign_dirs = sorted([d for d in benchmark_output_dir.iterdir() if d.is_dir() and d.name[:1].isdigit()])
    for sign_dir in sign_dirs:
        sign_code = _slug_to_code(sign_dir.name)
        if only_codes and sign_code not in only_codes:
            continue

        for backend in backends:
            manifest = _choose_manifest(sign_dir, backend)
            if manifest is None:
                continue

            for row in _load_jsonl_rows(manifest):
                if "valid" in row and not row["valid"]:
                    continue
                key = (backend, sign_code)
                if max_scenes_per_sign is not None and counts[key] >= max_scenes_per_sign:
                    continue
                row["_backend"] = backend
                row["_sign_code"] = sign_code
                rows.append(row)
                counts[key] += 1

    return rows


def _build_pgmap_env(row: dict, max_steps: int) -> TrafficSignEnv:
    from metadrive.component.pgblock.first_block import FirstPGBlock

    seed = int(row.get("seed") or row.get("deterministic_seed") or 0)
    _apply_manifest_profile_to_npcs(row)
    traffic_density = _manifest_traffic_density(row, default=0.1)
    horizon = _manifest_horizon(row, fallback=max_steps)
    spawn_lane = int(row["spawn_lane_index"])
    is_ramp_merge = row.get("block_id") == "r" and row.get("route_intent") == "merge"
    is_ramp_exit = row.get("block_id") == "R" and row.get("route_intent") == "exit"
    if is_ramp_merge:
        spawn_lane_tuple = ("2r1_0_", "2r1_1_", 0)
    elif is_ramp_exit:
        spawn_lane_tuple = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, int(row["lane_num"]) - 1)
    else:
        spawn_lane_tuple = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, spawn_lane)

    vehicle_config = {"show_lidar": False}
    spawn_vel = float(row.get("spawn_velocity_ms", 0.0) or 0.0)
    if spawn_vel > 0:
        vehicle_config["spawn_velocity"] = [spawn_vel, 0.0]
        vehicle_config["spawn_velocity_car_frame"] = True

    config = dict(
        start_seed=seed,
        use_render=False,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.CRITICAL,
        traffic_density=traffic_density,
        horizon=horizon,
        vehicle_config=vehicle_config,
        map_config={
            "type": "block_sequence",
            "config": row["block_sequence"],
            "lane_num": row["lane_num"],
            "lane_width": row["lane_width"],
        },
        random_spawn_lane_index=False,
        agent_configs={
            "default_agent": dict(
                use_special_color=True,
                spawn_lane_index=spawn_lane_tuple,
            )
        },
    )
    return TrafficSignEnv(config)


def _place_pgmap_sign(env: TrafficSignEnv, row: dict, seed: int) -> bool:
    from factorized_space.benchmark_runner import (
        BEGIN_TO_END,
        DETOUR_KEYS,
        LANE_CHANGE_KEYS,
        RESTRICTED_BEGIN_KEYS,
        SIGN_CLASS_MAP,
        _get_route_lanes,
        _override_route_intent,
        _pick_detour_lane,
        _pick_lane_for_lane_change,
        _pick_rightmost_lane,
        _pick_route_lane,
        _spawn_cyclists_on_lane,
    )
    from factorized_space.space_definition import BIKE_RELATED_SIGNS
    from traffic_signs.detour_obstacle import spawn_detour_obstacle

    if row.get("sign_type") is None and row.get("sign_type_start") and row.get("sign_type_end"):
        # Paired scene: place start + end signs on one lane segment.
        if not _override_route_intent(env, row):
            return False
        route_lanes = _get_route_lanes(env)
        if not route_lanes:
            return False

        zone_length = float(row.get("zone_length_m", 20.0))
        min_lane = 8.0 + zone_length + 4.0
        lane = _pick_route_lane(route_lanes, min_length=min_lane, road_network=env.current_map.road_network)
        if lane is None:
            return False

        sign_start_cls = SIGN_CLASS_MAP.get(row["sign_type_start"])
        sign_end_cls = SIGN_CLASS_MAP.get(row["sign_type_end"])
        if sign_start_cls is None or sign_end_cls is None:
            return False

        sign_mgr = env.engine.traffic_sign_manager
        half_w = lane.width_at(0) / 2 + 0.8
        start_long = 5.0
        end_long = start_long + zone_length
        # MetaDrive expects longitudinal offset from lane end.
        md_start = start_long - lane.length
        md_end = end_long - lane.length

        try:
            sign_mgr.add_sign(
                sign_start_cls,
                lane=lane,
                longitudinal_offset=md_start,
                lateral_offset=half_w,
                use_random_lane=False,
            )
            sign_mgr.add_sign(
                sign_end_cls,
                lane=lane,
                longitudinal_offset=md_end,
                lateral_offset=half_w,
                use_random_lane=False,
            )
            return True
        except Exception:
            return False

    sign_key = row["sign_type"]
    if sign_key not in SIGN_CLASS_MAP:
        return False
    sign_cls = SIGN_CLASS_MAP[sign_key]

    if not _override_route_intent(env, row):
        return False

    route_lanes = _get_route_lanes(env)
    if not route_lanes:
        return False

    veh = env.vehicle
    rn = env.current_map.road_network
    sign_mgr = env.engine.traffic_sign_manager

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
        return False

    if sign_key in RESTRICTED_BEGIN_KEYS:
        sign = sign_mgr.add_sign(sign_cls, lane=lane, use_random_lane=False)
        if sign_key in BEGIN_TO_END:
            sign_mgr.add_sign(BEGIN_TO_END[sign_key], lane=sign.lane, use_random_lane=False)
    else:
        sign = sign_mgr.add_sign(sign_cls, lane=lane, use_random_lane=False)

    if sign_key in DETOUR_KEYS and sign is not None:
        spawn_detour_obstacle(env.engine, sign.lane, sign)
    if sign_key in BIKE_RELATED_SIGNS and sign is not None:
        _spawn_cyclists_on_lane(env, sign.lane, seed, n=3)

    return True


def _build_sumo_env(row: dict, scenes_root: Path, max_steps: int) -> TrafficSignSumoEnv:
    SumoTrafficManager.EGO_SAFE_RADIUS = 15
    _apply_manifest_profile_to_npcs(row)
    traffic_density = _manifest_traffic_density(row, default=0.1)
    horizon = _manifest_horizon(row, fallback=max_steps)
    net_path = str(scenes_root / row["net_path"]) if not str(row["net_path"]).startswith("/") else str(row["net_path"])
    sign_spawn_distance = _resolve_sign_spawn_distance(row, scenes_root)

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
        vehicle_config={"show_lidar": False},
        debug_one_way_sign_selection=bool(row.get("debug_one_way_sign_selection", False)),
    )
    if row.get("road_id"):
        config["vehicle_config"]["spawn_lane_index"] = row["road_id"]
    if "spawn_lane_num" in row:
        config["spawn_lane_num"] = int(row["spawn_lane_num"])

    class _EnvWithTraffic(TrafficSignSumoEnv):
        @classmethod
        def default_config(cls):
            cfg = super().default_config()
            cfg["traffic_density"] = 0.0
            return cfg

        def setup_engine(self):
            super().setup_engine()
            self.engine.update_manager("traffic_manager", SumoTrafficManager())

    return _EnvWithTraffic(config)


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


def _try_apply_sumo_spawn_lane_from_manifest(env: TrafficSignSumoEnv, row: dict) -> None:
    road_id = row.get("road_id")
    lane_num = row.get("spawn_lane_num")
    if road_id is None or lane_num is None:
        return

    try:
        lane_key = f"lane_{str(road_id)}_{int(lane_num)}"
        lane = env.current_map.road_network.get_lane(lane_key)
        long = min(2.0, max(0.2, float(getattr(lane, "length", 1.0)) * 0.05))
        pos = lane.position(long, 0.0)
        heading = lane.heading_theta_at(long)
        env.vehicle.set_position(pos)
        env.vehicle.set_heading_theta(heading)
    except Exception:
        pass


def _wrap_for_policy(env, policy_type: str):
    if policy_type == "ppo_5ch":
        wrapped = AddStateObservationWrapper_5ch(
            env,
            debug=False,
            add_stop_signs=False,
            stop_sign_probability=0.0,
            stop_sign_min_lane_length=15.0,
        )
        wrapped = EnsureSuccessInfoWrapper(wrapped)
        return wrapped
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


def _unwrap_base_env(env):
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    return base_env


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
    # Bench2Drive/CARLA-like multiplicative penalty, with project-specific proxies.
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
    # Bench2Drive filters extreme spikes.
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


def _load_policy_models(policy: str, model_path: str | None):
    ppo_model = None
    carl_model = None
    plant2_model = None
    plant2_helpers = None

    if policy == "ppo_5ch":
        if not model_path:
            raise ValueError("--model-path is required for --policy ppo_5ch")
        ppo_model = PPO.load(
            model_path,
            device="cpu",
            custom_objects={"policy_kwargs": dict(features_extractor_class=CustomBEVCNN_5ch)},
        )
    elif policy == "carl":
        if not model_path:
            raise ValueError("--model-path is required for --policy carl")
        CARL_ADAPTER_PATH = SDC_ROOT / "carl_in_metadrive"
        aps = str(CARL_ADAPTER_PATH)
        if aps not in sys.path:
            sys.path.insert(0, aps)
        from agents.carl_in_metadrive.carl_adapter import CaRLMetaDriveAdapter

        carl_model = CaRLMetaDriveAdapter(model_path, device="cpu")
    elif policy == "plant2":
        if not model_path:
            raise ValueError("--model-path is required for --policy plant2")
        PLANT2_PATH = SDC_ROOT / "plant2"
        if str(PLANT2_PATH) not in sys.path:
            sys.path.insert(0, str(PLANT2_PATH))
        from scripts.agents.inference.eval_plant2_live_seeds_from_trajectories_gifs import load_plant_model
        from scripts.agents.train.train_metadrive_ppo_plant_stop_gifs import PlanTObsWrapper
        from carla_garage.plant2_control import get_target_speed_from_limit, plant2_predictions_to_action
        from metadrive.policy.metadrive_obs_to_plant2 import metadrive_obs_to_plant2_batch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        net, _ = load_plant_model(model_path, PLANT2_PATH / "PlanT", device=device)
        plant2_model = net.to(device)
        plant2_model.eval()
        plant2_helpers = {
            "PlanTObsWrapper": PlanTObsWrapper,
            "metadrive_obs_to_plant2_batch": metadrive_obs_to_plant2_batch,
            "plant2_predictions_to_action": plant2_predictions_to_action,
            "get_target_speed_from_limit": get_target_speed_from_limit,
            "device": device,
        }

    return {
        "ppo_model": ppo_model,
        "carl_model": carl_model,
        "plant2_model": plant2_model,
        "plant2_helpers": plant2_helpers,
    }


def run_one_episode(
    row: dict,
    backend: str,
    policy_type: str,
    models: dict,
    scenes_root: Path,
    max_steps: int,
    ego_variant: str,
    ego_sample_seed_base: int,
) -> dict:
    seed = int(row.get("seed") or row.get("deterministic_seed") or 0)
    np.random.seed(seed)
    random.seed(seed)

    if backend in ("pgmap", "paired", "citymap"):
        env = _build_pgmap_env(row, max_steps=max_steps)
    elif backend == "sumo":
        env = _build_sumo_env(row, scenes_root=scenes_root, max_steps=max_steps)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    raw_env = env
    if policy_type == "plant2":
        helpers = models["plant2_helpers"]
        env = helpers["PlanTObsWrapper"](env, max_objects=0, max_stop_signs=0, bev_resolution=128)
    env = _wrap_for_policy(env, policy_type)

    idm_policy = None
    if policy_type == "idm":
        idm_policy = IDMPolicy
    elif policy_type == "modified_idm":
        idm_policy = ModifiedIDMPolicy
    elif policy_type == "comprehensive_rule_expert":
        idm_policy = ComprehensiveRuleExpertPolicy
    elif policy_type == "rule_compliant":
        idm_policy = RuleCompliantExpertPolicy
    elif policy_type == "ppo_lidar":
        idm_policy = ExpertPolicy

    try:
        if backend == "sumo":
            env_seed = (int(row.get("sign_id", 0)) + int(row.get("var_idx", 0))) % 100000
        else:
            env_seed = seed
        obs, info = env.reset(seed=env_seed)
        base_env = _unwrap_base_env(env)

        if backend in ("pgmap", "paired", "citymap"):
            ok = _place_pgmap_sign(base_env, row, seed)
            if not ok:
                return {
                    "ok": False,
                    "error": "failed_to_place_pgmap_sign",
                    "backend": backend,
                    "scene_id": row.get("scene_id"),
                    "sign_type": row.get("_sign_code") or row.get("sign_code") or row.get("pdd_code") or row.get("sign_type"),
                    "seed": int(row.get("seed") or row.get("deterministic_seed") or -1),
                }

        policy_obj = None
        sampled_ego_params = None
        if idm_policy is not None:
            policy_obj = idm_policy(base_env.vehicle, seed)
            # NPC profile is applied via IDMPolicy class attrs globally.
            # Keep ego IDM-family policies deterministic by overriding defaults
            # on the policy instance.
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

        if policy_type == "carl" and models["carl_model"] is not None:
            models["carl_model"].reset()

        total_reward = 0.0
        violations = 0
        sign_violations = 0
        traffic_light_violations = 0
        crosswalk_violations = 0
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
        last_info: dict = {}

        for step in range(max_steps):
            if policy_type == "ppo_5ch":
                action, _ = models["ppo_model"].predict(obs, deterministic=True)
            elif policy_type == "carl":
                action = models["carl_model"].get_action(env.agent, env.engine)
            elif policy_type == "plant2":
                helpers = models["plant2_helpers"]
                ego = getattr(base_env, "agent", None) or getattr(base_env, "vehicle", None)
                if ego is None:
                    action = [0.0, 0.0]
                else:
                    batch = helpers["metadrive_obs_to_plant2_batch"](
                        base_env.engine,
                        ego,
                        route_ego_20x2=None,
                        speed_limit_kmh=None,
                        max_objects=0,
                        max_distance=75.0,
                        range_factor_front=16.0,
                        input_bev=True,
                        input_ego_speed=False,
                        bev_resolution=128,
                        bev_size_meters=64.0,
                        device=helpers["device"],
                    )
                    with torch.no_grad():
                        _, _, pred_plan, _ = models["plant2_model"](batch)
                    ego_speed = float(getattr(ego, "speed", 0.0))
                    speed_limit_idx = int(batch["speed_limit"][0].item())
                    target_speed_mps = helpers["get_target_speed_from_limit"](speed_limit_idx)

                    # Steer using pred_wps (expert-driven path) instead of pred_path
                    # (navigation route), which reduces detour-obstacle crashes.
                    pred_path, pred_wps, pred_speed = pred_plan
                    steer_source = pred_wps if pred_wps is not None else pred_path
                    pred_plan_for_action = (steer_source, pred_wps, pred_speed)

                    action = helpers["plant2_predictions_to_action"](
                        pred_plan_for_action,
                        current_speed=ego_speed,
                        target_speed_mps=target_speed_mps,
                        speed_limit_idx=speed_limit_idx,
                        speed_limits_kmh=(50, 80, 100, 120),
                        device=helpers["device"],
                        return_waypoints=False,
                    )
                    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            elif policy_obj is not None:
                action = policy_obj.act(base_env.vehicle.name)
            else:
                action = [0.0, 0.0]

            obs, reward, terminated, truncated, info = env.step(action)
            last_info = info
            total_reward += float(reward)
            steps += 1

            vehicle = base_env.agent
            sign_mgr = getattr(base_env.engine, "traffic_sign_manager", None)
            current_violation_texts = []
            if sign_mgr is not None and vehicle is not None:
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
                if current_violation_texts:
                    last_violation_texts = current_violation_texts[:3]
                    violation_text_ttl = 40
                elif violation_text_ttl > 0:
                    violation_text_ttl -= 1
                else:
                    last_violation_texts = []

            # Bench2Drive-like efficiency proxy.
            if vehicle is not None:
                sp = _nearby_speed_percentage(vehicle)
                if sp is not None:
                    speed_pct_samples.append(float(sp))

            # Additional quality/safety metrics.
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
                reached_dest = bool(info.get("arrive_dest", False))
                out_of_road = bool(info.get("out_of_road", False))
                crashed = bool(info.get("crash", False) or out_of_road)
                print("max_steps:   ", step)
                break
            
            text_dict = {
                "Step": step,
                "Speed": f"{vehicle.speed_km_h:.2f} km/h",
                "Current lane: ": vehicle.lane.index,
                "Current lane width: ": vehicle.lane.width
            }

            if current_violation_texts:
                text_dict["Violation"] = current_violation_texts[0]
                if len(current_violation_texts) > 1:
                    text_dict["Violation +"] = f"+{len(current_violation_texts) - 1} more"
            elif last_violation_texts:
                text_dict["Last violation"] = last_violation_texts[0]
                if len(last_violation_texts) > 1:
                    text_dict["Last violation +"] = f"+{len(last_violation_texts) - 1} more"
            
            # env.unwrapped.render(mode="topdown",
            #     # window=False,
            #     text=text_dict,
            #     screen_record=False,
            #     # film_size=(6000, 6000),
            #     # scaling=40,
            #     # screen_size=(600, 600),
            #     semantic_map=True,
            #     semantic_broken_line=True,
            #     draw_target_vehicle_trajectory=True,
            #     target_agent_heading_up=True)

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

        return {
            "ok": True,
            "backend": backend,
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
        }
    finally:
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
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in results:
        if not r.get("ok"):
            continue
        key = (str(r.get("backend")), str(r.get("sign_type")))
        grouped[key].append(r)

    summary: dict[str, dict] = {}
    for (backend, sign), runs in sorted(grouped.items()):
        success_rate = float(np.mean([x["success"] for x in runs])) if runs else 0.0
        crash_rate = float(np.mean([x["crashed"] for x in runs])) if runs else 0.0
        avg_violations = float(np.mean([x["violations"] for x in runs])) if runs else 0.0
        avg_sign_viol = float(np.mean([x.get("sign_violations", 0) for x in runs])) if runs else 0.0
        avg_tl_viol = float(np.mean([x.get("traffic_light_violations", 0) for x in runs])) if runs else 0.0
        avg_cw_viol = float(np.mean([x.get("crosswalk_violations", 0) for x in runs])) if runs else 0.0
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
        key = f"{backend}:{sign}"
        summary[key] = {
            "backend": backend,
            "sign_type": sign,
            "total_runs": len(runs),
            "success_rate": success_rate,
            "crash_rate": crash_rate,
            "average_violations": avg_violations,
            "average_sign_violations": avg_sign_viol,
            "average_traffic_light_violations": avg_tl_viol,
            "average_crosswalk_violations": avg_cw_viol,
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


def _episode_key_from_row(row: dict) -> tuple[str, str, str, int]:
    return (
        str(row.get("_backend", "")),
        str(row.get("scene_id", "")),
        str(row.get("_sign_code") or row.get("sign_code") or row.get("pdd_code") or row.get("sign_type") or ""),
        int(row.get("seed") or row.get("deterministic_seed") or -1),
    )


def _episode_key_from_result(r: dict) -> tuple[str, str, str, int]:
    return (
        str(r.get("backend", "")),
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
    parser = argparse.ArgumentParser(description="Run policies on per_sign_bench mini/full manifests")
    parser.add_argument("--policy", required=True,
                        choices=["idm", "modified_idm", "comprehensive_rule_expert", "rule_compliant", "ppo_5ch", "ppo_lidar", "carl", "plant2"])
    parser.add_argument("--model-path", type=str, default=None,
                        help="Required for ppo_5ch/carl/plant2")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--preset", type=str, default="mini", choices=["mini", "full", "full_last"])
    parser.add_argument("--benchmark-output", type=str, default="benchmark_output",
                        help="Base dir that contains <preset>/")
    parser.add_argument("--scenes-root", type=str, default=str(PDD_BENCH_DIR / "scenes"))
    parser.add_argument("--backends", type=str, default="sumo,pgmap,paired,citymap",
                        help="Comma-separated: sumo,pgmap,paired,citymap")
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
    parser.add_argument("--skip-error-episodes", action="store_true",
                        help="When used with --rerun-failed, keep previously errored episodes skipped")
    parser.add_argument("--debug-one-way-sign-selection", action="store_true",
                        help="Enable verbose lane-selection debug logs for signs 5.7.1/5.7.2")
    args = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, "CRITICAL"))

    benchmark_output_dir = (BENCH_DIR / args.benchmark_output / args.preset).resolve()
    if not benchmark_output_dir.exists():
        raise ValueError(f"Benchmark output not found: {benchmark_output_dir}")

    scenes_root = Path(args.scenes_root).resolve()
    if not scenes_root.exists():
        raise ValueError(f"Scenes root not found: {scenes_root}")

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    allowed = {"sumo", "pgmap", "paired", "citymap"}
    bad = [b for b in backends if b not in allowed]
    if bad:
        raise ValueError(f"Unsupported backends: {bad}; allowed={sorted(allowed)}")

    only_codes: set[str] = set()
    if args.sign_type:
        only_codes.add(args.sign_type)
    if args.sign_types.strip():
        only_codes.update([c.strip() for c in args.sign_types.split(",") if c.strip()])

    print(f"Policy: {args.policy}")
    print(f"Preset: {args.preset}")
    print(f"Backends: {backends}")
    print(f"Input: {benchmark_output_dir}")

    rows = collect_rows(
        benchmark_output_dir=benchmark_output_dir,
        backends=backends,
        only_codes=only_codes,
        max_scenes_per_sign=args.max_scenes_per_sign,
    )
    if not rows:
        raise RuntimeError("No scenes selected. Check --preset/--backends/--sign-type")

    print(f"Selected scenes: {len(rows)}")
    models = _load_policy_models(args.policy, args.model_path)

    out_dir = benchmark_output_dir / "policy_eval" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / f"episodes_{args.policy}.jsonl"

    existing_results = _load_existing_results(episodes_path)
    existing_by_key: dict[tuple[str, str, str, int], dict] = {}
    for r in existing_results:
        existing_by_key[_episode_key_from_result(r)] = r

    rows_to_run: list[dict] = []
    skipped = 0
    for row in rows:
        key = _episode_key_from_row(row)
        old = existing_by_key.get(key)
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

    print(f"Resume: loaded {len(existing_results)} existing episodes, skip {skipped}, run {len(rows_to_run)}")

    results_by_key: dict[tuple[str, str, str, int], dict] = dict(existing_by_key)
    write_mode = "a" if episodes_path.exists() else "w"
    with open(episodes_path, write_mode, encoding="utf-8") as f:
        for idx, row in enumerate(rows_to_run, start=1):
            backend = str(row["_backend"])
            scene_id = row.get("scene_id")
            sign_code = row.get("_sign_code")
            print(f"[{idx}/{len(rows_to_run)}] backend={backend} sign={sign_code} scene={scene_id}")
            try:
                if args.debug_one_way_sign_selection:
                    row["debug_one_way_sign_selection"] = True
                episode_t0 = time.time()
                r = run_one_episode(
                    row=row,
                    backend=backend,
                    policy_type=args.policy,
                    models=models,
                    scenes_root=scenes_root,
                    max_steps=args.max_steps,
                    ego_variant=args.ego_variant,
                    ego_sample_seed_base=args.ego_sample_seed_base,
                )
                episode_dt = time.time() - episode_t0
                print(f"{args.policy}  elapsed_s={episode_dt:.3f}")
            except Exception as exc:
                r = {
                    "ok": False,
                    "backend": backend,
                    "scene_id": scene_id,
                    "sign_type": sign_code,
                    "seed": int(row.get("seed") or row.get("deterministic_seed") or -1),
                    "error": str(exc),
                }
            results_by_key[_episode_key_from_result(r)] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
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
