import argparse
import json
import logging
import os
import random
from pathlib import Path
import sys
import logging
import numpy as np


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

for path in (PDD_BENCH_DIR, METADRIVE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

CARL_NUPLAN_DIR = SDC_ROOT / "CaRL" / "nuPlan"
if CARL_NUPLAN_DIR.exists():
    carl_path = str(CARL_NUPLAN_DIR)
    if carl_path not in sys.path:
        sys.path.insert(0, carl_path)

from envs.sumo_env import TrafficSignSumoEnv
from metadrive.policy.idm_policy import IDMPolicy, ModifiedIDMPolicy

from stable_baselines3 import PPO
from metadrive_core.bev_cnn import CustomBEVCNN as CustomBEVCNN_5ch
from metadrive_core.observation_wrappers import AddStateObservationWrapper as AddStateObservationWrapper_5ch
from metadrive_core.ppo_w_o_stop_sign.wrappers import EnsureSuccessInfoWrapper

from agents.policies.comprehensive_rule_expert import ComprehensiveRuleExpertPolicy

import torch
import cv2

from envs.sumo_traffic_manager import SumoTrafficManager

# ---------------------------------------------------------------------------
# Sign-group mapping for GCR (Group Compliance Rate)
# ---------------------------------------------------------------------------
SIGN_GROUP = {
    # Priority
    "2.1": "Priority", "2.2": "Priority", "2.3.1": "Priority",
    "2.3.2": "Priority", "2.3.3": "Priority", "2.4": "Priority",
    "2.5": "Priority",
    # Prohibitory
    "3.1": "Prohibitory", "3.2": "Prohibitory",
    "3.18.1": "Prohibitory", "3.18.2": "Prohibitory",
    "3.19": "Prohibitory", "3.20": "Prohibitory",
    "3.21": "Prohibitory", "3.24": "Prohibitory",
    "3.25": "Prohibitory", "3.27": "Prohibitory",
    "3.31": "Prohibitory",
    # Mandatory
    "4.1.1": "Mandatory", "4.1.2": "Mandatory",
    "4.1.3": "Mandatory", "4.1.4": "Mandatory",
    "4.1.5": "Mandatory", "4.1.6": "Mandatory",
    "4.2.1": "Mandatory", "4.2.2": "Mandatory",
    "4.2.3": "Mandatory", "4.6": "Mandatory",
    "5.3": "Mandatory", "5.4": "Mandatory",
    "5.5": "Mandatory", "5.7.1": "Mandatory",
    "5.7.2": "Mandatory", "5.11.1": "Mandatory",
    "5.11.2": "Mandatory", "5.12.1": "Mandatory",
    "5.12.2": "Mandatory", "5.13.1": "Mandatory",
    "5.13.2": "Mandatory", "5.13.3": "Mandatory",
    "5.13.4": "Mandatory", "5.14.1": "Mandatory",
    "5.14.2": "Mandatory", "5.14.3": "Mandatory",
    "5.14.4": "Mandatory", "5.15.1": "Mandatory",
    "5.15.2": "Mandatory", "5.16": "Mandatory",
    "5.31": "Mandatory", "5.32": "Mandatory",
}


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _route_completion_percent(info: dict, reached_dest: bool) -> float:
    for k in ("route_completion", "route_completion_rate", "route_completion_ratio", "route_completion_percentage"):
        if k in info:
            v = _safe_float(info.get(k), 0.0)
            if v <= 1.0:
                v *= 100.0
            return max(0.0, min(100.0, v))
    return 100.0 if reached_dest else 0.0


def _infraction_penalty(crashed: bool, out_of_road: bool, violations: int) -> float:
    p = 1.0
    if crashed:
        p *= 0.5
    if out_of_road:
        p *= 0.7
    if violations > 0:
        p *= 0.9 ** int(violations)
    return max(0.0, min(1.0, p))


def _compute_smoothness(step_vars: list, segment_len: int = 20) -> dict:
    if not step_vars:
        return {"smoothness_ratio": 0.0, "smooth_segments": 0, "total_segments": 0, "frame_smooth_ratio": 0.0}

    def _frame_ok(v):
        return (-4.05 <= v["long_acc"] <= 2.40 and abs(v["lat_acc"]) <= 4.89
                and abs(v["yaw_rate"]) <= 0.95 and abs(v["yaw_acc"]) <= 1.93
                and abs(v["long_jerk"]) <= 4.13 and abs(v["jerk_mag"]) <= 8.37)

    flags = [_frame_ok(v) for v in step_vars]
    frame_ratio = float(np.mean(flags)) if flags else 0.0
    total_seg = len(step_vars) // segment_len
    if total_seg <= 0:
        return {"smoothness_ratio": frame_ratio, "smooth_segments": int(sum(flags)),
                "total_segments": len(flags), "frame_smooth_ratio": frame_ratio}
    smooth_seg = sum(1 for i in range(total_seg) if all(flags[i * segment_len:(i + 1) * segment_len]))
    return {"smoothness_ratio": float(smooth_seg / total_seg), "smooth_segments": int(smooth_seg),
            "total_segments": int(total_seg), "frame_smooth_ratio": frame_ratio}


def _list_parallel_lane_nums(net_path: str, road_id: str):
    """Parse the SUMO .net.xml and return sorted lane_num ints available on road_id."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(net_path).getroot()
    except Exception:
        return [0]
    nums = []
    for edge in root.findall('edge'):
        if edge.get('id') != road_id:
            continue
        for lane in edge.findall('lane'):
            lid = lane.get('id', '')
            try:
                nums.append(int(lid.rsplit('_', 1)[1]))
            except (ValueError, IndexError):
                continue
    return sorted(set(nums)) or [0]


def _filter_relevant_lanes(map_path: str, road_id: str, sign_type: str, lane_nums):
    """Spin up a short-lived SUMO env to load the road graph, then use
    `is_lane_relevant_for_sign` to drop lanes whose forward path U-turns to the
    opposing flow before reaching the sign's drivable area. Returns the
    filtered sorted list of lane_num ints (always non-empty — falls back to
    [0] if every lane was rejected)."""
    from envs.sumo_env import TrafficSignSumoEnv
    env = TrafficSignSumoEnv(dict(
        use_render=False, horizon=3, map_name=map_path,
        sign_type=sign_type, sign_spawn_distance=0.0,
        vehicle_config={"spawn_lane_index": road_id},
        spawn_lane_num=0, log_level=50,
    ))
    try:
        env.reset()
        relevant = [
            n for n in lane_nums
            if env.is_lane_relevant_for_sign(f"lane_{road_id}_{n}", road_id)
        ]
    finally:
        env.close()
    return relevant or [0]


# ---------------------------------------------------------------------------
# Top-down GIF helpers (same behaviour as run_benchmark_v2.py)
# ---------------------------------------------------------------------------

def _pred_path_to_pixels(pred_path_xy, scaling, screen_size):
    """PlanT ego frame (x forward, y right) → pixel coords for heading-up view."""
    if pred_path_xy is None or len(pred_path_xy) < 1:
        return None
    arr = np.asarray(pred_path_xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    cx, cy = screen_size[0] // 2, screen_size[1] // 2
    out = np.empty((arr.shape[0], 2), dtype=np.int32)
    for i in range(arr.shape[0]):
        ex = float(arr[i, 0])
        ey = float(arr[i, 1])
        out[i, 0] = int(cx + ey * scaling)
        out[i, 1] = int(cy - ex * scaling)
    return out


def _draw_pred_path_on_frame(
    frame,
    pred_path_xy,
    scaling,
    screen_size,
    color=(0, 0, 255),
    thickness=2,
):
    import cv2
    pix = _pred_path_to_pixels(pred_path_xy, scaling, screen_size)
    if pix is None or len(pix) < 2:
        return
    cv2.polylines(
        frame, [pix], isClosed=False, color=color, thickness=thickness,
    )
    for j in (0, len(pix) - 1):
        cv2.circle(frame, (int(pix[j, 0]), int(pix[j, 1])), 3, color, -1)


def _extract_pred_path_numpy(pred_plan):
    pred_path_t = pred_plan[0]
    if pred_path_t is None:
        return None
    pp = pred_path_t.detach().cpu().numpy()
    if pp.ndim > 2:
        pp = pp.squeeze(0)
    if pp.size == 0 or pp.shape[-1] < 2:
        return None
    return pp.astype(np.float32)


class _EnvWithTraffic(TrafficSignSumoEnv):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg["traffic_density"] = 0.0
        return cfg

    def setup_engine(self):
        super().setup_engine()
        self.engine.update_manager("traffic_manager", SumoTrafficManager())


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


def create_env_for_policy(policy_type, map_path, road_id, sign_type, sign_spawn_distance, spawn_lane_num=0, model_path=None, manual_control=False, render_mode="topdown", traffic_density=0.1, plant2_no_mixin=False):
    """
    Create environment wrapped appropriately for the given policy.

    render_mode: "topdown" for semantic 2D view, "3d" for Panda3D 3D view
    traffic_density: 0.0 = no surrounding agents, 1.0 = maximum traffic
    """
    use_3d_render = (render_mode == "3d")

    base_config = dict(
        use_render=use_3d_render,
        vehicle_config={"spawn_lane_index": road_id},
        manual_control=manual_control,
        use_mesh_terrain=False,
        log_level=logging.CRITICAL,
        map_name=map_path,
        sign_type=sign_type,
        sign_spawn_distance=max(float(sign_spawn_distance or 0.0), 30.0),
        spawn_lane_num=int(spawn_lane_num or 0),
        traffic_density=traffic_density,
        tl_speed_factor=20.0,
        window_size=(1200, 800) if manual_control else (800, 800),
        show_coordinates=manual_control,
    )

    if traffic_density > 0:
        base_config["traffic_density"] = traffic_density
        EnvClass = _EnvWithTraffic
    else:
        EnvClass = TrafficSignSumoEnv

    if policy_type == "ppo_5ch":
        env = _EnvWithTraffic(base_config)
        env = AddStateObservationWrapper_5ch(
            env,
            debug=False,
            add_stop_signs=False,
            stop_sign_probability=0.0,
            stop_sign_min_lane_length=15.0
        )
        env = EnsureSuccessInfoWrapper(env)
        return env
    elif policy_type == "carl":
        env = _EnvWithTraffic(base_config)
        return env
    elif policy_type == "plant2":
        if plant2_no_mixin:
            from scripts.agents.train.train_metadrive_ppo_plant_stop_gifs import PlanTObsWrapper
            env = EnvClass(base_config)
            env = PlanTObsWrapper(env, max_objects=0, max_stop_signs=0, bev_resolution=128)
            return env
        from agents.policies.plant2_sign_compliant import PlanT2SignCompliantPolicy
        config = {**base_config, "agent_policy": PlanT2SignCompliantPolicy}
        return EnvClass(config)
    else:
        agent_policy = IDMPolicy if policy_type == "idm" else ComprehensiveRuleExpertPolicy
        config = {**base_config, "agent_policy": agent_policy}
        env = _EnvWithTraffic(config)
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


def run_single_episode(
    env_config,
    seed,
    policy_type,
    ppo_model=None,
    carl_model=None,
    plant2_model=None,
    plant2_no_mixin=False,
    gif_path=None,
    manual_control=False,
    render_mode="topdown",
    traffic_density=0.1,
):
    np.random.seed(seed)
    random.seed(seed)

    env = create_env_for_policy(
        policy_type=policy_type,
        map_path=env_config["map_name"],
        road_id=env_config["vehicle_config"]["spawn_lane_index"],
        sign_type=env_config["sign_type"],
        sign_spawn_distance=env_config["sign_spawn_distance"],
        spawn_lane_num=env_config.get("spawn_lane_num", 0),
        manual_control=manual_control,
        render_mode=render_mode,
        traffic_density=traffic_density,
        plant2_no_mixin=plant2_no_mixin,
    )

    try:
        obs, info = env.reset()
        if policy_type == "carl":
            carl_model.reset()
        total_reward = 0
        violations = 0
        sign_violations = 0
        steps = 0
        crashed = False
        out_of_road = False
        reached_dest = False
        violated_sign_ids = set()
        last_violation_texts = []
        violation_text_ttl = 0
        efficiency_samples = []
        smooth_step_vars = []
        violation_groups = set()  # sign groups that were violated this episode
        prev_long = None
        prev_steer = None
        prev_velocity = None

        if hasattr(env, "_get_base_env"):
            base_env = env._get_base_env()
        else:
            base_env = env.unwrapped
            while hasattr(base_env, "env"):
                base_env = base_env.env

        # Inline plant2 (legacy raw, no mixin)
        if policy_type == "plant2" and plant2_no_mixin:
            import torch as _torch  # noqa: F401  (kept on local scope below)
            from metadrive.policy.metadrive_obs_to_plant2 import metadrive_obs_to_plant2_batch
            from carla_garage.plant2_control import (
                plant2_predictions_to_action,
                get_target_speed_from_limit,
            )
            plant2_device = next(plant2_model.parameters()).device
            print("plant2_device:   ", plant2_device)

        save_gif = gif_path is not None
        RENDER_SCALING = 40
        SCREEN_SIZE = (600, 600)
        if save_gif:
            renderer = getattr(base_env, "top_down_renderer", None)
            if renderer is not None and hasattr(renderer, "_screen_frames"):
                renderer._screen_frames.clear()

        # Diagnostics: which verifiers are registered this episode
        _sm = base_env.engine.traffic_sign_manager
        _registered_signs = [type(s).__name__ for s in _sm.signs if type(s).__name__ != "TrafficLightSign"]
        _registered_rules = [str(getattr(r, "id", None) or type(r).__name__) for r in _sm.rules]
        print(f"[episode] seed={seed} sign_type={env_config['sign_type']} "
              f"signs={_registered_signs} rules={_registered_rules} "
              f"(TL signs hidden: {len(_sm.signs) - len(_registered_signs)})")

        output_dir = "./carl_v3_output"
        max_steps = 10000 if manual_control else 1500
        for step in range(600):
            pred_plan_for_viz = None
            if manual_control:
                action = [0.0, 0.0]
            elif policy_type == "ppo_5ch":
                action, _ = ppo_model.predict(obs, deterministic=True)
            elif policy_type == "carl":
                action = carl_model.get_action(env.agent, env.engine)
            elif policy_type == "plant2" and plant2_no_mixin:
                # Legacy raw plant2: no mixin, inline obs->model->action.
                import torch
                ego = getattr(base_env, "agent", None) or getattr(base_env, "vehicle", None)
                if ego is None:
                    break
                batch = metadrive_obs_to_plant2_batch(
                    base_env.engine, ego,
                    route_ego_20x2=None, speed_limit_kmh=None,
                    max_objects=30, max_distance=75.0, range_factor_front=16.0,
                    input_bev=True, input_ego_speed=False,
                    bev_resolution=128, bev_size_meters=64.0,
                    device=plant2_device,
                )
                with torch.no_grad():
                    _, _, pred_plan, _ = plant2_model(batch)
                pred_plan_for_viz = pred_plan
                ego_speed = float(getattr(ego, "speed", 0.0))
                speed_limit_idx = int(batch["speed_limit"][0].item())
                target_speed_mps = get_target_speed_from_limit(speed_limit_idx)
                action = plant2_predictions_to_action(
                    pred_plan,
                    current_speed=ego_speed,
                    target_speed_mps=target_speed_mps,
                    speed_limit_idx=speed_limit_idx,
                    speed_limits_kmh=(50, 80, 100, 120),
                    device=plant2_device,
                    return_waypoints=False,
                )
                action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            else:
                # idm / modified_idm / plant2(+mixin) driven by `agent_policy`
                # set on the env config; the action passed to env.step is ignored.
                action = [0.0, 0.0]

            sign_mgr = base_env.engine.traffic_sign_manager
            prev_violation_events = len(sign_mgr.violation_details)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Collect efficiency and smoothness step data
            vehicle = base_env.agent
            speed_kmh = getattr(vehicle, "speed_km_h", 0.0) or 0.0
            try:
                long_acc = info.get("long_acc", info.get("acceleration", 0.0))
                lat_acc = info.get("lat_acc", 0.0)
            except Exception:
                long_acc = lat_acc = 0.0
            if prev_velocity is not None:
                dt = max(float(info.get("dt", 0.1)), 0.01)
                long_jerk = (long_acc - prev_long) / dt
                jerk_mag = abs(long_jerk)
            else:
                long_jerk = jerk_mag = 0.0
            prev_long = long_acc
            prev_velocity = speed_kmh
            try:
                lane = getattr(vehicle, "lane", None)
                if lane is not None and hasattr(lane, "speed_limit"):
                    limit = lane.speed_limit
                else:
                    limit = info.get("speed_limit", 60.0)
                if limit and limit > 0:
                    efficiency_samples.append(speed_kmh / limit)
            except Exception:
                pass
            smooth_step_vars.append({
                "long_acc": long_acc, "lat_acc": lat_acc,
                "yaw_rate": info.get("yaw_rate", 0.0),
                "yaw_acc": info.get("yaw_acc", 0.0),
                "long_jerk": long_jerk, "jerk_mag": jerk_mag,
            })

            vehicle = base_env.agent
            current_violations = base_env.engine.traffic_sign_manager.check_all_violations(vehicle)
            current_violation_names, current_violation_texts = [], []
            compliance_lines = []
            _mgr_rules = set(id(r) for r in base_env.engine.traffic_sign_manager.rules)
            for sign, violated_for_report in current_violations:
                sign_type_name = type(sign).__name__
                if sign_type_name == "TrafficLightSign":
                    continue
                if id(sign) in _mgr_rules:
                    name = str(getattr(sign, "id", None) or sign_type_name)
                    is_rule = True
                else:
                    name = sign_type_name
                    is_rule = False

                currently_violating = False

                if violated_for_report:
                    violations += 1
                    step_has_sign_violation = True
                    current_violation_names.append(name)
                    violated_sign_ids.add(name)
                    current_violation_texts.append(_format_violation(sign, vehicle))
                    # Map to sign group for GCR
                    sign_type_str = env_config["sign_type"]
                    grp = SIGN_GROUP.get(sign_type_str)
                    if grp:
                        violation_groups.add(grp)

                compliance_lines.append(f"{name}: {'VIOLATED' if currently_violating else 'OK'}")

            status = ", ".join(compliance_lines) if compliance_lines else "no signs"
            marker = "VIOLATION" if current_violation_names else "ok      "
            # print(f"step={step:04d} {marker} speed={vehicle.speed_km_h:5.1f} km/h lane={vehicle.lane.index:>28} | {status}")
            if current_violation_texts:
                last_violation_texts = current_violation_texts[:3]
                violation_text_ttl = 40
            elif violation_text_ttl > 0:
                violation_text_ttl -= 1
            else:
                last_violation_texts = []

            done = terminated or truncated
            # done = False
            if done:
                reached_dest = info.get("arrive_dest", False)
                out_of_road = bool(info.get("out_of_road", False))
                crashed = (
                    info.get("crash", False)
                    or getattr(vehicle, "crash_vehicle", False)
                    or out_of_road
                )
            # elif done and manual_control:
            #     if info.get("arrive_dest", False):
            #         print("Arrived at destination! Press ESC to quit or continue exploring.")
            #     elif info.get("out_of_road", False):
            #         print("Went off road! Episode will reset...")
            #         obs, info = env.reset()
            #         continue

            if policy_type == "carl" and step % 10 == 0:
                carl_obs = carl_model.get_observation(env.agent, env.engine)
                vis = carl_model.visualize_bev(carl_obs["bev_semantics"])
                cv2.imwrite(f"{output_dir}/ep{0:02d}_step{step:04d}.png", vis)

            current_violations_str = ", ".join(current_violation_names) if current_violation_names else "-"
            text_dict = {
                "Step": step,
                "Speed": f"{vehicle.speed_km_h:.2f} km/h",
                "Current lane": vehicle.lane.index,
                "Current lane width": f"{vehicle.lane.width:.1f}m",
                "Violations (total)": violations,
                "Violation now": current_violations_str,
            }
            for i, line in enumerate(compliance_lines[:6]):
                text_dict[f"Compliance[{i}]"] = line

            if current_violation_texts:
                text_dict["Violation"] = current_violation_texts[0]
                if len(current_violation_texts) > 1:
                    text_dict["Violation +"] = f"+{len(current_violation_texts) - 1} more"
            elif last_violation_texts:
                text_dict["Last violation"] = last_violation_texts[0]
                if len(last_violation_texts) > 1:
                    text_dict["Last violation +"] = f"+{len(last_violation_texts) - 1} more"

            if manual_control:
                if render_mode == "3d":
                    env.unwrapped.render(text=text_dict)
                else:
                    env.unwrapped.render(
                        mode="top_down",
                        text=text_dict,
                        film_size=(6000, 6000),
                        scaling=RENDER_SCALING,
                        screen_size=SCREEN_SIZE,
                        semantic_map=True,
                        semantic_broken_line=True,
                        draw_target_vehicle_trajectory=True,
                        target_agent_heading_up=True,
                    )
            elif save_gif:
                env.unwrapped.render(
                    mode="topdown",
                    text=text_dict,
                    screen_record=True,
                    film_size=(6000, 6000),
                    scaling=RENDER_SCALING,
                    screen_size=SCREEN_SIZE,
                    semantic_map=True,
                    semantic_broken_line=True,
                    draw_target_vehicle_trajectory=True,
                    target_agent_heading_up=True,
                )
                renderer = getattr(base_env, "top_down_renderer", None)
                if (
                    policy_type == "plant2"
                    and pred_plan_for_viz is not None
                    and renderer is not None
                    and getattr(renderer, "_screen_frames", None)
                ):
                    frame = renderer._screen_frames[-1]
                    pred_path_viz = _extract_pred_path_numpy(pred_plan_for_viz)
                    _draw_pred_path_on_frame(
                        frame, pred_path_viz, RENDER_SCALING, SCREEN_SIZE,
                    )
            else:
                env.unwrapped.render(
                    mode="topdown",
                    text=text_dict,
                    screen_record=False,
                    film_size=(10000, 10000),
                    scaling=RENDER_SCALING,
                    screen_size=SCREEN_SIZE,
                    semantic_map=True,
                    semantic_broken_line=True,
                    draw_target_vehicle_trajectory=True,
                    target_agent_heading_up=True,
                )

            if done:
                print("Max step:    ", step)
                break

        if save_gif:
            renderer = getattr(base_env, "top_down_renderer", None)
            if renderer is not None:
                renderer.generate_gif(gif_path, duration=10)
                if hasattr(renderer, "clear"):
                    renderer.clear()
                elif hasattr(renderer, "_screen_frames"):
                    renderer._screen_frames.clear()
        # elif not manual_control:
        #     os.makedirs("./gifs", exist_ok=True)
        #     _ln = env_config.get("spawn_lane_num", 0)
        #     base_env.top_down_renderer.generate_gif(
        #         gif_name=f"./gifs/demo_{env_config.get('sign_type','x')}_{seed}_lane{_ln}.gif"
        #     )

        sm = _compute_smoothness(smooth_step_vars) if smooth_step_vars else {}
        eff_mean = float(np.mean(efficiency_samples)) * 100.0 if efficiency_samples else 0.0
        route_pct = _route_completion_percent(info, reached_dest)
        infr_pen = _infraction_penalty(crashed, out_of_road, violations)
        sign_violations = violations  # all tracked violations are sign violations here

        return {
            "seed": seed,
            "total_reward": total_reward,
            "steps": steps,
            "violations": violations,
            "sign_violations": violations,
            "crashed": crashed,
            "out_of_road": out_of_road,
            "reached_dest": reached_dest,
            "success": reached_dest and not crashed,
            "efficiency": eff_mean,
            "smoothness": sm.get("smoothness_ratio", 0.0) * 100.0,
            "route_completion_pct": route_pct,
            "infraction_penalty": infr_pen,
            "driving_score": route_pct * infr_pen / 100.0,
            "violation_groups": list(violation_groups),
            "has_sign_violation": int(violations > 0),
        }

    finally:
        env.close()

import time

import numpy as np

def main():
    logging.getLogger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, choices=["idm", "modified_idm", "ppo_5ch", "carl", "plant2", "comprehensive_rule_expert"], help="Policy to evaluate")
    parser.add_argument("--model-path", type=str, default=None, help="Path to PPO/carl/plant2 model (required for --policy ppo_5ch/carl/plant2)")
    parser.add_argument("--plant2-repo-dir", type=str, default=None,
                        help="Path to plant2 repo root (default: SDC_ROOT/plant2). Only used with --policy plant2.")
    parser.add_argument("--plant2-no-mixin", action="store_true",
                        help="Legacy raw plant2 mode: skip SignComplianceMixin, run obs->model->action inline. Only used with --policy plant2.")
    parser.add_argument("--sign-type", type=str, default=None, help='e.g. "2.5". If not provided, run ALL sign types.')
    parser.add_argument("--run-name", type=str, required=True, help='Name of this benchmark run')
    parser.add_argument("--seeds", type=str, default="", help='Comma-separated seeds. If empty, use sign_id as seed.')
    parser.add_argument("--scenes-dir", type=str, default="pdd-bench/scenes", help="Path to scenes root")
    parser.add_argument("--max-scenes-per-sign", type=int, default=None, help="Limit number of scenes per sign type")
    parser.add_argument("--spawn-all-lanes", action="store_true",
                        help="Run each scene once per parallel lane of the sign's road (default: rightmost only).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="carl_v3_output",
        help="If set, save top-down GIFs under <output-dir>/gifs/ (same layout as run_benchmark_v2.py).",
    )
    parser.add_argument(
        "--no-gifs",
        action="store_true",
        help="Do not record GIFs even if --output-dir is set.",
    )
    parser.add_argument("--manual", action="store_true", help="Enable manual control (keyboard W/A/S/D) with visualization")
    parser.add_argument("--render-mode", type=str, default="topdown", choices=["topdown", "3d"], help="Render mode for manual control")
    parser.add_argument("--traffic-density", type=float, default=0.0, help="Surrounding agent density (0.0 = no traffic, 1.0 = max). Default: 0.0")
    args = parser.parse_args()

    ppo_model = None
    carl_model = None
    plant2_model = None

    device = "cpu"

    if args.policy == "ppo_5ch":
        if not args.model_path:
            raise ValueError("--model-path is required for --policy ppo_5ch")
        try:
            ppo_model = PPO.load(
                args.model_path,
                device="cpu",
                custom_objects={"policy_kwargs": dict(features_extractor_class=CustomBEVCNN_5ch)}
            )
            print(f"Loaded 5-channel PPO model from {args.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load PPO model: {e}")

    elif args.policy == "carl":
        if not args.model_path:
            raise ValueError("--model-path is required for --policy carl")
        try:
            _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            CARL_PATH = os.environ.get("CARL_PATH", os.path.join(_repo_root, "CaRL", "nuPlan"))
            CARL_ADAPTER_PATH = os.environ.get(
                "CARL_ADAPTER_PATH",
                os.path.join(_repo_root, "pdd-bench", "agents", "carl_in_metadrive"),
            )
            for path in [CARL_PATH, CARL_ADAPTER_PATH]:
                if path not in sys.path:
                    sys.path.insert(0, path)
            from agents.carl_in_metadrive.carl_adapter import CaRLMetaDriveAdapter
            carl_model = CaRLMetaDriveAdapter(args.model_path, device="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load carl model: {e}")

    elif args.policy == "plant2":
        if not args.model_path:
            raise ValueError("--model-path is required for --policy plant2 (pdd-bench/checkpoints/epoch%3D029_final_3.ckpt)")
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        plant2_repo = Path(args.plant2_repo_dir) if args.plant2_repo_dir else (SDC_ROOT / "plant2")

        if args.plant2_no_mixin:
            plant2_path_str = str(plant2_repo)
            if plant2_path_str not in sys.path:
                sys.path.insert(0, plant2_path_str)
            from scripts.agents.inference.eval_plant2_live_seeds_from_trajectories_gifs import load_plant_model
            net, _ = load_plant_model(args.model_path, str(plant2_repo / "PlanT"), device=device)
            plant2_model = net.to(device)
            plant2_model.eval()
        else:
            from agents.policies.plant2_sign_compliant import PlanT2SignCompliantPolicy
            PlanT2SignCompliantPolicy.set_checkpoint(args.model_path, plant2_repo, device=device)


    scenes_root = Path(args.scenes_dir)
    if not scenes_root.exists():
        raise ValueError(f"Scenes root not found: {scenes_root}")

    if args.sign_type:
        sign_types = [args.sign_type]
    else:
        sign_types = sorted([d.name for d in scenes_root.iterdir() if d.is_dir()])

    print(f"Running benchmark for sign types: {sign_types}")

    gif_root = None
    if args.output_dir and not args.no_gifs:
        gif_root = Path(args.output_dir) / "gifs"
        gif_root.mkdir(parents=True, exist_ok=True)
        print(f"GIF output directory: {gif_root}")

    seen_scenes = set()

    results = []

    for sign_type in sign_types:
        sign_type_dir = scenes_root / sign_type
        if not sign_type_dir.exists():
            print(f"Skipping sign type '{sign_type}': directory not found")
            continue

        scene_dirs = sorted([d for d in sign_type_dir.iterdir() if d.is_dir()])
        if args.max_scenes_per_sign:
            scene_dirs = scene_dirs[:args.max_scenes_per_sign]

        if not scene_dirs:
            print(f"No scenes for sign type '{sign_type}'")
            continue

        if args.seeds.strip():
            seeds = [int(s) for s in args.seeds.split(",")]
        else:
            seeds = None
        
        scene_num = None
        if scene_num is not None:
            scene_dirs = [scene_dirs[scene_num]]

        for scene_dir in scene_dirs:
            meta_path = scene_dir / "meta.json"
            if not meta_path.exists():
                print(f"Skipping {scene_dir}: no meta.json")
                continue

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            net_file = meta["net_file"]
            map_path = scene_dir / net_file
            road_id = meta["road_id"]
            sign_id = meta["sign_id"]
            sign_spawn_distance = meta["distance_from_start"]

            scene_key = (sign_type, road_id)
            seen_scenes.add(scene_key)

            scene_seeds = seeds if seeds is not None else [int(sign_id)]
            if int(sign_id) not in scene_seeds:
                continue

            use_seed_suffix = seeds is not None or len(scene_seeds) > 1

            lane_nums = [0]
            if args.spawn_all_lanes:
                all_nums = _list_parallel_lane_nums(str(map_path.absolute()), road_id)
                lane_nums = _filter_relevant_lanes(
                    str(map_path.absolute()), road_id, sign_type, all_nums,
                )
                print(f"  spawn-all-lanes: parallel={all_nums}  relevant={lane_nums}")
            elapseds = []
            for seed in scene_seeds:
                for lane_num in lane_nums:
                    print(f"\nRunning scene {scene_dir.name} (sign_type={sign_type}) seed={seed} lane_num={lane_num}")

                    if gif_root is not None:
                        if use_seed_suffix or args.spawn_all_lanes:
                            gif_name = f"{sign_type}_{scene_dir.name}_seed{seed}_lane{lane_num}.gif"
                        else:
                            gif_name = f"{sign_type}_{scene_dir.name}.gif"
                        episode_gif_path = str((gif_root / gif_name).resolve())
                    else:
                        episode_gif_path = None

                    env_config = dict(
                        vehicle_config={"spawn_lane_index": road_id},
                        map_name=str(map_path.absolute()),
                        sign_type=sign_type,
                        sign_spawn_distance=sign_spawn_distance,
                        spawn_lane_num=lane_num,
                    )
                    episode_t0 = time.time()
                    result = run_single_episode(
                        env_config=env_config,
                        seed=seed,
                        policy_type=args.policy,
                        ppo_model=ppo_model,
                        carl_model=carl_model,
                        plant2_model=plant2_model,
                        plant2_no_mixin=args.plant2_no_mixin,
                        gif_path=episode_gif_path,
                        manual_control=args.manual,
                        render_mode=args.render_mode,
                        traffic_density=args.traffic_density,
                    )
                    episode_dt = time.time() - episode_t0
                    elapseds.append(episode_dt)
                    print(f"  elapsed_s={episode_dt:.3f}")
                    result["scene"] = scene_dir.name
                    result["sign_type"] = sign_type
                    result["spawn_lane_num"] = lane_num
                    results.append(result)
            print("elapseds mean:   ", np.mean(elapseds))
    from collections import defaultdict
    results_by_sign = defaultdict(list)

    for r in results:
        if "error" not in r and "sign_type" in r:
            results_by_sign[r["sign_type"]].append(r)

    summary_all = {}
    for sign_type, runs in results_by_sign.items():
        valid_runs = [r for r in runs if "error" not in r]
        if not valid_runs:
            metrics = {
                "run_name": args.run_name, "total_runs": 0,
                "success_rate": 0.0, "crash_rate": 0.0,
                "average_violations": 0.0, "average_reward": 0.0,
                "efficiency": 0.0, "smoothness": 0.0,
                "scr": 0.0, "route_completion_pct": 0.0,
            }
            rg = SIGN_GROUP.get(sign_type, "Special")
            metrics[f"gcr_{rg}"] = 0.0
        else:
            success_rate = np.mean([r["success"] for r in valid_runs])
            crash_rate = np.mean([r["crashed"] for r in valid_runs])
            avg_violations = np.mean([r["violations"] for r in valid_runs])
            avg_reward = np.mean([r["total_reward"] for r in valid_runs])
            avg_eff = np.mean([r.get("efficiency", 0.0) for r in valid_runs])
            avg_smooth = np.mean([r.get("smoothness", 0.0) for r in valid_runs])
            avg_route = np.mean([r.get("route_completion_pct", 0.0) for r in valid_runs])
            scr = np.mean([r.get("has_sign_violation", 1) == 0 for r in valid_runs])

            gcr = {}
            relevant_group = SIGN_GROUP.get(sign_type, "Special")
            for g in [relevant_group]:
                group_runs = [r for r in valid_runs if g in r.get("violation_groups", [])]
                gcr[g] = 1.0 - (len(group_runs) / max(len(valid_runs), 1))

            metrics = {
                "run_name": args.run_name, "total_runs": len(valid_runs),
                "success_rate": float(success_rate),
                "crash_rate": float(crash_rate),
                "average_violations": float(avg_violations),
                "average_reward": float(avg_reward),
                "efficiency": float(avg_eff),
                "smoothness": float(avg_smooth),
                "scr": float(scr),
                "route_completion_pct": float(avg_route),
            }
            for g in [relevant_group]:
                metrics[f"gcr_{g}"] = float(gcr[g])

            print(f"\n{'='*60}")
            print(f"Sign type: {sign_type}  |  Runs: {metrics['total_runs']}")
            print(f"{'='*60}")
            print(f"  Eff: {avg_eff:6.1f}  |  Comf: {avg_smooth:5.1f}  |  "
                  f"SCR: {scr*100:5.1f}%")
            print(f"  GCR {relevant_group}: {gcr[relevant_group]*100:5.1f}%")
            print(f"  Route: {avg_route:5.1f}%  |  "
                  f"Success: {success_rate*100:5.1f}%  |  "
                  f"Crash: {crash_rate*100:5.1f}%")
            print('='*60)

        output_file = scenes_root / f"benchmark_results_{sign_type}_{args.run_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"✅ Metrics for '{sign_type}' saved to {output_file}")

        summary_all[sign_type] = metrics

    if summary_all:
        summary_file = scenes_root / f"benchmark_summary_ALL_{args.run_name}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_all, f, indent=2, ensure_ascii=False)
        print(f"\n📁 Full summary saved to {summary_file}")


if __name__ == "__main__":
    main()
