#!/usr/bin/env python3
"""
Plant2 benchmark on SUMO scenes (v2).

Iterates over ``scenes/<sign_type>/<scene_dir>/`` folders, loads the .net.xml
map into :class:`SumoEnvV2`, runs Plant2 closed-loop inference, and collects
per-scene + aggregate metrics.

Usage
-----
python scripts/run_benchmark_v2.py \
    --checkpoint /path/to/epoch%3D029_final_3.ckpt \
    --scenes-dir /path/to/pdd-bench/scenes \
    --sign-type "3.27" \
    --seeds "811891,802029" \
    --max-scenes 3 \
    --max-steps 1500 \
    --output-dir ./benchmark_results

``--seeds`` (same as ``run_benchmark.py``): comma-separated integers matching
``meta.json`` ``sign_id``. Only those scenes run; each run sets
``np.random.seed`` / ``random.seed`` like ``run_single_episode``, then builds
the env and calls ``env.reset()`` with no seed argument. If omitted, every
scene runs once with ``seed = sign_id``.
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path bootstrap
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
PLANT2_DIR = SDC_ROOT / "plant2"
PLANT_PLAN_T_DIR = PLANT2_DIR / "PlanT"

for _p in (PDD_BENCH_DIR, METADRIVE_DIR, PLANT_PLAN_T_DIR, PLANT2_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

if os.environ.get("SDL_VIDEODRIVER") is None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# ---------------------------------------------------------------------------
# Mock CARLA (not installed)
# ---------------------------------------------------------------------------

def _mock_carla_modules():
    import unittest.mock as _mock
    for mod_name in (
        "carla", "agents", "agents.navigation",
        "agents.navigation.global_route_planner",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _mock.MagicMock()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_plant_model(
    checkpoint_path: str, plant_planT_path: str, device: str = "cpu"
):
    import yaml

    _mock_carla_modules()

    model_yaml = os.path.join(plant_planT_path, "config", "model", "PlanT.yaml")
    if not os.path.isfile(model_yaml):
        raise FileNotFoundError(f"PlanT config not found: {model_yaml}")
    with open(model_yaml) as f:
        plnt = yaml.safe_load(f)

    class DictAsMember(dict):
        def __getattr__(self, name):
            value = self.get(name)
            if isinstance(value, dict) and not isinstance(value, DictAsMember):
                return DictAsMember(value)
            return value

    config_all = DictAsMember({"model": plnt})
    config_net = config_all.model.network

    if plant_planT_path not in sys.path:
        sys.path.insert(0, plant_planT_path)
    elif sys.path[0] != plant_planT_path:
        sys.path.remove(plant_planT_path)
        sys.path.insert(0, plant_planT_path)
    from model import HFLM  # type: ignore

    net = HFLM(config_net, config_all)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint contains no state_dict")
    if list(sd.keys())[0].startswith("model."):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=False)
    net.to(device).eval()
    return net, config_all


# ---------------------------------------------------------------------------
# SUMO-compatible BEV renderer
# ---------------------------------------------------------------------------
# The upstream render_bev_plant2 crashes on EdgeRoadNetwork because it
# expects road_network.graph to be {from: {to: [lanes]}} (NodeRoadNetwork)
# while SUMO maps use {lane_id: edge_lane(lane=..., ...)}.  This version
# iterates the flat EdgeRoadNetwork graph directly.

BEV_COLORS = np.array([
    [0.485, 0.456, 0.406],  # 0 Background (ImageNet mean)
    [0.25,  0.25,  0.75],   # 1 Street
    [0.485, 0.456, 0.406],  # 2 Sidewalk  (ImageNet mean)
    [0.75,  0.25,  0.25],   # 3 Solid lines
    [0.25,  0.75,  0.25],   # 4 Broken lines
], dtype=np.float32)

BEV_IDX_STREET      = 1
BEV_IDX_ALL_LINES   = 3
BEV_IDX_BROKEN_LINES = 4


def render_bev_sumo(engine, ego_vehicle, resolution=128, size_meters=64.0,
                    device="cpu"):
    """
    Render a Plant2-style semantic BEV for a SUMO (EdgeRoadNetwork) map.

    Returns torch.Tensor of shape (1, 3, resolution, resolution).
    """
    from metadrive.constants import PGLineType

    road_network = getattr(engine.current_map, "road_network", None)
    if road_network is None:
        return torch.zeros(1, 3, resolution, resolution,
                           dtype=torch.float32, device=device)

    scale = resolution / size_meters
    half = size_meters / 2
    ego_pos = np.array(ego_vehicle.position[:2])
    ego_heading = float(ego_vehicle.heading_theta)
    c_neg, s_neg = np.cos(-ego_heading), np.sin(-ego_heading)

    sem_map = np.zeros((resolution, resolution), dtype=np.uint8)

    for lane_id, lane_info in road_network.graph.items():
        lane = lane_info.lane
        n_pts = max(50, int(lane.length))

        for i in range(n_pts):
            s = lane.length * i / max(n_pts - 1, 1)
            pt = lane.position(s, 0)
            dx, dy = pt[0] - ego_pos[0], pt[1] - ego_pos[1]
            ex = dx * c_neg - dy * s_neg
            ey = dx * s_neg + dy * c_neg
            if abs(ex) > half or abs(ey) > half:
                continue
            px = resolution // 2 - int(ey * scale)
            py = resolution // 2 - int(ex * scale)
            w = max(1, int(lane.width_at(s) / 2 * scale))
            for dw in range(-w, w + 1):
                for dh in range(-w, w + 1):
                    nx, ny = px + dw, py + dh
                    if 0 <= nx < resolution and 0 <= ny < resolution:
                        sem_map[ny, nx] = BEV_IDX_STREET

        line_types = getattr(lane, "line_types", None)
        if line_types is None:
            continue
        for side in range(2):
            if side >= len(line_types):
                break
            lt = line_types[side]
            idx = (BEV_IDX_ALL_LINES if lt == PGLineType.CONTINUOUS
                   else BEV_IDX_BROKEN_LINES)
            for i in range(n_pts):
                s = lane.length * i / max(n_pts - 1, 1)
                lat = (side - 0.5) * lane.width_at(s)
                pt = lane.position(s, lat)
                dx, dy = pt[0] - ego_pos[0], pt[1] - ego_pos[1]
                ex = dx * c_neg - dy * s_neg
                ey = dx * s_neg + dy * c_neg
                if abs(ex) > half or abs(ey) > half:
                    continue
                px = resolution // 2 - int(ey * scale)
                py = resolution // 2 - int(ex * scale)
                if 0 <= px < resolution and 0 <= py < resolution:
                    sem_map[py, px] = idx

    rgb = BEV_COLORS[sem_map]
    bev_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return bev_t


# ---------------------------------------------------------------------------
# SUMO-compatible Plant2 batch builder
# ---------------------------------------------------------------------------
# Reuses route / object / speed-limit logic from metadrive_obs_to_plant2
# but calls render_bev_sumo instead of the broken render_bev_plant2.
#
# Object search defaults match metadrive.policy.plant_policy.PlanTPolicy.act
# and metadrive_obs_to_plant2_batch.

PLANT2_MAX_DISTANCE = 50.0
PLANT2_RANGE_FACTOR_FRONT = 2.0

_SPEED_BUCKETS_KMH = [50, 80, 100, 120]


def _get_lane_speed_limit_kmh(engine, ego_vehicle):
    """Read the speed limit from the SUMO lane the ego is currently on.

    SUMO stores lane speed in m/s.  We convert to km/h and snap to the
    nearest PlanT speed bucket (50/80/100/120).  Returns None if the
    speed cannot be determined (falls back to default 80 km/h inside
    get_speed_limit_idx).
    """
    road_network = getattr(engine.current_map, "road_network", None)
    if road_network is None:
        return None

    lane = getattr(ego_vehicle, "lane", None)
    if lane is None:
        return None

    lane_index = getattr(lane, "index", None)
    if lane_index is None:
        return None

    lane_info = road_network.graph.get(lane_index)
    if lane_info is None:
        return None

    speed_mps = lane_info.speed if isinstance(lane_info.speed, (int, float)) else None
    if speed_mps is None or speed_mps <= 0:
        return None

    speed_kmh = speed_mps * 3.6
    closest = min(_SPEED_BUCKETS_KMH, key=lambda b: abs(b - speed_kmh))
    return closest


def build_plant2_batch_sumo(
    engine, ego_vehicle,
    max_objects=30,
    max_distance=PLANT2_MAX_DISTANCE,
    range_factor_front=PLANT2_RANGE_FACTOR_FRONT,
    bev_resolution=128, bev_size_meters=64.0,
    device="cpu",
):
    from metadrive.policy.plant_policy import get_route_points_ego_frame
    from metadrive.policy.metadrive_obs_to_plant2 import (
        collect_objects_ego_frame,
        objects_to_x_batch,
        get_speed_limit_idx,
    )

    num_route_points = 20
    route_ego, _ = get_route_points_ego_frame(ego_vehicle, num_route_points)
    route_ego = np.asarray(route_ego, dtype=np.float32)
    if route_ego.shape[0] < num_route_points:
        pad = np.tile(route_ego[-1],
                      (num_route_points - route_ego.shape[0], 1))
        route_ego = np.vstack([route_ego, pad])
    route_ego = route_ego[:num_route_points]

    objects = collect_objects_ego_frame(
        engine, ego_vehicle,
        max_objects=max_objects,
        max_distance=max_distance,
        range_factor_front=range_factor_front,
    )
    x_list, num_objs = objects_to_x_batch(objects, max_objects)

    pool_size = max_objects + 1
    if len(x_list) > pool_size:
        x_list = x_list[:pool_size]
        num_objs = min(num_objs, max_objects)
    elif len(x_list) < pool_size:
        x_list += [[0.0] * 7] * (pool_size - len(x_list))

    x_batch_objs = torch.tensor(x_list, dtype=torch.float32, device=device)
    batch_idxs = torch.zeros((1, max_objects), dtype=torch.int32, device=device)
    if num_objs > 0:
        batch_idxs[0, :num_objs] = torch.arange(
            1, 1 + num_objs, dtype=torch.int32, device=device
        )

    route_t = torch.tensor(route_ego, dtype=torch.float32,
                           device=device).unsqueeze(0)

    speed_limit_kmh = _get_lane_speed_limit_kmh(engine, ego_vehicle)
    speed_limit_idx = get_speed_limit_idx(speed_limit_kmh)
    speed_limit = torch.tensor(
        [min(3, max(0, int(speed_limit_idx)))],
        dtype=torch.long, device=device,
    )

    bev_t = render_bev_sumo(
        engine, ego_vehicle,
        resolution=bev_resolution,
        size_meters=bev_size_meters,
        device=device,
    )
    bev_t = torch.rot90(bev_t, k=-1, dims=[2, 3])

    batch = {
        "idxs": batch_idxs,
        "x_objs": x_batch_objs,
        "route_original": route_t,
        "speed_limit": speed_limit,
        "y_objs": None,
        "BEV": bev_t,
    }
    return batch


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(map_path: str, sign_type: str, sign_spawn_distance: float,
              road_id: str, traffic_density: float):
    from envs.sumo_env_v2 import SumoEnvV2
    from envs.sumo_traffic_manager import SumoTrafficManager

    class _EnvWithTraffic(SumoEnvV2):
        def setup_engine(self):
            super().setup_engine()
            if self.config["traffic_density"] > 0:
                self.engine.update_manager(
                    "traffic_manager", SumoTrafficManager()
                )

    env = _EnvWithTraffic(dict(
        use_render=False,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.CRITICAL,
        map_name=map_path,
        sign_type=sign_type,
        sign_spawn_distance=sign_spawn_distance,
        traffic_density=traffic_density,
        vehicle_config={"spawn_lane_index": road_id},
    ))
    return env


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def _find_crash_objects(engine, ego):
    """Return list of (position, width, length, heading) for objects ego collided with."""
    from metadrive.type import MetaDriveType

    ego_pos = np.array(ego.position[:2], dtype=np.float64)
    crashed = []

    for obj_id, obj in engine.get_objects().items():
        if obj is ego:
            continue
        if not hasattr(obj, "position"):
            continue
        obj_pos = np.array(obj.position[:2], dtype=np.float64)
        dist = np.linalg.norm(obj_pos - ego_pos)
        if dist > 15.0:
            continue

        is_crash_candidate = False
        if hasattr(obj, "crashed") and obj.crashed:
            is_crash_candidate = True
        obj_name = getattr(obj, "name", "")
        obj_type = getattr(obj, "type", "")
        if ego.crash_vehicle and (
            MetaDriveType.VEHICLE in str(obj_type)
            or MetaDriveType.VEHICLE in str(obj_name)
        ):
            is_crash_candidate = True
        if ego.crash_object and MetaDriveType.is_traffic_object(str(obj_name)):
            is_crash_candidate = True
        if ego.crash_building and MetaDriveType.BUILDING in str(obj_name):
            is_crash_candidate = True

        if not is_crash_candidate and dist < 5.0:
            is_crash_candidate = True

        if is_crash_candidate:
            w = getattr(obj, "top_down_width", getattr(obj, "WIDTH", 2.0))
            l = getattr(obj, "top_down_length", getattr(obj, "LENGTH", 4.5))
            h = getattr(obj, "heading_theta", 0.0)
            crashed.append((obj_pos, float(w), float(l), float(h)))

    return crashed


def _draw_crash_highlight(frame, ego_pos, ego_heading, crash_objects, scaling,
                          screen_size):
    """Draw red rectangles around crash objects on the rendered numpy frame.

    With target_agent_heading_up=True, ego is at the center of the frame and
    the frame is rotated so that ego faces up (-Y in pixel space).
    """
    cx, cy = screen_size[0] // 2, screen_size[1] // 2

    for obj_pos, obj_w, obj_l, obj_heading in crash_objects:
        dx = obj_pos[0] - ego_pos[0]
        dy = obj_pos[1] - ego_pos[1]

        c, s = np.cos(-ego_heading), np.sin(-ego_heading)
        ex = dx * c - dy * s
        ey = dx * s + dy * c

        # Heading-up: forward ex -> up (-py); physical right (+ey) -> screen +x (cv2 x right).
        px = int(cx + ey * scaling)
        py = int(cy - ex * scaling)

        half_w = max(int(obj_w / 2 * scaling), 6)
        half_l = max(int(obj_l / 2 * scaling), 6)
        radius = max(half_w, half_l) + 4

        rel_heading = obj_heading - ego_heading
        cos_r, sin_r = np.cos(-rel_heading), np.sin(-rel_heading)
        corners_local = np.array([
            [-half_l, -half_w],
            [-half_l,  half_w],
            [ half_l,  half_w],
            [ half_l, -half_w],
        ], dtype=np.float32)
        rotated = np.stack([
            corners_local[:, 0] * cos_r - corners_local[:, 1] * sin_r,
            corners_local[:, 0] * sin_r + corners_local[:, 1] * cos_r,
        ], axis=1)
        pts = (rotated + np.array([[px, py]], dtype=np.float32)).astype(np.int32)

        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255),
                      thickness=3)
        cv2.circle(frame, (px, py), radius, (0, 0, 255), 2)


def _pred_path_to_pixels(
    pred_path_xy: np.ndarray,
    scaling: float,
    screen_size,
) -> np.ndarray | None:
    """Map PlanT ego-frame path points to top-down pixel coordinates.

    ``pred_path_xy`` is (N, 2) with x = forward, y = right (same convention as
    ``route_original`` / :func:`get_route_points_ego_frame`).

    Same ex/ey → pixel mapping as :func:`_draw_crash_highlight` for
    ``target_agent_heading_up=True``: ego at screen center, forward ex →
    ``py = cy - ex * scaling``, lateral ey (PlanT / route_original: + = right) →
    ``px = cx + ey * scaling``.
    """
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
    pred_path_xy: np.ndarray | None,
    scaling: float,
    screen_size,
    color=(0, 0, 255),
    thickness=2,
):
    """Overlay ``pred_path`` (ego frame) on a top-down RGB frame."""
    pix = _pred_path_to_pixels(pred_path_xy, scaling, screen_size)
    if pix is None or len(pix) < 2:
        return
    cv2.polylines(
        frame, [pix], isClosed=False, color=color, thickness=thickness,
    )
    for j in (0, len(pix) - 1):
        cv2.circle(frame, (int(pix[j, 0]), int(pix[j, 1])), 3, color, -1)


def _extract_pred_path_numpy(pred_plan) -> np.ndarray | None:
    """Return ``pred_path`` as (N, 2) float32 numpy, or None."""
    pred_path_t = pred_plan[0]
    if pred_path_t is None:
        return None
    pp = pred_path_t.detach().cpu().numpy()
    if pp.ndim > 2:
        pp = pp.squeeze(0)
    if pp.size == 0 or pp.shape[-1] < 2:
        return None
    return pp.astype(np.float32)


def run_episode(
    env,
    net: torch.nn.Module,
    device: str,
    seed: int,
    max_steps: int = 1500,
    gif_path: str | None = None,
) -> Dict[str, Any]:
    """Caller must ``np.random.seed`` / ``random.seed`` before ``_make_env`` (same as v1)."""
    from carla_garage.plant2_control import (
        plant2_predictions_to_action,
        get_target_speed_from_limit,
    )

    RENDER_SCALING = 40
    SCREEN_SIZE = (600, 600)

    obs, info = env.reset()
    base_env = env.unwrapped

    save_gif = gif_path is not None
    if save_gif:
        renderer = getattr(base_env, "top_down_renderer", None)
        if renderer is not None and hasattr(renderer, "_screen_frames"):
            renderer._screen_frames.clear()

    total_reward = 0.0
    violations = 0
    outcome = "timeout"
    n_steps = 0

    for step_i in range(max_steps):
        ego = getattr(base_env, "agent", None) or getattr(base_env, "vehicle", None)
        if ego is None:
            break

        batch = build_plant2_batch_sumo(
            base_env.engine, ego,
            max_objects=30,
            max_distance=PLANT2_MAX_DISTANCE,
            range_factor_front=PLANT2_RANGE_FACTOR_FRONT,
            bev_resolution=128,
            bev_size_meters=64.0,
            device=device,
        )

        with torch.no_grad():
            _, _, pred_plan, _ = net(batch)

        pred_path_viz = _extract_pred_path_numpy(pred_plan)

        ego_speed = float(getattr(ego, "speed", 0.0))
        speed_limit_idx = int(batch["speed_limit"][0].item())
        target_speed = get_target_speed_from_limit(speed_limit_idx)

        action, _ = plant2_predictions_to_action(
            pred_plan,
            current_speed=ego_speed,
            target_speed_mps=target_speed,
            speed_limit_idx=speed_limit_idx,
            device=device,
            return_waypoints=True,
        )
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        n_steps = step_i + 1

        vehicle = base_env.agent
        sign_mgr = base_env.engine.traffic_sign_manager
        for _, violated in sign_mgr.check_all_violations(vehicle):
            if violated:
                violations += 1

        is_crash = (
            info.get("crash", False)
            or info.get("crash_vehicle", False)
            or vehicle.crash_vehicle
            or vehicle.crash_object
            or vehicle.crash_building
        )

        if save_gif:
            text_dict = {
                "Step": step_i,
                "Speed": f"{vehicle.speed_km_h:.1f} km/h",
            }
            if is_crash:
                text_dict["CRASH"] = "!"
            base_env.render(
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
            if renderer is not None and getattr(renderer, "_screen_frames", None):
                frame = renderer._screen_frames[-1]
                _draw_pred_path_on_frame(
                    frame, pred_path_viz, RENDER_SCALING, SCREEN_SIZE,
                )

        if terminated or truncated:
            if info.get("arrive_dest", False):
                outcome = "success"
            elif is_crash or info.get("out_of_road", False):
                outcome = "crash"
            elif info.get("max_step", False):
                outcome = "timeout"
            else:
                outcome = "truncated"
            break

    if save_gif:
        renderer = getattr(base_env, "top_down_renderer", None)
        if renderer is not None:
            renderer.generate_gif(gif_path, duration=10)
            if hasattr(renderer, "clear"):
                renderer.clear()
            elif hasattr(renderer, "_screen_frames"):
                renderer._screen_frames.clear()

    return {
        "steps": n_steps,
        "total_reward": total_reward,
        "violations": violations,
        "outcome": outcome,
        "success": outcome == "success",
        "crashed": outcome == "crash",
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    header = ["scene", "sign", "seed", "steps", "reward", "viol", "outcome"]
    col_w = [18, 6, 10, 6, 9, 5, 9]
    fmt_h = "  ".join(f"{h:<{w}}" for h, w in zip(header, col_w))
    sep = "-" * len(fmt_h)
    print(f"\n{sep}\n{fmt_h}\n{sep}")
    for r in rows:
        vals = [
            r["scene"][:18],
            r["sign_type"],
            r.get("seed", ""),
            r["steps"],
            f"{r['total_reward']:.2f}",
            r["violations"],
            r["outcome"],
        ]
        print("  ".join(f"{str(v):<{w}}" for v, w in zip(vals, col_w)))
    print(sep)

    n = len(rows)
    n_s = sum(r["outcome"] == "success" for r in rows)
    n_c = sum(r["outcome"] == "crash" for r in rows)
    n_t = sum(r["outcome"] == "timeout" for r in rows)
    print(f"Total     : {n}")
    print(f"Success   : {n_s}/{n}  ({100 * n_s / n:.1f}%)")
    print(f"Crash     : {n_c}/{n}  ({100 * n_c / n:.1f}%)")
    print(f"Timeout   : {n_t}/{n}  ({100 * n_t / n:.1f}%)")
    print(f"Mean reward: {np.mean([r['total_reward'] for r in rows]):.2f}")
    print(f"Mean steps : {np.mean([r['steps'] for r in rows]):.1f}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plant2 closed-loop benchmark on SUMO scenes (v2)."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to PlanT checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--scenes-dir", type=str, required=True,
        help="Root scenes directory (contains <sign_type>/<scene>/meta.json)",
    )
    parser.add_argument("--sign-type", type=str, default=None,
                        help="Run only this sign type (e.g. '3.27')")
    parser.add_argument(
        "--seeds", type=str, default="",
        help="Comma-separated integers (same as run_benchmark.py): only scenes "
        "whose meta.json sign_id is in this list; each run uses every listed "
        "seed for np.random/random before env.reset(). If empty, run all scenes "
        "once each with seed=sign_id.",
    )
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Max scenes per sign type")
    parser.add_argument("--max-steps", type=int, default=1500,
                        help="Max steps per episode")
    parser.add_argument("--traffic-density", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for CSV + JSON results")
    parser.add_argument("--run-name", type=str, default="plant2_v2")
    parser.add_argument("--no-gifs", action="store_true",
                        help="Skip GIF rendering (faster)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir or str(PDD_BENCH_DIR / "outputs" / "benchmark_v2")
    os.makedirs(out_dir, exist_ok=True)
    gif_dir = os.path.join(out_dir, "gifs")
    if not args.no_gifs:
        os.makedirs(gif_dir, exist_ok=True)

    scenes_root = Path(args.scenes_dir)
    if not scenes_root.exists():
        raise FileNotFoundError(f"Scenes dir not found: {scenes_root}")

    if args.sign_type:
        sign_types = [args.sign_type]
    else:
        sign_types = sorted(
            d.name for d in scenes_root.iterdir() if d.is_dir()
        )

    print("=" * 70)
    print("Plant2 Benchmark v2 — SUMO scenes")
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Scenes dir   : {scenes_root}")
    print(f"  Sign types   : {sign_types}")
    print(f"  Max scenes   : {args.max_scenes or 'all'}")
    print(f"  Seeds filter : {args.seeds.strip() or '(use each scene sign_id)'}")
    print(f"  Max steps    : {args.max_steps}")
    print(f"  Traffic dens.: {args.traffic_density}")
    print(f"  Device       : {device}")
    print(f"  Output dir   : {out_dir}")
    print(f"  GIFs         : {not args.no_gifs}")
    print("=" * 70)

    net, config_all = load_plant_model(
        args.checkpoint, str(PLANT_PLAN_T_DIR), device=device
    )

    csv_path = os.path.join(out_dir, f"results_{args.run_name}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "sign_type", "scene", "seed", "steps", "total_reward",
            "violations", "outcome", "success", "crashed",
        ],
    )
    csv_writer.writeheader()

    all_rows: List[Dict[str, Any]] = []

    # Same parsing as run_benchmark.py --seeds
    if args.seeds.strip():
        seeds_filter: List[int] | None = [
            int(s) for s in args.seeds.split(",")
        ]
    else:
        seeds_filter = None

    for sign_type in sign_types:
        sign_dir = scenes_root / sign_type
        if not sign_dir.exists():
            print(f"[SKIP] sign type '{sign_type}': directory not found")
            continue

        scene_dirs = sorted(d for d in sign_dir.iterdir() if d.is_dir())
        if args.max_scenes:
            scene_dirs = scene_dirs[: args.max_scenes]
        if not scene_dirs:
            print(f"[SKIP] sign type '{sign_type}': no scenes")
            continue

        print(f"\n--- sign type {sign_type}: {len(scene_dirs)} scene(s) ---")

        for scene_dir in scene_dirs:
            meta_path = scene_dir / "meta.json"
            if not meta_path.exists():
                print(f"  [SKIP] {scene_dir.name}: no meta.json")
                continue

            with open(meta_path, "r") as f:
                meta = json.load(f)

            if "sign_id" not in meta:
                print(f"  [SKIP] {scene_dir.name}: meta.json has no sign_id")
                continue
            sign_id = int(meta["sign_id"])

            scene_seeds = seeds_filter if seeds_filter is not None else [sign_id]
            if sign_id not in scene_seeds:
                continue

            net_file = meta["net_file"]
            map_path = str((scene_dir / net_file).resolve())
            road_id = meta["road_id"]
            sign_spawn_distance = meta["distance_from_start"]

            use_seed_suffix = seeds_filter is not None or len(scene_seeds) > 1

            for seed in scene_seeds:
                # Same RNG order as run_benchmark.run_single_episode: seed before env + reset.
                np.random.seed(seed)
                random.seed(seed)

                if not args.no_gifs:
                    if use_seed_suffix:
                        scene_gif = os.path.join(
                            gif_dir,
                            f"{sign_type}_{scene_dir.name}_seed{seed}.gif",
                        )
                    else:
                        scene_gif = os.path.join(
                            gif_dir, f"{sign_type}_{scene_dir.name}.gif",
                        )
                else:
                    scene_gif = None

                print(
                    f"  Scene {scene_dir.name}  seed={seed} ...",
                    end=" ",
                    flush=True,
                )

                env = _make_env(
                    map_path=map_path,
                    sign_type=sign_type,
                    sign_spawn_distance=sign_spawn_distance,
                    road_id=road_id,
                    traffic_density=args.traffic_density,
                )

                result = run_episode(
                    env,
                    net,
                    device=device,
                    seed=seed,
                    max_steps=args.max_steps,
                    gif_path=scene_gif,
                )
                env.close()

                result["sign_type"] = sign_type
                result["scene"] = scene_dir.name

                print(
                    f"{result['outcome']}  steps={result['steps']}  "
                    f"reward={result['total_reward']:.2f}  "
                    f"violations={result['violations']}"
                )

                all_rows.append(result)
                csv_writer.writerow(result)
                csv_file.flush()

    csv_file.close()

    by_sign: Dict[str, List] = defaultdict(list)
    for r in all_rows:
        by_sign[r["sign_type"]].append(r)

    summary_all = {}
    for st, runs in by_sign.items():
        valid = [r for r in runs if r["outcome"] != "error"]
        if valid:
            metrics = {
                "run_name": args.run_name,
                "total_runs": len(valid),
                "success_rate": float(np.mean([r["success"] for r in valid])),
                "crash_rate": float(np.mean([r["crashed"] for r in valid])),
                "average_violations": float(
                    np.mean([r["violations"] for r in valid])
                ),
                "average_reward": float(
                    np.mean([r["total_reward"] for r in valid])
                ),
            }
        else:
            metrics = {
                "run_name": args.run_name,
                "total_runs": 0,
                "success_rate": 0.0,
                "crash_rate": 0.0,
                "average_violations": 0.0,
                "average_reward": 0.0,
            }
        summary_all[st] = metrics

        out_json = os.path.join(out_dir, f"results_{st}_{args.run_name}.json")
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)

    summary_path = os.path.join(out_dir, f"summary_{args.run_name}.json")
    with open(summary_path, "w") as f:
        json.dump(summary_all, f, indent=2)

    _print_table(all_rows)

    print(f"\nCSV     : {csv_path}")
    print(f"Summary : {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
