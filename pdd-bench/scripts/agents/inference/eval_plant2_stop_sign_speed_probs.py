#!/usr/bin/env python3
"""
Diagnostic: run PlanT/HFLM closed-loop and, at every step where a stop-sign
(object type 4) is visible in x_objs, print the ego-speed classifier
probability distribution.

Usage
-----
python eval_plant2_stop_sign_speed_probs.py \\
    --checkpoint_file /path/to/plant2.ckpt \\
    --num-episodes 3 \\
    --start-seed 1000
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

def _find_sdc_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for parent in (current, *current.parents):
        if (parent / "pdd-bench").is_dir() and (parent / "metadrive").is_dir():
            return parent
    raise RuntimeError("Could not locate SDC root (expected pdd-bench and metadrive)")


FILE_PATH = Path(__file__).resolve()
SDC_ROOT = _find_sdc_root(FILE_PATH)
PDD_BENCH_DIR = SDC_ROOT / "pdd-bench"
METADRIVE_DIR = SDC_ROOT / "metadrive"
PLANT2_DIR = SDC_ROOT / "plant2"
PLANT_PLAN_T_DIR = PLANT2_DIR / "PlanT"
ADAPTER_PATH = PDD_BENCH_DIR / "agents" / "carl_in_metadrive"

for _p in (SDC_ROOT, METADRIVE_DIR, PDD_BENCH_DIR, PLANT_PLAN_T_DIR, PLANT2_DIR, ADAPTER_PATH):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

if os.environ.get("SDL_VIDEODRIVER") is None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Speed bin centres (same as train_plant2_from_carl_trajectories.py)
SPEED_BINS = np.array(
    [0, 0.025, 0.05472609, 1.0, 1.5, 2.0, 4.0, 8.0, 10.0, 20.0],
    dtype=np.float32,
)

OBJ_TYPE_STOP_SIGN = 4


# ---------------------------------------------------------------------------
# Mock + model loading (identical to eval_plant2_new_envs.py)
# ---------------------------------------------------------------------------

def _mock_carla_modules():
    import unittest.mock as _mock
    for mod_name in ("carla", "agents", "agents.navigation",
                     "agents.navigation.global_route_planner"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _mock.MagicMock()


def load_plant_model(checkpoint_path: str, device: str = "cpu"):
    import yaml
    _mock_carla_modules()

    model_yaml = os.path.join(str(PLANT_PLAN_T_DIR), "config", "model", "PlanT.yaml")
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
    plant_path = str(PLANT_PLAN_T_DIR)
    if plant_path not in sys.path:
        sys.path.insert(0, plant_path)
    elif sys.path[0] != plant_path:
        sys.path.remove(plant_path)
        sys.path.insert(0, plant_path)

    from model import HFLM  # type: ignore

    net = HFLM(config_all.model.network, config_all)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint contains no state_dict")
    if list(sd.keys())[0].startswith("model."):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=False)
    net = net.to(device)
    net.eval()
    return net, config_all


# ---------------------------------------------------------------------------
# Environment (copy of create_env from eval_plant2_new_envs.py)
# ---------------------------------------------------------------------------

import gymnasium as gym


class StopSignSpawnWrapper(gym.Wrapper):
    def __init__(self, env, stop_sign_probability=0.3, max_signs=10, min_distance=25.0):
        super().__init__(env)
        self.stop_sign_probability = stop_sign_probability
        self.max_signs = max_signs
        self.min_distance = min_distance

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._spawn_signs(seed=kwargs.get("seed"))
        return obs, info

    def _spawn_signs(self, seed=None):
        from traffic_signs.stop_sign import StopSign
        base = self.unwrapped
        sign_mgr = getattr(base.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return
        road_network = base.current_map.road_network
        lanes = [
            lane
            for roads in road_network.graph.values()
            for ends in roads.values()
            for lane in ends
            if lane.length >= 15.0
        ]
        rng = np.random.default_rng(seed)
        rng.shuffle(lanes)
        added_pos, n_added = [], 0
        for lane in lanes:
            if n_added >= self.max_signs:
                break
            if rng.random() >= self.stop_sign_probability:
                continue
            sign = sign_mgr.add_sign(StopSign, lane=lane, use_random_lane=False)
            pos = sign.position
            if any(np.hypot(pos[0] - ep[0], pos[1] - ep[1]) < self.min_distance for ep in added_pos):
                sign_mgr._cleanup_sign_object(sign)
                if sign in sign_mgr.signs:
                    sign_mgr.signs.remove(sign)
                continue
            added_pos.append(pos)
            n_added += 1


def create_env(seed=1000, traffic_density=0.1, horizon=500,
               stop_sign_probability=0.3, max_stop_signs=10, num_scenarios=200):
    from envs.traffic_sign_env import TrafficSignEnv
    config = dict(
        num_scenarios=num_scenarios, start_seed=seed, log_level=50,
        use_render=False, random_lane_width=True, random_lane_num=True,
        traffic_density=traffic_density, horizon=horizon, use_lateral_reward=True,
        success_reward=20.0, out_of_road_penalty=15.0, crash_vehicle_penalty=25.0,
        crash_object_penalty=20.0, crash_sidewalk_penalty=5.0,
        driving_reward=0.5, speed_reward=0.05,
    )
    env = TrafficSignEnv(config)
    env = StopSignSpawnWrapper(env, stop_sign_probability=stop_sign_probability,
                               max_signs=max_stop_signs)
    return env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stop_signs_in_batch(batch: Dict[str, Any]) -> bool:
    """Return True if x_objs contains at least one object of type 4 (stop sign)."""
    x_objs = batch["x_objs"]       # (pool_size, 7) — index 0 is the object type
    types = x_objs[..., 0]          # works for both 2-D and 3-D pools
    return bool((types == OBJ_TYPE_STOP_SIGN).any().item())


def _nearest_stop_sign_dist(batch: Dict[str, Any]) -> float:
    """Return ego-frame distance (m) to the nearest stop sign in x_objs, or inf."""
    x_objs = batch["x_objs"]
    types = x_objs[..., 0]
    mask = types == OBJ_TYPE_STOP_SIGN
    if not mask.any():
        return float("inf")
    sign_xy = x_objs[..., 1:3][mask]   # (N, 2)  x=forward, y=right
    dists = (sign_xy ** 2).sum(dim=-1).sqrt()
    return float(dists.min().item())


def _speed_probs(pred_speed: torch.Tensor) -> np.ndarray:
    """
    Convert ego-speed logits → probability vector.
    pred_speed shape: (1, C) or (C,)
    """
    logits = pred_speed.detach().float()
    if logits.dim() > 1:
        logits = logits.squeeze(0)      # (C,)
    return torch.softmax(logits, dim=0).cpu().numpy()


def _format_probs(probs: np.ndarray, bins: np.ndarray) -> str:
    """
    Return a compact one-liner with bin label + probability, e.g.:
    [0.05→0.72  1.84→0.18  3.37→0.06  ...]
    """
    parts = [f"{b:.2f}→{p:.3f}" for b, p in zip(bins, probs)]
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Print speed-classifier probs whenever a stop sign (type 4) is in x_objs."
    )
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--traffic-density", type=float, default=0.1)
    parser.add_argument("--stop-sign-probability", type=float, default=0.5,
                        help="Higher than default so stop signs appear more often")
    parser.add_argument("--num-scenarios", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--min-dist", type=float, default=float("inf"),
                        help="Only print when nearest stop sign is within this many metres")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save per-episode GIFs (default: outputs/eval_stop_sign_speed_probs)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir or str(PDD_BENCH_DIR / "outputs" / "eval_stop_sign_speed_probs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Stop-sign speed-prob diagnostic")
    print(f"  Checkpoint : {args.checkpoint_file}")
    print(f"  Episodes   : {args.num_episodes}  start_seed={args.start_seed}")
    print(f"  Device     : {device}")
    print(f"  Dist filter: {args.min_dist} m")
    print(f"  GIF dir    : {out_dir}")
    print("=" * 70)

    net, config_all = load_plant_model(args.checkpoint_file, device=device)
    training_cfg    = getattr(config_all.model, "training", {}) or {}
    input_bev       = True
    input_ego_speed = bool(training_cfg.get("input_ego_speed", False))

    from metadrive.policy.metadrive_obs_to_plant2 import metadrive_obs_to_plant2_batch
    from carla_garage.plant2_control import plant2_predictions_to_action, get_target_speed_from_limit

    env      = create_env(
        seed=args.start_seed,
        traffic_density=args.traffic_density,
        horizon=args.max_steps,
        stop_sign_probability=args.stop_sign_probability,
        num_scenarios=args.num_scenarios,
    )
    base_env = env.unwrapped

    total_stop_sign_steps = 0

    speed_dist = defaultdict(list)
    meters_to_speed = defaultdict(list)

    for ep in range(args.num_episodes):
        ep_seed = args.start_seed + ep
        print(f"\n{'─'*70}")
        print(f"Episode {ep + 1}/{args.num_episodes}  seed={ep_seed}")
        print(f"{'─'*70}")

        env.reset(seed=ep_seed)
        ep_stop_steps = 0

        # Clear renderer frames for this episode
        if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
            if hasattr(base_env.top_down_renderer, "_screen_frames"):
                base_env.top_down_renderer._screen_frames.clear()

        for step_i in range(args.max_steps):
            ego = getattr(base_env, "agent", None) or getattr(base_env, "vehicle", None)
            if ego is None:
                break

            batch = metadrive_obs_to_plant2_batch(
                base_env.engine,
                ego,
                route_ego_20x2=None,
                speed_limit_kmh=None,
                max_objects=30,
                max_distance=75.0,
                range_factor_front=16.0,
                input_bev=input_bev,
                input_ego_speed=input_ego_speed,
                bev_resolution=128,
                bev_size_meters=64.0,
                device=device,
            )

            with torch.no_grad():
                _, _, pred_plan, _ = net(batch)

            pred_path, pred_wps, pred_speed = pred_plan

            # Check for stop signs in the current observation
            has_stop_sign = _stop_signs_in_batch(batch)
            dist_to_sign  = _nearest_stop_sign_dist(batch)

            if has_stop_sign and dist_to_sign <= args.min_dist:
                ep_stop_steps += 1
                total_stop_sign_steps += 1

                ego_speed = float(getattr(ego, "speed", 0.0))

                if pred_speed is not None:
                    probs = _speed_probs(pred_speed)
                    expected_speed = float((probs * SPEED_BINS).sum())
                    top_bin = int(np.argmax(probs))
                    speed_dist[top_bin].append(dist_to_sign)
                    meters_to_speed[dist_to_sign].append(ego_speed)
                    print(
                        f"  step={step_i:4d} | ego_spd={ego_speed:.2f} m/s"
                        f" | sign_dist={dist_to_sign:.1f} m"
                        f" | top_bin={top_bin} ({SPEED_BINS[top_bin]:.2f} m/s)"
                        f" | E[speed]={expected_speed:.3f} m/s"
                    )
                    print(f"           probs: {_format_probs(probs, SPEED_BINS)}")
                else:
                    print(f"  step={step_i:4d} | stop sign visible but pred_speed is None")

            # Render top-down frame before stepping
            ego_pos_xy  = np.asarray(ego.position, dtype=np.float32)[:2]
            ego_heading = float(ego.heading_theta)
            pred_coords = pred_wps if pred_wps is not None else pred_path
            render_kw: Dict[str, Any] = dict(
                mode="top_down",
                screen_record=True,
                window=False,
                screen_size=(640, 640),
                overlay_ego_pos_world=ego_pos_xy,
                overlay_ego_heading_rad=ego_heading,
            )
            if pred_coords is not None:
                pred_ego = pred_coords.squeeze(0).detach().cpu().numpy().astype(np.float32)
                c, s = np.cos(ego_heading), np.sin(ego_heading)
                dx = pred_ego[:, 0] * c - (-pred_ego[:, 1]) * s
                dy = pred_ego[:, 0] * s + (-pred_ego[:, 1]) * c
                render_kw["overlay_pred_traj_world"] = np.stack([dx, dy], axis=-1) + ego_pos_xy
            base_env.render(**render_kw)

            # Step environment
            speed_limit_idx = int(batch["speed_limit"][0].item())
            target_speed    = get_target_speed_from_limit(speed_limit_idx)
            action_np, _    = plant2_predictions_to_action(
                pred_plan,
                current_speed=float(getattr(ego, "speed", 0.0)),
                target_speed_mps=target_speed,
                speed_limit_idx=speed_limit_idx,
                device=device,
                return_waypoints=True,
            )
            action_np = np.clip(np.asarray(action_np, dtype=np.float32), -1.0, 1.0)
            _, _, terminated, truncated, _ = env.step(action_np)

            if terminated or truncated:
                break

        # Save GIF for this episode
        gif_path = os.path.join(out_dir, f"ep{ep:03d}_seed{ep_seed}.gif")
        try:
            if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
                base_env.top_down_renderer.generate_gif(gif_path, duration=10)
                print(f"  [GIF] {gif_path}")
                if hasattr(base_env.top_down_renderer, "clear"):
                    base_env.top_down_renderer.clear()
                elif hasattr(base_env.top_down_renderer, "_screen_frames"):
                    base_env.top_down_renderer._screen_frames.clear()
        except Exception as exc:
            print(f"  [WARN] GIF failed: {exc}")

        print(f"  → {ep_stop_steps} steps with visible stop sign this episode")

    env.close()
    print(f"\n{'='*70}")
    print(f"Total steps with stop sign visible: {total_stop_sign_steps}")
    print("Done.")

    plt.figure(figsize=(10, 5))
    average_dists = []
    for speed_bin in SPEED_BINS:
        average_dists.append(np.mean(speed_dist.get(speed_bin, [0])))
    plt.bar(range(len(SPEED_BINS)), average_dists)
    plt.xticks(range(len(SPEED_BINS)), [f"{bin:.2f} m/s" for bin in SPEED_BINS])
    plt.ylabel("Average distance to stop sign (m)")
    plt.title("Average distance to stop sign at each speed bin")
    plt.savefig(os.path.join(out_dir, "average_dist_to_stop_sign.png"))

    plt.figure(figsize=(10, 5))
    for dist, speeds in meters_to_speed.items():
        plt.scatter(dist, np.mean(speeds))
    plt.xlabel("Distance to stop sign (m)")
    plt.ylabel("Average ego speed (m/s)")
    plt.title("Average ego speed vs. distance to stop sign")
    plt.savefig(os.path.join(out_dir, "average_ego_speed_vs_dist_to_stop_sign.png"))


if __name__ == "__main__":
    main()
