#!/usr/bin/env python3
"""
Closed-loop PlanT evaluation using **live** MetaDrive observations, with **seeds taken from .pt trajectories**.

This mirrors `train_metadrive_ppo_plant_stop_gifs.py` (model ← `metadrive_obs_to_plant2_batch` each step,
`plant2_predictions_to_action` → `env.step`), but for **each episode** uses `base_seed` / `reset_seed`
stored in files from `collect_metadrive_carl_plant2_trajectories.py` so maps and stop-sign placement match
collection — without feeding the saved `plant2_batch`.

Contrast: `eval_plant2_from_carl_trajectories_gifs.py` replays logged `action_env` and runs the model on
**saved** batches for visualization only.

Usage:
  python eval_plant2_live_seeds_from_trajectories_gifs.py \\
      --checkpoint_file /path/to.ckpt \\
      --data-dir /path/to/carl_plant2_trajectories \\
      --max_episodes 5
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


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

for _p in (SDC_ROOT, METADRIVE_DIR, PDD_BENCH_DIR, PLANT_PLAN_T_DIR, PLANT2_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)


def _mock_carla_modules():
    import unittest.mock as _mock
    for mod_name in ("carla", "agents", "agents.navigation",
                     "agents.navigation.global_route_planner"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _mock.MagicMock()


def load_plant_model(checkpoint_path: str, plant_planT_path: str, device: str = "cpu"):
    import importlib.util
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

    model_py = Path(plant_planT_path) / "model.py"
    if not model_py.exists():
        raise FileNotFoundError(f"PlanT model.py not found: {model_py}")
    spec = importlib.util.spec_from_file_location("plant2_plant_model", str(model_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {model_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    HFLM = getattr(mod, "HFLM", None)
    if HFLM is None:
        raise ImportError(f"HFLM not found in {model_py}")

    net = HFLM(config_net, config_all)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint contains no state_dict")
    if list(sd.keys())[0].startswith("model."):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=False)
    return net, config_all


def _ego_xy_to_world_xy(ego_pos_xy: np.ndarray, ego_heading: float, xy_ego: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy_ego, dtype=np.float32)
    x = xy[:, 0]
    y_right = xy[:, 1]
    c, s = float(np.cos(ego_heading)), float(np.sin(ego_heading))
    y_left = -y_right
    dx = x * c - y_left * s
    dy = x * s + y_left * c
    return np.stack([dx, dy], axis=-1) + np.asarray(ego_pos_xy, dtype=np.float32)[None, :]


def main():
    parser = argparse.ArgumentParser(
        description="PlanT closed-loop eval: live obs; episode seeds read from CaRL trajectory .pt files."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PDD_BENCH_DIR / "data"),
        help="Directory with .pt files (only base_seed / reset_seed / step count are required)",
    )
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--plant_planT_path", type=str, default=str(PLANT_PLAN_T_DIR))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument(
        "--cap-steps-to-recording",
        action="store_true",
        help="Each episode runs at most min(max_steps, len(steps)) from that .pt file",
    )
    parser.add_argument("--traffic-density", type=float, default=0.1)
    parser.add_argument("--stop-sign-probability", type=float, default=0.3)
    parser.add_argument("--num-scenarios", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--no-overlay-traj",
        action="store_true",
        help="Disable green/red trajectory overlays on top-down render",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir or str(PDD_BENCH_DIR / "outputs" / "plant2_eval_live_seeds_gifs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("PlanT LIVE inference — seeds from trajectory .pt files")
    print(f"  Data dir:       {args.data_dir}")
    print(f"  Checkpoint:     {args.checkpoint_file}")
    print(f"  Output dir:     {out_dir}")
    print(f"  Max episodes:   {args.max_episodes}")
    print(f"  Max steps/ep:   {args.max_steps}")
    print(f"  Cap to .pt len: {args.cap_steps_to_recording}")
    print(f"  Device:         {device}")
    print("=" * 70)

    net, config_all = load_plant_model(args.checkpoint_file, args.plant_planT_path, device=device)
    net = net.to(device)
    net.eval()

    from scripts.agents.train.train_metadrive_ppo_plant_stop_gifs import create_env  # type: ignore
    from metadrive.policy.metadrive_obs_to_plant2 import metadrive_obs_to_plant2_batch
    from carla_garage.plant2_control import plant2_predictions_to_action, get_target_speed_from_limit

    training_cfg = getattr(config_all.model, "training", {}) or {}
    input_bev = True
    input_ego_speed = bool(training_cfg.get("input_ego_speed", False))

    data_path = Path(args.data_dir)
    files = sorted(data_path.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {args.data_dir}")

    env = None
    base_env = None
    env_has_seed: Optional[int] = None
    episodes_done = 0

    for f in files:
        if episodes_done >= args.max_episodes:
            break

        ep: Dict[str, Any] = torch.load(f, weights_only=False, map_location="cpu")
        steps: List[Dict[str, Any]] = ep.get("steps", [])
        base_seed = int(ep.get("base_seed", 0))
        reset_seed = int(ep.get("reset_seed", base_seed + int(ep.get("episode_index", 0))))
        n_recorded = len(steps)

        if env is None or env_has_seed != base_seed:
            if env is not None:
                env.close()
            env = create_env(
                seed=base_seed,
                traffic_density=args.traffic_density,
                horizon=args.max_steps,
                stop_sign_probability=args.stop_sign_probability,
                num_scenarios=args.num_scenarios,
            )
            base_env = env.unwrapped
            env_has_seed = base_seed

        n_steps = args.max_steps
        if args.cap_steps_to_recording and n_recorded > 0:
            n_steps = min(n_steps, n_recorded)

        print(f"\n{f.name}  base_seed={base_seed} reset_seed={reset_seed} live_steps={n_steps} (recorded={n_recorded})")

        obs, info = env.reset(seed=reset_seed)
        if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
            if hasattr(base_env.top_down_renderer, "_screen_frames"):
                base_env.top_down_renderer._screen_frames.clear()

        ep_reward = 0.0
        for step_idx in range(n_steps):
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

            pred_path, pred_wps, _ = pred_plan
            pred_coords = pred_wps if pred_wps is not None else pred_path

            ego_speed = float(getattr(ego, "speed", 0.0))
            speed_limit_idx = int(batch["speed_limit"][0].item())
            target_speed_mps = get_target_speed_from_limit(speed_limit_idx)

            action = plant2_predictions_to_action(
                pred_plan,
                current_speed=ego_speed,
                target_speed_mps=target_speed_mps,
                speed_limit_idx=speed_limit_idx,
                speed_limits_kmh=(50, 80, 100, 120),
                device=device,
                return_waypoints=False,
            )

            action_np = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

            # Overlays (optional): live pose + predicted wps in world (before step)
            ego_pos_xy = np.asarray(ego.position, dtype=np.float32)[:2]
            ego_heading = float(ego.heading_theta)
            pred_wps_world = None
            target_world = None
            if not args.no_overlay_traj and pred_coords is not None:
                pred_wps_ego = pred_coords.squeeze(0).detach().cpu().numpy().astype(np.float32)
                pred_wps_world = _ego_xy_to_world_xy(ego_pos_xy, ego_heading, pred_wps_ego)
            if not args.no_overlay_traj and step_idx < len(steps):
                tw = steps[step_idx].get("ego_pos_world_future_4")
                if tw is not None:
                    target_world = np.asarray(tw, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action_np)
            ep_reward += float(reward)

            render_kw = dict(
                mode="top_down",
                screen_record=True,
                window=False,
                screen_size=(640, 640),
                overlay_ego_pos_world=ego_pos_xy,
                overlay_ego_heading_rad=ego_heading,
            )
            if pred_wps_world is not None:
                render_kw["overlay_pred_traj_world"] = pred_wps_world
            if target_world is not None:
                render_kw["overlay_target_traj_world"] = target_world
            base_env.render(**render_kw)

            if terminated or truncated:
                break

        gif_path = os.path.join(out_dir, f"plant2_live_{episodes_done:03d}.gif")
        if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
            base_env.top_down_renderer.generate_gif(gif_path, duration=10)
            print(f"[GIF] {gif_path}  return={ep_reward:.2f}")
        else:
            print("[WARN] top_down_renderer missing; GIF skipped")

        episodes_done += 1

    if env is not None:
        env.close()

    print("\nDone. Outputs:", out_dir)


if __name__ == "__main__":
    main()
