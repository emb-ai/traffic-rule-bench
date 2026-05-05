#!/usr/bin/env python3
"""
Evaluate Plant2 (PlanT) on offline CaRL MetaDrive trajectories — **logged-action replay**.

The environment is driven by the **recorded MetaDrive actions** (`action_env`) from
`collect_metadrive_carl_plant2_trajectories.py`, so the vehicle follows the same
trajectory as during data collection.

At each step:
  - Load the **saved** `plant2_batch` from the .pt file (exact inputs from collection).
  - Run PlanT to get `pred_plan` → **predicted waypoints** (`pred_wps`).
  - Optionally derive control via `plant2_predictions_to_action` for diagnostics only;
    the trajectory overlay uses **waypoints** → world frame for the top-down GIF.
  - Call `env.step(action_env)` with the **true** logged action.
  - Top-down GIF: yellow disk + crosshair = logged `ego_pos_world_before`; cyan heading ray =
    `ego_heading_before` (same pose used to map pred wps → world). Green = target future, red = pred.

Input:
  - .pt trajectory files from collect_metadrive_carl_plant2_trajectories.py

See also: eval_plant2_live_seeds_from_trajectories_gifs.py — same seeds from .pt, but **closed-loop**
live `metadrive_obs_to_plant2_batch` + model actions (previous evaluation style).
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import cv2


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
CARLA_GARAGE_DIR = PLANT2_DIR / "carla_garage"

for _p in (SDC_ROOT, METADRIVE_DIR, PDD_BENCH_DIR, PLANT_PLAN_T_DIR, PLANT2_DIR, CARLA_GARAGE_DIR):
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
    """Load HFLM from config + checkpoint (same as PPO script)."""
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
    return net, config_all


def plant2_saved_dict_to_model_batch(raw: Dict[str, Any], device: str) -> Dict[str, Any]:
    """
    Convert numpy dict saved in .pt (per-step plant2_batch) to HFLM batch (B=1).
    Mirrors train_plant2_from_carl_trajectories.LitHFLMSupervised._normalize_batch.
    """
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if v is None:
            out[k] = None
            continue
        if k == "y_objs":
            out[k] = None
            continue
        arr = np.asarray(v)

        if k == "idxs":
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            t = torch.as_tensor(arr, device=device, dtype=torch.long).unsqueeze(0)
        elif k == "speed_limit":
            t = torch.as_tensor(arr.reshape(-1), device=device, dtype=torch.long)
        elif k == "route_original":
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            t = torch.as_tensor(arr, device=device, dtype=torch.float32).unsqueeze(0)
        elif k == "x_objs":
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            t = torch.as_tensor(arr, device=device, dtype=torch.float32).unsqueeze(0)
        elif k == "BEV":
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            t = torch.as_tensor(arr, device=device, dtype=torch.float32).unsqueeze(0)
        elif k == "input_ego_speed":
            t = torch.as_tensor(arr, device=device, dtype=torch.float32)
            if t.dim() == 1:
                t = t.unsqueeze(0)
            if t.dim() == 2 and t.shape[-1] == 1:
                pass  # (1,1) ok
        else:
            t = torch.as_tensor(arr, device=device, dtype=torch.float32)

        out[k] = t

    out["y_objs"] = None
    return out


def _meters_to_pixels(xy: np.ndarray, resolution: int, size_meters: float) -> np.ndarray:
    m_per_px = size_meters / float(resolution)
    cx = cy = resolution / 2.0
    xs = xy[:, 0]
    ys = xy[:, 1]
    px = cx + ys / m_per_px
    py = cy - xs / m_per_px
    return np.stack([px, py], axis=-1)


def render_frame_from_obs(
    obs: Dict[str, Any],
    pred_wps_ego: np.ndarray,
    route_ego: np.ndarray,
    bev_size_meters: float = 64.0,
) -> np.ndarray:
    """BEV from stored obs + route (white) + predicted wps (red), ego frame."""
    bev = np.asarray(obs["bev"], dtype=np.float32)
    _, H, _ = bev.shape
    ch0 = bev[0]
    img = (np.clip((ch0 + 1.0) / 3.0, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = np.ascontiguousarray(np.stack([img, img, img], axis=-1), dtype=np.uint8)

    tgt_xy = np.asarray(route_ego, dtype=np.float32)
    pred_xy = np.asarray(pred_wps_ego, dtype=np.float32)
    tgt_px = _meters_to_pixels(tgt_xy, resolution=H, size_meters=bev_size_meters)
    pred_px = _meters_to_pixels(pred_xy, resolution=H, size_meters=bev_size_meters)

    for i in range(len(tgt_px) - 1):
        cv2.line(
            rgb,
            (int(round(tgt_px[i, 0])), int(round(tgt_px[i, 1]))),
            (int(round(tgt_px[i + 1, 0])), int(round(tgt_px[i + 1, 1]))),
            (255, 255, 255), 2,
        )
    for i in range(len(pred_px) - 1):
        cv2.line(
            rgb,
            (int(round(pred_px[i, 0])), int(round(pred_px[i, 1]))),
            (int(round(pred_px[i + 1, 0])), int(round(pred_px[i + 1, 1]))),
            (255, 0, 0), 2,
        )
    return rgb


def _ego_xy_to_world_xy(ego_pos_xy: np.ndarray, ego_heading: float, xy_ego: np.ndarray) -> np.ndarray:
    """Ego (x forward, y right) → world XY."""
    xy = np.asarray(xy_ego, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy_ego must be (N,2), got {xy.shape}")
    x = xy[:, 0]
    y_right = xy[:, 1]
    c, s = float(np.cos(ego_heading)), float(np.sin(ego_heading))
    y_left = -y_right
    dx = x * c - y_left * s
    dy = x * s + y_left * c
    return np.stack([dx, dy], axis=-1) + np.asarray(ego_pos_xy, dtype=np.float32)[None, :]


def _route_from_saved_batch(plant2_save: Dict[str, Any]) -> np.ndarray:
    r = plant2_save.get("route_original")
    if r is None:
        return np.zeros((20, 2), dtype=np.float32)
    a = np.asarray(r, dtype=np.float32)
    if a.ndim == 3:
        a = a.squeeze(0)
    return a


def main():
    parser = argparse.ArgumentParser(
        description="PlanT eval: model on saved plant2_batch; env stepped with logged action_env; GIF overlays."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PDD_BENCH_DIR / "data"),
        help=".pt trajectories from collect_metadrive_carl_plant2_trajectories.py",
    )
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--plant_planT_path", type=str, default=str(PLANT_PLAN_T_DIR))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--traffic-density", type=float, default=0.1)
    parser.add_argument("--stop-sign-probability", type=float, default=0.3)
    parser.add_argument("--num-scenarios", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--save-bev-frames", action="store_true",
        help="Also write per-episode BEV PNGs (stored obs + route + pred wps).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir or str(PDD_BENCH_DIR / "outputs" / "plant2_eval_gifs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("PlanT on saved batches + logged actions (true trajectory replay)")
    print(f"  Data dir:       {args.data_dir}")
    print(f"  Checkpoint:     {args.checkpoint_file}")
    print(f"  Output dir:     {out_dir}")
    print(f"  Max episodes:   {args.max_episodes}")
    print(f"  Max steps/ep:   {args.max_steps}")
    print(f"  Device:         {device}")
    print("=" * 70)

    net, _config_all = load_plant_model(args.checkpoint_file, args.plant_planT_path, device=device)
    net = net.to(device)
    net.eval()

    from scripts.agents.train.train_metadrive_ppo_plant_stop_gifs import create_env  # type: ignore
    from carla_garage.plant2_control import plant2_predictions_to_action, get_target_speed_from_limit

    data_path = Path(args.data_dir)
    files = sorted(data_path.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {args.data_dir}")

    env = None
    base_env = None
    episodes_done = 0

    for f in files:
        if episodes_done >= args.max_episodes:
            break

        ep = torch.load(f, weights_only=False, map_location="cpu")
        steps: List[Dict[str, Any]] = ep.get("steps", [])
        base_seed = int(ep.get("base_seed", 0))
        reset_seed = int(ep.get("reset_seed", base_seed + ep.get("episode_index", 0)))

        if env is None:
            env = create_env(
                seed=base_seed,
                traffic_density=args.traffic_density,
                horizon=args.max_steps,
                stop_sign_probability=args.stop_sign_probability,
                num_scenarios=args.num_scenarios,
            )
            base_env = env.unwrapped

        print(f"\nEpisode file: {f.name}  base_seed={base_seed} reset_seed={reset_seed} n_steps={len(steps)}")

        obs, info = env.reset(seed=reset_seed)
        if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
            if hasattr(base_env.top_down_renderer, "_screen_frames"):
                base_env.top_down_renderer._screen_frames.clear()

        ep_reward = 0.0
        n_play = min(len(steps), args.max_steps)
        bev_dir: Optional[Path] = None
        if args.save_bev_frames:
            bev_dir = Path(out_dir) / f"bev_{episodes_done:03d}"
            bev_dir.mkdir(parents=True, exist_ok=True)

        for step_idx in range(n_play):
            ego = getattr(base_env, "agent", None) or getattr(base_env, "vehicle", None)
            if ego is None:
                break

            step = steps[step_idx]
            plant_save = step.get("plant2_batch")
            if plant_save is None:
                print(f"[WARN] step {step_idx}: no plant2_batch, skipping model")
                action_np = np.asarray(step.get("action_env", [0.0, 0.0]), dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(np.clip(action_np, -1.0, 1.0))
                ep_reward += float(reward)
                ref_xy = np.asarray(step.get("ego_pos_world_before", ego.position[:2]), dtype=np.float32)[:2]
                ref_h = float(step.get("ego_heading_before", ego.heading_theta))
                base_env.render(
                    mode="top_down",
                    screen_record=True,
                    window=False,
                    screen_size=(640, 640),
                    overlay_ego_pos_world=ref_xy,
                    overlay_ego_heading_rad=ref_h,
                )
                if terminated or truncated:
                    break
                continue

            batch = plant2_saved_dict_to_model_batch(plant_save, device=device)

            with torch.no_grad():
                _, _, pred_plan, _ = net(batch)

            pred_path, pred_wps, _ = pred_plan
            pred_coords = pred_wps if pred_wps is not None else pred_path
            if pred_coords is None:
                raise RuntimeError("Model returned no pred_wps / pred_path")

            pred_wps_ego = pred_coords.squeeze(0).detach().cpu().numpy().astype(np.float32)

            # Optional: control from predictions (diagnostic only; env uses logged action)
            ego_speed = float(getattr(ego, "speed", 0.0))
            speed_limit_idx = int(batch["speed_limit"][0].item())
            target_speed_mps = get_target_speed_from_limit(speed_limit_idx)
            _pred_action = plant2_predictions_to_action(
                pred_plan,
                current_speed=ego_speed,
                target_speed_mps=target_speed_mps,
                speed_limit_idx=speed_limit_idx,
                speed_limits_kmh=(50, 80, 100, 120),
                device=device,
                return_waypoints=False,
            )
            ego_pos_xy = np.asarray(step.get("ego_pos_world_before", ego.position[:2]), dtype=np.float32)[:2]
            ego_heading = float(step.get("ego_heading_before", ego.heading_theta))

            if args.verbose:
                print(f"  step {step_idx}  pred_action (unused)={np.asarray(_pred_action)}")
                print(f"           overlay_ego_pos_world (logged before step)={ego_pos_xy}  heading={ego_heading:.4f}")

            pred_wps_world = _ego_xy_to_world_xy(ego_pos_xy, ego_heading, pred_wps_ego)

            target_world = step.get("ego_pos_world_future_4")
            if target_world is not None:
                target_world = np.asarray(target_world, dtype=np.float32)

            # Stored obs BEV viz (same ego frame as batch)
            plant_obs = step.get("obs")
            route_ego = _route_from_saved_batch(plant_save)
            if plant_obs is not None and bev_dir is not None:
                frame = render_frame_from_obs(plant_obs, pred_wps_ego, route_ego)
                cv2.imwrite(str(bev_dir / f"step_{step_idx:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            action_np = np.asarray(step["action_env"], dtype=np.float32)
            action_np = np.clip(action_np, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action_np)
            ep_reward += float(reward)

            base_env.render(
                mode="top_down",
                screen_record=True,
                window=False,
                screen_size=(640, 640),
                overlay_target_traj_world=target_world,
                overlay_pred_traj_world=pred_wps_world,
                overlay_ego_pos_world=ego_pos_xy,
                overlay_ego_heading_rad=ego_heading,
            )

            if terminated or truncated:
                break

        gif_path = os.path.join(out_dir, f"plant2_eval_{episodes_done:03d}.gif")
        if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
            base_env.top_down_renderer.generate_gif(gif_path, duration=10)
            print(f"[GIF] {gif_path}  return={ep_reward:.2f}")
        else:
            print("[WARN] No top_down_renderer; GIF skipped")

        episodes_done += 1

    if env is not None:
        env.close()

    print("\nDone. Outputs:", out_dir)


if __name__ == "__main__":
    main()
