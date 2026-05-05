#!/usr/bin/env python3
"""
Closed-loop evaluation of PlanT / HFLM on **fresh** MetaDrive environments.

Unlike eval_plant2_live_seeds_from_trajectories_gifs.py (which reads episode seeds
from pre-collected .pt files), this script creates brand-new episodes from a single
configurable start seed, so you can benchmark any checkpoint without having trajectory
data available.

Features
--------
- N episodes, each reset with seed = start_seed + episode_index
- Live metadrive_obs_to_plant2_batch observations (same pipeline as collection)
- Optional stop/speed-limit shields (--use-shields)
- Per-episode GIFs saved to output_dir
- Per-episode CSV row + printed summary table
- Aggregate metrics: success rate, crash rate, mean return, mean steps

Usage
-----
python eval_plant2_new_envs.py \\
    --checkpoint_file /path/to/plant2.ckpt \\
    --num-episodes 10 \\
    --start-seed 1000 \\
    --output_dir /path/to/outputs/eval_new_envs \\
    [--use-shields] [--no-gifs]
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup (mirrors collect / train scripts)
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
CARL_PATH = SDC_ROOT / "CaRL" / "nuPlan"

for _p in (SDC_ROOT, METADRIVE_DIR, PDD_BENCH_DIR, PLANT_PLAN_T_DIR, PLANT2_DIR, ADAPTER_PATH, CARL_PATH):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# Headless display
if os.environ.get("SDL_VIDEODRIVER") is None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


# ---------------------------------------------------------------------------
# Mock CARLA (not installed here)
# ---------------------------------------------------------------------------

def _mock_carla_modules():
    import unittest.mock as _mock
    for mod_name in ("carla", "agents", "agents.navigation",
                     "agents.navigation.global_route_planner"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _mock.MagicMock()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

import gymnasium as gym


class StopSignSpawnWrapper(gym.Wrapper):
    """Spawn random stop signs each episode reset."""

    def __init__(self, env, stop_sign_probability: float = 0.3,
                 max_signs: int = 10, min_distance: float = 25.0):
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


def create_env(
    seed: int = 1000,
    traffic_density: float = 0.1,
    horizon: int = 500,
    stop_sign_probability: float = 0.3,
    max_stop_signs: int = 10,
    num_scenarios: int = 200,
):
    from envs.traffic_sign_env import TrafficSignEnv

    config = dict(
        num_scenarios=num_scenarios,
        start_seed=seed,
        log_level=50,
        use_render=False,
        random_lane_width=True,
        random_lane_num=True,
        traffic_density=traffic_density,
        horizon=horizon,
        use_lateral_reward=True,
        success_reward=20.0,
        out_of_road_penalty=15.0,
        crash_vehicle_penalty=25.0,
        crash_object_penalty=20.0,
        crash_sidewalk_penalty=5.0,
        driving_reward=0.5,
        speed_reward=0.05,
    )
    env = TrafficSignEnv(config)
    env = StopSignSpawnWrapper(
        env,
        stop_sign_probability=stop_sign_probability,
        max_signs=max_stop_signs,
    )
    return env


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

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
# Helpers
# ---------------------------------------------------------------------------

def _parse_termination(info: Dict[str, Any]) -> str:
    """Return a short string describing why the episode ended."""
    if info.get("arrive_dest", False):
        return "success"
    if info.get("crash", False) or info.get("crash_vehicle", False) or info.get("crash_object", False):
        return "crash"
    if info.get("out_of_road", False):
        return "oor"
    if info.get("max_step", False):
        return "timeout"
    return "truncated"


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    header = ["ep", "seed", "steps", "return", "outcome"]
    col_w = [4, 6, 6, 9, 9]
    fmt_h = "  ".join(f"{h:<{w}}" for h, w in zip(header, col_w))
    sep = "-" * len(fmt_h)
    print(f"\n{sep}")
    print(fmt_h)
    print(sep)
    for r in rows:
        vals = [r["episode"], r["seed"], r["steps"], f"{r['return']:.2f}", r["outcome"]]
        print("  ".join(f"{str(v):<{w}}" for v, w in zip(vals, col_w)))
    print(sep)
    n = len(rows)
    n_success  = sum(r["outcome"] == "success" for r in rows)
    n_crash    = sum(r["outcome"] == "crash"   for r in rows)
    n_oor      = sum(r["outcome"] == "oor"     for r in rows)
    n_timeout  = sum(r["outcome"] == "timeout" for r in rows)
    mean_ret   = float(np.mean([r["return"] for r in rows]))
    mean_steps = float(np.mean([r["steps"]  for r in rows]))
    print(f"Episodes : {n}")
    print(f"Success  : {n_success}/{n}  ({100*n_success/n:.1f}%)")
    print(f"Crash    : {n_crash}/{n}  ({100*n_crash/n:.1f}%)")
    print(f"Out-road : {n_oor}/{n}  ({100*n_oor/n:.1f}%)")
    print(f"Timeout  : {n_timeout}/{n}  ({100*n_timeout/n:.1f}%)")
    print(f"Mean ret : {mean_ret:.3f}")
    print(f"Mean steps: {mean_steps:.1f}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PlanT/HFLM closed-loop on fresh MetaDrive environments."
    )
    parser.add_argument("--checkpoint_file", type=str, required=True,
                        help="Path to PlanT checkpoint (.pt or lightning .ckpt)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write GIFs and the results CSV")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--start-seed", type=int, default=1000,
                        help="Episode i uses reset seed = start_seed + i")
    parser.add_argument("--max-steps", type=int, default=1500,
                        help="Maximum steps per episode")
    parser.add_argument("--traffic-density", type=float, default=0.1)
    parser.add_argument("--stop-sign-probability", type=float, default=0.3)
    parser.add_argument("--num-scenarios", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-shields", action="store_true",
                        help="Apply stop-rule and speed-limit shields on top of model output")
    parser.add_argument("--no-gifs", action="store_true",
                        help="Skip GIF rendering (faster)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir or str(PDD_BENCH_DIR / "outputs" / "eval_plant2_new_envs")
    os.makedirs(out_dir, exist_ok=True)
    gif_dir = os.path.join(out_dir, "gifs")
    if not args.no_gifs:
        os.makedirs(gif_dir, exist_ok=True)

    print("=" * 70)
    print("PlanT CLOSED-LOOP EVAL — fresh environments")
    print(f"  Checkpoint : {args.checkpoint_file}")
    print(f"  Output dir : {out_dir}")
    print(f"  Episodes   : {args.num_episodes}")
    print(f"  Start seed : {args.start_seed}")
    print(f"  Max steps  : {args.max_steps}")
    print(f"  Device     : {device}")
    print(f"  Shields    : {args.use_shields}")
    print(f"  GIFs       : {not args.no_gifs}")
    print("=" * 70)

    net, config_all = load_plant_model(args.checkpoint_file, device=device)

    training_cfg = getattr(config_all.model, "training", {}) or {}
    input_bev       = True
    input_ego_speed = bool(training_cfg.get("input_ego_speed", False))

    from metadrive.policy.metadrive_obs_to_plant2 import metadrive_obs_to_plant2_batch
    from carla_garage.plant2_control import plant2_predictions_to_action, get_target_speed_from_limit

    # Optionally load shields
    if args.use_shields:
        from stop_rule_feat import StopRuleFeatureExtractor      # type: ignore
        from stop_shield import StopRuleShield, StopShieldConfig  # type: ignore
        from speed_limit_feat import SpeedLimitFeatureExtractor   # type: ignore
        from speed_limit_shield import SpeedLimitShield, SpeedLimitShieldConfig  # type: ignore

        stop_fx = StopRuleFeatureExtractor(
            dt=0.1, dist_norm_max_m=50.0, speed_norm_max_mps=20.0,
            required_hold_time_s=5.0, approach_dist_m=15.0,
        )
        stop_shield = StopRuleShield(
            stop_fx,
            StopShieldConfig(
                max_brake_norm=0.6, hold_brake_norm=0.4,
                release_accel_norm=0.5, hold_dist_m=0.5, post_clear_dist_m=2.0,
            ),
        )
        speed_fx     = SpeedLimitFeatureExtractor()
        speed_shield = SpeedLimitShield(speed_fx, SpeedLimitShieldConfig())
        print("[INFO] Safety shields loaded.")
    else:
        stop_fx = stop_shield = speed_fx = speed_shield = None

    env = create_env(
        seed=args.start_seed,
        traffic_density=args.traffic_density,
        horizon=args.max_steps,
        stop_sign_probability=args.stop_sign_probability,
        num_scenarios=args.num_scenarios,
    )
    base_env = env.unwrapped

    rows: List[Dict[str, Any]] = []
    csv_path = os.path.join(out_dir, "results.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=["episode", "seed", "steps", "return", "outcome"],
    )
    csv_writer.writeheader()

    for ep in range(args.num_episodes):
        ep_seed = args.start_seed + ep
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{args.num_episodes}  seed={ep_seed}")
        print(f"{'='*60}")

        obs, info = env.reset(seed=ep_seed)
        if args.use_shields:
            stop_fx.reset()
            speed_fx.reset()

        # Clear renderer frames
        if not args.no_gifs:
            if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
                if hasattr(base_env.top_down_renderer, "_screen_frames"):
                    base_env.top_down_renderer._screen_frames.clear()

        ep_return = 0.0
        outcome   = "timeout"
        n_steps   = 0

        for step_i in range(args.max_steps):
            ego = getattr(base_env, "agent", None) or getattr(base_env, "vehicle", None)
            if ego is None:
                break

            # Build PlanT batch from live MetaDrive state
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

            ego_speed       = float(getattr(ego, "speed", 0.0))
            speed_limit_idx = int(batch["speed_limit"][0].item())
            target_speed    = get_target_speed_from_limit(speed_limit_idx)

            action_np, _ = plant2_predictions_to_action(
                pred_plan,
                current_speed=ego_speed,
                target_speed_mps=target_speed,
                speed_limit_idx=speed_limit_idx,
                device=device,
                return_waypoints=True,
            )
            action_np = np.clip(np.asarray(action_np, dtype=np.float32), -1.0, 1.0)

            # Apply shields if enabled
            if args.use_shields:
                z_stop  = stop_fx.step_features(base_env.engine, ego)
                z_speed = speed_fx.step_features(base_env.engine, ego)
                action_np, _, stop_info = stop_shield.clip_action(z_stop, action_np.copy())
                stop_active = stop_info.get("must_stop", False) and not stop_info.get("stop_cleared", True)
                if ego.speed < 0.5 and stop_fx.state.stop_cleared:
                    action_np[0] = max(action_np[0], 0.3)
                action_np, _, _ = speed_shield.clip_action(z_speed, action_np, stop_shield_active=stop_active)

            # Render (before step so ego pose is aligned)
            if not args.no_gifs:
                # Top-down GIF frames (this MetaDrive TopDownRenderer has no plan-overlay kwargs;
                # use a fork that implements overlay_* if you need predicted trajectory on the GIF.)
                base_env.render(
                    mode="top_down",
                    screen_record=True,
                    window=False,
                    screen_size=(640, 640),
                )

            obs, reward, terminated, truncated, info = env.step(action_np)
            ep_return += float(reward)
            n_steps    = step_i + 1

            if terminated or truncated:
                outcome = _parse_termination(info)
                print(f"  step={n_steps}  outcome={outcome}  return={ep_return:.2f}")
                break

        # Save GIF
        if not args.no_gifs:
            gif_path = os.path.join(gif_dir, f"eval_ep{ep:03d}_seed{ep_seed}_{outcome}.gif")
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

        row = {"episode": ep, "seed": ep_seed, "steps": n_steps,
               "return": ep_return, "outcome": outcome}
        rows.append(row)
        csv_writer.writerow(row)
        csv_file.flush()

    csv_file.close()
    env.close()

    _print_summary(rows)
    print(f"\nResults CSV : {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
