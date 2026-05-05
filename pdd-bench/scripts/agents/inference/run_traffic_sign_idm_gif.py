#!/usr/bin/env python3
"""
Run one (or several) episodes in `TrafficSignEnv` using `ModifiedIDMPolicy`
and save a top-down GIF of the rollout.

Usage example (from pdd-bench root):
  python -m scripts.agents.inference.run_traffic_sign_idm_gif \
      --episodes 1 --max-steps 500
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np


def _setup_paths():
    """Add pdd-bench and metadrive roots to sys.path (same layout as training scripts)."""
    script_dir = Path(__file__).resolve().parent
    pdd_bench_dir = script_dir.parents[2]
    sdc_root = pdd_bench_dir.parent
    metadrive_dir = sdc_root / "metadrive"

    for p in (pdd_bench_dir, metadrive_dir, sdc_root):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)

    return pdd_bench_dir, metadrive_dir, sdc_root


def create_env(
    seed: int = 500,
    traffic_density: float = 0.1,
    horizon: int = 3000,
    stop_sign_probability: float = 0.3,
    max_stop_signs: int = 10,
    num_scenarios: int = 200,
):
    """
    Create `TrafficSignEnv` and wrap it with `StopSignSpawnWrapper`
    (pattern taken from train_metadrive_ppo_plant_stop_gifs.create_env).
    """
    from envs.traffic_sign_env import TrafficSignEnv
    from scripts.agents.train.train_metadrive_ppo_plant_stop_gifs import StopSignSpawnWrapper

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
    env = StopSignSpawnWrapper(env, stop_sign_probability=stop_sign_probability, max_signs=max_stop_signs)
    return env


def run_idm_and_save_gif(
    episodes: int,
    max_steps: int,
    output_dir: str,
    seed: int = 0,
    traffic_density: float = 0.1,
    stop_sign_probability: float = 0.3,
    num_scenarios: int = 200,
    max_stop_signs: int = 10,
):
    from metadrive.policy.idm_policy import ModifiedIDMPolicy

    os.makedirs(output_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    env = create_env(
        seed=seed,
        traffic_density=traffic_density,
        horizon=max_steps,
        stop_sign_probability=stop_sign_probability,
        max_stop_signs=max_stop_signs,
        num_scenarios=num_scenarios,
    )

    # Base MetaDrive env (under gym wrappers) exposes top_down_renderer/render.
    base_env = env.unwrapped

    for ep in range(episodes):
        obs, info = env.reset()

        # Clear any previous frames if renderer exists.
        if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
            if hasattr(base_env.top_down_renderer, "_screen_frames"):
                base_env.top_down_renderer._screen_frames.clear()

        # Get ego vehicle/agent and attach ModifiedIDMPolicy to it.
        ego = getattr(base_env, "vehicle", None) or getattr(base_env, "agent", None)
        if ego is None:
            raise RuntimeError("Could not find ego vehicle/agent on base_env.")

        policy = ModifiedIDMPolicy(control_object=ego, random_seed=seed)

        ep_reward = 0.0

        for step in range(max_steps):
            action = policy.act()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)

            # Record top-down frame (no on-screen window).
            base_env.render(
                mode="top_down",
                screen_record=True,
                window=False,
                screen_size=(640, 640),
            )

            if terminated or truncated:
                break

        gif_path = os.path.join(output_dir, f"traffic_sign_idm_ep{ep + 1}.gif")
        if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
            base_env.top_down_renderer.generate_gif(gif_path, duration=10)
            print(f"[GIF] Saved episode {ep + 1}/{episodes} to {gif_path} (return={ep_reward:.1f})")
        else:
            print("[WARN] top_down_renderer missing; cannot save GIF.")

    env.close()


def main():
    _, _, _ = _setup_paths()

    parser = argparse.ArgumentParser(
        description="Run ModifiedIDMPolicy in TrafficSignEnv and save top-down GIF rollouts."
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save GIFs (default: pdd-bench/outputs/traffic_sign_idm).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--traffic-density", type=float, default=0.1, help="Traffic density for MetaDrive config.")
    parser.add_argument(
        "--stop-sign-probability", type=float, default=0.3, help="Probability of placing a stop sign on a lane."
    )
    parser.add_argument("--max-stop-signs", type=int, default=10, help="Maximum number of stop signs per episode.")
    parser.add_argument("--num-scenarios", type=int, default=200, help="Number of scenarios in MetaDrive config.")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    pdd_bench_dir = script_dir.parents[2]
    default_output = pdd_bench_dir / "outputs" / "traffic_sign_idm"
    output_dir = args.output_dir or str(default_output)

    run_idm_and_save_gif(
        episodes=args.episodes,
        max_steps=args.max_steps,
        output_dir=output_dir,
        seed=args.seed,
        traffic_density=args.traffic_density,
        stop_sign_probability=args.stop_sign_probability,
        num_scenarios=args.num_scenarios,
        max_stop_signs=args.max_stop_signs,
    )


if __name__ == "__main__":
    main()

