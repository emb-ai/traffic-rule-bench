#!/usr/bin/env python3
"""
Run CaRL agent with right-turn verifier and save top-down GIF with compliance text.

Usage (from pdd-bench root):
    python scripts/agents/inference/run_carl_right_turn_verifier.py
    python scripts/agents/inference/run_carl_right_turn_verifier.py --episodes 3 --map OTX
"""

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PDD_BENCH_DIR = SCRIPT_DIR.parents[2]
SDC_ROOT = PDD_BENCH_DIR.parent
ADAPTER_PATH = PDD_BENCH_DIR / "agents" / "carl_in_metadrive"
CARL_PATH = SDC_ROOT / "CaRL" / "nuPlan"
METADRIVE_PATH = SDC_ROOT / "metadrive"

DEFAULT_CHECKPOINT = str(CARL_PATH / "checkpoints" / "nuplan_51479_1B" / "model_best.pth")
DEFAULT_OUTPUT_DIR = str(PDD_BENCH_DIR / "outputs" / "carl_right_turn_verifier")

for path in [PDD_BENCH_DIR, ADAPTER_PATH, CARL_PATH, METADRIVE_PATH]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from carl_adapter import CaRLMetaDriveAdapter
from envs.traffic_sign_env import TrafficSignEnv
from traffic_signs.right_turn_rule import RightTurnRule


def _safe_speed_kmh(vehicle) -> float:
    if hasattr(vehicle, "speed_km_h"):
        try:
            return float(vehicle.speed_km_h)
        except Exception:
            pass
    if hasattr(vehicle, "speed"):
        try:
            return float(vehicle.speed) * 3.6
        except Exception:
            pass
    return 0.0


def _compliance_label(status) -> str:
    planning = bool(status.get("is_planning_right_turn", False))
    compliant = bool(status.get("is_rule_compliant", True))
    if not planning:
        return "N/A"
    return "COMPLIANT" if compliant else "VIOLATION"


def run_carl_with_right_turn_verifier(
    checkpoint_path: str,
    output_dir: str,
    num_episodes: int,
    max_steps: int,
    start_seed: int,
    num_scenarios: int,
    map_name: str,
    traffic_density: float,
    lookahead_distance: float,
    device: str,
    render: bool,
    screen_size: int,
    debug_rule: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    env_config = {
        "use_render": False,
        "traffic_density": float(traffic_density),
        "start_seed": int(start_seed),
        "map": str(map_name),
        "num_scenarios": int(num_scenarios),
        "horizon": int(max_steps),
        "use_mesh_terrain": False,
        "show_coordinates": False,
    }

    env = TrafficSignEnv(config=env_config)
    adapter = CaRLMetaDriveAdapter(checkpoint_path=checkpoint_path, device=device)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")
    print(f"Map: {map_name}, episodes: {num_episodes}, max_steps: {max_steps}")

    for ep in range(num_episodes):
        seed = int(start_seed + ep)
        print(f"\n{'=' * 72}")
        print(f"Episode {ep + 1}/{num_episodes} | seed={seed}")
        print(f"{'=' * 72}")

        _obs, _info = env.reset(seed=seed)
        adapter.reset()
        verifier = RightTurnRule(None, debug=debug_rule, lookahead_distance=lookahead_distance)

        if render:
            try:
                import pygame
                if not pygame.display.get_init():
                    pygame.display.init()
                    pygame.display.set_mode((1, 1))
            except Exception:
                pass

        ep_reward = 0.0
        ep_len = 0
        planning_steps = 0
        compliant_steps = 0
        violation_steps = 0

        for step in range(max_steps):
            action = adapter.get_action(env.agent, env.engine, deterministic=True)
            _obs, reward, terminated, truncated, _info = env.step(action)
            ep_reward += float(reward)
            ep_len = step + 1

            status = verifier.get_status(env.agent)
            planning = bool(status.get("is_planning_right_turn", False))
            rightmost = bool(status.get("is_rightmost_lane", False))
            violation = bool(status.get("violation", False))
            compliant = bool(status.get("is_rule_compliant", True))

            if planning:
                planning_steps += 1
                if compliant:
                    compliant_steps += 1
                if violation:
                    violation_steps += 1

            if render:
                speed_kmh = _safe_speed_kmh(env.agent)
                lane_idx = getattr(env.agent, "lane_index", None)
                lane_id = lane_idx[2] if isinstance(lane_idx, tuple) and len(lane_idx) >= 3 else -1
                text = {
                    "Step": step,
                    "Speed": f"{speed_kmh:.1f} km/h",
                    "PlanRight": "YES" if planning else "NO",
                    "LaneIdx": lane_id,
                    "Rightmost": "YES" if rightmost else "NO",
                    "Rule": _compliance_label(status),
                }
                env.render(
                    mode="top_down",
                    screen_record=True,
                    window=False,
                    screen_size=(screen_size, screen_size),
                    text=text,
                )

            if terminated or truncated:
                print(f"Episode ended at step={step} terminated={terminated} truncated={truncated}")
                break

        compliance_rate = (compliant_steps / planning_steps) if planning_steps > 0 else 0.0
        print(
            f"Reward={ep_reward:.2f} | steps={ep_len} | "
            f"planning_steps={planning_steps} | compliant={compliant_steps} | "
            f"viol={violation_steps} | compliance_rate={compliance_rate:.3f}"
        )

        if render and hasattr(env, "top_down_renderer") and env.top_down_renderer is not None:
            frames = env.top_down_renderer._screen_frames
            if frames:
                gif_path = os.path.join(output_dir, f"ep{ep:02d}_right_turn.gif")
                env.top_down_renderer.generate_gif(gif_name=gif_path, duration=50)
                print(f"Saved GIF: {gif_path}")
            env.top_down_renderer.clear()

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/home/jovyan/shares/SR006.nfs2/smirnova/carl_exps/residual/checkpoints/rt_finetuned_from_default_sparse_style.pth") #DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=str, default=f"{DEFAULT_OUTPUT_DIR}_new__from_default_sparse_style")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--num-scenarios", type=int, default=100)
    parser.add_argument("--map", type=str, default="S")
    parser.add_argument("--traffic-density", type=float, default=0.05)
    parser.add_argument("--lookahead-distance", type=float, default=100.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--screen-size", type=int, default=800)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--debug-rule", action="store_true")
    args = parser.parse_args()

    run_carl_with_right_turn_verifier(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        start_seed=args.start_seed,
        num_scenarios=args.num_scenarios,
        map_name=args.map,
        traffic_density=args.traffic_density,
        lookahead_distance=args.lookahead_distance,
        device=args.device,
        render=not args.no_render,
        screen_size=args.screen_size,
        debug_rule=args.debug_rule,
    )


if __name__ == "__main__":
    main()
