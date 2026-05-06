#!/usr/bin/env python3
"""
Run CaRL agent in pdd-bench environment with stop signs.

Usage (from repo root):
    python pdd-bench/scripts/agents/inference/run_carl_pdd_bench.py
    python pdd-bench/scripts/agents/inference/run_carl_pdd_bench.py --checkpoint /path/to/model.pth
    python pdd-bench/scripts/agents/inference/run_carl_pdd_bench.py --no-render --episodes 5
"""

import sys
import os
import argparse
import numpy as np
import cv2
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PDD_BENCH_DIR = SCRIPT_DIR.parents[2]
SDC_ROOT = PDD_BENCH_DIR.parent
ADAPTER_PATH = PDD_BENCH_DIR / "agents" / "carl_in_metadrive"
CARL_PATH = SDC_ROOT / "CaRL" / "nuPlan"
METADRIVE_PATH = SDC_ROOT / "metadrive"
PDD_BENCH_PATH = PDD_BENCH_DIR
print(PDD_BENCH_DIR, ADAPTER_PATH)
DEFAULT_CHECKPOINT = str(
    CARL_PATH / "checkpoints" / "nuplan_51479_1B" / "model_best.pth"
)

for path in [PDD_BENCH_PATH, ADAPTER_PATH, CARL_PATH, METADRIVE_PATH]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from carl_adapter import CaRLMetaDriveAdapter
from metadrive_core.sign_utils import add_signs_by_count

import importlib.util

# TrafficSignEnv
pdd_bench_envs_path = os.path.join(str(PDD_BENCH_PATH), "envs", "traffic_sign_env.py")
spec_env = importlib.util.spec_from_file_location("pdd_bench_traffic_sign_env", pdd_bench_envs_path)
pdd_bench_envs_module = importlib.util.module_from_spec(spec_env)
spec_env.loader.exec_module(pdd_bench_envs_module)
TrafficSignEnv = pdd_bench_envs_module.TrafficSignEnv

# Import all traffic sign types
sign_modules = {}
sign_classes = {}

# StopSign
pdd_bench_signs_path = os.path.join(str(PDD_BENCH_PATH), "traffic_signs", "stop_sign.py")
spec_sign = importlib.util.spec_from_file_location("pdd_bench_stop_sign", pdd_bench_signs_path)
pdd_bench_signs_module = importlib.util.module_from_spec(spec_sign)
spec_sign.loader.exec_module(pdd_bench_signs_module)
StopSign = pdd_bench_signs_module.StopSign
sign_classes["2.5"] = StopSign

# SpeedLimitSign
speed_limit_path = os.path.join(str(PDD_BENCH_PATH), "traffic_signs", "speed_limit_sign.py")
spec_speed = importlib.util.spec_from_file_location("pdd_bench_speed_limit", speed_limit_path)
speed_limit_module = importlib.util.module_from_spec(spec_speed)
spec_speed.loader.exec_module(speed_limit_module)
SpeedLimitSign20 = speed_limit_module.SpeedLimitSign20
SpeedLimitSign40 = speed_limit_module.SpeedLimitSign40
SpeedLimitSign60 = speed_limit_module.SpeedLimitSign60
sign_classes["3.24_20"] = SpeedLimitSign20
sign_classes["3.24_40"] = SpeedLimitSign40
sign_classes["3.24_60"] = SpeedLimitSign60

# NoStoppingAllowedSign
no_stopping_path = os.path.join(str(PDD_BENCH_PATH), "traffic_signs", "no_stopping_allowed_sign.py")
spec_no_stop = importlib.util.spec_from_file_location("pdd_bench_no_stopping", no_stopping_path)
no_stopping_module = importlib.util.module_from_spec(spec_no_stop)
spec_no_stop.loader.exec_module(no_stopping_module)
NoStoppingAllowedSign = no_stopping_module.NoStoppingAllowedSign
sign_classes["3.27"] = NoStoppingAllowedSign

# DirectionSign
direction_path = os.path.join(str(PDD_BENCH_PATH), "traffic_signs", "direction_sign.py")
spec_dir = importlib.util.spec_from_file_location("pdd_bench_direction", direction_path)
direction_module = importlib.util.module_from_spec(spec_dir)
spec_dir.loader.exec_module(direction_module)
DirectionSign = direction_module.DirectionSign
sign_classes["5.15.2"] = DirectionSign



def run_carl_with_stop_signs(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    num_episodes: int = 1,
    max_steps: int = 500,
    render: bool = True,
    output_dir: str = None,
    device: str = "cuda",
    save_bev = False,
    sign_counts: str = None,
    min_distance: float = 15.0,
):
    """ CaRL agent in pdd-bench environment with traffic signs
    
    Args:
        sign_counts: comma-separated "type:count" pairs, e.g. "2.5:3,3.24_40:2,3.27:1"
    """
    
    if output_dir is None:
        output_dir = str(PDD_BENCH_DIR / "outputs" / "carl_pdd_bench")
    
    os.makedirs(output_dir, exist_ok=True)
    bev_output_dir = os.path.join(output_dir, "bev_channels")
    os.makedirs(bev_output_dir, exist_ok=True)
    
    env_config = {
        "use_render": False,
        "traffic_density": 0.0, 
        "start_seed": 42,
        "map": "S", 
        "num_scenarios": 10,
        "horizon": max_steps,
        "use_mesh_terrain": False,  
        "show_coordinates": False,  
        "vehicle_config": {
            "spawn_lane_index": (">", ">>", 0),
        }
    }
    
    env = TrafficSignEnv(config=env_config)
    
    print(f"Loading CaRL model from: {checkpoint_path}")
    adapter = CaRLMetaDriveAdapter(checkpoint_path, device=device)
    
    episode_stats = []
    
    for ep in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        obs, info = env.reset()
        adapter.reset()

        sign_counts_dict = {}
        if sign_counts:
            for pair in sign_counts.split(","):
                pair = pair.strip()
                if ":" not in pair:
                    continue
                sign_type, count_str = pair.split(":", 1)
                sign_type = sign_type.strip()
                try:
                    count = int(count_str.strip())
                    if sign_type in sign_classes and count > 0:
                        sign_counts_dict[sign_classes[sign_type]] = count
                except ValueError:
                    print(f"Warning: Invalid count for '{sign_type}', skipping")
        
        total_signs_added = 0
        result_counts = {}
        if sign_counts_dict:
            print(f"Adding signs: {[(sc.__name__, cnt) for sc, cnt in sign_counts_dict.items()]}...")
            result_counts = add_signs_by_count(env, sign_counts_dict, min_distance=min_distance)
            total_signs_added = sum(result_counts.values())
            print(f"Signs added: {result_counts} (total: {total_signs_added})")
            
            if hasattr(env, 'engine') and hasattr(env.engine, 'traffic_sign_manager'):
                sign_mgr = env.engine.traffic_sign_manager
                if sign_mgr:
                    actual_signs = len([s for s in sign_mgr.signs if s is not None])
                    print(f"  Verified: {actual_signs} signs in traffic_sign_manager")
        else:
            print("No signs specified. Running without traffic signs.")
        
        if render:
            try:
                import pygame
                if not pygame.display.get_init():
                    pygame.display.init()
                    pygame.display.set_mode((1, 1))
            except Exception as e:
                print(f"Warning: Failed to initialize pygame display: {e}")
            
            env.render(mode="top_down", screen_record=True, window=False, screen_size=(800, 800))
        
        ep_reward = 0
        ep_length = 0
        violations = []
        violation_details = []  
        
        sign_mgr = None
        if hasattr(env, 'engine') and hasattr(env.engine, 'traffic_sign_manager'):
            sign_mgr = env.engine.traffic_sign_manager
        
        if total_signs_added > 0:
            if sign_mgr:
                print(f"  Sign manager found. Signs count: {len(sign_mgr.signs) if sign_mgr.signs else 0}")
            else:
                print(f"  Warning: Sign manager not found!")
        
        for step in range(max_steps):
            carl_obs = adapter.get_observation(env.agent, env.engine)
            bev = carl_obs["bev_semantics"]
            
            action = adapter.get_action(env.agent, env.engine)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if total_signs_added > 0 and sign_mgr and sign_mgr.signs:
                try:

                    for sign in sign_mgr.signs:
                        if sign is None:
                            continue
                        try:
                            if hasattr(sign, '_is_violating'):
                                is_violating_now = sign._is_violating(env.agent)
                                if is_violating_now:
                                    sign_type = type(sign).__name__
                                    # a new violation?
                                    violation_key = (sign.id if hasattr(sign, 'id') else id(sign), sign_type)
                                    if violation_key not in [v[0] for v in violation_details]:
                                        violations.append(sign)
                                        violation_details.append((violation_key, sign_type, step))
                                        print(f"  [VIOLATION] {sign_type} violated at step {step}!")
                        except Exception as e:
                            print(f"  Warning: Error checking violations: {e}")
                            import traceback
                            traceback.print_exc()
                    
                except Exception as e:
                    # Only print error once to avoid spam
                    if step == 1:
                        print(f"  Warning: Error checking violations: {e}")
                        import traceback
                        traceback.print_exc()
            
            if save_bev and (step % 50 == 0 or step < 5 or terminated or truncated):
                # Visualize BEV channels
                bev_vis = adapter.visualize_bev(bev)
                
                # Save full BEV grid
                bev_path = os.path.join(bev_output_dir, f"ep{ep:02d}_step{step:04d}_bev_grid.png")
                cv2.imwrite(bev_path, bev_vis)
                
                if step % 50 == 0:
                    print(f"  Step {step}: Saved BEV channels to {bev_output_dir}")
            
            env.render(mode="top_down", screen_record=True, window=False, screen_size=(800, 800))
            
            ep_reward += reward
            ep_length += 1
            
            if terminated or truncated:
                print(f"  Episode ended. Terminated: {terminated}, Truncated: {truncated}")
                # Save final BEV
                final_bev_vis = adapter.visualize_bev(bev)
                final_bev_path = os.path.join(bev_output_dir, f"ep{ep:02d}_final_bev_grid.png")
                cv2.imwrite(final_bev_path, final_bev_vis)
                print(f"  Saved final BEV to {final_bev_path}")
                break
        
        # Print violation summary
        if total_signs_added > 0:
            print(f"Episode finished. Steps: {ep_length}. Reward: {ep_reward:.2f}")
            print(f"  Traffic signs: {total_signs_added} ({result_counts})")
            print(f"  Violations: {len(violations)}/{total_signs_added}")
            if violation_details:
                print(f"  Violation details:")
                for v_key, v_type, v_step in violation_details:
                    print(f"    - {v_type} at step {v_step}")
        else:
            print(f"Episode finished. Steps: {ep_length}. Reward: {ep_reward:.2f}")
            print(f"  No traffic signs in this episode.")
        print(f"  BEV channels saved to: {bev_output_dir}")
        
        episode_stats.append({
            "episode": ep + 1,
            "length": ep_length,
            "reward": ep_reward,
            "violations": len(violations),
            "total_signs": total_signs_added
        })
        
        # Save video if rendered
        if hasattr(env, "top_down_renderer") and env.top_down_renderer is not None:
            frames = env.top_down_renderer._screen_frames  
            if frames: 
                os.makedirs(output_dir, exist_ok=True)
                gif_path = os.path.join(output_dir, f"ep{ep:02d}_topdown.gif")
                env.top_down_renderer.generate_gif(gif_name=gif_path, duration=50)
                print(f"Saved top-down GIF: {gif_path}")
            env.top_down_renderer.clear()

    env.close()
    
    # Stats
    print(f"\n{'='*60}")
    print("Overall Statistics:")
    print(f"{'='*60}")
    for stat in episode_stats:
        if stat['total_signs'] > 0:
            print(f"Episode {stat['episode']}: Steps={stat['length']}, Reward={stat['reward']:.2f}, Violations={stat['violations']}/{stat['total_signs']}")
        else:
            print(f"Episode {stat['episode']}: Steps={stat['length']}, Reward={stat['reward']:.2f}, No signs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--save-bev", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--sign-counts",
        type=str,
        default=None,
        help='Comma-separated "type:count" pairs, e.g. "2.5:3,3.24_40:2,3.27:1". Available types: 2.5 (Stop), 3.24_20/40/60 (SpeedLimit), 3.27 (NoStopping), 5.15.2 (Direction). If not specified, no signs will be added.'
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=5.0,
        help="Minimum distance between signs (default: 15.0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: pdd-bench/outputs/carl_pdd_bench)",
    )
    args = parser.parse_args()
    
    run_carl_with_stop_signs(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        device=args.device,
        save_bev=args.save_bev,
        sign_counts=args.sign_counts,
        min_distance=args.min_distance,
        output_dir=args.output,
    )
