#!/usr/bin/env python3
"""
Run CaRL agent in pdd-bench environment with safety shields.

This script uses shields to correct CaRL actions for:
- Stop signs: ensures proper stopping behavior
- Speed limit signs: ensures speed limit compliance

Recording is headless (no display). By default a top-down GIF is saved per episode
under --output, so it works on remote machines without a display.

Usage (from repo root):
    python pdd-bench/scripts/agents/inference/run_carl_pdd_bench_shield.py
    python pdd-bench/scripts/agents/inference/run_carl_pdd_bench_shield.py --sign-counts "2.5:3,3.24_40:2"
    python pdd-bench/scripts/agents/inference/run_carl_pdd_bench_shield.py --episodes 5 --no-shields
    python pdd-bench/scripts/agents/inference/run_carl_pdd_bench_shield.py -o ./my_gifs --episodes 3
"""

# Headless: set before any imports that use pygame (e.g. MetaDrive) to avoid "No video mode has been set"
import os
if os.environ.get("SDL_VIDEODRIVER") is None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PDD_BENCH_DIR = SCRIPT_DIR.parents[2]
SDC_ROOT = PDD_BENCH_DIR.parent
ADAPTER_PATH = PDD_BENCH_DIR / "agents" / "carl_in_metadrive"
CARL_PATH = SDC_ROOT / "CaRL" / "nuPlan"
METADRIVE_PATH = SDC_ROOT / "metadrive"
PDD_BENCH_PATH = PDD_BENCH_DIR

DEFAULT_CHECKPOINT = str(
    CARL_PATH / "checkpoints" / "nuplan_51479_1B" / "model_best.pth"
)

for path in [PDD_BENCH_PATH, ADAPTER_PATH, CARL_PATH, METADRIVE_PATH]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from carl_adapter import CaRLMetaDriveAdapter
from metadrive_core.sign_utils import add_signs_by_count

# Import shields and feature extractors
from stop_rule_feat import StopRuleFeatureExtractor
from stop_shield import StopRuleShield, StopShieldConfig
from speed_limit_feat import SpeedLimitFeatureExtractor
from speed_limit_shield import SpeedLimitShield, SpeedLimitShieldConfig

import importlib.util

# TrafficSignEnv
pdd_bench_envs_path = os.path.join(str(PDD_BENCH_PATH), "envs", "traffic_sign_env.py")
spec_env = importlib.util.spec_from_file_location("pdd_bench_traffic_sign_env", pdd_bench_envs_path)
pdd_bench_envs_module = importlib.util.module_from_spec(spec_env)
spec_env.loader.exec_module(pdd_bench_envs_module)
TrafficSignEnv = pdd_bench_envs_module.TrafficSignEnv

# Import traffic sign types
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
SpeedLimitSign = speed_limit_module.SpeedLimitSign
sign_classes["3.24_20"] = SpeedLimitSign20
sign_classes["3.24_40"] = SpeedLimitSign40
sign_classes["3.24_60"] = SpeedLimitSign60


def create_env(
    seed: int = 42,
    traffic_density: float = 0.1,
    horizon: int = 500,
    stop_sign_probability: float = 0.3,
    max_stop_signs: int = 10,
    num_scenarios: int = 200,
):
    """Create TrafficSignEnv + StopSignSpawnWrapper with same config as run_traffic_sign_idm_gif.py."""
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
    env = StopSignSpawnWrapper(
        env,
        stop_sign_probability=stop_sign_probability,
        max_signs=max_stop_signs,
        min_distance=25.0,
    )
    return env


def run_carl_with_shields(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    num_episodes: int = 1,
    max_steps: int = 500,
    save_gif: bool = True,
    output_dir: str = None,
    device: str = "cuda",
    save_bev: bool = False,
    sign_counts: str = None,
    min_distance: float = 15.0,
    use_shields: bool = True,
    seed: int = 42,
    traffic_density: float = 0.1,
    num_scenarios: int = 200,
    stop_sign_probability: float = 0.3,
    max_stop_signs: int = 10,
):
    """Run CaRL agent with safety shields for stop signs and speed limits.
    
    Recording is headless (no display window). When save_gif is True, top-down
    frames are recorded each step and saved as a GIF per episode under output_dir.
    Works on remote machines without a display.
    
    Args:
        checkpoint_path: Path to CaRL model checkpoint
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        save_gif: Whether to record and save top-down GIF per episode (headless-friendly)
        output_dir: Output directory for results and GIFs
        device: Device to run model on (cuda/cpu)
        save_bev: Whether to save BEV channel visualizations
        sign_counts: Comma-separated "type:count" pairs, e.g. "2.5:3,3.24_40:2"
        min_distance: Minimum distance between signs
        use_shields: Whether to apply safety shields (if False, runs without shields)
    """
    
    if output_dir is None:
        output_dir = str(PDD_BENCH_DIR / "outputs" / "carl_pdd_bench_shield")
    
    os.makedirs(output_dir, exist_ok=True)
    bev_output_dir = os.path.join(output_dir, "bev_channels")
    os.makedirs(bev_output_dir, exist_ok=True)

    # Same env as run_traffic_sign_idm_gif: TrafficSignEnv + StopSignSpawnWrapper (map from MetaDrive default)
    env = create_env(
        seed=seed,
        traffic_density=traffic_density,
        horizon=max_steps,
        stop_sign_probability=stop_sign_probability,
        max_stop_signs=max_stop_signs,
        num_scenarios=num_scenarios,
    )
    
    print(f"Loading CaRL model from: {checkpoint_path}")
    adapter = CaRLMetaDriveAdapter(checkpoint_path, device=device)
    
    if use_shields:
        print("Initializing safety shields...")
        stop_fx = StopRuleFeatureExtractor(
            dt=0.1, 
            dist_norm_max_m=50.0, 
            speed_norm_max_mps=20.0,
            required_hold_time_s=1.0, 
            approach_dist_m=15.0,
        )
        stop_shield = StopRuleShield(
            stop_fx,
            StopShieldConfig(
                max_brake_norm=0.6,
                hold_brake_norm=0.4,
                release_accel_norm=0.5,
                hold_dist_m=0.5,
                post_clear_dist_m=2.0,
            ),
        )
        speed_fx = SpeedLimitFeatureExtractor()
        speed_shield = SpeedLimitShield(speed_fx, SpeedLimitShieldConfig())
        print("  Stop sign shield initialized")
        print("  Speed limit shield initialized")
    else:
        print("Running baseline CaRL")
        stop_fx = None
        stop_shield = None
        speed_fx = None
        speed_shield = None
    
    episode_stats = []
    base_env = getattr(env, "unwrapped", env)

    if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
        if hasattr(base_env.top_down_renderer, "_screen_frames"):
            base_env.top_down_renderer._screen_frames.clear()

    for ep in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{num_episodes}")
        print(f"{'='*60}")
  
        if hasattr(base_env.engine, 'traffic_sign_manager'):
            base_env.engine.traffic_sign_manager.before_reset()
        
        obs, info = env.reset()
        adapter.reset()
        
        if use_shields:
            stop_fx.reset()
            speed_fx.reset()
        
        # Parse signs
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
        else:
            print("No signs specified. Running without traffic signs.")
        
        # Start recording for this episode (headless; no window)
        if save_gif:
            if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
                if hasattr(base_env.top_down_renderer, "_screen_frames"):
                    base_env.top_down_renderer._screen_frames.clear()
        
        ep_reward = 0
        ep_length = 0
        shield_interventions = {"stop": 0, "speed": 0}
        
        sign_mgr = base_env.engine.traffic_sign_manager if hasattr(base_env.engine, 'traffic_sign_manager') else None
        if sign_mgr:
            sign_mgr.reset_violations()
        
        for step in range(max_steps):
            # Get CaRL observation
            carl_obs = adapter.get_observation(base_env.agent, base_env.engine)
            bev = carl_obs["bev_semantics"]
            
            obs_tensor = {
                "bev_semantics": torch.from_numpy(carl_obs["bev_semantics"][None, ...]).to(adapter.device, dtype=torch.float32),
                "measurements": torch.from_numpy(carl_obs["measurements"][None, ...]).to(adapter.device, dtype=torch.float32),
                "value_measurements": torch.from_numpy(carl_obs["value_measurements"][None, ...]).to(adapter.device, dtype=torch.float32),
            }
            with torch.no_grad():
                base_action = adapter.model.forward(obs_tensor, deterministic=True, lstm_state=None, done=None)[0].squeeze().cpu().numpy().astype(np.float32)
            
            if use_shields:
                z_stop = stop_fx.step_features(base_env.engine, base_env.agent)
                z_speed = speed_fx.step_features(base_env.engine, base_env.agent)
            else:
                z_stop = np.zeros(8, dtype=np.float32)
                z_speed = np.zeros(5, dtype=np.float32)
            
            # apply shields if enabled
            safe_action = base_action.copy()
            shield_active = False
            stop_shield_active = False
            if use_shields:
                
                # stop shield 
                safe_action, clip, stop_info = stop_shield.clip_action(z_stop, safe_action)
                if np.sum(np.abs(clip)) > 1e-6:
                    shield_interventions["stop"] += 1
                    shield_active = True
                    stop_shield_active = stop_info['must_stop'] and not stop_info['stop_cleared']
                    if step % 10 == 0:
                        print(f"  [Shield] Stop sign intervention at step {step} "
                              f"(must_stop={stop_info['must_stop']}, dist={stop_info['dist_to_stop_m']:.1f}m)")

                if base_env.agent.speed < 0.5 and stop_fx.state.stop_cleared:
                    safe_action[0] = max(safe_action[0], 0.3)   
                
                safe_action, clip, info = speed_shield.clip_action(z_speed, safe_action, stop_shield_active=stop_shield_active)
                if np.sum(np.abs(clip)) > 1e-6:
                    shield_interventions["speed"] += 1
                    shield_active = True
                    if step % 10 == 0:
                        print(f"  [Shield] Speed limit intervention at step {step} "
                              f"(speed: {info['ego_speed_mps']:.1f}, limit: {info['limit_speed_mps']:.1f} m/s)")
            
            # update adapter to metadrive action 
            adapter._last_action = safe_action.copy()
            md_action = adapter.carl_action_to_metadrive(base_env.agent, safe_action, base_env.engine)
            
            # action
            obs, reward, terminated, truncated, info = base_env.step(md_action)
            
            current_speed_kmh = base_env.agent.speed * 3.6
            
            stop_status = "N/A"
            if use_shields and z_stop[stop_fx.IDX_MUST_STOP] > 0.1:
                if hasattr(stop_fx, 'state') and stop_fx.state.stop_cleared:
                    stop_status = "COMPL"
                else:
                    stop_status = "APPROACH"
            
            speed_status = "N/A"
            limit_display = ""
            if use_shields and z_speed[speed_fx.IDX_IN_ZONE] > 0.5:
                limit_speed_kmh = float(z_speed[speed_fx.IDX_LIMIT_SPEED]) * speed_fx.speed_norm_max_mps * 3.6
                limit_display = f"/{limit_speed_kmh:.0f}"
                speed_status = f"ZONE{limit_display}"
            
            # Check violations using TrafficSignManager
            pre_stop_status = stop_status
            pre_speed_status = speed_status
            if sign_mgr:
                stop_violated, speed_violated, stop_status, speed_status = sign_mgr.track_violations(
                    base_env.agent, step, stop_status, speed_status,
                    stop_sign_class=StopSign, speed_limit_sign_class=SpeedLimitSign
                )
            else:
                stop_violated = False
                speed_violated = False

            if os.environ.get("TRAFFIC_SIGN_VIOLATIONS_DEBUG", "").lower() in {"1", "true", "yes", "y"}:
                try:
                    debug_every = int(os.environ.get("TRAFFIC_SIGN_VIOLATIONS_EVERY", "1"))
                except Exception:
                    debug_every = 1
                if step % max(1, debug_every) == 0:
                    limit_speed_kmh = float(z_speed[speed_fx.IDX_LIMIT_SPEED]) * speed_fx.speed_norm_max_mps * 3.6
                    print(
                        f"[gif-debug] step={step} speed={current_speed_kmh:.1f}km/h "
                        f"z_in_zone={z_speed[speed_fx.IDX_IN_ZONE]:.2f} limit={limit_speed_kmh:.1f} "
                        f"pre_stop={pre_stop_status} pre_speed={pre_speed_status} "
                        f"post_stop={stop_status} post_speed={speed_status} "
                        f"stop_viol={stop_violated} speed_viol={speed_violated}"
                    )
            
            if save_bev and (step % 50 == 0 or step < 5 or terminated or truncated):
                bev_vis = adapter.visualize_bev(bev)
                bev_path = os.path.join(bev_output_dir, f"ep{ep:02d}_step{step:04d}_bev_grid.png")
                cv2.imwrite(bev_path, bev_vis)
                if step % 50 == 0:
                    print(f"  Step {step}: Saved BEV channels")
            
            render_text = {
                "Speed": f"{current_speed_kmh:.1f}{limit_display} km/h",
                "Stop": stop_status,
                "SpeedLim": speed_status,
                "Shield": "ON" if shield_active else "OFF",
                "Step": step,
            }
            if os.environ.get("TRAFFIC_SIGN_VIOLATIONS_DEBUG", "").lower() in {"1", "true", "yes", "y"}:
                render_text["Viol"] = f"{'S' if stop_violated else '-'}{'V' if speed_violated else '-'}"
            if save_gif:
                base_env.render(
                    mode="top_down", 
                    screen_record=True, 
                    window=False, 
                    screen_size=(800, 800),
                )
            
            ep_reward += reward
            ep_length += 1
            
            if terminated or truncated:
                print(f"  Episode ended. Terminated: {terminated}, Truncated: {truncated}")
                if save_bev:
                    final_bev_vis = adapter.visualize_bev(bev)
                    final_bev_path = os.path.join(bev_output_dir, f"ep{ep:02d}_final_bev_grid.png")
                    cv2.imwrite(final_bev_path, final_bev_vis)
                    print(f"  Saved final BEV to {final_bev_path}")
                break
        
        # episode summary
        print(f"\nEpisode finished. Steps: {ep_length}. Reward: {ep_reward:.2f}")
        if total_signs_added > 0:
            violations_count = len(sign_mgr.violations) if sign_mgr else 0
            print(f"  Traffic signs: {total_signs_added} ({result_counts})")
            print(f"  Violations: {violations_count}/{total_signs_added}")
            if sign_mgr and sign_mgr.violation_details:
                print(f"  Violation details:")
                for v_key, v_type, v_step in sign_mgr.violation_details:
                    print(f"    - {v_type} at step {v_step}")
        if use_shields:
            print(f"  Shield interventions: Stop={shield_interventions['stop']}, Speed={shield_interventions['speed']}")
        
        violations_count = len(sign_mgr.violations) if sign_mgr else 0
        episode_stats.append({
            "episode": ep + 1,
            "length": ep_length,
            "reward": ep_reward,
            "violations": violations_count,
            "total_signs": total_signs_added,
            "shield_interventions": shield_interventions.copy(),
        })
        
        if save_gif and getattr(base_env, "top_down_renderer", None) is not None:
            frames = getattr(base_env.top_down_renderer, "_screen_frames", None)
            if frames and len(frames) > 0:
                os.makedirs(output_dir, exist_ok=True)
                shield_suffix = "_shield" if use_shields else "_noshield"
                gif_path = os.path.join(output_dir, f"ep{ep:02d}_topdown{shield_suffix}.gif")
                base_env.top_down_renderer.generate_gif(gif_name=gif_path, duration=50)
                print(f"  Saved top-down GIF: {gif_path}")
            if hasattr(base_env.top_down_renderer, "clear"):
                base_env.top_down_renderer.clear()
            elif hasattr(base_env.top_down_renderer, "_screen_frames"):
                base_env.top_down_renderer._screen_frames.clear()

    env.close()
    
    print(f"\n{'='*60}")
    print("Overall Statistics:")
    print(f"{'='*60}")
    for stat in episode_stats:
        if stat['total_signs'] > 0:
            shield_info = ""
            if use_shields:
                shield_info = f", Shield interventions: Stop={stat['shield_interventions']['stop']}, Speed={stat['shield_interventions']['speed']}"
            print(f"Episode {stat['episode']}: Steps={stat['length']}, Reward={stat['reward']:.2f}, "
                  f"Violations={stat['violations']}/{stat['total_signs']}{shield_info}")
        else:
            print(f"Episode {stat['episode']}: Steps={stat['length']}, Reward={stat['reward']:.2f}, No signs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CaRL agent with safety shields for traffic signs"
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--save-bev", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--sign-counts",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for GIFs and BEV images (default: pdd-bench/outputs/carl_pdd_bench_shield)",
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        default=True,
        help="Record and save top-down GIF per episode (headless-friendly; default: True)",
    )
    parser.add_argument(
        "--no-save-gif",
        action="store_false",
        dest="save_gif",
        help="Disable GIF recording and saving",
    )
    parser.add_argument("--no-shields", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Env seed (same as run_traffic_sign_idm_gif)")
    parser.add_argument("--traffic-density", type=float, default=0.1)
    parser.add_argument("--num-scenarios", type=int, default=200)
    parser.add_argument("--stop-sign-probability", type=float, default=0.3, help="StopSignSpawnWrapper")
    parser.add_argument("--max-stop-signs", type=int, default=10, help="StopSignSpawnWrapper max signs")
    args = parser.parse_args()

    run_carl_with_shields(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        save_gif=args.save_gif,
        device=args.device,
        save_bev=args.save_bev,
        sign_counts=args.sign_counts,
        min_distance=args.min_distance,
        output_dir=args.output,
        use_shields=not args.no_shields,
        seed=args.seed,
        traffic_density=args.traffic_density,
        num_scenarios=args.num_scenarios,
        stop_sign_probability=args.stop_sign_probability,
        max_stop_signs=args.max_stop_signs,
    )
