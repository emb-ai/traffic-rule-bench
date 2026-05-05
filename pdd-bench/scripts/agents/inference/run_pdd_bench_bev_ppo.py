#!/usr/bin/env python3
"""
Run BEV PPO agent with stop signs (pdd_bench).
"""

# CRITICAL: Set headless mode BEFORE any imports that might use pygame
# This must be done before MetaDrive imports pygame
import os
if 'SDL_VIDEODRIVER' not in os.environ:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

import argparse
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PDD_BENCH_DIR = SCRIPT_DIR.parents[2]
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"
DEFAULT_OUTPUT_DIR = PDD_BENCH_DIR / "outputs" / "pdd_bench_bev_ppo"

for path in (PDD_BENCH_DIR, METADRIVE_DIR, SDC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from metadrive.manager.traffic_manager import TrafficMode

from metadrive_core.bev_cnn import CustomBEVCNN
from metadrive_core.pdd_bench_bev_ppo.reward_wrapper import CustomRewardWrapper
from metadrive_core.ppo_w_stop_sign_6ch.env_wrapper import TopDownMetaDriveWithStopSigns
from metadrive_core.observation_wrappers import AddStateObservationWrapper
from metadrive_core.sign_utils import add_signs_by_count

# Import traffic sign classes
import importlib.util
import os

# Helper function to load sign classes
def load_sign_classes():
    """Load all available traffic sign classes"""
    sign_classes = {}
    
    # StopSign
    try:
        stop_sign_path = os.path.join(str(PDD_BENCH_DIR), "traffic_signs", "stop_sign.py")
        spec = importlib.util.spec_from_file_location("stop_sign", stop_sign_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sign_classes["2.5"] = module.StopSign
    except Exception as e:
        print(f"Warning: Could not load StopSign: {e}")
    
    # SpeedLimitSign
    try:
        speed_limit_path = os.path.join(str(PDD_BENCH_DIR), "traffic_signs", "speed_limit_sign.py")
        spec = importlib.util.spec_from_file_location("speed_limit_sign", speed_limit_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sign_classes["3.24_20"] = module.SpeedLimitSign20
        sign_classes["3.24_40"] = module.SpeedLimitSign40
        sign_classes["3.24_60"] = module.SpeedLimitSign60
    except Exception as e:
        print(f"Warning: Could not load SpeedLimitSign: {e}")
    
    # NoStoppingAllowedSign
    try:
        no_stopping_path = os.path.join(str(PDD_BENCH_DIR), "traffic_signs", "no_stopping_allowed_sign.py")
        spec = importlib.util.spec_from_file_location("no_stopping_allowed_sign", no_stopping_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sign_classes["3.27"] = module.NoStoppingAllowedSign
    except Exception as e:
        print(f"Warning: Could not load NoStoppingAllowedSign: {e}")
    
    # DirectionSign
    try:
        direction_path = os.path.join(str(PDD_BENCH_DIR), "traffic_signs", "direction_sign.py")
        spec = importlib.util.spec_from_file_location("direction_sign", direction_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sign_classes["5.15.2"] = module.DirectionSign
    except Exception as e:
        print(f"Warning: Could not load DirectionSign: {e}")
    
    return sign_classes


def visualize_channels_at_step(obs, step, episode, signs_count, save_dir):
    """Bev channels visualization at specific step"""
    if len(obs.shape) != 3:
        return None

    _, _, channels = obs.shape
    fig, axes = plt.subplots(1, channels, figsize=(5 * channels, 5))
    if channels == 1:
        axes = [axes]

    channel_names = []
    for i in range(channels):
        channel = obs[:, :, i]

        if i == 0:
            channel_name = "road_network"
        elif i == 1:
            channel_name = "past_pos"
        elif i == 2:
            channel_name = "traffic_flow"
        else:
            channel_name = f"channel_{i}"

        channel_names.append(channel_name)

        vmax = 1.0 if obs.max() <= 1.0 else 255
        im = axes[i].imshow(channel, cmap="gray", vmin=0, vmax=vmax)
        axes[i].set_title(
            f"Channel {i}: {channel_name}\nmin={channel.min():.3f}, max={channel.max():.3f}"
        )
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    plt.suptitle(
        f"Episode {episode}, Step {step}\nSTOP signs: {signs_count}", fontsize=14
    )
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"episode_{episode}_step_{step:03d}_channels.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return save_path



def run_model_and_visualize(
    model_path,
    num_episodes=1,
    max_steps=250,
    save_dir=None,
    max_signs=15,
    min_distance=15.0,
    stop_sign_probability=0.3,
    use_custom_reward=True,
    test_seed=None,
    save_bev_input=False,
    sign_counts=None,
):
    """
    Inference agents

    Args:
        model_path: path to a trained model
        num_episodes: number of episodes
        max_steps: max steps per episode
        save_dir: output directory
        use_custom_reward: use CustomRewardWrapper (True) or base reward (False)
        test_seed: fixed seed for reproducible episodes
        save_bev_input: save BEV input visualizations (True/False)
    """
    if save_dir is None:
        save_dir = str(DEFAULT_OUTPUT_DIR)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps: {max_steps}")
    print(f"Use CustomReward: {use_custom_reward}")
    print(f"Save BEV inputs: {save_bev_input}")
    if test_seed is not None:
        print(f"Test seed: {test_seed} (for reproducible maps)")
    print(f"Images will be saved to: {save_dir}")
    print()

    env_start_seed = test_seed if test_seed is not None else 1000

    env = TopDownMetaDriveWithStopSigns(
        dict(
            map="T",  # change if necessary
            num_scenarios=200,
            start_seed=env_start_seed,
            log_level=50,
            use_render=False,
            random_lane_width=True,
            random_lane_num=True,
            traffic_density=0.25,  # change if necessary
            traffic_mode=TrafficMode.Trigger,
        )
    )

    if use_custom_reward:
        env = CustomRewardWrapper(env, custom_reward_weight=1.0, stop_sign_penalty=-10.0)
    
    # Disable automatic sign addition in wrapper - we'll add signs manually after reset()
    env = AddStateObservationWrapper(
        env,
        debug=False,
        add_stop_signs=False,  # Disable automatic sign addition
        stop_sign_probability=0.0,
        stop_sign_min_lane_length=15.0,
        stop_sign_max_signs=0,
        stop_sign_min_distance=min_distance,
        include_stop_sign_radius=True,
    )

    base_env = env.unwrapped
    while hasattr(base_env, "env"):
        base_env = base_env.env

    try:
        model = PPO.load(
            model_path,
            device="cpu",
            custom_objects={
                "policy_kwargs": dict(features_extractor_class=CustomBEVCNN)
            },
        )
        print("Model loaded.")
    except Exception as exc:
        print(f"Failed to load model: {exc}")
        env.close()
        return

    # Load sign classes for parsing sign_counts
    sign_classes = load_sign_classes()
    # Import StopSign for default value
    try:
        stop_sign_path = os.path.join(str(PDD_BENCH_DIR), "traffic_signs", "stop_sign.py")
        spec = importlib.util.spec_from_file_location("stop_sign", stop_sign_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        StopSign = module.StopSign
    except Exception as e:
        print(f"Warning: Could not load StopSign: {e}")
        StopSign = None

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")

        if test_seed is not None:
            episode_seed = test_seed + episode
            obs_dict, _info = env.reset(seed=episode_seed)
            print(f"Seed episode: {episode_seed}")
        else:
            obs_dict, _info = env.reset()
        obs = obs_dict["image"]

        if test_seed is not None:
            random.seed(test_seed + episode)

        # Parse sign counts: "2.5:3,3.24_40:2,3.27:1" -> {StopSign: 3, SpeedLimitSign40: 2, ...}
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
        
        # Add signs only if sign_counts is specified
        total_signs_added = 0
        result_counts = {}
        if sign_counts_dict:
            print(f"Adding signs: {[(sc.__name__, cnt) for sc, cnt in sign_counts_dict.items()]}...")
            result_counts = add_signs_by_count(env, sign_counts_dict, min_distance=min_distance)
            total_signs_added = sum(result_counts.values())
            print(f"Signs added: {result_counts} (total: {total_signs_added})")
            
            # Verify signs are in manager
            if hasattr(base_env, 'engine') and hasattr(base_env.engine, 'traffic_sign_manager'):
                sign_mgr = base_env.engine.traffic_sign_manager
                if sign_mgr:
                    actual_signs = len([s for s in sign_mgr.signs if s is not None])
                    print(f"  Verified: {actual_signs} signs in traffic_sign_manager")
        else:
            print("No signs specified. Running without traffic signs.")
        
        # Count traffic signs for display
        signs_count = total_signs_added
        sign_type_counts = result_counts
        
        print(f"Observation shape: {obs.shape}")
        if sign_type_counts:
            print(f"Traffic signs on map: {signs_count} ({sign_type_counts})")
        else:
            print(f"Traffic signs on map: {signs_count}")

        # GIF render - initialize pygame display for headless mode BEFORE first render()
        # This is critical because renderer creates sign icons in _draw() which needs display
        try:
            import pygame
            if not pygame.get_init():
                pygame.init()
            if not pygame.display.get_init():
                pygame.display.init()
                pygame.display.set_mode((1, 1))
        except Exception as e:
            print(f"Warning: Failed to initialize pygame display: {e}")

        # GIF render - initialize screen recording
        if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
            base_env.top_down_renderer.screen_record = True
            base_env.top_down_renderer._screen_frames = []

        if save_bev_input:
            visualize_channels_at_step(obs, 0, episode + 1, signs_count, save_dir)

        if hasattr(base_env, "current_map") and base_env.current_map is not None:
            b_box = base_env.current_map.road_network.get_bounding_box()
            x_len = b_box[1] - b_box[0]
            y_len = b_box[3] - b_box[2]
        else:
            x_len = 0
            y_len = 0

        display_size_m = 100.0

        reward_wrapper = None
        temp_env = env
        while hasattr(temp_env, "env"):
            if isinstance(temp_env, CustomRewardWrapper):
                reward_wrapper = temp_env
                break
            temp_env = temp_env.env

        try:
            vehicle = None
            if hasattr(base_env, "vehicle"):
                vehicle = base_env.vehicle
            elif hasattr(base_env, "agents") and len(base_env.agents) > 0:
                vehicle = list(base_env.agents.values())[0]

            violation_recorded = reward_wrapper.episode_violations if reward_wrapper else 0

            text_info = {
                "Step": 0,
                "Reward": "0.00",
                "Total": "0.00",
            }

            if vehicle is not None:
                text_info["Position"] = f"({vehicle.position[0]:.1f}, {vehicle.position[1]:.1f})"
                text_info["Speed"] = f"{vehicle.speed:.1f} m/s"

            text_info["Violation"] = violation_recorded
            text_info["Vehicles"] = (
                len(base_env.engine.get_objects(lambda obj: hasattr(obj, "WIDTH")))
                if hasattr(base_env, "engine")
                else 0
            )
            text_info["Map"] = f"{x_len:.0f}m × {y_len:.0f}m"
            text_info["View"] = f"{display_size_m:.0f}m"

            # Ensure pygame display is initialized before render() to avoid "No video mode" error
            try:
                import pygame
                # Force reinitialize display to ensure it's active
                if pygame.display.get_init():
                    pygame.display.quit()
                pygame.display.init()
                pygame.display.set_mode((1, 1))
            except:
                pass

            # Ensure screen_record is enabled before rendering
            if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
                base_env.top_down_renderer.screen_record = True

            env.render(
                mode="topdown",
                screen_record=True,
                window=False,
                screen_size=(600, 600),
                camera_position=(50, 50),
                text=text_info
            )
        except Exception as e:
            pass  # Игнорируем ошибки рендеринга

        total_reward = 0.0
        cumulative_reward = 0.0
        violations = []
        violation_details = []  # Store detailed violation info
        
        # Get traffic sign manager for violation checking
        sign_mgr = None
        if hasattr(base_env, 'engine') and hasattr(base_env.engine, 'traffic_sign_manager'):
            sign_mgr = base_env.engine.traffic_sign_manager
        
        # Debug: print sign manager status
        if total_signs_added > 0:
            if sign_mgr:
                print(f"  Sign manager found. Signs count: {len(sign_mgr.signs) if sign_mgr.signs else 0}")
            else:
                print(f"  Warning: Sign manager not found!")
        
        for step in range(1, max_steps + 1):
            action, _ = model.predict(obs_dict, deterministic=True)

            obs_dict, reward, terminated, truncated, _info = env.step(action)
            obs = obs_dict["image"]
            total_reward += reward
            cumulative_reward += reward
            
            # Check violations if signs are present
            if total_signs_added > 0 and sign_mgr and sign_mgr.signs:
                vehicle = None
                if hasattr(base_env, "vehicle"):
                    vehicle = base_env.vehicle
                elif hasattr(base_env, "agents") and len(base_env.agents) > 0:
                    vehicle = list(base_env.agents.values())[0]
                
                if vehicle is not None:
                    try:
                        # Check violations directly using _is_violating to catch all violations
                        # This bypasses the deduplication in check_violation
                        for sign in sign_mgr.signs:
                            if sign is None:
                                continue
                            try:
                                # Check if currently violating (bypassing deduplication)
                                if hasattr(sign, '_is_violating'):
                                    is_violating_now = sign._is_violating(vehicle)
                                    if is_violating_now:
                                        sign_type = type(sign).__name__
                                        # Check if this is a new violation (not already recorded)
                                        violation_key = (sign.id if hasattr(sign, 'id') else id(sign), sign_type)
                                        if violation_key not in [v[0] for v in violation_details]:
                                            violations.append(sign)
                                            violation_details.append((violation_key, sign_type, step))
                                            print(f"  [VIOLATION] {sign_type} violated at step {step}!")
                            except Exception as e:
                                # Skip individual sign errors silently
                                pass
                        
                        # Also check using the standard method (for completeness)
                        violation_results = sign_mgr.check_all_violations(vehicle, for_reward=False)
                        for sign, violated in violation_results:
                            if violated:
                                sign_type = type(sign).__name__
                                violation_key = (sign.id if hasattr(sign, 'id') else id(sign), sign_type)
                                if violation_key not in [v[0] for v in violation_details]:
                                    violations.append(sign)
                                    violation_details.append((violation_key, sign_type, step))
                                    print(f"  [VIOLATION] {sign_type} violated at step {step} (via check_all_violations)!")
                    except Exception as e:
                        # Only print error once to avoid spam
                        if step == 1:
                            print(f"  Warning: Error checking violations: {e}")
                            import traceback
                            traceback.print_exc()

            try:
                vehicle = None
                if hasattr(base_env, "vehicle"):
                    vehicle = base_env.vehicle

                violation_recorded = reward_wrapper.episode_violations if reward_wrapper else 0

                text_info = {
                    "Step": step,
                    "Reward": f"{reward:.2f}",
                    "Cumulative": f"{cumulative_reward:.2f}",
                    "Total": f"{total_reward:.2f}",
                }

                if vehicle is not None:
                    text_info["Position"] = (
                        f"({vehicle.position[0]:.1f}, {vehicle.position[1]:.1f})"
                    )
                    text_info["Speed"] = f"{vehicle.speed:.1f} m/s"

                text_info["Violation"] = violation_recorded

                # Ensure pygame display is initialized before render() to avoid "No video mode" error
                # Only initialize every N steps to avoid performance issues
                if step % 10 == 0:
                    try:
                        import pygame
                        # Force reinitialize display to ensure it's active
                        if pygame.display.get_init():
                            pygame.display.quit()
                        pygame.display.init()
                        pygame.display.set_mode((1, 1))
                    except:
                        pass

                # Ensure screen_record is enabled before rendering
                if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
                    base_env.top_down_renderer.screen_record = True

                env.render(
                    mode="topdown",
                    screen_record=True,
                    window=False,
                    screen_size=(600, 600),
                    camera_position=(50, 50),
                    text=text_info
                )
            except Exception as e:
                pass  # Игнорируем ошибки рендеринга

            if save_bev_input and step % 10 == 0:
                visualize_channels_at_step(obs, step, episode + 1, signs_count, save_dir)
                print(f"  Step {step}: saved, reward={total_reward:.2f}")

            if terminated or truncated:
                print(f"  Episode finished at step {step}")
                print(f"     Total reward: {total_reward:.2f}")
                break

        # Save GIF if rendered (exactly like in working code)
        if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
            try:
                frames = base_env.top_down_renderer._screen_frames
                num_frames = len(frames) if frames else 0
                if num_frames > 0:
                    base_env.top_down_renderer.generate_gif(gif_name="demo.gif", duration=30)

                    # Copy working code approach: save to demo.gif first, then rename
                demo_gif_path = "demo.gif"
                if os.path.exists(demo_gif_path):
                        save_path = os.path.join(save_dir, f"{episode:03d}")
                        os.makedirs(save_path, exist_ok=True)
                        gif_path = os.path.join(save_path, f"episode_{episode + 1}_run.gif")
                    os.rename(demo_gif_path, gif_path)
                        print(f"GIF: {gif_path} ({num_frames} frames)")
                    else:
                        print(f"demo.gif error")
                else:
                    print(f"Warning: No frames recorded ({num_frames} frames, screen_record={base_env.top_down_renderer.screen_record})")

                base_env.top_down_renderer._screen_frames = []
            except Exception as e:
                import traceback
                traceback.print_exc()

        print("  Episode summary:")
        print(f"  Steps: {step}")
        print(f"  Total reward: {total_reward:.2f}")
        if total_signs_added > 0:
            print(f"  Traffic signs: {signs_count} ({sign_type_counts})")
            print(f"  Violations: {len(violations)}/{total_signs_added}")
            if violation_details:
                print(f"  Violation details:")
                for v_key, v_type, v_step in violation_details:
                    print(f"    - {v_type} at step {v_step}")
        else:
            print(f"  No traffic signs in this episode.")
        if save_bev_input:
            print(f"     Images saved: {step // 10 + 1}")

    env.close()

    print("\n" + "=" * 70)
    print(f"\n Results saved to: {save_dir}")
    if save_bev_input:
        print(f"   BEV channels format: episode_{episode + 1}_step_XXX_channels.png")
    print(f"   GIF: episode_{episode + 1}_run.gif")
    print("=" * 70)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BEV PPO inference with stop signs (pdd_bench)."
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to PPO model (.zip)",
    )
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=1,
        help="Number of episodes",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=250,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--max-signs",
        type=int,
        default=15,
        help="Maximum stop signs per episode",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=15.0,
        help="Minimum distance between stop signs",
    )
    parser.add_argument(
        "--stop-prob",
        type=float,
        default=0.3,
        help="Probability of adding a stop sign to a lane",
    )
    parser.add_argument(
        "--no-custom-reward",
        action="store_true",
        help="Disable custom reward wrapper",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fixed seed for reproducible episodes",
    )
    parser.add_argument(
        "--save-bev-input",
        action="store_true",
        help="Save BEV input visualizations",
    )
    parser.add_argument(
        "--sign-counts",
        type=str,
        default=None,
        help='Comma-separated "type:count" pairs, e.g. "2.5:3,3.24_40:2,3.27:1". Available types: 2.5 (Stop), 3.24_20/40/60 (SpeedLimit), 3.27 (NoStopping), 5.15.2 (Direction). If not specified, no signs will be added.'
    )

    args = parser.parse_args()

    run_model_and_visualize(
        model_path=args.model,
        num_episodes=args.episodes,
        max_steps=args.steps,
        save_dir=args.output,
        max_signs=args.max_signs,
        min_distance=args.min_distance,
        stop_sign_probability=args.stop_prob,
        use_custom_reward=not args.no_custom_reward,
        test_seed=args.seed,
        save_bev_input=args.save_bev_input,
        sign_counts=args.sign_counts,
    )


if __name__ == "__main__":
    main()
