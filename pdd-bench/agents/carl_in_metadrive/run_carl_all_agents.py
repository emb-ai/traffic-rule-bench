#!/usr/bin/env python3
"""
Run CaRL agent (nuplan_51479_1B) for ALL agents in MetaDrive (ego + traffic).

Usage:
    python run_carl_all_agents.py
    python run_carl_all_agents.py --checkpoint /path/to/model.pth
    python run_carl_all_agents.py --no-render --episodes 5
"""

import sys
import os
import argparse
import numpy as np
import cv2
from typing import Dict, List

# Paths to CaRL and MetaDrive
CARL_BASE_PATH = "/home/jovyan/shares/SR006.nfs2/smirnova/CaRL/nuPlan"  # Parent directory containing carl_nuplan
CARL_NUPLAN_PATH = os.path.join(CARL_BASE_PATH, "carl_nuplan")  # Actual carl_nuplan module directory
METADRIVE_PATH = "/home/gbuhtuev/airi/pdd/sdc/metadrive"

# automatic -- i tried to make it more flexible
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
ADAPTER_PATH = os.path.dirname(os.path.abspath(__file__))               
# Alternative paths (commented out)
# CARL_BASE_PATH = os.path.join(BASE_DIR, "CaRL", "nuPlan")
# METADRIVE_PATH = os.path.join(BASE_DIR, "metadrive")

# Path to checkpoint (checkpoints are typically in the parent directory, not in carl_nuplan)
DEFAULT_CHECKPOINT = os.path.join(
    CARL_BASE_PATH, "checkpoints", "nuplan_51479_1B", "model_best.pth"
)

# Add parent directory to sys.path so we can import carl_nuplan
# We need CARL_BASE_PATH (not CARL_NUPLAN_PATH) so Python can find the carl_nuplan module
for path in [ADAPTER_PATH, CARL_BASE_PATH, METADRIVE_PATH]:
    if path not in sys.path:
        sys.path.insert(0, path)


def run_carl_all_agents(
    checkpoint_path: str,
    num_episodes: int = 1,
    max_steps: int = 5000,
    render: bool = False,
    save_bev: bool = True,
    output_dir: str = "./carl_all_agents_output",
    device: str = "cuda",
):
    """Run CaRL agent for ALL agents in MetaDrive using MultiAgentMetaDrive."""
    
    from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
    from carl_adapter import CaRLMetaDriveAdapter
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Environment configuration for multi-agent
    # Map: Roundabout (кольцо) - one roundabout block
    from metadrive.component.map.base_map import BaseMap
    from metadrive.component.algorithm.BIG import BigGenerateMethod
    
    env_config = {
        "use_render": render,
        "start_seed": 42,
        "num_scenarios": 1,
        "horizon": max_steps,
        "num_agents": 6,  # Number of agents to control
        "traffic_density": 0.0,  # No automatic traffic, we control all agents
        "allow_respawn": False,  # Don't respawn agents automatically
        
        # Map configuration for roundabout (кольцо)
        # "O" = Roundabout block
        "map_config": {
            BaseMap.GENERATE_TYPE: BigGenerateMethod.BLOCK_SEQUENCE,
            BaseMap.GENERATE_CONFIG: "O",  # One Roundabout block (кольцо)
            "exit_length": 50,  # Exit length for roundabout
        },
        
        "vehicle_config": {
            "spawn_lane_index": (">", ">>", 0),
        }
    }
    
    print(f"\nCreating MultiAgentMetaDrive environment...")
    env = MultiAgentMetaDrive(config=env_config)
    
    print(f"Loading CaRL model from: {checkpoint_path}")
    
    episode_stats = []
    
    for ep in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        obs, info = env.reset()
        
        # Get all agents (in multi-agent env, all vehicles are agents)
        agent_ids = list(env.agents.keys())
        print(f"Total agents: {len(agent_ids)}")
        
        # Create CaRL adapters for each agent
        agent_adapters: Dict[str, CaRLMetaDriveAdapter] = {}
        
        for agent_id in agent_ids:
            print(f"  Creating CaRL adapter for {agent_id}...")
            adapter = CaRLMetaDriveAdapter(checkpoint_path, device=device)
            adapter.reset()
            agent_adapters[agent_id] = adapter
        
        # Setup rendering - ensure top_down_renderer is initialized
        # For multi-agent, use film_size instead of screen_size
        env.render(
            mode="top_down",
            screen_record=True,
            window=False,
            film_size=(600, 600),  # Use film_size for multi-agent
        )
        # Verify top_down_renderer is available
        if hasattr(env, "top_down_renderer") and env.top_down_renderer is not None:
            print(f"  Top-down renderer initialized successfully")
        else:
            print(f"  Warning: Top-down renderer not initialized!")
        
        ep_reward = 0
        ep_length = 0
        all_terminated = False
        all_truncated = False
        
        print(f"Starting episode with max_steps={max_steps}")
        
        for step in range(max_steps):
            # Check for new agents that may have spawned (if respawn is enabled)
            current_agent_ids = list(env.agents.keys())
            for agent_id in current_agent_ids:
                if agent_id not in agent_adapters:
                    # New agent spawned, create adapter for it
                    print(f"  New agent {agent_id} detected, creating CaRL adapter...")
                    adapter = CaRLMetaDriveAdapter(checkpoint_path, device=device)
                    adapter.reset()
                    agent_adapters[agent_id] = adapter
            
            # Get actions for all agents from CaRL
            actions_dict = {}
            for agent_id in env.agents.keys():
                if agent_id in agent_adapters:
                    vehicle = env.agents[agent_id]
                    action = agent_adapters[agent_id].get_action(vehicle, env.engine)
                    actions_dict[agent_id] = action
                else:
                    # Fallback to zero action if adapter not ready
                    actions_dict[agent_id] = np.array([0.0, 0.0], dtype=np.float32)
            
            # Step environment with actions for all agents
            obs, reward, terminated, truncated, info = env.step(actions_dict)
            
            # Render - must be called on every step to capture frames
            # For multi-agent, use film_size and ensure screen_record is True
            env.render(
                mode="top_down",
                screen_record=True,
                window=False,
                film_size=(600, 600),  # Use film_size for multi-agent
            )
            
            # Calculate total reward (sum of all agents' rewards)
            total_reward = sum(reward.values()) if isinstance(reward, dict) else reward
            ep_reward += total_reward
            ep_length += 1
            
            # Debug and log (using first agent as example)
            if step % 50 == 0 and len(env.agents) > 0:
                first_agent_id = list(env.agents.keys())[0]
                vehicle = env.agents[first_agent_id]
                speed = getattr(vehicle, "speed_km_h", 0.0)
                
                if first_agent_id in agent_adapters:
                    carl_obs = agent_adapters[first_agent_id].get_observation(
                        vehicle, env.engine
                    )
                    bev = carl_obs["bev_semantics"]
                    bev_nonzero = [np.count_nonzero(bev[i]) for i in range(9)]
                    measurements = carl_obs["measurements"]
                    
                    action = actions_dict.get(first_agent_id, [0.0, 0.0])
                    
                    print(f"  Step {step:4d}: total_reward={total_reward:6.3f}, "
                          f"cumulative={ep_reward:7.2f}, active_agents={len(env.agents)}, "
                          f"first_agent_speed={speed:5.1f} km/h, "
                          f"first_agent_action=[{action[0]:.2f}, {action[1]:.2f}]")
                    print(f"    BEV non-zero pixels per channel: {bev_nonzero}")
                    print(f"    Measurements: vel_x={measurements[2]:.2f}, vel_y={measurements[3]:.2f}, "
                          f"steer={measurements[6]:.3f}, yaw_rate={measurements[8]:.3f}")
                
                # Debug info for first agent
                if isinstance(info, dict) and first_agent_id in info:
                    agent_info = info[first_agent_id]
                    crash = agent_info.get("crash", False) or agent_info.get("crash_vehicle", False) or agent_info.get("crash_object", False)
                    out_of_road = agent_info.get("out_of_road", False)
                    arrive_dest = agent_info.get("arrive_dest", False)
                    print(f"    Info (first agent): crash={crash}, out_of_road={out_of_road}, arrive_dest={arrive_dest}")
            
            # BEV visualization for first agent
            if save_bev and step % 10 == 0 and len(env.agents) > 0:
                first_agent_id = list(env.agents.keys())[0]
                if first_agent_id in agent_adapters:
                    vehicle = env.agents[first_agent_id]
                    carl_obs = agent_adapters[first_agent_id].get_observation(
                        vehicle, env.engine
                    )
                    vis = agent_adapters[first_agent_id].visualize_bev(
                        carl_obs["bev_semantics"]
                    )
                    cv2.imwrite(f"{output_dir}/ep{ep:02d}_step{step:04d}.png", vis)
            
            # Check if all agents are done
            if isinstance(terminated, dict):
                all_terminated = terminated.get("__all__", all(terminated.values()))
            else:
                all_terminated = terminated
                
            if isinstance(truncated, dict):
                all_truncated = truncated.get("__all__", all(truncated.values()))
            else:
                all_truncated = truncated
            
            if all_terminated or all_truncated:
                print(f"  Episode ended: terminated={all_terminated}, truncated={all_truncated}")
                break
        
        # Save GIF
        print(f"\nSaving GIF...")
        if hasattr(env, "top_down_renderer") and env.top_down_renderer is not None:
            frames = env.top_down_renderer._screen_frames
            print(f"  Number of frames captured: {len(frames) if frames else 0}")
            if frames and len(frames) > 0:
                # Check if frames are not empty
                first_frame = frames[0] if frames else None
                if first_frame is not None:
                    print(f"  First frame shape: {first_frame.shape if hasattr(first_frame, 'shape') else type(first_frame)}")
                    # Check if frame is not all white/black
                    if hasattr(first_frame, 'shape') and len(first_frame.shape) >= 2:
                        frame_sum = np.sum(first_frame) if isinstance(first_frame, np.ndarray) else 0
                        print(f"  First frame sum (should not be 0 or max): {frame_sum}")
                
                os.makedirs(output_dir, exist_ok=True)
                gif_path = os.path.join(output_dir, f"ep{ep + 1:02d}_topdown.gif")
                try:
                    env.top_down_renderer.generate_gif(gif_name=gif_path, duration=50)
                    print(f"  Saved top-down GIF: {gif_path}")
                except Exception as e:
                    print(f"  Error generating GIF: {e}")
            else:
                print(f"  Warning: No frames captured! Check if render() is being called correctly.")
            env.top_down_renderer.clear()
        else:
            print(f"  Warning: top_down_renderer not available!")
        
        episode_stats.append({
            "episode": ep,
            "reward": ep_reward,
            "length": ep_length,
            "terminated": all_terminated,
            "truncated": all_truncated,
        })
        
        print(f"\nEpisode {ep + 1} summary:")
        print(f"  Total reward: {ep_reward:.2f}")
        print(f"  Length: {ep_length} steps")
        print(f"  Final active agents: {len(env.agents)}")
        
        # Clean up adapters for this episode
        agent_adapters.clear()
    
    env.close()
    
    print(f"\n{'='*60}")
    print("Final Summary")
    print(f"{'='*60}")
    
    avg_reward = np.mean([s["reward"] for s in episode_stats])
    avg_length = np.mean([s["length"] for s in episode_stats])
    success_rate = np.mean([not s["terminated"] for s in episode_stats])
    
    print(f"Episodes: {num_episodes}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average length: {avg_length:.1f}")
    print(f"Success rate: {success_rate:.1%}")
    
    return episode_stats


def main():
    parser = argparse.ArgumentParser(description="Run CaRL agent for ALL agents in MetaDrive")
    
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to CaRL checkpoint"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=1,
        help="Number of episodes"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=1000,
        help="Max steps per episode"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering"
    )
    parser.add_argument(
        "--save-bev",
        action="store_true",
        help="Save BEV visualizations"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./carl_all_agents_output",
        help="Output directory"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        help="Torch device"
    )
    
    args = parser.parse_args()
    
    run_carl_all_agents(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        max_steps=args.steps,
        render=not args.no_render,
        save_bev=args.save_bev,
        output_dir=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
