#!/usr/bin/env python3
"""
Collect K trajectories from CaRL policy in MetaDrive, with observations
formatted for Plant2 (PlanT) training.

Per step we store:
- Plant2-style observation dict (see PlanTObsWrapper)
- plant2_batch: batch dict from metadrive_obs_to_plant2_batch (same args as eval script) for exact replay
- Env action applied (MetaDrive action)
- Raw CaRL action before adapter (for reference)
- Reward
- Terminated / truncated flags

Per episode we also store:
- engine.current_map.road_network
- World coordinates of all stop signs spawned for that episode

Output: one .pt file per episode with a Python dict, saved via torch.save.
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import torch
import gymnasium as gym


# ---------------------------------------------------------------------------
# Headless setup for MetaDrive (no display needed)
# ---------------------------------------------------------------------------

if os.environ.get("SDL_VIDEODRIVER") is None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


# ---------------------------------------------------------------------------
# Path setup (mirrors existing train_* scripts)
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

for _p in (SDC_ROOT, METADRIVE_DIR, PDD_BENCH_DIR, PLANT_PLAN_T_DIR, ADAPTER_PATH, CARL_PATH):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)


from carl_adapter import CaRLMetaDriveAdapter  # type: ignore
from stop_rule_feat import StopRuleFeatureExtractor  # type: ignore
from stop_shield import StopRuleShield, StopShieldConfig  # type: ignore
from speed_limit_feat import SpeedLimitFeatureExtractor  # type: ignore
from speed_limit_shield import SpeedLimitShield, SpeedLimitShieldConfig  # type: ignore


# ---------------------------------------------------------------------------
# Stop sign spawn wrapper (reused for CaRL env)
# ---------------------------------------------------------------------------


class StopSignSpawnWrapper(gym.Wrapper):
    def __init__(self, env, stop_sign_probability: float = 0.3, max_signs: int = 10, min_distance: float = 25.0):
        super().__init__(env)
        self.stop_sign_probability = stop_sign_probability
        self.max_signs = max_signs
        self.min_distance = min_distance

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Use reset seed for reproducible sign placement when reconstructing the same map
        spawn_seed = kwargs.get("seed") if kwargs else None
        self._spawn_signs(seed=spawn_seed)
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


# ---------------------------------------------------------------------------
# PlanT-style observation wrapper (copied from PlanTObsWrapper in train scripts)
# ---------------------------------------------------------------------------


class PlanTObsWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        max_objects: int = 30,
        max_stop_signs: int = 10,
        bev_resolution: int = 128,
        bev_size_meters: float = 64.0,
        max_distance: float = 75.0,
        range_factor_front: float = 16.0,
    ):
        super().__init__(env)
        self.max_objects = max_objects
        self.max_stop_signs = max_stop_signs
        self.bev_resolution = bev_resolution
        self.bev_size_meters = bev_size_meters
        self.max_distance = max_distance
        self.range_factor_front = range_factor_front

        ps = max_objects + 1
        self.observation_space = gym.spaces.Dict(
            {
                "bev": gym.spaces.Box(-1, 2, (3, bev_resolution, bev_resolution), np.float32),
                "objects": gym.spaces.Box(-500, 500, (ps, 7), np.float32),
                "obj_idxs": gym.spaces.Box(0, ps, (max_objects,), np.float32),
                "route": gym.spaces.Box(-500, 500, (20, 2), np.float32),
                "speed_limit": gym.spaces.Box(0, 3, (1,), np.float32),
                "ego_speed": gym.spaces.Box(-10, 100, (1,), np.float32),
                "stop_signs": gym.spaces.Box(-500, 500, (max_stop_signs, 3), np.float32),
                "n_stop_signs": gym.spaces.Box(0, max_stop_signs, (1,), np.float32),
            }
        )

    def _base(self):
        e = self.env
        while hasattr(e, "env"):
            e = e.env
        return e

    def _empty(self) -> Dict[str, np.ndarray]:
        ps = self.max_objects + 1
        return {
            "bev": np.zeros((3, self.bev_resolution, self.bev_resolution), np.float32),
            "objects": np.zeros((ps, 7), np.float32),
            "obj_idxs": np.zeros(self.max_objects, np.float32),
            "route": np.zeros((20, 2), np.float32),
            "speed_limit": np.array([3.0], np.float32),
            "ego_speed": np.array([0.0], np.float32),
            "stop_signs": np.zeros((self.max_stop_signs, 3), np.float32),
            "n_stop_signs": np.array([0.0], np.float32),
        }

    def _build(self) -> Dict[str, np.ndarray]:
        from metadrive.policy.metadrive_obs_to_plant2 import (
            collect_objects_ego_frame,
            objects_to_x_batch,
            get_speed_limit_idx,
            render_bev_plant2,
            OBJ_TYPE_STOP_SIGN,
        )
        from metadrive.policy.plant_policy import get_route_points_ego_frame

        base = self._base()
        engine = base.engine
        ego = getattr(base, "vehicle", None) or getattr(base, "agent", None)
        if ego is None:
            return self._empty()

        objs = collect_objects_ego_frame(engine, ego, self.max_objects, self.max_distance, self.range_factor_front)
        xl, n_obj = objects_to_x_batch(objs, self.max_objects)
        ps = self.max_objects + 1
        while len(xl) < ps:
            xl.append([0.0] * 7)
        xl = xl[:ps]

        idxs = np.zeros(self.max_objects, np.float32)
        n = min(n_obj, self.max_objects)
        if n > 0:
            idxs[:n] = np.arange(1, 1 + n, dtype=np.float32)

        rt, _ = get_route_points_ego_frame(ego, 20)
        route = np.zeros((20, 2), np.float32)
        nr = min(len(rt), 20)
        route[:nr] = rt[:nr]
        if 0 < nr < 20:
            route[nr:] = route[nr - 1]

        bev_t = render_bev_plant2(engine, ego, self.bev_resolution, self.bev_size_meters)
        assert bev_t is not None
        bev_t = torch.rot90(bev_t, k=-1, dims=[2, 3])
        bev = bev_t.squeeze(0).numpy()

        stop_arr = np.zeros((self.max_stop_signs, 3), np.float32)
        ns = 0
        sign_mgr = getattr(engine, "traffic_sign_manager", None)
        if sign_mgr and hasattr(sign_mgr, "signs") and sign_mgr.signs:
            ego_pos = np.array(ego.position[:2])
            raw = []
            for s in sign_mgr.signs:
                if not hasattr(s, "position"):
                    continue
                rel = np.array([s.position[0] - ego_pos[0], s.position[1] - ego_pos[1]])
                loc = ego.convert_to_local_coordinates(rel, 0.0)
                raw.append((float(OBJ_TYPE_STOP_SIGN), float(loc[0]), -float(loc[1])))
            raw.sort(key=lambda r: r[1] ** 2 + r[2] ** 2)
            ns = min(len(raw), self.max_stop_signs)
            for i in range(ns):
                stop_arr[i] = raw[i]

        return {
            "bev": bev.astype(np.float32),
            "objects": np.array(xl, np.float32),
            "obj_idxs": idxs,
            "route": route,
            "speed_limit": np.array([float(get_speed_limit_idx(None))], np.float32),
            "ego_speed": np.array([float(getattr(ego, "speed", 0.0))], np.float32),
            "stop_signs": stop_arr,
            "n_stop_signs": np.array([float(ns)], np.float32),
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        plant_obs = self._build()
        return plant_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        plant_obs = self._build()
        return plant_obs, info


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def create_env(
    seed: int = 500,
    traffic_density: float = 0.1,
    horizon: int = 3000,
    stop_sign_probability: float = 0.3,
    max_stop_signs: int = 10,
    max_objects: int = 30,
    bev_resolution: int = 128,
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
    env = StopSignSpawnWrapper(env, stop_sign_probability=stop_sign_probability, max_signs=max_stop_signs)
    env = PlanTObsWrapper(env, max_objects=max_objects, max_stop_signs=max_stop_signs, bev_resolution=bev_resolution)
    return env


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------


DEFAULT_CARL_CHECKPOINT = str(CARL_PATH / "checkpoints" / "nuplan_51479_1B" / "model_best.pth")


def collect_trajectories(
    checkpoint_path: str,
    output_dir: str,
    num_episodes: int = 10,
    max_steps: int = 500,
    device: str = "cuda",
    seed: int = 42,
    traffic_density: float = 0.1,
    num_scenarios: int = 200,
    stop_sign_probability: float = 0.3,
    max_stop_signs: int = 10,
    use_shields: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    gif_dir = os.path.join(output_dir, "gifs")
    os.makedirs(gif_dir, exist_ok=True)

    # Fix global RNGs so collection is reproducible for the same seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        print("Initializing safety shields (StopRule + SpeedLimit)...")
        stop_fx = StopRuleFeatureExtractor(
            dt=0.1,
            dist_norm_max_m=50.0,
            speed_norm_max_mps=20.0,
            required_hold_time_s=2.0,
            approach_dist_m=5.0,
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
    else:
        stop_fx = None
        stop_shield = None
        speed_fx = None
        speed_shield = None

    base_env = getattr(env, "unwrapped", env)

    from metadrive.policy.metadrive_obs_to_plant2 import metadrive_obs_to_plant2_batch

    # Same batch args as eval_plant2_from_carl_trajectories_gifs.py for exact replay
    _input_bev = True
    _input_ego_speed = True

    speed_distribution = []

    for ep in range(num_episodes):
        print(f"\n{'=' * 60}")
        print(f"Episode {ep + 1}/{num_episodes}")
        print(f"{'=' * 60}")

        # Fixed seed per episode so the same map can be reconstructed later
        episode_seed = seed + ep
        obs, info = env.reset(seed=episode_seed)
        if use_shields:
            stop_fx.reset()
            speed_fx.reset()

        # Clear any previous recording frames if renderer exists
        if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
            if hasattr(base_env.top_down_renderer, "_screen_frames"):
                base_env.top_down_renderer._screen_frames.clear()

        # Gather static episode info: road network + spawned stop signs (world coordinates)
        road_network = getattr(base_env.current_map, "road_network", None)
        sign_mgr = getattr(base_env.engine, "traffic_sign_manager", None)
        stop_sign_world_positions: List[Any] = []
        if sign_mgr and hasattr(sign_mgr, "signs") and sign_mgr.signs:
            for s in sign_mgr.signs:
                if hasattr(s, "position"):
                    stop_sign_world_positions.append(np.array(s.position, dtype=np.float32))

        ep_data: Dict[str, Any] = {
            "episode_index": ep,
            "reset_seed": episode_seed,
            "base_seed": seed,
            "steps": [],
            "road_network": road_network,
            "stop_sign_world_positions": stop_sign_world_positions,
        }

        ep_reward = 0.0

        for step in range(max_steps):
            # Ground-truth ego state BEFORE action (aligns with plant2_batch / CaRL obs)
            ego_pos_before = np.asarray(getattr(base_env.agent, "position")[:2], dtype=np.float32)
            ego_heading_before = float(getattr(base_env.agent, "heading_theta", 0.0))

            # Build PlanT batch exactly like eval script (before step) for exact replay
            plant2_batch = metadrive_obs_to_plant2_batch(
                base_env.engine,
                base_env.agent,
                route_ego_20x2=None,
                speed_limit_kmh=None,
                max_objects=30,
                max_distance=75.0,
                range_factor_front=16.0,
                input_bev=_input_bev,
                input_ego_speed=_input_ego_speed,
                bev_resolution=128,
                bev_size_meters=64.0,
                device="cpu",
            )
            # Serialize batch for saving (tensors -> numpy)
            plant2_batch_save = {}
            for k, v in plant2_batch.items():
                if v is None:
                    plant2_batch_save[k] = None
                elif torch.is_tensor(v):
                    plant2_batch_save[k] = v.cpu().numpy()
                else:
                    plant2_batch_save[k] = v

            # Build CaRL observation from MetaDrive engine/agent
            carl_obs = adapter.get_observation(base_env.agent, base_env.engine)
            obs_tensor = {
                "bev_semantics": torch.from_numpy(carl_obs["bev_semantics"][None, ...]).to(
                    adapter.device, dtype=torch.float32
                ),
                "measurements": torch.from_numpy(carl_obs["measurements"][None, ...]).to(
                    adapter.device, dtype=torch.float32
                ),
                "value_measurements": torch.from_numpy(carl_obs["value_measurements"][None, ...]).to(
                    adapter.device, dtype=torch.float32
                ),
            }

            with torch.no_grad():
                base_action = (
                    adapter.model.forward(obs_tensor, deterministic=True, lstm_state=None, done=None)[0]
                    .squeeze()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            if use_shields:
                z_stop = stop_fx.step_features(base_env.engine, base_env.agent)
                z_speed = speed_fx.step_features(base_env.engine, base_env.agent)
                safe_action, _, stop_info = stop_shield.clip_action(z_stop, base_action.copy())
                stop_shield_active = stop_info.get("must_stop", False) and not stop_info.get("stop_cleared", True)
                if base_env.agent.speed < 0.5 and stop_fx.state.stop_cleared:
                    safe_action[0] = max(safe_action[0], 0.3)
                safe_action, _, _ = speed_shield.clip_action(z_speed, safe_action, stop_shield_active=stop_shield_active)
            else:
                safe_action = base_action.copy()

            adapter._last_action = safe_action.copy()
            md_action = adapter.carl_action_to_metadrive(base_env.agent, safe_action, base_env.engine)

            # Supervision target: ego speed (m/s) at this timestep (same moment as plant2_batch / CaRL obs)
            plant2_batch_save["target_speed"] = np.array([[float(getattr(base_env.agent, "speed", 0.0))]], dtype=np.float32)
            speed_distribution.append(float(getattr(base_env.agent, "speed", 0.0)))

            plant_obs, reward, terminated, truncated, info = env.step(md_action)
            ep_reward += float(reward)

            # Ground-truth ego state AFTER action (next state)
            ego_pos_after = np.asarray(getattr(base_env.agent, "position", [0.0, 0.0, 0.0])[:2], dtype=np.float32)
            ego_heading_after = float(getattr(base_env.agent, "heading_theta", 0.0))

            # Record top-down frame for GIF, headless-friendly
            if hasattr(base_env, "render"):
                base_env.render(
                    mode="top_down",
                    screen_record=True,
                    window=False,
                    screen_size=(640, 640),
                )

            step_record = {
                "obs": plant_obs,
                "plant2_batch": plant2_batch_save,
                "ego_pos_world_before": ego_pos_before,
                "ego_heading_before": ego_heading_before,
                "ego_pos_world_after": ego_pos_after,
                "ego_heading_after": ego_heading_after,
                "action_env": np.asarray(md_action, dtype=np.float32),
                "action_carl": base_action,
                "action_carl_safe": np.asarray(safe_action, dtype=np.float32),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "step_idx": np.array([float(step)], np.float32),
            }
            ep_data["steps"].append(step_record)

            if terminated or truncated:
                print(f"  Episode ended at step {step} (terminated={terminated}, truncated={truncated})")
                break

        # Build supervised targets for each step.
        # MetaDrive runs at 10 Hz, but the original HFLM nav_planner was calibrated for
        # ~4 Hz waypoints (factor-of-4 in _desired_speed_from_waypoints assumes 0.25 s
        # between waypoints).  We store TWO variants:
        #
        #   ego_pos_world_future_4       — consecutive steps i+1..i+4  (0.1 s each, 10 Hz)
        #                                  REFERENCE ONLY.  Do NOT use directly as
        #                                  pred_wps supervision targets — the 0.1 s interval
        #                                  is 2.5× too short for the HFLM speed calibration
        #                                  and causes crashes after fine-tuning.
        #
        #   ego_pos_world_future_4_s3    — stride-3 steps i+3, i+6, i+9, i+12  (~0.3 s each)
        #                                  These approximate the original 4 Hz interval
        #                                  (target = 0.25 s → stride-2.5 at 10 Hz).
        #                                  Use these for pred_wps supervision to match the
        #                                  HFLM's speed-estimation calibration.
        steps_list = ep_data["steps"]
        n = len(steps_list)
        if n > 0:
            pos_after = [np.asarray(s["ego_pos_world_after"], dtype=np.float32) for s in steps_list]
            for i in range(n):
                # Consecutive (reference)
                fut_consec = []
                for k in range(4):
                    j = i + k
                    fut_consec.append(pos_after[j] if j < n else pos_after[-1])
                steps_list[i]["ego_pos_world_future_4"] = np.stack(fut_consec, axis=0).astype(np.float32)

                # Stride-3 (~0.3 s each ≈ 4 Hz target)
                fut_s3 = []
                for k in range(1, 5):          # offsets: 3, 6, 9, 12
                    j = i + k * 3
                    fut_s3.append(pos_after[j] if j < n else pos_after[-1])
                steps_list[i]["ego_pos_world_future_4_s3"] = np.stack(fut_s3, axis=0).astype(np.float32)

        ep_data["return"] = ep_reward
        ep_data["num_steps"] = len(ep_data["steps"])

        out_path = os.path.join(output_dir, f"carl_plant2_traj_ep{ep:03d}.pt")
        torch.save(ep_data, out_path)
        print(f"Saved episode {ep + 1}/{num_episodes} to {out_path} (return={ep_reward:.2f})")

        # Save GIF for this episode if renderer is available
        if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
            gif_path = os.path.join(gif_dir, f"carl_plant2_traj_ep{ep:03d}.gif")
            try:
                base_env.top_down_renderer.generate_gif(gif_path, duration=10)
                print(f"Saved GIF for episode {ep + 1}/{num_episodes} to {gif_path}")
            except Exception as e:
                print(f"[WARN] Failed to save GIF for episode {ep + 1}: {e}")
            # Clear frames for the next episode if possible
            if hasattr(base_env.top_down_renderer, "clear"):
                base_env.top_down_renderer.clear()
            elif hasattr(base_env.top_down_renderer, "_screen_frames"):
                base_env.top_down_renderer._screen_frames.clear()

    env.close()
    print("\nDone. All trajectories saved.")
    plt.hist(speed_distribution, bins=10)
    plt.savefig("speed_distribution.png")
    plt.close()
    print("Saved speed distribution to speed_distribution.png")


def main():
    parser = argparse.ArgumentParser(
        description="Collect MetaDrive trajectories from CaRL policy with Plant2-style observations"
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CARL_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=10, help="Number of trajectories (episodes) to collect")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PDD_BENCH_DIR / "outputs" / "carl_plant2_trajectories"),
        help="Directory to save .pt trajectory files",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for CaRL model (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--traffic-density", type=float, default=0.1)
    parser.add_argument("--num-scenarios", type=int, default=200)
    parser.add_argument("--stop-sign-probability", type=float, default=0.3)
    parser.add_argument("--max-stop-signs", type=int, default=10)
    parser.add_argument(
        "--no-shields",
        action="store_true",
        help="Disable stop/speed-limit shields (run raw CaRL actions)",
    )
    args = parser.parse_args()

    collect_trajectories(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
        traffic_density=args.traffic_density,
        num_scenarios=args.num_scenarios,
        stop_sign_probability=args.stop_sign_probability,
        max_stop_signs=args.max_stop_signs,
        use_shields=not args.no_shields,
    )


if __name__ == "__main__":
    main()

