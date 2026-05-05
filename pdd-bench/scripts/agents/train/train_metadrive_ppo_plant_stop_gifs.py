"""
PlanTwSign in MetaDrive — collect top-down GIF rollouts (no training).

Usage:
  python train_metadrive_ppo_plant_stop_gifs.py \\
      --checkpoint_file /path/to/plant.ckpt \\
      --num_gifs 5 --max_steps_per_episode 1000
"""
import os
import sys
import argparse
import random
import multiprocessing as mp
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


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
TRAIN_OUTPUT_DIR = PDD_BENCH_DIR / "outputs" / "metadrive_ppo_plant"

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


def load_plant_model(checkpoint_path, plant_planT_path, device="cpu"):
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
    from model import HFLM

    net = HFLM(config_net, config_all)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint contains no state_dict")
    if list(sd.keys())[0].startswith("model."):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    return net, config_all


class StopSignSpawnWrapper(gym.Wrapper):
    def __init__(self, env, stop_sign_probability=0.3, max_signs=10, min_distance=25.0):
        super().__init__(env)
        self.stop_sign_probability = stop_sign_probability
        self.max_signs = max_signs
        self.min_distance = min_distance

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Use reset seed for reproducible sign placement (e.g. when reconstructing same map in eval)
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


class PlanTObsWrapper(gym.Wrapper):
    def __init__(self, env, max_objects=30, max_stop_signs=10,
                 bev_resolution=128, bev_size_meters=64.0,
                 max_distance=75.0, range_factor_front=16.0):
        super().__init__(env)
        self.max_objects = max_objects
        self.max_stop_signs = max_stop_signs
        self.bev_resolution = bev_resolution
        self.bev_size_meters = bev_size_meters
        self.max_distance = max_distance
        self.range_factor_front = range_factor_front

        ps = max_objects + 1
        self.observation_space = gym.spaces.Dict({
            "bev":          gym.spaces.Box(-1, 2, (3, bev_resolution, bev_resolution), np.float32),
            "objects":      gym.spaces.Box(-500, 500, (ps, 7), np.float32),
            "obj_idxs":     gym.spaces.Box(0, ps, (max_objects,), np.float32),
            "route":        gym.spaces.Box(-500, 500, (20, 2), np.float32),
            "speed_limit":  gym.spaces.Box(0, 3, (1,), np.float32),
            "ego_speed":    gym.spaces.Box(-10, 100, (1,), np.float32),
            "stop_signs":   gym.spaces.Box(-500, 500, (max_stop_signs, 3), np.float32),
            "n_stop_signs": gym.spaces.Box(0, max_stop_signs, (1,), np.float32),
        })

    def _base(self):
        e = self.env
        while hasattr(e, "env"):
            e = e.env
        return e

    def _empty(self):
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

    def _build(self):
        from metadrive.policy.metadrive_obs_to_plant2 import (
            collect_objects_ego_frame, objects_to_x_batch,
            get_speed_limit_idx, render_bev_plant2, OBJ_TYPE_STOP_SIGN,
        )
        from metadrive.policy.plant_policy import get_route_points_ego_frame

        base = self._base()
        engine = base.engine
        ego = getattr(base, "vehicle", None) or getattr(base, "agent", None)
        if ego is None:
            return self._empty()

        objs = collect_objects_ego_frame(engine, ego, self.max_objects,
                                         self.max_distance, self.range_factor_front)
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


def create_env(
    seed=500,
    traffic_density=0.1,
    horizon=3000,
    stop_sign_probability=0.3,
    max_stop_signs=10,
    max_objects=30,
    bev_resolution=128,
    num_scenarios=200,
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
    env = StopSignSpawnWrapper(env, stop_sign_probability=stop_sign_probability,
                               max_signs=max_stop_signs)
    env = PlanTObsWrapper(env, max_objects=max_objects, max_stop_signs=max_stop_signs,
                          bev_resolution=bev_resolution)
    return env


def main():
    parser = argparse.ArgumentParser(description="PlanTwSign rollouts — save GIFs, no training")
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--plant_planT_path", type=str, default=str(PLANT_PLAN_T_DIR))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_gifs", type=int, default=5, help="Number of trajectories to dump as GIFs")
    parser.add_argument("--max_steps_per_episode", type=int, default=1000)
    parser.add_argument("--stop_sign_probability", type=float, default=0.3)
    parser.add_argument("--traffic_density", type=float, default=0.1)
    parser.add_argument("--num_scenarios", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = args.output_dir or str(TRAIN_OUTPUT_DIR / "plant_stop_gifs")
    os.makedirs(MODEL_PATH, exist_ok=True)

    print("=" * 70)
    print("PlanTwSign in MetaDrive — collect trajectories to GIFs (no training)")
    print(f"  Checkpoint:     {args.checkpoint_file}")
    print(f"  Stop sign prob: {args.stop_sign_probability}")
    print(f"  Traffic density:{args.traffic_density}")
    print(f"  Num scenarios:  {args.num_scenarios}")
    print(f"  Num GIFs:       {args.num_gifs}")
    print(f"  Max steps/ep:   {args.max_steps_per_episode}")
    print(f"  Device:         {DEVICE}")
    print(f"  Output:         {MODEL_PATH}")
    print("=" * 70)

    net, config_all = load_plant_model(args.checkpoint_file, args.plant_planT_path)
    net = net.to(DEVICE)
    net.eval()

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen_bev = sum(p.numel() for p in net.bev_encoder.parameters()) if hasattr(net, "bev_encoder") else 0
    print(f"[Model] Total: {total_params:,}  Trainable: {trainable_params:,}  Frozen BEV: {frozen_bev:,}")

    env = create_env(
        seed=args.seed,
        traffic_density=args.traffic_density,
        stop_sign_probability=args.stop_sign_probability,
        num_scenarios=args.num_scenarios,
    )

    # Use original PlanT (HFLM) + plant2_control to turn predicted waypoints into MetaDrive actions.
    from metadrive.policy.metadrive_obs_to_plant2 import metadrive_obs_to_plant2_batch
    from carla_garage.plant2_control import plant2_predictions_to_action, get_target_speed_from_limit

    num_gifs = args.num_gifs
    max_steps = args.max_steps_per_episode

    # Base MetaDrive env exposes render(mode=..., ...) and top_down_renderer; gymnasium wrappers do not
    base_env = env.unwrapped

    # Whether to feed BEV / ego speed based on PlanT config
    training_cfg = getattr(config_all.model, "training", {}) or {}
    input_bev = True
    input_ego_speed = bool(training_cfg.get("input_ego_speed", False))

    for ep in range(num_gifs):
        obs, info = env.reset()
        if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None and hasattr(base_env.top_down_renderer, "_screen_frames"):
            base_env.top_down_renderer._screen_frames.clear()

        ep_reward = 0.0
        for step in range(max_steps):
            ego = getattr(base_env, "vehicle", None) or getattr(base_env, "agent", None)
            if ego is None:
                break

            # Build PlanT batch directly from engine + ego state
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
                device=DEVICE,
            )

            with torch.no_grad():
                _, _, pred_plan, _ = net(batch)

            ego_speed = float(getattr(ego, "speed", 0.0))
            speed_limit_idx = int(batch["speed_limit"][0].item())
            target_speed_mps = get_target_speed_from_limit(speed_limit_idx)

            action = plant2_predictions_to_action(
                pred_plan,
                current_speed=ego_speed,
                target_speed_mps=target_speed_mps,
                speed_limit_idx=speed_limit_idx,
                speed_limits_kmh=(50, 80, 100, 120),
                device=DEVICE,
                return_waypoints=False,
            )

            action_np = np.asarray(action, dtype=np.float32)
            action_np = np.clip(action_np, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action_np)
            ep_reward += float(reward)

            base_env.render(
                mode="top_down",
                screen_record=True,
                window=False,
                screen_size=(640, 640),
            )

            if terminated or truncated:
                break

        gif_path = os.path.join(MODEL_PATH, f"traj_{ep + 1}.gif")
        if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
            base_env.top_down_renderer.generate_gif(gif_path, duration=10)
            print(f"[GIF] Saved episode {ep + 1}/{num_gifs} to {gif_path} (return={ep_reward:.1f})")
        else:
            print(f"[WARN] top_down_renderer missing; cannot save GIF for episode {ep + 1}")

    env.close()
    print("\nDone. All GIFs saved.")


if __name__ == "__main__":
    main()

