"""
PPO fine-tuning of PlanTwSign in MetaDrive — manual PPO, no stable_baselines3.

Env: TrafficSignEnv (reward includes stop-sign violations natively).
Signs: StopSignSpawnWrapper randomly places stop signs each episode.
Obs:  PlanTObsWrapper builds PlanT-format observations from engine state.
Model: PlanTActorCritic = PlanTwSign (frozen BEV) + actor/critic heads.

Usage:
  python train_metadrive_ppo_plant_stop.py \\
      --checkpoint_file /path/to/plant.ckpt \\
      --total_timesteps 500000 --num_envs 4
"""
import os
import sys
import argparse
import time
import random
import math
import multiprocessing as mp
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym

# ---------------------------------------------------------------------------
# Path setup
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
TRAIN_OUTPUT_DIR = PDD_BENCH_DIR / "outputs" / "metadrive_ppo_plant"

for _p in (SDC_ROOT, METADRIVE_DIR, PDD_BENCH_DIR, PLANT_PLAN_T_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)


# ===========================================================================
# Model loader
# ===========================================================================

def _mock_carla_modules():
    import unittest.mock as _mock
    for mod_name in ("carla", "agents", "agents.navigation",
                     "agents.navigation.global_route_planner"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _mock.MagicMock()


def load_plant_model(checkpoint_path, plant_planT_path, device="cpu"):
    """Load HFLM from config + checkpoint."""
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


# ===========================================================================
# PlanTwSign — PlanT with stop-sign module
# ===========================================================================

class PlanTwSign(nn.Module):
    """HFLM + new stop-sign layers (Embedding + Linear). Original class untouched."""

    def __init__(self, plant_model: nn.Module):
        super().__init__()
        self.plant = plant_model
        n_embd = plant_model.n_embd
        self.sign_type_emb = nn.Embedding(10, 32)
        self.sign_coord_linear = nn.Linear(3, 32)
        self.sign_fusion = nn.Linear(64, n_embd)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self.plant, "bev_encoder"):
            self.plant.bev_encoder.eval()
        return self

    def forward_features(self, batch: dict) -> torch.Tensor:
        plant = self.plant
        x_objs = batch["objects"]
        b_idxs = batch["obj_idxs"].long()
        route = batch["route"]
        spd_lim = batch["speed_limit"].long()
        if spd_lim.dim() == 2:
            spd_lim = spd_lim.squeeze(-1)

        B, pool, device = b_idxs.shape[0], x_objs.shape[1], x_objs.device

        emb = torch.zeros(B, pool, plant.n_embd, device=device)
        for i in range(len(plant.tok_emb)):
            m = x_objs[..., 0] == i
            if m.any():
                emb[m] = plant.tok_emb[i](x_objs[m][:, 1:])
        emb = torch.gather(emb, 1, b_idxs.unsqueeze(-1).expand(-1, -1, plant.n_embd))

        emb = torch.cat((plant.route_emb(route.flatten(1))[:, None], emb), 1)
        emb = torch.cat((plant.speed_emb(spd_lim)[:, None], emb), 1)

        if plant.input_ego_speed and "ego_speed" in batch:
            es = batch["ego_speed"]
            if es.dim() == 1:
                es = es.unsqueeze(-1)
            emb = torch.cat((plant.ego_speed_emb(es)[:, None], emb), 1)

        if plant.input_bev and "bev" in batch:
            with torch.no_grad():
                bev_tok = plant.bev_encoder(batch["bev"]).detach()
            emb = torch.cat((bev_tok[:, None], emb), 1)

        # stop-sign tokens
        signs = batch["stop_signs"]
        n_signs = batch["n_stop_signs"]
        if n_signs.dim() == 2:
            n_signs = n_signs.squeeze(-1)
        n_signs = n_signs.long()

        t_emb = self.sign_type_emb(signs[..., 0].long().clamp(0, 9))
        c_emb = self.sign_coord_linear(signs)
        s_emb = self.sign_fusion(torch.cat([t_emb, c_emb], -1))
        mask = torch.arange(s_emb.shape[1], device=device).unsqueeze(0) < n_signs.unsqueeze(1)
        s_emb = s_emb * mask.unsqueeze(-1).float()
        emb = torch.cat((s_emb, emb), 1)

        emb = torch.cat((plant.wp_token.expand(B, -1, -1), emb), 1)
        if plant.config_net.get("use_dropout", False):
            emb = plant.drop(emb)

        x = plant.model(inputs_embeds=emb, output_attentions=False).last_hidden_state
        return x.mean(1)


# ===========================================================================
# Actor-Critic wrapping PlanTwSign
# ===========================================================================

class PlanTActorCritic(nn.Module):
    def __init__(self, plant_w_sign: PlanTwSign, action_dim: int = 2, freeze_bev: bool = True):
        super().__init__()
        self.features = plant_w_sign
        n = plant_w_sign.plant.n_embd

        self.actor_mean = nn.Sequential(nn.Linear(n, 256), nn.Tanh(), nn.Linear(256, action_dim))
        self.actor_logstd = nn.Parameter(-0.5 * torch.ones(action_dim))
        self.critic = nn.Sequential(nn.Linear(n, 256), nn.Tanh(), nn.Linear(256, 1))

        if freeze_bev and hasattr(plant_w_sign.plant, "bev_encoder"):
            for p in plant_w_sign.plant.bev_encoder.parameters():
                p.requires_grad = False
            plant_w_sign.plant.bev_encoder.eval()

    def get_action_and_value(self, obs, action=None):
        feat = self.features.forward_features(obs)
        mean = self.actor_mean(feat)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return (
            action,
            dist.log_prob(action).sum(-1),
            dist.entropy().sum(-1),
            self.critic(feat).squeeze(-1),
        )

    def get_value(self, obs):
        feat = self.features.forward_features(obs)
        return self.critic(feat).squeeze(-1)


# ===========================================================================
# StopSignSpawnWrapper — randomly place stop signs on reset
# ===========================================================================

class StopSignSpawnWrapper(gym.Wrapper):
    def __init__(self, env, stop_sign_probability=0.3, max_signs=10, min_distance=25.0):
        super().__init__(env)
        self.stop_sign_probability = stop_sign_probability
        self.max_signs = max_signs
        self.min_distance = min_distance

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._spawn_signs()
        return obs, info

    def _spawn_signs(self):
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
        random.shuffle(lanes)
        added_pos, n_added = [], 0
        for lane in lanes:
            if n_added >= self.max_signs:
                break
            if random.random() >= self.stop_sign_probability:
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


# ===========================================================================
# PlanTObsWrapper — engine state ➜ PlanT-format Dict observation
# ===========================================================================

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
        self._ep_reward = 0.0
        self._ep_len = 0

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
        self._ep_reward += reward
        self._ep_len += 1

        if terminated or truncated:
            info["episode"] = {"r": self._ep_reward, "l": self._ep_len}
            base = self._base()
            sm = getattr(base.engine, "traffic_sign_manager", None)
            info["stop_signs_count"] = len(sm.signs) if sm else 0
            info["stop_sign_violations"] = len(sm.violations) if sm else 0
            self._ep_reward = 0.0
            self._ep_len = 0

        plant_obs = self._build()
        return plant_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ep_reward = 0.0
        self._ep_len = 0
        plant_obs = self._build()
        return plant_obs, info


# ===========================================================================
# Subprocess vectorized environment (no SB3)
# ===========================================================================

def _worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs, rew, term, trunc, info = env.step(data)
            if term or trunc:
                final_info = info.copy()
                obs, _ = env.reset()
                info = final_info
            remote.send((obs, rew, term, trunc, info))
        elif cmd == "reset":
            obs, info = env.reset()
            remote.send((obs, info))
        elif cmd == "spaces":
            remote.send((env.observation_space, env.action_space))
        elif cmd == "close":
            env.close()
            remote.close()
            break


def _stack_obs(obs_list):
    if isinstance(obs_list[0], dict):
        return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
    return np.stack(obs_list)


class SubprocVecEnv:
    """Minimal subprocess-based vectorized env (auto-resets on done)."""

    def __init__(self, env_fns):
        try:
            ctx = mp.get_context("forkserver")
        except ValueError:
            ctx = mp.get_context("spawn")
        self.n = len(env_fns)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n)])
        self.ps = []
        for wr, r, fn in zip(self.work_remotes, self.remotes, env_fns):
            p = ctx.Process(target=_worker, args=(wr, r, fn), daemon=True)
            p.start()
            self.ps.append(p)
        for wr in self.work_remotes:
            wr.close()
        self.remotes[0].send(("spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        results = [r.recv() for r in self.remotes]
        return _stack_obs([r[0] for r in results])

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        results = [r.recv() for r in self.remotes]
        obs = _stack_obs([r[0] for r in results])
        rews = np.array([r[1] for r in results], np.float32)
        dones = np.array([r[2] or r[3] for r in results], np.float32)
        infos = [r[4] for r in results]
        return obs, rews, dones, infos

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.ps:
            p.join(timeout=5)


# ===========================================================================
# GAE + PPO helpers
# ===========================================================================

def compute_gae(rewards, values, dones, next_values, gamma, lam):
    """GAE-Lambda. All arrays (n_steps, n_envs)."""
    T, N = rewards.shape
    adv = np.zeros((T, N), np.float32)
    last = np.zeros(N, np.float32)
    for t in reversed(range(T)):
        nv = next_values if t == T - 1 else values[t + 1]
        nnt = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nnt - values[t]
        last = delta + gamma * lam * nnt * last
        adv[t] = last
    return adv, adv + values


def obs_to_device(obs, device):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}


# ===========================================================================
# Environment factory
# ===========================================================================

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
    parser = argparse.ArgumentParser(description="PPO fine-tuning of PlanTwSign")
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--plant_planT_path", type=str, default=str(PLANT_PLAN_T_DIR))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.001)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--stop_sign_probability", type=float, default=0.3)
    parser.add_argument("--traffic_density", type=float, default=0.1)
    parser.add_argument("--num_scenarios", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = args.output_dir or str(TRAIN_OUTPUT_DIR / f"plant_stop_{args.total_timesteps}_ts")
    os.makedirs(MODEL_PATH, exist_ok=True)

    print("=" * 70)
    print("PPO Fine-Tuning: PlanTwSign in MetaDrive (manual PPO)")
    print(f"  Checkpoint:     {args.checkpoint_file}")
    print(f"  Timesteps:      {args.total_timesteps:,}")
    print(f"  Envs:           {args.num_envs}")
    print(f"  N steps:        {args.n_steps}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Stop sign prob: {args.stop_sign_probability}")
    print(f"  LR:             {args.lr}")
    print(f"  Device:         {DEVICE}")
    print(f"  Output:         {MODEL_PATH}")
    print("=" * 70)

    envs = SubprocVecEnv([
        partial(create_env, seed=500 + i,
                traffic_density=args.traffic_density,
                stop_sign_probability=args.stop_sign_probability,
                num_scenarios=args.num_scenarios)
        for i in range(args.num_envs)
    ])

    net, _ = load_plant_model(args.checkpoint_file, args.plant_planT_path)
    net.train()
    plant_w_sign = PlanTwSign(net)
    model = PlanTActorCritic(plant_w_sign, action_dim=2, freeze_bev=True).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_bev = sum(p.numel() for p in net.bev_encoder.parameters()) if hasattr(net, "bev_encoder") else 0
    print(f"[Model] Total: {total_params:,}  Trainable: {trainable_params:,}  Frozen BEV: {frozen_bev:,}")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, eps=1e-5)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join(MODEL_PATH, "tb"))

    N_STEPS = args.n_steps
    N_ENVS = args.num_envs
    obs_space = envs.observation_space
    buf_obs = {k: np.zeros((N_STEPS, N_ENVS, *sp.shape), np.float32) for k, sp in obs_space.items()}
    buf_actions = np.zeros((N_STEPS, N_ENVS, 2), np.float32)
    buf_logprobs = np.zeros((N_STEPS, N_ENVS), np.float32)
    buf_rewards = np.zeros((N_STEPS, N_ENVS), np.float32)
    buf_dones = np.zeros((N_STEPS, N_ENVS), np.float32)
    buf_values = np.zeros((N_STEPS, N_ENVS), np.float32)

    num_iterations = math.ceil(args.total_timesteps / (N_STEPS * N_ENVS))
    global_step = 0
    ep_rewards, ep_lengths, ep_violations, ep_sign_counts = [], [], [], []

    obs = envs.reset()
    start_time = time.time()

    print(f"\nTraining for {num_iterations} iterations ({args.total_timesteps:,} steps)\n")

    for iteration in range(1, num_iterations + 1):
        frac = 1.0 - (iteration - 1) / num_iterations
        lr_now = frac * args.lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        model.eval()
        for step in range(N_STEPS):
            for k in buf_obs:
                buf_obs[k][step] = obs[k]

            with torch.no_grad():
                obs_t = obs_to_device(obs, DEVICE)
                action, logprob, _, value = model.get_action_and_value(obs_t)
            action_np = action.cpu().numpy()
            action_np = np.clip(action_np, -1.0, 1.0)

            buf_actions[step] = action_np
            buf_logprobs[step] = logprob.cpu().numpy()
            buf_values[step] = value.cpu().numpy()

            obs, rewards, dones, infos = envs.step(action_np)
            buf_rewards[step] = rewards
            buf_dones[step] = dones
            global_step += N_ENVS

            for info in infos:
                if "episode" in info:
                    ep_rewards.append(info["episode"]["r"])
                    ep_lengths.append(info["episode"]["l"])
                if "stop_sign_violations" in info:
                    ep_violations.append(info["stop_sign_violations"])
                if "stop_signs_count" in info:
                    ep_sign_counts.append(info["stop_signs_count"])

        with torch.no_grad():
            next_val = model.get_value(obs_to_device(obs, DEVICE)).cpu().numpy()
        advantages, returns = compute_gae(buf_rewards, buf_values, buf_dones, next_val,
                                          args.gamma, args.gae_lambda)

        total = N_STEPS * N_ENVS
        flat_obs = {k: v.reshape(total, *v.shape[2:]) for k, v in buf_obs.items()}
        flat_act = buf_actions.reshape(total, 2)
        flat_lp = buf_logprobs.reshape(total)
        flat_adv = advantages.reshape(total)
        flat_ret = returns.reshape(total)

        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        model.train()
        pg_losses, v_losses, entropies, clip_fracs = [], [], [], []

        for _epoch in range(args.n_epochs):
            indices = np.random.permutation(total)
            for start in range(0, total, args.batch_size):
                end = min(start + args.batch_size, total)
                mb = indices[start:end]

                mb_obs = {k: torch.as_tensor(flat_obs[k][mb], device=DEVICE) for k in flat_obs}
                mb_act = torch.as_tensor(flat_act[mb], device=DEVICE)
                mb_old_lp = torch.as_tensor(flat_lp[mb], device=DEVICE)
                mb_adv = torch.as_tensor(flat_adv[mb], device=DEVICE)
                mb_ret = torch.as_tensor(flat_ret[mb], device=DEVICE)

                _, new_lp, entropy, new_val = model.get_action_and_value(mb_obs, mb_act)

                ratio = (new_lp - mb_old_lp).exp()
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * (new_val - mb_ret).pow(2).mean()
                ent = entropy.mean()

                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(ent.item())
                with torch.no_grad():
                    clip_fracs.append(((ratio - 1).abs() > args.clip_range).float().mean().item())

        if iteration % args.log_interval == 0:
            elapsed = time.time() - start_time
            sps = global_step / elapsed

            writer.add_scalar("train/policy_loss", np.mean(pg_losses), global_step)
            writer.add_scalar("train/value_loss", np.mean(v_losses), global_step)
            writer.add_scalar("train/entropy", np.mean(entropies), global_step)
            writer.add_scalar("train/clip_fraction", np.mean(clip_fracs), global_step)
            writer.add_scalar("train/learning_rate", lr_now, global_step)
            writer.add_scalar("train/steps_per_second", sps, global_step)

            msg = (f"[iter {iteration}/{num_iterations}] step={global_step:,}  "
                   f"pg={np.mean(pg_losses):.4f}  vf={np.mean(v_losses):.4f}  "
                   f"ent={np.mean(entropies):.4f}  clip={np.mean(clip_fracs):.3f}  "
                   f"lr={lr_now:.2e}  sps={sps:.1f}")

            if ep_rewards:
                mean_r = np.mean(ep_rewards[-100:])
                mean_l = np.mean(ep_lengths[-100:])
                writer.add_scalar("rollout/ep_reward_mean", mean_r, global_step)
                writer.add_scalar("rollout/ep_length_mean", mean_l, global_step)
                msg += f"  ep_r={mean_r:.1f}  ep_l={mean_l:.0f}"
            if ep_violations:
                mean_v = np.mean(ep_violations[-100:])
                writer.add_scalar("stop_signs/violations_mean", mean_v, global_step)
                msg += f"  viol={mean_v:.2f}"
            if ep_sign_counts:
                writer.add_scalar("stop_signs/count_mean", np.mean(ep_sign_counts[-100:]), global_step)

            print(msg)

        if iteration % args.save_interval == 0 or iteration == num_iterations:
            ckpt_path = os.path.join(MODEL_PATH, f"ppo_plant_iter{iteration}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iteration": iteration,
                "global_step": global_step,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    envs.close()
    writer.close()
    final_path = os.path.join(MODEL_PATH, "ppo_plant_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nDone. Final model: {final_path}")
    print(f"TensorBoard: {os.path.join(MODEL_PATH, 'tb')}")


if __name__ == "__main__":
    main()
