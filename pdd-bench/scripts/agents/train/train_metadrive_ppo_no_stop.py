"""
Train a PPO agent in MetaDrive without stop signs and with the default MetaDrive reward.
"""
import os
import sys
import torch
from pathlib import Path


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
TRAIN_OUTPUT_DIR = PDD_BENCH_DIR / "outputs" / "metadrive_ppo"

for path in (SDC_ROOT, METADRIVE_DIR, PDD_BENCH_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.envs.top_down_env import TopDownMetaDrive

from metadrive_core.bev_cnn import CustomBEVCNN
from metadrive_core.observation_wrappers import AddStateObservationWrapper
from metadrive_core.ppo_w_o_stop_sign.wrappers import EnsureSuccessInfoWrapper

def create_env(need_monitor=True, debug=False, traffic_density=0.1, seed=None, horizon=None):
    """
    Create a MetaDrive environment without stop signs and with the default reward.

    Args:
        need_monitor: wrap with Monitor
        debug:
        traffic_density:
        seed:
        horizon:
    """
    config = dict(
        num_scenarios=200,
        start_seed=seed if seed is not None else 500,
        log_level=50,
        use_render=False,
        random_lane_width=True,
        random_lane_num=True,
        traffic_density=traffic_density,
        traffic_mode=TrafficMode.Trigger,
    )
    
    if horizon is not None:
        config['horizon'] = horizon
    
    env = TopDownMetaDrive(config)
    
    env = AddStateObservationWrapper(
        env, 
        debug=debug,
        add_stop_signs=False, 
        stop_sign_probability=0.0,  
        stop_sign_min_lane_length=15.0
    )
    
    env = EnsureSuccessInfoWrapper(env)
    
    if need_monitor:
        env = Monitor(env, info_keywords=("is_success",))
    return env


if __name__ == "__main__":
    # Model output paths
    set_random_seed(0)
    # Atari hyperparameters from table 9 https://arxiv.org/pdf/2504.17838
    # Traffic density
    TRAFFIC_DENSITY = 0.25
    
    ENV_HORIZON = 3000  
    BASE_SEED = 500  
    
    DEBUG_OBSERVATIONS = False
    TRAIN_TIMESTEPS = 12_000_000  
    
    MODEL_PATH = str(TRAIN_OUTPUT_DIR / f"bev_no_stop_{TRAIN_TIMESTEPS}_ts")
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    
    
    # Atari hyperparameters:(a little bit changed)
    INITIAL_LEARNING_RATE = 0.00025  # Initial Learning rate: 0.00025
    NUM_ENVS = 64  # num envs: 8
    N_STEPS = 1024  # env steps per iteration: 128
    BATCH_SIZE = 1024  # batch size: 1024
    N_EPOCHS = 10  # Epochs: 4
    CLIP_RANGE = 0.2  # clipping coefficient: 0.1
    GAMMA = 0.99  # discount factor: 0.99
    GAE_LAMBDA = 0.95  # GAE : 0.95
    ENT_COEF = 0.001  # entropy coefficient: 0.01
    VF_COEF = 0.5  # value function coefficient: 0.5
    MAX_GRAD_NORM = 0.5  # Max gradient norm: 0.5
    NORMALIZE_ADVANTAGE = True  # norm advantage
    CLIP_RANGE_VF = CLIP_RANGE  # clip value loss
    
    # Learning rate schedule: linear 
    def linear_schedule(initial_value):
        def func(progress_remaining):
            return progress_remaining * initial_value
        return func
    
    LEARNING_RATE = linear_schedule(INITIAL_LEARNING_RATE)
    
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    else:
        DEVICE = "cpu"
        print("CUDA not available, using CPU")
    
    print("=" * 70)
    print("Training PPO agent in MetaDrive")
    print(f"Training for: {TRAIN_TIMESTEPS:,} timesteps")
    print(f"Traffic density: {TRAFFIC_DENSITY} (fixed)")
    print(f"Environment settings:")
    print(f"  Horizon (max episode length): {ENV_HORIZON}")
    print(f"  Number of environments: {NUM_ENVS}")
    print(f"  Base seed: {BASE_SEED}")
    print(f"  Seed range (train): {BASE_SEED} to {BASE_SEED + NUM_ENVS - 1}")
    print(f"  Seed range (eval): {BASE_SEED + NUM_ENVS} to {BASE_SEED + NUM_ENVS + 7}")
    print(f"Stop signs: DISABLED")
    print(f"Custom rewards: DISABLED (using default MetaDrive rewards)")
    print(f"Output path: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    
    train_env = SubprocVecEnv([
        partial(
            create_env, 
            need_monitor=True,
            debug=(DEBUG_OBSERVATIONS and i == 0), 
            traffic_density=TRAFFIC_DENSITY,
            seed=BASE_SEED + i,  # seed for each 
            horizon=ENV_HORIZON, 
        ) 
        for i in range(NUM_ENVS)
    ])

    print("\nCreating new PPO model with Atari hyperparameters...")
    ppo_kwargs = {
        "policy": "MultiInputPolicy",
        "env": train_env,
        "n_steps": N_STEPS,  # 128
        "batch_size": BATCH_SIZE,  # 1024
        "n_epochs": N_EPOCHS,  # 4
        "verbose": 1,
        "device": DEVICE,
        "learning_rate": LEARNING_RATE,  # linear schedule starting at 0.00025
        "gamma": GAMMA,  # 0.99
        "gae_lambda": GAE_LAMBDA,  # 0.95
        "clip_range": CLIP_RANGE,  # 0.1
        "ent_coef": ENT_COEF,  # 0.01
        "vf_coef": VF_COEF,  # 0.5
        "max_grad_norm": MAX_GRAD_NORM,  # 0.5
        "policy_kwargs": dict(features_extractor_class=CustomBEVCNN),
        "tensorboard_log": MODEL_PATH,
        "clip_range_vf": CLIP_RANGE_VF
    }
    
    model = PPO(**ppo_kwargs)
    
    print("Model created successfully!")
    print("\n=== Atari Hyperparameters ===")
    print(f"Initial learning rate: {INITIAL_LEARNING_RATE}")
    print(f"Learning rate schedule: linear")
    print(f"n_steps (env steps per iteration): {model.n_steps}")
    print(f"batch_size: {model.batch_size}")
    print(f"n_epochs: {model.n_epochs}")
    print(f"num_envs: {NUM_ENVS}")
    print(f"clip_range: {model.clip_range}")
    print(f"gamma: {model.gamma}")
    print(f"gae_lambda: {model.gae_lambda}")
    print(f"ent_coef: {model.ent_coef}")
    print(f"vf_coef: {model.vf_coef}")
    print(f"max_grad_norm: {model.max_grad_norm}")
    if hasattr(model, 'normalize_advantage'):
        print(f"normalize_advantage: {model.normalize_advantage}")
    if hasattr(model, 'clip_range_vf'):
        print(f"clip_range_vf: {model.clip_range_vf}")
    print("=" * 70)

    
    if DEBUG_OBSERVATIONS:
        if hasattr(model.policy, 'features_extractor'):
            model.policy.features_extractor.set_debug(True)
    
    # eval
    eval_traffic_density = TRAFFIC_DENSITY
    eval_env = SubprocVecEnv([
        partial(
            create_env, 
            need_monitor=True,
            debug=False,
            traffic_density=eval_traffic_density,
            seed=BASE_SEED + NUM_ENVS + i,  
            horizon=ENV_HORIZON,
        ) 
        for i in range(8) 
    ])
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_PATH, "best_model"),
        log_path=os.path.join(MODEL_PATH, "eval_logs"),
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1
    )
    
    callback_list = eval_callback

    print(f"\nStarting training for: {TRAIN_TIMESTEPS:,} steps")
    print()
    
    model.learn(
        total_timesteps=TRAIN_TIMESTEPS,
        log_interval=100,
        callback=callback_list,
        reset_num_timesteps=True
    )
    
    print("Training completed!")
    
    final_model_path = os.path.join(MODEL_PATH, "ppo_model")
    model.save(final_model_path)
    print(f"Model saved to: {final_model_path}")
    
    train_env.close()
    eval_env.close()
    
    print("\n" + "=" * 70)
    print("All done!")
    print(f"TensorBoard: {MODEL_PATH}")
    print("=" * 70)
