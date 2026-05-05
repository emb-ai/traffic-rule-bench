"""
PPO agent training in MetaDrive with stop signs and custom rewards
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

DEFAULT_PRETRAINED_MODEL_PATH = (
    PDD_BENCH_DIR
    / "checkpoints"
    / "metadrive_custom"
    / "bev_no_stop_12000000_ts"
    / "ppo_model_6ch_20state.zip"
)

from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from metadrive.manager.traffic_manager import TrafficMode

from metadrive_core.bev_cnn import CustomBEVCNN
from metadrive_core.observation_wrappers import AddStateObservationWrapper
from metadrive_core.ppo_w_stop_sign_6ch.env_wrapper import TopDownMetaDriveWithStopSigns
from metadrive_core.ppo_w_stop_sign_6ch.reward_wrapper import CustomRewardWrapper

from metadrive_core.ppo_w_o_stop_sign.wrappers import EnsureSuccessInfoWrapper




def create_env(need_monitor=True, debug=False, traffic_density=0.1, seed=None, horizon=None, 
               use_custom_reward=True, stop_sign_probability=0.3):
    """
    Creates a MetaDrive environment with stop signs and a custom reward.

    need_monitor: Use Monitor wrapper
    debug: Enable debug output
    traffic_density
    seed
    horizon
    use_custom_reward: Use CustomRewardWrapper -- True
    stop_sign_probability: Probability of adding a sign to a lane 
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

        # Params of custom base term of reward
        success_reward=20.0,
        out_of_road_penalty=15.0,
        crash_vehicle_penalty=25.0,
        crash_object_penalty=20.0,
        crash_sidewalk_penalty=5.0,
        driving_reward=0.5,
        speed_reward=0.05,
    )
    
    if horizon is not None:
        config['horizon'] = horizon
    
    env = TopDownMetaDriveWithStopSigns(config)
    
    env = AddStateObservationWrapper(
        env,
        debug=debug,
        add_stop_signs=True,
        stop_sign_probability=stop_sign_probability,
        stop_sign_min_lane_length=15.0,
        stop_sign_max_signs=15,
        stop_sign_min_distance=25.0,
        include_stop_sign_radius=True,
    )
    
    if use_custom_reward:
        # meaning in reward_wrapper.py
        env = CustomRewardWrapper(
            env, 
            custom_reward_weight=1.0,
            R_far=50.0,           
            R_near=10.0,          
            max_speed_far=30.0,   
            max_speed_near=5.0,   
            speed_penalty_coef=-1.0,  
            stop_reward=2.0,     
            movement_penalty=-4.0 
        )
    # kostyl' for monitor and success rate
    env = EnsureSuccessInfoWrapper(env)
    
    if need_monitor:
        env = Monitor(env, info_keywords=("is_success",))
    return env


if __name__ == "__main__":
    # 2 modes: fine-tuning and from scratch
    if len(sys.argv) > 1 and '--fine-tune' in sys.argv:
        FINE_TUNE_MODE = True
    elif 'FINE_TUNE_MODE' in os.environ:
        FINE_TUNE_MODE = os.environ.get('FINE_TUNE_MODE', '0').lower() in ('1', 'true', 'yes')
    else:
        FINE_TUNE_MODE = False  # Default: train from scratch
        
    # Path to the converted model for fine-tuning (6 channels, 20 vectors)
    PRETRAINED_MODEL_PATH = os.environ.get(
        "PRETRAINED_MODEL_PATH",
        str(DEFAULT_PRETRAINED_MODEL_PATH),
    )

    set_random_seed(0)

    # Traffic density
    TRAFFIC_DENSITY = 0.1
    
    # env params
    ENV_HORIZON = 3000  
    BASE_SEED = 500  
    
    DEBUG_OBSERVATIONS = False

    TRAIN_TIMESTEPS = 5_000_000  
    
    # stop signs params
    STOP_SIGN_PROBABILITY = 0.3  
    USE_CUSTOM_REWARD = True 
    # save paths
    if FINE_TUNE_MODE:
        MODEL_PATH = str(TRAIN_OUTPUT_DIR / f"bev_stop_sign_6ch_{TRAIN_TIMESTEPS}_ts_fine_tune")
    else:
        TRAIN_TIMESTEPS = 12_000_000
        MODEL_PATH = str(TRAIN_OUTPUT_DIR / f"bev_stop_sign_6ch_{TRAIN_TIMESTEPS}_ts")

    os.makedirs(MODEL_PATH, exist_ok=True)
    
    
    
    # Atari hyperparameters:
    INITIAL_LEARNING_RATE = 0.00025/5  # Initial Learning rate: 0.00025
    NUM_ENVS = 64  # num envs: 8
    N_STEPS = 512  # env steps per iteration: 128
    BATCH_SIZE = 1024  # batch size: 1024
    N_EPOCHS = 3  # Epochs: 4
    CLIP_RANGE = 0.15  # clipping coefficient: 0.1
    GAMMA = 0.99  # discount factor: 0.99
    GAE_LAMBDA = 0.95  # GAE: 0.95
    ENT_COEF = 0.001  # entropy coefficient: 0.01
    VF_COEF = 0.5  # value function coefficient: 0.5
    MAX_GRAD_NORM = 0.5  # Max gradient norm: 0.5
    NORMALIZE_ADVANTAGE = True  # norm advantage
    CLIP_RANGE_VF = CLIP_RANGE  # clip value loss
    
    
    if not FINE_TUNE_MODE:
        INITIAL_LEARNING_RATE = 1e-4
        N_STEPS= 1024
        N_EPOCHS=6
        ENT_COEF=0.01
        TRAFFIC_DENSITY = 0.05
    # Learning rate schedule: linear -- lr(progress) = initial_lr * (1 - progress) 
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
    print(f"Mode: {'FINE-TUNING' if FINE_TUNE_MODE else 'TRAINING FROM SCRATCH'}")
    if FINE_TUNE_MODE:
        print(f"Pretrained model: {PRETRAINED_MODEL_PATH}")
    print(f"Training for: {TRAIN_TIMESTEPS:,} timesteps")
    print(f"Traffic density: {TRAFFIC_DENSITY} (fixed)")
    print(f"Environment settings:")
    print(f"  Horizon (max episode length): {ENV_HORIZON}")
    print(f"  Number of environments: {NUM_ENVS}")
    print(f"  Base seed: {BASE_SEED}")
    print(f"  Seed range (train): {BASE_SEED} to {BASE_SEED + NUM_ENVS - 1}")
    print(f"  Seed range (eval): {BASE_SEED + NUM_ENVS} to {BASE_SEED + NUM_ENVS + 7}")
    print(f"Stop signs: ENABLED (probability: {STOP_SIGN_PROBABILITY})")
    print(f"Custom rewards: {'ENABLED' if USE_CUSTOM_REWARD else 'DISABLED'} (using {'CustomRewardWrapper' if USE_CUSTOM_REWARD else 'default MetaDrive rewards'})")
    print(f"BEV channels: 6 (road_network, past_pos, traffic_flow, target_vehicle, additional, traffic_signs)")
    print(f"State features: includes stop_sign radius information")
    print(f"Output path: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    
    # Create a train environment 

    train_env = SubprocVecEnv([
        partial(
            create_env, 
            need_monitor=True,
            debug=(DEBUG_OBSERVATIONS and i == 0), 
            traffic_density=TRAFFIC_DENSITY,
            seed=BASE_SEED + i,  # seed for each 
            horizon=ENV_HORIZON,  
            use_custom_reward=USE_CUSTOM_REWARD,
            stop_sign_probability=STOP_SIGN_PROBABILITY,
        ) 
        for i in range(NUM_ENVS)
    ])

    # Model loading 
    if FINE_TUNE_MODE:
        # Fine-tuning
        print("Fine-tuning mode: Loading pretrained model...")
        print(f"Model path: {PRETRAINED_MODEL_PATH}")
        
        if os.path.exists(PRETRAINED_MODEL_PATH):
            try:
                model = PPO.load(
                    PRETRAINED_MODEL_PATH,
                    env=train_env,
                    device=DEVICE,
                    custom_objects={
                        "learning_rate": LEARNING_RATE,
                        "policy_kwargs": dict(features_extractor_class=CustomBEVCNN)
                    }
                )
                print("Pretrained model loaded successfully!")
                print(f"   Learning rate: {INITIAL_LEARNING_RATE}")
                
                # fine-tuning params
                model.learning_rate = LEARNING_RATE
                model.tensorboard_log = MODEL_PATH

        
            except Exception as e:
                print(f"Error loading pretrained model: {e}")
                import traceback
                traceback.print_exc()
                print("   Falling back to training from scratch...")
                FINE_TUNE_MODE = False  # if no fine-tuning --> train from scratch
                
        else:
            print(f"Pretrained model not found: {PRETRAINED_MODEL_PATH}")
            print("   Falling back to training from scratch...")
            FINE_TUNE_MODE = False # if no fine-tuning --> train from scratch
    
    if not FINE_TUNE_MODE:
        print("\n🏗️  Training from scratch: Creating new PPO model...")
        #  Atari hyperparams
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
            "tensorboard_log": MODEL_PATH
        }
        
        ppo_kwargs["clip_range_vf"] = CLIP_RANGE_VF
        
        model = PPO(**ppo_kwargs)
        

    if DEBUG_OBSERVATIONS:
        if hasattr(model.policy, 'features_extractor'):
            model.policy.features_extractor.set_debug(True)
    
    # eval environment
    eval_traffic_density = TRAFFIC_DENSITY
    eval_env = SubprocVecEnv([
        partial(
            create_env, 
            need_monitor=True,
            debug=False,
            traffic_density=eval_traffic_density,
            seed=1000 + NUM_ENVS + i, 
            horizon=ENV_HORIZON,
            use_custom_reward=USE_CUSTOM_REWARD,
            stop_sign_probability=STOP_SIGN_PROBABILITY,
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
