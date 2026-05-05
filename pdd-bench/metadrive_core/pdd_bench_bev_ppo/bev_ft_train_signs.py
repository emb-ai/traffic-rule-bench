"""
Utilities for BEV PPO fine-tuning (pdd_bench).
"""
import sys
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
SDC_ROOT = FILE_PATH.parents[3]
PDD_BENCH_DIR = SDC_ROOT / "pdd-bench"
METADRIVE_DIR = SDC_ROOT / "metadrive"
for path in (PDD_BENCH_DIR, METADRIVE_DIR, SDC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from stable_baselines3.common.monitor import Monitor
from metadrive.manager.traffic_manager import TrafficMode

from metadrive_core.ppo_w_o_stop_sign.env_wrapper import TopDownMetaDriveWithStopSigns
from metadrive_core.pdd_bench_bev_ppo.reward_wrapper import CustomRewardWrapper
from metadrive_core.observation_wrappers import AddStateObservationWrapper


def create_env(need_monitor=True, debug=False, traffic_density=0.1,
               use_custom_reward=True, custom_reward_weight=1.0,
               add_stop_signs=True, stop_sign_probability=0.3, stop_sign_penalty=-10.0,
               speed_reward_coef=None, driving_reward_coef=None):

    config = dict(
        num_scenarios=200,
        start_seed=500,
        log_level=50,
        use_render=False,
        random_lane_width=True,
        random_lane_num=True,
        traffic_density=traffic_density,
        traffic_mode=TrafficMode.Trigger,
    )
    ### my heuristics
    if speed_reward_coef is not None:
        config['speed_reward'] = speed_reward_coef
    if driving_reward_coef is not None:
        config['driving_reward'] = driving_reward_coef
    
    env = TopDownMetaDriveWithStopSigns(config)
    
    if use_custom_reward:
        env = CustomRewardWrapper(env, custom_reward_weight=custom_reward_weight, 
                                  stop_sign_penalty=stop_sign_penalty)
    
    env = AddStateObservationWrapper(
        env, 
        debug=debug,
        add_stop_signs=add_stop_signs,
        stop_sign_probability=stop_sign_probability,
        stop_sign_min_lane_length=15.0
    )
    
    if need_monitor:
        env = Monitor(env, info_keywords=("is_success",))
    return env
    
