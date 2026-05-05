"""
CustomRewardWrapper:
-  STOP sign penalty
- Reward за progress
"""
import sys
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
SDC_ROOT = FILE_PATH.parents[3]
PDD_BENCH_DIR = SDC_ROOT / "pdd-bench"
METADRIVE_DIR = SDC_ROOT / "metadrive"

for path in (FILE_PATH.parent, PDD_BENCH_DIR, METADRIVE_DIR, SDC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import gymnasium as gym
from traffic_signs.stop_sign import StopSign


class CustomRewardWrapper(gym.RewardWrapper):
    """
    Params:
        custom_reward_weight: вес для кастомных reward (по умолчанию 1.0)
        stop_sign_penalty: штраф за нарушение знака STOP (по умолчанию -10.0)
    """
    
    def __init__(self, env, custom_reward_weight=1.0, stop_sign_penalty=-10.0):
        super().__init__(env)
        self.custom_reward_weight = custom_reward_weight
        self.stop_sign_penalty = stop_sign_penalty
        self.violation_count = 0
        self.episode_violations = 0
        # ID signs that were violated in the current episode
        self.violated_sign_ids = set()
        
    def reset(self, **kwargs):
        if self.episode_violations > 0:
            print(f"    Episode violations: {self.episode_violations}")
        self.episode_violations = 0
        self.violation_count = 0
        self.violated_sign_ids.clear()
        obs, info = self.env.reset(**kwargs)
        info['stop_sign_violations'] = self.episode_violations
        return obs, info
        
    def reward(self, reward):
        """
        Args:
            reward:  reward from MetaDrive
            
        Returns:
             reward = reward from MetaDrive + custom reward
        """
        base_env = self.unwrapped
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        vehicle = None
        if hasattr(base_env, 'vehicle'):
            vehicle = base_env.vehicle
        elif hasattr(base_env, 'agents') and len(base_env.agents) > 0:
            vehicle = list(base_env.agents.values())[0]
        
        if vehicle is None:
            return reward
        
        custom_reward = 0.0
        
        #  STOP sign violation
        try:
            if hasattr(base_env, 'engine') and hasattr(base_env.engine, 'traffic_sign_manager'):
                sign_mgr = base_env.engine.traffic_sign_manager
                if sign_mgr and len(sign_mgr.signs) > 0:
                    violations = sign_mgr.check_all_violations(vehicle, for_reward=True)
                    for sign, violated in violations:
                        if violated and isinstance(sign, StopSign):
                            sign_id = getattr(sign, 'id', id(sign))  
                            if sign_id not in self.violated_sign_ids:
                                custom_reward += self.stop_sign_penalty
                                self.violation_count += 1
                                self.episode_violations += 1
                                self.violated_sign_ids.add(sign_id)
        except Exception as e:
            pass
        # print("###"*10)
        # print(custom_reward, "custom_reward_stop_sign")
        # print("###"*10)
        try:
            if hasattr(vehicle, 'navigation') and hasattr(vehicle.navigation, 'route_completion'):
                route_completion = vehicle.navigation.route_completion
                route_reward = 0.02 * route_completion
                custom_reward += route_reward
        except:
            pass
        
        modified_reward = reward + (custom_reward * self.custom_reward_weight)
        # print("////"*5)
        # print("custom_reward", custom_reward)
        # print("modified_reward", modified_reward)
        # print("////"*5)
        return modified_reward
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    @property
    def top_down_renderer(self):
        """Pass through top_down_renderer property from wrapped environment"""
        return getattr(self.env, 'top_down_renderer', None)