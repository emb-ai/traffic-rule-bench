"""
CustomRewardWrapper:
- STOP sign penalty 
- Reward for progress
"""
import sys
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

for path in (SDC_ROOT, METADRIVE_DIR, PDD_BENCH_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import gymnasium as gym
import numpy as np
from traffic_signs.stop_sign import StopSign


class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, custom_reward_weight=1.0, 
                 R_far=70.0, R_near=15.0, max_speed_far=20.0, max_speed_near=5.0,
                 speed_penalty_coef=-1.0, stop_reward=2.0, movement_penalty=-4.0):
        super().__init__(env)
        self.custom_reward_weight = custom_reward_weight
        self.R_far = R_far
        self.R_near = R_near
        self.max_speed_far = max_speed_far
        self.max_speed_near = max_speed_near
        self.speed_penalty_coef = speed_penalty_coef
        self.stop_reward = stop_reward
        self.movement_penalty = movement_penalty
        
        self.violation_count = 0
        self.episode_violations = 0
        self.violated_sign_ids = set()
        self._prev_route_completion = 0.0
        self.total_violations = 0  # Total number of violations for all time
        self.stop_signs_count = 0  # Number of characters in the current episode
        
        # Save the last observation to access the radius to the sign
        self._last_observation = None
    
    def _count_stop_signs(self):
        try:
            base_env = self.unwrapped
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            if hasattr(base_env, 'engine') and hasattr(base_env.engine, 'traffic_sign_manager'):
                sign_mgr = base_env.engine.traffic_sign_manager
                if sign_mgr is not None:
                    signs = [sign for sign in sign_mgr.signs if isinstance(sign, StopSign)]
                    return len(signs)
        except Exception:
            pass
        return 0
        
    def reset(self, **kwargs):
        # statistics for the previous episode
        if self.episode_violations > 0:
            print(f"    Episode violations: {self.episode_violations}")
        
        self.episode_violations = 0
        self.violation_count = 0
        self.violated_sign_ids.clear()
        obs, info = self.env.reset(**kwargs)
        
        # number of characters in the new episode
        self.stop_signs_count = self._count_stop_signs()
        if self.stop_signs_count > 0:
            print(f"    Stop signs on map: {self.stop_signs_count}")
        
        info['stop_sign_violations'] = self.episode_violations
        info['stop_signs_count'] = self.stop_signs_count
        self._prev_route_completion = 0.0
        self._last_observation = obs
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_observation = obs
        modified_reward = self.reward(reward)
        
        info['stop_sign_violations'] = self.episode_violations
        info['stop_signs_count'] = self.stop_signs_count
        info['total_violations'] = self.total_violations
        
        return obs, modified_reward, terminated, truncated, info
        
    def reward(self, reward):
        """
        Args:
            reward: reward from MetaDrive
            
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
        
        # distance to the sign
        distance_to_sign = 1000.0  
        if self._last_observation is not None and "state" in self._last_observation:
            state = self._last_observation["state"]
            if len(state) > 19:
                distance_to_sign = float(state[19]) 
        
        speed_kmh = 0.0
        if hasattr(vehicle, 'speed_km_h'):
            speed_kmh = vehicle.speed_km_h
        elif hasattr(vehicle, 'speed'):
            speed_kmh = vehicle.speed * 3.6  # km/h
        
        # Stop sign reward depending on distance
        if distance_to_sign < 1000.0:
            stop_sign_reward = self._calculate_stop_sign_reward(
                distance_to_sign, speed_kmh
            )
            custom_reward += stop_sign_reward
        
        try:
            if hasattr(base_env, 'engine') and hasattr(base_env.engine, 'traffic_sign_manager'):
                sign_mgr = base_env.engine.traffic_sign_manager
                if sign_mgr and len(sign_mgr.signs) > 0:
                    violations = sign_mgr.check_all_violations(vehicle, for_reward=True)
                    for sign, violated in violations:
                        if violated and isinstance(sign, StopSign):
                            sign_id = getattr(sign, 'id', id(sign))  
                            if sign_id not in self.violated_sign_ids:
                                self.violation_count += 1
                                self.episode_violations += 1
                                self.total_violations += 1
                                self.violated_sign_ids.add(sign_id)
                                print(f" !!! Stop sign violation detected! Total violations in episode: {self.episode_violations}")
        except Exception as e:
            pass
        
        # Reward for progress 
        try:
            if (hasattr(vehicle, 'navigation') and 
                vehicle.navigation is not None and 
                hasattr(vehicle.navigation, 'route_completion')):
                route_completion = vehicle.navigation.route_completion
                delta = route_completion - self._prev_route_completion
                if delta > 0:
                    route_reward = 0.1 * delta
                    custom_reward += route_reward
                self._prev_route_completion = route_completion
        except Exception as e:
            pass
        
        modified_reward = reward + (custom_reward * self.custom_reward_weight)
        return modified_reward
    
    def _calculate_stop_sign_reward(self, distance, speed_kmh):
        """
        Stop sign reward based on distance and speed.

        Logic:
        - distance > R_far: nothing (0)
        - R_near < distance <= R_far: penalty for excessive speed
        - distance <= R_near: reward for slow/stopping, penalty for moving    
        """
        reward = 0.0
        
        if distance > self.R_far:
            return 0.0
        
        elif distance > self.R_near:
            # R_near < distance <= R_far
            if speed_kmh > self.max_speed_far:
                excess_speed = speed_kmh - self.max_speed_far
                # Normalize: the closer to the sign, the higher the fine
                distance_factor = 1.0 - ((distance - self.R_near) / (self.R_far - self.R_near))
                reward = self.speed_penalty_coef * excess_speed * distance_factor
        else:
            # distance <= R_near
            if speed_kmh <= self.max_speed_near:
                # Reward for slow speed/stopping
                speed_factor = 1.0 - (speed_kmh / self.max_speed_near)
                reward = self.stop_reward * speed_factor
            else:
                # Fine for driving near a sign
                excess_speed = speed_kmh - self.max_speed_near
                reward = self.movement_penalty * (1.0 + excess_speed / self.max_speed_near)
        
        return reward
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
