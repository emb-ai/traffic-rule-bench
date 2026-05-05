"""
Common wrappers for PPO training.
"""

import gymnasium as gym


class EnsureSuccessInfoWrapper(gym.Wrapper):
    """
    Ensure info contains is_success for SB3 Monitor.
    """

    def _get_base_env(self):
        base_env = self.unwrapped
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        return base_env

    def reset(self, **kwargs):
        filtered_kwargs = {}
        if 'seed' in kwargs:
            filtered_kwargs['seed'] = kwargs['seed']

        obs, info = self.env.reset(**filtered_kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            is_success = False

            if "arrive_dest" in info:
                is_success = bool(info["arrive_dest"])
            elif "is_success" in info:
                is_success = bool(info["is_success"])
            else:
                try:
                    base_env = self._get_base_env()
                    if hasattr(base_env, 'vehicle') and base_env.vehicle is not None:
                        vehicle = base_env.vehicle
                        if hasattr(vehicle, 'arrive_destination'):
                            is_success = bool(vehicle.arrive_destination)
                        elif hasattr(vehicle, 'crash_vehicle') or hasattr(vehicle, 'crash_object'):
                            is_success = False
                        elif hasattr(vehicle, 'out_of_road'):
                            is_success = False
                except Exception:
                    is_success = False

            info["is_success"] = is_success

        return obs, reward, terminated, truncated, info
