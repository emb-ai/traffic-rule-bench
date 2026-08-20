from metadrive.obs.observation_base import BaseObservation
from metadrive.obs.state_obs import StateObservation
import gymnasium as gym
import numpy as np
from pdd_bench.signs.stop_sign import StopSign
from pdd_bench.signs.no_stopping_allowed_sign import NoStoppingAllowedSign
from metadrive.obs.state_obs import LidarStateObservation

class TrafficSignObservation(LidarStateObservation):
    def __init__(self, config):
        super().__init__(config)

    @property
    def observation_space(self):
        # 259 from StateObservation + 2 channels for sign state
        return gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(259 + 2,),
            dtype=np.float32
        )

    def observe(self, vehicle):
        base_obs = super().observe(vehicle) 

        sign_type = 0.0
        sign_dist = 1.0
        if hasattr(self.engine, 'traffic_sign_manager'):
            min_d = float('inf')
            for sign in self.engine.traffic_sign_manager.signs:
                if sign is None or getattr(sign, '_is_destroyed', False):
                    continue
                if hasattr(sign, "is_in_drivable_area") and not sign.is_in_drivable_area(vehicle):
                    continue
                dx = sign.position[0] - vehicle.position[0]
                dy = sign.position[1] - vehicle.position[1]
                dist = np.hypot(dx, dy)
                if dist < min_d and dist < 20.0:
                    min_d = dist
                    if isinstance(sign, StopSign):
                        sign_type = 1.0
                    elif isinstance(sign, NoStoppingAllowedSign):
                        sign_type = 2.0
            if min_d != float('inf'):
                sign_dist = min_d / 20.0
        #print("sign_type, sign_dist:    ", sign_type, sign_dist)
        extra = np.array([sign_type, sign_dist], dtype=np.float32)
        return np.concatenate([base_obs, extra])
