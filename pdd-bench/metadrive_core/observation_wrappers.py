import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

from metadrive.obs.state_obs import StateObservation
from traffic_signs.stop_sign import StopSign
from metadrive_core.sign_utils import add_stop_signs_to_map, add_signs_by_count


class AddStateObservationWrapper(gym.ObservationWrapper):
    """
    Generic BEV+state observation wrapper with optional stop-sign features.
    """

    def __init__(
        self,
        env,
        debug=False,
        add_stop_signs=True,
        stop_sign_probability=0.3,
        stop_sign_min_lane_length=15.0,
        stop_sign_max_signs=10,
        stop_sign_min_distance=25.0,
        include_stop_sign_radius=False,
        stop_sign_radius_max=1000.0,
        sign_class=StopSign,
        sign_classes_with_weights=None,
    ):
        super().__init__(env)
        self.state_obs = None
        self._state_obs_space = None
        self.debug = debug
        self.add_stop_signs = add_stop_signs
        self.stop_sign_probability = stop_sign_probability
        self.stop_sign_min_lane_length = stop_sign_min_lane_length
        self.stop_sign_max_signs = stop_sign_max_signs
        self.stop_sign_min_distance = stop_sign_min_distance
        self.include_stop_sign_radius = include_stop_sign_radius
        self.stop_sign_radius_max = stop_sign_radius_max
        self.sign_class = sign_class
        self.sign_classes_with_weights = sign_classes_with_weights  # List of (sign_class, weight) tuples

        temp_state_obs = StateObservation(env.unwrapped.config)
        base_space = temp_state_obs.observation_space
        base_state_dim = base_space.shape[0]
        temp_state_obs.destroy()

        if include_stop_sign_radius:
            extended_state_dim = base_state_dim + 1
            self._state_obs_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(extended_state_dim,),
                dtype=np.float32,
            )
        else:
            self._state_obs_space = base_space

        self.observation_space = gym.spaces.Dict({
            "image": env.observation_space,
            "state": self._state_obs_space,
        })

    def reset(self, **kwargs):
        filtered_kwargs = {}
        if "seed" in kwargs:
            filtered_kwargs["seed"] = kwargs["seed"]

        obs, info = self.env.reset(**filtered_kwargs)

        if self.state_obs is None:
            self.state_obs = StateObservation(self.unwrapped.config)
        if hasattr(self.state_obs, "reset"):
            self.state_obs.reset(self.unwrapped, self.unwrapped.vehicle)

        if self.add_stop_signs:
            if self.sign_classes_with_weights is not None and len(self.sign_classes_with_weights) > 0:
                # Add multiple sign types (same mechanism as run_carl_pdd_bench.py)
                sign_counts = add_multiple_sign_types_to_map(
                    self.env,
                    sign_classes_with_weights=self.sign_classes_with_weights,
                    max_signs=self.stop_sign_max_signs,
                    min_distance=self.stop_sign_min_distance,
                    sign_probability=self.stop_sign_probability,
                )
                total_added = sum(sign_counts.values())
                if self.debug and total_added > 0:
                    print(f"  Added {total_added} traffic sign(s) to the map: {sign_counts}")
            else:
                # Add single sign type (backward compatibility)
                signs_added = add_stop_signs_to_map(
                    self.env,
                    stop_sign_probability=self.stop_sign_probability,
                    max_signs=self.stop_sign_max_signs,
                    min_distance=self.stop_sign_min_distance,
                    min_lane_length=self.stop_sign_min_lane_length,
                    sign_class=self.sign_class,
                )
                if self.debug and signs_added > 0:
                    print(f"  Added {signs_added} stop sign(s) to the map")

        obs_dict = self.observation(obs)
        return obs_dict, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_dict = self.observation(obs)
        return obs_dict, reward, terminated, truncated, info

    def _get_stop_sign_radius(self):
        try:
            base_env = self.unwrapped
            while hasattr(base_env, "env"):
                base_env = base_env.env

            if not hasattr(base_env, "engine") or not hasattr(
                base_env.engine, "traffic_sign_manager"
            ):
                return self.stop_sign_radius_max

            sign_mgr = base_env.engine.traffic_sign_manager
            if sign_mgr is None or len(sign_mgr.signs) == 0:
                return self.stop_sign_radius_max

            vehicle = base_env.vehicle
            if vehicle is None:
                return self.stop_sign_radius_max

            vehicle_pos = np.array(vehicle.position)
            min_distance = float("inf")

            for sign in sign_mgr.signs:
                if hasattr(sign, "position"):
                    sign_pos = np.array(sign.position)
                    distance = np.linalg.norm(vehicle_pos - sign_pos)
                    min_distance = min(min_distance, distance)

            if min_distance == float("inf"):
                return self.stop_sign_radius_max
            return min(min_distance, self.stop_sign_radius_max)
        except Exception as e:
            if self.debug:
                print(f"  Warning: Error getting stop_sign radius: {e}")
            return self.stop_sign_radius_max

    def observation(self, observation):
        if self.state_obs is None:
            self.state_obs = StateObservation(self.unwrapped.config)
        vehicle = self.unwrapped.vehicle
        image = np.asarray(observation, dtype=np.float32)
        base_state = np.asarray(self.state_obs.observe(vehicle), dtype=np.float32)

        if self.include_stop_sign_radius:
            stop_sign_radius = self._get_stop_sign_radius()
            state = np.concatenate(
                [base_state, np.array([stop_sign_radius], dtype=np.float32)]
            )
        else:
            state = base_state

        return {
            "image": image,
            "state": state,
        }

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped
    
    @property
    def top_down_renderer(self):
        """Pass through top_down_renderer property from wrapped environment"""
        return getattr(self.env, 'top_down_renderer', None)


__all__ = ["AddStateObservationWrapper"]
