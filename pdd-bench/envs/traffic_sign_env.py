from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.envs import MetaDriveEnv
from traffic_signs.traffic_sign_manager import TrafficSignManager
from traffic_signs.stop_sign import StopSign
from traffic_signs.speed_limit_sign import SpeedLimitSign, SpeedLimitSign20, SpeedLimitSign40, SpeedLimitSign60
from traffic_signs.no_stopping_allowed_sign import NoStoppingAllowedSign
import numpy as np
from metadrive.utils import clip, Config
from envs.pedestrian_manager import CrosswalkPedestrianManager
from envs.socket_crosswalk_manager import SocketCrosswalkManager
from envs.crosswalk_yield_enforcer_manager import CrosswalkYieldEnforcerManager
from traffic_signs.pedestrian_yield_rule import PedestrianYieldRule


TRAFFIC_SIGN_DEFAULT_CONFIG = dict(
    use_crosswalk_manager=False,
    use_pedestrian_manager=False,
    use_pedestrian_yield_rule=True,
    enforce_pedestrian_yield_for_traffic=False,
    crosswalk_manager=dict(
        enabled=True,
        num_crosswalks=-1,
        replace_existing=True,
        random_select=False,
        include_pre_block_socket=False,
        include_all_sockets=True,
        skip_first_block=True,
        socket_margin=1.5,
        socket_anchor="end",
        debug_rejection_log=False,
        debug_max_rejections=64,
    ),
    pedestrian_manager=dict(
        enabled=True,
        initial_pedestrians=2,
        max_pedestrians=6,
        spawn_by_interval=True,
        crossing_interval_range=[6.0, 12.0],
        max_active_per_crosswalk=1,
        max_new_tracks_per_step=1,
        spawn_probability=0.08,
        min_spawn_gap=1.5,
        speed_mean=1.2,
        speed_std=0.2,
        arrive_dist=0.35,
        yield_to_vehicles=True,
        yield_on_crosswalk=False,
        yield_distance=12.0,
        yield_speed_kmh=8.0,
        no_stop_before_crosswalk_m=8.0,
        no_stop_speed_kmh=1.0,
        no_stop_min_duration_s=1.0,
        wait_time_range=[1.5, 4.0],
        pause_time_range=[0.8, 2.0],
    ),
    pedestrian_yield_enforcer=dict(
        enabled=True,
        steer_value=0.0,
        brake_value=-1.0,
        max_forced_per_step=-1,
    ),
    crash_human_penalty=20.0,
)


class TrafficSignEnv(MetaDriveEnv):
    @classmethod
    def default_config(cls) -> Config:
        config = super(TrafficSignEnv, cls).default_config()
        config.update(TRAFFIC_SIGN_DEFAULT_CONFIG)
        return config

    def __init__(self, config=None):
        # Pull `scene_row` out of the user-supplied config before MetaDrive's
        # Config validator runs — it rejects nested dict values for keys that
        # aren't in its allow-list (agent_configs / sensors). We carry it on
        # the instance instead so reset() can drive sign placement (see below).
        self._scene_row = None
        if config is not None and hasattr(config, "pop"):
            try:
                row = config.pop("scene_row", None)
                if row:
                    self._scene_row = row
            except Exception:
                pass
        super().__init__(config)

    def setup_engine(self):
        super().setup_engine()
        self.engine.register_manager("traffic_sign_manager", TrafficSignManager())
        if self.config.get("use_crosswalk_manager", False):
            self.engine.register_manager("crosswalk_manager", SocketCrosswalkManager())
        if self.config.get("use_pedestrian_yield_rule", True):
            ped_cfg = self.config.get("pedestrian_manager", {})
            ped_cfg = ped_cfg.get_dict() if hasattr(ped_cfg, "get_dict") else dict(ped_cfg)
            self.engine.traffic_sign_manager.add_rule(
                PedestrianYieldRule(
                    yield_distance=float(ped_cfg.get("yield_distance", 12.0)),
                    yield_speed_kmh=float(ped_cfg.get("yield_speed_kmh", 8.0)),
                )
            )
        if self.config["use_pedestrian_manager"]:
            self.engine.register_manager("pedestrian_manager", CrosswalkPedestrianManager())
            if self.config.get("enforce_pedestrian_yield_for_traffic", False):
                self.engine.register_manager("crosswalk_yield_enforcer", CrosswalkYieldEnforcerManager())

    def _is_near_stop_sign(self, vehicle, radius=10.0):
        sign_mgr = getattr(self.engine, 'traffic_sign_manager', None)
        if sign_mgr is None:
            return False
        for sign in sign_mgr.signs:
            if isinstance(sign, StopSign):
                dx = sign.position[0] - vehicle.position[0]
                dy = sign.position[1] - vehicle.position[1]
                dist = np.hypot(dx, dy)
                if dist < radius:
                    return True
        return False

    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0

        speed_multiplier = 1.0
        if self._is_near_stop_sign(vehicle, radius=15.0):
            speed_multiplier = max(0.0, 1.0 - vehicle.speed_km_h / 30.0)

        reward = 0.0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
        # reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * speed_multiplier * positive_road

        base_reward = reward
        sign_mgr = self.engine.traffic_sign_manager
        violations = sign_mgr.check_all_violations(vehicle, for_reward=True)
        total_violation_penalty = 0.0
        ped_yield_violated = False
        direction_sign_violated = False
        traffic_violations = 0
        for sign, violated in violations:
            if violated:
                traffic_violations += 1
                total_violation_penalty -= 3
                if getattr(sign, "id", None) == "pedestrian_yield_rule":
                    ped_yield_violated = True
                if type(sign).__name__ == "PGDirectionSign":
                    direction_sign_violated = True
                # print(sign, violated, total_violation_penalty, base_reward)

        reward = base_reward + total_violation_penalty
        step_info["traffic_violations"] = traffic_violations
        step_info["direction_sign_violated"] = direction_sign_violated
        step_info["pedestrian_yield_violated"] = ped_yield_violated

        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_human:
            reward = -self.config.get("crash_human_penalty", self.config["crash_vehicle_penalty"])
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        elif vehicle.crash_sidewalk:
            reward = -self.config["crash_sidewalk_penalty"]
        step_info["route_completion"] = vehicle.navigation.route_completion

        return reward, step_info





    def reset(self, *, seed=None):
        obs, info = super().reset(seed=seed)

        # Place sign(s) inside reset() — symmetric with TrafficSignSumoEnv.
        # When self._scene_row is None this is a no-op and the env behaves
        # exactly as before (callers add signs externally post-reset).
        if self._scene_row:
            try:
                from factorized_space.benchmark_runner import place_pgmap_sign_from_row
            except ImportError:
                # factorized_space not on sys.path (caller set scene_row but
                # the helper module isn't reachable) — silently skip.
                place_pgmap_sign_from_row = None
            if place_pgmap_sign_from_row is not None:
                ok, reason = place_pgmap_sign_from_row(self, self._scene_row, seed=seed)
                info["sign_placement_ok"] = ok
                info["sign_placement_reason"] = reason

        return obs, info
