"""
Minimal SUMO-map environment for Plant2 inference on pdd-bench scenes.

Fixes over the original sumo_env.py
-------------------------------------
- No reference to ``graph.lane_to_tl_signals`` (attribute does not exist on
  ``RoadLaneJunctionGraph``).
- Config keys (``map_name``, ``sign_type``, …) are read from config *without*
  ``pop`` so BaseEnv's machinery is never confused.
- ``TrafficSignManager.signs`` is cleared on every ``reset`` so signs do not
  accumulate across episodes.
- Sign type lookup falls back gracefully when the code is unknown.
"""

import json
import logging
import os

from metadrive.envs import BaseEnv
from metadrive.engine.asset_loader import AssetLoader
from metadrive.manager.sumo_map_manager import SumoMapManager
from metadrive.manager.base_manager import BaseManager
from metadrive.constants import TerminationState
from metadrive.component.navigation_module.edge_network_navigation import EdgeNetworkNavigation
from metadrive.utils import clip, Config
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel

from traffic_signs.traffic_sign_manager import TrafficSignManager
from traffic_signs.stop_sign import StopSign
from traffic_signs.direction_sign import DirectionSign
from traffic_signs.no_entry_sign import NoEntrySign
from traffic_signs.min_speed_limit_sign import MinimumSpeedLimitSign
from traffic_signs.no_traffic_sign import NoTrafficSign
from traffic_signs.speed_limit_sign import SpeedLimitSign
from traffic_signs.no_stopping_allowed_sign import NoStoppingAllowedSign
from traffic_signs.no_overtaking_sign import NoOvertakingSign
from traffic_signs.zone_signs import ZoneSpeedLimitSign
from traffic_signs.end_of_zone_signs import (
    EndOfZoneSpeedLimitSign, EndOfSpeedLimitSign, EndOfAllRestrictionsSign,
)
from traffic_signs.priority_signs import (
    MainRoadSign, EndMainRoadSign, SecondaryRoadSign,
    SecondaryRoadRightSign, SecondaryRoadLeftSign, YieldSign,
)
from traffic_signs.no_turn_allowed import *
from traffic_signs.one_way_entry_sign import *
from traffic_signs.lane_allowed_direction_sign import *

SIGN_TYPE_TO_CLASS = {
    "2.1":    MainRoadSign,
    "2.2":    EndMainRoadSign,
    "2.3.1":  SecondaryRoadSign,
    "2.3.2":  SecondaryRoadRightSign,
    "2.3.3":  SecondaryRoadLeftSign,
    "2.4":    YieldSign,
    "2.5":    StopSign,
    "3.1":    NoEntrySign,
    "3.2":    NoTrafficSign,
    "3.18.1": NoRightTurnSign,
    "3.18.2": NoLeftTurnSign,
    "3.19":   NoUTurnSign,
    "3.20":   NoOvertakingSign,
    "3.24":   SpeedLimitSign,
    "3.25":   EndOfSpeedLimitSign,
    "3.27":   NoStoppingAllowedSign,
    "3.31":   EndOfAllRestrictionsSign,
    "4.1.1":  LaneAllowedDirectionSign4_1_1,
    "4.1.2":  LaneAllowedDirectionSign4_1_2,
    "4.1.3":  LaneAllowedDirectionSign4_1_3,
    "4.1.4":  LaneAllowedDirectionSign4_1_4,
    "4.1.5":  LaneAllowedDirectionSign4_1_5,
    "4.1.6":  LaneAllowedDirectionSign4_1_6,
    "4.6":    MinimumSpeedLimitSign,
    "5.5":    OneWayEntrySignS,
    "5.7.1":  OneWayEntrySignR,
    "5.7.2":  OneWayEntrySignL,
    "5.15.2": DirectionSign,
    "5.31":   ZoneSpeedLimitSign,
    "5.32":   EndOfZoneSpeedLimitSign,
}


class _NoOpManager(BaseManager):
    """Placeholder traffic manager that does nothing."""
    def after_reset(self):
        pass

    def before_step(self):
        pass


SUMO_V2_CONFIG = dict(
    success_reward=10.0,
    out_of_road_penalty=5.0,
    crash_vehicle_penalty=5.0,
    crash_object_penalty=5.0,
    crash_sidewalk_penalty=0.0,
    driving_reward=1.0,
    speed_reward=0.1,
    use_lateral_reward=False,

    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,

    out_of_route_done=False,
    out_of_road_done=True,
    on_continuous_line_done=True,
    on_broken_line_done=False,
    crash_vehicle_done=True,
    crash_object_done=True,
    crash_human_done=True,

    vehicle_config=dict(
        navigation_module=EdgeNetworkNavigation,
        max_steering=50,
    ),
)


class SumoEnvV2(BaseEnv):
    """
    SUMO-map environment for pdd-bench scenes.

    Required config keys (on top of MetaDrive defaults)::

        map_name            – absolute path to the .net.xml file
        sign_type           – e.g. "2.5", "3.27"
        sign_spawn_distance – metres from the start of the sign lane
        traffic_density     – float, 0 = no surrounding traffic (default)
    """

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(SUMO_V2_CONFIG)
        config["map_name"] = "map.net.xml"
        config["sign_type"] = "2.5"
        config["sign_spawn_distance"] = 0.0
        config["traffic_density"] = 0.0
        return config

    def __init__(self, config):
        self._default_snapshot = Config(self.default_config(), unchangeable=True)
        super().__init__(config)

        self._map_name = self.config["map_name"]
        self._sign_type = self.config["sign_type"]
        self._sign_spawn_distance = self.config["sign_spawn_distance"]
        self._meta = self._load_meta()

    # ------------------------------------------------------------------
    # Engine / managers
    # ------------------------------------------------------------------

    def setup_engine(self):
        super().setup_engine()
        if os.path.isabs(self._map_name) and os.path.exists(self._map_name):
            map_path = self._map_name
        else:
            map_path = AssetLoader.file_path(
                "carla", self._map_name, unix_style=False
            )
        self.engine.register_manager("map_manager", SumoMapManager(map_path))
        self.engine.register_manager("traffic_manager", _NoOpManager())
        self.engine.register_manager("traffic_sign_manager", TrafficSignManager())

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, *, seed=None):
        obs, info = super().reset(seed=seed)

        sign_mgr = self.engine.traffic_sign_manager
        self._clear_signs(sign_mgr)

        sign_class = SIGN_TYPE_TO_CLASS.get(self._sign_type)
        if sign_class is None:
            logging.warning(
                "Unknown sign type '%s' – skipping sign placement", self._sign_type
            )
            return obs, info

        road_network = self.engine.current_map.road_network

        sign_lane = None
        if self._meta and "road_id" in self._meta:
            road_id = str(self._meta["road_id"])
            try:
                lane_key = road_network.find_rightmost_lane_by_road_id(road_id)
                if lane_key is None:
                    raise KeyError(f"No graph lane for road_id={road_id!r}")
                sign_lane = road_network.get_lane(lane_key)
            except Exception:
                logging.warning(
                    "Could not find lane for road_id=%s, falling back to ego lane",
                    road_id,
                )

        if sign_lane is None:
            sign_lane = self.vehicle.lane

        sign_mgr.add_sign(
            sign_class,
            lane=sign_lane,
            longitudinal_offset=-sign_lane.length + self._sign_spawn_distance,
            lateral_offset=sign_lane.width_at(0) / 2 + 0.8,
        )

        return obs, info

    @staticmethod
    def _clear_signs(sign_mgr):
        for sign in list(sign_mgr.signs):
            try:
                sign_mgr._cleanup_sign_object(sign)
            except Exception:
                pass
        sign_mgr.signs.clear()
        sign_mgr.rules.clear()
        sign_mgr.violations.clear()
        sign_mgr.violation_details.clear()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_single_observation(self, _=None):
        return TopDownMultiChannel(
            self.config["vehicle_config"],
            onscreen=self.config["use_render"],
            clip_rgb=True,
            frame_stack=3,
            post_stack=5,
            frame_skip=5,
            resolution=(84, 84),
            max_distance=30,
        )

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def done_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        done = False
        max_step = (
            self.config["horizon"] is not None
            and self.episode_lengths[vehicle_id] >= self.config["horizon"]
        )
        done_info = {
            TerminationState.CRASH_VEHICLE:  vehicle.crash_vehicle,
            TerminationState.CRASH_OBJECT:   vehicle.crash_object,
            TerminationState.CRASH_BUILDING: vehicle.crash_building,
            TerminationState.CRASH_HUMAN:    vehicle.crash_human,
            TerminationState.CRASH_SIDEWALK: vehicle.crash_sidewalk,
            TerminationState.OUT_OF_ROAD:    self._is_out_of_road(vehicle),
            TerminationState.SUCCESS:        self._is_arrive_destination(vehicle),
            TerminationState.MAX_STEP:       max_step,
            TerminationState.ENV_SEED:       self.current_seed,
        }
        done_info[TerminationState.CRASH] = (
            done_info[TerminationState.CRASH_VEHICLE]
            or done_info[TerminationState.CRASH_OBJECT]
            or done_info[TerminationState.CRASH_BUILDING]
            or done_info[TerminationState.CRASH_SIDEWALK]
            or done_info[TerminationState.CRASH_HUMAN]
        )

        if done_info[TerminationState.SUCCESS]:
            done = True
        if done_info[TerminationState.OUT_OF_ROAD] and self.config["out_of_road_done"]:
            done = True
        if done_info[TerminationState.CRASH_VEHICLE] and self.config["crash_vehicle_done"]:
            done = True
        if done_info[TerminationState.CRASH_OBJECT] and self.config["crash_object_done"]:
            done = True
        if done_info[TerminationState.CRASH_BUILDING]:
            done = True
        if done_info[TerminationState.CRASH_HUMAN] and self.config["crash_human_done"]:
            done = True
        if max_step and self.config["truncate_as_terminate"]:
            done = True

        return done, done_info

    # ------------------------------------------------------------------
    # Reward / cost
    # ------------------------------------------------------------------

    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = {}

        current_lane = (
            vehicle.navigation.current_lane
            if hasattr(vehicle, "navigation") and vehicle.navigation
            else vehicle.lane
        )
        if current_lane is None:
            return 0.0, {"step_reward": 0.0, "route_completion": 0.0}

        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        if self.config["use_lateral_reward"]:
            lateral_factor = clip(
                1 - 2 * abs(lateral_now) / current_lane.width, 0.0, 1.0
            )
        else:
            lateral_factor = 1.0

        reward = self.config["driving_reward"] * (long_now - long_last) * lateral_factor
        reward += self.config["speed_reward"] * (
            vehicle.speed_km_h / vehicle.max_speed_km_h
        )

        sign_mgr = self.engine.traffic_sign_manager
        for _, violated in sign_mgr.check_all_violations(vehicle, for_reward=True):
            if violated:
                reward -= 3.0

        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        elif vehicle.crash_sidewalk:
            reward = -self.config["crash_sidewalk_penalty"]

        step_info["route_completion"] = getattr(
            vehicle.navigation, "route_completion", 0.0
        )
        return reward, step_info

    def cost_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = {"cost": 0}
        if self._is_out_of_road(vehicle):
            step_info["cost"] = self.config["out_of_road_cost"]
        elif vehicle.crash_vehicle:
            step_info["cost"] = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
        return step_info["cost"], step_info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_arrive_destination(vehicle):
        long, lat = vehicle.navigation.final_lane.local_coordinates(vehicle.position)
        fl = vehicle.navigation.final_lane
        return (
            fl.length - 5 < long < fl.length + 5
            and vehicle.navigation.get_current_lane_width() / 2
            >= lat
            >= (0.5 - vehicle.navigation.get_current_lane_num())
            * vehicle.navigation.get_current_lane_width()
        )

    def _is_out_of_road(self, vehicle):
        ret = not vehicle.on_lane
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        elif self.config["on_continuous_line_done"]:
            ret = ret or (
                vehicle.on_yellow_continuous_line
                or vehicle.on_white_continuous_line
                or vehicle.crash_sidewalk
            )
        if self.config["on_broken_line_done"]:
            ret = ret or vehicle.on_broken_line
        return ret

    def _load_meta(self):
        map_name = self._map_name
        if os.path.isabs(map_name):
            meta_path = os.path.join(os.path.dirname(map_name), "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    return json.load(f)
        return None
