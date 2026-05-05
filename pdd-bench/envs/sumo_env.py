import json
import logging
import os
import numpy as np
from typing import Optional
from metadrive.envs import BaseEnv
from metadrive.engine.asset_loader import AssetLoader
from metadrive.manager.sumo_map_manager import SumoMapManager
from metadrive.manager.base_manager import BaseManager
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.component.navigation_module.edge_network_navigation import EdgeNetworkNavigation
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from metadrive.utils import clip, Config
from traffic_signs.traffic_sign_manager import TrafficSignManager
from traffic_signs.stop_sign import StopSign
from traffic_signs.direction_sign import DirectionSign
from traffic_signs.no_entry_sign import NoEntrySign
from traffic_signs.min_speed_limit_sign import MinimumSpeedLimitSign
from traffic_signs.no_traffic_sign import NoTrafficSign
from traffic_signs.speed_limit_sign import SpeedLimitSign, SpeedLimitSign20, SpeedLimitSign40, SpeedLimitSign60
from traffic_signs.traffic_light_sign import TrafficLightSign
from traffic_signs.no_stopping_allowed_sign import NoStoppingAllowedSign
from traffic_signs.no_overtaking_sign import NoOvertakingSign
from traffic_signs.zone_signs import ZoneSpeedLimitSign
from traffic_signs.end_of_zone_signs import EndOfZoneSpeedLimitSign
from traffic_signs.speed_limit_sign import SpeedLimitSign
from traffic_signs.no_stopping_allowed_sign import NoStoppingAllowedSign
from traffic_signs.lane_allowed_direction_sign import *
from traffic_signs.zone_signs import ZoneSpeedLimitSign
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from envs.pedestrian_manager import CrosswalkPedestrianManager
from envs.crosswalk_yield_enforcer_manager import CrosswalkYieldEnforcerManager
from traffic_signs.pedestrian_yield_rule import PedestrianYieldRule
import numpy as np

from traffic_signs.end_of_zone_signs import *
from traffic_signs.priority_signs import *
from traffic_signs.no_turn_allowed import *
from traffic_signs.one_way_entry_sign import *
from traffic_signs.detour_sign import DetourRightSign, DetourLeftSign, DetourEitherSign
from traffic_signs.bus_station_sign import BusStationSign
from traffic_signs.only_auto_sign import OnlyAutoSign
from traffic_signs.restricted_lane_sign import (
    BusLaneRoadSign, BikeLaneRoadSign,
    EndBusLaneRoadSign, EndBikeLaneRoadSign,
    ExitToBusLaneSign, ExitToBusLaneSignLeft,
    ExitToBikeLaneSign, ExitToBikeLaneSignLeft,
    BusLaneSign, BikeLaneSign,
    EndBusLaneSign, EndBikeLaneSign,
)
from traffic_signs.only_auto_sign import OnlyAutoSign

SIGN_TYPE_TO_CLASS = {
    "2.1": MainRoadSign,
    "2.2": EndMainRoadSmartSign,
    "2.3.1": SecondaryRoadSign,
    "2.3.2": SecondaryRoadRightSign,
    "2.3.3": SecondaryRoadLeftSign,
    "2.4": YieldSign,
    "2.5": StopSign,
    "3.24": SpeedLimitSign,
    "3.25": EndOfSpeedLimitSign,
    "3.27": NoStoppingAllowedSign,
    "3.31": EndOfAllRestrictionsSign,
    "5.15.2": DirectionSign,
    "5.31": ZoneSpeedLimitSign,
    "5.32": EndOfZoneSpeedLimitSign,
    "5.16": BusStationSign,
    "5.3":  OnlyAutoSign,
    "3.1" : NoEntrySign,
    "4.6" : MinimumSpeedLimitSign,
    "3.2" : NoTrafficSign,
    "3.20": NoOvertakingSign,
    "3.21": EndOfNoOvertakingSign,
    "4.1.1" : LaneAllowedDirectionSign4_1_1,
    "4.1.2" : LaneAllowedDirectionSign4_1_2,
    "4.1.3": LaneAllowedDirectionSign4_1_3,
    "4.1.4": LaneAllowedDirectionSign4_1_4,
    "4.1.5": LaneAllowedDirectionSign4_1_5,
    "4.1.6": LaneAllowedDirectionSign4_1_6,
    "3.18.1": NoRightTurnSign,
    "3.18.2": NoLeftTurnSign,
    "3.19": NoUTurnSign,
    "5.7.1": OneWayEntrySignR,
    "5.7.2": OneWayEntrySignL,
    "5.3": OnlyAutoSign,
    "5.4": EndOfOnlyAutoSign,
    "5.5": OneWayEntrySignS,
    "4.2.1": DetourRightSign,
    "4.2.2": DetourLeftSign,
    "4.2.3": DetourEitherSign,
    "5.11.1": BusLaneRoadSign,
    "5.11.2": BikeLaneRoadSign,
    "5.12.1": EndBusLaneRoadSign,
    "5.12.2": EndBikeLaneRoadSign,
    "5.13.1": ExitToBusLaneSign,
    "5.13.2": ExitToBusLaneSignLeft,
    "5.13.3": ExitToBikeLaneSign,
    "5.13.4": ExitToBikeLaneSignLeft,
    "5.14.1": BusLaneSign,
    "5.14.2": BikeLaneSign,
    "5.14.3": EndBusLaneSign,
    "5.14.4": EndBikeLaneSign,
}

class SimpleTrafficManager(BaseManager):
    def after_reset(self):
        pass
    def before_step(self):
        pass
    
SUMO_DEFAULT_CONFIG = dict(
    success_reward=10.0,
    out_of_road_penalty=5.0,
    crash_vehicle_penalty=5.0,
    crash_object_penalty=5.0,
    crash_sidewalk_penalty=0.0,
    crash_human_penalty=20.0,
    driving_reward=1.0,
    speed_reward=0.1,
    use_lateral_reward=False,

    # ===== Cost Scheme =====
    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,

    # ===== Termination Scheme =====
    out_of_route_done=False,
    out_of_road_done=True,
    on_continuous_line_done=False,
    on_broken_line_done=False,
    crash_vehicle_done=True,
    crash_object_done=True,
    crash_human_done=True,
    vehicle_config=dict(
        navigation_module=EdgeNetworkNavigation,
               max_steering=50,
    ),

    min_lane_width=0,

    # ===== Pedestrians & yield rule =====
    use_pedestrian_manager=True,
    use_pedestrian_yield_rule=True,
    enforce_pedestrian_yield_for_traffic=True,
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
        # Disabled by default: SUMO ego often spawns near a crosswalk at 0 km/h
        # and would instantly accumulate a no-stop violation, zeroing the reward.
        no_stop_before_crosswalk_m=0.0,
        no_stop_speed_kmh=1.0,
        no_stop_min_duration_s=1.0,
        wait_time_range=[1.5, 4.0],
        pause_time_range=[0.8, 2.0],
        # Suppress spawn on crosswalks whose adjacent TL is green for cars.
        green_tl_spawn_probability=0.05,
        tl_match_radius=40.0,
    ),
    pedestrian_yield_enforcer=dict(
        enabled=True,
        steer_value=0.0,
        brake_value=-1.0,
        max_forced_per_step=-1,
    ),
)


class TrafficSignSumoEnv(BaseEnv):
    @classmethod
    def default_config(cls):
        config = super(TrafficSignSumoEnv, cls).default_config()
        config.update(SUMO_DEFAULT_CONFIG)
        config["map_name"] = "map.net.xml"
        config["sign_type"] = "2.5"
        config["sign_spawn_distance"] = 0.0
        config["tl_speed_factor"] = 1.0
        # place ego onto a  parallel lane_num
        # sign's road_id after reset:  0 = rightmost
        config["spawn_lane_num"] = 0
        config["debug_one_way_sign_selection"] = False
        config["min_route_hops_after_spawn"] = 10
        config["max_route_hops_after_spawn"] = 10
        return config
    
    def __init__(self, config):
        self.default_config_copy = Config(self.default_config(), unchangeable=True)
        super(TrafficSignSumoEnv, self).__init__(config)
        self.custom_map_name = config.pop("map_name", "map.net.xml")
        self.sign_type = config.pop("sign_type", "2.5")
        self.sign_spawn_distance = config.pop("sign_spawn_distance", 0.0)
        self.meta = self._load_meta()

    def done_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        done = False
        max_step = self.config["horizon"] is not None and self.episode_lengths[vehicle_id] >= self.config["horizon"]
        done_info = {
            TerminationState.CRASH_VEHICLE: vehicle.crash_vehicle,
            TerminationState.CRASH_OBJECT: vehicle.crash_object,
            TerminationState.CRASH_BUILDING: vehicle.crash_building,
            TerminationState.CRASH_HUMAN: vehicle.crash_human,
            TerminationState.CRASH_SIDEWALK: vehicle.crash_sidewalk,
            TerminationState.OUT_OF_ROAD: self._is_out_of_road(vehicle),
            TerminationState.SUCCESS: self._is_arrive_destination(vehicle),
            TerminationState.MAX_STEP: max_step,
            TerminationState.ENV_SEED: self.current_seed,
            # TerminationState.CURRENT_BLOCK: self.agent.navigation.current_road.block_ID(),
            # crash_vehicle=False, crash_object=False, crash_building=False, out_of_road=False, arrive_dest=False,
        }

        # for compatibility
        # crash almost equals to crashing with vehicles
        done_info[TerminationState.CRASH] = (
            done_info[TerminationState.CRASH_VEHICLE] or done_info[TerminationState.CRASH_OBJECT]
            or done_info[TerminationState.CRASH_BUILDING] or done_info[TerminationState.CRASH_SIDEWALK]
            or done_info[TerminationState.CRASH_HUMAN]
        )

        # determine env return
        if done_info[TerminationState.SUCCESS]:
            done = True
            self.logger.debug(
                "Episode ended! Scenario Index: {} Reason: arrive_dest.".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.OUT_OF_ROAD] and self.config["out_of_road_done"]:
            done = True
            self.logger.debug(
                "Episode ended! Scenario Index: {} Reason: out_of_road.".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_VEHICLE] and self.config["crash_vehicle_done"]:
            done = True
            self.logger.debug(
                "Episode ended! Scenario Index: {} Reason: crash vehicle ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_OBJECT] and self.config["crash_object_done"]:
            done = True
            self.logger.debug(
                "Episode ended! Scenario Index: {} Reason: crash object ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_BUILDING]:
            done = True
            self.logger.debug(
                "Episode ended! Scenario Index: {} Reason: crash building ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_HUMAN] and self.config["crash_human_done"]:
            done = True
            self.logger.debug(
                "Episode ended! Scenario Index: {} Reason: crash human".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.MAX_STEP]:
            # single agent horizon has the same meaning as max_step_per_agent
            if self.config["truncate_as_terminate"]:
                done = True
            self.logger.debug(
                "Episode ended! Scenario Index: {} Reason: max step ".format(self.current_seed),
                extra={"log_once": True}
            )
        return done, done_info

    def cost_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        step_info["cost"] = 0
        if self._is_out_of_road(vehicle):
            step_info["cost"] = self.config["out_of_road_cost"]
        elif vehicle.crash_vehicle:
            step_info["cost"] = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
        return step_info['cost'], step_info

    @staticmethod
    def _is_arrive_destination(vehicle):
        """
        Args:
            vehicle: The BaseVehicle instance.

        Returns:
            flag: Whether this vehicle arrives its destination.
        """
        long, lat = vehicle.navigation.final_lane.local_coordinates(vehicle.position)
        flag = (vehicle.navigation.final_lane.length - 5 < long < vehicle.navigation.final_lane.length + 5) and (
            vehicle.navigation.get_current_lane_width() / 2 >= lat >=
            (0.5 - vehicle.navigation.get_current_lane_num()) * vehicle.navigation.get_current_lane_width()
        )
        return flag

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        # Yellow-continuous contact is intentionally ignored in SUMO env: the axial divider
        # is a rendering hint, not a hard barrier, and some edge geometries place the
        # solid-yellow polyline close to valid driving surface (false positives).
        ret = not vehicle.on_lane
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        elif self.config["on_continuous_line_done"]:
            ret = ret or vehicle.on_white_continuous_line or vehicle.crash_sidewalk
        if self.config["on_broken_line_done"]:
            ret = ret or vehicle.on_broken_line
        return ret

    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = {}

        if hasattr(vehicle, 'navigation') and vehicle.navigation:
            current_lane = vehicle.navigation.current_lane
        else:
            current_lane = vehicle.lane

        if current_lane is None:
            return 0.0, {"step_reward": 0.0, "route_completion": 0.0}

        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # Reward for moving forward
        if self.config["use_lateral_reward"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / current_lane.width, 0.0, 1.0)
        else:
            lateral_factor = 1.0

        reward = self.config["driving_reward"] * (long_now - long_last) * lateral_factor
        reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h)

        base_reward = reward
        sign_mgr = self.engine.traffic_sign_manager
        violations = sign_mgr.check_all_violations(vehicle, for_reward=True)
        total_violation_penalty = 0.0
        for _, violated in violations:
            if violated:
                total_violation_penalty -= 3

        reward = base_reward + total_violation_penalty

        step_info["step_reward"] = reward

        # Terminal rewards
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

        step_info["route_completion"] = getattr(vehicle.navigation, "route_completion", 0.0)
        return reward, step_info

    def _load_meta(self):
        """Load meta.json from the same directory as the .net.xml file."""
        map_name = self.custom_map_name
        if os.path.isabs(map_name):
            meta_path = os.path.join(os.path.dirname(map_name), "meta.json")
        else:
            meta_path = None
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                return json.load(f)
        return None

    def setup_engine(self):
        super().setup_engine()
        if os.path.isabs(self.custom_map_name) and os.path.exists(self.custom_map_name):
            map_path = self.custom_map_name
        else:
            map_path = AssetLoader.file_path("carla", self.custom_map_name, unix_style=False)
        # Must be set before SumoMapManager constructs the lane graph — LaneNode
        # reads this class attribute at __init__.
        from metadrive.utils.sumo.map_utils import LaneNode
        LaneNode.MIN_LANE_WIDTH = float(self.config.get("min_lane_width", 0.0))
        self.engine.register_manager("map_manager", SumoMapManager(map_path))
        self.engine.register_manager("traffic_manager", SimpleTrafficManager())
        self.engine.register_manager("traffic_sign_manager", TrafficSignManager())
        if self.config.get("use_pedestrian_yield_rule", True):
            ped_cfg = self.config.get("pedestrian_manager", {})
            ped_cfg = ped_cfg.get_dict() if hasattr(ped_cfg, "get_dict") else dict(ped_cfg)
            self.engine.traffic_sign_manager.add_rule(
                PedestrianYieldRule(
                    yield_distance=float(ped_cfg.get("yield_distance", 12.0)),
                    yield_speed_kmh=float(ped_cfg.get("yield_speed_kmh", 8.0)),
                    no_stop_before_m=float(ped_cfg.get("no_stop_before_crosswalk_m", 0.0)),
                    no_stop_speed_kmh=float(ped_cfg.get("no_stop_speed_kmh", 1.0)),
                    no_stop_min_duration_s=float(ped_cfg.get("no_stop_min_duration_s", 1.0)),
                )
            )
        if self.config.get("use_pedestrian_manager", True):
            self.engine.register_manager("pedestrian_manager", CrosswalkPedestrianManager())
            if self.config.get("enforce_pedestrian_yield_for_traffic", True):
                self.engine.register_manager("crosswalk_yield_enforcer", CrosswalkYieldEnforcerManager())
        print(f"[sumo_env] LaneNode.MIN_LANE_WIDTH = {LaneNode.MIN_LANE_WIDTH}")

    @staticmethod
    def _normalize_turn_direction(raw_dir: str) -> str:
        d = str(raw_dir or "").strip().lower()
        if d in ("r", "right"):
            return "r"
        if d in ("l", "left"):
            return "l"
        if d in ("s", "straight"):
            return "s"
        if d in ("t", "u", "uturn", "u-turn"):
            return "t"
        return d

    @staticmethod
    def _edge_id_from_lane_key(lane_key: str) -> Optional[str]:
        if not isinstance(lane_key, str) or not lane_key.startswith("lane_"):
            return None
        parts = lane_key.split("_")
        if len(parts) < 3:
            return None
        return parts[1]

    @staticmethod
    def _reverse_edge_id(edge_id: Optional[str]) -> Optional[str]:
        if not edge_id:
            return None
        e = str(edge_id)
        return e[1:] if e.startswith("-") else f"-{e}"

    @staticmethod
    def _wrap_pi(x: float) -> float:
        import math

        return float((x + math.pi) % (2 * math.pi) - math.pi)

    @staticmethod
    def _lane_heading_near_end(lane) -> Optional[float]:
        try:
            s = max(0.0, float(lane.length) - 1.0)
            return float(lane.heading_theta_at(s))
        except Exception:
            return None

    @staticmethod
    def _lane_heading_near_start(lane) -> Optional[float]:
        try:
            s = min(1.0, max(0.0, float(lane.length) * 0.1))
            return float(lane.heading_theta_at(s))
        except Exception:
            return None

    @staticmethod
    def _normalize_junction_id(junction_id) -> Optional[str]:
        if not junction_id:
            return None
        jid = str(junction_id)
        return jid if jid.startswith("junction_") else f"junction_{jid}"

    def _has_reverse_inflow_for_turn(self, road_network, from_lane_key: str, to_lane_key: str) -> bool:
        """Return True if target corridor seems to have opposite inflow into junction.

        Heuristic: for turn `from_lane -> to_lane`, check lanes on `to_lane` edge and
        see if any of them has turns back to the source edge. This suggests two-way
        movement at this junction and weakens one-way-entry assumption.
        """
        src_edge = self._edge_id_from_lane_key(from_lane_key)
        dst_edge = self._edge_id_from_lane_key(to_lane_key)
        if src_edge is None or dst_edge is None:
            return False

        for other_lane_key, other_info in road_network.graph.items():
            if self._edge_id_from_lane_key(other_lane_key) != dst_edge:
                continue
            turns = getattr(other_info, "turns", None) or []
            for t in turns:
                back_to = t.get("to_lane")
                if self._edge_id_from_lane_key(back_to) == src_edge:
                    return True
        return False

    def _has_reverse_inflow_via_junction(self, road_network, from_lane_key: str, to_lane_key: str) -> bool:
        """Check reverse inflow at the same junction using lane-level junction metadata."""
        try:
            from_lane = road_network.get_lane(from_lane_key)
            to_lane = road_network.get_lane(to_lane_key)
        except Exception:
            # Fallback to edge-based check when lane lookup is unavailable.
            return self._has_reverse_inflow_for_turn(road_network, from_lane_key, to_lane_key)

        if from_lane is None or to_lane is None:
            return self._has_reverse_inflow_for_turn(road_network, from_lane_key, to_lane_key)

        # Fast SUMO edge-id check: opposite direction edge is usually represented
        # as the same edge id with '-' prefix toggled.
        dst_edge = self._edge_id_from_lane_key(to_lane_key)
        opp_edge = self._reverse_edge_id(dst_edge)
        if opp_edge is not None:
            junction_id = self._normalize_junction_id(getattr(from_lane, "incoming_junction_id", None))
            for lane_key, info in road_network.graph.items():
                if self._edge_id_from_lane_key(lane_key) != opp_edge:
                    continue
                lane_obj = getattr(info, "lane", None)
                if lane_obj is None:
                    continue
                lane_jid = self._normalize_junction_id(getattr(lane_obj, "incoming_junction_id", None))
                if junction_id is not None and lane_jid != junction_id:
                    continue
                return True


        
        junction_id = self._normalize_junction_id(getattr(from_lane, "incoming_junction_id", None))
        incoming_lanes = set(getattr(from_lane, "incoming_junction_lanes", None) or [])
        if not incoming_lanes and junction_id is not None:
            # Build incoming set by scanning graph for same incoming junction.
            for lane_key, info in road_network.graph.items():
                lane = getattr(info, "lane", None)
                if lane is None:
                    continue
                lane_jid = self._normalize_junction_id(getattr(lane, "incoming_junction_id", None))
                if lane_jid == junction_id:
                    incoming_lanes.add(lane_key)

        if not incoming_lanes:
            return self._has_reverse_inflow_for_turn(road_network, from_lane_key, to_lane_key)

        exit_heading = self._lane_heading_near_start(to_lane)
        if exit_heading is None:
            return self._has_reverse_inflow_for_turn(road_network, from_lane_key, to_lane_key)

        import math

        opposite = self._wrap_pi(exit_heading + math.pi)
        threshold = math.radians(35.0)

        for in_lane_key in incoming_lanes:
            if in_lane_key == from_lane_key:
                continue
            try:
                in_lane = road_network.get_lane(in_lane_key)
            except Exception:
                continue
            if in_lane is None:
                continue
            h = self._lane_heading_near_end(in_lane)
            if h is None:
                continue
            if abs(self._wrap_pi(h - opposite)) < threshold:
                return True
        return False

    def _pick_one_way_entry_lane(self, road_network, preferred_road_id: Optional[str] = None):
        """Pick a lane for 5.7.1/5.7.2 by turn topology at intersections.

        Heuristic:
        - 5.7.1: lane has right turn and has no left turn
        - 5.7.2: lane has left turn and has no right turn
        Prefer lanes that match meta road_id when available.
        """
        if self.sign_type not in ("5.7.1", "5.7.2"):
            return None

        need = "r" if self.sign_type == "5.7.1" else "l"
        forbid = "l" if need == "r" else "r"
        target = str(preferred_road_id) if preferred_road_id is not None else None

        best = None
        best_key = None
        best_score = -10**9
        debug_rows = []
        candidate_lane_keys = []

        for lane_key, info in road_network.graph.items():
            
            if not isinstance(lane_key, str) or not lane_key.startswith("lane_"):
                continue
            if ":" in lane_key:
                continue

            turns = getattr(info, "turns", None) or []
            if not turns:
                continue

            dirs = set()
            need_turn_targets = []
            for t in turns:
                d = self._normalize_turn_direction(t.get("direction"))
                dirs.add(d)
                if d == need and t.get("to_lane"):
                    need_turn_targets.append(t.get("to_lane"))

            if need not in dirs or forbid in dirs:
                continue

            # Requested constraint: no reverse inflow from target corridor
            # back into this junction for the chosen turn direction.
            if need_turn_targets:
                if all(self._has_reverse_inflow_via_junction(road_network, lane_key, t) for t in need_turn_targets):
                    continue


            # Straight/U-turn availability is allowed and does not penalize candidate.
            # Prefer longer approach lane before junction.
            lane = info.lane

            if float(getattr(lane, "length", 0.0)) > best_score:
                best_score = float(getattr(lane, "length", 0.0))
                best = lane
                best_key = lane_key

            debug_rows.append((lane_key, float(getattr(lane, "length", 0.0)), sorted(dirs)))
            candidate_lane_keys.append(lane_key)

        self._one_way_candidate_lane_keys = candidate_lane_keys

        if self.config.get("debug_one_way_sign_selection", False):
            self._log_one_way_selection_debug(
                preferred_road_id=target,
                best_key=best_key,
                best_score=best_score,
                debug_rows=debug_rows,
            )

        return best

    @staticmethod
    def _allowed_dirs_for_lane_sign(sign_type: str) -> Optional[set[str]]:
        mapping = {
            "4.1.1": {"s"},
            "4.1.2": {"r"},
            "4.1.3": {"l"},
            "4.1.4": {"s", "r"},
            "4.1.5": {"s", "l"},
            "4.1.6": {"l", "r"},
        }
        return mapping.get(sign_type)

    @staticmethod
    def _lane_priority_flag(lane) -> Optional[bool]:
        turns = getattr(lane, "turns", None) or []
        unmanaged = [
            t for t in turns
            if t.get("junction_type") in ("priority", "right_before_left") and isinstance(t.get("priority"), dict)
        ]
        if not unmanaged:
            return None
        has_any_yield = any(not bool(t["priority"].get("has_priority", False)) for t in unmanaged)
        return not has_any_yield

    def _pick_lane_direction_candidates(self, road_network, preferred_road_id: Optional[str] = None):
        allowed = self._allowed_dirs_for_lane_sign(self.sign_type)
        if allowed is None:
            return None, []

        target = str(preferred_road_id) if preferred_road_id is not None else None
        best_lane = None
        best_key = None
        best_score = -10**9
        candidate_lane_keys = []
        debug_rows = []

        for lane_key, info in road_network.graph.items():
            if not isinstance(lane_key, str) or not lane_key.startswith("lane_"):
                continue
            if ":" in lane_key:
                continue
            turns = getattr(info, "turns", None) or []
            if not turns:
                continue

            dirs = set()
            for t in turns:
                d = self._normalize_turn_direction(t.get("direction"))
                if d in {"l", "r", "s"}:
                    dirs.add(d)
            if dirs != allowed:
                continue

            score = 0.0
            lane = info.lane
            score += min(float(getattr(lane, "length", 0.0)), 80.0)
            if target:
                parts = lane_key.split("_")
                edge_id = parts[1] if len(parts) >= 3 else None
                if edge_id == target:
                    score += 1000.0

            candidate_lane_keys.append(lane_key)
            debug_rows.append((lane_key, score, sorted(dirs)))
            if score > best_score:
                best_score = score
                best_lane = lane
                best_key = lane_key

        if self.config.get("debug_one_way_sign_selection", False):
            top_rows = sorted(debug_rows, key=lambda x: x[1], reverse=True)[:8]
            print(
                f"[LaneSignDebug] sign={self.sign_type} map={self.custom_map_name} "
                f"preferred_road_id={target} chosen={best_key} score={best_score}"
            )
            for lane_key, score, dirs in top_rows:
                print(f"[LaneSignDebug] candidate lane={lane_key} score={score:.1f} dirs={dirs}")

        return best_lane, candidate_lane_keys

    @staticmethod
    def _prohibited_dir_for_no_turn_sign(sign_type: str) -> Optional[str]:
        mapping = {
            "3.18.1": "r",
            "3.18.2": "l",
            "3.19": "t",
        }
        return mapping.get(sign_type)

    @staticmethod
    def _is_junction_like_lane_id(lane_id) -> bool:
        if lane_id is None:
            return False
        s = str(lane_id)
        return s.startswith("junction") or s.startswith("lane_:") or s.startswith(":")

    def _pick_no_turn_candidates(self, road_network, preferred_road_id: Optional[str] = None):
        prohibited = self._prohibited_dir_for_no_turn_sign(self.sign_type)
        if prohibited is None:
            return None, []

        target = str(preferred_road_id) if preferred_road_id is not None else None
        best_lane = None
        best_key = None
        best_score = -10**9
        candidate_lane_keys = []
        debug_rows = []

        for lane_key, info in road_network.graph.items():
            if not isinstance(lane_key, str) or not lane_key.startswith("lane_"):
                continue
            if ":" in lane_key:
                continue
            turns = getattr(info, "turns", None) or []
            if not turns:
                continue

            dirs = set()
            for t in turns:
                d = self._normalize_turn_direction(t.get("direction"))
                if d in {"l", "r", "s", "t"}:
                    dirs.add(d)

            # Candidate lane must contain prohibited maneuver and at least one
            # alternative maneuver that does not violate the sign.
            if prohibited not in dirs:
                continue
            if len(dirs - {prohibited}) == 0:
                continue

            score = 0.0
            lane = info.lane
            score += min(float(getattr(lane, "length", 0.0)), 80.0)
            if target:
                parts = lane_key.split("_")
                edge_id = parts[1] if len(parts) >= 3 else None
                if edge_id == target:
                    score += 1000.0

            candidate_lane_keys.append(lane_key)
            debug_rows.append((lane_key, score, sorted(dirs)))
            if score > best_score:
                best_score = score
                best_lane = lane
                best_key = lane_key

        if self.config.get("debug_one_way_sign_selection", False):
            top_rows = sorted(debug_rows, key=lambda x: x[1], reverse=True)[:8]
            print(
                f"[NoTurnSignDebug] sign={self.sign_type} map={self.custom_map_name} "
                f"preferred_road_id={target} chosen={best_key} score={best_score}"
            )
            for lane_key, score, dirs in top_rows:
                print(f"[NoTurnSignDebug] candidate lane={lane_key} score={score:.1f} dirs={dirs}")

        return best_lane, candidate_lane_keys

    @staticmethod
    def _counterpart_dir_for_no_turn_anchor(sign_type: str) -> Optional[str]:
        mapping = {
            "3.18.1": "l",  # no-right: anchor lane should be target of a left turn
            "3.18.2": "r",  # no-left: anchor lane should be target of a right turn
            "3.19": "t",    # no-u-turn: anchor lane should be target of a u-turn
        }
        return mapping.get(sign_type)

    def _pick_no_turn_anchor_lane(
        self,
        road_network,
        no_turn_candidate_lane_keys,
        preferred_road_id: Optional[str] = None,
    ):
        prohibited = self._prohibited_dir_for_no_turn_sign(self.sign_type)
        counterpart = self._counterpart_dir_for_no_turn_anchor(self.sign_type)
        if prohibited is None or counterpart is None:
            return None

        target = str(preferred_road_id) if preferred_road_id is not None else None
        candidate_set = set(no_turn_candidate_lane_keys or [])
        if not candidate_set:
            return None

        best_lane = None
        best_key = None
        best_score = -10**9
        debug_rows = []

        for lane_key, info in road_network.graph.items():
            if not isinstance(lane_key, str) or not lane_key.startswith("lane_"):
                continue
            if ":" in lane_key:
                continue

            turns = getattr(info, "turns", None) or []
            dirs = set()
            for t in turns:
                d = self._normalize_turn_direction(t.get("direction"))
                if d in {"l", "r", "s", "t"}:
                    dirs.add(d)

            # User-requested placement heuristic:
            # anchor lane does NOT have prohibited maneuver.
            if prohibited in dirs:
                continue

            # There must exist an incoming no-turn candidate lane that turns
            # into this anchor lane with counterpart direction.
            has_counterpart_entry = False
            for src_key in candidate_set:
                src_info = road_network.graph.get(src_key)
                if src_info is None:
                    continue
                for t in (getattr(src_info, "turns", None) or []):
                    d = self._normalize_turn_direction(t.get("direction"))
                    if d != counterpart:
                        continue
                    to_lane = t.get("to_lane")
                    via_lane = t.get("via_lane")
                    if to_lane == lane_key or via_lane == lane_key:
                        if self._is_junction_like_lane_id(to_lane) or self._is_junction_like_lane_id(via_lane):
                            continue
                        has_counterpart_entry = True
                        break

                    # SUMO often stores turn links via an internal junction lane.
                    # Accept anchor lane if it is reachable in one hop from
                    # to_lane/via_lane through exit_lanes.
                    for mid_lane in (to_lane, via_lane):
                        if not mid_lane:
                            continue
                        if self._is_junction_like_lane_id(mid_lane):
                            continue
                        try:
                            mid_obj = road_network.get_lane(mid_lane)
                        except Exception:
                            continue
                        if mid_obj is None:
                            continue
                        next_lanes = set(getattr(mid_obj, "exit_lanes", None) or [])
                        next_lanes = {n for n in next_lanes if not self._is_junction_like_lane_id(n)}
                        if lane_key in next_lanes:
                            has_counterpart_entry = True
                            break
                    if has_counterpart_entry:
                        break
                if has_counterpart_entry:
                    break

            if not has_counterpart_entry:
                continue

            lane = info.lane
            score = min(float(getattr(lane, "length", 0.0)), 80.0)
            if target:
                parts = lane_key.split("_")
                edge_id = parts[1] if len(parts) >= 3 else None
                if edge_id == target:
                    score += 1000.0

            debug_rows.append((lane_key, score, sorted(dirs)))
            if score > best_score:
                best_score = score
                best_lane = lane
                best_key = lane_key

        if self.config.get("debug_one_way_sign_selection", False):
            top_rows = sorted(debug_rows, key=lambda x: x[1], reverse=True)[:8]
            print(
                f"[NoTurnAnchorDebug] sign={self.sign_type} map={self.custom_map_name} "
                f"preferred_road_id={target} chosen={best_key} score={best_score}"
            )
            for lane_key, score, dirs in top_rows:
                print(f"[NoTurnAnchorDebug] candidate lane={lane_key} score={score:.1f} dirs={dirs}")

        return best_lane

    def _pick_priority_candidates(self, road_network, preferred_road_id: Optional[str] = None):
        if self.sign_type not in ("2.1", "2.2", "2.4", "2.3.1", "2.3.2", "2.3.3"):
            return None, []
        
        yield_priority = ("2.1", "2.3.1", "2.3.2", "2.3.3")
        want_priority = True if self.sign_type in yield_priority  else False
        target = str(preferred_road_id) if preferred_road_id is not None else None
        best_lane = None
        best_key = None
        best_score = -10**9
        candidate_lane_keys = []

        for lane_key, info in road_network.graph.items():
            if not isinstance(lane_key, str) or not lane_key.startswith("lane_"):
                continue
            if ":" in lane_key:
                continue
            lane = getattr(info, "lane", None)
            if lane is None:
                continue

            pf = self._lane_priority_flag(lane)
            if pf is None or pf is not want_priority:
                continue

            turns = getattr(info, "turns", None) or []
            exit_lanes = set(getattr(info, "exit_lanes", None) or [])
            maneuver_targets = set()
            for t in turns:
                d = self._normalize_turn_direction(t.get("direction"))
                if d not in {"l", "r", "s"}:
                    continue
                to_lane = t.get("to_lane")
                via_lane = t.get("via_lane")
                if to_lane:
                    maneuver_targets.add(to_lane)
                if via_lane:
                    maneuver_targets.add(via_lane)

            # Keep only lanes that have at least one outgoing movement among
            # left/right/straight reflected in exit_lanes.
            if not (exit_lanes and maneuver_targets and len(exit_lanes.intersection(maneuver_targets)) > 0):
                continue

            candidate_lane_keys.append(lane_key)

            score = min(float(getattr(lane, "length", 0.0)), 80.0)
            if target:
                parts = lane_key.split("_")
                edge_id = parts[1] if len(parts) >= 3 else None
                if edge_id == target:
                    score += 1000.0

            if score > best_score:
                best_score = score
                best_lane = lane
                best_key = lane_key

        if self.config.get("debug_one_way_sign_selection", False):
            print(
                f"[PrioritySignDebug] sign={self.sign_type} candidates={len(candidate_lane_keys)} "
                f"preferred_road_id={target} chosen={best_key}"
            )

        return best_lane, candidate_lane_keys

    def _log_one_way_selection_debug(self, preferred_road_id, best_key, best_score, debug_rows):
        top_rows = sorted(debug_rows, key=lambda x: x[1], reverse=True)[:8]
        header = (
            f"[OneWaySignDebug] sign={self.sign_type} map={self.custom_map_name} "
            f"preferred_road_id={preferred_road_id} chosen={best_key} score={best_score}"
        )
        print(header)
        for lane_key, score, dirs in top_rows:
            print(f"[OneWaySignDebug] candidate lane={lane_key} score={score:.1f} dirs={dirs}")

    def _pick_spawn_lane_from_sign_attachment(self, road_network, sign_lane, sign_kwargs):
        lane_ids = []
        preset = sign_kwargs.get("applicable_lane_indices")
        base_idx = getattr(sign_lane, "index", None)

        # For no-turn signs we intentionally place/render signs on anchor-side
        # lanes, while preset applicable lanes may represent source/topology
        # lanes. Spawn should prefer the physical sign lane first so route
        # naturally passes the sign location.
        prefer_base_first = self.sign_type in ("3.18.1", "3.18.2", "3.19")

        if prefer_base_first and base_idx is not None:
            lane_ids.append(base_idx)
        if preset:
            lane_ids.extend([x for x in preset if x is not None])
        if (not prefer_base_first) and base_idx is not None and base_idx not in lane_ids:
            lane_ids.append(base_idx)

        lanes = []
        for lane_id in lane_ids:
            try:
                lane_obj = road_network.get_lane(lane_id)
            except Exception:
                continue
            if lane_obj is not None:
                lanes.append(lane_obj)

        if not lanes:
            return None
        # Deterministic pick to keep rollouts reproducible across resets.
        idx = int(getattr(self, "current_seed", 0) or 0) % len(lanes)
        return lanes[idx]

    def _spawn_ego_on_lane(self, lane):
        if lane is None:
            return
        try:
            long = min(2.0, max(0.2, float(getattr(lane, "length", 1.0)) * 0.05))
            pos = lane.position(long, 0.0)
            heading = lane.heading_theta_at(long)
            self.vehicle.set_position(pos)
            self.vehicle.set_heading_theta(heading)
        except Exception:
            pass

    def _pick_destination_with_min_hops(self, start_lane_index, road_network, min_hops: int, max_hops: int):
        if start_lane_index not in road_network.graph:
            return None
        min_hops = max(1, int(min_hops))
        max_hops = max(min_hops, int(max_hops))

        visited = {start_lane_index}
        queue = [(start_lane_index, 0)]
        fallback = None

        def _is_routable_destination(lane_idx):
            return lane_idx is not None and not str(lane_idx).startswith("lane_:")

        while queue:
            lane_idx, depth = queue.pop(0)
            if depth >= max_hops:
                continue
            info = road_network.graph.get(lane_idx)
            if info is None:
                continue
            next_lanes = sorted(set(getattr(info, "exit_lanes", None) or []))
            for nxt in next_lanes:
                if nxt not in road_network.graph or nxt in visited:
                    continue
                visited.add(nxt)
                next_depth = depth + 1
                if _is_routable_destination(nxt):
                    fallback = nxt
                if next_depth >= min_hops and _is_routable_destination(nxt):
                    return nxt
                queue.append((nxt, next_depth))

        return fallback
    
    def _pick_secondary_road_candidates(self, road_network, preferred_road_id=None):
        """Return (best_main_lane, list_of_main_lane_keys) for sign type 2.3.1.
        
        Selects main-priority lanes approaching junctions that also have
        at least one secondary-priority lane.
        """
        target = str(preferred_road_id) if preferred_road_id is not None else None
        best_lane = None
        best_score = -1e9
        candidate_keys = []

        for lane_key, info in road_network.graph.items():
            if not isinstance(lane_key, str) or not lane_key.startswith("lane_"):
                continue
            if ":" in lane_key:
                continue
            lane = getattr(info, "lane", None)
            if lane is None:
                continue

            # только главные полосы
            if self._lane_priority_flag(lane) is not True:
                continue

            junction_id = self._normalize_junction_id(
                getattr(lane, "incoming_junction_id", None)
            )
            if junction_id is None:
                continue

            # Проверяем, есть ли на этом перекрёстке второстепенная полоса
            has_secondary = False
            for other_key, other_info in road_network.graph.items():
                other_lane = getattr(other_info, "lane", None)
                if other_lane is None or other_key == lane_key:
                    continue
                other_jid = self._normalize_junction_id(
                    getattr(other_lane, "incoming_junction_id", None)
                )
                if other_jid != junction_id:
                    continue
                if self._lane_priority_flag(other_lane) is False:
                    has_secondary = True
                    break

            if not has_secondary:
                continue

            # Проверяем наличие манёвров (знак должен быть осмысленным)
            turns = getattr(info, "turns", None) or []
            exit_lanes = set(getattr(info, "exit_lanes", None) or [])
            maneuver_targets = set()
            for t in turns:
                to_lane = t.get("to_lane")
                via_lane = t.get("via_lane")
                if to_lane:
                    maneuver_targets.add(to_lane)
                if via_lane:
                    maneuver_targets.add(via_lane)
            if not (exit_lanes and maneuver_targets and exit_lanes.intersection(maneuver_targets)):
                continue

            candidate_keys.append(lane_key)
            score = min(float(getattr(lane, "length", 0.0)), 80.0)
            if target:
                parts = lane_key.split("_")
                edge_id = parts[1] if len(parts) >= 3 else None
                if edge_id == target:
                    score += 1000.0

            if score > best_score:
                best_score = score
                best_lane = lane

        return best_lane, candidate_keys

    def _refresh_navigation_after_spawn(self, spawn_lane):
        if spawn_lane is None:
            return
        nav = getattr(self.vehicle, "navigation", None)
        if nav is None:
            return
        try:
            road_network = self.engine.current_map.road_network
            min_hops = int(self.config.get("min_route_hops_after_spawn", 8))
            max_hops = int(self.config.get("max_route_hops_after_spawn", 10))
            destination = self._pick_destination_with_min_hops(
                spawn_lane.index,
                road_network,
                min_hops=min_hops,
                max_hops=max_hops,
            )
            explicit_destination = getattr(self.vehicle, "config", {}).get("destination", None)
            if explicit_destination is not None:
                destination = explicit_destination
            destination = None
            nav.set_route(spawn_lane.index, destination)
            nav.update_localization(self.vehicle)
        except Exception:
            pass
        
    def reset(self, *, seed=None):
        obs, info = super().reset(seed=seed)

        sign_mgr = self.engine.traffic_sign_manager
        road_network = self.engine.current_map.road_network
        sign_class = SIGN_TYPE_TO_CLASS[self.sign_type]
        self._one_way_candidate_lane_keys = []
        self._lane_direction_candidate_lane_keys = []
        self._no_turn_candidate_lane_keys = []
        self._priority_candidate_lane_keys = []

        # Determine the lane for sign placement using road_id from meta.json
        sign_lane = None
        preferred_road_id = None
        if self.meta and "road_id" in self.meta:
            preferred_road_id = str(self.meta["road_id"])

        # For one-way entry signs at junctions, road_id in meta can be ambiguous.
        # Prefer topology-based lane selection.
        sign_lane = self._pick_one_way_entry_lane(road_network, preferred_road_id)
        lane_source = "topology"

        lane_sign_candidate, lane_sign_keys = self._pick_lane_direction_candidates(road_network, preferred_road_id)
        if lane_sign_candidate is not None:
            sign_lane = lane_sign_candidate
            self._lane_direction_candidate_lane_keys = lane_sign_keys
            lane_source = "lane_direction_topology"

        no_turn_lane_candidate, no_turn_lane_keys = self._pick_no_turn_candidates(road_network, preferred_road_id)
        if no_turn_lane_candidate is not None:
            self._no_turn_candidate_lane_keys = no_turn_lane_keys
            no_turn_anchor_lane = self._pick_no_turn_anchor_lane(
                road_network,
                no_turn_lane_keys,
                preferred_road_id,
            )
            if no_turn_anchor_lane is not None:
                sign_lane = no_turn_anchor_lane
                lane_source = "no_turn_anchor_topology"
            else:
                sign_lane = no_turn_lane_candidate
                lane_source = "no_turn_topology"

        original_lane = getattr(getattr(self, "vehicle", None), "lane", None)
        original_lane_priority = self._lane_priority_flag(original_lane) if original_lane is not None else None

        priority_lane, priority_keys = self._pick_priority_candidates(road_network, preferred_road_id)
        if priority_keys:
            self._priority_candidate_lane_keys = priority_keys
        if priority_lane is not None:
            if self.sign_type in ("2.1", "2.4"):
                sign_lane = priority_lane
                lane_source = "priority_topology"
            elif self.sign_type == "2.2":
                # For 2.2 follow 2.4-style placement unless original lane is
                # already secondary-priority and can be used directly.
                if original_lane is not None and original_lane_priority is False:
                    sign_lane = original_lane
                    lane_source = "original_secondary_lane"
                else:
                    sign_lane = priority_lane
                    lane_source = "priority_topology"
            elif self.sign_type == "2.3.1" or self.sign_type == "2.3.2" or self.sign_type == "2.3.3":
                # Находим главные полосы перекрёстков со второстепенной дорогой
                secondary_lane, secondary_keys = self._pick_secondary_road_candidates(
                    road_network, preferred_road_id
                )
                if secondary_keys:
                    self._priority_candidate_lane_keys = secondary_keys
                if secondary_lane is not None:
                    sign_lane = secondary_lane
                    lane_source = "secondary_road_topology"

        if self.meta and "road_id" in self.meta:
            road_id = str(self.meta["road_id"])
            try:
                if sign_lane is None:
                    lane_key = road_network.find_rightmost_lane_by_road_id(road_id)
                    sign_lane = road_network.get_lane(lane_key)
                    lane_source = f"meta_road_id({road_id})"
            except Exception:
                logging.warning(f"Could not find lane for road_id={road_id}, falling back to vehicle lane")

        if sign_lane is None:
            sign_lane = self.vehicle.lane
            lane_source = "vehicle_lane_fallback"

        if self.config.get("debug_one_way_sign_selection", False) and self.sign_type in ("5.7.1", "5.7.2"):
            print(
                f"[OneWaySignDebug] final lane source={lane_source} "
                f"lane={getattr(sign_lane, 'index', None)} "
                f"sign_spawn_distance={float(self.sign_spawn_distance):.2f}"
            )

        sign_kwargs = {}
        if self.sign_type in ("5.7.1", "5.7.2") and self._one_way_candidate_lane_keys:
            sign_kwargs["applicable_lane_indices"] = list(self._one_way_candidate_lane_keys)
        if self.sign_type in ("4.1.1", "4.1.2", "4.1.3", "4.1.4", "4.1.5", "4.1.6") and self._lane_direction_candidate_lane_keys:
            sign_kwargs["applicable_lane_indices"] = list(self._lane_direction_candidate_lane_keys)
        if self.sign_type in ("3.18.1", "3.18.2", "3.19") and self._no_turn_candidate_lane_keys:
            sign_kwargs["applicable_lane_indices"] = list(self._no_turn_candidate_lane_keys)
        if self.sign_type in ("2.1", "2.4", "2.3.1", "2.3.2", "2.3.3") and self._priority_candidate_lane_keys:
            sign_kwargs["applicable_lane_indices"] = list(self._priority_candidate_lane_keys)
        # if self.sign_type == "2.2" and self._priority_candidate_lane_keys:
        #     sign_kwargs["applicable_lane_indices"] = list(self._priority_candidate_lane_keys)

        # Signs that manage their own placement offsets internally
        from traffic_signs.restricted_lane_sign import (
            RestrictedLaneSign as _RLS,
            IntersectionRestrictedLaneSign as _IRLS,
            EndOfRestrictedLaneSign as _EORLS,
        )
        from traffic_signs.detour_sign import DetourSign as _DS

        if issubclass(sign_class, _RLS):
            # RestrictedLaneSign keeps "sign_spawn_distance metres from lane
            # start" semantics — the zone extends onward from the sign.
            spawn_dist = max(0.0, float(self.sign_spawn_distance))
            sign_mgr.add_sign(
                sign_class, lane=sign_lane,
                longitudinal_offset=-sign_lane.length + spawn_dist,
            )
        elif issubclass(sign_class, (_IRLS, _EORLS)):
            sign_mgr.add_sign(sign_class, lane=sign_lane)
        elif issubclass(sign_class, _DS):
            sign_mgr.add_sign(sign_class, lane=sign_lane)
        else:
            # For approach-style signs (Stop, NoEntry, NoTraffic, TL, etc.)
            # the stop-line is at the intersection entrance, which in SUMO
            # is the END of the sign's edge. If the sign is on the ego's
            # initial lane we offset by `sign_spawn_distance` from the lane
            # start (= near the end on this short edge); otherwise place at
            # lane start so the sign isn't pushed off the parallel edge.
            initial_lane_idx = getattr(getattr(self, "vehicle", None), "lane", None)
            initial_lane_idx = getattr(initial_lane_idx, "index", None)
            sign_lane_idx = getattr(sign_lane, "index", None)
            is_initial_sign_lane = sign_lane_idx == initial_lane_idx
            sign_longitudinal_offset = (
                -sign_lane.length + self.sign_spawn_distance
                if is_initial_sign_lane
                else 0.0
            )
            sign_obj = sign_mgr.add_sign(
                sign_class,
                lane=sign_lane,
                longitudinal_offset=sign_longitudinal_offset,
                lateral_offset=sign_lane.width_at(0) / 2 + 0.8,
                **sign_kwargs,
            )

        # If sign was attached to a concrete lane set, spawn ego on one of those
        # lanes instead of an unrelated default lane.
        if lane_source != "vehicle_lane_fallback":
            # For no-turn signs, spawn only from render lanes.
            if self.sign_type in ("3.18.1", "3.18.2", "3.19"):
                render_lanes = [
                    l for l in list(getattr(locals().get("sign_obj", None), "render_lanes", None) or [])
                    if not str(getattr(l, "index", "")).startswith("junction")
                ]
                if render_lanes:
                    idx = int(getattr(self, "current_seed", 0) or 0) % len(render_lanes)
                    spawn_lane = render_lanes[idx]
                else:
                    spawn_lane = self._pick_spawn_lane_from_sign_attachment(road_network, sign_lane, sign_kwargs)
            else:
                spawn_lane = self._pick_spawn_lane_from_sign_attachment(road_network, sign_lane, sign_kwargs)
            start_entry_lanes = [
                    lane_id
                    for lane_id in spawn_lane.entry_lanes
                    if ":" not in lane_id
                ]
            if len(start_entry_lanes) > 0:
                spawn_lane = road_network.get_lane(start_entry_lanes[0])
            # ==== Очистка зоны от трафика перед респавном эго ====
            # Вычисляем будущую позицию эго (аналогично _spawn_ego_on_lane)
            spawn_long = min(2.0, max(0.2, spawn_lane.length * 0.05))
            ego_spawn_pos = spawn_lane.position(spawn_long, 0.0)
            
            traffic_mgr = self.engine.traffic_manager
            if hasattr(traffic_mgr, 'traffic_vehicles'):
                to_remove = []
                for v in list(traffic_mgr.traffic_vehicles):
                    dist = np.linalg.norm(np.array(v.position) - np.array(ego_spawn_pos))
                    if dist < 15.0:   # радиус очистки (можно уменьшить до 10–12 м)
                        to_remove.append(v)
                for v in to_remove:
                    traffic_mgr.clear_objects([v.id])
                    traffic_mgr._traffic_vehicles.remove(v)
            # =====================================================
            self._spawn_ego_on_lane(spawn_lane)
            self._refresh_navigation_after_spawn(spawn_lane)

        graph = self.engine.map_manager.graph
        for lane_name, lane_node in graph.lanes.items():
            if lane_node.type == 'driving' and lane_name in graph.lane_to_tl_signals:
                sign_mgr.add_sign(
                    TrafficLightSign,
                    lane=road_network.get_lane("lane_" + lane_name),
                    lateral_offset=0,
                    check_zone_overlap=False,
                    sim_step_duration=self.engine.global_config["physics_world_step_size"],
                    tl_speed_factor=self.config.get("tl_speed_factor", 1.0),
                )

        # Optional multi-lane spawn: teleport ego onto a specific parallel lane of the
        # sign's road (by lane_num). Base vehicle was spawned on rightmost (lane_0) via
        # find_rightmost_lane_by_road_id; we move it to the requested lane_num after
        # reset so per-scene benchmarks can verify every parallel lane.
        # Important: after teleport we MUST rebuild navigation, otherwise the route
        # still points to lane_0's checkpoints and the IDM/policy steers ego back
        # toward the original lane (often into oncoming traffic).
        lane_num = self.config.get("spawn_lane_num", None)
        if lane_num is not None and int(lane_num) > 0 and self.meta and "road_id" in self.meta:
            road_id = str(self.meta["road_id"])
            target_key = f"lane_{road_id}_{int(lane_num)}"
            try:
                target_lane = road_network.get_lane(target_key)
                start_long = min(1.0, target_lane.length - 0.1)
                start_pos = np.asarray(target_lane.position(start_long, 0.0), dtype=np.float64)
                start_heading = float(target_lane.heading_theta_at(start_long))
                # Physical teleport
                self.vehicle.set_position(start_pos)
                self.vehicle.set_heading_theta(start_heading)
                # Update spawn_place so a navigation.reset() uses the new point for
                # ray_localization (otherwise nav falls back to original lane_0).
                try:
                    self.vehicle.spawn_place = start_pos.copy()
                except Exception:
                    pass
                # Full navigation rebuild from the new lane using the same
                # destination policy as the initial spawn.
                self._refresh_navigation_after_spawn(target_lane)
            except Exception as e:
                logging.warning(f"spawn_lane_num={lane_num}: could not teleport to {target_key}: {e}")

        return obs, info

    @staticmethod
    def _lane_key_edge(lane_key: str):
        """Extract SUMO edge_id (including leading '-' for reverse direction)
        from a lane key like 'lane_-123#0_2' or '123#0_2'."""
        raw = lane_key[5:] if lane_key.startswith("lane_") else lane_key
        if ":" in raw:
            return None
        return raw.rsplit("_", 1)[0]

    def is_lane_relevant_for_sign(self, lane_key: str, sign_road_id: str, max_depth: int = 8) -> bool:
        """Walk the road graph forward from `lane_key` via exit_lanes (skipping
        internal junction lanes) and return True IFF the first non-internal
        successor is NOT the reverse edge of the sign's road.

        This filters out lanes whose SUMO topology U-turns to the opposing flow
        before crossing the sign's drivable area. The lane itself is assumed to
        sit on the sign's edge (callers pass parallel lanes of `sign_road_id`)
        so the current segment already counts as "on the sign"; we check where
        the graph leads AFTER this lane to detect U-turn routing.
        """
        road_network = getattr(self.engine.current_map, "road_network", None)
        graph = getattr(road_network, "graph", None) if road_network is not None else None
        if graph is None:
            return True  # Fail-open: keep the lane if we can't evaluate the graph
        reverse_edge = ("-" + sign_road_id) if not sign_road_id.startswith("-") else sign_road_id[1:]
        current = lane_key
        visited = {current}
        for _ in range(max_depth):
            info = graph.get(current)
            if info is None:
                return True
            exits = getattr(info, "exit_lanes", []) or []
            next_lane = None
            for e in exits:
                if e in visited:
                    continue
                next_lane = e
                break
            if next_lane is None:
                return True  # Dead end on graph — not a U-turn, keep lane
            visited.add(next_lane)
            current = next_lane
            edge = self._lane_key_edge(current)
            if edge is None:
                continue  # Still inside a junction; keep walking
            if edge == reverse_edge:
                return False  # Forward path leads to opposing flow — NOT relevant
            return True  # Reached a forward-direction successor — relevant
        return True  # Exhausted depth — keep lane (conservative default)

    def list_parallel_lane_nums(self, road_id: str):
        """Return sorted list of available lane_num ints for a SUMO edge road_id."""
        graph = getattr(getattr(self.engine, "map_manager", None), "graph", None)
        if graph is None:
            return [0]
        prefix = f"{road_id}_"
        nums = []
        for k in graph.lanes.keys():
            if k.startswith(prefix):
                try:
                    nums.append(int(k.rsplit("_", 1)[1]))
                except (ValueError, IndexError):
                    continue
        return sorted(set(nums)) or [0]
    
    def get_single_observation(self, _=None):
        return TopDownMultiChannel(
            self.config["vehicle_config"],
            onscreen=self.config["use_render"],
            clip_rgb=True,
            frame_stack=3,
            post_stack=5,
            frame_skip=5,
            resolution=(84, 84),
            max_distance=30
        )
