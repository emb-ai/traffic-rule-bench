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
from traffic_bench.envs.auto_spawn import AutoSpawnMixin
from traffic_bench.envs.lane_node_patch import apply_sumo_lane_node_patch

apply_sumo_lane_node_patch()

from traffic_bench.signs.manager import TrafficSignManager
from traffic_bench.signs.junction.yield_sign import StopSign
from traffic_bench.signs.extra.direction_legacy import DirectionSign
from traffic_bench.signs.extra.lane_directions import LaneDirectionsSign
from traffic_bench.signs.dual_path.no_entry import NoEntrySign
from traffic_bench.signs.speed.min_speed import MinimumSpeedLimitSign
from traffic_bench.signs.blocked.no_traffic import NoTrafficSign
from traffic_bench.signs.speed.limit import SpeedLimitSign, SpeedLimitSign20, SpeedLimitSign40, SpeedLimitSign60
from traffic_bench.signs.extra.traffic_light import TrafficLightSign
from traffic_bench.signs.extra.no_stopping import NoStoppingAllowedSign
from traffic_bench.signs.extra.no_overtaking import NoOvertakingSign
from traffic_bench.signs.speed.zone import ZoneSpeedLimitSign
from traffic_bench.signs.speed.end_of_zone import EndOfZoneSpeedLimitSign
from traffic_bench.signs.speed.residential import ResidentialZoneSign, EndOfResidentialZoneSign
from traffic_bench.signs.speed.limit import SpeedLimitSign
from traffic_bench.signs.extra.no_stopping import NoStoppingAllowedSign
from traffic_bench.signs.dual_path.direction import *
from traffic_bench.signs.speed.zone import ZoneSpeedLimitSign
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from traffic_bench.envs.pedestrians import CrosswalkPedestrianManager
from traffic_bench.envs.crosswalk_enforce import CrosswalkYieldEnforcerManager
from traffic_bench.signs.crosswalk.yield_rule import PedestrianYieldRule
import numpy as np

from traffic_bench.signs.speed.end_of_zone import *
from traffic_bench.signs.junction import *
from traffic_bench.signs.dual_path.no_turn import *
from traffic_bench.signs.dual_path.one_way import *
from traffic_bench.signs.detour.plate import DetourRightSign, DetourLeftSign, DetourEitherSign
from traffic_bench.signs.extra.bus_station import BusStationSign
from traffic_bench.signs.extra.only_auto import OnlyAutoSign
from traffic_bench.signs.extra.restricted_lane import (
    BusLaneRoadSign, BikeLaneRoadSign,
    EndBusLaneRoadSign, EndBikeLaneRoadSign,
    ExitToBusLaneSign, ExitToBusLaneSignLeft,
    ExitToBikeLaneSign, ExitToBikeLaneSignLeft,
    BusLaneSign, BikeLaneSign,
    EndBusLaneSign, EndBikeLaneSign,
)
from traffic_bench.signs.extra.only_auto import OnlyAutoSign

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
    "5.15.1": LaneDirectionsSign,
    "5.15.2": DirectionSign,
    "5.31": ZoneSpeedLimitSign,
    "5.32": EndOfZoneSpeedLimitSign,
    "5.21": ResidentialZoneSign,
    "5.22": EndOfResidentialZoneSign,
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

# Zone-ENTRY signs (PDD 5.21 residential zone). For these the scene must read
# "big road -> sign -> courtyard": the ego spawns on the approaching big road,
# crosses the junction INTO the courtyard, and passes the sign on the INBOUND
# carriageway. Gated separately from BRAKING_SPAWN_CODES so 3.24 (a normal
# through-road speed sign) keeps its existing upstream-spawn behaviour.
# Parameterized so 5.31 (zone speed limit, same entry semantics) can be added.
ZONE_ENTRY_SIGN_CODES = {"5.21"}

# Detour signs (mandatory obstacle detour): the sign lane is the
# OBSTACLE lane (meta sign_lane_index), cones are spawned on it, and ego is
# pinned to that lane so the prescribed lane change is actually exercised.
DETOUR_SIGN_CODES = ("4.2.1", "4.2.2", "4.2.3")


def _edge_base(edge_id: str) -> str:
    """Way base of a directed SUMO edge id: '-794#0' -> '794'."""
    return edge_id.lstrip("-").split("#")[0]


def _edges_are_reverse(a: str, b: str) -> bool:
    """True iff a and b are the two directions of the same way segment (a U-turn:
    same base way id, opposite leading '-')."""
    return _edge_base(a) == _edge_base(b) and a.startswith("-") != b.startswith("-")


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
    # NPC background traffic: metres. >0 → SumoTrajectoryIDMPolicy brakes for ego.
    # Off by default so other benches keep realistic NPC priority; set in
    # direction_signs (and similar skill benches) where NPC→ego crashes poison eval.
    npc_ego_yield_radius=0.0,
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
        spawn_mode="interval",
        ego_spawn_distance_m=15.0,
        target_pedestrian_count=1,
        pedestrian_spawn_gap_s=2.5,
        pedestrian_spawn_chain="time_gap",
        crosswalk_active_tolerance_m=0.05,
    ),
    pedestrian_yield_enforcer=dict(
        enabled=True,
        steer_value=0.0,
        brake_value=-1.0,
        max_forced_per_step=-1,
    ),
)


class TrafficSignSumoEnv(AutoSpawnMixin, BaseEnv):
    @classmethod
    def default_config(cls):
        config = super(TrafficSignSumoEnv, cls).default_config()
        config.update(SUMO_DEFAULT_CONFIG)
        config["map_name"] = "map.net.xml"
        config["sign_type"] = "2.5"
        config["sign_spawn_distance"] = 0.0
        # Speed scenes: background traffic spawns only PAST the plate on the
        # ego's edge, so the approach the plate is judged on is clear of NPCs.
        # -1 / "" disables it (every other family).
        config["traffic_spawn_after_lng"] = -1.0
        config["traffic_spawn_after_edge"] = ""
        # The plate's value (km/h) for the same purpose: cars spawned past the
        # plate start at a speed it allows. 0 = unknown.
        config["traffic_spawn_after_kmh"] = 0.0
        # Share of background cars that obey the plate (speed families; the
        # manifest samples it per variant). 1.0 = every car, as before.
        config["traffic_npc_compliance_rate"] = 1.0
        config["tl_speed_factor"] = 1.0
        # place ego onto a  parallel lane_num
        # sign's road_id after reset:  0 = rightmost
        config["spawn_lane_num"] = 0
        config["debug_one_way_sign_selection"] = False
        config["min_route_hops_after_spawn"] = 2
        config["max_route_hops_after_spawn"] = 4
        # How far PAST the sign the route destination is placed (edges). Keeps the
        # route short: approach -> sign -> a few edges into the zone, instead of
        # winding to the far end of a multi-edge zone. The in-zone metric / zone
        # of effect is independent (see _configure_standalone_zone).
        config["route_forward_edges"] = 3
        # When True (default), after sign placement ego is teleported onto the
        # sign-topology lane. Neural policies (plant2/carl) need False — keep ego
        # on vehicle_config spawn_lane_index / meta road_id instead.
        config["relocate_ego_to_sign_lane"] = True
        # Spawn the physical cone cluster for detour signs (4.2.x).
        config["spawn_detour_cones"] = True
        # NPCs are cleared from the obstacle lane starting this many metres
        # before the detour sign (up to the far end of the cones). Traffic
        # elsewhere around the sign is kept.
        config["detour_clear_before_sign_m"] = 5.0
        # Braking-spawn (3.24): ego starts above the limit, placed d_required
        # before the sign (resolved up the road graph). Disabled by default.
        config["ego_braking_spawn"] = False
        config["ego_spawn_mode"] = "brake"   # "brake" (3.24/5.21/5.31) | "accel" (4.6)
        config["ego_spawn_v0_ms"] = 0.0
        config["ego_brake_d_required"] = 0.0
        config["ego_v_target_kmh"] = 0.0
        config["ego_brake_decel"] = 2.5
        config["ego_brake_delay"] = 1.0
        config["ego_brake_margin"] = 5.0
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
        LaneNode.TREAT_LIGHT_VEHICLE_AS_DRIVING = bool(
            self.config.get("treat_light_vehicle_lanes_as_driving", True)
        )
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
        print(
            f"[sumo] LaneNode.MIN_LANE_WIDTH = {LaneNode.MIN_LANE_WIDTH}, "
            f"TREAT_LIGHT_VEHICLE_AS_DRIVING = {LaneNode.TREAT_LIGHT_VEHICLE_AS_DRIVING}"
        )

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
            nav.set_route(spawn_lane.index, destination)
            route_failed = (
                destination is not None
                and (
                    len(nav.checkpoints) <= 1
                    or nav.checkpoints[-1] == spawn_lane.index
                    or nav.checkpoints[0] == nav.checkpoints[-1]
                )
            )
            if route_failed:
                if explicit_destination is not None:
                    # Don't override explicit destination - log warning instead
                    logging.warning(
                        f"[Navigation] Route from {spawn_lane.index} to explicit destination "
                        f"{explicit_destination} failed (checkpoints loop back). "
                        f"Scene may have invalid routing."
                    )
                else:
                    # Only fall back to random destination if no explicit one was given
                    destination = self._pick_destination_with_min_hops(
                        spawn_lane.index,
                        road_network,
                        min_hops=min_hops,
                        max_hops=max_hops,
                        exclude_lane_indices=[explicit_destination],
                    )
                    nav.set_route(spawn_lane.index, destination)
            nav.update_localization(self.vehicle)
        except Exception:
            pass
        
    def _resolve_detour_lane_and_s(self, road_network, road_id):
        """Resolve the OBSTACLE lane + sign longitudinal position for 4.2.x.

        Uses meta ``sign_lane_index`` (written by detour_scene_editor.py) when
        present; legacy scenes fall back to a feasibility-aware auto-pick: the
        obstacle lane must have an adjacent same-direction lane on the side
        prescribed by the sign (MetaDrive lane_info semantics: ``right_lanes``
        is the physically-right neighbour, i.e. lower SUMO lane index).
        """
        meta = self.meta or {}
        nums = self.list_parallel_lane_nums(road_id)
        lane = None
        idx = meta.get("sign_lane_index")
        if idx is not None and int(idx) in nums:
            try:
                lane = road_network.get_lane(f"lane_{road_id}_{int(idx)}")
            except Exception:
                lane = None
        if lane is None:
            graph = getattr(road_network, "graph", {}) or {}
            need_r = self.sign_type in ("4.2.1", "4.2.3")
            need_l = self.sign_type in ("4.2.2", "4.2.3")

            def feasible(n):
                info = graph.get(f"lane_{road_id}_{n}")
                if info is None:
                    return False
                return bool((need_r and getattr(info, "right_lanes", None)) or
                            (need_l and getattr(info, "left_lanes", None)))

            cand = [n for n in nums if feasible(n)]
            if not cand:
                raise RuntimeError(
                    "detour_infeasible: no adjacent same-direction lane "
                    f"on road_id={road_id} for sign {self.sign_type}"
                )
            above = [c for c in cand if c > 0]
            pick = min(above) if self.sign_type != "4.2.2" and above else cand[0]
            lane = road_network.get_lane(f"lane_{road_id}_{pick}")
        sign_s = float(meta.get("sign_s", meta.get("distance_from_start", 0.0)) or 0.0)
        # Keep the obstacle cluster (sign_s + 3.5 + half-span) on the lane.
        sign_s = min(max(0.5, sign_s), max(0.5, lane.length - 4.5))
        return lane, sign_s

    def reset(self, *, seed=None):
        if self.meta and self.meta.get("excluded"):
            raise RuntimeError(
                f"scene_excluded: {self.meta.get('excluded_reason', 'unspecified')}"
            )
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
                    # Zone-entry signs (5.21): place on the EXACT inbound directed
                    # edge so the sign sits on the entering carriageway, never the
                    # opposite/outbound one. Fall back to the generic picker.
                    if self.sign_type in ZONE_ENTRY_SIGN_CODES:
                        sign_lane = self._lane_for_exact_edge(road_network, road_id)
                        if sign_lane is not None:
                            lane_source = f"meta_road_id_exact({road_id})"
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

        # For DirectionSign (5.15.2), set up trap lane with parallel alternatives
        trap_lane_id = trap_violation_target = trap_adjacent_lane_id = None
        if self.sign_type == "5.15.2":
            trap_lane_id, trap_candidate, trap_violation_target, trap_adjacent_lane_id = self._setup_direction_sign_trap(
                road_network, sign_lane
            )
            if trap_candidate is not None:
                sign_lane = trap_candidate
                lane_source = "direction_sign_trap"
            if trap_lane_id is not None:
                sign_kwargs["trap_lane_id"] = trap_lane_id
            if trap_violation_target is not None:
                sign_kwargs["trap_violation_target"] = trap_violation_target
            if trap_adjacent_lane_id is not None:
                sign_kwargs["trap_adjacent_lane_id"] = trap_adjacent_lane_id

        # Signs that manage their own placement offsets internally
        from traffic_bench.signs.extra.restricted_lane import (
            RestrictedLaneSign as _RLS,
            IntersectionRestrictedLaneSign as _IRLS,
            EndOfRestrictedLaneSign as _EORLS,
        )
        from traffic_bench.signs.detour.plate import DetourSign as _DS

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
            detour_s = None
            if preferred_road_id:
                try:
                    sign_lane, detour_s = self._resolve_detour_lane_and_s(
                        road_network, preferred_road_id
                    )
                    lane_source = f"meta_detour_lane({preferred_road_id})"
                except Exception as e:
                    logging.warning(f"detour lane resolution failed: {e}")
            detour_kwargs = {}
            if detour_s is not None:
                detour_kwargs = {
                    "longitudinal_offset": detour_s,
                    "longitudinal_from_start": True,
                }
            sign_obj = sign_mgr.add_sign(sign_class, lane=sign_lane, **detour_kwargs)
            self._detour_sign_obj = sign_obj
            if sign_obj is not None and self.config.get("spawn_detour_cones", True):
                from traffic_bench.signs.detour.obstacle import spawn_detour_obstacle
                spawn_detour_obstacle(self.engine, sign_lane, sign_obj)
                # Clear NPCs ONLY from the obstacle-lane corridor between
                # `detour_clear_before_sign_m` before the sign and the far end
                # of the cone cluster — otherwise cones spawn on top of an NPC.
                # Traffic on the adjacent lanes and elsewhere near the sign is
                # deliberately kept (ego must merge into real traffic).
                clear_before = float(self.config.get("detour_clear_before_sign_m", 5.0))
                win_start = float(sign_obj.placement_long) - clear_before
                win_end = float(sign_obj.obstacle_long) + 5.0
                half_w = sign_lane.width_at(0) / 2 + 0.3
                traffic_mgr = self.engine.traffic_manager
                if hasattr(traffic_mgr, 'traffic_vehicles'):
                    to_remove = []
                    for v in list(traffic_mgr.traffic_vehicles):
                        try:
                            v_long, v_lat = sign_lane.local_coordinates(v.position)
                        except Exception:
                            continue
                        if win_start <= v_long <= win_end and abs(v_lat) <= half_w:
                            to_remove.append(v)
                    for v in to_remove:
                        traffic_mgr.clear_objects([v.id])
                        traffic_mgr._traffic_vehicles.remove(v)
        elif issubclass(sign_class, (SpeedLimitSign, ZoneSpeedLimitSign, BaseEndOfZoneSign)):
            # Speed/zone/end-of-zone signs use the unified `longitudinal_from_start`
            # convention (offset = meters from the edge START). For a combined
            # pair, place the start sign at its re-projected offset `s_start`;
            # otherwise `sign_spawn_distance` into the edge for approach room.
            if (self.meta or {}).get("is_paired") and (self.meta or {}).get("s_start") is not None:
                spawn_dist = max(0.1, float(self.meta["s_start"]))
            else:
                spawn_dist = max(0.1, float(self.sign_spawn_distance))
            sign_longitudinal_offset = min(spawn_dist, max(0.1, sign_lane.length - 1.0))
            speed_kwargs = dict(sign_kwargs)
            # 3.24 / 5.31: force the enforced limit to the bucketed value
            # ({20,30,40}) so the verifier checks the canonical limit (not the raw
            # road speed) and the icon resolves. 5.31's catalog v_target is now
            # bucketed too, so its runtime sign must use the same value.
            if self.sign_type in ("3.24", "5.31") and float(self.config.get("ego_v_target_kmh", 0) or 0) > 0:
                speed_kwargs["speed_limit_override"] = float(self.config.get("ego_v_target_kmh"))
            sign_obj = sign_mgr.add_sign(
                sign_class,
                lane=sign_lane,
                longitudinal_offset=sign_longitudinal_offset,
                lateral_offset=sign_lane.width_at(0) / 2 + 0.8,
                **speed_kwargs,
            )
        elif issubclass(sign_class, MinimumSpeedLimitSign):
            # 4.6 opens a minimum-speed zone, so it belongs `sign_spawn_distance`
            # into the edge like the other speed plates. Its base still reads the
            # offset from the lane END (it never opted into
            # `longitudinal_from_start`), so convert here. The generic branch
            # below placed it at the lane END whenever the sign lane was not the
            # ego's initial lane -- off-screen, with an empty zone.
            spawn_dist = min(max(0.1, float(self.sign_spawn_distance)),
                             max(0.1, sign_lane.length - 1.0))
            accel_kwargs = dict(sign_kwargs)
            # Enforce the catalog's achievable-capped minimum (20/40) so the
            # verifier checks the value the acceleration scene targets.
            if float(self.config.get("ego_v_target_kmh", 0) or 0) > 0:
                accel_kwargs["min_speed_override"] = float(self.config.get("ego_v_target_kmh"))
            sign_obj = sign_mgr.add_sign(
                sign_class,
                lane=sign_lane,
                longitudinal_offset=spawn_dist - sign_lane.length,
                lateral_offset=sign_lane.width_at(0) / 2 + 0.8,
                **accel_kwargs,
            )
        else:
            # For approach-style signs (Stop, NoEntry, NoTraffic, TL, etc.)
            # the stop-line is at the intersection entrance, which in SUMO
            # is the END of the sign's edge. If the sign is on the ego's
            # initial lane we offset by `sign_spawn_distance` from the lane
            # start (= near the end on this short edge); otherwise place at
            # lane start so the sign isn't pushed off the parallel edge.
            # These signs keep the legacy from-END convention.
            initial_lane_idx = getattr(getattr(self, "vehicle", None), "lane", None)
            initial_lane_idx = getattr(initial_lane_idx, "index", None)
            sign_lane_idx = getattr(sign_lane, "index", None)
            is_initial_sign_lane = sign_lane_idx == initial_lane_idx
            sign_longitudinal_offset = (
                -sign_lane.length + self.sign_spawn_distance
                if is_initial_sign_lane
                else 0.0
            )
            approach_kwargs = dict(sign_kwargs)
            sign_obj = sign_mgr.add_sign(
                sign_class,
                lane=sign_lane,
                longitudinal_offset=sign_longitudinal_offset,
                lateral_offset=sign_lane.width_at(0) / 2 + 0.8,
                **approach_kwargs,
            )

        # If sign was attached to a concrete lane set, spawn ego on one of those
        # lanes instead of an unrelated default lane.
        if self.config.get("relocate_ego_to_sign_lane", True) and lane_source != "vehicle_lane_fallback":
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
            approach_lanes = []
            if trap_lane_id is not None:
                # Walk backward through the road graph from trap_lane_id until
                # we have accumulated at least 80 m of total lane length, so
                # the ego has room to change lanes before the junction.
                seen = set()
                current_idx = trap_lane_id
                total_len = 0.0
                spawn_lane = road_network.get_lane(trap_lane_id)
                spawn_long = 0.0
                approach_lanes = [trap_lane_id]
                for _ in range(10):
                    lane_obj = road_network.get_lane(current_idx)
                    total_len += float(getattr(lane_obj, "length", 0.0))
                    if total_len >= 80.0:
                        spawn_lane = lane_obj
                        spawn_long = total_len - 80.0
                        break
                    spawn_lane = lane_obj
                    spawn_long = 0.0
                    seen.add(current_idx)
                    found = None
                    for k, v in road_network.graph.items():
                        if k in seen:
                            continue
                        if current_idx in set(getattr(v, "exit_lanes", None) or []):
                            found = k
                            break
                    if found is None:
                        break
                    current_idx = found
                    approach_lanes.append(found)
            else:
                start_entry_lanes = [
                    lane_id
                    for lane_id in spawn_lane.entry_lanes
                    if ":" not in lane_id
                ]
                if len(start_entry_lanes) > 0:
                    spawn_lane = road_network.get_lane(start_entry_lanes[0])
                spawn_long = min(2.0, max(0.2, spawn_lane.length * 0.05))
            ego_spawn_pos = spawn_lane.position(spawn_long, 0.0)
            
            traffic_mgr = self.engine.traffic_manager
            if hasattr(traffic_mgr, 'traffic_vehicles'):
                to_remove = []
                for v in list(traffic_mgr.traffic_vehicles):
                    dist = np.linalg.norm(np.array(v.position) - np.array(ego_spawn_pos))
                    if dist < 15.0:
                        to_remove.append(v)
                for v in to_remove:
                    traffic_mgr.clear_objects([v.id])
                    traffic_mgr._traffic_vehicles.remove(v)
            # =====================================================
            if trap_lane_id is not None:
                # For trap scenarios, manually set spawn position from the
                # walkback spawn_lane so the ego has room to change lanes.
                try:
                    heading = spawn_lane.heading_theta_at(spawn_long)
                    self.vehicle.set_position(ego_spawn_pos)
                    self.vehicle.set_heading_theta(heading)
                except Exception:
                    self._spawn_ego_on_lane(spawn_lane)
            else:
                self._spawn_ego_on_lane(spawn_lane)
            self._refresh_navigation_after_spawn(spawn_lane)
            # Detour (4.2.x): ego spawns ON the obstacle lane, so the BFS route
            # from _refresh_navigation_after_spawn already passes the sign edge.
            # Only repair a degenerate [spawn, spawn] route (dead-end BFS pick):
            # route explicitly through the sign edge and onward if possible.
            if self.sign_type in DETOUR_SIGN_CODES and sign_lane is not None:
                nav = getattr(self.vehicle, "navigation", None)
                ckpts = [str(c) for c in (getattr(nav, "checkpoints", None) or [])]
                if len(set(ckpts)) < 2:
                    self._route_through_sign(spawn_lane, sign_lane.index)

            # Force navigation checkpoints through the trap lane to the
            # violation target (which is NOT in the trap lane's allowed set),
            # then extend the route from the violation target via BFS.
            if trap_lane_id is not None and trap_violation_target is not None:
                nav = getattr(self.vehicle, "navigation", None)
                if nav is not None:
                    # Detect if the policy is a sign-compliant expert (has
                    # _handle_direction_compliance) or a vanilla IDM / neural
                    # policy.  Expert route includes the adjacent lane as an
                    # explicit checkpoint so the agent can switch to it and
                    # still follow the navigation.  Non-expert route skips it
                    # and goes straight to violation_target (route-based
                    # violation check triggers).
                    is_expert = False
                    ap = self.config.get("agent_policy")
                    if ap is not None:
                        name = getattr(ap, "__name__", "")
                        if "Expert" in name or "SignCompliant" in name or "Rule" in name:
                            is_expert = True
                    approach_prefix = list(reversed(approach_lanes))
                    if trap_adjacent_lane_id is not None and is_expert:
                        tail = road_network.find_path(trap_adjacent_lane_id, None, max_len=10)
                        if tail:
                            tail = tail[1:]
                        else:
                            tail = []
                        forced_route = approach_prefix + [trap_adjacent_lane_id] + tail
                    else:
                        tail = road_network.find_path(trap_violation_target, None, max_len=10)
                        if tail:
                            tail = tail[1:]
                        else:
                            tail = []
                        forced_route = approach_prefix + [trap_violation_target] + tail
                    nav.checkpoints = forced_route
                    nav.final_lane = road_network.get_lane(forced_route[-1])
                    nav._target_checkpoints_index = [0, 1]
                    if nav._dest_node_path is not None:
                        ref_lane = nav.final_lane
                        later_middle = (float(nav.get_current_lane_num()) / 2 - 0.5) * nav.get_current_lane_width()
                        check_point = ref_lane.position(ref_lane.length, later_middle)
                        from metadrive.utils.math import panda_vector
                        nav._dest_node_path.setPos(panda_vector(check_point[0], check_point[1], nav.MARK_HEIGHT))
                    nav.update_localization(self.vehicle)

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
        road_id = None
        vehicle_cfg = self.config.get("vehicle_config") or {}
        # Detour (4.2.x): when ego was relocated to the obstacle lane above,
        # the parallel-lane teleport must not move it off that lane — the guard
        # covers BOTH entry points (spawn_lane_index is set by the catalog eval
        # path for every row, so it must not bypass the guard). With relocation
        # disabled (NN policies) the teleport IS the pinning mechanism (catalog
        # sets spawn_lane_num = sign_lane_index) — keep it.
        if (self.sign_type in DETOUR_SIGN_CODES
                and self.config.get("relocate_ego_to_sign_lane", True)):
            pass
        elif vehicle_cfg.get("spawn_lane_index"):
            # yield scenes: explicit spawn_lane_index (lane 0 included)
            road_id = str(vehicle_cfg["spawn_lane_index"])
        elif (lane_num is not None and int(lane_num) > 0
                and self.meta and self.meta.get("road_id")):
            # legacy path: relocate via meta.road_id only for lanes > 0
            road_id = str(self.meta["road_id"])
        if lane_num is not None and road_id:
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

        # Paired zone scene: the start sign was placed above (self.sign_type = the
        # start code). If meta carries the matching END sign, place it on its own
        # edge at its re-projected offset so the zone is terminated on the same
        # road (see sumo_space pairing script). Both use the from-start convention.
        self._place_paired_end_sign(sign_mgr, road_network)

        # Build zone boundaries now that all signs are placed: this runs
        # update_zones()/_terminate_zones() so an end-of-zone sign actually
        # truncates the preceding speed/zone limit (otherwise zone_end stays at
        # the edge end / infinity and the end sign has no effect).
        try:
            sign_mgr.build_zones()
        except Exception as exc:
            logging.warning(f"build_zones failed: {exc}")

        # Standalone zone signs (5.21/5.31) with no end-of-zone partner in the
        # scene: their zone is in effect until the end sign (which isn't here),
        # so extend it forward along the connected corridor instead of leaving it
        # clipped to the sign's own edge. Paired scenes already get a multi-edge
        # zone via _place_paired_end_sign (skip those — zone_edges already set).
        if not (self.meta or {}).get("sign_type_end"):
            for _sg in list(sign_mgr.signs):
                if isinstance(_sg, ZoneSpeedLimitSign) and not getattr(_sg, "zone_edges", None):
                    self._configure_standalone_zone(_sg, road_network)

        # Braking-spawn (3.24): place ego above the limit, d_required before the
        # sign (resolved up the road graph). Done LAST so the spawn_lane_num
        # teleport above doesn't clobber it.
        if self.config.get("ego_braking_spawn", False):
            # Spawn-upstream start sign: 3.24 (SpeedLimitSign), 5.31
            # (ZoneSpeedLimitSign), 5.21 (ResidentialZoneSign ⊂ ZoneSpeedLimitSign)
            # for braking; 4.6 (MinimumSpeedLimitSign) for acceleration (ego starts
            # below the min and must speed up). End-of-zone signs excluded.
            start_sign = next(
                (s for s in sign_mgr.signs
                 if isinstance(s, (SpeedLimitSign, ZoneSpeedLimitSign,
                                   MinimumSpeedLimitSign))),
                None,
            )
            if start_sign is not None:
                self._spawn_ego_before_sign(start_sign, road_network)

        return obs, info

    def _place_paired_end_sign(self, sign_mgr, road_network):
        """Place the end-of-zone partner sign for a combined SUMO pair scene.

        Reads paired fields from meta.json (written by the sumo_space pairing
        script): `sign_type_end`, `road_id_end`, `s_end`. No-op when absent.
        """
        meta = self.meta or {}
        end_code = meta.get("sign_type_end")
        if not end_code:
            return
        end_cls = SIGN_TYPE_TO_CLASS.get(end_code)
        if end_cls is None:
            logging.warning(f"paired end sign: unknown sign_type_end={end_code!r}")
            return
        road_id_end = str(meta.get("road_id_end") or meta.get("road_id") or "")
        if not road_id_end:
            return
        try:
            lane_key = road_network.find_rightmost_lane_by_road_id(road_id_end)
            end_lane = road_network.get_lane(lane_key)
        except Exception as exc:
            logging.warning(f"paired end sign: lane for road_id_end={road_id_end} not found: {exc}")
            return
        s_end = float(meta.get("s_end", meta.get("distance_from_start", 0.0)) or 0.0)
        s_end = min(max(0.1, s_end), max(0.1, end_lane.length - 0.5))
        try:
            sign_mgr.add_sign(
                end_cls,
                lane=end_lane,
                longitudinal_offset=s_end,
                lateral_offset=end_lane.width_at(0) / 2 + 0.8,
                use_random_lane=False,
                check_zone_overlap=False,
            )
        except Exception as exc:
            logging.warning(f"paired end sign: add_sign failed: {exc}")

        # Configure the start sign's multi-edge zone so the verifier treats the
        # whole connected route start_edge..end_edge as the zone (the end sign on
        # a downstream edge can't truncate via single-lane get_signs_before).
        zone_edges = meta.get("zone_edges")
        if zone_edges:
            s_start = float(meta.get("s_start", meta.get("distance_from_start", 0.0)) or 0.0)
            for sg in sign_mgr.signs:
                if (isinstance(sg, (SpeedLimitSign, ZoneSpeedLimitSign))
                        and hasattr(sg, "configure_multi_edge_zone")):
                    sg.configure_multi_edge_zone(zone_edges, s_start, s_end)
                    break

    def _configure_standalone_zone(self, sign, road_network, max_edges: int = 12):
        """Extend a zone sign's zone forward along the connected corridor when no
        end-of-zone partner is present in the scene.

        Walks the road graph from the sign's lane via `exit_lanes` (crossing
        internal junction lanes, skipping the reverse-direction U-turn), collects
        the ordered directed edge ids, and calls `configure_multi_edge_zone` so
        the in-zone / violation checks span the whole corridor (matching the
        multi-edge zone that paired scenes get). Mirrors the edge-id format of
        `_sumo_edge_id_from_lane_index`. No-op if nothing reachable downstream.
        """
        if getattr(sign, "zone_edges", None) or not hasattr(sign, "configure_multi_edge_zone"):
            return
        sign_lane = getattr(sign, "lane", None)
        graph = getattr(road_network, "graph", None)
        if sign_lane is None or graph is None:
            return
        e0 = sign._sumo_edge_id_from_lane_index(getattr(sign_lane, "index", None))
        if e0 is None:
            return

        _is_rev = _edges_are_reverse

        zone_edges = [e0]
        seen = {e0}
        cur_key = str(getattr(sign_lane, "index", None))
        cur_edge = e0
        last_lane = sign_lane
        hops = 0
        while len(zone_edges) < max_edges and hops < 60:
            hops += 1
            info = graph.get(cur_key)
            if info is None:
                break
            nxt_key = None
            for ek in sorted(str(e) for e in (getattr(info, "exit_lanes", None) or [])):
                if ek == cur_key:
                    continue
                ee = sign._sumo_edge_id_from_lane_index(ek)
                internal = ":" in ek
                if ee is not None and not internal and (ee in seen or _is_rev(cur_edge, ee)):
                    continue
                nxt_key = ek
                break
            if nxt_key is None:
                break
            cur_key = nxt_key
            ee = sign._sumo_edge_id_from_lane_index(nxt_key)
            if ee is not None and ":" not in nxt_key and ee not in seen:
                zone_edges.append(ee)
                seen.add(ee)
                cur_edge = ee
                try:
                    last_lane = road_network.get_lane(nxt_key)
                except Exception:
                    pass
        if len(zone_edges) <= 1:
            return  # nothing downstream — single-edge default stands
        zone_end_s = float(getattr(last_lane, "length", 0.0) or 0.0)
        try:
            sign.configure_multi_edge_zone(
                zone_edges, float(getattr(sign, "zone_start", 0.0) or 0.0), zone_end_s)
        except Exception as exc:
            logging.warning(f"standalone zone config failed: {exc}")

    @staticmethod
    def _lane_key_edge(lane_key: str):
        """Extract SUMO edge_id (including leading '-' for reverse direction)
        from a lane key like 'lane_-123#0_2' or '123#0_2'."""
        raw = lane_key[5:] if lane_key.startswith("lane_") else lane_key
        if ":" in raw:
            return None
        return raw.rsplit("_", 1)[0]

    def _lane_for_exact_edge(self, road_network, edge_id):
        """Rightmost lane (lowest lane index) on the EXACT directed edge `edge_id`.

        Direction-aware (keeps the leading '-'), so a zone-entry sign lands on the
        INBOUND carriageway and never on the opposite (outbound) directed edge.
        Returns the lane object, or None if `edge_id` has no lane in the graph
        (caller falls back to the generic picker)."""
        graph = getattr(road_network, "graph", None)
        if graph is None:
            return None
        best_key, best_idx = None, None
        for lane_key in graph.keys():
            if not isinstance(lane_key, str) or not lane_key.startswith("lane_"):
                continue
            if self._lane_key_edge(lane_key) != edge_id:
                continue
            try:
                idx = int(lane_key.rsplit("_", 1)[1])
            except (ValueError, IndexError):
                continue
            if best_idx is None or idx < best_idx:
                best_idx, best_key = idx, lane_key
        if best_key is None:
            return None
        try:
            return road_network.get_lane(best_key)
        except Exception:
            return None

    def _forward_reachable_destination(self, sign_lane_index, road_network, max_edges=None):
        """Furthest lane reachable FORWARD from the sign LANE via the live routing
        graph (`road_network.graph` exit_lanes), crossing internal junction lanes
        and skipping the reverse/U-turn edge. Reachable by construction — so
        `set_route` to it won't degenerate — which lets the route CONTINUE past
        the sign even when the catalog's edge-level destination isn't reachable
        from the sign's own lane (different junction branch). Returns a lane index
        or None."""
        graph = getattr(road_network, "graph", None)
        if graph is None or sign_lane_index is None:
            return None
        if max_edges is None:
            max_edges = max(1, int(self.config.get("route_forward_edges", 3)))
        cur_key = str(sign_lane_index)
        cur_edge = self._lane_key_edge(cur_key)
        seen = {cur_edge} if cur_edge else set()
        last_real = None
        for _ in range(40):
            info = graph.get(cur_key)
            if info is None:
                break
            nxt = None
            for ek in sorted(str(e) for e in (getattr(info, "exit_lanes", None) or [])):
                if ek == cur_key:
                    continue
                ee = self._lane_key_edge(ek)  # None for internal ':' lanes
                if ee is not None and (ee in seen
                                       or (cur_edge and _edges_are_reverse(cur_edge, ee))):
                    continue
                nxt = ek
                break
            if nxt is None:
                break
            cur_key = nxt
            ee = self._lane_key_edge(nxt)
            if ee is not None and ":" not in nxt:
                seen.add(ee)
                cur_edge = ee
                last_real = nxt
                if len(seen) - 1 >= max_edges:
                    break
        return last_real

    def _forward_courtyard_destination(self, sign_obj, road_network):
        """Lane index of the furthest courtyard edge reachable FORWARD from the
        sign (skipping the reverse/U-turn edge), for use as the route destination
        so the path is big-road -> sign -> courtyard interior (not ending at the
        sign). Reuses the forward corridor already computed by
        `_configure_standalone_zone` (sign.zone_edges). None if nothing forward."""
        zone_edges = getattr(sign_obj, "zone_edges", None)
        if not zone_edges or len(zone_edges) <= 1:
            return None
        # Cap how far past the sign the route goes: target ~route_forward_edges
        # edges in, closest-resolvable within the cap (never zone_edges[0] = the
        # sign edge), so the route doesn't wind to the far end of the zone.
        cap = max(1, int(self.config.get("route_forward_edges", 3)))
        for k in range(min(cap, len(zone_edges) - 1), 0, -1):
            lane = self._lane_for_exact_edge(road_network, zone_edges[k])
            if lane is not None:
                return lane.index
        return None

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
