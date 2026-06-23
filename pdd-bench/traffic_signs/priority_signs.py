from traffic_signs.base_traffic_sign import BaseTrafficSign, ICONS_DIR
import numpy as np
import os
import re


class MainRoadSign(BaseTrafficSign):
    def __init__(
        self, 
        lane, 
        intersection_name: str = None, 
        **kwargs
    ):
        super().__init__(
            lane, 
            icon_path="2.1.png", 
            **kwargs
        )
        self.intersection_name = intersection_name
        self.is_priority_sign = True
        self.priority_type = "main"
    
    def _is_violating(self, vehicle) -> bool:
        return False
    
    def get_rule_description(self) -> str:
        return "Main road - you have priority at the intersection"
    
    @property
    def top_down_color(self):
        return [255, 204, 0]
    
    @property  
    def top_down_color_name(self):
        return "yellow"


class EndMainRoadSign(BaseTrafficSign):
    def __init__(
        self, 
        lane, 
        intersection_name: str = None, 
        **kwargs
    ):
        super().__init__(
            lane, 
            icon_path="2.2.png", 
            **kwargs
        )
        self.intersection_name = intersection_name
        self.is_priority_sign = True
        self.priority_type = "end_main"
    
    def _is_violating(self, vehicle) -> bool:
        return False
    
    def get_rule_description(self) -> str:
        return "End of main road - priority road ends"
    
    @property
    def top_down_color(self):
        return [200, 200, 200]
    
    @property
    def top_down_color_name(self):
        return "grey"


class YieldSign(BaseTrafficSign):

    EGO_ZONE_BEFORE = 30.0        
    MAIN_ROAD_ZONE_BEFORE = 20.0     
    MAIN_ROAD_ZONE_AFTER = 15.0     

    def __init__(
        self,
        lane,
        intersection_name: str = None,
        main_road_lanes: list = None,
        auto_detect_main_roads: bool = True,
        **kwargs,
    ):
        icon_path = kwargs.pop("icon_path", "2.4.png")
        super().__init__(
            lane, 
            icon_path=icon_path,
            **kwargs
        )
        self.intersection_name = intersection_name
        self.is_priority_sign = True
        self.priority_type = "secondary"
        
        self._vehicle_states = {}

        self.main_road_lanes = main_road_lanes or []
        self._main_road_node_set: set[str] | None = None

        self.zone_start = max(0.0, self.lane.length - self.EGO_ZONE_BEFORE)
        self.zone_end = self.lane.length
        
        # PG intersection topology cache
        self._intersection_type = None  # "X" or "T"
        self._incoming_roads = []       # [(from_node, to_node), ...] in circular order
        self._outgoing_roads = []       # [(from_node, to_node), ...] in circular order
        self._sign_incoming_road = None
        self._main_incoming_roads = []  # Roads that are "main" (perpendicular to sign road)
        self._pg_initialized = False
        self._auto_detect = auto_detect_main_roads
        
    def set_main_road_lanes(self, lanes: list):
        self.main_road_lanes = lanes
        self._main_road_node_set = None
        
    # =========================================================================
    # PG Intersection Topology Detection (similar to no_turn_allowed.py)
    # =========================================================================
    
    @staticmethod
    def _lane_to_road(lane_index):
        """Convert lane index to road tuple (from_node, to_node)."""
        if lane_index is None:
            return None
        if isinstance(lane_index, tuple) and len(lane_index) >= 2:
            return lane_index[0], lane_index[1]
        return None
    
    @staticmethod
    def _opposite_node(node_name):
        """Get the opposite direction node name."""
        node_name = str(node_name)
        if node_name.startswith("-"):
            return node_name[1:]
        return f"-{node_name}"
    
    @staticmethod
    def _extract_intersection_tag(node_name):
        """
        Return ("X"|"T", idx) for nodes like "...X0_0_" / "...T1_0_".
        """
        node_upper = str(node_name).upper()
        match = re.search(r"([XT])(\d)_0_", node_upper)
        if match:
            return match.group(1), int(match.group(2))
        return None
    
    def _infer_intersection_type(self, road_network, sign_road):
        """Infer whether this is an X or T intersection."""
        approach_node = sign_road[1]
        outgoing = road_network.graph.get(approach_node, {})
        tags = set()
        for to_node, lanes in outgoing.items():
            if not lanes:
                continue
            tag = self._extract_intersection_tag(to_node)
            if tag is not None:
                tags.add(tag[0])

        if not tags:
            for from_node, to_dict in road_network.graph.items():
                for to_node, lanes in to_dict.items():
                    if not lanes:
                        continue
                    for node in (from_node, to_node):
                        tag = self._extract_intersection_tag(node)
                        if tag is not None:
                            tags.add(tag[0])

        if "X" in tags:
            return "X"
        if "T" in tags:
            return "T"
        return None
    
    def _road_rank(self, node_name):
        """
        Circular order for intersection roads:
        - X: S(0) -> X0(1) -> X1(2) -> X2(3)
        - T: S(0) -> T0(1) -> T1(2)
        """
        node_upper = str(node_name).upper()
        if "S0_0_" in node_upper:
            return 0
        tag = self._extract_intersection_tag(node_upper)
        if tag is not None and tag[0] == self._intersection_type:
            return 1 + tag[1]
        return 999
    
    def _incoming_road_rank(self, road):
        """Rank for incoming road (uses the approach node)."""
        return self._road_rank(road[1])
    
    def _expected_road_count(self):
        """Expected number of roads at the intersection."""
        return 4 if self._intersection_type == "X" else 3
    
    def _compute_pg_intersection_roads(self):
        """
        Compute all incoming/outgoing roads at the intersection.
        Returns: (incoming_roads, outgoing_roads, sign_road)
        """
        try:
            road_network = self.engine.current_map.road_network
        except Exception:
            return [], [], None
            
        sign_road = self._lane_to_road(self.lane.index)
        if sign_road is None:
            return [], [], None

        self._intersection_type = self._infer_intersection_type(road_network, sign_road)
        if self._intersection_type is None:
            return [], [], sign_road
            
        expected_count = self._expected_road_count()

        # Find all incoming roads to approach nodes
        incoming_to_node = {}
        for from_node, to_dict in road_network.graph.items():
            for to_node, lanes in to_dict.items():
                if lanes:
                    incoming_to_node.setdefault(to_node, []).append((from_node, to_node))

        incoming_roads = []
        for approach_node, to_dict in road_network.graph.items():
            outgoing_branches = [(to_node, lanes) for to_node, lanes in to_dict.items() if lanes]
            if len(outgoing_branches) < 2:
                continue
            candidates = sorted(incoming_to_node.get(approach_node, []), key=lambda r: r[0])
            if candidates:
                incoming_roads.append(candidates[0])

        # Sort by circular order and filter to expected count
        incoming_roads = sorted(set(incoming_roads), key=self._incoming_road_rank)
        incoming_roads = [r for r in incoming_roads if self._incoming_road_rank(r) != 999]
        incoming_roads = incoming_roads[:expected_count]

        # Compute outgoing roads (opposite direction)
        outgoing_roads = []
        for in_from, in_to in incoming_roads:
            out_road = (self._opposite_node(in_to), self._opposite_node(in_from))
            lanes = road_network.graph.get(out_road[0], {}).get(out_road[1], [])
            if lanes:
                outgoing_roads.append(out_road)

        return incoming_roads, outgoing_roads, sign_road
    
    def _identify_main_roads(self):
        """
        Identify which roads are "main roads" that ego must yield to.
        
        For yield sign scenarios:
        - Ego is on a secondary road approaching the intersection
        - Main roads are perpendicular to the ego's approach
        
        Road ordering (circular):
        - X intersection: S(0) -> X0(1) -> X1(2) -> X2(3)
          If sign is at S(0), main roads are X0(1) and X2(3) - the perpendicular ones
        - T intersection: S(0) -> T0(1) -> T1(2)
          If sign is at S(0), main road is T0(1) and T1(2) - the through road
        """
        if self._pg_initialized:
            return
            
        incoming_roads, outgoing_roads, sign_road = self._compute_pg_intersection_roads()
        
        self._incoming_roads = incoming_roads
        self._outgoing_roads = outgoing_roads
        self._sign_incoming_road = sign_road
        
        if not incoming_roads or sign_road is None:
            print(f"[YieldSign] Could not identify intersection roads")
            self._pg_initialized = True
            return
            
        # Find the index of the sign's road in the circular order
        try:
            sign_idx = incoming_roads.index(sign_road)
        except ValueError:
            print(f"[YieldSign] Sign road {sign_road} not in incoming roads {incoming_roads}")
            self._pg_initialized = True
            return
        
        n_roads = len(incoming_roads)
        
        if self._intersection_type == "X":
            # X intersection (4 roads): main roads are at positions ±1 from sign
            # (the two roads perpendicular to the sign's road)
            main_indices = [(sign_idx + 1) % n_roads, (sign_idx + 3) % n_roads]
        else:
            # T intersection (3 roads): main roads are the other two
            main_indices = [(sign_idx + 1) % n_roads, (sign_idx + 2) % n_roads]
        
        self._main_incoming_roads = [incoming_roads[i] for i in main_indices]
        
        print(f"[YieldSign] Intersection type: {self._intersection_type}")
        print(f"[YieldSign] Incoming roads (circular order): {incoming_roads}")
        print(f"[YieldSign] Sign road: {sign_road} (index {sign_idx})")
        print(f"[YieldSign] Main roads (must yield to): {self._main_incoming_roads}")
        
        # Now get the actual lanes for these main roads
        self._auto_set_main_road_lanes()
        
        self._pg_initialized = True
    
    def _auto_set_main_road_lanes(self):
        """Set main_road_lanes based on identified main roads."""
        if not self._main_incoming_roads:
            return
            
        try:
            road_network = self.engine.current_map.road_network
        except Exception:
            return
            
        main_lanes = []
        for road in self._main_incoming_roads:
            from_node, to_node = road
            # Get lanes for this road segment
            lanes = road_network.graph.get(from_node, {}).get(to_node, [])
            main_lanes.extend(lanes)
            
            # Also get the outgoing lanes (opposite direction of this incoming road)
            # These are vehicles that have already entered the intersection from the main road
            out_from = self._opposite_node(to_node)
            out_to = self._opposite_node(from_node)
            out_lanes = road_network.graph.get(out_from, {}).get(out_to, [])
            main_lanes.extend(out_lanes)
        
        if main_lanes:
            self.main_road_lanes = main_lanes
            print(f"[YieldSign] Auto-detected {len(main_lanes)} main road lanes")
            for lane in main_lanes[:4]:  # Print first 4
                print(f"[YieldSign]   - {lane.index}, length={lane.length:.1f}m")

    def _get_all_vehicles(self):
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        return list(
            self.engine.get_objects(
                filter=lambda o: isinstance(o, BaseVehicle)
            ).values()
        )

    def _is_vehicle_in_main_road_conflict_zone(self, vehicle) -> bool:
        """Check if vehicle is in the conflict zone on/near a main road lane."""
        if not self.main_road_lanes:
            return False
            
        try:
            vehicle_pos = vehicle.position
            vehicle_heading = vehicle.heading_theta
            vehicle_lane_idx = getattr(vehicle.lane, 'index', None)
            if vehicle_lane_idx is None:
                return False
            v_segment = (vehicle_lane_idx[0], vehicle_lane_idx[1])
        except Exception:
            return False
        
        main_segments = set()
        for ln in self.main_road_lanes:
            ln_idx = getattr(ln, 'index', None)
            if ln_idx and len(ln_idx) >= 2:
                main_segments.add((ln_idx[0], ln_idx[1]))
        
        for lane in self.main_road_lanes:
            try:
                lane_idx = getattr(lane, 'index', None)
                if lane_idx is None or len(lane_idx) < 2:
                    continue
                lane_segment = (lane_idx[0], lane_idx[1])
                long_pos, lat_pos = lane.local_coordinates(vehicle_pos)
                
                zone_start = max(0.0, lane.length - self.MAIN_ROAD_ZONE_BEFORE)
                zone_end = lane.length + self.MAIN_ROAD_ZONE_AFTER
                
                if zone_start <= long_pos <= zone_end:
                    if abs(lat_pos) <= lane.width * 1.5:  # Allow some lateral tolerance
                        if v_segment == lane_segment:
                            return True
                        if v_segment not in main_segments:
                            lane_heading = lane.heading_theta_at(min(long_pos, lane.length))
                            heading_diff = abs(vehicle_heading - lane_heading)
                            heading_diff = min(heading_diff, 2 * np.pi - heading_diff)
                            if heading_diff < np.pi / 2:
                                return True
            except Exception:
                continue
        return False
    
    def is_vehicle_on_main_road(self, vehicle) -> bool:
        """
        Public method to check if a vehicle is on the main road.
        Useful for external status displays.
        """
        # Ensure main roads are identified
        if self._auto_detect and not self._pg_initialized:
            self._identify_main_roads()
        return self._is_vehicle_in_main_road_conflict_zone(vehicle)
    
    def has_main_road_traffic(self, exclude_vehicle=None) -> tuple:
        """
        Public method to check if there's traffic on the main road.
        Returns (has_traffic: bool, vehicles: list)
        """
        # Ensure main roads are identified
        if self._auto_detect and not self._pg_initialized:
            self._identify_main_roads()
            
        if not self.main_road_lanes:
            return False, []
            
        conflicting = []
        for v in self._get_all_vehicles():
            if exclude_vehicle is not None and v.id == exclude_vehicle.id:
                continue
            if self._is_vehicle_in_main_road_conflict_zone(v):
                conflicting.append(v)
        
        return len(conflicting) > 0, conflicting

    def _check_main_road_traffic(self, ego_vehicle) -> tuple:
        """Check if there are vehicles on the main road in the conflict zone."""
        # Ensure main roads are identified (lazy initialization)
        if self._auto_detect and not self._pg_initialized:
            self._identify_main_roads()
        
        if not self.main_road_lanes:
            return False, []
            
        conflicting = []
        for v in self._get_all_vehicles():
            if v.id == ego_vehicle.id:
                continue
            if self._is_vehicle_in_main_road_conflict_zone(v):
                conflicting.append(v)
        
        return len(conflicting) > 0, conflicting

    def _is_vehicle_in_zone(self, vehicle) -> bool:
        """Check if the vehicle is within the yield zone."""
        vehicle_idx = vehicle.lane.index
        sign_idx = self.lane.index

        same_road = (vehicle_idx[0] == sign_idx[0] and vehicle_idx[1] == sign_idx[1])
        if not same_road:
            return False
        
        veh_long = vehicle.lane.local_coordinates(vehicle.position)[0]
        return self.zone_start <= veh_long <= self.zone_end

    def _is_violating(self, vehicle) -> bool:
        """Check if the vehicle is violating the yield sign."""
        default_state = {
            "reported_for_reward": False,
            "reported_for_metrics": False,
            "had_traffic_while_in_zone": False,
            "last_violation_step": -1,
        }
        state = self._vehicle_states.setdefault(vehicle.id, default_state)
        
        # Ensure YieldSign-specific keys exist
        state.setdefault("had_traffic_while_in_zone", False)
        state.setdefault("last_violation_step", -1)
        state.setdefault("last_violation_result", False)

        current_step = self.engine.episode_step
        if state["last_violation_step"] == current_step:        # cache violation result
            return state.get("last_violation_result", False)

        in_zone_now = self._is_vehicle_in_zone(vehicle)
        has_traffic, _ = self._check_main_road_traffic(vehicle)

        violation = not in_zone_now and state["had_traffic_while_in_zone"]
        state["had_traffic_while_in_zone"] = has_traffic and in_zone_now

        state["last_violation_step"] = current_step
        state["last_violation_result"] = violation
        return violation

    def get_rule_description(self) -> str:
        return (
            "Yield sign (2.4) - must not leave yield zone "
            "while traffic is present on main road"
        )

    @property
    def top_down_color(self):
        return [255, 0, 0]

    @property
    def top_down_color_name(self):
        return "red"


class RightHandYieldSign(YieldSign):
    """Equal-priority intersection rule tracker (right-hand yield).

    Not a separate PDD plate — used with MainRoadSign (2.1) on all approaches.
    Reuses YieldSign zone logic with ``right_road_lanes`` passed as
    ``main_road_lanes``. Violation = leaving the approach zone while traffic
    was present on the conflicting approach from the right.
  """

    def __init__(
        self,
        lane,
        intersection_name: str = None,
        right_road_lanes: list = None,
        **kwargs,
    ):
        kwargs.setdefault("show_model", False)
        kwargs.setdefault("icon_path", "2.1.png")
        super().__init__(
            lane,
            intersection_name=intersection_name,
            main_road_lanes=right_road_lanes,
            auto_detect_main_roads=False,
            **kwargs,
        )
        self.priority_type = "right_hand_yield"

    def get_rule_description(self) -> str:
        return (
            "Right-hand rule at equal-priority intersection — "
            "must not leave approach zone while traffic is on the right"
        )


class StopSign(YieldSign):
    """Stop sign (2.5) — yield to main-road traffic in zone + mandatory stop at line."""

    STOP_SPEED_THRESHOLD_MPS = 0.5
    STOP_LINE_PAST_MARGIN_M = 0.3

    def __init__(
        self,
        lane,
        intersection_name: str = None,
        main_road_lanes: list = None,
        auto_detect_main_roads: bool = True,
        **kwargs,
    ):
        icon_path = kwargs.pop("icon_path", "2.5.png")
        super().__init__(
            lane,
            intersection_name=intersection_name,
            main_road_lanes=main_road_lanes,
            auto_detect_main_roads=auto_detect_main_roads,
            icon_path=icon_path,
            **kwargs,
        )
        self.priority_type = "stop_secondary"
        self.stop_line_position = float(self.placement_long)
        self._vehicle_states_stop: dict = {}

    def _create_visual_model(self):
        from metadrive.engine.asset_loader import AssetLoader

        model_path = AssetLoader.file_path("models", "traffic_sign", "stop_sign.gltf")
        model = self.loader.loadModel(model_path)
        model.setPos(0, 0, self.sign_height)
        model.setH(-90)
        self._visual_model = model.instanceTo(self.origin)

    def _is_on_sign_road(self, vehicle) -> bool:
        veh_lane = getattr(vehicle, "lane", None)
        if veh_lane is None:
            return False
        from traffic_signs.base_traffic_sign import same_road_check

        return same_road_check(
            getattr(veh_lane, "index", None),
            getattr(self.lane, "index", None),
        )

    def _track_stop_before_line(self, vehicle) -> bool:
        """Return True once the vehicle has made a complete stop before the line."""
        if not self._is_on_sign_road(vehicle):
            return False
        try:
            veh_long = self.lane.local_coordinates(vehicle.position)[0]
        except Exception:
            return False

        vid = vehicle.id
        in_zone = self.zone_start <= veh_long <= self.zone_end
        if not in_zone:
            self._vehicle_states_stop.pop(vid, None)
            return False

        state = self._vehicle_states_stop.setdefault(vid, {"stopped_before_line": False})
        stop_long = self.stop_line_position
        speed = float(getattr(vehicle, "speed", 0.0) or 0.0)
        if veh_long < stop_long and speed < self.STOP_SPEED_THRESHOLD_MPS:
            state["stopped_before_line"] = True
        return bool(state["stopped_before_line"])

    def _is_stop_line_violating(self, vehicle) -> bool:
        """True if the vehicle crossed the stop line without stopping first."""
        if not self._is_on_sign_road(vehicle):
            return False
        try:
            veh_long = self.lane.local_coordinates(vehicle.position)[0]
        except Exception:
            return False

        if veh_long < self.stop_line_position + self.STOP_LINE_PAST_MARGIN_M:
            self._track_stop_before_line(vehicle)
            return False

        stopped = self._track_stop_before_line(vehicle)
        return not stopped

    def _is_violating(self, vehicle) -> bool:
        """Yield-zone traffic rule + mandatory stop before the sign line."""
        if super()._is_violating(vehicle):
            return True
        return self._is_stop_line_violating(vehicle)

    def get_rule_description(self) -> str:
        return (
            "Stop sign (2.5) - must stop before the sign line and must not leave "
            "the approach zone while traffic is present on the main road"
        )

    @property
    def top_down_color(self):
        return [255, 255, 255]

    @property
    def top_down_color_name(self):
        return "white"


class SecondaryRoadSign(BaseTrafficSign):
    """Sign 2.3.1 – Intersection with a secondary road (X crossroad variant)."""

    def __init__(self, lane, intersection_name: str = None, **kwargs):
        super().__init__(lane, icon_path="2.3.1.png", **kwargs)
        self.intersection_name = intersection_name
        self.is_priority_sign = True
        self.priority_type = "secondary_road_ahead"

    def _is_violating(self, vehicle) -> bool:
        return False

    def get_rule_description(self) -> str:
        return "Intersection with secondary road ahead (2.3.1) - you have priority"

    @property
    def top_down_color(self):
        return [255, 204, 0]

    @property
    def top_down_color_name(self):
        return "yellow"


class SecondaryRoadLeftSign(BaseTrafficSign):
    """Sign 2.3.2 – Secondary road on the left (T crossroad variant)."""

    def __init__(self, lane, intersection_name: str = None, **kwargs):
        super().__init__(lane, icon_path="2.3.3.png", **kwargs)
        self.intersection_name = intersection_name
        self.is_priority_sign = True
        self.priority_type = "secondary_road_left"

    def _is_violating(self, vehicle) -> bool:
        return False

    def get_rule_description(self) -> str:
        return "Secondary road on the left (2.3.2) - you have priority"

    @property
    def top_down_color(self):
        return [255, 204, 0]

    @property
    def top_down_color_name(self):
        return "yellow"


class SecondaryRoadRightSign(BaseTrafficSign):
    """Sign 2.3.3 – Secondary road on the right (T crossroad variant)."""

    def __init__(self, lane, intersection_name: str = None, **kwargs):
        super().__init__(lane, icon_path="2.3.2.png", **kwargs)
        self.intersection_name = intersection_name
        self.is_priority_sign = True
        self.priority_type = "secondary_road_right"

    def _is_violating(self, vehicle) -> bool:
        return False

    def get_rule_description(self) -> str:
        return "Secondary road on the right (2.3.3) - you have priority"

    @property
    def top_down_color(self):
        return [255, 204, 0]

    @property
    def top_down_color_name(self):
        return "yellow"


class EndMainRoadSmartSign(YieldSign):
    """2.2 logic:
    - if lane has traffic-light signals -> check as traffic-light rule
    - else -> behave like yield (2.4)
    """

    def __init__(self, lane, intersection_name: str = None, debug_priority: bool = True, **kwargs):
        tl_speed_factor = kwargs.get("tl_speed_factor", 1.0)
        icon_path = kwargs.pop("icon_path", "2.2.png")
        super().__init__(
            lane,
            intersection_name=intersection_name,
            main_road_lanes=None,
            debug_priority=debug_priority,
            icon_path=icon_path,
            **kwargs,
        )
        self.priority_type = "end_main_smart"
        self._tl_rule = None
        if getattr(lane, "tl_signals", None):
            self._tl_rule = TrafficLightSign(
                lane,
                sim_step_duration=self.engine.global_config.get("physics_world_step_size", 0.1),
                tl_speed_factor=tl_speed_factor,
                show_model=False,
            )
            
    def get_top_down_icon_poses(self):
        """Draw the icon only on the lane where the sign is placed."""
        road_network = getattr(getattr(self.engine, "current_map", None), "road_network", None)
        if road_network is None or self.lane is None:
            return []
        
        try:
            pos = self.lane.position(self.placement_long, self._lateral_offset)
            heading = self.lane.heading_theta_at(self.placement_long) + np.pi / 2
            return [(pos, heading)]
        except Exception:
            return []

    def update_state(self):
        if self._tl_rule is not None:
            self._tl_rule.update_state()

    def _is_violating(self, vehicle) -> bool:
        if self._tl_rule is not None:
            return self._tl_rule._is_violating(vehicle)
        return super()._is_violating(vehicle)


__all__ = [
    "MainRoadSign",
    "EndMainRoadSign",
    "EndMainRoadSmartSign",
    "YieldSign",
    "RightHandYieldSign",
    "StopSign",
    "SecondaryRoadSign",
    "SecondaryRoadLeftSign",
    "SecondaryRoadRightSign",
]
