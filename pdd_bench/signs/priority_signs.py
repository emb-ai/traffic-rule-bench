from pdd_bench.signs.base_traffic_sign import BaseTrafficSign, ICONS_DIR, same_road_check
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
    YIELD_STOP_BEFORE_END = 5.0
    # Treat yield obligation as cleared when the vehicle center passes this far before
    # the lane end (~half car length), so a junction entry/crash is not missed while
    # the rear is still geometrically on the approach lane.
    EGO_ZONE_END_CENTER_INSET = 4.0
    MAIN_ROAD_ZONE_BEFORE = 15.0
    MAIN_ROAD_ZONE_AFTER = 5.0
    # Path-geometry sticky yield: sample routes and require foe to clear the
    # ego/foe path intersection after the coarse MAIN_ROAD_ZONE prefilter.
    PATH_SAMPLE_STEP_M = 2.0
    PATH_AHEAD_M = 100.0
    # Crossing / near-miss: treat as conflict when polylines come this close.
    # ~5.5m covers adjacent same-direction lane centers that never literally cross.
    CONFLICT_PATH_TOLERANCE_M = 5.5
    # Extra metres past the conflict point before releasing yield (0 = release
    # as soon as the conflict is behind the foe).
    CONFLICT_CLEARANCE_M = 0.0

    def __init__(
        self,
        lane,
        intersection_name: str = None,
        main_road_lanes: list = None,
        auto_detect_main_roads: bool = True,
        **kwargs,
    ):
        # Explicit deny-list of outgoing / post-junction lane keys or edge ids.
        outgoing_lane_keys = kwargs.pop("outgoing_lane_keys", None)
        outgoing_edge_ids = kwargs.pop("outgoing_edge_ids", None)
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
        # foe_id -> {"conflict_point": np.ndarray|None} once seen in main zone
        self._path_sticky_foes: dict = {}
        # Foes already cleared for this passage through the main zone (no re-arm).
        self._path_released_foes: set = set()

        self.main_road_lanes = main_road_lanes or []
        self._main_road_node_set: set[str] | None = None
        self._main_approach_lane_ids: set = set()
        self._main_approach_edge_ids: set[str] = set()
        self._outgoing_lane_ids: set = set(outgoing_lane_keys or [])
        self._outgoing_edge_ids: set[str] = set(str(e) for e in (outgoing_edge_ids or []))
        self._refresh_main_approach_index()

        self.zone_start = max(0.0, self.lane.length - self.EGO_ZONE_BEFORE)
        self.zone_end = self.lane.length
        self.stop_line_position = max(
            0.0, float(self.lane.length) - float(self.YIELD_STOP_BEFORE_END)
        )
        
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
        self._refresh_main_approach_index()

    def set_outgoing_exclusions(
        self,
        outgoing_lane_keys: list | None = None,
        outgoing_edge_ids: list | None = None,
    ) -> None:
        """Record post-junction / outgoing edges that must never trigger yield."""
        if outgoing_lane_keys is not None:
            self._outgoing_lane_ids = set(outgoing_lane_keys)
        if outgoing_edge_ids is not None:
            self._outgoing_edge_ids = set(str(e) for e in outgoing_edge_ids)

    @staticmethod
    def _edge_id_from_lane_index(lane_index) -> str | None:
        if lane_index is None:
            return None
        if isinstance(lane_index, str):
            raw = lane_index[5:] if lane_index.startswith("lane_") else lane_index
            return raw.rsplit("_", 1)[0] if "_" in raw else raw
        if isinstance(lane_index, (tuple, list)) and len(lane_index) >= 2:
            return f"{lane_index[0]}->{lane_index[1]}"
        return str(lane_index)

    def _refresh_main_approach_index(self) -> None:
        """Cache lane-id / edge-id allowlists for incoming main approaches only."""
        self._main_approach_lane_ids = set()
        self._main_approach_edge_ids = set()
        for lane in self.main_road_lanes or []:
            idx = getattr(lane, "index", None)
            if idx is None:
                continue
            self._main_approach_lane_ids.add(idx)
            edge = self._edge_id_from_lane_index(idx)
            if edge is not None:
                self._main_approach_edge_ids.add(str(edge))

    def _is_outgoing_lane_index(self, lane_index) -> bool:
        """True for explicit outgoing exclusions (never a yield-conflict lane)."""
        if lane_index is None:
            return False
        if lane_index in self._outgoing_lane_ids:
            return True
        edge = self._edge_id_from_lane_index(lane_index)
        return edge is not None and str(edge) in self._outgoing_edge_ids

    def _is_on_main_approach(self, lane_index) -> bool:
        """Vehicle lane must be one of the monitored incoming main approaches."""
        if lane_index is None:
            return False
        if self._is_outgoing_lane_index(lane_index):
            return False
        if lane_index in self._main_approach_lane_ids:
            return True
        edge = self._edge_id_from_lane_index(lane_index)
        return edge is not None and str(edge) in self._main_approach_edge_ids

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
        """True only while the vehicle is still on a main *incoming approach*.

        Outgoing edges and junction connectors never count — even if an aux is
        on them after clearing the intersection.
        """
        if not self.main_road_lanes:
            return False

        try:
            vehicle_pos = vehicle.position
            vehicle_lane_idx = getattr(vehicle.lane, "index", None)
            if vehicle_lane_idx is None:
                return False
        except Exception:
            return False

        # Hard reject: outgoing / post-junction pieces.
        if self._is_outgoing_lane_index(vehicle_lane_idx):
            return False
        if not self._is_on_main_approach(vehicle_lane_idx):
            return False

        for lane in self.main_road_lanes:
            try:
                lane_idx = getattr(lane, "index", None)
                if lane_idx is None:
                    continue
                if not same_road_check(vehicle_lane_idx, lane_idx):
                    continue
                long_pos, lat_pos = lane.local_coordinates(vehicle_pos)
                zone_start = max(0.0, float(lane.length) - float(self.MAIN_ROAD_ZONE_BEFORE))
                zone_end = float(lane.length) + float(self.MAIN_ROAD_ZONE_AFTER)
                # Still on the approach geometry (not past the lane end).
                if zone_start <= long_pos <= zone_end and abs(lat_pos) <= lane.width * 1.5:
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

    @staticmethod
    def _xy(vehicle) -> np.ndarray | None:
        try:
            pos = vehicle.position
            return np.asarray([float(pos[0]), float(pos[1])], dtype=float)
        except Exception:
            return None

    def _route_polyline(
        self,
        vehicle,
        *,
        ahead_m: float | None = None,
        step_m: float | None = None,
    ) -> list[np.ndarray]:
        """Sample world XY points along the vehicle's remaining navigation route."""
        ahead = float(self.PATH_AHEAD_M if ahead_m is None else ahead_m)
        step = float(self.PATH_SAMPLE_STEP_M if step_m is None else step_m)
        origin = self._xy(vehicle)
        if origin is None:
            return []

        points: list[np.ndarray] = [origin]
        lane = getattr(vehicle, "lane", None)
        nav = getattr(vehicle, "navigation", None)
        checkpoints = list(getattr(nav, "checkpoints", None) or []) if nav is not None else []
        road_network = None
        try:
            road_network = nav.map.road_network if nav is not None else None
        except Exception:
            road_network = None

        remaining = ahead
        if road_network is not None and checkpoints:
            current_idx = getattr(lane, "index", None)
            try:
                start_i = checkpoints.index(current_idx) if current_idx in checkpoints else 0
            except Exception:
                start_i = 0
            for i in range(start_i, len(checkpoints)):
                if remaining <= 0.0:
                    break
                try:
                    ck_lane = road_network.get_lane(checkpoints[i])
                except Exception:
                    continue
                if ck_lane is None:
                    continue
                if i == start_i and lane is not None:
                    try:
                        long0 = float(ck_lane.local_coordinates(vehicle.position)[0])
                    except Exception:
                        long0 = 0.0
                else:
                    long0 = 0.0
                long0 = max(0.0, min(long0, float(ck_lane.length) - 1e-3))
                s = long0
                lane_len = float(ck_lane.length)
                # Roundabout ego finish cap: stop the sampled path at the
                # configured longitude on the final navigation lane.
                dest_cap = getattr(vehicle, "_priority_bench_dest_along_m", None)
                if dest_cap is not None and i == len(checkpoints) - 1:
                    try:
                        lane_len = min(lane_len, float(dest_cap))
                    except (TypeError, ValueError):
                        pass
                while s < lane_len - 1e-3 and remaining > 0.0:
                    try:
                        p = ck_lane.position(s, 0.0)
                        points.append(np.asarray([float(p[0]), float(p[1])], dtype=float))
                    except Exception:
                        break
                    ds = min(step, lane_len - s, remaining)
                    if ds <= 1e-6:
                        break
                    s += ds
                    remaining -= ds
        elif lane is not None:
            try:
                long0 = float(lane.local_coordinates(vehicle.position)[0])
            except Exception:
                long0 = 0.0
            s = max(0.0, long0)
            lane_len = float(lane.length)
            while s < lane_len - 1e-3 and remaining > 0.0:
                try:
                    p = lane.position(s, 0.0)
                    points.append(np.asarray([float(p[0]), float(p[1])], dtype=float))
                except Exception:
                    break
                ds = min(step, lane_len - s, remaining)
                if ds <= 1e-6:
                    break
                s += ds
                remaining -= ds

        return points

    @staticmethod
    def _segment_closest_points(
        a0: np.ndarray,
        a1: np.ndarray,
        b0: np.ndarray,
        b1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Closest points on segments a0–a1 and b0–b1, plus distance."""
        a = a1 - a0
        b = b1 - b0
        r = a0 - b0
        aa = float(np.dot(a, a))
        bb = float(np.dot(b, b))
        ab = float(np.dot(a, b))
        ar = float(np.dot(a, r))
        br = float(np.dot(b, r))
        denom = aa * bb - ab * ab
        if denom < 1e-12:
            s = 0.0
        else:
            s = (ab * br - bb * ar) / denom
        s = max(0.0, min(1.0, s))
        if bb < 1e-12:
            t = 0.0
        else:
            t = (ab * s + br) / bb
            t = max(0.0, min(1.0, t))
            if denom >= 1e-12:
                s = (ab * t - ar) / aa if aa >= 1e-12 else 0.0
                s = max(0.0, min(1.0, s))
        pa = a0 + s * a
        pb = b0 + t * b
        return pa, pb, float(np.linalg.norm(pa - pb))

    def _paths_conflict_point(
        self,
        ego_path: list[np.ndarray],
        foe_path: list[np.ndarray],
        *,
        tolerance_m: float | None = None,
    ) -> np.ndarray | None:
        """First point along the ego route where paths come within tolerance.

        Walking ego-forward (not a global closest) keeps the conflict at the
        junction merge/crossing instead of sliding down a shared exit arm.
        """
        if len(ego_path) < 2 or len(foe_path) < 2:
            return None
        tol = float(
            self.CONFLICT_PATH_TOLERANCE_M if tolerance_m is None else tolerance_m
        )
        for i in range(len(ego_path) - 1):
            best_dist = float("inf")
            best_mid: np.ndarray | None = None
            for j in range(len(foe_path) - 1):
                pa, pb, dist = self._segment_closest_points(
                    ego_path[i], ego_path[i + 1], foe_path[j], foe_path[j + 1]
                )
                if dist < best_dist:
                    best_dist = dist
                    best_mid = 0.5 * (pa + pb)
            if best_mid is not None and best_dist <= tol:
                return best_mid
        return None

    def _paths_closest_midpoint(
        self,
        ego_path: list[np.ndarray],
        foe_path: list[np.ndarray],
    ) -> tuple[np.ndarray | None, float]:
        """Return (midpoint, distance) of the globally closest path-segment pair."""
        if len(ego_path) < 2 or len(foe_path) < 2:
            return None, float("inf")
        best_dist = float("inf")
        best_mid: np.ndarray | None = None
        for i in range(len(ego_path) - 1):
            for j in range(len(foe_path) - 1):
                pa, pb, dist = self._segment_closest_points(
                    ego_path[i], ego_path[i + 1], foe_path[j], foe_path[j + 1]
                )
                if dist < best_dist:
                    best_dist = dist
                    best_mid = 0.5 * (pa + pb)
        return best_mid, best_dist

    def _future_route_edge_ids(self, vehicle) -> set[str]:
        """Edge ids on the remaining navigation route (any lane on an edge)."""
        edges: set[str] = set()
        nav = getattr(vehicle, "navigation", None)
        checkpoints = list(getattr(nav, "checkpoints", None) or []) if nav else []
        lane = getattr(vehicle, "lane", None)
        current_idx = getattr(lane, "index", None)
        start_i = 0
        if checkpoints and current_idx in checkpoints:
            try:
                start_i = checkpoints.index(current_idx)
            except ValueError:
                start_i = 0
        for cp in checkpoints[start_i:]:
            edge = self._edge_id_from_lane_index(cp)
            if edge:
                edges.add(str(edge))
        cur_edge = self._edge_id_from_lane_index(current_idx)
        if cur_edge:
            edges.add(str(cur_edge))
        return edges

    def _routes_share_future_edge(self, ego_vehicle, foe_vehicle) -> bool:
        """True when both routes will use the same edge (any lane), e.g. same exit arm."""
        return bool(self._shared_future_edges(ego_vehicle, foe_vehicle))

    def _shared_future_edges(self, ego_vehicle, foe_vehicle) -> set[str]:
        ego_edges = self._future_route_edge_ids(ego_vehicle)
        foe_edges = self._future_route_edge_ids(foe_vehicle)
        sign_edge = self._edge_id_from_lane_index(getattr(self.lane, "index", None))
        shared = ego_edges & foe_edges
        if sign_edge:
            shared.discard(str(sign_edge))
        return shared

    def _first_point_on_shared_edge(
        self,
        vehicle,
        shared_edges: set[str],
        path: list[np.ndarray],
    ) -> np.ndarray | None:
        """Merge point: start of the first shared edge on the vehicle's route."""
        if not shared_edges or len(path) < 1:
            return None
        nav = getattr(vehicle, "navigation", None)
        checkpoints = list(getattr(nav, "checkpoints", None) or []) if nav else []
        try:
            road_network = nav.map.road_network if nav is not None else None
        except Exception:
            road_network = None
        if road_network is None or not checkpoints:
            return path[min(1, len(path) - 1)]

        lane = getattr(vehicle, "lane", None)
        current_idx = getattr(lane, "index", None)
        start_i = 0
        if current_idx in checkpoints:
            try:
                start_i = checkpoints.index(current_idx)
            except ValueError:
                start_i = 0

        for cp in checkpoints[start_i:]:
            edge = self._edge_id_from_lane_index(cp)
            if edge and str(edge) in shared_edges:
                try:
                    ck_lane = road_network.get_lane(cp)
                    p = ck_lane.position(0.0, 0.0)
                    return np.asarray([float(p[0]), float(p[1])], dtype=float)
                except Exception:
                    break
        return path[min(1, len(path) - 1)]

    def _resolve_path_conflict_point(
        self,
        ego_vehicle,
        foe_vehicle,
        ego_path: list[np.ndarray],
        foe_path: list[np.ndarray],
    ) -> np.ndarray | None:
        """Conflict point from near-crossing paths, else shared-exit merge.

        Prefer geometric route intersection (the yield conflict at the junction).
        Shared-future-edge merge is only a fallback for parallel merges that
        never literally cross within ``CONFLICT_PATH_TOLERANCE_M``. Preferring
        shared edges first wrongly locked conflict at a far exit when ego and
        aux shared a destination (e.g. roundabout), so ego waited until aux
        cleared the exit instead of the entry crossing.
        """
        geom = self._paths_conflict_point(ego_path, foe_path)
        if geom is not None:
            return geom

        shared = self._shared_future_edges(ego_vehicle, foe_vehicle)
        if shared:
            merge = self._first_point_on_shared_edge(foe_vehicle, shared, foe_path)
            if merge is not None:
                return merge
            merge = self._first_point_on_shared_edge(ego_vehicle, shared, ego_path)
            if merge is not None:
                return merge
        return None

    def _min_distance_path_to_point(
        self,
        path: list[np.ndarray],
        point: np.ndarray,
    ) -> float:
        """Minimum distance from ``point`` to any segment of ``path``."""
        if not path:
            return float("inf")
        pt = np.asarray(point, dtype=float)
        if len(path) == 1:
            return float(np.linalg.norm(path[0] - pt))
        best = float("inf")
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom < 1e-9:
                dist = float(np.linalg.norm(a - pt))
            else:
                t = float(np.dot(pt - a, ab) / denom)
                t = max(0.0, min(1.0, t))
                dist = float(np.linalg.norm(a + t * ab - pt))
            if dist < best:
                best = dist
        return best

    def _has_cleared_conflict_point(
        self,
        vehicle,
        conflict_point: np.ndarray,
        *,
        clearance_m: float | None = None,
        remaining_path: list[np.ndarray] | None = None,
    ) -> bool:
        """True when the recorded conflict is behind / off the remaining route.

        Heading projection alone fails on sharp roundabout curves (forward axis
        is tangential while the sticky X sits laterally). Also treat the point
        as cleared when the remaining route no longer comes near it.
        """
        pos = self._xy(vehicle)
        if pos is None:
            return False
        clearance = float(
            self.CONFLICT_CLEARANCE_M if clearance_m is None else clearance_m
        )
        try:
            heading = float(vehicle.heading_theta)
        except Exception:
            heading = None
        if heading is not None:
            forward = np.asarray([np.cos(heading), np.sin(heading)], dtype=float)
            along = float(np.dot(conflict_point - pos, forward))
            if along < -clearance:
                return True

        path = remaining_path
        if path is None:
            path = self._route_polyline(vehicle)
        # Past the sticky X: remaining polyline starts at the vehicle and no
        # longer skirts the recorded entry conflict (shared-exit near-misses
        # further along must not keep this foe armed).
        near_tol = float(self.CONFLICT_PATH_TOLERANCE_M) + max(0.0, clearance)
        if self._min_distance_path_to_point(path, conflict_point) > near_tol:
            return True
        return False

    def _is_foe_blocking_ego(self, ego_vehicle, foe_vehicle) -> bool:
        """Main-zone arm + sticky path-clearance until the conflict is passed.

        Semantics:
        1. Foe enters the main conflict zone → start tracking if ego/foe routes
           currently intersect (record conflict point).
        2. Keep blocking while that foe is tracked.
        3. Drop the foe once it has cleared the conflict point, or remaining
           routes no longer geometrically intersect — even if still in the zone.
        4. Do not re-arm the same foe until it has left the main zone.
        """
        if self._is_waiting_gated_aux(foe_vehicle):
            return False

        foe_id = getattr(foe_vehicle, "id", None)
        if foe_id is None:
            return False

        in_main = self._is_vehicle_in_main_road_conflict_zone(foe_vehicle)
        if not in_main:
            self._path_released_foes.discard(foe_id)

        if foe_id in self._path_released_foes:
            return False

        ego_path = self._route_polyline(ego_vehicle)
        foe_path = self._route_polyline(foe_vehicle)
        # Arming may use shared-exit merge; release must not — otherwise a
        # common destination keeps the foe sticky after the entry crossing.
        geom_conflict = self._paths_conflict_point(ego_path, foe_path)
        arm_conflict = self._resolve_path_conflict_point(
            ego_vehicle, foe_vehicle, ego_path, foe_path
        )

        sticky = self._path_sticky_foes.get(foe_id)

        # Arm tracking on first entry into the main zone with a path conflict.
        if in_main and sticky is None:
            if arm_conflict is None:
                return False
            self._path_sticky_foes[foe_id] = {"conflict_point": arm_conflict}
            sticky = self._path_sticky_foes[foe_id]

        if sticky is None:
            return False

        # Fill conflict point once (do not chase along a shared exit arm).
        if sticky.get("conflict_point") is None and arm_conflict is not None:
            sticky["conflict_point"] = arm_conflict

        conflict_point = sticky.get("conflict_point")

        def _release() -> bool:
            self._path_sticky_foes.pop(foe_id, None)
            self._path_released_foes.add(foe_id)
            return False

        if conflict_point is None:
            return _release()

        # Foe has driven past the recorded conflict → stop tracking.
        if self._has_cleared_conflict_point(
            foe_vehicle, conflict_point, remaining_path=foe_path
        ):
            return _release()

        # Remaining trajectories no longer intersect (geometry only).
        if geom_conflict is None:
            return _release()

        return True

    def is_vehicle_blocking_yield(self, ego_vehicle, foe_vehicle) -> bool:
        """Public sticky path-conflict check used by expert / overlays."""
        if self._auto_detect and not self._pg_initialized:
            self._identify_main_roads()
        return self._is_foe_blocking_ego(ego_vehicle, foe_vehicle)

    def get_top_down_path_conflict_overlay(self, ego_vehicle) -> dict:
        """Snapshot for GIF debug: ego/foe route polylines + conflict points.

        Uses the same geometry as ``_is_foe_blocking_ego`` (route sample + sticky
        conflict point). Does not change sticky state beyond a normal yield check.
        """
        if self._auto_detect and not self._pg_initialized:
            self._identify_main_roads()

        ego_path = self._route_polyline(ego_vehicle)
        foes: list[dict] = []
        for v in self._get_all_vehicles():
            if getattr(v, "id", None) == getattr(ego_vehicle, "id", None):
                continue
            # Draw gated (held) aux too — GIF debug must show their planned path
            # even before release. Yield / traffic checks still ignore them.
            foe_path = self._route_polyline(v)
            conflict = self._resolve_path_conflict_point(
                ego_vehicle, v, ego_path, foe_path
            )
            sticky = self._path_sticky_foes.get(getattr(v, "id", None)) or {}
            sticky_pt = sticky.get("conflict_point")
            waiting = bool(self._is_waiting_gated_aux(v))
            blocking = (
                False
                if waiting
                else bool(self._is_foe_blocking_ego(ego_vehicle, v))
            )
            # After the blocking call, sticky may have been updated/cleared.
            sticky = self._path_sticky_foes.get(getattr(v, "id", None)) or {}
            sticky_pt = sticky.get("conflict_point")
            draw_pt = sticky_pt if sticky_pt is not None else conflict
            foes.append(
                {
                    "vehicle_id": getattr(v, "id", None),
                    "path": foe_path,
                    # Only show the X while this foe is still yielding-relevant.
                    "conflict_point": draw_pt if blocking else None,
                    "in_main_zone": bool(
                        self._is_vehicle_in_main_road_conflict_zone(v)
                    ),
                    "blocking": blocking,
                    "waiting_gated": waiting,
                    "sticky": bool(sticky_pt is not None),
                }
            )

        zones: list[dict] = []
        if hasattr(self, "get_top_down_aux_conflict_zones"):
            try:
                zones = list(self.get_top_down_aux_conflict_zones() or [])
            except Exception:
                zones = []

        return {
            "ego_path": ego_path,
            "foes": foes,
            "zones": zones,
        }

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

        ego = exclude_vehicle
        if ego is None:
            # Fall back to coarse main-zone only when no ego is provided.
            conflicting = []
            for v in self._get_all_vehicles():
                if self._is_waiting_gated_aux(v):
                    continue
                if self._is_vehicle_in_main_road_conflict_zone(v):
                    conflicting.append(v)
            return len(conflicting) > 0, conflicting

        return self._check_main_road_traffic(ego)

    def _is_waiting_gated_aux(self, vehicle) -> bool:
        """True for gated aux that has not been released yet (still held at spawn).

        Such vehicles must not count as conflicting main-road traffic: otherwise a
        yielding ego stops far from the junction while aux waits for ego to get
        closer — a deadlock.
        """
        try:
            engine = getattr(self, "engine", None)
            if engine is None:
                return False
            policy = engine.get_policy(getattr(vehicle, "id", None))
        except Exception:
            return False
        if policy is None:
            return False
        # GatedAuxiliaryIDMPolicy exposes ``released``; default True = treat as traffic.
        if not hasattr(policy, "ego_distance_to_spawn_lane_end"):
            return False
        return not bool(getattr(policy, "released", True))

    def _check_main_road_traffic(self, ego_vehicle) -> tuple:
        """Conflict if foe is in main zone or sticky until path conflict is cleared."""
        # Ensure main roads are identified (lazy initialization)
        if self._auto_detect and not self._pg_initialized:
            self._identify_main_roads()
        
        if not self.main_road_lanes:
            return False, []

        live_ids = set()
        conflicting = []
        for v in self._get_all_vehicles():
            if v.id == ego_vehicle.id:
                continue
            live_ids.add(v.id)
            if self._is_foe_blocking_ego(ego_vehicle, v):
                conflicting.append(v)

        # Drop sticky / released entries for despawned vehicles.
        for foe_id in list(self._path_sticky_foes.keys()):
            if foe_id not in live_ids:
                self._path_sticky_foes.pop(foe_id, None)
        for foe_id in list(self._path_released_foes):
            if foe_id not in live_ids:
                self._path_released_foes.discard(foe_id)

        return len(conflicting) > 0, conflicting

    def _vehicle_half_length(self, vehicle) -> float:
        return 0.5 * float(
            getattr(vehicle, "LENGTH", self.EGO_ZONE_END_CENTER_INSET * 2)
        )

    def _obligation_zone_end(self, vehicle) -> float:
        """Max vehicle-center longitudinal still fully inside the yield zone."""
        half_len = self._vehicle_half_length(vehicle)
        return max(self.zone_start + half_len, float(self.zone_end) - half_len)

    def _is_vehicle_in_zone(self, vehicle) -> bool:
        """True when the whole vehicle footprint is inside the yield zone."""
        vehicle_lane = getattr(vehicle, "lane", None)
        if vehicle_lane is None or not same_road_check(
            getattr(vehicle_lane, "index", None),
            getattr(self.lane, "index", None),
        ):
            return False

        veh_long = self.lane.local_coordinates(vehicle.position)[0]
        half_len = self._vehicle_half_length(vehicle)
        front = float(veh_long) + half_len
        rear = float(veh_long) - half_len
        return rear >= self.zone_start and front <= self.zone_end

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
        # Invisible metrics tracker only: no 3D model and no top-down icon.
        # A 2.1 icon here duplicated MainRoadSign on the ego approach whenever
        # ego spawned off the rightmost lane (edge plate + mid-lane tracker).
        kwargs.setdefault("show_model", False)
        kwargs["icon_path"] = None
        super().__init__(
            lane,
            intersection_name=intersection_name,
            main_road_lanes=right_road_lanes,
            auto_detect_main_roads=False,
            **kwargs,
        )
        self.priority_type = "right_hand_yield"
        self.icon_path = None

    def get_rule_description(self) -> str:
        return (
            "Right-hand rule at equal-priority intersection — "
            "must not leave approach zone while traffic is on the right"
        )


class StopSign(YieldSign):
    """Stop sign (2.5) — yield to main-road traffic in zone + mandatory stop at line."""

    STOP_SPEED_THRESHOLD_MPS = 0.5
    STOP_LINE_PAST_MARGIN = 0.3

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
        from pdd_bench.signs.base_traffic_sign import same_road_check

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

        if veh_long < self.stop_line_position + self.STOP_LINE_PAST_MARGIN:
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
    """Sign 2.3.3 – Secondary road on the left (T crossroad variant)."""

    def __init__(self, lane, intersection_name: str = None, **kwargs):
        super().__init__(lane, icon_path="2.3.3.png", **kwargs)
        self.intersection_name = intersection_name
        self.is_priority_sign = True
        self.priority_type = "secondary_road_left"

    def _is_violating(self, vehicle) -> bool:
        return False

    def get_rule_description(self) -> str:
        return "Secondary road on the left (2.3.3) - you have priority"

    @property
    def top_down_color(self):
        return [255, 204, 0]

    @property
    def top_down_color_name(self):
        return "yellow"


class RoundaboutSign(BaseTrafficSign):
    """Sign 4.3 — roundabout ahead (informational plate on the approach)."""

    def __init__(self, lane, intersection_name: str = None, **kwargs):
        super().__init__(lane, icon_path="4.3.png", **kwargs)
        self.intersection_name = intersection_name
        self.is_priority_sign = True
        self.priority_type = "roundabout_ahead"

    def _is_violating(self, vehicle) -> bool:
        return False

    def get_rule_description(self) -> str:
        return "Roundabout ahead (4.3) — yield to traffic on the circle"

    @property
    def top_down_color(self):
        return [255, 204, 0]

    @property
    def top_down_color_name(self):
        return "yellow"


class RoundaboutYieldSign(YieldSign):
    """Invisible yield tracker for 4.3 — ego on spoke yields to ring traffic."""

    ENTRY_CONFLICT_BEFORE_M = 20.0
    ENTRY_CONFLICT_AFTER_M = 5.0

    def __init__(
        self,
        lane,
        intersection_name: str = None,
        ring_road_lanes: list = None,
        entry_incoming_lanes: list = None,
        entry_junction_xy: tuple[float, float] | list[float] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("show_model", False)
        kwargs["icon_path"] = None
        incoming = list(entry_incoming_lanes or [])
        conflict_lanes = incoming
        if not conflict_lanes:
            conflict_lanes = list(ring_road_lanes or [])
        super().__init__(
            lane,
            intersection_name=intersection_name,
            main_road_lanes=conflict_lanes,
            auto_detect_main_roads=False,
            **kwargs,
        )
        self.priority_type = "roundabout_yield"
        self.icon_path = None
        if entry_junction_xy is not None:
            self._entry_junction_xy = np.array(
                [float(entry_junction_xy[0]), float(entry_junction_xy[1])],
                dtype=np.float64,
            )
        else:
            self._entry_junction_xy = None

    def _junction_at_lane_end(self, lane) -> bool:
        """True when the entry junction is at ``lane.length`` (SUMO to-node end)."""
        if self._entry_junction_xy is None:
            return True
        try:
            p0 = np.array(lane.position(0.0, 0.0), dtype=np.float64)
            p1 = np.array(lane.position(float(lane.length), 0.0), dtype=np.float64)
        except Exception:
            return True
        return float(np.linalg.norm(p1 - self._entry_junction_xy)) <= float(
            np.linalg.norm(p0 - self._entry_junction_xy)
        )

    def _conflict_longitudinal_range(self, lane) -> tuple[float, float]:
        """Ring tail upstream of ego entry, plus a short downstream tail past lane end."""
        before_m = self.ENTRY_CONFLICT_BEFORE_M
        after_m = self.ENTRY_CONFLICT_AFTER_M
        return (
            max(0.0, float(lane.length) - before_m),
            float(lane.length) + after_m,
        )

    def _is_vehicle_in_main_road_conflict_zone(self, vehicle) -> bool:
        """True when ring traffic is in the monitored conflict arc at ego's entry."""
        if not self.main_road_lanes:
            return False

        try:
            vehicle_pos = vehicle.position
            vehicle_heading = vehicle.heading_theta
            vehicle_lane = getattr(vehicle, "lane", None)
            vehicle_lane_idx = getattr(vehicle_lane, "index", None)
            if vehicle_lane_idx is None:
                return False
            v_segment = (vehicle_lane_idx[0], vehicle_lane_idx[1])
        except Exception:
            return False

        main_segments = set()
        for ln in self.main_road_lanes:
            ln_idx = getattr(ln, "index", None)
            if ln_idx and len(ln_idx) >= 2:
                main_segments.add((ln_idx[0], ln_idx[1]))

        if v_segment in main_segments and vehicle_lane is not None:
            try:
                long_pos, lat_pos = vehicle_lane.local_coordinates(vehicle_pos)
                zone_start, zone_end = self._conflict_longitudinal_range(vehicle_lane)
                if zone_start <= long_pos <= zone_end and abs(lat_pos) <= vehicle_lane.width * 1.5:
                    return True
            except Exception:
                pass

        for lane in self.main_road_lanes:
            try:
                lane_idx = getattr(lane, "index", None)
                if lane_idx is None or len(lane_idx) < 2:
                    continue
                lane_segment = (lane_idx[0], lane_idx[1])
                long_pos, lat_pos = lane.local_coordinates(vehicle_pos)
                zone_start, zone_end = self._conflict_longitudinal_range(lane)

                if zone_start <= long_pos <= zone_end:
                    if abs(lat_pos) <= lane.width * 1.5:
                        if v_segment == lane_segment:
                            return True
                        if v_segment not in main_segments:
                            lane_heading = lane.heading_theta_at(
                                min(max(long_pos, 0.0), lane.length)
                            )
                            heading_diff = abs(vehicle_heading - lane_heading)
                            heading_diff = min(heading_diff, 2 * np.pi - heading_diff)
                            if heading_diff < np.pi / 2:
                                return True
            except Exception:
                continue
        return False

    def get_top_down_aux_conflict_zones(self) -> list[dict]:
        """Lane segments where auxiliary ring traffic triggers a 4.3 violation."""
        zones: list[dict] = []
        for lane in self.main_road_lanes:
            try:
                long_start, long_end = self._conflict_longitudinal_range(lane)
                zones.append(
                    {
                        "lane": lane,
                        "long_start": float(long_start),
                        "long_end": float(long_end),
                        "kind": "incoming",
                    }
                )
            except Exception:
                continue
        return zones

    def get_rule_description(self) -> str:
        return (
            "Roundabout (4.3) — must not leave the approach zone "
            "while traffic is present on the ring within "
            f"{self.ENTRY_CONFLICT_BEFORE_M:.0f} m upstream of the ego entry junction "
            f"(plus {self.ENTRY_CONFLICT_AFTER_M:.0f} m past the lane end)"
        )


class SecondaryRoadRightSign(BaseTrafficSign):
    """Sign 2.3.2 – Secondary road on the right (T crossroad variant)."""

    def __init__(self, lane, intersection_name: str = None, **kwargs):
        super().__init__(lane, icon_path="2.3.2.png", **kwargs)
        self.intersection_name = intersection_name
        self.is_priority_sign = True
        self.priority_type = "secondary_road_right"

    def _is_violating(self, vehicle) -> bool:
        return False

    def get_rule_description(self) -> str:
        return "Secondary road on the right (2.3.2) - you have priority"

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
    "RoundaboutSign",
    "RoundaboutYieldSign",
    "StopSign",
    "SecondaryRoadSign",
    "SecondaryRoadLeftSign",
    "SecondaryRoadRightSign",
]
