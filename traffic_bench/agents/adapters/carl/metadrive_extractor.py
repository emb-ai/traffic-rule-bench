"""
Extracts road geometry, vehicles, pedestrians, and other data from MetaDrive
 for CaRL BEV rendering.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import traceback

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
from metadrive.constants import Decoration, MetaDriveType
from metadrive.scenario.scenario_description import ScenarioDescription


class MetaDriveExtractor:
    """
    Extracts data from MetaDrive environment for CaRL BEV rendering.
    """
    
    def __init__(
        self,
        engine,
        fov_range: float = 80.0,
        lane_speed_limit_unit: str = "kmh",
        min_speed_limit_kmh: float = 120.0,
    ):
        """
        Initialize extractor.
        
        Args:
            engine: MetaDrive engine instance
            fov_range: Maximum distance to extract objects (meters)
            lane_speed_limit_unit: Unit for `lane.speed_limit` when falling back to lane limits.
                Use "kmh" if lane.speed_limit is in km/h, or "mps" if it's already in m/s.
        """
        self.engine = engine
        self.fov_range = fov_range
        unit = str(lane_speed_limit_unit).strip().lower()
        if unit not in {"kmh", "mps"}:
            raise ValueError('lane_speed_limit_unit must be "kmh" or "mps"')
        self._lane_speed_limit_unit = unit
        self._min_speed_limit_kmh = float(min_speed_limit_kmh)

        # Cache static lane geometry for performance on large (multi-block) maps.
        self._cached_map_id = None
        self._cached_lane_objs: List[AbstractLane] = []
        self._cached_lane_indices: List[Any] = []
        self._cached_lane_centers = np.zeros((0, 2), dtype=np.float64)
        self._lane_by_index: Dict[Any, AbstractLane] = {}
        self._lane_center_by_index: Dict[Any, np.ndarray] = {}
        self._lane_polygon_cache: Dict[Any, np.ndarray] = {}
        self._lane_boundaries_cache: Dict[Any, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
        self._lane_centerline_cache: Dict[Any, np.ndarray] = {}

        # Cache explicit speed limits from traffic-bench SpeedLimitSign objects (edge -> limit_mps).
        self._explicit_speed_limit_mps_by_lane: Dict[Any, float] = {}
        self._explicit_speed_limit_cache_key: Optional[int] = None

    def _get_current_map(self):
        current_map = getattr(self.engine, "current_map", None)
        if current_map is None and hasattr(self.engine, "map_manager"):
            current_map = getattr(self.engine.map_manager, "current_map", None)
        return current_map

    def _ensure_lane_cache(self) -> None:
        current_map = self._get_current_map()
        map_id = id(current_map) if current_map is not None else None
        if map_id == self._cached_map_id:
            return

        self._cached_map_id = map_id
        self._cached_lane_objs = []
        self._cached_lane_indices = []
        self._cached_lane_centers = np.zeros((0, 2), dtype=np.float64)
        self._lane_by_index = {}
        self._lane_center_by_index = {}
        self._lane_polygon_cache = {}
        self._lane_boundaries_cache = {}
        self._lane_centerline_cache = {}
        self._explicit_speed_limit_mps_by_lane = {}
        self._explicit_speed_limit_cache_key = None

        if current_map is None:
            return
        road_network = getattr(current_map, "road_network", None)
        if not isinstance(road_network, NodeRoadNetwork):
            return

        lane_centers: List[np.ndarray] = []
        for _from in road_network.graph.keys():
            if _from == Decoration.start:
                continue
            for _to in road_network.graph[_from].keys():
                for lane in road_network.graph[_from][_to]:
                    lane_index = getattr(lane, "index", None)
                    if lane_index is None:
                        continue
                    self._cached_lane_objs.append(lane)
                    self._cached_lane_indices.append(lane_index)
                    self._lane_by_index[lane_index] = lane
                    center = self._get_lane_center(lane)
                    if center is None:
                        center = np.array([np.nan, np.nan], dtype=np.float64)
                    else:
                        center = np.array(center[:2], dtype=np.float64)
                    self._lane_center_by_index[lane_index] = center
                    lane_centers.append(center)

        if lane_centers:
            self._cached_lane_centers = np.stack(lane_centers, axis=0).astype(np.float64, copy=False)

    def _lane_ids_in_fov(self, ego_pos: np.ndarray) -> List[int]:
        self._ensure_lane_cache()
        if self._cached_lane_centers.size == 0:
            return []
        ego_xy = np.array(ego_pos[:2], dtype=np.float64)
        d = self._cached_lane_centers - ego_xy
        dist2 = np.einsum("ij,ij->i", d, d)
        mask = np.isfinite(dist2) & (dist2 < float(self.fov_range) ** 2)
        return list(np.nonzero(mask)[0])

    def _get_lane_polygon_cached(self, lane: AbstractLane) -> Optional[np.ndarray]:
        lane_index = getattr(lane, "index", None)
        if lane_index is None:
            return self._lane_to_polygon(lane)
        poly = self._lane_polygon_cache.get(lane_index)
        if poly is None:
            poly = self._lane_to_polygon(lane)
            if poly is not None:
                self._lane_polygon_cache[lane_index] = poly
        return poly

    def _get_lane_boundaries_cached(self, lane: AbstractLane) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        lane_index = getattr(lane, "index", None)
        if lane_index is None:
            return self._get_lane_boundaries(lane)
        if lane_index not in self._lane_boundaries_cache:
            self._lane_boundaries_cache[lane_index] = self._get_lane_boundaries(lane)
        return self._lane_boundaries_cache[lane_index]

    def _get_lane_centerline_cached(self, lane: AbstractLane) -> Optional[np.ndarray]:
        lane_index = getattr(lane, "index", None)
        if lane_index is None:
            return self._get_lane_centerline(lane)
        centerline = self._lane_centerline_cache.get(lane_index)
        if centerline is None:
            centerline = self._get_lane_centerline(lane)
            if centerline is not None:
                self._lane_centerline_cache[lane_index] = centerline
        return centerline

    def _ensure_explicit_speed_limit_cache(self) -> None:
        try:
            if not hasattr(self.engine, "traffic_sign_manager"):
                self._explicit_speed_limit_mps_by_lane = {}
                self._explicit_speed_limit_cache_key = 0
                return
            mgr = self.engine.traffic_sign_manager
            signs = getattr(mgr, "signs", None)
            if not signs:
                self._explicit_speed_limit_mps_by_lane = {}
                self._explicit_speed_limit_cache_key = 0
                return
            key = int(len(signs))
            if self._explicit_speed_limit_cache_key == key:
                return

            SpeedLimitSignCls = None
            try:
                from traffic_bench.signs.speed.limit import SpeedLimitSign as SpeedLimitSignCls  # type: ignore
            except Exception:
                SpeedLimitSignCls = None

            mapping: Dict[Any, float] = {}
            for sign in list(signs):
                if sign is None or getattr(sign, "_is_destroyed", False):
                    continue
                if SpeedLimitSignCls is not None:
                    if not isinstance(sign, SpeedLimitSignCls):
                        continue
                else:
                    if "SpeedLimit" not in type(sign).__name__:
                        continue

                lane = getattr(sign, "lane", None)
                lane_index = getattr(lane, "index", None) if lane is not None else None
                speed_kmh = getattr(sign, "speed_limit", None)
                if lane_index is None or speed_kmh is None:
                    continue
                try:
                    edge_key = (lane_index[0], lane_index[1])
                except Exception:
                    continue
                mapping[edge_key] = float(speed_kmh) / 3.6

            self._explicit_speed_limit_mps_by_lane = mapping
            self._explicit_speed_limit_cache_key = key
        except Exception:
            # Don't fail observation building if sign parsing breaks.
            return
    
    def get_ego_state(self, vehicle: BaseVehicle) -> Dict[str, Any]:
        """
        Extract ego vehicle state.
        
        Returns:
            Dict with position, heading, speed, steering, acceleration, etc.
        """
        return {
            "position": np.array(vehicle.position),
            "heading": vehicle.heading_theta,
            "speed": vehicle.speed_km_h / 3.6,  # Convert to m/s
            "velocity": np.array(vehicle.velocity) if hasattr(vehicle, 'velocity') else np.array([vehicle.speed_km_h / 3.6, 0]),
            "steering": vehicle.steering if hasattr(vehicle, 'steering') else 0.0,
            "length": vehicle.LENGTH if hasattr(vehicle, 'LENGTH') else 4.5,
            "width": vehicle.WIDTH if hasattr(vehicle, 'WIDTH') else 2.0,
        }
    
    def get_road_polygons(self, ego_pos: np.ndarray) -> List[np.ndarray]:
        """
        Extract drivable area polygons near ego vehicle.
        
        Returns:
            List of (N, 2) polygon arrays in global coordinates
        """
        polygons = []
        ego_xy = np.array(ego_pos[:2], dtype=np.float64)
        
        try:
            road_network = self.engine.current_map.road_network
            
            if isinstance(road_network, NodeRoadNetwork):
                # PG/Node-based road network
                for _from in road_network.graph.keys():
                    if _from == Decoration.start:
                        continue
                    for _to in road_network.graph[_from].keys():
                        for lane in road_network.graph[_from][_to]:
                            poly = self._lane_to_polygon(lane)
                            if poly is None or len(poly) < 3:
                                continue
                            dist = np.min(np.linalg.norm(poly[:, :2] - ego_xy, axis=1))
                            if dist < self.fov_range:
                                polygons.append(poly)
            
            elif isinstance(road_network, EdgeRoadNetwork):
                # Edge-based road network
                map_data = None
                if hasattr(self.engine, 'map_manager') and hasattr(self.engine.map_manager, 'current_map'):
                    current_map = self.engine.map_manager.current_map
                    if hasattr(current_map, 'blocks') and len(current_map.blocks) > 0:
                        if hasattr(current_map.blocks[-1], 'map_data'):
                            map_data = current_map.blocks[-1].map_data
                
                if map_data:
                    for lane_id, lane_info in map_data.items():
                        try:
                            if not isinstance(lane_info, dict):
                                continue
                            if not MetaDriveType.is_lane(lane_info.get("type")):
                                continue
                            if "polygon" not in lane_info:
                                continue

                            poly = np.array(lane_info["polygon"])
                            if poly.ndim != 2 or poly.shape[0] < 3:
                                continue
                            if poly.shape[1] > 2:
                                poly = poly[:, :2]

                            dist = np.min(np.linalg.norm(poly - ego_xy, axis=1))
                            if dist < self.fov_range:
                                polygons.append(poly)
                        except Exception as lane_err:
                            print(f"[MetaDriveExtractor] get_road_polygons skip lane {lane_id}: {lane_err!r}")
                            continue

        except Exception as e:
            print(f"[MetaDriveExtractor] get_road_polygons failed: {e!r}")
            traceback.print_exc()
            raise
        
        return polygons
    
    # ---- range helpers ------------------------------------------------------
    # Everything map-side is kept when ANY of its points lies within fov_range
    # of the ego. The earlier checks used one representative point (the lane
    # midpoint, the polyline mean, the first vertex), which on the long SUMO
    # edges of the segment scenes dropped the very lane the ego was driving on:
    # with the ego at the start of a 640 m edge the midpoint is 320 m away, so
    # the route channel came out empty and the lane-line channel flickered
    # between roads. Measured on seg_1158657284: route 0 px in every frame,
    # lane lines 1652 px at spawn and 121 px forty steps later.
    @staticmethod
    def _min_dist(points, ego_pos) -> float:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or len(pts) == 0:
            return float("inf")
        ego = np.asarray(ego_pos, dtype=np.float64)[:2]
        return float(np.min(np.linalg.norm(pts[:, :2] - ego, axis=1)))

    @staticmethod
    def _polyline2d(raw) -> Optional[np.ndarray]:
        if raw is None:
            return None
        pl = np.asarray(raw, dtype=np.float64)
        if pl.ndim != 2 or pl.shape[0] < 2:
            return None
        return pl[:, :2]

    def _map_data(self):
        mm = getattr(self.engine, "map_manager", None)
        current_map = getattr(mm, "current_map", None)
        blocks = getattr(current_map, "blocks", None)
        if blocks:
            return getattr(blocks[-1], "map_data", None)
        return None

    @staticmethod
    def _is_internal_lane_id(lane_id) -> bool:
        return isinstance(lane_id, str) and lane_id.startswith("lane_:")

    def _append_lane_polygon_if_near(self, polygons: List[np.ndarray], lane, ego_pos) -> None:
        poly = self._get_lane_polygon_cached(lane)
        if poly is None or len(poly) < 3:
            return
        if self._min_dist(poly, ego_pos) < self.fov_range:
            polygons.append(poly)

    def get_route_polygons(self, vehicle: BaseVehicle, ego_pos: np.ndarray) -> List[np.ndarray]:
        """Polygons of the lanes on the ego route (every peer lane of each
        checkpoint edge), for the route channel -- the one channel that tells
        the policy which side of the roadway is its own.
        """
        polygons: List[np.ndarray] = []
        try:
            navigation = getattr(vehicle, "navigation", None)
            if navigation is None:
                return polygons
            road_network = self.engine.current_map.road_network
            checkpoints = list(getattr(navigation, "checkpoints", None) or [])

            if isinstance(road_network, NodeRoadNetwork):
                for i, checkpoint in enumerate(checkpoints[:-1]):
                    next_checkpoint = checkpoints[i + 1]
                    if checkpoint in road_network.graph and next_checkpoint in road_network.graph.get(checkpoint, {}):
                        for lane in road_network.graph[checkpoint][next_checkpoint]:
                            self._append_lane_polygon_if_near(polygons, lane, ego_pos)

            elif isinstance(road_network, EdgeRoadNetwork):
                seen = set()
                for lane_index in checkpoints:
                    try:
                        peer_lanes = road_network.get_peer_lanes_from_index(lane_index)
                    except KeyError:
                        continue
                    for lane in peer_lanes:
                        if id(lane) in seen:
                            continue
                        seen.add(id(lane))
                        self._append_lane_polygon_if_near(polygons, lane, ego_pos)
        except Exception as e:
            print(f"[MetaDriveExtractor] get_route_polygons failed: {e!r}")

        return polygons

    def get_lane_boundaries(self, ego_pos: np.ndarray) -> List[np.ndarray]:
        """Lane boundary polylines for the lane-line channel.

        On SUMO maps the map data carries the painted lines (lane dividers,
        the yellow axial line between opposing flows): they ARE boundaries and
        are drawn where they lie. The previous code took each of them for a
        lane centreline and drew two lines 1.75 m to either side -- through
        the middle of the neighbouring lanes, with nothing on the paint itself.
        Lane edges are added from every lane's centreline +- half its width,
        which is what nuPlan's lane-boundary channel contains; that also puts a
        line on the outer road edge. SUMO's internal junction lanes are skipped
        (their edges criss-cross the intersection).
        """
        boundaries: List[np.ndarray] = []

        try:
            road_network = self.engine.current_map.road_network

            if isinstance(road_network, NodeRoadNetwork):
                for i in self._lane_ids_in_fov(ego_pos):
                    lane = self._cached_lane_objs[i]
                    left, right = self._get_lane_boundaries_cached(lane)
                    if left is not None and len(left) >= 2:
                        boundaries.append(left)
                    if right is not None and len(right) >= 2:
                        boundaries.append(right)

            elif isinstance(road_network, EdgeRoadNetwork):
                map_data = self._map_data()
                if map_data:
                    for lane_id, lane_info in map_data.items():
                        if not isinstance(lane_info, dict) or self._is_internal_lane_id(lane_id):
                            continue
                        kind = lane_info.get("type")
                        polyline = self._polyline2d(lane_info.get("polyline"))
                        if polyline is None or self._min_dist(polyline, ego_pos) >= self.fov_range:
                            continue
                        if MetaDriveType.is_road_line(kind):
                            boundaries.append(polyline)
                        elif MetaDriveType.is_lane(kind):
                            width = float(lane_info.get("width", 3.5) or 3.5)
                            left, right = self._create_boundaries_from_centerline(polyline, width)
                            if left is not None and len(left) >= 2:
                                boundaries.append(left)
                            if right is not None and len(right) >= 2:
                                boundaries.append(right)
        except Exception as e:
            print(f"[MetaDriveExtractor] get_lane_boundaries failed: {e!r}")

        return boundaries

    def get_vehicles(self, ego_vehicle: BaseVehicle, ego_pos: np.ndarray) -> List[Dict]:
        """
        Extract other vehicles in scene.
        """
        vehicles = []
        
        try:
            for obj in self.engine.get_objects(
                lambda o: isinstance(o, BaseVehicle) and o is not ego_vehicle
            ).values():
                pos = np.array(obj.position)
                dist = np.linalg.norm(pos - ego_pos)
                if dist < self.fov_range:
                    vehicles.append({
                        "position": pos,
                        "heading": obj.heading_theta if hasattr(obj, 'heading_theta') else 0.0,
                        "length": obj.LENGTH if hasattr(obj, 'LENGTH') else 4.5,
                        "width": obj.WIDTH if hasattr(obj, 'WIDTH') else 2.0,
                        "speed": obj.speed_km_h / 3.6 if hasattr(obj, 'speed_km_h') else 0.0,
                    })
        except Exception as e:
            pass
        
        return vehicles
    
    def get_pedestrians(self, ego_pos: np.ndarray) -> List[Dict]:
        """
        Extract pedestrians in scene.
        """
        pedestrians = []
        
        try:
            for obj in self.engine.get_objects(
                lambda o: isinstance(o, BaseTrafficParticipant) and not isinstance(o, BaseVehicle)
            ).values():
                pos = np.array(obj.position)
                dist = np.linalg.norm(pos - ego_pos)
                if dist < self.fov_range:
                    pedestrians.append({
                        "position": pos,
                        "heading": obj.heading_theta if hasattr(obj, 'heading_theta') else 0.0,
                        "length": 0.8,
                        "width": 0.8,
                        "speed": obj.speed_km_h / 3.6 if hasattr(obj, 'speed_km_h') else 0.0,
                    })
        except Exception as e:
            pass
        
        return pedestrians
    
    def get_traffic_lights(self, ego_pos: np.ndarray) -> List[Dict]:
        """
        Extract traffic light states.
        
        Returns:
            List of traffic light dicts with position and state
        """
        traffic_lights = []
        
        try:
            #  traffic lights from traffic manager
            if hasattr(self.engine, 'traffic_manager') and self.engine.traffic_manager is not None:
                tm = self.engine.traffic_manager
                
                tl_list = None
                if hasattr(tm, 'get_traffic_lights'):
                    tl_list = tm.get_traffic_lights()
                
                if tl_list is not None:
                    for tl in tl_list:
                        pos = None
                        if hasattr(tl, 'position'):
                            pos = np.array(tl.position)
                        
                        if pos is not None:
                            dist = np.linalg.norm(pos - ego_pos)
                            if dist < self.fov_range:
                                state = "unknown"
                                if hasattr(tl, 'status'):
                                    status = tl.status
                                    if status == "green" or status == 0:
                                        state = "green"
                                    elif status == "yellow" or status == 1:
                                        state = "yellow"
                                    elif status == "red" or status == 2:
                                        state = "red"
                                traffic_lights.append({
                                    "position": pos,
                                    "state": state,
                                })
        except Exception as e:
            pass
        
        return traffic_lights
    
    def get_stop_signs(self, ego_pos: np.ndarray) -> List[Dict]:
        """
        Extract stop sign locations from TrafficSignManager (traffic-bench).
        """
        stop_signs = []
        
        try:
            # Check if TrafficSignManager exists in the engine
            if not hasattr(self.engine, 'traffic_sign_manager'):
                return stop_signs
            
            traffic_sign_manager = self.engine.traffic_sign_manager
            
            if traffic_sign_manager is None or not hasattr(traffic_sign_manager, 'signs'):
                return stop_signs
            
            StopSignCls = None
            try:
                from traffic_bench.signs.junction.yield_sign import StopSign as StopSignCls  # type: ignore
            except Exception:
                StopSignCls = None

            # Extract stop signs from manager
            for sign in traffic_sign_manager.signs:
                if sign is None or getattr(sign, "_is_destroyed", False):
                    continue
                if StopSignCls is not None:
                    if not isinstance(sign, StopSignCls):
                        continue
                else:
                    if type(sign).__name__ != "StopSign":
                        continue
                
                # Check if this is a stop sign
                is_stop_sign = False
                if hasattr(sign, '__class__'):
                    class_name = sign.__class__.__name__
                    if "StopSign" in class_name:
                        is_stop_sign = True
                
                # For EdgeRoadNetwork, check by type if available
                if not is_stop_sign and hasattr(sign, 'type') and sign.type == "STOP":
                    is_stop_sign = True
                
                if not is_stop_sign:
                    continue
                
                # Get position
                sign_pos = np.array(sign.position[:2], dtype=np.float64)
                
                dist = np.linalg.norm(sign_pos - ego_pos)
                if dist > self.fov_range:
                    continue
                
                # Get dimensions and heading
                sign_length = sign.top_down_length if hasattr(sign, 'top_down_length') else 4.0
                sign_width = sign.top_down_width if hasattr(sign, 'top_down_width') else 4.0
                
                sign_heading = sign.heading_theta if hasattr(sign, 'heading_theta') else 0.0
                
                # Create polygon around the sign
                half_length = sign_length / 2.0
                half_width = sign_width / 2.0
                
                corners_local = np.array([
                    [-half_length, -half_width],  
                    [half_length, -half_width],    
                    [half_length, half_width],     
                    [-half_length, half_width],    
                ], dtype=np.float64)
                
                cos_h = np.cos(sign_heading)
                sin_h = np.sin(sign_heading)
                R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
                corners_rotated = corners_local @ R.T
                
                polygon = corners_rotated + sign_pos
                
                stop_signs.append({
                    "polygon": polygon,
                    "type": "stop_sign"
                })
        
        except Exception as e:
            pass
        
        return stop_signs
    
    def get_speed_limits(self, ego_pos: np.ndarray) -> List[Dict]:
        """
        Extract speed limit information along lanes
        """
        self._ensure_lane_cache()
        self._ensure_explicit_speed_limit_cache()

        speed_limits: List[Dict] = []
        
        try:
            road_network = self.engine.current_map.road_network
            
            if isinstance(road_network, NodeRoadNetwork):
                for _from in road_network.graph.keys():
                    if _from == Decoration.start:
                        continue
                    for _to in road_network.graph[_from].keys():
                        for lane in road_network.graph[_from][_to]:
                            lane_center = self._get_lane_center(lane)
                            if lane_center is not None:
                                dist = np.linalg.norm(lane_center - ego_pos)
                                if dist < self.fov_range:
                                    centerline = self._get_lane_centerline(lane)
                                    if centerline is not None and len(centerline) >= 2:
                                        speed_limit = lane.speed_limit if hasattr(lane, 'speed_limit') else 30.0
                                        speed_limits.append({
                                            "centerline": centerline,
                                            "limit_mps": speed_limit / 3.6 if speed_limit > 10 else speed_limit,
                                        })
            
            elif isinstance(road_network, EdgeRoadNetwork):
                # For EdgeRoadNetwork, extract speed limits from map data
                map_data = None
                if hasattr(self.engine, 'map_manager') and hasattr(self.engine.map_manager, 'current_map'):
                    current_map = self.engine.map_manager.current_map
                    if hasattr(current_map, 'blocks') and len(current_map.blocks) > 0:
                        if hasattr(current_map.blocks[-1], 'map_data'):
                            map_data = current_map.blocks[-1].map_data
                
                if map_data:
                    for lane_id, lane_info in map_data.items():
                        # Check if this is a lane with speed limit
                        if MetaDriveType.is_lane(lane_info.get("type")) and "speed" in lane_info:
                            if "polyline" in lane_info:
                                polyline = np.array(lane_info["polyline"])
                                if polyline.shape[1] > 2:
                                    polyline = polyline[:, :2]
                                
                                # Any point of the lane within range: the
                                # first-vertex check dropped the ego's own lane
                                # once it was 100 m past the lane start.
                                if len(polyline) > 0:
                                    dist = self._min_dist(polyline, ego_pos)
                                    if dist < self.fov_range:
                                        # Calculate approximate speed limit
                                        speed_limit_kmh = lane_info["speed"] * 3.6
                                        speed_limit_mps = lane_info["speed"]
                                        
                                        speed_limits.append({
                                            "centerline": polyline,
                                            "limit_mps": speed_limit_mps
                                        })
        except Exception as e:
            pass

        return speed_limits
    
    def get_static_objects(self, ego_pos: np.ndarray) -> List[Dict]:
        """Cones, barriers and other static traffic objects near the ego, for
        the pedestrians+static channel (where CaRL's nuPlan renderer puts them).

        Traffic signs are TrafficObjects too, but nuPlan never draws sign posts
        as obstacles and this benchmark stands its plates on the lane
        centreline, so they are left out explicitly.
        """
        objects: List[Dict] = []
        try:
            from metadrive.component.static_object.traffic_object import TrafficObject
            try:
                from traffic_bench.signs.base import BaseTrafficSign
            except Exception:  # pragma: no cover - signs package optional
                BaseTrafficSign = ()
            ego = np.asarray(ego_pos, dtype=np.float64)[:2]
            for obj in self.engine.get_objects(
                lambda o: isinstance(o, TrafficObject) and not isinstance(o, BaseTrafficSign)
            ).values():
                pos = np.asarray(obj.position, dtype=np.float64)[:2]
                if np.linalg.norm(pos - ego) >= self.fov_range:
                    continue
                length = getattr(obj, "top_down_length", None) or getattr(obj, "LENGTH", None) or 1.0
                width = getattr(obj, "top_down_width", None) or getattr(obj, "WIDTH", None) or 1.0
                objects.append({
                    "position": pos,
                    "heading": float(getattr(obj, "heading_theta", 0.0) or 0.0),
                    "length": float(length),
                    "width": float(width),
                })
        except Exception as e:
            print(f"[MetaDriveExtractor] get_static_objects failed: {e!r}")
        return objects

    def extract_all(self, vehicle: BaseVehicle) -> Dict[str, Any]:
        ego_state = self.get_ego_state(vehicle)
        ego_pos = ego_state["position"]
        
        return {
            "ego_state": ego_state,
            "road_polygons": self.get_road_polygons(ego_pos),
            "route_polygons": self.get_route_polygons(vehicle, ego_pos),
            "lane_boundaries": self.get_lane_boundaries(ego_pos),
            "traffic_lights": self.get_traffic_lights(ego_pos),
            "stop_signs": self.get_stop_signs(ego_pos),
            "speed_limits": self.get_speed_limits(ego_pos),
            "vehicles": self.get_vehicles(vehicle, ego_pos),
            "pedestrians": self.get_pedestrians(ego_pos),
            "static_objects": self.get_static_objects(ego_pos),
        }
    def _get_lane_center(self, lane: AbstractLane) -> Optional[np.ndarray]:
        """Get approximate center of lane."""
        try:
            if hasattr(lane, 'position'):
                length = lane.length if hasattr(lane, 'length') else 10.0
                return np.array(lane.position(length / 2, 0))[:2]  # Ensure 2D
            return None
        except:
            return None
    
    def _get_lane_centerline(self, lane: AbstractLane) -> Optional[np.ndarray]:
        """Get lane centerline as polyline."""
        try:
            if hasattr(lane, 'position'):
                length = lane.length if hasattr(lane, 'length') else 10.0
                num_points = max(2, int(length / 2))  # Point every 2 meters
                s_values = np.linspace(0, length, num_points)
                points = []
                for s in s_values:
                    pos = lane.position(s, 0)
                    points.append(pos[:2])
                return np.array(points)
            return None
        except:
            return None

    def _get_lane_centerline_segment(self, lane: AbstractLane, start_s: float, end_s: float) -> Optional[np.ndarray]:
        """Get lane centerline polyline on [start_s, end_s]."""
        try:
            if not hasattr(lane, "position"):
                return None
            lane_len = float(getattr(lane, "length", 0.0))
            s0 = float(np.clip(start_s, 0.0, lane_len))
            s1 = float(np.clip(end_s, 0.0, lane_len))
            if s1 < s0:
                s0, s1 = s1, s0
            seg_len = max(0.0, s1 - s0)
            if seg_len < 0.5:
                s1 = float(np.clip(s0 + 1.0, 0.0, lane_len))
                seg_len = max(0.0, s1 - s0)
            num_points = max(2, int(seg_len / 2) + 1)  # point every 2m
            s_values = np.linspace(s0, s1, num_points)
            points = []
            for s in s_values:
                pos = lane.position(float(s), 0)
                points.append(pos[:2])
            return np.array(points)
        except Exception:
            return None
    
    def _lane_to_polygon(self, lane: AbstractLane) -> Optional[np.ndarray]:
        """Convert lane to polygon."""
        try:
            if hasattr(lane, 'position') and hasattr(lane, 'width_at'):
                length = lane.length if hasattr(lane, 'length') else 10.0
                num_points = max(2, int(length / 2))
                s_values = np.linspace(0, length, num_points)
                
                left_points = []
                right_points = []
                
                for s in s_values:
                    # Handle width differently for different lane types
                    if callable(getattr(lane, 'width_at', None)):
                        width = lane.width_at(s)
                    elif hasattr(lane, 'width'):
                        width = lane.width
                    else:
                        width = 3.5
                    
                    left_pos = lane.position(s, width / 2)
                    right_pos = lane.position(s, -width / 2)
                    left_points.append(left_pos[:2])
                    right_points.append(right_pos[:2])
                
                polygon = left_points + right_points[::-1]
                return np.array(polygon)
            return None
        except:
            return None
    
    def _get_lane_boundaries(self, lane: AbstractLane) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get left and right lane boundaries."""
        try:
            if hasattr(lane, 'position'):
                length = lane.length if hasattr(lane, 'length') else 10.0
                num_points = max(2, int(length / 2))
                s_values = np.linspace(0, length, num_points)
                
                # Get width using the most reliable method
                if callable(getattr(lane, 'width_at', None)):
                    width = lane.width_at(length / 2)
                elif hasattr(lane, 'width'):
                    width = lane.width
                else:
                    width = 3.5
                
                half_width = width / 2
                
                left_points = []
                right_points = []
                
                for s in s_values:
                    left_pos = lane.position(s, half_width)
                    right_pos = lane.position(s, -half_width)
                    left_points.append(left_pos[:2])
                    right_points.append(right_pos[:2])
                
                return np.array(left_points), np.array(right_points)
            return None, None
        except:
            return None, None
    
    def _create_route_corridor(self, centerline: np.ndarray, width: float) -> Optional[np.ndarray]:
        """Create corridor polygon around route centerline."""
        try:
            if len(centerline) < 2:
                return None
            
            left_points = []
            right_points = []
            
            for i in range(len(centerline)):
                if i == 0:
                    direction = centerline[1] - centerline[0]
                elif i == len(centerline) - 1:
                    direction = centerline[-1] - centerline[-2]
                else:
                    direction = centerline[i + 1] - centerline[i - 1]
                
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                normal = np.array([-direction[1], direction[0]])
                
                left_points.append(centerline[i] + normal * width / 2)
                right_points.append(centerline[i] - normal * width / 2)
            
            polygon = left_points + right_points[::-1]
            return np.array(polygon)
        except:
            return None
    
    def _create_boundaries_from_centerline(self, centerline: np.ndarray, width: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create left and right boundaries from a centerline and width.
        
        Args:
            centerline: (N, 2) array of points
            width: width of the lane
        
        Returns:
            Tuple of left and right boundaries as (N, 2) arrays
        """
        try:
            if len(centerline) < 2:
                return None, None
            
            half_width = width / 2
            
            left_boundary = []
            right_boundary = []
            
            for i in range(len(centerline)):
                if i == 0:
                    next_point = centerline[1]
                    current_point = centerline[0]
                    direction = next_point - current_point
                elif i == len(centerline) - 1:
                    prev_point = centerline[-2]
                    current_point = centerline[-1]
                    direction = current_point - prev_point
                else:
                    next_point = centerline[i + 1]
                    prev_point = centerline[i - 1]
                    current_point = centerline[i]
                    direction = next_point - prev_point
                
                # Get perpendicular direction
                direction_norm = np.linalg.norm(direction)
                if direction_norm < 1e-6:
                    continue
                
                direction = direction / direction_norm
                perpendicular = np.array([-direction[1], direction[0]])
                
                # Calculate boundary points
                left_point = current_point + perpendicular * half_width
                right_point = current_point - perpendicular * half_width
                
                left_boundary.append(left_point)
                right_boundary.append(right_point)
            
            return np.array(left_boundary), np.array(right_boundary)
        except:
            return None, None
