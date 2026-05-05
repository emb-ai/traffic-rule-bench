"""
Main road traffic manager for yield sign scenarios.

Spawns a single NPC vehicle on the incoming main road (the road that crosses
ego's path at the intersection) with proper navigation to drive through
the intersection and respawn at the start.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from metadrive.manager.base_manager import BaseManager
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.policy.idm_policy import IDMPolicy


class YieldMainRoadTrafficManager(BaseManager):
    """
    Traffic manager that spawns a vehicle on the incoming main road at intersections.
    
    For yield sign scenarios, the ego vehicle is on the secondary road and must
    yield to traffic on the main road. This manager creates that main road traffic
    with a proper route through the intersection.
    
    The vehicle is spawned when ego is within `spawn_trigger_distance` meters
    of the yield sign (not immediately at scene start).
    """
    
    RESPAWN_DISTANCE_THRESHOLD = 5.0  # Respawn when within this distance of destination
    
    def __init__(
        self,
        num_vehicles: int = 1,
        spawn_velocity: float = 10.0,
        spawn_trigger_distance: float = 15.0,
    ):
        """
        Args:
            num_vehicles: Number of vehicles to spawn (typically 1)
            spawn_velocity: Initial velocity in m/s
            spawn_trigger_distance: Spawn when ego is within this distance of yield sign (meters)
        """
        super().__init__()
        self._traffic_vehicles: List[BaseVehicle] = []
        self._vehicle_routes: Dict[str, dict] = {}  # vehicle_id -> route info
        self._num_vehicles = num_vehicles
        self._spawn_velocity = spawn_velocity
        self._spawn_trigger_distance = spawn_trigger_distance
        self._spawn_lane = None
        self._destination_node = None
        self._spawn_longitude = 5.0
        self._spawn_triggered = False
        self._yield_sign = None
        
    def reset(self):
        """Reset manager state."""
        self._traffic_vehicles = []
        self._vehicle_routes = {}
        self._spawn_lane = None
        self._destination_node = None
        self._spawn_triggered = False
        self._yield_sign = None
        
    def after_reset(self):
        """Called after environment reset - identify main road route (but don't spawn yet)."""
        self._identify_main_road_route()
        self._find_yield_sign()
        # Don't spawn immediately - wait for ego to approach the yield sign
        
    def _identify_main_road_route(self):
        """
        Identify the incoming lane on the main road and the destination.
        
        For T/X intersections:
        - Find the perpendicular road (main road)
        - Get the START of that road (where traffic comes from)
        - Get the END of the opposite side (where traffic goes to)
        - Use shortest_path to find the route through the intersection
        """
        print("[YieldTraffic] _identify_main_road_route called")
        
        current_map = self.engine.map_manager.current_map
        if current_map is None:
            print("[YieldTraffic] ERROR: No current map")
            return
            
        blocks = current_map.blocks
        print(f"[YieldTraffic] Found {len(blocks)} blocks")
        
        if len(blocks) < 2:
            print("[YieldTraffic] ERROR: Need at least 2 blocks")
            return
            
        intersection_block = blocks[-1]
        block_id = getattr(intersection_block, 'ID', None)
        print(f"[YieldTraffic] Intersection block ID: {block_id}")
        
        if block_id not in ('T', 'X'):
            print(f"[YieldTraffic] ERROR: Expected T or X intersection, got {block_id}")
            return
            
        road_network = current_map.road_network
        
        try:
            sockets = intersection_block.get_socket_list()
            print(f"[YieldTraffic] Found {len(sockets)} sockets")
            for i, s in enumerate(sockets):
                print(f"[YieldTraffic]   Socket {i}: pos_road={s.positive_road}, neg_road={s.negative_road}")
        except Exception as e:
            print(f"[YieldTraffic] ERROR: Failed to get sockets: {e}")
            return
            
        if len(sockets) < 2:
            print("[YieldTraffic] ERROR: Need at least 2 sockets")
            return
            
        # For yield sign: ego comes from spawn road (south), main road is perpendicular
        # 
        # X-intersection layout:
        #   Socket 1 (up/north)
        #        |
        # Socket 2 (left) --- X --- Socket 0 (right)
        #        |
        #   Spawn road (south) - where ego comes from
        #
        # For a vehicle to cross ego's path (from right to left):
        # - Start: socket 0's negative_road end_node (outer edge of right road)
        # - End: socket 2's positive_road end_node (outer edge of left road)
        #
        # T-intersection:
        # Socket 0 --- T --- Socket 1
        #        |
        #   Spawn road
        
        if block_id == 'X':
            incoming_socket_idx = 0  # Right
            outgoing_socket_idx = 2  # Left
        else:  # T-intersection
            incoming_socket_idx = 0
            outgoing_socket_idx = 1
        
        print(f"[YieldTraffic] Using incoming socket {incoming_socket_idx}, outgoing socket {outgoing_socket_idx}")
            
        if incoming_socket_idx >= len(sockets) or outgoing_socket_idx >= len(sockets):
            print(f"[YieldTraffic] ERROR: Socket indices out of range (have {len(sockets)} sockets)")
            return
            
        incoming_socket = sockets[incoming_socket_idx]
        outgoing_socket = sockets[outgoing_socket_idx]
        
        try:
            # The negative_road of a socket goes FROM the outer edge TO the intersection center
            # So negative_road.start_node is at the outer edge (where we want to spawn)
            # And negative_road.end_node is at the intersection center
            
            # For spawning: we need a lane on the negative_road
            # The vehicle starts at the outer edge and drives toward the center,
            # then through the intersection to the other side
            
            start_node = incoming_socket.negative_road.start_node
            end_node = outgoing_socket.positive_road.end_node
            
            print(f"[YieldTraffic] Route: {start_node} -> {end_node}")
            
            # Find shortest path through the intersection
            try:
                checkpoints = road_network.shortest_path(
                    (start_node, incoming_socket.negative_road.end_node, 0),  # lane index
                    end_node
                )
                print(f"[YieldTraffic] Checkpoints: {checkpoints}")
            except Exception as e:
                print(f"[YieldTraffic] shortest_path failed: {e}, trying direct lane lookup")
                checkpoints = None
            
            # Get the first lane segment (spawn lane)
            neg_road = incoming_socket.negative_road
            print(f"[YieldTraffic] Negative road: {neg_road.start_node} -> {neg_road.end_node}")
            
            # Get lanes on this road segment
            try:
                spawn_lanes = road_network.graph[neg_road.start_node][neg_road.end_node]
                print(f"[YieldTraffic] Found {len(spawn_lanes)} spawn lanes")
                for i, lane in enumerate(spawn_lanes):
                    print(f"[YieldTraffic]   Lane {i}: {lane.index}, length={lane.length:.1f}m")
                    
                if spawn_lanes:
                    self._spawn_lane = spawn_lanes[0]
                    self._destination_node = end_node
                    print(f"[YieldTraffic] Selected spawn lane: {self._spawn_lane.index}")
                    print(f"[YieldTraffic] Destination: {self._destination_node}")
                else:
                    print("[YieldTraffic] ERROR: No spawn lanes found")
            except KeyError as e:
                print(f"[YieldTraffic] ERROR: Road segment not in graph: {e}")
                # Try to find any available lane from the socket
                print(f"[YieldTraffic] Trying get_negative_lanes fallback...")
                try:
                    spawn_lanes = incoming_socket.get_negative_lanes(road_network)
                    if spawn_lanes:
                        self._spawn_lane = spawn_lanes[0]
                        self._destination_node = end_node
                        print(f"[YieldTraffic] Fallback spawn lane: {self._spawn_lane.index}")
                except Exception as e2:
                    print(f"[YieldTraffic] Fallback also failed: {e2}")
            
        except Exception as e:
            import traceback
            print(f"[YieldTraffic] ERROR: Failed to identify route: {e}")
            traceback.print_exc()
            return
        
    def _spawn_main_road_vehicles(self):
        """Spawn NPC vehicle(s) on the main road with proper navigation."""
        print(f"[YieldTraffic] _spawn_main_road_vehicles called, num_vehicles={self._num_vehicles}")
        
        if self._spawn_lane is None or self._destination_node is None:
            print("[YieldTraffic] ERROR: No valid route identified (spawn_lane or destination is None)")
            return
            
        for i in range(self._num_vehicles):
            print(f"[YieldTraffic] Spawning vehicle {i+1}/{self._num_vehicles}")
            vehicle = self._spawn_vehicle_with_route(
                spawn_offset=i * 15.0  # Space vehicles apart
            )
            if vehicle is not None:
                print(f"[YieldTraffic] Successfully spawned vehicle {vehicle.id}")
            else:
                print(f"[YieldTraffic] Failed to spawn vehicle {i+1}")
                
    def _spawn_vehicle_with_route(self, spawn_offset: float = 0.0) -> Optional[BaseVehicle]:
        """
        Spawn a single vehicle with proper navigation.
        
        Args:
            spawn_offset: Additional longitudinal offset for spacing multiple vehicles
            
        Returns:
            The spawned vehicle, or None on failure
        """
        from metadrive.component.vehicle.vehicle_type import DefaultVehicle
        
        print(f"[YieldTraffic] _spawn_vehicle_with_route called, offset={spawn_offset}")
        
        try:
            lane = self._spawn_lane
            spawn_long = self._spawn_longitude + spawn_offset
            spawn_long = max(2.0, min(spawn_long, lane.length - 5.0))
            
            print(f"[YieldTraffic] Spawning at lane {lane.index}, longitude={spawn_long:.1f}")
            
            traffic_v_config = {
                "spawn_lane_index": lane.index,
                "spawn_longitude": spawn_long,
                "enable_reverse": False,
                "destination": self._destination_node,
            }
            
            print(f"[YieldTraffic] Vehicle config: {traffic_v_config}")
            
            vehicle = self.spawn_object(DefaultVehicle, vehicle_config=traffic_v_config)
            print(f"[YieldTraffic] Vehicle spawned: {vehicle.id}")
            print(f"[YieldTraffic] Vehicle position: {vehicle.position}")
            
            # Initialize navigation with the route
            try:
                if hasattr(vehicle, 'navigation') and vehicle.navigation is not None:
                    vehicle.navigation.set_route(lane.index, self._destination_node)
                    print(f"[YieldTraffic] Navigation route set")
            except Exception as e:
                print(f"[YieldTraffic] Navigation setup note: {e}")
            
            # Set initial velocity
            try:
                vehicle.set_velocity([self._spawn_velocity, 0.0], in_local_frame=True)
                print(f"[YieldTraffic] Velocity set to {self._spawn_velocity} m/s")
            except Exception as e:
                print(f"[YieldTraffic] Failed to set velocity: {e}")
                
            # Add IDM policy for autonomous driving
            self.add_policy(vehicle.id, IDMPolicy, vehicle, self.generate_seed())
            print(f"[YieldTraffic] IDM policy added")
            
            self._traffic_vehicles.append(vehicle)
            self._vehicle_routes[vehicle.id] = {
                "spawn_lane": lane,
                "destination": self._destination_node,
                "spawn_longitude": spawn_long,
            }
            
            return vehicle
            
        except Exception as e:
            import traceback
            print(f"[YieldTraffic] ERROR: Failed to spawn vehicle: {e}")
            traceback.print_exc()
            return None
            
    def _find_yield_sign(self):
        """Find the yield sign in the scene."""
        try:
            sign_mgr = getattr(self.engine, "traffic_sign_manager", None)
            if sign_mgr is None:
                print("[YieldTraffic] No traffic_sign_manager found")
                return
                
            for sign in sign_mgr.signs:
                sign_class_name = type(sign).__name__.lower()
                if "yield" in sign_class_name:
                    self._yield_sign = sign
                    print(f"[YieldTraffic] Found yield sign: {type(sign).__name__}")
                    return
                    
            print("[YieldTraffic] No yield sign found in scene")
        except Exception as e:
            print(f"[YieldTraffic] Error finding yield sign: {e}")
            
    def _get_ego_distance_to_yield_sign(self) -> Optional[float]:
        """Get distance from ego to the yield sign."""
        if self._yield_sign is None:
            return None
            
        try:
            # Get ego vehicle
            ego = None
            for agent in self.engine.agent_manager.active_agents.values():
                ego = agent
                break
                
            if ego is None:
                return None
                
            # Get sign position
            sign_lane = getattr(self._yield_sign, 'lane', None)
            sign_long = getattr(self._yield_sign, 'placement_long', None) or getattr(self._yield_sign, 'zone_start', None)
            
            if sign_lane is None or sign_long is None:
                return None
                
            # Check if ego is on the same road as the sign
            ego_lane = getattr(ego, 'lane', None)
            if ego_lane is None:
                return None
                
            # Get ego's longitudinal position on its lane
            try:
                ego_long, _ = ego_lane.local_coordinates(ego.position)
            except:
                return None
                
            # If ego is on a lane leading to the sign's lane, calculate distance
            # Simple approximation: if on same road segment, use longitudinal difference
            ego_idx = getattr(ego_lane, 'index', None)
            sign_idx = getattr(sign_lane, 'index', None)
            
            if ego_idx is not None and sign_idx is not None:
                # Same road segment
                if ego_idx[0] == sign_idx[0] and ego_idx[1] == sign_idx[1]:
                    distance = sign_long - ego_long
                    return distance if distance > 0 else None
                    
            # Different segments - use Euclidean distance as fallback
            sign_pos = sign_lane.position(sign_long, 0)
            ego_pos = ego.position
            distance = float(np.linalg.norm(np.array(sign_pos) - np.array(ego_pos)))
            return distance
            
        except Exception as e:
            return None
    
    def before_step(self):
        """Execute driving decisions for all traffic vehicles."""
        # Check if we should spawn (ego approaching yield sign)
        if not self._spawn_triggered and self._spawn_lane is not None:
            distance = self._get_ego_distance_to_yield_sign()
            if distance is not None and distance <= self._spawn_trigger_distance:
                print(f"[YieldTraffic] Ego is {distance:.1f}m from yield sign - spawning traffic!")
                self._spawn_main_road_vehicles()
                self._spawn_triggered = True
        
        for vehicle in self._traffic_vehicles:
            try:
                policy = self.engine.get_policy(vehicle.name)
                if policy is not None:
                    vehicle.before_step(policy.act())
            except Exception as e:
                logging.debug(f"YieldMainRoadTrafficManager: Policy error: {e}")
        return {}
    
    def is_vehicle_in_main_zone(self) -> bool:
        """Check if any main road traffic vehicle is in the intersection zone."""
        for vehicle in self._traffic_vehicles:
            try:
                if not vehicle.on_lane:
                    continue
                # Vehicle is considered "in main zone" if it exists and is on a lane
                return True
            except:
                pass
        return False
    
    def get_main_road_vehicle_info(self) -> Optional[dict]:
        """Get info about the main road vehicle for display."""
        if not self._traffic_vehicles:
            return None
            
        try:
            v = self._traffic_vehicles[0]
            return {
                "exists": True,
                "on_lane": v.on_lane if hasattr(v, 'on_lane') else None,
                "speed_kmh": float(v.speed_km_h) if hasattr(v, 'speed_km_h') else None,
                "position": list(v.position) if hasattr(v, 'position') else None,
            }
        except:
            return None
        
    def after_step(self, *args, **kwargs):
        """
        Update vehicle states and respawn vehicles that have completed their route.
        """
        vehicles_to_respawn = []
        
        for vehicle in list(self._traffic_vehicles):
            try:
                vehicle.after_step()
                
                # Check if vehicle should respawn
                if self._should_respawn(vehicle):
                    vehicles_to_respawn.append(vehicle)
                    
            except Exception as e:
                logging.debug(f"YieldMainRoadTrafficManager: after_step error: {e}")
                vehicles_to_respawn.append(vehicle)
                
        # Respawn vehicles
        for vehicle in vehicles_to_respawn:
            self._respawn_vehicle(vehicle)
            
        return {}
        
    def _should_respawn(self, vehicle: BaseVehicle) -> bool:
        """
        Check if a vehicle should be respawned.
        
        Returns True if:
        - Vehicle is off-road
        - Vehicle has reached the destination
        - Vehicle navigation indicates arrival
        """
        try:
            # Off-road check
            if not vehicle.on_lane:
                return True
            
            # Check navigation arrival
            if hasattr(vehicle, 'navigation') and vehicle.navigation is not None:
                nav = vehicle.navigation
                if hasattr(nav, 'arrive_destination') and nav.arrive_destination:
                    return True
                    
            # Check if near the end of final road
            route_info = self._vehicle_routes.get(vehicle.id)
            if route_info is None:
                return True
                
            # Check progress along route
            lane = vehicle.lane
            if lane is not None:
                try:
                    local_coords = lane.local_coordinates(vehicle.position)
                    progress = local_coords[0] / lane.length if lane.length > 0 else 0
                    
                    # If we're at the end of a lane and it's likely the final segment
                    if progress > 0.95:
                        # Check if this is near the destination area
                        lane_idx = getattr(lane, 'index', None)
                        if lane_idx is not None:
                            # The destination node should match the end of current lane
                            dest = route_info.get('destination')
                            if lane_idx[1] == dest:
                                return True
                except Exception:
                    pass
                
            return False
            
        except Exception:
            return True
            
    def _respawn_vehicle(self, vehicle: BaseVehicle):
        """Remove old vehicle and spawn a new one at the start of the route."""
        try:
            vehicle_id = vehicle.id
            self.clear_objects([vehicle_id])
            self._traffic_vehicles.remove(vehicle)
            if vehicle_id in self._vehicle_routes:
                del self._vehicle_routes[vehicle_id]
        except Exception as e:
            logging.debug(f"YieldMainRoadTrafficManager: Error removing vehicle: {e}")
            return
            
        # Spawn new vehicle
        new_vehicle = self._spawn_vehicle_with_route()
        if new_vehicle is not None:
            logging.debug(f"YieldMainRoadTrafficManager: Respawned vehicle {new_vehicle.id}")
        
    def before_reset(self):
        """Clear all traffic vehicles before reset."""
        super().before_reset()
        self.clear_objects([v.id for v in self._traffic_vehicles])
        self._traffic_vehicles = []
        self._vehicle_routes = {}
        self._spawn_lane = None
        self._destination_node = None
        
    def destroy(self):
        """Clean up resources."""
        self.clear_objects([v.id for v in self._traffic_vehicles])
        self._traffic_vehicles = []
        self._vehicle_routes = {}
        
    @property
    def traffic_vehicles(self):
        return list(self._traffic_vehicles)


def add_main_road_traffic(
    env,
    num_vehicles: int = 1,
    spawn_velocity: float = 10.0,
    spawn_trigger_distance: float = 15.0,
) -> Optional[YieldMainRoadTrafficManager]:
    """
    Helper function to add main road traffic to an existing environment.
    
    Call this AFTER env.reset() to add traffic on the main road.
    The vehicle will spawn when ego is within spawn_trigger_distance of the yield sign.
    
    Args:
        env: TrafficSignEnv instance
        num_vehicles: Number of vehicles (typically 1)
        spawn_velocity: Initial velocity in m/s
        spawn_trigger_distance: Spawn when ego is within this distance of yield sign (meters)
        
    Returns:
        The traffic manager instance, or None if registration failed
        
    Example:
        env = TrafficSignEnv(config)
        obs, info = env.reset()
        traffic_mgr = add_main_road_traffic(env, num_vehicles=1, spawn_velocity=10.0)
    """
    print(f"[YieldTraffic] add_main_road_traffic called with num_vehicles={num_vehicles}, velocity={spawn_velocity}, trigger_dist={spawn_trigger_distance}")
    
    if not hasattr(env, 'engine') or env.engine is None:
        print("[YieldTraffic] ERROR: Environment has no engine")
        return None
        
    manager = YieldMainRoadTrafficManager(
        num_vehicles=num_vehicles,
        spawn_velocity=spawn_velocity,
        spawn_trigger_distance=spawn_trigger_distance,
    )
    
    try:
        # Register manager with engine
        env.engine.register_manager("yield_traffic_manager", manager)
        print("[YieldTraffic] Manager registered with engine")
        
        # Initialize the manager
        manager.reset()
        manager.after_reset()
        
        print(f"[YieldTraffic] Spawned {len(manager.traffic_vehicles)} vehicles")
        if manager._spawn_lane:
            print(f"[YieldTraffic] Spawn lane: {manager._spawn_lane.index}")
        if manager._destination_node:
            print(f"[YieldTraffic] Destination: {manager._destination_node}")
        
        if len(manager.traffic_vehicles) == 0:
            print("[YieldTraffic] WARNING: No vehicles spawned!")
            
        return manager
        
    except Exception as e:
        import traceback
        print(f"[YieldTraffic] ERROR: Failed to initialize: {e}")
        traceback.print_exc()
        return None


def get_main_road_traffic_summary(manager: YieldMainRoadTrafficManager) -> dict:
    """Get a summary of main road traffic state for logging/debugging."""
    if manager is None:
        return {"enabled": False}
        
    vehicles_info = []
    for v in manager.traffic_vehicles:
        try:
            vehicles_info.append({
                "id": v.id,
                "position": list(v.position) if hasattr(v, 'position') else None,
                "speed_kmh": float(v.speed_km_h) if hasattr(v, 'speed_km_h') else None,
                "on_lane": v.on_lane if hasattr(v, 'on_lane') else None,
            })
        except Exception:
            pass
            
    return {
        "enabled": True,
        "num_vehicles": len(manager.traffic_vehicles),
        "spawn_lane": str(manager._spawn_lane.index) if manager._spawn_lane else None,
        "destination": manager._destination_node,
        "vehicles": vehicles_info,
    }
