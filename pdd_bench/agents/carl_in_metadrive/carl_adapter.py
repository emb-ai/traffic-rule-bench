"""
CaRL agent adarter to run in metadrive.

Based on:
- carl_nuplan/planning/gym/environment/observation_builder/default/default_renderer.py
- carl_nuplan/planning/gym/environment/observation_builder/default/default_observation_builder.py
- carl_nuplan/planning/gym/environment/helper/environment_area.py
- carl_nuplan/planning/gym/environment/trajectory_builder/action_trajectory_builder.py

Parameters nuPlan checkpoint:
- BEV shape: (9, 256, 256) - (C, W, H)
- FOV: front=78m, back=50m, left=64m, right=64m
- pixel_per_meter = 2.0
- Origin is centered in the raster (14m ahead of ego)
- Measurements: 10 values
- Value measurements: 4 values
- Action: [acceleration, steering_angle] in [-1, 1]
"""

import os
import sys
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# CaRL constants
MIN_VALUE = 0
MAX_VALUE = 255
LINE_THICKNESS = 1

# traffic light  (default_renderer.py lines 33-37)
TRAFFIC_LIGHT_VALUE = {
    "green": 80,
    "yellow": 170,
    "red": 255,
    "unknown": 0,
}


@dataclass
class CaRLConfig:
    """
    CaRL nuPlan configuration.
    
    carl_nuplan/planning/script/config/gym/observation_builder/default_observation_builder.yaml:
    - FOV: front=78, back=50, left=64, right=64
    - pixel_per_meter=2.0
    - max_vehicle_speed=30.0
    - max_pedestrian_speed=4.0
    """
    front: float = 78.0
    back: float = 50.0
    left: float = 64.0
    right: float = 64.0
    
    pixel_per_meter: float = 2.0
    output_pixel_width: Optional[int] = 256
    output_pixel_height: Optional[int] = 256
    
    max_vehicle_speed: float = 30.0
    max_pedestrian_speed: float = 4.0
    
    vehicle_scaling: float = 1.0
    pedestrian_scaling: float = 1.0
    static_scaling: float = 1.0
    
    # Action space ( action_trajectory_builder.py)
    scale_max_acceleration: float = 2.4  # m/s²
    scale_max_deceleration: float = 3.2  # m/s²
    scale_max_steering_angle: float = 0.83775804096  # rad (48 deg)
    dt_control: float = 0.1  # s
    
    @property
    def frame_width(self) -> float:
        """Width of environment area in meters."""
        return self.left + self.right  # 128m
    
    @property
    def frame_height(self) -> float:
        """Height of environment area in meters."""
        return self.front + self.back  # 128m
    
    @property
    def pixel_width(self) -> int:
        """Width of raster in pixels."""
        return int(self.frame_width * self.pixel_per_meter)  # 256
    
    @property
    def pixel_height(self) -> int:
        """Height of raster in pixels."""
        return int(self.frame_height * self.pixel_per_meter)  # 256
    
    @property
    def bev_shape(self) -> Tuple[int, int, int]:
        """BEV shape (C, W, H)"""
        return (9, self.pixel_width, self.pixel_height)
    
    @property
    def origin_longitudinal_offset(self) -> float:
        """
        Longitudinal offset of origin from ego.
        """
        return (self.frame_height / 2.0) - self.back
    
    @property
    def origin_lateral_offset(self) -> float:
        """
        Lateral offset of origin from ego.
        """
        return (self.frame_width / 2.0) - self.right


class CaRLRenderer:
    """
    BEV Renderer
    
    Coordinate systems:
    Global: world coordinates (x, y) with arbitrary origin
    Local (relative to origin): 
       - Origin is center of raster, 14m ahead of ego
       - x = lateral (right positive)
       - y = longitudinal (forward positive)
    Pixel:
       - pixel = local * ppm + pixel_center
       - pixel_center = (pixel_height/2, pixel_width/2) = (128, 128)
    """
    
    def __init__(self, config: Optional[CaRLConfig] = None):
        self.config = config or CaRLConfig()
        self.c = self.config  
        
    def _get_empty_raster(self) -> np.ndarray:
        return np.zeros((self.c.pixel_width, self.c.pixel_height), dtype=np.uint8)
    
    def _scale_to_color(self, value: Optional[float], max_value: float) -> int:
        """        
        default_renderer.py line 211
        """
        if value is None:
            value = max_value
        normed = float(np.clip(value / max_value, 0.0, 1.0))
        color = int((MAX_VALUE / 2) * normed + (MAX_VALUE / 2))
        return int(np.clip(color, MIN_VALUE, MAX_VALUE))
    
    def _global_to_local(self, global_coords: np.ndarray, 
                         ego_pos: np.ndarray, ego_heading: float) -> np.ndarray:
        """
        global coordinates to local 
                
        Args:
            global_coords: (N, 2) array in global frame
            ego_pos: (2,) ego position in global frame
            ego_heading: ego heading in radians
        """
        if global_coords.ndim == 1:
            global_coords = global_coords.reshape(1, -1)
        
        origin_x = ego_pos[0] + self.c.origin_longitudinal_offset * np.cos(ego_heading)
        origin_y = ego_pos[1] + self.c.origin_longitudinal_offset * np.sin(ego_heading)
        origin = np.array([origin_x, origin_y])
        
        translated = global_coords - origin
        
        theta = -ego_heading
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        local = translated @ R.T
        
        return local
    
    def _local_to_pixel(self, local_coords: np.ndarray) -> np.ndarray:
        """
        local coordinates to pixel coordinates for cv2
        
        Coordinate Systems:
        Local Frame (from _global_to_local):
           - Index 0 (x): Longitudinal (Forward = positive)
           - Index 1 (y): Lateral (Left = positive)
           
        CaRL Pixel Frame (DefaultRenderer):
           - Raster shape: (Lateral, Longitudinal) = (pixel_width, pixel_height)
           - cv2 uses (col, row) = (x, y)
           - x (col) corresponds to Longitudinal dimension
           - y (row) corresponds to Lateral dimension
           - Ego faces RIGHT (increasing col/x)
           - Left is DOWN (increasing row/y)
        """
        if local_coords.ndim == 1:
            local_coords = local_coords.reshape(1, -1)
        
        local_long = local_coords[:, 0]  # lon
        local_lat = local_coords[:, 1]   # lat
        
        # Image centers
        cx = self.c.pixel_height / 2.0
        cy = self.c.pixel_width / 2.0
        
        # Map to pixels
        px = cx + local_long * self.c.pixel_per_meter
        py = cy + local_lat * self.c.pixel_per_meter
        
        # cv2 expects (x, y) = (col, row)
        pixel_coords = np.stack([px, py], axis=-1)
        
        return pixel_coords.astype(np.int32)
    
    def _render_polygon(self, raster: np.ndarray, local_coords: np.ndarray, 
                        color: int = MAX_VALUE) -> None:
        if len(local_coords) < 3:
            return
        pixels = self._local_to_pixel(local_coords)
        cv2.fillPoly(raster, [pixels], color)
    
    def _render_convex_polygon(self, raster: np.ndarray, local_coords: np.ndarray,
                               color: int = MAX_VALUE) -> None:
        if len(local_coords) < 3:
            return
        pixels = self._local_to_pixel(local_coords)
        cv2.fillConvexPoly(raster, pixels, color)
    
    def _render_polyline(self, raster: np.ndarray, local_coords: np.ndarray,
                         color: int = MAX_VALUE, thickness: int = LINE_THICKNESS) -> None:
        """Render polyline onto raster."""
        if len(local_coords) < 2:
            return
        pixels = self._local_to_pixel(local_coords)
        cv2.polylines(
            raster,
            [pixels],
            isClosed=False,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    
    def _create_bbox_polygon(self, center_local: np.ndarray, heading_local: float,
                             length: float, width: float) -> np.ndarray:
        """Create rotated bbox polygon in local coordinates.
        """
        half_len = length / 2
        half_wid = width / 2
        
        corners = np.array([
            [half_len, half_wid],    
            [half_len, -half_wid],   
            [-half_len, -half_wid], 
            [-half_len, half_wid], 
        ], dtype=np.float64)
        
        cos_h = np.cos(heading_local)
        sin_h = np.sin(heading_local)
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        rotated = corners @ R.T
        
        return rotated + center_local
    
    def render(
        self,
        ego_pos: np.ndarray,
        ego_heading: float,
        ego_speed: float,
        ego_length: float = 4.5,
        ego_width: float = 2.0,
        road_polygons: Optional[List[np.ndarray]] = None,
        route_polygons: Optional[List[np.ndarray]] = None,
        lane_boundaries: Optional[List[np.ndarray]] = None,
        traffic_lights: Optional[List[Dict]] = None,
        stop_signs: Optional[List[Dict]] = None,
        speed_limits: Optional[List[Dict]] = None,
        vehicles: Optional[List[Dict]] = None,
        pedestrians: Optional[List[Dict]] = None,
        static_objects: Optional[List[Dict]] = None,
    ) -> np.ndarray:
        """
        9-channel BEV like CaRL format.
        
        Channel order (from default_renderer.py):
        0: drivable_area - roads, intersections, car parks
        1: route - planned path
        2: lane_boundaries - lane boundary lines
        3: traffic_lights - red=255, yellow=170, green=80
        4: stop_signs
        5: speed_limits - intensity = limit/max_speed
        6: vehicles - intensity = speed/max_speed
        7: pedestrians + static - intensity = speed/max_speed
        8: ego - intensity = speed/max_speed 
        """
        ego_pos = np.array(ego_pos, dtype=np.float64)
        
        ch_drivable = self._get_empty_raster()
        ch_route = self._get_empty_raster()
        ch_lanes = self._get_empty_raster()
        ch_tl = self._get_empty_raster()
        ch_stop = self._get_empty_raster()
        ch_speed = self._get_empty_raster()
        ch_vehicles = self._get_empty_raster()
        ch_pedestrians = self._get_empty_raster()
        ch_ego = self._get_empty_raster()
        
        # Channel 0: Drivable area
        if road_polygons:
            for poly in road_polygons:
                poly = np.array(poly, dtype=np.float64)
                local = self._global_to_local(poly, ego_pos, ego_heading)
                self._render_polygon(ch_drivable, local, MAX_VALUE)
        
        # Channel 1: Route
        if route_polygons:
            for poly in route_polygons:
                poly = np.array(poly, dtype=np.float64)
                local = self._global_to_local(poly, ego_pos, ego_heading)
                self._render_polygon(ch_route, local, MAX_VALUE)
        
        # Channel 2: Lane boundaries
        if lane_boundaries:
            for boundary in lane_boundaries:
                boundary = np.array(boundary, dtype=np.float64)
                local = self._global_to_local(boundary, ego_pos, ego_heading)
                self._render_polyline(ch_lanes, local, MAX_VALUE)
        
        # Channel 3: Traffic lights
        if traffic_lights:
            for tl in traffic_lights:
                pos = np.array(tl["position"], dtype=np.float64)
                local = self._global_to_local(pos, ego_pos, ego_heading)
                state = tl.get("state", "unknown")
                color = TRAFFIC_LIGHT_VALUE.get(state, 0)
                if color > 0:
                    pixels = self._local_to_pixel(local)[0]
                    cv2.circle(ch_tl, tuple(pixels), radius=3, color=color, thickness=-1)
        
        # Channel 4: Stop signs
        if stop_signs:
            for sign in stop_signs:
                if "polygon" in sign:
                    poly = np.array(sign["polygon"], dtype=np.float64)
                    local = self._global_to_local(poly, ego_pos, ego_heading)
                    self._render_polygon(ch_stop, local, MAX_VALUE)
                elif "position" in sign:
                    pos = np.array(sign["position"], dtype=np.float64)
                    local = self._global_to_local(pos, ego_pos, ego_heading)
                    pixels = self._local_to_pixel(local)[0]
                    cv2.circle(ch_stop, tuple(pixels), radius=3, color=MAX_VALUE, thickness=-1)
        
        # Channel 5: Speed limits
        if speed_limits:
            for sl in speed_limits:
                centerline = np.array(sl["centerline"], dtype=np.float64)
                limit_mps = sl.get("limit_mps", self.c.max_vehicle_speed)
                local = self._global_to_local(centerline, ego_pos, ego_heading)
                color = self._scale_to_color(limit_mps, self.c.max_vehicle_speed)
                self._render_polyline(ch_speed, local, color)
        
        # Channel 6: Vehicles
        if vehicles:
            for veh in vehicles:
                pos = np.array(veh["position"], dtype=np.float64)
                local_pos = self._global_to_local(pos, ego_pos, ego_heading)
                
                veh_heading = veh.get("heading", 0.0)
                local_heading = veh_heading - ego_heading
                
                length = veh.get("length", 4.5) * self.c.vehicle_scaling
                width = veh.get("width", 2.0) * self.c.vehicle_scaling
                speed = veh.get("speed", 0.0)
                
                bbox = self._create_bbox_polygon(local_pos[0], local_heading, length, width)
                color = self._scale_to_color(abs(speed), self.c.max_vehicle_speed)
                self._render_convex_polygon(ch_vehicles, bbox, color)
        
        # Channel 7: Pedestrians + Static
        if pedestrians:
            for ped in pedestrians:
                pos = np.array(ped["position"], dtype=np.float64)
                local_pos = self._global_to_local(pos, ego_pos, ego_heading)
                
                ped_heading = ped.get("heading", 0.0)
                local_heading = ped_heading - ego_heading
                
                length = ped.get("length", 0.8) * self.c.pedestrian_scaling
                width = ped.get("width", 0.8) * self.c.pedestrian_scaling
                speed = ped.get("speed", 0.0)
                
                bbox = self._create_bbox_polygon(local_pos[0], local_heading, length, width)
                color = self._scale_to_color(abs(speed), self.c.max_pedestrian_speed)
                self._render_convex_polygon(ch_pedestrians, bbox, color)
        
        if static_objects:
            for obj in static_objects:
                pos = np.array(obj["position"], dtype=np.float64)
                local_pos = self._global_to_local(pos, ego_pos, ego_heading)
                
                obj_heading = obj.get("heading", 0.0)
                local_heading = obj_heading - ego_heading
                
                length = obj.get("length", 1.0) * self.c.static_scaling
                width = obj.get("width", 1.0) * self.c.static_scaling
                
                bbox = self._create_bbox_polygon(local_pos[0], local_heading, length, width)
                # Static objects have 0 speed, so color = 128
                color = self._scale_to_color(0.0, self.c.max_vehicle_speed)
                self._render_convex_polygon(ch_pedestrians, bbox, color)
        
        # Channel 8: Ego vehicle
        ego_local_pos = np.array([-self.c.origin_longitudinal_offset, 0.0])
        ego_bbox = self._create_bbox_polygon(ego_local_pos, 0.0, ego_length, ego_width)
        ego_color = self._scale_to_color(abs(ego_speed), self.c.max_vehicle_speed)
        self._render_convex_polygon(ch_ego, ego_bbox, ego_color)
        
        bev = np.stack([
            ch_drivable,     
            ch_route,        
            ch_lanes,        
            ch_tl,          
            ch_stop,        
            ch_speed,      
            ch_vehicles,    
            ch_pedestrians, 
            ch_ego,        
        ], axis=0)

        if self.c.output_pixel_width and self.c.output_pixel_height:
            out_w = int(self.c.output_pixel_width)
            out_h = int(self.c.output_pixel_height)
            if (out_w, out_h) != (self.c.pixel_width, self.c.pixel_height):
                bev = np.stack(
                    [
                        cv2.resize(ch, (out_h, out_w), interpolation=cv2.INTER_AREA)
                        for ch in bev
                    ],
                    axis=0,
                )

        return bev


class CaRLMeasurementsBuilder:
    """
    Build CaRL measurements vector.
    
    from default_observation_builder.py lines 216-230:
    [0] last_acceleration
    [1] last_steering_rate
    [2] velocity_x (longitudinal)
    [3] velocity_y (lateral)
    [4] acceleration_x
    [5] acceleration_y
    [6] steering_angle
    [7] steering_rate
    [8] angular_velocity (yaw_rate)
    [9] angular_acceleration
    """
    
    def __init__(self):
        self._prev_angular_velocity: Optional[float] = None
        self._prev_velocity: Optional[np.ndarray] = None
        self._prev_steering_angle: Optional[float] = None
        self._dt = 0.1  # 10 Hz (CaRL dt_control)
    
    def reset(self):
        self._prev_angular_velocity = None
        self._prev_velocity = None
        self._prev_steering_angle = None
    
    def build(
        self,
        last_action: np.ndarray,
        velocity: np.ndarray,
        steering_angle: float,
        angular_velocity: float,
        acceleration: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build measurements vector.
        
        Args:
            last_action: previous CaRL action in normalized space [-1, 1]
            velocity: [vx, vy] in ego frame (longitudinal, lateral)
            steering_angle: tire steering angle in radians
            angular_velocity: yaw rate in rad/s
            acceleration: [ax, ay] in ego frame (optional)
        """
        last_accel = float(last_action[0]) if last_action is not None else 0.0
        last_steer_rate = float(last_action[1]) if last_action is not None else 0.0
        
        vx = float(velocity[0])
        vy = float(velocity[1]) if len(velocity) > 1 else 0.0
        
        if acceleration is not None:
            ax = float(acceleration[0])
            ay = float(acceleration[1])
        else:
            if self._prev_velocity is None:
                ax = 0.0
                ay = 0.0
            else:
                ax = (vx - float(self._prev_velocity[0])) / self._dt
                ay = (vy - float(self._prev_velocity[1])) / self._dt
        
        self._prev_velocity = np.array([vx, vy], dtype=np.float32)
        
        if self._prev_steering_angle is None:
            steering_rate = 0.0
        else:
            steering_rate = (float(steering_angle) - float(self._prev_steering_angle)) / self._dt
        self._prev_steering_angle = float(steering_angle)
        
        if self._prev_angular_velocity is None:
            angular_accel = 0.0
        else:
            angular_accel = (float(angular_velocity) - float(self._prev_angular_velocity)) / self._dt
        self._prev_angular_velocity = float(angular_velocity)
        
        measurements = np.array([
            last_accel,         
            last_steer_rate,  
            vx,              
            vy,                
            ax,           
            ay,            
            steering_angle,  
            steering_rate,    
            angular_velocity, 
            angular_accel,   
        ], dtype=np.float32)
        
        return measurements


class CaRLValueMeasurementsBuilder:
    """
    Build value measurements.
    
    Returns [1.0, 1.0, 1.0, 1.0] as placeholder values:
    [0] remaining_time
    [1] remaining_progress
    [2] comfort_score
    [3] ttc_score
    """
    
    @staticmethod
    def build() -> np.ndarray:
        return np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

class CaRLMetaDriveAdapter:
    """
    Usage:
        adapter = CaRLMetaDriveAdapter(checkpoint_path)
        
        obs, info = env.reset()
        adapter.reset()
        
        while not done:
            action = adapter.get_action(env.agent, env.engine)
            obs, reward, done, info = env.step(action)
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        extractor_fov_range: float = 100.0,
        lane_speed_limit_unit: str = "kmh",
        min_speed_limit_kmh: float = 120.0,
        load_model: bool = True,
    ):
        self.config = CaRLConfig()
        self.renderer = CaRLRenderer(self.config)
        self.measurements_builder = CaRLMeasurementsBuilder()
        
        self.device = device
        self.model = None
        self._last_action = np.array([0.0, 0.0], dtype=np.float32)
        self._extractor = None
        
        self._obs_cache_step = None
        self._obs_cache = None
        self._prev_heading = None

        self._extractor_fov_range = float(extractor_fov_range)
        self._lane_speed_limit_unit = str(lane_speed_limit_unit).strip().lower()
        if self._lane_speed_limit_unit not in {"kmh", "mps"}:
            self._lane_speed_limit_unit = "kmh"
        self._min_speed_limit_kmh = float(min_speed_limit_kmh)

        # Longitudinal mode: "tracking" (closed-loop acceleration tracking, matches
        # CaRL's nuPlan action semantics) or "legacy" (old open-loop accel→throttle
        # interp + zeroed acceleration measurements) for A/B and rollback.
        self._longitudinal_mode = os.environ.get("CARL_LONGITUDINAL", "tracking").strip().lower()
        if self._longitudinal_mode not in {"tracking", "legacy"}:
            self._longitudinal_mode = "tracking"
        # Controller state (reset per episode)
        self._a_cmd = 0.0
        self._prev_speed_mps = None
        self._a_meas_ema = 0.0
        self._i_term = 0.0
        
        if load_model:
            if not checkpoint_path:
                raise ValueError("checkpoint_path must be provided when load_model=True")
            self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path: str):
        """Load CaRL PPO model."""
        import torch
        import gymnasium as gym

        # CaRL lives at <sdc>/CaRL/nuPlan; this file is at
        # <sdc>/pdd-bench/agents/carl_in_metadrive/carl_adapter.py — three
        # dirname()s up from THIS_DIR.
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        SDC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS_DIR)))
        CARL_PATH = os.path.join(SDC_ROOT, "third_party", "CaRL", "nuPlan")

        if CARL_PATH not in sys.path:
            sys.path.insert(0, CARL_PATH)
        
        from carl_nuplan.planning.gym.policy.ppo.ppo_model import PPOPolicy
        from carl_nuplan.planning.gym.policy.ppo.ppo_config import GlobalConfig
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        config = GlobalConfig()
        
        observation_space = gym.spaces.Dict({
            "bev_semantics": gym.spaces.Box(
                low=0, high=255,
                shape=(9, 256, 256),  # (C, W, H)
                dtype=np.uint8
            ),
            "measurements": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(10,),
                dtype=np.float32
            ),
            "value_measurements": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(4,),
                dtype=np.float32
            ),
        })
        
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        self.model = PPOPolicy(
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )
        
        self.model.load_state_dict(checkpoint, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded CaRL model from {checkpoint_path}")
    
    def reset(self):
        self._last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.measurements_builder.reset()
        self._extractor = None

        self._obs_cache_step = None
        self._obs_cache = None
        self._prev_heading = None
        self._a_cmd = 0.0
        self._prev_speed_mps = None
        self._a_meas_ema = 0.0
        self._i_term = 0.0
        
    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _get_vehicle_max_steering_rad(vehicle) -> Optional[float]:
        """Return MetaDrive vehicle max steering angle in radians."""
        max_steering = getattr(vehicle, "max_steering", None)
        if max_steering is None:
            return None
        try:
            max_steering = float(max_steering)
        except Exception:
            return None
        if max_steering <= 0:
            return None
        # MetaDrive typically stores max steering in degrees (e.g. 40-50). If it looks like degrees, convert.
        return float(np.deg2rad(max_steering) if max_steering > np.pi else max_steering)
    

    def _init_extractor(self, engine):
        if self._extractor is None:
            from pdd_bench.agents.carl_in_metadrive.metadrive_extractor import MetaDriveExtractor
            fov_range = float(getattr(self, "_extractor_fov_range", 100.0) or 100.0)
            self._extractor = MetaDriveExtractor(
                engine,
                fov_range=fov_range,
            )
    
    def get_observation(self, vehicle, engine) -> Dict[str, np.ndarray]:
        """Get CaRL observation from MetaDrive."""
        
        self._init_extractor(engine)
        
        engine_step = getattr(engine, "episode_step", None) if engine is not None else None
        if engine_step is not None and self._obs_cache_step == int(engine_step) and self._obs_cache is not None:
            return self._obs_cache
        
        data = self._extractor.extract_all(vehicle)
        ego_state = data["ego_state"]
        
        bev = self.renderer.render(
            ego_pos=ego_state["position"],
            ego_heading=ego_state["heading"],
            ego_speed=ego_state["speed"],
            ego_length=ego_state.get("length", 4.5),
            ego_width=ego_state.get("width", 2.0),
            road_polygons=data["road_polygons"],
            route_polygons=data["route_polygons"],
            lane_boundaries=data["lane_boundaries"],
            traffic_lights=data["traffic_lights"],
            stop_signs=data["stop_signs"],
            speed_limits=data["speed_limits"],
            vehicles=data["vehicles"],
            pedestrians=data["pedestrians"],
        )
        
        # Measurements
        heading = float(ego_state["heading"])
        velocity_world = np.array(
            ego_state.get("velocity", getattr(vehicle, "velocity", [ego_state["speed"], 0.0])),
            dtype=np.float64,
        )
        if velocity_world.size < 2:
            velocity_world = np.array([float(ego_state["speed"]), 0.0], dtype=np.float64)
        
        # Transform velocity to ego frame (longitudinal, lateral)
        # In MetaDrive/nuPlan, longitudinal is forward (along heading), lateral is left.
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        # Longitudinal velocity (vx)
        vx = float(velocity_world[0] * cos_h + velocity_world[1] * sin_h)
        # Lateral velocity (vy)
        vy = float(-velocity_world[0] * sin_h + velocity_world[1] * cos_h)
        velocity = np.array([vx, vy], dtype=np.float32)

        # acceleration in ego frame. MetaDrive's BaseVehicle exposes no
        # `acceleration` attribute, so the old getattr fallback fed constant
        # zeros into measurements[4:5] every step (CaRL was trained with real
        # rear-axle acceleration). Passing None lets the measurements builder
        # derive ax/ay by finite difference of the ego-frame velocity.
        if self._longitudinal_mode == "legacy":
            acceleration_world = np.array(getattr(vehicle, "acceleration", [0.0, 0.0]), dtype=np.float64)
            ax = float(acceleration_world[0] * cos_h + acceleration_world[1] * sin_h)
            ay = float(-acceleration_world[0] * sin_h + acceleration_world[1] * cos_h)
            acceleration = np.array([ax, ay], dtype=np.float32)
        else:
            acceleration = None

        max_steering_rad = self._get_vehicle_max_steering_rad(vehicle)
        steering_cmd = float(getattr(vehicle, "steering", 0.0))
        steering = steering_cmd * (max_steering_rad if max_steering_rad is not None else 0.5)

        if self._prev_heading is None:
            angular_velocity = 0.0
        else:
            angular_velocity = self._wrap_to_pi(heading - float(self._prev_heading)) / float(self.config.dt_control)
        self._prev_heading = heading

        measurements = self.measurements_builder.build(
            last_action=self._last_action,
            velocity=velocity,
            steering_angle=steering,
            angular_velocity=angular_velocity,
            acceleration=acceleration,
        )

        value_measurements = CaRLValueMeasurementsBuilder.build()

        obs = {
            "bev_semantics": bev,
            "measurements": measurements,
            "value_measurements": value_measurements,
        }
        if engine_step is not None:
            self._obs_cache_step = int(engine_step)
            self._obs_cache = obs
        return obs

    # --- closed-loop longitudinal tracking (CARL_LONGITUDINAL=tracking) -------
    # In nuPlan, CaRL's action[0] is a TARGET ACCELERATION (×2.4 m/s² gas /
    # ×3.2 brake) realized EXACTLY by the kinematic bicycle model (low-pass
    # tau=0.2 s, reverse driving disabled). MetaDrive's throttle is engine-force
    # based, so the commanded acceleration is tracked with feedforward + PI.
    FF_GAS_MPS2_PER_THROTTLE = 3.0   # ~full-throttle accel at urban speeds
    FF_BRAKE_MPS2_PER_BRAKE = 6.0    # ~full-brake decel
    ACCEL_LOWPASS_TAU_S = 0.2        # nuPlan accel_time_constant
    KP_THROTTLE_PER_MPS2 = 0.25
    KI_THROTTLE_PER_MPS2 = 0.08
    I_TERM_CLAMP = 0.4               # anti-windup bound on the integral term

    def _track_acceleration(self, vehicle, accel_normed: float) -> float:
        """Return a MetaDrive throttle_brake that tracks CaRL's commanded accel."""
        dt = float(self.config.dt_control)
        scale = (self.config.scale_max_acceleration if accel_normed >= 0
                 else self.config.scale_max_deceleration)
        a_target = accel_normed * scale
        alpha = dt / (dt + self.ACCEL_LOWPASS_TAU_S)
        self._a_cmd += alpha * (a_target - self._a_cmd)

        v = float(vehicle.speed_km_h) / 3.6
        a_meas = 0.0 if self._prev_speed_mps is None else (v - self._prev_speed_mps) / dt
        self._prev_speed_mps = v
        # finite-difference accel at 10 Hz is noisy — smooth before feedback
        self._a_meas_ema += 0.5 * (a_meas - self._a_meas_ema)

        # disable_reverse_driving (as in nuPlan): a negative throttle at ~zero
        # speed would put the MetaDrive vehicle into reverse — hold still instead.
        if v < 0.15 and self._a_cmd < 0.0:
            self._i_term = 0.0
            return 0.0

        err = self._a_cmd - self._a_meas_ema
        ff = self._a_cmd / (self.FF_GAS_MPS2_PER_THROTTLE if self._a_cmd >= 0
                            else self.FF_BRAKE_MPS2_PER_BRAKE)
        self._i_term = float(np.clip(self._i_term + self.KI_THROTTLE_PER_MPS2 * err * dt,
                                     -self.I_TERM_CLAMP, self.I_TERM_CLAMP))
        return float(np.clip(ff + self.KP_THROTTLE_PER_MPS2 * err + self._i_term, -1.0, 1.0))

    def carl_action_to_metadrive(self, vehicle, carl_action: np.ndarray, engine=None) -> np.ndarray:
        """Convert a CaRL action to a MetaDrive [steering, throttle] action.

        Steering: target tire angle → normalized MetaDrive steering command.
        Throttle: closed-loop acceleration tracking (default), or the legacy
        open-loop interp when CARL_LONGITUDINAL=legacy.
        """
        max_steering_rad = self._get_vehicle_max_steering_rad(vehicle)
        accel_normed = float(np.clip(carl_action[0], -1.0, 1.0))
        steer_normed = float(np.clip(carl_action[1], -1.0, 1.0))

        if max_steering_rad is not None and max_steering_rad > 0:
            target_steering_angle = steer_normed * self.config.scale_max_steering_angle
            steering_cmd = target_steering_angle / float(max_steering_rad)
        else:
            steering_cmd = steer_normed
        steering_cmd = np.clip(steering_cmd, -1.0, 1.0)

        if self._longitudinal_mode == "legacy":
            xp = [-1.0, -0.1, 0.0, 1.0]
            fp = [-1.0, 0.0, 0.2, 1.0]
            throttle = float(np.interp(accel_normed, xp, fp))
        else:
            throttle = self._track_acceleration(vehicle, accel_normed)

        return np.array([steering_cmd, throttle], dtype=np.float32)
    
    def get_action(self, vehicle, engine, deterministic: bool = True) -> np.ndarray:
        import torch
        if self.model is None:
            raise RuntimeError("CaRL model is not loaded. Initialize adapter with load_model=True.")
        
        obs = self.get_observation(vehicle, engine)
        
        obs_tensor = {
            "bev_semantics": torch.from_numpy(obs["bev_semantics"][None, ...]).to(
                self.device, dtype=torch.float32
            ),
            "measurements": torch.from_numpy(obs["measurements"][None, ...]).to(
                self.device, dtype=torch.float32
            ),
            "value_measurements": torch.from_numpy(obs["value_measurements"][None, ...]).to(
                self.device, dtype=torch.float32
            ),
        }
        
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            output = self.model.forward(
                obs_tensor,
                deterministic=deterministic,
                lstm_state=None,
                done=None,
            )
            carl_action = output[0].squeeze().cpu().numpy()
        
        self._last_action = carl_action.copy()
        
        metadrive_action = self.carl_action_to_metadrive(vehicle, carl_action, engine)
        
        return metadrive_action
    
    def visualize_bev(self, bev: np.ndarray) -> np.ndarray:
        """Visualize BEV channels as a 3x3 grid."""
        c, w, h = bev.shape
        grid = np.zeros((3 * h, 3 * w), dtype=np.uint8)

        for i in range(9):
            row, col = i // 3, i % 3
            ch = bev[i]
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = ch
        
        return cv2.cvtColor(grid, cv2.COLOR_GRAY2RGB)


if __name__ == "__main__":
    """Test configuration."""
    config = CaRLConfig()
    print(f"BEV shape: {config.bev_shape}")
    print(f"Pixel size: {config.pixel_width}x{config.pixel_height}")
    print(f"Origin offset: {config.origin_longitudinal_offset}m ahead of ego")
