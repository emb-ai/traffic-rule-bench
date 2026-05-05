"""
features: must_stop, dist_to_stop, time_to_stop, ego_speed, stop_cleared, hold_remaining, required_decel, lead_gap
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Set, Tuple
import numpy as np
from traffic_signs.base_traffic_sign import BaseTrafficSign


def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def clamp_signed(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


@dataclass
class StopRuleState:
    """Tracks per-sign state for STOP handling."""
    active_sign_id: Optional[Any] = None
    
    approached: bool = False           
    stopped_before_line: bool = False  
    stop_hold_time: float = 0.0        
    stop_cleared: bool = False         
    
    handled_sign_ids: Set[Any] = field(default_factory=set)
    violated_sign_ids: Set[Any] = field(default_factory=set)



class StopRuleFeatureExtractor:

    IDX_MUST_STOP = 0
    IDX_DIST_TO_STOP = 1
    IDX_TIME_TO_STOP = 2
    IDX_EGO_SPEED = 3
    IDX_STOP_CLEARED = 4
    IDX_HOLD_REMAINING = 5
    IDX_REQUIRED_DECEL = 6
    IDX_LEAD_GAP = 7
    
    N_FEATURES = 8
    
    def __init__(
        self,
        dt: float = 0.1,
        # Normalization constants
        dist_norm_max_m: float = 50.0,
        time_norm_max_s: float = 10.0,
        speed_norm_max_mps: float = 10.0,
        decel_norm_max_mps2: float = 8.0,
        lead_gap_max_m: float = 50.0,
        approach_dist_m: float = 30.0,      
        stop_speed_thresh_mps: float = 0.1, 
        required_hold_time_s: float = 1.0, 
    ):
        self.dt = dt
        
        self.dist_norm_max_m = dist_norm_max_m
        self.time_norm_max_s = time_norm_max_s
        self.speed_norm_max_mps = speed_norm_max_mps
        self.decel_norm_max_mps2 = decel_norm_max_mps2
        self.lead_gap_max_m = lead_gap_max_m
        
        self.approach_dist_m = approach_dist_m
        self.stop_speed_thresh_mps = stop_speed_thresh_mps
        self.required_hold_time_s = required_hold_time_s
        
        self.state = StopRuleState()
    
    def reset(self):
        self.state = StopRuleState()
        
    
    def _sign_id(self, sign) -> Any:
        sid = getattr(sign, "id", None)
        return sid if sid is not None else id(sign)
    
    def _get_nearest_stop_sign(self, engine, agent) -> Tuple[float, Optional[Any]]:

        manager = getattr(engine, "traffic_sign_manager", None)
        if manager is None or not hasattr(manager, "signs") or not manager.signs:
            return 1e9, None

        ego_heading = float(getattr(agent, "heading_theta", 0.0))

        best_dist = 1e9
        best_sign = None

        def _wrap_pi(x: float) -> float:
            return float((x + np.pi) % (2 * np.pi) - np.pi)

        def _direction_match(sign_lane_index):
            """
             True if direction matches, False if mismatched or None if can't detect
            """
            if sign_lane_index is None:
                return None
            agent_lane = getattr(agent, "lane", None)
            agent_lane_index = getattr(agent_lane, "index", None) if agent_lane is not None else None
            if agent_lane_index is not None:
                return BaseTrafficSign._same_road_direction(agent_lane_index, sign_lane_index)
            navigation = getattr(agent, "navigation", None)
            if navigation is not None and hasattr(navigation, "current_ref_lanes"):
                ref_lanes = navigation.current_ref_lanes or []
                if len(ref_lanes) == 0:
                    return None
                for ref_lane in ref_lanes:
                    ref_idx = getattr(ref_lane, "index", None)
                    if BaseTrafficSign._same_road_direction(ref_idx, sign_lane_index):
                        return True
                return False
            return None
        
        for sign in manager.signs:
            if sign is None:
                continue

            try:
                from traffic_signs.stop_sign import StopSign
            except Exception:
                StopSign = None
            sign_type = type(sign).__name__
            if StopSign is not None:
                if not (isinstance(sign, StopSign) or sign_type == "StopSign"):
                    continue
            else:
                if sign_type != "StopSign":
                    continue
            
            try:
                lane = getattr(sign, "lane", None)
                if lane is None:
                    continue
                sign_lane_index = getattr(lane, "index", None)
                dir_match = _direction_match(sign_lane_index)
                if dir_match is False:
                    continue
                if dir_match is None:
                    if hasattr(sign, "is_in_drivable_area") and not sign.is_in_drivable_area(agent):
                        continue

                veh_long, _ = lane.local_coordinates(agent.position)
                s = float(np.clip(veh_long, 0.1, lane.length - 0.1))

                lane_heading = float(lane.heading_theta_at(s))
                heading_diff = abs(_wrap_pi(lane_heading - ego_heading))
                if heading_diff > 1.1:  # 63 deg
                    continue

                stop_long = float(getattr(sign, "stop_line_position", lane.length - 0.5))
                dist_to_line = float(stop_long - veh_long)

                if dist_to_line < -5.0:
                    continue

                if dist_to_line < best_dist:
                    best_dist = dist_to_line
                    best_sign = sign
            except Exception:
                continue
        
        return best_dist, best_sign
        
    def _get_lead_vehicle_gap(self, engine, agent) -> float:

        try:
            from metadrive.component.vehicle.base_vehicle import BaseVehicle
        except Exception:
            return self.lead_gap_max_m
        
        ego_pos = np.array(agent.position[:2], dtype=np.float64)
        ego_heading = float(getattr(agent, "heading_theta", 0.0))
        forward = np.array([np.cos(ego_heading), np.sin(ego_heading)])
        
        min_gap = self.lead_gap_max_m
        
        try:
            vehicles = engine.get_objects(
                lambda o: isinstance(o, BaseVehicle) and o is not agent
            ).values()
        except Exception:
            return self.lead_gap_max_m
        
        for veh in vehicles:
            try:
                veh_pos = np.array(veh.position[:2], dtype=np.float64)
                rel = veh_pos - ego_pos
                dist = float(np.linalg.norm(rel))
                
                if dist < 1.0 or dist > self.lead_gap_max_m:
                    continue
                
                ahead_proj = float(np.dot(rel, forward))
                if ahead_proj < 0:
                    continue  
                
                # if in same lane 
                lateral = abs(float(np.dot(rel, np.array([-forward[1], forward[0]]))))
                if lateral > 3.0:  
                    continue
                
                if ahead_proj < min_gap:
                    min_gap = ahead_proj
            except Exception:
                continue
        
        return min_gap
    
    def _update_stop_state(
        self,
        sign: Optional[Any],
        dist_to_line: float,
        ego_speed: float,
    ):
        """Update internal stop tracking state."""
        if sign is None:
            if self.state.active_sign_id is not None:
                self.state = StopRuleState()
            return
        
        sid = self._sign_id(sign)
        
        if self.state.active_sign_id != sid:
            old_violated_ids = self.state.violated_sign_ids.copy()
            old_handled_ids = self.state.handled_sign_ids.copy()
            
            self.state = StopRuleState()
            self.state.active_sign_id = sid
            
            self.state.violated_sign_ids = old_violated_ids
            self.state.handled_sign_ids = old_handled_ids
        
        if sid in self.state.handled_sign_ids:
            self.state.stop_cleared = True
            return
        
        if dist_to_line < -3.0 and not self.state.stop_cleared:
            if sid in self.state.violated_sign_ids:
                self.state.handled_sign_ids.add(sid)
                self.state.stop_cleared = False 
            else:
                self.state.handled_sign_ids.add(sid)
            return
        
        if not self.state.approached and dist_to_line <= self.approach_dist_m:
            self.state.approached = True
        
        is_before_line = dist_to_line > -0.5
        is_stopped = ego_speed < self.stop_speed_thresh_mps
        
        if self.state.approached and is_before_line and is_stopped:
            if not self.state.stopped_before_line:
                self.state.stopped_before_line = True
                self.state.stop_hold_time = 0.0
            else:
                self.state.stop_hold_time += self.dt
            
            if self.state.stop_hold_time >= self.required_hold_time_s:
                self.state.stop_cleared = True
                self.state.handled_sign_ids.add(sid)
        else:
            if self.state.stopped_before_line and not is_stopped and not self.state.stop_cleared:
                self.state.stop_hold_time = 0.0
    
    def step_features(self, engine, agent) -> np.ndarray:
 
        dist_to_line, sign = self._get_nearest_stop_sign(engine, agent)
        
        ego_speed = float(getattr(agent, "speed", 0.0) or 0.0)
        if not np.isfinite(ego_speed) or ego_speed < 0:
            ego_speed = 0.0
        
        self._update_stop_state(sign, dist_to_line, ego_speed)
        
        sign_exists = sign is not None
        sign_is_ahead = dist_to_line > -2.0 
        sign_in_approach_zone = sign_exists and dist_to_line <= self.approach_dist_m
        
        # must_stop
        must_stop = 1.0 if (sign_in_approach_zone and sign_is_ahead and not self.state.stop_cleared) else 0.0
        
        # dist_to_stop (signed, normalized to [-1, 1])
        if sign_exists:
            dist_norm = clamp_signed(dist_to_line / self.dist_norm_max_m, -1.0, 1.0)
        else:
            dist_norm = 1.0 
        
        # time_to_stop = dist / speed
        if sign_exists and ego_speed > 0.1 and dist_to_line > 0:
            time_to_stop = dist_to_line / ego_speed
            time_to_stop_norm = clamp01(time_to_stop / self.time_norm_max_s)
        elif sign_exists and dist_to_line <= 0:
            time_to_stop_norm = 0.0 
        else:
            time_to_stop_norm = 1.0 
        
        # ego_speed (normalized)
        ego_speed_norm = clamp01(ego_speed / self.speed_norm_max_mps)
        
        # stop_cleared (0/1)
        stop_cleared = 1.0 if self.state.stop_cleared else 0.0
        
        # hold_remaining (how much longer to hold)
        if self.state.stop_cleared:
            hold_remaining_norm = 0.0
        elif self.state.stopped_before_line:
            remaining = max(0.0, self.required_hold_time_s - self.state.stop_hold_time)
            hold_remaining_norm = clamp01(remaining / self.required_hold_time_s)
        else:
            hold_remaining_norm = 1.0 
        
        # required_decel = v**2 / (2 * d)
        if sign_exists and dist_to_line > 0.5:
            required_decel = (ego_speed ** 2) / (2.0 * dist_to_line)
            required_decel_norm = clamp01(required_decel / self.decel_norm_max_mps2)
        else:
            required_decel_norm = 0.0
        
        # lead_gap (distance to vehicle ahead)
        lead_gap = self._get_lead_vehicle_gap(engine, agent)
        lead_gap_norm = clamp01(lead_gap / self.lead_gap_max_m)
        
        return np.array([
            must_stop,            
            dist_norm,           
            time_to_stop_norm, 
            ego_speed_norm,    
            stop_cleared, 
            hold_remaining_norm,
            required_decel_norm,
            lead_gap_norm,
        ], dtype=np.float32)
    
