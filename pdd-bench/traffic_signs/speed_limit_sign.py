from traffic_signs.base_traffic_sign import BaseTrafficSign
from typing import Optional


class SpeedLimitSign(BaseTrafficSign):
    def __init__(
        self, 
        lane, 
        speed_limit: float = 20,
        zone_length: Optional[float] = None,
        longitudinal_offset: float = 0.0,
        **kwargs
    ):
        if hasattr(lane, 'speed') and lane.speed is not None:
            self.speed_limit = round(lane.speed * 3.6)
        else:
            self.speed_limit = speed_limit
        lane.speed_limit = self.speed_limit

        base_longitudinal_offset = -lane.length + longitudinal_offset
        super().__init__(
            lane, 
            longitudinal_offset=base_longitudinal_offset, 
            icon_path=f"3.24_{self.speed_limit:02d}.png", 
            **kwargs
        )
        self.sign_lane = lane.index
        self.zone_start = self.placement_long
        self.zone_length = max(0.0, self._calculate_zone_length())
        self.zone_end = self.zone_start + self.zone_length

    def _calculate_zone_length(self) -> float:
        if self._is_sumo_network():
            return self.lane.length - self.placement_long
        return self._find_next_intersection_distance()

    def _is_violating(self, vehicle) -> bool:
        if not self._is_inside_zone(vehicle):
            return False
        return vehicle.speed_km_h > self.speed_limit

    def _is_inside_zone(self, vehicle) -> bool:
        veh_long = self.lane.local_coordinates(vehicle.position)[0]

        if self._is_sumo_network():
            is_same_lane = self._is_sumo_lane_on_sign_road(
                self.sign_lane,
                vehicle.lane_index,
                allowed_directions="same"
            )
        else:
            vehicle_lane_signature = self._lane_direction_signature(vehicle.lane_index)
            is_same_lane = not self._is_opposite_signature(
                self._lane_direction_signature(self.sign_lane),
                vehicle_lane_signature
            )
        return self.zone_start <= veh_long <= self.zone_end and is_same_lane

    def _create_visual_model(self):
        pass

    def get_rule_description(self) -> str:
        return f"Exceeding the speed limit ({self.speed_limit} km/h)"

    @property
    def top_down_color(self):
        return [255, 255, 255]

    @property
    def top_down_color_name(self):
        return "blue"
    
    def update_zones(self):
        """Update zone boundaries based on other signs on the route."""
        self._terminate_previous_speed_limits()

    def _terminate_previous_speed_limits(self):
        """Terminate earlier speed limit zones at this sign's position."""
        for sign in self.engine.traffic_sign_manager.get_signs_before(self, SpeedLimitSign):
            sign.zone_end = min(sign.zone_end, self.zone_start)
            sign.zone_length = max(0.0, sign.zone_end - sign.zone_start)


class SpeedLimitSign20(SpeedLimitSign):
    def __init__(self, lane, zone_length=None, **kwargs):
        super().__init__(lane, speed_limit=20, zone_length=zone_length, **kwargs)
        

class SpeedLimitSign30(SpeedLimitSign):
    def __init__(self, lane, zone_length=None, **kwargs):
        super().__init__(lane, speed_limit=30, zone_length=zone_length, **kwargs)        
        

class SpeedLimitSign40(SpeedLimitSign):
    def __init__(self, lane, zone_length=None, **kwargs):
        super().__init__(lane, speed_limit=40, zone_length=zone_length, **kwargs)
        

class SpeedLimitSign60(SpeedLimitSign):
    def __init__(self, lane, zone_length=None, **kwargs):
        super().__init__(lane, speed_limit=60, zone_length=zone_length, **kwargs)
