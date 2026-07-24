from traffic_signs.base_traffic_sign import BaseTrafficSign
from typing import Optional


class SpeedLimitSign(BaseTrafficSign):
    def __init__(
        self,
        lane,
        speed_limit: float = 20,
        zone_length: Optional[float] = None,
        longitudinal_offset: float = 0.0,
        speed_limit_override: Optional[float] = None,
        **kwargs
    ):
        # `speed_limit_override` forces the enforced limit (e.g. the bucketed
        # 20/40/60 value for 3.24 scenes), bypassing the lane's raw OSM speed.
        if speed_limit_override is not None:
            self.speed_limit = int(round(float(speed_limit_override)))
        elif hasattr(lane, 'speed') and lane.speed is not None:
            self.speed_limit = round(lane.speed * 3.6)
        else:
            self.speed_limit = speed_limit
        lane.speed_limit = self.speed_limit

        super().__init__(
            lane,
            longitudinal_offset=longitudinal_offset,
            longitudinal_from_start=True,
            icon_path=f"3.24_{self.speed_limit:02d}.png",
            **kwargs
        )
        self.sign_lane = lane.index
        self.zone_start = self.placement_long
        self.zone_length = max(0.0, self._calculate_zone_length())
        self.zone_end = self.zone_start + self.zone_length
        # Multi-edge zone (set by configure_multi_edge_zone for combined SUMO
        # pairs whose zone spans several connected edges). None = single-lane.
        self.zone_edges = None
        self.zone_end_s = None

    def configure_multi_edge_zone(self, zone_edges, zone_start, zone_end_s):
        """Configure a zone that spans connected edges start_edge..end_edge.

        `zone_edges` is the ordered list of directed edge ids, `zone_start` the
        offset on the first edge, `zone_end_s` the offset on the last edge.
        """
        self.zone_edges = list(zone_edges)
        self.zone_start = float(zone_start)
        self.zone_end_s = float(zone_end_s)

    def _calculate_zone_length(self) -> float:
        if self._is_sumo_network():
            return self.lane.length - self.placement_long
        return self._find_next_intersection_distance()

    def _is_violating(self, vehicle) -> bool:
        if not self._is_inside_zone(vehicle):
            return False
        return vehicle.speed_km_h > self.speed_limit

    def is_vehicle_in_zone(self, vehicle) -> bool:
        """Public zone-membership check used by the verifier (manager)."""
        return self._is_inside_zone(vehicle)

    def _is_inside_zone(self, vehicle) -> bool:
        # Multi-edge zone (combined pairs): directed edge ids already encode
        # direction, so this subsumes the same-lane/direction check.
        multi = self._in_multi_edge_zone(vehicle)
        if multi is not None:
            return multi

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
