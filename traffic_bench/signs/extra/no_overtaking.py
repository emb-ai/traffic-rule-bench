import numpy as np

from traffic_bench.signs.base import BaseTrafficSign


_OPPOSITE_HEADING_THRESHOLD_RAD = np.pi / 2


class NoOvertakingSign(BaseTrafficSign):
    """(Overtaking prohibited)."""

    def __init__(
        self,
        lane,
        longitudinal_offset: float = 0.0,
        **kwargs
    ):
        base_longitudinal_offset = -lane.length + longitudinal_offset
        super().__init__(
            lane,
            longitudinal_offset=base_longitudinal_offset,
            icon_path="no_overtaking.png",
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

        if self._is_sumo_network():
            is_opposite_heading = self._is_heading_opposite_to_sign_lane(vehicle)
            sign_key = self._sumo_direction_key(self.sign_lane)
            vehicle_key = self._sumo_direction_key(vehicle.lane_index)
            return (not is_opposite_heading) and sign_key != vehicle_key
        else:
            current_lane_index, _ = self.engine.current_map.road_network.get_closest_lane_index(vehicle.position)
            current_signature = self._lane_direction_signature(current_lane_index)
            return self._is_opposite_signature(self._lane_direction_signature(self.sign_lane), current_signature)

    def _is_inside_zone(self, vehicle) -> bool:
        veh_long = self.lane.local_coordinates(vehicle.position)[0]

        if self._is_sumo_network():
            is_same_lane = self._is_sumo_lane_on_sign_road(
                self.sign_lane,
                vehicle.lane_index,
                allowed_directions="all"
            )
        else:
            vehicle_lane_signature = self._lane_direction_signature(vehicle.lane_index)
            is_same_lane = not self._is_opposite_signature(
                self._lane_direction_signature(self.sign_lane),
                vehicle_lane_signature
            )
        return self.zone_start <= veh_long <= self.zone_end and is_same_lane

    def _heading_diff_to_sign_lane(self, vehicle):
        """Absolute wrapped angle between the vehicle heading and the sign lane."""
        veh_long = self.lane.local_coordinates(vehicle.position)[0]
        veh_long = float(np.clip(veh_long, 0.0, self.lane.length))
        lane_heading = float(self.lane.heading_theta_at(veh_long))
        return abs(self._wrap_pi(vehicle.heading_theta - lane_heading))

    def _is_heading_opposite_to_sign_lane(self, vehicle) -> bool:
        return self._heading_diff_to_sign_lane(vehicle) > _OPPOSITE_HEADING_THRESHOLD_RAD

    def update_zones(self):
        """Shorten earlier no-overtaking zones so they end at this sign (same lane, upstream)."""
        self._terminate_previous_no_overtaking()

    def _terminate_previous_no_overtaking(self):
        for sign in self.engine.traffic_sign_manager.get_signs_before(self, NoOvertakingSign):
            sign.zone_end = min(sign.zone_end, self.zone_start)
            sign.zone_length = max(0.0, sign.zone_end - sign.zone_start)

    def _create_visual_model(self):
        pass

    def get_rule_description(self) -> str:
        return "Overtaking prohibited"

    @property
    def top_down_color(self):
        return [255, 200, 0]

    @property
    def top_down_color_name(self):
        return "yellow"

    @property
    def top_down_length(self):
        return 1.0

    @property
    def top_down_width(self):
        return 1.0


__all__ = ["NoOvertakingSign"]
