from pdd_bench.signs.base_traffic_sign import BaseTrafficSign
from metadrive.component.vehicle.vehicle_type import XLVehicle


class OnlyAutoSign(BaseTrafficSign):
    """Only passenger cars allowed on this lane; trucks are not allowed."""

    def __init__(self, lane, lane_index=None, **kwargs):
        super().__init__(
            lane, 
            icon_path="5.3.png", 
            **kwargs
        )
        self._lane_index = lane_index if lane_index is not None else getattr(lane, "index", None)
        self.sign_direction_signature = self._lane_direction_signature(self._lane_index)
        self.sign_lane = lane.index
        self.zone_start = self.placement_long
        self.zone_length = max(0.0, self._calculate_zone_length())
        self.zone_end = self.zone_start + self.zone_length

    def _calculate_zone_length(self) -> float:
        if self._is_sumo_network():
            return self.lane.length - self.placement_long
        return self._find_next_intersection_distance()

    def _create_visual_model(self):
        pass

    def _is_truck(self, vehicle) -> bool:
        """Is the vehicle a truck (XL) — it is not allowed to pass on the sign lane."""
        if isinstance(vehicle, XLVehicle):
            return True
        return False

    def _is_violating(self, vehicle) -> bool:
        """Only passenger cars (S, M, L, Default) are allowed. Trucks (XL) are not allowed."""
        if not self._is_inside_zone(vehicle):
            return False
        return self._is_truck(vehicle)

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

    def update_zones(self):
        """Shorten earlier only-auto zones so they end at this sign (same lane, upstream)."""
        self._terminate_previous_only_auto()

    def _terminate_previous_only_auto(self):
        for sign in self.engine.traffic_sign_manager.get_signs_before(self, OnlyAutoSign):
            sign.zone_end = min(sign.zone_end, self.zone_start)
            sign.zone_length = max(0.0, sign.zone_end - sign.zone_start)

    def get_rule_description(self) -> str:
        return "Only passenger cars allowed on this lane; trucks are not allowed"
