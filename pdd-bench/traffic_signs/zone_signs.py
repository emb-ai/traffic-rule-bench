from traffic_signs.base_traffic_sign import BaseTrafficSign
from traffic_signs.speed_limit_sign import SpeedLimitSign


class ZoneSpeedLimitSign(BaseTrafficSign):
    """
    Zone speed limit sign.

    Zone speed limit applies to the entire road in both directions and remains in effect
    until it is explicitly ended by an end-of-zone sign.
    """

    def __init__(
        self, 
        lane, 
        speed_limit: float = 20,
        longitudinal_offset: float = 0.0,
        **kwargs
    ):
        if hasattr(lane, 'speed') and lane.speed is not None:
            self.speed_limit = round(lane.speed * 3.6)
        else:
            self.speed_limit = speed_limit
        lane.speed_limit = self.speed_limit

        super().__init__(
            lane,
            longitudinal_offset=longitudinal_offset,
            longitudinal_from_start=True,
            icon_path=f'5.31_{int(self.speed_limit)}.png',
            **kwargs
        )
        self.zone_start = self.placement_long
        self.zone_length = float('inf')
        self.zone_end = float('inf')
        self._is_active = True
        # Multi-edge zone (combined SUMO pairs). None = single-lane.
        self.zone_edges = None
        self.zone_end_s = None

    def configure_multi_edge_zone(self, zone_edges, zone_start, zone_end_s):
        """Configure a zone spanning connected edges (see SpeedLimitSign)."""
        self.zone_edges = list(zone_edges)
        self.zone_start = float(zone_start)
        self.zone_end_s = float(zone_end_s)

    def _create_visual_model(self):
        pass

    def is_vehicle_in_zone(self, vehicle) -> bool:
        multi = self._in_multi_edge_zone(vehicle)
        if multi is not None:
            return multi
        veh_long = self.lane.local_coordinates(vehicle.position)[0]
        return self.zone_start <= veh_long <= self.zone_end

    def _is_violating(self, vehicle) -> bool:
        return self.is_vehicle_in_zone(vehicle) and vehicle.speed_km_h > self.speed_limit

    def get_rule_description(self) -> str:
        return f"Exceeding the zone speed limit ({self.speed_limit} km/h)"

    @property
    def top_down_color(self):
        return [255, 200, 0]

    @property
    def top_down_color_name(self):
        return "orange"
    
    def update_zones(self):
        self._terminate_previous_zone_speed_limits()

    def _terminate_previous_zone_speed_limits(self):
        from traffic_signs.zone_signs import ZoneSpeedLimitSign
        for sign in self.engine.traffic_sign_manager.get_signs_before(self, (ZoneSpeedLimitSign, SpeedLimitSign)):
            sign.zone_end = min(sign.zone_end, self.zone_start)
            sign.zone_length = sign.zone_end - sign.zone_start


class ZoneSpeedLimitSign20(ZoneSpeedLimitSign):
    def __init__(self, lane, **kwargs):
        super().__init__(lane, speed_limit=20, **kwargs)


class ZoneSpeedLimitSign30(ZoneSpeedLimitSign):
    def __init__(self, lane, **kwargs):
        super().__init__(lane, speed_limit=30, **kwargs)


class ZoneSpeedLimitSign40(ZoneSpeedLimitSign):
    def __init__(self, lane, **kwargs):
        super().__init__(lane, speed_limit=40, **kwargs)


class ZoneSpeedLimitSign60(ZoneSpeedLimitSign):
    def __init__(self, lane, **kwargs):
        super().__init__(lane, speed_limit=60, **kwargs)

