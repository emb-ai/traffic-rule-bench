from traffic_signs.base_traffic_sign import BaseTrafficSign


class BusStationSign(BaseTrafficSign):
    """Bus stop only: non-buses must not *stop* in the marked zone; driving through is allowed.

    Violation = low speed (standing) in zone for at least min_zero_speed_duration. Not an entry ban.
    """

    def __init__(self, lane, zone_length=5.0, min_zero_speed_duration=0.01, zone_start=None, zone_end=None, **kwargs):
        kwargs.pop("zone_start", None)
        kwargs.pop("zone_end", None)
        super().__init__(lane, icon_path="bus_station.png", **kwargs)
        self.zone_length = float(zone_length)
        self.min_zero_speed_duration = float(min_zero_speed_duration)
        if zone_start is not None and zone_end is not None:
            self.zone_start = float(zone_start)
            self.zone_end = float(zone_end)
        else:
            # Bus bay often extends both before and after the pole along the lane; a
            # forward-only strip missed vehicles stopped slightly upstream of the sign.
            half = self.zone_length
            self.zone_start = max(0.0, self.placement_long - half)
            self.zone_end = min(self.lane.length, self.placement_long + half)
        self._zero_speed_accumulated = {}  # vehicle_id -> accumulated time with speed ~0 in zone

    def _create_visual_model(self):
        pass

    def _get_dt(self, vehicle):
        """Simulation time step in seconds."""
        engine = getattr(vehicle, "engine", None)
        if engine is not None:
            cfg = getattr(engine, "global_config", {}) or {}
            return float(
                cfg.get("physics_world_step_size", cfg.get("physics_dt", 0.02))
            )
        return 0.02

    def _is_bus(self, vehicle) -> bool:
        """Is the vehicle a bus (allowed to stop in the zone)."""
        if getattr(vehicle, "is_bus", False):
            return True
        config = getattr(vehicle, "config", None)
        if config is not None and config.get("is_bus", False):
            return True
        return False

    def _is_violating(self, vehicle) -> bool:
        if not self.is_in_drivable_area(vehicle):
            return False
        if self._is_bus(vehicle):
            return False
        # Violation only on the sign lane (like OnlyAutoSign).
        sign_lane_index = self._get_lane_index(self.lane)
        vehicle_lane_index = self._get_lane_index(vehicle)
        if sign_lane_index is not None and vehicle_lane_index is not None and sign_lane_index != vehicle_lane_index:
            return False
        try:
            veh_long = self.lane.local_coordinates(vehicle.position)[0]
        except Exception:
            return False
        if not (self.zone_start <= veh_long <= self.zone_end):
            self._zero_speed_accumulated.pop(getattr(vehicle, "id", id(vehicle)), None)
            return False
        vid = getattr(vehicle, "id", id(vehicle))
        speed = getattr(vehicle, "speed", 0.0)
        # MetaDrive can report small non-zero speed when visually stopped; 0.1 m/s was too strict.
        speed_threshold = 0.5
        dt = self._get_dt(vehicle)
        if speed < speed_threshold:
            acc = self._zero_speed_accumulated.get(vid, 0.0) + dt
            self._zero_speed_accumulated[vid] = acc
            if acc >= self.min_zero_speed_duration:
                return True
        else:
            self._zero_speed_accumulated.pop(vid, None)
        return False

    def get_rule_description(self) -> str:
        return (
            f"Bus stop only: non-bus vehicles must not stand still "
            f"for {self.min_zero_speed_duration}s in the {self.zone_length}m along-lane band "
            f"(~{self.zone_length}m before and after the sign); driving through is allowed"
        )
