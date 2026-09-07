from traffic_bench.signs.base import BaseTrafficSign

class MinimumSpeedLimitSign(BaseTrafficSign):
    def __init__(self, lane, min_speed=40,
                 icon_path="min_speed.png", zone_length=None,
                 min_speed_override=None, **kwargs):

        # `min_speed_override` forces the enforced minimum (e.g. the catalog's
        # achievable-capped 20/40) so the verifier checks the same value the
        # acceleration scene was built for, instead of road_speed - 10.
        if min_speed_override is not None:
            self.min_speed = int(round(float(min_speed_override)))
        elif hasattr(lane, 'speed'):
            self.min_speed = round(lane.speed * 3.6) - 10
        else:
            self.min_speed = min_speed
        lane.min_speed_limit = self.min_speed
        
        super().__init__(lane, icon_path=icon_path or "min_speed.png", **kwargs)
        
        self.zone_start = self.placement_long + 10
        if zone_length is None:
            self.zone_length = self.lane.length - self.zone_start
        else:
            self.zone_length = max(0.0, float(zone_length))
        
        self.zone_end = min(self.lane.length, self.zone_start + self.zone_length)

    def _create_visual_model(self):
        pass

    # A minimum-speed violation is a sustained one. Measured on the v6 eval:
    # in the episodes flagged under 4.6 only 1-2% of the in-zone steps were
    # below the minimum, i.e. the flag came from a few-step dip at the zone
    # entry or behind a braking car, not from driving slowly. The plate holds
    # "when conditions permit" (PDD 10.1), so a car ahead that itself drives
    # below the minimum within braking distance exempts the ego.
    MIN_UNDERSPEED_STEPS = 10        # 1 s at 10 Hz
    LEADER_EXEMPT_MIN_M = 15.0       # look at least this far ahead
    LEADER_EXEMPT_DECEL = 4.0        # m/s^2, braking distance for the look-ahead

    def _is_violating(self, vehicle) -> bool:
        try:
            veh_long = self.lane.local_coordinates(vehicle.position)[0]
        except Exception:
            return False
        states = self.__dict__.setdefault("_under_floor", {})
        vid = getattr(vehicle, "id", id(vehicle))
        st = states.setdefault(vid, {"count": 0, "step": -1})
        under = (self.zone_start <= veh_long <= self.zone_end
                 and vehicle.speed_km_h < self.min_speed - 2.0
                 and not self._slower_leader_ahead(vehicle))
        try:
            step = int(getattr(self.engine, "episode_step", 0) or 0)
        except Exception:
            step = st["step"] + 1
        if not under:
            st["count"] = 0
            st["step"] = step
            return False
        if st["step"] != step:          # check_violation runs twice per step
            st["count"] += 1
            st["step"] = step
        return st["count"] >= self.MIN_UNDERSPEED_STEPS

    def _slower_leader_ahead(self, vehicle) -> bool:
        """A car (or static obstacle) on the ego lane within braking distance
        that itself moves below the minimum."""
        try:
            from metadrive.policy.idm_policy import FrontBackObjects

            lane = vehicle.lane
            if lane is None:
                return False
            objs = vehicle.lidar.get_surrounding_objects(vehicle)
            v_ms = max(0.0, float(vehicle.speed_km_h)) / 3.6
            look = max(self.LEADER_EXEMPT_MIN_M,
                       v_ms * v_ms / (2.0 * self.LEADER_EXEMPT_DECEL) + self.LEADER_EXEMPT_MIN_M)
            fb = FrontBackObjects.get_find_front_back_objs_single_lane(
                objs, lane, vehicle.position, max_distance=look)
            front = fb.front_object()
            if front is None:
                return False
            return float(getattr(front, "speed_km_h", 0.0) or 0.0) < float(self.min_speed)
        except Exception:
            return False

    def get_rule_description(self) -> str:
        return f"Speed below minimum limit ({self.min_speed} km/h)"

    @property
    def top_down_color(self):
        return [255, 255, 255]

    @property
    def top_down_color_name(self):
        return "cyan"

class MinimumSpeedLimit30(MinimumSpeedLimitSign):
    def __init__(self, lane, zone_length=None, **kwargs):
        super().__init__(lane, min_speed=30, icon_path="min_speed.png",
                         zone_length=zone_length, **kwargs)

class MinimumSpeedLimit40(MinimumSpeedLimitSign):
    def __init__(self, lane, zone_length=None, **kwargs):
        super().__init__(lane, min_speed=40, icon_path="min_speed.png",
                         zone_length=zone_length, **kwargs)

class MinimumSpeedLimit50(MinimumSpeedLimitSign):
    def __init__(self, lane, zone_length=None, **kwargs):
        super().__init__(lane, min_speed=50, icon_path="min_speed.png",
                         zone_length=zone_length, **kwargs)

class MinimumSpeedLimit60(MinimumSpeedLimitSign):
    def __init__(self, lane, zone_length=None, **kwargs):
        super().__init__(lane, min_speed=60, icon_path="min_speed.png",
                         zone_length=zone_length, **kwargs)