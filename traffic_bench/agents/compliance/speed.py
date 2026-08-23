from traffic_bench.agents.compliance.kinematics import (
    BRAKE_BIAS,
    BRAKE_PROP_GAIN,
    BRAKING_MARGIN,
    COMFORT_DECEL,
    END_MAIN_ROAD_LOOKAHEAD,
    FALLBACK_FACTOR,
    FALLBACK_MIN_KMH,
    FLOOR_BIAS,
    FLOOR_OVERSHOOT_KMH,
    FLOOR_PROP_GAIN,
    LANE_CHANGE_LOOKAHEAD,
    LC_COMPLETE_LAT,
    SLOW_APPROACH_FACTOR,
    SLOW_APPROACH_MIN_KMH,
    SPEED_SIGN_LOOKAHEAD,
    STOP_ENGAGE_DISTANCE_M,
    STOP_PAST_THRESHOLD,
    UTURN_ZONE_CENTER_REMAINING_M,
    UTURN_ZONE_CREEP_KMH,
    UTURN_ZONE_DESIRED_LAT_M,
    UTURN_ZONE_FORCE_NAV_REMAINING_M,
    UTURN_ZONE_HOLD_STEPS,
    UTURN_ZONE_LOOKAHEAD_M,
    UTURN_ZONE_MAX_STEER,
    UTURN_ZONE_MAX_STEERING_DEG,
    UTURN_ZONE_MIDROAD_TOL_M,
    UTURN_ZONE_MIN_KMH,
    UTURN_ZONE_SOFT_STEER,
    UTURN_ZONE_SPEED_CAP_KMH,
    UTURN_ZONE_SPIN_ALIGN_RAD,
    UTURN_ZONE_SPIN_HOLD_STEP_M,
    UTURN_ZONE_SPIN_RAD_PER_STEP,
    UTURN_ZONE_SPIN_REMAINING_M,
    accel_distance,
    braking_distance,
    lane_index_num,
    on_same_road,
    same_lane,
)


class SpeedCompliance:
        def _handle_speed_limit(self, sign):
            limit = float(sign.speed_limit)
            # In-zone check via the SIGN's own is_vehicle_in_zone (multi-edge aware,
            # exactly what the verifier uses). The previous lane-local check
            # (zone_start<=veh_long<=zone_end on sign.lane + on_same_road) lost the
            # limit when the ego crossed into another edge of a multi-edge zone
            # (5.21/5.31), so the ego sped up mid-zone. Asking the sign keeps the cap
            # applied across the WHOLE zone, matching where violations are measured.
            if sign.is_vehicle_in_zone(self.control_object):
                self._cap_speed(limit)
                return
            # Approach phase (before entering the zone): brake early within lookahead.
            if not on_same_road(self.control_object.lane, sign.lane):
                if not self._is_sign_on_route(sign):
                    return
            veh_long = self._veh_long(sign.lane)
            if veh_long < sign.zone_start:
                approach = max(self._approach_dist(limit), SPEED_SIGN_LOOKAHEAD)
                if 0 < (sign.zone_start - veh_long) < approach:
                    self._cap_speed(limit)

        def _handle_min_speed(self, sign):
            if not on_same_road(self.control_object.lane, sign.lane):
                if not self._is_sign_on_route(sign):
                    return
            min_spd = float(sign.min_speed)
            veh_long = self._veh_long(sign.lane)
            if sign.zone_start <= veh_long <= sign.zone_end:
                self._raise_floor(min_spd)
            elif veh_long < sign.zone_start:
                approach = max(
                    accel_distance(self.control_object.speed_km_h, min_spd),
                    SPEED_SIGN_LOOKAHEAD,
                )
                if 0 < (sign.zone_start - veh_long) < approach:
                    self._raise_floor(min_spd)
