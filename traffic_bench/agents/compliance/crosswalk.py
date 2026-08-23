import logging

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

from traffic_bench.signs.crosswalk.yield_rule import PedestrianYieldRule

logger = logging.getLogger(__name__)


class CrosswalkCompliance:
        def _process_rules(self):
            """Process non-sign rules such as PedestrianYieldRule."""
            engine = getattr(self, "engine", None)
            if engine is None or not hasattr(engine, "traffic_sign_manager"):
                return
            for rule in engine.traffic_sign_manager.rules:
                try:
                    self._handle_pedestrian_yield(rule)
                except Exception as exc:
                    logger.debug("Error processing rule %s: %s", type(rule).__name__, exc)

        def _handle_pedestrian_yield(self, rule):
            """Stop ``yield_distance`` metres before an occupied crosswalk.

            ``PedestrianYieldRule.should_vehicle_stop`` uses Euclidean distance
            plus a heading cone. Around a bend the zebra is not "ahead" until
            the last few metres, so this handler never fired and IDM then
            braked for the pedestrian at the painted stop line
            (``no_stop_before_crosswalk_m``, ~3 m). Changing
            ``pedestrian.yield_distance`` therefore had no effect.

            Injected 5.19 zebras sit at the approach lane end — treat
            ``lane.length - s`` as the stop geometry so the config actually
            controls the rest point.
            """
            if not hasattr(rule, "should_vehicle_stop"):
                return
            along = self._along_distance_to_occupied_crosswalk(rule)
            if along is None:
                if rule.should_vehicle_stop(self.control_object):
                    self._cap_speed(0.001)
                return
            try:
                engine = getattr(self, "engine", None) or getattr(
                    self.control_object, "engine", None
                )
                yield_d = float(rule._resolve_all_thresholds(engine)["yield_distance"])
            except Exception:
                yield_d = float(
                    getattr(rule, "_defaults", {}).get("yield_distance", 12.0)
                )
            # Virtual stop line is ``yield_distance`` before the zebra. Brake
            # from braking-distance away; keep the cap once at/past that line
            # so we do not roll on to the painted 3 m mark.
            dist_to_stop = float(along) - yield_d
            approach = max(float(self._approach_dist(0.0)), 2.0)
            if (0.0 < dist_to_stop <= approach) or (0.0 < float(along) <= yield_d):
                self._cap_speed(0.001)

        def _along_distance_to_occupied_crosswalk(self, rule):
            """Metres remaining along the current lane to an occupied zebra at lane end.

            Returns None when no occupied crosswalk sits at this lane's end
            (caller falls back to heading-based ``should_vehicle_stop``).
            """
            get_state = getattr(rule, "_get_crosswalk_state", None)
            dist_fn = getattr(rule, "_distance_to_polygon", None)
            if get_state is None or dist_fn is None:
                return None
            ego = self.control_object
            _engine, state = get_state(ego)
            if not state:
                return None
            lane = getattr(ego, "lane", None)
            if lane is None:
                return None
            try:
                long, _ = lane.local_coordinates(ego.position)
                along = float(lane.length) - float(long)
                end_xy = np.asarray(
                    lane.position(max(0.5, float(lane.length) - 0.5), 0.0)[:2],
                    dtype=np.float64,
                )
            except Exception:
                return None
            if along <= 0.0:
                return None
            for st in state.values():
                if not bool(st.get("active", False)):
                    continue
                poly = np.asarray(st.get("polygon", []), dtype=np.float64)
                try:
                    if float(dist_fn(end_xy, poly)) > 8.0:
                        continue
                except Exception:
                    continue
                return along
            return None
