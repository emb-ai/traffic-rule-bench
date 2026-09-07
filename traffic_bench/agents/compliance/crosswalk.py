import logging

import numpy as np

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

logger = logging.getLogger(__name__)


class CrosswalkCompliance:
        def _process_rules(self):
            """Process non-sign rules such as PedestrianYieldRule."""
            # Cleared each step; set again if an occupied zebra demands a stop.
            self._pedestrian_yield_hold_steer = False
            engine = getattr(self, "engine", None)
            if engine is None or not hasattr(engine, "traffic_sign_manager"):
                return
            for rule in engine.traffic_sign_manager.rules:
                try:
                    self._handle_pedestrian_yield(rule)
                except Exception as exc:
                    logger.debug("Error processing rule %s: %s", type(rule).__name__, exc)

        def _handle_pedestrian_yield(self, rule):
            """Brake to a stop ``stop_before`` metres before an occupied zebra.

            ``stop_before`` is ``max(yield_distance, no_stop_before_m)`` so the
            rest point sits at/outside the no-stop band (default 3 m). Distance
            is measured along the current lane to the zebra polygon center when
            possible (mid-block and lane-end injects); falls back to the rule's
            heading-cone helper otherwise.

            Also sets ``_pedestrian_yield_hold_steer`` so NN policies (CaRL/PPO/
            PlanT2) keep lane-center steering instead of swerving around peds.
            """
            if not hasattr(rule, "should_vehicle_stop"):
                return
            ego = self.control_object
            try:
                engine = getattr(self, "engine", None) or getattr(ego, "engine", None)
                thr = rule._resolve_all_thresholds(engine)
                yield_d = float(thr["yield_distance"])
                no_stop_m = float(thr["no_stop_before_m"])
            except Exception:
                yield_d = float(
                    getattr(rule, "_defaults", {}).get("yield_distance", 12.0)
                )
                no_stop_m = float(
                    getattr(rule, "_defaults", {}).get("no_stop_before_m", 3.0)
                )
            # Rest outside/at the no-stop boundary so we do not creep into the
            # 3 m band and then eat an occupied-crosswalk tip-in.
            stop_before = max(yield_d, no_stop_m)

            along = self._along_distance_to_occupied_crosswalk(rule)
            must_stop = False
            if along is not None:
                dist_to_stop = float(along) - stop_before
                approach = max(float(self._approach_dist(0.0)), 2.0)
                # Brake from braking-distance away; hold once at/past the line
                # (including slightly past — still before the zebra paint).
                if (0.0 < dist_to_stop <= approach) or (0.0 < float(along) <= stop_before):
                    must_stop = True
                elif float(along) <= 0.5:
                    # Nose already at/on the zebra while it is occupied.
                    must_stop = True
            elif rule.should_vehicle_stop(ego):
                must_stop = True

            if must_stop:
                self._cap_speed(0.001)
                self._pedestrian_yield_hold_steer = True

        def _along_distance_to_occupied_crosswalk(self, rule):
            """Metres along the current lane from ego to an occupied zebra ahead.

            Uses the zebra polygon center projected onto the ego lane so mid-block
            (no-split) and lane-end injects both work. Returns None when no
            occupied crosswalk lies ahead on this lane.
            """
            get_state = getattr(rule, "_get_crosswalk_state", None)
            if get_state is None:
                return None
            ego = self.control_object
            _engine, state = get_state(ego)
            if not state:
                return None
            lane = getattr(ego, "lane", None)
            if lane is None:
                return None
            try:
                ego_long, _ = lane.local_coordinates(ego.position)
                ego_long = float(ego_long)
                lane_len = float(getattr(lane, "length", 0.0) or 0.0)
            except Exception:
                return None

            best_along = None
            for st in state.values():
                if not bool(st.get("active", False)):
                    continue
                poly = np.asarray(st.get("polygon", []), dtype=np.float64)
                if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] < 2:
                    continue
                try:
                    center = np.mean(poly[:, :2], axis=0)
                    cw_long, cw_lat = lane.local_coordinates(center)
                    cw_long = float(cw_long)
                    # Must sit on / near this lane corridor (not a parallel road).
                    half_w = float(lane.width_at(cw_long)) * 0.5 + 4.0
                    if abs(float(cw_lat)) > half_w:
                        continue
                    along = cw_long - ego_long
                    if along <= -1.0:
                        # Clearly behind.
                        continue
                    # Clamp to a sane horizon; ignore far-away zebras.
                    if along > max(lane_len, 80.0):
                        continue
                    if best_along is None or along < best_along:
                        best_along = along
                except Exception:
                    continue
            return best_along
