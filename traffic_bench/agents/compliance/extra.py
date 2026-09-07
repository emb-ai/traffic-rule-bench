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

logger = logging.getLogger(__name__)


class ExtraCompliance:
        def _handle_no_stopping(self, sign):
            if not on_same_road(self.control_object.lane, sign.lane):
                return
            veh_long = self._veh_long(sign.lane)
            if sign.zone_start <= veh_long <= sign.zone_end:
                self._raise_floor(self.NO_STOP_MIN_SPEED_KMH)

        def _handle_bus_station(self, sign):
            if getattr(self.control_object, "is_bus", False):
                return
            if not on_same_road(self.control_object.lane, sign.lane):
                return
            veh_long = self._veh_long(sign.lane)
            if sign.zone_start <= veh_long <= sign.zone_end:
                self._raise_floor(self.NO_STOP_MIN_SPEED_KMH)

        def _handle_traffic_light(self, sign):
            # Cross-edge pre-brake: if ego is approaching sign.lane from an
            # upstream edge and any signal is currently red, start braking so
            # we don't enter the junction. Conservative — it brakes even when
            # the exact direction is green, but only when ego hasn't yet
            # entered the approach zone.
            if not same_lane(self.control_object.lane, sign.lane):
                states = getattr(sign, "current_states", None) or {}
                if states and any(s in ("r", "R", "y", "Y") for s in states.values()):
                    self._cross_edge_brake_for(sign, stop_long=sign.lane.length)
                return
            veh_long = self._veh_long(sign.lane)
            if not (sign.zone_start <= veh_long <= sign.zone_end):
                return
            if not sign.current_states:
                return
            # Determine which to_lane the ego is actually heading towards
            # by matching the sign's signals against the navigation route
            nav = getattr(self.control_object, "navigation", None)
            route_checkpoints = set()
            if nav is not None:
                for ckpt in (getattr(nav, "checkpoints", None) or []):
                    route_checkpoints.add(ckpt)
            # Collect to_lanes reachable from the ego's current lane
            ego_to_lanes = set()
            for turn in getattr(self.control_object.lane, "turns", []):
                to = turn.get("to_lane")
                if to:
                    ego_to_lanes.add(to)
            # Priority 1: check signal for the direction that is BOTH
            # reachable from ego lane AND on the navigation route
            route_directed = [
                state for to_lane, state in sign.current_states.items()
                if to_lane in ego_to_lanes and to_lane in route_checkpoints
            ]
            if route_directed:
                is_green = any(s in ("g", "G") for s in route_directed)
            elif ego_to_lanes:
                # Priority 2: ego lane has turns but none matched route —
                # check all ego-reachable directions
                relevant = [
                    state for to_lane, state in sign.current_states.items()
                    if to_lane in ego_to_lanes
                ]
                if not relevant:
                    return
                is_green = any(s in ("g", "G") for s in relevant)
            else:
                # No turn info (e.g. ego is on a junction lane) — skip
                return
            if not is_green:
                dist_to_end = sign.lane.length - veh_long
                if dist_to_end < self._approach_dist(0.0):
                    self._cap_speed(0.001)

        def _restricted_preempt_m(self) -> float:
            """Per-episode pre-empt distance before a reserved-lane zone."""
            if getattr(self, "_restricted_preempt_cache", None) is None:
                rng = getattr(self.engine, "np_random", None)
                rng_range = getattr(self, "PREEMPT_RESTRICTED_LANE_RANGE_M", None)
                if rng_range is None or rng is None:
                    self._restricted_preempt_cache = float(self.PREEMPT_RESTRICTED_LANE_M)
                else:
                    lo, hi = rng_range
                    self._restricted_preempt_cache = float(rng.uniform(lo, hi))
            return self._restricted_preempt_cache

        def _handle_restricted_lane(self, sign):
            sign_idx = getattr(sign.lane, "index", None)
            if sign_idx is not None:
                self._restricted_lanes.add(sign_idx)
            if not on_same_road(self.control_object.lane, sign.lane):
                return
            sign_ln = lane_index_num(sign.lane)
            cur = self._cur_lane_num()
            if cur is None or sign_ln is None or cur != sign_ln:
                if self._lc_target_lane is None:
                    self._lc_steer_limit = None   # merge done: full steering again
                return
            veh_long = self._veh_long(sign.lane)
            in_zone = sign.zone_start <= veh_long <= sign.zone_end
            # On the reserved lane ahead of its zone the ego stays slow for the
            # whole run-up: the lateral controller is stable for a lane change
            # at 18-25 km/h and ran the ego off the road when one began at 36.
            # The sampled pre-empt distance below only decides WHEN the change
            # starts; the speed is already right when it does.
            if veh_long < sign.zone_start and (sign.zone_start - veh_long) < self.RESTRICTED_APPROACH_M:
                self._cap_speed(self.RESTRICTED_LC_KMH)
            # Preemptive: start lane change ~50 m before the restricted zone so
            # NN policies (CaRL/PlanT2) don't enter the bus/bike lane and trigger
            # a violation. Reactive case (already in zone) keeps the same logic.
            approaching = (veh_long < sign.zone_start
                           and (sign.zone_start - veh_long) < self._restricted_preempt_m())
            if in_zone or approaching:
                safe = self._find_safe_lane_num()
                if safe is not None:
                    target = self._ref_lanes_by_num().get(safe)
                    # The run-up is 60 m: wait for a real gap instead of cutting in.
                    # A position-only 20 m check still put the ego in front of a
                    # 50 km/h car closing from 25 m; the time-to-close rule below
                    # refuses that. The approach cap stays at the slow-approach value:
                    # the lateral controller over-steers a merge at 30 km/h.
                    to_zone = float(sign.zone_start) - float(veh_long)
                    v_ms = max(0.0, float(self.control_object.speed_km_h)) / 3.6
                    # Straight after spawn the ego is still slow: merging at 5 m/s
                    # into 40 km/h traffic is what most expert crashes were.
                    # Reach traffic speed first unless the zone is close.
                    too_slow = v_ms < 5.5 and to_zone > 25.0
                    if (target is not None and self._lc_target_lane is None
                            and (too_slow or not self._merge_gap_ok(target))):
                        if to_zone < 8.0:
                            self._cap_speed(0.0)      # never roll into the reserved zone
                        elif to_zone < 20.0:
                            self._cap_speed(8.0)
                        elif not too_slow:
                            self._cap_speed(max(SLOW_APPROACH_MIN_KMH,
                                                self.control_object.speed_km_h * SLOW_APPROACH_FACTOR))
                        return
                    self._lc_steer_limit = self.RESTRICTED_LC_STEER
                    self._begin_lane_change(safe)
                    self._cap_speed(max(SLOW_APPROACH_MIN_KMH,
                                        self.control_object.speed_km_h * SLOW_APPROACH_FACTOR))

        def _merge_gap_ok(self, target_lane, ahead=15.0, behind=20.0, t_close=3.5, look_behind=80.0):
            """Gap in ``target_lane`` for a merge: nothing within [-behind, ahead] m
            of the ego's projection, and no car further back that would close the
            gap in under ``t_close`` seconds at the current speed difference."""
            try:
                ego = self.control_object
                ego_long, _ = target_lane.local_coordinates(ego.position)
                tm = getattr(self.engine, "traffic_manager", None)
                vehicles = list(getattr(tm, "traffic_vehicles", None) or [])
            except Exception:
                return False
            half_w = target_lane.width_at(0) / 2 + 0.3
            v_ego = max(0.0, float(getattr(ego, "speed_km_h", 0.0))) / 3.6
            for v in vehicles:
                try:
                    v_long, v_lat = target_lane.local_coordinates(v.position)
                except Exception:
                    continue
                if abs(v_lat) > half_w:
                    continue
                gap = float(v_long) - float(ego_long)
                if -behind < gap < ahead:
                    return False
                if -look_behind < gap <= -behind:
                    v_other = max(0.0, float(getattr(v, "speed_km_h", 0.0))) / 3.6
                    closing = v_other - v_ego
                    if closing > 0.1 and (-gap) / closing < t_close:
                        return False
            return True

        def _handle_intersection_restricted_lane(self, sign):
            if hasattr(sign, "is_valid_placement") and not sign.is_valid_placement:
                return
            for lane_idx in sign.forbidden_to_lanes:
                self._blocked_lanes.add(lane_idx)

        def _handle_only_auto(self, sign):
            try:
                if not sign._is_truck(self.control_object):
                    return
            except Exception:
                logger.debug("OnlyAutoSign._is_truck() failed for %s", self.control_object.name)
                return
            sign_idx = getattr(sign.lane, "index", None)
            if sign_idx is not None:
                self._restricted_lanes.add(sign_idx)
            if not on_same_road(self.control_object.lane, sign.lane):
                return
            sign_ln = lane_index_num(sign.lane)
            cur = self._cur_lane_num()
            if cur is None or sign_ln is None or cur != sign_ln:
                return
            # Only enforce within the sign's zone
            veh_long = self._veh_long(sign.lane)
            if not (sign.zone_start <= veh_long <= sign.zone_end):
                return
            safe = self._find_safe_lane_num()
            if safe is not None:
                self._begin_lane_change(safe)
                self._cap_speed(max(SLOW_APPROACH_MIN_KMH,
                                    self.control_object.speed_km_h * SLOW_APPROACH_FACTOR))

        def _handle_right_turn_rule(self, sign):
            try:
                status = sign.get_status(self.control_object)
            except Exception:
                logger.debug("RightTurnRule.get_status() failed")
                return
            if status.get("is_planning_right_turn", False) and not status.get(
                "is_rightmost_lane", True
            ):
                ref = self._get_ref_lanes()
                if ref:
                    self._begin_lane_change(len(ref) - 1)

        def _handle_no_overtaking(self, sign):
            """Handle NoOvertakingSign — block the opposite lane and flag."""
            opposite = getattr(sign, "opposite_lane", None)
            if opposite is not None:
                self._blocked_lanes.add(opposite)
            if on_same_road(self.control_object.lane, sign.lane):
                self._no_overtaking_active = True
