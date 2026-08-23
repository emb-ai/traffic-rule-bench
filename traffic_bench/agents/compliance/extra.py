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

        def _handle_restricted_lane(self, sign):
            sign_idx = getattr(sign.lane, "index", None)
            if sign_idx is not None:
                self._restricted_lanes.add(sign_idx)
            if not on_same_road(self.control_object.lane, sign.lane):
                return
            sign_ln = lane_index_num(sign.lane)
            cur = self._cur_lane_num()
            if cur is None or sign_ln is None or cur != sign_ln:
                return
            veh_long = self._veh_long(sign.lane)
            in_zone = sign.zone_start <= veh_long <= sign.zone_end
            # Preemptive: start lane change ~50 m before the restricted zone so
            # NN policies (CaRL/PlanT2) don't enter the bus/bike lane and trigger
            # a violation. Reactive case (already in zone) keeps the same logic.
            approaching = (veh_long < sign.zone_start
                           and (sign.zone_start - veh_long) < self.PREEMPT_RESTRICTED_LANE_M)
            if in_zone or approaching:
                safe = self._find_safe_lane_num()
                if safe is not None:
                    self._begin_lane_change(safe)
                    self._cap_speed(max(SLOW_APPROACH_MIN_KMH,
                                        self.control_object.speed_km_h * SLOW_APPROACH_FACTOR))

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
