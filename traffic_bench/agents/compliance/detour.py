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

import numpy as np

class DetourCompliance:
        def _detour_preempt_m(self) -> float:
            """Per-episode preempt distance, cached after the first draw."""
            if getattr(self, "_detour_preempt_cache", None) is None:
                rng = getattr(self.engine, "np_random", None)
                if self.PREEMPT_DETOUR_RANGE_M is None or rng is None:
                    self._detour_preempt_cache = float(self.PREEMPT_DETOUR_M)
                else:
                    lo, hi = self.PREEMPT_DETOUR_RANGE_M
                    self._detour_preempt_cache = float(rng.uniform(lo, hi))
            return self._detour_preempt_cache

        def _handle_detour(self, sign):
            self._blocked_lanes.add(getattr(sign.lane, "index", None))
            if not on_same_road(self.control_object.lane, sign.lane):
                return
            sign_ln = lane_index_num(sign.lane)
            cur = self._cur_lane_num()
            if cur is None or sign_ln is None:
                return
            veh_long = self._veh_long(sign.lane)
            if cur != sign_ln:
                # Ego is on the detour lane. Once the cone cluster is fully
                # cleared, merge back into the original (route) lane — completes
                # the manoeuvre and restores route-lane arrival checks.
                on_detour_lane = (
                    getattr(self.control_object.lane, "index", None)
                    in (getattr(sign, "_allowed_lane_indices", None) or set()))
                cleared = veh_long > sign.obstacle_long + self.DETOUR_RETURN_CLEARANCE_M
                if on_detour_lane and cleared and self._lc_target_lane is None \
                        and self._detour_gap_ok(sign.lane):
                    self._lc_target_lane = sign.lane
                    self._get_heading_pid().reset()
                    self._get_lateral_pid().reset()
                return
            in_zone = sign.zone_start <= veh_long <= sign.zone_end
            approaching = (veh_long < sign.zone_start
                           and sign.zone_start - veh_long < self._detour_preempt_m())
            if not (in_zone or approaching):
                # Queue at the cones: NPCs on the obstacle lane pile up behind the
                # cluster. Merge BEFORE reaching the queue tail instead of joining
                # it and crawling behind a stopped leader.
                if not (veh_long < sign.zone_start
                        and self._detour_queue_ahead(sign, veh_long)):
                    return
            violation_long = getattr(
                sign, "violation_long",
                sign.obstacle_long + getattr(sign, "OBSTACLE_OFFSET", 0.0),
            )
            if in_zone and violation_long - veh_long <= 0:
                return
            target = self._detour_target_lane(sign)
            if not (target is not None and self._detour_gap_ok(target)):
                # Target gap blocked (or no target): bleed speed off while
                # waiting. With a free gap the ego keeps momentum and merges at
                # speed instead of crawling up to the plate.
                self._cap_speed(max(self.DETOUR_APPROACH_MIN_KMH,
                                    self.control_object.speed_km_h
                                    * self.DETOUR_APPROACH_FACTOR))
            if target is not None:
                if self._lc_target_lane is None and not same_lane(
                    self.control_object.lane, target
                ):
                    self._lc_target_lane = target
                    self._get_heading_pid().reset()
                    self._get_lateral_pid().reset()
                # Keep momentum while merging: base IDM brakes for the queued
                # leader on the lane being vacated. With a clear gap in the target
                # lane, floor the speed so the merge completes instead of stalling.
                if self._lc_target_lane is not None \
                        and self._detour_gap_ok(self._lc_target_lane):
                    self._raise_floor(15.0)
            else:
                self._cap_speed(max(FALLBACK_MIN_KMH,
                                    self.control_object.speed_km_h * FALLBACK_FACTOR))

        def _detour_gap_ok(self, target_lane, ahead=15.0, behind=8.0):
            """True if no vehicle occupies the target-lane corridor within
            `ahead` m in front / `behind` m behind the ego's projection."""
            try:
                ego_long, _ = target_lane.local_coordinates(self.control_object.position)
                tm = getattr(self.engine, "traffic_manager", None)
                vehicles = list(getattr(tm, "traffic_vehicles", None) or [])
            except Exception:
                return False
            half_w = target_lane.width_at(0) / 2 + 0.3
            for v in vehicles:
                try:
                    v_long, v_lat = target_lane.local_coordinates(v.position)
                except Exception:
                    continue
                if abs(v_lat) > half_w:
                    continue
                if -behind < v_long - ego_long < ahead:
                    return False
            return True

        def _detour_queue_ahead(self, sign, veh_long):
            """True if a slow/stopped vehicle occupies the obstacle lane between
            ego and the cones — the tail of a queue formed behind the obstacle."""
            try:
                tm = getattr(self.engine, "traffic_manager", None)
                vehicles = list(getattr(tm, "traffic_vehicles", None) or [])
            except Exception:
                return False
            lane = sign.lane
            half_w = lane.width_at(0) / 2 + 0.3
            for v in vehicles:
                try:
                    v_long, v_lat = lane.local_coordinates(v.position)
                except Exception:
                    continue
                if abs(v_lat) > half_w:
                    continue
                if not (veh_long < v_long <= sign.obstacle_long + 3.0):
                    continue
                if v_long - veh_long > self.DETOUR_QUEUE_LOOKAHEAD_M:
                    continue
                if float(getattr(v, "speed_km_h", 99.0)) < 10.0:
                    return True
            return False

        def _detour_target_lane(self, sign):
            """DETOUR-ONLY: pick the physically-correct adjacent lane from the
            sign's own resolved allowed set (``_allowed_lane_indices`` is correct
            on both SUMO and PG networks, unlike raw ``cur±1`` arithmetic — SUMO
            lane 0 is the rightmost, PG lane 0 is the leftmost). Prefers the
            right-hand option for 4.2.3."""
            allowed = list(getattr(sign, "_allowed_lane_indices", None) or [])
            if not allowed:
                return None
            rn = getattr(getattr(getattr(self, "engine", None), "current_map", None),
                         "road_network", None)
            if rn is None:
                return None
            sign_num = lane_index_num(sign.lane)

            def _num_kind(idx):
                if isinstance(idx, tuple) and len(idx) >= 3 and isinstance(idx[2], int):
                    return idx[2], "pg"
                try:
                    return int(str(idx).rsplit("_", 1)[1]), "sumo"
                except (ValueError, IndexError):
                    return None, None

            def _right_first(idx):
                # SUMO: right = lower lane num; PG: right = higher lane num.
                n, kind = _num_kind(idx)
                if n is None or sign_num is None:
                    return 1
                is_right = (kind == "sumo" and n < sign_num) or \
                           (kind == "pg" and n > sign_num)
                return 0 if is_right else 1

            for idx in sorted(allowed, key=_right_first):
                try:
                    lane = rn.get_lane(idx)
                except Exception:
                    lane = None
                if lane is not None and getattr(lane, "index", None) not in self._restricted_lanes:
                    return lane
            return None
