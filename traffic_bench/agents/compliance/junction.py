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

import numpy as np
from metadrive.component.vehicle.base_vehicle import BaseVehicle

from traffic_bench.signs.junction import StopSign, YieldSign

logger = logging.getLogger(__name__)


class JunctionCompliance:
        def _handle_stop_sign(self, sign):
            ego_lane = self.control_object.lane
            sign_lane = sign.lane
            stop_long = sign.stop_line_position
            sid = id(sign)
            st = self._stop_states.setdefault(
                sid, {"waiting": False, "steps": 0, "done": False}
            )

            if on_same_road(ego_lane, sign_lane):
                veh_long = self._veh_long(sign_lane)
                dist = stop_long - veh_long
                # Past the stop line — reset state so the sign is inert for this vehicle.
                if veh_long >= stop_long + STOP_PAST_THRESHOLD:
                    st.update(waiting=False, steps=0, done=False)
                    return
            else:
                # Ego on an upstream lane that eventually leads to sign.lane —
                # compute accumulated distance through the exit-lanes chain.
                dist = self._distance_to_sign(ego_lane, sign_lane, stop_long)
                if dist is None:
                    return

            if st["done"]:
                return

            # Engage window: braking distance OR a fixed near-line radius.
            # Critical: at speed≈0, ``_approach_dist(0)`` is 0, so a pure
            # approach_dist gate would drop the FSM mid-wait and the kick-start
            # in ComprehensiveRuleExpert would inch forward then re-brake (stutter).
            approach = max(float(self._approach_dist(0.0)), STOP_ENGAGE_DISTANCE_M)
            in_stop_zone = dist is not None and 0 < float(dist) < approach
            if st["waiting"] and dist is not None and float(dist) > 0:
                in_stop_zone = True
            if not in_stop_zone:
                return

            if st["waiting"]:
                st["steps"] += 1
                if st["steps"] >= self.STOP_WAIT_STEPS:
                    # Mandatory dwell counted from first speed≈0. Further holding
                    # for conflicting traffic is yield/TTC (``_handle_yield_sign``),
                    # not a second post-clearance wait.
                    st.update(done=True, waiting=False)
                    return
                self._cap_speed(0.001)
            elif self.control_object.speed < 0.1:
                st.update(waiting=True, steps=0)
                self._cap_speed(0.001)
            else:
                self._cap_speed(0.001)

        def _handle_yield_sign(self, sign):
            """Yield / right-hand rule: creep to a stop near the junction.

            Hard-stop only near ``stop_line_position`` (YieldSign.YIELD_STOP_BEFORE_END)
            when conflicting main-road traffic is present. Coarse MAIN_ROAD_ZONE is a
            prefilter; sticky path-clearance in ``_check_main_road_traffic`` keeps
            yielding until the foe has passed the ego/foe route intersection.
            """
            if not on_same_road(self.control_object.lane, sign.lane):
                return
            veh_long = self._veh_long(sign.lane)
            stop_before = float(
                getattr(sign, "YIELD_STOP_BEFORE_END", YieldSign.YIELD_STOP_BEFORE_END)
            )
            main_before = float(
                getattr(sign, "MAIN_ROAD_ZONE_BEFORE", YieldSign.MAIN_ROAD_ZONE_BEFORE)
            )
            stop_long = float(
                getattr(
                    sign,
                    "stop_line_position",
                    max(0.0, float(sign.lane.length) - stop_before),
                )
            )
            # Past the stop line: keep blocking while still in the obligation zone.
            if veh_long >= stop_long:
                if veh_long <= float(getattr(sign, "zone_end", sign.lane.length)):
                    has_traffic, _ = sign._check_main_road_traffic(self.control_object)
                    if has_traffic:
                        self._cap_speed(0.001)
                return

            dist_to_stop = stop_long - veh_long
            approach = self._approach_dist(0.0)
            # Ignore until within braking range or the main-road watch window.
            if dist_to_stop > max(approach, main_before):
                return

            has_traffic, _ = sign._check_main_road_traffic(self.control_object)
            if not has_traffic:
                return

            # At / past the preferential stop: full stop.
            if dist_to_stop <= 1.5 or self.control_object.speed < 0.15:
                self._cap_speed(0.001)
                return

            # Soft approach toward the stop line (do not freeze far from the junction).
            creep_kmh = max(8.0, min(SLOW_APPROACH_MIN_KMH, dist_to_stop * 1.5))
            self._cap_speed(creep_kmh)

        def _yield_conflict_window_m(self, sign=None):
            """(before, after) metres for main-road foe proximity, from YieldSign."""
            before = float(
                getattr(sign, "MAIN_ROAD_ZONE_BEFORE", YieldSign.MAIN_ROAD_ZONE_BEFORE)
                if sign is not None
                else YieldSign.MAIN_ROAD_ZONE_BEFORE
            )
            after = float(
                getattr(sign, "MAIN_ROAD_ZONE_AFTER", YieldSign.MAIN_ROAD_ZONE_AFTER)
                if sign is not None
                else YieldSign.MAIN_ROAD_ZONE_AFTER
            )
            return before, after

        def _foe_in_main_conflict_window(self, other_dist_to_end: float, sign=None) -> bool:
            """True if foe distance-to-lane-end falls in MAIN_ROAD_ZONE_BEFORE/AFTER."""
            before, after = self._yield_conflict_window_m(sign)
            return -after <= float(other_dist_to_end) <= before

        @staticmethod
        def _is_junction_internal_lane(lane) -> bool:
            """SUMO junction connectors / vias — not main approaches."""
            idx = getattr(lane, "index", None)
            if idx is None:
                return True
            s = str(idx).lower()
            return (
                s.startswith("junction")
                or ":_:" in s
                or "junction" in s
            )

        def _is_outgoing_sumo_lane(self, lane) -> bool:
            """True for post-junction outgoing edges (enter from junction, leave road)."""
            if self._is_junction_internal_lane(lane):
                return True
            try:
                engine = getattr(self, "engine", None)
                graph = getattr(
                    getattr(getattr(engine, "current_map", None), "road_network", None),
                    "graph",
                    None,
                )
                if graph is None:
                    return False
                info = graph.get(getattr(lane, "index", None))
                if info is None:
                    return False
                entry = list(getattr(info, "entry_lanes", None) or [])
                exit_ = list(getattr(info, "exit_lanes", None) or [])
                from_junction = any(str(e).lower().startswith("junction") for e in entry)
                to_junction = any(str(e).lower().startswith("junction") for e in exit_)
                return bool(from_junction and not to_junction)
            except Exception:
                return False

        def _foe_on_yield_main_approach(self, other_lane) -> bool | None:
            """Filter foes using YieldSign allow/deny lists.

            Returns:
              True  — foe is on a monitored main *incoming* approach
              False — foe is on outgoing / non-approach (must not trigger yield)
              None  — no YieldSign with main_road_lanes; caller uses fallback
            """
            decided = False
            for sign in self._get_signs():
                if not isinstance(sign, YieldSign):
                    continue
                if isinstance(sign, StopSign):
                    continue
                if not getattr(sign, "main_road_lanes", None):
                    continue
                decided = True
                idx = getattr(other_lane, "index", None)
                if sign._is_outgoing_lane_index(idx):
                    return False
                if sign._is_on_main_approach(idx):
                    return True
            if decided:
                return False
            return None

        def _handle_main_road(self, sign):
            if on_same_road(self.control_object.lane, sign.lane):
                self._has_priority = True

        def _handle_end_main_road(self, sign):
            if not on_same_road(self.control_object.lane, sign.lane):
                return
            veh_long = self._veh_long(sign.lane)
            dist = sign.placement_long - veh_long
            if dist <= 0:
                # Past the sign — priority is gone
                self._has_priority = False
            elif dist < END_MAIN_ROAD_LOOKAHEAD:
                self._cap_speed(max(SLOW_APPROACH_MIN_KMH,
                                    self.control_object.speed_km_h * SLOW_APPROACH_FACTOR))

        def _init_priority_network(self):
            """Lazily initialize the priority network from RoadPriorityAssigner."""
            if getattr(self, "_priority_network", None) is not None:
                return
            self._priority_network = None
            try:
                from priority.road_priority_assigner import RoadPriorityAssigner
                engine = getattr(self, "engine", None)
                if engine is None:
                    return
                current_map = getattr(engine, "current_map", None)
                if current_map is None:
                    return
                assigner = RoadPriorityAssigner()
                self._priority_network = assigner.assign_priorities(current_map)
            except Exception as exc:
                logger.debug("Failed to init priority network: %s", exc)

        def _handle_intersection_priority(self):
            """Right-hand rule at equal-priority intersections.

            At intersections where ``sign_style == "equal"``, check for vehicles
            approaching from the right.  If one is found in the conflict zone,
            yield by capping speed.
            """
            self._init_priority_network()
            if self._priority_network is None:
                return
            ego_lane = self.control_object.lane
            if ego_lane is None:
                return
            intersection = self._priority_network.get_intersection_for_lane(ego_lane)
            if intersection is None:
                return
            if intersection.sign_style != "equal":
                return
            # Check if we're close enough to the intersection to care
            veh_long = ego_lane.local_coordinates(self.control_object.position)[0]
            dist_to_end = ego_lane.length - veh_long
            if dist_to_end > YieldSign.EGO_ZONE_BEFORE:
                return
            # Find our approach
            ego_approach = None
            ego_idx = getattr(ego_lane, "index", None)
            for approach in intersection.approaches:
                for lane in approach.lanes:
                    if getattr(lane, "index", None) == ego_idx:
                        ego_approach = approach
                        break
                if ego_approach is not None:
                    break
            if ego_approach is None:
                return
            # Right-hand rule: yield to vehicles on approaches from the right
            ego_heading = ego_approach.heading
            from metadrive.component.vehicle.base_vehicle import BaseVehicle
            engine = getattr(self, "engine", None)
            if engine is None:
                return
            stop_before = YieldSign.YIELD_STOP_BEFORE_END
            for obj in engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values():
                if getattr(obj, "id", None) == getattr(self.control_object, "id", None):
                    continue
                try:
                    other_policy = engine.get_policy(getattr(obj, "id", None))
                except Exception:
                    other_policy = None
                if (
                    other_policy is not None
                    and hasattr(other_policy, "ego_distance_to_spawn_lane_end")
                    and not bool(getattr(other_policy, "released", True))
                ):
                    continue
                other_lane = getattr(obj, "lane", None)
                if other_lane is None:
                    continue
                if self._is_junction_internal_lane(other_lane):
                    continue
                if self._is_outgoing_sumo_lane(other_lane):
                    continue
                yield_filter = self._foe_on_yield_main_approach(other_lane)
                if yield_filter is False:
                    continue
                # Check if this vehicle is on a different approach of the same intersection
                other_approach = None
                other_idx = getattr(other_lane, "index", None)
                for approach in intersection.approaches:
                    if approach is ego_approach:
                        continue
                    for lane in approach.lanes:
                        if getattr(lane, "index", None) == other_idx:
                            other_approach = approach
                            break
                    if other_approach is not None:
                        break
                if other_approach is None:
                    continue
                # Same MAIN_ROAD_ZONE_BEFORE/AFTER window as YieldSign.
                other_long = other_lane.local_coordinates(obj.position)[0]
                other_dist = other_lane.length - other_long
                if not self._foe_in_main_conflict_window(other_dist):
                    continue
                # Is the other vehicle on our right?
                heading_diff = other_approach.heading - ego_heading
                heading_diff = heading_diff % (2.0 * np.pi)
                # Vehicle from the right means its approach heading is roughly
                # 90° clockwise (≈ π/2) from ours.
                if 0.3 < heading_diff < np.pi:
                    if dist_to_end > stop_before:
                        creep_kmh = max(8.0, min(SLOW_APPROACH_MIN_KMH, dist_to_end * 1.2))
                        self._cap_speed(creep_kmh)
                    else:
                        self._cap_speed(0.001)
                    return

        def _handle_intersection_priority_sumo(self):
            """SUMO-native priority handler.

            Reads lane.turns[i]["priority"] + junction_type (already extracted from
            SUMO .net.xml by map_utils.extract_map_features) and yields to priority
            traffic even when no explicit YieldSign/MainRoadSign is placed in the
            scene. Complements the PG-only `_handle_intersection_priority`.

            Actions:
              * has_priority=True AND junction_type != right_before_left → ego is on
                main road → no yield.
              * junction_type == right_before_left → equal priority → yield to any
                foe on the right (per Russian traffic rules).
              * Otherwise → ego must yield → cap speed to 0 if any NPC is on a
                must_yield_to lane close to the junction.
            """
            from metadrive.component.vehicle.base_vehicle import BaseVehicle

            ego = self.control_object
            ego_lane = ego.lane
            if ego_lane is None:
                return
            info = self._get_sumo_priority_info(ego_lane)
            if info is None:
                return

            # Approach distance: only act when ego is near the end of its current
            # lane (close to the junction) — same window as YieldSign.EGO_ZONE_BEFORE.
            try:
                veh_long = self._veh_long(ego_lane)
                dist_to_end = float(ego_lane.length) - veh_long
            except Exception:
                return
            if dist_to_end > YieldSign.EGO_ZONE_BEFORE:
                return

            jt = info["junction_type"]
            pri = info["priority"]
            has_priority = bool(pri.get("has_priority", False))
            must_yield_to = set(pri.get("must_yield_to") or [])
            foes = set(pri.get("foes") or [])

            if jt == "right_before_left":
                # Equal-priority — yield to any foe to the right of ego.
                watch_lanes = foes
                check_right = True
            elif has_priority:
                # Ego on main road (priority/allway_stop/priority_stop/zipper).
                return
            else:
                # Ego on secondary road — yield to must_yield_to.
                watch_lanes = must_yield_to
                check_right = False

            if not watch_lanes:
                return

            engine = getattr(self, "engine", None)
            if engine is None:
                return

            stop_before = YieldSign.YIELD_STOP_BEFORE_END
            # Scan surrounding vehicles for priority conflict.
            ego_id = getattr(ego, "id", None)
            for obj in engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values():
                if getattr(obj, "id", None) == ego_id:
                    continue
                # Gated aux still held at spawn must not freeze a yielding ego.
                try:
                    other_policy = engine.get_policy(getattr(obj, "id", None))
                except Exception:
                    other_policy = None
                if (
                    other_policy is not None
                    and hasattr(other_policy, "ego_distance_to_spawn_lane_end")
                    and not bool(getattr(other_policy, "released", True))
                ):
                    continue
                other_lane = getattr(obj, "lane", None)
                if other_lane is None:
                    continue
                if self._is_junction_internal_lane(other_lane):
                    continue
                if self._is_outgoing_sumo_lane(other_lane):
                    continue
                yield_filter = self._foe_on_yield_main_approach(other_lane)
                if yield_filter is False:
                    continue
                other_idx = getattr(other_lane, "index", None)
                if other_idx not in watch_lanes:
                    continue
                # Same MAIN_ROAD_ZONE_BEFORE/AFTER window as YieldSign.
                try:
                    other_long = other_lane.local_coordinates(obj.position)[0]
                    other_dist_to_end = float(other_lane.length) - other_long
                except Exception:
                    continue
                if not self._foe_in_main_conflict_window(other_dist_to_end):
                    continue
                # Right-hand rule: additionally require NPC to be on ego's right.
                if check_right:
                    try:
                        vec = np.asarray(obj.position[:2], dtype=np.float64) - \
                              np.asarray(ego.position[:2], dtype=np.float64)
                        ego_heading = float(getattr(ego, "heading_theta", 0.0))
                        rel_angle = float(np.arctan2(vec[1], vec[0])) - ego_heading
                        rel_angle = (rel_angle + np.pi) % (2.0 * np.pi) - np.pi
                        # Right = negative angle in standard coords (y flipped in
                        # MetaDrive, but heading convention matches priority_signs
                        # `_is_other_on_right`).
                        if not (-np.pi + 1e-4 < rel_angle < -1e-4):
                            continue
                    except Exception:
                        continue
                # Priority conflict → hard stop only near the yield stop line.
                if dist_to_end > stop_before:
                    creep_kmh = max(8.0, min(SLOW_APPROACH_MIN_KMH, dist_to_end * 1.2))
                    self._cap_speed(creep_kmh)
                else:
                    self._cap_speed(0.001)
                return
