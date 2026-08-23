from metadrive.manager.base_manager import BaseManager
import os
import random

from traffic_bench.signs.base import BaseTrafficSign
from traffic_bench.signs.speed.limit import SpeedLimitSign
from traffic_bench.signs.speed.end_of_zone import EndOfSpeedLimitSign, EndOfAllRestrictionsSign, BaseEndOfZoneSign
from traffic_bench.signs.speed.zone import ZoneSpeedLimitSign


class TrafficSignManager(BaseManager):
    def __init__(self):
        super().__init__()
        self.signs = []
        self.rules = []
        self.violations = []
        self.violation_details = []

    @staticmethod
    def _same_road_direction(idx_a, idx_b):
        if idx_a is None or idx_b is None:
            return False
        try:
            return idx_a[0] == idx_b[0] and idx_a[1] == idx_b[1]
        except (IndexError, TypeError):
            return False

    def before_step(self, *args, **kwargs):
        """Called before each simulation step - update traffic light states"""
        for sign in self.signs:
            if hasattr(sign, 'update_state'):
                sign.update_state()

        return {}

    def _lane_has_continuation(self, lane):
        """Check that the lane's end node has outgoing edges (not a dead end).

        Works for both SUMO (EdgeRoadNetwork) and MetaDrive PG (NodeRoadNetwork).
        For SUMO flat graphs the check is skipped (edges always span intersections).
        """
        road_network = self.engine.current_map.road_network
        graph = getattr(road_network, "graph", None)
        if graph is None:
            return True

        idx = getattr(lane, "index", None)
        if idx is None:
            return True

        # --- NodeRoadNetwork (PG): index = (from_node, to_node, lane_num) ---
        if isinstance(idx, tuple) and len(idx) >= 2:
            end_node = idx[1]
            outgoing = graph.get(end_node)
            if outgoing is None:
                return False  # dead-end outer socket
            if isinstance(outgoing, dict) and len(outgoing) == 0:
                return False
            return True

        # --- EdgeRoadNetwork (SUMO): flat graph, always has context ---
        return True

    def _get_random_lane(self, min_length=10.0, reference_lane_index=None,
                         require_continuation=True):
        """Pick a random lane, optionally filtering out dead-end outer sockets."""
        road_network = self.engine.current_map.road_network
        lanes = []
        for start_road in road_network.graph.values():
            if not isinstance(start_road, dict):
                continue
            for end_road in start_road.values():
                for lane in end_road:
                    if lane.length >= min_length:
                        lanes.append(lane)
        if reference_lane_index is not None:
            lanes = [
                lane for lane in lanes
                if self._same_road_direction(getattr(lane, "index", None), reference_lane_index)
            ]
        if require_continuation:
            lanes = [l for l in lanes if self._lane_has_continuation(l)]
        if not lanes:
            raise RuntimeError("No lanes meet the criteria!")
        return random.choice(lanes)

    def _zones_overlap(self, sign1, sign2):
        """Check if two signs' zones overlap on the same lane."""
        #may need change
        if sign1.lane != sign2.lane:
            return False

        zone1_start = getattr(sign1, "zone_start", getattr(sign1, "placement_long", 0))
        zone1_end = getattr(sign1, "zone_end", zone1_start + getattr(sign1, "zone_length", 0))
        zone2_start = getattr(sign2, "zone_start", getattr(sign2, "placement_long", 0))
        zone2_end = getattr(sign2, "zone_end", zone2_start + getattr(sign2, "zone_length", 0))
        return not (zone1_end < zone2_start or zone2_end < zone1_start)

    def _is_speed_limit_sign(self, sign):
        """Check if sign is a speed limit sign."""
        sign_type = type(sign).__name__
        return "SpeedLimitSign" in sign_type

    def _cleanup_sign_object(self, sign):
        if sign is None:
            return
        sign_id = getattr(sign, "id", None)
        if sign_id is not None and getattr(self, "spawned_objects", None) is not None:
            self.spawned_objects.pop(sign_id, None)
            try:
                self.engine.clear_objects([sign_id])
            except Exception:
                pass
        if hasattr(sign, "destroy"):
            try:
                sign.destroy()
            except Exception:
                pass

    def _purge_missing_spawned_objects(self):
        if getattr(self, "spawned_objects", None) is None:
            return
        engine_objs = getattr(self.engine, "_spawned_objects", None)
        if engine_objs is None:
            return
        missing = [obj_id for obj_id in list(self.spawned_objects.keys()) if obj_id not in engine_objs]
        for obj_id in missing:
            self.spawned_objects.pop(obj_id, None)

    def add_sign(
        self,
        sign_class,
        lane=None,
        use_random_lane=True,
        check_zone_overlap=True,
        reference_lane=None,
        reference_lane_index=None,
        longitudinal_offset=None,
        zone_start=None,
        zone_end=None,
        **kwargs
    ):
        """Add a sign and optionally validate zone overlap rules."""
        if reference_lane_index is None and reference_lane is not None:
            reference_lane_index = getattr(reference_lane, "index", None)

        if lane is None and use_random_lane:
            lane = self._get_random_lane(reference_lane_index=reference_lane_index)

        if lane is None:
            raise ValueError("No lane provided and random lane selection is disabled.")

        kwargs.pop("use_random_lane", None)
        if longitudinal_offset is not None:
            kwargs["longitudinal_offset"] = longitudinal_offset
        if zone_start is not None:
            kwargs["zone_start"] = zone_start
        if zone_end is not None:
            kwargs["zone_end"] = zone_end

        # force_spawn=True: always create a new object and call __init__ of the sign (not reset from the pool)
        sign = self.engine.spawn_object(sign_class, lane=lane, force_spawn=True, **kwargs)
        self.spawned_objects[sign.id] = sign

        if check_zone_overlap:
            try:
                sign_type = type(sign).__name__
                is_stop_sign = sign_type == "StopSign"
                is_speed_sign = self._is_speed_limit_sign(sign)

                for existing_sign in self.signs:
                    existing_type = type(existing_sign).__name__
                    existing_is_stop = existing_type == "StopSign"
                    existing_is_speed = self._is_speed_limit_sign(existing_sign)

                    if is_stop_sign and not existing_is_speed and self._zones_overlap(sign, existing_sign):
                        raise ValueError(
                            f"StopSign zone overlaps with existing {existing_type} "
                        )

                    # if not is_speed_sign and existing_is_stop and self._zones_overlap(sign, existing_sign):
                    #     raise ValueError(
                    #         f"{sign_type} cannot be placed in StopSign zone. "
                    #     )
            except Exception:
                self._cleanup_sign_object(sign)
                raise

        self.signs.append(sign)
        return sign

    def add_rule(self, rule):
        """
        Add a non-spawned logical verifier (e.g. pedestrian-yield rule).
        """
        if rule is None:
            raise ValueError("rule can not be None")
        self.rules.append(rule)
        return rule

    def build_zones(self):
        """Build zones for all signs, handling cross-lane interactions."""
        route_to_signs = {}
        for sign in self.signs:
            route_to_signs.setdefault(sign.lane.id, []).append(sign)

        for signs in route_to_signs.values():
            self._build_zones_for_route(signs)

    def _build_zones_for_route(self, signs):
        """Build zones for signs on the same route."""
        signs_sorted = sorted(signs, key=lambda s: s.placement_long)

        for sign in signs_sorted:
            if isinstance(sign, SpeedLimitSign | ZoneSpeedLimitSign | BaseEndOfZoneSign):
                sign.update_zones()

    def get_signs_before(self, reference_sign, sign_type=None) -> list[BaseTrafficSign]:
        """Get all signs that come before reference_sign on the same route."""
        result = []
        for sign in self.signs:
            if sign is reference_sign:
                continue
            if sign.lane != reference_sign.lane or \
                sign.placement_long >= reference_sign.placement_long or \
                (sign_type is not None and not isinstance(sign, sign_type)):
                continue

            result.append(sign)
        return sorted(result, key=lambda s: s.placement_long)

    def check_all_violations(self, vehicle, for_reward=False):
        checks = []
        for obj in list(self.signs) + list(self.rules):
            if not hasattr(obj, "check_violation"):
                continue
            try:
                violated = bool(obj.check_violation(vehicle, for_reward))
            except Exception:
                violated = False
            checks.append((obj, violated))
        return checks

    def track_violations(self, vehicle, step, stop_status="N/A", speed_status="N/A",
                         stop_sign_class=None, speed_limit_sign_class=None, verbose=True):
        """Track sign violations and update realtime stop/speed statuses."""
        stop_violated = False
        speed_violated = False

        violation_results = self.check_all_violations(vehicle, for_reward=False)
        seen_violation_keys = {v[0] for v in self.violation_details}
        for sign, violated in violation_results:
            if not violated:
                continue
            sign_type = type(sign).__name__
            sign_id = sign.id if hasattr(sign, 'id') else id(sign)
            # Include step so each distinct violation event (re-fired after
            # re-arming in BaseTrafficSign.check_violation) is counted, while
            # still deduping multiple calls within the same step.
            violation_key = (sign_id, sign_type, step)
            if violation_key not in seen_violation_keys:
                self.violations.append(sign)
                self.violation_details.append((violation_key, sign_type, step))
                seen_violation_keys.add(violation_key)
                if verbose:
                    print(f"  [VIOLATION] {sign_type} violated at step {step}!")

        realtime_stop_status = None
        realtime_speed_status = None
        speed_zone_seen = False
        agent_speed_kmh = float(getattr(vehicle, "speed_km_h", 0.0) or 0.0)
        vehicle_id = getattr(vehicle, "id", None)

        def _direction_match(sign_lane_index):
            if sign_lane_index is None:
                return None
            agent_lane = getattr(vehicle, "lane", None)
            agent_lane_idx = getattr(agent_lane, "index", None) if agent_lane is not None else None
            if agent_lane_idx is not None:
                return BaseTrafficSign._same_road_direction(agent_lane_idx, sign_lane_index)
            navigation = getattr(vehicle, "navigation", None)
            if navigation is not None and hasattr(navigation, "current_ref_lanes"):
                ref_lanes = navigation.current_ref_lanes or []
                if len(ref_lanes) == 0:
                    return None
                for ref_lane in ref_lanes:
                    ref_idx = getattr(ref_lane, "index", None)
                    if BaseTrafficSign._same_road_direction(ref_idx, sign_lane_index):
                        return True
                return False
            return None

        for sign in self.signs:
            sign_type = type(sign).__name__
            is_stop_sign = isinstance(sign, stop_sign_class) if stop_sign_class is not None else sign_type == "StopSign"
            is_speed_sign = (
                isinstance(sign, speed_limit_sign_class)
                if speed_limit_sign_class is not None
                else "SpeedLimitSign" in sign_type
            )
            if not is_stop_sign and not is_speed_sign:
                continue

            lane = getattr(sign, "lane", None)
            if lane is None:
                continue

            # Multi-edge zones (combined SUMO pairs) span edges other than the
            # sign's own lane, so the direction gate / sign-lane veh_long don't
            # apply — defer entirely to the sign's own membership check.
            is_multi_edge = is_speed_sign and bool(getattr(sign, "zone_edges", None))

            if not is_multi_edge:
                sign_lane_idx = getattr(lane, "index", None)
                dir_match = _direction_match(sign_lane_idx)
                if dir_match is False:
                    continue
                if dir_match is None:
                    if not sign.is_in_drivable_area(vehicle):
                        continue
            try:
                veh_long = float(lane.local_coordinates(vehicle.position)[0])
            except Exception:
                veh_long = None
            if veh_long is None and not is_multi_edge:
                continue

            if is_speed_sign:
                if hasattr(sign, "is_vehicle_in_zone"):
                    in_zone = bool(sign.is_vehicle_in_zone(vehicle))
                else:
                    zone_start = getattr(sign, "zone_start", None)
                    zone_end = getattr(sign, "zone_end", None)
                    if zone_start is None or zone_end is None:
                        continue
                    in_zone = zone_start <= veh_long <= zone_end
                if not in_zone:
                    continue
                speed_zone_seen = True
                limit_kmh = float(getattr(sign, "speed_limit", 0.0) or 0.0)
                if agent_speed_kmh > limit_kmh + 1e-3:
                    realtime_speed_status = "VIOL"
                    speed_violated = True
                elif realtime_speed_status != "VIOL":
                    realtime_speed_status = "COMPL"

            if is_stop_sign:
                zone_start = getattr(sign, "zone_start", None)
                zone_end = getattr(sign, "zone_end", None)
                stop_long = float(getattr(sign, "stop_line_position", getattr(sign, "placement_long", 0.0)))
                if zone_start is not None and zone_end is not None:
                    if not (zone_start <= veh_long <= zone_end):
                        continue

                if veh_long < stop_long - 0.3:
                    if realtime_stop_status != "VIOL":
                        realtime_stop_status = "APPROACH"
                else:
                    state = getattr(sign, "_vehicle_states_stop", {}).get(vehicle_id, {})
                    stopped_before_line = bool(state.get("stopped_before_line", False) or state.get("stopped", False))
                    if stopped_before_line:
                        if realtime_stop_status != "VIOL":
                            realtime_stop_status = "COMPL"
                    else:
                        realtime_stop_status = "VIOL"
                        stop_violated = True

        if realtime_stop_status is not None:
            stop_status = realtime_stop_status
        if realtime_speed_status is not None:
            speed_status = realtime_speed_status
        elif speed_status.startswith("ZONE") and speed_zone_seen:
            speed_status = "COMPL"

        return stop_violated, speed_violated, stop_status, speed_status

    def reset_violations(self):
        """Reset violation tracking for a new episode."""
        self.violations.clear()
        self.violation_details.clear()

    def before_reset(self):
        self._purge_missing_spawned_objects()
        engine_objs = getattr(self.engine, "_spawned_objects", None)
        valid_ids = []
        if engine_objs is not None:
            valid_ids = [obj_id for obj_id in list(self.spawned_objects.keys()) if obj_id in engine_objs]
        if valid_ids:
            try:
                self.clear_objects(valid_ids)
            except Exception:
                pass
        self.spawned_objects = {}
        self.signs.clear()
        for rule in self.rules:
            if hasattr(rule, "before_reset"):
                try:
                    rule.before_reset()
                except Exception:
                    pass
        self.reset_violations()

    def destroy(self):
        self.rules.clear()
        self.before_reset()
        super().destroy()