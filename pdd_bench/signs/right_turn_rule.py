import numpy as np
from collections import deque
from pdd_bench.signs.base_traffic_sign import BaseTrafficSign


class RightTurnRule(BaseTrafficSign):
    """Right turn is correct only from the most right line"""
    
    def __init__(self, lane=None, debug=False, lookahead_distance=60.0, **kwargs):
        if lane is None:
            self.lane = None
            self._show_model = False
            self.placement_long = 0
        else:
            super().__init__(lane, show_model=False, **kwargs)

        self._vehicle_states = {}
        self.debug = debug
        self.lookahead_distance = float(lookahead_distance)
        self._violation_states = {}
        self._status_cache = {}
        self._extractor = None
        self._extractor_engine_id = None
        self._extractor_cls = None
        self._turn_signal_history = {}
        self._prev_planning_right = {}
        self._planning_streak = {}
        self._turn_non_rightmost_seen = {}

        self._turn_horizons_m = (15.0, 30.0, 45.0)
        self._distance_band_m = 4.0
        self._right_lateral_threshold_m = 1.75
        self._right_trend_threshold_m = 1.0
        self._min_forward_distance_m = 8.0
        self._turn_history_window = 5
        self._turn_history_required = 3
        self._min_turn_steps_for_completion = 3

    def get_rule_description(self) -> str:
        return "Right turn must be made from the rightmost lane"

    def _create_visual_model(self):
        pass

    def _import_extractor_cls(self):
        if self._extractor_cls is not None:
            return self._extractor_cls

        candidates = (
            "metadrive_extractor",
            "carl_in_metadrive.metadrive_extractor",
            "agents.carl_in_metadrive.metadrive_extractor",
        )
        for module_name in candidates:
            try:
                module = __import__(module_name, fromlist=["MetaDriveExtractor"])
                self._extractor_cls = getattr(module, "MetaDriveExtractor")
                return self._extractor_cls
            except Exception:
                continue
        raise ImportError("MetaDriveExtractor is not available on PYTHONPATH")

    def _get_extractor(self, vehicle):
        engine = getattr(vehicle, "engine", None)
        if engine is None:
            return None
        engine_id = id(engine)
        if self._extractor is not None and self._extractor_engine_id == engine_id:
            return self._extractor

        extractor_cls = self._import_extractor_cls()
        self._extractor = extractor_cls(engine, fov_range=self.lookahead_distance)
        self._extractor_engine_id = engine_id
        return self._extractor

    def get_status(self, vehicle):
        engine = getattr(vehicle, "engine", None)
        step = getattr(engine, "episode_step", None) if engine is not None else None
        vehicle_id = getattr(vehicle, "id", id(vehicle))

        if step == 0:
            self._violation_states[vehicle_id] = False
            self._prev_planning_right[vehicle_id] = False
            self._planning_streak[vehicle_id] = 0
            self._turn_non_rightmost_seen[vehicle_id] = False
            if vehicle_id in self._turn_signal_history:
                self._turn_signal_history[vehicle_id].clear()
            self._status_cache.pop(vehicle_id, None)

        cached = self._status_cache.get(vehicle_id)
        if step is not None and cached is not None and cached.get("step") == step:
            return cached

        if getattr(vehicle, "navigation", None) is None:
            status = {
                "step": step,
                "is_planning_right_turn": False,
                "is_rightmost_lane": True,
                "is_leftmost_lane": False,
                "violation": False,
                "turn_completed_with_violation": False,
                "should_terminate": False,
                "is_rule_compliant": True,
            }
            self._status_cache[vehicle_id] = status
            return status

        is_turning_right = self._is_planning_right_turn(vehicle)
        is_rightmost = self._is_rightmost_lane(vehicle)
        is_leftmost = self._is_leftmost_lane(vehicle)
        prev_planning = self._prev_planning_right.get(vehicle_id, False)
        prev_planning_streak = self._planning_streak.get(vehicle_id, 0)
        turn_non_rightmost_seen = self._turn_non_rightmost_seen.get(vehicle_id, False)

        if is_turning_right:
            if not prev_planning:
                turn_non_rightmost_seen = not is_rightmost
            else:
                turn_non_rightmost_seen = turn_non_rightmost_seen or (not is_rightmost)
            self._planning_streak[vehicle_id] = prev_planning_streak + 1
            self._turn_non_rightmost_seen[vehicle_id] = turn_non_rightmost_seen
        else:
            self._planning_streak[vehicle_id] = 0

        turn_completed_with_violation = False
        if prev_planning and (not is_turning_right):
            had_turn_violation = turn_non_rightmost_seen or self._violation_states.get(vehicle_id, False)
            if had_turn_violation and prev_planning_streak >= self._min_turn_steps_for_completion:
                turn_completed_with_violation = True
            self._turn_non_rightmost_seen[vehicle_id] = False

        violation_state = self._violation_states.setdefault(vehicle_id, False)

        if is_turning_right and not is_rightmost:
            violation_state = True
            self._violation_states[vehicle_id] = True
            if self.debug:
                print("[DEBUG VIOLATION] Violation active: planning right turn from non-rightmost lane")

        violation = is_turning_right and violation_state

        if not is_turning_right and violation_state:
            if self.debug:
                print("[DEBUG VIOLATION] Violation cleared: no longer planning right turn")
            self._violation_states[vehicle_id] = False

        self._prev_planning_right[vehicle_id] = is_turning_right

        status = {
            "step": step,
            "is_planning_right_turn": is_turning_right,
            "is_rightmost_lane": is_rightmost,
            "is_leftmost_lane": is_leftmost,
            "violation": violation,
            "turn_completed_with_violation": turn_completed_with_violation,
            "should_terminate": turn_completed_with_violation,
            "is_rule_compliant": (not is_turning_right) or (is_rightmost and not violation),
        }
        self._status_cache[vehicle_id] = status
        return status

    def check_violation(self, vehicle, for_reward=False) -> bool:
        _ = for_reward
        status = self.get_status(vehicle)
        return status["violation"]

    def _is_violating(self, vehicle) -> bool:
        status = self.get_status(vehicle)
        return status["violation"]

    def _sample_lateral_at_distance(self, local_x, local_y, target_distance):
        if local_x.size == 0:
            return None

        band = float(self._distance_band_m)
        mask = np.abs(local_x - float(target_distance)) <= band
        if np.any(mask):
            return float(np.median(local_y[mask]))

        order = np.argsort(np.abs(local_x - float(target_distance)))
        k = min(25, int(order.size))
        if k <= 0:
            return None
        nearest_idx = order[:k]
        return float(np.median(local_y[nearest_idx]))

    def _smooth_turn_signal(self, vehicle_id, raw_signal):
        hist = self._turn_signal_history.get(vehicle_id)
        if hist is None:
            hist = deque(maxlen=self._turn_history_window)
            self._turn_signal_history[vehicle_id] = hist
        hist.append(1 if raw_signal else 0)

        if len(hist) < self._turn_history_required:
            return bool(raw_signal)
        return sum(hist) >= self._turn_history_required

    def _is_planning_right_turn(self, vehicle):
        try:
            if getattr(vehicle, "navigation", None) is None:
                if self.debug:
                    print("[DEBUG VISUAL] No navigation")
                return False

            ego_pos = np.array(vehicle.position[:2], dtype=np.float64)
            ego_heading = float(vehicle.heading_theta)
            ego_forward = np.array([np.cos(ego_heading), np.sin(ego_heading)], dtype=np.float64)
            ego_left = np.array([-np.sin(ego_heading), np.cos(ego_heading)], dtype=np.float64)

            try:
                extractor = self._get_extractor(vehicle)
            except ImportError as e:
                if self.debug:
                    print(f"[DEBUG VISUAL] Cannot import extractor: {e}")
                return False
            if extractor is None:
                return False

            route_polygons = extractor.get_route_polygons(vehicle, ego_pos)
            if not route_polygons:
                if self.debug:
                    print("[DEBUG VISUAL] No route polygons")
                return False

            route_points = [point[:2] for poly in route_polygons if len(poly) >= 3 for point in poly]
            if not route_points:
                return False

            route_points = np.asarray(route_points, dtype=np.float64)
            rel_positions = route_points - ego_pos
            local_x = np.dot(rel_positions, ego_forward)
            local_y = np.dot(rel_positions, ego_left)

            ahead_mask = local_x > float(self._min_forward_distance_m)
            if not np.any(ahead_mask):
                return False
            local_x = local_x[ahead_mask]
            local_y = local_y[ahead_mask]

            y15 = self._sample_lateral_at_distance(local_x, local_y, self._turn_horizons_m[0])
            y30 = self._sample_lateral_at_distance(local_x, local_y, self._turn_horizons_m[1])
            y45 = self._sample_lateral_at_distance(local_x, local_y, self._turn_horizons_m[2])
            if y30 is None:
                return False

            right_thr = float(self._right_lateral_threshold_m)
            is_right_at_30 = y30 < -right_thr
            is_left_at_30 = y30 > right_thr

            trend_right = False
            if y15 is not None and y45 is not None:
                trend_right = (y45 - y15) < -float(self._right_trend_threshold_m)

            raw_right_turn = bool(is_right_at_30 or (trend_right and y30 < -0.5))
            if is_left_at_30:
                raw_right_turn = False

            vehicle_id = getattr(vehicle, "id", id(vehicle))
            is_right_turn = self._smooth_turn_signal(vehicle_id, raw_right_turn)

            if self.debug:
                print(
                    "[DEBUG VISUAL] "
                    f"y15={None if y15 is None else round(y15, 2)} "
                    f"y30={round(y30, 2)} "
                    f"y45={None if y45 is None else round(y45, 2)} "
                    f"right30={is_right_at_30} trend_right={trend_right} "
                    f"raw={raw_right_turn} smooth={is_right_turn}"
                )

            return is_right_turn

        except Exception as e:
            if self.debug:
                print(f"[DEBUG VISUAL] Exception: {e}")
                import traceback
                traceback.print_exc()
            return False

    def _is_rightmost_lane(self, vehicle):
        """True if lane is the most right"""
        try:
            lane_index = vehicle.lane_index
            road_network = vehicle.engine.current_map.road_network
            lanes = road_network.graph[lane_index[0]][lane_index[1]]
            max_lane_id = len(lanes) - 1
            return lane_index[2] == max_lane_id
        except Exception:
            return True

    def _is_leftmost_lane(self, vehicle):
        try:
            lane_index = vehicle.lane_index
            road_network = vehicle.engine.current_map.road_network
            lanes = road_network.graph[lane_index[0]][lane_index[1]]
            if not lanes:
                return False
            return lane_index[2] == 0
        except Exception:
            return False
