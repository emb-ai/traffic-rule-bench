from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class PedestrianYieldRule:
    """
    Rule verifier: vehicle must yield to pedestrians occupying a crosswalk.
    Additionally, standing/parking in configured no-stop zone before crosswalk is a violation
    even when no pedestrian is currently active.
    """

    def __init__(
        self,
        yield_distance: float = 12.0,
        yield_speed_kmh: float = 8.0,
        crosswalk_enter_tolerance: float = 0.2,
        no_stop_before_m: float = 8.0,
        no_stop_speed_kmh: float = 1.0,
        no_stop_min_duration_s: float = 1.0,
    ):
        self.id = "pedestrian_yield_rule"
        self._defaults = {
            "yield_distance": float(yield_distance),
            "yield_speed_kmh": float(yield_speed_kmh),
            "no_stop_before_m": float(no_stop_before_m),
            "no_stop_speed_kmh": float(no_stop_speed_kmh),
            "no_stop_min_duration_s": float(no_stop_min_duration_s),
        }
        self._crosswalk_enter_tolerance = float(crosswalk_enter_tolerance)
        self._vehicle_states: Dict[Tuple[str, str], dict] = {}

    def get_rule_description(self) -> str:
        return "Must yield on crosswalk; no standing in no-stop zone before crosswalk"

    def before_reset(self):
        self._vehicle_states.clear()

    def check_violation(self, vehicle, for_reward: bool = False) -> bool:
        engine, crosswalk_state = self._get_crosswalk_state(vehicle)
        if engine is None or crosswalk_state is None:
            return False

        if not crosswalk_state:
            self._cleanup_vehicle_states(getattr(vehicle, "id", None), {}, None)
            return False

        vehicle_pos = self._vehicle_pos(vehicle)
        heading = float(getattr(vehicle, "heading_theta", 0.0) or 0.0)
        forward = np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float64)
        vehicle_id = str(getattr(vehicle, "id", id(vehicle)))
        speed_kmh = float(getattr(vehicle, "speed_km_h", 0.0) or 0.0)
        thresholds = self._resolve_all_thresholds(engine)
        yield_distance = thresholds["yield_distance"]
        yield_speed_kmh = thresholds["yield_speed_kmh"]
        no_stop_before_m = thresholds["no_stop_before_m"]
        no_stop_speed_kmh = thresholds["no_stop_speed_kmh"]
        monitor_distance = max(yield_distance, no_stop_before_m) + 3.0

        self._cleanup_vehicle_states(vehicle_id, crosswalk_state, vehicle_pos, monitor_distance)

        for crosswalk_id, state in crosswalk_state.items():
            polygon = np.asarray(state.get("polygon", []), dtype=np.float64)
            if not self._valid_polygon(polygon):
                continue

            dist = self._distance_to_polygon(vehicle_pos, polygon)
            in_crosswalk = self._point_in_polygon(vehicle_pos, polygon) or dist <= self._crosswalk_enter_tolerance
            if (not in_crosswalk) and dist > monitor_distance:
                continue

            ahead = self._is_polygon_ahead(vehicle_pos, forward, polygon)
            active = bool(state.get("active", False))
            in_before_no_stop_zone = (
                not in_crosswalk
                and no_stop_before_m > 0.0
                and self._signed_distance_to_no_stop_zone(
                    vehicle=vehicle, point=vehicle_pos, polygon=polygon, no_stop_before_m=no_stop_before_m,
                ) <= self._crosswalk_enter_tolerance
            )
            in_yield_zone = active and (in_crosswalk or (ahead and dist <= yield_distance))
            if not in_before_no_stop_zone and not in_yield_zone and not in_crosswalk:
                continue

            key = (vehicle_id, str(crosswalk_id))
            v_state = self._vehicle_states.setdefault(
                key,
                {
                    "seen": False,
                    "stopped_before_crosswalk": False,
                    "violated": False,
                    "reported_for_reward": False,
                    "reported_for_metrics": False,
                },
            )
            v_state["seen"] = True
            if speed_kmh <= yield_speed_kmh:
                v_state["stopped_before_crosswalk"] = True

            # Occupied crosswalk is non-passable: entering it while pedestrian is active is a violation.
            if active and in_crosswalk:
                v_state["violated"] = True

            # Standing in no-stop zone (<=8m before crosswalk) is a violation.
            if in_before_no_stop_zone and speed_kmh <= no_stop_speed_kmh:
                v_state["stopped_before_crosswalk"] = True
                v_state["violated"] = True

            if bool(getattr(vehicle, "crash_human", False)):
                v_state["violated"] = True

            if v_state["violated"]:
                report_key = "reported_for_reward" if for_reward else "reported_for_metrics"
                if not v_state[report_key]:
                    v_state[report_key] = True
                    return True

        return False

    def should_vehicle_stop(self, vehicle) -> bool:
        """
        Helper for traffic enforcers (e.g. IDM override):
        returns True if vehicle should stop before an occupied crosswalk.
        """
        engine, crosswalk_state = self._get_crosswalk_state(vehicle)
        if engine is None or not crosswalk_state:
            return False
        yield_distance = self._resolve_all_thresholds(engine)["yield_distance"]
        relevant = self._get_relevant_active_crosswalks(vehicle, crosswalk_state, yield_distance)
        # All relevant crosswalks are already within yield_distance, so any match means stop.
        return len(relevant) > 0

    def get_status(self, vehicle) -> dict:
        """
        Lightweight verifier status for logging/debug overlays.
        Does not mutate internal violation state.
        """
        speed_kmh = float(getattr(vehicle, "speed_km_h", 0.0) or 0.0)
        status = {
            "rule_id": self.id,
            "speed_kmh": speed_kmh,
            "crosswalk_total_count": 0,
            "crosswalk_active_count": 0,
            "target_crosswalk_id": None,
            "target_distance_m": None,
            "target_active": False,
            "target_ahead": False,
            "in_crosswalk": False,
            "in_no_stop_zone": False,
            "in_yield_zone": False,
            "must_stop": False,
            "stopped_before_crosswalk": False,
            "stop_before_steps": 0,
            "min_stop_steps": 0,
            "no_stop_before_m": self._defaults["no_stop_before_m"],
            "yield_distance_m": self._defaults["yield_distance"],
            "violation_latched": False,
        }

        engine, crosswalk_state = self._get_crosswalk_state(vehicle)
        if engine is None or crosswalk_state is None:
            return status

        crosswalk_state = crosswalk_state or {}
        status["crosswalk_total_count"] = len(crosswalk_state)
        status["crosswalk_active_count"] = sum(1 for v in crosswalk_state.values() if v.get("active", False))
        if not crosswalk_state:
            return status

        thresholds = self._resolve_all_thresholds(engine)
        yield_distance = thresholds["yield_distance"]
        no_stop_before_m = thresholds["no_stop_before_m"]
        no_stop_min_steps = thresholds["no_stop_min_steps"]
        status["yield_distance_m"] = yield_distance
        status["no_stop_before_m"] = no_stop_before_m
        status["min_stop_steps"] = no_stop_min_steps

        vehicle_pos = self._vehicle_pos(vehicle)
        heading = float(getattr(vehicle, "heading_theta", 0.0) or 0.0)
        forward = np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float64)
        monitor_distance = max(yield_distance, no_stop_before_m) + 3.0

        best = self._find_target_crosswalk_status(
            vehicle_pos=vehicle_pos,
            forward=forward,
            crosswalk_state=crosswalk_state,
            yield_distance=yield_distance,
            no_stop_before_m=no_stop_before_m,
            monitor_distance=monitor_distance,
            vehicle=vehicle,
        )
        if best is None:
            return status

        vehicle_id = str(getattr(vehicle, "id", id(vehicle)))
        status["violation_latched"] = any(
            bool(v.get("violated", False))
            for (vid, _), v in self._vehicle_states.items()
            if str(vid) == vehicle_id
        )
        target_id = str(best["crosswalk_id"])
        state_key = (vehicle_id, target_id)
        v_state = self._vehicle_states.get(state_key, {})

        status.update(
            {
                "target_crosswalk_id": target_id,
                # Ped_Dist(m): signed distance to no-stop zone boundary (stop-line geometry).
                # >0 outside zone, ~=0 on boundary, <0 inside zone.
                "target_distance_m": float(best["no_stop_dist"]),
                "target_active": bool(best["active"]),
                "target_ahead": bool(best["ahead"]),
                "in_crosswalk": bool(best["in_crosswalk"]),
                "in_no_stop_zone": bool(best["in_before_no_stop_zone"]),
                "in_yield_zone": bool(best["in_yield_zone"]),
                "must_stop": bool(best["active"] and (best["in_crosswalk"] or (best["ahead"] and best["dist"] <= yield_distance))),
                "stopped_before_crosswalk": bool(v_state.get("stopped_before_crosswalk", False)),
                "stop_before_steps": int(v_state.get("stop_before_steps", 0)),
            }
        )
        return status

    @staticmethod
    def _valid_polygon(polygon: np.ndarray) -> bool:
        return polygon.ndim == 2 and polygon.shape[0] >= 3 and polygon.shape[1] >= 2

    def _resolve_all_thresholds(self, engine) -> dict:
        cfg = engine.global_config.get("pedestrian_manager", {})
        if hasattr(cfg, "get_dict"):
            cfg = cfg.get_dict()
        cfg = dict(cfg)
        dt = float(engine.global_config["physics_world_step_size"]) * float(engine.global_config["decision_repeat"])
        no_stop_min_duration_s = float(cfg.get("no_stop_min_duration_s", self._defaults["no_stop_min_duration_s"]))
        return {
            "yield_distance": float(cfg.get("yield_distance", self._defaults["yield_distance"])),
            "yield_speed_kmh": float(cfg.get("yield_speed_kmh", self._defaults["yield_speed_kmh"])),
            "no_stop_before_m": float(cfg.get("no_stop_before_crosswalk_m", self._defaults["no_stop_before_m"])),
            "no_stop_speed_kmh": float(cfg.get("no_stop_speed_kmh", self._defaults["no_stop_speed_kmh"])),
            "no_stop_min_steps": max(1, int(round(no_stop_min_duration_s / max(dt, 1e-3)))),
        }

    def _get_crosswalk_state(self, vehicle):
        engine = getattr(vehicle, "engine", None)
        if engine is None:
            return None, None

        map_state = self._map_crosswalk_state(engine)
        ped_manager = getattr(engine, "pedestrian_manager", None)
        if ped_manager is None or not getattr(ped_manager, "enabled", False):
            return engine, map_state
        try:
            state = ped_manager.get_active_crosswalk_state()
        except Exception:
            state = {}

        # Merge in map crosswalks so verifier still works when manager state is partial.
        if map_state:
            merged = dict(map_state)
            for key, value in (state or {}).items():
                merged[str(key)] = value
            state = merged
        return engine, state

    @staticmethod
    def _map_crosswalk_state(engine) -> Dict[str, dict]:
        current_map = getattr(engine, "current_map", None)
        crosswalks = getattr(current_map, "crosswalks", {}) if current_map is not None else {}
        ret: Dict[str, dict] = {}
        for crosswalk_id, feat in (crosswalks or {}).items():
            polygon = np.asarray(feat.get("polygon", []), dtype=np.float64)
            if not PedestrianYieldRule._valid_polygon(polygon):
                continue
            polygon = polygon[:, :2]
            ret[str(crosswalk_id)] = {
                "polygon": polygon,
                "center": np.mean(polygon, axis=0),
                "pedestrian_count": 0,
                "pedestrian_positions": [],
                "pedestrian_ids": [],
                "active": False,
            }
        return ret


    def _get_relevant_active_crosswalks(self, vehicle, crosswalk_state: Dict[str, dict], yield_distance: float) -> List[tuple]:
        vehicle_pos = self._vehicle_pos(vehicle)
        heading = float(getattr(vehicle, "heading_theta", 0.0) or 0.0)
        forward = np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float64)
        ret = []

        for crosswalk_id, state in crosswalk_state.items():
            if not bool(state.get("active", False)):
                continue
            polygon = np.asarray(state.get("polygon", []), dtype=np.float64)
            if not self._valid_polygon(polygon):
                continue

            dist = self._distance_to_polygon(vehicle_pos, polygon)
            if dist > yield_distance:
                continue

            in_crosswalk = self._point_in_polygon(vehicle_pos, polygon) or dist <= self._crosswalk_enter_tolerance
            # Ignore crosswalk clearly behind vehicle.
            if (not in_crosswalk) and (not self._is_polygon_ahead(vehicle_pos, forward, polygon, tolerance=-1.0)):
                continue

            ret.append((str(crosswalk_id), polygon, float(dist)))
        return ret

    def _find_target_crosswalk_status(
        self,
        vehicle_pos: np.ndarray,
        forward: np.ndarray,
        crosswalk_state: Dict[str, dict],
        yield_distance: float,
        no_stop_before_m: float,
        monitor_distance: float,
        vehicle=None,
    ) -> Optional[dict]:
        best = None
        for crosswalk_id, state in crosswalk_state.items():
            polygon = np.asarray(state.get("polygon", []), dtype=np.float64)
            if not self._valid_polygon(polygon):
                continue
            dist = self._distance_to_polygon(vehicle_pos, polygon)
            in_crosswalk = self._point_in_polygon(vehicle_pos, polygon) or dist <= self._crosswalk_enter_tolerance
            if (not in_crosswalk) and dist > monitor_distance:
                continue

            ahead = self._is_polygon_ahead(vehicle_pos, forward, polygon)
            active = bool(state.get("active", False))
            signed_no_stop_dist = self._signed_distance_to_no_stop_zone(
                vehicle=vehicle,
                point=vehicle_pos,
                polygon=polygon,
                no_stop_before_m=no_stop_before_m,
            )
            in_before_no_stop_zone = (
                not in_crosswalk
                and no_stop_before_m > 0.0
                and signed_no_stop_dist <= self._crosswalk_enter_tolerance
            )
            in_yield_zone = active and (in_crosswalk or (ahead and dist <= yield_distance))

            # Prefer active-ahead crosswalks, then active, then ahead, then nearest.
            priority = 3
            if active and ahead:
                priority = 0
            elif active:
                priority = 1
            elif ahead:
                priority = 2
            candidate = {
                "crosswalk_id": str(crosswalk_id),
                "dist": float(dist),
                "active": active,
                "ahead": ahead,
                "in_crosswalk": in_crosswalk,
                "in_before_no_stop_zone": in_before_no_stop_zone,
                "in_yield_zone": in_yield_zone,
                "no_stop_dist": float(signed_no_stop_dist),
                "priority": priority,
            }
            if best is None:
                best = candidate
                continue
            if (candidate["priority"], candidate["dist"]) < (best["priority"], best["dist"]):
                best = candidate
        return best

    def _cleanup_vehicle_states(
        self,
        vehicle_id: str,
        crosswalk_state: Dict[str, dict],
        vehicle_pos: np.ndarray,
        monitor_distance: float = 12.0,
    ):
        if vehicle_id is None:
            return
        vehicle_id = str(vehicle_id)
        to_delete = []
        for key in list(self._vehicle_states.keys()):
            vid, crosswalk_id = key
            if vid != vehicle_id:
                continue
            current = crosswalk_state.get(crosswalk_id, None)
            if current is None:
                to_delete.append(key)
                continue
            if vehicle_pos is None:
                continue
            polygon = np.asarray(current.get("polygon", []), dtype=np.float64)
            if not self._valid_polygon(polygon):
                to_delete.append(key)
                continue
            if self._distance_to_polygon(vehicle_pos, polygon) > (monitor_distance + 8.0):
                to_delete.append(key)
        for key in to_delete:
            self._vehicle_states.pop(key, None)

    @staticmethod
    def _vehicle_pos(vehicle) -> np.ndarray:
        return np.asarray(vehicle.position[:2], dtype=np.float64)

    @staticmethod
    def _is_polygon_ahead(
        vehicle_pos: np.ndarray, forward: np.ndarray, polygon: np.ndarray, tolerance: float = 0.0
    ) -> bool:
        polygon = np.asarray(polygon, dtype=np.float64)
        if not PedestrianYieldRule._valid_polygon(polygon):
            return False
        rel = polygon[:, :2] - np.asarray(vehicle_pos[:2], dtype=np.float64)
        proj = rel @ np.asarray(forward[:2], dtype=np.float64)
        return float(np.max(proj)) > float(tolerance)

    def _signed_distance_to_no_stop_zone(
        self,
        vehicle,
        point: np.ndarray,
        polygon: np.ndarray,
        no_stop_before_m: float,
    ) -> float:
        if float(no_stop_before_m) <= 0.0:
            return float("inf")
        probe_points = self._vehicle_probe_points(vehicle, fallback_point=point)
        zone_polygon = self._build_renderer_no_stop_zone_polygon(polygon, no_stop_before_m)
        if zone_polygon is None:
            # Fallback for non-quad polygons: keep old boundary-distance behavior.
            min_dist = min(self._distance_to_polygon(p, polygon) for p in probe_points)
            return float(min_dist - float(no_stop_before_m))
        min_dist = min(self._distance_to_polygon(p, zone_polygon) for p in probe_points)
        inside = any(self._point_in_polygon(p, zone_polygon) for p in probe_points)
        return float(-min_dist if inside else min_dist)

    @staticmethod
    def _build_renderer_no_stop_zone_polygon(polygon: np.ndarray, no_stop_before_m: float) -> Optional[np.ndarray]:
        """
        Build the same expanded quad used by top-down stop-line rendering:
        [a - f*d, b + f*d, c + f*d, d - f*d], where f = normalize((b-a) + (c-d)).
        """
        poly = PedestrianYieldRule._crosswalk_reference_quad(polygon)
        if poly is None:
            return None
        dist = float(no_stop_before_m)
        a, b, c, d = poly
        forward = (b - a) + (c - d)
        norm = float(np.linalg.norm(forward))
        if norm <= 1e-6:
            return None
        forward = forward / norm
        return np.asarray([a - forward * dist, b + forward * dist, c + forward * dist, d - forward * dist], dtype=np.float64)

    @staticmethod
    def _crosswalk_reference_quad(polygon: np.ndarray) -> Optional[np.ndarray]:
        """
        Build a stable 4-point quad for crosswalk geometry calculations.
        SUMO crossing polygons may contain many points; find the longest edge
        as primary axis, then build an oriented bounding rectangle from
        min/max projections onto that axis and its perpendicular.
        """
        poly = np.asarray(polygon, dtype=np.float64)
        if poly.ndim != 2 or poly.shape[1] < 2 or poly.shape[0] < 4:
            return None
        poly = poly[:, :2]
        if np.linalg.norm(poly[0] - poly[-1]) < 1e-6:
            poly = poly[:-1]
        if poly.shape[0] < 4:
            return None
        if poly.shape[0] == 4:
            return poly

        # Use the longest edge as the primary axis direction.
        n = len(poly)
        best_len, best_axis = 0.0, None
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            length = float(np.linalg.norm(edge))
            if length > best_len:
                best_len, best_axis = length, edge
        if best_axis is None or best_len < 1e-6:
            return poly[:4]

        axis_long = best_axis / best_len
        axis_short = np.array([-axis_long[1], axis_long[0]])

        center = np.mean(poly, axis=0)
        centered = poly - center
        proj_long = centered @ axis_long
        proj_short = centered @ axis_short
        l_min, l_max = float(np.min(proj_long)), float(np.max(proj_long))
        s_min, s_max = float(np.min(proj_short)), float(np.max(proj_short))

        a = center + axis_long * l_min + axis_short * s_min
        b = center + axis_long * l_min + axis_short * s_max
        c = center + axis_long * l_max + axis_short * s_max
        d = center + axis_long * l_max + axis_short * s_min
        return np.asarray([a, b, c, d], dtype=np.float64)

    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        if polygon is None:
            return False
        polygon = np.asarray(polygon, dtype=np.float64)
        if not PedestrianYieldRule._valid_polygon(polygon):
            return False
        x = float(point[0])
        y = float(point[1])
        inside = False
        n = len(polygon)
        j = n - 1
        for i in range(n):
            xi, yi = float(polygon[i][0]), float(polygon[i][1])
            xj, yj = float(polygon[j][0]), float(polygon[j][1])
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            if intersect:
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _distance_to_polygon(point: np.ndarray, polygon: np.ndarray) -> float:
        if polygon is None:
            return float("inf")
        polygon = np.asarray(polygon, dtype=np.float64)
        if not PedestrianYieldRule._valid_polygon(polygon):
            return float("inf")
        if PedestrianYieldRule._point_in_polygon(point, polygon):
            return 0.0
        p = np.asarray(point[:2], dtype=np.float64)
        n = len(polygon)
        return min(
            PedestrianYieldRule._distance_point_to_segment(
                p,
                np.asarray(polygon[i][:2], dtype=np.float64),
                np.asarray(polygon[(i + 1) % n][:2], dtype=np.float64),
            )
            for i in range(n)
        )

    @staticmethod
    def _distance_point_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-9:
            return float(np.linalg.norm(point - a))
        t = float(np.dot(point - a, ab) / denom)
        t = min(1.0, max(0.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(point - proj))

    @staticmethod
    def _vehicle_probe_points(vehicle, fallback_point: np.ndarray) -> List[np.ndarray]:
        """Use footprint corners (if available) to align zone checks with rendered stop lines."""
        points: List[np.ndarray] = [np.asarray(fallback_point[:2], dtype=np.float64)]
        if vehicle is None:
            return points
        try:
            bb = getattr(vehicle, "bounding_box", None)
            if bb is not None:
                for p in bb:
                    arr = np.asarray(p, dtype=np.float64).reshape(-1)
                    if arr.size >= 2:
                        points.append(arr[:2])
        except Exception:
            pass
        return points
