from typing import Optional, Tuple, Any
import numpy as np
from traffic_signs.base_traffic_sign import BaseTrafficSign


def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


class SpeedLimitFeatureExtractor:
    """
    features: in_zone, ego_speed, limit_speed, overspeed, dist_to_end  
    """
    IDX_IN_ZONE = 0
    IDX_EGO_SPEED = 1
    IDX_LIMIT_SPEED = 2
    IDX_OVERSPEED = 3
    IDX_DIST_TO_END = 4

    N_FEATURES = 5

    def __init__(
        self,
        dist_norm_max_m: float = 200.0,
        speed_norm_max_mps: float = 20.0,
    ):
        self.dist_norm_max_m = dist_norm_max_m
        self.speed_norm_max_mps = speed_norm_max_mps
        # Active speed limit persists until replaced by another speed sign.
        self._active_limit_kmh: Optional[float] = None
        self._active_lane_index = None
        self._active_zone_end = None
        self._active_heading = None

    def reset(self):
        self._active_limit_kmh = None
        self._active_lane_index = None
        self._active_zone_end = None
        self._active_heading = None

    def _iter_speed_signs(self, engine):
        manager = getattr(engine, "traffic_sign_manager", None)
        if manager is None or not hasattr(manager, "signs"):
            return []
        return [
            s for s in manager.signs
            if hasattr(s, "speed_limit") and hasattr(s, "zone_start") and hasattr(s, "zone_end")
        ]

    def _select_sign(self, engine, agent) -> Tuple[Optional[Any], Optional[float], Optional[float]]:
        best_sign = None
        best_dist = float("inf")
        best_long = None
        best_in_zone = False

        for sign in self._iter_speed_signs(engine):
            if not sign.is_in_drivable_area(agent):
                continue

            lane = getattr(sign, "lane", None)
            if lane is None:
                continue
            try:
                veh_long, _ = lane.local_coordinates(agent.position)
            except Exception:
                continue

            zone_start = getattr(sign, "zone_start", None)
            zone_end = getattr(sign, "zone_end", None)
            if zone_start is None or zone_end is None:
                continue

            in_zone = zone_start <= veh_long <= zone_end
            if in_zone:
                dist = zone_end - veh_long
                if dist < best_dist:
                    best_sign = sign
                    best_dist = dist
                    best_long = veh_long
                    best_in_zone = True
            elif not best_in_zone and veh_long < zone_start:
                dist = zone_start - veh_long
                if dist < best_dist:
                    best_sign = sign
                    best_dist = dist
                    best_long = veh_long

        return best_sign, best_long, best_dist

    def step_features(self, engine, agent) -> np.ndarray:
        sign, veh_long, _ = self._select_sign(engine, agent)
        ego_speed = float(getattr(agent, "speed", 0.0) or 0.0)
        if not np.isfinite(ego_speed) or ego_speed < 0:
            ego_speed = 0.0

        in_zone = 0.0
        limit_mps = 0.0
        overspeed = 0.0
        dist_to_end = self.dist_norm_max_m

        active_updated_this_step = False
        if sign is not None and veh_long is not None:
            zone_start = getattr(sign, "zone_start", None)
            zone_end = getattr(sign, "zone_end", None)
            if zone_start is not None and zone_end is not None:
                if zone_start <= veh_long <= zone_end:
                    limit_kmh = float(getattr(sign, "speed_limit", 0.0) or 0.0)
                    if limit_kmh > 0.0:
                        self._active_limit_kmh = limit_kmh
                        self._active_lane_index = getattr(getattr(sign, "lane", None), "index", None)
                        # Speed limit acts until end of the lane.
                        self._active_zone_end = getattr(getattr(sign, "lane", None), "length", None)
                        try:
                            self._active_heading = float(sign.lane.heading_theta_at(sign.placement_long))
                        except Exception:
                            self._active_heading = None
                        active_updated_this_step = True
                    dist_to_end = max(0.0, zone_end - veh_long)

        # Validate active limit against current lane direction and lane end.
        keep_active = self._active_limit_kmh is not None and self._active_limit_kmh > 0.0
        if keep_active:
            try:
                agent_lane = getattr(agent, "lane", None)
                agent_lane_idx = getattr(agent_lane, "index", None) if agent_lane is not None else None
                if self._active_lane_index is not None and agent_lane_idx is not None:
                    if not BaseTrafficSign._same_road_direction(agent_lane_idx, self._active_lane_index):
                        # Different road direction; keep active unless heading indicates a turn.
                        pass
                if self._active_heading is not None:
                    agent_heading = getattr(agent, "heading_theta", None)
                    if agent_heading is not None:
                        diff = abs(BaseTrafficSign._wrap_pi(self._active_heading - float(agent_heading)))
                        if diff > 1.3:
                            keep_active = False
            except Exception:
                pass

        if keep_active:
            # If we can compute longitudinal along current lane, enforce end-of-lane cutoff.
            if self._active_zone_end is not None:
                try:
                    agent_lane = getattr(agent, "lane", None)
                    if agent_lane is not None:
                        curr_long = float(agent_lane.local_coordinates(agent.position)[0])
                        # Apply end-of-lane cutoff only if still on the same road direction.
                        agent_lane_idx = getattr(agent_lane, "index", None)
                        if self._active_lane_index is None or agent_lane_idx is None:
                            pass
                        elif BaseTrafficSign._same_road_direction(agent_lane_idx, self._active_lane_index):
                            if curr_long > float(self._active_zone_end) + 0.1:
                                keep_active = False
                except Exception:
                    pass

        if keep_active:
            in_zone = 1.0
            limit_mps = self._active_limit_kmh / 3.6
            overspeed = max(0.0, ego_speed - limit_mps)
            if not active_updated_this_step:
                dist_to_end = self.dist_norm_max_m
        else:
            self._active_limit_kmh = None
            self._active_lane_index = None
            self._active_zone_end = None
            self._active_heading = None

        return np.array([
            in_zone,
            clamp01(ego_speed / self.speed_norm_max_mps),
            clamp01(limit_mps / self.speed_norm_max_mps),
            clamp01(overspeed / self.speed_norm_max_mps),
            clamp01(dist_to_end / self.dist_norm_max_m),
        ], dtype=np.float32)
