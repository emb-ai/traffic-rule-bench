"""
4.2.1 — pass obstacle on the right
4.2.2 — pass obstacle on the left
4.2.3 — pass on either side
"""

import numpy as np
from traffic_signs.base_traffic_sign import BaseTrafficSign


class DetourSign(BaseTrafficSign):
    """
    Base class for detour signs (4.2.x).

    The sign is placed on a lane where an obstacle exists.
    The vehicle must change to an adjacent lane in the prescribed direction
    before reaching the obstacle point.

    Subclasses set ``allowed_directions`` to ``{"right"}``, ``{"left"}``,
    or ``{"left", "right"}``.
    """

    allowed_directions: set  # set by subclasses

    # How far ahead of the obstacle the vehicle must have already changed lane.
    OBSTACLE_OFFSET = 2.0
    # Distance from sign icon to the centre of the cone cluster.
    # The first cone is at obstacle_long − 2.25 m, so the sign sits ~1.25 m
    # before the first cone (3.5 − 2.25 = 1.25).
    SIGN_TO_OBSTACLE = 3.5
    # Zone parameters (relative to obstacle position)
    ZONE_BEFORE = 30.0
    ZONE_AFTER = 15.0

    def __init__(self, lane, zone_length=None, icon_path=None, **kwargs):
        # Place sign before the cone cluster (about 1.25 m before first cone).
        # With SIGN_TO_OBSTACLE=3.5 and cone half-span=2.25:
        # sign -> first cone gap = (SIGN_TO_OBSTACLE + 3.0) - 2.25 = 1.25 m.
        kwargs.setdefault("longitudinal_offset", -(self.SIGN_TO_OBSTACLE + 3.0))
        # Place sign at the centre of the lane (lateral_offset=0).
        kwargs.setdefault("lateral_offset", 0)
        super().__init__(lane, icon_path=icon_path, **kwargs)
        self._vehicle_states_detour = {}
        self._violation_events = 0

        # Obstacle cones are placed ahead of the sign.
        self.obstacle_long = min(
            float(self.placement_long) + self.SIGN_TO_OBSTACLE,
            self.lane.length - 0.5,
        )
        self.zone_start = max(0.0, self.obstacle_long - self.ZONE_BEFORE)
        if zone_length is not None:
            self.zone_end = self.zone_start + max(0.0, float(zone_length))
        else:
            self.zone_end = min(self.lane.length, self.obstacle_long + self.ZONE_AFTER)
        self.zone_length = self.zone_end - self.zone_start
        # If obstacle is near lane end, obstacle_long + offset may exceed lane length.
        # Clamp the violation trigger so it remains reachable.
        self.violation_long = min(self.lane.length - 0.1, self.obstacle_long + self.OBSTACLE_OFFSET)

        # Resolve allowed adjacent lane indices from the road network.
        self._allowed_lane_indices = set()
        self._resolve_allowed_lanes()

    # ------------------------------------------------------------------
    # Lane resolution helpers
    # ------------------------------------------------------------------

    def _resolve_allowed_lanes(self):
        """Populate ``_allowed_lane_indices`` using the road-network graph.

        Supports both EdgeRoadNetwork (SUMO maps, lane_info with
        ``left_lanes`` / ``right_lanes``) and NodeRoadNetwork (PG maps,
        index = ``(from_node, to_node, lane_num)``).
        """
        road_network = None
        try:
            road_network = self.engine.current_map.road_network
        except Exception:
            pass
        if road_network is None:
            return

        sign_lane_index = getattr(self.lane, "index", None)
        if sign_lane_index is None:
            return

        graph = getattr(road_network, "graph", None)
        if graph is None:
            return

        lane_info = graph.get(sign_lane_index)
        if lane_info is not None:
            if "right" in self.allowed_directions:
                right_lanes = getattr(lane_info, "right_lanes", None) or []
                for rl in right_lanes:
                    self._allowed_lane_indices.add(rl)
            if "left" in self.allowed_directions:
                left_lanes = getattr(lane_info, "left_lanes", None) or []
                for ll in left_lanes:
                    self._allowed_lane_indices.add(ll)
            return

        if isinstance(sign_lane_index, tuple) and len(sign_lane_index) == 3:
            from_node, to_node, lane_num = sign_lane_index
            road_lanes = graph.get(from_node, {}).get(to_node, [])
            num_lanes = len(road_lanes)
            if "right" in self.allowed_directions and lane_num < num_lanes - 1:
                self._allowed_lane_indices.add((from_node, to_node, lane_num + 1))
            if "left" in self.allowed_directions and lane_num > 0:
                self._allowed_lane_indices.add((from_node, to_node, lane_num - 1))

    @property
    def detour_feasible(self) -> bool:
        """``True`` if at least one adjacent lane exists in the prescribed direction."""
        return bool(self._allowed_lane_indices)
    @property
    def top_down_icon_rotation_deg(self) -> float:
        """pygame rotation angle (degrees, CCW) to orient the sign correctly"""
        h = np.rad2deg(getattr(self, "_heading_theta", 0.0))
        return float(h - 180.0)

    def _get_vehicle_lane_index(self, vehicle):
        lane = getattr(vehicle, "lane", None)
        if lane is not None:
            idx = getattr(lane, "index", None)
            if idx is not None:
                return idx
        return getattr(vehicle, "lane_index", None)

    def check_violation(self, vehicle, for_reward=False) -> bool:
        """Per-step violation flag.

        Returns True on EVERY step the vehicle is inside the zone of effect
        without having changed to an allowed adjacent lane. ``_violation_events``
        is still edge-counted (incremented only on the not-violating →
        violating transition) so existing event-based metrics keep their
        semantics; the per-step return value drives ``violations_by_class_step``
        in the runner — that becomes "steps spent in the zone without proper
        lane change".
        """
        violating_now = bool(self._is_violating(vehicle))
        vid = vehicle.id
        state = self._vehicle_states_detour.setdefault(
            vid,
            {
                "entered_zone": False,
                "changed_correctly": False,
                "violation_active": False,
            },
        )
        was_violating = bool(state.get("violation_active", False))
        state["violation_active"] = violating_now
        if violating_now and not was_violating:
            self._violation_events += 1
        return violating_now

    def _is_violating(self, vehicle) -> bool:
        # An infeasible detour (no adjacent lane in the prescribed direction)
        # can never be satisfied, so it must not count as a violation.
        if not self._allowed_lane_indices:
            return False
        if not self.is_in_drivable_area(vehicle):
            return False

        try:
            veh_long = self.lane.local_coordinates(vehicle.position)[0]
        except Exception:
            return False

        # inside the zone
        if not (self.zone_start <= veh_long <= self.zone_end):
            return False

        vid = vehicle.id
        if vid not in self._vehicle_states_detour:
            self._vehicle_states_detour[vid] = {
                "entered_zone": False,
                "changed_correctly": False,
                "violation_active": False,
            }

        state = self._vehicle_states_detour[vid]

        # If agent moved back before the zone, restart detour evaluation.
        if state["entered_zone"] and veh_long < self.zone_start - 1.0:
            state["entered_zone"] = False
            state["changed_correctly"] = False

        sign_lane_index = getattr(self.lane, "index", None)
        current_lane_index = self._get_vehicle_lane_index(vehicle)

        if not state["entered_zone"] and veh_long >= self.zone_start:
            state["entered_zone"] = True

        if (
            current_lane_index is not None
            and current_lane_index != sign_lane_index
            and self._allowed_lane_indices
            and current_lane_index in self._allowed_lane_indices
        ):
            state["changed_correctly"] = True

        # Violating on every step inside the zone while not yet changed
        # correctly. No per-step "compliance credit" is granted to vehicles
        # that haven't yet performed the prescribed lane change.
        return state["entered_zone"] and not state["changed_correctly"]
    
    def get_rule_description(self) -> str:
        return "Sign 4.2.x (obstacle detour) — pass the obstacle in the prescribed direction"

    @property
    def violation_events(self) -> int:
        return int(self._violation_events)

    @property
    def top_down_color(self):
        return [0, 100, 255]

    @property
    def top_down_color_name(self):
        return "blue"


class DetourRightSign(DetourSign):
    """4.2.1 — Detour obstacle on the right."""

    allowed_directions = {"right"}

    def __init__(self, lane, **kwargs):
        kwargs.setdefault("icon_path", "4.2.1.png")
        super().__init__(lane, **kwargs)

    def get_rule_description(self) -> str:
        return (
            "Sign 4.2.1 (detour to the right) — "
            "passing is allowed on the right only"
        )



class DetourLeftSign(DetourSign):
    """4.2.2 — Detour obstacle on the left."""

    allowed_directions = {"left"}

    def __init__(self, lane, **kwargs):
        kwargs.setdefault("icon_path", "4.2.2.png")
        super().__init__(lane, **kwargs)

    def get_rule_description(self) -> str:
        return (
            "Sign 4.2.2 (detour to the left) — "
            "passing is allowed on the left only"
        )


class DetourEitherSign(DetourSign):
    """4.2.3 — Detour obstacle on the right or left."""

    allowed_directions = {"left", "right"}

    def __init__(self, lane, **kwargs):
        kwargs.setdefault("icon_path", "4.2.3.png")
        super().__init__(lane, **kwargs)

    def get_rule_description(self) -> str:
        return (
            "Sign 4.2.3 (detour right or left) — "
            "passing is allowed on either side"
        )



__all__ = ["DetourSign", "DetourRightSign", "DetourLeftSign", "DetourEitherSign"]