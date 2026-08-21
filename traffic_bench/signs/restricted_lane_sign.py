"""Signs 5.11–5.14: lanes for specific vehicle categories (code index).

5.11.1 — road with route-vehicle (bus) lane
5.11.2 — road with bicycle lane
5.12.1 — end of road with route-vehicle lane
5.12.2 — end of road with bicycle lane
5.13.1 / 5.13.2 — exit to road with route-vehicle lane
5.13.3 / 5.13.4 — exit to road with bicycle lane
5.14.1 — route-vehicle lane
5.14.2 — bicycle lane
5.14.3 — end of route-vehicle lane
5.14.4 — end of bicycle lane

Semantic families:

  5.11.x — Road-level, OPPOSITE-direction restricted lane (counter-flow).
  5.12.x — End of that road-level counter-flow mode.
  5.13.x — Intersection-only turn restriction signs.
           5.13.1/3: forbid left turn; 5.13.2/4: forbid right turn.
           U-turn always allowed.  NOT begin-of-road signs.
  5.14.x — Lane-level, SAME-direction restricted lane.
           5.14.1/2 = begin; 5.14.3/4 = end.

5.14.x numbering (website vs actual):
  website 5.14   = actual 5.14.1
  website 5.14.1 = actual 5.14.3
  website 5.14.2 = actual 5.14.2
  website 5.14.3 = actual 5.14.4

Active signs (detect violations): 5.11.x, 5.13.x, 5.14.1, 5.14.2
End signs (informational only):   5.12.x, 5.14.3, 5.14.4
"""

import math
import re
from enum import Enum
from traffic_bench.signs.base_traffic_sign import BaseTrafficSign


# ======================================================================
# Semantic enums
# ======================================================================

class LaneUser(Enum):
    BUS = "bus"
    BICYCLE = "bicycle"


class FlowDirection(Enum):
    OPPOSITE = "opposite"   # counter-flow (5.11.x, 5.12.x, 5.13.x)
    SAME = "same"           # same direction (5.14.x)


class SignScope(Enum):
    ROAD = "road"                  # road segment (5.11.x, 5.12.x)
    INTERSECTION = "intersection"  # intersection only (5.13.x)
    LANE = "lane"                  # single lane (5.14.x)


class SignEffect(Enum):
    BEGIN = "begin"
    END = "end"
    INTERSECTION_EXIT = "intersection_exit"


# ======================================================================
# Declarative tables
# ======================================================================

# (lane_user, flow_direction, scope, effect)
SIGN_SEMANTICS = {
    "5.11.1": (LaneUser.BUS,     FlowDirection.OPPOSITE, SignScope.ROAD,         SignEffect.BEGIN),
    "5.11.2": (LaneUser.BICYCLE, FlowDirection.OPPOSITE, SignScope.ROAD,         SignEffect.BEGIN),
    "5.12.1": (LaneUser.BUS,     FlowDirection.OPPOSITE, SignScope.ROAD,         SignEffect.END),
    "5.12.2": (LaneUser.BICYCLE, FlowDirection.OPPOSITE, SignScope.ROAD,         SignEffect.END),
    "5.13.1": (LaneUser.BUS,     FlowDirection.OPPOSITE, SignScope.INTERSECTION, SignEffect.INTERSECTION_EXIT),
    "5.13.2": (LaneUser.BUS,     FlowDirection.OPPOSITE, SignScope.INTERSECTION, SignEffect.INTERSECTION_EXIT),
    "5.13.3": (LaneUser.BICYCLE, FlowDirection.OPPOSITE, SignScope.INTERSECTION, SignEffect.INTERSECTION_EXIT),
    "5.13.4": (LaneUser.BICYCLE, FlowDirection.OPPOSITE, SignScope.INTERSECTION, SignEffect.INTERSECTION_EXIT),
    "5.14.1": (LaneUser.BUS,     FlowDirection.SAME,     SignScope.LANE,         SignEffect.BEGIN),
    "5.14.2": (LaneUser.BICYCLE, FlowDirection.SAME,     SignScope.LANE,         SignEffect.BEGIN),
    "5.14.3": (LaneUser.BUS,     FlowDirection.SAME,     SignScope.LANE,         SignEffect.END),
    "5.14.4": (LaneUser.BICYCLE, FlowDirection.SAME,     SignScope.LANE,         SignEffect.END),
}

_SIGN_DESCRIPTIONS = {
    "5.11.1": "Sign 5.11.1 'Road with bus lane' — dedicated lane in the opposite direction, entry forbidden",
    "5.11.2": "Sign 5.11.2 'Road with bicycle lane' — dedicated lane in the opposite direction, entry forbidden",
    "5.12.1": "Sign 5.12.1 'End of road with bus lane'",
    "5.12.2": "Sign 5.12.2 'End of road with bicycle lane'",
    "5.13.1": "Sign 5.13.1 'Exit onto road with bus lane' — left turn forbidden at the intersection",
    "5.13.2": "Sign 5.13.2 'Exit onto road with bus lane' — right turn forbidden at the intersection",
    "5.13.3": "Sign 5.13.3 'Exit onto road with bicycle lane' — left turn forbidden at the intersection",
    "5.13.4": "Sign 5.13.4 'Exit onto road with bicycle lane' — right turn forbidden at the intersection",
    "5.14.1": "Sign 5.14.1 'Bus lane' — co-directional dedicated lane, other vehicles forbidden",
    "5.14.2": "Sign 5.14.2 'Bicycle lane' — co-directional dedicated lane, other vehicles forbidden",
    "5.14.3": "Sign 5.14.3 'End of bus lane'",
    "5.14.4": "Sign 5.14.4 'End of bicycle lane'",
}


# ======================================================================
# Helpers
# ======================================================================

def _is_tuple_lane_index(idx):
    """True if idx looks like a MetaDrive (from, to, num) lane index."""
    if idx is None or isinstance(idx, str):
        return False
    try:
        return len(idx) >= 3
    except TypeError:
        return False


def _parse_sumo_lane_index(lane_idx):
    """Parse SUMO lane index string into (edge_id, lane_num) or None.
    Handles both 'lane_<edge>_<num>' and '<edge>_<num>' formats.
    """
    if not isinstance(lane_idx, str):
        return None
    raw = lane_idx[5:] if lane_idx.startswith("lane_") else lane_idx
    # Skip internal/junction lanes (contain ':')
    if ":" in raw:
        return None
    parts = raw.rsplit("_", 1)
    if len(parts) != 2:
        return None
    try:
        return parts[0], int(parts[1])
    except (ValueError, IndexError):
        return None


def _is_max_lane_index_for(lane_idx, engine):
    """Check if lane_idx is the rightmost (max index) on its road segment.
    Returns True (fail-open) when the check cannot be performed.
    """
    # PG maps: tuple index (from_node, to_node, lane_num)
    if _is_tuple_lane_index(lane_idx):
        try:
            graph = engine.current_map.road_network.graph
            if graph is None:
                return True
            lanes = graph.get(lane_idx[0], {}).get(lane_idx[1], [])
            return not lanes or lane_idx[2] == len(lanes) - 1
        except Exception:
            return True

    # SUMO: string index "lane_<edge>_<num>" or "<edge>_<num>"
    # SUMO convention: lane 0 is the rightmost; indices increase to the left.
    parsed = _parse_sumo_lane_index(lane_idx)
    if parsed is not None:
        _, lane_num = parsed
        return lane_num == 0

    return True  # fail-open


def _get_vehicle_lane_index(vehicle):
    """Best-effort lane index fetch from vehicle."""
    lane = getattr(vehicle, "lane", None)
    if lane is not None:
        idx = getattr(lane, "index", None)
        if idx is not None:
            return idx
    return getattr(vehicle, "lane_index", None)


def _wrap_pi(x):
    """Wrap angle to [-pi, pi]."""
    return float((x + math.pi) % (2 * math.pi) - math.pi)


# PG node parsing
_PG_NODE_RE = re.compile(r'-?(\d+)([A-Za-z$])(\d+)_(\d+)_')
_INTERSECTION_BLOCK_TYPES = frozenset({"X", "T"})


def _pg_node_parse(node_str):
    """Parse PG node string into component dict or None.
    '2X1_0_' -> {'block_index': 2, 'block_type': 'X', 'part_index': 1, 'road_index': 0}
    """
    m = _PG_NODE_RE.match(str(node_str))
    if not m:
        return None
    return {
        'block_index': int(m.group(1)),
        'block_type': m.group(2).upper(),
        'part_index': int(m.group(3)),
        'road_index': int(m.group(4)),
    }


def _pg_node_block_type(node_str):
    """Extract block type letter from PG node string, or None."""
    info = _pg_node_parse(node_str)
    return info['block_type'] if info else None


def _pg_intersection_block_key(node_str):
    """Return (block_type, block_index) if node belongs to an X/T intersection, else None."""
    info = _pg_node_parse(str(node_str))
    if info and info['block_type'] in _INTERSECTION_BLOCK_TYPES:
        return (info['block_type'], info['block_index'])
    return None


def _auto_fill_semantics(cls):
    """Auto-populate LANE_USER, FLOW_DIRECTION, SIGN_SCOPE, SIGN_EFFECT
    from SIGN_SEMANTICS when a subclass defines SIGN_CODE."""
    code = getattr(cls, 'SIGN_CODE', None)
    if code and code in SIGN_SEMANTICS:
        user, flow, scope, effect = SIGN_SEMANTICS[code]
        for attr, val in [('LANE_USER', user), ('FLOW_DIRECTION', flow),
                          ('SIGN_SCOPE', scope), ('SIGN_EFFECT', effect)]:
            if getattr(cls, attr, None) is None:
                setattr(cls, attr, val)


# ======================================================================
# Base: active restricted-lane sign (5.11.x, 5.14.x)
# ======================================================================

class RestrictedLaneSign(BaseTrafficSign):
    """Base for signs that actively restrict a lane (5.11.x, 5.14.x).

    Violation: vehicle on the restricted (rightmost) lane within the active zone.
    For LANE-scoped signs (5.14.x), restriction is continuous across segments.
    """

    SIGN_CODE = None
    LANE_USER = None
    FLOW_DIRECTION = None
    SIGN_SCOPE = None
    SIGN_EFFECT = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _auto_fill_semantics(cls)

    def __init__(self, lane, zone_length=None, icon_path=None, **kwargs):
        kwargs.setdefault("longitudinal_offset", -lane.length)
        kwargs["lateral_offset"] = 0
        if icon_path is None and self.SIGN_CODE:
            icon_path = f"z{self.SIGN_CODE}.png"
        super().__init__(lane, icon_path=icon_path, **kwargs)

        self.zone_start = float(self.placement_long)
        if zone_length is not None:
            self.zone_end = min(self.lane.length,
                                self.zone_start + max(0.0, float(zone_length)))
        else:
            self.zone_end = self.lane.length
        self.zone_length = self.zone_end - self.zone_start

    @staticmethod
    def _get_vehicle_lane_index(vehicle):
        return _get_vehicle_lane_index(vehicle)

    def _is_max_lane_index(self):
        return _is_max_lane_index_for(
            getattr(self.lane, "index", None),
            getattr(self, "engine", None),
        )

    # -- 5.14.x lane continuity --

    def _get_continuous_lane_indices(self):
        """For LANE-scoped signs, follow lane graph forward to find all
        connected lane indices. Caches the result."""
        cached = getattr(self, "_continuous_lanes_cache", None)
        if cached is not None:
            return cached

        sign_lane = getattr(self.lane, "index", None)
        if self.SIGN_SCOPE != SignScope.LANE:
            result = {sign_lane} if sign_lane is not None else set()
            self._continuous_lanes_cache = result
            return result

        # SUMO: string lane index
        if isinstance(sign_lane, str):
            continuous = {sign_lane}
            try:
                graph = self.engine.current_map.road_network.graph
                if graph is not None:
                    self._follow_lane_forward_sumo(graph, sign_lane, continuous)
            except Exception:
                pass
            self._continuous_lanes_cache = continuous
            return continuous

        # PG: tuple lane index
        if not _is_tuple_lane_index(sign_lane):
            result = {sign_lane} if sign_lane is not None else set()
            self._continuous_lanes_cache = result
            return result

        continuous = {sign_lane}
        try:
            graph = self.engine.current_map.road_network.graph
            if graph is not None:
                self._follow_lane_forward(graph, sign_lane, continuous)
        except Exception:
            pass

        self._continuous_lanes_cache = continuous
        return continuous

    @staticmethod
    def _follow_lane_forward(graph, start_idx, result_set, max_depth=50):
        current_to = start_idx[1]
        lane_num = start_idx[2]
        visited_roads = {(start_idx[0], start_idx[1])}

        for _ in range(max_depth):
            next_segments = graph.get(current_to, {})
            found = False
            for next_to, lanes in next_segments.items():
                road_key = (current_to, next_to)
                if road_key in visited_roads:
                    continue
                if lane_num < len(lanes):
                    next_idx = getattr(lanes[lane_num], "index", None)
                    if next_idx is not None:
                        result_set.add(next_idx)
                        visited_roads.add(road_key)
                        current_to = next_to
                        found = True
                        break
            if not found:
                break

    @staticmethod
    def _follow_lane_forward_sumo(graph, start_idx, result_set, max_depth=50):
        """Follow exit_lanes in SUMO graph, keeping the same lane number."""
        parsed = _parse_sumo_lane_index(start_idx)
        if parsed is None:
            return
        _, lane_num = parsed
        current = start_idx
        visited = {start_idx}

        for _ in range(max_depth):
            info = graph.get(current)
            if info is None or not hasattr(info, "exit_lanes") or not info.exit_lanes:
                break
            found = False
            for exit_lane in info.exit_lanes:
                if exit_lane in visited:
                    continue
                exit_parsed = _parse_sumo_lane_index(exit_lane)
                if exit_parsed is None:
                    continue
                # Follow lane with the same lane number on the next edge
                if exit_parsed[1] == lane_num:
                    result_set.add(exit_lane)
                    visited.add(exit_lane)
                    current = exit_lane
                    found = True
                    break
            if not found:
                break

    # -- Violation detection --

    def _is_violating(self, vehicle) -> bool:
        if not self.is_in_drivable_area(vehicle):
            return False

        current_lane = _get_vehicle_lane_index(vehicle)
        if current_lane is None:
            return False

        sign_lane = getattr(self.lane, "index", None)

        if self.SIGN_SCOPE == SignScope.LANE:
            target_lanes = self._get_continuous_lane_indices()
        else:
            target_lanes = {sign_lane} if sign_lane is not None else set()

        if current_lane not in target_lanes:
            return False

        if not _is_max_lane_index_for(current_lane, getattr(self, "engine", None)):
            return False

        if current_lane == sign_lane:
            try:
                veh_long = self.lane.local_coordinates(vehicle.position)[0]
            except Exception:
                return False
            if not (self.zone_start <= veh_long <= self.zone_end):
                return False

        return True

    def get_rule_description(self) -> str:
        return _SIGN_DESCRIPTIONS.get(self.SIGN_CODE, f"Sign {self.SIGN_CODE}")

    @property
    def top_down_color(self):
        return [0, 100, 210]

    @property
    def top_down_color_name(self):
        return "blue"


# ======================================================================
# Base: intersection-only turn-restriction sign (5.13.x)
# ======================================================================

class IntersectionRestrictedLaneSign(BaseTrafficSign):
    """Base for 5.13.x intersection-only turn-restriction signs.

    Verification uses lane-transition tracking:
    1. Vehicle on approach lane -> start tracking
    2. Track through intermediate intersection lanes
    3. On known exit lane -> check if forbidden
    4. Heading-based fallback when no turns data available

    NOT a begin-of-road sign.  Must not be confused with 5.11.x or 5.14.x.
    """

    SIGN_CODE = None
    LANE_USER = None
    FLOW_DIRECTION = FlowDirection.OPPOSITE
    SIGN_SCOPE = SignScope.INTERSECTION
    SIGN_EFFECT = SignEffect.INTERSECTION_EXIT
    FORBIDDEN_TURNS = frozenset()
    REQUIRED_EXIT_LANE_USER = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _auto_fill_semantics(cls)

    def __init__(self, lane, icon_path=None, **kwargs):
        if "longitudinal_offset" not in kwargs:
            kwargs["longitudinal_offset"] = -(lane.length - 2.0)
        if "lateral_offset" not in kwargs:
            _long = lane.length + kwargs["longitudinal_offset"]
            _long = max(0.1, min(_long, lane.length - 0.1))
            _hw = (lane.width_at(_long) if hasattr(lane, "width_at") else 3.7) / 2
            kwargs["lateral_offset"] = -(_hw + 0.8)
        self._active_agents = {}

        self.forbidden_to_lanes = set()
        self.allowed_to_lanes = set()
        self._approach_lane_indices = set()
        self._collect_turns_from_lane(lane)

        if icon_path is None and self.SIGN_CODE:
            icon_path = f"z{self.SIGN_CODE}.png"
        super().__init__(lane, icon_path=icon_path, **kwargs)

        self.zone_start = float(self.placement_long)
        self.zone_end = float(self.placement_long)
        self.zone_length = 0.0
        self._valid_placement = None
        self._approach_lanes_resolved = False

    # -- Graph / turns --

    def _get_graph(self):
        engine = getattr(self, "engine", None)
        if engine is None:
            return None
        try:
            return engine.current_map.road_network.graph
        except (AttributeError, TypeError):
            return None

    # SUMO direction codes ('l'/'r'/'s'/'t'/'L'/'R') -> PG-style maneuver strings
    _SUMO_DIR_TO_MANEUVER = {
        "l": "left", "L": "left",
        "r": "right", "R": "right",
        "s": "straight",
        "t": "uturn",
    }

    def _collect_turns_from_lane(self, lane):
        for turn in getattr(lane, "turns", []) or []:
            to_lane = turn.get("to_lane")
            if to_lane is None:
                continue
            maneuver = turn.get("maneuver", "")
            if not maneuver:
                # SUMO-style: translate 'direction' short code
                maneuver = self._SUMO_DIR_TO_MANEUVER.get(turn.get("direction", ""), "")
            if maneuver in self.FORBIDDEN_TURNS:
                self.forbidden_to_lanes.add(to_lane)
            else:
                self.allowed_to_lanes.add(to_lane)

    def _resolve_approach_lanes(self):
        if self._approach_lanes_resolved:
            return
        self._approach_lanes_resolved = True
        sign_idx = getattr(self.lane, "index", None)

        # SUMO: find lanes whose exit_lanes include the sign lane
        if isinstance(sign_idx, str):
            self._resolve_approach_lanes_sumo(sign_idx)
            return

        # PG: tuple index
        if not _is_tuple_lane_index(sign_idx):
            return
        arm_entry_node = str(sign_idx[0])
        graph = self._get_graph()
        if graph is None:
            return
        for from_node, successors in graph.items():
            if not isinstance(successors, dict):
                continue
            lanes = successors.get(arm_entry_node)
            if lanes is None:
                continue
            for i, lane_obj in enumerate(lanes):
                self._approach_lane_indices.add((str(from_node), arm_entry_node, i))
                self._collect_turns_from_lane(lane_obj)

    def _resolve_approach_lanes_sumo(self, sign_lane_idx):
        """For SUMO: find all lanes that have sign_lane_idx in their exit_lanes."""
        graph = self._get_graph()
        if graph is None:
            return
        # Strip "lane_" prefix for comparison with graph keys
        sign_raw = sign_lane_idx[5:] if sign_lane_idx.startswith("lane_") else sign_lane_idx
        for lane_name in graph:
            info = graph.get(lane_name)
            if info is None or not hasattr(info, "exit_lanes"):
                continue
            for exit_lane in (info.exit_lanes or []):
                exit_raw = exit_lane[5:] if isinstance(exit_lane, str) and exit_lane.startswith("lane_") else exit_lane
                if exit_raw == sign_raw or exit_lane == sign_lane_idx:
                    self._approach_lane_indices.add(lane_name)
                    # Collect turns from approach lane object
                    try:
                        road_network = self.engine.current_map.road_network
                        lane_obj = road_network.get_lane("lane_" + lane_name if not lane_name.startswith("lane_") else lane_name)
                        self._collect_turns_from_lane(lane_obj)
                    except Exception:
                        pass
                    break

    # -- Placement validation --

    def _check_placement(self, lane_idx, graph=None):
        # SUMO: string index — check that exit_lanes contain junction lanes
        if isinstance(lane_idx, str):
            try:
                rn_graph = self.engine.current_map.road_network.graph
                info = rn_graph.get(lane_idx)
                if info is None or not hasattr(info, "exit_lanes") or not info.exit_lanes:
                    return False
                # Valid if at least one exit lane goes through a junction
                return any(":" in str(e) for e in (info.exit_lanes or []))
            except Exception:
                return True  # fail-open

        if not _is_tuple_lane_index(lane_idx):
            return True  # fail-open for non-MetaDrive envs

        from_type = _pg_node_block_type(str(lane_idx[0]))
        to_type = _pg_node_block_type(str(lane_idx[1]))

        if to_type in _INTERSECTION_BLOCK_TYPES:  # T or X
            if from_type in _INTERSECTION_BLOCK_TYPES:
                from_info = _pg_node_parse(str(lane_idx[0]))
                to_info = _pg_node_parse(str(lane_idx[1]))
                if not from_info or not to_info:
                    return False
                return (from_info['block_type'] == to_info['block_type']
                        and from_info['block_index'] == to_info['block_index']
                        and from_info['part_index'] == to_info['part_index']
                        and from_info['road_index'] == 0
                        and to_info['road_index'] == 1)
            return True  # approach from non-intersection → intersection

        if to_type is not None:
            return False  # non-intersection PG block (S, C, etc.)

        return True  # non-PG nodes: fail-open

    @property
    def is_valid_placement(self):
        if self._valid_placement is None:
            self._valid_placement = self._check_placement(
                getattr(self.lane, "index", None)
            )
        return self._valid_placement

    # -- Violation detection --

    def check_violation(self, vehicle, for_reward=False) -> bool:
        vid = getattr(vehicle, "id", None)
        if vid is None:
            return False
        key = "reported_for_reward" if for_reward else "reported_for_metrics"
        state = self._vehicle_states.setdefault(
            vid, {"reported_for_reward": False, "reported_for_metrics": False},
        )
        if not self._is_violating(vehicle) or state[key]:
            return False
        state[key] = True
        return True

    def _is_violating(self, vehicle) -> bool:
        if not self.is_valid_placement:
            return False
        self._resolve_approach_lanes()

        agent_id = getattr(vehicle, "name", None) or getattr(vehicle, "id", None)
        if agent_id is None:
            return False

        current_lane = getattr(vehicle, "lane_index", None)
        sign_lane = getattr(self.lane, "index", None)

        # Normalize for comparison with approach set
        cl_norm = None
        if current_lane is not None and _is_tuple_lane_index(current_lane):
            cl_norm = (str(current_lane[0]), str(current_lane[1]), current_lane[2])

        # Vehicle on approach -> start tracking
        on_approach = (current_lane == sign_lane)
        if not on_approach and cl_norm is not None:
            on_approach = cl_norm in self._approach_lane_indices
        # SUMO: string-based approach check
        if not on_approach and isinstance(current_lane, str):
            cl_raw = current_lane[5:] if current_lane.startswith("lane_") else current_lane
            on_approach = (cl_raw in self._approach_lane_indices
                           or current_lane in self._approach_lane_indices)
        if on_approach:
            self._active_agents[agent_id] = {"on_approach": True, "violated": False}
            return False

        state = self._active_agents.get(agent_id)
        if state is None:
            return False
        if not state["on_approach"]:
            return state["violated"]

        # Vehicle left approach -> determine turn direction
        all_known = self.forbidden_to_lanes | self.allowed_to_lanes
        if all_known:
            if current_lane in self.forbidden_to_lanes:
                state.update(on_approach=False, violated=True)
                return True
            if current_lane in self.allowed_to_lanes:
                state.update(on_approach=False, violated=False)
                return False
            return False  # intermediate lane — keep tracking

        # No turns data -> use block-level tracking + heading fallback
        # SUMO: internal/junction lanes contain ':' — keep tracking
        if isinstance(current_lane, str) and ":" in current_lane:
            return False
        if cl_norm is not None:
            to_key = _pg_intersection_block_key(str(cl_norm[1]))
            if to_key is not None:
                return False  # still inside intersection block, keep tracking

        state["on_approach"] = False
        turn_dir = self._detect_turn_from_heading(vehicle)
        violated = turn_dir is not None and turn_dir in self.FORBIDDEN_TURNS
        state["violated"] = violated
        return violated

    def _detect_turn_from_heading(self, vehicle):
        try:
            approach_heading = float(self.lane.heading_theta_at(self.lane.length))
            vehicle_heading = getattr(vehicle, "heading_theta", None)
            if vehicle_heading is None:
                return None
            diff = _wrap_pi(float(vehicle_heading) - approach_heading)
            if abs(diff) > 2.5:
                return "uturn"
            elif diff > 0.4:
                return "left"
            elif diff < -0.4:
                return "right"
            return "straight"
        except Exception:
            return None

    @staticmethod
    def _get_vehicle_lane_index(vehicle):
        return _get_vehicle_lane_index(vehicle)

    def get_rule_description(self) -> str:
        return _SIGN_DESCRIPTIONS.get(self.SIGN_CODE, f"Sign {self.SIGN_CODE}")

    @property
    def top_down_color(self):
        return [0, 100, 210]

    @property
    def top_down_color_name(self):
        return "blue"


# ======================================================================
# Base: end-of-restriction sign (informational, no violations)
# ======================================================================

class EndOfRestrictedLaneSign(BaseTrafficSign):
    """Base for informational end-of-restriction signs. No violations."""

    SIGN_CODE = None
    LANE_USER = None
    FLOW_DIRECTION = None
    SIGN_SCOPE = None
    SIGN_EFFECT = SignEffect.END
    CANCELS_SIGN_CODE = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _auto_fill_semantics(cls)

    def __init__(self, lane, icon_path=None, **kwargs):
        kwargs["lateral_offset"] = 0
        if icon_path is None and self.SIGN_CODE:
            icon_path = f"z{self.SIGN_CODE}.png"
        super().__init__(lane, icon_path=icon_path, **kwargs)
        self.zone_start = self.placement_long
        self.zone_end = self.placement_long
        self.zone_length = 0.0

    def _is_violating(self, vehicle) -> bool:
        return False

    def get_rule_description(self) -> str:
        return _SIGN_DESCRIPTIONS.get(self.SIGN_CODE, f"Sign {self.SIGN_CODE}")

    @property
    def top_down_color(self):
        return [100, 150, 210]

    @property
    def top_down_color_name(self):
        return "blue"


# ======================================================================
# 5.11 — road with reserved lane (counter-flow, road-level begin)
# ======================================================================

class BusLaneRoadSign(RestrictedLaneSign):
    """5.11.1 — Road with counter-flow bus lane (begin)."""
    SIGN_CODE = "5.11.1"


class BikeLaneRoadSign(RestrictedLaneSign):
    """5.11.2 — Road with counter-flow bicycle lane (begin)."""
    SIGN_CODE = "5.11.2"


# ======================================================================
# 5.12 — end of road with reserved lane (counter-flow, road-level end)
# ======================================================================

class EndBusLaneRoadSign(EndOfRestrictedLaneSign):
    """5.12.1 — End of road with counter-flow bus lane."""
    SIGN_CODE = "5.12.1"
    CANCELS_SIGN_CODE = "5.11.1"


class EndBikeLaneRoadSign(EndOfRestrictedLaneSign):
    """5.12.2 — End of road with counter-flow bicycle lane."""
    SIGN_CODE = "5.12.2"
    CANCELS_SIGN_CODE = "5.11.2"


# ======================================================================
# 5.13 — exit to road with reserved lane (intersection turn restriction)
# ======================================================================

class ExitToBusLaneSign(IntersectionRestrictedLaneSign):
    """5.13.1 — Forbids left turn. Cars right, route vehicles left."""
    SIGN_CODE = "5.13.1"
    FORBIDDEN_TURNS = frozenset({"left"})
    REQUIRED_EXIT_LANE_USER = LaneUser.BUS


class ExitToBusLaneSignLeft(IntersectionRestrictedLaneSign):
    """5.13.2 — Forbids right turn. Cars left, route vehicles right."""
    SIGN_CODE = "5.13.2"
    FORBIDDEN_TURNS = frozenset({"right"})
    REQUIRED_EXIT_LANE_USER = LaneUser.BUS


class ExitToBikeLaneSign(IntersectionRestrictedLaneSign):
    """5.13.3 — Forbids left turn. Cars right, bicycles left."""
    SIGN_CODE = "5.13.3"
    FORBIDDEN_TURNS = frozenset({"left"})
    REQUIRED_EXIT_LANE_USER = LaneUser.BICYCLE


class ExitToBikeLaneSignLeft(IntersectionRestrictedLaneSign):
    """5.13.4 — Forbids right turn. Cars left, bicycles right."""
    SIGN_CODE = "5.13.4"
    FORBIDDEN_TURNS = frozenset({"right"})
    REQUIRED_EXIT_LANE_USER = LaneUser.BICYCLE


# ======================================================================
# 5.14 — lane for specific vehicle classes (same-direction, lane-level)
# ======================================================================

class BusLaneSign(RestrictedLaneSign):
    """5.14.1 — Same-direction bus lane (begin)."""
    SIGN_CODE = "5.14.1"


class BikeLaneSign(RestrictedLaneSign):
    """5.14.2 — Same-direction bicycle lane (begin)."""
    SIGN_CODE = "5.14.2"


class EndBusLaneSign(EndOfRestrictedLaneSign):
    """5.14.3 — End of same-direction bus lane."""
    SIGN_CODE = "5.14.3"
    CANCELS_SIGN_CODE = "5.14.1"


class EndBikeLaneSign(EndOfRestrictedLaneSign):
    """5.14.4 — End of same-direction bicycle lane."""
    SIGN_CODE = "5.14.4"
    CANCELS_SIGN_CODE = "5.14.2"


__all__ = [
    "LaneUser", "FlowDirection", "SignScope", "SignEffect",
    "SIGN_SEMANTICS",
    "RestrictedLaneSign", "IntersectionRestrictedLaneSign", "EndOfRestrictedLaneSign",
    "BusLaneRoadSign", "BikeLaneRoadSign",
    "EndBusLaneRoadSign", "EndBikeLaneRoadSign",
    "ExitToBusLaneSign", "ExitToBusLaneSignLeft",
    "ExitToBikeLaneSign", "ExitToBikeLaneSignLeft",
    "BusLaneSign", "BikeLaneSign",
    "EndBusLaneSign", "EndBikeLaneSign",
]
