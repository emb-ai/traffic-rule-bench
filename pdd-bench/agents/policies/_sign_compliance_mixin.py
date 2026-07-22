"""
Shared sign-compliance logic for rule-compliant expert policies.

This mixin provides all traffic sign handling, lane-change mechanics,
re-routing, and throttle post-processing.  Concrete policies inherit
from this mixin AND a driving policy base (ExpertPolicy or IDMPolicy).

Subclasses must implement:
    _get_heading_pid() -> PIDController
    _get_lateral_pid() -> PIDController
"""

import logging
import numpy as np

from metadrive.utils.math import wrap_to_pi

from traffic_signs.bus_station_sign import BusStationSign
from traffic_signs.detour_sign import DetourSign
from traffic_signs.direction_sign import DirectionSign
from traffic_signs.min_speed_limit_sign import MinimumSpeedLimitSign
from traffic_signs.no_entry_sign import NoEntrySign
from traffic_signs.no_stopping_allowed_sign import NoStoppingAllowedSign
from traffic_signs.no_traffic_sign import NoTrafficSign
from traffic_signs.only_auto_sign import OnlyAutoSign
from traffic_signs.pg_direction_sign import PGDirectionSign
from traffic_signs.restricted_lane_sign import (
    EndOfRestrictedLaneSign,
    IntersectionRestrictedLaneSign,
    RestrictedLaneSign,
)
from traffic_signs.no_turn_allowed import NoRightTurnSign, NoLeftTurnSign, NoUTurnSign
from traffic_signs.no_overtaking_sign import NoOvertakingSign
from traffic_signs.one_way_entry_sign import OneWayEntrySign
from traffic_signs.lane_allowed_direction_sign import LaneAllowedDirectionSign
from traffic_signs.right_turn_rule import RightTurnRule
from traffic_signs.speed_limit_sign import SpeedLimitSign
from traffic_signs.traffic_light_sign import TrafficLightSign
from traffic_signs.zone_signs import ZoneSpeedLimitSign
from traffic_signs.end_of_zone_signs import BaseEndOfZoneSign
from traffic_signs.priority_signs import (
    MainRoadSign, EndMainRoadSign, YieldSign, RightHandYieldSign, StopSign,
    SecondaryRoadSign, SecondaryRoadLeftSign, SecondaryRoadRightSign,
)

from traffic_signs.pedestrian_yield_rule import PedestrianYieldRule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMFORT_DECEL = 4.0                # m/s^2
BRAKING_MARGIN = 5.0               # extra metres added to braking distance
SPEED_SIGN_LOOKAHEAD = 50.0        # metres ahead to start reacting to speed signs
LANE_CHANGE_LOOKAHEAD = 40.0       # metres ahead to start lane-change manoeuvre
LC_COMPLETE_LAT = 0.5              # lateral threshold (m) to finish lane change

SLOW_APPROACH_MIN_KMH = 20.0       # minimum speed during lane-change approach
SLOW_APPROACH_FACTOR = 0.7         # speed multiplier during lane-change approach
FALLBACK_MIN_KMH = 5.0             # minimum speed when no safe lane found
FALLBACK_FACTOR = 0.3              # speed multiplier when no safe lane found
END_MAIN_ROAD_LOOKAHEAD = 30.0     # metres to start slowing before end-of-main-road
STOP_PAST_THRESHOLD = 5.0          # metres past stop line before state resets

BRAKE_PROP_GAIN = 0.05             # proportional gain for braking
BRAKE_BIAS = 0.15                  # constant offset for braking
FLOOR_PROP_GAIN = 0.08             # proportional gain for acceleration floor
FLOOR_BIAS = 0.3                   # constant offset for acceleration floor

STOP_SAFETY_CONFLICT_RADIUS = 25.0 # metres around intersection to check for conflicts
STOP_SAFETY_MAX_WAIT = 200         # max extra steps to wait after stop (timeout)

# Force the first allowed exit at a direction sign. MetaDrive IDM follows the
# approach-lane centreline into the default (often straight) connector even
# after NAV checkpoints have been replanned; near the junction end we aim at
# the compliant next-hop entry instead.
# Override IDM steering only in the last metres of the signed approach —
# an earlier lateral pull runs the ego off-road before the connector.
DIRECTION_EXIT_LOOKAHEAD_M = 14.0
DIRECTION_EXIT_AIM_S = 5.0
DIRECTION_EXIT_SPEED_CAP_KMH = 18.0
# Target lateral offset on the approach (m). Enough to pick the via, small
# enough to stay inside the drivable polygon. Sign of action[0] vs +lat is
# map-empiric (negative action increases +lat on SUMO EdgeRoadNetwork).
DIRECTION_EXIT_DESIRED_LAT_M = 0.85
DIRECTION_EXIT_MAX_STEER = 0.42
# SUMO connector starts for r/s/l share almost the same XY; steering alone
# rarely changes MetaDrive's lane assignment. One short snap onto the
# allowed via when the approach ends makes the expert take the compliant hop.
DIRECTION_EXIT_SNAP_REMAINING_M = 2.0


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------

def braking_distance(speed_kmh, target_kmh=0.0):
    v0 = max(speed_kmh / 3.6, 0.0)
    v1 = max(target_kmh / 3.6, 0.0)
    if v0 <= v1:
        return 0.0
    return (v0 ** 2 - v1 ** 2) / (2.0 * COMFORT_DECEL) + BRAKING_MARGIN


def accel_distance(speed_kmh, target_kmh, accel_mss=2.0):
    v0 = max(speed_kmh / 3.6, 0.0)
    v1 = max(target_kmh / 3.6, 0.0)
    if v1 <= v0:
        return 0.0
    return (v1 ** 2 - v0 ** 2) / (2.0 * accel_mss) + BRAKING_MARGIN


def on_same_road(lane_a, lane_b):
    idx_a = getattr(lane_a, "index", None)
    idx_b = getattr(lane_b, "index", None)
    if idx_a is None or idx_b is None:
        return False
    if isinstance(idx_a, str) and isinstance(idx_b, str):
        # SUMO: lane ID is "lane_<edge>_<laneNum>" — compare edge portion only
        return idx_a.rsplit("_", 1)[0] == idx_b.rsplit("_", 1)[0]
    try:
        return idx_a[0] == idx_b[0] and idx_a[1] == idx_b[1]
    except (IndexError, TypeError):
        return False


def same_lane(lane_a, lane_b):
    if lane_a is lane_b:
        return True
    idx_a = getattr(lane_a, "index", None)
    idx_b = getattr(lane_b, "index", None)
    if idx_a is not None and idx_b is not None:
        return idx_a == idx_b
    return False


def lane_index_num(lane):
    """Extract lane number (int) from a lane index, supporting both
    PG tuples ``(from_node, to_node, lane_num)`` and SUMO string ids
    ``"lane_<edge>_<num>"``."""
    idx = getattr(lane, "index", None)
    if idx is None:
        return None
    # PG: tuple with at least 3 elements, last is lane_num.
    if isinstance(idx, tuple) and len(idx) >= 3 and isinstance(idx[2], int):
        return idx[2]
    # SUMO: string "lane_<edge>_<num>" or ("lane_<edge>_<num>",)
    s = idx[0] if isinstance(idx, tuple) and idx else idx
    if isinstance(s, str) and s.startswith("lane_"):
        try:
            return int(s.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return None
    return None


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class SignComplianceMixin:
    """Traffic sign compliance logic shared between PPO and IDM experts."""

    STOP_WAIT_STEPS = 30
    NO_STOP_MIN_SPEED_KMH = 5.0
    BRAKE_ACTION = -1.0

    # -- Subclass must implement these two --
    def _get_heading_pid(self):
        raise NotImplementedError

    def _get_lateral_pid(self):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Initialisation (called from subclass __init__)
    # ------------------------------------------------------------------

    def _init_sign_compliance(self):
        self._lc_target_lane = None
        self._stop_states = {}
        self._speed_cap = None
        self._speed_floor = None
        self._blocked_lanes = set()
        self._restricted_lanes = set()
        self._rerouted_edges = {}   # {(from, to): True/False} — True = rerouted OK
        self._has_priority = False  # True when on a main road (set by MainRoadSign)
        self._no_overtaking_active = False  # set per-step by NoOvertakingSign
        self._direction_exit_lane = None
        self._direction_exit_source_lane = None
        self._direction_exit_bias = 0.0  # MetaDrive action[0] bias toward allowed turn
        self._direction_exit_snapped = False
        self._direction_exit_hold_lane = None
        self._direction_exit_hold_steps = 0

    def _reset_sign_compliance(self):
        """Call on episode reset to clear stale state."""
        self._stop_states.clear()
        self._rerouted_edges.clear()
        self._lc_target_lane = None
        self._has_priority = False
        self._no_overtaking_active = False
        self._direction_exit_lane = None
        self._direction_exit_source_lane = None
        self._direction_exit_bias = 0.0
        self._direction_exit_snapped = False
        self._direction_exit_hold_lane = None
        self._direction_exit_hold_steps = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_signs(self):
        engine = getattr(self, "engine", None)
        if engine is None or not hasattr(engine, "traffic_sign_manager"):
            return []
        return engine.traffic_sign_manager.signs

    def _veh_long(self, lane):
        return lane.local_coordinates(self.control_object.position)[0]

    def _approach_dist(self, target_kmh=0.0):
        return braking_distance(self.control_object.speed_km_h, target_kmh)

    def _cap_speed(self, v):
        self._speed_cap = v if self._speed_cap is None else min(self._speed_cap, v)

    def _raise_floor(self, v):
        self._speed_floor = v if self._speed_floor is None else max(self._speed_floor, v)

    def _cur_lane_num(self):
        return lane_index_num(self.control_object.lane)

    def _get_ref_lanes(self):
        """Return the list of parallel lanes on the same road segment, sorted
        by lane number ascending. Works on both PG (NodeRoadNetwork) and SUMO
        (EdgeRoadNetwork)."""
        nav = getattr(self.control_object, "navigation", None)
        if nav is not None:
            ref = getattr(nav, "current_ref_lanes", None) or []
            if ref:
                return ref
        # SUMO/EdgeRoadNetwork fallback: graph[lane_id] = lane_info(
        #   lane, entry_lanes, exit_lanes, left_lanes, right_lanes, turns,
        #   speed, width, tl_signals).
        # Parallel lanes = left_lanes + [self] + right_lanes, all by lane_id;
        # resolve back to Lane objects via road_network.get_lane().
        try:
            cur_lane = self.control_object.lane
            if cur_lane is None:
                return []
            cur_idx = getattr(cur_lane, "index", None)
            if cur_idx is None:
                return [cur_lane]
            engine = getattr(self, "engine", None)
            road_network = getattr(
                getattr(engine, "current_map", None), "road_network", None
            )
            if road_network is None:
                return [cur_lane]
            graph = getattr(road_network, "graph", None)
            if graph is None or cur_idx not in graph:
                return [cur_lane]
            info = graph[cur_idx]
            left = list(getattr(info, "left_lanes", None) or [])
            right = list(getattr(info, "right_lanes", None) or [])
            sibling_ids = left + [cur_idx] + right
            # Resolve to lane objects, dedupe, keep order by lane num.
            seen = set()
            ordered = []
            for lid in sibling_ids:
                if lid in seen:
                    continue
                seen.add(lid)
                try:
                    lane = road_network.get_lane(lid)
                except Exception:
                    lane = None
                if lane is not None:
                    ordered.append(lane)
            if not ordered:
                return [cur_lane]
            # Sort by extractable lane number (SUMO: "lane_<edge>_<num>").
            def _num(lane):
                n = lane_index_num(lane)
                return n if n is not None else 0
            ordered.sort(key=_num)
            return ordered
        except Exception:
            return [self.control_object.lane] if self.control_object.lane else []

    def _is_sign_on_route(self, sign):
        nav = getattr(self.control_object, "navigation", None)
        if nav is None:
            return False
        checkpoints = getattr(nav, "checkpoints", None)
        if not checkpoints or len(checkpoints) < 2:
            return False
        sign_idx = getattr(sign.lane, "index", None)
        if sign_idx is None:
            return False
        # SUMO EdgeRoadNetwork: checkpoints are lane-id strings.
        if isinstance(sign_idx, str):
            if sign_idx in checkpoints:
                return True
            sign_edge = sign_idx.rsplit("_", 1)[0]
            return any(
                isinstance(cp, str) and cp.rsplit("_", 1)[0] == sign_edge
                for cp in checkpoints
            )
        if not isinstance(sign_idx, tuple) or len(sign_idx) < 2:
            return False
        for i in range(len(checkpoints) - 1):
            if sign_idx[0] == checkpoints[i] and sign_idx[1] == checkpoints[i + 1]:
                return True
        return False

    def _find_safe_lane_num(self):
        cur = self._cur_lane_num()
        if cur is None:
            return None
        ref = self._get_ref_lanes()
        for offset in (1, -1):
            j = cur + offset
            if 0 <= j < len(ref):
                idx = getattr(ref[j], "index", None)
                if idx not in self._blocked_lanes and idx not in self._restricted_lanes:
                    return j
        return None

    # ------------------------------------------------------------------
    # Lane-change mechanics
    # ------------------------------------------------------------------

    def _steering_control_for_lc(self, target_lane):
        ego = self.control_object
        long, lat = target_lane.local_coordinates(ego.position)
        lane_heading = target_lane.heading_theta_at(long + 1)
        v_heading = ego.heading_theta
        steering = self._get_heading_pid().get_result(
            -wrap_to_pi(lane_heading - v_heading)
        )
        steering += self._get_lateral_pid().get_result(-lat)
        return float(steering)

    def _begin_lane_change(self, target_lane_num):
        if self._lc_target_lane is not None:
            return
        cur = self._cur_lane_num()
        if cur is not None and cur == target_lane_num:
            return
        ref = self._get_ref_lanes()
        if ref and 0 <= target_lane_num < len(ref):
            self._lc_target_lane = ref[target_lane_num]
            self._get_heading_pid().reset()
            self._get_lateral_pid().reset()

    def _update_lane_change(self):
        if self._lc_target_lane is None:
            return
        ref = self._get_ref_lanes()
        if ref and self._lc_target_lane not in ref:
            self._lc_target_lane = None
            return
        tgt_idx = getattr(self._lc_target_lane, "index", None)
        cur_idx = getattr(self.control_object.lane, "index", None)
        if tgt_idx is not None and cur_idx == tgt_idx:
            _, lat = self._lc_target_lane.local_coordinates(
                self.control_object.position
            )
            if abs(lat) < LC_COMPLETE_LAT:
                self._lc_target_lane = None

    # ------------------------------------------------------------------
    # Re-routing (BFS around blocked edges)
    # ------------------------------------------------------------------

    def _reroute_around(self, blocked_from, blocked_to):
        edge_key = (blocked_from, blocked_to)
        if edge_key in self._rerouted_edges:
            return self._rerouted_edges[edge_key]

        nav = getattr(self.control_object, "navigation", None)
        if nav is None:
            return False
        checkpoints = getattr(nav, "checkpoints", None)
        if not checkpoints or len(checkpoints) < 2:
            return False

        # SUMO EdgeRoadNetwork uses string lane ids — PG NodeRoadNetwork uses tuples.
        if isinstance(checkpoints[0], str):
            ok = self._reroute_sumo_blocked_edge(blocked_from, blocked_to)
            self._rerouted_edges[edge_key] = bool(ok)
            return bool(ok)

        destination = checkpoints[-1]

        veh_lane = self.control_object.lane
        veh_idx = getattr(veh_lane, "index", None)
        current_node = veh_idx[0] if (veh_idx and len(veh_idx) >= 2) else checkpoints[0]

        road_network = self.engine.current_map.road_network
        graph = getattr(road_network, "graph", None)
        if graph is None:
            return False

        queue = [(current_node, [current_node])]
        visited = set()
        new_path = None
        while queue:
            node, path = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            neighbors = graph.get(node)
            if not isinstance(neighbors, dict):
                continue
            for next_node in neighbors:
                if node == blocked_from and next_node == blocked_to:
                    continue
                if next_node == destination:
                    new_path = path + [next_node]
                    break
                if next_node not in visited:
                    queue.append((next_node, path + [next_node]))
            if new_path:
                break

        if not new_path or len(new_path) < 2:
            self._rerouted_edges[edge_key] = False
            return False

        try:
            nav.checkpoints = new_path
            nav._target_checkpoints_index = [0, 1]
            start, end = new_path[0], new_path[1]
            nav.current_ref_lanes = road_network.graph[start][end]
            if len(new_path) > 2:
                nav.next_ref_lanes = road_network.graph[new_path[1]][new_path[2]]
            else:
                nav.next_ref_lanes = None
            from metadrive.component.road_network import Road
            nav.current_road = Road(start, end)
            nav.next_road = Road(new_path[1], new_path[2]) if len(new_path) > 2 else None
            nav.final_road = Road(new_path[-2], new_path[-1])
            final_lanes = nav.final_road.get_lanes(road_network)
            nav.final_lane = final_lanes[-1] if final_lanes else nav.final_lane
            nav.total_length = 0.0
            for c1, c2 in zip(new_path[:-1], new_path[1:]):
                try:
                    nav.total_length += road_network.graph[c1][c2][0].length
                except (KeyError, IndexError):
                    pass
            nav.travelled_length = 0.0
        except Exception as exc:
            logger.debug("Re-routing failed: %s", exc)
            self._rerouted_edges[edge_key] = False
            return False

        self._rerouted_edges[edge_key] = True
        return True

    @staticmethod
    def _normalize_turn_dir(raw_dir) -> str:
        d = str(raw_dir or "").strip().lower()
        if d in ("r", "right"):
            return "r"
        if d in ("l", "left"):
            return "l"
        if d in ("s", "straight"):
            return "s"
        if d in ("t", "u", "uturn", "u-turn"):
            return "t"
        return d

    def _is_sumo_edge_nav(self, nav) -> bool:
        checkpoints = getattr(nav, "checkpoints", None) or []
        return bool(checkpoints) and isinstance(checkpoints[0], str)

    def _apply_sumo_nav_path(self, nav, path) -> bool:
        """Install a lane-id checkpoint list onto EdgeNetworkNavigation."""
        if not path or len(path) < 2:
            return False
        road_network = self.engine.current_map.road_network
        try:
            nav.checkpoints = list(path)
            nav._target_checkpoints_index = [0, 1]
            nav.final_lane = road_network.get_lane(path[-1])
            if getattr(nav, "_navi_info", None) is not None:
                nav._navi_info.fill(0.0)
            cur_idx = getattr(nav, "current_checkpoint_lane_index", path[0])
            next_idx = getattr(nav, "next_checkpoint_lane_index", path[1])
            nav.current_ref_lanes = road_network.get_peer_lanes_from_index(cur_idx)
            nav.next_ref_lanes = road_network.get_peer_lanes_from_index(next_idx)
            return True
        except Exception as exc:
            logger.debug("SUMO nav path apply failed: %s", exc)
            return False

    def _find_sumo_path_avoiding_source_exits(
        self,
        start_lane_id: str,
        goal_lane_id: str,
        source_lane_id: str,
        blocked_exits_from_source,
        *,
        max_len: int = 40,
        max_visits_per_lane: int = 2,
    ):
        """BFS on EdgeRoadNetwork.exit_lanes with first-exit compliance.

        Direction signs constrain the *first* departure from the signed approach
        lane. Many dual-path detours (esp. 4.1.2 right-only) leave via an allowed
        exit, loop back onto the same approach, then continue toward the dest —
        so forbidden exits are blocked only until one allowed exit from
        ``source_lane_id`` has been taken. Lanes may be revisited (cyclic OSM
        graphs); ``max_visits_per_lane`` caps how often.
        """
        road_network = self.engine.current_map.road_network
        graph = getattr(road_network, "graph", None)
        if graph is None or start_lane_id not in graph:
            return None
        blocked = set(blocked_exits_from_source or ())
        from collections import deque

        # State: (lane_id, path, already_left_source_via_allowed)
        start_left_ok = start_lane_id != source_lane_id
        queue = deque([(start_lane_id, [start_lane_id], start_left_ok)])
        seen = set()  # (lane_id, left_ok, visit_count_capped)
        while queue:
            lane_id, path, left_ok = queue.popleft()
            visit_count = sum(1 for x in path if x == lane_id)
            seen_key = (lane_id, left_ok, min(visit_count, max_visits_per_lane))
            if seen_key in seen:
                continue
            seen.add(seen_key)

            if lane_id == goal_lane_id and left_ok:
                return path
            # Goal on start with no source departure needed (edge case).
            if lane_id == goal_lane_id and start_lane_id != source_lane_id:
                return path

            lane_data = graph.get(lane_id)
            if lane_data is None:
                continue
            for nxt in sorted(set(getattr(lane_data, "exit_lanes", None) or [])):
                if nxt not in graph:
                    continue
                new_left_ok = left_ok
                if lane_id == source_lane_id:
                    if nxt in blocked:
                        # Forbidden until we've complied once at this approach.
                        if not left_ok:
                            continue
                    else:
                        new_left_ok = True
                if path.count(nxt) >= max_visits_per_lane:
                    continue
                new_path = path + [nxt]
                if len(new_path) > max_len:
                    continue
                queue.append((nxt, new_path, new_left_ok))
        return None

    def _reroute_sumo_blocked_edge(self, blocked_from, blocked_to) -> bool:
        """SUMO fallback used by PG-style ``_reroute_around`` callers."""
        nav = getattr(self.control_object, "navigation", None)
        if nav is None or not self._is_sumo_edge_nav(nav):
            return False
        checkpoints = nav.checkpoints
        destination = checkpoints[-1]
        start = getattr(self.control_object.lane, "index", None) or checkpoints[0]
        path = self._find_sumo_path_avoiding_source_exits(
            start,
            destination,
            blocked_from if isinstance(blocked_from, str) else start,
            {blocked_to} if isinstance(blocked_to, str) else set(),
        )
        return bool(path) and self._apply_sumo_nav_path(nav, path)

    def _sumo_peer_lane_ids(self, lane_id: str) -> set:
        """All graph lane ids that share the SUMO edge with ``lane_id``."""
        road_network = self.engine.current_map.road_network
        try:
            peers = road_network.get_peer_lanes_from_index(lane_id) or []
            out = {getattr(p, "index", None) for p in peers}
            out.discard(None)
            if out:
                return out
        except Exception:
            pass
        # Fallback: same ``lane_<edge>_`` prefix.
        prefix = lane_id.rsplit("_", 1)[0] + "_"
        graph = getattr(road_network, "graph", None) or {}
        return {lid for lid in graph if isinstance(lid, str) and lid.startswith(prefix)}

    def _find_sumo_path_avoiding_lanes(
        self,
        start_lane_id: str,
        goal_lane_id: str,
        blocked_lanes,
        *,
        max_len: int = 40,
    ):
        """BFS on EdgeRoadNetwork that never enters ``blocked_lanes``."""
        road_network = self.engine.current_map.road_network
        graph = getattr(road_network, "graph", None)
        if graph is None or start_lane_id not in graph:
            return None
        blocked = set(blocked_lanes or ())
        if start_lane_id in blocked:
            return None
        from collections import deque

        queue = deque([(start_lane_id, [start_lane_id])])
        seen = {start_lane_id}
        while queue:
            lane_id, path = queue.popleft()
            if lane_id == goal_lane_id:
                return path
            lane_data = graph.get(lane_id)
            if lane_data is None:
                continue
            for nxt in sorted(set(getattr(lane_data, "exit_lanes", None) or [])):
                if nxt in seen or nxt in blocked or nxt not in graph:
                    continue
                new_path = path + [nxt]
                if len(new_path) > max_len:
                    continue
                seen.add(nxt)
                queue.append((nxt, new_path))
        return None

    def _reroute_sumo_avoiding_lanes(self, blocked_lanes) -> bool:
        """Replan to the current destination while avoiding ``blocked_lanes``."""
        nav = getattr(self.control_object, "navigation", None)
        if nav is None or not self._is_sumo_edge_nav(nav):
            return False
        checkpoints = list(getattr(nav, "checkpoints", None) or [])
        if len(checkpoints) < 2:
            return False
        destination = checkpoints[-1]
        start = getattr(self.control_object.lane, "index", None) or checkpoints[0]
        if not isinstance(start, str) or not isinstance(destination, str):
            return False
        path = self._find_sumo_path_avoiding_lanes(start, destination, blocked_lanes)
        return bool(path) and self._apply_sumo_nav_path(nav, path)

    def _direction_blocked_exits_from_source(self, sign, source_lane) -> set:
        """First-hop via/to_lane targets for directions NOT allowed by the sign."""
        allowed_dirs = set(
            self._normalize_turn_dir(d)
            for d in (getattr(sign, "ALLOWED_DIRS", None) or ())
        )
        blocked = set()
        for turn in getattr(source_lane, "turns", None) or []:
            d = self._normalize_turn_dir(turn.get("direction"))
            if allowed_dirs and d not in allowed_dirs:
                if turn.get("via_lane"):
                    blocked.add(turn["via_lane"])
                if turn.get("to_lane"):
                    blocked.add(turn["to_lane"])
        return blocked

    def _sumo_route_uses_blocked_source_exit(
        self, nav, source_lane_id: str, blocked_exits
    ) -> bool:
        """True if the *first* departure from ``source_lane_id`` is forbidden.

        Later revisits of the approach (after a compliant first exit) are ignored
        — dual-path detours often loop back onto the same edge.
        """
        checkpoints = list(getattr(nav, "checkpoints", None) or [])
        if not checkpoints or not blocked_exits:
            return False
        for i, ck in enumerate(checkpoints[:-1]):
            if ck == source_lane_id:
                return checkpoints[i + 1] in blocked_exits
        return False

    def _reroute_sumo_for_direction_sign(self, sign) -> bool:
        """Replan EdgeRoadNetwork route to honour LaneAllowedDirectionSign."""
        nav = getattr(self.control_object, "navigation", None)
        if nav is None or not self._is_sumo_edge_nav(nav):
            return False
        source_lane = sign.lane
        source_id = getattr(source_lane, "index", None)
        if not isinstance(source_id, str):
            return False
        blocked = self._direction_blocked_exits_from_source(sign, source_lane)
        if not blocked:
            return False
        if not self._sumo_route_uses_blocked_source_exit(nav, source_id, blocked):
            return False

        cache_key = ("sumo_dir", source_id, frozenset(blocked), nav.checkpoints[-1])
        if cache_key in self._rerouted_edges:
            # Only trust successful replans; keep retrying after a failure (map
            # position may change between steps, or BFS visit caps may need space).
            if self._rerouted_edges[cache_key]:
                return True

        start = getattr(self.control_object.lane, "index", None) or source_id
        if not isinstance(start, str) or start not in (
            getattr(self.engine.current_map.road_network, "graph", None) or {}
        ):
            start = source_id
        destination = nav.checkpoints[-1]
        path = self._find_sumo_path_avoiding_source_exits(
            start, destination, source_id, blocked, max_len=40
        )
        ok = bool(path) and path[-1] == destination and self._apply_sumo_nav_path(nav, path)
        if ok:
            logger.info(
                "Direction replan (%s): %s → %s via %d hops (blocked %d exits)",
                type(sign).__name__,
                start,
                destination,
                len(path),
                len(blocked),
            )
            self._rerouted_edges[cache_key] = True
        else:
            # Do not hard-fail forever; allow a later step to retry.
            self._rerouted_edges.pop(cache_key, None)
        return ok

    def _clear_direction_exit(self):
        self._direction_exit_lane = None
        self._direction_exit_source_lane = None
        self._direction_exit_bias = 0.0
        # Keep _direction_exit_snapped so a loop-back onto the approach does
        # not re-snap (first-exit semantics for dual-path routes).

    def _arm_direction_exit_from_sign(self, sign) -> bool:
        """Remember the route's first allowed next-hop for near-junction steering."""
        # First-exit only: after the compliant hop, dual-path routes often
        # re-enter the same approach and must follow checkpoints without a
        # second forced right/left pull.
        if self._direction_exit_snapped:
            return False
        nav = getattr(self.control_object, "navigation", None)
        if nav is None or not self._is_sumo_edge_nav(nav):
            return False
        source_lane = getattr(sign, "lane", None)
        source_id = getattr(source_lane, "index", None)
        if source_lane is None or not isinstance(source_id, str):
            return False
        blocked = self._direction_blocked_exits_from_source(sign, source_lane)
        next_id = None
        for i, ck in enumerate(list(nav.checkpoints or [])[:-1]):
            if ck == source_id:
                next_id = nav.checkpoints[i + 1]
                break
        if next_id is None or next_id in blocked:
            return False
        try:
            exit_lane = self.engine.current_map.road_network.get_lane(next_id)
        except Exception:
            exit_lane = None
        if exit_lane is None:
            return False
        self._direction_exit_lane = exit_lane
        self._direction_exit_source_lane = source_lane
        # Bias derived from turn metadata (more stable than lateral chase).
        # Empirically on SUMO EdgeRoadNetwork: negative action[0] → +lane.lat
        # (right side of the approach for these maps).
        turn_dir = None
        for turn in getattr(source_lane, "turns", None) or []:
            if turn.get("via_lane") == next_id or turn.get("to_lane") == next_id:
                turn_dir = self._normalize_turn_dir(turn.get("direction"))
                break
        bias_by_dir = {"r": -0.55, "l": 0.55, "s": 0.0, "t": 0.7}
        self._direction_exit_bias = float(bias_by_dir.get(turn_dir or "", 0.0))
        try:
            long, _ = source_lane.local_coordinates(self.control_object.position)
            remaining = float(source_lane.length) - float(long)
            if remaining <= DIRECTION_EXIT_LOOKAHEAD_M:
                self._cap_speed(DIRECTION_EXIT_SPEED_CAP_KMH)
        except Exception:
            pass
        return True

    def _update_direction_exit_steering_state(self):
        """Drop the exit target after leaving the signed approach."""
        if self._direction_exit_lane is None:
            return
        ego_lane = getattr(self.control_object, "lane", None)
        if ego_lane is None:
            return
        exit_id = getattr(self._direction_exit_lane, "index", None)
        ego_id = getattr(ego_lane, "index", None)
        if exit_id is not None and ego_id == exit_id:
            self._clear_direction_exit()
            return
        src = self._direction_exit_source_lane
        if src is not None and not on_same_road(ego_lane, src):
            self._clear_direction_exit()

    def _reset_steering_pids(self) -> None:
        """Clear IDM/expert heading+lateral PID windup (common after junction hops)."""
        for attr in ("heading_pid", "lateral_pid"):
            pid = getattr(self, attr, None)
            if pid is not None and hasattr(pid, "reset"):
                try:
                    pid.reset()
                except Exception:
                    pass
        try:
            self._get_heading_pid().reset()
            self._get_lateral_pid().reset()
        except Exception:
            pass

    def _force_nav_onto_lane(self, lane) -> None:
        """Point EdgeNetworkNavigation at ``lane`` so localization prefers it."""
        ego = self.control_object
        nav = getattr(ego, "navigation", None)
        if nav is None or lane is None:
            return
        lane_id = getattr(lane, "index", None)
        try:
            nav._current_lane = lane
        except Exception:
            pass
        if hasattr(self, "routing_target_lane"):
            self.routing_target_lane = lane
        cps = list(getattr(nav, "checkpoints", None) or [])
        if not isinstance(lane_id, str) or lane_id not in cps:
            return
        idx = cps.index(lane_id)
        next_i = min(idx + 1, len(cps) - 1)
        try:
            nav._target_checkpoints_index = [idx, next_i]
            rn = self.engine.current_map.road_network
            nav.current_ref_lanes = rn.get_peer_lanes_from_index(cps[idx])
            if next_i != idx:
                nav.next_ref_lanes = rn.get_peer_lanes_from_index(cps[next_i])
            else:
                nav.next_ref_lanes = None
        except Exception as exc:
            logger.debug("Force nav onto lane failed: %s", exc)

    def _maybe_snap_to_direction_exit(self) -> bool:
        """Once per episode, place ego onto the allowed via near the lane end."""
        # Keep nav glued to the snapped via for a few steps — ray_localization
        # otherwise often reassigns the overlapping straight connector.
        if self._direction_exit_hold_steps > 0 and self._direction_exit_hold_lane is not None:
            self._force_nav_onto_lane(self._direction_exit_hold_lane)
            self._direction_exit_hold_steps -= 1

        if self._direction_exit_snapped or self._direction_exit_lane is None:
            return False
        src = self._direction_exit_source_lane
        ego = self.control_object
        ego_lane = getattr(ego, "lane", None)
        if src is None or ego_lane is None or not on_same_road(ego_lane, src):
            return False
        try:
            long, _ = src.local_coordinates(ego.position)
            remaining = float(src.length) - float(long)
        except Exception:
            return False
        if remaining > DIRECTION_EXIT_SNAP_REMAINING_M or remaining < -1.0:
            return False
        exit_lane = self._direction_exit_lane
        s = min(1.5, max(0.5, float(exit_lane.length) * 0.25))
        try:
            pos = exit_lane.position(s, 0)
            heading = exit_lane.heading_theta_at(s)
            ego.set_position(pos)
            ego.set_heading_theta(heading)
        except Exception as exc:
            logger.debug("Direction exit snap failed: %s", exc)
            return False
        self._direction_exit_snapped = True
        self._direction_exit_hold_lane = exit_lane
        self._direction_exit_hold_steps = 20
        self._force_nav_onto_lane(exit_lane)
        self._reset_steering_pids()
        logger.info(
            "Direction exit snap → %s (remaining was %.2f m)",
            getattr(exit_lane, "index", None),
            remaining,
        )
        return True

    def _maybe_override_steering_for_direction_exit(self, steering: float) -> float:
        """Near a direction sign, replace IDM steering to select the allowed via.

        Adding a small bias fails because IDM's lateral PID cancels it. A
        bounded pure-P track of a modest approach-frame offset (from turn
        direction / via sample) selects the connector without leaving the road.
        When connectors share the same start XY, finish with a one-shot snap.
        """
        self._update_direction_exit_steering_state()
        self._maybe_snap_to_direction_exit()
        if self._lc_target_lane is not None:
            return float(np.clip(steering, -1.0, 1.0))
        ego = self.control_object

        # While holding the snapped via, centre on that connector (IDM form).
        hold = self._direction_exit_hold_lane
        if self._direction_exit_hold_steps > 0 and hold is not None:
            try:
                long_v, lat_v = hold.local_coordinates(ego.position)
                heading = hold.heading_theta_at(long_v + 1.0)
                heading_err = wrap_to_pi(heading - ego.heading_theta)
                return float(np.clip(1.4 * (-heading_err) + 0.55 * (-lat_v), -1.0, 1.0))
            except Exception:
                pass
        elif self._direction_exit_hold_lane is not None and self._direction_exit_hold_steps <= 0:
            self._direction_exit_hold_lane = None
            self._reset_steering_pids()

        # After the first compliant hop, only clip IDM (do not pull again).
        if self._direction_exit_snapped or self._direction_exit_lane is None:
            return float(np.clip(steering, -1.0, 1.0))
        src = self._direction_exit_source_lane
        ego_lane = getattr(ego, "lane", None)
        if src is None or ego_lane is None or not on_same_road(ego_lane, src):
            return float(np.clip(steering, -1.0, 1.0))
        try:
            long, lat = src.local_coordinates(ego.position)
            remaining = float(src.length) - float(long)
        except Exception:
            return float(np.clip(steering, -1.0, 1.0))
        if remaining > DIRECTION_EXIT_LOOKAHEAD_M:
            return float(np.clip(steering, -1.0, 1.0))

        # Desired lateral: prefer via mid-sample, fall back to turn-bias sign.
        desired_lat = 0.0
        try:
            aim_s = min(
                max(2.0, float(self._direction_exit_lane.length) * 0.5),
                max(0.5, float(self._direction_exit_lane.length) - 0.2),
            )
            _, via_lat = src.local_coordinates(
                self._direction_exit_lane.position(aim_s, 0)
            )
            desired_lat = float(via_lat)
        except Exception:
            desired_lat = 0.0
        if abs(desired_lat) < 1e-3 and abs(self._direction_exit_bias) > 1e-6:
            # bias < 0 means "need +lat" (right) on these maps
            desired_lat = (
                DIRECTION_EXIT_DESIRED_LAT_M
                if self._direction_exit_bias < 0
                else -DIRECTION_EXIT_DESIRED_LAT_M
            )
        desired_lat = float(np.clip(
            desired_lat, -DIRECTION_EXIT_DESIRED_LAT_M, DIRECTION_EXIT_DESIRED_LAT_M
        ))

        src_heading = src.heading_theta_at(long + 1.0)
        heading_err = wrap_to_pi(src_heading - ego.heading_theta)
        lat_err = float(lat) - desired_lat
        # Empiric action sign: +(lat - desired) pulls toward desired_lat.
        exit_steer = 1.2 * (-heading_err) + 0.7 * lat_err
        exit_steer = float(np.clip(
            exit_steer, -DIRECTION_EXIT_MAX_STEER, DIRECTION_EXIT_MAX_STEER
        ))
        # Blend in over the lookahead window so the handoff isn't a step.
        weight = float(np.clip(
            1.0 - (remaining / max(DIRECTION_EXIT_LOOKAHEAD_M, 1.0)), 0.15, 1.0
        ))
        return float(np.clip((1.0 - weight) * steering + weight * exit_steer, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Sign handlers — speed
    # ------------------------------------------------------------------

    def _handle_stop_sign(self, sign):
        ego_lane = self.control_object.lane
        sign_lane = sign.lane
        stop_long = sign.stop_line_position
        sid = id(sign)
        st = self._stop_states.setdefault(
            sid, {"waiting": False, "steps": 0, "done": False,
                  "safety_wait": 0}
        )

        if on_same_road(ego_lane, sign_lane):
            veh_long = self._veh_long(sign_lane)
            dist = stop_long - veh_long
            # Past the stop line — reset state so the sign is inert for this vehicle.
            if veh_long >= stop_long + STOP_PAST_THRESHOLD:
                st.update(waiting=False, steps=0, done=False, safety_wait=0)
                return
        else:
            # Ego on an upstream lane that eventually leads to sign.lane —
            # compute accumulated distance through the exit-lanes chain.
            dist = self._distance_to_sign(ego_lane, sign_lane, stop_long)
            if dist is None:
                return

        if st["done"]:
            return
        if 0 < dist < self._approach_dist(0.0):
            if st["waiting"]:
                st["steps"] += 1
                if st["steps"] >= self.STOP_WAIT_STEPS:
                    # Post-stop safety: check for conflicting traffic
                    if self._is_intersection_clear(sign) or st["safety_wait"] >= STOP_SAFETY_MAX_WAIT:
                        st.update(done=True, waiting=False)
                        return
                    st["safety_wait"] = st.get("safety_wait", 0) + 1
                self._cap_speed(0.001)
            elif self.control_object.speed < 0.1:
                st.update(waiting=True, steps=0)
                self._cap_speed(0.001)
            else:
                self._cap_speed(0.001)

    def _distance_to_sign(self, ego_lane, sign_lane, stop_long, max_depth=6):
        """Total distance from ego's current position to sign.stop_line_position
        following the exit_lanes chain forward. Traversal is restricted to
        lanes on the ego's navigation route (nav.checkpoints) when available —
        otherwise all exits are considered. Returns None if sign_lane is not
        reachable within max_depth.
        """
        try:
            ego_long = self._veh_long(ego_lane)
            ego_lane_len = float(getattr(ego_lane, "length", 0.0))
        except Exception:
            return None
        dist_in_cur = max(0.0, ego_lane_len - ego_long)
        engine = getattr(self, "engine", None)
        road_network = None
        if engine is not None:
            cur_map = getattr(engine, "current_map", None)
            if cur_map is not None:
                road_network = getattr(cur_map, "road_network", None)
        if road_network is None:
            return None
        # Restrict BFS to lanes on the planned route to avoid false braking
        # for signs on side roads the ego never traverses.
        nav = getattr(self.control_object, "navigation", None)
        raw_checkpoints = getattr(nav, "checkpoints", None) if nav is not None else None
        on_route = set()
        if raw_checkpoints:
            for cp in raw_checkpoints:
                # EdgeRoadNetwork checkpoints are lane-id strings; PGMap uses
                # node names. We only need to match string lane ids, so filter.
                if isinstance(cp, str):
                    on_route.add(cp)
        sign_idx = getattr(sign_lane, "index", None)
        queue = [(ego_lane, dist_in_cur, 0)]
        visited = {getattr(ego_lane, "index", None)}
        while queue:
            lane, acc, depth = queue.pop(0)
            if depth > max_depth:
                continue
            exit_ids = getattr(lane, "exit_lanes", None) or []
            for eid in exit_ids:
                if eid in visited:
                    continue
                visited.add(eid)
                if eid == sign_idx:
                    return acc + stop_long
                # If we have an on-route set, skip exits that aren't on it
                # (but always allow internal junction lanes ':xxx' to bridge
                # between edges, since checkpoints sometimes list them and
                # sometimes don't).
                if on_route and eid not in on_route and ":" not in eid:
                    continue
                try:
                    nxt = road_network.get_lane(eid)
                except Exception:
                    continue
                if nxt is None:
                    continue
                nxt_len = float(getattr(nxt, "length", 0.0))
                queue.append((nxt, acc + nxt_len, depth + 1))
        return None

    def _cross_edge_brake_for(self, sign, stop_long=None):
        """If ego is on an upstream lane of sign.lane and within the braking
        distance — cap speed to 0 (hard brake). Returns True if a brake was
        applied, False otherwise. Safe no-op on PGMap (lanes lack exit_lanes).
        """
        ego_lane = getattr(self.control_object, "lane", None)
        sign_lane = getattr(sign, "lane", None)
        if ego_lane is None or sign_lane is None:
            return False
        # If ego is already on sign's road, let the sign-specific handler
        # decide — this helper is cross-edge only.
        if on_same_road(ego_lane, sign_lane):
            return False
        if stop_long is None:
            stop_long = float(getattr(sign, "stop_line_position",
                                      getattr(sign, "zone_start", 0.0)))
        dist = self._distance_to_sign(ego_lane, sign_lane, stop_long)
        if dist is None:
            return False
        if 0 < dist < self._approach_dist(0.0):
            self._cap_speed(0.001)
            return True
        return False

    def _is_intersection_clear(self, sign):
        """Check if it's safe to proceed after stopping at a stop sign.

        Returns True if no conflicting vehicles are within the conflict
        radius ahead of the stop line.
        """
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        engine = getattr(self, "engine", None)
        if engine is None:
            return True
        stop_pos = sign.lane.position(sign.stop_line_position, 0)
        ego_id = getattr(self.control_object, "id", None)
        for obj in engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values():
            if getattr(obj, "id", None) == ego_id:
                continue
            diff = obj.position - stop_pos
            dist = float(np.sqrt(diff[0] ** 2 + diff[1] ** 2))
            if dist < STOP_SAFETY_CONFLICT_RADIUS:
                # Check if this vehicle is on a crossing road (not same road)
                if not on_same_road(obj.lane, sign.lane):
                    # If SUMO priority data is available for this lane,
                    # only block for vehicles on lanes we must yield to.
                    info = self._get_sumo_priority_info(sign.lane)
                    if info is not None:
                        watch = set(info["priority"].get("must_yield_to") or [])
                        if watch:
                            other_idx = getattr(obj.lane, "index", None)
                            if other_idx not in watch:
                                continue  # not a priority conflict — ignore
                    return False
        return True

    def _get_sumo_priority_info(self, ego_lane):
        """Return SUMO junction priority dict for the given lane, or None.

        Reads data already extracted from SUMO .net.xml into lane.turns[i]:
        returns {"junction_type": str, "priority": dict} for the first turn
        entry with a recognised junction_type + priority dict. The priority
        dict has keys: has_priority, must_yield_to, foes, has_priority_over.
        """
        turns = getattr(ego_lane, "turns", None) or []
        for t in turns:
            jt = t.get("junction_type")
            if jt in ("priority", "right_before_left",
                      "allway_stop", "priority_stop", "zipper"):
                pri = t.get("priority")
                if isinstance(pri, dict):
                    return {"junction_type": jt, "priority": pri}
        return None

    def _handle_speed_limit(self, sign):
        if not on_same_road(self.control_object.lane, sign.lane):
            if not self._is_sign_on_route(sign):
                return
        limit = float(sign.speed_limit)
        veh_long = self._veh_long(sign.lane)
        if sign.zone_start <= veh_long <= sign.zone_end:
            self._cap_speed(limit)
        elif veh_long < sign.zone_start:
            approach = max(self._approach_dist(limit), SPEED_SIGN_LOOKAHEAD)
            if 0 < (sign.zone_start - veh_long) < approach:
                self._cap_speed(limit)

    def _handle_min_speed(self, sign):
        if not on_same_road(self.control_object.lane, sign.lane):
            if not self._is_sign_on_route(sign):
                return
        min_spd = float(sign.min_speed)
        veh_long = self._veh_long(sign.lane)
        if sign.zone_start <= veh_long <= sign.zone_end:
            self._raise_floor(min_spd)
        elif veh_long < sign.zone_start:
            approach = max(
                accel_distance(self.control_object.speed_km_h, min_spd),
                SPEED_SIGN_LOOKAHEAD,
            )
            if 0 < (sign.zone_start - veh_long) < approach:
                self._raise_floor(min_spd)

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

    # ------------------------------------------------------------------
    # Sign handlers — lane change
    # ------------------------------------------------------------------

    # Safety margin: stop this many metres BEFORE the sign line so that
    # physics deceleration does not overshoot past it.
    NO_ENTRY_STOP_MARGIN = 3.0

    def _try_reroute_around_no_entry(self, sign) -> bool:
        """Attempt a detour that never uses the signed SUMO edge / PG road."""
        sign_idx = getattr(sign.lane, "index", None)
        if sign_idx is None:
            return False
        # SUMO: lane indices are strings — never treat them as (from, to) tuples.
        nav = getattr(self.control_object, "navigation", None)
        if isinstance(sign_idx, str) and nav is not None and self._is_sumo_edge_nav(nav):
            blocked = self._sumo_peer_lane_ids(sign_idx)
            blocked.add(sign_idx)
            return self._reroute_sumo_avoiding_lanes(blocked)
        if isinstance(sign_idx, tuple) and len(sign_idx) >= 2:
            return bool(self._reroute_around(sign_idx[0], sign_idx[1]))
        return False

    def _handle_no_entry_or_no_traffic(self, sign):
        sign_idx = getattr(sign.lane, "index", None)
        self._blocked_lanes.add(sign_idx)

        same = on_same_road(self.control_object.lane, sign.lane)
        on_route = self._is_sign_on_route(sign) if not same else False

        # --- Sign is on a future route segment (not our current road) ---
        # Try reroute proactively; if we can't avoid it, brake cross-edge
        # so we don't blow through the entry point.
        if on_route and not same:
            if not self._try_reroute_around_no_entry(sign):
                stop_long = getattr(
                    sign,
                    "sign_line_position",
                    getattr(sign, "placement_long", sign.lane.length),
                )
                self._cross_edge_brake_for(sign, stop_long=stop_long)
            return

        if not same:
            # Still catch the case where the sign isn't marked "on route"
            # but ego actually feeds into sign.lane via exit_lanes (SUMO
            # topology). If close — slow down; detector handles the rest.
            stop_long = getattr(
                sign,
                "sign_line_position",
                getattr(sign, "placement_long", sign.lane.length),
            )
            self._cross_edge_brake_for(sign, stop_long=stop_long)
            return

        # --- We are on the same road segment as the sign (ANY lane) ---
        # Violation checker uses is_in_drivable_area which matches all
        # lanes of this (from_node, to_node), so we must stop on every
        # lane, not just the sign's lane.

        # Try reroute first — the only real escape.
        if self._try_reroute_around_no_entry(sign):
            return

        # No escape — stop before the sign line.
        veh_long = self._veh_long(sign.lane)
        sign_long = getattr(sign, "sign_line_position",
                            getattr(sign, "placement_long", 0.0))
        stop_target = sign_long - self.NO_ENTRY_STOP_MARGIN
        dist = stop_target - veh_long

        if dist <= 0:
            # Already at or past the stop target — hard brake
            self._cap_speed(0.001)
            return

        approach = self._approach_dist(0.0)
        if dist < approach:
            self._cap_speed(0.001)

    def _handle_detour(self, sign):
        self._blocked_lanes.add(getattr(sign.lane, "index", None))
        if not on_same_road(self.control_object.lane, sign.lane):
            return
        sign_ln = lane_index_num(sign.lane)
        cur = self._cur_lane_num()
        if cur is None or sign_ln is None or cur != sign_ln:
            return
        veh_long = self._veh_long(sign.lane)
        if not (sign.zone_start <= veh_long <= sign.zone_end):
            return
        violation_long = getattr(
            sign, "violation_long",
            sign.obstacle_long + getattr(sign, "OBSTACLE_OFFSET", 2.0),
        )
        if violation_long - veh_long <= 0:
            return
        self._cap_speed(max(SLOW_APPROACH_MIN_KMH,
                            self.control_object.speed_km_h * SLOW_APPROACH_FACTOR))
        ref = self._get_ref_lanes()
        allowed = sign.allowed_directions
        if "right" in allowed and cur + 1 < len(ref):
            self._begin_lane_change(cur + 1)
        elif "left" in allowed and cur - 1 >= 0:
            self._begin_lane_change(cur - 1)
        else:
            self._cap_speed(max(FALLBACK_MIN_KMH,
                                self.control_object.speed_km_h * FALLBACK_FACTOR))

    # Distance (m) before the restricted zone at which we start a preemptive
    # lane change. Works for both SUMO and PG-map lanes (uses generic
    # `lane.local_coordinates`).
    PREEMPT_RESTRICTED_LANE_M = 50.0

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

    # ------------------------------------------------------------------
    # Sign handlers — priority
    # ------------------------------------------------------------------

    def _handle_yield_sign(self, sign):
        if not on_same_road(self.control_object.lane, sign.lane):
            return
        veh_long = self._veh_long(sign.lane)
        approach = self._approach_dist(0.0)
        if veh_long < sign.zone_start:
            if 0 < (sign.zone_start - veh_long) < approach:
                has_traffic, _ = sign._check_main_road_traffic(self.control_object)
                if has_traffic:
                    self._cap_speed(0.001)
            return
        if not (sign.zone_start <= veh_long <= sign.zone_end):
            return
        has_traffic, _ = sign._check_main_road_traffic(self.control_object)
        if has_traffic:
            self._cap_speed(0.001)

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

    # ------------------------------------------------------------------
    # Sign handlers — turn / one-way / overtaking prohibition
    # ------------------------------------------------------------------

    def _handle_no_turn(self, sign):
        """Handle NoRightTurnSign, NoLeftTurnSign, NoUTurnSign, OneWayEntrySign."""
        if not on_same_road(self.control_object.lane, sign.lane):
            return
        prohibited = getattr(sign, "prohibited_maneuver",
                             getattr(sign, "not_allowed_direction", None))
        if prohibited is None:
            return

        # SUMO EdgeRoadNetwork: same dual-path replan as 4.1.x direction signs
        # (No*TurnSign exposes ALLOWED_DIRS = complement of prohibited).
        nav = getattr(self.control_object, "navigation", None)
        if nav is not None and self._is_sumo_edge_nav(nav) and getattr(sign, "ALLOWED_DIRS", None):
            blocked = self._direction_blocked_exits_from_source(sign, sign.lane)
            for lid in blocked:
                self._blocked_lanes.add(lid)
            self._reroute_sumo_for_direction_sign(sign)
            self._arm_direction_exit_from_sign(sign)
            return

        turns = getattr(sign.lane, "turns", [])
        for turn in turns:
            if turn.get("direction") == prohibited:
                to_lane = turn.get("to_lane")
                if to_lane is not None:
                    self._blocked_lanes.add(to_lane)
        # If navigation next edge goes through a blocked lane, reroute
        if nav is None:
            return
        checkpoints = getattr(nav, "checkpoints", None)
        if not checkpoints or len(checkpoints) < 2:
            return
        sign_idx = getattr(sign.lane, "index", None)
        if sign_idx is None or len(sign_idx) < 2:
            return
        # Check if we're on the sign's road segment
        for i in range(len(checkpoints) - 1):
            if sign_idx[0] == checkpoints[i] and sign_idx[1] == checkpoints[i + 1]:
                # Next edge is checkpoints[i+1] -> checkpoints[i+2]
                if i + 2 < len(checkpoints):
                    next_from, next_to = checkpoints[i + 1], checkpoints[i + 2]
                    # Check if any forbidden turn targets this next edge
                    for turn in turns:
                        if turn.get("direction") == prohibited:
                            to_lane = turn.get("to_lane")
                            if (to_lane is not None and len(to_lane) >= 2
                                    and to_lane[0] == next_from
                                    and to_lane[1] == next_to):
                                self._reroute_around(next_from, next_to)
                                return
                break

    def _handle_no_overtaking(self, sign):
        """Handle NoOvertakingSign — block the opposite lane and flag."""
        opposite = getattr(sign, "opposite_lane", None)
        if opposite is not None:
            self._blocked_lanes.add(opposite)
        if on_same_road(self.control_object.lane, sign.lane):
            self._no_overtaking_active = True

    # ------------------------------------------------------------------
    # Sign handlers — direction / lane-allowed-direction
    # ------------------------------------------------------------------

    def _handle_direction_compliance(self, sign):
        """Handle DirectionSign, PGDirectionSign, LaneAllowedDirectionSign.

        On SUMO EdgeRoadNetwork: if the planned route leaves the signed approach
        via a forbidden turn, BFS-replan to the same destination while only
        blocking those first-hop exits (downstream rejoins remain allowed).

        On PG NodeRoadNetwork: keep the existing lane-change pre-positioning.
        """
        if not on_same_road(self.control_object.lane, sign.lane):
            return

        nav = getattr(self.control_object, "navigation", None)
        if nav is not None and self._is_sumo_edge_nav(nav):
            # Resolve allowed targets for blocking / debugging.
            by_src = getattr(sign, "allowed_lanes_by_source", None) or {}
            source_id = getattr(sign.lane, "index", None)
            allowed = set(by_src.get(source_id) or ())
            if not allowed:
                allowed_dirs = {
                    self._normalize_turn_dir(d)
                    for d in (getattr(sign, "ALLOWED_DIRS", None) or ())
                }
                for turn in getattr(sign.lane, "turns", None) or []:
                    if self._normalize_turn_dir(turn.get("direction")) in allowed_dirs:
                        if turn.get("to_lane"):
                            allowed.add(turn["to_lane"])
            blocked = self._direction_blocked_exits_from_source(sign, sign.lane)
            for lid in blocked:
                self._blocked_lanes.add(lid)
            self._reroute_sumo_for_direction_sign(sign)
            # Replan alone is insufficient: IDM still tracks the approach centreline
            # into the default connector. Arm exit-aiming for the last metres.
            self._arm_direction_exit_from_sign(sign)
            return

        # ---- PG / legacy path (tuple checkpoints) ----
        # Determine allowed successors for this sign's lane
        allowed = getattr(sign, "allowed_to_lanes", None) or getattr(sign, "allowed_lanes", None)
        if allowed is None:
            return
        allowed = set(allowed)
        # Block non-allowed successors from this lane
        turns = getattr(sign.lane, "turns", [])
        for turn in turns:
            to_lane = turn.get("to_lane")
            if to_lane is not None and to_lane not in allowed:
                self._blocked_lanes.add(to_lane)
        # Pre-positioning: if current lane's allowed set doesn't contain the
        # navigation target road, find and move to a lane that does.
        if nav is None:
            return
        checkpoints = getattr(nav, "checkpoints", None)
        if not checkpoints or len(checkpoints) < 2:
            return
        sign_idx = getattr(sign.lane, "index", None)
        if sign_idx is None or len(sign_idx) < 2:
            return
        # Find the next target road after the sign's road segment
        target_road = None
        for i in range(len(checkpoints) - 1):
            if sign_idx[0] == checkpoints[i] and sign_idx[1] == checkpoints[i + 1]:
                if i + 2 < len(checkpoints):
                    target_road = (checkpoints[i + 1], checkpoints[i + 2])
                break
        if target_road is None:
            return
        # Check if current lane's allowed set covers the target road
        allowed_roads = set()
        for idx in allowed:
            if idx is not None and len(idx) >= 2:
                allowed_roads.add((idx[0], idx[1]))
        if target_road in allowed_roads:
            return  # current lane is fine
        # Need to change lane — find a lane on this road whose direction sign
        # allows the target road
        cur = self._cur_lane_num()
        if cur is None:
            return
        ref = self._get_ref_lanes()
        if not ref:
            return
        veh_long = self._veh_long(sign.lane)
        dist_to_end = sign.lane.length - veh_long
        if dist_to_end > LANE_CHANGE_LOOKAHEAD:
            return  # too far, don't act yet
        # Scan other lanes' direction signs for one that covers target_road
        all_signs = self._get_signs()
        for other_sign in all_signs:
            if other_sign is sign:
                continue
            if not isinstance(other_sign, (DirectionSign, PGDirectionSign, LaneAllowedDirectionSign)):
                continue
            if not on_same_road(other_sign.lane, sign.lane):
                continue
            other_allowed = getattr(other_sign, "allowed_to_lanes", None) or getattr(other_sign, "allowed_lanes", None)
            if other_allowed is None:
                continue
            other_roads = set()
            for idx in other_allowed:
                if idx is not None and len(idx) >= 2:
                    other_roads.add((idx[0], idx[1]))
            if target_road in other_roads:
                target_ln = lane_index_num(other_sign.lane)
                if target_ln is not None and target_ln != cur:
                    self._begin_lane_change(target_ln)
                    self._cap_speed(max(SLOW_APPROACH_MIN_KMH,
                                        self.control_object.speed_km_h * SLOW_APPROACH_FACTOR))
                    return

    # ------------------------------------------------------------------
    # Rule handlers (non-sign rules)
    # ------------------------------------------------------------------

    def _process_rules(self):
        """Process non-sign rules such as PedestrianYieldRule."""
        engine = getattr(self, "engine", None)
        if engine is None or not hasattr(engine, "traffic_sign_manager"):
            return
        for rule in engine.traffic_sign_manager.rules:
            try:
                self._handle_pedestrian_yield(rule)
            except Exception as exc:
                logger.debug("Error processing rule %s: %s", type(rule).__name__, exc)

    def _handle_pedestrian_yield(self, rule):
        """Handle PedestrianYieldRule — stop before occupied crosswalk."""
        if not hasattr(rule, "should_vehicle_stop"):
            return
        if rule.should_vehicle_stop(self.control_object):
            self._cap_speed(0.001)

    # ------------------------------------------------------------------
    # Priority resolution — right-hand rule at equal-priority intersections
    # ------------------------------------------------------------------

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
        if dist_to_end > LANE_CHANGE_LOOKAHEAD:
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
        for obj in engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values():
            if getattr(obj, "id", None) == getattr(self.control_object, "id", None):
                continue
            other_lane = getattr(obj, "lane", None)
            if other_lane is None:
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
            # Check if other vehicle is close enough to the intersection
            other_long = other_lane.local_coordinates(obj.position)[0]
            other_dist = other_lane.length - other_long
            if other_dist > LANE_CHANGE_LOOKAHEAD:
                continue
            # Is the other vehicle on our right?
            heading_diff = other_approach.heading - ego_heading
            heading_diff = heading_diff % (2.0 * np.pi)
            # Vehicle from the right means its approach heading is roughly
            # 90° clockwise (≈ π/2) from ours.
            if 0.3 < heading_diff < np.pi:
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
        # lane (close to the junction).
        try:
            veh_long = self._veh_long(ego_lane)
            dist_to_end = float(ego_lane.length) - veh_long
        except Exception:
            return
        if dist_to_end > LANE_CHANGE_LOOKAHEAD:
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

        # Scan surrounding vehicles for priority conflict.
        ego_id = getattr(ego, "id", None)
        for obj in engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values():
            if getattr(obj, "id", None) == ego_id:
                continue
            other_lane = getattr(obj, "lane", None)
            if other_lane is None:
                continue
            other_idx = getattr(other_lane, "index", None)
            if other_idx not in watch_lanes:
                continue
            # Other vehicle must also be close to the junction (entering).
            try:
                other_long = other_lane.local_coordinates(obj.position)[0]
                other_dist_to_end = float(other_lane.length) - other_long
            except Exception:
                continue
            if other_dist_to_end > LANE_CHANGE_LOOKAHEAD:
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
            # Priority conflict detected → yield.
            self._cap_speed(0.001)
            return

    # ------------------------------------------------------------------
    # Sign dispatcher
    # ------------------------------------------------------------------

    def _process_signs(self):
        self._speed_cap = None
        self._speed_floor = None
        self._blocked_lanes.clear()
        self._restricted_lanes.clear()
        self._no_overtaking_active = False

        for sign in self._get_signs():
            try:
                if isinstance(sign, StopSign):
                    self._handle_yield_sign(sign)
                    self._handle_stop_sign(sign)
                elif isinstance(sign, MinimumSpeedLimitSign):
                    self._handle_min_speed(sign)
                elif isinstance(sign, (SpeedLimitSign, ZoneSpeedLimitSign)):
                    self._handle_speed_limit(sign)
                elif isinstance(sign, (NoEntrySign, NoTrafficSign)):
                    self._handle_no_entry_or_no_traffic(sign)
                elif isinstance(sign, TrafficLightSign):
                    self._handle_traffic_light(sign)
                elif isinstance(sign, DetourSign):
                    self._handle_detour(sign)
                elif isinstance(sign, IntersectionRestrictedLaneSign):
                    self._handle_intersection_restricted_lane(sign)
                elif isinstance(sign, EndOfRestrictedLaneSign):
                    pass
                elif isinstance(sign, RestrictedLaneSign):
                    self._handle_restricted_lane(sign)
                elif isinstance(sign, NoStoppingAllowedSign):
                    self._handle_no_stopping(sign)
                elif isinstance(sign, BusStationSign):
                    self._handle_bus_station(sign)
                elif isinstance(sign, OnlyAutoSign):
                    self._handle_only_auto(sign)
                elif isinstance(sign, RightTurnRule):
                    self._handle_right_turn_rule(sign)
                elif isinstance(sign, RightHandYieldSign):
                    self._handle_yield_sign(sign)
                elif isinstance(sign, YieldSign):
                    self._handle_yield_sign(sign)
                elif isinstance(sign, EndMainRoadSign):
                    self._handle_end_main_road(sign)
                elif isinstance(sign, (NoRightTurnSign, NoLeftTurnSign, NoUTurnSign)):
                    self._handle_no_turn(sign)
                elif isinstance(sign, OneWayEntrySign):
                    self._handle_no_turn(sign)
                elif isinstance(sign, NoOvertakingSign):
                    self._handle_no_overtaking(sign)
                elif isinstance(sign, (DirectionSign, PGDirectionSign, LaneAllowedDirectionSign)):
                    self._handle_direction_compliance(sign)
                elif isinstance(sign, MainRoadSign):
                    self._handle_main_road(sign)
                elif isinstance(sign, (SecondaryRoadSign,
                                       SecondaryRoadLeftSign, SecondaryRoadRightSign,
                                       BaseEndOfZoneSign)):
                    pass  # informational — no action needed
            except Exception as exc:
                logger.debug("Error processing %s: %s", type(sign).__name__, exc)

        self._process_rules()
        self._handle_intersection_priority()       # PG (RoadPriorityAssigner)
        self._handle_intersection_priority_sumo()  # SUMO (lane.turns[i]["priority"])
        # Keep / apply the direction-exit snap *before* base IDM steering so
        # routing_target_lane and current_lane see the allowed via this step.
        self._maybe_snap_to_direction_exit()
        # After a forced connector hop, MetaDrive IDM PIDs wind up on the
        # curved vias and drive the ego off-road on the next edges. Zero the
        # integrators each step so lane-keeping stays P-dominated.
        if self._direction_exit_snapped:
            self._reset_steering_pids()

    # ------------------------------------------------------------------
    # Throttle post-processing
    # ------------------------------------------------------------------

    def _apply_speed_constraints(self, throttle, speed_kmh):
        if self._speed_cap is not None:
            if self._speed_cap < 1.0:
                throttle = self.BRAKE_ACTION
            elif speed_kmh > self._speed_cap:
                overshoot = speed_kmh - self._speed_cap
                brake = np.clip(-BRAKE_PROP_GAIN * overshoot - BRAKE_BIAS,
                                self.BRAKE_ACTION, 0.0)
                throttle = min(throttle, brake)
            elif speed_kmh > self._speed_cap - 5.0:
                throttle = min(throttle, 0.0)

        if self._speed_floor is not None:
            if speed_kmh < self._speed_floor:
                deficit = self._speed_floor - speed_kmh
                accel = min(FLOOR_PROP_GAIN * deficit + FLOOR_BIAS, 1.0)
                throttle = max(throttle, accel)

        return throttle
