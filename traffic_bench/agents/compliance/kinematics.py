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
from typing import Optional

import numpy as np

from metadrive.utils.math import wrap_to_pi



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
FLOOR_BIAS = 0.4                   # constant offset for acceleration floor
FLOOR_PUSH_CLEAR_M = 25.0          # push to a 4.6 floor against the policy's own
                                   # braking only with at least this much (or the
                                   # braking distance) of free lane ahead
FLOOR_OVERSHOOT_KMH = 3.0          # aim this far ABOVE the min so a policy's
                                   # pull-back (its own target is below the min)
                                   # doesn't dip below min - tolerance

# Once this close to the stop line, keep the stop FSM engaged even when
# speed≈0 makes ``_approach_dist(0)`` collapse to 0 (otherwise the expert
# drops out of the wait loop, kick-starts, then re-brakes — stutter after stop).
STOP_ENGAGE_DISTANCE_M = 12.0

# Mid-route U-turns on compliant no-turn (3.18.1 / 3.18.2) detours only.
# Rule-based phases (PlanT2 opt-in via ``APPLY_UTURN_ZONE_ASSIST``):
#   approach → mid-road (between own / oncoming) → 180° spin → release.
# No body teleports. Not used for 3.19 / one-way / direction signs.
UTURN_ZONE_LOOKAHEAD_M = 40.0
UTURN_ZONE_SPEED_CAP_KMH = 4.0
UTURN_ZONE_CREEP_KMH = 2.0
UTURN_ZONE_MIN_KMH = 1.5  # crawl floor — keep moving, never freeze at the lip
UTURN_ZONE_MAX_STEER = 1.0
UTURN_ZONE_SOFT_STEER = 0.45  # far approach: stay on-lane, do not circle
UTURN_ZONE_DESIRED_LAT_M = 1.2
UTURN_ZONE_HOLD_STEPS = 70
UTURN_ZONE_FORCE_NAV_REMAINING_M = 12.0
UTURN_ZONE_CENTER_REMAINING_M = 12.0  # start drifting to mid-road
UTURN_ZONE_SPIN_REMAINING_M = 5.0  # begin in-place 180° only near the via
UTURN_ZONE_MIDROAD_TOL_M = 0.55
UTURN_ZONE_SPIN_ALIGN_RAD = np.radians(20.0)
UTURN_ZONE_SPIN_RAD_PER_STEP = np.radians(7.0)  # kinematic yaw while holding mid-road
UTURN_ZONE_SPIN_HOLD_STEP_M = 0.18  # max XY correction toward mid per step
UTURN_ZONE_MAX_STEERING_DEG = 90.0  # temporary vehicle limit for tight OSM U-turns


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

class Kinematics:
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

        def _steering_control_for_lc(self, target_lane):
            ego = self.control_object
            long, lat = target_lane.local_coordinates(ego.position)
            lane_heading = target_lane.heading_theta_at(long + 1)
            v_heading = ego.heading_theta
            steering = self._get_heading_pid().get_result(
                -wrap_to_pi(lane_heading - v_heading)
            )
            lat_term = self._get_lateral_pid().get_result(-lat)
            # Peer LC on short 5.15.1 approaches needs a firmer lateral pull or
            # the merge never finishes before the junction.
            if abs(lat) > 0.35:
                lat_term *= 1.6
            steering += lat_term
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
                self._lc_final_sumo_num = lane_index_num(self._lc_target_lane)
                self._get_heading_pid().reset()
                self._get_lateral_pid().reset()

        def _begin_lane_change_by_sumo_num(self, sumo_lane_num: int) -> bool:
            """Lane-change to the peer whose SUMO lane number equals ``sumo_lane_num``."""
            if self._lc_target_lane is not None:
                return True
            cur = self._cur_lane_num()
            if cur is not None and int(cur) == int(sumo_lane_num):
                return True
            self._lc_final_sumo_num = int(sumo_lane_num)
            # Try ref lanes first (adjacent lanes from navigation).
            ref = self._get_ref_lanes() or []
            for lane in ref:
                if lane_index_num(lane) == int(sumo_lane_num):
                    self._lc_target_lane = lane
                    self._get_heading_pid().reset()
                    self._get_lateral_pid().reset()
                    return True
            # Fallback: search ALL lanes on the same edge (for multi-lane-change).
            try:
                cur_lane = self.control_object.lane
                if cur_lane is not None:
                    cur_idx = getattr(cur_lane, "index", None)
                    # Extract edge ID from lane index: "lane_<edge>_<num>" -> "<edge>"
                    if isinstance(cur_idx, str) and cur_idx.startswith("lane_"):
                        parts = cur_idx[5:].rsplit("_", 1)
                        if len(parts) == 2:
                            edge_id = parts[0]
                            road_network = self.control_object.navigation.map.road_network
                            # Try to find target lane directly: "lane_<edge>_<target_num>"
                            target_lid = f"lane_{edge_id}_{sumo_lane_num}"
                            try:
                                target_lane = road_network.get_lane(target_lid)
                                if target_lane is not None:
                                    self._lc_target_lane = target_lane
                                    self._get_heading_pid().reset()
                                    self._get_lateral_pid().reset()
                                    return True
                            except Exception:
                                pass
            except Exception:
                pass
            return False

        def _update_lane_change(self):
            if self._lc_target_lane is None and self._lc_final_sumo_num is None:
                return
            final_num = self._lc_final_sumo_num
            if final_num is None:
                final_num = lane_index_num(self._lc_target_lane)
            cur_num = self._cur_lane_num()
            cur_idx = getattr(self.control_object.lane, "index", None)
            ref = self._get_ref_lanes() or []

            # Done when on the final SUMO lane and centered.
            if cur_num is not None and final_num is not None and int(cur_num) == int(final_num):
                lane = self.control_object.lane
                if lane is not None:
                    _, lat = lane.local_coordinates(self.control_object.position)
                    if abs(lat) < LC_COMPLETE_LAT:
                        self._lc_target_lane = None
                        self._lc_final_sumo_num = None
                        return

            # Aim at the next hop toward the final lane (supports L0→L2).
            if final_num is None or cur_num is None:
                return
            if int(cur_num) == int(final_num):
                # On final lane but not yet centered — keep aiming at it.
                aim_num = int(final_num)
            else:
                step = 1 if int(final_num) > int(cur_num) else -1
                aim_num = int(cur_num) + step

            aim = None
            for lane in ref:
                if lane_index_num(lane) == aim_num:
                    aim = lane
                    break
            if aim is None and self._lc_target_lane is not None:
                # Keep previous aim object if peers aren't in ref this tick.
                if lane_index_num(self._lc_target_lane) == aim_num:
                    aim = self._lc_target_lane
            if aim is None:
                # Resolve by id on the current edge.
                try:
                    cur_lane = self.control_object.lane
                    cur_lid = getattr(cur_lane, "index", None)
                    if isinstance(cur_lid, str) and cur_lid.startswith("lane_"):
                        edge_id = cur_lid[5:].rsplit("_", 1)[0]
                        road_network = self.control_object.navigation.map.road_network
                        aim = road_network.get_lane(f"lane_{edge_id}_{aim_num}")
                except Exception:
                    aim = None
            if aim is not None:
                self._lc_target_lane = aim
            # Drop LC only if we left the approach entirely (no usable aim).
            elif ref and self._lc_target_lane is not None:
                tgt_idx = getattr(self._lc_target_lane, "index", None)
                ref_idxs = {getattr(l, "index", None) for l in ref}
                if tgt_idx not in ref_idxs and cur_idx not in ref_idxs:
                    self._lc_target_lane = None
                    self._lc_final_sumo_num = None

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

        def _lanes_on_edges(self, edges) -> set:
            """All EdgeRoadNetwork lane ids that belong to the given SUMO edges.

            Lane ids look like ``lane_<edge>_<num>`` or ``<edge>_<num>``; strip the
            optional prefix and the trailing lane number to recover the raw SUMO
            edge id (e.g. ``539307698#1``).
            """
            want = {str(e) for e in (edges or ())}
            if not want:
                return set()
            graph = getattr(self.engine.current_map.road_network, "graph", None) or {}
            out = set()
            for lid in graph:
                if not isinstance(lid, str):
                    continue
                raw = lid[5:] if lid.startswith("lane_") else lid
                edge = raw.rsplit("_", 1)[0] if "_" in raw else raw
                if edge in want:
                    out.add(lid)
            return out

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

        def _reroute_sumo_from_current_lane(self) -> bool:
            """Replan spawn→dest starting from the vehicle's current lane (no blocks)."""
            nav = getattr(self.control_object, "navigation", None)
            if nav is None or not self._is_sumo_edge_nav(nav):
                return False
            checkpoints = list(getattr(nav, "checkpoints", None) or [])
            if len(checkpoints) < 2:
                return False
            destination = checkpoints[-1]
            start = getattr(self.control_object.lane, "index", None)
            if not isinstance(start, str) or not isinstance(destination, str):
                return False
            cache_key = ("sumo_from_cur", start, destination)
            if cache_key in self._rerouted_edges and self._rerouted_edges[cache_key]:
                return True
            path = self._find_sumo_path_avoiding_lanes(start, destination, blocked_lanes=())
            ok = bool(path) and self._apply_sumo_nav_path(nav, path)
            if ok:
                self._rerouted_edges[cache_key] = True
            return ok

        def _soft_cap_into_next_checkpoint_via(self) -> None:
            """Slow before a short next-hop connector."""
            ego = self.control_object
            nav = getattr(ego, "navigation", None)
            lane = getattr(ego, "lane", None)
            if nav is None or lane is None:
                return
            ckpts = list(getattr(nav, "checkpoints", None) or [])
            cur_id = getattr(lane, "index", None)
            if not isinstance(cur_id, str) or cur_id not in ckpts:
                return
            try:
                i = ckpts.index(cur_id)
            except ValueError:
                return
            if i + 1 >= len(ckpts):
                return
            next_id = ckpts[i + 1]
            try:
                long, _ = lane.local_coordinates(ego.position)
                remaining = float(lane.length) - float(long)
            except Exception:
                return
            next_len = None
            try:
                next_lane = self.engine.current_map.road_network.get_lane(next_id)
                next_len = float(getattr(next_lane, "length", 0.0) or 0.0)
            except Exception:
                next_len = None
            # Short internal vias / connectors: creep in the last metres.
            short_via = (
                isinstance(next_id, str)
                and (next_id.startswith("lane_:") or (next_len is not None and next_len < 12.0))
            )
            if not short_via:
                if remaining <= 15.0:
                    self._cap_speed(16.0)
                return
            if remaining <= 25.0:
                self._cap_speed(14.0)
            if remaining <= 12.0:
                self._cap_speed(10.0)
            if remaining <= 6.0:
                self._cap_speed(8.0)

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

        @staticmethod
        def _sumo_edge_from_lane_id(lane_id) -> Optional[str]:
            if not isinstance(lane_id, str) or not lane_id.startswith("lane_"):
                return None
            raw = lane_id[5:]
            if "_" not in raw:
                return raw or None
            edge, _ = raw.rsplit("_", 1)
            return edge or None

        @staticmethod
        def _sumo_edges_are_reverse(a: str, b: str) -> bool:
            """Opposite carriageways of the same OSM edge (incl. ``#segment``)."""
            if not a or not b:
                return False
            return (
                a.lstrip("-") == b.lstrip("-")
                and a.startswith("-") != b.startswith("-")
            )

        @staticmethod
        def _is_internal_lane_id(lane_id) -> bool:
            if not isinstance(lane_id, str):
                return False
            return lane_id.startswith("lane_:") or lane_id.startswith(":")

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

        def _floor_road_clear(self, speed_kmh) -> bool:
            """No vehicle or obstacle on the ego lane within the braking distance
            (at least FLOOR_PUSH_CLEAR_M). Conservative: any failure = not clear."""
            try:
                from metadrive.policy.idm_policy import FrontBackObjects

                ego = self.control_object
                lane = getattr(self, "_lc_target_lane", None) or ego.lane
                if lane is None:
                    return False
                look = max(FLOOR_PUSH_CLEAR_M, braking_distance(speed_kmh, 0.0))
                objs = ego.lidar.get_surrounding_objects(ego)
                fb = FrontBackObjects.get_find_front_back_objs_single_lane(
                    objs, lane, ego.position, max_distance=look)
                return fb.front_object() is None
            except Exception:
                return False

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
                # Aim slightly ABOVE the minimum so a policy whose own desired speed
                # is below the min doesn't keep dipping under min - tolerance. NN
                # policies (carl/plant2) have no internal target to raise, so this
                # firm throttle floor is their only lever to reach/hold the minimum.
                #
                # Never turn a braking request into acceleration. The target speed
                # is already raised to the floor upstream, so a negative throttle
                # arriving here means the controller is braking for something --
                # a slower leader, a curve cap -- and the floor must yield to it.
                # Overriding it drove every rule expert into the car ahead on the
                # 4.6 scenes with traffic (20 of 24 rows, crash at step ~25, while
                # the sign-unaware idm on the same rows arrived 24 of 24).
                #
                # The exception is a clear road: CaRL and PPO hold their own
                # cruise (30 and 45 km/h) with a zero or slightly negative
                # throttle, so under the rule above the push never engaged and
                # both failed nearly every 50/60 plate even without traffic
                # (v6 eval: carl_rule 98%, ppo_rule 60% of nominal episodes).
                # With no leader within braking distance the negative throttle
                # is a preference, not a safety brake, and the floor wins.
                floor_target = self._speed_floor + FLOOR_OVERSHOOT_KMH
                if speed_kmh < floor_target and (throttle >= 0.0 or self._floor_road_clear(speed_kmh)):
                    deficit = floor_target - speed_kmh
                    accel = min(FLOOR_PROP_GAIN * deficit + FLOOR_BIAS, 1.0)
                    throttle = max(throttle, accel)

            return throttle
