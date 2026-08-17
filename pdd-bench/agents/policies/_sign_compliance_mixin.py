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

from traffic_signs.bus_station_sign import BusStationSign
from traffic_signs.detour_sign import DetourSign
from traffic_signs.direction_sign import DirectionSign
from traffic_signs.lane_directions_sign import LaneDirectionsSign
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
YIELD_STOP_BEFORE_END_M = 5.0      # hold yield / RH-rule stop this far before lane end

BRAKE_PROP_GAIN = 0.05             # proportional gain for braking
BRAKE_BIAS = 0.15                  # constant offset for braking
FLOOR_PROP_GAIN = 0.08             # proportional gain for acceleration floor
FLOOR_BIAS = 0.4                   # constant offset for acceleration floor
FLOOR_OVERSHOOT_KMH = 3.0          # aim this far ABOVE the min so a policy's
                                   # pull-back (its own target is below the min)
                                   # doesn't dip below min - tolerance

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
# 4 m (was 2): CaRL often brakes near the curb ~3–4 m before the end after the
# rightward exit-aim pull, and never reaches a 2 m snap → stuck on the shoulder.
DIRECTION_EXIT_SNAP_REMAINING_M = 4.0
# A one-shot teleport (1.5–2 m in a single frame) reads as a visible "jump"
# on GIFs. Instead, glide: pull ego toward the allowed via by at most
# GLIDE_MAX_STEP_M per step (plus a bounded heading turn) until it actually
# sits on the via, then finalize nav/hold. Also un-sticks a stalled CaRL,
# since the glide moves the body regardless of throttle.
DIRECTION_EXIT_GLIDE_MAX_STEP_M = 0.40
DIRECTION_EXIT_GLIDE_MAX_TURN_RAD = np.radians(6.0)
DIRECTION_EXIT_GLIDE_DONE_LAT_M = 0.4
DIRECTION_EXIT_GLIDE_MAX_STEPS = 40  # safety: hard-snap if glide drags on

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

class SignComplianceMixin:
    """Traffic sign compliance logic shared between PPO and IDM experts."""

    STOP_WAIT_STEPS = 30
    NO_STOP_MIN_SPEED_KMH = 5.0
    BRAKE_ACTION = -1.0
    # Mid-route U-turn crawl/spin assist (3.18 detours). Opt-in — PlanT2 only.
    APPLY_UTURN_ZONE_ASSIST = False

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
        self._lc_final_sumo_num = None
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
        # Whether exit snap should physically relocate ego to the connector.
        # For plain 5.15.2 DirectionSign this creates a visible "forward jump";
        # keep only nav/checkpoint snap there.
        self._direction_exit_position_snap = True
        self._direction_exit_glide_steps = 0
        self._direction_exit_hold_lane = None
        self._direction_exit_hold_steps = 0
        # One-way first hop: slow into the short connector.
        self._direction_exit_creep = False
        # Set per-step when 5.7.x nav already avoids wrong-way lanes.
        self._one_way_nav_clean = False
        # Mid-route U-turn assist (3.18.1/3.18.2 only) — steering only.
        self._no_turn_318_context = False
        self._uturn_via_lane = None
        self._uturn_source_lane = None
        self._uturn_hold_lane = None
        self._uturn_hold_steps = 0
        self._uturn_bias = 0.0
        self._uturn_phase = None  # "approach" | "center" | "spin"
        self._uturn_spinning = False
        self._uturn_spin_dir = None
        self._uturn_spin_dir_flipped = False
        self._uturn_saved_max_steering = None
        # 5.15.1: after the one post-LC compliant replan, never rewrite nav
        # again (NN policies often oscillate across peer lanes mid-merge).
        self._lane_dirs_nav_locked = False
        self._lane_dirs_hold_applied = False

    # When True (IDM experts), stub nav to [lane, dest] during 5.15.1 peer LC
    # so IDM does not dive into an injected connector. NN policies set False.
    APPLY_LANE_DIRS_NAV_HOLD = True

    def _reset_sign_compliance(self):
        """Call on episode reset to clear stale state."""
        self._stop_states.clear()
        self._rerouted_edges.clear()
        self._lc_target_lane = None
        self._lc_final_sumo_num = None
        self._has_priority = False
        self._no_overtaking_active = False
        self._direction_exit_lane = None
        self._direction_exit_source_lane = None
        self._lane_dirs_nav_locked = False
        self._lane_dirs_hold_applied = False
        self._direction_exit_bias = 0.0
        self._direction_exit_snapped = False
        self._direction_exit_position_snap = True
        self._direction_exit_glide_steps = 0
        self._direction_exit_hold_lane = None
        self._direction_exit_hold_steps = 0
        self._direction_exit_creep = False
        self._one_way_nav_clean = False
        self._no_turn_318_context = False
        self._restore_uturn_steering_limit()
        self._uturn_via_lane = None
        self._uturn_source_lane = None
        self._uturn_hold_lane = None
        self._uturn_hold_steps = 0
        self._uturn_bias = 0.0
        self._uturn_phase = None
        self._uturn_spinning = False
        self._uturn_spin_dir = None
        self._uturn_spin_dir_flipped = False
        self._uturn_saved_max_steering = None

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

    def _install_lane_dirs_compliant_route(self, sign) -> bool:
        """Install target-lane → dest route for 5.15.1 after a soft peer LC.

        Prefers ``sign.compliant_edge_path`` (manifest ``dual_path.straight_path``)
        when present; otherwise BFS from the current lane. Nav-only — never
        relocates the body. Once successful, locks for the rest of the episode
        so peer-lane oscillation cannot trigger a second replan.
        """
        if getattr(self, "_lane_dirs_nav_locked", False):
            return True
        nav = getattr(self.control_object, "navigation", None)
        if nav is None or not self._is_sumo_edge_nav(nav):
            return False
        start = getattr(self.control_object.lane, "index", None)
        checkpoints = list(getattr(nav, "checkpoints", None) or [])
        destination = checkpoints[-1] if checkpoints else None
        if not isinstance(start, str) or not isinstance(destination, str):
            return False

        blocked = self._lane_directions_blocked_exits(sign, self.control_object.lane)
        edge_hint = tuple(getattr(sign, "compliant_edge_path", None) or ())
        cache_key = (
            "lane_dirs_compliant",
            start,
            destination,
            frozenset(blocked),
            edge_hint,
        )
        if cache_key in self._rerouted_edges and self._rerouted_edges[cache_key]:
            self._lane_dirs_nav_locked = True
            return True

        path = None
        # CaRL/PlanT2: keep MetaDrive shortest_path (no dual-path edge rewrite).
        # IDM experts still prefer compliant_edge_path — MetaDrive BFS often
        # still picks a short illegal spur even after the peer LC.
        use_metadrive = not getattr(self, "APPLY_LANE_DIRS_NAV_HOLD", True)
        if use_metadrive:
            try:
                nav.set_route(start, destination)
                path = list(getattr(nav, "checkpoints", None) or [])
                if path and path[-1] == destination and len(path) >= 2:
                    # Drop illegal first hops if MetaDrive still chose them.
                    if blocked and self._sumo_route_uses_blocked_source_exit(
                        nav, start, blocked
                    ):
                        path = None
                    else:
                        ok = True
                        logger.info(
                            "LaneDirections MetaDrive route: %s → %s via %d hops",
                            start,
                            destination,
                            len(path),
                        )
                        self._rerouted_edges[cache_key] = True
                        self._lane_dirs_nav_locked = True
                        self._lane_dirs_hold_applied = False
                        return True
            except Exception:
                path = None

        # Prefer waypoint edges from the dual-path crop — MetaDrive BFS often
        # still prefers a short illegal spur even after the peer LC.
        if edge_hint:
            path = self._find_sumo_path_via_edge_hint(
                start, destination, edge_hint, blocked_lanes=blocked, max_len=80
            )
        if not path:
            path = self._find_sumo_path_avoiding_lanes(
                start, destination, blocked_lanes=blocked, max_len=80
            )
        ok = bool(path) and path[-1] == destination and self._apply_sumo_nav_path(nav, path)
        if ok:
            logger.info(
                "LaneDirections compliant replan: %s → %s via %d hops",
                start,
                destination,
                len(path),
            )
            self._rerouted_edges[cache_key] = True
            self._lane_dirs_nav_locked = True
            self._lane_dirs_hold_applied = False
        else:
            self._rerouted_edges.pop(cache_key, None)
        return ok

    def _find_sumo_path_via_edge_hint(
        self,
        start_lane_id: str,
        goal_lane_id: str,
        edge_hint,
        blocked_lanes=None,
        *,
        max_len: int = 80,
    ):
        """BFS that prefers hops onto successive edges in ``edge_hint``."""
        want_edges = [str(e) for e in (edge_hint or ()) if e]
        if not want_edges:
            return None
        road_network = self.engine.current_map.road_network
        graph = getattr(road_network, "graph", None) or {}
        if start_lane_id not in graph:
            return None
        blocked = set(blocked_lanes or ())

        def _edge_of(lid: str) -> str:
            raw = lid[5:] if lid.startswith("lane_") else lid
            return raw.rsplit("_", 1)[0] if "_" in raw else raw

        # Progress = how many hint edges we've matched in order.
        from collections import deque

        start_prog = 0
        se = _edge_of(start_lane_id)
        for i, e in enumerate(want_edges):
            if se == e:
                start_prog = i + 1
                break
        queue = deque([(start_lane_id, [start_lane_id], start_prog)])
        seen = {(start_lane_id, start_prog)}
        while queue:
            lid, path, prog = queue.popleft()
            if lid == goal_lane_id and prog >= min(1, len(want_edges)):
                return path
            if len(path) > max_len:
                continue
            lane_data = graph.get(lid)
            if lane_data is None:
                continue
            exits = list(set(getattr(lane_data, "exit_lanes", None) or []))
            next_want = want_edges[prog] if prog < len(want_edges) else None

            def _sort_key(x: str):
                xe = _edge_of(x)
                prefer = 0 if (next_want is not None and xe == next_want) else 1
                return (prefer, 0 if x.endswith("_0") else 1, x)

            for nxt in sorted(exits, key=_sort_key):
                if nxt in blocked or nxt not in graph:
                    continue
                nprog = prog
                if next_want is not None and _edge_of(nxt) == next_want:
                    nprog = prog + 1
                key = (nxt, nprog)
                if key in seen:
                    continue
                seen.add(key)
                queue.append((nxt, path + [nxt], nprog))
        return None

    def _lane_directions_blocked_exits(self, sign, source_lane) -> set:
        """First-hop via/to targets not allowed for this approach lane (5.15.1)."""
        source_id = getattr(source_lane, "index", None)
        by_src = getattr(sign, "allowed_lanes_by_source", None) or {}
        allowed = set(by_src.get(source_id) or ())
        if not allowed:
            return set()
        blocked = set()
        for turn in getattr(source_lane, "turns", None) or []:
            to_lane = turn.get("to_lane")
            via = turn.get("via_lane")
            if to_lane and to_lane not in allowed:
                blocked.add(to_lane)
                if via:
                    blocked.add(via)
        return blocked

    def _hold_on_lane_until_lc(self, source_lane, blocked=None) -> bool:
        """Park nav on the current lane while a peer LC is in progress.

        Used mid lane-change so IDM does not dive into an injected connector
        (or a long alternate spur). Once on the target lane,
        ``_install_lane_dirs_compliant_route`` installs the real dest route.
        Idempotent: does not rewrite checkpoints if already held on ``source``.
        """
        if getattr(self, "_lane_dirs_nav_locked", False):
            return True
        nav = getattr(self.control_object, "navigation", None)
        if nav is None or not self._is_sumo_edge_nav(nav):
            return False
        source_id = getattr(source_lane, "index", None)
        if not isinstance(source_id, str):
            return False
        checkpoints = list(getattr(nav, "checkpoints", None) or [])
        dest = checkpoints[-1] if checkpoints else None
        # Already holding on this source — do not reset checkpoint indices.
        if (
            len(checkpoints) >= 2
            and checkpoints[0] == source_id
            and isinstance(dest, str)
            and checkpoints[-1] == dest
            and len(checkpoints) <= 2
        ):
            return True
        path = [source_id]
        if isinstance(dest, str) and dest != source_id:
            path.append(dest)
        if len(path) < 2:
            return False
        return self._apply_sumo_nav_path(nav, path)

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
        self._direction_exit_position_snap = True
        self._direction_exit_glide_steps = 0
        if self._direction_exit_hold_steps <= 0:
            self._direction_exit_creep = False
        # Keep _direction_exit_snapped so a loop-back onto the approach does
        # not re-snap (first-exit semantics for dual-path routes).

    def _soft_cap_into_next_checkpoint_via(self) -> None:
        """Slow before a short next-hop connector (nav-only, no body snap)."""
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

    def _arm_direction_exit_from_sign(self, sign) -> bool:
        """Remember the route's first allowed next-hop for near-junction steering."""
        return self._arm_direction_exit_from_lane(sign, getattr(sign, "lane", None))

    def _arm_direction_exit_from_lane(self, sign, source_lane) -> bool:
        """Arm exit aiming using ``source_lane`` as the departure lane."""
        # First-exit only: after the compliant hop, dual-path routes often
        # re-enter the same approach and must follow checkpoints without a
        # second forced right/left pull.
        if self._direction_exit_snapped:
            return False
        nav = getattr(self.control_object, "navigation", None)
        if nav is None or not self._is_sumo_edge_nav(nav):
            return False
        source_id = getattr(source_lane, "index", None)
        if source_lane is None or not isinstance(source_id, str):
            return False
        if isinstance(sign, LaneDirectionsSign):
            blocked = self._lane_directions_blocked_exits(sign, source_lane)
        else:
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
        # Direction boards (5.15.1 LaneDirectionsSign / 5.15.2 DirectionSign) and
        # one-way (5.7.x): never physically teleport onto the via. Route/nav +
        # steering bias are enough; body snaps read as visible "jumps" on GIFs.
        is_one_way = isinstance(sign, OneWayEntrySign)
        is_direction_board = isinstance(sign, DirectionSign)  # includes LaneDirectionsSign
        self._direction_exit_position_snap = not (is_direction_board or is_one_way)
        self._direction_exit_creep = is_one_way
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
            if is_one_way:
                # Compliant first hops are often short sharp vias (~5m).
                if remaining <= 20.0:
                    self._cap_speed(12.0)
                if remaining <= 10.0:
                    self._cap_speed(8.0)
            elif remaining <= DIRECTION_EXIT_LOOKAHEAD_M:
                self._cap_speed(DIRECTION_EXIT_SPEED_CAP_KMH)
        except Exception:
            pass
        return True

    def _update_direction_exit_steering_state(self):
        """Drop the exit target after leaving the signed approach."""
        if self._direction_exit_lane is None:
            return
        # Keep arming while hold is active so creep cap / nav glue still apply.
        if self._direction_exit_hold_steps > 0:
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

    def _finalize_direction_exit_snap(self) -> bool:
        exit_lane = self._direction_exit_lane
        self._direction_exit_snapped = True
        self._direction_exit_glide_steps = 0
        self._direction_exit_hold_lane = exit_lane
        self._direction_exit_hold_steps = 20
        self._force_nav_onto_lane(exit_lane)
        self._reset_steering_pids()
        logger.info(
            "Direction exit snap → %s", getattr(exit_lane, "index", None)
        )
        return True

    def _maybe_snap_to_direction_exit(self) -> bool:
        """Once per episode, guide ego onto the allowed via near the lane end.

        Instead of a one-shot teleport (a visible "jump" on GIFs), pull the
        body toward the via with a bounded per-step displacement/turn until it
        actually sits there, then finalize the nav/checkpoint switch.
        """
        # Keep nav glued to the snapped via for a few steps — ray_localization
        # otherwise often reassigns the overlapping straight connector.
        if self._direction_exit_hold_steps > 0 and self._direction_exit_hold_lane is not None:
            self._force_nav_onto_lane(self._direction_exit_hold_lane)
            self._direction_exit_hold_steps -= 1

        if self._direction_exit_snapped or self._direction_exit_lane is None:
            return False
        ego = self.control_object
        exit_lane = self._direction_exit_lane
        if self._direction_exit_glide_steps == 0:
            # Not gliding yet: arm only within the snap window on the approach.
            src = self._direction_exit_source_lane
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
            if not self._direction_exit_position_snap:
                # 5.15.2: pure nav/checkpoint snap, no physical relocation.
                return self._finalize_direction_exit_snap()

        self._direction_exit_glide_steps += 1
        try:
            long_v, lat_v = exit_lane.local_coordinates(ego.position)
        except Exception:
            long_v, lat_v = -1.0, 10.0
        if float(long_v) >= 0.35 and abs(float(lat_v)) <= DIRECTION_EXIT_GLIDE_DONE_LAT_M:
            # Final exact placement on the via centerline (≤ DONE_LAT lateral,
            # invisible on GIFs) so IDM starts the connector from the same
            # clean state the old one-shot snap provided.
            s = float(np.clip(float(long_v), 0.35, max(0.35, float(exit_lane.length) - 0.3)))
            try:
                ego.set_position(exit_lane.position(s, 0))
                ego.set_heading_theta(exit_lane.heading_theta_at(s))
            except Exception as exc:
                logger.debug("Direction exit final placement failed: %s", exc)
            return self._finalize_direction_exit_snap()
        if self._direction_exit_glide_steps > DIRECTION_EXIT_GLIDE_MAX_STEPS:
            # Glide never converged (physics fighting back) — hard snap as a
            # last resort so the episode still takes the compliant exit.
            s = min(1.5, max(0.5, float(exit_lane.length) * 0.25))
            try:
                ego.set_position(exit_lane.position(s, 0))
                ego.set_heading_theta(exit_lane.heading_theta_at(s))
            except Exception as exc:
                logger.debug("Direction exit snap failed: %s", exc)
                return False
            return self._finalize_direction_exit_snap()
        # One bounded glide increment toward a point slightly ahead on the via.
        s_cap = max(0.5, float(exit_lane.length) - 0.5)
        s_t = float(np.clip(float(long_v) + 0.6, 0.35, s_cap))
        try:
            target = np.asarray(exit_lane.position(s_t, 0), dtype=float)[:2]
            target_heading = float(exit_lane.heading_theta_at(s_t))
        except Exception as exc:
            logger.debug("Direction exit glide failed: %s", exc)
            return False
        try:
            pos = np.asarray(ego.position, dtype=float)[:2]
            delta = target - pos
            dist = float(np.linalg.norm(delta))
            if dist > 1e-6:
                step = delta * (min(DIRECTION_EXIT_GLIDE_MAX_STEP_M, dist) / dist)
                ego.set_position(pos + step)
            heading = float(ego.heading_theta)
            dh = float(wrap_to_pi(target_heading - heading))
            dh = float(np.clip(dh, -DIRECTION_EXIT_GLIDE_MAX_TURN_RAD, DIRECTION_EXIT_GLIDE_MAX_TURN_RAD))
            ego.set_heading_theta(heading + dh)
        except Exception as exc:
            logger.debug("Direction exit glide failed: %s", exc)
        return False

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
                if getattr(self, "_direction_exit_creep", False):
                    self._cap_speed(8.0)
                long_v, lat_v = hold.local_coordinates(ego.position)
                heading = hold.heading_theta_at(max(0.0, float(long_v)) + 0.5)
                heading_err = wrap_to_pi(heading - ego.heading_theta)
                return float(np.clip(1.4 * (-heading_err) + 0.55 * (-lat_v), -1.0, 1.0))
            except Exception:
                pass
        elif self._direction_exit_hold_lane is not None and self._direction_exit_hold_steps <= 0:
            self._direction_exit_hold_lane = None
            self._direction_exit_creep = False
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
    # Mid-route U-turn assist (no-turn 3.18 compliant detours)
    # ------------------------------------------------------------------

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

    def _clear_uturn_assist(self) -> None:
        self._restore_uturn_steering_limit()
        self._uturn_via_lane = None
        self._uturn_source_lane = None
        self._uturn_hold_lane = None
        self._uturn_hold_steps = 0
        self._uturn_bias = 0.0
        self._uturn_phase = None
        self._uturn_spinning = False
        self._uturn_spin_dir = None
        self._uturn_spin_dir_flipped = False

    def _scene_has_no_turn_318(self) -> bool:
        """True iff this episode has a 3.18.1 / 3.18.2 sign (not 3.19)."""
        if getattr(self, "_no_turn_318_context", False):
            return True
        try:
            return any(
                isinstance(s, (NoRightTurnSign, NoLeftTurnSign))
                for s in self._get_signs()
            )
        except Exception:
            return False

    def _boost_uturn_steering_limit(self) -> None:
        """Temporarily widen vehicle max steering for a tight OSM U-turn."""
        ego = self.control_object
        if self._uturn_saved_max_steering is None:
            self._uturn_saved_max_steering = getattr(ego, "max_steering", None)
        try:
            cur = float(getattr(ego, "max_steering", 50.0) or 50.0)
            ego.max_steering = max(cur, float(UTURN_ZONE_MAX_STEERING_DEG))
        except Exception:
            pass

    def _restore_uturn_steering_limit(self) -> None:
        saved = getattr(self, "_uturn_saved_max_steering", None)
        if saved is None:
            return
        try:
            self.control_object.max_steering = saved
        except Exception:
            pass
        self._uturn_saved_max_steering = None

    def _find_uturn_next_hop(self, ego_lane, ckpts: list, cur_idx: int):
        """Return (via_lane, bias) only for true U-turns (``dir=t`` / reverse)."""
        if ego_lane is None or cur_idx + 1 >= len(ckpts):
            return None, 0.0
        next_id = ckpts[cur_idx + 1]
        turn_dir = None
        for turn in getattr(ego_lane, "turns", None) or []:
            via_id = turn.get("via_lane")
            to_id = turn.get("to_lane")
            if via_id == next_id or to_id == next_id:
                turn_dir = self._normalize_turn_dir(turn.get("direction"))
                break
        try:
            via = self.engine.current_map.road_network.get_lane(next_id)
        except Exception:
            via = None
        if turn_dir == "t" and via is not None:
            return via, 0.85

        cur_edge = self._sumo_edge_from_lane_id(getattr(ego_lane, "index", None))
        probe_ids = [next_id]
        if self._is_internal_lane_id(next_id) and cur_idx + 2 < len(ckpts):
            probe_ids.append(ckpts[cur_idx + 2])
        for pid in probe_ids:
            nxt_edge = self._sumo_edge_from_lane_id(pid)
            if cur_edge and nxt_edge and self._sumo_edges_are_reverse(cur_edge, nxt_edge):
                if via is not None:
                    return via, 0.85

        # Heading reversal across a short via (~U-turn geometry).
        try:
            if via is None:
                return None, 0.0
            via_len = float(getattr(via, "length", 0.0) or 0.0)
            if via_len > 16.0:
                return None, 0.0
            h0 = float(ego_lane.heading_theta_at(max(0.0, float(ego_lane.length) - 0.5)))
            h1 = float(via.heading_theta_at(min(1.0, max(0.2, via_len * 0.55))))
            if abs(float(wrap_to_pi(h1 - h0))) >= (130.0 * np.pi / 180.0):
                return via, 0.85
        except Exception:
            pass
        return None, 0.0

    def _arm_uturn_from_nav(self) -> bool:
        """Arm mid-route U-turn assist from current nav next-hop.

        Only for 3.18.1 / 3.18.2 scenes (plant2 opt-in). Never for 3.19 /
        one-way / direction signs.
        """
        if not getattr(self, "APPLY_UTURN_ZONE_ASSIST", False):
            return False
        if not self._scene_has_no_turn_318():
            return False
        if self._uturn_via_lane is not None:
            # Sticky until `_clear_uturn_assist`: re-arming on the connector
            # would replace the via with the reverse carriageway.
            return True
        if self._uturn_hold_steps > 0 and self._uturn_via_lane is not None:
            return True
        # First signed exit owns the approach until it snaps/clears.
        if (
            self._direction_exit_lane is not None
            and not self._direction_exit_snapped
        ):
            return False
        # While still physically on the first-exit hold lane, do not steal
        # steering. Once ego is past it (e.g. on the U-turn approach), arm.
        if self._direction_exit_hold_steps > 0:
            hold = self._direction_exit_hold_lane
            ego_lane = getattr(self.control_object, "lane", None)
            if hold is not None and ego_lane is not None and (
                same_lane(ego_lane, hold) or on_same_road(ego_lane, hold)
            ):
                return False

        ego = self.control_object
        nav = getattr(ego, "navigation", None)
        lane = getattr(ego, "lane", None)
        if nav is None or lane is None or not self._is_sumo_edge_nav(nav):
            return False
        ckpts = list(getattr(nav, "checkpoints", None) or [])
        cur_id = getattr(lane, "index", None)
        if not isinstance(cur_id, str) or cur_id not in ckpts:
            return False
        try:
            cur_idx = ckpts.index(cur_id)
        except ValueError:
            return False

        via, bias = self._find_uturn_next_hop(lane, ckpts, cur_idx)
        if via is None:
            return False

        self._uturn_via_lane = via
        self._uturn_source_lane = lane
        self._uturn_bias = float(bias)
        self._uturn_phase = "approach"
        return True

    def _uturn_spin_dir_from_geometry(self, via, src, rev) -> float:
        """Pick the U-turn steer sign from via curvature / lateral offset.

        Shortest-path heading to reverse is ambiguous near ±180° and often
        picks the wrong side → OOR. Via bend / side-of-road is stable.
        Convention (this codebase): ``steer = -heading_err`` so negative
        steer increases heading (CCW).
        """
        # 1) Via curvature: which way the connector bends.
        if via is not None:
            try:
                via_len = float(getattr(via, "length", 0.0) or 0.0)
                if via_len > 0.4:
                    h0 = float(via.heading_theta_at(min(0.35, via_len * 0.15)))
                    h1 = float(via.heading_theta_at(
                        min(via_len - 0.05, max(via_len * 0.55, via_len * 0.35))
                    ))
                    dh = float(wrap_to_pi(h1 - h0))
                    if abs(dh) >= (12.0 * np.pi / 180.0):
                        return -1.0 if dh > 0.0 else 1.0
            except Exception:
                pass

        # 2) Which side of the approach the via sits on.
        # Empiric on SUMO EdgeRoadNetwork: negative action → +lane.lat.
        if via is not None and src is not None:
            try:
                via_len = float(getattr(via, "length", 1.0) or 1.0)
                aim = min(max(0.4, via_len * 0.45), max(0.3, via_len - 0.1))
                _, via_lat = src.local_coordinates(via.position(aim, 0))
                if abs(float(via_lat)) >= 0.25:
                    return -1.0 if float(via_lat) > 0.0 else 1.0
            except Exception:
                pass

        # 3) Cross product approach_heading × (via - ego) in XY.
        if via is not None:
            try:
                ego = self.control_object
                pos = np.asarray(ego.position, dtype=float)[:2]
                via_len = float(getattr(via, "length", 1.0) or 1.0)
                tgt = np.asarray(
                    via.position(min(via_len * 0.5, max(0.3, via_len - 0.1)), 0),
                    dtype=float,
                )[:2]
                delta = tgt - pos
                if src is not None:
                    long_a, _ = src.local_coordinates(ego.position)
                    hx = float(np.cos(src.heading_theta_at(
                        min(float(src.length) - 0.1, max(0.0, float(long_a)))
                    )))
                    hy = float(np.sin(src.heading_theta_at(
                        min(float(src.length) - 0.1, max(0.0, float(long_a)))
                    )))
                else:
                    hx = float(np.cos(ego.heading_theta))
                    hy = float(np.sin(ego.heading_theta))
                cross = hx * float(delta[1]) - hy * float(delta[0])
                if abs(cross) >= 1e-3:
                    # Positive cross = target left of heading = CCW = neg steer.
                    return -1.0 if cross > 0.0 else 1.0
            except Exception:
                pass

        return 1.0 if self._uturn_bias >= 0 else -1.0

    def _uturn_pure_pursuit_steer(
        self, ego, via, fallback_steer: float, *, aggressive: bool = True
    ) -> float:
        """Steer toward a point ahead on the U-turn via (smooth, no teleport)."""
        try:
            via_len = float(getattr(via, "length", 1.0) or 1.0)
            # Close: aim deep into the hook. Far: aim near via entry so we
            # drive along the approach instead of spinning toward mid-via.
            if aggressive:
                aim_s = min(max(0.8, via_len * 0.75), max(0.5, via_len - 0.1))
            else:
                aim_s = min(0.6, max(0.2, via_len * 0.15))
            target = np.asarray(via.position(aim_s, 0), dtype=float)[:2]
            pos = np.asarray(ego.position, dtype=float)[:2]
            delta = target - pos
            dist = float(np.linalg.norm(delta))
            if dist < 1e-4:
                return float(np.clip(fallback_steer, -1.0, 1.0))
            desired_heading = float(np.arctan2(delta[1], delta[0]))
            heading_err = wrap_to_pi(desired_heading - float(ego.heading_theta))
            if aggressive:
                if dist < 5.0:
                    gain = 4.0
                elif dist < 10.0:
                    gain = 3.2
                else:
                    gain = 2.4
                bias = 0.55 * float(self._uturn_bias)
                cap = UTURN_ZONE_MAX_STEER
            else:
                gain = 1.1
                bias = 0.15 * float(self._uturn_bias)
                cap = UTURN_ZONE_SOFT_STEER
            steer = gain * (-heading_err) + bias
            return float(np.clip(steer, -cap, cap))
        except Exception:
            return float(np.clip(fallback_steer, -1.0, 1.0))

    def _uturn_follow_approach_steer(self, ego, src, via, steering: float) -> float:
        """Stay on the approach lane and ease toward the via entry."""
        try:
            long, lat = src.local_coordinates(ego.position)
            # Look ahead along the approach toward the via.
            aim = min(float(src.length) - 0.1, max(0.5, float(long) + 4.0))
            heading = src.heading_theta_at(aim)
            heading_err = wrap_to_pi(heading - ego.heading_theta)
            desired_lat = 0.0
            try:
                _, via_lat = src.local_coordinates(via.position(0.3, 0))
                desired_lat = float(np.clip(
                    via_lat, -UTURN_ZONE_DESIRED_LAT_M, UTURN_ZONE_DESIRED_LAT_M
                ))
            except Exception:
                desired_lat = (
                    -0.6 * UTURN_ZONE_DESIRED_LAT_M
                    if self._uturn_bias >= 0
                    else 0.6 * UTURN_ZONE_DESIRED_LAT_M
                )
            lat_err = float(lat) - desired_lat
            lane_steer = 1.6 * (-heading_err) + 0.9 * lat_err
            pp = self._uturn_pure_pursuit_steer(
                ego, via, steering, aggressive=False
            )
            # Prefer lane following far out; mild via pull only.
            steer = 0.75 * lane_steer + 0.25 * pp
            return float(np.clip(
                steer, -UTURN_ZONE_SOFT_STEER, UTURN_ZONE_SOFT_STEER
            ))
        except Exception:
            return self._uturn_pure_pursuit_steer(
                ego, via, steering, aggressive=False
            )

    def _uturn_reverse_lane_after_via(self):
        """Lane after the U-turn via on the nav path (opposite carriageway)."""
        via = self._uturn_via_lane
        if via is None:
            return None
        nav = getattr(self.control_object, "navigation", None)
        if nav is None or not self._is_sumo_edge_nav(nav):
            return None
        ckpts = list(getattr(nav, "checkpoints", None) or [])
        via_id = getattr(via, "index", None)
        if not isinstance(via_id, str) or via_id not in ckpts:
            return None
        try:
            idx = ckpts.index(via_id)
        except ValueError:
            return None
        rn = self.engine.current_map.road_network
        for pid in ckpts[idx + 1 : idx + 3]:
            if self._is_internal_lane_id(pid):
                continue
            try:
                return rn.get_lane(pid)
            except Exception:
                return None
        return None

    def _uturn_midroad_target(self, src, rev, via, *, at_via: bool = False):
        """World XY of the mid-road point between approach and oncoming.

        Halfway between the approach centerline and the reverse carriageway.
        Default: at the ego's current longitudinal position (lateral mid-road).
        ``at_via=True``: at the U-turn via entry (used only as a far waypoint).
        """
        ego = self.control_object
        try:
            if at_via and via is not None:
                via_len = float(getattr(via, "length", 1.0) or 1.0)
                anchor = np.asarray(via.position(min(0.4, via_len * 0.2), 0), dtype=float)[:2]
            else:
                anchor = np.asarray(ego.position, dtype=float)[:2]
        except Exception:
            anchor = np.asarray(ego.position, dtype=float)[:2]

        p_src = None
        p_rev = None
        if src is not None:
            try:
                long_s, _ = src.local_coordinates(anchor)
                long_s = float(np.clip(long_s, 0.0, max(0.1, float(src.length) - 0.1)))
                p_src = np.asarray(src.position(long_s, 0), dtype=float)[:2]
            except Exception:
                p_src = None
        if rev is not None:
            try:
                long_r, _ = rev.local_coordinates(anchor if p_src is None else p_src)
                long_r = float(np.clip(long_r, 0.0, max(0.1, float(rev.length) - 0.1)))
                p_rev = np.asarray(rev.position(long_r, 0), dtype=float)[:2]
            except Exception:
                p_rev = None

        if p_src is not None and p_rev is not None:
            return 0.5 * (p_src + p_rev)
        if p_src is not None and via is not None:
            try:
                via_len = float(getattr(via, "length", 1.0) or 1.0)
                p_via = np.asarray(
                    via.position(min(via_len * 0.35, max(0.3, via_len - 0.1)), 0),
                    dtype=float,
                )[:2]
                return 0.5 * (p_src + p_via)
            except Exception:
                return p_src
        if p_src is not None:
            return p_src
        return anchor

    def _uturn_steer_toward_xy(self, ego, target_xy, *, gain: float, cap: float) -> float:
        pos = np.asarray(ego.position, dtype=float)[:2]
        delta = np.asarray(target_xy, dtype=float)[:2] - pos
        dist = float(np.linalg.norm(delta))
        if dist < 1e-4:
            return 0.0
        desired = float(np.arctan2(delta[1], delta[0]))
        heading_err = float(wrap_to_pi(desired - float(ego.heading_theta)))
        return float(np.clip(gain * (-heading_err), -cap, cap))

    def _uturn_phase_approach_steer(self, ego, src, via, steering: float) -> float:
        """Mostly plant2 on approach; curb guard only when drifting wide.

        Continuous centerline P weaves. Pure plant2 walks off ~3 m OSM roads
        (lat → ±4). Soft graduated guard: no touch near center, stronger only
        as |lat| grows.
        """
        steer = float(steering)
        if src is not None:
            try:
                long_a, lat_a = src.local_coordinates(ego.position)
                rem = float(src.length) - float(long_a)
                lat_a = float(lat_a)
            except Exception:
                rem = None
                lat_a = None
            self._force_nav_onto_lane(src)
            if rem is not None and rem <= 18.0:
                self._cap_speed(max(UTURN_ZONE_SPEED_CAP_KMH, 5.0))
            if rem is not None and rem <= 14.0:
                self._cap_speed(UTURN_ZONE_CREEP_KMH)
                self._raise_floor(UTURN_ZONE_MIN_KMH)
            # Empiric MetaDrive: positive steer decreases lane.lat.
            if lat_a is not None:
                abs_lat = abs(lat_a)
                if abs_lat <= 0.40:
                    pass  # plant2 free near center
                elif abs_lat <= 0.85:
                    corr = float(np.clip(0.9 * lat_a, -0.22, 0.22))
                    steer = 0.75 * steer + 0.25 * corr
                else:
                    corr = float(np.clip(1.3 * lat_a, -0.55, 0.55))
                    steer = 0.25 * steer + 0.75 * corr
        return float(np.clip(steer, -1.0, 1.0))

    def _uturn_phase_center_steer(self, ego, src, via, rev, steering: float) -> float:
        """Drift onto the mid-road strip between approach and oncoming.

        Lateral error dominates — heading-keep at crawl speed previously
        overshot mid-road all the way to lat≈±4 (OOR) on narrow OSM.
        """
        self._cap_speed(UTURN_ZONE_CREEP_KMH)
        self._raise_floor(UTURN_ZONE_MIN_KMH)
        if src is not None:
            self._force_nav_onto_lane(src)
        desired_lat = 0.0
        try:
            mid = self._uturn_midroad_target(src, rev, via, at_via=False)
            if src is not None:
                _, desired_lat = src.local_coordinates(mid)
                # Do not ask for more than ~half a lane toward the median.
                desired_lat = float(np.clip(desired_lat, -1.15, 1.15))
        except Exception:
            desired_lat = 0.0
        try:
            long_a, lat_a = src.local_coordinates(ego.position)
            aim = min(float(src.length) - 0.1, max(0.5, float(long_a) + 2.0))
            h_err = float(wrap_to_pi(
                float(src.heading_theta_at(aim)) - float(ego.heading_theta)
            ))
            lat_err = float(lat_a) - float(desired_lat)
            # Empiric: positive steer decreases lane.lat. Lat first, mild heading.
            steer = 0.5 * (-h_err) + 1.6 * lat_err
            return float(np.clip(steer, -0.45, 0.45))
        except Exception:
            mid = self._uturn_midroad_target(src, rev, via, at_via=False)
            return self._uturn_steer_toward_xy(
                ego, mid, gain=1.4, cap=0.45
            )

    def _uturn_phase_spin_steer(self, ego, via, rev, steering: float) -> float:
        """In-place 180° yaw at mid-road until aligned with the opposite lane.

        Ackermann full-lock on ~3 m OSM roads sweeps into the curb. Instead:
        hold XY near the mid-road point and rotate heading kinematically
        (same pattern as direction-exit glide), then release to plant2.
        """
        self._uturn_spinning = True
        # Near-stop: no forward crawl that would arc off the mid-road.
        self._cap_speed(0.6)
        src = self._uturn_source_lane
        if self._uturn_spin_dir is None:
            self._uturn_spin_dir = self._uturn_spin_dir_from_geometry(via, src, rev)

        mid = self._uturn_midroad_target(src, rev, via, at_via=False)
        try:
            pos = np.asarray(ego.position, dtype=float)[:2]
            delta = np.asarray(mid, dtype=float)[:2] - pos
            dist = float(np.linalg.norm(delta))
            if dist > 1e-4:
                step = delta * (
                    min(UTURN_ZONE_SPIN_HOLD_STEP_M, dist) / dist
                )
                ego.set_position(pos + step)
            # Kill residual velocity so the hold does not fight physics.
            try:
                ego.set_velocity([1.0, 0.0], 0.0)
            except Exception:
                pass
        except Exception:
            pass

        # Target heading = reverse carriageway (or approach + π).
        target_h = None
        if rev is not None:
            try:
                long_r, _ = rev.local_coordinates(ego.position)
                aim = min(
                    float(rev.length) - 0.1,
                    max(0.5, float(long_r) + 1.5),
                )
                target_h = float(rev.heading_theta_at(aim))
                self._force_nav_onto_lane(rev)
            except Exception:
                target_h = None
        if target_h is None and src is not None:
            try:
                long_a, _ = src.local_coordinates(ego.position)
                h0 = float(src.heading_theta_at(
                    min(float(src.length) - 0.1, max(0.0, float(long_a)))
                ))
                target_h = float(wrap_to_pi(h0 + np.pi))
            except Exception:
                target_h = None

        if target_h is not None:
            err = float(wrap_to_pi(target_h - float(ego.heading_theta)))
            if abs(err) < UTURN_ZONE_SPIN_ALIGN_RAD:
                self._clear_uturn_assist()
                return float(np.clip(steering, -1.0, 1.0))
            # Prefer geometric shortest remaining yaw; fall back to locked dir.
            if abs(err) > (np.pi * 0.5) and self._uturn_spin_dir is not None:
                # Keep committed side once past 90° so we do not dither.
                dh = -float(np.sign(self._uturn_spin_dir)) * UTURN_ZONE_SPIN_RAD_PER_STEP
            else:
                dh = float(np.clip(
                    err, -UTURN_ZONE_SPIN_RAD_PER_STEP, UTURN_ZONE_SPIN_RAD_PER_STEP
                ))
            try:
                ego.set_heading_theta(float(ego.heading_theta) + dh)
            except Exception:
                pass
        else:
            # No reverse geometry — yaw by locked spin dir.
            try:
                dh = -float(self._uturn_spin_dir) * UTURN_ZONE_SPIN_RAD_PER_STEP
                ego.set_heading_theta(float(ego.heading_theta) + dh)
            except Exception:
                pass

        if via is not None:
            self._force_nav_onto_lane(via)
        # Steering unused during kinematic spin; return mild lock for logging.
        return float(np.clip(
            float(self._uturn_spin_dir or 0.0) * 0.3, -0.3, 0.3
        ))

    def _maybe_override_steering_for_uturn_zone(self, steering: float) -> float:
        """Rule-based mid-route U-turn for 3.18 detours (PlanT2 only).

        Phases:
          1. approach — plant2 steers; we only soft-cap speed
          2. center   — move to mid-road (between own / oncoming)
          3. spin     — in-place ~180° until aligned with reverse lane
          4. release  — clear assist, resume base policy
        """
        if not getattr(self, "APPLY_UTURN_ZONE_ASSIST", False):
            return float(np.clip(steering, -1.0, 1.0))
        if not self._scene_has_no_turn_318():
            return float(np.clip(steering, -1.0, 1.0))

        # U-turn assist outranks an in-progress lane-change once armed.
        if self._lc_target_lane is not None and self._uturn_via_lane is None:
            return float(np.clip(steering, -1.0, 1.0))

        ego = self.control_object
        self._arm_uturn_from_nav()
        via = self._uturn_via_lane
        if via is None:
            return float(np.clip(steering, -1.0, 1.0))

        self._boost_uturn_steering_limit()
        ego_lane = getattr(ego, "lane", None)
        rev = self._uturn_reverse_lane_after_via()
        src = self._uturn_source_lane
        phase = getattr(self, "_uturn_phase", None) or "approach"

        approach_rem = None
        if src is not None:
            try:
                long_a, _ = src.local_coordinates(ego.position)
                approach_rem = float(src.length) - float(long_a)
            except Exception:
                approach_rem = None
        geo_near_via = False
        try:
            long_v, lat_v = via.local_coordinates(ego.position)
            geo_near_via = (
                float(long_v) >= -0.8
                and float(long_v) <= float(via.length) + 1.5
                and abs(float(lat_v)) <= 2.5
            )
        except Exception:
            geo_near_via = False

        # Lateral-only distance to mid-road at current long (not via waypoint).
        mid_dist = 99.0
        try:
            mid = self._uturn_midroad_target(src, rev, via, at_via=False)
            if src is not None:
                _, cur_lat = src.local_coordinates(ego.position)
                _, mid_lat = src.local_coordinates(mid)
                mid_dist = abs(float(cur_lat) - float(mid_lat))
            else:
                mid_dist = float(np.linalg.norm(
                    np.asarray(ego.position, dtype=float)[:2]
                    - np.asarray(mid, dtype=float)[:2]
                ))
        except Exception:
            mid_dist = 99.0

        # Sticky phase advances only forward.
        if phase == "approach":
            if (
                geo_near_via
                or same_lane(ego_lane, via)
                or (
                    approach_rem is not None
                    and approach_rem <= UTURN_ZONE_CENTER_REMAINING_M
                )
            ):
                phase = "center"
        if phase == "center":
            # Spin only at the U-turn location (near via), after mid-road.
            at_uturn = geo_near_via or (
                approach_rem is not None
                and approach_rem <= UTURN_ZONE_SPIN_REMAINING_M
            )
            centered = mid_dist <= (UTURN_ZONE_MIDROAD_TOL_M + 0.35)
            if at_uturn and centered:
                phase = "spin"
            elif at_uturn and (
                approach_rem is not None
                and approach_rem <= max(2.0, UTURN_ZONE_SPIN_REMAINING_M - 2.0)
            ):
                # Very close to via — spin even if mid-road not perfect.
                phase = "spin"
        if getattr(self, "_uturn_spinning", False):
            phase = "spin"
        self._uturn_phase = phase

        # False reverse localization mid-approach: glue nav, keep plant2 steer.
        if (
            phase == "approach"
            and rev is not None
            and ego_lane is not None
            and on_same_road(ego_lane, rev)
            and src is not None
        ):
            self._force_nav_onto_lane(src)

        if phase == "spin":
            return self._uturn_phase_spin_steer(ego, via, rev, steering)
        if phase == "center":
            return self._uturn_phase_center_steer(ego, src, via, rev, steering)

        # Approach: plant2 steering + optional speed soft-cap only.
        return self._uturn_phase_approach_steer(ego, src, via, steering)

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
        limit = float(sign.speed_limit)
        # In-zone check via the SIGN's own is_vehicle_in_zone (multi-edge aware,
        # exactly what the verifier uses). The previous lane-local check
        # (zone_start<=veh_long<=zone_end on sign.lane + on_same_road) lost the
        # limit when the ego crossed into another edge of a multi-edge zone
        # (5.21/5.31), so the ego sped up mid-zone. Asking the sign keeps the cap
        # applied across the WHOLE zone, matching where violations are measured.
        if sign.is_vehicle_in_zone(self.control_object):
            self._cap_speed(limit)
            return
        # Approach phase (before entering the zone): brake early within lookahead.
        if not on_same_road(self.control_object.lane, sign.lane):
            if not self._is_sign_on_route(sign):
                return
        veh_long = self._veh_long(sign.lane)
        if veh_long < sign.zone_start:
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

    # Distance (m) before the detour zone at which the preemptive lane change
    # starts. The zone itself starts violating immediately, so the manoeuvre
    # must begin before zone_start for a zero-violation run.
    PREEMPT_DETOUR_M = 10.0
    # Longitudinal clearance past the cone cluster centre before merging back
    # into the original lane (cone half-span 2.25 m + ego half-length + margin).
    DETOUR_RETURN_CLEARANCE_M = 8.0

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
                       and sign.zone_start - veh_long < self.PREEMPT_DETOUR_M)
        if not (in_zone or approaching):
            # Queue at the cones: NPCs on the obstacle lane pile up behind the
            # cluster. Merge BEFORE reaching the queue tail instead of joining
            # it and crawling behind a stopped leader.
            if not (veh_long < sign.zone_start
                    and self._detour_queue_ahead(sign, veh_long)):
                return
        violation_long = getattr(
            sign, "violation_long",
            sign.obstacle_long + getattr(sign, "OBSTACLE_OFFSET", 2.0),
        )
        if in_zone and violation_long - veh_long <= 0:
            return
        self._cap_speed(max(SLOW_APPROACH_MIN_KMH,
                            self.control_object.speed_km_h * SLOW_APPROACH_FACTOR))
        target = self._detour_target_lane(sign)
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

    # How far ahead (m) to look for a stopped queue on the obstacle lane.
    DETOUR_QUEUE_LOOKAHEAD_M = 35.0

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
        # Explicit stop line (StopSign) or default: stop 5 m before the lane end /
        # intersection entry, matching roundabout / right-hand yield behaviour.
        stop_long = getattr(sign, "stop_line_position", None)
        if stop_long is None:
            stop_long = float(sign.lane.length) - YIELD_STOP_BEFORE_END_M
        stop_long = max(
            float(sign.zone_start),
            min(float(stop_long), float(sign.lane.length) - 0.5),
        )

        if veh_long < sign.zone_start:
            if 0 < (sign.zone_start - veh_long) < approach:
                has_traffic, _ = sign._check_main_road_traffic(self.control_object)
                if has_traffic:
                    self._cap_speed(0.001)
            return
        if not (sign.zone_start <= veh_long <= sign.zone_end):
            return
        has_traffic, _ = sign._check_main_road_traffic(self.control_object)
        if not has_traffic:
            return
        # Brake toward / hold at the stop line; do not creep into the junction.
        if veh_long >= stop_long - 0.35:
            self._cap_speed(0.001)
        elif 0 < (stop_long - veh_long) < approach:
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
            # One-way entry (5.7.x): the crossing road is one-way. Unlike a
            # no-turn sign — where the dual-path detour may legally loop back and
            # take the once-forbidden turn on a second pass — here the wrong-way
            # carriageway must NEVER be driven (no oncoming lane exists).
            # Prefer the compliant route already installed at episode start
            # (one_way_signs.run_benchmark._install_one_way_compliant_nav_route).
            #
            # When nav is already clean: do NOT arm direction-exit creep/steer
            # and skip hard intersection-priority braking this step. Arming from
            # spawn (often ≤20 m to junction) + continuous cross-traffic yield
            # (_cap_speed 0.001) stalls CaRL/PlanT2 (~12 m / 1500 steps) while
            # the same compliant path with APPLY_RULE_OVERLAY=False succeeds.
            # Intervene only if the route still touches the wrong-way carriageway.
            forbidden_edges = getattr(sign, "one_way_forbidden_edges", None)
            if forbidden_edges:
                wrong_lanes = self._lanes_on_edges(forbidden_edges)
                if wrong_lanes:
                    ckpts = list(getattr(nav, "checkpoints", None) or [])
                    dirty = any(ck in wrong_lanes for ck in ckpts)
                    if dirty:
                        self._reroute_sumo_avoiding_lanes(wrong_lanes)
                        self._arm_direction_exit_from_sign(sign)
                    else:
                        self._one_way_nav_clean = True
                    return
            # No-turn / missing forbidden_edges: replan+arm only when the first
            # hop from the signed approach is still a blocked exit.
            source_id = getattr(sign.lane, "index", None)
            if self._sumo_route_uses_blocked_source_exit(nav, source_id, blocked):
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
        """Handle DirectionSign, PGDirectionSign, LaneAllowedDirectionSign, LaneDirectionsSign.

        On SUMO EdgeRoadNetwork: if the planned route leaves the signed approach
        via a forbidden turn, BFS-replan to the same destination while only
        blocking those first-hop exits (downstream rejoins remain allowed).

        For 5.15.1 (``LaneDirectionsSign``) with ``target_lane_num``: first
        peer-lane-change onto the lane that can reach the destination, then
        replan from that lane.

        On PG NodeRoadNetwork: keep the existing lane-change pre-positioning.
        """
        if not on_same_road(self.control_object.lane, sign.lane):
            return

        # 5.15.1: force peer lane-change onto the crop-time target lane.
        # Always starts on the WRONG lane — the whole point of the task.
        if isinstance(sign, LaneDirectionsSign):
            target_ln = getattr(sign, "target_lane_num", None)
            if target_ln is not None:
                # After the one post-LC compliant install, never rewrite nav
                # again. NN policies (CaRL/Plant2) often oscillate across peers
                # mid-merge; a second hold/replan looks like the route "jumps".
                if getattr(self, "_lane_dirs_nav_locked", False):
                    cur = self._cur_lane_num()
                    if cur is not None and int(cur) != int(target_ln):
                        # Soft recenter only — keep the locked checkpoints.
                        self._begin_lane_change_by_sumo_num(int(target_ln))
                        try:
                            self._cap_speed(12.0)
                        except Exception:
                            pass
                    else:
                        self._soft_cap_into_next_checkpoint_via()
                    return

                cur = self._cur_lane_num()
                if cur is not None and int(cur) != int(target_ln):
                    # While lane-changing, block the CURRENT lane's illegal
                    # first-hops (injected connectors). Hold nav once on this
                    # lane — re-applying every step resets checkpoint indices
                    # and fights CaRL.
                    blocked = self._lane_directions_blocked_exits(
                        sign, self.control_object.lane
                    )
                    for lid in blocked:
                        self._blocked_lanes.add(lid)
                    if (
                        getattr(self, "APPLY_LANE_DIRS_NAV_HOLD", True)
                        and not getattr(self, "_lane_dirs_hold_applied", False)
                    ):
                        if self._hold_on_lane_until_lc(
                            self.control_object.lane, blocked
                        ):
                            self._lane_dirs_hold_applied = True
                    # Soft peer LC via steering only (no body teleport).
                    self._begin_lane_change_by_sumo_num(int(target_ln))
                    # Slow enough to finish a 1-lane merge in ~20–40 m without
                    # curb overshoot; CRE default cruise (~36) OORs mid-LC.
                    try:
                        self._cap_speed(12.0)
                    except Exception:
                        pass
                    return
                # On target lane — install legal dest route once, then lock.
                if not self._install_lane_dirs_compliant_route(sign):
                    blocked = self._lane_directions_blocked_exits(
                        sign, self.control_object.lane
                    )
                    nav = getattr(self.control_object, "navigation", None)
                    if (
                        blocked
                        and nav is not None
                        and not getattr(self, "_lane_dirs_nav_locked", False)
                        and self._sumo_route_uses_blocked_source_exit(
                            nav,
                            getattr(self.control_object.lane, "index", None),
                            blocked,
                        )
                    ):
                        if self._reroute_sumo_from_current_lane():
                            self._lane_dirs_nav_locked = True
                self._soft_cap_into_next_checkpoint_via()
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
        if sign_idx is None:
            return

        # --- Edge-based (SUMO) path: string lane indices ---
        if isinstance(sign_idx, str):
            # Skip peer lanes on the same road edge; find first checkpoint
            # on a different edge (the actual turn target).
            next_target = None
            sign_edge = sign_idx.rsplit("_", 1)[0] if ":" not in sign_idx else None
            for i, cp in enumerate(checkpoints):
                if cp == sign_idx:
                    for j in range(i + 1, len(checkpoints)):
                        cj = checkpoints[j]
                        if isinstance(cj, str) and ":" not in cj:
                            cj_edge = cj.rsplit("_", 1)[0]
                            if cj_edge != sign_edge:
                                next_target = cj
                                break
                    break
            if next_target is None:
                return
            if next_target in allowed:
                return
            veh_long = self._veh_long(sign.lane)
            dist_to_end = sign.lane.length - veh_long
            if dist_to_end > LANE_CHANGE_LOOKAHEAD:
                return
            rn = self.engine.current_map.road_network
            peer_lanes = rn.get_peer_lanes_from_index(sign_idx)
            ref = self._get_ref_lanes()
            for peer_lane in peer_lanes:
                peer_idx = getattr(peer_lane, "index", None)
                if peer_idx is None or peer_idx == sign_idx:
                    continue
                if peer_lane not in ref:
                    continue
                peer_info = rn.graph.get(peer_idx)
                if peer_info is None:
                    continue
                for turn in (getattr(peer_info, "turns", None) or []):
                    if turn.get("to_lane") == next_target:
                        if self._lc_target_lane is None:
                            self._lc_target_lane = peer_lane
                            self._get_heading_pid().reset()
                            self._get_lateral_pid().reset()
                            self._cap_speed(max(SLOW_APPROACH_MIN_KMH,
                                                self.control_object.speed_km_h * SLOW_APPROACH_FACTOR))
                        return
            return

        # --- Tuple-based (PG) path ---
        if len(sign_idx) < 2:
            return
        target_road = None
        for i in range(len(checkpoints) - 1):
            if sign_idx[0] == checkpoints[i] and sign_idx[1] == checkpoints[i + 1]:
                if i + 2 < len(checkpoints):
                    target_road = (checkpoints[i + 1], checkpoints[i + 2])
                break
        if target_road is None:
            return
        allowed_roads = set()
        for idx in allowed:
            if idx is not None and len(idx) >= 2:
                allowed_roads.add((idx[0], idx[1]))
        if target_road in allowed_roads:
            return
        cur = self._cur_lane_num()
        if cur is None:
            return
        ref = self._get_ref_lanes()
        if not ref:
            return
        veh_long = self._veh_long(sign.lane)
        dist_to_end = sign.lane.length - veh_long
        if dist_to_end > LANE_CHANGE_LOOKAHEAD:
            return
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
        if (self._direction_exit_hold_steps > 0
                and getattr(self, "_direction_exit_creep", False)):
            self._cap_speed(8.0)
        self._no_overtaking_active = False
        # Sticky within a step: any 3.18.1/3.18.2 sign in this scene.
        # Cleared each `_process_signs` call; re-set when those signs are seen.
        self._no_turn_318_context = False
        self._one_way_nav_clean = False

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
                elif isinstance(sign, (NoRightTurnSign, NoLeftTurnSign)):
                    # 3.18.1 / 3.18.2 only — enables mid-route U-turn assist.
                    self._no_turn_318_context = True
                    self._handle_no_turn(sign)
                elif isinstance(sign, NoUTurnSign):
                    # 3.19: same turn blocking, but no mid-route U-turn assist.
                    self._handle_no_turn(sign)
                elif isinstance(sign, OneWayEntrySign):
                    self._handle_no_turn(sign)
                elif isinstance(sign, NoOvertakingSign):
                    self._handle_no_overtaking(sign)
                elif isinstance(sign, (DirectionSign, PGDirectionSign, LaneAllowedDirectionSign, LaneDirectionsSign)):
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
        # Pre-installed one-way compliant nav: skip hard priority yield so
        # continuous cross-traffic cannot freeze NN policies at the approach.
        if not getattr(self, "_one_way_nav_clean", False):
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
            # Aim slightly ABOVE the minimum so a policy whose own desired speed
            # is below the min doesn't keep dipping under min - tolerance. NN
            # policies (carl/plant2) have no internal target to raise, so this
            # firm throttle floor is their only lever to reach/hold the minimum.
            floor_target = self._speed_floor + FLOOR_OVERSHOOT_KMH
            if speed_kmh < floor_target:
                deficit = floor_target - speed_kmh
                accel = min(FLOOR_PROP_GAIN * deficit + FLOOR_BIAS, 1.0)
                throttle = max(throttle, accel)

        return throttle
