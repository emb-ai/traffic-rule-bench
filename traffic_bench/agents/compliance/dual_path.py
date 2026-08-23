import logging
from typing import Optional

import numpy as np
from metadrive.utils.math import wrap_to_pi

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

from traffic_bench.signs.dual_path.direction import LaneAllowedDirectionSign
from traffic_bench.signs.dual_path.no_turn import NoLeftTurnSign, NoRightTurnSign
from traffic_bench.signs.dual_path.pg_direction import PGDirectionSign
from traffic_bench.signs.extra.direction_legacy import DirectionSign
from traffic_bench.signs.extra.lane_directions import LaneDirectionsSign

logger = logging.getLogger(__name__)


class DualPathCompliance:
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
            if getattr(self, "_lane_dirs_active", False):
                return False
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
            hold XY near the mid-road point and rotate heading kinematically,
            then release to plant2.
            """
            self._uturn_spinning = True
            # Near-stop: no forward crawl that would arc off the mid-road.
            self._cap_speed(0.6)
            src = self._uturn_source_lane
            if self._uturn_spin_dir is None:
                self._uturn_spin_dir = self._uturn_spin_dir_from_geometry(via, src, rev)

            mid = self._uturn_midroad_target(src, rev, via, at_via=False)
            try:
                # 5.15.1: never move the body. U-turn spin-hold is 3.18-only.
                if not getattr(self, "_lane_dirs_active", False):
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
            if getattr(self, "_lane_dirs_active", False):
                return float(np.clip(steering, -1.0, 1.0))
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
                # Prefer the compliant route already installed at episode start.
                # When nav is already clean: skip hard intersection-priority braking
                # this step. Yield (_cap_speed 0.001) stalls CaRL/PlanT2 at spawn.
                # Replan only if the route still touches the wrong-way carriageway.
                forbidden_edges = getattr(sign, "one_way_forbidden_edges", None)
                if forbidden_edges:
                    wrong_lanes = self._lanes_on_edges(forbidden_edges)
                    if wrong_lanes:
                        ckpts = list(getattr(nav, "checkpoints", None) or [])
                        dirty = any(ck in wrong_lanes for ck in ckpts)
                        if dirty:
                            self._reroute_sumo_avoiding_lanes(wrong_lanes)
                        else:
                            self._one_way_nav_clean = True
                        return
                # No-turn / missing forbidden_edges: replan only when the first
                # hop from the signed approach is still a blocked exit.
                source_id = getattr(sign.lane, "index", None)
                if self._sumo_route_uses_blocked_source_exit(nav, source_id, blocked):
                    self._reroute_sumo_for_direction_sign(sign)
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

        def _handle_direction_compliance(self, sign):
            """Handle DirectionSign, PGDirectionSign, LaneAllowedDirectionSign, LaneDirectionsSign.

            On SUMO EdgeRoadNetwork: if the planned route leaves the signed approach
            via a forbidden turn, BFS-replan to the same destination while only
            blocking those first-hop exits (downstream rejoins remain allowed).

            For 5.15.1 (``LaneDirectionsSign``) with ``target_lane_num``: first
            peer-lane-change onto the lane that can reach the destination, then
            replan from that lane. Steering-only — never a body snap onto the
            via (the old 4.1 direction-exit teleport must not run here).

            On PG NodeRoadNetwork: keep the existing lane-change pre-positioning.
            """
            if isinstance(sign, LaneDirectionsSign):
                self._lane_dirs_active = True
            if not on_same_road(self.control_object.lane, sign.lane):
                return

            # 5.15.1: force peer lane-change onto the crop-time target lane.
            # Always starts on the WRONG lane — the whole point of the task.
            # Always return: LaneDirectionsSign subclasses DirectionSign, so a
            # missing target_lane_num must not fall through into 4.1/PG via-aim.
            if isinstance(sign, LaneDirectionsSign):
                target_ln = getattr(sign, "target_lane_num", None)
                if target_ln is None:
                    return
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
                if self._sumo_route_uses_blocked_source_exit(nav, source_id, blocked):
                    self._reroute_sumo_for_direction_sign(sign)
                else:
                    self._one_way_nav_clean = True
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
