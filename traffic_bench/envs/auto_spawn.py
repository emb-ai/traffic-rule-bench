"""Auto-spawn / trap / pick-lane helpers for tools/run_simulation.

Official eval sets skip_auto_signs=True and places plates via eval/signs.
"""
from __future__ import annotations

import logging
from typing import Optional

from traffic_bench.eval.engine.map.lane_keys import lane_edge_id


class AutoSpawnMixin:
    def _normalize_turn_direction(raw_dir: str) -> str:
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

    @staticmethod
    def _edge_id_from_lane_key(lane_key: str) -> Optional[str]:
        if not isinstance(lane_key, str) or not lane_key.startswith("lane_"):
            return None
        return lane_edge_id(lane_key)

    @staticmethod
    def _reverse_edge_id(edge_id: Optional[str]) -> Optional[str]:
        if not edge_id:
            return None
        e = str(edge_id)
        return e[1:] if e.startswith("-") else f"-{e}"

    @staticmethod
    def _wrap_pi(x: float) -> float:
        import math

        return float((x + math.pi) % (2 * math.pi) - math.pi)

    @staticmethod
    def _lane_heading_near_end(lane) -> Optional[float]:
        try:
            s = max(0.0, float(lane.length) - 1.0)
            return float(lane.heading_theta_at(s))
        except Exception:
            return None

    @staticmethod
    def _lane_heading_near_start(lane) -> Optional[float]:
        try:
            s = min(1.0, max(0.0, float(lane.length) * 0.1))
            return float(lane.heading_theta_at(s))
        except Exception:
            return None

    @staticmethod
    def _normalize_junction_id(junction_id) -> Optional[str]:
        if not junction_id:
            return None
        jid = str(junction_id)
        return jid if jid.startswith("junction_") else f"junction_{jid}"

    def _has_reverse_inflow_for_turn(self, road_network, from_lane_key: str, to_lane_key: str) -> bool:
        """Return True if target corridor seems to have opposite inflow into junction.

        Heuristic: for turn `from_lane -> to_lane`, check lanes on `to_lane` edge and
        see if any of them has turns back to the source edge. This suggests two-way
        movement at this junction and weakens one-way-entry assumption.
        """
        src_edge = self._edge_id_from_lane_key(from_lane_key)
        dst_edge = self._edge_id_from_lane_key(to_lane_key)
        if src_edge is None or dst_edge is None:
            return False

        for other_lane_key, other_info in road_network.graph.items():
            if self._edge_id_from_lane_key(other_lane_key) != dst_edge:
                continue
            turns = getattr(other_info, "turns", None) or []
            for t in turns:
                back_to = t.get("to_lane")
                if self._edge_id_from_lane_key(back_to) == src_edge:
                    return True
        return False

    def _has_reverse_inflow_via_junction(self, road_network, from_lane_key: str, to_lane_key: str) -> bool:
        """Check reverse inflow at the same junction using lane-level junction metadata."""
        try:
            from_lane = road_network.get_lane(from_lane_key)
            to_lane = road_network.get_lane(to_lane_key)
        except Exception:
            # Fallback to edge-based check when lane lookup is unavailable.
            return self._has_reverse_inflow_for_turn(road_network, from_lane_key, to_lane_key)

        if from_lane is None or to_lane is None:
            return self._has_reverse_inflow_for_turn(road_network, from_lane_key, to_lane_key)

        # Fast SUMO edge-id check: opposite direction edge is usually represented
        # as the same edge id with '-' prefix toggled.
        dst_edge = self._edge_id_from_lane_key(to_lane_key)
        opp_edge = self._reverse_edge_id(dst_edge)
        if opp_edge is not None:
            junction_id = self._normalize_junction_id(getattr(from_lane, "incoming_junction_id", None))
            for lane_key, info in road_network.graph.items():
                if self._edge_id_from_lane_key(lane_key) != opp_edge:
                    continue
                lane_obj = getattr(info, "lane", None)
                if lane_obj is None:
                    continue
                lane_jid = self._normalize_junction_id(getattr(lane_obj, "incoming_junction_id", None))
                if junction_id is not None and lane_jid != junction_id:
                    continue
                return True


        
        junction_id = self._normalize_junction_id(getattr(from_lane, "incoming_junction_id", None))
        incoming_lanes = set(getattr(from_lane, "incoming_junction_lanes", None) or [])
        if not incoming_lanes and junction_id is not None:
            # Build incoming set by scanning graph for same incoming junction.
            for lane_key, info in road_network.graph.items():
                lane = getattr(info, "lane", None)
                if lane is None:
                    continue
                lane_jid = self._normalize_junction_id(getattr(lane, "incoming_junction_id", None))
                if lane_jid == junction_id:
                    incoming_lanes.add(lane_key)

        if not incoming_lanes:
            return self._has_reverse_inflow_for_turn(road_network, from_lane_key, to_lane_key)

        exit_heading = self._lane_heading_near_start(to_lane)
        if exit_heading is None:
            return self._has_reverse_inflow_for_turn(road_network, from_lane_key, to_lane_key)

        import math

        opposite = self._wrap_pi(exit_heading + math.pi)
        threshold = math.radians(35.0)

        for in_lane_key in incoming_lanes:
            if in_lane_key == from_lane_key:
                continue
            try:
                in_lane = road_network.get_lane(in_lane_key)
            except Exception:
                continue
            if in_lane is None:
                continue
            h = self._lane_heading_near_end(in_lane)
            if h is None:
                continue
            if abs(self._wrap_pi(h - opposite)) < threshold:
                return True
        return False

    def _pick_one_way_entry_lane(self, road_network, preferred_road_id: Optional[str] = None):
        """Pick a lane for 5.7.1/5.7.2 by turn topology at intersections.

        Heuristic:
        - 5.7.1: lane has right turn and has no left turn
        - 5.7.2: lane has left turn and has no right turn
        Prefer lanes that match meta road_id when available.
        """
        if self.sign_type not in ("5.7.1", "5.7.2"):
            return None

        need = "r" if self.sign_type == "5.7.1" else "l"
        forbid = "l" if need == "r" else "r"
        target = str(preferred_road_id) if preferred_road_id is not None else None

        best = None
        best_key = None
        best_score = -10**9
        debug_rows = []
        candidate_lane_keys = []

        for lane_key, info in road_network.graph.items():
            
            if not isinstance(lane_key, str) or not lane_key.startswith("lane_"):
                continue
            if ":" in lane_key:
                continue

            turns = getattr(info, "turns", None) or []
            if not turns:
                continue

            dirs = set()
            need_turn_targets = []
            for t in turns:
                d = self._normalize_turn_direction(t.get("direction"))
                dirs.add(d)
                if d == need and t.get("to_lane"):
                    need_turn_targets.append(t.get("to_lane"))

            if need not in dirs or forbid in dirs:
                continue

            # Requested constraint: no reverse inflow from target corridor
            # back into this junction for the chosen turn direction.
            if need_turn_targets:
                if all(self._has_reverse_inflow_via_junction(road_network, lane_key, t) for t in need_turn_targets):
                    continue


            # Straight/U-turn availability is allowed and does not penalize candidate.
            # Prefer longer approach lane before junction.
            lane = info.lane

            if float(getattr(lane, "length", 0.0)) > best_score:
                best_score = float(getattr(lane, "length", 0.0))
                best = lane
                best_key = lane_key

            debug_rows.append((lane_key, float(getattr(lane, "length", 0.0)), sorted(dirs)))
            candidate_lane_keys.append(lane_key)

        self._one_way_candidate_lane_keys = candidate_lane_keys

        if self.config.get("debug_one_way_sign_selection", False):
            self._log_one_way_selection_debug(
                preferred_road_id=target,
                best_key=best_key,
                best_score=best_score,
                debug_rows=debug_rows,
            )

        return best

    @staticmethod
    def _allowed_dirs_for_lane_sign(sign_type: str) -> Optional[set[str]]:
        mapping = {
            "4.1.1": {"s"},
            "4.1.2": {"r"},
            "4.1.3": {"l"},
            "4.1.4": {"s", "r"},
            "4.1.5": {"s", "l"},
            "4.1.6": {"l", "r"},
        }
        return mapping.get(sign_type)

    @staticmethod
    def _lane_priority_flag(lane) -> Optional[bool]:
        turns = getattr(lane, "turns", None) or []
        unmanaged = [
            t for t in turns
            if t.get("junction_type") in ("priority", "right_before_left") and isinstance(t.get("priority"), dict)
        ]
        if not unmanaged:
            return None
        has_any_yield = any(not bool(t["priority"].get("has_priority", False)) for t in unmanaged)
        return not has_any_yield

    def _pick_lane_direction_candidates(self, road_network, preferred_road_id: Optional[str] = None):
        allowed = self._allowed_dirs_for_lane_sign(self.sign_type)
        if allowed is None:
            return None, []

        target = str(preferred_road_id) if preferred_road_id is not None else None
        best_lane = None
        best_key = None
        best_score = -10**9
        candidate_lane_keys = []
        debug_rows = []

        for lane_key, info in road_network.graph.items():
            if not isinstance(lane_key, str) or not lane_key.startswith("lane_"):
                continue
            if ":" in lane_key:
                continue
            turns = getattr(info, "turns", None) or []
            if not turns:
                continue

            dirs = set()
            for t in turns:
                d = self._normalize_turn_direction(t.get("direction"))
                if d in {"l", "r", "s"}:
                    dirs.add(d)
            if dirs != allowed:
                continue

            score = 0.0
            lane = info.lane
            score += min(float(getattr(lane, "length", 0.0)), 80.0)
            if target:
                parts = lane_key.split("_")
                edge_id = parts[1] if len(parts) >= 3 else None
                if edge_id == target:
                    score += 1000.0

            candidate_lane_keys.append(lane_key)
            debug_rows.append((lane_key, score, sorted(dirs)))
            if score > best_score:
                best_score = score
                best_lane = lane
                best_key = lane_key

        if self.config.get("debug_one_way_sign_selection", False):
            top_rows = sorted(debug_rows, key=lambda x: x[1], reverse=True)[:8]
            print(
                f"[LaneSignDebug] sign={self.sign_type} map={self.custom_map_name} "
                f"preferred_road_id={target} chosen={best_key} score={best_score}"
            )
            for lane_key, score, dirs in top_rows:
                print(f"[LaneSignDebug] candidate lane={lane_key} score={score:.1f} dirs={dirs}")

        return best_lane, candidate_lane_keys

    @staticmethod
    def _prohibited_dir_for_no_turn_sign(sign_type: str) -> Optional[str]:
        mapping = {
            "3.18.1": "r",
            "3.18.2": "l",
            "3.19": "t",
        }
        return mapping.get(sign_type)

    @staticmethod
    def _is_junction_like_lane_id(lane_id) -> bool:
        if lane_id is None:
            return False
        s = str(lane_id)
        return s.startswith("junction") or s.startswith("lane_:") or s.startswith(":")

    def _pick_no_turn_candidates(self, road_network, preferred_road_id: Optional[str] = None):
        prohibited = self._prohibited_dir_for_no_turn_sign(self.sign_type)
        if prohibited is None:
            return None, []

        target = str(preferred_road_id) if preferred_road_id is not None else None
        best_lane = None
        best_key = None
        best_score = -10**9
        candidate_lane_keys = []
        debug_rows = []

        for lane_key, info in road_network.graph.items():
            if not isinstance(lane_key, str) or not lane_key.startswith("lane_"):
                continue
            if ":" in lane_key:
                continue
            turns = getattr(info, "turns", None) or []
            if not turns:
                continue

            dirs = set()
            for t in turns:
                d = self._normalize_turn_direction(t.get("direction"))
                if d in {"l", "r", "s", "t"}:
                    dirs.add(d)

            # Candidate lane must contain prohibited maneuver and at least one
            # alternative maneuver that does not violate the sign.
            if prohibited not in dirs:
                continue
            if len(dirs - {prohibited}) == 0:
                continue

            score = 0.0
            lane = info.lane
            score += min(float(getattr(lane, "length", 0.0)), 80.0)
            if target:
                parts = lane_key.split("_")
                edge_id = parts[1] if len(parts) >= 3 else None
                if edge_id == target:
                    score += 1000.0

            candidate_lane_keys.append(lane_key)
            debug_rows.append((lane_key, score, sorted(dirs)))
            if score > best_score:
                best_score = score
                best_lane = lane
                best_key = lane_key

        if self.config.get("debug_one_way_sign_selection", False):
            top_rows = sorted(debug_rows, key=lambda x: x[1], reverse=True)[:8]
            print(
                f"[NoTurnSignDebug] sign={self.sign_type} map={self.custom_map_name} "
                f"preferred_road_id={target} chosen={best_key} score={best_score}"
            )
            for lane_key, score, dirs in top_rows:
                print(f"[NoTurnSignDebug] candidate lane={lane_key} score={score:.1f} dirs={dirs}")

        return best_lane, candidate_lane_keys

    def _setup_direction_sign_trap(self, road_network, current_lane):
        """For sign 5.15.2, find a lane_A (fewer turn options) and a parallel
        lane_B (more turn options). Returns (trap_lane_id, trap_lane_obj,
        violation_target_id, adjacent_lane_id) or (None, None, None, None)."""
        def _dirs_and_targets(lane_key):
            info = road_network.graph.get(lane_key)
            if info is None:
                return set(), set()
            dirs = set()
            targets = set()
            for t in (getattr(info, "turns", None) or []):
                d = t.get("direction")
                to = t.get("to_lane")
                if d:
                    dirs.add(d)
                if to:
                    targets.add(to)
            return dirs, targets

        def _peers_by_edge(candidate_key):
            edge = candidate_key.rsplit("_", 1)[0]
            return [
                k for k in road_network.graph
                if k != candidate_key and k.startswith(edge + "_") and ":" not in k
            ]

        # Build edge groups first
        edge_groups = {}
        for key in road_network.graph:
            if not isinstance(key, str) or not key.startswith("lane_") or ":" in key:
                continue
            e = key.rsplit("_", 1)[0]
            edge_groups.setdefault(e, []).append(key)

        for edge, lane_keys in edge_groups.items():
            if len(lane_keys) < 2:
                continue
            lane_data = {}
            for lk in lane_keys:
                dirs, targets = _dirs_and_targets(lk)
                if dirs and targets:
                    lane_data[lk] = (dirs, targets)
            keys = list(lane_data.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    kA, kDirsA, kTargetsA = keys[i], lane_data[keys[i]][0], lane_data[keys[i]][1]
                    kB, kDirsB, kTargetsB = keys[j], lane_data[keys[j]][0], lane_data[keys[j]][1]
                    extra_B = kDirsB - kDirsA
                    extra_A = kDirsA - kDirsB
                    if len(extra_B) > 0:
                        # kB (peer) has extra direction(s) — kA is trap
                        violation_target = self._find_extra_target(kB, extra_B, road_network)
                        if violation_target is not None:
                            return kA, road_network.graph[kA].lane, violation_target, kB
                    if len(extra_A) > 0:
                        # kA (peer) has extra direction(s) — kB is trap
                        violation_target = self._find_extra_target(kA, extra_A, road_network)
                        if violation_target is not None:
                            return kB, road_network.graph[kB].lane, violation_target, kA

        return None, None, None, None

    def _find_extra_target(self, lane_key, extra_dirs, road_network):
        """Find a to_lane from lane_key's turns that matches any of
        extra_dirs and is NOT reachable from peers of lane_key."""
        info = road_network.graph.get(lane_key)
        if info is None:
            return None
        edge = lane_key.rsplit("_", 1)[0]
        peer_keys = [
            k for k in road_network.graph
            if k != lane_key and k.startswith(edge + "_") and ":" not in k
        ]
        for t in (getattr(info, "turns", None) or []):
            d = t.get("direction")
            to = t.get("to_lane")
            if d in extra_dirs and to:
                # Verify this target is NOT reachable from any peer
                is_unique = True
                for pk in peer_keys:
                    pi = road_network.graph.get(pk)
                    if pi is None:
                        continue
                    for pt in (getattr(pi, "turns", None) or []):
                        if pt.get("to_lane") == to:
                            is_unique = False
                            break
                    if not is_unique:
                        break
                if is_unique:
                    return to
        return None

    @staticmethod
    def _counterpart_dir_for_no_turn_anchor(sign_type: str) -> Optional[str]:
        mapping = {
            "3.18.1": "l",  # no-right: anchor lane should be target of a left turn
            "3.18.2": "r",  # no-left: anchor lane should be target of a right turn
            "3.19": "t",    # no-u-turn: anchor lane should be target of a u-turn
        }
        return mapping.get(sign_type)

    def _pick_no_turn_anchor_lane(
        self,
        road_network,
        no_turn_candidate_lane_keys,
        preferred_road_id: Optional[str] = None,
    ):
        prohibited = self._prohibited_dir_for_no_turn_sign(self.sign_type)
        counterpart = self._counterpart_dir_for_no_turn_anchor(self.sign_type)
        if prohibited is None or counterpart is None:
            return None

        target = str(preferred_road_id) if preferred_road_id is not None else None
        candidate_set = set(no_turn_candidate_lane_keys or [])
        if not candidate_set:
            return None

        best_lane = None
        best_key = None
        best_score = -10**9
        debug_rows = []

        for lane_key, info in road_network.graph.items():
            if not isinstance(lane_key, str) or not lane_key.startswith("lane_"):
                continue
            if ":" in lane_key:
                continue

            turns = getattr(info, "turns", None) or []
            dirs = set()
            for t in turns:
                d = self._normalize_turn_direction(t.get("direction"))
                if d in {"l", "r", "s", "t"}:
                    dirs.add(d)

            # User-requested placement heuristic:
            # anchor lane does NOT have prohibited maneuver.
            if prohibited in dirs:
                continue

            # There must exist an incoming no-turn candidate lane that turns
            # into this anchor lane with counterpart direction.
            has_counterpart_entry = False
            for src_key in candidate_set:
                src_info = road_network.graph.get(src_key)
                if src_info is None:
                    continue
                for t in (getattr(src_info, "turns", None) or []):
                    d = self._normalize_turn_direction(t.get("direction"))
                    if d != counterpart:
                        continue
                    to_lane = t.get("to_lane")
                    via_lane = t.get("via_lane")
                    if to_lane == lane_key or via_lane == lane_key:
                        if self._is_junction_like_lane_id(to_lane) or self._is_junction_like_lane_id(via_lane):
                            continue
                        has_counterpart_entry = True
                        break

                    # SUMO often stores turn links via an internal junction lane.
                    # Accept anchor lane if it is reachable in one hop from
                    # to_lane/via_lane through exit_lanes.
                    for mid_lane in (to_lane, via_lane):
                        if not mid_lane:
                            continue
                        if self._is_junction_like_lane_id(mid_lane):
                            continue
                        try:
                            mid_obj = road_network.get_lane(mid_lane)
                        except Exception:
                            continue
                        if mid_obj is None:
                            continue
                        next_lanes = set(getattr(mid_obj, "exit_lanes", None) or [])
                        next_lanes = {n for n in next_lanes if not self._is_junction_like_lane_id(n)}
                        if lane_key in next_lanes:
                            has_counterpart_entry = True
                            break
                    if has_counterpart_entry:
                        break
                if has_counterpart_entry:
                    break

            if not has_counterpart_entry:
                continue

            lane = info.lane
            score = min(float(getattr(lane, "length", 0.0)), 80.0)
            if target:
                parts = lane_key.split("_")
                edge_id = parts[1] if len(parts) >= 3 else None
                if edge_id == target:
                    score += 1000.0

            debug_rows.append((lane_key, score, sorted(dirs)))
            if score > best_score:
                best_score = score
                best_lane = lane
                best_key = lane_key

        if self.config.get("debug_one_way_sign_selection", False):
            top_rows = sorted(debug_rows, key=lambda x: x[1], reverse=True)[:8]
            print(
                f"[NoTurnAnchorDebug] sign={self.sign_type} map={self.custom_map_name} "
                f"preferred_road_id={target} chosen={best_key} score={best_score}"
            )
            for lane_key, score, dirs in top_rows:
                print(f"[NoTurnAnchorDebug] candidate lane={lane_key} score={score:.1f} dirs={dirs}")

        return best_lane

    def _pick_priority_candidates(self, road_network, preferred_road_id: Optional[str] = None):
        if self.sign_type not in ("2.1", "2.2", "2.4", "2.3.1", "2.3.2", "2.3.3"):
            return None, []
        
        yield_priority = ("2.1", "2.3.1", "2.3.2", "2.3.3")
        want_priority = True if self.sign_type in yield_priority  else False
        target = str(preferred_road_id) if preferred_road_id is not None else None
        best_lane = None
        best_key = None
        best_score = -10**9
        candidate_lane_keys = []

        for lane_key, info in road_network.graph.items():
            if not isinstance(lane_key, str) or not lane_key.startswith("lane_"):
                continue
            if ":" in lane_key:
                continue
            lane = getattr(info, "lane", None)
            if lane is None:
                continue

            pf = self._lane_priority_flag(lane)
            if pf is None or pf is not want_priority:
                continue

            turns = getattr(info, "turns", None) or []
            exit_lanes = set(getattr(info, "exit_lanes", None) or [])
            maneuver_targets = set()
            for t in turns:
                d = self._normalize_turn_direction(t.get("direction"))
                if d not in {"l", "r", "s"}:
                    continue
                to_lane = t.get("to_lane")
                via_lane = t.get("via_lane")
                if to_lane:
                    maneuver_targets.add(to_lane)
                if via_lane:
                    maneuver_targets.add(via_lane)

            # Keep only lanes that have at least one outgoing movement among
            # left/right/straight reflected in exit_lanes.
            if not (exit_lanes and maneuver_targets and len(exit_lanes.intersection(maneuver_targets)) > 0):
                continue

            candidate_lane_keys.append(lane_key)

            score = min(float(getattr(lane, "length", 0.0)), 80.0)
            if target:
                parts = lane_key.split("_")
                edge_id = parts[1] if len(parts) >= 3 else None
                if edge_id == target:
                    score += 1000.0

            if score > best_score:
                best_score = score
                best_lane = lane
                best_key = lane_key

        if self.config.get("debug_one_way_sign_selection", False):
            print(
                f"[PrioritySignDebug] sign={self.sign_type} candidates={len(candidate_lane_keys)} "
                f"preferred_road_id={target} chosen={best_key}"
            )

        return best_lane, candidate_lane_keys

    def _log_one_way_selection_debug(self, preferred_road_id, best_key, best_score, debug_rows):
        top_rows = sorted(debug_rows, key=lambda x: x[1], reverse=True)[:8]
        header = (
            f"[OneWaySignDebug] sign={self.sign_type} map={self.custom_map_name} "
            f"preferred_road_id={preferred_road_id} chosen={best_key} score={best_score}"
        )
        print(header)
        for lane_key, score, dirs in top_rows:
            print(f"[OneWaySignDebug] candidate lane={lane_key} score={score:.1f} dirs={dirs}")

    def _pick_spawn_lane_from_sign_attachment(self, road_network, sign_lane, sign_kwargs):
        lane_ids = []
        preset = sign_kwargs.get("applicable_lane_indices")
        base_idx = getattr(sign_lane, "index", None)

        # For no-turn signs we intentionally place/render signs on anchor-side
        # lanes, while preset applicable lanes may represent source/topology
        # lanes. Spawn should prefer the physical sign lane first so route
        # naturally passes the sign location.
        prefer_base_first = self.sign_type in ("3.18.1", "3.18.2", "3.19")

        if prefer_base_first and base_idx is not None:
            lane_ids.append(base_idx)
        if preset:
            lane_ids.extend([x for x in preset if x is not None])
        if (not prefer_base_first) and base_idx is not None and base_idx not in lane_ids:
            lane_ids.append(base_idx)

        lanes = []
        for lane_id in lane_ids:
            try:
                lane_obj = road_network.get_lane(lane_id)
            except Exception:
                continue
            if lane_obj is not None:
                lanes.append(lane_obj)

        if not lanes:
            return None
        # Deterministic pick to keep rollouts reproducible across resets.
        idx = int(getattr(self, "current_seed", 0) or 0) % len(lanes)
        return lanes[idx]

    def _spawn_ego_on_lane(self, lane):
        if lane is None:
            return
        try:
            long = min(2.0, max(0.2, float(getattr(lane, "length", 1.0)) * 0.05))
            pos = lane.position(long, 0.0)
            heading = lane.heading_theta_at(long)
            self.vehicle.set_position(pos)
            self.vehicle.set_heading_theta(heading)
        except Exception:
            pass

    def _upstream_real_lane(self, lane, road_network, visited,
                            skip_reverse=False, prefer_big_road=False):
        """Nearest upstream NON-internal lane feeding `lane`, crossing at most one
        junction. Returns (lane, gap_len) where gap_len is the internal junction
        lane length traversed (0.0 if directly connected). (None, 0.0) if none.

        SUMO edges connect THROUGH junctions via internal ':' lanes; walking only
        real edges would dead-end at every intersection. This follows the internal
        lane one hop to reach the real upstream edge after the intersection.

        `skip_reverse`: never follow the reverse (U-turn) direction of `lane`'s own
        edge, so an inbound courtyard lane can't walk back onto its own outbound
        carriageway. `prefer_big_road`: among eligible predecessors choose the
        longest lane (a proxy for a main road over a short courtyard stub), so the
        ego ends up approaching from the big road.
        """
        cur_edge = self._lane_key_edge(str(getattr(lane, "index", None)))

        def _eligible(l):
            if l is None or str(getattr(l, "index", None)) in visited:
                return False
            if skip_reverse and cur_edge is not None:
                e = self._lane_key_edge(str(getattr(l, "index", None)))
                if e is not None and _edges_are_reverse(cur_edge, e):
                    return False
            return True

        def _pick(cands):
            if prefer_big_road and len(cands) > 1:
                cands = sorted(cands, key=lambda l: float(getattr(l, "length", 0.0)),
                               reverse=True)
            return cands[0]

        entries = list(getattr(lane, "entry_lanes", None) or [])
        # Direct real predecessors first.
        direct = []
        for e in entries:
            if ":" in str(e):
                continue
            try:
                l = road_network.get_lane(e)
            except Exception:
                l = None
            if _eligible(l):
                direct.append(l)
        if direct:
            return _pick(direct), 0.0
        # Otherwise cross an internal junction lane to its real upstream edge.
        for e in entries:
            if ":" not in str(e):
                continue
            try:
                il = road_network.get_lane(e)
            except Exception:
                il = None
            if il is None:
                continue
            gap = float(getattr(il, "length", 0.0))
            cross = []
            for e2 in (getattr(il, "entry_lanes", None) or []):
                if ":" in str(e2):
                    continue
                try:
                    l2 = road_network.get_lane(e2)
                except Exception:
                    l2 = None
                if _eligible(l2):
                    cross.append(l2)
            if cross:
                return _pick(cross), gap
        return None, 0.0

    def _route_through_sign(self, spawn_lane, sign_lane_index, forward_dest=None):
        """Route ego from spawn_lane so the path passes THROUGH the sign edge.

        Tries the configured downstream destination first (gives post-sign road),
        but only accepts it if the resulting route actually contains the sign
        edge. Otherwise routes directly to the sign edge (always reachable — we
        walked upstream from it). Returns True if the sign edge is on the route.

        For zone-ENTRY signs (5.21) `forward_dest` (a lane deep in the courtyard)
        is tried FIRST so the route is big-road -> sign -> courtyard interior, and
        a degenerate `[spawn, spawn]` route (the sign edge present but the route
        not actually continuing past it) is rejected — the sign must be strictly
        mid-route.
        """
        nav = getattr(self.vehicle, "navigation", None)
        if nav is None:
            return False
        sign_key = str(sign_lane_index)
        zone_entry = self.sign_type in ZONE_ENTRY_SIGN_CODES

        def _route_has_sign():
            ckpts = [str(c) for c in (getattr(nav, "checkpoints", None) or [])]
            return sign_key in ckpts

        def _n_checkpoints():
            return len(getattr(nav, "checkpoints", None) or [])

        dest = (getattr(self.vehicle, "config", {}) or {}).get("destination")
        # Reachable forward destination walked from the sign's own lane in the live
        # graph — used as a fallback so the route CONTINUES past the sign even when
        # the catalog destination isn't reachable from this lane (junction branch).
        try:
            road_network = self.engine.current_map.road_network
        except Exception:
            road_network = None
        fwd_reach = (self._forward_reachable_destination(sign_lane_index, road_network)
                     if road_network is not None else None)
        targets = ([forward_dest, dest, fwd_reach, sign_lane_index] if zone_entry
                   else [dest, fwd_reach, sign_lane_index])
        for target in targets:
            if not target:
                continue
            try:
                nav.set_route(spawn_lane.index, target)
            except Exception:
                continue
            has_sign = _route_has_sign()
            if zone_entry:
                # Sign strictly mid-route (path continues into the courtyard):
                # present AND more than the degenerate 2-checkpoint route.
                if has_sign and _n_checkpoints() > 2:
                    nav.update_localization(self.vehicle)
                    return True
                # Last resort: routing to the sign edge itself, only if it yields
                # a real (>1 edge) route — never the [spawn, spawn] degenerate.
                if str(target) == sign_key and has_sign and _n_checkpoints() > 1:
                    nav.update_localization(self.vehicle)
                    return True
            else:
                if has_sign or str(target) == sign_key:
                    nav.update_localization(self.vehicle)
                    return True
        self._refresh_navigation_after_spawn(spawn_lane)
        return _route_has_sign()

    def _restrict_npcs_to_zone_or_adjacent(self, corridor_lanes, sign_lane_index,
                                           sign_s, lane_lat=2.0):
        """Keep surrounding NPCs only on ADJACENT lanes or INSIDE the sign zone.

        `corridor_lanes` are the lanes ego traverses from its braking spawn up to
        the sign. An NPC is removed iff it sits on one of those lanes BEFORE the
        sign:
          - on an upstream corridor lane (not the sign's lane) → always before
            the sign → remove;
          - on the sign's own lane with longitudinal < sign_s → before sign →
            remove; at/after sign_s it is in the zone → keep.
        NPCs on any other (adjacent / non-corridor) lane are kept regardless of
        position. Returns the number removed.

        To preserve the sampled traffic_density, every NPC removed from the
        corridor is RELOCATED: an equal number is respawned on allowed lanes
        (corridor lanes excluded → adjacent lanes / elsewhere). Spawn/removal/
        relocation counts are recorded in `self._npc_density_stats` so the
        realized density can be reported in the manifest.

        Lane membership is by local coordinates: |lateral| < `lane_lat` (half a
        lane width) so a car one lane over (~3.5 m) is treated as adjacent.
        """
        try:
            traffic_mgr = self.engine.traffic_manager
            if not hasattr(traffic_mgr, "traffic_vehicles"):
                return 0
            n_before = len(traffic_mgr.traffic_vehicles)
            sign_key = str(sign_lane_index)
            # Dedup corridor lanes by index, drop None.
            seen = set()
            lanes = []
            for ln in corridor_lanes:
                if ln is None:
                    continue
                k = str(getattr(ln, "index", None))
                if k in seen:
                    continue
                seen.add(k)
                lanes.append(ln)
            to_remove = []
            for v in list(traffic_mgr.traffic_vehicles):
                pos = np.asarray(v.position, dtype=np.float64)
                drop = False
                for ln in lanes:
                    try:
                        lng, lat = ln.local_coordinates(pos)
                    except Exception:
                        continue
                    if not (0.0 <= lng <= float(getattr(ln, "length", 0.0)) and abs(lat) < lane_lat):
                        continue  # not on this corridor lane
                    # On a corridor lane. Keep only if it's the sign lane AND the
                    # NPC is at/after the sign (already in the enforced zone).
                    if str(getattr(ln, "index", None)) == sign_key and lng >= float(sign_s):
                        drop = False
                        break
                    drop = True
                    break
                if drop:
                    to_remove.append(v)
            for v in to_remove:
                traffic_mgr.clear_objects([v.id])
                if v in getattr(traffic_mgr, "_traffic_vehicles", []):
                    traffic_mgr._traffic_vehicles.remove(v)

            # Relocate: respawn the same count on allowed lanes (corridor lanes
            # excluded) so the realized density matches the sampled profile.
            n_removed = len(to_remove)
            relocated = 0
            if n_removed > 0 and hasattr(traffic_mgr, "_try_respawn"):
                corridor_keys = [str(getattr(ln, "index", None)) for ln in lanes]
                try:
                    relocated = int(traffic_mgr._try_respawn(
                        n_removed, forbidden_keys=corridor_keys) or 0)
                except Exception as exc:
                    logging.warning(f"braking-spawn NPC relocation failed: {exc}")
            n_after = len(traffic_mgr.traffic_vehicles)
            self._npc_density_stats = {
                "npc_before": int(n_before),
                "npc_removed_corridor": int(n_removed),
                "npc_relocated": int(relocated),
                "npc_after": int(n_after),
                "npc_density_preserved": bool(relocated >= n_removed),
                "traffic_density": float(self.config.get("traffic_density", 0.0) or 0.0),
            }
            return n_removed
        except Exception as exc:
            logging.warning(f"braking-spawn NPC restriction failed: {exc}")
            return 0

    def _spawn_ego_before_sign(self, sign_obj, road_network):
        """Place ego `d_required` UPSTREAM of the sign along the road graph at v0.

        Reads the braking spec from config (ego_spawn_v0_ms, ego_brake_d_required,
        ego_v_target_kmh, brake params). Walks up `entry_lanes` when the required
        distance exceeds the sign's offset on its own edge. On a dead-end
        (insufficient runway) it clamps to the furthest reachable point and lowers
        v0 to what fits (still above the limit). Returns an info dict or None.
        """
        try:
            v0 = float(self.config.get("ego_spawn_v0_ms", 0.0) or 0.0)
            d_req = float(self.config.get("ego_brake_d_required", 0.0) or 0.0)
            v_target_mps = float(self.config.get("ego_v_target_kmh", 0.0) or 0.0) / 3.6
            # "accel" (4.6): ego starts BELOW the target and must speed up; "brake"
            # (3.24/5.21/5.31): ego starts above and must slow down.
            accel_mode = str(self.config.get("ego_spawn_mode", "brake")) == "accel"
            sign_lane = sign_obj.lane
            sign_s = float(getattr(sign_obj, "placement_long", 0.0))
        except Exception:
            return None
        if sign_lane is None or v0 <= 0.0:
            return None

        # Zone-entry signs (5.21): keep the upstream walk on the inbound
        # carriageway (no U-turn) and bias toward the big road, so the ego
        # approaches FROM the big road instead of deeper in the courtyard.
        zone_entry = self.sign_type in ZONE_ENTRY_SIGN_CODES

        lane_num = int(self.config.get("spawn_lane_num", 0) or 0)
        spawn_lane = sign_lane
        road_id = str((self.meta or {}).get("road_id") or "")
        if road_id:
            try:
                cand = road_network.get_lane(f"lane_{road_id}_{lane_num}")
                if cand is not None:
                    spawn_lane = cand
            except Exception:
                pass

        # Walk upstream until d_required is covered, crossing junctions (internal
        # ':' lanes) to reach the real upstream edge AFTER each intersection.
        remaining = d_req
        cur = spawn_lane
        avail_here = min(sign_s, float(getattr(cur, "length", sign_s)))
        visited = {str(getattr(cur, "index", None))}
        # Lanes ego traverses from spawn up to the sign — the braking corridor.
        # NPCs on these (before the sign) get removed so ego, starting above the
        # limit, doesn't rear-end a slower car it can't avoid. The sign lane is
        # included but only its pre-sign part counts as corridor (the zone after
        # the sign keeps its traffic).
        corridor_lanes = [cur]
        insufficient = False
        spawn_long = None
        for _ in range(30):
            if remaining <= avail_here:
                spawn_long = avail_here - remaining
                spawn_lane = cur
                remaining = 0.0
                break
            remaining -= avail_here
            # skip_reverse is correct for EVERY braking sign: the ego approaches
            # on the sign's own carriageway, so the upstream walk must never
            # U-turn onto the oncoming (reverse) edge (else the sign ends up "on
            # the other side"). prefer_big_road stays zone-entry-only.
            pred, gap = self._upstream_real_lane(
                cur, road_network, visited,
                skip_reverse=True, prefer_big_road=zone_entry)
            if pred is None:
                insufficient = True
                spawn_lane = cur
                spawn_long = 0.2
                break
            # If the spawn point falls inside the junction, place it just before
            # the junction at the end of the upstream edge.
            if remaining <= gap:
                spawn_lane = pred
                spawn_long = max(0.2, float(getattr(pred, "length", 1.0)) - 0.5)
                corridor_lanes.append(pred)
                remaining = 0.0
                break
            remaining -= gap
            cur = pred
            visited.add(str(getattr(cur, "index", None)))
            corridor_lanes.append(cur)
            avail_here = float(getattr(cur, "length", 0.0))
        if spawn_long is None:
            insufficient = True
            spawn_lane = cur
            spawn_long = 0.2

        # Insufficient runway → lower v0 to what the achieved distance allows
        # (kept above the limit). If even v0 = limit+ε doesn't fit, the scene
        # can't test braking to the limit → mark braking_invalid for filtering.
        d_achieved = max(0.0, d_req - remaining)
        braking_invalid = False
        # Braking only: on insufficient runway lower v0 to what can be braked in
        # d_achieved. For ACCEL (4.6) keep v0 — a short runway just means the ego
        # reaches the minimum later (within the zone), which is still valid.
        if not accel_mode:
            if insufficient and d_achieved > 0.0:
                try:
                    from traffic_bench.eval.engine.traffic.agent_profile_bank import max_v0_for_distance
                    v0_fit = max_v0_for_distance(
                        d_achieved, v_target_mps,
                        float(self.config.get("ego_brake_decel", 2.5)),
                        float(self.config.get("ego_brake_delay", 1.0)),
                        float(self.config.get("ego_brake_margin", 5.0)),
                    )
                    if v0_fit > v_target_mps:
                        v0 = min(v0, v0_fit)
                    else:
                        braking_invalid = True
                except Exception:
                    pass
            elif insufficient:
                braking_invalid = True

        spawn_long = max(0.2, min(float(spawn_long),
                                  float(getattr(spawn_lane, "length", spawn_long)) - 0.1))
        routed = False
        try:
            pos = np.asarray(spawn_lane.position(spawn_long, 0.0), dtype=np.float64)
            heading = float(spawn_lane.heading_theta_at(spawn_long))
            self.vehicle.set_position(pos)
            self.vehicle.set_heading_theta(heading)
            try:
                self.vehicle.spawn_place = pos.copy()
            except Exception:
                pass
            # Restrict surrounding traffic: NPCs may only sit on ADJACENT lanes
            # or already INSIDE the sign's zone (from the sign onward). Any NPC on
            # ego's own braking-corridor lanes BEFORE the sign is removed — ego
            # starts above the limit among slower (limit-capped) NPCs, so such a
            # car is an unavoidable rear-end before ego can brake → spurious crash.
            n_cleared = self._restrict_npcs_to_zone_or_adjacent(
                corridor_lanes, getattr(sign_lane, "index", None), sign_s)
            fwd_dest = (self._forward_courtyard_destination(sign_obj, road_network)
                        if zone_entry else None)
            routed = self._route_through_sign(
                spawn_lane, getattr(sign_lane, "index", None), forward_dest=fwd_dest)
            try:
                self.vehicle.set_velocity([float(v0), 0.0], in_local_frame=True)
            except TypeError:
                self.vehicle.set_velocity([float(v0), 0.0])
        except Exception as exc:
            logging.warning(f"braking spawn failed: {exc}")
            return None

        info = {
            "ego_spawn_edge": str(getattr(spawn_lane, "index", None)),
            "ego_spawn_long": round(float(spawn_long), 3),
            "ego_spawn_v0_ms": round(float(v0), 4),
            "ego_d_required_m": round(float(d_req), 3),
            "ego_d_achieved_m": round(float(d_achieved), 3),
            "insufficient_runway": bool(insufficient),
            "braking_invalid": bool(braking_invalid),
            "routed_through_sign": bool(routed),
            "npc_cleared_corridor": int(n_cleared),
        }
        # Density bookkeeping (spawn/removal/relocation) so the realized traffic
        # density can be audited per scene.
        info.update(getattr(self, "_npc_density_stats", {}) or {})
        self._braking_spawn_info = info
        return info

    @staticmethod
    def _spawn_edge_id(lane_index: str) -> str:
        raw = lane_index[5:] if lane_index.startswith("lane_") else lane_index
        return raw.rsplit("_", 1)[0] if "_" in raw else raw

    def _pick_destination_with_min_hops(
        self,
        start_lane_index,
        road_network,
        min_hops: int,
        max_hops: int,
        *,
        exclude_lane_indices=None,
    ):
        if start_lane_index not in road_network.graph:
            return None
        min_hops = max(1, int(min_hops))
        max_hops = max(min_hops, int(max_hops))
        excluded = set(exclude_lane_indices or [])
        excluded.add(start_lane_index)
        spawn_edge = self._spawn_edge_id(start_lane_index)

        visited = {start_lane_index}
        queue = [(start_lane_index, 0)]
        fallback = None

        def _is_routable_destination(lane_idx):
            if lane_idx is None or str(lane_idx).startswith("lane_:"):
                return False
            if lane_idx in excluded:
                return False
            if self._spawn_edge_id(lane_idx) == spawn_edge:
                return False
            return True

        while queue:
            lane_idx, depth = queue.pop(0)
            if depth >= max_hops:
                continue
            info = road_network.graph.get(lane_idx)
            if info is None:
                continue
            next_lanes = sorted(set(getattr(info, "exit_lanes", None) or []))
            for nxt in next_lanes:
                if nxt not in road_network.graph or nxt in visited:
                    continue
                visited.add(nxt)
                next_depth = depth + 1
                if _is_routable_destination(nxt):
                    fallback = nxt
                if next_depth >= min_hops and _is_routable_destination(nxt):
                    return nxt
                queue.append((nxt, next_depth))

        return fallback
    
    def _pick_secondary_road_candidates(self, road_network, preferred_road_id=None):
        """Return (best_main_lane, list_of_main_lane_keys) for sign type 2.3.1.
        
        Selects main-priority lanes approaching junctions that also have
        at least one secondary-priority lane.
        """
        target = str(preferred_road_id) if preferred_road_id is not None else None
        best_lane = None
        best_score = -1e9
        candidate_keys = []

        for lane_key, info in road_network.graph.items():
            if not isinstance(lane_key, str) or not lane_key.startswith("lane_"):
                continue
            if ":" in lane_key:
                continue
            lane = getattr(info, "lane", None)
            if lane is None:
                continue

            if self._lane_priority_flag(lane) is not True:
                continue

            junction_id = self._normalize_junction_id(
                getattr(lane, "incoming_junction_id", None)
            )
            if junction_id is None:
                continue

            has_secondary = False
            for other_key, other_info in road_network.graph.items():
                other_lane = getattr(other_info, "lane", None)
                if other_lane is None or other_key == lane_key:
                    continue
                other_jid = self._normalize_junction_id(
                    getattr(other_lane, "incoming_junction_id", None)
                )
                if other_jid != junction_id:
                    continue
                if self._lane_priority_flag(other_lane) is False:
                    has_secondary = True
                    break

            if not has_secondary:
                continue

            turns = getattr(info, "turns", None) or []
            exit_lanes = set(getattr(info, "exit_lanes", None) or [])
            maneuver_targets = set()
            for t in turns:
                to_lane = t.get("to_lane")
                via_lane = t.get("via_lane")
                if to_lane:
                    maneuver_targets.add(to_lane)
                if via_lane:
                    maneuver_targets.add(via_lane)
            if not (exit_lanes and maneuver_targets and exit_lanes.intersection(maneuver_targets)):
                continue

            candidate_keys.append(lane_key)
            score = min(float(getattr(lane, "length", 0.0)), 80.0)
            if target:
                parts = lane_key.split("_")
                edge_id = parts[1] if len(parts) >= 3 else None
                if edge_id == target:
                    score += 1000.0

            if score > best_score:
                best_score = score
                best_lane = lane

        return best_lane, candidate_keys

