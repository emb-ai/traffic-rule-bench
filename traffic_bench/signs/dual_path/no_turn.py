from traffic_bench.signs.base import BaseTrafficSign
from traffic_bench.signs.outgoing import SumoOutgoingMixin
import re


class _BaseNoTurnSign(SumoOutgoingMixin, BaseTrafficSign):
    """
    Shared logic for no-turn signs.
    - SUMO: uses lane.turns direction labels.
    - PG intersections: supports X (4 roads) and T (3 roads).
    """
    ICON_PATH = None
    PROHIBITED_MANEUVER = None
    RULE_DESCRIPTION = ""
    PG_FORBIDDEN_MODE = None  # "left" | "right" | "uturn"

    def __init__(self, lane, **kwargs):
        assert self.ICON_PATH is not None
        assert self.PROHIBITED_MANEUVER in ("l", "r", "t")
        assert self.PG_FORBIDDEN_MODE in ("left", "right", "uturn")
        kwargs.pop("applicable_lane_indices", None)
        super().__init__(lane=lane, icon_path=self.ICON_PATH, **kwargs)
        self.prohibited_maneuver = self.PROHIBITED_MANEUVER
        self.active_agents = {}
        self._semantic_forbidden_targets_by_render = {}
        # SUMO: outgoing edges after the signed approach, keyed by turn dir.
        self._sumo_approach_roads = set()
        self._sumo_outgoing_by_dir = {"l": set(), "r": set(), "s": set(), "t": set()}
        self._sumo_all_outgoing = set()
        self._sumo_forbidden_outgoing = set()
        self._sumo_outgoing_ready = False
        self.render_lanes = self._collect_render_lanes()
        self.enforcement_lanes = self._collect_enforcement_lanes()
        self.enforcement_lane_ids = {getattr(l, "index", None) for l in self.enforcement_lanes}

        # Cached PG topology around the sign.
        self._intersection_type = None         # "X" or "T"
        self._incoming_roads = []              # [(from_node, to_node)]
        self._outgoing_roads = []              # [(from_node, to_node)]
        self._sign_incoming_road = None        # (from_node, to_node)
        self._sign_outgoing_road = None        # (from_node, to_node)
        self._forbidden_target_road = None     # (from_node, to_node)
        self._pg_initialized = False

    @staticmethod
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

    def _lane_for_id(self, lane_id):
        if lane_id is None:
            return None
        for lane in self.enforcement_lanes:
            if getattr(lane, "index", None) == lane_id:
                return lane
        return None

    def _lane_dirs(self, lane_obj):
        return {
            self._normalize_turn_direction(t.get("direction"))
            for t in (getattr(lane_obj, "turns", None) or [])
            if self._normalize_turn_direction(t.get("direction")) in {"l", "r", "s", "t"}
        }

    def _all_main_lanes(self, road_network):
        lanes = []
        for lane_key, info in getattr(road_network, "graph", {}).items():
            if not isinstance(lane_key, str) or not lane_key.startswith("lane_") or ":" in lane_key:
                continue
            lane_obj = getattr(info, "lane", None)
            if lane_obj is not None:
                lanes.append((lane_key, lane_obj, getattr(info, "turns", None) or []))
        return lanes

    def _road_from_lane_id(self, lane_id):
        if isinstance(lane_id, tuple) and len(lane_id) >= 2:
            return lane_id[0], lane_id[1]
        if isinstance(lane_id, str) and lane_id.startswith("lane_"):
            # SUMO lane ids usually look like: lane_<edge_id>_<lane_idx>
            # Edge id can contain '-', so split from the right once.
            body = lane_id[5:]
            if "_" in body:
                edge_id, _lane_idx = body.rsplit("_", 1)
                if edge_id:
                    return edge_id
        return None

    def _lane_number_from_lane_id(self, lane_id):
        if isinstance(lane_id, tuple) and len(lane_id) >= 3:
            try:
                return int(lane_id[2])
            except Exception:
                return None
        if isinstance(lane_id, str) and lane_id.startswith("lane_"):
            body = lane_id[5:]
            if "_" in body:
                _edge_id, lane_idx = body.rsplit("_", 1)
                try:
                    return int(lane_idx)
                except Exception:
                    return None
        return None

    def _is_neighbor_lane(self, lane_a, lane_b):
        road_a = self._road_from_lane_id(lane_a)
        road_b = self._road_from_lane_id(lane_b)
        if road_a is None or road_b is None or road_a != road_b:
            return False
        idx_a = self._lane_number_from_lane_id(lane_a)
        idx_b = self._lane_number_from_lane_id(lane_b)
        if idx_a is None or idx_b is None:
            return False
        return abs(idx_a - idx_b) == 1

    def _is_opposite_road(self, road_a, road_b) -> bool:
        if road_a is None or road_b is None:
            return False
        if isinstance(road_a, str) and isinstance(road_b, str):
            return road_a == self._opposite_node(road_b)
        return road_a[0] == self._opposite_node(road_b[1]) and road_a[1] == self._opposite_node(road_b[0])

    def _opposite_lane_ids_of(self, lane_obj, all_lanes):
        lane_road = self._road_from_lane_id(getattr(lane_obj, "index", None))
        if lane_road is None:
            return set()
        opposite_lane_ids = set()
        for _lane_key, other_lane, _other_turns in all_lanes:
            other_id = getattr(other_lane, "index", None)
            if other_id is None:
                continue
            other_road = self._road_from_lane_id(other_id)
            if self._is_opposite_road(other_road, lane_road):
                opposite_lane_ids.add(other_id)
        return opposite_lane_ids

    def _same_road_lane_ids_of(self, lane_obj, all_lanes):
        lane_road = self._road_from_lane_id(getattr(lane_obj, "index", None))
        if lane_road is None:
            return set()
        same_road_lane_ids = set()
        for _lane_key, other_lane, _other_turns in all_lanes:
            other_id = getattr(other_lane, "index", None)
            if other_id is None:
                continue
            other_road = self._road_from_lane_id(other_id)
            if other_road == lane_road:
                same_road_lane_ids.add(other_id)
        return same_road_lane_ids

    def _reachable_lane_ids(self, src_lane):
        try:
            road_network = self.engine.current_map.road_network
        except Exception:
            return set()

        reachable = set()
        for t in (getattr(src_lane, "turns", None) or []):
            to_lane = t.get("to_lane")
            via_lane = t.get("via_lane")
            for lane_id in (to_lane, via_lane):
                if lane_id:
                    reachable.add(lane_id)
            for mid_lane in (to_lane, via_lane):
                if not mid_lane:
                    continue
                try:
                    mid_obj = road_network.get_lane(mid_lane)
                except Exception:
                    continue
                if mid_obj is None:
                    continue
                reachable.update(set(getattr(mid_obj, "exit_lanes", None) or []))
        return reachable

    def _is_render_lane_by_semantics(self, lane_obj, all_lanes):
        return True

    def _semantic_forbidden_targets_for_render(self, render_lane, all_lanes):
        return set()

    def _has_rotated_match_same_to_lane(self, render_lane, cand_lane, rotate_map):
        render_turns = getattr(render_lane, "turns", None) or []
        cand_turns = getattr(cand_lane, "turns", None) or []

        render_targets_by_dir = {}
        for t in render_turns:
            d = self._normalize_turn_direction(t.get("direction"))
            to_lane = t.get("to_lane")
            if d not in {"l", "r", "s", "t"} or not to_lane:
                continue
            render_targets_by_dir.setdefault(d, set()).add(to_lane)

        cand_targets_by_dir = {}
        for t in cand_turns:
            d = self._normalize_turn_direction(t.get("direction"))
            to_lane = t.get("to_lane")
            if d not in {"l", "r", "s", "t"} or not to_lane:
                continue
            cand_targets_by_dir.setdefault(d, set()).add(to_lane)

        for render_dir, render_targets in render_targets_by_dir.items():
            mapped_dir = rotate_map(render_dir)
            if not mapped_dir:
                continue
            cand_targets = cand_targets_by_dir.get(mapped_dir, set())
            if render_targets & cand_targets:
                return True
        return False

    def _collect_sumo_signed_approach_lanes(self):
        """Lanes on the explicitly signed SUMO approach (placement edge).

        Dual-path benches place the sign ON the ego approach that offers the
        prohibited maneuver. The legacy PG ``_collect_render_lanes`` path skips
        any lane whose turns include ``prohibited_maneuver`` and then hunts for
        counterpart approaches — that puts icons/enforcement on the wrong road
        and violations never arm.
        """
        lanes = [self.lane]
        ids = {getattr(self.lane, "index", None)}
        placement_road = self._road_from_lane_id(getattr(self.lane, "index", None))
        if placement_road is None:
            return lanes
        try:
            all_lanes = self._all_main_lanes(self.engine.current_map.road_network)
        except Exception:
            return lanes
        for _key, lane_obj, _turns in all_lanes:
            lid = getattr(lane_obj, "index", None)
            if lid is None or lid in ids:
                continue
            if self._road_from_lane_id(lid) != placement_road:
                continue
            lanes.append(lane_obj)
            ids.add(lid)
        # Cache forbidden first-hop targets for violation checks.
        for lane_obj in lanes:
            lid = getattr(lane_obj, "index", None)
            targets = set()
            for turn in getattr(lane_obj, "turns", None) or []:
                if self._normalize_turn_direction(turn.get("direction")) != self.prohibited_maneuver:
                    continue
                for key in ("to_lane", "via_lane"):
                    tgt = turn.get(key)
                    if tgt is not None:
                        targets.add(tgt)
            if targets:
                self._semantic_forbidden_targets_by_render[lid] = targets
            else:
                self._semantic_forbidden_targets_by_render.pop(lid, None)
        self._cache_sumo_outgoing_roads(lanes)
        return lanes

    def _collect_render_lanes(self):
        if self._is_sumo_network():
            return self._collect_sumo_signed_approach_lanes()

        try:
            road_network = self.engine.current_map.road_network
        except Exception:
            return [self.lane]

        all_lanes = self._all_main_lanes(road_network)

        render = []
        render_ids = set()
        for _lane_key, lane_obj, _turns in all_lanes:
            dirs = self._lane_dirs(lane_obj)
            if self.prohibited_maneuver in dirs:
                continue

            semantic_targets = self._semantic_forbidden_targets_for_render(lane_obj, all_lanes)
            if semantic_targets:
                self._semantic_forbidden_targets_by_render[getattr(lane_obj, "index", None)] = semantic_targets
            else:
                self._semantic_forbidden_targets_by_render.pop(getattr(lane_obj, "index", None), None)

            if not self._is_render_lane_by_semantics(lane_obj, all_lanes):
                continue

            lane_id = getattr(lane_obj, "index", None)
            if lane_id not in render_ids:
                render.append(lane_obj)
                render_ids.add(lane_id)

        return render if render else [self.lane]

    def _collect_enforcement_lanes(self):
        if self._is_sumo_network():
            return self._collect_sumo_signed_approach_lanes()
        return list(self.render_lanes)

    def _cache_sumo_outgoing_roads(self, approach_lanes) -> None:
        """Store junction outgoing edges and mark the prohibited turn's road."""
        mapped = self._map_sumo_outgoing_from_lanes(approach_lanes)
        forbidden = self._forbidden_outgoing_edges(mapped, self.prohibited_maneuver)
        self._sumo_approach_roads = mapped["approach_roads"]
        self._sumo_outgoing_by_dir = mapped["by_dir"]
        self._sumo_all_outgoing = mapped["all_outgoing"]
        self._sumo_forbidden_outgoing = forbidden
        self._sumo_outgoing_ready = bool(self._sumo_approach_roads)

        if forbidden:
            try:
                all_lanes = self._all_main_lanes(self.engine.current_map.road_network)
            except Exception:
                all_lanes = []
            extra = {
                getattr(other, "index", None)
                for _key, other, _turns in all_lanes
                if self._sumo_edge_id_from_lane_index(getattr(other, "index", None)) in forbidden
            }
            extra.discard(None)
            for lane_obj in approach_lanes or []:
                lid = getattr(lane_obj, "index", None)
                if lid is None:
                    continue
                self._semantic_forbidden_targets_by_render.setdefault(lid, set()).update(extra)

    def _ensure_sumo_outgoing_context(self) -> None:
        if self._sumo_outgoing_ready and self._sumo_all_outgoing:
            return
        lanes = self.enforcement_lanes or self.render_lanes or [self.lane]
        self._cache_sumo_outgoing_roads(lanes)

    def get_top_down_icon_poses(self):
        poses = []
        offset_from_end = max(0.1, float(self.lane.length) - float(self.placement_long))
        for lane in self.render_lanes:
            try:
                lane_len = float(lane.length)
                place_long = min(max(0.1, lane_len - offset_from_end), lane_len - 0.1)
                lat = lane.width_at(place_long) / 2 + 0.8
                pos = lane.position(place_long, lat)
                heading = lane.heading_theta_at(place_long) + 3.141592653589793 / 2
                poses.append((pos, heading))
            except Exception:
                continue
        return poses

    @staticmethod
    def _lane_to_road(lane_index):
        if lane_index is None:
            return None
        if isinstance(lane_index, tuple) and len(lane_index) >= 2:
            return lane_index[0], lane_index[1]
        return None

    @staticmethod
    def _opposite_node(node_name):
        node_name = str(node_name)
        if node_name.startswith("-"):
            return node_name[1:]
        return f"-{node_name}"

    @staticmethod
    def _extract_intersection_tag(node_name):
        """
        Return ("X"|"T", idx) for nodes like "...X0_0_" / "...T1_0_".
        """
        node_upper = str(node_name).upper()
        match = re.search(r"([XT])(\d)_0_", node_upper)
        if match:
            return match.group(1), int(match.group(2))
        return None

    def _infer_intersection_type(self, road_network, sign_road):
        approach_node = sign_road[1]
        outgoing = road_network.graph.get(approach_node, {})
        tags = set()
        for to_node, lanes in outgoing.items():
            if not lanes:
                continue
            tag = self._extract_intersection_tag(to_node)
            if tag is not None:
                tags.add(tag[0])

        if not tags:
            for from_node, to_dict in road_network.graph.items():
                for to_node, lanes in to_dict.items():
                    if not lanes:
                        continue
                    for node in (from_node, to_node):
                        tag = self._extract_intersection_tag(node)
                        if tag is not None:
                            tags.add(tag[0])

        assert tags, f"Could not infer intersection type for sign road: {sign_road}"
        if "X" in tags:
            return "X"
        if "T" in tags:
            return "T"
        raise AssertionError(f"Unsupported intersection type tags: {sorted(tags)}")

    def _road_rank(self, node_name):
        """
        Circular order:
        - X: S -> X0 -> X1 -> X2
        - T: S -> T0 -> T1
        """
        node_upper = str(node_name).upper()
        if "S0_0_" in node_upper:
            return 0
        tag = self._extract_intersection_tag(node_upper)
        if tag is not None and tag[0] == self._intersection_type:
            return 1 + tag[1]
        return 999

    def _incoming_road_rank(self, road):
        return self._road_rank(road[1])  # incoming ends at approach node

    def _outgoing_road_rank(self, road):
        return self._road_rank(road[0])  # outgoing starts at opposite approach node

    def _expected_road_count(self):
        return 4 if self._intersection_type == "X" else 3

    def _compute_pg_intersection_roads(self):
        road_network = self.engine.current_map.road_network
        sign_road = self._lane_to_road(self.lane.index)
        assert sign_road is not None, f"Invalid sign lane index for {type(self).__name__}: {self.lane.index}"

        self._intersection_type = self._infer_intersection_type(road_network, sign_road)
        expected_count = self._expected_road_count()

        incoming_to_node = {}
        for from_node, to_dict in road_network.graph.items():
            for to_node, lanes in to_dict.items():
                if lanes:
                    incoming_to_node.setdefault(to_node, []).append((from_node, to_node))

        # Candidate approach nodes: intersection decisions usually have 2 or 3 outgoing branches.
        incoming_roads = []
        for approach_node, to_dict in road_network.graph.items():
            outgoing_branches = [(to_node, lanes) for to_node, lanes in to_dict.items() if lanes]
            if len(outgoing_branches) < 2:
                continue
            candidates = sorted(incoming_to_node.get(approach_node, []), key=lambda r: r[0])
            if candidates:
                incoming_roads.append(candidates[0])

        # Keep only roads participating in our intersection ring order.
        incoming_roads = sorted(set(incoming_roads), key=self._incoming_road_rank)
        incoming_roads = [r for r in incoming_roads if self._incoming_road_rank(r) != 999]
        incoming_roads = incoming_roads[:expected_count]

        assert len(incoming_roads) == expected_count, (
            f"Expected {expected_count} incoming roads for {self._intersection_type}, got: {incoming_roads}"
        )
        assert sign_road in incoming_roads, (
            f"{type(self).__name__} must be attached to one of {expected_count} incoming roads of "
            f"{self._intersection_type}. Sign road: {sign_road}, incoming roads: {incoming_roads}"
        )

        outgoing_roads = []
        for in_from, in_to in incoming_roads:
            out_road = (self._opposite_node(in_to), self._opposite_node(in_from))
            lanes = road_network.graph.get(out_road[0], {}).get(out_road[1], [])
            assert lanes, f"Outgoing road not found for incoming road {in_from}->{in_to}: {out_road}"
            outgoing_roads.append(out_road)
        outgoing_roads = sorted(set(outgoing_roads), key=self._outgoing_road_rank)
        outgoing_roads = [r for r in outgoing_roads if self._outgoing_road_rank(r) != 999]

        assert len(outgoing_roads) == expected_count, (
            f"Expected {expected_count} outgoing roads for {self._intersection_type}, got: {outgoing_roads}"
        )

        sign_outgoing = (self._opposite_node(sign_road[1]), self._opposite_node(sign_road[0]))
        assert sign_outgoing in outgoing_roads, (
            f"Sign outgoing road {sign_outgoing} not among {self._intersection_type} outgoing roads: {outgoing_roads}"
        )

        return incoming_roads, outgoing_roads, sign_road, sign_outgoing

    def _select_forbidden_target_road(self, outgoing_roads, sign_outgoing):
        idx = outgoing_roads.index(sign_outgoing)
        if self.PG_FORBIDDEN_MODE == "left":
            return outgoing_roads[(idx - 1) % len(outgoing_roads)]
        if self.PG_FORBIDDEN_MODE == "right":
            return outgoing_roads[(idx + 1) % len(outgoing_roads)]
        return sign_outgoing

    def _ensure_pg_context(self):
        if self._pg_initialized:
            return
        incoming_roads, outgoing_roads, sign_road, sign_outgoing = self._compute_pg_intersection_roads()
        self._incoming_roads = incoming_roads
        self._outgoing_roads = outgoing_roads
        self._sign_incoming_road = sign_road
        self._sign_outgoing_road = sign_outgoing
        self._forbidden_target_road = self._select_forbidden_target_road(outgoing_roads, sign_outgoing)
        self._pg_initialized = True

    def _is_violating_sumo(self, vehicle, agent_id, current_lane) -> bool:
        """Arm on the signed approach; violate if ego then takes the forbidden outgoing road."""
        self._ensure_sumo_outgoing_context()
        return self._judge_sumo_outgoing(
            agent_id,
            current_lane,
            approach_roads=self._sumo_approach_roads,
            all_outgoing=self._sumo_all_outgoing,
            violate_roads=self._sumo_forbidden_outgoing,
            states=self.active_agents,
        )

    def _is_violating(self, vehicle) -> bool:
        agent_id = vehicle.name
        current_lane = vehicle.lane_index

        if self._is_sumo_network():
            return self._is_violating_sumo(vehicle, agent_id, current_lane)

        self._ensure_pg_context()
        current_road = self._lane_to_road(current_lane)
        if current_road is None:
            return False

        state = self.active_agents.setdefault(agent_id, {"armed": False, "last_road": None})

        if current_road == self._sign_incoming_road:
            state["armed"] = True
            state["last_road"] = current_road
            return False

        if state["armed"] and current_road in self._outgoing_roads and state["last_road"] != current_road:
            state["armed"] = False
            state["last_road"] = current_road
            if current_road == self._forbidden_target_road:
                print(
                    f"Violation: {self.get_rule_description()} "
                    f"(type: {self._intersection_type}, incoming: {self._sign_incoming_road}, "
                    f"forbidden outgoing: {self._forbidden_target_road}, ego outgoing: {current_road})"
                )
                return True
            return False

        return False

    def get_rule_description(self) -> str:
        return self.RULE_DESCRIPTION


class NoRightTurnSign(_BaseNoTurnSign):
    """Sign 3.18.1 — right turn prohibited."""
    ICON_PATH = "no_right_turn.png"
    PROHIBITED_MANEUVER = "r"
    # Used by SignComplianceMixin SUMO replan (same helper as 4.1.x).
    ALLOWED_DIRS = frozenset({"s", "l", "t"})
    RULE_DESCRIPTION = "Right turn prohibited (sign 3.18.1)."
    PG_FORBIDDEN_MODE = "right"

    @staticmethod
    def _rotate_map(direction: str) -> str:
        return {"l": "s", "s": "r", "t": "l"}.get(direction, "")

    def _semantic_forbidden_targets_for_render(self, lane_obj, all_lanes):
        forbidden = set()
        reachable = self._reachable_lane_ids(lane_obj)
        for _cand_key, cand_lane, _cand_turns in all_lanes:
            if cand_lane is lane_obj:
                continue
            has_rotated_match = self._has_rotated_match_same_to_lane(lane_obj, cand_lane, self._rotate_map)
            if not has_rotated_match:
                continue
            cand_road = self._road_from_lane_id(getattr(cand_lane, "index", None))
            has_turn_to_opposite = any(
                self._is_opposite_road(self._road_from_lane_id(reached_lane_id), cand_road)
                for reached_lane_id in reachable
            )
            if not has_turn_to_opposite:
                forbidden.update(self._same_road_lane_ids_of(cand_lane, all_lanes))
                forbidden.update(self._opposite_lane_ids_of(cand_lane, all_lanes))
        return forbidden

    def _is_render_lane_by_semantics(self, lane_obj, all_lanes):
        return bool(self._semantic_forbidden_targets_for_render(lane_obj, all_lanes))


class NoLeftTurnSign(_BaseNoTurnSign):
    """Sign 3.18.2 — left turn prohibited."""
    ICON_PATH = "no_left_turn.png"
    PROHIBITED_MANEUVER = "l"
    ALLOWED_DIRS = frozenset({"s", "r", "t"})
    RULE_DESCRIPTION = "Left turn prohibited (sign 3.18.2)."
    PG_FORBIDDEN_MODE = "left"

    @staticmethod
    def _rotate_map(direction: str) -> str:
        return {"r": "s", "s": "l", "t": "r"}.get(direction, "")

    def _semantic_forbidden_targets_for_render(self, lane_obj, all_lanes):
        forbidden = set()
        reachable = self._reachable_lane_ids(lane_obj)
        for _cand_key, cand_lane, _cand_turns in all_lanes:
            if cand_lane is lane_obj:
                continue
            has_rotated_match = self._has_rotated_match_same_to_lane(lane_obj, cand_lane, self._rotate_map)
            if not has_rotated_match:
                continue
            cand_road = self._road_from_lane_id(getattr(cand_lane, "index", None))
            has_turn_to_opposite = any(
                self._is_opposite_road(self._road_from_lane_id(reached_lane_id), cand_road)
                for reached_lane_id in reachable
            )
            if not has_turn_to_opposite:
                forbidden.update(self._same_road_lane_ids_of(cand_lane, all_lanes))
                forbidden.update(self._opposite_lane_ids_of(cand_lane, all_lanes))
        return forbidden

    def _is_render_lane_by_semantics(self, lane_obj, all_lanes):
        return bool(self._semantic_forbidden_targets_for_render(lane_obj, all_lanes))


class NoUTurnSign(_BaseNoTurnSign):
    """Sign 3.19 — U-turn prohibited."""
    ICON_PATH = "no_uturn.png"
    PROHIBITED_MANEUVER = "t"
    ALLOWED_DIRS = frozenset({"s", "r", "l"})
    RULE_DESCRIPTION = "U-turn prohibited (sign 3.19)."
    PG_FORBIDDEN_MODE = "uturn"

    def _semantic_forbidden_targets_for_render(self, lane_obj, all_lanes):
        forbidden = set()
        reachable = self._reachable_lane_ids(lane_obj)
        for _cand_key, cand_lane, _cand_turns in all_lanes:
            if cand_lane is lane_obj:
                continue
            cand_road = self._road_from_lane_id(getattr(cand_lane, "index", None))
            lane_road = self._road_from_lane_id(getattr(lane_obj, "index", None))
            if not self._is_opposite_road(cand_road, lane_road):
                continue
            has_turn_to_opposite = any(
                self._is_opposite_road(self._road_from_lane_id(reached_lane_id), cand_road)
                for reached_lane_id in reachable
            )
            if not has_turn_to_opposite:
                forbidden.update(self._same_road_lane_ids_of(cand_lane, all_lanes))
                forbidden.update(self._opposite_lane_ids_of(cand_lane, all_lanes))
        return forbidden

    def _is_render_lane_by_semantics(self, lane_obj, all_lanes):
        return bool(self._semantic_forbidden_targets_for_render(lane_obj, all_lanes))
