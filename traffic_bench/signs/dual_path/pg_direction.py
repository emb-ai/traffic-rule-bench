from __future__ import annotations

from typing import Any, Sequence

from traffic_bench.signs.base import BaseTrafficSign

MANEUVER_TO_SYMBOL = {
    "left": "L",
    "straight": "S",
    "right": "R",
    "uturn": "U",
}

class PGDirectionSign(BaseTrafficSign):
    """
    PG-specific direction sign.

    This class keeps DirectionSign violation semantics while exposing class-level
    helpers to generate/apply PG lane turns and regex-valid lane layouts.
    """

    # Filled from the selected regex layout: lane_index -> allowed next lane indices
    _layout_allowed_to_lanes = {}
    # Same mapping but projected to roads only: lane_index -> {(to_from, to_to), ...}
    _layout_allowed_to_roads = {}
    # Lane indices where PGDirectionSign is currently spawned.
    _registered_sign_lanes = set()
    # Default (non-layout) allowed moves for spawned sign lanes.
    _base_allowed_to_lanes = {}
    _base_allowed_to_roads = {}
    # Per-agent global tracking context:
    #   agent_id -> {"owner": lane_index_with_sign, "tracked": latest_lane_within_same_source}
    _agent_context = {}
    def __init__(self, lane, **kwargs):
        self.lane = lane
        self._lane_index_norm = self._normalize_lane_index(getattr(lane, "index", None))
        if self._lane_index_norm is not None:
            self.__class__._registered_sign_lanes.add(self._lane_index_norm)
        self.allowed_to_lanes = set()
        for turn in getattr(lane, "turns", []):
            to_lane = turn.get("to_lane")
            if to_lane:
                self.allowed_to_lanes.add(self._normalize_lane_index(to_lane))
        self.allowed_to_roads = {
            (idx[0], idx[1]) for idx in self.allowed_to_lanes if idx is not None
        }
        if self._lane_index_norm is not None:
            self.__class__._base_allowed_to_lanes[self._lane_index_norm] = set(self.allowed_to_lanes)
            self.__class__._base_allowed_to_roads[self._lane_index_norm] = set(self.allowed_to_roads)
        layout_allowed = self.__class__._layout_allowed_to_lanes.get(self._lane_index_norm)
        if layout_allowed is not None:
            self.allowed_to_lanes = set(layout_allowed)
        layout_allowed_roads = self.__class__._layout_allowed_to_roads.get(self._lane_index_norm)
        if layout_allowed_roads is not None:
            self.allowed_to_roads = set(layout_allowed_roads)
        super().__init__(lane=lane, **kwargs)

    def check_violation(self, vehicle, for_reward=False) -> bool:
        """
        Continue checking tracked vehicles even after they leave sign lane.

        BaseTrafficSign.check_violation() gates by is_in_drivable_area(), which is
        too strict for turn-compliance checks: at the moment of lane transition the
        vehicle often already has different road direction, and violation is missed.
        """
        agent_id = getattr(vehicle, "name", None)
        is_tracked = False
        if agent_id is not None:
            ctx = self.__class__._agent_context.get(agent_id)
            is_tracked = bool(ctx and ctx.get("owner") == self._lane_index_norm)
        if (not is_tracked) and (not self.is_in_drivable_area(vehicle)):
            return False

        vid = getattr(vehicle, "id", None)
        if vid is None:
            vid = agent_id
        key = "reported_for_reward" if for_reward else "reported_for_metrics"
        state = self._vehicle_states.setdefault(
            vid,
            {"reported_for_reward": False, "reported_for_metrics": False},
        )

        if (not self._is_violating(vehicle)) or state[key]:
            return False
        state[key] = True
        return True

    def _is_violating(self, vehicle) -> bool:
        """
        Logic: layout (generator) gives allowed next sockets (roads out of the block).
        Store that set; only check violation when agent EXITS the internal block:
        if exit road not in allowed set → violation.
        """
        agent_id = getattr(vehicle, "name", None)
        if agent_id is None:
            return False
        cls = self.__class__
        current_lane = self._normalize_lane_index(getattr(vehicle, "lane_index", None))
        current_road = (current_lane[0], current_lane[1]) if current_lane and len(current_lane) >= 2 else None
        ctx = cls._agent_context.get(agent_id)

        if ctx is None:
            if current_lane == self._lane_index_norm:
                cls._agent_context[agent_id] = {"owner": self._lane_index_norm}
            return False

        owner_lane = ctx.get("owner")
        if owner_lane is None or owner_lane != self._lane_index_norm:
            return False

        block_key = ctx.get("block_key")
        if current_lane and cls._is_entering_or_within_intersection(current_lane):
            if block_key is None:
                block_key = cls._intersection_node_key(str(current_lane[1])) or cls._intersection_node_key(str(current_lane[0]))
                if block_key is not None:
                    ctx["block_key"] = block_key
            return False  # inside block: never violate

        if block_key is None or not current_lane:
            return False  # not yet entered any block

        from_key = cls._intersection_node_key(str(current_lane[0]))
        to_key = cls._intersection_node_key(str(current_lane[1])) if current_lane and len(current_lane) >= 2 else None
        is_leaving = from_key == block_key and to_key != block_key
        if not is_leaving:
            cls._agent_context.pop(agent_id, None)
            return False  # already past exit or on other road

        allowed_lanes = cls._layout_allowed_to_lanes.get(owner_lane) or cls._base_allowed_to_lanes.get(owner_lane)
        allowed_roads = set(cls._layout_allowed_to_roads.get(owner_lane) or cls._base_allowed_to_roads.get(owner_lane) or ())
        if allowed_lanes:
            allowed_roads |= {(idx[0], idx[1]) for idx in allowed_lanes if idx is not None}
        exit_roads = {r for r in allowed_roads if cls._intersection_node_key(str(r[0])) == block_key and cls._intersection_node_key(str(r[1])) != block_key}
        cls._agent_context.pop(agent_id, None)
        return current_road not in exit_roads if current_road else False

    @staticmethod
    def _normalize_lane_index(index):
        if isinstance(index, tuple) and len(index) >= 3:
            try:
                return (str(index[0]), str(index[1]), int(index[2]))
            except Exception:
                return None
        if isinstance(index, str):
            import re

            m = re.fullmatch(
                r"^\(\s*'(?P<from>[^']+)'\s*,\s*'(?P<to>[^']+)'\s*,\s*(?P<lane>-?\d+)\s*\)$",
                index.strip(),
            )
            if m:
                return (m.group("from"), m.group("to"), int(m.group("lane")))
        return None

    @staticmethod
    def _parse_node(node: str):
        from metadrive_core.pg_direction_sign_generator import parse_node_token

        return parse_node_token(node)

    @classmethod
    def _arm_index(cls, node: str):
        try:
            token = cls._parse_node(str(node))
            block_type = str(getattr(token, "block_type", "") or "").upper()
            part_index = getattr(token, "part_index", None)
            if block_type in {"X", "T"} and part_index is not None:
                return int(part_index)
        except Exception:
            return None
        return None

    @classmethod
    def _infer_source_arm(cls, lane_idx, turn_targets):
        """
        Infer incoming arm for a source lane.
        1) Use arm on source to-node, then from-node.
        2) If unavailable (e.g. chevron before X), infer as the missing arm among targets.
        """
        if lane_idx is None:
            return None
        src_from, src_to, _ = lane_idx
        arm = cls._arm_index(src_to)
        if arm is not None:
            return arm
        arm = cls._arm_index(src_from)
        if arm is not None:
            return arm

        target_arms = {cls._arm_index(t[1]) for t in turn_targets}
        target_arms.discard(None)
        missing = sorted({0, 1, 2, 3} - target_arms)
        if len(missing) == 1:
            return int(missing[0])
        return None

    @classmethod
    def _symbol_by_relative_arm(cls, lane_idx, to_lane, source_arm):
        """
        Resolve symbol from relative topology:
          right=(src+1)%4, straight=(src+2)%4, left=(src+3)%4
        """
        if lane_idx is None or to_lane is None:
            return None
        src_from, _, _ = lane_idx
        dst_from, dst_to, _ = to_lane

        # U-turn-like edge back toward source node.
        if str(dst_to) == str(src_from):
            return "U"
        if source_arm is None:
            return None

        target_arm = cls._arm_index(dst_to)
        if target_arm is None:
            target_arm = cls._arm_index(dst_from)
        if target_arm is None:
            return None

        delta = (int(target_arm) - int(source_arm)) % 4
        if delta == 1:
            return "R"
        if delta == 2:
            return "S"
        if delta == 3:
            return "L"
        if delta == 0:
            # Same arm (e.g. 1X1_0_ -> 1X1_1_): not a real U-turn,
            # just an internal arm traversal.  Fall back to maneuver.
            return None
        return None

    @classmethod
    def _symbol_for_verifier_turn(cls, lane_idx, to_lane, maneuver, source_arm):
        """
        Resolve turn symbol for verifier.

        Priority:
        1) Maneuver from lane.turns (matches generator's inference)
        2) Fallback to relative arm topology
        """
        symbol = MANEUVER_TO_SYMBOL.get(maneuver)
        if symbol is not None:
            return symbol
        return cls._symbol_by_relative_arm(
            lane_idx=lane_idx,
            to_lane=to_lane,
            source_arm=source_arm,
        )

    @classmethod
    def _is_intersection_core_node(cls, node: str) -> bool:
        try:
            t = cls._parse_node(node)
            block_type = str(getattr(t, "block_type", "") or "").upper()
            road_index = getattr(t, "road_index", None)
            return block_type in {"X", "T"} and road_index is not None and int(road_index) == 0
        except Exception:
            return False

    @classmethod
    def _is_intersection_core_entry_lane(cls, lane_idx) -> bool:
        if lane_idx is None:
            return False
        try:
            return cls._is_intersection_core_node(str(lane_idx[1]))
        except Exception:
            return False

    @classmethod
    def _is_fully_within_intersection_block(cls, lane_idx) -> bool:
        """Both from-node and to-node belong to the same X/T intersection block."""
        if lane_idx is None:
            return False
        try:
            from_key = cls._intersection_node_key(str(lane_idx[0]))
            to_key = cls._intersection_node_key(str(lane_idx[1]))
            return from_key is not None and to_key is not None and from_key == to_key
        except Exception:
            return False

    @classmethod
    def _is_entering_or_within_intersection(cls, lane_idx) -> bool:
        """
        To-node belongs to an X/T intersection block.

        Covers entering lanes (>>>->1X0_0_), fully internal lanes,
        and cross-connectors.  Excludes leaving lanes where only the
        from-node is in the block (-1X0_0_->->>>), so the exit check
        fires there.
        """
        if lane_idx is None:
            return False
        try:
            return cls._intersection_node_key(str(lane_idx[1])) is not None
        except Exception:
            return False

    @classmethod
    def _is_intersection_internal_lane(cls, lane_idx) -> bool:
        """
        Internal arm segment inside one X/T block:
        same block, same arm(part_index), road_index transition 0<->1.
        Example: 1X1_0_ -> 1X1_1_ (or reverse)
        """
        if lane_idx is None:
            return False
        try:
            from_t = cls._parse_node(str(lane_idx[0]))
            to_t = cls._parse_node(str(lane_idx[1]))
            from_type = str(getattr(from_t, "block_type", "") or "").upper()
            to_type = str(getattr(to_t, "block_type", "") or "").upper()
            if from_type not in {"X", "T"} or to_type not in {"X", "T"}:
                return False
            if from_type != to_type:
                return False
            if getattr(from_t, "block_index", None) is None or getattr(to_t, "block_index", None) is None:
                return False
            if int(from_t.block_index) != int(to_t.block_index):
                return False
            if getattr(from_t, "part_index", None) is None or getattr(to_t, "part_index", None) is None:
                return False
            if int(from_t.part_index) != int(to_t.part_index):
                return False
            if getattr(from_t, "road_index", None) is None or getattr(to_t, "road_index", None) is None:
                return False
            a = int(from_t.road_index)
            b = int(to_t.road_index)
            return {a, b} == {0, 1}
        except Exception:
            return False

    @classmethod
    def _intersection_node_key(cls, node: str):
        try:
            t = cls._parse_node(node)
            block_type = str(getattr(t, "block_type", "") or "").upper()
            block_index = getattr(t, "block_index", None)
            if block_type in {"X", "T"} and block_index is not None:
                return (block_type, int(block_index))
        except Exception:
            return None
        return None

    @classmethod
    def _intersection_block_key_for_lane(cls, lane_idx):
        if lane_idx is None:
            return None
        try:
            from_key = cls._intersection_node_key(str(lane_idx[0]))
            to_key = cls._intersection_node_key(str(lane_idx[1]))
            if from_key is not None and to_key is not None:
                return from_key if from_key == to_key else to_key
            return to_key or from_key
        except Exception:
            return None

    @classmethod
    def _is_intersection_lane(cls, lane_idx) -> bool:
        return cls._intersection_block_key_for_lane(lane_idx) is not None

    @classmethod
    def _lane_in_same_intersection_block(cls, lane_idx, block_key) -> bool:
        if lane_idx is None or block_key is None:
            return False
        try:
            from_key = cls._intersection_node_key(str(lane_idx[0]))
            to_key = cls._intersection_node_key(str(lane_idx[1]))
            return from_key == block_key or to_key == block_key
        except Exception:
            return False

    @classmethod
    def _graph_continuations(cls, road_network: Any, idx):
        """
        Find graph-based continuations for a lane whose turns are empty.

        MetaDrive intersection arms use positive/negative node pairs
        (e.g. 1X0_0_->1X0_1_ and -1X0_1_->-1X0_0_).  The positive arm
        is a dead end in the graph; the continuation lives under the
        corresponding negative node.
        """
        graph = getattr(road_network, "graph", None)
        if not graph:
            return []

        to_node = str(idx[1])
        lane_num = int(idx[2])
        result = []

        for candidate_node in (to_node, cls._flip_node_sign(to_node)):
            for next_node, next_lanes in (graph.get(candidate_node) or {}).items():
                for i, next_lane in enumerate(list(next_lanes or [])):
                    if i != lane_num:
                        continue
                    norm = cls._normalize_lane_index(
                        getattr(next_lane, "index", None)
                    )
                    if norm is None:
                        norm = (candidate_node, str(next_node), i)
                    result.append(norm)
        return result

    @staticmethod
    def _flip_node_sign(node: str) -> str:
        if node.startswith("-"):
            return node[1:]
        return "-" + node

    @classmethod
    def _expand_past_intersection_core(cls, road_network: Any, lane_idx):
        """
        Expand one allowed lane until leaving the same X/T intersection block.

        Inside the block the expansion uses graph edges (including the
        positive ↔ negative node flip) so that it always reaches the
        physical exit roads regardless of which per-lane maneuver turns
        are assigned.
        """
        start = cls._normalize_lane_index(lane_idx)
        if start is None or road_network is None:
            return set()
        start_block = cls._intersection_block_key_for_lane(start)
        if start_block is None:
            return {start}

        terminals = set()
        frontier = [start]
        visited = set()
        max_depth = 20

        for _ in range(max_depth):
            if not frontier:
                break
            next_frontier = []
            for idx in frontier:
                if idx in visited:
                    continue
                visited.add(idx)

                if not cls._lane_in_same_intersection_block(idx, start_block):
                    terminals.add(idx)
                    continue

                for cont in cls._graph_continuations(road_network, idx):
                    if cont not in visited:
                        next_frontier.append(cont)
            frontier = next_frontier

        if not terminals and frontier:
            terminals.update(frontier)
        return terminals

    @classmethod
    def _project_allowed_roads(cls, road_network: Any, allowed_to_lanes):
        roads = set()
        for lane_idx in list(allowed_to_lanes or []):
            idx = cls._normalize_lane_index(lane_idx)
            if idx is None:
                continue
            expanded = cls._expand_past_intersection_core(road_network, idx)
            if expanded:
                roads.update((x[0], x[1]) for x in expanded if x is not None)
            else:
                roads.add((idx[0], idx[1]))
        return roads

    @classmethod
    def configure_layout_verifier_from_report(cls, road_network: Any, report: Any, layout_index: int = 0):
        """
        Build lane_index -> allowed_to_lanes map from selected regex layout.

        Selection rule:
        - if regex_valid_layouts is present, pick one by layout_index
        - otherwise fallback to dsl_full
        """
        cls._layout_allowed_to_lanes = {}
        cls._layout_allowed_to_roads = {}
        cls._registered_sign_lanes = set()
        cls._base_allowed_to_lanes = {}
        cls._base_allowed_to_roads = {}
        cls._agent_context = {}
        if road_network is None or report is None:
            return

        specs = list(getattr(report, "approach_dsl_specs", []) or [])
        for spec in specs:
            layouts = tuple(getattr(spec, "regex_valid_layouts", ()) or ())
            if layouts:
                chosen = layouts[max(0, min(int(layout_index), len(layouts) - 1))]
            else:
                chosen = str(getattr(spec, "dsl_full", "") or "")

            tokens = chosen.split("|") if chosen else []
            lane_indices = tuple(getattr(spec, "lane_indices", ()) or ())
            if len(tokens) != len(lane_indices):
                continue

            for lane_idx, token in zip(lane_indices, tokens):
                lane_idx_norm = cls._normalize_lane_index(lane_idx)
                if lane_idx_norm is None:
                    continue
                lane = road_network.get_lane(lane_idx_norm)
                if lane is None:
                    continue

                allowed_symbols = set(token)
                allowed_to_lanes = set()
                turn_targets = []
                for turn in getattr(lane, "turns", []) or []:
                    to_lane = cls._normalize_lane_index(turn.get("to_lane"))
                    if to_lane is not None:
                        turn_targets.append(to_lane)
                source_arm = cls._infer_source_arm(
                    lane_idx=lane_idx_norm,
                    turn_targets=turn_targets,
                )
                for turn in getattr(lane, "turns", []) or []:
                    maneuver = turn.get("maneuver")
                    to_lane = cls._normalize_lane_index(turn.get("to_lane"))
                    if to_lane is None:
                        continue
                    symbol = cls._symbol_for_verifier_turn(
                        lane_idx=lane_idx_norm,
                        to_lane=to_lane,
                        maneuver=maneuver,
                        source_arm=source_arm,
                    )
                    if symbol in allowed_symbols:
                        allowed_to_lanes.add(to_lane)

                cls._layout_allowed_to_lanes[lane_idx_norm] = allowed_to_lanes
                cls._layout_allowed_to_roads[lane_idx_norm] = cls._project_allowed_roads(
                    road_network=road_network,
                    allowed_to_lanes=allowed_to_lanes,
                )

    def get_rule_description(self) -> str:
        return "Vehicle violated direction sign: turned into a disallowed lane."

    @property
    def top_down_length(self):
        return 0

    @property
    def top_down_width(self):
        return 0

    @classmethod
    def build_turn_plan(
        cls,
        road_network: Any,
        allow_uturn: bool = False,
        strict: bool = False,
        regex_layout_mode: str = "all",
        regex_max_layouts: int = 0,
        regex_seed: int = 42,
        allow_s_block_multi_exit: bool = False,
    ):
        from metadrive_core.pg_direction_sign_generator import build_pg_direction_sign_plan

        return build_pg_direction_sign_plan(
            road_network=road_network,
            allow_uturn=allow_uturn,
            strict=strict,
            regex_layout_mode=regex_layout_mode,
            regex_max_layouts=regex_max_layouts,
            regex_seed=regex_seed,
            allow_s_block_multi_exit=bool(allow_s_block_multi_exit),
        )

    @classmethod
    def generate_and_apply_turns(
        cls,
        road_network: Any,
        allow_uturn: bool = False,
        strict: bool = False,
        overwrite_existing: bool = True,
        regex_layout_mode: str = "all",
        regex_max_layouts: int = 0,
        regex_seed: int = 42,
        layout_index: int = 0,
        allow_s_block_multi_exit: bool = False,
    ):
        from metadrive_core.pg_direction_sign_generator import generate_and_apply_pg_direction_turns

        report = generate_and_apply_pg_direction_turns(
            road_network=road_network,
            allow_uturn=allow_uturn,
            strict=strict,
            overwrite_existing=overwrite_existing,
            regex_layout_mode=regex_layout_mode,
            regex_max_layouts=regex_max_layouts,
            regex_seed=regex_seed,
            allow_s_block_multi_exit=bool(allow_s_block_multi_exit),
        )
        cls.configure_layout_verifier_from_report(
            road_network=road_network,
            report=report,
            layout_index=layout_index,
        )
        return report

    @classmethod
    def generate_regex_layout_variants(
        cls,
        lane_tokens: Sequence[str],
        mode: str = "all",
        max_layouts: int = 64,
        seed: int = 42,
    ):
        from metadrive_core.pg_direction_sign_generator import (
            SOCKET_LAYOUT_RE,
            SOCKET_LAYOUT_TOKENS,
            generate_layout_variants,
        )

        return generate_layout_variants(
            lane_tokens=lane_tokens,
            socket_layout_re=SOCKET_LAYOUT_RE,
            tokens=SOCKET_LAYOUT_TOKENS,
            mode=mode,
            max_layouts=max_layouts,
            seed=seed,
        )
