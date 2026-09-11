"""PDD 4.1.1–4.1.6 lane-allowed-direction signs.

Enforcement model (SUMO dual-path):
  * Exactly one plate is placed on the ego approach arm (placement lane).
  * Top-down icon is drawn only on that placement lane (not on peer/other arms).
  * Approach scope for arming = every lane on the signed SUMO edge.
  * All real outgoing roads reachable from that approach are collected, then
    classified geometrically (heading delta → l/r/s/t) relative to the approach.
  * Allowed exits = those whose geometric label is in ``ALLOWED_DIRS``;
    every other outgoing is a violation.
"""

from traffic_bench.signs.base import BaseTrafficSign, same_road_check
from traffic_bench.signs.outgoing import (
    SumoOutgoingMixin,
    allow_forbid_outgoing,
    filter_same_edge_lane_ids,
    is_internal_lane_id,
    normalize_turn_direction,
)


def _normalize_turn_direction(raw_dir: str) -> str:
    return normalize_turn_direction(raw_dir)


class LaneAllowedDirectionSign(SumoOutgoingMixin, BaseTrafficSign):
    ALLOWED_DIRS = frozenset()

    def __init__(self, lane, **kwargs):
        turns = list(getattr(lane, "turns", []) or [])
        self._has_turn_metadata = bool(turns)
        self.allowed_lanes = set()
        for turn in turns:
            to_lane = turn.get("to_lane")
            if to_lane:
                self.allowed_lanes.add(to_lane)
        self._preset_applicable_lane_indices = kwargs.pop("applicable_lane_indices", None)
        self.intersection_name = kwargs.pop("intersection_name", None)
        # lane_id -> last seen approach lane while agent still under this sign
        self.active_agents = {}
        self._sumo_agent_states = {}
        # Agents whose *first* departure from the signed approach was already
        # judged. Dual-path compliant routes (esp. 4.1.2) often loop back onto
        # the same approach and continue with a different exit; that second
        # pass must not re-trigger the rule.
        self._cleared_agents = set()
        super().__init__(lane=lane, **kwargs)
        # Approach = ego arm (all lanes on the signed edge). Not "lanes whose
        # turn-set equals ALLOWED_DIRS" — that hid forbidden exits on peer lanes.
        self.applicable_lanes = self._collect_approach_lanes()
        self.applicable_lane_ids = {getattr(l, "index", None) for l in self.applicable_lanes}
        self.allowed_lanes_by_source = self._build_allowed_targets()
        self._sumo_outgoing_mapped = None

    def _collect_approach_lanes(self):
        """Lanes on the signed ego approach arm only (one edge / one arm)."""
        lanes = [self.lane]
        try:
            road_network = self.engine.current_map.road_network
            graph = getattr(road_network, "graph", {}) or {}
        except Exception:
            return lanes

        if self._preset_applicable_lane_indices:
            preset = []
            for lane_id in self._preset_applicable_lane_indices:
                try:
                    lane_obj = road_network.get_lane(lane_id)
                except Exception:
                    continue
                if lane_obj is not None:
                    preset.append(lane_obj)
            if preset:
                return preset

        sign_idx = getattr(self.lane, "index", None)
        sign_edge = self._sumo_edge_id_from_lane_index(sign_idx)
        candidate_ids = set(getattr(self.lane, "incoming_junction_lanes", None) or [])
        if sign_idx is not None:
            candidate_ids.add(sign_idx)

        # Same-edge peers may not appear in incoming_junction_lanes; scan graph.
        if self._is_sumo_network() and sign_edge:
            for lane_id in graph:
                if not isinstance(lane_id, str) or is_internal_lane_id(lane_id):
                    continue
                if self._sumo_edge_id_from_lane_index(lane_id) == sign_edge:
                    candidate_ids.add(lane_id)

        out = []
        seen = set()
        if self._is_sumo_network() and sign_edge:
            for lane_id in filter_same_edge_lane_ids(
                sign_idx, candidate_ids, self._sumo_edge_id_from_lane_index
            ):
                if lane_id in seen:
                    continue
                try:
                    lane_obj = road_network.get_lane(lane_id)
                except Exception:
                    continue
                if lane_obj is None:
                    continue
                out.append(lane_obj)
                seen.add(lane_id)
            return out or lanes

        for lane_id in candidate_ids:
            if lane_id in seen:
                continue
            if sign_idx is not None and not same_road_check(sign_idx, lane_id):
                continue
            try:
                lane_obj = road_network.get_lane(lane_id)
            except Exception:
                continue
            if lane_obj is None:
                continue
            out.append(lane_obj)
            seen.add(lane_id)
        return out or lanes

    # Back-compat alias used by older call sites / docs.
    _collect_applicable_lanes = _collect_approach_lanes

    def _build_allowed_targets(self):
        out = {}
        for lane in self.applicable_lanes:
            lane_id = getattr(lane, "index", None)
            allowed = set()
            for turn in (getattr(lane, "turns", None) or []):
                d = _normalize_turn_direction(turn.get("direction"))
                if d not in self.ALLOWED_DIRS:
                    continue
                to_lane = turn.get("to_lane")
                if to_lane:
                    allowed.add(to_lane)
                    to_lane_obj = self.engine.current_map.road_network.get_lane(to_lane)
                    next_lanes = set(getattr(to_lane_obj, "exit_lanes", None) or [])
                    for next_lane in next_lanes:
                        allowed.add(next_lane)
                        if ":" in str(next_lane):
                            to_lane_obj2 = self.engine.current_map.road_network.get_lane(next_lane)
                            next_lanes2 = set(getattr(to_lane_obj2, "exit_lanes", None) or [])
                            for next_lane2 in next_lanes2:
                                allowed.add(next_lane2)

            out[lane_id] = allowed
        return out

    def _lane_for_id(self, lane_id):
        if lane_id is None:
            return None
        for lane in self.applicable_lanes:
            if getattr(lane, "index", None) == lane_id:
                return lane
        return None

    @staticmethod
    def _lane_index_parts(lane_idx):
        if lane_idx is None:
            return None, None
        if isinstance(lane_idx, tuple) and len(lane_idx) >= 3:
            return (lane_idx[0], lane_idx[1]), int(lane_idx[2])
        if isinstance(lane_idx, str) and lane_idx.startswith("lane_"):
            core = lane_idx[5:]
            if "_" not in core:
                return None, None
            prefix, last = core.rsplit("_", 1)
            try:
                return prefix, int(last)
            except Exception:
                return prefix, None
        return None, None

    @classmethod
    def _is_pre_junction_lane_change(cls, src_lane_obj, current_lane_id) -> bool:
        if src_lane_obj is None or current_lane_id is None:
            return False
        incoming = set(getattr(src_lane_obj, "incoming_junction_lanes", None) or [])
        if not incoming:
            return False
        if current_lane_id not in incoming:
            return False

        src_key, src_lane_num = cls._lane_index_parts(getattr(src_lane_obj, "index", None))
        cur_key, cur_lane_num = cls._lane_index_parts(current_lane_id)
        if src_key is None or cur_key is None:
            return False
        if src_key != cur_key:
            return False
        if src_lane_num is None or cur_lane_num is None:
            return False
        return abs(src_lane_num - cur_lane_num) == 1

    def get_top_down_icon_poses(self):
        """Single icon on the placement lane — one plate on the ego arm."""
        lane = self.lane
        try:
            lane_len = float(lane.length)
            place_long = float(self.placement_long)
            place_long = min(max(0.1, place_long), lane_len - 0.1)
            lat = getattr(self, "_lateral_offset", None)
            if lat is None:
                lat = lane.width_at(place_long) / 2 + 0.8
            pos = lane.position(place_long, lat)
            heading = lane.heading_theta_at(place_long) + 3.141592653589793 / 2
            return [(pos, heading)]
        except Exception:
            return []

    def _ensure_sumo_outgoing_context(self) -> None:
        if self._sumo_outgoing_mapped and self._sumo_outgoing_mapped.get("all_outgoing"):
            return
        # Collect every real outgoing reachable from the ego-arm approach lanes.
        mapped = self._map_sumo_outgoing_from_lanes(self.applicable_lanes)
        all_outgoing = set(mapped.get("all_outgoing") or ())
        if not all_outgoing:
            self._sumo_outgoing_mapped = mapped
            return
        # Reclassify *all* outgoings geometrically from the approach heading.
        # Turn-table labels on a single peer lane must not define the forbid set.
        geo = self._classify_outgoing_geometrically(all_outgoing)
        labeled_any = any(geo.get(d) for d in ("l", "r", "s", "t"))
        if labeled_any:
            # Edges we could not heading-classify keep their turn-table label.
            turn_by_dir = mapped.get("by_dir") or {}
            covered = set().union(*(geo.get(d, set()) for d in geo))
            for d, edges in turn_by_dir.items():
                for edge in edges:
                    if edge not in covered and d in geo:
                        geo[d].add(edge)
            mapped = {
                "approach_roads": mapped.get("approach_roads") or set(),
                "by_dir": geo,
                "all_outgoing": all_outgoing,
            }
        self._sumo_outgoing_mapped = mapped

    def _is_violating_lane_targets(self, vehicle) -> bool:
        agent_id = vehicle.name
        current_lane = vehicle.lane_index

        if agent_id in self._cleared_agents:
            return False

        if current_lane in self.applicable_lane_ids:
            self.active_agents[agent_id] = current_lane
            return False

        if agent_id in self.active_agents:
            prev_lane = self.active_agents[agent_id]
            if prev_lane in self.applicable_lane_ids and current_lane != prev_lane:
                if isinstance(current_lane, str) and (
                    "junction_" in current_lane or "lane_:" in current_lane
                ):
                    return False
                allowed_targets = self.allowed_lanes_by_source.get(prev_lane, set())
                src_lane_obj = self._lane_for_id(prev_lane)
                self.active_agents.pop(agent_id, None)
                self._cleared_agents.add(agent_id)
                if self._is_pre_junction_lane_change(src_lane_obj, current_lane):
                    return False
                if current_lane not in allowed_targets:
                    return True
                return False
        return False

    def _is_violating(self, vehicle) -> bool:
        # Without turn metadata (typical for MetaDrive PG maps) we cannot
        # enumerate allowed exit lanes, so the rule is unverifiable and must
        # NOT report a violation on every lane transition.
        if not self._has_turn_metadata:
            return False

        if self._is_sumo_network():
            self._ensure_sumo_outgoing_context()
            mapped = self._sumo_outgoing_mapped or {}
            all_outgoing = set(mapped.get("all_outgoing") or ())
            if all_outgoing:
                _allowed, violate_roads = allow_forbid_outgoing(
                    all_outgoing, mapped.get("by_dir") or {}, self.ALLOWED_DIRS
                )
                return self._judge_sumo_outgoing(
                    vehicle.name,
                    vehicle.lane_index,
                    approach_roads=mapped.get("approach_roads") or set(),
                    all_outgoing=all_outgoing,
                    violate_roads=violate_roads,
                    states=self._sumo_agent_states,
                    cleared=self._cleared_agents,
                )

        return self._is_violating_lane_targets(vehicle)

    def get_rule_description(self) -> str:
        return "Violation! not following allowed line."

    @property
    def top_down_length(self):
        return 0

    @property
    def top_down_width(self):
        return 0


class LaneAllowedDirectionSign4_1_1(LaneAllowedDirectionSign):
    ALLOWED_DIRS = frozenset({"s"})

    def __init__(self, lane, **kwargs):
        super().__init__(lane, icon_path="direction_straight.png", **kwargs)


class LaneAllowedDirectionSign4_1_2(LaneAllowedDirectionSign):
    ALLOWED_DIRS = frozenset({"r"})

    def __init__(self, lane, **kwargs):
        super().__init__(lane, icon_path="direction_right.png", **kwargs)


class LaneAllowedDirectionSign4_1_3(LaneAllowedDirectionSign):
    # Per PDD, signs permitting a left turn also permit a U-turn ("t").
    ALLOWED_DIRS = frozenset({"l", "t"})

    def __init__(self, lane, **kwargs):
        super().__init__(lane, icon_path="direction_left.png", **kwargs)


class LaneAllowedDirectionSign4_1_4(LaneAllowedDirectionSign):
    ALLOWED_DIRS = frozenset({"s", "r"})

    def __init__(self, lane, **kwargs):
        super().__init__(lane, icon_path="direction_straight_right.png", **kwargs)


class LaneAllowedDirectionSign4_1_5(LaneAllowedDirectionSign):
    ALLOWED_DIRS = frozenset({"s", "l", "t"})

    def __init__(self, lane, **kwargs):
        super().__init__(lane, icon_path="direction_straight_left.png", **kwargs)


class LaneAllowedDirectionSign4_1_6(LaneAllowedDirectionSign):
    ALLOWED_DIRS = frozenset({"l", "r", "t"})

    def __init__(self, lane, **kwargs):
        super().__init__(lane, icon_path="direction_left_right.png", **kwargs)
