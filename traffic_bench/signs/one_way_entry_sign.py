from traffic_bench.signs.base_traffic_sign import BaseTrafficSign
from traffic_bench.signs.sumo_outgoing import SumoOutgoingMixin, normalize_turn_direction


def _normalize_turn_direction(raw_dir: str) -> str:
    return normalize_turn_direction(raw_dir)


def _edge_id_from_lane_index(lane_index):
    if not isinstance(lane_index, str) or not lane_index.startswith("lane_"):
        return None
    parts = lane_index.split("_")
    if len(parts) < 3:
        return None
    return parts[1]


def _allowed_turn_dirs_for_sign(not_allowed_direction: str):
    blocked = _normalize_turn_direction(not_allowed_direction)
    if blocked == "l":
        # 5.7.1: forbid left entry only; straight/U-turn are allowed if present.
        return {"r", "s", "t"}
    if blocked == "r":
        # 5.7.2: forbid right entry only; straight/U-turn are allowed if present.
        return {"l", "s", "t"}
    if blocked == "t":
        # 5.5: U-turn is forbidden; left/right/straight are acceptable.
        return {"l", "r", "s"}
    return {"l", "r", "s", "t"}


class OneWayEntrySign(SumoOutgoingMixin, BaseTrafficSign):
    """
    Sign 5.7.1 / 5.7.2: entry to a one-way road from a multi-direction approach.

    ``not_allowed_direction`` is the blocked exit direction at the approach:
    ``'l'`` (left blocked → one-way to the right, 5.7.1) or
    ``'r'`` (right blocked → one-way to the left, 5.7.2).
    A violation is recorded if the vehicle leaves the approach lane by a turn
    whose direction matches the blocked direction.

    Exposes ``ALLOWED_DIRS`` / ``prohibited_maneuver`` so SignComplianceMixin
    can reuse the same SUMO dual-path replan path as NoLeftTurn / NoRightTurn.
    """

    # Overridden on subclasses; base defaults match 5.7.1 (left forbidden).
    ALLOWED_DIRS = frozenset({"r", "s", "t"})

    def __init__(self, lane, not_allowed_direction='l', icon_path="5.7.1.png", **kwargs):
        self._preset_applicable_lane_indices = kwargs.pop("applicable_lane_indices", None)
        super().__init__(lane=lane, icon_path=icon_path, **kwargs)
        self.not_allowed_direction = _normalize_turn_direction(not_allowed_direction)
        # Alias used by SignComplianceMixin / GIF overlays (same as No*TurnSign).
        self.prohibited_maneuver = self.not_allowed_direction
        self.active_agents = {}
        self._sumo_agent_states = {}
        self.applicable_lanes = self._collect_applicable_lanes()
        self.applicable_lane_ids = {getattr(l, "index", None) for l in self.applicable_lanes}
        self._sumo_outgoing_mapped = None

    def _collect_applicable_lanes(self):
        """Collect all approach lanes where this sign should be visible/active.

        We keep lanes from the same incoming junction and same approach edge as
        the seed lane, and require they support the allowed turn direction.
        """
        base_lane = self.lane
        base_idx = getattr(base_lane, "index", None)
        base_edge = _edge_id_from_lane_index(base_idx)
        allowed_dirs = set(self.ALLOWED_DIRS) or _allowed_turn_dirs_for_sign(
            self.not_allowed_direction
        )

        lanes = [base_lane]
        try:
            road_network = self.engine.current_map.road_network
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

        candidate_ids = set(getattr(base_lane, "incoming_junction_lanes", None) or [])
        if base_idx is not None:
            candidate_ids.add(base_idx)

        for lane_id in candidate_ids:
            if lane_id == base_idx:
                continue
            try:
                lane_obj = road_network.get_lane(lane_id)
            except Exception:
                continue
            if lane_obj is None:
                continue

            if base_edge is not None:
                if _edge_id_from_lane_index(getattr(lane_obj, "index", None)) != base_edge:
                    continue

            turns = getattr(lane_obj, "turns", None) or []
            dirs = {_normalize_turn_direction(t.get("direction")) for t in turns}
            if not (dirs & allowed_dirs):
                continue

            lanes.append(lane_obj)

        return lanes

    def _lane_for_id(self, lane_id):
        if lane_id is None:
            return None
        for lane in self.applicable_lanes:
            if getattr(lane, "index", None) == lane_id:
                return lane
        if getattr(self.lane, "index", None) == lane_id:
            return self.lane
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
    def _is_pre_junction_adjacent_lane_change(cls, src_lane_obj, current_lane_id) -> bool:
        if src_lane_obj is None or current_lane_id is None:
            return False
        incoming = set(getattr(src_lane_obj, "incoming_junction_lanes", None) or [])
        if not incoming or current_lane_id not in incoming:
            return False
        src_key, src_lane_num = cls._lane_index_parts(getattr(src_lane_obj, "index", None))
        cur_key, cur_lane_num = cls._lane_index_parts(current_lane_id)
        if src_key is None or cur_key is None or src_key != cur_key:
            return False
        if src_lane_num is None or cur_lane_num is None:
            return False
        return abs(src_lane_num - cur_lane_num) == 1

    def get_top_down_icon_poses(self):
        """Positions/headings for rendering this sign on all applicable lanes.

        Lateral offset matches physical placement (half lane width + 1.8 m
        shoulder) so the top-down icon sits off the pavement, not on it.
        """
        poses = []
        shoulder_m = 1.8
        offset_from_end = max(0.1, float(self.lane.length) - float(self.placement_long))
        for lane in self.applicable_lanes:
            try:
                lane_len = float(lane.length)
                place_long = min(max(0.1, lane_len - offset_from_end), lane_len - 0.1)
                lat = lane.width_at(place_long) / 2 + shoulder_m
                pos = lane.position(place_long, lat)
                heading = lane.heading_theta_at(place_long) + 3.141592653589793 / 2
                poses.append((pos, heading))
            except Exception:
                continue
        return poses

    def _is_neighbor_lane(self, lane_a, lane_b) -> bool:
        """True if two lane ids are adjacent lanes on the same SUMO edge."""
        key_a, num_a = self._lane_index_parts(lane_a)
        key_b, num_b = self._lane_index_parts(lane_b)
        if key_a is None or key_b is None or key_a != key_b:
            return False
        if num_a is None or num_b is None:
            return False
        return abs(num_a - num_b) == 1

    def _ensure_sumo_outgoing_context(self) -> None:
        if self._sumo_outgoing_mapped and self._sumo_outgoing_mapped.get("all_outgoing"):
            return
        mapped = self._map_sumo_outgoing_from_lanes(self.applicable_lanes)
        forbidden = self._forbidden_outgoing_edges(mapped, self.not_allowed_direction)
        extra = getattr(self, "one_way_forbidden_edges", None) or ()
        forbidden |= {str(e) for e in extra}
        all_outgoing = set(mapped.get("all_outgoing") or ()) | forbidden
        mapped["all_outgoing"] = all_outgoing
        mapped["forbidden"] = forbidden
        self._sumo_outgoing_mapped = mapped

    def _is_violating_lane_targets(self, vehicle) -> bool:
        agent_id = vehicle.name
        current_lane = vehicle.lane_index
        prohibited = self.not_allowed_direction

        if current_lane in self.applicable_lane_ids:
            self.active_agents[agent_id] = current_lane
            return False

        if agent_id not in self.active_agents:
            return False

        prev_lane = self.active_agents[agent_id]
        if prev_lane not in self.applicable_lane_ids or current_lane == prev_lane:
            return False

        if self._is_neighbor_lane(prev_lane, current_lane):
            self.active_agents.pop(agent_id, None)
            return False

        src_lane_obj = self._lane_for_id(prev_lane)
        if src_lane_obj is None:
            self.active_agents.pop(agent_id, None)
            return False

        turn_info = next(
            (
                turn
                for turn in (getattr(src_lane_obj, "turns", None) or [])
                if (
                    turn.get("to_lane") == current_lane
                    or turn.get("via_lane") == current_lane
                )
            ),
            None,
        )
        self.active_agents.pop(agent_id, None)
        if turn_info and _normalize_turn_direction(turn_info.get("direction")) == prohibited:
            return True
        return False

    def _is_violating(self, vehicle) -> bool:
        """Same SUMO check as ``NoLeftTurnSign`` / ``NoRightTurnSign``.

        5.7.1 ≡ no-left (forbidden ``l``); 5.7.2 ≡ no-right (forbidden ``r``).
        Judge the first real outgoing *edge* after the signed approach, not the
        exact ``to_lane`` / ``via_lane`` ids (NN policies often land on a
        sibling lane of the same road).
        """
        if self._is_sumo_network():
            self._ensure_sumo_outgoing_context()
            mapped = self._sumo_outgoing_mapped or {}
            extra = {str(e) for e in (getattr(self, "one_way_forbidden_edges", None) or ())}
            forbidden = set(mapped.get("forbidden") or ()) | extra
            all_outgoing = set(mapped.get("all_outgoing") or ()) | forbidden
            if all_outgoing:
                return self._judge_sumo_outgoing(
                    vehicle.name,
                    vehicle.lane_index,
                    approach_roads=mapped.get("approach_roads") or set(),
                    all_outgoing=all_outgoing,
                    violate_roads=forbidden,
                    states=self._sumo_agent_states,
                )
        return self._is_violating_lane_targets(vehicle)

    def get_rule_description(self) -> str:
        side = "right" if self.not_allowed_direction == "l" else "left"
        return f"Exit onto a one-way road, turn {side}."
    
class OneWayEntrySignL(OneWayEntrySign):
    """5.7.2 — exit onto one-way road to the left (right turn blocked)."""

    ALLOWED_DIRS = frozenset({"l", "s", "t"})

    def __init__(self, lane, **kwargs):
        super().__init__(lane, not_allowed_direction='r', icon_path="5.7.2.png", **kwargs)


class OneWayEntrySignR(OneWayEntrySign):
    """5.7.1 — exit onto one-way road to the right (left turn blocked)."""

    ALLOWED_DIRS = frozenset({"r", "s", "t"})

    def __init__(self, lane, **kwargs):
        super().__init__(lane, not_allowed_direction='l', icon_path="5.7.1.png", **kwargs)


class OneWayEntrySignS(OneWayEntrySign):
    """5.5 — one-way road ahead (U-turn blocked); kept for catalog compatibility."""

    ALLOWED_DIRS = frozenset({"l", "r", "s"})

    def __init__(self, lane, **kwargs):
        super().__init__(lane, not_allowed_direction='t', icon_path="5.5.png", **kwargs)
