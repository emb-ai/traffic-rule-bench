"""Sign 5.15.1 — directions of movement by lanes (board over a multi-lane approach).

Per-lane allowed exits come from SUMO ``lane.turns`` (same ground truth as
pavement arrows / 5.15.2). The sign covers the whole approach: peer lanes share
one board. Leaving an approach lane into a target that is not among that lane's
turn targets is a violation. Pre-junction peer lane-changes are allowed.

For 5.15.1 benchmark scenes with injected connectors, the sign uses the
ORIGINAL allowed exits stored in ``original_allowed_exits_by_lane`` (before
connector injection) to detect violations.
"""

from __future__ import annotations

from traffic_signs.direction_sign import DirectionSign


class LaneDirectionsSign(DirectionSign):
    """PDD 5.15.1 — multi-lane approach board of allowed directions."""

    def __init__(self, lane, **kwargs):
        self._preset_applicable_lane_indices = kwargs.pop("applicable_lane_indices", None)
        # Original allowed exits from meta (before connector injection for baseline).
        # Format: {lane_index_str: {dir: [to_edge, ...], ...}, ...}
        self._original_allowed_exits = kwargs.pop("original_allowed_exits_by_lane", None)
        # Never show a board icon for 5.15.1 (top-down or 3D).
        kwargs.pop("icon_path", None)
        kwargs["show_model"] = False
        # DirectionSign.__init__ calls BaseTrafficSign.__init__
        super().__init__(lane, **kwargs)
        self.icon_path = None
        self.applicable_lanes = self._collect_applicable_lanes()
        self.applicable_lane_ids = {
            getattr(l, "index", None) for l in self.applicable_lanes if l is not None
        }
        self.allowed_lanes_by_source = {
            getattr(l, "index", None): self._build_allowed_targets_for_sign(l)
            for l in self.applicable_lanes
            if getattr(l, "index", None) is not None
        }
        # Any peer with turn metadata makes the board enforceable.
        self._has_turn_metadata = any(
            bool(getattr(l, "turns", None)) for l in self.applicable_lanes
        )

    def _build_allowed_targets_for_sign(self, lane):
        """Build allowed targets using ORIGINAL exits when available."""
        lane_id = getattr(lane, "index", None)
        if lane_id is None:
            return self._build_allowed_targets(lane)

        # If original allowed exits were provided, use those (pre-injection truth).
        if self._original_allowed_exits:
            # Extract lane number from lane_id (e.g. "lane_376380753_1" -> "1")
            try:
                lane_num_str = str(lane_id).rsplit("_", 1)[-1]
                original_exits = self._original_allowed_exits.get(lane_num_str, {})
                if original_exits:
                    # Build allowed set from original exits (only first-hop edges).
                    allowed = set()
                    road_network = self.engine.current_map.road_network
                    for direction, to_edges in original_exits.items():
                        for to_edge in to_edges:
                            # Find lane(s) on the to_edge that this lane can reach.
                            for turn in (getattr(lane, "turns", None) or []):
                                to_lane = turn.get("to_lane", "")
                                # Match if to_lane is on to_edge (format: lane_<edge>_<num>).
                                if f"_{to_edge}_" in to_lane or to_lane.endswith(f"_{to_edge}_0"):
                                    allowed.add(to_lane)
                                    # Also add exit lanes of connector.
                                    to_lane_obj = road_network.get_lane(to_lane)
                                    for exit_l in (getattr(to_lane_obj, "exit_lanes", None) or []):
                                        allowed.add(exit_l)
                                        if ":" in exit_l:
                                            exit_obj = road_network.get_lane(exit_l)
                                            for exit2 in (getattr(exit_obj, "exit_lanes", None) or []):
                                                allowed.add(exit2)
                    if allowed:
                        return allowed
            except Exception:
                pass

        # Fallback to current lane.turns (standard behavior).
        return self._build_allowed_targets(lane)

    def _collect_applicable_lanes(self):
        lanes = [self.lane]
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

        # All peer lanes on the signed approach edge.
        try:
            peers = road_network.get_peer_lanes_from_index(self.lane.index) or []
        except Exception:
            peers = []
        out = []
        seen = set()
        for lane_obj in [self.lane, *peers]:
            idx = getattr(lane_obj, "index", None)
            if idx is None or idx in seen:
                continue
            seen.add(idx)
            out.append(lane_obj)
        return out or lanes

    def _is_violating(self, vehicle) -> bool:
        if not self._has_turn_metadata:
            return False

        agent_id = vehicle.name
        current_lane = vehicle.lane_index

        if "lane_:" in str(current_lane) or "junction" in str(current_lane):
            return False

        if agent_id not in self.active_agents:
            # Arm only once the vehicle is on the signed approach.
            if current_lane in self.applicable_lane_ids:
                self.active_agents[agent_id] = current_lane
            return False

        prev_lane = self.active_agents[agent_id]
        if current_lane == prev_lane:
            return False

        src_lane_obj = self.engine.current_map.road_network.get_lane(prev_lane)
        # Peer lane-change on the approach is required for this skill.
        if self._is_pre_junction_lane_change(src_lane_obj, current_lane):
            self.active_agents[agent_id] = current_lane
            return False

        self.active_agents.pop(agent_id, None)
        # Only judge departures that started on the signed approach.
        if prev_lane not in self.applicable_lane_ids:
            return False
        allowed = self.allowed_lanes_by_source.get(prev_lane) or self._build_allowed_targets(
            src_lane_obj
        )
        return current_lane not in allowed

    def get_rule_description(self) -> str:
        return (
            "Vehicle violated lane-direction board (5.15.1): left an approach "
            "lane into a disallowed direction for that lane."
        )

    def get_top_down_icon_poses(self):
        # Never draw a top-down board icon for 5.15.1 (pavement arrows only).
        return []

    @property
    def top_down_length(self):
        return 0

    @property
    def top_down_width(self):
        return 0


# Alias kept for registry clarity.
LaneDirectionsSign5_15_1 = LaneDirectionsSign
