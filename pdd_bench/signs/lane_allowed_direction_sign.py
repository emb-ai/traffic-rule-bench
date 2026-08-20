from pdd_bench.signs.base_traffic_sign import BaseTrafficSign
from pdd_bench.signs.sumo_outgoing import SumoOutgoingMixin, normalize_turn_direction


def _normalize_turn_direction(raw_dir: str) -> str:
    return normalize_turn_direction(raw_dir)


_CARDINAL_DIRS = frozenset({"l", "r", "s"})


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
        # lane_id -> last seen approach lane while agent still under this sign
        self.active_agents = {}
        self._sumo_agent_states = {}
        # Agents whose *first* departure from the signed approach was already
        # judged. Dual-path compliant routes (esp. 4.1.2) often loop back onto
        # the same approach and continue with a different exit; that second
        # pass must not re-trigger the rule.
        self._cleared_agents = set()
        super().__init__(lane=lane, **kwargs)
        self.applicable_lanes = self._collect_applicable_lanes()
        self.applicable_lane_ids = {getattr(l, "index", None) for l in self.applicable_lanes}
        self.allowed_lanes_by_source = self._build_allowed_targets()
        self._sumo_outgoing_mapped = None

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

        candidate_ids = set(getattr(self.lane, "incoming_junction_lanes", None) or [])
        lane_idx = getattr(self.lane, "index", None)
        if lane_idx is not None:
            candidate_ids.add(lane_idx)

        for lane_id in candidate_ids:
            if lane_id == lane_idx:
                continue
            try:
                lane_obj = road_network.get_lane(lane_id)
            except Exception:
                continue
            if lane_obj is None:
                continue
            dirs = {
                _normalize_turn_direction(t.get("direction"))
                for t in (getattr(lane_obj, "turns", None) or [])
                if _normalize_turn_direction(t.get("direction")) in _CARDINAL_DIRS
            }
            # Compare cardinal directions only: "t" (U-turn) is an enforcement
            # extra for left-turn signs, never part of lane turn matching.
            if dirs == (set(self.ALLOWED_DIRS) & _CARDINAL_DIRS):
                lanes.append(lane_obj)
        return lanes

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
                via_lane = turn.get("via_lane")
                if to_lane:
                    allowed.add(to_lane)
                    to_lane_obj = self.engine.current_map.road_network.get_lane(to_lane)
                    next_lanes = set(getattr(to_lane_obj, "exit_lanes", None) or [])
                    for next_lane in next_lanes:
                        allowed.add(next_lane)
                        if ":" in next_lane:
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
        poses = []
        offset_from_end = max(0.1, float(self.lane.length) - float(self.placement_long))
        for lane in self.applicable_lanes:
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

    def _ensure_sumo_outgoing_context(self) -> None:
        if self._sumo_outgoing_mapped and self._sumo_outgoing_mapped.get("all_outgoing"):
            return
        self._sumo_outgoing_mapped = self._map_sumo_outgoing_from_lanes(self.applicable_lanes)

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
                allowed = self._allowed_outgoing_edges(mapped, self.ALLOWED_DIRS)
                return self._judge_sumo_outgoing(
                    vehicle.name,
                    vehicle.lane_index,
                    approach_roads=mapped.get("approach_roads") or set(),
                    all_outgoing=all_outgoing,
                    violate_roads=all_outgoing - allowed,
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
        super().__init__(lane, icon_path="4.1.1.png", **kwargs)
        
class LaneAllowedDirectionSign4_1_2(LaneAllowedDirectionSign):
    ALLOWED_DIRS = frozenset({"r"})

    def __init__(self, lane, **kwargs):
        super().__init__(lane, icon_path="4.1.2.png", **kwargs)  
        
class LaneAllowedDirectionSign4_1_3(LaneAllowedDirectionSign):
    # Per PDD, signs permitting a left turn also permit a U-turn ("t").
    ALLOWED_DIRS = frozenset({"l", "t"})

    def __init__(self, lane, **kwargs):
        super().__init__(lane, icon_path="4.1.3.png", **kwargs)  
        
class LaneAllowedDirectionSign4_1_4(LaneAllowedDirectionSign):
    ALLOWED_DIRS = frozenset({"s", "r"})

    def __init__(self, lane, **kwargs):
        super().__init__(lane, icon_path="4.1.4.png", **kwargs)  
        
class LaneAllowedDirectionSign4_1_5(LaneAllowedDirectionSign):
    ALLOWED_DIRS = frozenset({"s", "l", "t"})

    def __init__(self, lane, **kwargs):
        super().__init__(lane, icon_path="4.1.5.png", **kwargs)  
        
class LaneAllowedDirectionSign4_1_6(LaneAllowedDirectionSign):
    ALLOWED_DIRS = frozenset({"l", "r", "t"})

    def __init__(self, lane, **kwargs):
        super().__init__(lane, icon_path="4.1.6.png", **kwargs)  
