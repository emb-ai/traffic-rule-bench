from pdd_bench.signs.base_traffic_sign import BaseTrafficSign


class DirectionSign(BaseTrafficSign):
    """Sign 5.15.2 — allowed lane directions."""
    
    def __init__(self, lane, **kwargs):
        self.lane = lane
        turns = list(getattr(lane, 'turns', []) or [])
        self._has_turn_metadata = bool(turns)
        self.active_agents = {}
        self._trap_lane_id = kwargs.pop("trap_lane_id", None)
        super().__init__(lane=lane, **kwargs)

    @property
    def allowed_to_lanes(self):
        if not hasattr(self, '_allowed_cache'):
            self._allowed_cache = self._build_allowed_targets(self.lane)
        return self._allowed_cache

    def _build_allowed_targets(self, lane):
        lane_id = getattr(lane, "index", None)
        allowed = set()
        for turn in (getattr(lane, "turns", None) or []):
            to_lane = turn.get("to_lane")
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

        return allowed

    @staticmethod
    def _is_internal_lane(lane_id):
        """True for SUMO internal / via lanes (IDs like ``lane_:node_0_0``)."""
        if not isinstance(lane_id, str):
            return False
        raw = lane_id[5:] if lane_id.startswith("lane_") else lane_id
        return ":" in raw

    def _is_violating(self, vehicle) -> bool:
        if not self._has_turn_metadata:
            return False

        current_lane = vehicle.lane_index

        if "lane_:" in current_lane or "junction" in current_lane:
            return False

        agent_id = vehicle.name
        if agent_id not in self.active_agents:
            self.active_agents[agent_id] = current_lane
            return False

        prev_lane = self.active_agents[agent_id]

        if current_lane != prev_lane:
            src_lane_obj = self.engine.current_map.road_network.get_lane(prev_lane)
            if self._is_pre_junction_lane_change(src_lane_obj, current_lane):
                self.active_agents[agent_id] = current_lane
                return False
            self.active_agents.pop(agent_id, None)
            allowed_to_lanes = self._build_allowed_targets(src_lane_obj)
            if current_lane not in allowed_to_lanes:
                return True
            return False

        return False

    # ------------------------------------------------------------------
    def get_rule_description(self) -> str:
        return "Vehicle violated direction sign: turned into a disallowed lane."

    @property
    def top_down_length(self):
        return 0

    @property
    def top_down_width(self):
        return 0