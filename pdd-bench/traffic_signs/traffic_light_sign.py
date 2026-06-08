from traffic_signs.base_traffic_sign import BaseTrafficSign, same_road_check

class TrafficLightSign(BaseTrafficSign):
    ZONE_BEFORE = 30.0  # meters before lane end to start zone

    def __init__(self, lane, sim_step_duration=0.1, tl_speed_factor=1.0, **kwargs):
        self.lane = lane
        self.tl_signals = getattr(lane, 'tl_signals', [])
        self.signals_map = {sig['to_lane']: sig for sig in self.tl_signals}
        self.active_agents = {}
        self.sim_step_duration = sim_step_duration
        self.tl_speed_factor = tl_speed_factor

        self.current_states = {}  # to_lane -> current state

        # Define zone for violation checking
        self.zone_start = max(0.0, lane.length - self.ZONE_BEFORE)
        self.zone_end = lane.length

        super().__init__(lane=lane, **kwargs)

    def update_state(self):
        """Update traffic light states based on current simulation time"""
        self._update_current_states()
    
    def _update_current_states(self):
        """Update current state for all target lanes"""
        for to_lane, signal_info in self.signals_map.items():
            phases = signal_info.get('phases', [])
            if phases:
                current_state = self._get_current_state(phases)
                self.current_states[to_lane] = current_state
            else:
                self.current_states[to_lane] = 'O'  # off

    def _is_violating(self, vehicle) -> bool:
        agent_id = vehicle.name
        current_lane = getattr(vehicle, "lane_index", None)
        veh_lane = getattr(vehicle, "lane", None)
        on_sign_road = veh_lane is not None and same_road_check(
            getattr(veh_lane, "index", None),
            getattr(self.lane, "index", None),
        )

        if on_sign_road:
            try:
                veh_long = self.lane.local_coordinates(vehicle.position)[0]
            except Exception:
                return False

            in_zone = self.zone_start <= veh_long <= self.zone_end
            if in_zone:
                self.active_agents[agent_id] = {
                    'entry_time': self.engine.episode_step,
                    'entry_lane': self.lane.index
                }
                return False

        if agent_id in self.active_agents:
            if current_lane == self.active_agents[agent_id]['entry_lane']:
                return False
            prev_info = self.active_agents.pop(agent_id)
            
            to_lane = current_lane
            if to_lane in self.signals_map:
                state = self.current_states.get(to_lane, 'O')
                if state not in ('g', 'G'):
                    return True
            return False

        return False

    def _get_current_state(self, phases):
        """Return current signal phase from global simulation time."""
        if not phases:
            return 'O'

        cycle_time = sum(dur for _, dur in phases)
        if cycle_time == 0:
            return phases[0][0]
        current_time = (float(self.engine.episode_step) * self.sim_step_duration * float(self.tl_speed_factor)) % cycle_time
        accumulated = 0
        for state, dur in phases:
            if current_time < accumulated + dur:
                return state
            accumulated += dur
        return phases[0][0]

    def get_rule_description(self) -> str:
        return "Vehicle violated traffic light: passed on red."

    # @property
    # def top_down_length(self):
    #     return 0

    # @property
    # def top_down_width(self):
    #     return 0

    @staticmethod
    def _state_to_color(state):
        if state == 'r':
            return [255, 0, 0]
        elif state in ('y', 'u'):
            return [255, 255, 0]
        elif state in ('g', 'G', 's'):
            return [0, 255, 0]
        return [180, 180, 180]

    @property
    def top_down_color(self):
        if not self.current_states:
            return [255, 255, 255]
        # Show color for directions reachable from THIS lane only
        own_to_lanes = set()
        for turn in getattr(self.lane, "turns", []):
            to = turn.get("to_lane")
            if to:
                own_to_lanes.add(to)
        if own_to_lanes:
            states = [s for to_lane, s in self.current_states.items()
                      if to_lane in own_to_lanes]
        else:
            states = list(self.current_states.values())
        if not states:
            states = list(self.current_states.values())
        priority = {'r': 3, 'y': 2, 'u': 2, 'g': 1, 'G': 1, 's': 1}
        max_priority = 0
        chosen_color = [255, 255, 255]
        for state in states:
            p = priority.get(state, 0)
            if p > max_priority:
                max_priority = p
                chosen_color = self._state_to_color(state)
        return chosen_color

    def get_per_direction_info(self):
        """Return list of (direction_label, color) ordered: left, straight, right.
        direction_label is 's' (straight), 'r' (right), 'l' (left), 't' (u-turn).
        """
        if not self.current_states:
            return []
        turns = getattr(self.lane, "turns", [])
        to_lane_direction = {}
        for turn in turns:
            to = turn.get("to_lane")
            d = turn.get("direction", "?")
            if to:
                to_lane_direction[to] = d
        result = []
        for to_lane, state in self.current_states.items():
            direction = to_lane_direction.get(to_lane, "?")
            color = self._state_to_color(state)
            result.append((direction, color))
        # Order: left/L first, then straight/u-turn, then right/R
        order = {'l': 0, 'L': 0, 't': 1, 's': 2, '?': 3, 'r': 4, 'R': 4}
        result.sort(key=lambda x: order.get(x[0], 3))
        return result