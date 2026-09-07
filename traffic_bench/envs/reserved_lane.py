"""Reserved-lane users for the 5.11.x / 5.14.x scenes: buses or cyclists that own the lane.

Regular background cars never use the reserved lane (``SumoTrafficManager`` keeps
it out of its spawn ladder and ``NPCIDMPolicy`` reroutes around it). What drives
on it are the vehicles the plate is for:

* 5.14.1 / 5.14.2 (``flow="same"``): buses / cyclists moving with the ego;
* 5.11.1 / 5.11.2 (``flow="opposite"``): the lane is a counter-flow lane on a
  one-way street, so the buses / cyclists come towards the ego.

They are moved kinematically along the lane centreline at a constant speed and
wrap around at the lane end, so the lane is occupied for the whole episode. They
have physics bodies: an ego that stays on the lane collides with them.

Engine config (all set by ``eval/run/env.py`` from the manifest row):

    reserved_lane_edge    SUMO edge id of the plate's road ("" = manager idle)
    reserved_lane_index   lane number of the reserved lane on that edge
    reserved_lane_flow    "same" | "opposite"
    reserved_lane_user    "bus" | "bicycle"
    reserved_zone_start   plate longitude on the lane (m)
    reserved_zone_end     end of the reserved zone (m)
    reserved_ego_s        ego spawn longitude on the lane (m); kept clear at reset
    reserved_agents_n     how many users share the lane (default 3)
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from metadrive.manager.base_manager import BaseManager

BUS_SPEED_MS = 8.5          # ~30 km/h, the pace of a city bus in its own lane
BICYCLE_SPEED_MS = 4.5      # ~16 km/h
SPEED_JITTER = 0.15         # ± share of the nominal speed, per agent
EGO_CLEAR_BEHIND_M = 60.0   # no user this close behind the ego spawn (same flow)
EGO_CLEAR_AHEAD_M = 30.0    # no user this close ahead of the ego spawn
# Counter-flow (5.11.x): the users come as a stream from the far end of the
# edge. The first one is released so that a car that stays on the lane meets it
# in the middle of the zone, i.e. after its violation has been registered, not on
# the run-up where nobody has had a chance to change lane yet.
OPPOSITE_FIRST_MEET_IN_ZONE_M = 30.0
OPPOSITE_EGO_SPEED_MS = 10.0     # assumed ego pace for the release timing
OPPOSITE_HEADWAY_S = 6.0         # gap between successive users
PARK_OFFSET_M = 5000.0           # where a not-yet-released user waits
# Lateral offset of the users towards the far side of their lane: a 2.3 m wide
# body on a 3.2 m lane leaves 0.45 m to the line, and the IDM leader test of the
# neighbouring lane fires on any corner that touches it.
LATERAL_OFFSET_M = 0.6
# Same flow: the first user waits this far past the plate, so a car that keeps
# the lane enters the zone and is scored before it meets anyone.
SAME_FLOW_START_AFTER_PLATE_M = 25.0
# A user (re)enters the lane only where no vehicle is within this distance.
RESPAWN_CLEAR_M = 20.0


class ReservedLaneAgentManager(BaseManager):
    """Spawns and drives the buses / cyclists of a reserved lane."""

    PRIORITY = 15  # after the map (SumoMapManager) and the traffic manager

    def __init__(self):
        super().__init__()
        self._agents: List[dict] = []
        self._lane = None
        self._dt = 0.1
        self._opp_spacing = 50.0
        self._opp_exit_s = 1.0
        self._lat = 0.0
        self._same_start = 1.0

    # ------------------------------------------------------------------ helpers

    @property
    def enabled(self) -> bool:
        return bool(str(self.engine.global_config.get("reserved_lane_edge", "") or ""))

    @property
    def agents(self):
        return [a["obj"] for a in self._agents]

    def _resolve_lane(self):
        cfg = self.engine.global_config
        edge = str(cfg.get("reserved_lane_edge", "") or "")
        idx = int(cfg.get("reserved_lane_index", 0) or 0)
        rn = self.engine.current_map.road_network
        for key in (f"lane_{edge}_{idx}", f"{edge}_{idx}"):
            try:
                lane = rn.get_lane(key)
                if lane is not None:
                    return lane
            except Exception:
                continue
        return None

    def _away_from_neighbour_offset(self) -> float:
        """+/-LATERAL_OFFSET_M, whichever moves the body away from the neighbouring lane."""
        lane = self._lane
        cfg = self.engine.global_config
        edge = str(cfg.get("reserved_lane_edge", "") or "")
        idx = int(cfg.get("reserved_lane_index", 0) or 0)
        rn = self.engine.current_map.road_network
        neighbour = None
        for cand in (idx - 1, idx + 1):
            if cand < 0:
                continue
            for key in (f"lane_{edge}_{cand}", f"{edge}_{cand}"):
                try:
                    neighbour = rn.get_lane(key)
                except Exception:
                    neighbour = None
                if neighbour is not None:
                    break
            if neighbour is not None:
                break
        if neighbour is None:
            return 0.0
        mid = float(lane.length) / 2.0
        nb = neighbour.position(min(mid, float(neighbour.length) - 1.0), 0.0)
        best, best_d = 0.0, -1.0
        for off in (LATERAL_OFFSET_M, -LATERAL_OFFSET_M):
            p = lane.position(mid, off)
            d = math.hypot(float(p[0]) - float(nb[0]), float(p[1]) - float(nb[1]))
            if d > best_d:
                best, best_d = off, d
        return best

    def _spawn_bus(self, s: float, heading: float, name_hint: str):
        from metadrive.component.vehicle.vehicle_type import XLVehicle

        cfg = dict(self.engine.global_config.get("traffic_vehicle_config", {}) or {})
        cfg.update(
            {
                "spawn_lane_index": str(self.engine.global_config.get("reserved_lane_edge", "")),
                "spawn_longitude": float(s),
                "destination": None,
                "show_navi_mark": False,
                "show_dest_mark": False,
                "show_line_to_dest": False,
            }
        )
        obj = self.spawn_object(XLVehicle, vehicle_config=cfg, force_spawn=True)
        obj.is_bus = True
        return obj

    def _spawn_cyclist(self, position, heading: float):
        from metadrive.component.traffic_participants.cyclist import Cyclist

        obj = self.spawn_object(
            Cyclist,
            position=[float(position[0]), float(position[1])],
            heading_theta=float(heading),
            force_spawn=True,
        )
        obj.is_cyclist = True
        return obj

    def _spot_clear(self, s: float) -> bool:
        """No vehicle (background car or ego) within RESPAWN_CLEAR_M of lane point ``s``."""
        try:
            p = self._lane.position(min(max(s, 0.5), float(self._lane.length) - 0.5), self._lat)
            px, py = float(p[0]), float(p[1])
            others = []
            tm = getattr(self.engine, "traffic_manager", None)
            others += list(getattr(tm, "traffic_vehicles", None) or [])
            agents = getattr(getattr(self.engine, "agent_manager", None), "active_agents", None) or {}
            others += list(agents.values())
            for v in others:
                q = v.position
                if (float(q[0]) - px) ** 2 + (float(q[1]) - py) ** 2 < RESPAWN_CLEAR_M ** 2:
                    return False
        except Exception:
            return True
        return True

    def _place(self, agent: dict) -> None:
        lane = self._lane
        if agent["s"] > float(lane.length) - 0.5:
            # Not released yet: park far away from the scene, standing still.
            pos = lane.position(float(lane.length) - 0.5, 0.0)
            try:
                agent["obj"].set_position([float(pos[0]) + PARK_OFFSET_M, float(pos[1]) + PARK_OFFSET_M])
                agent["obj"].set_velocity([0.0, 0.0], in_local_frame=True)
            except Exception:
                pass
            return
        s = min(max(agent["s"], 0.5), float(lane.length) - 0.5)
        pos = lane.position(s, self._lat)
        heading = float(lane.heading_theta_at(s))
        if agent["dir"] < 0:
            heading += math.pi
        obj = agent["obj"]
        try:
            obj.set_position([float(pos[0]), float(pos[1])])
            obj.set_heading_theta(heading)
            obj.set_velocity([float(agent["v"]), 0.0], in_local_frame=True)
        except Exception:
            pass

    # ------------------------------------------------------------------ lifecycle

    def before_reset(self):
        self._clear()

    def reset(self):
        self._clear()
        if not self.enabled:
            return
        cfg = self.engine.global_config
        self._lane = self._resolve_lane()
        if self._lane is None:
            print(f"[ReservedLane] lane {cfg.get('reserved_lane_edge')}_{cfg.get('reserved_lane_index')} not found")
            return
        self._dt = float(cfg.get("physics_world_step_size", 0.02)) * int(cfg.get("decision_repeat", 5))
        self._lat = self._away_from_neighbour_offset()
        flow = str(cfg.get("reserved_lane_flow", "same") or "same")
        user = str(cfg.get("reserved_lane_user", "bus") or "bus")
        n = int(cfg.get("reserved_agents_n", 0) or 0)
        if n <= 0:
            n = 3 if user == "bicycle" else 2
        direction = -1 if flow == "opposite" else 1
        base_v = BICYCLE_SPEED_MS if user == "bicycle" else BUS_SPEED_MS
        length = float(self._lane.length)
        ego_s = float(cfg.get("reserved_ego_s", -1.0))

        zone_start = float(cfg.get("reserved_zone_start", 0.0) or 0.0)
        # The lane is reserved from the plate on. Same flow: the users are
        # spread over the stretch past the plate (random phase) and wrap back
        # to the plate at the edge end; before the plate the lane is ordinary.
        # Counter flow: a stream from the far end, first user timed to meet a
        # lane-keeping ego inside the zone, turning off at the plate.
        self._same_start = min(max(1.0, zone_start + SAME_FLOW_START_AFTER_PLATE_M), length - 2.0)
        stretch = max(5.0, length - self._same_start)
        spacing = stretch / n
        phase = float(self.np_random.uniform(0.0, spacing))
        if direction < 0 and ego_s >= 0.0:
            # The first user must be at meet_point when the ego gets there: it
            # starts base_v * t_ego before that point, usually beyond the edge
            # end, where it waits parked until its longitude enters the lane.
            meet_point = zone_start + OPPOSITE_FIRST_MEET_IN_ZONE_M
            t_ego = max(0.0, (meet_point - ego_s) / OPPOSITE_EGO_SPEED_MS)
            first_s = meet_point + base_v * t_ego
            self._opp_spacing = base_v * OPPOSITE_HEADWAY_S
            self._opp_exit_s = max(1.0, zone_start - 2.0)
        spawned = 0
        for k in range(n):
            if direction < 0 and ego_s >= 0.0:
                s = first_s + k * self._opp_spacing
            else:
                s = self._same_start + (phase + k * spacing) % stretch
            v = base_v * float(self.np_random.uniform(1.0 - SPEED_JITTER, 1.0 + SPEED_JITTER))
            heading = float(self._lane.heading_theta_at(s)) + (math.pi if direction < 0 else 0.0)
            try:
                if user == "bicycle":
                    obj = self._spawn_cyclist(self._lane.position(s, 0.0), heading)
                else:
                    obj = self._spawn_bus(s, heading, f"bus{k}")
            except Exception as exc:  # noqa: BLE001
                print(f"[ReservedLane] spawn failed: {exc!r}")
                continue
            obj._trb_reserved_agent = True
            obj._trb_reserved_lane_key = str(getattr(self._lane, "index", ""))
            agent = {"obj": obj, "s": s, "v": v, "dir": direction}
            self._agents.append(agent)
            self._place(agent)
            spawned += 1
        print(
            f"[ReservedLane] {spawned} {user}(s) on lane {getattr(self._lane, 'index', '?')} "
            f"flow={flow} v={base_v:.1f} m/s"
        )

    def before_step(self):
        if not self._agents or self._lane is None:
            return {}
        length = float(self._lane.length)
        for agent in self._agents:
            was_parked = agent["s"] > length - 0.5
            agent["s"] += agent["dir"] * agent["v"] * self._dt
            if agent["dir"] > 0 and agent["s"] > length - 1.0:
                # Back to the start of the reserved stretch, but only into a clear
                # spot: a user teleported onto a car is a crash nobody caused.
                if self._spot_clear(self._same_start):
                    agent["s"] = self._same_start
                else:
                    agent["s"] = length - 1.0   # hold at the end until the spot clears
            elif agent["dir"] < 0 and agent["s"] < self._opp_exit_s:
                # The counter-flow lane starts at the plate: at the plate the user
                # turns off. Back into the stream one headway behind the last user.
                agent["s"] = max(length + 1.0, max(a["s"] for a in self._agents) + getattr(self, "_opp_spacing", 50.0))
            elif agent["dir"] < 0 and was_parked and agent["s"] <= length - 0.5 and not self._spot_clear(length - 0.5):
                agent["s"] = length + 0.5      # release point occupied: wait one more step
            self._place(agent)
        return {}

    def _clear(self):
        if self._agents:
            try:
                self.clear_objects([a["obj"].id for a in self._agents])
            except Exception:
                pass
        self._agents = []
        self._lane = None

    def destroy(self):
        self._clear()
        super().destroy()
