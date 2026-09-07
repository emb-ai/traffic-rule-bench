"""Reserved-lane users for the 5.11.x / 5.14.x scenes: buses or cyclists that own the lane.

Regular background cars never use the reserved lane (``SumoTrafficManager`` keeps
it out of its spawn ladder and ``NPCIDMPolicy`` reroutes around it). What drives
on it are the vehicles the plate is for:

* 5.14.1 / 5.14.2 (``flow="same"``): buses / cyclists moving with the ego, on
  the lane from the plate on (before the plate the lane is ordinary);
* 5.11.1 / 5.11.2 (``flow="opposite"``): the lane is a counter-flow lane on a
  one-way street, so the buses / cyclists come towards the ego and turn off at
  the plate.

They form a stream (``LaneStream``): one user every ``HEADWAY_S`` seconds at a
constant speed, no overtaking, re-entry only into a clear spot. They are moved
kinematically along the lane centreline and have physics bodies: an ego that
stays on the lane collides with them.

Engine config (all set by ``eval/run/env.py`` from the manifest row):

    reserved_lane_edge           SUMO edge id of the plate's road ("" = manager idle)
    reserved_lane_index          lane number of the reserved lane on that edge
    reserved_lane_flow           "same" | "opposite"
    reserved_lane_user           "bus" | "bicycle"
    reserved_zone_start          plate longitude on the SUMO edge (m)
    reserved_zone_end            end of the reserved zone (m)
    reserved_ego_s               ego spawn longitude on the SUMO edge (m)
    reserved_agents_n            how many users share the lane (default 1)
    reserved_ego_v0_ms           ego spawn speed (counter-flow release timing)
    reserved_sumo_edge_length_m  SUMO edge length; longitudes are remapped onto
                                 the MetaDrive lane when it is longer (stitched prefix)
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from metadrive.manager.base_manager import BaseManager

from traffic_bench.envs.lane_stream import LaneStream, ego_travel_time_s
from traffic_bench.eval.engine.map.sumo_metadrive_along import remap_sumo_along_to_metadrive

USER_SPEED_MS = {"bus": 8.5, "bicycle": 4.5}     # ~30 km/h city bus, ~16 km/h cyclist
SPEED_JITTER = 0.15                              # ± share of the nominal speed, per user
HEADWAY_S = {"bus": 6.0, "bicycle": 4.0}         # time gap between successive users
MIN_GAP_M = {"bus": 8.0, "bicycle": 3.0}         # bumper-to-bumper floor (leader clamp)
BODY_LEN_M = {"bus": 5.8, "bicycle": 1.75}       # XLVehicle / Cyclist
# Counter-flow (5.11.x): the first user is released so that a car that stays on
# the lane meets it this far inside the zone, i.e. after its violation has been
# registered, not on the run-up where nobody has had a chance to change lane yet.
OPPOSITE_FIRST_MEET_IN_ZONE_M = 30.0
OPPOSITE_EGO_ACCEL_MS2 = 1.5      # ego pace model for the release timing:
OPPOSITE_EGO_CRUISE_MS = 10.0     # spawn speed -> cruise at this acceleration
DEFAULT_EGO_V0_MS = 5.0
PARK_OFFSET_M = 5000.0            # where a parked user waits
PARK_SPACING_M = 15.0             # one parking spot per user (no stacked bodies)
# Lateral offset of the users towards the far side of their lane: a 2.3 m wide
# body on a 3.2 m lane leaves 0.45 m to the line, and the IDM leader test of the
# neighbouring lane fires on any corner that touches it.
LATERAL_OFFSET_M = 0.6
# Same flow: the stream starts this far past the plate, so a car that keeps the
# lane enters the zone and is scored before it meets anyone.
SAME_FLOW_START_AFTER_PLATE_M = 25.0
# A user (re)enters the lane only where no vehicle is within this distance on
# THIS lane (plus a time headway behind for moving cars).
RESPAWN_CLEAR_M = 20.0
CAR_CLEAR_HEADWAY_S = 2.5
LANE_HALF_WIDTH_MARGIN_M = 0.5


class ReservedLaneAgentManager(BaseManager):
    """Spawns and drives the buses / cyclists of a reserved lane."""

    PRIORITY = 15  # after the map (SumoMapManager) and the traffic manager

    def __init__(self):
        super().__init__()
        self._agents: List[dict] = []
        self._lane = None
        self._stream: Optional[LaneStream] = None
        self._dt = 0.1
        self._lat = 0.0
        self._entry_s = 0.0
        self._dir = 1
        self._user = "bus"
        self._delta_m = 0.0

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

    def _lane_s(self, k: int) -> float:
        return self._entry_s + self._dir * float(self._stream.r[k])

    def _spot_clear(self, s: float) -> bool:
        """No vehicle on THIS lane close to lane point ``s``.

        Cars and the ego count only when their centre is inside the lane (the
        compliant ego driving past in the neighbour lane must not hold the
        stream); a car approaching from behind the spot needs a time headway.
        Other active users are kept apart by the stream itself, the check here
        is a guard against a user held mid-lane by a car.
        """
        lane = self._lane
        try:
            half_w = float(getattr(lane, "width", 3.5)) / 2.0 + LANE_HALF_WIDTH_MARGIN_M
            others = []
            tm = getattr(self.engine, "traffic_manager", None)
            others += list(getattr(tm, "traffic_vehicles", None) or [])
            agents = getattr(getattr(self.engine, "agent_manager", None), "active_agents", None) or {}
            others += list(agents.values())
            for v in others:
                lon, lat = lane.local_coordinates(v.position)
                if abs(float(lat)) > half_w:
                    continue
                d = self._dir * (float(lon) - float(s))   # >0: ahead of the spot (travel dir)
                v_ms = float(getattr(v, "speed", 0.0) or 0.0)
                behind = max(RESPAWN_CLEAR_M, CAR_CLEAR_HEADWAY_S * v_ms + BODY_LEN_M[self._user])
                if -behind <= d <= RESPAWN_CLEAR_M:
                    return False
            if self._stream is not None:
                floor = BODY_LEN_M[self._user] + MIN_GAP_M[self._user]
                for k in self._stream.active_indices():
                    d = self._dir * (self._lane_s(k) - float(s))
                    if -floor <= d <= floor:
                        return False
        except Exception:
            return True
        return True

    def _place(self, agent: dict) -> None:
        lane = self._lane
        k = int(agent["k"])
        obj = agent["obj"]
        length = float(lane.length)
        if not self._stream.is_active(k):
            # Parked off-scene, standing still, one spot per user.
            pos = lane.position(length - 0.5, 0.0)
            try:
                obj.set_position(
                    [float(pos[0]) + PARK_OFFSET_M + k * PARK_SPACING_M, float(pos[1]) + PARK_OFFSET_M]
                )
                obj.set_velocity([0.0, 0.0], in_local_frame=True)
            except Exception:
                pass
            return
        s = min(max(self._lane_s(k), 0.5), length - 0.5)
        pos = lane.position(s, self._lat)
        heading = float(lane.heading_theta_at(s))
        if self._dir < 0:
            heading += math.pi
        try:
            obj.set_position([float(pos[0]), float(pos[1])])
            obj.set_heading_theta(heading)
            obj.set_velocity([float(self._stream.v[k]), 0.0], in_local_frame=True)
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
        lane = self._lane
        length = float(lane.length)
        self._dt = float(cfg.get("physics_world_step_size", 0.02)) * int(cfg.get("decision_repeat", 5))
        self._lat = self._away_from_neighbour_offset()
        flow = str(cfg.get("reserved_lane_flow", "same") or "same")
        user = str(cfg.get("reserved_lane_user", "bus") or "bus")
        if user not in USER_SPEED_MS:
            user = "bus"
        self._user = user
        n = int(cfg.get("reserved_agents_n", 0) or 0) or 1
        self._dir = -1 if flow == "opposite" else 1

        # Manifest longitudes are SUMO-edge metres; the MetaDrive lane may carry
        # a stitched prefix, in which case every mark shifts by the difference.
        sumo_len = float(cfg.get("reserved_sumo_edge_length_m", 0.0) or 0.0)

        def _md(s_sumo: float) -> float:
            return float(
                remap_sumo_along_to_metadrive(
                    float(s_sumo),
                    sumo_edge_length_m=sumo_len if sumo_len > 0.0 else None,
                    metadrive_lane_length_m=length,
                )
            )

        zone_start_raw = float(cfg.get("reserved_zone_start", 0.0) or 0.0)
        zone_start = _md(zone_start_raw)
        self._delta_m = zone_start - zone_start_raw
        ego_s_raw = float(cfg.get("reserved_ego_s", -1.0))
        ego_s = _md(ego_s_raw) if ego_s_raw >= 0.0 else -1.0

        v_nom = USER_SPEED_MS[user]
        body = BODY_LEN_M[user]
        headway_m = max(body + MIN_GAP_M[user], v_nom * HEADWAY_S[user])
        if self._dir > 0:
            entry_s = min(max(1.0, zone_start + SAME_FLOW_START_AFTER_PLATE_M), length - 2.0)
            exit_s = length - 1.0
        else:
            entry_s = length - 1.0
            exit_s = max(1.0, zone_start - 2.0)
        self._entry_s = entry_s
        stretch = max(1.0, abs(exit_s - entry_s))
        stream = LaneStream(
            n=n,
            v_nom_ms=v_nom,
            headway_m=headway_m,
            min_gap_m=MIN_GAP_M[user],
            body_len_m=body,
            stretch_m=stretch,
            speed_jitter=SPEED_JITTER,
            rng=self.np_random,
        )
        t_ego = None
        if self._dir < 0 and ego_s >= 0.0:
            meet = zone_start + OPPOSITE_FIRST_MEET_IN_ZONE_M
            v0 = float(cfg.get("reserved_ego_v0_ms", 0.0) or 0.0) or DEFAULT_EGO_V0_MS
            t_ego = ego_travel_time_s(
                meet - ego_s, v0, accel_ms2=OPPOSITE_EGO_ACCEL_MS2, cruise_ms=OPPOSITE_EGO_CRUISE_MS
            )
            stream.init_lead((entry_s - meet) - v_nom * t_ego)
        else:
            stream.init_spread(self.np_random)
        self._stream = stream

        spawned = 0
        for k in range(n):
            s = min(max(self._lane_s(k), 0.5), length - 0.5)
            heading = float(lane.heading_theta_at(s)) + (math.pi if self._dir < 0 else 0.0)
            try:
                if user == "bicycle":
                    obj = self._spawn_cyclist(lane.position(s, 0.0), heading)
                else:
                    obj = self._spawn_bus(s, heading, f"bus{k}")
            except Exception as exc:  # noqa: BLE001
                print(f"[ReservedLane] spawn failed: {exc!r}")
                continue
            obj._trb_reserved_agent = True
            obj._trb_reserved_lane_key = str(getattr(lane, "index", ""))
            agent = {"obj": obj, "k": k}
            self._agents.append(agent)
            self._place(agent)
            spawned += 1
        print(
            f"[ReservedLane] {spawned} {user}(s) on lane {getattr(lane, 'index', '?')} "
            f"flow={flow} v={v_nom:.1f} m/s headway={headway_m:.0f} m "
            f"entry={entry_s:.1f} stretch={stretch:.0f} active={len(stream.active_indices())}"
            + (f" first_meet_in={t_ego:.1f}s" if t_ego is not None else "")
            + (f" delta={self._delta_m:+.1f} m" if abs(self._delta_m) > 0.05 else "")
        )

    def before_step(self):
        if not self._agents or self._lane is None or self._stream is None:
            return {}
        self._stream.step(self._dt, entry_clear=lambda: self._spot_clear(self._entry_s))
        for agent in self._agents:
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
        self._stream = None
        self._entry_s = 0.0
        self._delta_m = 0.0

    def destroy(self):
        self._clear()
        super().destroy()
