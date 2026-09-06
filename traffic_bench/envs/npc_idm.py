"""
IDM policies adapted for SUMO maps.

SumoTrajectoryIDMPolicy — extends TrajectoryIDMPolicy with intersection-aware
collision avoidance: on intersections it checks ALL nearby objects (not just
those on its own PointLane) and brakes for the closest one in its path.
"""
import json
import math
import os

import numpy as np

from metadrive.policy.idm_policy import TrajectoryIDMPolicy, FrontBackObjects
from metadrive.utils.math import wrap_to_pi
from metadrive.component.vehicle.PID_controller import PIDController


class SumoTrajectoryIDMPolicy(TrajectoryIDMPolicy):
    """TrajectoryIDMPolicy + intersection collision avoidance + curvature speed.

    On straight road sections, behaves identically to TrajectoryIDMPolicy:
    only considers vehicles on its PointLane for braking.

    Near intersections (detected by lane index containing ':'), also scans
    ALL lidar-detected objects and brakes for any that are:
      - within INTERSECTION_SCAN_RADIUS
      - roughly ahead of the vehicle (within ±60° of heading)
    This prevents T-bone collisions at intersections.

    Additionally reduces speed based on road curvature so vehicles don't
    overshoot turns and leave the road surface.
    """

    NORMAL_SPEED = 30  # km/h — urban SUMO maps
    INTERSECTION_SCAN_RADIUS = 5.0   # metres — only imminent collisions
    INTERSECTION_HALF_ANGLE = math.pi / 4  # 45° — tighter cone
    # Front hemisphere for ego-yield (wider than intersection cone so T-bones slow).
    EGO_YIELD_HALF_ANGLE = math.pi / 2  # 90°
    CURVATURE_LOOK_AHEAD = 8.0  # metres — look further ahead
    CURVATURE_MU = 0.03  # conservative — slow down more for tight turns
    CURVATURE_MIN_SPEED = 3.0  # km/h

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Same PID gains as MetaDrive's proven IDMPolicy
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, 0.002, 0.05)

    def _curvature_target_speed(self):
        """Safe speed from heading change over look-ahead distance."""
        lane = self.routing_target_lane
        long, _ = lane.local_coordinates(self.control_object.position)
        h0 = lane.heading_theta_at(long)
        h1 = lane.heading_theta_at(min(long + self.CURVATURE_LOOK_AHEAD, lane.length))
        delta = abs(wrap_to_pi(h1 - h0))
        if delta > 0.02:
            radius = self.CURVATURE_LOOK_AHEAD / delta
            safe_kmh = math.sqrt(self.CURVATURE_MU * 9.81 * radius) * 3.6
            return max(min(self.NORMAL_SPEED, safe_kmh), self.CURVATURE_MIN_SPEED)
        return self.NORMAL_SPEED

    def steering_control(self, target_lane) -> float:
        """PID steering with 2 m lookahead (parent uses 1 m)."""
        ego = self.control_object
        long, lat = target_lane.local_coordinates(ego.position)
        lane_heading = target_lane.heading_theta_at(long + 2)
        v_heading = ego.heading_theta
        steering = self.heading_pid.get_result(-wrap_to_pi(lane_heading - v_heading))
        steering += self.lateral_pid.get_result(-lat)
        return float(steering)

    # Sign compliance for background traffic. Speed plates bound the target
    # speed while the car is inside the zone and heading the plate's way;
    # a detour plate reroutes the car onto the allowed adjacent lane before the
    # cones. Signs are placed after env.reset(), i.e. after this policy was
    # constructed, so both lookups are lazy and cached on first sight.
    SIGN_LATERAL_TOL_M = 7.5      # the plate governs the whole carriageway
    DETOUR_ON_LANE_TOL_M = 1.9    # centre within the lane the cones sit on
    FLOOR_PUSH_CLEAR_M = 25.0     # push to a 4.6 floor only with this much road ahead

    def _speed_signs(self):
        cached = getattr(self, "_speed_signs_cache", None)
        if cached is not None:
            return cached
        out = []
        mgr = getattr(self.engine, "traffic_sign_manager", None)
        for s in (getattr(mgr, "signs", None) or []):
            if hasattr(s, "zone_start") and (hasattr(s, "speed_limit") or hasattr(s, "min_speed")):
                if type(s).__name__.startswith("EndOf"):
                    continue
                out.append(s)
        if mgr is not None and out:
            self._speed_signs_cache = out
        return out

    def _inside_sign_zone(self, sign) -> bool:
        veh = self.control_object
        inside = None
        try:
            inside = sign._in_multi_edge_zone(veh)
        except Exception:
            inside = None
        if inside is not None:
            return bool(inside)
        try:
            s, lat = sign.lane.local_coordinates(veh.position)
        except Exception:
            return False
        if abs(float(lat)) > self.SIGN_LATERAL_TOL_M:
            return False
        try:
            if sign._heading_aligned(veh) is False:
                return False
        except Exception:
            pass
        end = float(getattr(sign, "zone_end", float("inf")))
        return float(sign.zone_start) <= float(s) <= end

    def _sign_speed_bounds(self):
        """(ceiling_kmh, floor_kmh) from the speed plates whose zone holds us."""
        ceiling, floor = None, None
        for sign in self._speed_signs():
            if not self._inside_sign_zone(sign):
                continue
            lim = getattr(sign, "speed_limit", None)
            if lim is not None:
                ceiling = float(lim) if ceiling is None else min(ceiling, float(lim))
            mn = getattr(sign, "min_speed", None)
            if mn is not None:
                floor = float(mn) if floor is None else max(floor, float(mn))
        return ceiling, floor

    def _maybe_reroute_for_detour(self):
        """Once: if we drive on the lane a detour plate's cones block, and the
        cones are still ahead, swap our trajectory for one that moves onto
        the allowed lane before the zone and comes back after the cluster."""
        if getattr(self, "_detour_checked", False):
            return
        mgr = getattr(self.engine, "traffic_sign_manager", None)
        signs = getattr(mgr, "signs", None) or []
        if mgr is None or not signs:
            return  # signs not placed yet; look again next step
        self._detour_checked = True
        veh = self.control_object
        for sign in signs:
            if not hasattr(sign, "obstacle_long") or not getattr(sign, "_allowed_lane_indices", None):
                continue
            try:
                s, lat = sign.lane.local_coordinates(veh.position)
            except Exception:
                continue
            if abs(float(lat)) > self.DETOUR_ON_LANE_TOL_M:
                continue
            if float(s) >= float(sign.obstacle_long) - 10.0:
                continue
            traffic_mgr = None
            for m in (getattr(self.engine, "managers", None) or {}).values():
                if hasattr(m, "rebuild_detour_trajectory"):
                    traffic_mgr = m
                    break
            if traffic_mgr is None:
                return
            new_traj = traffic_mgr.rebuild_detour_trajectory(veh, sign, self.routing_target_lane)
            self._probe("detour", rerouted=new_traj is not None,
                        s_on_sign_lane=round(float(s), 1),
                        obstacle_long=round(float(sign.obstacle_long), 1))
            if new_traj is not None:
                self.traj_to_follow = new_traj
                self.routing_target_lane = new_traj
                self.destination = np.asarray(new_traj.end)
            return

    def _probe(self, kind, **fields):
        """One JSON line per call when TRB_NPC_SIGN_PROBE names a file: what the
        traffic actually did around the plates. Off unless asked."""
        path = os.environ.get("TRB_NPC_SIGN_PROBE")
        if not path:
            return
        try:
            rec = {"kind": kind, "step": int(getattr(self.engine, "episode_step", 0) or 0),
                   "id": str(self.control_object.id)[:8]}
            rec.update(fields)
            with open("%s.%d" % (path, os.getpid()), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def act(self, do_speed_control, *args, **kwargs):
        comply_all = os.environ.get("TRB_NPC_SIGN_COMPLIANCE", "1") != "0"
        if comply_all:
            # Leaving the closed lane is never optional: a car that stays
            # drives into the cones. The per-car flag below covers plates only.
            self._maybe_reroute_for_detour()
        # Per-car plate compliance, drawn at spawn from the row's
        # npc_compliance_rate (traffic.py). Missing flag = compliant.
        comply = comply_all and bool(getattr(self.control_object, "_trb_sign_compliant", True))
        self.target_speed = self._curvature_target_speed()
        ceiling, floor = self._sign_speed_bounds()
        if ceiling is not None or floor is not None:
            self._probe("zone", speed_kmh=round(float(self.control_object.speed_km_h), 1),
                        ceiling=ceiling, floor=floor, comply=comply)
        if comply and ceiling is not None:
            self.target_speed = min(self.target_speed, ceiling)
        if comply and floor is not None:
            # The floor raises the desired speed only; car-following in
            # acceleration() still brakes for a slower leader.
            self.target_speed = max(self.target_speed, floor)

        front_dist = None
        try:
            if do_speed_control:
                all_objects = self.control_object.lidar.get_surrounding_objects(
                    self.control_object
                )
                surrounding = FrontBackObjects.get_find_front_back_objs_single_lane(
                    all_objects, self.routing_target_lane,
                    self.control_object.position, max_distance=self.IDM_MAX_DIST
                )
                acc_front_obj = surrounding.front_object()
                acc_front_dist = surrounding.front_min_distance()

                # Only check crossing traffic if no same-lane vehicle nearby.
                # Otherwise IDM already handles car-following correctly and
                # the crossing check would brake for adjacent-lane vehicles.
                if acc_front_dist > self.INTERSECTION_SCAN_RADIUS:
                    cross_obj, cross_dist = self._find_crossing_obstacle(all_objects)
                    if cross_obj is not None and cross_dist < acc_front_dist:
                        acc_front_obj = cross_obj
                        acc_front_dist = cross_dist

                # Optional skill-bench mode: treat ego as a hard obstacle when
                # nearby so NPC T-bones don't poison sign-compliance eval.
                ego_obj, ego_dist = self._find_ego_to_yield()
                if ego_obj is not None and ego_dist < acc_front_dist:
                    acc_front_obj = ego_obj
                    acc_front_dist = ego_dist

                acc = self.acceleration(acc_front_obj, acc_front_dist)
                front_dist = acc_front_dist
            else:
                acc = self.last_action[-1]
        except Exception:
            acc = 0

        # Soft speed cap: if over target, clamp acc to at most 0 (coast/brake)
        if self.control_object.speed_km_h > self.target_speed:
            acc = min(acc, 0.0)

        # Plates need more than a lowered target. Coasting alone left the 30
        # and 40 km/h zones with exactly the same overspeed steps as with
        # compliance off, and the raised 4.6 target barely moved a queue of
        # cars spawned at 3-12 m/s. Brake in proportion to the excess above a
        # ceiling; push up to a floor only while the road ahead is clear.
        if comply:
            v_kmh = float(self.control_object.speed_km_h)
            if ceiling is not None and v_kmh > ceiling + 1.0:
                acc = min(acc, -min(1.0, 0.25 + 0.05 * (v_kmh - ceiling)))
            if floor is not None and v_kmh < floor - 1.0 \
                    and (front_dist is None or front_dist > self.FLOOR_PUSH_CLEAR_M):
                acc = max(acc, min(1.0, 0.3 + 0.03 * (floor - v_kmh)))

        # Traffic light compliance (DISABLED — investigating stuck NPCs)
        # is_red = self._should_stop_for_red()
        # if is_red:
        #     acc = min(acc, -1.0)
        # elif self.control_object.speed_km_h < 3.0 and acc < 0.5:
        #     acc = 0.5

        steering = self.steering_control(self.routing_target_lane)
        self.last_action = [steering, acc]
        self.action_info["action"] = [steering, acc]
        return [steering, acc]

    def _should_stop_for_red(self):
        """Check if the vehicle should stop for a red traffic light."""
        try:
            engine = self.engine
            if not hasattr(engine, "traffic_sign_manager"):
                return False
            sign_mgr = engine.traffic_sign_manager
            ego = self.control_object
            ego_lane = ego.lane
            if ego_lane is None:
                return False
            ego_idx = getattr(ego_lane, "index", None)

            for sign in sign_mgr.signs:
                if type(sign).__name__ != "TrafficLightSign":
                    continue
                sign_idx = getattr(sign.lane, "index", None)
                # Only check signs on the same road (same from/to nodes)
                if ego_idx is None or sign_idx is None:
                    continue
                try:
                    if ego_idx[0] != sign_idx[0] or ego_idx[1] != sign_idx[1]:
                        continue
                except (IndexError, TypeError):
                    continue

                veh_long = sign.lane.local_coordinates(ego.position)[0]
                if not (sign.zone_start <= veh_long <= sign.zone_end):
                    continue
                if not sign.current_states:
                    continue

                # Check signals for directions reachable from ego's lane.
                # NPC vehicles on PointLane may not have `turns` → fall back
                # to checking ALL signal states (optimistic: any green → go).
                ego_to_lanes = set()
                for turn in getattr(ego_lane, "turns", []) or []:
                    to = turn.get("to_lane") if isinstance(turn, dict) else None
                    if to:
                        ego_to_lanes.add(to)

                if ego_to_lanes:
                    relevant = [s for to_lane, s in sign.current_states.items()
                                if to_lane in ego_to_lanes]
                else:
                    # No turn info → use all states (any green = go)
                    relevant = list(sign.current_states.values())
                if not relevant:
                    continue
                any_green = any(s in ("g", "G") for s in relevant)
                if not any_green:
                    dist_to_end = sign.lane.length - veh_long
                    if dist_to_end < 30:
                        return True
        except Exception:
            pass
        return False

    def _find_crossing_obstacle(self, all_objects):
        """Find the closest object ahead that is NOT on our PointLane
        but is within intersection scan zone."""
        ego = self.control_object
        ego_pos = ego.position
        ego_heading = ego.heading_theta

        best_obj = None
        best_dist = self.INTERSECTION_SCAN_RADIUS

        for obj in all_objects:
            if obj is ego:
                continue
            dx = obj.position[0] - ego_pos[0]
            dy = obj.position[1] - ego_pos[1]
            dist = math.hypot(dx, dy)
            if dist > self.INTERSECTION_SCAN_RADIUS or dist < 1.0:
                continue

            # Check if object is roughly ahead of us
            angle_to_obj = math.atan2(dy, dx)
            angle_diff = abs(wrap_to_pi(angle_to_obj - ego_heading))
            if angle_diff > self.INTERSECTION_HALF_ANGLE:
                continue

            if dist < best_dist:
                best_dist = dist
                best_obj = obj

        return best_obj, best_dist

    def _find_ego_to_yield(self):
        """If npc_ego_yield_radius > 0, return the ego agent when it is inside
        that radius and in the NPC's front hemisphere.

        Returns (ego_vehicle, distance) or (None, inf). Distance fed into IDM
        is slightly tightened so NPCs start braking earlier than for peers.
        """
        try:
            radius = float(self.engine.global_config.get("npc_ego_yield_radius", 0.0) or 0.0)
        except Exception:
            radius = 0.0
        if radius <= 0.0:
            return None, float("inf")

        agents = getattr(self.engine.agent_manager, "active_agents", None) or {}
        if not agents:
            return None, float("inf")
        ego = next(iter(agents.values()))
        if ego is None or ego is self.control_object:
            return None, float("inf")

        npc = self.control_object
        dx = float(ego.position[0] - npc.position[0])
        dy = float(ego.position[1] - npc.position[1])
        dist = math.hypot(dx, dy)
        if dist > radius or dist < 0.5:
            return None, float("inf")

        angle_to_ego = math.atan2(dy, dx)
        angle_diff = abs(wrap_to_pi(angle_to_ego - npc.heading_theta))
        if angle_diff > self.EGO_YIELD_HALF_ANGLE:
            return None, float("inf")

        # Slightly under-report distance so IDM brakes with more margin.
        return ego, max(0.5, dist * 0.7)
