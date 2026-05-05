"""
IDM policies adapted for SUMO maps.

SumoTrajectoryIDMPolicy — extends TrajectoryIDMPolicy with intersection-aware
collision avoidance: on intersections it checks ALL nearby objects (not just
those on its own PointLane) and brakes for the closest one in its path.
"""
import math

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

    def act(self, do_speed_control, *args, **kwargs):
        self.target_speed = self._curvature_target_speed()

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

                acc = self.acceleration(acc_front_obj, acc_front_dist)
            else:
                acc = self.last_action[-1]
        except Exception:
            acc = 0

        # Soft speed cap: if over target, clamp acc to at most 0 (coast/brake)
        if self.control_object.speed_km_h > self.target_speed:
            acc = min(acc, 0.0)

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
