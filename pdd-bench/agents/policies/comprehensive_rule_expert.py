"""
Comprehensive Rule Expert Policy — IDM + full sign compliance.

Deterministic rule-based expert suitable for collecting compliant
driving trajectories.  All traffic signs and rules are enforced
via the extended SignComplianceMixin as post-processing over the
IDM base driver.

Covers:
  - Speed limits (regular, zone, minimum, end-of-zone)
  - Stop sign with post-stop safety check
  - Yield sign + right-hand rule at equal-priority intersections
  - Traffic lights
  - No-entry / no-traffic
  - No right/left/U-turn
  - One-way entry
  - No overtaking
  - Direction signs + lane-allowed-direction (pre-positioning)
  - Lane restrictions (bus/bike lane)
  - Intersection restricted lanes (exit-to-bus/bike)
  - Detour signs
  - Bus station
  - Only auto
  - Right turn rule
  - Pedestrian yield
  - End-of-main-road

Defensive add-ons (bring ego up to par with SumoTrajectoryIDMPolicy
which is used for surrounding traffic in SUMO maps):
  - Extended planning horizon (MAX_LONG_DIST = 60 m, was 30 m)
  - Larger desired time-gap (TIME_WANTED = 2.0 s, was 1.5 s)
  - Curvature-aware target speed (slow down before tight turns)
  - Intersection crossing-traffic emergency brake
"""

import math

import numpy as np

from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils.math import wrap_to_pi

from agents.policies._sign_compliance_mixin import SignComplianceMixin


class ComprehensiveRuleExpertPolicy(SignComplianceMixin, IDMPolicy):
    """IDM-based expert policy with comprehensive traffic-sign compliance."""

    # Per-backend steering lookahead. PGMap uses the wider 3m (validated on
    # pgmap scenes); SUMO uses 2m — tight junction geometries in SUMO made
    # 3m cause PID oversteer on curved internal lanes.
    STEERING_LOOKAHEAD_PG = 3
    STEERING_LOOKAHEAD_SUMO = 2

    # ---- Defensive overrides on top of base IDMPolicy ---------------------
    # Wider planning horizon: the base IDM looked only 30 m ahead which on
    # SUMO maps is too short — by the time a slow/stopped vehicle entered
    # the cone IDM had no time to brake. Doubling it gives ~2 s extra at
    # 30 km/h to react.
    MAX_LONG_DIST = 60
    # Larger desired time gap to leader (was 1.5 s).
    TIME_WANTED = 2.0
    # Larger jam distance (was 10 m) — gives more slack at low speed.
    DISTANCE_WANTED = 12.0
    # Be more eager to bail out of unsafe lane changes.
    SAFE_LANE_CHANGE_DISTANCE = 20

    # Crossing-traffic detection (mirrors SumoTrajectoryIDMPolicy but with
    # a slightly wider search radius because the ego does not have a
    # PointLane and benefits from earlier reaction).
    INTERSECTION_SCAN_RADIUS = 12.0      # m
    INTERSECTION_HALF_ANGLE = math.pi / 3  # 60° forward cone

    # Curvature speed reduction. PGMap keeps the original conservative
    # 0.05; SUMO uses 0.03 (same as SumoTrajectoryIDMPolicy) — on SUMO the
    # 0.05 value was over-braking in internal junction lanes and causing
    # the ego to lose momentum / drift off-route.
    CURVATURE_LOOK_AHEAD = 12.0  # m — look further ahead than surrounding
    CURVATURE_MU_PG = 0.05
    CURVATURE_MU_SUMO = 0.03
    CURVATURE_MIN_SPEED = 5.0    # km/h — never slower than walking

    # Ego-specific floors so per-scene profile (apply_profile_to_idm_class
    # writes onto IDMPolicy class attrs) doesn't leave ego with 8 km/h target
    # after stop-sign releases. NPCs still inherit the raw profile from
    # IDMPolicy — these floors only apply to the expert instance.
    EGO_NORMAL_SPEED_FLOOR_KMH = 30.0
    EGO_ACC_FACTOR_FLOOR = 1.5

    def __init__(self, control_object, random_seed=None, config=None):
        IDMPolicy.__init__(self, control_object, random_seed)
        self._init_sign_compliance()
        # Stronger PID for SUMO maps with tight turns
        from metadrive.component.vehicle.PID_controller import PIDController
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, 0.002, 0.05)
        self._is_sumo = self._detect_sumo()
        # Floor ego-facing speed/accel as instance attrs — shadow IDMPolicy
        # class attrs that were clobbered by the per-scene profile.
        self.NORMAL_SPEED = max(float(IDMPolicy.NORMAL_SPEED), self.EGO_NORMAL_SPEED_FLOOR_KMH)
        self.ACC_FACTOR = max(float(IDMPolicy.ACC_FACTOR), self.EGO_ACC_FACTOR_FLOOR)

    def _detect_sumo(self) -> bool:
        """True iff ego is on a SUMO (EdgeRoadNetwork / ScenarioLane) map.
        Cached once at init; PGMap tuning stays untouched."""
        try:
            lane = getattr(self.control_object, "lane", None)
            if lane is not None:
                idx = getattr(lane, "index", None)
                if isinstance(idx, str):
                    return True
            engine = getattr(self, "engine", None)
            if engine is not None and hasattr(engine, "current_map"):
                from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
                rn = getattr(engine.current_map, "road_network", None)
                if rn is not None and isinstance(rn, EdgeRoadNetwork):
                    return True
        except Exception:
            pass
        return False

    @property
    def STEERING_LOOKAHEAD(self) -> float:
        return self.STEERING_LOOKAHEAD_SUMO if self._is_sumo else self.STEERING_LOOKAHEAD_PG

    @property
    def CURVATURE_MU(self) -> float:
        return self.CURVATURE_MU_SUMO if self._is_sumo else self.CURVATURE_MU_PG

    def _get_heading_pid(self):
        return self.heading_pid

    def _get_lateral_pid(self):
        return self.lateral_pid

    def steering_control(self, target_lane) -> float:
        """PID steering with extended lookahead for junction turns."""
        ego = self.control_object
        long, lat = target_lane.local_coordinates(ego.position)
        lane_heading = target_lane.heading_theta_at(long + self.STEERING_LOOKAHEAD)
        v_heading = ego.heading_theta
        from metadrive.utils.math import wrap_to_pi
        steering = self.heading_pid.get_result(-wrap_to_pi(lane_heading - v_heading))
        steering += self.lateral_pid.get_result(-lat)
        return float(steering)

    def _sync_routing_target(self):
        """Keep IDM routing_target_lane in sync with the ego's lane.

        On PGMap the original eager sync is kept (old behaviour preserved —
        PGMap was already validated at dest_rate ~= 1.0 before SUMO work).

        On SUMO, prefer to hold the planned target lane when it's still in
        ref_lanes (mirrors SumoTrajectoryIDMPolicy) — transient lateral
        drift in tight junctions would otherwise cause the target to jump
        to the drifted lane and abandon the BFS route.
        """
        ego = self.control_object
        nav = getattr(ego, "navigation", None)
        if nav is None:
            return
        current_lane = ego.lane
        if current_lane is None:
            return
        ref = getattr(nav, "current_ref_lanes", None) or []

        if self._is_sumo:
            # Conservative: don't overwrite a still-valid target on drift.
            if not ref:
                return
            if self.routing_target_lane in ref:
                return
            if current_lane in ref:
                self.routing_target_lane = current_lane
                return
            self.routing_target_lane = ref[0]
            return

        # PGMap — original eager sync
        if self.routing_target_lane is current_lane:
            return
        if current_lane in ref:
            self.routing_target_lane = current_lane
        elif self.routing_target_lane not in ref:
            self.routing_target_lane = current_lane

    # ---- Defensive helpers (mirror SumoTrajectoryIDMPolicy) ---------------
    def _curvature_target_speed(self) -> float:
        """Safe speed from heading change over the look-ahead distance.

        Slows the ego down before tight curves so PID steering does not
        overshoot the lane on SUMO maps with sharp junctions / curves.
        Falls back to NORMAL_SPEED if anything goes wrong (no lane,
        missing methods, etc.).
        """
        lane = self.routing_target_lane
        if lane is None or not hasattr(lane, "heading_theta_at"):
            return self.NORMAL_SPEED
        try:
            long, _ = lane.local_coordinates(self.control_object.position)
            far = min(long + self.CURVATURE_LOOK_AHEAD, getattr(lane, "length", long + self.CURVATURE_LOOK_AHEAD))
            h0 = lane.heading_theta_at(long)
            h1 = lane.heading_theta_at(far)
            delta = abs(wrap_to_pi(h1 - h0))
        except Exception:
            return self.NORMAL_SPEED
        if delta > 0.02:
            radius = self.CURVATURE_LOOK_AHEAD / delta
            safe_kmh = math.sqrt(self.CURVATURE_MU * 9.81 * radius) * 3.6
            return max(min(self.NORMAL_SPEED, safe_kmh), self.CURVATURE_MIN_SPEED)
        return self.NORMAL_SPEED

    def _find_crossing_obstacle(self):
        """Closest object inside the forward cone, regardless of lane.

        Used as a defensive layer on top of IDM car-following: when a
        crossing vehicle / pedestrian enters the ego's path at an
        intersection (and is therefore NOT on the same lane that IDM
        scanned), we still need to brake for it.

        Returns (obj, dist) or (None, inf).
        """
        ego = self.control_object
        try:
            all_objects = ego.lidar.get_surrounding_objects(ego)
        except Exception:
            return None, float("inf")

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
            angle_to_obj = math.atan2(dy, dx)
            angle_diff = abs(wrap_to_pi(angle_to_obj - ego_heading))
            if angle_diff > self.INTERSECTION_HALF_ANGLE:
                continue
            # Ignore stationary non-close objects: base IDM car-following already
            # handles a stopped leader on the same lane. Perpetual braking here
            # for static signs / parked NPCs / red-light waiters several metres
            # ahead was causing ego to freeze in place mid-route.
            # Still trigger for close (<6 m) static objects as a last-resort
            # collision guard, and for anything actually moving (possible
            # crossing traffic / pedestrian stepping in).
            try:
                obj_speed_kmh = float(getattr(obj, "speed_km_h", 0.0))
            except Exception:
                obj_speed_kmh = 0.0
            # Mid lane-change the relevant corridor is the TARGET lane: the
            # queue crawling on the lane being vacated (detour 4.2.x) must not
            # stall the merge. Applies to moving objects too.
            lc_target = getattr(self, "_lc_target_lane", None)
            if lc_target is not None and obj_speed_kmh >= 1.0:
                try:
                    s_obj, lat_obj = lc_target.local_coordinates(obj.position)
                    if abs(lat_obj) > lc_target.width_at(s_obj) / 2 + 0.5:
                        continue
                except Exception:
                    pass
            if obj_speed_kmh < 1.0:
                if dist > 6.0:
                    continue
                # Close static object: brake only if it actually sits in the
                # ego's own lane corridor. A detour sign / cone cluster (4.2.x)
                # stands mid-roadway on the ADJACENT blocked lane — without
                # this check the ego passing alongside freezes forever.
                # During an active lane change the corridor is the TARGET lane:
                # ego is vacating the current one, so the detour sign/cones on
                # it must not brake the ego into a crawl mid-manoeuvre.
                try:
                    lane = self._lc_target_lane or ego.lane
                    s_obj, lat_obj = lane.local_coordinates(obj.position)
                    if abs(lat_obj) > lane.width_at(s_obj) / 2 + 0.3:
                        continue
                except Exception:
                    pass
            if dist < best_dist:
                best_dist = dist
                best_obj = obj

        return best_obj, best_dist

    def act(self, *args, **kwargs):
        # Reset IDM target speed each step, capped by upcoming-curve limit.
        self.target_speed = min(self.NORMAL_SPEED, self._curvature_target_speed())

        # Direction / no-entry replan must run before base IDM steering.
        self._process_signs()

        # Base IDM action (car-following + PID steering + lane-change logic)
        action = IDMPolicy.act(self, *args, **kwargs)
        steering = float(action[0])
        throttle = float(action[1])

        # Defensive layer: if a crossing object is closer than what IDM
        # found on our own lane, recompute acceleration treating it as
        # the leader and clamp throttle. This catches T-bones, pedestrians
        # stepping into the cone, and adjacent-lane drift on intersections.
        crossing_brake_active = False
        try:
            cross_obj, cross_dist = self._find_crossing_obstacle()
            if cross_obj is not None:
                cross_acc = self.acceleration(cross_obj, cross_dist)
                if cross_acc < throttle:
                    throttle = cross_acc
                    crossing_brake_active = True
        except Exception:
            pass

        # Keep routing in sync
        self._sync_routing_target()

        # Update any active lane-change manoeuvre
        self._update_lane_change()
        if self._lc_target_lane is not None:
            steering = np.clip(
                self._steering_control_for_lc(self._lc_target_lane), -1.0, 1.0
            )

        steering = self._maybe_override_steering_for_direction_exit(steering)

        # Sync IDM target_speed with sign constraints
        if self._speed_cap is not None:
            if self._speed_cap < 1.0:
                self.target_speed = 0.0
            else:
                self.target_speed = min(self.target_speed, self._speed_cap)

        # Minimum-speed sign (4.6): raise the IDM target UP to the floor so the
        # controller actively cruises at the minimum and HOLDS it (symmetric to
        # the cap-lowering above). Without this IDM keeps aiming for NORMAL_SPEED
        # (~30) below the min, and the reactive throttle floor only nudges it,
        # so the ego hovers below the min - tolerance and violates.
        if self._speed_floor is not None:
            self.target_speed = max(self.target_speed, self._speed_floor)

        # Apply final speed constraints (braking / acceleration floor)
        throttle = self._apply_speed_constraints(
            throttle, self.control_object.speed_km_h
        )

        # Kick-start: if target speed is above walking speed but ego is near
        # stopped, give a firm throttle. Skip only if IDM actively wanted to
        # brake OR the crossing-traffic layer actually clamped (i.e. a real
        # threat was detected).
        idm_throttle = float(action[1])
        if (self.target_speed >= 5.0
                and self.control_object.speed_km_h < 2.0
                and throttle < 0.6
                and idm_throttle >= 0.0
                and not crossing_brake_active):
            throttle = 0.6

        # Stuck watchdog: if we've been at ~0 speed for many steps with no
        # active speed cap from signs and no crossing brake, force a positive
        # throttle to unstick. Covers edge cases where IDM leader detection
        # latches onto a phantom obstacle or curvature slowdown floors out.
        if not hasattr(self, "_stuck_steps"):
            self._stuck_steps = 0
        if (self.control_object.speed_km_h < 1.0
                and self._speed_cap is None
                and not crossing_brake_active):
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        if self._stuck_steps >= 20:  # ~2 s stalled without a reason
            throttle = max(throttle, 0.8)

        action = [steering, throttle]
        self.action_info["action"] = action
        return action
