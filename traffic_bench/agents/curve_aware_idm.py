"""Curve-aware base IDM — an ego baseline with the same defensive layer as sumoidm.

Bare MetaDrive IDMPolicy steers with a 1 m preview and knows nothing about
road curvature — at 40+ km/h it fails to hold OSM-map turns (43% OOR for
idm_default). Meanwhile the NPCs (SumoTrajectoryIDMPolicy) and the rule
expert (ComprehensiveRuleExpertPolicy) have long been driving with a
curvature speed cap, an extended steering lookahead, and braking for
crossing traffic.

This class gives the BASE idm exactly the expert's defensive layer — the
methods and constants are reused from ComprehensiveRuleExpertPolicy
directly (not copied) so the two never drift apart. There is NO sign
knowledge here: the idm/idm_rule pair now differs only in sign compliance,
not in handling — the pair gap becomes clean.

Rollback: EGO_CURVE_AWARE=0 in the environment restores the raw IDMPolicy
(see bench/policy_factory.py).
"""
from __future__ import annotations

from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.policy.idm_policy import IDMPolicy

from traffic_bench.agents.idm_rule import ComprehensiveRuleExpertPolicy as _EXPERT


class CurveAwareIDMPolicy(IDMPolicy):
    # Defensive-layer constants — same as the expert's (single source).
    STEERING_LOOKAHEAD_PG = _EXPERT.STEERING_LOOKAHEAD_PG
    STEERING_LOOKAHEAD_SUMO = _EXPERT.STEERING_LOOKAHEAD_SUMO
    MAX_LONG_DIST = _EXPERT.MAX_LONG_DIST
    SAFE_LANE_CHANGE_DISTANCE = _EXPERT.SAFE_LANE_CHANGE_DISTANCE
    INTERSECTION_SCAN_RADIUS = _EXPERT.INTERSECTION_SCAN_RADIUS
    INTERSECTION_HALF_ANGLE = _EXPERT.INTERSECTION_HALF_ANGLE
    CURVATURE_LOOK_AHEAD = _EXPERT.CURVATURE_LOOK_AHEAD
    CURVATURE_MU_PG = _EXPERT.CURVATURE_MU_PG
    CURVATURE_MU_SUMO = _EXPERT.CURVATURE_MU_SUMO
    CURVATURE_MIN_SPEED = _EXPERT.CURVATURE_MIN_SPEED

    # Reuse the expert's implementation (plain functions/descriptors — they
    # work on instances of this class; no sign logic is pulled in).
    _detect_sumo = _EXPERT._detect_sumo
    STEERING_LOOKAHEAD = _EXPERT.STEERING_LOOKAHEAD          # property (PG/SUMO)
    CURVATURE_MU = _EXPERT.CURVATURE_MU                      # property (PG/SUMO)
    steering_control = _EXPERT.steering_control              # 2–3 m lookahead
    _curvature_target_speed = _EXPERT._curvature_target_speed
    _find_crossing_obstacle = _EXPERT._find_crossing_obstacle

    def __init__(self, control_object, random_seed=None, config=None):
        super().__init__(control_object, random_seed)
        # Same PID gains as the expert/NPC (they match the defaults, but we
        # pin them explicitly — one shared tuning for the pair).
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, 0.002, 0.05)
        self._is_sumo = self._detect_sumo()

    def acceleration(self, front_obj, dist_to_front) -> float:
        # (1) Speed cap from the curvature ahead. Hooked exactly here: the
        # base act() overwrites target_speed in its lane-change branches
        # BEFORE calling acceleration, so we clamp as the last word.
        try:
            cap = self._curvature_target_speed()
            if cap < self.target_speed:
                self.target_speed = cap
        except Exception:
            pass
        # (2) Defensive braking for crossing traffic at a junction
        # (mirror of the expert/SumoTrajectoryIDMPolicy).
        try:
            if dist_to_front is None or dist_to_front > self.INTERSECTION_SCAN_RADIUS:
                obj, dist = self._find_crossing_obstacle()
                if obj is not None and (dist_to_front is None or dist < dist_to_front):
                    front_obj, dist_to_front = obj, dist
        except Exception:
            pass
        return super().acceleration(front_obj, dist_to_front)
