"""Modified IDM + sign compliance (same overlay pattern as CarlSignCompliantPolicy).

ModifiedIDMPolicy handles defensive driving (intersection crossing brake,
curvature limits, stop-sign wait). SignComplianceMixin adds yield / priority
and other traffic-sign rules as post-processing on top of each act() step.
"""

from __future__ import annotations

import numpy as np

from metadrive.policy.idm_policy import ModifiedIDMPolicy

from agents.policies._sign_compliance_mixin import SignComplianceMixin


class ModifiedIDMSignCompliantPolicy(SignComplianceMixin, ModifiedIDMPolicy):
    """ModifiedIDMPolicy with SignComplianceMixin post-processing."""

    APPLY_LANE_CHANGE_OVERRIDE = True
    APPLY_RULE_OVERLAY = True

    def __init__(self, control_object, random_seed=None, config=None):
        ModifiedIDMPolicy.__init__(self, control_object, random_seed)
        self._init_sign_compliance()

    def _get_heading_pid(self):
        return self.heading_pid

    def _get_lateral_pid(self):
        return self.lateral_pid

    def act(self, *args, **kwargs):
        action = ModifiedIDMPolicy.act(self, *args, **kwargs)
        steering = float(action[0])
        throttle = float(action[1])

        if self.APPLY_RULE_OVERLAY:
            self._process_signs()

            if self.APPLY_LANE_CHANGE_OVERRIDE:
                self._update_lane_change()
                if self._lc_target_lane is not None:
                    steering = float(
                        np.clip(
                            self._steering_control_for_lc(self._lc_target_lane),
                            -1.0,
                            1.0,
                        )
                    )

            if self._no_overtaking_active and self._lc_target_lane is None:
                ego = self.control_object
                cur_lane = ego.lane
                if cur_lane is not None:
                    _, lat = cur_lane.local_coordinates(ego.position)
                    if lat < -0.5 and steering < 0:
                        lane_steer = self._steering_control_for_lc(cur_lane)
                        steering = max(steering, lane_steer)
                    elif lat > 0.5 and steering > 0:
                        lane_steer = self._steering_control_for_lc(cur_lane)
                        steering = min(steering, lane_steer)

            throttle = self._apply_speed_constraints(
                throttle, self.control_object.speed_km_h
            )

        action_out = [steering, throttle]
        self.action_info["action"] = action_out
        return action_out
