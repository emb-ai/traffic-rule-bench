"""
Rule-Compliant Expert Policy — PPO + sign compliance.

Normal driving is handled by the PPO ExpertPolicy neural network.
Sign compliance (stop, speed limit, lane changes, etc.) is provided
by SignComplianceMixin as post-processing.

PPO parity notes:
  - PPO has no target_speed feed-forward (unlike IDM), so speed
    compliance is purely reactive via _apply_speed_constraints().
    The mixin's SPEED_SIGN_LOOKAHEAD (50 m) compensates.
  - When _no_overtaking_active is set, PPO steering is clamped to
    prevent drifting into the opposite lane.
  - _reset_sign_compliance() clears stale state across episodes.
"""

import numpy as np

from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.utils.math import wrap_to_pi

from agents.policies._sign_compliance_mixin import SignComplianceMixin


class RuleCompliantExpertPolicy(SignComplianceMixin, ExpertPolicy):

    def __init__(self, control_object, random_seed=None, config=None):
        ExpertPolicy.__init__(self, control_object, random_seed, config)
        self._heading_pid = PIDController(1.7, 0.01, 3.5)
        self._lateral_pid = PIDController(0.3, 0.002, 0.05)
        self._init_sign_compliance()

    def _get_heading_pid(self):
        return self._heading_pid

    def _get_lateral_pid(self):
        return self._lateral_pid

    def act(self, agent_id=None):
        # Base PPO action (neural network)
        action = ExpertPolicy.act(self, agent_id)
        steering = float(action[0])
        throttle = float(action[1])

        # Process all signs -> sets _speed_cap, _speed_floor,
        # _blocked_lanes, _lc_target_lane, _no_overtaking_active, etc.
        self._process_signs()

        # Lane-change override
        self._update_lane_change()
        if self._lc_target_lane is not None:
            steering = np.clip(
                self._steering_control_for_lc(self._lc_target_lane), -1.0, 1.0
            )

        # No-overtaking steering guard: if PPO steers toward the opposite
        # lane heading, clamp it to keep the vehicle in its lane.
        if self._no_overtaking_active and self._lc_target_lane is None:
            ego = self.control_object
            cur_lane = ego.lane
            if cur_lane is not None:
                long, lat = cur_lane.local_coordinates(ego.position)
                # If vehicle is drifting left (toward opposite lane) and PPO
                # steers further left, clamp steering to lane-following.
                if lat < -0.5 and steering < 0:
                    lane_steer = self._steering_control_for_lc(cur_lane)
                    steering = max(steering, lane_steer)
                elif lat > 0.5 and steering > 0:
                    lane_steer = self._steering_control_for_lc(cur_lane)
                    steering = min(steering, lane_steer)

        # Apply speed constraints (reactive throttle clamp)
        throttle = self._apply_speed_constraints(
            throttle, self.control_object.speed_km_h
        )

        action = [steering, throttle]
        self.action_info["action"] = action
        return action
