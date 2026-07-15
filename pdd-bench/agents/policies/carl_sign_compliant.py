"""Carl-Sign-Compliant Expert Policy — CaRL PPO + sign compliance.

Normal driving is handled by the CaRL PPO network (loaded from a checkpoint).
Sign compliance (stop, speed limit, priority, etc.) is applied as
post-processing by SignComplianceMixin — throttle-side only by default to
avoid fighting the NN on steering.

Usage (from replay_mini_new.py):
    CarlSignCompliantPolicy.set_checkpoint("/path/to/carl.ckpt", device="cpu")
    config["agent_policy"] = CarlSignCompliantPolicy
    ...

The CaRLMetaDriveAdapter is loaded lazily and shared as a class attribute:
heavy (~100 MB PPO net), only loaded once per process regardless of scene.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.policy.base_policy import BasePolicy

from agents.policies._sign_compliance_mixin import SignComplianceMixin


class CarlSignCompliantPolicy(SignComplianceMixin, BasePolicy):
    """CaRL PPO + SignComplianceMixin post-processing."""

    # --- shared CaRL adapter, loaded once per process ---
    _carl_adapter = None
    _carl_checkpoint: Optional[str] = None
    _carl_device: str = "cpu"

    # Let mixin override steering for sign-driven lane changes
    # (pre-positioning at 5.15.2 direction signs, lane bans, etc.).
    APPLY_LANE_CHANGE_OVERRIDE = True

    # When False, skip ALL sign-compliance post-processing (process_signs,
    # lane-change override, no-overtaking guard, speed-cap clamp). Used by
    # PlainCarlPolicy to expose raw CaRL behaviour without rule overlay.
    APPLY_RULE_OVERLAY = True

    @classmethod
    def set_checkpoint(cls, checkpoint_path: str, device: str = "cpu"):
        """Configure the CaRL checkpoint path + device BEFORE env construction.

        Must be called before the first `agent_policy` instantiation. The
        adapter is constructed lazily on the first `act()` so env setup is
        fast.
        """
        cls._carl_checkpoint = checkpoint_path
        cls._carl_device = device

    @classmethod
    def _get_adapter(cls):
        if cls._carl_adapter is None:
            if cls._carl_checkpoint is None:
                raise RuntimeError(
                    "CaRL checkpoint not configured. Call "
                    "CarlSignCompliantPolicy.set_checkpoint(<path>) before env build."
                )
            from agents.carl_in_metadrive.carl_adapter import CaRLMetaDriveAdapter
            cls._carl_adapter = CaRLMetaDriveAdapter(
                cls._carl_checkpoint, device=cls._carl_device
            )
        return cls._carl_adapter

    def __init__(self, control_object, random_seed=None, config=None):
        BasePolicy.__init__(self, control_object, random_seed, config)
        # PID controllers used by the mixin for optional lane-change override.
        self._heading_pid = PIDController(1.7, 0.01, 3.5)
        self._lateral_pid = PIDController(0.3, 0.002, 0.05)
        self._init_sign_compliance()
        # New episode → reset adapter's internal state (last action, measurements).
        adapter = self._get_adapter()
        try:
            adapter.reset()
        except Exception:
            pass

    # --- Abstract-method impl from SignComplianceMixin ---
    def _get_heading_pid(self):
        return self._heading_pid

    def _get_lateral_pid(self):
        return self._lateral_pid

    def act(self, agent_id=None):
        if self.APPLY_RULE_OVERLAY:
            # Replan / speed-cap state before CaRL so route is updated early.
            self._process_signs()

        adapter = self._get_adapter()
        # CaRL returns a MetaDrive-compatible [steering, throttle] numpy array.
        try:
            action = adapter.get_action(self.control_object, self.engine)
        except Exception as exc:
            # Adapter-side error — fallback to zero action; sign-compliance
            # layer below may still clamp to brake if a sign demands it.
            print(f"[CarlSignCompliantPolicy] get_action failed: {exc}")
            action = np.asarray([0.0, 0.0], dtype=np.float32)

        steering = float(action[0])
        throttle = float(action[1])

        if self.APPLY_RULE_OVERLAY:
            # Lane-change steering override from sign logic.
            if self.APPLY_LANE_CHANGE_OVERRIDE:
                self._update_lane_change()
                if self._lc_target_lane is not None:
                    steering = float(np.clip(
                        self._steering_control_for_lc(self._lc_target_lane), -1.0, 1.0
                    ))

            # No-overtaking steering guard (matches RuleCompliantExpertPolicy:59-71):
            # if CaRL steers toward the opposite lane while overtaking is forbidden,
            # clamp it back toward lane-following.
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

            # Clamp throttle by sign-driven speed_cap / speed_floor.
            throttle = self._apply_speed_constraints(
                throttle, self.control_object.speed_km_h
            )

        action_out = [steering, throttle]
        self.action_info["action"] = action_out
        return action_out
