"""
SpeedLimitShield: safety layer for speed limit handling.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np

from speed_limit_feat import SpeedLimitFeatureExtractor


@dataclass
class SpeedLimitShieldConfig:
    # Proportional braking gain for overspeed [m/s] -> action brake magnitude.
    kp_overspeed: float = 0.35
    # Hard cap on braking from speed shield so it does not try to "stop" the car.
    max_brake_norm: float = 0.45
    # Near-limit hold band: suppress acceleration to avoid crossing the limit.
    hold_margin_mps: float = 0.3


class SpeedLimitShield:
    """
    In the speed zone:
      - speed > limit  -> apply bounded proportional braking (cap max speed)
      - near limit     -> suppress positive acceleration
      - below limit    -> no intervention
      - if stop shield is active, do not weaken its braking
    """

    def __init__(self, rule_extractor: SpeedLimitFeatureExtractor, cfg: SpeedLimitShieldConfig = None):
        self.fx = rule_extractor
        self.cfg = cfg or SpeedLimitShieldConfig()

    def clip_action(
        self,
        z: np.ndarray,
        carl_action: np.ndarray,
        stop_shield_active: bool = False,
    ):

        in_zone = z[self.fx.IDX_IN_ZONE] > 0.5
        ego_speed = float(z[self.fx.IDX_EGO_SPEED]) * self.fx.speed_norm_max_mps
        limit_speed = float(z[self.fx.IDX_LIMIT_SPEED]) * self.fx.speed_norm_max_mps
        overspeed = max(0.0, ego_speed - limit_speed)

        accel = float(carl_action[0])
        steer = float(carl_action[1])
        accel_new = accel

        if in_zone and limit_speed > 0.0:
            if overspeed > 0.0:
                # Bounded braking to pull speed back to the limit without full stop behavior.
                desired_brake = -min(self.cfg.max_brake_norm, self.cfg.kp_overspeed * overspeed)
                accel_new = min(accel_new, desired_brake)
            elif ego_speed >= max(0.0, limit_speed - self.cfg.hold_margin_mps):
                # Near speed limit: do not add more acceleration.
                accel_new = min(accel_new, 0.0)

            if stop_shield_active:
                # Keep stronger braking decisions made by stop shield.
                accel_new = min(accel_new, accel)

        safe = np.array(
            [np.clip(accel_new, -1.0, 1.0), np.clip(steer, -1.0, 1.0)],
            dtype=np.float32,
        )
        clip = safe - carl_action

        info = {
            "in_zone": in_zone,
            "ego_speed_mps": ego_speed,
            "limit_speed_mps": limit_speed,
            "overspeed_mps": overspeed,
            "accel_in": accel,
            "accel_out": accel_new,
            "clip_norm": float(np.sum(np.abs(clip))),
        }
        return safe, clip, info


__all__ = ["SpeedLimitShield", "SpeedLimitShieldConfig"]
