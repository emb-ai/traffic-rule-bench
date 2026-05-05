"""
StopRuleShield:   STOP sign handling for carl 
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np

from stop_rule_feat import StopRuleFeatureExtractor


@dataclass
class StopShieldConfig:
    max_brake_norm: float = 1.0     
    hold_brake_norm: float = 0.7  
    release_accel_norm: float = 0.3 
    hold_dist_m: float = 0.7    
    post_clear_dist_m: float = 3.0 
    min_required_decel_norm: float = 0.15 

class StopRuleShield:

    def __init__(self, rule_extractor: StopRuleFeatureExtractor, cfg: StopShieldConfig = None):
        self.fx = rule_extractor
        self.cfg = cfg or StopShieldConfig()

    def clip_action(
        self,
        z: np.ndarray,
        carl_action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:

        must_stop = z[self.fx.IDX_MUST_STOP] > 0.5
        dist_norm = float(z[self.fx.IDX_DIST_TO_STOP])
        stop_cleared = z[self.fx.IDX_STOP_CLEARED] > 0.5
        hold_remaining = float(z[self.fx.IDX_HOLD_REMAINING])
        req_decel_norm = float(z[self.fx.IDX_REQUIRED_DECEL])


        dist_m = dist_norm * self.fx.dist_norm_max_m

        accel = float(carl_action[0])
        steer = float(carl_action[1])

        accel_new = accel

        sign_ahead = dist_m > 0.0 and dist_m < self.fx.approach_dist_m * 1.5  

        # enforce braking
        if must_stop and not stop_cleared:
            target_brake = -self.cfg.max_brake_norm * max(req_decel_norm, self.cfg.min_required_decel_norm)
            accel_new = min(accel_new, target_brake)

            # close to  line
            if dist_m <= self.cfg.hold_dist_m:
                accel_new = min(accel_new, -self.cfg.hold_brake_norm)

            # gentle brake
            if hold_remaining > 0.0 and abs(accel_new) < self.cfg.hold_brake_norm:
                accel_new = -self.cfg.hold_brake_norm
        elif sign_ahead and not stop_cleared and req_decel_norm > 0.01:

            target_brake = -self.cfg.max_brake_norm * 0.5 * max(req_decel_norm, self.cfg.min_required_decel_norm * 0.5)
            accel_new = min(accel_new, target_brake)

        if stop_cleared:
            accel_new = max(accel_new, self.cfg.release_accel_norm)
            if dist_m < -self.cfg.post_clear_dist_m:
                accel_new = accel  

        safe = np.array(
            [np.clip(accel_new, -1.0, 1.0), np.clip(steer, -1.0, 1.0)],
            dtype=np.float32,
        )
        clip = safe - carl_action

        info = {
            "must_stop": must_stop,
            "stop_cleared": stop_cleared,
            "dist_to_stop_m": dist_m,
            "req_decel_norm": req_decel_norm,
            "clip_norm": float(np.sum(np.abs(clip))),
        }
        return safe, clip, info


__all__ = ["StopRuleShield", "StopShieldConfig"]
