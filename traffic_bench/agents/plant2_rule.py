"""PlanT2-Sign-Compliant Policy — PlanT2 + sign compliance.

Normal driving comes from the PlanT2 transformer model (loaded from a
checkpoint). Sign compliance (stop, speed limit, priority, lane changes,
no-overtaking, etc.) is applied as post-processing by SignComplianceMixin.

Unlike CarlSignCompliantPolicy, lane-change override is enabled here
(APPLY_LANE_CHANGE_OVERRIDE = True) so the mixin can re-route ego at
5.15.2 direction signs and similar — same behavior as
RuleCompliantExpertPolicy. The no-overtaking guard from
RuleCompliantExpertPolicy is also reused.

Usage (from run_benchmark.py / replay_mini_new.py):
    PlanT2SignCompliantPolicy.set_checkpoint(
        "/path/to/plant2.ckpt",
        "/path/to/plant2_repo",
        device="cuda",
    )
    config["agent_policy"] = PlanT2SignCompliantPolicy

The PlanT2MetaDriveAdapter is loaded lazily and shared as a class
attribute: heavy net, only loaded once per process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.policy.base_policy import BasePolicy

from traffic_bench.agents.compliance.mixin import SignComplianceMixin


class PlanT2SignCompliantPolicy(SignComplianceMixin, BasePolicy):
    """PlanT2 transformer + SignComplianceMixin post-processing."""

    # --- shared PlanT2 adapter, loaded once per process ---
    _plant2_adapter = None
    _plant2_checkpoint: Optional[str] = None
    _plant2_repo_dir: Optional[Path] = None
    _plant2_device: str = "cpu"
    # "pid" (default, matches plant2_predictions_to_action — PCHIP + LateralPID
    # on pred_path) or "wps_pure_pursuit" (eval_plant2_wps_steer.py path —
    # pure-pursuit on pred_wps[1], throttle from softmax(pred_speed)).
    _plant2_action_mode: str = "pid"
    # Speed cap passed to metadrive_obs_to_plant2_batch as speed_limit_kmh.
    # None = use batch default (80 km/h).  Supported: 50, 80, 100, 120.
    _plant2_max_speed_kmh: Optional[int] = 50
    # Route step size (meters between sampled route points).
    # Default 1.0 = 20m lookahead (training distribution).
    # Use 4.0 for pg_direction maps where turns are 37m+ ahead → 80m lookahead.
    _plant2_route_step_m: float = 1.0

    # Let the mixin override steering for sign-driven lane changes
    # (5.15.2 direction signs, lane bans, etc.). Mirror RuleCompliantExpertPolicy.
    APPLY_LANE_CHANGE_OVERRIDE = True
    # Mid-route rule-based U-turn on 3.18.1/3.18.2 compliant detours only.
    APPLY_UTURN_ZONE_ASSIST = True

    # When False, skip ALL sign-compliance post-processing. Used by
    # PlainPlanT2Policy to expose raw PlanT2 behaviour without rule overlay.
    APPLY_RULE_OVERLAY = True

    # 5.15.1: keep MetaDrive checkpoints during peer LC (no [spawn, dest] stub).
    APPLY_LANE_DIRS_NAV_HOLD = False

    @classmethod
    def set_checkpoint(
        cls,
        checkpoint_path,
        plant_repo_dir,
        device: str = "cpu",
        action_mode: str = "wps_pure_pursuit",
        max_speed_kmh: Optional[int] = 50,
        route_step_m: float = 1.0,
    ):
        """Configure the PlanT2 checkpoint + plant2 repo path BEFORE env construction.

        Must be called before the first `agent_policy` instantiation. The
        adapter is created lazily on the first `act()`, so env setup stays fast.

        action_mode: "pid" (default) or "wps_pure_pursuit". See
        `_plant2_action_mode` class-attr docstring.
        max_speed_kmh: speed cap applied after model inference (km/h). None = no cap.
        route_step_m: meters between route sample points (default 1.0 = training distribution).
        """
        if action_mode not in ("pid", "wps_pure_pursuit"):
            raise ValueError(
                f"action_mode must be 'pid' or 'wps_pure_pursuit', got {action_mode!r}"
            )
        cls._plant2_checkpoint = str(checkpoint_path)
        cls._plant2_repo_dir = Path(plant_repo_dir)
        cls._plant2_device = device
        cls._plant2_action_mode = action_mode
        cls._plant2_max_speed_kmh = max_speed_kmh
        cls._plant2_route_step_m = route_step_m
        cls._plant2_adapter = None  # reset so next act() re-creates with new params

    @classmethod
    def _get_adapter(cls):
        if cls._plant2_adapter is None:
            if cls._plant2_checkpoint is None or cls._plant2_repo_dir is None:
                raise RuntimeError(
                    "PlanT2 checkpoint not configured. Call "
                    "PlanT2SignCompliantPolicy.set_checkpoint(<ckpt>, <plant2_repo>) "
                    "before env build."
                )
            from traffic_bench.agents.adapters.plant2.plant2_adapter import PlanT2MetaDriveAdapter
            cls._plant2_adapter = PlanT2MetaDriveAdapter(
                cls._plant2_checkpoint,
                cls._plant2_repo_dir,
                device=cls._plant2_device,
                action_mode=cls._plant2_action_mode,
                max_speed_kmh=cls._plant2_max_speed_kmh,
                route_step_m=cls._plant2_route_step_m,
            )
        return cls._plant2_adapter

    def __init__(self, control_object, random_seed=None, config=None):
        BasePolicy.__init__(self, control_object, random_seed, config)
        # PIDs used by the mixin for lane-change steering override.
        self._heading_pid = PIDController(1.7, 0.01, 3.5)
        self._lateral_pid = PIDController(0.3, 0.002, 0.05)
        self._init_sign_compliance()
        # New episode → reset adapter state (no-op for PlanT2, but consistent).
        try:
            self._get_adapter().reset()
        except Exception:
            pass

    # --- Abstract-method impl from SignComplianceMixin ---
    def _get_heading_pid(self):
        return self._heading_pid

    def _get_lateral_pid(self):
        return self._lateral_pid

    def act(self, agent_id=None):
        if self.APPLY_RULE_OVERLAY:
            self._process_signs()

        adapter = self._get_adapter()
        try:
            action = adapter.get_action(self.control_object, self.engine)
        except Exception as exc:
            print(f"[PlanT2SignCompliantPolicy] get_action failed: {exc}")
            action = np.asarray([0.0, 0.0], dtype=np.float32)

        steering = float(action[0])
        throttle = float(action[1])

        if self.APPLY_RULE_OVERLAY:
            # Steering override for sign-driven lane changes (mixin path).
            if self.APPLY_LANE_CHANGE_OVERRIDE:
                self._update_lane_change()
                self._arm_uturn_from_nav()
                uturn_armed = self._uturn_via_lane is not None
                if self._lc_target_lane is not None and not uturn_armed:
                    steering = float(np.clip(
                        self._steering_control_for_lc(self._lc_target_lane), -1.0, 1.0
                    ))
            else:
                self._arm_uturn_from_nav()

            steering = self._maybe_override_steering_for_uturn_zone(steering)

            # No-overtaking steering guard (matches RuleCompliantExpertPolicy:59-71):
            # if PlanT2 is steering toward the opposite lane while overtaking is
            # forbidden, clamp it back toward lane-following.
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

            # Clamp throttle by sign-driven _speed_cap / _speed_floor.
            throttle = self._apply_speed_constraints(
                throttle, self.control_object.speed_km_h
            )

        action_out = [steering, throttle]
        self.action_info["action"] = action_out
        return action_out
