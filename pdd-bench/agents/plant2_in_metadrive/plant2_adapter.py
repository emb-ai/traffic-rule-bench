"""PlanT2 <-> MetaDrive adapter — mirror of CaRLMetaDriveAdapter.

Holds the PlanT2 model and wraps the obs->action conversion, so a higher-level
policy class (e.g. PlanT2SignCompliantPolicy) can call `get_action(vehicle, engine)`
each step without thinking about plant2 batch construction or path setup.

The adapter is meant to be created once per process and shared across episodes
(model is heavy). Construction is lazy: the PlanT2 model is only loaded on the
first `get_action()` call.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _ensure_plant2_paths(plant_repo_dir: Path) -> None:
    """Insert plant2 dirs onto sys.path so `carla_garage.plant2_control` and
    PlanT's `model` module become importable. Idempotent."""
    plant_repo_dir = Path(plant_repo_dir).resolve()
    plant_planT_dir = plant_repo_dir / "PlanT"
    for p in (plant_repo_dir, plant_planT_dir):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)


# --- pure-pursuit constants (match eval_plant2_wps_steer.py) ---
_SPEED_BINS = np.array(
    [0.0, 0.025, 0.05472609, 1.0, 1.5, 2.0, 4.0, 8.0, 10.0, 20.0],
    dtype=np.float32,
)
_WHEELBASE_M = 2.5
_LOOKAHEAD_IDX = 1


def _wps_to_action(pred_plan, current_speed: float, target_speed_mps: float) -> np.ndarray:
    """Compute MetaDrive [steer, throttle] directly from pred_wps via pure-pursuit.

    Mirrors eval_plant2_wps_steer.py:_wps_to_action — bypasses LateralPID/PCHIP
    used by plant2_predictions_to_action. Steering from pred_wps[lookahead_idx]
    via pure-pursuit; throttle from softmax(pred_speed) capped by target_speed.
    Ego frame: x=forward, y=right (CARLA / metadrive_obs_to_plant2 convention).
    MetaDrive action[0]: +1=left, -1=right.
    """
    pred_path, pred_wps, pred_speed = pred_plan

    wps_tensor = pred_wps if pred_wps is not None else pred_path
    steer = 0.0
    wps_np = None
    if wps_tensor is not None:
        wps_np = wps_tensor.detach().cpu().numpy()
        if wps_np.ndim > 2:
            wps_np = wps_np.squeeze(0)
        if wps_np.shape[0] > 0:
            idx = min(_LOOKAHEAD_IDX, wps_np.shape[0] - 1)
            tx, ty = float(wps_np[idx, 0]), float(wps_np[idx, 1])
            dist = max(np.hypot(tx, ty), 1e-3)
            alpha = np.arctan2(ty, max(tx, 1e-3))
            delta = np.arctan2(2.0 * _WHEELBASE_M * np.sin(alpha), dist)
            steer = float(np.clip(delta / (np.pi / 4.0), -1.0, 1.0))
            # pred_wps use CARLA y=right; MetaDrive steer +1=left (mirrors plant2_control.py:198)
            steer = -steer

    if pred_speed is not None:
        logits = pred_speed.detach().float()
        if logits.dim() > 1:
            logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        desired_speed = float((probs * _SPEED_BINS).sum())
    elif wps_np is not None and wps_np.shape[0] >= 4:
        desired_speed = float(np.linalg.norm(wps_np[2] - wps_np[3]) * 4.0)
        if current_speed < 0.01:
            mean_sp = float(np.linalg.norm(wps_np[:-1] - wps_np[1:], axis=-1).mean() * 4.0)
            desired_speed = min(mean_sp, 0.1)
    else:
        desired_speed = target_speed_mps

    desired_speed = min(desired_speed, target_speed_mps)

    if desired_speed < 0.05:
        throttle_brake = -0.5
    else:
        speed_err = desired_speed - current_speed
        throttle_brake = float(np.clip(speed_err * 0.5, -1.0, 1.0))

    return np.array([steer, throttle_brake], dtype=np.float32)


class PlanT2MetaDriveAdapter:
    """Lazy-loaded PlanT2 model wrapper that produces MetaDrive [steering, throttle]."""

    def __init__(
        self,
        checkpoint_path: str,
        plant_repo_dir,
        device: str = "cpu",
        action_mode: str = "wps_pure_pursuit",
        max_speed_kmh: Optional[int] = 50,
        route_step_m: float = 1.0,
    ):
        if action_mode not in ("pid", "wps_pure_pursuit"):
            raise ValueError(
                f"action_mode must be 'pid' or 'wps_pure_pursuit', got {action_mode!r}"
            )
        self.checkpoint_path = str(checkpoint_path)
        self.plant_repo_dir = Path(plant_repo_dir)
        self.plant_planT_dir = self.plant_repo_dir / "PlanT"
        self.device = device
        self.action_mode = action_mode
        # Speed cap: None = use batch speed_limit as-is; int (km/h) = override target speed.
        # Supported values matching SPEED_CATS: 50, 80, 100, 120.  Other values default to 80.
        self.max_speed_kmh: Optional[int] = max_speed_kmh
        # Route lookahead: step_m=1.0 samples 20m ahead (default training value).
        # pg_direction maps have turns 37m+ ahead → step_m=4.0 gives 80m lookahead so
        # the model can see and react to intersection turns.
        self.route_step_m: float = route_step_m
        self._model = None
        self._config = None
        # Persistent controllers for action_mode="pid". Created once (lazy) and
        # reused across steps so the lateral PID's error-history window survives
        # between frames (restores the derivative-damping term). Mirrors the
        # canonical CARLA PlanT agent, which holds lat_pid/lon_pid as members.
        self._lat_pid = None
        self._lon_pid = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        _ensure_plant2_paths(self.plant_repo_dir)

        import os as _os
        import yaml as _yaml
        import importlib.util as _ilu
        import unittest.mock as _mock

        # Mock CARLA dependencies (same as all eval scripts).
        for _mod_name in ("carla", "agents", "agents.navigation",
                          "agents.navigation.global_route_planner"):
            if _mod_name not in sys.modules:
                sys.modules[_mod_name] = _mock.MagicMock()

        # ── Step 1: inspect checkpoint keys ──────────────────────────────────
        # Load raw state-dict to detect which optional components were saved.
        # We do this BEFORE building HFLM so the config can be patched correctly.
        _raw = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        _sd = _raw.get("state_dict", _raw.get("model_state_dict", _raw))
        if not isinstance(_sd, dict):
            raise ValueError("Checkpoint contains no state_dict")
        if list(_sd.keys())[0].startswith("model."):
            _sd = {k.replace("model.", "", 1): v for k, v in _sd.items()}

        # speed_classifier / ego_speed_classifier → pred_speed head exists
        _speed_keys = {k for k in _sd if "speed_classifier" in k}
        self._has_trained_speed_head: bool = bool(_speed_keys)

        # ego_speed_emb → checkpoint was trained with input_ego_speed=True
        _has_ego_speed_emb: bool = any("ego_speed_emb" in k for k in _sd)

        # speed_token → new-style HFLM that appends a dedicated speed token at end
        _has_speed_token: bool = "speed_token" in _sd

        print(
            f"[PlanT2Adapter] ckpt keys: speed_classifier={self._has_trained_speed_head}  "
            f"ego_speed_emb={_has_ego_speed_emb}  speed_token={_has_speed_token}"
        )
        if not self._has_trained_speed_head:
            print(
                "[PlanT2Adapter] WARNING: no speed_classifier in ckpt — "
                "pred_speed will be ignored (waypoint-spacing fallback)."
            )

        # ── Step 2: build config, patching input_ego_speed from checkpoint ───
        # PlanT.yaml may be stale (the model forward was rewritten after training).
        # Derive input_ego_speed from the checkpoint itself, not from the YAML.
        _model_yaml = _os.path.join(str(self.plant_planT_dir), "config", "model", "PlanT.yaml")
        if not _os.path.isfile(_model_yaml):
            raise FileNotFoundError(f"PlanT config not found: {_model_yaml}")
        with open(_model_yaml) as _f:
            _plnt = _yaml.safe_load(_f)

        class _DictAsMember(dict):
            def __getattr__(self, name):
                value = self.get(name)
                if isinstance(value, dict) and not isinstance(value, _DictAsMember):
                    return _DictAsMember(value)
                return value

        config_all = _DictAsMember({"model": _plnt})
        # Patch input_ego_speed: honour checkpoint over stale YAML
        if config_all["model"].get("training") is None:
            config_all["model"]["training"] = {}
        config_all["model"]["training"]["input_ego_speed"] = _has_ego_speed_emb
        # Store for get_action() so it doesn't re-read the (now-correct) config
        self._input_ego_speed: bool = _has_ego_speed_emb

        # ── Step 3: instantiate HFLM with the corrected config ────────────────
        _model_py = self.plant_planT_dir / "model.py"
        if not _model_py.exists():
            raise FileNotFoundError(f"PlanT model.py not found: {_model_py}")
        _spec = _ilu.spec_from_file_location("plant2_adapter_hflm", str(_model_py))
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        HFLM = _mod.HFLM

        net = HFLM(config_all.model.network, config_all)
        net.load_state_dict(_sd, strict=False)
        del _raw  # free raw checkpoint memory

        # ── Step 4: fix randomly-initialised speed_token for old checkpoints ──
        # Old HFLM didn't have speed_token; the new forward always appends it to
        # the sequence.  If the checkpoint lacks it the parameter is random noise
        # → zero it out so it doesn't corrupt the attention of the WP tokens.
        if not _has_speed_token and hasattr(net, "speed_token"):
            print(
                "[PlanT2Adapter] INFO: speed_token absent in ckpt — "
                "zeroing parameter to suppress noise in attention."
            )
            with torch.no_grad():
                net.speed_token.data.zero_()

        net = net.to(self.device)
        net.eval()
        self._model = net
        self._config = config_all

    def reset(self) -> None:
        """Clear the persistent lateral PID error-history window between episodes.

        The controllers persist across episodes (created once); only the sliding
        window must be cleared so a new episode doesn't inherit stale heading
        errors. (_lon_pid is a stateless linear-regression controller — no reset.)"""
        if self._lat_pid is not None:
            self._lat_pid.error_history = []

    def get_action(self, vehicle, engine) -> np.ndarray:
        """Run one PlanT2 inference step → MetaDrive `[steering, throttle]` in [-1, 1]."""
        self._ensure_loaded()

        from metadrive.policy.metadrive_obs_to_plant2 import metadrive_obs_to_plant2_batch
        from carla_garage.plant2_control import get_target_speed_from_limit

        # input_ego_speed is detected from the checkpoint in _ensure_loaded()
        # and stored in self._input_ego_speed — do NOT re-read from PlanT.yaml
        # (that config may be stale since the forward was rewritten after training).
        input_ego_speed = getattr(self, "_input_ego_speed", False)

        # Pre-compute route with configurable step_m.
        # get_route_points_ego_frame already returns CARLA convention (y=right),
        # matching collect_objects_ego_frame — do NOT flip y here.
        from metadrive.policy.plant_policy import get_route_points_ego_frame
        route_ego, _ = get_route_points_ego_frame(vehicle, num_points=20, step_m=self.route_step_m)

        # Mirrors the reference inference path in eval_plant2_rule_sign_speed_probs.py:736-748.
        # max_objects=30 — PlanT2 must see surrounding NPCs and sign tokens via x_objs;
        # 0 silently degrades it to "drive forward by BEV alone".
        batch = metadrive_obs_to_plant2_batch(
            engine,
            vehicle,
            route_ego_20x2=route_ego,
            speed_limit_kmh=None,     # keep model input unchanged (80 km/h default token)
            max_objects=30,
            max_distance=75.0,
            range_factor_front=16.0,
            input_bev=True,
            input_ego_speed=input_ego_speed,
            bev_resolution=128,
            bev_size_meters=64.0,
            device=self.device,
        )

        with torch.no_grad():
            _, _, pred_plan, _ = self._model(batch)

        # If the speed classifier head was not saved in the checkpoint its weights are
        # random — null out pred_speed so _wps_to_action falls back to waypoint spacing.
        if not getattr(self, "_has_trained_speed_head", True):
            pred_path, pred_wps, _ = pred_plan
            pred_plan = (pred_path, pred_wps, None)

        ego_speed = float(getattr(vehicle, "speed", 0.0))
        speed_limit_idx = int(batch["speed_limit"][0].item())
        target_speed_mps = get_target_speed_from_limit(speed_limit_idx)

        # Apply max_speed_kmh cap without changing the model's batch input.
        if self.max_speed_kmh is not None:
            target_speed_mps = min(target_speed_mps, self.max_speed_kmh / 3.6)

        if self.action_mode == "wps_pure_pursuit":
            action = _wps_to_action(pred_plan, ego_speed, target_speed_mps)
        else:
            action = self._pid_action_persistent(pred_plan, ego_speed, target_speed_mps)

        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    def _ensure_controllers(self) -> None:
        """Lazily create the persistent lateral/longitudinal controllers.

        Imported the same way carla_garage.plant2_control imports them internally
        (carla_garage is already on sys.path once the model is loaded)."""
        if self._lat_pid is not None:
            return
        from config import GlobalConfig
        from lateral_controller import LateralPIDController
        from longitudinal_controller import LongitudinalLinearRegressionController
        cfg = GlobalConfig()
        self._lat_pid = LateralPIDController(cfg)
        self._lon_pid = LongitudinalLinearRegressionController(cfg)

    def _pid_action_persistent(self, pred_plan, current_speed, target_speed_mps) -> np.ndarray:
        """Mirror of carla_garage.plant2_control.plant2_predictions_to_action, with
        two faithfulness fixes vs that per-step function:

          1. Reuses PERSISTENT controllers (self._lat_pid/_lon_pid) instead of
             re-instantiating each step, so the lateral PID's error_history
             accumulates across frames and its derivative (damping) term works —
             this is what kills the lane-change wobble. Matches the canonical
             CARLA PlanT agent (PlanT_agent.py:_get_control).
          2. Applies FULL brake (-1.0 == CARLA control.brake=1.0) instead of the
             half-brake (-0.5) the per-step function used.

        Helpers (interpolate_waypoints, SPEED_BINS) are reused from the submodule —
        plant2 is NOT modified."""
        from carla_garage.plant2_control import interpolate_waypoints, SPEED_BINS

        self._ensure_controllers()
        pred_path, pred_wps, pred_speed = pred_plan

        # 1. Desired speed (softmax over 8 speed bins, else waypoint-spacing heuristic)
        if pred_speed is not None:
            logits = pred_speed.detach().float()
            if logits.dim() > 1:
                logits = logits.squeeze(0)
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            desired_speed = float((probs * SPEED_BINS).sum())
        else:
            _wp = pred_wps if pred_wps is not None else pred_path
            if _wp is not None:
                wp_arr = _wp.detach().cpu().numpy()
                if wp_arr.ndim > 2:
                    wp_arr = wp_arr.squeeze(0)
                if len(wp_arr) >= 4:
                    desired_speed = float(np.linalg.norm(wp_arr[2] - wp_arr[3]) * 4.0)
                    mean_speed = float(np.linalg.norm(wp_arr[:-1] - wp_arr[1:], axis=-1).mean() * 4.0)
                    if current_speed < 0.01:
                        desired_speed = min(mean_speed, 0.1)
                else:
                    desired_speed = target_speed_mps
            else:
                desired_speed = target_speed_mps
        desired_speed = min(desired_speed, target_speed_mps)

        # 2. Longitudinal control (persistent, stateless linear-regression controller)
        hazard_brake = desired_speed < 0.05
        throttle, brake = self._lon_pid.get_throttle_and_brake(
            hazard_brake, desired_speed, current_speed
        )

        # 3. Select steering waypoints — prefer pred_path, fallback pred_wps
        steer_tensor = pred_path if pred_path is not None else pred_wps
        if steer_tensor is None:
            return np.array([0.0, 0.5], dtype=np.float32)
        steer_np = steer_tensor.detach().cpu().numpy()
        if steer_np.ndim > 2:
            steer_np = steer_np.squeeze(0)
        if steer_np.shape[0] == 0:
            return np.array([0.0, 0.5], dtype=np.float32)

        # 4. Interpolate steering path at 0.1 m (PCHIP) — reused from submodule
        interp_wp = interpolate_waypoints(steer_np)

        # 5. Lateral control (persistent PID; creep dummy path when stopped + braking)
        if current_speed < 0.05 and brake:
            steer_input = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
                                   dtype=np.float32)
        else:
            steer_input = interp_wp
        steer = self._lat_pid.step(
            steer_input, current_speed, np.array([0.0, 0.0]), 0.0, False
        )

        # 6. Assemble MetaDrive action
        steer = np.clip(float(steer), -1.0, 1.0)
        steer = -steer                       # PID positive=right; MetaDrive action[0] positive=left
        if brake:
            throttle_brake = -1.0            # full brake (= CARLA control.brake=1.0); was -0.5 half-brake
        else:
            throttle_brake = float(np.clip(throttle, 0.0, 1.0))
        throttle_brake = float(np.clip(throttle_brake, -1.0, 1.0))
        return np.array([steer, throttle_brake], dtype=np.float32)
