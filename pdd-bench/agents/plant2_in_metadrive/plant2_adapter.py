"""PlanT2 <-> MetaDrive adapter — mirror of CaRLMetaDriveAdapter.

Holds the PlanT2 model and wraps the obs->action conversion, so a higher-level
policy class (e.g. PlanT2SignCompliantPolicy) can call `get_action(vehicle, engine)`
each step without thinking about plant2 batch construction or path setup.

The adapter is meant to be created once per process and shared across episodes
(model is heavy). Construction is lazy: the PlanT2 model is only loaded on the
first `get_action()` call.
"""

from __future__ import annotations

import json
import os
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


# --- pure-pursuit constants (match PlanTVariables.target_speeds / bins_speed=8) ---
_SPEED_BINS = np.array(
    [0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0],
    dtype=np.float32,
)
_WHEELBASE_M = 2.5
_LOOKAHEAD_IDX = 1

# Pure pursuit aims at a point on the predicted trajectory. Picking it by index
# ties the aim distance to the waypoint spacing: at WPS_STRIDE=1 index 1 sits
# ~1.8 m ahead, which is inside the wheelbase's turning scale and makes the
# loop oscillate -- measured as steer_delta 0.032 against the expert's 0.047 at
# a third of its smoothness, i.e. the car wobbles instead of changing lane.
# Choosing the point by ARC LENGTH, growing with speed, is the textbook form and
# is independent of how the model was trained.
_LOOKAHEAD_K_V = float(os.environ.get("PLANT2_LOOKAHEAD_K_V", 0.8))    # seconds of travel
_LOOKAHEAD_L0_M = float(os.environ.get("PLANT2_LOOKAHEAD_L0_M", 3.0))  # floor at standstill
_LOOKAHEAD_MIN_M = float(os.environ.get("PLANT2_LOOKAHEAD_MIN_M", 4.0))
_LOOKAHEAD_MAX_M = float(os.environ.get("PLANT2_LOOKAHEAD_MAX_M", 12.0))


def _lookahead_point(wps_np: np.ndarray, speed_mps: float):
    """Point at the speed-scaled lookahead distance along the predicted path.

    Falls back to the far end when the path is shorter than the target distance
    rather than extrapolating: a short trajectory is exactly where extrapolation
    is least trustworthy.
    """
    if wps_np is None or wps_np.shape[0] == 0:
        return None
    pts = np.vstack([np.zeros((1, 2), dtype=wps_np.dtype), wps_np[:, :2]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    target = float(np.clip(_LOOKAHEAD_K_V * max(speed_mps, 0.0) + _LOOKAHEAD_L0_M,
                           _LOOKAHEAD_MIN_M, _LOOKAHEAD_MAX_M))
    if arc[-1] <= 1e-6:
        return None
    if target >= arc[-1]:
        return pts[-1]
    i = int(np.searchsorted(arc, target))
    lo, hi = arc[i - 1], arc[i]
    t = 0.0 if hi <= lo else (target - lo) / (hi - lo)
    return pts[i - 1] + t * (pts[i] - pts[i - 1])


def _env_float_or_none(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def _apply_stop_prob_threshold(probs: np.ndarray, desired_speed: float) -> float:
    """Force desired_speed=0 when stop-bin mass is high / other-bin mass is low.

    Env (default off / None):
      PLANT2_STOP_PROB_THR      — if probs[0] >= thr → stop
      PLANT2_STOP_OTHER_MASS_THR — if sum(probs[1:]) <= thr → stop
    """
    stop_thr = _env_float_or_none("PLANT2_STOP_PROB_THR")
    other_thr = _env_float_or_none("PLANT2_STOP_OTHER_MASS_THR")
    if stop_thr is None and other_thr is None:
        return desired_speed
    p0 = float(probs[0]) if probs is not None and len(probs) else 0.0
    other = float(probs[1:].sum()) if probs is not None and len(probs) > 1 else 1.0
    force = False
    if stop_thr is not None and p0 >= stop_thr:
        force = True
    if other_thr is not None and other <= other_thr:
        force = True
    return 0.0 if force else desired_speed


def _maybe_log_speed_pred(
    probs: np.ndarray,
    desired_speed: float,
    ego_speed: float,
    extra: Optional[dict] = None,
) -> None:
    """Append one JSON line when PLANT2_SPEED_LOG_PATH is set."""
    path = os.environ.get("PLANT2_SPEED_LOG_PATH")
    if not path:
        return
    rec = {
        "probs": [float(x) for x in probs.tolist()],
        "p0": float(probs[0]),
        "other_mass": float(probs[1:].sum()) if len(probs) > 1 else 0.0,
        "desired_speed": float(desired_speed),
        "ego_speed": float(ego_speed),
    }
    if extra:
        rec.update(extra)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


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
            aim = _lookahead_point(wps_np, current_speed)
            if aim is None:
                idx = min(_LOOKAHEAD_IDX, wps_np.shape[0] - 1)
                aim = wps_np[idx, :2]
            tx, ty = float(aim[0]), float(aim[1])
            dist = max(np.hypot(tx, ty), 1e-3)
            alpha = np.arctan2(ty, max(tx, 1e-3))
            delta = np.arctan2(2.0 * _WHEELBASE_M * np.sin(alpha), dist)
            steer = float(np.clip(delta / (np.pi / 4.0), -1.0, 1.0))
            # pred_wps use CARLA y=right; MetaDrive steer +1=left (mirrors plant2_control.py:198)
            steer = -steer

    probs = None
    if pred_speed is not None:
        logits = pred_speed.detach().float()
        if logits.dim() > 1:
            logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        desired_speed = float((probs * _SPEED_BINS).sum())
        desired_speed = _apply_stop_prob_threshold(probs, desired_speed)
        _maybe_log_speed_pred(probs, desired_speed, current_speed)
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
        lateral_lookahead_scale: float = 2.0,
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
        # Lateral PID lookahead multiplier. CARLA tuned the lookahead for 20 Hz;
        # MetaDrive runs the ego at ~10 Hz (decision_repeat=5), so the native
        # ~4.5 m lookahead is too short -> lateral weave and missed turns at
        # junctions. 2.0 compensates (3.24: route tracking settles ~±0.5 m vs
        # ±5 m and divergence at 1.0). Env PLANT2_LOOKAHEAD_MULT overrides.
        self._lateral_lookahead_scale: float = float(lateral_lookahead_scale)
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

        # sign_emb → checkpoint trained with explicit PDD sign_id token
        _has_sign_emb: bool = any(k.startswith("sign_emb.") for k in _sd)
        self._use_sign_id: bool = _has_sign_emb

        print(
            f"[PlanT2Adapter] ckpt keys: speed_classifier={self._has_trained_speed_head}  "
            f"ego_speed_emb={_has_ego_speed_emb}  speed_token={_has_speed_token}  "
            f"sign_emb={_has_sign_emb}"
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

        import os as _os
        if _os.environ.get("PLANT2_ROUTE_YFLIP"):
            # A/B test: re-apply the pre-1300c1e route y-flip (route -> MetaDrive
            # y=left). If this stops the pred_path oscillation, the checkpoint was
            # trained expecting the route in y=left, and 1300c1e's route change is
            # wrong for it. Temporary diagnostic toggle.
            route_ego = route_ego.copy()
            route_ego[:, 1] = -route_ego[:, 1]
        if _os.environ.get("PLANT2_DEBUG_STEER"):
            _r = route_ego[:, 1]
            print(f"[routedbg] route_y[min/max/last]="
                  f"{_r.min():+.2f}/{_r.max():+.2f}/{_r[-1]:+.2f}", flush=True)

        # Mirrors the reference inference path in eval_plant2_rule_sign_speed_probs.py:736-748.
        # max_objects=30 — PlanT2 must see surrounding NPCs and sign tokens via x_objs;
        # 0 silently degrades it to "drive forward by BEV alone".
        batch = metadrive_obs_to_plant2_batch(
            engine,
            vehicle,
            route_ego_20x2=route_ego,
            speed_limit_kmh=None,     # keep model input unchanged (80 km/h default token)
            max_objects=30,
            # Training filtered objects with range 50 / front factor 2
            # (PlanT.yaml model.training). These eval defaults are wider; the env
            # vars exist to A/B whether that train/eval gap costs anything.
            # Training filters at range 50 with front factor 2 (PlanT.yaml
            # model.training). The eval defaults used to be 75 and 16, a forward
            # ellipse an order of magnitude longer than anything the model was
            # trained on; the env vars keep that available for an A/B.
            max_distance=float(_os.environ.get("PLANT2_OBJ_MAX_DIST", 50.0)),
            range_factor_front=float(_os.environ.get("PLANT2_OBJ_FRONT_FACTOR", 2.0)),
            input_bev=True,
            input_ego_speed=input_ego_speed,
            bev_resolution=128,
            # The dump writes 256 px over 64 m and PlanTDataset keeps the central
            # 128 px (dataset.py: bev[0, 64:-64, 64:-64]), so training sees 32 m
            # at 0.25 m/px. Rendering 128 px over 64 m here gives the same tensor
            # shape at half the zoom — twice the area, silently. PLANT2_BEV_METERS=32
            # reproduces the training geometry; the default keeps the old behaviour
            # so the difference can be A/B'd.
            bev_size_meters=float(_os.environ.get("PLANT2_BEV_METERS", 64.0)),
            device=self.device,
            # PLANT2_SIGN_TOKEN=0 drops the global sign token, so the A/B can
            # separate it from the per-object PDD classes (PLANT2_SIGN_OBJS).
            include_sign_id=bool(getattr(self, "_use_sign_id", False))
            and _os.environ.get("PLANT2_SIGN_TOKEN", "1") not in ("0", "false", "False"),
            # PLANT2_FORCE_SIGN_CODE rewrites the sign token without touching
            # the geometry: the counterfactual that proved the speed channel is
            # read, never yet run for detour. 4.2.1 and 4.2.2 prescribe opposite
            # sides, so swapping them must flip the predicted manoeuvre if the
            # model uses the sign at all.
            sign_code=(_os.environ.get("PLANT2_FORCE_SIGN_CODE")
                       or getattr(self, "sign_code", None)),
        )

        # Object pool as the model sees it — to compare the eval-side convention
        # against the training dumps (boxes/NNNN.json.gz) row by row.
        if _os.environ.get("PLANT2_DEBUG_OBJS"):
            _x = batch.get("x_objs")
            if _x is not None:
                _rows = _x[0] if _x.dim() == 3 else _x
                for _r in _rows.tolist():
                    if _r[0] > 0:
                        print(f"[objdbg] type={_r[0]:.0f} x={_r[1]:+.1f} y={_r[2]:+.1f} "
                              f"yaw={_r[3]:+.0f} spd={_r[4]:.1f}", flush=True)
                print("[objdbg] ---", flush=True)

        with torch.no_grad():
            _, _, pred_plan, _ = self._model(batch)

        # If the speed classifier head was not saved in the checkpoint its weights are
        # random — null out pred_speed so _wps_to_action falls back to waypoint spacing.
        if not getattr(self, "_has_trained_speed_head", True):
            pred_path, pred_wps, _ = pred_plan
            pred_plan = (pred_path, pred_wps, None)

        # The dumps used for finetuning store route AND targets in MetaDrive
        # convention (y=left), while the controllers below read the prediction as
        # CARLA (y=right). Flipping only the route input leaves the output
        # mirrored — steering comes out inverted. This flips the prediction too,
        # so route in / path out can be put in one convention together.
        if _os.environ.get("PLANT2_PRED_YFLIP"):
            _pp, _pw, _ps = pred_plan
            if _pp is not None:
                _pp = _pp.clone(); _pp[..., 1] = -_pp[..., 1]
            if _pw is not None:
                _pw = _pw.clone(); _pw[..., 1] = -_pw[..., 1]
            pred_plan = (_pp, _pw, _ps)

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

        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        trace_dir = _os.environ.get("PLANT2_TRACE_DIR")
        if trace_dir:
            self._append_trace(trace_dir, vehicle, engine, batch, pred_plan, action,
                               ego_speed, route_ego)

        return action

    def _append_trace(self, trace_dir, vehicle, engine, batch, pred_plan, action,
                      ego_speed, route_ego=None):
        """Per-step JSONL trace: command, plan and visible objects.

        Aggregate episode metrics cannot separate a plan that oscillates from a
        controller that oscillates, nor tell an obstacle-driven offset from an
        offset that happens to have the right average. Both need the step series.
        """
        import json as _json
        import os as _os

        try:
            _os.makedirs(trace_dir, exist_ok=True)
            pred_path, pred_wps, _ = pred_plan

            def _xy(t):
                if t is None:
                    return []
                a = t.detach().cpu().numpy()
                if a.ndim > 2:
                    a = a.squeeze(0)
                return [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in a[:, :2]]

            objs = []
            x_objs = batch.get("x_objs")
            if x_objs is not None:
                rows = x_objs[0] if x_objs.dim() == 3 else x_objs
                for r in rows.tolist():
                    if r[0] > 0:
                        objs.append([round(r[0], 1), round(r[1], 2), round(r[2], 2),
                                     round(r[4], 1)])

            pos = getattr(vehicle, "position", (0.0, 0.0))
            rec = {
                "ep_step": int(getattr(engine, "episode_step", -1)),
                "seed": getattr(engine, "current_seed", None),
                "ego_x": round(float(pos[0]), 3),
                "ego_y": round(float(pos[1]), 3),
                "heading": round(float(getattr(vehicle, "heading_theta", 0.0)), 4),
                "speed": round(ego_speed, 3),
                "steer": round(float(action[0]), 4),
                "throttle": round(float(action[1]), 4),
                "wps": _xy(pred_wps),
                "path": _xy(pred_path),
                "objs": objs,
                # The route the model is given here, to compare against the route
                # stored in the training dumps for the same scene.
                "route": ([[round(float(p[0]), 3), round(float(p[1]), 3)]
                           for p in route_ego] if route_ego is not None else []),
            }
            fname = _os.path.join(trace_dir, f"trace_{_os.getpid()}.jsonl")
            with open(fname, "a") as fh:
                fh.write(_json.dumps(rec) + "\n")
        except Exception as exc:      # tracing must never break an eval run
            if not getattr(self, "_trace_warned", False):
                self._trace_warned = True
                print(f"[trace] disabled after error: {exc}", flush=True)

    def _ensure_controllers(self) -> None:
        """Lazily create the persistent lateral/longitudinal controllers.

        Imported the same way carla_garage.plant2_control imports them internally
        (carla_garage is already on sys.path once the model is loaded)."""
        if self._lat_pid is not None:
            return
        import os as _os
        from config import GlobalConfig
        from lateral_controller import LateralPIDController
        from longitudinal_controller import LongitudinalLinearRegressionController
        cfg = GlobalConfig()
        # Scale the lateral PID lookahead to compensate MetaDrive's ~10 Hz control
        # (CARLA tuned it for 20 Hz). Default = self._lateral_lookahead_scale (2.0);
        # a larger lookahead lowers the effective gain -> stable tracking + makes
        # junction turns. Env PLANT2_LOOKAHEAD_MULT overrides for experiments.
        _m = float(_os.environ.get("PLANT2_LOOKAHEAD_MULT", str(self._lateral_lookahead_scale)))
        if _m != 1.0:
            cfg.lateral_pid_speed_scale *= _m
            cfg.lateral_pid_speed_offset *= _m
            cfg.lateral_pid_minimum_lookahead_distance *= _m
            cfg.lateral_pid_maximum_lookahead_distance *= _m
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
            soft_desired = desired_speed
            desired_speed = _apply_stop_prob_threshold(probs, desired_speed)
            _maybe_log_speed_pred(
                probs,
                desired_speed,
                current_speed,
                extra={"soft_desired_speed": soft_desired, "ctrl": "pid"},
            )
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

        # Temporary diagnostic (env-gated): distinguish controller-wobble from
        # pred_path-wobble. path_y = lateral spread of the model's raw pred_path
        # (CARLA y=right). If path_y swings sign step-to-step → the MODEL's path
        # oscillates (input/convention/model issue). If path_y is smooth but
        # steer swings → controller. deriv = lateral PID derivative term.
        import os as _os
        if _os.environ.get("PLANT2_DEBUG_STEER"):
            _lat = steer_np[:, 1]
            _hist = self._lat_pid.error_history
            _deriv = 0.0 if len(_hist) < 2 else (_hist[-1] - _hist[-2])
            print(f"[steerdbg] v={current_speed:5.1f} steer={steer:+.3f} "
                  f"deriv={_deriv:+.4f} path_y[min/max/last]="
                  f"{_lat.min():+.2f}/{_lat.max():+.2f}/{_lat[-1]:+.2f} "
                  f"npts={steer_np.shape[0]} brake={int(bool(brake))}", flush=True)

        return np.array([steer, throttle_brake], dtype=np.float32)
