"""PlanT2 frame capture helpers for live dump during expert_replay rollouts.

Layout written under ``plant2_dir``::

    data/<scene_uid>_<variant>/
        boxes/NNNN.json.gz
        measurements/NNNN.json.gz
        bev_no_car_semantics/NNNN.png              # semantic indices 0-4 (256²)
        bev_no_car_semantics_augmented/NNNN.png    # same when aug offsets are 0
        results.json.gz
    slurm/run_files/logs/qsub_out2025_07.log   # required by PlanTDataset(filter_routes=True)

Frames are captured *before* each ``env.step`` (PlanTDataset convention).

BEV is required for default PlanT training (``model.training.input_bev=True``):
``PlanTDataset`` loads the PNG, crops ``[64:-64, 64:-64]`` → 128², then
colours via ``PlanTVariables.bev_colors``.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
from pathlib import Path

import numpy as np

_MAX_TARGET_SPEED = 20.0  # max from PlanTVariables.target_speeds (m/s)
_TIMESTAMP = "2025_07_20_00_00_00"
_LOG_SUFFIX = "_".join(_TIMESTAMP.split("_")[:2])  # "2025_07"
# 256 so PlanTDataset crop 64:-64 yields 128 (model / CARLA convention).
_BEV_RESOLUTION = 256
_BEV_SIZE_METERS = 64.0


def build_ego_matrix(position, heading: float) -> list:
    """4×4 world transform: LOCAL=(x-forward, y-left)."""
    c, s = math.cos(heading), math.sin(heading)
    px, py = float(position[0]), float(position[1])
    return [
        [c, -s, 0.0, px],
        [s,  c, 0.0, py],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


# MetaDrive sign class → PDD code (fallback when icon_path / attrs missing).
_CLASS_TO_PDD = {
    "MainRoadSign": "2.1",
    "SecondaryRoadSign": "2.3.1",
    "SecondaryRoadRightSign": "2.3.2",
    "SecondaryRoadLeftSign": "2.3.3",
    "YieldSign": "2.4",
    "RightHandYieldSign": "2.4",
    "StopSign": "2.5",
    "NoEntrySign": "3.1",
    "NoTrafficSign": "3.2",
    "NoRightTurnSign": "3.18.1",
    "NoLeftTurnSign": "3.18.2",
    "SpeedLimitSign": "3.24",
    "SpeedLimitSign20": "3.24",
    "SpeedLimitSign30": "3.24",
    "SpeedLimitSign40": "3.24",
    "SpeedLimitSign60": "3.24",
    # The mandatory-direction plates carry their code in the class name, not in
    # the icon file (those are named "direction_straight.png" and the like), so
    # the icon regex cannot resolve them and every one of them was written as a
    # nameless box: 875 dumped routes across six families carried no sign at all.
    "LaneAllowedDirectionSign4_1_1": "4.1.1",
    "LaneAllowedDirectionSign4_1_2": "4.1.2",
    "LaneAllowedDirectionSign4_1_3": "4.1.3",
    "LaneAllowedDirectionSign4_1_4": "4.1.4",
    "LaneAllowedDirectionSign4_1_5": "4.1.5",
    "LaneAllowedDirectionSign4_1_6": "4.1.6",
    "DetourRightSign": "4.2.1",
    "DetourLeftSign": "4.2.2",
    "DetourEitherSign": "4.2.3",
    "RoundaboutSign": "4.3",
    "RoundaboutYieldSign": "4.3",
    "MinimumSpeedLimitSign": "4.6",
    "OneWayEntrySign": "5.7.1",
    "OneWayEntrySignR": "5.7.1",
    "OneWayEntrySignL": "5.7.2",
    "LaneDirectionsSign": "5.15.1",
    "DirectionSign": "5.15.2",
    "ResidentialZoneSign": "5.21",
    "ZoneSpeedLimitSign": "5.31",
    "ZoneSpeedLimitSign20": "5.31",
    "ZoneSpeedLimitSign30": "5.31",
    "ZoneSpeedLimitSign40": "5.31",
    "ZoneSpeedLimitSign60": "5.31",
}

_PDD_ICON_RE = re.compile(r"^(\d+(?:\.\d+)*)")

# How far a traffic sign stays in the ego-centric object list. Deliberately
# wider than max_distance: a sign kept only inside the generic object range
# reaches the model in a small share of frames and its token cannot be learnt
# (measured: 2% of frames at the narrow radius against 33% at 120 m). The zone
# is also symmetric, so a sign behind the ego is still the reason it is slowing.
_SIGN_RADIUS_M = float(os.environ.get("PLANT2_SIGN_RADIUS_M", 120.0))

# Codes whose plate carries a number, and the attribute holding it (km/h).
# Restricted by code on purpose: reading any `speed_limit` attribute that
# happened to exist would put a road speed on plates that prescribe nothing.
_SIGN_VALUE_ATTR = {
    "3.24": "speed_limit",   # maximum speed
    "5.31": "speed_limit",   # zone maximum speed
    "4.6": "min_speed",      # minimum speed
}


def _sign_value_kmh(sign, pdd):
    """The number written on the plate, in km/h, or None if it carries none."""
    attr = _SIGN_VALUE_ATTR.get(str(pdd))
    if attr is None:
        return None
    raw = getattr(sign, attr, None)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def resolve_pdd_code_from_sign(sign):
    """Best-effort PDD code for a BaseTrafficSign instance."""
    if sign is None:
        return None
    for attr in ("pdd_code", "sign_code", "sign_type", "code"):
        val = getattr(sign, attr, None)
        if val:
            return str(val).strip()
    icon = getattr(sign, "icon_path", None)
    if icon:
        m = _PDD_ICON_RE.match(Path(str(icon)).name)
        if m:
            return m.group(1)
    return _CLASS_TO_PDD.get(type(sign).__name__)


def collect_boxes(engine, vehicle,
                  max_distance: float = 50.0,
                  range_factor_front: float = 2.0) -> list:
    """Objects in ego frame for boxes/NNNN.json.gz.

    Ego is first (position=[0,0,0]). Positions use CARLA convention: x=forward,
    y=right. Yaw is radians relative to ego.
    """
    from metadrive.utils.math import wrap_to_pi
    from metadrive.component.vehicle.base_vehicle import BaseVehicle
    from metadrive.component.traffic_participants.pedestrian import Pedestrian
    from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight

    ego_pos = np.array(vehicle.position[:2])
    ego_heading = float(vehicle.heading_theta)
    ego_speed = float(getattr(vehicle, "speed", 0.0))
    ego_w = float(getattr(vehicle, "top_down_width", 2.0) or 2.0)
    ego_l = float(getattr(vehicle, "top_down_length", 4.0) or 4.0)

    boxes = [{
        "class": "car",
        # PlanTDataset indexes type_id for emergency-vehicle filtering.
        "type_id": "vehicle.tesla.model3",
        "position": [0.0, 0.0, 0.0],
        "yaw": 0.0,
        "speed": ego_speed,
        "extent": [ego_l / 2.0, ego_w / 2.0, 0.75],
        "id": 0,
    }]

    def _cls(obj) -> str:
        if isinstance(obj, BaseVehicle):
            return "car"
        if isinstance(obj, Pedestrian):
            return "walker"
        if isinstance(obj, BaseTrafficLight):
            return "traffic_light"
        return "static"

    obj_id = 1
    seen: set = set()

    # The plates are claimed before the generic sweep runs. `engine.get_objects`
    # hands them over like any other prop, and the sweep would file them as
    # `static` / constructioncone -- indistinguishable from the detour obstacle,
    # with the PDD code and the plate value lost. They are written below, each
    # under its own class.
    _sign_mgr = getattr(engine, "traffic_sign_manager", None)
    _sign_objs = [s for s in list(getattr(_sign_mgr, "signs", []) or [])
                  if s is not None and hasattr(s, "position")]
    _sign_ids = {id(s) for s in _sign_objs}

    def _add(obj):
        nonlocal obj_id
        oid = id(obj)
        if oid in seen or obj is vehicle:
            return
        if oid in _sign_ids:
            return
        seen.add(oid)
        if not hasattr(obj, "position"):
            return
        rel = np.array([obj.position[0] - ego_pos[0],
                        obj.position[1] - ego_pos[1]])
        local = vehicle.convert_to_local_coordinates(rel, 0.0)
        x = float(local[0])
        y = -float(local[1])  # y=right (CARLA; invert MetaDrive left)
        cls = _cls(obj)

        if cls == "traffic_light":
            if x * x + y * y > 900.0:  # >30m
                return
        else:
            x_div = range_factor_front ** 2 if x > 0.0 else 1.0
            if x * x / x_div + y * y > max_distance ** 2:
                return

        yaw = float(wrap_to_pi(float(obj.heading_theta) - ego_heading))
        speed = float(getattr(obj, "speed", 0.0))
        w = float(getattr(obj, "top_down_width", 2.0) or 2.0)
        l = float(getattr(obj, "top_down_length", 4.0) or 4.0)

        entry: dict = {
            "class": cls,
            "position": [x, y, 0.0],
            "yaw": yaw,
            "speed": speed,
            "extent": [l / 2.0, w / 2.0, 0.75],
            "id": obj_id,
        }
        # CARLA-compatible type_id (required by PlanTDataset; MetaDrive has no asset ids).
        if cls == "car":
            entry["type_id"] = "vehicle.tesla.model3"
        elif cls == "walker":
            entry["type_id"] = "walker.pedestrian.0001"
        elif cls == "static":
            entry["type_id"] = "static.prop.constructioncone"
        if cls == "traffic_light":
            from metadrive.constants import MetaDriveType
            status = getattr(obj, "status", MetaDriveType.LIGHT_UNKNOWN)
            if status not in (MetaDriveType.LIGHT_RED, MetaDriveType.LIGHT_YELLOW):
                return
            entry["state"] = ("Red" if status == MetaDriveType.LIGHT_RED else "Yellow")
            entry["affects_ego"] = True
            entry["type_id"] = "traffic.traffic_light"

        boxes.append(entry)
        obj_id += 1

    traffic_mgr = getattr(engine, "traffic_manager", None)
    if traffic_mgr is not None:
        for v in list(getattr(traffic_mgr, "vehicles", [])):
            _add(v)
    if hasattr(engine, "get_objects"):
        try:
            for obj in engine.get_objects(lambda o: hasattr(o, "position")).values():
                _add(obj)
        except Exception:
            pass
    obj_mgr = getattr(engine, "object_manager", None)
    if obj_mgr is not None:
        for obj in getattr(obj_mgr, "spawned_objects", {}).values():
            _add(obj)

    # Explicit PDD signs, written as their own object class rather than as the
    # generic `static` the loop above would give them. Without this the sign is
    # indistinguishable from a traffic cone in the dump, and no later fix
    # recovers it: the box was never written.
    try:
        from PlanT.util.sign_id import SIGN_CODES as _known
        known_pdd = set(_known)
    except Exception:
        known_pdd = set(_CLASS_TO_PDD.values())

    for sign in _sign_objs:
        oid = id(sign)
        if oid in seen:
            continue
        seen.add(oid)
        pdd = resolve_pdd_code_from_sign(sign)
        if not pdd or pdd not in known_pdd:
            continue
        rel = np.array([sign.position[0] - ego_pos[0],
                        sign.position[1] - ego_pos[1]])
        local = vehicle.convert_to_local_coordinates(rel, 0.0)
        x = float(local[0])
        y = -float(local[1])  # y=right (CARLA), as for every other box
        if x * x + y * y > _SIGN_RADIUS_M ** 2:
            continue
        if hasattr(sign, "_fallback_heading"):
            heading = float(sign._fallback_heading())
        else:
            heading = float(getattr(sign, "heading_theta", ego_heading))
        # The code alone cannot say WHICH limit a speed plate prescribes:
        # SpeedLimitSign20 and SpeedLimitSign60 are both "3.24", and the
        # sequence's speed_limit token carries the road's raw speed, not the
        # sign's. Carry the number in the box's own speed slot, as for vehicles.
        value_kmh = _sign_value_kmh(sign, pdd)
        w = float(getattr(sign, "WIDTH", 0.6) or 0.6)
        l = float(getattr(sign, "DEPTH", 0.1) or 0.1)
        entry = {
            "class": pdd,
            "position": [x, y, 0.0],
            "yaw": float(wrap_to_pi(heading - ego_heading)),
            "speed": (value_kmh or 0.0) / 3.6,
            "extent": [max(l, 0.2) / 2.0, max(w, 0.2) / 2.0, 0.75],
            "id": obj_id,
            "type_id": f"traffic.sign.{pdd}",
            "pdd_code": pdd,
            "affects_ego": True,
        }
        if value_kmh is not None:
            # Readable in the dump and immune to the m/s convention above.
            entry["sign_value_kmh"] = float(value_kmh)
        boxes.append(entry)
        obj_id += 1

    return boxes


def get_route(vehicle, num_points: int = 20, step_m: float = 1.0) -> np.ndarray:
    """Route in ego frame: x=forward, y=left (training convention).

    ``get_route_points_ego_frame`` returns y=right → invert.
    """
    try:
        from metadrive.policy.plant_policy import get_route_points_ego_frame
        pts, _ = get_route_points_ego_frame(vehicle, num_points=num_points, step_m=step_m)
        pts = np.asarray(pts, dtype=np.float32)
        if pts.shape[0] < num_points:
            pad = np.tile(pts[-1:], (num_points - pts.shape[0], 1))
            pts = np.vstack([pts, pad])
        pts = pts[:num_points].copy()
        pts[:, 1] = -pts[:, 1]  # y=right → y=left
        return pts
    except Exception:
        return np.array([[i * step_m, 0.0] for i in range(1, num_points + 1)],
                        dtype=np.float32)


def target_speed_mps(vehicle, engine, row: dict) -> float:
    """Target speed m/s: v_target_kmh in sign zone, else v_target_raw_kmh."""
    from traffic_bench.eval.engine.sim.sign_eval import _ego_in_sign_zone
    v_raw = float(row.get("v_target_raw_kmh", 80)) / 3.6
    v_sign = float(row.get("v_target_kmh", v_raw * 3.6)) / 3.6
    sign_mgr = getattr(engine, "traffic_sign_manager", None)
    if sign_mgr is not None and vehicle is not None:
        in_zone = any(_ego_in_sign_zone(s, vehicle) for s in sign_mgr.signs)
        target = v_sign if in_zone else v_raw
    else:
        target = v_raw
    return min(target, _MAX_TARGET_SPEED)


def write_gz_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def ensure_slurm_dummy(plant2_dir: Path) -> Path:
    """Create the dummy qsub log PlanTDataset expects when filter_routes=True."""
    log_dir = Path(plant2_dir) / "slurm" / "run_files" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    dummy_log = log_dir / f"qsub_out{_LOG_SUFFIX}.log"
    if not dummy_log.exists():
        dummy_log.write_text("# dummy log\n", encoding="utf-8")
    return dummy_log


def plant2_route_dir(plant2_dir: Path, scene_uid: str, variant: str) -> Path:
    """``<plant2_dir>/data/<scene_uid>_<variant>/``."""
    return Path(plant2_dir) / "data" / f"{scene_uid}_{variant}"


def render_bev_semantics(engine, vehicle,
                         resolution: int = _BEV_RESOLUTION,
                         size_meters: float = _BEV_SIZE_METERS):
    """Semantic BEV index map (H, W) uint8, or None if render fails."""
    try:
        from metadrive.policy.metadrive_obs_to_plant2 import render_bev_plant2
        sem = render_bev_plant2(
            engine, vehicle,
            resolution=resolution,
            size_meters=size_meters,
            device="cpu",
            return_semantic_map=True,
        )
        if sem is None:
            return None
        return np.asarray(sem, dtype=np.uint8)
    except Exception:
        return None


def write_bev_png(path: Path, sem_map: np.ndarray) -> None:
    """Write PlanTDataset-compatible semantic index PNG (mode L)."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(sem_map, dtype=np.uint8), mode="L").save(str(path))


class Plant2FrameCollector:
    """Accumulate pre-step PlanT2 boxes/measurements(/BEV) and flush to disk."""

    def __init__(self, row: dict, save_bev: bool = True):
        self.row = row
        self.save_bev = bool(save_bev)
        # (boxes, measurements, optional sem_map)
        self.step_records: list[tuple[list, dict, np.ndarray | None]] = []

    def on_step(self, base_env, row: dict | None = None) -> None:
        """Capture one frame from the current pre-action env state."""
        row = row if row is not None else self.row
        vehicle = getattr(base_env, "vehicle", None) or getattr(base_env, "agent", None)
        engine = getattr(base_env, "engine", None)
        if vehicle is None or engine is None:
            return

        pos = vehicle.position
        heading = float(vehicle.heading_theta)
        speed = float(getattr(vehicle, "speed", 0.0))

        ego_matrix = build_ego_matrix(pos, heading)
        boxes = collect_boxes(engine, vehicle)
        route_pts = get_route(vehicle)
        target_speed = target_speed_mps(vehicle, engine, row)

        v_limit_raw_kmh = float(row.get("v_target_raw_kmh", 80))
        speed_limit_mps = v_limit_raw_kmh / 3.6

        measurements = {
            "ego_matrix": ego_matrix,
            "pos_global": [float(pos[0]), float(pos[1])],
            "theta": heading,
            "speed": speed,
            "target_speed": target_speed,
            "speed_limit": speed_limit_mps,
            "route": route_pts.tolist(),
            "route_original": route_pts.tolist(),
            "brake": False,
            "augmentation_translation": 0.0,
            "augmentation_rotation": 0.0,
        }
        sem_map = None
        if self.save_bev:
            sem_map = render_bev_semantics(engine, vehicle)
        self.step_records.append((boxes, measurements, sem_map))

    def flush(self, route_dir: Path, success: bool, reason: dict | None = None) -> int:
        """Write boxes/measurements/BEV/results under ``route_dir``. Returns frame count.

        ``reason`` carries the flags behind ``success`` (arrived_dest, crashed,
        out_of_road). Without them the dataset can only see status=Failed and
        drops the whole route: measured on the previous corpus, 142 of 146
        dropped routes were "left the road near the end" with zero sign
        violations and the entire sign zone recorded, which is usable data.
        """
        route_dir = Path(route_dir)
        n_bev = 0
        for idx, (bxs, meas, sem) in enumerate(self.step_records):
            fname = f"{idx:04d}.json.gz"
            write_gz_json(route_dir / "boxes" / fname, bxs)
            write_gz_json(route_dir / "measurements" / fname, meas)
            if sem is not None:
                png = f"{idx:04d}.png"
                # PlanTDataset with augment=True also opens the *_augmented path.
                # Our aug offsets are 0 → same semantic map is a valid stand-in.
                write_bev_png(route_dir / "bev_no_car_semantics" / png, sem)
                write_bev_png(
                    route_dir / "bev_no_car_semantics_augmented" / png, sem)
                n_bev += 1

        score = 100.0 if success else 0.0
        results = {
            "scores": {"score_composed": score, "score_route": score},
            "num_infractions": 0,
            "infractions": {"min_speed_infractions": []},
            "status": "Completed" if success else "Failed",
            "timestamp": _TIMESTAMP,
        }
        if reason:
            results["failure_reason"] = {k: bool(v) for k, v in reason.items()}
        write_gz_json(route_dir / "results.json.gz", results)
        if self.save_bev and n_bev == 0 and self.step_records:
            print("[plant2] WARNING: save_bev=True but no BEV frames written "
                  "(render_bev_semantics returned None)")
        elif self.save_bev:
            print(f"[plant2] wrote {n_bev} BEV frames under {route_dir}")
        return len(self.step_records)
