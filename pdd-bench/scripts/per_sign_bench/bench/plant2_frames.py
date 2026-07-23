"""PlanT2 frame capture helpers for live dump during expert_replay rollouts.

Layout written under ``plant2_dir``::

    data/<scene_uid>_<variant>/
        boxes/NNNN.json.gz
        measurements/NNNN.json.gz
        results.json.gz
    slurm/run_files/logs/qsub_out2025_07.log   # required by PlanTDataset(filter_routes=True)

Frames are captured *before* each ``env.step`` (PlanTDataset convention).
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np

_MAX_TARGET_SPEED = 20.0  # max from PlanTVariables.target_speeds (m/s)
_TIMESTAMP = "2025_07_20_00_00_00"
_LOG_SUFFIX = "_".join(_TIMESTAMP.split("_")[:2])  # "2025_07"


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

    def _add(obj):
        nonlocal obj_id
        oid = id(obj)
        if oid in seen or obj is vehicle:
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
        if cls == "traffic_light":
            from metadrive.constants import MetaDriveType
            status = getattr(obj, "status", MetaDriveType.LIGHT_UNKNOWN)
            if status not in (MetaDriveType.LIGHT_RED, MetaDriveType.LIGHT_YELLOW):
                return
            entry["state"] = ("Red" if status == MetaDriveType.LIGHT_RED else "Yellow")
            entry["affects_ego"] = True

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
    from bench.sign_eval import _ego_in_sign_zone
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


class Plant2FrameCollector:
    """Accumulate pre-step PlanT2 boxes/measurements and flush to disk."""

    def __init__(self, row: dict):
        self.row = row
        self.step_records: list[tuple[list, dict]] = []

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
        self.step_records.append((boxes, measurements))

    def flush(self, route_dir: Path, success: bool) -> int:
        """Write boxes/measurements/results under ``route_dir``. Returns frame count."""
        route_dir = Path(route_dir)
        for idx, (bxs, meas) in enumerate(self.step_records):
            fname = f"{idx:04d}.json.gz"
            write_gz_json(route_dir / "boxes" / fname, bxs)
            write_gz_json(route_dir / "measurements" / fname, meas)

        score = 100.0 if success else 0.0
        results = {
            "scores": {"score_composed": score, "score_route": score},
            "num_infractions": 0,
            "infractions": {"min_speed_infractions": []},
            "status": "Completed" if success else "Failed",
            "timestamp": _TIMESTAMP,
        }
        write_gz_json(route_dir / "results.json.gz", results)
        return len(self.step_records)
