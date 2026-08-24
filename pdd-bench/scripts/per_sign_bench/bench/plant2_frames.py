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

_DUMP_DEBUG = os.environ.get("PLANT2_DUMP_DEBUG", "").lower() in ("1", "true", "yes")

# Only these PDD codes may become sign boxes (and therefore x_objs tokens).
# Restoring the recorded signs puts MainRoadSign (2.1) and YieldSign (2.4) into
# the env alongside the StopSign, but the stop benchmark must train on 2.5
# alone. Set PLANT2_DUMP_SIGN_CLASSES to a comma-separated list of codes, or to
# "all" to disable the filter.
_DUMP_SIGN_CLASSES_ENV = os.environ.get("PLANT2_DUMP_SIGN_CLASSES", "2.5").strip()
DUMP_SIGN_CLASS_ALLOWLIST = (
    None if _DUMP_SIGN_CLASSES_ENV.lower() == "all"
    else {c.strip() for c in _DUMP_SIGN_CLASSES_ENV.split(",") if c.strip()}
)


def _dbg(msg: str) -> None:
    if _DUMP_DEBUG:
        print(f"[plant2_frames] {msg}", flush=True)


def _iter_traffic_vehicles(traffic_mgr) -> list:
    """NPC list from traffic manager (PGTrafficManager vs SumoTrafficManager)."""
    if traffic_mgr is None:
        return []
    for attr in ("vehicles", "traffic_vehicles", "_traffic_vehicles"):
        val = getattr(traffic_mgr, attr, None)
        if val is None:
            continue
        if callable(val) and not isinstance(val, list):
            try:
                val = val()
            except TypeError:
                continue
        if val:
            return list(val)
    return []


def _iter_auxiliary_vehicles(engine) -> list:
    """Convoy NPCs spawned by ``AuxiliaryAgentsManager`` (2.5 stop/yield scenes)."""
    mgr = getattr(engine, "auxiliary_agent_manager", None)
    if mgr is None:
        return []
    val = getattr(mgr, "auxiliary_vehicles", None)
    if val is None:
        return []
    if callable(val) and not isinstance(val, list):
        try:
            val = val()
        except TypeError:
            return []
    return list(val)


def _engine_object_sources(engine) -> dict[str, list]:
    """Live object inventory for debug (no side effects)."""
    out: dict[str, list] = {}
    traffic_mgr = getattr(engine, "traffic_manager", None)
    out["traffic_mgr"] = _iter_traffic_vehicles(traffic_mgr)
    out["aux_mgr"] = _iter_auxiliary_vehicles(engine)
    if hasattr(engine, "get_objects"):
        out["get_objects"] = list(
            engine.get_objects(lambda o: hasattr(o, "position")).values()
        )
    obj_mgr = getattr(engine, "object_manager", None)
    if obj_mgr is not None:
        out["object_manager"] = list(getattr(obj_mgr, "spawned_objects", {}).values())
    sign_mgr = getattr(engine, "traffic_sign_manager", None)
    out["sign_mgr"] = list(getattr(sign_mgr, "signs", []) or [])
    return out

_MAX_TARGET_SPEED = 20.0  # max from PlanTVariables.target_speeds (m/s)
# Near-stop threshold (m/s): PlanTDataset zeros target_speed when brake=True.
_BRAKE_SPEED_EPS = 0.5
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
    "SpeedLimitSign": "3.24",
    "SpeedLimitSign20": "3.24",
    "SpeedLimitSign30": "3.24",
    "SpeedLimitSign40": "3.24",
    "SpeedLimitSign60": "3.24",
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


def resolve_pdd_code_from_sign(sign) -> str | None:
    """Best-effort PDD code for a MetaDrive BaseTrafficSign instance."""
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

    Traffic signs from ``traffic_sign_manager`` are written as PDD object
    classes (``class`` / ``pdd_code`` = e.g. ``\"2.5\"``) with ``affects_ego``
    so PlanTDataset can map them via ``PlanTVariables.class_nums``.
    """
    from metadrive.utils.math import wrap_to_pi
    from metadrive.component.vehicle.base_vehicle import BaseVehicle
    from metadrive.component.traffic_participants.pedestrian import Pedestrian
    from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight

    try:
        from traffic_signs.base_traffic_sign import BaseTrafficSign
    except ImportError:  # pragma: no cover
        BaseTrafficSign = ()  # type: ignore

    # Known PDD codes that have a dedicated class_emb index.
    try:
        from util.sign_id import SIGN_CODES as _SIGN_CODES
        known_pdd = set(_SIGN_CODES)
    except Exception:
        known_pdd = set(_CLASS_TO_PDD.values())

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
        if BaseTrafficSign and isinstance(obj, BaseTrafficSign):
            return "_traffic_sign"  # dumped via traffic_sign_manager below
        if isinstance(obj, BaseVehicle):
            return "car"
        if isinstance(obj, Pedestrian):
            return "walker"
        if isinstance(obj, BaseTrafficLight):
            return "traffic_light"
        return "static"

    obj_id = 1
    seen: set = set()

    def _ego_xy(obj):
        rel = np.array([obj.position[0] - ego_pos[0],
                        obj.position[1] - ego_pos[1]])
        local = vehicle.convert_to_local_coordinates(rel, 0.0)
        return float(local[0]), -float(local[1])  # y=right (CARLA)

    def _add(obj):
        nonlocal obj_id
        oid = id(obj)
        if oid in seen or obj is vehicle:
            return
        if not hasattr(obj, "position"):
            return
        cls = _cls(obj)
        # Do NOT mark traffic signs as seen here: they also appear in
        # engine.get_objects(), and marking them would make the sign_mgr
        # dump loop below skip every PDD sign (empty spatial tokens).
        if cls == "_traffic_sign":
            return
        seen.add(oid)
        x, y = _ego_xy(obj)

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

    n_from_traffic = n_from_aux = n_from_engine = n_from_obj_mgr = 0
    traffic_mgr = getattr(engine, "traffic_manager", None)
    if traffic_mgr is not None:
        tm_type = type(traffic_mgr).__name__
        tm_vehicles = _iter_traffic_vehicles(traffic_mgr)
        _dbg(f"traffic_manager={tm_type} n_vehicles={len(tm_vehicles)} "
             f"(Sumo uses .traffic_vehicles; 0 is normal when convoy is aux-only)")
        for v in tm_vehicles:
            before = len(boxes)
            _add(v)
            if len(boxes) > before:
                n_from_traffic += 1
    aux_vehicles = _iter_auxiliary_vehicles(engine)
    if aux_vehicles or _DUMP_DEBUG:
        _dbg(f"auxiliary_agent_manager n_vehicles={len(aux_vehicles)}")
    for v in aux_vehicles:
        before = len(boxes)
        _add(v)
        if len(boxes) > before:
            n_from_aux += 1
    if hasattr(engine, "get_objects"):
        engine_objs = engine.get_objects(lambda o: hasattr(o, "position")).values()
        _dbg(f"engine.get_objects n={len(list(engine_objs))}")
        for obj in engine_objs:
            before = len(boxes)
            _add(obj)
            if len(boxes) > before:
                n_from_engine += 1
    obj_mgr = getattr(engine, "object_manager", None)
    if obj_mgr is not None:
        spawned = getattr(obj_mgr, "spawned_objects", {})
        _dbg(f"object_manager.spawned_objects n={len(spawned)}")
        for obj in spawned.values():
            before = len(boxes)
            _add(obj)
            if len(boxes) > before:
                n_from_obj_mgr += 1
    elif _DUMP_DEBUG:
        _dbg("object_manager: absent on engine (using get_objects only)")

    # Explicit PDD signs (spatial tokens for PlanT tok_emb).
    sign_mgr = getattr(engine, "traffic_sign_manager", None)
    n_signs_added = 0
    for sign in list(getattr(sign_mgr, "signs", []) or []):
        if sign is None or not hasattr(sign, "position"):
            continue
        oid = id(sign)
        if oid in seen:
            continue
        seen.add(oid)
        pdd = resolve_pdd_code_from_sign(sign)
        sign_type = type(sign).__name__
        sign_id = getattr(sign, "id", None) or getattr(sign, "name", None)
        if not pdd or pdd not in known_pdd:
            _dbg(f"sign skip: type={sign_type} id={sign_id} pdd={pdd!r} (unknown)")
            continue
        if DUMP_SIGN_CLASS_ALLOWLIST is not None and pdd not in DUMP_SIGN_CLASS_ALLOWLIST:
            _dbg(f"sign skip: pdd={pdd} id={sign_id} "
                 f"(not in DUMP_SIGN_CLASS_ALLOWLIST={sorted(DUMP_SIGN_CLASS_ALLOWLIST)})")
            continue
        x, y = _ego_xy(sign)
        if x * x + y * y > 900.0:  # 30m, same as stop_sign / TL in PlanTDataset
            _dbg(f"sign skip: pdd={pdd} id={sign_id} dist={math.hypot(x, y):.1f}m > 30m")
            continue
        if hasattr(sign, "_fallback_heading"):
            heading = float(sign._fallback_heading())
        else:
            heading = float(getattr(sign, "heading_theta", getattr(sign, "_heading_theta", ego_heading)))
        yaw = float(wrap_to_pi(heading - ego_heading))
        w = float(getattr(sign, "WIDTH", 0.6) or 0.6)
        l = float(getattr(sign, "DEPTH", 0.1) or 0.1)
        boxes.append({
            "class": pdd,
            "position": [x, y, 0.0],
            "yaw": yaw,
            "speed": 0.0,
            "extent": [max(l, 0.2) / 2.0, max(w, 0.2) / 2.0, 0.75],
            "id": obj_id,
            "type_id": f"traffic.sign.{pdd}",
            "pdd_code": pdd,
            "affects_ego": True,
        })
        obj_id += 1
        n_signs_added += 1
        _dbg(f"sign added: pdd={pdd} id={sign_id} type={sign_type} "
             f"ego_xy=({x:.1f},{y:.1f}) box_id={obj_id - 1}")

    if _DUMP_DEBUG:
        classes = [b["class"] for b in boxes]
        _dbg(
            f"collect_boxes: total={len(boxes)} cars={classes.count('car')} "
            f"from_traffic={n_from_traffic} from_aux={n_from_aux} "
            f"from_engine={n_from_engine} from_obj_mgr={n_from_obj_mgr} "
            f"signs={n_signs_added} classes={classes}"
        )

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


def row_has_speed_target(row: dict) -> bool:
    """True when catalog row carries an explicit speed-limit target.

    Priority / stop / yield / etc. dumps usually omit these fields. The old
    default (80 km/h → capped to 20 m/s) made ``target_speed`` constantly 20
    and broke ego-speed supervision for stops.
    """
    return row.get("v_target_kmh") is not None or row.get("v_target_raw_kmh") is not None


def target_speed_mps(vehicle, engine, row: dict) -> float:
    """Target speed (m/s) for PlanT ego-speed head.

    * Speed-limit scenes (catalog has ``v_target_*``): ``v_target_kmh`` in the
      sign zone, else ``v_target_raw_kmh``, capped at ``_MAX_TARGET_SPEED``.
    * Other scenes: imitate expert ego speed so stop/yield profiles are
      supervised (instead of the bogus constant 20 m/s default).
    """
    ego_speed = float(getattr(vehicle, "speed", 0.0) or 0.0) if vehicle is not None else 0.0
    if not row_has_speed_target(row):
        return min(max(ego_speed, 0.0), _MAX_TARGET_SPEED)

    from bench.sign_eval import _ego_in_sign_zone

    v_raw = float(row["v_target_raw_kmh"] if row.get("v_target_raw_kmh") is not None else 80) / 3.6
    v_sign = float(row["v_target_kmh"] if row.get("v_target_kmh") is not None else v_raw * 3.6) / 3.6
    sign_mgr = getattr(engine, "traffic_sign_manager", None) if engine is not None else None
    if sign_mgr is not None and vehicle is not None:
        in_zone = any(_ego_in_sign_zone(s, vehicle) for s in sign_mgr.signs)
        target = v_sign if in_zone else v_raw
    else:
        target = v_raw
    return min(max(target, 0.0), _MAX_TARGET_SPEED)


def expert_brake_flag(speed_mps: float, row: dict) -> bool:
    """Set ``brake`` so PlanTDataset can force ``target_speed=0`` near stops.

    Only for non-speed-limit catalogs; speed-limit scenes keep limit supervision.
    """
    if row_has_speed_target(row):
        return False
    return float(speed_mps) < _BRAKE_SPEED_EPS


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


def known_pdd_codes() -> set[str]:
    try:
        from util.sign_id import SIGN_CODES
        return set(SIGN_CODES)
    except Exception:
        return set(_CLASS_TO_PDD.values())


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
        step_idx = len(self.step_records)
        if vehicle is None or engine is None:
            _dbg(f"step={step_idx}: skip (vehicle={vehicle is not None} engine={engine is not None})")
            return

        if _DUMP_DEBUG and step_idx % 10 == 0:
            src = _engine_object_sources(engine)
            _dbg(
                f"step={step_idx}: engine sources "
                f"traffic={len(src.get('traffic_mgr', []))} "
                f"aux={len(src.get('aux_mgr', []))} "
                f"get_objects={len(src.get('get_objects', []))} "
                f"signs={len(src.get('sign_mgr', []))}"
            )

        pos = vehicle.position
        heading = float(vehicle.heading_theta)
        speed = float(getattr(vehicle, "speed", 0.0) or 0.0)

        ego_matrix = build_ego_matrix(pos, heading)
        boxes = collect_boxes(engine, vehicle)
        route_pts = get_route(vehicle)
        target_speed = target_speed_mps(vehicle, engine, row)
        brake = expert_brake_flag(speed, row)

        if row_has_speed_target(row):
            v_limit_raw_kmh = float(
                row["v_target_raw_kmh"] if row.get("v_target_raw_kmh") is not None else 80
            )
        else:
            # No catalog limit: keep a neutral highway-ish embedding input;
            # ego-speed supervision comes from target_speed (= expert speed).
            v_limit_raw_kmh = 80.0
        speed_limit_mps = v_limit_raw_kmh / 3.6

        measurements = {
            "ego_matrix": ego_matrix,
            "pos_global": [float(pos[0]), float(pos[1])],
            "theta": heading,
            # Always the expert/live ego speed (m/s). Kept separate from
            # target_speed so FV catalog limits can change later without a
            # re-dump of velocities.
            "speed": speed,
            "ego_speed": speed,
            "target_speed": target_speed,
            "speed_limit": speed_limit_mps,
            "route": route_pts.tolist(),
            "route_original": route_pts.tolist(),
            "brake": brake,
            "augmentation_translation": 0.0,
            "augmentation_rotation": 0.0,
        }
        sem_map = None
        if self.save_bev:
            sem_map = render_bev_semantics(engine, vehicle)
            if _DUMP_DEBUG and step_idx % 10 == 0:
                if sem_map is None:
                    _dbg(f"step={step_idx}: BEV render returned None")
                else:
                    uniq = np.unique(sem_map)
                    _dbg(f"step={step_idx}: BEV shape={sem_map.shape} classes={uniq.tolist()}")

        if _DUMP_DEBUG and step_idx % 10 == 0:
            npc = [b for b in boxes[1:] if b.get("class") == "car"]
            signs = [b for b in boxes if str(b.get("class", "")) in known_pdd_codes()]
            _dbg(
                f"step={step_idx}: boxes n={len(boxes)} npc_cars={len(npc)} signs={len(signs)} "
                f"target_speed={target_speed:.2f} brake={brake} route_pts={len(route_pts)}"
            )
            for s in signs:
                _dbg(f"  sign box: class={s.get('class')} id={s.get('id')} "
                     f"pdd_code={s.get('pdd_code')} pos={s.get('position')[:2]}")

        self.step_records.append((boxes, measurements, sem_map))

    def flush(self, route_dir: Path, success: bool) -> int:
        """Write boxes/measurements/BEV/results under ``route_dir``. Returns frame count."""
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
        write_gz_json(route_dir / "results.json.gz", results)
        if self.save_bev and n_bev == 0 and self.step_records:
            print("[plant2] WARNING: save_bev=True but no BEV frames written "
                  "(render_bev_semantics returned None)")
        elif self.save_bev:
            print(f"[plant2] wrote {n_bev} BEV frames under {route_dir}")
        return len(self.step_records)
