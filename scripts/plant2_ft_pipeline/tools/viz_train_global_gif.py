#!/usr/bin/env python3
"""GIF of a training route: model BEV + boxes.

Modes (--canvas):
  ego   — per-frame ego-centric BEV (model input); static reprojected each frame.
  world — BEV tiles projected into a fixed world map; static/objects stay put on screen.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

PIPELINE_DIR = Path(__file__).resolve().parent
# _ROOT is plant2_ft_pipeline/; parents[1] is traffic-rule-bench repo root.
TRB_ROOT = _ROOT.parents[1]
PLAN_T = TRB_ROOT / "plant2" / "PlanT"
sys.path.insert(0, str(PLAN_T))

from plant_variables import PlanTVariables  # noqa: E402
from util.sign_id import SIGN_CODES  # noqa: E402

DEFAULT_ROUTE = (
    Path(__file__).resolve().parents[3]
    / "plant2_l1_fv_experts_split_signs_2.5/train/data/"
    "sign_100062_j2_lane0_seed1413785215_v0_default"
)

_RAW_BEV_RES = 256
_BEV_CROP = 64
_MODEL_BEV_SIDE = _RAW_BEV_RES - 2 * _BEV_CROP
_MODEL_BEV_SPAN_M = 32.0

SIGN_LIKE = set(SIGN_CODES) | {"stop_sign", "traffic_light"}
WORLD_FIXED_CLASSES = SIGN_LIKE | {"static"}
_CLUSTER_MATCH_M = 1.0


def type_id_for_obj(obj: dict) -> float | None:
    cls = obj.get("class")
    if cls is None:
        return None
    nums = PlanTVariables.class_nums
    if cls in nums:
        return float(nums[cls])
    low = str(cls).lower()
    if low in nums:
        return float(nums[low])
    return None


def type_id_to_name(tid: float | None, raw_class: str) -> str:
    if tid is None:
        return str(raw_class)
    inv = {v: k for k, v in PlanTVariables.class_nums.items()}
    name = inv.get(float(tid), f"type_{int(tid)}")
    if name in SIGN_CODES:
        return f"PDD {name}"
    return name.replace("_", " ")


CLASS_COLORS_BGR: dict[float, tuple[int, int, int]] = {
    1.0: (0, 0, 220),
    2.0: (0, 220, 220),
    3.0: (180, 180, 180),
    4.0: (220, 0, 220),
    5.0: (0, 0, 255),
    6.0: (0, 140, 255),
}
for i, code in enumerate(SIGN_CODES):
    hue = int(180 * i / max(len(SIGN_CODES), 1))
    hsv = np.uint8([[[hue, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    CLASS_COLORS_BGR[float(7 + i)] = tuple(int(x) for x in bgr)


def ego_carla_to_world(fwd_x: float, y_right: float, ego_matrix: np.ndarray) -> tuple[float, float]:
    p = ego_matrix @ np.array([fwd_x, -y_right, 0.0, 1.0])
    return float(p[0]), float(p[1])


def world_to_ego_carla(wx: float, wy: float, ego_matrix: np.ndarray) -> tuple[float, float]:
    p = np.linalg.inv(ego_matrix) @ np.array([wx, wy, 0.0, 1.0])
    return float(p[0]), float(-p[1])


def is_world_fixed(obj: dict) -> bool:
    cls = obj.get("class")
    if cls in WORLD_FIXED_CLASSES:
        return True
    return cls == "static"


def is_ego(obj: dict, idx: int) -> bool:
    if idx == 0 and obj.get("class") == "car":
        pos = obj.get("position", [0, 0, 0])
        if abs(pos[0]) < 1e-3 and abs(pos[1]) < 1e-3:
            return True
    return obj.get("id") == 0 and obj.get("class") == "car"


def load_model_bev(bev_path: Path) -> np.ndarray:
    """(128, 128, 3) uint8 RGB — same as sample['BEV']."""
    import torch

    bev_colors = np.array(PlanTVariables.bev_colors, dtype=np.float32)
    bev = pil_to_tensor(Image.open(bev_path))
    bev = torch.rot90(bev, dims=(1, 2))
    idx = bev[0, _BEV_CROP : -_BEV_CROP, _BEV_CROP : -_BEV_CROP].numpy().astype(np.int64)
    rgb = bev_colors[idx]
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def carla_ego_to_model_bev_pixel(
    fwd_x: float,
    y_right: float,
    *,
    upscale: int = 1,
) -> tuple[int, int] | None:
    scale = _RAW_BEV_RES / 64.0
    ey_md = -y_right
    px256 = _RAW_BEV_RES // 2 - int(ey_md * scale)
    py256 = _RAW_BEV_RES // 2 - int(fwd_x * scale)
    row128 = (_RAW_BEV_RES - 1 - px256) - _BEV_CROP
    col128 = py256 - _BEV_CROP
    if not (0 <= col128 < _MODEL_BEV_SIDE and 0 <= row128 < _MODEL_BEV_SIDE):
        return None
    if upscale == 1:
        return col128, row128
    return col128 * upscale, row128 * upscale


def ego_yaw_to_image_angle(
    fwd_x: float,
    y_right: float,
    yaw_deg: float,
    *,
    upscale: int = 1,
    probe_m: float = 3.0,
) -> float:
    rad = math.radians(yaw_deg)
    p0 = carla_ego_to_model_bev_pixel(fwd_x, y_right, upscale=upscale)
    p1 = carla_ego_to_model_bev_pixel(
        fwd_x + math.cos(rad) * probe_m,
        y_right + math.sin(rad) * probe_m,
        upscale=upscale,
    )
    if p0 is None or p1 is None:
        return yaw_deg
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))


def get_coords_bb_px(cx, cy, angle_deg, extent_x, extent_y):
    angle = angle_deg
    endx1 = cx - extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy1 = cy + extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))
    endx2 = cx + extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy2 = cy - extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))
    endx3 = cx + extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy3 = cy - extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))
    endx4 = cx - extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy4 = cy + extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))
    return np.array(
        [(endx1, endy1), (endx2, endy2), (endx3, endy3), (endx4, endy4)],
        dtype=np.int32,
    )


def bev_pixel_to_ego_carla(col: int, row: int) -> tuple[float, float]:
    """Inverse of carla_ego_to_model_bev_pixel (128² model BEV grid)."""
    px256 = _RAW_BEV_RES - 1 - (row + _BEV_CROP)
    py256 = col + _BEV_CROP
    scale = _RAW_BEV_RES / 64.0
    ey_md = (_RAW_BEV_RES // 2 - px256) / scale
    fwd_x = (_RAW_BEV_RES // 2 - py256) / scale
    return fwd_x, -ey_md


_BEV_BG_RGB = tuple(int(c * 255) for c in PlanTVariables.bev_colors[0])


class WorldMap:
    """Fixed world canvas: BEV semantics mosaic + world↔pixel mapping."""

    def __init__(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        ppm: float,
        margin_m: float = 6.0,
    ) -> None:
        self.min_x = min_x - margin_m
        self.min_y = min_y - margin_m
        self.max_x = max_x + margin_m
        self.max_y = max_y + margin_m
        self.ppm = ppm
        self.w = int(math.ceil((self.max_x - self.min_x) * ppm))
        self.h = int(math.ceil((self.max_y - self.min_y) * ppm))
        bg = _BEV_BG_RGB[::-1]
        self.mosaic = np.full((self.h, self.w, 3), bg, dtype=np.uint8)
        self.ego_trail: list[tuple[float, float]] = []

    def world_to_px(self, wx: float, wy: float) -> tuple[int, int]:
        col = int(round((wx - self.min_x) * self.ppm))
        row = int(round((self.max_y - wy) * self.ppm))
        return col, row

    def world_heading_to_px_angle(self, wx: float, wy: float, world_yaw: float, probe_m: float = 2.0) -> float:
        p0 = self.world_to_px(wx, wy)
        p1 = self.world_to_px(wx + math.cos(world_yaw) * probe_m, wy + math.sin(world_yaw) * probe_m)
        return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))

    def stamp_bev(self, bev_rgb: np.ndarray, ego_matrix: np.ndarray) -> None:
        """Project one model BEV tile into the world mosaic."""
        em = np.array(ego_matrix, dtype=np.float64)
        bev_bgr = cv2.cvtColor(bev_rgb, cv2.COLOR_RGB2BGR)
        for row in range(bev_rgb.shape[0]):
            for col in range(bev_rgb.shape[1]):
                rgb = tuple(bev_rgb[row, col])
                if rgb == _BEV_BG_RGB:
                    continue
                fx, fy = bev_pixel_to_ego_carla(col, row)
                wx, wy = ego_carla_to_world(fx, fy, em)
                px, py = self.world_to_px(wx, wy)
                if 0 <= px < self.w and 0 <= py < self.h:
                    self.mosaic[py, px] = bev_bgr[row, col]

    def object_world_pose(
        self,
        obj: dict,
        measurements: dict,
        anchors: GlobalAnchors,
    ) -> tuple[float, float, float]:
        em = np.array(measurements["ego_matrix"], dtype=np.float64)
        ego_theta = float(measurements["theta"])
        ex, ey = float(obj["position"][0]), float(obj["position"][1])
        wx, wy = ego_carla_to_world(ex, ey, em)
        yaw = ego_theta + float(obj.get("yaw", 0.0))
        if is_world_fixed(obj):
            cluster = anchors.nearest_cluster(obj, wx, wy)
            if cluster is not None:
                wx, wy = cluster.world_xy
                yaw = cluster.world_yaw
        return wx, wy, yaw


def draw_world_frame(
    world_map: WorldMap,
    labels: list,
    measurements: dict,
    anchors: GlobalAnchors,
    *,
    seq: int,
    route_name: str,
) -> np.ndarray:
    ppm = world_map.ppm
    img = world_map.mosaic.copy()

    ego_wx = float(measurements["pos_global"][0])
    ego_wy = float(measurements["pos_global"][1])
    ego_theta = float(measurements["theta"])
    world_map.ego_trail.append((ego_wx, ego_wy))

    # trajectory
    if len(world_map.ego_trail) > 1:
        pts = np.array([world_map.world_to_px(x, y) for x, y in world_map.ego_trail], dtype=np.int32)
        cv2.polylines(img, [pts], False, (0, 180, 0), 2)

    seen: set[str] = set()
    n_drawn = 0
    drawn_clusters: set[tuple[str, tuple[float, float]]] = set()

    for idx, obj in enumerate(labels):
        if is_ego(obj, idx):
            continue
        raw_cls = str(obj.get("class"))
        wx, wy, world_yaw = world_map.object_world_pose(obj, measurements, anchors)
        if is_world_fixed(obj):
            key = (raw_cls, (round(wx, 2), round(wy, 2)))
            if key in drawn_clusters:
                continue
            drawn_clusters.add(key)

        cx, cy = world_map.world_to_px(wx, wy)
        if not (0 <= cx < world_map.w and 0 <= cy < world_map.h):
            continue

        n_drawn += 1
        tid = type_id_for_obj(obj)
        seen.add(type_id_to_name(tid, raw_cls))
        color = CLASS_COLORS_BGR.get(tid, (200, 200, 200)) if tid is not None else (160, 160, 160)

        ext = obj.get("extent", [1.0, 1.0, 1.0])
        width_m = float(ext[1]) * 2.0
        length_m = float(ext[0]) * 2.0
        img_yaw = world_map.world_heading_to_px_angle(wx, wy, world_yaw)
        extent_x_px = max(width_m * ppm / 2, 3)
        extent_y_px = max(length_m * ppm / 2, 3)
        box = get_coords_bb_px(cx, cy, img_yaw, extent_x_px, extent_y_px)

        overlay = img.copy()
        cv2.fillPoly(overlay, [box], color)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        cv2.polylines(img, [box], True, color, 2)

        name = type_id_to_name(tid, raw_cls)
        cv2.putText(img, name, (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2)
        cv2.putText(img, name, (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # ego marker
    ex_px, ey_px = world_map.world_to_px(ego_wx, ego_wy)
    ego_angle = world_map.world_heading_to_px_angle(ego_wx, ego_wy, ego_theta)
    tip = (
        int(ex_px + 10 * math.cos(math.radians(ego_angle))),
        int(ey_px + 10 * math.sin(math.radians(ego_angle))),
    )
    cv2.arrowedLine(img, (ex_px, ey_px), tip, (0, 255, 0), 2, tipLength=0.4)
    cv2.circle(img, (ex_px, ey_px), 5, (0, 255, 0), -1)

    legend_h = 56
    out_h = img.shape[0] + legend_h
    out = np.zeros((out_h, img.shape[1], 3), dtype=np.uint8)
    out[: img.shape[0], : img.shape[1]] = img

    title = f"frame {seq:04d} | world canvas | ego=({ego_wx:.1f},{ego_wy:.1f}) | {route_name[:32]}"
    cv2.putText(out, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(out, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    cv2.putText(
        out,
        f"world-fixed BEV mosaic | static/sign fixed on map | objects={n_drawn} | {world_map.w}x{world_map.h}px @ {ppm:.1f}px/m",
        (8, img.shape[0] + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (200, 200, 200),
        1,
    )
    y0 = img.shape[0] + 38
    for i, name in enumerate(sorted(seen)):
        cv2.putText(out, name, (8 + i * 160, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def build_world_map(
    route: Path,
    seq_list: list[int],
    bev_dir: Path,
    meas_dir: Path,
    anchors: GlobalAnchors,
    ppm: float | None,
) -> WorldMap:
    xs: list[float] = []
    ys: list[float] = []
    for seq in seq_list:
        meas = json.load(gzip.open(meas_dir / f"{seq:04d}.json.gz"))
        wx, wy = float(meas["pos_global"][0]), float(meas["pos_global"][1])
        xs.append(wx)
        ys.append(wy)
        em = np.array(meas["ego_matrix"])
        for fx, fy in [(-16, -16), (-16, 16), (16, -16), (16, 16)]:
            cx, cy = ego_carla_to_world(fx, fy, em)
            xs.append(cx)
            ys.append(cy)
    for (_cls, _i), (wx, wy) in anchors.pos.items():
        xs.append(wx)
        ys.append(wy)

    if ppm is None:
        span = max(max(xs) - min(xs), max(ys) - min(ys), 20.0)
        ppm = min(900.0 / span, 12.0)
        ppm = max(ppm, 4.0)

    world_map = WorldMap(min(xs), min(ys), max(xs), max(ys), ppm)
    for seq in seq_list:
        bev_f = bev_dir / f"{seq:04d}.png"
        meas_f = meas_dir / f"{seq:04d}.json.gz"
        if not bev_f.is_file() or not meas_f.is_file():
            continue
        bev_rgb = load_model_bev(bev_f)
        meas = json.load(gzip.open(meas_f))
        world_map.stamp_bev(bev_rgb, meas["ego_matrix"])
    return world_map


def rad2deg(yaw_rad: float) -> float:
    return float(np.rad2deg(yaw_rad))


class WorldCluster:
    __slots__ = ("cls", "world_xy", "world_yaw", "_xy", "_yaws")

    def __init__(self, cls: str, wx: float, wy: float, yaw: float) -> None:
        self.cls = cls
        self._xy = [(wx, wy)]
        self._yaws = [yaw]
        self.world_xy = (wx, wy)
        self.world_yaw = yaw

    def add(self, wx: float, wy: float, yaw: float) -> None:
        self._xy.append((wx, wy))
        self._yaws.append(yaw)

    def finalize(self) -> None:
        arr = np.array(self._xy, dtype=np.float64)
        span = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
        if span > 1.0:
            self.world_xy = tuple(np.median(arr, axis=0))
            self.world_yaw = float(np.median(self._yaws))
        else:
            self.world_xy = self._xy[0]
            self.world_yaw = self._yaws[0]


class GlobalAnchors:
    """World-space anchors clustered by class + proximity (dump ids are unstable)."""

    def __init__(self) -> None:
        self.clusters: dict[str, list[WorldCluster]] = {}
        self._finalized = False

    def record(self, obj: dict, measurements: dict) -> None:
        if not is_world_fixed(obj):
            return
        em = np.array(measurements["ego_matrix"], dtype=np.float64)
        ex, ey = float(obj["position"][0]), float(obj["position"][1])
        wx, wy = ego_carla_to_world(ex, ey, em)
        cls = str(obj.get("class"))
        yaw = float(measurements["theta"]) + float(obj.get("yaw", 0.0))

        # PDD / TL: one world anchor per class (dump origin can drift tens of metres).
        if cls in SIGN_LIKE:
            clusters = self.clusters.setdefault(cls, [])
            if not clusters:
                clusters.append(WorldCluster(cls, wx, wy, yaw))
            else:
                clusters[0].add(wx, wy, yaw)
            return

        for cluster in self.clusters.setdefault(cls, []):
            cx, cy = cluster.world_xy
            if math.hypot(wx - cx, wy - cy) <= _CLUSTER_MATCH_M:
                cluster.add(wx, wy, yaw)
                return

        self.clusters[cls].append(WorldCluster(cls, wx, wy, yaw))

    def finalize(self) -> None:
        if self._finalized:
            return
        for clusters in self.clusters.values():
            for cluster in clusters:
                cluster.finalize()
        self._finalized = True

    def nearest_cluster(self, obj: dict, wx: float, wy: float) -> WorldCluster | None:
        cls = str(obj.get("class"))
        if cls in SIGN_LIKE:
            clusters = self.clusters.get(cls, [])
            return clusters[0] if clusters else None
        best: WorldCluster | None = None
        best_d = _CLUSTER_MATCH_M
        for cluster in self.clusters.get(cls, []):
            cx, cy = cluster.world_xy
            d = math.hypot(wx - cx, wy - cy)
            if d <= best_d:
                best_d = d
                best = cluster
        return best

    def ego_pose_for_draw(
        self,
        obj: dict,
        measurements: dict,
    ) -> tuple[float, float, float, float, float]:
        """Return (ex, ey, yaw_deg, wx, wy) in CARLA ego frame for BEV drawing."""
        em = np.array(measurements["ego_matrix"], dtype=np.float64)
        ego_theta = float(measurements["theta"])
        ex, ey = float(obj["position"][0]), float(obj["position"][1])
        wx, wy = ego_carla_to_world(ex, ey, em)
        yaw_rad = float(obj.get("yaw", 0.0))

        if is_world_fixed(obj):
            cluster = self.nearest_cluster(obj, wx, wy)
            if cluster is not None:
                wx, wy = cluster.world_xy
                yaw_rad = cluster.world_yaw - ego_theta

        ex2, ey2 = world_to_ego_carla(wx, wy, em)
        return ex2, ey2, rad2deg(yaw_rad), wx, wy

    @property
    def pos(self) -> dict[tuple[str, int], tuple[float, float]]:
        """Legacy debug listing: class + cluster index -> world xy."""
        out: dict[tuple[str, int], tuple[float, float]] = {}
        for cls, clusters in self.clusters.items():
            for i, cluster in enumerate(clusters):
                out[(cls, i)] = cluster.world_xy
        return out


def draw_frame(
    bev_rgb: np.ndarray,
    labels: list,
    measurements: dict,
    anchors: GlobalAnchors,
    *,
    seq: int,
    route_name: str,
    upscale: int = 4,
) -> np.ndarray:
    side = bev_rgb.shape[0]
    canvas_side = side * upscale
    pix_per_m = side / _MODEL_BEV_SPAN_M

    img = cv2.cvtColor(
        cv2.resize(bev_rgb, (canvas_side, canvas_side), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_RGB2BGR,
    )

    ego_col = ego_row = canvas_side // 2
    cv2.drawMarker(img, (ego_col, ego_row), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2)

    ego_wx, ego_wy = float(measurements["pos_global"][0]), float(measurements["pos_global"][1])
    n_drawn = n_oob = 0
    seen: set[str] = set()
    drawn_clusters: set[tuple[str, tuple[float, float]]] = set()

    for idx, obj in enumerate(labels):
        if is_ego(obj, idx):
            continue

        ex, ey, yaw_deg, wx, wy = anchors.ego_pose_for_draw(obj, measurements)
        raw_cls = str(obj.get("class"))
        if is_world_fixed(obj):
            cluster_key = (raw_cls, (round(wx, 2), round(wy, 2)))
            if cluster_key in drawn_clusters:
                continue
            drawn_clusters.add(cluster_key)
        pt = carla_ego_to_model_bev_pixel(ex, ey, upscale=upscale)
        if pt is None:
            n_oob += 1
            continue

        cx, cy = pt
        n_drawn += 1
        tid = type_id_for_obj(obj)
        seen.add(type_id_to_name(tid, raw_cls))
        color = CLASS_COLORS_BGR.get(tid, (200, 200, 200)) if tid is not None else (160, 160, 160)

        ext = obj.get("extent", [1.0, 1.0, 1.0])
        width_m = float(ext[1]) * 2.0
        length_m = float(ext[0]) * 2.0
        img_yaw = ego_yaw_to_image_angle(ex, ey, yaw_deg, upscale=upscale)
        extent_x_px = max(width_m * pix_per_m * upscale / 2, 3)
        extent_y_px = max(length_m * pix_per_m * upscale / 2, 3)
        box = get_coords_bb_px(cx, cy, img_yaw, extent_x_px, extent_y_px)

        overlay = img.copy()
        cv2.fillPoly(overlay, [box], color)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        cv2.polylines(img, [box], True, color, 2)

        name = type_id_to_name(tid, raw_cls)
        label = f"{name} w=({wx:.1f},{wy:.1f})"
        cv2.putText(img, label, (cx + 4, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 2)
        cv2.putText(img, label, (cx + 4, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        speed = float(obj.get("speed", 0.0))
        if speed > 0.5:
            vel = speed
            rad = math.radians(img_yaw)
            endx = cx + vel * pix_per_m * upscale * math.cos(rad)
            endy = cy + vel * pix_per_m * upscale * math.sin(rad)
            cv2.arrowedLine(img, (cx, cy), (int(endx), int(endy)), color, 1, tipLength=0.35)

    legend_h = 72 + 18 * max((len(seen) + 2) // 3, 1)
    out = np.zeros((canvas_side + legend_h, canvas_side, 3), dtype=np.uint8)
    out[:canvas_side, :canvas_side] = img

    title = f"frame {seq:04d} | ego w=({ego_wx:.1f},{ego_wy:.1f}) | {route_name[:36]}"
    cv2.putText(out, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(out, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    cv2.putText(
        out,
        f"world anchor -> ego -> BEV px | static/sign clustered by world xy | drawn={n_drawn} oob={n_oob}",
        (8, canvas_side + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (200, 200, 200),
        1,
    )

    y0 = canvas_side + 40
    for i, name in enumerate(sorted(seen)):
        x0 = 8 + (i % 3) * 210
        y = y0 + (i // 3) * 18
        cv2.putText(out, name, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route", type=Path, default=DEFAULT_ROUTE, help="Route dir with boxes/ bev/ measurements/")
    ap.add_argument("--out-dir", type=Path, default=PIPELINE_DIR / "viz_outputs")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--upscale", type=int, default=4)
    ap.add_argument(
        "--canvas",
        choices=("ego", "world"),
        default="ego",
        help="ego=per-frame model BEV; world=fixed map from projected BEV tiles",
    )
    ap.add_argument(
        "--world-ppm",
        type=float,
        default=None,
        help="World canvas pixels per meter (auto if unset)",
    )
    args = ap.parse_args()

    route = args.route.resolve()
    boxes_dir = route / "boxes"
    bev_dir = route / "bev_no_car_semantics"
    meas_dir = route / "measurements"
    for d, name in [(boxes_dir, "boxes"), (bev_dir, "bev_no_car_semantics"), (meas_dir, "measurements")]:
        if not d.is_dir():
            raise SystemExit(f"Missing {name}/ in {route}")

    frames = sorted(int(p.name.split(".")[0]) for p in boxes_dir.glob("*.json.gz"))
    end = args.end if args.end is not None else max(frames)
    seq_range = range(max(args.start, min(frames)), min(end, max(frames)) + 1, args.step)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = "train_world" if args.canvas == "world" else "train_global"
    out_gif = args.out_dir / f"{tag}_{route.name[:60]}.gif"

    anchors = GlobalAnchors()
    seq_list = []
    for seq in seq_range:
        box_f = boxes_dir / f"{seq:04d}.json.gz"
        meas_f = meas_dir / f"{seq:04d}.json.gz"
        if not box_f.is_file() or not meas_f.is_file():
            continue
        with gzip.open(box_f, "rt") as f:
            labels = json.load(f)
        with gzip.open(meas_f, "rt") as f:
            measurements = json.load(f)
        for idx, obj in enumerate(labels):
            if not is_ego(obj, idx):
                anchors.record(obj, measurements)
        seq_list.append(seq)

    anchors.finalize()
    gif_frames = []

    world_map = None
    if args.canvas == "world":
        world_map = build_world_map(route, seq_list, bev_dir, meas_dir, anchors, args.world_ppm)
        print(f"World mosaic: {world_map.w}x{world_map.h}px @ {world_map.ppm:.1f}px/m")

    for seq in seq_list:
        box_f = boxes_dir / f"{seq:04d}.json.gz"
        bev_f = bev_dir / f"{seq:04d}.png"
        meas_f = meas_dir / f"{seq:04d}.json.gz"
        with gzip.open(box_f, "rt") as f:
            labels = json.load(f)
        with gzip.open(meas_f, "rt") as f:
            measurements = json.load(f)
        if args.canvas == "world":
            assert world_map is not None
            gif_frames.append(
                draw_world_frame(
                    world_map,
                    labels,
                    measurements,
                    anchors,
                    seq=seq,
                    route_name=route.name,
                )
            )
        else:
            bev_rgb = load_model_bev(bev_f)
            gif_frames.append(
                draw_frame(
                    bev_rgb,
                    labels,
                    measurements,
                    anchors,
                    seq=seq,
                    route_name=route.name,
                    upscale=args.upscale,
                )
            )

    if not gif_frames:
        raise SystemExit("No frames rendered")

    duration_ms = int(1000 / max(args.fps, 1))
    pil_frames = [Image.fromarray(f) for f in gif_frames]
    pil_frames[0].save(
        out_gif,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Wrote {len(gif_frames)} frames -> {out_gif}")
    if anchors.pos:
        print("World clusters (static / signs):")
        for k, (wx, wy) in sorted(anchors.pos.items()):
            print(f"  {k[0]} cluster={k[1]}: ({wx:.4f}, {wy:.4f})")


if __name__ == "__main__":
    main()
