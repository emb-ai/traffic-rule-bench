#!/usr/bin/env python3
"""Validate a PlanT2 L1 dump route: BEV, x_objs (from boxes), sign type_ids."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PLAN_T = _ROOT.parents[1] / "plant2" / "PlanT"
if str(_PLAN_T) not in sys.path:
    sys.path.insert(0, str(_PLAN_T))

from plant_variables import PlanTVariables
from util.sign_id import SIGN_CODES, sign_code_to_id

_RAW_BEV_RES = 256
_BEV_CROP = 64
_MODEL_BEV_SIDE = _RAW_BEV_RES - 2 * _BEV_CROP
_MODEL_BEV_SPAN_M = 32.0

# BGR colors keyed by x_objs type_id (matches viz_train_global_gif.py).
_CLASS_COLORS_BGR: dict[float, tuple[int, int, int]] = {
    1.0: (0, 0, 220),      # car
    2.0: (0, 220, 220),    # walker
    3.0: (180, 180, 180),  # static
    4.0: (220, 0, 220),    # stop_sign
    5.0: (0, 0, 255),      # traffic_light
    6.0: (0, 140, 255),    # emergency
}
for _i, _code in enumerate(SIGN_CODES):
    _hue = int(180 * _i / max(len(SIGN_CODES), 1))
    _bgr = cv2.cvtColor(np.uint8([[[_hue, 200, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]
    _CLASS_COLORS_BGR[float(7 + _i)] = tuple(int(x) for x in _bgr)


def rad2deg(theta: float) -> float:
    deg = float(np.rad2deg(theta)) % 360.0
    return deg - 360.0 if deg > 180.0 else deg


def load_bev_semantic_indices(bev_path: Path) -> np.ndarray:
    """Load 256x256 semantic class indices (0..4) from mode-L PNG."""
    im = Image.open(bev_path)
    if im.mode != "L":
        im = im.convert("L")
    arr = np.asarray(im, dtype=np.uint8)
    if arr.shape != (_RAW_BEV_RES, _RAW_BEV_RES):
        raise ValueError(f"expected {_RAW_BEV_RES}x{_RAW_BEV_RES} BEV, got {arr.shape} from {bev_path}")
    return arr


def load_model_bev_rgb(bev_path: Path, *, upscale: int = 1) -> np.ndarray:
    """(128, 128, 3) uint8 RGB — same transform as PlanTDataset / load_model_bev."""
    bev_colors = np.array(PlanTVariables.bev_colors, dtype=np.float32)
    bev = pil_to_tensor(Image.open(bev_path))
    bev = torch.rot90(bev, dims=(1, 2))
    idx = bev[0, _BEV_CROP : -_BEV_CROP, _BEV_CROP : -_BEV_CROP].numpy().astype(np.int64)
    rgb = (bev_colors[idx] * 255.0).clip(0, 255).astype(np.uint8)
    if upscale > 1:
        side = rgb.shape[0] * upscale
        rgb = np.asarray(
            Image.fromarray(rgb, mode="RGB").resize((side, side), Image.NEAREST),
            dtype=np.uint8,
        )
    return rgb


def summarize_bev_debug_rgb(rgb: np.ndarray) -> dict:
    flat = rgb.reshape(-1, rgb.shape[-1])
    uniq = np.unique(flat, axis=0)
    return {
        "min": int(rgb.min()),
        "max": int(rgb.max()),
        "mean": float(rgb.mean()),
        "n_unique_rgb": int(len(uniq)),
    }


def carla_ego_to_model_bev_pixel(
    fwd_x: float,
    y_right: float,
    *,
    upscale: int = 1,
) -> tuple[int, int] | None:
    """Ego-relative meters (x=forward, y=right) -> col,row on model BEV canvas."""
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


def oriented_box_px(
    cx: int,
    cy: int,
    angle_deg: float,
    extent_x: float,
    extent_y: float,
) -> np.ndarray:
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


def draw_x_objs_on_bev(bev_rgb: np.ndarray, x_objs: list, *, upscale: int = 1) -> np.ndarray:
    """Overlay x_objs boxes on model-view BEV. x_objs coords are ego-relative meters."""
    img = cv2.cvtColor(bev_rgb, cv2.COLOR_RGB2BGR)
    canvas_side = img.shape[0]
    pix_per_m = _MODEL_BEV_SIDE / _MODEL_BEV_SPAN_M

    ego_col = ego_row = canvas_side // 2
    cv2.drawMarker(img, (ego_col, ego_row), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 12, 2)

    n_drawn = 0
    for o in x_objs:
        tid, ex, ey, yaw_deg, speed, width_m, length_m = o[0], o[1], o[2], o[3], o[4], o[5], o[6]
        pt = carla_ego_to_model_bev_pixel(ex, ey, upscale=upscale)
        if pt is None:
            continue
        cx, cy = pt
        n_drawn += 1
        color = _CLASS_COLORS_BGR.get(float(tid), (200, 200, 200))
        img_yaw = ego_yaw_to_image_angle(ex, ey, yaw_deg, upscale=upscale)
        extent_x_px = max(width_m * pix_per_m * upscale / 2, 3)
        extent_y_px = max(length_m * pix_per_m * upscale / 2, 3)
        box = oriented_box_px(cx, cy, img_yaw, extent_x_px, extent_y_px)

        overlay = img.copy()
        cv2.fillPoly(overlay, [box], color)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        cv2.polylines(img, [box], True, color, 2)

        label = str(int(tid)) if float(tid).is_integer() else f"{tid:g}"
        cv2.putText(img, label, (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2)
        cv2.putText(img, label, (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        if speed > 0.5:
            rad = math.radians(img_yaw)
            endx = cx + speed / 3.6 * pix_per_m * upscale * math.cos(rad)
            endy = cy + speed / 3.6 * pix_per_m * upscale * math.sin(rad)
            cv2.arrowedLine(img, (cx, cy), (int(endx), int(endy)), color, 1, tipLength=0.35)

    if n_drawn == 0 and x_objs:
        cv2.putText(
            img,
            "x_objs oob",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def boxes_to_x_objs(boxes: list, *, range_m: float = 50.0, range_factor_front: float = 2.0) -> list:
    """Mirror PlanTDataset filtering for dynamic + sign objects."""
    type_nums = PlanTVariables.class_nums
    sign_like = {"stop_sign"} | set(SIGN_CODES)
    labels = boxes[1:]  # drop ego
    out = []
    for x in labels:
        cls = x.get("class", "")
        pos_x, pos_y = x["position"][:2]
        if cls in (["traffic_light"] + list(sign_like)):
            if pos_x ** 2 + pos_y ** 2 > 30 ** 2:
                continue
        else:
            x_div = range_factor_front ** 2 if pos_x > 0 else 1
            if pos_x ** 2 / x_div + pos_y ** 2 > range_m ** 2:
                continue
        if cls == "car":
            out.append([
                type_nums["car"],
                pos_x, pos_y,
                rad2deg(x["yaw"]),
                x["speed"] * 3.6,
                x["extent"][1] * 2,
                x["extent"][0] * 2,
            ])
        elif cls in sign_like or cls in SIGN_CODES:
            if not x.get("affects_ego"):
                continue
            key = cls if cls in type_nums else cls.lower()
            out.append([
                type_nums[key],
                pos_x, pos_y,
                rad2deg(x["yaw"]),
                0.0,
                x["extent"][1] * 2,
                x["extent"][0] * 2,
            ])
    return out


def validate_route(route_dir: Path, *, frames: list[int] | None = None, upscale: int = 4) -> dict:
    route_dir = Path(route_dir)
    boxes_dir = route_dir / "boxes"
    bev_dir = route_dir / "bev_no_car_semantics"
    n_frames = len(list(boxes_dir.glob("*.json.gz")))
    if not frames:
        frames = list(range(n_frames))
    else:
        frames = sorted(set(frames))
        frames = [f for f in frames if 0 <= f < n_frames]

    debug_dir = bev_dir / "debug"
    if debug_dir.is_dir():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    report = {"route": str(route_dir), "n_frames": n_frames, "sampled_frames": frames, "frames": {}}
    for seq in frames:
        bx_path = boxes_dir / f"{seq:04d}.json.gz"
        bev_path = bev_dir / f"{seq:04d}.png"
        bev_path_debug = debug_dir / f"{seq:04d}.png"
        bx = json.load(gzip.open(bx_path))
        x_objs = boxes_to_x_objs(bx)
        npc = [o for o in x_objs if o[0] == PlanTVariables.class_nums["car"]]
        sign_boxes = [b for b in bx if str(b.get("class")) in SIGN_CODES]

        bev_ok = False
        bev_classes: list[int] = []
        debug_rgb_stats: dict | None = None
        arr: np.ndarray | None = None
        debug_rgb: np.ndarray | None = None
        if bev_path.is_file():
            arr = load_bev_semantic_indices(bev_path)
            bev_classes = sorted(int(x) for x in np.unique(arr))
            bev_ok = len(bev_classes) > 1 and arr.shape == (_RAW_BEV_RES, _RAW_BEV_RES)
            debug_rgb = load_model_bev_rgb(bev_path, upscale=upscale)
            debug_rgb = draw_x_objs_on_bev(debug_rgb, x_objs, upscale=upscale)
            Image.fromarray(debug_rgb, mode="RGB").save(bev_path_debug)
            debug_rgb_stats = summarize_bev_debug_rgb(debug_rgb)

        pdd_codes = [str(b.get("class")) for b in sign_boxes]
        type_ids = {code: PlanTVariables.class_nums.get(code) for code in pdd_codes}
        sign_emb_id = sign_code_to_id("2.5") if "2.5" in pdd_codes else None

        report["frames"][seq] = {
            "n_boxes": len(bx),
            "box_classes": [b.get("class") for b in bx],
            "n_x_objs": len(x_objs),
            "n_npc_x_objs": len(npc),
            "npc_x_objs": npc,
            "sign_boxes": [
                {
                    "class": b.get("class"),
                    "id": b.get("id"),
                    "pdd_code": b.get("pdd_code"),
                    "position": b.get("position")[:2],
                    "x_obj_type_id": type_ids.get(str(b.get("class"))),
                }
                for b in sign_boxes
            ],
            "sign_emb_id_2p5": sign_emb_id,
            "bev_ok": bev_ok,
            "bev_shape": list(arr.shape) if arr is not None else None,
            "bev_semantic_classes": bev_classes,
            "bev_debug_path": str(bev_path_debug) if bev_path.is_file() else None,
            "bev_debug_shape": list(debug_rgb.shape) if debug_rgb is not None else None,
            "bev_debug_stats": debug_rgb_stats,
        }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route-dir", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    ap.add_argument("--upscale", type=int, default=4, help="NEAREST upscale for debug PNGs (default 4)")
    args = ap.parse_args()
    rep = validate_route(args.route_dir, frames=args.frames, upscale=args.upscale)
    print(json.dumps(rep, indent=2))
    ok = True
    if rep["n_frames"] == 0:
        print("FAIL: no frames", file=sys.stderr)
        return 1
    any_bev = any(f["bev_ok"] for f in rep["frames"].values())
    any_npc = any(f["n_npc_x_objs"] > 0 for f in rep["frames"].values())
    any_sign = any(f["sign_boxes"] for f in rep["frames"].values())
    print(f"\nSummary: frames={rep['n_frames']} bev_ok={any_bev} npc_in_x_objs={any_npc} sign_2.5={any_sign}")
    if not any_bev:
        print("WARN: BEV missing or single-class", file=sys.stderr)
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
