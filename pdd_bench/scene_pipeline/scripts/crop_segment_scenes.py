#!/usr/bin/env python3
"""Crop indexed segments into scenes/segment/{straight,curved}/<scene_id>/.

Each scene contains the segment edge cropped to an XY boundary that ends
BEFORE the junction (margin 10m), so the scene is a straight road without
any intersection.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]


def json_dumps(obj) -> str:
    """Compact JSON dump."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _find_netconvert() -> str:
    """Find netconvert executable."""
    for path in (
        shutil.which("netconvert"),
        str(Path.home() / ".local" / "bin" / "netconvert"),
        "/usr/local/bin/netconvert",
        "/usr/bin/netconvert",
    ):
        if path and Path(path).exists():
            return path
    raise FileNotFoundError(
        "netconvert not found. Install SUMO or add netconvert to PATH."
    )


class CropError(Exception):
    """Error during scene cropping."""
    pass


def crop_net_to_xy_boundary(
    net_path: Path,
    bbox_xy: Tuple[float, float, float, float],
    out_path: Path,
) -> None:
    """Crop a SUMO net to cartesian boundary (xmin, ymin, xmax, ymax)."""
    xmin, ymin, xmax, ymax = bbox_xy
    if xmax <= xmin or ymax <= ymin:
        raise CropError(f"Degenerate XY boundary: {bbox_xy}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    boundary = f"{xmin},{ymin},{xmax},{ymax}"
    cmd = [
        _find_netconvert(),
        "--sumo-net-file",
        str(net_path),
        "-o",
        str(out_path),
        "--keep-edges.in-boundary",
        boundary,
        "--geometry.remove",
        "true",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise CropError(
            f"netconvert XY-boundary crop failed for {net_path}:\n"
            f"{result.stderr or result.stdout}"
        )
    if not out_path.is_file():
        raise CropError(f"netconvert did not write {out_path}")

    # Refresh location convBoundary from remaining geometry
    tree = ET.parse(out_path)
    root = tree.getroot()
    xs: list[float] = []
    ys: list[float] = []
    for lane in root.findall("./edge/lane"):
        for token in (lane.get("shape") or "").split():
            if "," not in token:
                continue
            x_s, y_s = token.split(",", 1)
            xs.append(float(x_s))
            ys.append(float(y_s))
    loc = root.find("location")
    if loc is not None and xs and ys:
        loc.set("convBoundary", f"{min(xs)},{min(ys)},{max(xs)},{max(ys)}")
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="unicode", xml_declaration=True)

DEFAULT_NET = ROOT / "nets" / "moscow.net.xml"
DEFAULT_INDEX = ROOT / "index" / "segments.jsonl"
DEFAULT_OUT = ROOT / "scenes" / "segment"
DEFAULT_MAX_PER_TYPE = 500

# Margin before junction (meters) — segment ends this far before the junction
JUNCTION_MARGIN_M = 10.0
# Margin around segment for cropping (meters)
CROP_MARGIN_M = 30.0


def load_segments_index(path: Path) -> List[Dict[str, Any]]:
    """Load segments.jsonl."""
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_crop_bbox(
    start_xy: tuple,
    end_xy: tuple,
    junction_xy: tuple,
    margin_m: float = CROP_MARGIN_M,
    junction_margin_m: float = JUNCTION_MARGIN_M,
) -> tuple:
    """Compute bounding box for segment crop.

    The segment runs from start_xy toward end_xy (which is near the junction).
    We want to include most of the segment but stop before the junction.
    """
    # Collect all relevant points
    xs = [start_xy[0], end_xy[0]]
    ys = [start_xy[1], end_xy[1]]

    # Add margin
    xmin = min(xs) - margin_m
    xmax = max(xs) + margin_m
    ymin = min(ys) - margin_m
    ymax = max(ys) + margin_m

    return (xmin, ymin, xmax, ymax)


def write_scene_meta(scene_dir: Path, meta: dict) -> None:
    """Write meta.json and center.json."""
    (scene_dir / "meta.json").write_text(
        json_dumps(meta) + "\n", encoding="utf-8"
    )
    (scene_dir / "center.json").write_text(
        json_dumps({"lat": meta["latitude"], "lon": meta["longitude"]}) + "\n",
        encoding="utf-8",
    )


def render_segment_preview(
    net_path: Path,
    out_png: Path,
    road_id: str,
    center_xy: Tuple[float, float],
) -> None:
    """Same top-down preview as junction / dual_path / lane_direction scenes."""
    import matplotlib

    matplotlib.use("Agg")
    from tools.render_map import parse_sumo_net, render_network

    edges, junctions = parse_sumo_net(net_path)
    render_network(
        edges,
        junctions,
        out_png,
        figsize=(6, 6),
        dpi=120,
        marker_xy=center_xy,
        compliant_edge_ids=[road_id],
        legend=True,
    )


def crop_segment_scene(
    row: Dict[str, Any],
    *,
    source_net: Path,
    scenes_root: Path,
    skip_existing: bool,
) -> tuple:
    """Crop a single segment scene. Returns (status, scene_id, detail)."""
    scene_id = row["scene_id"]
    segment_type = row["segment_type"]
    scene_dir = scenes_root / segment_type / scene_id
    out_net = scene_dir / "map.net.xml"

    if skip_existing and out_net.is_file():
        return ("skip", scene_id, "exists")

    start_xy = tuple(row["start_xy"])
    end_xy = tuple(row["end_xy"])
    junction_xy = tuple(row["to_junction_xy"])

    bbox = compute_crop_bbox(start_xy, end_xy, junction_xy)

    scene_dir.mkdir(parents=True, exist_ok=True)
    try:
        crop_net_to_xy_boundary(source_net, bbox, out_net)
    except Exception as exc:
        return ("fail", scene_id, str(exc))

    if not out_net.is_file():
        return ("fail", scene_id, "netconvert did not write output")

    meta = {
        "scene_name": scene_id,
        "scene_kind": "segment",
        "segment_type": segment_type,
        "road_id": row["edge_id"],
        "junction_id": row["junction_id"],
        "osm_way_id": row["osm_way_id"],
        "length_m": row["length_m"],
        "straightness": row["straightness"],
        "lane_count": row["lane_count"],
        "center_xy": row["center_xy"],
        "start_xy": row["start_xy"],
        "end_xy": row["end_xy"],
        "to_junction_xy": row["to_junction_xy"],
        "crop_bbox": list(bbox),
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "net_file": "map.net.xml",
        "source_net": row.get("source_net", source_net.name),
        "source_project": "moscow_scenes",
        "harvest": "sign_free_moscow_osm",
    }
    write_scene_meta(scene_dir, meta)

    # Render PNG preview
    out_png = scene_dir / "custom_cropped.png"
    try:
        render_segment_preview(
            out_net,
            out_png,
            road_id=row["edge_id"],
            center_xy=(float(row["center_xy"][0]), float(row["center_xy"][1])),
        )
    except Exception as exc:
        print(f"  [png warn] {scene_id}: {exc}")

    return ("ok", scene_id, str(scene_dir))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", type=Path, default=DEFAULT_NET)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--segment-types",
        default="straight,curved",
        help="Comma-separated segment types to crop (default: straight,curved)",
    )
    ap.add_argument(
        "--max-per-type",
        type=int,
        default=DEFAULT_MAX_PER_TYPE,
        help=f"Max scenes per segment type (default {DEFAULT_MAX_PER_TYPE})",
    )
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--png-only",
        action="store_true",
        help="Re-render custom_cropped.png for existing scenes (no netconvert)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.png_only:
        _rerender_existing_pngs(args.out, args.segment_types)
        return

    if not args.net.is_file():
        sys.exit(f"ERROR: net not found: {args.net}")
    if not args.index.is_file():
        sys.exit(f"ERROR: segments index not found: {args.index}")

    want_types = {s.strip() for s in args.segment_types.split(",") if s.strip()}
    rows = [
        r for r in load_segments_index(args.index)
        if r.get("segment_type") in want_types
    ]

    print(f"[crop_segment] net={args.net}")
    print(f"[crop_segment] index={args.index} ({len(rows)} segments of types {want_types})")
    print(f"[crop_segment] max_per_type={args.max_per_type}, skip_existing={args.skip_existing}")

    # Count existing scenes
    existing: Dict[str, int] = defaultdict(int)
    if args.skip_existing:
        for seg_type in want_types:
            type_dir = args.out / seg_type
            if type_dir.is_dir():
                existing[seg_type] = sum(
                    1 for p in type_dir.iterdir()
                    if (p / "map.net.xml").is_file()
                )
        if existing:
            print(f"[crop_segment] existing on disk: {dict(existing)}")

    # Shuffle for variety (stable seed)
    import random
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    # Cap per type
    filled: Dict[str, int] = dict(existing)
    jobs: List[Dict] = []
    for row in rows:
        seg_type = row["segment_type"]
        if filled.get(seg_type, 0) >= args.max_per_type:
            continue
        jobs.append(row)
        filled[seg_type] = filled.get(seg_type, 0) + 1

    print(f"[crop_segment] jobs to process: {len(jobs)} (target fill: {filled})")

    stats = {"ok": 0, "fail": 0, "skip": 0}
    t0 = time.time()

    for i, row in enumerate(jobs, 1):
        status, scene_id, detail = crop_segment_scene(
            row,
            source_net=args.net,
            scenes_root=args.out,
            skip_existing=args.skip_existing,
        )
        if status == "ok":
            stats["ok"] += 1
        elif status == "skip":
            stats["skip"] += 1
        else:
            stats["fail"] += 1
            print(f"  [fail] {scene_id}: {detail}")

        if i % 25 == 0 or i == len(jobs):
            print(f"  [{i}/{len(jobs)}] ok={stats['ok']} fail={stats['fail']} skip={stats['skip']}")

    elapsed = time.time() - t0
    print(f"[crop_segment] Done in {elapsed:.1f}s: ok={stats['ok']} fail={stats['fail']} skip={stats['skip']}")
    print(f"[crop_segment] Output: {args.out}")


def _rerender_existing_pngs(scenes_root: Path, segment_types: str) -> None:
    """Rewrite custom_cropped.png for scenes already on disk."""
    want_types = {s.strip() for s in segment_types.split(",") if s.strip()}
    jobs: List[Path] = []
    for seg_type in sorted(want_types):
        type_dir = scenes_root / seg_type
        if not type_dir.is_dir():
            continue
        for scene_dir in sorted(type_dir.iterdir()):
            if (scene_dir / "map.net.xml").is_file() and (scene_dir / "meta.json").is_file():
                jobs.append(scene_dir)

    print(f"[crop_segment] png-only: {len(jobs)} scenes under {scenes_root}")
    ok = fail = 0
    t0 = time.time()
    for i, scene_dir in enumerate(jobs, 1):
        try:
            meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
            render_segment_preview(
                scene_dir / "map.net.xml",
                scene_dir / "custom_cropped.png",
                road_id=str(meta["road_id"]),
                center_xy=(float(meta["center_xy"][0]), float(meta["center_xy"][1])),
            )
            ok += 1
        except Exception as exc:
            fail += 1
            print(f"  [png fail] {scene_dir.name}: {exc}")
        if i % 10 == 0 or i == len(jobs):
            print(f"  [{i}/{len(jobs)}] ok={ok} fail={fail}")
    print(f"[crop_segment] png-only done in {time.time() - t0:.1f}s: ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
