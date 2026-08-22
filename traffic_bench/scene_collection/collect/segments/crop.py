#!/usr/bin/env python3
"""Crop indexed segments into crops/segment/<scene_id>/.

Each scene contains the segment edge cropped to an XY boundary that ends
BEFORE the junction (margin 10m), so the scene is a corridor without
an intersection. ``segment_type`` lives in meta.json, not in the path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from traffic_bench.eval.core.layout.junction_priority_layout import JunctionLayoutError as CropError
from traffic_bench.scene_collection.collect.lib.crop_xy import crop_net_to_xy_boundary
from traffic_bench.scene_collection.collect.segments.metrics import enrich_lane_fields
from traffic_bench.scene_collection.paths import MOSCOW_NET, SEGMENT_CROPS, SEGMENTS_INDEX
from traffic_bench.scene_collection.preview import parse_sumo_net, render_network


def json_dumps(obj) -> str:
    """Compact JSON dump."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


DEFAULT_NET = MOSCOW_NET
DEFAULT_INDEX = SEGMENTS_INDEX
DEFAULT_OUT = SEGMENT_CROPS
DEFAULT_MAX_SCENES = 0  # 0 = no cap; crop all of P

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


def flatten_legacy_segment_layout(scenes_root: Path) -> int:
    """Move crops/segment/{straight,curved}/<id>/ → crops/segment/<id>/."""
    moved = 0
    if not scenes_root.is_dir():
        return moved
    for nested in ("straight", "curved"):
        type_dir = scenes_root / nested
        if not type_dir.is_dir():
            continue
        for scene_dir in list(type_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            dest = scenes_root / scene_dir.name
            if dest.exists():
                continue
            shutil.move(str(scene_dir), str(dest))
            moved += 1
        try:
            next(type_dir.iterdir())
        except StopIteration:
            type_dir.rmdir()
        except OSError:
            pass
    return moved


def backfill_segment_metas(scenes_root: Path) -> int:
    """Write vehicle_lane_indices / pass_* onto existing meta.json files."""
    updated = 0
    for scene_dir in iter_segment_scene_dirs(scenes_root):
        meta_path = scene_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        new_meta = enrich_lane_fields(meta)
        if new_meta != meta:
            write_scene_meta(scene_dir, new_meta)
            updated += 1
    return updated


def iter_segment_scene_dirs(scenes_root: Path):
    """Yield scene dirs; supports flat layout and leftover straight/curved nests."""
    if not scenes_root.is_dir():
        return
    for child in sorted(scenes_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in {"straight", "curved"}:
            for inner in sorted(child.iterdir()):
                if inner.is_dir() and (inner / "meta.json").is_file():
                    yield inner
            continue
        if (child / "meta.json").is_file():
            yield child


def write_scene_meta(scene_dir: Path, meta: dict) -> None:
    """Write meta.json and center.json."""
    (scene_dir / "meta.json").write_text(
        json_dumps(meta) + "\n", encoding="utf-8"
    )
    if "latitude" in meta and "longitude" in meta:
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
    scene_dir = scenes_root / scene_id
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

    row = enrich_lane_fields(row)
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
        "vehicle_lane_indices": row.get("vehicle_lane_indices") or [],
        "pass_right_ok": bool(row.get("pass_right_ok")),
        "pass_left_ok": bool(row.get("pass_left_ok")),
        "center_xy": row["center_xy"],
        "start_xy": row["start_xy"],
        "end_xy": row["end_xy"],
        "to_junction_xy": row["to_junction_xy"],
        "crop_bbox": list(bbox),
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "net_file": "map.net.xml",
        "source_project": "scene_collection",
        "harvest": "sign_free_moscow_osm",
        "source_net": row.get("source_net", source_net.name),
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


def _crop_one(args_tuple: tuple) -> tuple:
    """Worker for parallel crop. Returns (status, scene_id, detail)."""
    row, net, scenes_root, skip_existing = args_tuple
    try:
        return crop_segment_scene(
            row,
            source_net=Path(net),
            scenes_root=Path(scenes_root),
            skip_existing=bool(skip_existing),
        )
    except Exception as exc:  # noqa: BLE001 — collect failures in worker
        scene_id = str(row.get("scene_id") or "?")
        return ("fail", scene_id, str(exc))


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
        "--max-scenes",
        type=int,
        default=DEFAULT_MAX_SCENES,
        help="Max scenes to crop (0 = no cap, harvest all of P)",
    )
    ap.add_argument(
        "--max-per-type",
        type=int,
        default=None,
        help="Deprecated alias for --max-scenes (ignored if --max-scenes is set)",
    )
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel netconvert workers (default 4)",
    )
    ap.add_argument(
        "--png-only",
        action="store_true",
        help="Re-render custom_cropped.png for existing scenes (no netconvert)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.png_only:
        moved = flatten_legacy_segment_layout(args.out)
        if moved:
            print(f"[crop_segment] flattened {moved} legacy nested scenes")
        _rerender_existing_pngs(
            args.out,
            skip_existing=args.skip_existing,
            workers=args.workers,
        )
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
    rows = [enrich_lane_fields(r) for r in rows]

    moved = flatten_legacy_segment_layout(args.out)
    if moved:
        print(f"[crop_segment] flattened {moved} legacy nested scenes → {args.out}")
    n_backfill = backfill_segment_metas(args.out)
    if n_backfill:
        print(f"[crop_segment] backfilled lane fields on {n_backfill} metas")

    max_scenes = int(args.max_scenes or 0)
    if max_scenes <= 0 and args.max_per_type:
        max_scenes = int(args.max_per_type)

    print(f"[crop_segment] net={args.net}")
    print(f"[crop_segment] index={args.index} ({len(rows)} segments of types {want_types})")
    cap_note = "unlimited" if max_scenes <= 0 else str(max_scenes)
    print(f"[crop_segment] max_scenes={cap_note}, skip_existing={args.skip_existing}, workers={args.workers}")

    existing_ids: Set[str] = set()
    if args.skip_existing:
        for scene_dir in iter_segment_scene_dirs(args.out):
            if (scene_dir / "map.net.xml").is_file():
                existing_ids.add(scene_dir.name)
        print(f"[crop_segment] existing on disk: {len(existing_ids)}")

    import random
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    jobs: List[Dict] = []
    for row in rows:
        sid = str(row.get("scene_id") or "")
        if args.skip_existing and sid in existing_ids:
            continue
        jobs.append(row)
        if max_scenes > 0 and len(jobs) >= max_scenes:
            break

    print(f"[crop_segment] jobs to process: {len(jobs)}")

    job_args = [
        (row, str(args.net), str(args.out), bool(args.skip_existing))
        for row in jobs
    ]

    stats = {"ok": 0, "fail": 0, "skip": 0}
    t0 = time.time()
    workers = max(1, int(args.workers))

    def _consume(i: int, status: str, scene_id: str, detail: str) -> None:
        if status == "ok":
            stats["ok"] += 1
        elif status == "skip":
            stats["skip"] += 1
        else:
            stats["fail"] += 1
            print(f"  [fail] {scene_id}: {detail}")
        if i % 25 == 0 or i == len(job_args):
            print(
                f"  [{i}/{len(job_args)}] ok={stats['ok']} fail={stats['fail']} skip={stats['skip']}"
            )

    if workers == 1:
        for i, job in enumerate(job_args, 1):
            status, scene_id, detail = _crop_one(job)
            _consume(i, status, scene_id, detail)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_crop_one, job) for job in job_args]
            for i, fut in enumerate(as_completed(futures), 1):
                status, scene_id, detail = fut.result()
                _consume(i, status, scene_id, detail)

    elapsed = time.time() - t0
    print(f"[crop_segment] Done in {elapsed:.1f}s: ok={stats['ok']} fail={stats['fail']} skip={stats['skip']}")
    print(f"[crop_segment] Output: {args.out}")


def _render_one_png(scene_dir: Path) -> tuple:
    """Worker: write custom_cropped.png. Returns (status, scene_id, detail)."""
    try:
        meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
        render_segment_preview(
            scene_dir / "map.net.xml",
            scene_dir / "custom_cropped.png",
            road_id=str(meta["road_id"]),
            center_xy=(float(meta["center_xy"][0]), float(meta["center_xy"][1])),
        )
        return ("ok", scene_dir.name, "")
    except Exception as exc:  # noqa: BLE001
        return ("fail", scene_dir.name, str(exc))


def _rerender_existing_pngs(
    scenes_root: Path,
    *,
    skip_existing: bool = False,
    workers: int = 1,
) -> None:
    """Write custom_cropped.png for cropped scenes (optionally only missing)."""
    jobs = []
    skipped = 0
    for scene_dir in iter_segment_scene_dirs(scenes_root):
        if not (scene_dir / "map.net.xml").is_file():
            continue
        if skip_existing and (scene_dir / "custom_cropped.png").is_file():
            skipped += 1
            continue
        jobs.append(scene_dir)

    print(
        f"[crop_segment] png-only: {len(jobs)} to render "
        f"(skip_existing={skip_existing}, already_have={skipped}, workers={workers})"
    )
    ok = fail = 0
    t0 = time.time()
    workers = max(1, int(workers))

    def _consume(i: int, status: str, scene_id: str, detail: str) -> None:
        nonlocal ok, fail
        if status == "ok":
            ok += 1
        else:
            fail += 1
            print(f"  [png fail] {scene_id}: {detail}")
        if i % 50 == 0 or i == len(jobs):
            print(f"  [{i}/{len(jobs)}] ok={ok} fail={fail}")

    if workers == 1:
        for i, scene_dir in enumerate(jobs, 1):
            status, scene_id, detail = _render_one_png(scene_dir)
            _consume(i, status, scene_id, detail)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_render_one_png, scene_dir) for scene_dir in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                status, scene_id, detail = fut.result()
                _consume(i, status, scene_id, detail)

    print(f"[crop_segment] png-only done in {time.time() - t0:.1f}s: ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
