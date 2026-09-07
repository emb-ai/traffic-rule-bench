#!/usr/bin/env python3
"""Crop corridor maps into crops/segment/<scene_id>/.

Reads ``index/segments.jsonl`` (full candidate pool from ``enumerate``). Each
scene is an XY neighborhood around the corridor window.

Jobs are **round-robin interleaved by subtype**
``(straight|curved) × (1|2|3plus)`` so an interrupted / resumed run
(``--skip-existing``) leaves a balanced partial pool on disk. Per-sign
quotas are still applied later by ``assign``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from traffic_bench.scene_collection.collect.lib.crop_xy import crop_net_to_xy_boundary
from traffic_bench.scene_collection.collect.segments.metrics import (
    enrich_lane_fields,
    lane_bucket,
)
from traffic_bench.scene_collection.paths import (
    MOSCOW_NET,
    SEGMENT_CROPS,
    SEGMENTS_INDEX,
)
from traffic_bench.scene_collection.preview import parse_sumo_net, render_network


def json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


DEFAULT_NET = MOSCOW_NET
DEFAULT_INDEX = SEGMENTS_INDEX
DEFAULT_OUT = SEGMENT_CROPS
CROP_MARGIN_M = 40.0

SUBTYPE_ORDER = (
    "straight|1",
    "straight|2",
    "straight|3plus",
    "curved|1",
    "curved|2",
    "curved|3plus",
)


def _row_subtype(row: Dict[str, Any]) -> str:
    if row.get("subtype"):
        return str(row["subtype"])
    seg = str(row.get("segment_type") or "unknown")
    bucket = str(row.get("lane_bucket") or "")
    if not bucket:
        bucket = lane_bucket(int(row.get("lane_count") or 0))
    return f"{seg}|{bucket}"


def interleave_by_subtype(
    rows: List[Dict[str, Any]],
    *,
    seed: int,
    on_disk_by_subtype: Dict[str, int] | None = None,
) -> List[Dict[str, Any]]:
    """Round-robin jobs across subtypes so partial crops stay balanced.

    When ``on_disk_by_subtype`` is set (resume with --skip-existing), subtypes
    that are furthest behind their fair share of the *remaining* work are
    visited first each round.
    """
    import random
    from collections import defaultdict

    pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pools[_row_subtype(row)].append(row)

    rng = random.Random(seed)
    for key in pools:
        rng.shuffle(pools[key])

    keys = [k for k in SUBTYPE_ORDER if pools.get(k)]
    for k in sorted(pools):
        if k not in keys:
            keys.append(k)

    on_disk = dict(on_disk_by_subtype or {})
    remaining = {k: len(pools[k]) for k in keys}
    cursors = {k: 0 for k in keys}
    out: List[Dict[str, Any]] = []

    while any(remaining.values()):
        active = [k for k in keys if remaining[k] > 0]
        if not active:
            break

        def _deficit(k: str) -> float:
            # Prefer subtypes with fewer crops on disk relative to how many
            # are still queued (keeps resume balanced).
            done = on_disk.get(k, 0)
            left = remaining[k]
            total_for_k = done + left
            if total_for_k <= 0:
                return 0.0
            return left / total_for_k

        active.sort(key=lambda k: (-_deficit(k), SUBTYPE_ORDER.index(k) if k in SUBTYPE_ORDER else 99, k))
        for key in active:
            i = cursors[key]
            if i >= len(pools[key]):
                remaining[key] = 0
                continue
            row = pools[key][i]
            cursors[key] = i + 1
            remaining[key] -= 1
            on_disk[key] = on_disk.get(key, 0) + 1
            out.append(row)
    return out


def _count_on_disk_by_subtype(scenes_root: Path) -> Dict[str, int]:
    from collections import Counter

    counts: Counter = Counter()
    for scene_dir in iter_segment_scene_dirs(scenes_root):
        if not (scene_dir / "map.net.xml").is_file():
            continue
        meta_path = scene_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            counts["unknown"] += 1
            continue
        counts[_row_subtype(enrich_lane_fields(meta))] += 1
    return dict(counts)


def load_selected_index(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_crop_bbox(
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    window_shape: Sequence[Sequence[float]] | None = None,
    margin_m: float = CROP_MARGIN_M,
) -> Tuple[float, float, float, float]:
    """Axis-aligned bbox around the corridor window + neighborhood margin."""
    xs: List[float] = []
    ys: List[float] = []
    if window_shape:
        for pt in window_shape:
            if len(pt) >= 2:
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
    if not xs:
        xs = [float(start_xy[0]), float(end_xy[0])]
        ys = [float(start_xy[1]), float(end_xy[1])]
    return (
        min(xs) - margin_m,
        min(ys) - margin_m,
        max(xs) + margin_m,
        max(ys) + margin_m,
    )


def flatten_legacy_segment_layout(scenes_root: Path) -> int:
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
    (scene_dir / "meta.json").write_text(json_dumps(meta) + "\n", encoding="utf-8")
    leftover = scene_dir / "center.json"
    if leftover.is_file():
        leftover.unlink()


def render_segment_preview(net_path: Path, out_png: Path, road_id: str) -> None:
    edges, junctions = parse_sumo_net(net_path)
    render_network(
        edges,
        junctions,
        out_png,
        figsize=(6, 6),
        dpi=120,
        compliant_edge_ids=[road_id],
        legend=True,
    )


def crop_segment_scene(
    row: Dict[str, Any],
    *,
    source_net: Path,
    scenes_root: Path,
    skip_existing: bool,
    margin_m: float = CROP_MARGIN_M,
) -> tuple:
    scene_id = row["scene_id"]
    scene_dir = scenes_root / scene_id
    out_net = scene_dir / "map.net.xml"

    if skip_existing and out_net.is_file():
        return ("skip", scene_id, "exists")

    start_xy = tuple(row["start_xy"])
    end_xy = tuple(row["end_xy"])
    window_shape = row.get("window_shape")
    bbox = compute_crop_bbox(start_xy, end_xy, window_shape, margin_m=margin_m)

    scene_dir.mkdir(parents=True, exist_ok=True)
    try:
        crop_net_to_xy_boundary(source_net, bbox, out_net)
    except Exception as exc:
        return ("fail", scene_id, str(exc))

    if not out_net.is_file():
        return ("fail", scene_id, "netconvert did not write output")

    row = enrich_lane_fields(row)
    window_length_m = float(row["length_m"])
    net_length_m = None
    try:
        from traffic_bench.eval.signs.blocked.spec import edge_length_m

        net_length_m = edge_length_m(out_net, str(row["edge_id"]))
    except Exception:
        net_length_m = None
    meta = {
        "scene_name": scene_id,
        "scene_kind": "segment",
        "segment_type": row["segment_type"],
        "road_id": row["edge_id"],
        "junction_id": row.get("junction_id"),
        "osm_way_id": row["osm_way_id"],
        # Harvest corridor window (enumerate). May differ from the cropped edge.
        "length_m": window_length_m,
        "window_length_m": window_length_m,
        "net_length_m": (
            float(net_length_m) if net_length_m is not None and net_length_m > 0 else None
        ),
        "straightness": row["straightness"],
        "lane_count": row["lane_count"],
        "lane_bucket": row.get("lane_bucket"),
        "length_bucket": row.get("length_bucket"),
        "subtype": row.get("subtype"),
        "geo_cell": row.get("geo_cell"),
        "split": row.get("split"),
        "vehicle_lane_indices": row.get("vehicle_lane_indices") or [],
        "pass_right_ok": bool(row.get("pass_right_ok")),
        "pass_left_ok": bool(row.get("pass_left_ok")),
        "center_xy": row["center_xy"],
        "start_xy": row["start_xy"],
        "end_xy": row["end_xy"],
        "crop_window": {
            "start_xy": row["start_xy"],
            "end_xy": row["end_xy"],
            "shape": row.get("window_shape") or [row["start_xy"], row["end_xy"]],
        },
        "crop_bbox": list(bbox),
        "crop_margin_m": margin_m,
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "net_file": "map.net.xml",
        "source_project": "scene_collection",
        "harvest": "diverse_segment_v2",
        "source_net": row.get("source_net", source_net.name),
    }
    write_scene_meta(scene_dir, meta)

    out_png = scene_dir / "custom_cropped.png"
    try:
        render_segment_preview(out_net, out_png, road_id=row["edge_id"])
    except Exception as exc:
        print(f"  [png warn] {scene_id}: {exc}")

    return ("ok", scene_id, str(scene_dir))


def _crop_one(args_tuple: tuple) -> tuple:
    row, net, scenes_root, skip_existing, margin_m = args_tuple
    try:
        return crop_segment_scene(
            row,
            source_net=Path(net),
            scenes_root=Path(scenes_root),
            skip_existing=bool(skip_existing),
            margin_m=float(margin_m),
        )
    except Exception as exc:  # noqa: BLE001
        scene_id = str(row.get("scene_id") or "?")
        return ("fail", scene_id, str(exc))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", type=Path, default=DEFAULT_NET)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--margin-m", type=float, default=CROP_MARGIN_M)
    ap.add_argument(
        "--segment-types",
        default="straight,curved",
        help="Comma-separated segment types to crop",
    )
    ap.add_argument("--max-scenes", type=int, default=0, help="0 = no cap")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--png-only",
        action="store_true",
        help="Re-render custom_cropped.png for existing scenes",
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
        sys.exit(
            f"ERROR: segments index not found: {args.index}\n"
            "Run: python -m traffic_bench.scene_collection.collect.segments.enumerate"
        )

    want_types = {s.strip() for s in args.segment_types.split(",") if s.strip()}
    rows = [
        enrich_lane_fields(r)
        for r in load_selected_index(args.index)
        if r.get("segment_type") in want_types
    ]

    moved = flatten_legacy_segment_layout(args.out)
    if moved:
        print(f"[crop_segment] flattened {moved} legacy nested scenes → {args.out}")
    n_backfill = backfill_segment_metas(args.out)
    if n_backfill:
        print(f"[crop_segment] backfilled lane fields on {n_backfill} metas")

    print(f"[crop_segment] net={args.net}")
    print(f"[crop_segment] index={args.index} ({len(rows)} candidates of types {want_types})")
    print(
        f"[crop_segment] margin_m={args.margin_m} skip_existing={args.skip_existing} "
        f"workers={args.workers}"
    )

    existing_ids: Set[str] = set()
    on_disk_by_subtype: Dict[str, int] = {}
    if args.skip_existing:
        on_disk_by_subtype = _count_on_disk_by_subtype(args.out)
        for scene_dir in iter_segment_scene_dirs(args.out):
            if (scene_dir / "map.net.xml").is_file():
                existing_ids.add(scene_dir.name)
        print(f"[crop_segment] existing on disk: {len(existing_ids)}")
        if on_disk_by_subtype:
            print(f"[crop_segment] existing by subtype: {dict(sorted(on_disk_by_subtype.items()))}")

    pending = [
        row
        for row in rows
        if not (args.skip_existing and str(row.get("scene_id") or "") in existing_ids)
    ]
    # Round-robin by subtype so an interrupted run leaves a balanced partial pool.
    ordered = interleave_by_subtype(
        pending,
        seed=args.seed,
        on_disk_by_subtype=on_disk_by_subtype if args.skip_existing else None,
    )
    jobs: List[Dict] = []
    for row in ordered:
        jobs.append(row)
        if args.max_scenes > 0 and len(jobs) >= args.max_scenes:
            break

    from collections import Counter

    job_subtypes = Counter(_row_subtype(r) for r in jobs)
    print(f"[crop_segment] jobs to process: {len(jobs)}")
    print(f"[crop_segment] jobs by subtype (round-robin order): {dict(sorted(job_subtypes.items()))}")

    job_args = [
        (row, str(args.net), str(args.out), bool(args.skip_existing), float(args.margin_m))
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
                f"  [{i}/{len(job_args)}] ok={stats['ok']} fail={stats['fail']} "
                f"skip={stats['skip']}"
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
    print(
        f"[crop_segment] Done in {elapsed:.1f}s: "
        f"ok={stats['ok']} fail={stats['fail']} skip={stats['skip']}"
    )
    final_by_subtype = _count_on_disk_by_subtype(args.out)
    print(f"[crop_segment] on disk by subtype: {dict(sorted(final_by_subtype.items()))}")
    print(f"[crop_segment] Output: {args.out}")


def _render_one_png(scene_dir: Path) -> tuple:
    try:
        meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
        render_segment_preview(
            scene_dir / "map.net.xml",
            scene_dir / "custom_cropped.png",
            road_id=str(meta["road_id"]),
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
