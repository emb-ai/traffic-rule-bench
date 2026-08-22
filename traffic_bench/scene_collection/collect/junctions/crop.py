#!/usr/bin/env python3
"""Crop indexed junctions into crops/junction/{T,X,O}/<scene_id>/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from traffic_bench.eval.core.layout.junction_crop import (
    JunctionPick,
    _find_netconvert,
    crop_net_to_junction_only,
    json_dumps,
)
from traffic_bench.eval.core.layout.junction_priority_layout import JunctionLayoutError
from traffic_bench.scene_collection.paths import JUNCTION_CROPS, JUNCTIONS_INDEX, MOSCOW_NET

DEFAULT_NET = MOSCOW_NET
DEFAULT_INDEX = JUNCTIONS_INDEX
DEFAULT_SCENES = JUNCTION_CROPS


def _load_index(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_scene_meta(scene_dir: Path, meta: dict) -> None:
    (scene_dir / "meta.json").write_text(
        json_dumps(meta) + "\n", encoding="utf-8"
    )
    leftover = scene_dir / "center.json"
    if leftover.is_file():
        leftover.unlink()


def _crop_keep_edges(source_net: Path, edge_ids: List[str], out_net: Path) -> None:
    import tempfile
    import subprocess

    out_net.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(sorted(set(edge_ids))))
        edge_list = Path(handle.name)
    try:
        cmd = [
            _find_netconvert(),
            "--sumo-net-file",
            str(source_net),
            "-o",
            str(out_net),
            "--keep-edges.input-file",
            str(edge_list),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise JunctionLayoutError(
                f"netconvert crop failed for {source_net}:\n"
                f"{result.stderr or result.stdout}"
            )
    finally:
        edge_list.unlink(missing_ok=True)
    if not out_net.is_file():
        raise JunctionLayoutError(f"netconvert did not write {out_net}")


def _roundabout_keep_edges(source_net: Path, row: Dict[str, Any]) -> List[str]:
    """Ring edges + any normal edge that touches a ring node (spokes)."""
    import xml.etree.ElementTree as ET

    ring_nodes = set(row.get("ring_node_ids") or [])
    keep = set(row.get("ring_edge_ids") or [])
    root = ET.parse(source_net).getroot()
    for edge in root.findall("edge"):
        eid = edge.get("id")
        if not eid or eid.startswith(":") or edge.get("function") == "internal":
            continue
        if edge.get("from") in ring_nodes or edge.get("to") in ring_nodes:
            keep.add(eid)
    return sorted(keep)


def crop_tx_row(
    row: Dict[str, Any],
    *,
    source_net: Path,
    scenes_root: Path,
    radius_m: float,
    skip_existing: bool,
) -> Optional[Path]:
    shape = row["shape"]
    scene_id = row["scene_id"]
    scene_dir = scenes_root / shape / scene_id
    out_net = scene_dir / "map.net.xml"
    if skip_existing and out_net.is_file():
        return scene_dir

    scene_dir.mkdir(parents=True, exist_ok=True)
    pick = JunctionPick(
        junction_id=str(row["junction_id"]),
        center_xy=(float(row["center_xy"][0]), float(row["center_xy"][1])),
        total_lanes=int(row.get("total_lanes") or 0),
        incoming_edge_ids=tuple(row.get("incoming_edge_ids") or ()),
        arm_count=int(row["arm_count"]),
    )
    crop_net_to_junction_only(
        source_net,
        pick.junction_id,
        out_net,
        arm_length_m=radius_m,
    )
    meta = {
        "scene_name": scene_id,
        "scene_kind": "junction",
        "shape": shape,
        "junction_id": pick.junction_id,
        "junction_arm_count": pick.arm_count,
        "junction_type": row.get("junction_type"),
        "total_lanes": pick.total_lanes,
        "incoming_edge_ids": list(pick.incoming_edge_ids),
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "crop_radius_m": radius_m,
        "net_file": "map.net.xml",
        "source_net": source_net.name,
        "source_project": "scene_collection",
        "harvest": "sign_free_moscow_osm",
    }
    _write_scene_meta(scene_dir, meta)
    return scene_dir


def crop_o_row(
    row: Dict[str, Any],
    *,
    source_net: Path,
    scenes_root: Path,
    radius_m: float,
    skip_existing: bool,
) -> Optional[Path]:
    scene_id = row["scene_id"]
    scene_dir = scenes_root / "O" / scene_id
    out_net = scene_dir / "map.net.xml"
    if skip_existing and out_net.is_file():
        return scene_dir

    scene_dir.mkdir(parents=True, exist_ok=True)
    keep = _roundabout_keep_edges(source_net, row)
    if len(keep) < 2:
        raise JunctionLayoutError(
            f"roundabout {scene_id}: fewer than 2 edges to keep"
        )
    _crop_keep_edges(source_net, keep, out_net)
    meta = {
        "scene_name": scene_id,
        "scene_kind": "roundabout",
        "shape": "O",
        "junction_id": row["junction_id"],
        "roundabout_fingerprint": row.get("roundabout_fingerprint"),
        "ring_node_ids": row.get("ring_node_ids"),
        "ring_edge_ids": row.get("ring_edge_ids"),
        "kept_edge_ids": keep,
        "junction_type": "roundabout",
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "crop_radius_m": radius_m,
        "net_file": "map.net.xml",
        "source_net": source_net.name,
        "source_project": "scene_collection",
        "harvest": "sign_free_moscow_osm",
    }
    _write_scene_meta(scene_dir, meta)
    return scene_dir


def _crop_one(args_tuple):
    """Worker for parallel crop. Returns (status, shape, scene_id, detail)."""
    row, net, scenes_root, radius_m, skip_existing = args_tuple
    shape = row["shape"]
    scene_id = row["scene_id"]
    try:
        if shape in {"T", "X"}:
            path = crop_tx_row(
                row,
                source_net=Path(net),
                scenes_root=Path(scenes_root),
                radius_m=radius_m,
                skip_existing=skip_existing,
            )
        elif shape == "O":
            path = crop_o_row(
                row,
                source_net=Path(net),
                scenes_root=Path(scenes_root),
                radius_m=max(radius_m, 100.0),
                skip_existing=skip_existing,
            )
        else:
            return ("skip", shape, scene_id, "unknown_shape")
        if path is None:
            return ("skip", shape, scene_id, "exists")
        return ("ok", shape, scene_id, str(path))
    except Exception as exc:  # noqa: BLE001 — collect failures in worker
        return ("fail", shape, scene_id, str(exc))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", type=Path, default=DEFAULT_NET)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--scenes-root", type=Path, default=DEFAULT_SCENES)
    ap.add_argument("--radius-m", type=float, default=80.0)
    ap.add_argument("--max-per-shape", type=int, default=None)
    ap.add_argument("--shapes", default="T,X,O")
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
        help="Write custom_cropped.png for existing crops (no netconvert)",
    )
    args = ap.parse_args()

    if args.png_only:
        from traffic_bench.scene_collection.preview import backfill_previews

        backfill_previews(
            args.scenes_root,
            skip_existing=args.skip_existing,
            workers=args.workers,
        )
        return

    if not args.net.is_file():
        sys.exit(f"ERROR: net not found: {args.net}")
    if not args.index.is_file():
        sys.exit(f"ERROR: index not found: {args.index} (run: python -m traffic_bench.scene_collection collect)")

    want = {s.strip().upper() for s in args.shapes.split(",") if s.strip()}
    rows = [r for r in _load_index(args.index) if r.get("shape") in want]

    # Cap per shape (stable order already in index).
    if args.max_per_shape is not None:
        capped: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        for row in rows:
            sh = row["shape"]
            n = counts.get(sh, 0)
            if n >= args.max_per_shape:
                continue
            counts[sh] = n + 1
            capped.append(row)
        rows = capped

    print(
        f"[crop] {len(rows)} scenes from {args.index.name} "
        f"(shapes={sorted(want)}, max_per_shape={args.max_per_shape}, "
        f"workers={args.workers})"
    )

    jobs = [
        (
            row,
            str(args.net),
            str(args.scenes_root),
            float(args.radius_m),
            bool(args.skip_existing),
        )
        for row in rows
    ]

    ok = fail = skip = 0
    workers = max(1, int(args.workers))
    if workers == 1:
        results = (_crop_one(job) for job in jobs)
        for i, (status, shape, scene_id, detail) in enumerate(results, 1):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"  [fail] {shape}/{scene_id}: {detail}")
            if i % 25 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] ok={ok} fail={fail} skip={skip}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_crop_one, job) for job in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                status, shape, scene_id, detail = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    fail += 1
                    print(f"  [fail] {shape}/{scene_id}: {detail}")
                if i % 25 == 0 or i == len(jobs):
                    print(f"  [{i}/{len(jobs)}] ok={ok} fail={fail} skip={skip}")

    print(f"[crop] Done: ok={ok} fail={fail} skip={skip} → {args.scenes_root}")


if __name__ == "__main__":
    main()
