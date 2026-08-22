#!/usr/bin/env python3
"""Enumerate T / X / O junctions from moscow.net.xml → index/junctions.jsonl."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from traffic_bench.scene_collection.collect.lib.junction_crop import (
    collect_intersection_junction_candidates,
)
from traffic_bench.eval.engine.map.junction_priority_layout import INTERSECTION_JUNCTION_TYPES
from traffic_bench.scene_collection.collect.lib.geo import net_xy_to_latlon_proj
from traffic_bench.scene_collection.paths import INDEX, JUNCTIONS_INDEX, MOSCOW_NET

DEFAULT_NET = MOSCOW_NET
DEFAULT_INDEX = JUNCTIONS_INDEX
DEFAULT_SUMMARY = INDEX / "junctions_summary.json"


def _roundabout_fingerprint(node_ids: Iterable[str], edge_ids: Iterable[str]) -> str:
    payload = "nodes=" + ",".join(sorted(node_ids)) + "|edges=" + ",".join(sorted(edge_ids))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _junction_centers(net_path: Path) -> Dict[str, Tuple[float, float]]:
    root = ET.parse(net_path).getroot()
    centers: Dict[str, Tuple[float, float]] = {}
    for j in root.findall("junction"):
        jid = j.get("id")
        if not jid:
            continue
        try:
            centers[jid] = (float(j.get("x", 0)), float(j.get("y", 0)))
        except ValueError:
            continue
    return centers


def enumerate_tx(
    net_path: Path,
    *,
    min_lane_m: float,
    shapes: Set[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not (shapes & {"T", "X"}):
        return rows

    arm_counts: Tuple[int, ...] = tuple(
        n for n, label in ((4, "X"), (3, "T")) if label in shapes
    )
    picks = collect_intersection_junction_candidates(
        net_path,
        min_lane_length_m=min_lane_m,
        arm_counts=arm_counts,
    )
    # Prefer richer junctions first (stable for --max-per-shape cropping).
    picks.sort(key=lambda pick: (-pick.arm_count, -pick.total_lanes, pick.junction_id))
    root = ET.parse(net_path).getroot()
    jtype = {
        j.get("id"): j.get("type", "unknown")
        for j in root.findall("junction")
        if j.get("id")
    }

    for rank, pick in enumerate(picks):
        shape = "X" if pick.arm_count == 4 else "T" if pick.arm_count == 3 else "2"
        if shape not in shapes:
            continue
        lat, lon = net_xy_to_latlon_proj(
            net_path, pick.center_xy[0], pick.center_xy[1]
        )
        rows.append(
            {
                "shape": shape,
                "junction_id": pick.junction_id,
                "scene_id": f"junc_{pick.junction_id}",
                "arm_count": pick.arm_count,
                "total_lanes": pick.total_lanes,
                "incoming_edge_ids": list(pick.incoming_edge_ids),
                "junction_type": jtype.get(pick.junction_id, "unknown"),
                "center_xy": [pick.center_xy[0], pick.center_xy[1]],
                "latitude": lat,
                "longitude": lon,
                "rank": rank,
                "source_net": str(net_path.name),
            }
        )
    return rows


def enumerate_roundabouts(net_path: Path) -> List[Dict[str, Any]]:
    root = ET.parse(net_path).getroot()
    centers = _junction_centers(net_path)
    rows: List[Dict[str, Any]] = []

    for rb in root.findall("roundabout"):
        node_ids = [t for t in (rb.get("nodes") or "").split() if t]
        edge_ids = [t for t in (rb.get("edges") or "").split() if t]
        if len(edge_ids) < 2 or len(node_ids) < 2:
            continue
        xs, ys = [], []
        for nid in node_ids:
            if nid in centers:
                xs.append(centers[nid][0])
                ys.append(centers[nid][1])
        if not xs:
            continue
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        lat, lon = net_xy_to_latlon_proj(net_path, cx, cy)
        fp = _roundabout_fingerprint(node_ids, edge_ids)
        # Prefer a stable representative junction id (first sorted ring node).
        rep_id = sorted(node_ids)[0]
        rows.append(
            {
                "shape": "O",
                "junction_id": rep_id,
                "roundabout_fingerprint": fp,
                "scene_id": f"rb_{fp}",
                "arm_count": None,
                "ring_node_ids": node_ids,
                "ring_edge_ids": edge_ids,
                "n_ring_edges": len(edge_ids),
                "n_ring_nodes": len(node_ids),
                "junction_type": "roundabout",
                "center_xy": [cx, cy],
                "latitude": lat,
                "longitude": lon,
                "source_net": str(net_path.name),
            }
        )
    return rows


def write_index(
    rows: List[Dict[str, Any]],
    index_path: Path,
    summary_path: Path,
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_shape: Dict[str, int] = {}
    for row in rows:
        by_shape[row["shape"]] = by_shape.get(row["shape"], 0) + 1
    summary = {
        "total": len(rows),
        "by_shape": by_shape,
        "index_file": str(index_path.name),
        "intersection_types_kept": sorted(INTERSECTION_JUNCTION_TYPES),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[enumerate] Wrote {index_path} ({len(rows)} rows)")
    print(f"[enumerate] Summary: {summary['by_shape']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", type=Path, default=DEFAULT_NET)
    ap.add_argument("--index-out", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--min-lane-m", type=float, default=10.0)
    ap.add_argument(
        "--shapes",
        default="T,X,O",
        help="Comma-separated subset of T,X,O",
    )
    args = ap.parse_args()

    if not args.net.is_file():
        sys.exit(f"ERROR: net not found: {args.net} (run build_net.py first)")

    shapes = {s.strip().upper() for s in args.shapes.split(",") if s.strip()}
    print(f"[enumerate] Scanning {args.net} for shapes={sorted(shapes)}")

    rows: List[Dict[str, Any]] = []
    rows.extend(enumerate_tx(args.net, min_lane_m=args.min_lane_m, shapes=shapes))
    if "O" in shapes:
        rows.extend(enumerate_roundabouts(args.net))

    # Stable order: shape then id
    rows.sort(key=lambda r: (r["shape"], r.get("scene_id") or r["junction_id"]))
    write_index(rows, args.index_out, args.summary_out)


if __name__ == "__main__":
    main()
