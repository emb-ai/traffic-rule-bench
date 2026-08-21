#!/usr/bin/env python3
"""Enumerate straight road segments from incoming edges → index/segments.jsonl.

Source: incoming_edge_ids from index/junctions.jsonl (already harvested).
This reuses the existing junction index to find long straight road segments
suitable for speed/pedestrian/detour sign scenes.

Segments are edges leading INTO junctions, filtered by:
- Length >= 150m (configurable, based on braking physics)
- Straightness >= 0.97 (curved) or >= 0.99 (straight)

Train/test split uses osm_way_id (not edge_id) to prevent data leakage
between segments of the same physical road.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]

from traffic_bench.scene_collection.map_pool.lib.segment import (
    CURVED_THRESHOLD,
    MIN_SEGMENT_LENGTH_M,
    STRAIGHT_THRESHOLD,
    SegmentCandidate,
    build_edge_metrics_cache,
    build_junction_positions_cache,
    osm_way_id_from_edge,
)
from traffic_bench.scene_collection.map_pool.scripts.geo_utils import net_xy_to_latlon_proj

DEFAULT_NET = ROOT / "nets" / "moscow.net.xml"
DEFAULT_JUNCTIONS_INDEX = ROOT / "index" / "junctions.jsonl"
DEFAULT_OUT = ROOT / "index" / "segments.jsonl"
DEFAULT_SUMMARY = ROOT / "index" / "segments_summary.json"


def load_junction_index(path: Path) -> List[Dict[str, Any]]:
    """Load junctions.jsonl."""
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def enumerate_segments(
    net_path: Path,
    junctions_index: List[Dict[str, Any]],
    *,
    min_length_m: float = MIN_SEGMENT_LENGTH_M,
    min_straightness: float = CURVED_THRESHOLD,
) -> List[SegmentCandidate]:
    """Find segment candidates from incoming edges of indexed junctions."""
    print(f"[enumerate_segments] Building edge metrics cache from {net_path}...")
    edge_cache = build_edge_metrics_cache(net_path)
    print(f"[enumerate_segments] Cached {len(edge_cache)} edges")

    print("[enumerate_segments] Building junction positions cache...")
    junction_cache = build_junction_positions_cache(net_path)
    print(f"[enumerate_segments] Cached {len(junction_cache)} junctions")

    seen_edges: Set[str] = set()
    candidates: List[SegmentCandidate] = []

    for row in junctions_index:
        junction_id = str(row.get("junction_id", ""))
        incoming = row.get("incoming_edge_ids") or []

        junction_xy = junction_cache.get(junction_id)
        if junction_xy is None:
            continue

        for edge_id in incoming:
            if edge_id in seen_edges:
                continue
            seen_edges.add(edge_id)

            metrics = edge_cache.get(edge_id)
            if metrics is None:
                continue

            length_m = float(metrics["length_m"])
            straightness = float(metrics["straightness"])

            if length_m < min_length_m:
                continue
            if straightness < min_straightness:
                continue

            osm_way = osm_way_id_from_edge(edge_id)

            candidate = SegmentCandidate(
                edge_id=edge_id,
                junction_id=junction_id,
                osm_way_id=osm_way,
                length_m=length_m,
                straightness=straightness,
                lane_count=int(metrics["lane_count"]),
                center_xy=tuple(metrics["center_xy"]),
                start_xy=tuple(metrics["start_xy"]),
                end_xy=tuple(metrics["end_xy"]),
                to_junction_xy=junction_xy,
                vehicle_lane_indices=tuple(metrics.get("vehicle_lane_indices") or ()),
                pass_right_ok=bool(metrics.get("pass_right_ok", False)),
                pass_left_ok=bool(metrics.get("pass_left_ok", False)),
            )
            candidates.append(candidate)

    return candidates


def write_segments_index(
    candidates: List[SegmentCandidate],
    net_path: Path,
    out_path: Path,
    summary_path: Path,
) -> None:
    """Write segments.jsonl and summary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        lat, lon = net_xy_to_latlon_proj(
            net_path, cand.center_xy[0], cand.center_xy[1]
        )
        rows.append({
            "edge_id": cand.edge_id,
            "scene_id": cand.scene_id(),
            "junction_id": cand.junction_id,
            "osm_way_id": cand.osm_way_id,
            "length_m": round(cand.length_m, 2),
            "straightness": round(cand.straightness, 5),
            "segment_type": cand.segment_type,
            "lane_count": cand.lane_count,
            "vehicle_lane_indices": list(cand.vehicle_lane_indices),
            "pass_right_ok": cand.pass_right_ok,
            "pass_left_ok": cand.pass_left_ok,
            "center_xy": [round(cand.center_xy[0], 2), round(cand.center_xy[1], 2)],
            "start_xy": [round(cand.start_xy[0], 2), round(cand.start_xy[1], 2)],
            "end_xy": [round(cand.end_xy[0], 2), round(cand.end_xy[1], 2)],
            "to_junction_xy": [round(cand.to_junction_xy[0], 2), round(cand.to_junction_xy[1], 2)],
            "latitude": lat,
            "longitude": lon,
            "source_net": net_path.name,
        })

    # Sort by segment_type, then by length (longest first)
    rows.sort(key=lambda r: (r["segment_type"], -r["length_m"]))

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary statistics
    by_type: Dict[str, int] = {}
    by_type_lanes: Dict[str, Dict[int, int]] = {}
    lengths: List[float] = []
    for row in rows:
        seg_type = row["segment_type"]
        by_type[seg_type] = by_type.get(seg_type, 0) + 1
        lengths.append(row["length_m"])

        if seg_type not in by_type_lanes:
            by_type_lanes[seg_type] = {}
        lc = row["lane_count"]
        by_type_lanes[seg_type][lc] = by_type_lanes[seg_type].get(lc, 0) + 1

    summary = {
        "total": len(rows),
        "by_segment_type": by_type,
        "by_type_and_lanes": {k: dict(sorted(v.items())) for k, v in by_type_lanes.items()},
        "length_stats": {
            "min": round(min(lengths), 1) if lengths else 0,
            "max": round(max(lengths), 1) if lengths else 0,
            "median": round(sorted(lengths)[len(lengths) // 2], 1) if lengths else 0,
        },
        "thresholds": {
            "straight": STRAIGHT_THRESHOLD,
            "curved": CURVED_THRESHOLD,
            "min_length_m": MIN_SEGMENT_LENGTH_M,
        },
        "index_file": out_path.name,
        "source_net": net_path.name,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[enumerate_segments] Wrote {out_path} ({len(rows)} segments)")
    print(f"[enumerate_segments] Summary: {by_type}")
    print(f"[enumerate_segments] By type and lanes: {by_type_lanes}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", type=Path, default=DEFAULT_NET)
    ap.add_argument("--junctions-index", type=Path, default=DEFAULT_JUNCTIONS_INDEX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument(
        "--min-length-m",
        type=float,
        default=MIN_SEGMENT_LENGTH_M,
        help=f"Minimum segment length (default {MIN_SEGMENT_LENGTH_M}m)",
    )
    ap.add_argument(
        "--min-straightness",
        type=float,
        default=CURVED_THRESHOLD,
        help=f"Minimum straightness ratio (default {CURVED_THRESHOLD})",
    )
    args = ap.parse_args()

    if not args.net.is_file():
        sys.exit(f"ERROR: net not found: {args.net}")
    if not args.junctions_index.is_file():
        sys.exit(f"ERROR: junctions index not found: {args.junctions_index}")

    print(f"[enumerate_segments] net={args.net}")
    print(f"[enumerate_segments] junctions_index={args.junctions_index}")
    print(f"[enumerate_segments] min_length_m={args.min_length_m}, min_straightness={args.min_straightness}")

    junctions = load_junction_index(args.junctions_index)
    print(f"[enumerate_segments] Loaded {len(junctions)} junctions from index")

    candidates = enumerate_segments(
        args.net,
        junctions,
        min_length_m=args.min_length_m,
        min_straightness=args.min_straightness,
    )
    print(f"[enumerate_segments] Found {len(candidates)} segment candidates")

    write_segments_index(candidates, args.net, args.out, args.summary_out)


if __name__ == "__main__":
    main()
