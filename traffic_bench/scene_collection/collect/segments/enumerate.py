#!/usr/bin/env python3
"""Enumerate corridor candidates from the full Moscow SUMO net.

Unlike the legacy junction-incoming harvest, this scans every vehicle edge,
cuts a mid-corridor window (≥150 m), and keeps one candidate per osm_way_id.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from traffic_bench.scene_collection.collect.lib.geo import net_xy_to_latlon_proj
from traffic_bench.scene_collection.collect.segments.metrics import (
    CURVED_THRESHOLD,
    MIN_SEGMENT_LENGTH_M,
    TARGET_WINDOW_M,
    SegmentCandidate,
    dedupe_by_osm_way,
    enumerate_edge_candidates,
)
from traffic_bench.scene_collection.paths import INDEX, MOSCOW_NET, SEGMENTS_INDEX

DEFAULT_NET = MOSCOW_NET
DEFAULT_OUT = SEGMENTS_INDEX
DEFAULT_SUMMARY = INDEX / "segments_summary.json"


def candidate_to_row(cand: SegmentCandidate, net_path: Path) -> Dict[str, Any]:
    lat, lon = net_xy_to_latlon_proj(net_path, cand.center_xy[0], cand.center_xy[1])
    return {
        "edge_id": cand.edge_id,
        "scene_id": cand.scene_id(),
        "junction_id": cand.junction_id or None,
        "osm_way_id": cand.osm_way_id,
        "length_m": round(cand.length_m, 2),
        "straightness": round(cand.straightness, 5),
        "segment_type": cand.segment_type,
        "lane_count": cand.lane_count,
        "lane_bucket": cand.lane_bucket,
        "length_bucket": cand.length_bucket,
        "subtype": cand.subtype,
        "geo_cell": cand.geo_cell,
        "vehicle_lane_indices": list(cand.vehicle_lane_indices),
        "pass_right_ok": cand.pass_right_ok,
        "pass_left_ok": cand.pass_left_ok,
        "center_xy": [round(cand.center_xy[0], 2), round(cand.center_xy[1], 2)],
        "start_xy": [round(cand.start_xy[0], 2), round(cand.start_xy[1], 2)],
        "end_xy": [round(cand.end_xy[0], 2), round(cand.end_xy[1], 2)],
        "window_shape": [
            [round(x, 2), round(y, 2)] for x, y in cand.window_shape
        ],
        "latitude": lat,
        "longitude": lon,
        "source_net": net_path.name,
        "harvest": "diverse_segment_v2",
    }


def write_segments_index(
    candidates: List[SegmentCandidate],
    net_path: Path,
    out_path: Path,
    summary_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [candidate_to_row(c, net_path) for c in candidates]
    rows.sort(key=lambda r: (r["subtype"], -r["length_m"]))

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_type: Counter = Counter()
    by_subtype: Counter = Counter()
    by_lanes: Dict[str, Counter] = {}
    lengths: List[float] = []
    for row in rows:
        seg_type = row["segment_type"]
        by_type[seg_type] += 1
        by_subtype[row["subtype"]] += 1
        lengths.append(row["length_m"])
        by_lanes.setdefault(seg_type, Counter())[row["lane_count"]] += 1

    summary = {
        "total": len(rows),
        "n_osm_ways": len({r["osm_way_id"] for r in rows}),
        "by_segment_type": dict(by_type),
        "by_subtype": dict(sorted(by_subtype.items())),
        "by_type_and_lanes": {
            k: dict(sorted(v.items())) for k, v in sorted(by_lanes.items())
        },
        "length_stats": {
            "min": round(min(lengths), 1) if lengths else 0,
            "max": round(max(lengths), 1) if lengths else 0,
            "median": round(sorted(lengths)[len(lengths) // 2], 1) if lengths else 0,
        },
        "thresholds": {
            "straight": 0.99,
            "curved": CURVED_THRESHOLD,
            "min_length_m": MIN_SEGMENT_LENGTH_M,
            "target_window_m": TARGET_WINDOW_M,
        },
        "index_file": out_path.name,
        "source_net": net_path.name,
        "harvest": "diverse_segment_v2",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[enumerate_segments] Wrote {out_path} ({len(rows)} ways)")
    print(f"[enumerate_segments] by_subtype: {dict(sorted(by_subtype.items()))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", type=Path, default=DEFAULT_NET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--min-length-m", type=float, default=MIN_SEGMENT_LENGTH_M)
    ap.add_argument("--min-straightness", type=float, default=CURVED_THRESHOLD)
    ap.add_argument("--target-window-m", type=float, default=TARGET_WINDOW_M)
    args = ap.parse_args()

    if not args.net.is_file():
        sys.exit(f"ERROR: net not found: {args.net}")

    print(f"[enumerate_segments] net={args.net}")
    print(
        f"[enumerate_segments] min_length_m={args.min_length_m} "
        f"min_straightness={args.min_straightness} "
        f"target_window_m={args.target_window_m}"
    )

    raw = enumerate_edge_candidates(
        args.net,
        min_length_m=args.min_length_m,
        min_straightness=args.min_straightness,
        target_window_m=args.target_window_m,
    )
    print(f"[enumerate_segments] raw edge windows: {len(raw)}")
    candidates = dedupe_by_osm_way(raw)
    print(f"[enumerate_segments] unique osm_way_id: {len(candidates)}")

    write_segments_index(candidates, args.net, args.out, args.summary_out)


if __name__ == "__main__":
    main()
