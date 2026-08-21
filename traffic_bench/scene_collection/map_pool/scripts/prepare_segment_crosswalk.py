#!/usr/bin/env python3
"""Add pedestrian crosswalks to segment scenes for sign 5.19 benchmarking.

This script transforms segment scenes (straight roads without intersections)
into crosswalk scenes by:
1. Splitting the main road edge to create a mid-block junction
2. Adding sidewalk lanes to the edges
3. Defining a pedestrian crossing at the new junction
4. Running netconvert to generate the full pedestrian infrastructure

Input:  crops/segment/{straight,curved}/<scene_id>/
Output: crops/segment_crosswalk/{straight,curved}/<scene_id>_cw_<position>/

Examples:
    # Process all segment scenes, all positions
    python scripts/prepare_segment_crosswalk.py

    # Process specific scenes
    python scripts/prepare_segment_crosswalk.py --scenes seg_72843492_0 seg_31298749_0

    # Specific position only (near_start, middle, near_end)
    python scripts/prepare_segment_crosswalk.py --positions middle

    # Limit per segment type
    python scripts/prepare_segment_crosswalk.py --max-per-type 10

    # Skip existing scenes
    python scripts/prepare_segment_crosswalk.py --skip-existing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

from traffic_bench.scene_collection.map_pool.lib.crosswalk_inject import (
    CrosswalkInjection,
    calculate_crosswalk_positions,
    find_paired_edges,
    identify_main_edges,
    inject_crosswalk,
    validate_crosswalk_net,
)

DEFAULT_SEGMENT_SCENES = ROOT / "crops" / "segment"
DEFAULT_OUTPUT = ROOT / "crops" / "segment_crosswalk"
DEFAULT_POSITIONS = ["near_start", "middle", "near_end"]
DEFAULT_MAX_PER_TYPE = 500


def json_dumps(obj) -> str:
    """Compact JSON dump."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def discover_segment_scenes(
    scenes_root: Path,
    segment_types: Optional[List[str]] = None,
) -> List[Path]:
    """Find segment scene directories (flat or nested straight/curved)."""
    if segment_types is None:
        segment_types = ["straight", "curved"]

    scenes: List[Path] = []
    if not scenes_root.is_dir():
        return scenes
    for child in sorted(scenes_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in set(segment_types):
            for scene_dir in sorted(child.iterdir()):
                if not scene_dir.is_dir():
                    continue
                if not (scene_dir / "meta.json").is_file():
                    continue
                if not (scene_dir / "map.net.xml").is_file():
                    continue
                scenes.append(scene_dir)
            continue
        if (child / "meta.json").is_file() and (child / "map.net.xml").is_file():
            meta = json.loads((child / "meta.json").read_text(encoding="utf-8"))
            if str(meta.get("segment_type") or "") in set(segment_types) or not meta.get("segment_type"):
                scenes.append(child)
    return scenes


def load_scene_meta(scene_dir: Path) -> Dict[str, Any]:
    """Load meta.json from a scene directory."""
    meta_path = scene_dir / "meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def render_crosswalk_preview(
    net_path: Path,
    out_png: Path,
    road_id: str,
    center_xy: Tuple[float, float],
    crosswalk_xy: Optional[Tuple[float, float]] = None,
) -> None:
    """Render preview PNG with crosswalk marker."""
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
        marker_xy=crosswalk_xy or center_xy,
        compliant_edge_ids=[road_id],
        legend=True,
    )


def get_crosswalk_center_xy(
    meta: Dict,
    position_m: float,
) -> Tuple[float, float]:
    """Estimate crosswalk center XY from segment metadata."""
    start_xy = tuple(meta["start_xy"])
    end_xy = tuple(meta["end_xy"])
    length_m = meta["length_m"]

    if length_m <= 0:
        return tuple(meta["center_xy"])

    t = position_m / length_m
    t = max(0.0, min(1.0, t))

    x = start_xy[0] + t * (end_xy[0] - start_xy[0])
    y = start_xy[1] + t * (end_xy[1] - start_xy[1])
    return (x, y)


def process_segment_scene(
    scene_dir: Path,
    output_root: Path,
    positions: List[str],
    *,
    skip_existing: bool = False,
    crosswalk_width_m: float = 4.0,
    sidewalk_width_m: float = 2.0,
    flat: bool = False,
) -> List[Dict]:
    """Process a single segment scene, creating crosswalk variants.

    Returns:
        List of result dicts with status, scene_id, output_dir, error
    """
    results: List[Dict] = []

    try:
        meta = load_scene_meta(scene_dir)
    except Exception as exc:
        return [{"status": "fail", "scene_id": scene_dir.name, "error": str(exc)}]

    scene_id = meta.get("scene_name", scene_dir.name)
    segment_type = meta.get("segment_type", "straight")
    road_id = meta.get("road_id", "")
    length_m = meta.get("length_m", 0.0)

    if length_m < 80:
        return [{"status": "skip", "scene_id": scene_id, "error": "segment too short"}]

    # Identify main edges in the network
    source_net = scene_dir / meta.get("net_file", "map.net.xml")
    if not source_net.is_file():
        return [{"status": "fail", "scene_id": scene_id, "error": "net file not found"}]

    edges = identify_main_edges(source_net)
    if not edges:
        return [{"status": "fail", "scene_id": scene_id, "error": "no main edges found"}]

    # Find the edge that matches road_id from meta.json
    # This ensures we split the correct edge
    target_edge = None
    for e in edges:
        if e["edge_id"] == road_id or e["edge_id"] == f"-{road_id}":
            target_edge = e
            break
    
    if target_edge is None:
        # Fall back to finding paired edges
        pairs = find_paired_edges(edges)
        if pairs:
            edge_ids = pairs[0]
            # Use the length of the first edge in the pair
            edge_length = next(
                (e["length_m"] for e in edges if e["edge_id"] == edge_ids[0]),
                max(e["length_m"] for e in edges)
            )
        else:
            edges.sort(key=lambda e: e["length_m"], reverse=True)
            edge_ids = (edges[0]["edge_id"],)
            edge_length = edges[0]["length_m"]
    else:
        # Use the road_id edge and its reverse
        reverse_id = f"-{road_id}" if not road_id.startswith("-") else road_id[1:]
        if any(e["edge_id"] == reverse_id for e in edges):
            edge_ids = (road_id, reverse_id)
        else:
            edge_ids = (road_id,)
        edge_length = target_edge["length_m"]

    # Calculate crosswalk positions using ACTUAL edge length from network
    pos_map = calculate_crosswalk_positions(edge_length, positions)

    for pos_name, pos_m in pos_map.items():
        out_scene_id = f"{scene_id}_cw_{pos_name}"
        out_dir = output_root / out_scene_id if flat else output_root / segment_type / out_scene_id

        if skip_existing and (out_dir / "map.net.xml").is_file():
            results.append({
                "status": "skip",
                "scene_id": out_scene_id,
                "output_dir": str(out_dir),
            })
            continue

        # Prepare injection
        injection = CrosswalkInjection(
            source_net=source_net,
            crosswalk_position_m=pos_m,
            edge_ids=edge_ids,
            crosswalk_width_m=crosswalk_width_m,
            sidewalk_width_m=sidewalk_width_m,
            priority=True,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        out_net = out_dir / "map.net.xml"

        # Inject crosswalk
        inject_result = inject_crosswalk(injection, out_net)

        if not inject_result.success:
            results.append({
                "status": "fail",
                "scene_id": out_scene_id,
                "error": inject_result.error,
            })
            continue

        # Write updated meta.json
        crosswalk_xy = get_crosswalk_center_xy(meta, pos_m)
        out_meta = dict(meta)
        out_meta.update({
            "scene_name": out_scene_id,
            "scene_kind": "segment_crosswalk",
            "source_segment_scene": scene_id,
            "crosswalk_position": pos_name,
            "crosswalk_position_m": pos_m,
            "crosswalk_node_id": inject_result.crosswalk_node_id,
            "crosswalk_edge_id": inject_result.crosswalk_edge_id,
            "crossed_edge_ids": list(inject_result.crossed_edge_ids),
            "crosswalk_width_m": crosswalk_width_m,
            "sidewalk_width_m": sidewalk_width_m,
            "crosswalk_xy": list(crosswalk_xy),
            "pdd_code": "5.19",
        })

        (out_dir / "meta.json").write_text(
            json_dumps(out_meta) + "\n", encoding="utf-8"
        )

        # Write center.json
        (out_dir / "center.json").write_text(
            json_dumps({
                "lat": meta.get("latitude", 0),
                "lon": meta.get("longitude", 0),
            }) + "\n",
            encoding="utf-8",
        )

        # Render preview PNG
        try:
            render_crosswalk_preview(
                out_net,
                out_dir / "custom_cropped.png",
                road_id=road_id,
                center_xy=tuple(meta.get("center_xy", [0, 0])),
                crosswalk_xy=crosswalk_xy,
            )
        except Exception as exc:
            print(f"  [png warn] {out_scene_id}: {exc}")

        results.append({
            "status": "ok",
            "scene_id": out_scene_id,
            "output_dir": str(out_dir),
            "crosswalk_position_m": pos_m,
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_SEGMENT_SCENES,
        help=f"Input segment scenes directory (default: {DEFAULT_SEGMENT_SCENES})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for crosswalk scenes (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        help="Specific scene IDs to process (default: all)",
    )
    parser.add_argument(
        "--segment-types",
        default="straight,curved",
        help="Comma-separated segment types (default: straight,curved)",
    )
    parser.add_argument(
        "--positions",
        default="near_start,middle,near_end",
        help="Comma-separated crosswalk positions (default: near_start,middle,near_end)",
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=DEFAULT_MAX_PER_TYPE,
        help=f"Max scenes per segment type (default: {DEFAULT_MAX_PER_TYPE})",
    )
    parser.add_argument(
        "--crosswalk-width",
        type=float,
        default=4.0,
        help="Crosswalk width in meters (default: 4.0)",
    )
    parser.add_argument(
        "--sidewalk-width",
        type=float,
        default=2.0,
        help="Sidewalk width in meters (default: 2.0)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scenes that already exist in output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total number of input scenes to process",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        sys.exit(f"ERROR: Input directory not found: {args.input}")

    segment_types = [s.strip() for s in args.segment_types.split(",") if s.strip()]
    positions = [s.strip() for s in args.positions.split(",") if s.strip()]

    # Discover scenes
    all_scenes = discover_segment_scenes(args.input, segment_types)
    print(f"[prepare_crosswalk] Found {len(all_scenes)} segment scenes in {args.input}")

    # Filter to specific scenes if requested
    if args.scenes:
        scene_set = set(args.scenes)
        all_scenes = [s for s in all_scenes if s.name in scene_set]
        print(f"[prepare_crosswalk] Filtered to {len(all_scenes)} specified scenes")

    # Apply limit
    if args.limit:
        all_scenes = all_scenes[:args.limit]
        print(f"[prepare_crosswalk] Limited to {len(all_scenes)} scenes")

    # Count existing by type
    existing: Dict[str, int] = defaultdict(int)
    if args.skip_existing:
        for seg_type in segment_types:
            type_dir = args.output / seg_type
            if type_dir.is_dir():
                existing[seg_type] = sum(
                    1 for p in type_dir.iterdir()
                    if (p / "map.net.xml").is_file()
                )
        if existing:
            print(f"[prepare_crosswalk] Existing scenes: {dict(existing)}")

    # Cap by type
    filled: Dict[str, int] = dict(existing)
    jobs: List[Path] = []
    for scene_dir in all_scenes:
        meta = load_scene_meta(scene_dir)
        seg_type = meta.get("segment_type", "straight")
        # Each scene produces len(positions) outputs
        if filled.get(seg_type, 0) >= args.max_per_type:
            continue
        jobs.append(scene_dir)
        filled[seg_type] = filled.get(seg_type, 0) + len(positions)

    print(f"[prepare_crosswalk] Processing {len(jobs)} segment scenes")
    print(f"[prepare_crosswalk] Positions: {positions}")
    print(f"[prepare_crosswalk] Output: {args.output}")

    stats = {"ok": 0, "fail": 0, "skip": 0}
    t0 = time.time()

    for i, scene_dir in enumerate(jobs, 1):
        results = process_segment_scene(
            scene_dir,
            args.output,
            positions,
            skip_existing=args.skip_existing,
            crosswalk_width_m=args.crosswalk_width,
            sidewalk_width_m=args.sidewalk_width,
        )

        for r in results:
            status = r.get("status", "fail")
            if status == "ok":
                stats["ok"] += 1
            elif status == "skip":
                stats["skip"] += 1
            else:
                stats["fail"] += 1
                if r.get("error"):
                    print(f"  [fail] {r['scene_id']}: {r['error']}")

        if i % 10 == 0 or i == len(jobs):
            print(
                f"  [{i}/{len(jobs)}] ok={stats['ok']} fail={stats['fail']} skip={stats['skip']}"
            )

    elapsed = time.time() - t0
    print(
        f"[prepare_crosswalk] Done in {elapsed:.1f}s: "
        f"ok={stats['ok']} fail={stats['fail']} skip={stats['skip']}"
    )


if __name__ == "__main__":
    main()
