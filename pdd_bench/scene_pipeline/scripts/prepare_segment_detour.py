#!/usr/bin/env python3
"""Prepare detour scenes from segment scenes for signs 4.2.1/4.2.2/4.2.3.

This script transforms segment scenes (multi-lane roads) into detour scenes by:
1. Filtering segments with lane_count >= 2
2. Determining valid obstacle lane index based on sign code
3. Calculating sign position (sign_s)
4. Copying the network and writing updated meta.json

Input:  scenes/segment/{straight,curved}/<scene_id>/
Output: scenes/segment_detour/{straight,curved}/<scene_id>_detour_<code>/

SUMO convention: lane 0 is the rightmost; physically further right = LOWER index.
  4.2.1 (pass on the right) -> the obstacle lane must have a lower-index neighbor;
  4.2.2 (pass on the left)  -> a higher-index neighbor;
  4.2.3 (either side)       -> any neighbor.

Examples:
    # Process all segment scenes for all detour codes
    python scripts/prepare_segment_detour.py

    # Specific code only
    python scripts/prepare_segment_detour.py --codes 4.2.1

    # Limit per code
    python scripts/prepare_segment_detour.py --max-per-code 100

    # Skip existing scenes
    python scripts/prepare_segment_detour.py --skip-existing
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sumolib

ROOT = Path(__file__).resolve().parents[1]

DETOUR_CODES = ("4.2.1", "4.2.2", "4.2.3")
MIN_EDGE_LENGTH_M = 45.0
TARGET_SIGN_S = 60.0
EDGE_TAIL_MARGIN = 12.0
SHORT_RUNWAY_S = 40.0

DEFAULT_SEGMENT_SCENES = ROOT / "scenes" / "segment"
DEFAULT_OUTPUT = ROOT / "scenes" / "segment_detour"
DEFAULT_MAX_PER_CODE = 500


def json_dumps(obj) -> str:
    """Compact JSON dump."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def discover_segment_scenes(
    scenes_root: Path,
    segment_types: Optional[List[str]] = None,
) -> List[Path]:
    """Find all segment scene directories."""
    if segment_types is None:
        segment_types = ["straight", "curved"]

    scenes: List[Path] = []
    for seg_type in segment_types:
        type_dir = scenes_root / seg_type
        if not type_dir.is_dir():
            continue
        for scene_dir in sorted(type_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if not (scene_dir / "meta.json").is_file():
                continue
            if not (scene_dir / "map.net.xml").is_file():
                continue
            scenes.append(scene_dir)

    return scenes


def load_scene_meta(scene_dir: Path) -> Dict[str, Any]:
    """Load meta.json from a scene directory."""
    meta_path = scene_dir / "meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def drivable_lane_indices(edge) -> set:
    """Return lane indices that allow passenger vehicles."""
    return {l.getIndex() for l in edge.getLanes() if l.allows("passenger")}


def valid_obstacle_lane_indices(edge, code: str) -> List[int]:
    """Lane indices where an obstacle may be placed for the given code.
    
    Among passenger lanes there must be a neighbor on the prescribed side:
    - 4.2.1 (pass right): obstacle lane must have lower-index neighbor
    - 4.2.2 (pass left): obstacle lane must have higher-index neighbor
    - 4.2.3 (either): any neighbor
    """
    drivable = drivable_lane_indices(edge)
    n = len(edge.getLanes())
    out = []
    for i in sorted(drivable):
        has_right = any(j in drivable for j in range(0, i))
        has_left = any(j in drivable for j in range(i + 1, n))
        if code == "4.2.1" and has_right:
            out.append(i)
        elif code == "4.2.2" and has_left:
            out.append(i)
        elif code == "4.2.3" and (has_right or has_left):
            out.append(i)
    return out


def preferred_obstacle_lane(edge, code: str) -> Optional[int]:
    """Pick the best obstacle lane for the given code."""
    valid = valid_obstacle_lane_indices(edge, code)
    if not valid:
        return None
    if code in ("4.2.1", "4.2.2"):
        return valid[0]
    n = len(edge.getLanes())
    if n >= 3 and (n // 2) in valid:
        return n // 2
    ge1 = [i for i in valid if i >= 1]
    return ge1[0] if ge1 else valid[0]


def clamp_sign_s(edge_len: float, target_sign_s: float = TARGET_SIGN_S) -> float:
    """Keep the sign at least EDGE_TAIL_MARGIN from the edge end."""
    upper = max(0.5, edge_len - EDGE_TAIL_MARGIN)
    lower = min(float(target_sign_s), upper)
    return max(lower, min(target_sign_s, upper))


def render_detour_preview(
    net_path: Path,
    out_png: Path,
    road_id: str,
    center_xy: Tuple[float, float],
    sign_xy: Optional[Tuple[float, float]] = None,
) -> None:
    """Render preview PNG with detour marker."""
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
        marker_xy=sign_xy or center_xy,
        compliant_edge_ids=[road_id],
        legend=True,
    )


def get_sign_xy(
    meta: Dict,
    sign_s: float,
) -> Tuple[float, float]:
    """Estimate sign position XY from segment metadata."""
    start_xy = tuple(meta["start_xy"])
    end_xy = tuple(meta["end_xy"])
    length_m = meta["length_m"]

    if length_m <= 0:
        return tuple(meta["center_xy"])

    t = sign_s / length_m
    t = max(0.0, min(1.0, t))

    x = start_xy[0] + t * (end_xy[0] - start_xy[0])
    y = start_xy[1] + t * (end_xy[1] - start_xy[1])
    return (x, y)


def process_segment_scene(
    scene_dir: Path,
    output_root: Path,
    codes: List[str],
    *,
    skip_existing: bool = False,
) -> List[Dict]:
    """Process a single segment scene, creating detour variants.

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
    lane_count = meta.get("lane_count", 1)

    if length_m < MIN_EDGE_LENGTH_M:
        return [{"status": "skip", "scene_id": scene_id, "error": "segment too short"}]

    if lane_count < 2:
        return [{"status": "skip", "scene_id": scene_id, "error": "single lane"}]

    source_net_path = scene_dir / meta.get("net_file", "map.net.xml")
    if not source_net_path.is_file():
        return [{"status": "fail", "scene_id": scene_id, "error": "net file not found"}]

    try:
        net = sumolib.net.readNet(str(source_net_path), withInternal=False)
    except Exception as exc:
        return [{"status": "fail", "scene_id": scene_id, "error": f"net parse: {exc}"}]

    edge = net.getEdge(road_id) if net.hasEdge(road_id) else None
    if edge is None:
        return [{"status": "fail", "scene_id": scene_id, "error": f"edge not found: {road_id}"}]

    for code in codes:
        out_scene_id = f"{scene_id}_detour_{code.replace('.', '_')}"
        out_dir = output_root / segment_type / out_scene_id

        if skip_existing and (out_dir / "map.net.xml").is_file():
            results.append({
                "status": "skip",
                "scene_id": out_scene_id,
                "output_dir": str(out_dir),
                "code": code,
            })
            continue

        valid_lanes = valid_obstacle_lane_indices(edge, code)
        if not valid_lanes:
            results.append({
                "status": "skip",
                "scene_id": out_scene_id,
                "error": f"no valid obstacle lane for {code}",
                "code": code,
            })
            continue

        obstacle_lane = preferred_obstacle_lane(edge, code)
        if obstacle_lane is None:
            results.append({
                "status": "skip",
                "scene_id": out_scene_id,
                "error": f"could not pick obstacle lane for {code}",
                "code": code,
            })
            continue

        sign_s = clamp_sign_s(edge.getLength())
        sign_xy = get_sign_xy(meta, sign_s)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_net = out_dir / "map.net.xml"

        try:
            shutil.copy2(source_net_path, out_net)
        except Exception as exc:
            results.append({
                "status": "fail",
                "scene_id": out_scene_id,
                "error": f"copy net: {exc}",
                "code": code,
            })
            continue

        out_meta = dict(meta)
        out_meta.update({
            "scene_name": out_scene_id,
            "scene_kind": "segment_detour",
            "source_segment_scene": scene_id,
            "detour_code": code,
            "pdd_code": code,
            "sign_lane_index": obstacle_lane,
            "sign_s": round(sign_s, 2),
            "sign_xy": list(sign_xy),
            "valid_obstacle_lanes": valid_lanes,
            "short_runway": sign_s < SHORT_RUNWAY_S,
        })

        (out_dir / "meta.json").write_text(
            json_dumps(out_meta) + "\n", encoding="utf-8"
        )

        (out_dir / "center.json").write_text(
            json_dumps({
                "lat": meta.get("latitude", 0),
                "lon": meta.get("longitude", 0),
            }) + "\n",
            encoding="utf-8",
        )

        try:
            render_detour_preview(
                out_net,
                out_dir / "custom_cropped.png",
                road_id=road_id,
                center_xy=tuple(meta.get("center_xy", [0, 0])),
                sign_xy=sign_xy,
            )
        except Exception as exc:
            print(f"  [png warn] {out_scene_id}: {exc}")

        results.append({
            "status": "ok",
            "scene_id": out_scene_id,
            "output_dir": str(out_dir),
            "code": code,
            "sign_lane_index": obstacle_lane,
            "sign_s": sign_s,
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
        help=f"Output directory for detour scenes (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--codes",
        default="4.2.1,4.2.2,4.2.3",
        help="Comma-separated detour codes (default: 4.2.1,4.2.2,4.2.3)",
    )
    parser.add_argument(
        "--segment-types",
        default="straight,curved",
        help="Comma-separated segment types (default: straight,curved)",
    )
    parser.add_argument(
        "--max-per-code",
        type=int,
        default=DEFAULT_MAX_PER_CODE,
        help=f"Max scenes per detour code (default: {DEFAULT_MAX_PER_CODE})",
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

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    for c in codes:
        if c not in DETOUR_CODES:
            sys.exit(f"ERROR: Invalid detour code: {c}. Valid: {DETOUR_CODES}")

    segment_types = [s.strip() for s in args.segment_types.split(",") if s.strip()]

    all_scenes = discover_segment_scenes(args.input, segment_types)
    print(f"[prepare_detour] Found {len(all_scenes)} segment scenes in {args.input}")

    multilane = []
    for scene_dir in all_scenes:
        meta = load_scene_meta(scene_dir)
        if meta.get("lane_count", 1) >= 2:
            multilane.append(scene_dir)

    print(f"[prepare_detour] Filtered to {len(multilane)} multi-lane segments")

    if args.limit:
        multilane = multilane[:args.limit]
        print(f"[prepare_detour] Limited to {len(multilane)} scenes")

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
            print(f"[prepare_detour] Existing scenes: {dict(existing)}")

    filled_by_code: Dict[str, int] = defaultdict(int)
    jobs: List[Tuple[Path, List[str]]] = []

    for scene_dir in multilane:
        meta = load_scene_meta(scene_dir)
        seg_type = meta.get("segment_type", "straight")
        lane_count = meta.get("lane_count", 1)
        road_id = meta.get("road_id", "")

        source_net_path = scene_dir / meta.get("net_file", "map.net.xml")
        if not source_net_path.is_file():
            continue

        try:
            net = sumolib.net.readNet(str(source_net_path), withInternal=False)
            edge = net.getEdge(road_id) if net.hasEdge(road_id) else None
        except Exception:
            continue

        if edge is None:
            continue

        codes_for_scene = []
        for code in codes:
            if filled_by_code[code] >= args.max_per_code:
                continue
            if valid_obstacle_lane_indices(edge, code):
                codes_for_scene.append(code)
                filled_by_code[code] += 1

        if codes_for_scene:
            jobs.append((scene_dir, codes_for_scene))

    print(f"[prepare_detour] Processing {len(jobs)} segment scenes")
    print(f"[prepare_detour] Codes: {codes}")
    print(f"[prepare_detour] Target per code: {dict(filled_by_code)}")
    print(f"[prepare_detour] Output: {args.output}")

    stats = {"ok": 0, "fail": 0, "skip": 0}
    stats_by_code: Dict[str, int] = defaultdict(int)
    t0 = time.time()

    for i, (scene_dir, scene_codes) in enumerate(jobs, 1):
        results = process_segment_scene(
            scene_dir,
            args.output,
            scene_codes,
            skip_existing=args.skip_existing,
        )

        for r in results:
            status = r.get("status", "fail")
            code = r.get("code", "")
            if status == "ok":
                stats["ok"] += 1
                stats_by_code[code] += 1
            elif status == "skip":
                stats["skip"] += 1
            else:
                stats["fail"] += 1
                if r.get("error"):
                    print(f"  [fail] {r['scene_id']}: {r['error']}")

        if i % 20 == 0 or i == len(jobs):
            print(
                f"  [{i}/{len(jobs)}] ok={stats['ok']} fail={stats['fail']} skip={stats['skip']}"
            )

    elapsed = time.time() - t0
    print(
        f"[prepare_detour] Done in {elapsed:.1f}s: "
        f"ok={stats['ok']} fail={stats['fail']} skip={stats['skip']}"
    )
    print(f"[prepare_detour] By code: {dict(stats_by_code)}")


if __name__ == "__main__":
    main()
