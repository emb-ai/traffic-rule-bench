#!/usr/bin/env python3
"""Crop a scene net around the best 4- or 3-arm junction and write center.json + preview PNG.

Selection rules:
  - Prefer a junction with exactly 4 incoming arms; fall back to 3 arms (T junction).
  - Each arm must have at least one lane longer than --min-lane-length (default 10 m).
  - Among valid junctions of the chosen arm count, pick the one with the most lanes.

Outputs per scene:
  - map.net.xml (only the picked junction + its incoming/outgoing arms)
  - center.json  {"lat": ..., "lon": ...} at junction center
  - custom_cropped.png   cropped-map preview with red dot at junction center
  - meta.json    updated with crop metadata

Examples:
    python tools/crop_junction_scene.py sign_72424
    python tools/crop_junction_scene.py sign_72424 sign_73117 --radius 100
    python tools/crop_junction_scene.py --limit 3 --source ../../../../../scenes/2.4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
YIELD_SIGN_DIR = TOOLS_DIR.parent.parent
SCENES_DIR_DEFAULT = YIELD_SIGN_DIR / "scenes"

sys.path.insert(0, str(YIELD_SIGN_DIR))

from lib.junction_crop import (  # noqa: E402
    JunctionLayoutError,
    crop_scene_to_junction,
    find_best_intersection_junction,
)
from lib.sumo_utils import load_scene_meta, resolve_net_file, resolve_scene_dir  # noqa: E402
from tools.render_map import parse_sumo_net, render_network  # noqa: E402


def discover_scene_dirs(scenes_root: Path) -> list[Path]:
    out: list[Path] = []
    for entry in sorted(scenes_root.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "meta.json").is_file():
            continue
        if not any(entry.glob("*.net.xml")):
            continue
        out.append(entry)
    return out


def render_preview(scene_dir: Path, marker_xy: tuple[float, float], out_path: Path) -> None:
    meta = load_scene_meta(scene_dir)
    net_file = resolve_net_file(scene_dir, meta)
    edges, junctions = parse_sumo_net(scene_dir / net_file)
    render_network(
        edges,
        junctions,
        out_path,
        marker_xy=marker_xy,
    )


def process_scene(
    scene_dir: Path,
    *,
    radius_m: float,
    min_lane_length_m: float,
    preview_path: Path,
    dry_run: bool,
) -> bool:
    scene_name = scene_dir.name
    print(f"\n=== {scene_name} ===")
    try:
        meta = load_scene_meta(scene_dir)
        source_net = scene_dir / resolve_net_file(scene_dir, meta)
        pick = find_best_intersection_junction(source_net, min_lane_length_m=min_lane_length_m)
    except (FileNotFoundError, JunctionLayoutError) as exc:
        print(f"  [skip] {exc}")
        return False

    print(
        f"  junction {pick.junction_id} ({pick.arm_count}-arm): {pick.total_lanes} lane(s) on "
        f"{len(pick.incoming_edge_ids)} arms"
    )
    print(f"  center xy=({pick.center_xy[0]:.2f}, {pick.center_xy[1]:.2f})")

    if dry_run:
        return True

    pick = crop_scene_to_junction(
        scene_dir,
        radius_m=radius_m,
        min_lane_length_m=min_lane_length_m,
    )
    render_preview(scene_dir, pick.center_xy, preview_path)
    print(f"  wrote center.json, map.net.xml, {preview_path.name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop scenes to the best 4- or 3-arm junction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "scenes",
        nargs="*",
        help="Scene folder name(s) under --scenes-dir",
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=SCENES_DIR_DEFAULT,
        help=f"Scenes root (default: {SCENES_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional alternate source root (process dirs in-place there)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process first N scenes when none named")
    parser.add_argument("--radius", type=float, default=80.0, help="Max arm length in meters (default: 80)")
    parser.add_argument(
        "--min-lane-length",
        type=float,
        default=10.0,
        help="Each arm must have a lane longer than this (default: 10 m)",
    )
    parser.add_argument(
        "--preview-name",
        default="custom_cropped.png",
        help="Cropped-map preview filename inside scene dir (default: custom_cropped.png)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report junction picks, do not write files")
    args = parser.parse_args()

    scenes_root = (args.source or args.scenes_dir).expanduser().resolve()

    if args.scenes:
        scene_dirs = [resolve_scene_dir(scenes_root, name) for name in args.scenes]
    else:
        scene_dirs = discover_scene_dirs(scenes_root)
        if args.limit is not None:
            scene_dirs = scene_dirs[: args.limit]

    if not scene_dirs:
        sys.exit(f"No scenes found under {scenes_root}")

    ok = 0
    for scene_dir in scene_dirs:
        preview = scene_dir / args.preview_name
        if process_scene(
            scene_dir,
            radius_m=args.radius,
            min_lane_length_m=args.min_lane_length,
            preview_path=preview,
            dry_run=args.dry_run,
        ):
            ok += 1

    print(f"\nDone: {ok}/{len(scene_dirs)} scene(s) processed.")
    if ok < len(scene_dirs):
        sys.exit(1)


if __name__ == "__main__":
    main()
