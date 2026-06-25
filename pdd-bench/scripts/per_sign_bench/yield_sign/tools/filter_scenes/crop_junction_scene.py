#!/usr/bin/env python3
"""Crop core catalog scenes into separate junction scene folders under scenes/.

Reads full maps from scenes/core/ (import_catalog_scenes.py output) and writes
one folder per picked junction directly under scenes/, e.g.:
  scenes/core/sign_72424/           # untouched core map
  scenes/sign_72424_j0/             # best 4-arm junction crop
  scenes/sign_72424_j1/             # next-ranked junction crop

Selection rules:
  - Consider all junctions with exactly 3 or 4 incoming arms.
  - Each arm must have at least one lane longer than --min-lane-length (default 10 m).
  - Sort: 4-arm junctions first, then 3-arm; within each group by total lane count (desc).
  - Keep at most --max-junctions picks per core scene (default 5).

Examples:
    python tools/filter_scenes/crop_junction_scene.py
    python tools/filter_scenes/crop_junction_scene.py sign_72424 sign_73117 --radius 100
    python tools/filter_scenes/crop_junction_scene.py --limit 3 --max-junctions 5
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
YIELD_SIGN_DIR = TOOLS_DIR.parent.parent
SCENES_DIR_DEFAULT = YIELD_SIGN_DIR / "scenes"
CORE_DIR_DEFAULT = SCENES_DIR_DEFAULT / "core"

sys.path.insert(0, str(YIELD_SIGN_DIR))

from lib.junction_crop import (  # noqa: E402
    JunctionLayoutError,
    JunctionPick,
    crop_scene_to_junction_pick,
    find_ranked_intersection_junctions,
    resolve_full_source_net,
)
from lib.sumo_utils import (  # noqa: E402
    is_core_scene_name,
    junction_scene_name,
    load_scene_meta,
    resolve_net_file,
    resolve_scene_dir,
)
from tools.render_map import parse_sumo_net, render_network  # noqa: E402


def discover_core_scene_dirs(core_root: Path) -> list[Path]:
    out: list[Path] = []
    if not core_root.is_dir():
        return out
    for entry in sorted(core_root.iterdir()):
        if not entry.is_dir():
            continue
        if not is_core_scene_name(entry.name):
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


def write_junctions_index(
    core_scene_dir: Path,
    core_scene_name: str,
    picks: list[JunctionPick],
    *,
    scenes_root: Path,
    preview_name: str,
) -> None:
    entries = []
    for rank, pick in enumerate(picks):
        scene_name = junction_scene_name(core_scene_name, rank)
        entries.append(
            {
                "rank": rank,
                "scene_name": scene_name,
                "junction_id": pick.junction_id,
                "arm_count": pick.arm_count,
                "total_lanes": pick.total_lanes,
                "incoming_edge_ids": list(pick.incoming_edge_ids),
                "center_xy": [pick.center_xy[0], pick.center_xy[1]],
                "output_dir": scene_name,
                "preview": f"{scene_name}/{preview_name}",
            }
        )
    index_path = core_scene_dir / "junctions.json"
    index_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def process_core_scene(
    core_scene_dir: Path,
    scenes_root: Path,
    *,
    radius_m: float,
    min_lane_length_m: float,
    max_junctions: int,
    preview_name: str,
    dry_run: bool,
    overwrite: bool,
) -> int:
    """Crop junction variants for one core scene. Returns number of scenes written."""
    core_scene_name = core_scene_dir.name
    print(f"\n=== {core_scene_name} (core) ===")
    try:
        meta = load_scene_meta(core_scene_dir)
        source_net = resolve_full_source_net(core_scene_dir, meta)
        picks = find_ranked_intersection_junctions(
            source_net,
            min_lane_length_m=min_lane_length_m,
            max_junctions=max_junctions,
        )
    except (FileNotFoundError, JunctionLayoutError) as exc:
        print(f"  [skip] {exc}")
        return 0

    print(f"  source net: {source_net.name}")
    print(f"  picked {len(picks)} junction(s) (max {max_junctions}):")
    for rank, pick in enumerate(picks):
        scene_name = junction_scene_name(core_scene_name, rank)
        print(
            f"    [{rank}] {pick.junction_id} ({pick.arm_count}-arm, "
            f"{pick.total_lanes} lane(s)) -> scenes/{scene_name}"
        )

    if dry_run:
        return len(picks)

    base_meta = meta
    created = 0
    for rank, pick in enumerate(picks):
        scene_name = junction_scene_name(core_scene_name, rank)
        out_dir = scenes_root / scene_name
        if out_dir.exists():
            if not overwrite:
                print(f"  [skip existing] {scene_name}")
                continue
            shutil.rmtree(out_dir)

        crop_scene_to_junction_pick(
            core_scene_dir,
            pick,
            source_net=source_net,
            radius_m=radius_m,
            min_lane_length_m=min_lane_length_m,
            output_dir=out_dir,
            output_scene_name=scene_name,
            base_meta=base_meta,
            backup_original=False,
            junction_rank=rank,
            core_scene_name=core_scene_name,
        )
        preview_path = out_dir / preview_name
        render_preview(out_dir, pick.center_xy, preview_path)
        print(f"  wrote scenes/{scene_name}/ ({preview_name})")
        created += 1

    write_junctions_index(
        core_scene_dir,
        core_scene_name,
        picks,
        scenes_root=scenes_root,
        preview_name=preview_name,
    )
    print(f"  wrote core/{core_scene_name}/junctions.json")
    return created


def uncropped_core_dirs(core_root: Path) -> list[Path]:
    """Core scenes that have not been cropped yet (no junctions.json)."""
    return [
        core_dir
        for core_dir in discover_core_scene_dirs(core_root)
        if not (core_dir / "junctions.json").is_file()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop core scenes into separate junction folders under scenes/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "scenes",
        nargs="*",
        help="Core scene folder name(s) under --core-dir (e.g. sign_72424)",
    )
    parser.add_argument(
        "--core-dir",
        type=Path,
        default=CORE_DIR_DEFAULT,
        help=f"Core scenes root (default: {CORE_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=SCENES_DIR_DEFAULT,
        help=f"Output scenes root for junction crops (default: {SCENES_DIR_DEFAULT})",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process first N core scenes when none named")
    parser.add_argument("--radius", type=float, default=80.0, help="Max arm length in meters (default: 80)")
    parser.add_argument(
        "--min-lane-length",
        type=float,
        default=10.0,
        help="Each arm must have a lane longer than this (default: 10 m)",
    )
    parser.add_argument(
        "--max-junctions",
        type=int,
        default=5,
        help="Maximum junctions to crop per core scene (default: 5)",
    )
    parser.add_argument(
        "--preview-name",
        default="custom_cropped.png",
        help="Cropped-map preview filename inside each junction scene (default: custom_cropped.png)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing junction scene folders",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report junction picks, do not write files")
    args = parser.parse_args()

    if args.max_junctions < 1:
        sys.exit("--max-junctions must be at least 1")

    core_root = args.core_dir.expanduser().resolve()
    scenes_root = args.scenes_dir.expanduser().resolve()
    scenes_root.mkdir(parents=True, exist_ok=True)

    if not core_root.is_dir():
        sys.exit(f"Core scenes directory not found: {core_root}\nRun import_catalog_scenes.py first.")

    if args.scenes:
        core_scene_dirs = [resolve_scene_dir(core_root, name) for name in args.scenes]
    else:
        core_scene_dirs = discover_core_scene_dirs(core_root)
        if args.limit is not None:
            core_scene_dirs = core_scene_dirs[: args.limit]

    if not core_scene_dirs:
        sys.exit(f"No core scenes found under {core_root}")

    ok = 0
    created_total = 0
    for core_scene_dir in core_scene_dirs:
        created = process_core_scene(
            core_scene_dir,
            scenes_root,
            radius_m=args.radius,
            min_lane_length_m=args.min_lane_length,
            max_junctions=args.max_junctions,
            preview_name=args.preview_name,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        if created > 0:
            ok += 1
        created_total += created

    print(f"\nDone: {ok}/{len(core_scene_dirs)} core scene(s) processed, {created_total} junction scene(s) written.")
    if ok < len(core_scene_dirs):
        sys.exit(1)


if __name__ == "__main__":
    main()
