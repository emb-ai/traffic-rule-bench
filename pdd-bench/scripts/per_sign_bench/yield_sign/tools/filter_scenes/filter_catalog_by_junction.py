#!/usr/bin/env python3
"""Analyze catalog scenes and copy only those with valid 3- and/or 4-arm junctions.

Scans every scene under a catalog folder (default: pdd-bench/scenes/2.4),
checks for an intersection with the requested arm count(s) where each arm has
a lane longer than --min-lane-length, and copies qualifying scenes into a
separate folder.

When both 3 and 4 are requested, a scene matches if it has a qualifying
4-arm junction; otherwise a qualifying 3-arm junction is accepted.

Examples:
    # Copy scenes with a valid 4-arm junction (default)
    python tools/filter_scenes/filter_catalog_by_junction.py

    # Copy scenes with a valid 3-arm (T) junction
    python tools/filter_scenes/filter_catalog_by_junction.py --arms 3

    # Copy scenes with either 4-arm or 3-arm junction
    python tools/filter_scenes/filter_catalog_by_junction.py --arms 4 3

    # Dry-run on 2.5 catalog
    python tools/filter_scenes/filter_catalog_by_junction.py \\
        --source ../../../../scenes/2.5 --arms 4 --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

FILTER_SCENES_DIR = Path(__file__).resolve().parent
TOOLS_DIR = FILTER_SCENES_DIR.parent
YIELD_SIGN_DIR = TOOLS_DIR.parent
PDD_BENCH_DIR = YIELD_SIGN_DIR.parent.parent.parent
DEFAULT_SOURCE = PDD_BENCH_DIR / "scenes" / "2.4"

sys.path.insert(0, str(YIELD_SIGN_DIR))

from lib.junction_crop import try_find_junction_for_arm_counts  # noqa: E402
from lib.sumo_utils import load_scene_meta, resolve_net_file  # noqa: E402


@dataclass
class SceneAnalysis:
    scene_name: str
    matched: bool
    arm_count: Optional[int] = None
    junction_id: Optional[str] = None
    total_lanes: Optional[int] = None
    incoming_edge_ids: Optional[list[str]] = None
    reason: Optional[str] = None


def default_dest_for_source(source_dir: Path, arm_counts: list[int]) -> Path:
    catalog_name = source_dir.name
    unique = sorted(set(arm_counts))
    if unique == [4]:
        suffix = "four_arm"
    elif unique == [3]:
        suffix = "three_arm"
    else:
        suffix = f"junction_{'_'.join(str(n) for n in unique)}arm"
    return source_dir.parent / f"{catalog_name}_{suffix}"


def arm_counts_label(arm_counts: list[int]) -> str:
    unique = sorted(set(arm_counts))
    if len(unique) == 1:
        return f"{unique[0]}-arm"
    return "/".join(f"{n}-arm" for n in unique)


def discover_catalog_scenes(source_dir: Path) -> list[Path]:
    scenes: list[Path] = []
    if not source_dir.is_dir():
        return scenes
    for entry in sorted(source_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "meta.json").is_file():
            continue
        if not any(entry.glob("*.net.xml")):
            continue
        scenes.append(entry)
    return scenes


def analyze_scene(
    scene_dir: Path,
    *,
    arm_counts: tuple[int, ...],
    min_lane_length_m: float,
) -> SceneAnalysis:
    scene_name = scene_dir.name
    try:
        meta = load_scene_meta(scene_dir)
        net_path = scene_dir / resolve_net_file(scene_dir, meta)
    except (FileNotFoundError, ValueError) as exc:
        return SceneAnalysis(scene_name=scene_name, matched=False, reason=str(exc))

    if not net_path.is_file():
        return SceneAnalysis(
            scene_name=scene_name,
            matched=False,
            reason=f"net file not found: {net_path.name}",
        )

    pick = try_find_junction_for_arm_counts(
        net_path,
        arm_counts=arm_counts,
        min_lane_length_m=min_lane_length_m,
    )
    if pick is None:
        label = arm_counts_label(list(arm_counts))
        return SceneAnalysis(
            scene_name=scene_name,
            matched=False,
            reason=f"no {label} junction with all arms longer than min lane length",
        )

    return SceneAnalysis(
        scene_name=scene_name,
        matched=True,
        arm_count=pick.arm_count,
        junction_id=pick.junction_id,
        total_lanes=pick.total_lanes,
        incoming_edge_ids=list(pick.incoming_edge_ids),
    )


def copy_scene(source_dir: Path, dest_root: Path, *, overwrite: bool) -> Path:
    dest = dest_root / source_dir.name
    if dest.exists():
        if not overwrite:
            print(f"  [skip copy] already exists: {dest}")
            return dest
        shutil.rmtree(dest)
    shutil.copytree(source_dir, dest)
    return dest


def write_report(report_path: Path, analyses: list[SceneAnalysis]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        for row in analyses:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def parse_arm_counts(raw: list[int]) -> tuple[int, ...]:
    allowed = {3, 4}
    counts = sorted({int(n) for n in raw})
    invalid = [n for n in counts if n not in allowed]
    if invalid:
        raise SystemExit(f"Unsupported --arms value(s): {invalid}; allowed: 3, 4")
    if not counts:
        raise SystemExit("At least one --arms value is required (3 and/or 4)")
    return tuple(counts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy catalog scenes that contain a valid 3- and/or 4-arm junction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Catalog scenes root (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Output folder for matched scenes (default: <source>_<four_arm|three_arm|...>)",
    )
    parser.add_argument(
        "--arms",
        type=int,
        nargs="+",
        default=[4],
        metavar="N",
        help="Junction arm count(s) to accept: 3, 4, or both (default: 4)",
    )
    parser.add_argument(
        "--min-lane-length",
        type=float,
        default=10.0,
        help="Each arm must have a lane longer than this (default: 10 m)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process first N scenes only")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing dest folders")
    parser.add_argument("--dry-run", action="store_true", help="Analyze and report only, do not copy")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write JSONL analysis report (default: <dest>/analysis.jsonl)",
    )
    args = parser.parse_args()

    arm_counts = parse_arm_counts(args.arms)
    source_dir = args.source.expanduser().resolve()
    dest_root = (
        args.dest.expanduser().resolve()
        if args.dest is not None
        else default_dest_for_source(source_dir, list(arm_counts))
    )
    report_path = args.report or (dest_root / "analysis.jsonl")

    if not source_dir.is_dir():
        sys.exit(f"Source catalog not found: {source_dir}")

    scene_dirs = discover_catalog_scenes(source_dir)
    if args.limit is not None:
        scene_dirs = scene_dirs[: args.limit]

    if not scene_dirs:
        sys.exit(f"No scenes found under {source_dir}")

    arms_label = arm_counts_label(list(arm_counts))
    print(f"Source: {source_dir}")
    print(f"Dest:   {dest_root}")
    print(f"Arms:   {arms_label}")
    print(f"Scenes: {len(scene_dirs)}")
    print(f"Dry run: {args.dry_run}")

    analyses: list[SceneAnalysis] = []
    copied = 0
    for scene_dir in scene_dirs:
        analysis = analyze_scene(
            scene_dir,
            arm_counts=arm_counts,
            min_lane_length_m=args.min_lane_length,
        )
        analyses.append(analysis)

        if analysis.matched:
            print(
                f"  [{analysis.arm_count}-arm] {analysis.scene_name}: "
                f"junction {analysis.junction_id}, {analysis.total_lanes} lane(s)"
            )
            if not args.dry_run:
                dest_root.mkdir(parents=True, exist_ok=True)
                copy_scene(scene_dir, dest_root, overwrite=args.overwrite)
                copied += 1
        else:
            print(f"  [skip]  {analysis.scene_name}: {analysis.reason}")

    if not args.dry_run:
        write_report(report_path, analyses)
        print(f"\nReport: {report_path}")

    matched = sum(1 for row in analyses if row.matched)
    print(
        f"\nDone: {matched}/{len(analyses)} scene(s) with {arms_label} junction"
        + (f", {copied} copied" if not args.dry_run else "")
    )


if __name__ == "__main__":
    main()
