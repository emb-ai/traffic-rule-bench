#!/usr/bin/env python3
"""Grow a roundabout scene pool to a target size (default 100) with review cycles.

Workflow:
  1. Import core maps (roundabout filter required):
       python tools/filter_scenes/import_catalog_scenes.py --limit 40

  2. Create initial candidate pool (crop roundabouts until >= target scenes exist):
       python tools/filter_scenes/build_scene_pool.py crop --target 100

  3. Review and mark keep/reject:
       python tools/filter_scenes/review_junction_scenes.py

  4. After review, add more crops from unused core maps if kept < target:
       python tools/filter_scenes/build_scene_pool.py fill --target 100

  Repeat steps 3–4 until kept >= target or core maps are exhausted.

  Check progress anytime:
       python tools/filter_scenes/build_scene_pool.py status --target 100

Examples:
    python tools/filter_scenes/build_scene_pool.py status
    python tools/filter_scenes/build_scene_pool.py crop --target 100
    python tools/filter_scenes/build_scene_pool.py fill --target 100 --force
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

FILTER_SCENES_DIR = Path(__file__).resolve().parent
ROUNDABOUT_SIGN_DIR = FILTER_SCENES_DIR.parent.parent
DEFAULT_TARGET = 100

sys.path.insert(0, str(ROUNDABOUT_SIGN_DIR))

from lib.manifest_config import (  # noqa: E402
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_SPAWN_DISTANCE_BEFORE_END,
)
from lib.manifest_viability import check_scene_dir_viability  # noqa: E402
from lib.scene_selection import VERDICT_PENDING  # noqa: E402
from tools.filter_scenes.crop_junction_scene import (  # noqa: E402
    CORE_DIR_DEFAULT,
    SCENES_DIR_DEFAULT,
    discover_core_scene_dirs,
    process_core_scene,
    uncropped_core_dirs,
)
from tools.filter_scenes.review_junction_scenes import (  # noqa: E402
    PREVIEW_NAME_DEFAULT,
    discover_review_scenes,
    kept_scene_names,
    scene_records,
)


@dataclass
class PoolStatus:
    target: int
    candidates: int
    manifest_viable: int
    kept: int
    rejected: int
    pending: int
    cores_total: int
    cores_cropped: int
    cores_remaining: int


def collect_pool_status(
    scenes_root: Path,
    core_root: Path,
    *,
    target: int,
    preview_name: str,
    min_ego_lane_m: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
) -> PoolStatus:
    records = scene_records(scenes_root, preview_name=preview_name)
    kept = sum(1 for r in records if r["verdict"] != "reject")
    rejected = sum(1 for r in records if r["verdict"] == "reject")
    pending = sum(1 for r in records if r["verdict"] == VERDICT_PENDING)
    manifest_viable = 0
    for record in records:
        scene_dir = scenes_root / record["name"]
        if not scene_dir.is_dir():
            continue
        result = check_scene_dir_viability(
            scene_dir,
            min_ego_lane_m=min_ego_lane_m,
            aux_distance_from_intersection=aux_distance_from_intersection,
        )
        if result.viable:
            manifest_viable += 1
    cores = discover_core_scene_dirs(core_root)
    remaining = uncropped_core_dirs(core_root)
    return PoolStatus(
        target=target,
        candidates=len(records),
        manifest_viable=manifest_viable,
        kept=kept,
        rejected=rejected,
        pending=pending,
        cores_total=len(cores),
        cores_cropped=len(cores) - len(remaining),
        cores_remaining=len(remaining),
    )


def print_pool_status(status: PoolStatus) -> None:
    print(f"Target kept scenes:  {status.target}")
    print(f"Candidates (total):  {status.candidates}")
    print(f"  manifest-viable:   {status.manifest_viable}")
    print(f"  kept:              {status.kept}")
    print(f"  rejected:          {status.rejected}")
    print(f"  pending review:    {status.pending}")
    print(f"Core maps:           {status.cores_cropped}/{status.cores_total} cropped, "
          f"{status.cores_remaining} remaining")
    if status.kept >= status.target:
        print(f"\nOK: kept count reached target ({status.kept} >= {status.target}).")
    elif status.cores_remaining == 0:
        print(
            f"\nWarning: only {status.kept} kept scene(s) and no uncropped core maps left "
            f"(target {status.target})."
        )
    else:
        need = status.target - status.kept
        print(f"\nNeed {need} more kept scene(s) to reach target.")


def crop_core_batch(
    core_root: Path,
    scenes_root: Path,
    *,
    max_cores: int | None,
    spoke_length_m: float,
    min_lane_length_m: float,
    preview_name: str,
    dry_run: bool,
    overwrite: bool,
    require_manifest_viable: bool = True,
    min_ego_lane_m: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    one_per_core: bool = True,
) -> tuple[int, int]:
    """Crop uncropped core maps. Returns (roundabout_scenes_written, cores_processed)."""
    cores = uncropped_core_dirs(core_root)
    if max_cores is not None:
        cores = cores[:max_cores]

    written = 0
    processed = 0
    for core_dir in cores:
        created = process_core_scene(
            core_dir,
            scenes_root,
            spoke_extension_m=spoke_length_m,
            max_spoke_length_m=spoke_length_m,
            min_lane_length_m=min_lane_length_m,
            preview_name=preview_name,
            dry_run=dry_run,
            overwrite=overwrite,
            require_manifest_viable=require_manifest_viable,
            min_ego_lane_m=min_ego_lane_m,
            aux_distance_from_intersection=aux_distance_from_intersection,
            one_per_core=one_per_core,
        )
        processed += 1
        written += created
    return written, processed


def crop_until_candidates(
    scenes_root: Path,
    core_root: Path,
    *,
    target: int,
    spoke_length_m: float,
    min_lane_length_m: float,
    preview_name: str,
    dry_run: bool,
    overwrite: bool,
    require_manifest_viable: bool = True,
    min_ego_lane_m: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    one_per_core: bool = True,
) -> tuple[int, int]:
    """Crop uncropped cores until at least ``target`` reviewable scenes exist."""
    candidates = len(discover_review_scenes(scenes_root, preview_name=preview_name))
    if candidates >= target:
        print(f"Already have {candidates} candidate scene(s) (target {target}).")
        return 0, 0

    total_written = 0
    total_cores = 0
    while candidates < target:
        remaining = uncropped_core_dirs(core_root)
        if not remaining:
            print(
                f"\nWarning: only {candidates} candidate scene(s) available "
                f"and no uncropped core maps left (target {target})."
            )
            break

        written, cores = crop_core_batch(
            core_root,
            scenes_root,
            max_cores=1,
            spoke_length_m=spoke_length_m,
            min_lane_length_m=min_lane_length_m,
            preview_name=preview_name,
            dry_run=dry_run,
            overwrite=overwrite,
            require_manifest_viable=require_manifest_viable,
            min_ego_lane_m=min_ego_lane_m,
            aux_distance_from_intersection=aux_distance_from_intersection,
            one_per_core=one_per_core,
        )
        total_written += written
        total_cores += cores
        candidates = len(discover_review_scenes(scenes_root, preview_name=preview_name))
        if written == 0:
            print("  [skip core] no manifest-viable roundabout; trying next core map")

    return total_written, total_cores


def fill_after_review(
    scenes_root: Path,
    core_root: Path,
    *,
    target: int,
    spoke_length_m: float,
    min_lane_length_m: float,
    preview_name: str,
    dry_run: bool,
    overwrite: bool,
    force: bool,
    require_manifest_viable: bool = True,
    min_ego_lane_m: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    one_per_core: bool = True,
) -> int:
    """Add roundabout crops from new core maps when kept count is below target."""
    status = collect_pool_status(
        scenes_root,
        core_root,
        target=target,
        preview_name=preview_name,
        min_ego_lane_m=min_ego_lane_m,
        aux_distance_from_intersection=aux_distance_from_intersection,
    )
    print_pool_status(status)

    if status.kept >= status.target:
        return 0

    if status.pending > 0 and not force:
        print(
            f"\n{status.pending} scene(s) still pending review. "
            "Finish review in the UI, then run fill again.\n"
            "Use --force to add more cores anyway."
        )
        return 0

    need = status.target - status.kept
    print(f"\nAdding at least {need} new candidate scene(s) from uncropped core maps...")

    total_written = 0
    while total_written < need:
        remaining = uncropped_core_dirs(core_root)
        if not remaining:
            print(
                f"\nWarning: cannot reach target {status.target}: "
                f"kept {status.kept}, added {total_written} new scene(s) this run, "
                "no uncropped core maps left."
            )
            break

        written, _ = crop_core_batch(
            core_root,
            scenes_root,
            max_cores=1,
            spoke_length_m=spoke_length_m,
            min_lane_length_m=min_lane_length_m,
            preview_name=preview_name,
            dry_run=dry_run,
            overwrite=overwrite,
            require_manifest_viable=require_manifest_viable,
            min_ego_lane_m=min_ego_lane_m,
            aux_distance_from_intersection=aux_distance_from_intersection,
            one_per_core=one_per_core,
        )
        total_written += written
        if written == 0:
            print("  [skip core] no manifest-viable roundabout; trying next core map")

    status = collect_pool_status(
        scenes_root,
        core_root,
        target=target,
        preview_name=preview_name,
        min_ego_lane_m=min_ego_lane_m,
        aux_distance_from_intersection=aux_distance_from_intersection,
    )
    print(f"\nAdded {total_written} roundabout scene(s) this run.")
    print_pool_status(status)
    if status.kept < status.target:
        print(
            "\nNext: python tools/filter_scenes/review_junction_scenes.py\n"
            "Then: python tools/filter_scenes/build_scene_pool.py fill --target "
            f"{status.target}"
        )
    return total_written


def add_crop_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--spoke-length",
        type=float,
        default=80.0,
        help="Max upstream length on each spoke arm in meters (default: 80)",
    )
    parser.add_argument(
        "--min-lane-length",
        type=float,
        default=10.0,
        help="Minimum approach lane length when picking spawn meta (default: 10 m)",
    )
    parser.add_argument(
        "--preview-name",
        default=PREVIEW_NAME_DEFAULT,
        help=f"Preview image filename (default: {PREVIEW_NAME_DEFAULT})",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing roundabout scene folders")
    parser.add_argument(
        "--per-spoke",
        action="store_true",
        help="Crop one folder per spoke (_rb_s00, _s01, …) instead of one per core (_rb)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report picks, do not write files")
    parser.add_argument(
        "--no-require-manifest-viable",
        action="store_true",
        help="Write all cropped roundabouts even if they would be dropped by generate_manifest.py",
    )


def _crop_kwargs(args: argparse.Namespace) -> dict:
    return {
        "spoke_length_m": args.spoke_length,
        "min_lane_length_m": args.min_lane_length,
        "preview_name": args.preview_name,
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "require_manifest_viable": not args.no_require_manifest_viable,
        "min_ego_lane_m": args.min_ego_lane,
        "aux_distance_from_intersection": args.aux_distance,
        "one_per_core": not args.per_spoke,
    }


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help=f"Target number of kept scenes (default: {DEFAULT_TARGET})",
    )
    common.add_argument(
        "--core-dir",
        type=Path,
        default=CORE_DIR_DEFAULT,
        help=f"Core scenes root (default: {CORE_DIR_DEFAULT})",
    )
    common.add_argument(
        "--scenes-dir",
        type=Path,
        default=SCENES_DIR_DEFAULT,
        help=f"Roundabout scenes root (default: {SCENES_DIR_DEFAULT})",
    )
    common.add_argument(
        "--min-ego-lane",
        type=float,
        default=DEFAULT_SPAWN_DISTANCE_BEFORE_END,
        help=f"Min vehicle approach lane for manifest check (default: {DEFAULT_SPAWN_DISTANCE_BEFORE_END})",
    )
    common.add_argument(
        "--aux-distance",
        type=float,
        default=DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
        help=(
            "Aux spawn distance for manifest check "
            f"(default: {DEFAULT_AUX_DISTANCE_FROM_INTERSECTION})"
        ),
    )

    parser = argparse.ArgumentParser(
        description="Build a reviewed roundabout scene pool up to a target size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", parents=[common], help="Show pool progress toward target")

    crop_parser = sub.add_parser(
        "crop",
        parents=[common],
        help="Crop uncropped core maps until enough candidate scenes exist",
    )
    add_crop_args(crop_parser)

    fill_parser = sub.add_parser(
        "fill",
        parents=[common],
        help="After review, crop more core maps if kept count is below target",
    )
    add_crop_args(fill_parser)
    fill_parser.add_argument(
        "--force",
        action="store_true",
        help="Add cores even when some scenes are still pending review",
    )

    args = parser.parse_args()
    if args.target < 1:
        sys.exit("--target must be at least 1")

    core_root = args.core_dir.expanduser().resolve()
    scenes_root = args.scenes_dir.expanduser().resolve()
    scenes_root.mkdir(parents=True, exist_ok=True)

    if not core_root.is_dir():
        sys.exit(f"Core scenes directory not found: {core_root}\nRun import_catalog_scenes.py first.")

    if args.command == "status":
        status = collect_pool_status(
            scenes_root,
            core_root,
            target=args.target,
            preview_name=PREVIEW_NAME_DEFAULT,
            min_ego_lane_m=args.min_ego_lane,
            aux_distance_from_intersection=args.aux_distance,
        )
        print_pool_status(status)
        if status.kept < status.target and status.cores_remaining == 0:
            sys.exit(1)
        return

    if args.command == "crop":
        crop_kw = _crop_kwargs(args)
        written, cores = crop_until_candidates(
            scenes_root,
            core_root,
            target=args.target,
            **crop_kw,
        )
        print(f"\nCrop done: {written} roundabout scene(s) from {cores} core map(s).")
        status = collect_pool_status(
            scenes_root,
            core_root,
            target=args.target,
            preview_name=args.preview_name,
            min_ego_lane_m=args.min_ego_lane,
            aux_distance_from_intersection=args.aux_distance,
        )
        print_pool_status(status)
        print(
            "\nNext: python tools/filter_scenes/review_junction_scenes.py\n"
            f"Then: python tools/filter_scenes/build_scene_pool.py fill --target {args.target}"
        )
        return

    if args.command == "fill":
        fill_after_review(
            scenes_root,
            core_root,
            target=args.target,
            force=args.force,
            **_crop_kwargs(args),
        )


if __name__ == "__main__":
    main()
