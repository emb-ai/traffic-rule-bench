#!/usr/bin/env python3
"""Crop core catalog scenes into separate crosswalk scene folders under scenes/.

Reads full maps from scenes/core/ (import_catalog_scenes.py output) and writes
one folder per picked pedestrian crossing directly under scenes/, e.g.:
  scenes/core/sign_71853/           # untouched core map
  scenes/sign_71853_cw0/            # first crossing crop
  scenes/sign_71853_cw1/            # second crossing crop

Selection rules:
  - Consider all SUMO pedestrian crossings with at least one vehicle approach lane.
  - Sort by longest approach lane, then number of approach arms.
  - Keep at most --max-crosswalks picks per core scene (default 8).
  - Each crop keeps only the picked crossing; other crossings are pruned from the net.
  - By default, skip crops that would fail generate_manifest.py.

Examples:
    python tools/filter_scenes/crop_crosswalk_scene.py
    python tools/filter_scenes/crop_crosswalk_scene.py sign_71853 --radius 80
    python tools/filter_scenes/crop_crosswalk_scene.py --limit 3 --max-crosswalks 5
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CROSSWALK_SIGN_DIR = TOOLS_DIR.parent.parent
SCENES_DIR_DEFAULT = CROSSWALK_SIGN_DIR / "scenes"
CORE_DIR_DEFAULT = SCENES_DIR_DEFAULT / "core"

sys.path.insert(0, str(CROSSWALK_SIGN_DIR))

from lib.crosswalk_crop import (  # noqa: E402
    CrosswalkPick,
    crop_scene_to_crosswalk_pick,
    crosswalk_scene_name,
    find_ranked_crosswalks,
)
from lib.junction_crop import JunctionLayoutError, resolve_full_source_net  # noqa: E402
from lib.manifest_config import DEFAULT_CROSSWALK_CROP_RADIUS_M, DEFAULT_SPAWN_DISTANCE_BEFORE_END
from lib.manifest_viability import ManifestViabilityResult, check_manifest_viability  # noqa: E402
from lib.sumo_utils import (  # noqa: E402
    is_core_scene_name,
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
    render_network(edges, junctions, out_path, marker_xy=marker_xy)


def write_crosswalks_index(
    core_scene_dir: Path,
    core_scene_name: str,
    pick_records: list[dict],
) -> None:
    entries = []
    for record in pick_records:
        pick: CrosswalkPick = record["pick"]
        rank = record["rank"]
        scene_name = crosswalk_scene_name(core_scene_name, rank)
        entry = {
            "rank": rank,
            "scene_name": scene_name,
            "crosswalk_id": pick.crosswalk_id,
            "junction_id": pick.junction_id,
            "crossed_edge_ids": list(pick.crossed_edge_ids),
            "approach_edge_ids": list(pick.approach_edge_ids),
            "max_approach_lane_m": pick.max_approach_lane_m,
            "center_xy": [pick.center_xy[0], pick.center_xy[1]],
            "output_dir": scene_name,
            "preview": f"{scene_name}/{record.get('preview_name', 'custom_cropped.png')}",
        }
        viability: ManifestViabilityResult | None = record.get("viability")
        if viability is not None:
            entry["manifest_viable"] = viability.viable
            if not viability.viable:
                entry["manifest_skip_reason"] = viability.reason
                entry["manifest_skip_detail"] = viability.detail
        if record.get("written"):
            entry["written"] = True
        crop_skip = record.get("crop_skip_reason")
        if crop_skip:
            entry["crop_skip_reason"] = crop_skip
        entries.append(entry)
    index_path = core_scene_dir / "crosswalks.json"
    index_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def mark_core_crop_attempted(
    core_scene_dir: Path,
    core_scene_name: str,
    *,
    dry_run: bool,
    skipped: str | None = None,
) -> None:
    """Record that this core was processed so build_scene_pool does not retry forever."""
    if dry_run:
        return
    index_path = core_scene_dir / "crosswalks.json"
    payload = [
        {
            "written": False,
            "skipped": skipped or "no_viable_crop",
            "scene_name": core_scene_name,
        }
    ]
    index_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def process_core_scene(
    core_scene_dir: Path,
    scenes_root: Path,
    *,
    radius_m: float,
    crop_mode: str,
    trim_geometry: bool,
    min_approach_lane_m: float,
    max_crosswalks: int,
    preview_name: str,
    dry_run: bool,
    overwrite: bool,
    require_manifest_viable: bool = True,
    min_ego_lane_m: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END,
) -> int:
    """Crop crosswalk variants for one core scene. Returns number of scenes written."""
    core_scene_name = core_scene_dir.name
    print(f"\n=== {core_scene_name} (core) ===")
    try:
        meta = load_scene_meta(core_scene_dir)
        source_net = resolve_full_source_net(core_scene_dir, meta)
        picks = find_ranked_crosswalks(
            source_net,
            min_approach_lane_m=min_approach_lane_m,
            max_crosswalks=max_crosswalks,
        )
    except (FileNotFoundError, JunctionLayoutError) as exc:
        print(f"  [skip] {exc}")
        mark_core_crop_attempted(
            core_scene_dir,
            core_scene_name,
            dry_run=dry_run,
            skipped=str(exc),
        )
        return 0

    print(f"  source net: {source_net.name}")
    print(f"  picked {len(picks)} crossing(s) (max {max_crosswalks}):")
    for rank, pick in enumerate(picks):
        scene_name = crosswalk_scene_name(core_scene_name, rank)
        print(
            f"    [{rank}] {pick.crosswalk_id} @ junction {pick.junction_id} "
            f"(max approach {pick.max_approach_lane_m:.1f}m, "
            f"{pick.approach_lane_count} arm(s)) -> scenes/{scene_name}"
        )

    if dry_run:
        return len(picks)

    if not picks:
        print("  [skip] no qualifying crossings")
        mark_core_crop_attempted(
            core_scene_dir,
            core_scene_name,
            dry_run=dry_run,
            skipped="no_qualifying_crossings",
        )
        return 0

    base_meta = meta
    created = 0
    skipped_manifest = 0
    skipped_crop = 0
    pick_records: list[dict] = []
    for rank, pick in enumerate(picks):
        scene_name = crosswalk_scene_name(core_scene_name, rank)
        out_dir = scenes_root / scene_name
        record: dict = {
            "rank": rank,
            "pick": pick,
            "preview_name": preview_name,
            "written": False,
            "viability": None,
        }
        pick_records.append(record)

        if out_dir.exists():
            if not overwrite:
                print(f"  [skip existing] {scene_name}")
                continue
            shutil.rmtree(out_dir)

        try:
            with tempfile.TemporaryDirectory(prefix="crosswalk_crop_") as tmp:
                tmp_dir = Path(tmp)
                crop_scene_to_crosswalk_pick(
                    core_scene_dir,
                    pick,
                    source_net=source_net,
                    radius_m=radius_m,
                    crop_mode=crop_mode,
                    trim_geometry=trim_geometry,
                    output_dir=tmp_dir,
                    output_scene_name=scene_name,
                    base_meta=base_meta,
                    backup_original=False,
                    crosswalk_rank=rank,
                    core_scene_name=core_scene_name,
                )

                viability: ManifestViabilityResult | None = None
                if require_manifest_viable:
                    scene_meta = json.loads((tmp_dir / "meta.json").read_text(encoding="utf-8"))
                    net_file = scene_meta.get("net_file", "map.net.xml")
                    viability = check_manifest_viability(
                        tmp_dir / net_file,
                        meta=scene_meta,
                        min_ego_lane_m=min_ego_lane_m,
                    )
                    record["viability"] = viability
                    if not viability.viable:
                        print(
                            f"  [skip manifest] {scene_name}: "
                            f"{viability.reason} — {viability.detail}"
                        )
                        skipped_manifest += 1
                        continue

                shutil.copytree(tmp_dir, out_dir)
        except JunctionLayoutError as exc:
            print(f"  [skip crop] {scene_name}: {exc}")
            record["crop_skip_reason"] = str(exc)
            skipped_crop += 1
            continue
        except Exception as exc:
            print(f"  [skip crop] {scene_name}: {type(exc).__name__}: {exc}")
            record["crop_skip_reason"] = f"{type(exc).__name__}: {exc}"
            skipped_crop += 1
            continue

        preview_path = out_dir / preview_name
        render_preview(out_dir, pick.center_xy, preview_path)
        print(f"  wrote scenes/{scene_name}/ ({preview_name})")
        record["written"] = True
        created += 1

    write_crosswalks_index(core_scene_dir, core_scene_name, pick_records)
    print(f"  wrote core/{core_scene_name}/crosswalks.json")
    if skipped_crop:
        print(f"  skipped {skipped_crop} crossing(s) that failed crop/prune")
    if skipped_manifest:
        print(f"  skipped {skipped_manifest} crossing(s) that would fail manifest generation")
    return created


def uncropped_core_dirs(core_root: Path) -> list[Path]:
    """Core scenes that have not been cropped yet (no crosswalks.json)."""
    return [
        core_dir
        for core_dir in discover_core_scene_dirs(core_root)
        if not (core_dir / "crosswalks.json").is_file()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop core scenes into separate crosswalk folders under scenes/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "scenes",
        nargs="*",
        help="Core scene folder name(s) under --core-dir (e.g. sign_71853)",
    )
    parser.add_argument("--core-dir", type=Path, default=CORE_DIR_DEFAULT)
    parser.add_argument("--scenes-dir", type=Path, default=SCENES_DIR_DEFAULT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_CROSSWALK_CROP_RADIUS_M,
        help=f"Geo/junction crop radius in meters (default: {DEFAULT_CROSSWALK_CROP_RADIUS_M})",
    )
    parser.add_argument(
        "--crop-mode",
        choices=("geo", "junction"),
        default="geo",
        help="geo = square bbox around crossing center (default); junction = junction arms only",
    )
    parser.add_argument(
        "--trim-geometry",
        action="store_true",
        help="Clip lane shapes at the geo boundary (default: keep full edges that extend past radius)",
    )
    parser.add_argument(
        "--min-approach-lane",
        type=float,
        default=10.0,
        help="Approach lane must be longer than this to qualify (default: 10 m)",
    )
    parser.add_argument(
        "--max-crosswalks",
        type=int,
        default=8,
        help="Maximum crossings to crop per core scene (default: 8)",
    )
    parser.add_argument("--preview-name", default="custom_cropped.png")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-require-manifest-viable", action="store_true")
    parser.add_argument(
        "--min-ego-lane",
        type=float,
        default=DEFAULT_SPAWN_DISTANCE_BEFORE_END,
        help="Min approach lane length for manifest viability check",
    )
    args = parser.parse_args()

    if args.max_crosswalks < 1:
        sys.exit("--max-crosswalks must be at least 1")

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
            crop_mode=args.crop_mode,
            trim_geometry=args.trim_geometry,
            min_approach_lane_m=args.min_approach_lane,
            max_crosswalks=args.max_crosswalks,
            preview_name=args.preview_name,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            require_manifest_viable=not args.no_require_manifest_viable,
            min_ego_lane_m=args.min_ego_lane,
        )
        if created > 0:
            ok += 1
        created_total += created

    print(
        f"\nDone: {ok}/{len(core_scene_dirs)} core scene(s) processed, "
        f"{created_total} crosswalk scene(s) written."
    )
    if ok < len(core_scene_dirs):
        sys.exit(1)


if __name__ == "__main__":
    main()
