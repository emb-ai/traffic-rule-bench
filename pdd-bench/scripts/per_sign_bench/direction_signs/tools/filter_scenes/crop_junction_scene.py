#!/usr/bin/env python3
"""Crop core maps into 4.1.1 dual-path scenes (variant 1).

Selection first: on the full core net find an X-junction approach where the same
destination is reachable via a *shorter* left/right turn and a *longer* straight
path through the junction. Then crop to the XY bbox of both paths (+ margin).

Examples:
    python tools/filter_scenes/crop_junction_scene.py --limit 5
    python tools/filter_scenes/crop_junction_scene.py sign_72915 --overwrite
    python tools/filter_scenes/crop_junction_scene.py --min-gain 25 --margin 50
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
DIRECTION_SIGNS_DIR = TOOLS_DIR.parent.parent
SCENES_DIR_DEFAULT = DIRECTION_SIGNS_DIR / "scenes"
CORE_DIR_DEFAULT = SCENES_DIR_DEFAULT / "core"

sys.path.insert(0, str(DIRECTION_SIGNS_DIR))

from lib.direction_dual_path import (  # noqa: E402
    DualPathScenario,
    crop_scene_to_dual_path_scenario,
    find_ranked_dual_path_picks,
)
from lib.junction_crop import (  # noqa: E402
    JunctionLayoutError,
    JunctionPick,
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
    pick_records: list[dict],
) -> None:
    entries = []
    for record in pick_records:
        pick: JunctionPick = record["pick"]
        scenario: DualPathScenario | None = record.get("scenario")
        rank = record["rank"]
        scene_name = junction_scene_name(core_scene_name, rank)
        entry = {
            "rank": rank,
            "scene_name": scene_name,
            "junction_id": pick.junction_id,
            "arm_count": pick.arm_count,
            "total_lanes": pick.total_lanes,
            "incoming_edge_ids": list(pick.incoming_edge_ids),
            "center_xy": [pick.center_xy[0], pick.center_xy[1]],
            "output_dir": scene_name,
            "preview": f"{scene_name}/{record.get('preview_name', 'custom_cropped.png')}",
            "crop_mode": "dual_path_bbox",
        }
        if scenario is not None:
            entry["dual_path"] = {
                "ego_edge_id": scenario.ego_edge_id,
                "dest_edge_id": scenario.dest_edge_id,
                "turn_dir": scenario.turn_dir,
                "turn_length_m": scenario.turn_length_m,
                "straight_length_m": scenario.straight_length_m,
                "gain_m": scenario.gain_m,
            }
        if record.get("skip_reason"):
            entry["skip_reason"] = record["skip_reason"]
            entry["skip_detail"] = record.get("skip_detail", "")
        if record.get("written"):
            entry["written"] = True
        entries.append(entry)
    index_path = core_scene_dir / "junctions.json"
    index_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def process_core_scene(
    core_scene_dir: Path,
    scenes_root: Path,
    *,
    margin_m: float,
    min_lane_length_m: float,
    min_gain_m: float,
    max_scenarios: int,
    preview_name: str,
    dry_run: bool,
    overwrite: bool,
) -> int:
    """Find dual-path scenarios and crop each to its path-union bbox."""
    core_scene_name = core_scene_dir.name
    print(f"\n=== {core_scene_name} (core) ===")
    try:
        meta = load_scene_meta(core_scene_dir)
        source_net = resolve_full_source_net(core_scene_dir, meta)
        ranked = find_ranked_dual_path_picks(
            source_net,
            min_lane_length_m=min_lane_length_m,
            min_gain_m=min_gain_m,
            max_scenarios=max_scenarios,
        )
    except (FileNotFoundError, JunctionLayoutError) as exc:
        print(f"  [skip] {exc}")
        return 0

    if not ranked:
        print("  [skip] no dual-path (turn shorter / straight longer) scenario found")
        write_junctions_index(core_scene_dir, core_scene_name, [])
        return 0

    print(f"  source net: {source_net.name}")
    print(f"  found {len(ranked)} dual-path scenario(s) (max {max_scenarios}):")
    for rank, (scenario, pick) in enumerate(ranked):
        scene_name = junction_scene_name(core_scene_name, rank)
        print(
            f"    [{rank}] j={pick.junction_id} ego={scenario.ego_edge_id} "
            f"dest={scenario.dest_edge_id} turn={scenario.turn_dir} "
            f"Lt={scenario.turn_length_m:.0f}m Ls={scenario.straight_length_m:.0f}m "
            f"gain={scenario.gain_m:.0f}m -> scenes/{scene_name}"
        )

    if dry_run:
        return len(ranked)

    base_meta = meta
    created = 0
    pick_records: list[dict] = []
    for rank, (scenario, pick) in enumerate(ranked):
        scene_name = junction_scene_name(core_scene_name, rank)
        out_dir = scenes_root / scene_name
        record: dict = {
            "rank": rank,
            "pick": pick,
            "scenario": scenario,
            "preview_name": preview_name,
            "written": False,
        }
        pick_records.append(record)

        if out_dir.exists():
            if not overwrite:
                print(f"  [skip existing] {scene_name}")
                continue
            shutil.rmtree(out_dir)

        try:
            with tempfile.TemporaryDirectory(prefix="direction_crop_") as tmp:
                tmp_dir = Path(tmp)
                cropped = crop_scene_to_dual_path_scenario(
                    core_scene_dir,
                    scenario,
                    source_net=source_net,
                    margin_m=margin_m,
                    output_dir=tmp_dir,
                    output_scene_name=scene_name,
                    base_meta=base_meta,
                    junction_rank=rank,
                    core_scene_name=core_scene_name,
                )
                record["scenario"] = cropped
                shutil.copytree(tmp_dir, out_dir)
        except JunctionLayoutError as exc:
            print(f"  [skip crop] {scene_name}: {exc}")
            record["skip_reason"] = "dual_path_lost_after_crop"
            record["skip_detail"] = str(exc)
            continue

        preview_path = out_dir / preview_name
        render_preview(out_dir, pick.center_xy, preview_path)
        print(f"  wrote scenes/{scene_name}/ ({preview_name})")
        record["written"] = True
        created += 1

    write_junctions_index(core_scene_dir, core_scene_name, pick_records)
    print(f"  wrote core/{core_scene_name}/junctions.json")
    return created


def uncropped_core_dirs(core_root: Path, *, retry_failed: bool = False) -> list[Path]:
    """Core scenes not yet cropped (no ``junctions.json``)."""
    uncropped: list[Path] = []
    for core_dir in discover_core_scene_dirs(core_root):
        index_path = core_dir / "junctions.json"
        if not index_path.is_file():
            uncropped.append(core_dir)
            continue
        if not retry_failed:
            continue
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            uncropped.append(core_dir)
            continue
        if any(entry.get("written") for entry in entries):
            continue
        uncropped.append(core_dir)
    return uncropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop core scenes into 4.1.1 dual-path folders under scenes/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "scenes",
        nargs="*",
        help="Core scene folder name(s) under --core-dir (e.g. sign_72915)",
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
        help=f"Output scenes root (default: {SCENES_DIR_DEFAULT})",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process first N core scenes when none named")
    parser.add_argument(
        "--margin",
        type=float,
        default=40.0,
        help="XY margin (m) around turn+straight path union bbox (default: 40)",
    )
    parser.add_argument(
        "--min-lane-length",
        type=float,
        default=8.0,
        help="Min ego approach lane length (m) for dual-path selection (default: 8)",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=20.0,
        help="Min straight_length - turn_length (m) (default: 20)",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=5,
        help="Maximum dual-path scenarios to crop per core scene (default: 5)",
    )
    parser.add_argument(
        "--preview-name",
        default="custom_cropped.png",
        help="Cropped-map preview filename (default: custom_cropped.png)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing junction scene folders",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report dual-path picks")
    args = parser.parse_args()

    if args.max_scenarios < 1:
        sys.exit("--max-scenarios must be at least 1")

    core_root = args.core_dir.expanduser().resolve()
    scenes_root = args.scenes_dir.expanduser().resolve()
    scenes_root.mkdir(parents=True, exist_ok=True)

    if not core_root.is_dir():
        sys.exit(
            f"Core scenes directory not found: {core_root}\n"
            "Run import_catalog_scenes.py --arms 4 first."
        )

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
            margin_m=args.margin,
            min_lane_length_m=args.min_lane_length,
            min_gain_m=args.min_gain,
            max_scenarios=args.max_scenarios,
            preview_name=args.preview_name,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        if created > 0 or args.dry_run:
            ok += 1
        created_total += created

    print(
        f"\nDone: {ok}/{len(core_scene_dirs)} core scene(s) processed, "
        f"{created_total} dual-path scene(s) written."
    )


if __name__ == "__main__":
    main()
