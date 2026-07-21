#!/usr/bin/env python3
"""Crop core maps into dual-path no-turn-sign scenes.

Selection first: on the full core net find an X-junction approach where the same
destination is reachable via a *shorter* forbidden (baseline) first exit and a
*longer* allowed (compliant) path. Roles come from ``--pdd-code``:

  * 3.18.1: baseline r, compliant s/l
  * 3.18.2: baseline l, compliant s/r

Then crop to the XY bbox of both paths (+ margin).

Examples:
    python tools/filter_scenes/crop_junction_scene.py --limit 5
    python tools/filter_scenes/crop_junction_scene.py --pdd-code 3.18.1 --limit 5
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
NO_TURN_SIGNS_DIR = TOOLS_DIR.parent.parent
SCENES_BASE_DEFAULT = NO_TURN_SIGNS_DIR / "scenes"
SCENES_DIR_DEFAULT = SCENES_BASE_DEFAULT  # resolved per --pdd-code in main()
CORE_DIR_DEFAULT = SCENES_BASE_DEFAULT / "core"  # legacy; prefer local_core_scenes_root

sys.path.insert(0, str(NO_TURN_SIGNS_DIR))

from lib.direction_dual_path import (  # noqa: E402
    DualPathScenario,
    crop_scene_to_dual_path_scenario,
    dual_path_role_dirs,
    find_ranked_dual_path_picks,
)
from lib.no_turn_sign_spec import (  # noqa: E402
    DEFAULT_PDD_CODE,
    NO_TURN_SIGN_CODES,
    get_no_turn_sign_spec,
    local_core_scenes_root,
    local_scenes_root,
)
from lib.junction_crop import (  # noqa: E402
    JunctionLayoutError,
    JunctionPick,
    resolve_full_source_net,
)
from lib.metadrive_route_check import filter_dual_paths_metadrive  # noqa: E402
from lib.sumo_utils import (  # noqa: E402
    is_core_scene_name,
    junction_scene_name,
    load_scene_meta,
    resolve_net_file,
    resolve_scene_dir,
)
from tools.render_map import (  # noqa: E402
    dual_path_overlays,
    parse_sumo_net,
    point_on_edge,
    edge_shapes_by_id,
    render_network,
)


def _core_name_of_scene_dir(scene_dir: Path, meta: dict) -> str:
    core = meta.get("core_scene_name")
    if core:
        return str(core)
    name = scene_dir.name
    return name.rsplit("_j", 1)[0] if "_j" in name else name


def claimed_scenario_keys(
    scenes_root: Path,
    *,
    exclude_cores: set[str] | None = None,
) -> set[tuple[str, str]]:
    """(junction_id, ego_edge_id) keys already used by existing cropped scenes.

    Different catalog signs near the same junction produce identical dual-path
    scenes; one (junction, approach) pair must yield at most one scene across
    ALL core maps. ``exclude_cores`` skips scenes about to be re-cropped with
    ``--overwrite`` so they can re-claim their own keys.
    """
    keys: set[tuple[str, str]] = set()
    if not scenes_root.is_dir():
        return keys
    for entry in sorted(scenes_root.iterdir()):
        if not entry.is_dir() or entry.name in ("core", "_rejected"):
            continue
        meta_path = entry / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if exclude_cores and _core_name_of_scene_dir(entry, meta) in exclude_cores:
            continue
        jid = meta.get("junction_id")
        ego = meta.get("road_id")
        if jid and ego:
            keys.add((str(jid), str(ego)))
    return keys


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


def render_dual_path_preview(
    scene_dir: Path,
    scenario: DualPathScenario,
    out_path: Path,
    *,
    junction_xy: tuple[float, float] | None = None,
) -> None:
    """Render cropped net with straight / turn / shared-to-dest overlays.

    Both routes end at the same destination. Shared edges are drawn in their
    own color so the turn overlay does not hide that the straight path arrives.
    """
    meta = load_scene_meta(scene_dir)
    net_file = resolve_net_file(scene_dir, meta)
    edges, junctions = parse_sumo_net(scene_dir / net_file)
    shapes = edge_shapes_by_id(edges)
    # Spawn near end of ego approach (toward junction); dest at end of dest edge.
    spawn_xy = point_on_edge(shapes, scenario.ego_edge_id, at="end")
    dest_xy = point_on_edge(shapes, scenario.dest_edge_id, at="end")
    render_network(
        edges,
        junctions,
        out_path,
        marker_xy=junction_xy or scenario.junction_center_xy,
        spawn_xy=spawn_xy,
        dest_xy=dest_xy,
        legend=True,
        path_overlays=dual_path_overlays(
            scenario.ego_edge_id,
            scenario.turn_path,
            scenario.straight_path,
            turn_dir=scenario.turn_dir,
            turn_length_m=scenario.turn_length_m,
            straight_length_m=scenario.straight_length_m,
        ),
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
    validate_metadrive: bool,
    pdd_code: str = DEFAULT_PDD_CODE,
    claimed: set[tuple[str, str]] | None = None,
) -> int:
    """Find dual-path scenarios and crop each to its path-union bbox.

    Chosen spawn/dest + both paths are written into ``meta.json`` and become the
    canonical endpoints for ``generate_manifest`` (no rediscovery).

    ``claimed`` deduplicates (junction, ego approach) pairs across core scenes:
    picks colliding with an already-written scene are skipped, and written
    scenes claim their key.
    """
    sign_spec = get_direction_sign_spec(pdd_code)
    pdd_code = sign_spec.pdd_code
    baseline_dirs, compliant_dirs = dual_path_role_dirs(pdd_code)
    core_scene_name = core_scene_dir.name
    print(
        f"\n=== {core_scene_name} (core) === "
        f"sign={pdd_code} ({sign_spec.title}); "
        f"baseline={baseline_dirs} compliant={compliant_dirs}"
    )
    try:
        meta = load_scene_meta(core_scene_dir)
        source_net = resolve_full_source_net(core_scene_dir, meta)
        # Several dests per arm so MetaDrive hop-cap can pick a routable one.
        ranked = find_ranked_dual_path_picks(
            source_net,
            pdd_code=pdd_code,
            min_lane_length_m=min_lane_length_m,
            min_gain_m=min_gain_m,
            max_scenarios=max(max_scenarios * 8, 40),
            dests_per_arm=8,
        )
    except (FileNotFoundError, JunctionLayoutError) as exc:
        print(f"  [skip] {exc}")
        return 0

    if claimed:
        deduped = []
        for sc, pick in ranked:
            if (str(sc.junction_id), str(sc.ego_edge_id)) in claimed:
                print(
                    f"  [skip duplicate] junction {sc.junction_id} "
                    f"ego {sc.ego_edge_id}: already covered by another scene"
                )
                continue
            deduped.append((sc, pick))
        ranked = deduped

    if not ranked:
        print(
            "  [skip] no dual-path (baseline shorter / compliant longer) scenario found"
        )
        write_junctions_index(core_scene_dir, core_scene_name, [])
        return 0

    scenarios = [sc for sc, _pick in ranked]
    if validate_metadrive:
        try:
            kept, dropped = filter_dual_paths_metadrive(
                scenarios,
                source_net,
                one_per_ego=False,
                max_keep=None,
                pdd_code=pdd_code,
            )
            print(
                f"  MetaDrive filter on core net: kept {len(kept)}, dropped {dropped}"
            )
            if kept:
                # Prefer larger length gain among MetaDrive-routable dests.
                kept.sort(key=lambda s: (-s.gain_m, s.turn_length_m))
                by_key = {
                    (s.junction_id, s.ego_edge_id, s.dest_edge_id): (s, p)
                    for s, p in ranked
                }
                seen_ego: set[str] = set()
                ranked = []
                for s in kept:
                    if s.ego_edge_id in seen_ego:
                        continue
                    key = (s.junction_id, s.ego_edge_id, s.dest_edge_id)
                    if key not in by_key:
                        continue
                    seen_ego.add(s.ego_edge_id)
                    ranked.append(by_key[key])
                    if len(ranked) >= max_scenarios:
                        break
            else:
                print("  [warn] no MetaDrive-routable dual-path; keeping shortest SUMO picks")
                # Fall back: first dest per ego from already short-preferring list.
                seen_ego = set()
                fallback = []
                for sc, pick in ranked:
                    if sc.ego_edge_id in seen_ego:
                        continue
                    seen_ego.add(sc.ego_edge_id)
                    fallback.append((sc, pick))
                    if len(fallback) >= max_scenarios:
                        break
                ranked = fallback
        except Exception as exc:
            print(f"  [warn] MetaDrive validation skipped: {exc}")
            seen_ego = set()
            fallback = []
            for sc, pick in ranked:
                if sc.ego_edge_id in seen_ego:
                    continue
                seen_ego.add(sc.ego_edge_id)
                fallback.append((sc, pick))
                if len(fallback) >= max_scenarios:
                    break
            ranked = fallback
    else:
        seen_ego = set()
        trimmed = []
        for sc, pick in ranked:
            if sc.ego_edge_id in seen_ego:
                continue
            seen_ego.add(sc.ego_edge_id)
            trimmed.append((sc, pick))
            if len(trimmed) >= max_scenarios:
                break
        ranked = trimmed

    print(f"  source net: {source_net.name}")
    print(f"  selected {len(ranked)} dual-path scenario(s) (max {max_scenarios}):")
    for rank, (scenario, pick) in enumerate(ranked):
        scene_name = junction_scene_name(core_scene_name, rank)
        print(
            f"    [{rank}] j={pick.junction_id} ego={scenario.ego_edge_id} "
            f"dest={scenario.dest_edge_id} "
            f"baseline={scenario.turn_dir} compliant={scenario.compliant_dir} "
            f"Lb={scenario.turn_length_m:.0f}m Lc={scenario.straight_length_m:.0f}m "
            f"gain={scenario.gain_m:.0f}m -> {scenes_root.name}/{scene_name}"
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
        render_dual_path_preview(
            out_dir,
            record["scenario"],
            preview_path,
            junction_xy=pick.center_xy,
        )
        print(
            f"  wrote {scenes_root.name}/{scene_name}/ "
            f"(spawn={record['scenario'].ego_edge_id} "
            f"dest={record['scenario'].dest_edge_id}, {preview_name})"
        )
        record["written"] = True
        created += 1
        if claimed is not None:
            written_sc = record["scenario"]
            claimed.add((str(written_sc.junction_id), str(written_sc.ego_edge_id)))

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
        description="Crop core scenes into dual-path folders under scenes/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "scenes",
        nargs="*",
        help="Core scene folder name(s) under --core-dir (e.g. sign_72915)",
    )
    parser.add_argument(
        "--pdd-code",
        default=DEFAULT_PDD_CODE,
        choices=list(DIRECTION_SIGN_CODES),
        help=f"Direction-sign member for dual-path roles (default: {DEFAULT_PDD_CODE})",
    )
    parser.add_argument(
        "--core-dir",
        type=Path,
        default=None,
        help="Core scenes root (default: scenes/<slug>/core)",
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=None,
        help="Output cropped-scenes root (default: scenes/<slug>)",
    )
    parser.add_argument(
        "--scenes-base",
        type=Path,
        default=SCENES_BASE_DEFAULT,
        help=f"Parent of per-sign folders (default: {SCENES_BASE_DEFAULT})",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process first N core scenes when none named")
    parser.add_argument(
        "--margin",
        type=float,
        default=40.0,
        help="XY margin (m) around baseline+compliant path union bbox (default: 40)",
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
        help="Min compliant_length - baseline_length (m) (default: 20)",
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
    parser.add_argument(
        "--skip-metadrive-check",
        action="store_true",
        help="Do not require MetaDrive-routable spawn→dest when selecting endpoints",
    )
    args = parser.parse_args()

    if args.max_scenarios < 1:
        sys.exit("--max-scenarios must be at least 1")

    sign_spec = get_direction_sign_spec(args.pdd_code)
    scenes_base = args.scenes_base.expanduser().resolve()
    core_root = (
        args.core_dir.expanduser().resolve()
        if args.core_dir is not None
        else local_core_scenes_root(scenes_base, sign_spec.pdd_code).resolve()
    )
    scenes_root = (
        args.scenes_dir.expanduser().resolve()
        if args.scenes_dir is not None
        else local_scenes_root(scenes_base, sign_spec.pdd_code).resolve()
    )
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

    print(
        f"Cropping dual-path for {sign_spec.pdd_code} ({sign_spec.title}); "
        f"roles baseline={dual_path_role_dirs(sign_spec.pdd_code)[0]} "
        f"compliant={dual_path_role_dirs(sign_spec.pdd_code)[1]}"
    )

    # Cross-core dedup: one scene per (junction, ego approach). Scenes being
    # re-cropped with --overwrite may re-claim their own keys.
    exclude_cores = (
        {d.name for d in core_scene_dirs} if args.overwrite else set()
    )
    claimed = claimed_scenario_keys(scenes_root, exclude_cores=exclude_cores)
    if claimed:
        print(f"Dedup: {len(claimed)} (junction, approach) key(s) already claimed")

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
            validate_metadrive=not args.skip_metadrive_check,
            pdd_code=sign_spec.pdd_code,
            claimed=claimed,
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
