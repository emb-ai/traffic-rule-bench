#!/usr/bin/env python3
"""Crop core 4.3 catalog scenes into roundabout benchmark folders.

By default writes one cropped folder per core map (catalog sign spoke):
  scenes/core/sign_77277/     # untouched core map
  scenes/sign_77277_rb/       # ring + spokes, sign on catalog road

Use --per-spoke for the legacy layout (one folder per attached road):
  scenes/sign_77277_rb_s00/, sign_77277_rb_s01/, ...

Each crop keeps only the traffic circle and attached spoke roads.

Examples:
    python tools/filter_scenes/crop_junction_scene.py
    python tools/filter_scenes/crop_junction_scene.py sign_77277 --spoke-length 100
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROUNDABOUT_SIGN_DIR = TOOLS_DIR.parent.parent
SCENES_DIR_DEFAULT = ROUNDABOUT_SIGN_DIR / "scenes"
CORE_DIR_DEFAULT = SCENES_DIR_DEFAULT / "core"

sys.path.insert(0, str(ROUNDABOUT_SIGN_DIR))

from lib.manifest_config import (  # noqa: E402
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_SPAWN_DISTANCE_BEFORE_END,
)
from lib.manifest_viability import check_manifest_viability  # noqa: E402
from lib.junction_crop import (  # noqa: E402
    JunctionLayoutError,
    crop_scene_to_roundabout,
    resolve_catalog_sign_spoke,
    resolve_full_source_net,
    roundabout_scene_name,
    roundabout_spoke_scene_name,
)
from lib.roundabout_fingerprint import (  # noqa: E402
    RoundaboutFingerprintRegistry,
    fingerprint_from_sumo_roundabout,
)
from lib.roundabout_topology import detect_roundabout, resolve_sumo_roundabout  # noqa: E402
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


def render_preview(scene_dir: Path, out_path: Path) -> None:
    meta = load_scene_meta(scene_dir)
    net_file = resolve_net_file(scene_dir, meta)
    edges, junctions = parse_sumo_net(scene_dir / net_file)
    render_network(edges, junctions, out_path)


def cropped_spoke_edges_for_core(scenes_root: Path, core_scene_name: str) -> set[str]:
    """Spoke edge ids already covered by cropped folders for this core map."""
    covered: set[str] = set()
    for scene_dir in existing_roundabout_scenes(scenes_root, core_scene_name):
        meta_path = scene_dir / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        spoke = (
            meta.get("roundabout_sign_spoke_edge")
            or meta.get("roundabout_approach_edge")
            or meta.get("road_id")
        )
        if spoke:
            covered.add(str(spoke))
    return covered


def per_spoke_expand_jobs(
    core_scene_dir: Path,
    scenes_root: Path,
) -> list[tuple[int, str, str]]:
    """Crop jobs for spokes not yet covered by any folder from this core."""
    core_scene_name = core_scene_dir.name
    meta = load_scene_meta(core_scene_dir)
    source_net = resolve_full_source_net(core_scene_dir, meta)
    pick = detect_roundabout(source_net, sign_edge_id=meta.get("road_id"))
    covered = cropped_spoke_edges_for_core(scenes_root, core_scene_name)
    jobs: list[tuple[int, str, str]] = []
    for rank, spoke_edge_id in enumerate(pick.spoke_edge_ids):
        if spoke_edge_id in covered:
            continue
        jobs.append((rank, spoke_edge_id, roundabout_spoke_scene_name(core_scene_name, rank)))
    return jobs


def existing_roundabout_scenes(scenes_root: Path, core_scene_name: str) -> list[Path]:
    patterns = [
        f"{core_scene_name}_rb_s*",
        f"{core_scene_name}_rb",
    ]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(scenes_root.glob(pattern))
    return sorted({p for p in found if p.is_dir()})


def write_roundabout_index(core_scene_dir: Path, records: list[dict]) -> None:
    index_path = core_scene_dir / "junctions.json"
    index_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def mark_core_crop_attempted(
    core_scene_dir: Path,
    records: list[dict],
    *,
    dry_run: bool,
    skipped: str | None = None,
) -> None:
    """Record that this core was processed so build_scene_pool does not retry forever."""
    if dry_run:
        return
    if not records:
        records = [{"written": False, "skipped": skipped or "no_viable_crop"}]
    write_roundabout_index(core_scene_dir, records)


def process_core_scene(
    core_scene_dir: Path,
    scenes_root: Path,
    *,
    spoke_extension_m: float,
    max_spoke_length_m: float,
    min_lane_length_m: float,
    preview_name: str,
    dry_run: bool,
    overwrite: bool,
    require_manifest_viable: bool = True,
    min_ego_lane_m: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    one_per_core: bool = True,
    skip_duplicate_fingerprints: bool = True,
    crop_jobs: list[tuple[int, str, str]] | None = None,
    write_junction_index: bool = True,
) -> int:
    """Crop roundabout scene(s) from a core map. Returns scenes written."""
    core_scene_name = core_scene_dir.name
    print(f"\n=== {core_scene_name} (core) ===")

    try:
        meta = load_scene_meta(core_scene_dir)
        source_net = resolve_full_source_net(core_scene_dir, meta)
        source_pick = detect_roundabout(source_net, sign_edge_id=meta.get("road_id"))
    except (FileNotFoundError, JunctionLayoutError) as exc:
        print(f"  [skip] {exc}")
        mark_core_crop_attempted(core_scene_dir, [], dry_run=dry_run, skipped=str(exc))
        return 0

    spokes = list(source_pick.spoke_edge_ids)
    print(
        f"  source net: {source_net.name}, "
        f"{len(source_pick.ring_edge_ids)} ring edge(s), {len(spokes)} spoke(s)"
    )
    if not spokes:
        print("  [skip] no spokes on traffic circle")
        mark_core_crop_attempted(core_scene_dir, [], dry_run=dry_run, skipped="no_spokes")
        return 0

    try:
        sumo_rb = resolve_sumo_roundabout(source_net, sign_edge_id=meta.get("road_id"))
        fingerprint = fingerprint_from_sumo_roundabout(sumo_rb)
    except JunctionLayoutError as exc:
        print(f"  [skip] {exc}")
        mark_core_crop_attempted(core_scene_dir, [], dry_run=dry_run, skipped=str(exc))
        return 0

    registry = RoundaboutFingerprintRegistry.for_scenes_root(scenes_root)

    if crop_jobs is None:
        if one_per_core:
            try:
                sign_spoke = resolve_catalog_sign_spoke(
                    source_pick, meta.get("road_id"), spokes
                )
            except JunctionLayoutError as exc:
                print(f"  [skip] {exc}")
                mark_core_crop_attempted(core_scene_dir, [], dry_run=dry_run, skipped=str(exc))
                return 0
            crop_jobs = [(0, sign_spoke, roundabout_scene_name(core_scene_name))]
            print(f"  one crop on catalog spoke {sign_spoke!r} -> {crop_jobs[0][2]}")
        else:
            crop_jobs = [
                (rank, spoke_edge_id, roundabout_spoke_scene_name(core_scene_name, rank))
                for rank, spoke_edge_id in enumerate(spokes)
            ]
    else:
        one_per_core = False
        print(f"  spoke expansion: {len(crop_jobs)} additional crop(s)")

    if dry_run:
        if skip_duplicate_fingerprints and crop_jobs:
            probe_name = crop_jobs[0][2]
            duplicate = registry.duplicate_owner(
                fingerprint,
                scene_name=probe_name,
                core_scene_name=core_scene_name,
                one_per_core=one_per_core,
            )
            if duplicate is not None:
                print(
                    f"  [dry-run duplicate] would skip; same ring as "
                    f"{duplicate.get('scene_name')!r}"
                )
                return 0
        return len(crop_jobs)

    records: list[dict] = []
    written = 0
    try:
        if overwrite:
            for old_dir in existing_roundabout_scenes(scenes_root, core_scene_name):
                registry.remove_scene(old_dir.name)
                shutil.rmtree(old_dir)
            registry.save()

        for rank, spoke_edge_id, scene_name in crop_jobs:
            out_dir = scenes_root / scene_name
            if skip_duplicate_fingerprints:
                duplicate = registry.duplicate_owner(
                    fingerprint,
                    scene_name=scene_name,
                    core_scene_name=core_scene_name,
                    one_per_core=one_per_core,
                )
                if duplicate is not None:
                    owner = duplicate.get("scene_name", "?")
                    print(
                        f"  [skip duplicate roundabout] {scene_name}: "
                        f"same SUMO nodes as {owner!r}"
                    )
                    records.append(
                        {
                            "scene_name": scene_name,
                            "spoke_edge": spoke_edge_id,
                            "spoke_rank": rank,
                            "written": False,
                            "skipped": "duplicate_fingerprint",
                            "duplicate_of": owner,
                            "sumo_roundabout_fingerprint": fingerprint,
                        }
                    )
                    continue

            if out_dir.exists() and not overwrite:
                print(f"  [skip existing] {scene_name}")
                records.append(
                    {
                        "scene_name": scene_name,
                        "spoke_edge": spoke_edge_id,
                        "spoke_rank": rank,
                        "written": False,
                        "skipped": "exists",
                    }
                )
                continue

            record: dict = {
                "scene_name": scene_name,
                "spoke_edge": spoke_edge_id,
                "spoke_rank": rank,
                "written": False,
            }

            with tempfile.TemporaryDirectory(prefix="roundabout_crop_") as tmp:
                tmp_dir = Path(tmp)
                try:
                    pick = crop_scene_to_roundabout(
                        core_scene_dir,
                        ego_spoke_edge_id=spoke_edge_id,
                        spoke_rank=rank,
                        spoke_extension_m=spoke_extension_m,
                        max_spoke_length_m=max_spoke_length_m,
                        min_lane_length_m=min_lane_length_m,
                        output_dir=tmp_dir,
                        output_scene_name=scene_name,
                        backup_original=False,
                        source_pick=source_pick,
                        sumo_roundabout=sumo_rb,
                    )
                except JunctionLayoutError as exc:
                    print(f"  [skip] {scene_name}: {exc}")
                    record["error"] = str(exc)
                    records.append(record)
                    continue

                record.update(
                    {
                        "entry_junction": pick.entry_junction_id,
                        "ring_edges": len(pick.ring_edge_ids),
                        "spokes": len(pick.spoke_edge_ids),
                        "center_xy": [pick.center_xy[0], pick.center_xy[1]],
                    }
                )

                if require_manifest_viable:
                    scene_meta = json.loads((tmp_dir / "meta.json").read_text(encoding="utf-8"))
                    net_file = scene_meta.get("net_file", "map.net.xml")
                    viability = check_manifest_viability(
                        tmp_dir / net_file,
                        meta=scene_meta,
                        min_ego_lane_m=min_ego_lane_m,
                        aux_distance_from_intersection=aux_distance_from_intersection,
                    )
                    record["manifest_viable"] = viability.viable
                    if not viability.viable:
                        print(
                            f"  [skip manifest] {scene_name} ({spoke_edge_id}): "
                            f"{viability.reason} — {viability.detail}"
                        )
                        records.append(record)
                        continue

                if out_dir.exists():
                    shutil.rmtree(out_dir)
                shutil.copytree(tmp_dir, out_dir)

            preview_path = out_dir / preview_name
            render_preview(out_dir, preview_path)
            registry.upsert(
                fingerprint,
                scene_name=scene_name,
                core_scene_name=core_scene_name,
                kind="cropped",
                sign_id=meta.get("sign_id"),
                sumo_roundabout_nodes=sumo_rb.node_ids,
                sumo_roundabout_ring_edges=sumo_rb.ring_edge_ids,
            )
            registry.save()
            print(f"  wrote scenes/{scene_name}/ sign on {spoke_edge_id} ({preview_name})")
            record["written"] = True
            record["preview"] = f"{scene_name}/{preview_name}"
            records.append(record)
            written += 1

        return written
    except Exception as exc:
        print(f"  [error] {core_scene_name}: {exc}")
        records.append({"written": False, "skipped": "error", "error": str(exc)})
        return 0
    finally:
        if write_junction_index:
            mark_core_crop_attempted(core_scene_dir, records, dry_run=dry_run)


def cores_with_cropped_scenes(scenes_root: Path, core_root: Path) -> list[Path]:
    """Core maps that already have at least one cropped roundabout folder."""
    cores: list[Path] = []
    for core_dir in discover_core_scene_dirs(core_root):
        if existing_roundabout_scenes(scenes_root, core_dir.name):
            cores.append(core_dir)
    return cores


def expand_spokes_until(
    scenes_root: Path,
    core_root: Path,
    *,
    target: int,
    preview_name: str,
    dry_run: bool,
    overwrite: bool,
    require_manifest_viable: bool = True,
    min_ego_lane_m: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    spoke_length_m: float = 80.0,
    min_lane_length_m: float = 10.0,
    skip_duplicate_fingerprints: bool = True,
) -> int:
    """Add per-spoke crops for cores that already have a base ``_rb`` scene."""
    from tools.filter_scenes.review_junction_scenes import discover_review_scenes

    candidates = len(discover_review_scenes(scenes_root, preview_name=preview_name))
    if candidates >= target:
        print(f"Already have {candidates} candidate scene(s) (target {target}).")
        return 0

    total_written = 0
    cores = cores_with_cropped_scenes(scenes_root, core_root)
    for core_dir in cores:
        if candidates >= target:
            break
        try:
            jobs = per_spoke_expand_jobs(core_dir, scenes_root)
        except (FileNotFoundError, JunctionLayoutError) as exc:
            print(f"  [skip expand] {core_dir.name}: {exc}")
            continue
        if not jobs:
            continue
        written = process_core_scene(
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
            one_per_core=False,
            skip_duplicate_fingerprints=skip_duplicate_fingerprints,
            crop_jobs=jobs,
            write_junction_index=False,
        )
        total_written += written
        candidates = len(discover_review_scenes(scenes_root, preview_name=preview_name))
    return total_written


def uncropped_core_dirs(core_root: Path) -> list[Path]:
    """Core scenes that have not been cropped yet (no junctions.json)."""
    return [
        core_dir
        for core_dir in discover_core_scene_dirs(core_root)
        if not (core_dir / "junctions.json").is_file()
    ]


def retryable_core_dirs(core_root: Path) -> list[Path]:
    """Cores whose last crop attempt wrote no roundabout scene (safe to retry)."""
    retryable: list[Path] = []
    for core_dir in discover_core_scene_dirs(core_root):
        index_path = core_dir / "junctions.json"
        if not index_path.is_file():
            continue
        try:
            records = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not records:
            retryable.append(core_dir)
            continue
        if any(record.get("written") for record in records):
            continue
        retryable.append(core_dir)
    return retryable


def clear_crop_attempt(core_dir: Path) -> None:
    index_path = core_dir / "junctions.json"
    if index_path.is_file():
        index_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop core scenes into roundabout folders under scenes/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--per-spoke",
        action="store_true",
        help="Emit one folder per spoke (sign_<id>_rb_s00, _s01, …) instead of one per core",
    )
    parser.add_argument(
        "--allow-duplicate-roundabout",
        action="store_true",
        help="Allow cropping even when scenes/roundabout_fingerprints.json already has this ring",
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
        help=f"Output scenes root (default: {SCENES_DIR_DEFAULT})",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process first N core scenes when none named")
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
        default="custom_cropped.png",
        help="Cropped-map preview filename (default: custom_cropped.png)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing cropped scene folders")
    parser.add_argument("--dry-run", action="store_true", help="Only report, do not write files")
    parser.add_argument(
        "--no-require-manifest-viable",
        action="store_true",
        help="Write crops even if generate_manifest would drop them",
    )
    parser.add_argument(
        "--min-ego-lane",
        type=float,
        default=DEFAULT_SPAWN_DISTANCE_BEFORE_END,
        help=f"Min ego approach lane length for manifest check (default: {DEFAULT_SPAWN_DISTANCE_BEFORE_END})",
    )
    parser.add_argument(
        "--aux-distance",
        type=float,
        default=DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
        help=f"Aux spawn distance for manifest check (default: {DEFAULT_AUX_DISTANCE_FROM_INTERSECTION})",
    )
    args = parser.parse_args()

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
            spoke_extension_m=args.spoke_length,
            max_spoke_length_m=args.spoke_length,
            min_lane_length_m=args.min_lane_length,
            preview_name=args.preview_name,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            require_manifest_viable=not args.no_require_manifest_viable,
            min_ego_lane_m=args.min_ego_lane,
            aux_distance_from_intersection=args.aux_distance,
            one_per_core=not args.per_spoke,
            skip_duplicate_fingerprints=not args.allow_duplicate_roundabout,
        )
        if created > 0:
            ok += 1
        created_total += created

    print(
        f"\nDone: {ok}/{len(core_scene_dirs)} core scene(s) produced scenes, "
        f"{created_total} roundabout variant(s) written."
    )
    if ok < len(core_scene_dirs):
        sys.exit(1)


if __name__ == "__main__":
    main()
