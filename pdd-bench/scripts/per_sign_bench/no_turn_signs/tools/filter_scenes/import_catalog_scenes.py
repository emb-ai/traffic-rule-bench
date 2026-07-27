#!/usr/bin/env python3
"""Import direction-sign catalog scenes into scenes/<slug>/core with junction filtering.

Scans the catalog (default: pdd-bench/scenes/<pdd_code>), keeps only scenes with a
valid 3- and/or 4-arm junction (each arm has a lane longer than --min-lane-length),
then copies them into ``direction_signs/scenes/<4_1_x>/core/``, renders custom.png,
and optionally runs a simulation GIF.

Use crop_junction_scene.py afterward to emit junction crops as siblings under
``scenes/<4_1_x>/`` (e.g. ``scenes/4_1_2/sign_72424_j0``).

Examples:
    python tools/filter_scenes/import_catalog_scenes.py --limit 10
    python tools/filter_scenes/import_catalog_scenes.py --pdd-code 4.1.2 --limit 10
    python tools/filter_scenes/import_catalog_scenes.py --limit 10 --arms 4 3
    python tools/filter_scenes/import_catalog_scenes.py sign_79054 75605
    python tools/filter_scenes/import_catalog_scenes.py --sign-ids 79054 75605
    python tools/filter_scenes/import_catalog_scenes.py sign_79054 --no-simulation
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

FILTER_SCENES_DIR = Path(__file__).resolve().parent
TOOLS_DIR = FILTER_SCENES_DIR.parent
NO_TURN_SIGNS_DIR = TOOLS_DIR.parent
PDD_BENCH_DIR = NO_TURN_SIGNS_DIR.parent.parent.parent
SCENES_BASE_DEFAULT = NO_TURN_SIGNS_DIR / "scenes"
DEFAULT_SOURCE = PDD_BENCH_DIR / "scenes" / "3.18.1"

sys.path.insert(0, str(NO_TURN_SIGNS_DIR))

from lib.junction_crop import (  # noqa: E402
    try_find_junction_for_arm_counts,
    try_find_junction_with_arm_count,
)
from lib.no_turn_sign_spec import (  # noqa: E402
    DEFAULT_PDD_CODE,
    get_no_turn_sign_spec,
    local_core_scenes_root,
)
from lib.sumo_utils import load_scene_meta, resolve_net_file  # noqa: E402
from tools.render_map import parse_sumo_net, render_network  # noqa: E402


@dataclass
class SceneAnalysis:
    scene_name: str
    matched: bool
    arm_count: Optional[int] = None
    junction_id: Optional[str] = None
    total_lanes: Optional[int] = None
    incoming_edge_ids: Optional[list[str]] = None
    reason: Optional[str] = None


def _scene_name_from_sign_id(sign_id: int | str) -> str:
    return f"sign_{int(sign_id)}"


def arm_counts_label(arm_counts: tuple[int, ...]) -> str:
    unique = sorted(set(arm_counts))
    if len(unique) == 1:
        return f"{unique[0]}-arm"
    return "/".join(f"{n}-arm" for n in unique)


def parse_arm_counts(raw: list[int]) -> tuple[int, ...]:
    allowed = {3, 4}
    counts = sorted({int(n) for n in raw})
    invalid = [n for n in counts if n not in allowed]
    if invalid:
        raise SystemExit(f"Unsupported --arms value(s): {invalid}; allowed: 3, 4")
    if not counts:
        raise SystemExit("At least one --arms value is required (3 and/or 4)")
    return tuple(counts)


def discover_source_scenes(source_dir: Path) -> list[Path]:
    """Return sorted scene directories that contain meta.json + a net.xml."""
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


def existing_dest_scene_names(dest_root: Path) -> set[str]:
    """Scene folder names already present under the destination root."""
    if not dest_root.is_dir():
        return set()
    return {
        entry.name
        for entry in dest_root.iterdir()
        if entry.is_dir() and (entry / "meta.json").is_file()
    }


def analyze_scene_strict(
    scene_dir: Path,
    *,
    require_arm_count: int,
    min_lane_length_m: float,
) -> SceneAnalysis:
    """Match only when the scene has a qualifying junction with exactly ``require_arm_count`` arms."""
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

    pick = try_find_junction_with_arm_count(
        net_path,
        arm_count=require_arm_count,
        min_lane_length_m=min_lane_length_m,
    )
    if pick is None:
        return SceneAnalysis(
            scene_name=scene_name,
            matched=False,
            reason=f"no {require_arm_count}-arm junction with all arms longer than min lane length",
        )

    return SceneAnalysis(
        scene_name=scene_name,
        matched=True,
        arm_count=pick.arm_count,
        junction_id=pick.junction_id,
        total_lanes=pick.total_lanes,
        incoming_edge_ids=list(pick.incoming_edge_ids),
    )


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
        label = arm_counts_label(arm_counts)
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


def import_arm_pass_order(arm_counts: tuple[int, ...]) -> tuple[int, ...]:
    """Order arm-count passes: 4-arm catalog sweep before 3-arm when both are requested."""
    return tuple(sorted(set(arm_counts), reverse=True))


def resolve_import_candidates_bulk(
    available: dict[str, Path],
    candidate_names: list[str],
    *,
    limit: int | None,
    arm_counts: tuple[int, ...],
    min_lane_length_m: float,
) -> list[SceneAnalysis]:
    """Scan the full catalog: all qualifying 4-arm scenes first, then 3-arm."""
    matched: list[SceneAnalysis] = []
    picked: set[str] = set()
    arm_passes = import_arm_pass_order(arm_counts)

    for arm_count in arm_passes:
        if limit is not None and len(matched) >= limit:
            break
        print(f"  pass: {arm_count}-arm junctions")
        for name in candidate_names:
            if name in picked:
                continue
            analysis = analyze_scene_strict(
                available[name],
                require_arm_count=arm_count,
                min_lane_length_m=min_lane_length_m,
            )
            if not analysis.matched:
                continue
            matched.append(analysis)
            picked.add(name)
            if limit is not None and len(matched) >= limit:
                break

    for name in candidate_names:
        if name in picked:
            continue
        analysis = analyze_scene(
            available[name],
            arm_counts=arm_counts,
            min_lane_length_m=min_lane_length_m,
        )
        if not analysis.matched:
            print(f"  [skip]  {name}: {analysis.reason}")

    return matched


def resolve_import_candidates(
    source_dir: Path,
    dest_root: Path,
    names: list[str],
    sign_ids: list[int],
    limit: int | None,
    *,
    skip_existing: bool,
    arm_counts: tuple[int, ...],
    min_lane_length_m: float,
    junction_filter: bool,
) -> list[SceneAnalysis]:
    available = {p.name: p for p in discover_source_scenes(source_dir)}
    if not available:
        raise SystemExit(f"No valid scenes found under {source_dir}")

    requested: list[str] = []
    for raw in names:
        name = raw.strip()
        if name.isdigit():
            name = _scene_name_from_sign_id(int(name))
        if name not in available:
            raise SystemExit(f"Scene not found in catalog: {name!r} (source: {source_dir})")
        requested.append(name)

    for sign_id in sign_ids:
        name = _scene_name_from_sign_id(sign_id)
        if name not in available:
            raise SystemExit(f"Sign id {sign_id} not found in catalog ({name!r})")
        requested.append(name)

    if requested:
        candidate_names = requested
    else:
        candidate_names = sorted(available)

    if skip_existing:
        already = existing_dest_scene_names(dest_root)
        candidate_names = [name for name in candidate_names if name not in already]

    if not junction_filter:
        matched = [SceneAnalysis(scene_name=name, matched=True) for name in candidate_names]
        if limit is not None and not requested:
            matched = matched[:limit]
        if requested and limit is not None:
            matched = matched[:limit]
        return matched

    if not requested and set(arm_counts) >= {3, 4}:
        return resolve_import_candidates_bulk(
            available,
            candidate_names,
            limit=limit,
            arm_counts=arm_counts,
            min_lane_length_m=min_lane_length_m,
        )

    matched: list[SceneAnalysis] = []

    for name in candidate_names:
        scene_dir = available[name]
        analysis = analyze_scene(
            scene_dir,
            arm_counts=arm_counts,
            min_lane_length_m=min_lane_length_m,
        )
        if not analysis.matched:
            print(f"  [skip]  {name}: {analysis.reason}")
            continue

        matched.append(analysis)
        if limit is not None and not requested and len(matched) >= limit:
            break

    if requested and limit is not None:
        matched = matched[:limit]
    return matched


def normalize_meta(meta: dict, scene_name: str, analysis: SceneAnalysis | None = None) -> dict:
    """Ensure direction_signs-compatible meta fields."""
    out = dict(meta)
    out["scene_name"] = scene_name
    out["scene_kind"] = "core"
    if out.get("sign_type") and not out.get("pdd_code"):
        out["pdd_code"] = out["sign_type"]
    if analysis is not None and analysis.matched:
        out["catalog_junction_id"] = analysis.junction_id
        out["catalog_junction_arm_count"] = analysis.arm_count
        out["catalog_junction_total_lanes"] = analysis.total_lanes
    return out


def copy_scene(
    source_dir: Path,
    dest_root: Path,
    scene_name: str,
    *,
    overwrite: bool,
    analysis: SceneAnalysis | None = None,
) -> Path:
    src = source_dir / scene_name
    dst = dest_root / scene_name
    if dst.exists():
        if not overwrite:
            print(f"  [skip copy] already exists: {dst}")
            return dst
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    meta_path = dst / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = normalize_meta(meta, scene_name, analysis)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"  copied -> {dst}")
    return dst


def render_scene_preview(scene_dir: Path, *, dpi: int, figsize: float) -> Path:
    meta = load_scene_meta(scene_dir)
    net_file = resolve_net_file(scene_dir, meta)
    net_path = scene_dir / net_file
    out_path = scene_dir / "custom.png"

    edges, junctions = parse_sumo_net(net_path)
    render_network(edges, junctions, out_path, figsize=(figsize, figsize), dpi=dpi)
    return out_path


def run_simulation(
    scene_name: str,
    *,
    policy: str,
    max_steps: int,
    model_path: str | None,
    plant2_action_mode: str,
) -> None:
    cmd = [
        sys.executable,
        str(TOOLS_DIR / "run_simulation.py"),
        scene_name,
        "--policy",
        policy,
        "--max-steps",
        str(max_steps),
        "--plant2-action-mode",
        plant2_action_mode,
    ]
    if model_path:
        cmd += ["--model-path", model_path]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(NO_TURN_SIGNS_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import catalog scenes with 3/4-arm junction filter into scenes/core",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "scenes",
        nargs="*",
        help="Scene folder names (e.g. sign_79054) or numeric sign ids",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=f"Catalog root (default: pdd-bench/scenes/<pdd_code>, initially {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--pdd-code",
        type=str,
        default=DEFAULT_PDD_CODE,
        help="No-turn sign (3.18.1 / 3.18.2); sets default --source catalog",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination core-scenes root "
        f"(default: scenes/<slug>/core under {SCENES_BASE_DEFAULT})",
    )
    parser.add_argument(
        "--scenes-base",
        type=Path,
        default=SCENES_BASE_DEFAULT,
        help=f"Parent of per-sign scene folders (default: {SCENES_BASE_DEFAULT})",
    )
    parser.add_argument(
        "--sign-ids",
        type=int,
        nargs="*",
        default=[],
        help="Import by numeric sign id (e.g. 79054 -> sign_79054)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="When no scenes are named, import up to N qualifying catalog scenes "
        "not already present in --dest (unless --overwrite)",
    )
    parser.add_argument(
        "--arms",
        type=int,
        nargs="+",
        default=[4, 3],
        metavar="N",
        help="Junction arm count(s) to require: 3 (T), 4 (X), or both (default: 4 3)",
    )
    parser.add_argument(
        "--min-lane-length",
        type=float,
        default=0.5,
        help="Each arm must have a lane longer than this (default: 0.5 m; OSM stubs are often short)",
    )
    parser.add_argument(
        "--no-junction-filter",
        action="store_true",
        help="Import without checking for 3/4-arm junctions",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination scene folders",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip static map preview (custom.png)",
    )
    parser.add_argument(
        "--run-simulation",
        action="store_true",
        help="Run run_simulation.py and save simulation-<policy>.gif",
    )
    parser.add_argument(
        "--no-simulation",
        action="store_true",
        help="Alias for not passing --run-simulation (default)",
    )
    parser.add_argument("--policy", default="idm", help="Policy for simulation (default: idm)")
    parser.add_argument("--model-path", default=None, help="Checkpoint for NN policies")
    parser.add_argument("--max-steps", type=int, default=400, help="Simulation steps (default: 400)")
    parser.add_argument(
        "--plant2-action-mode",
        default="pid",
        choices=["pid", "wps_pure_pursuit"],
        help="PLANT2 action mode",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Preview image DPI (default: 150)")
    parser.add_argument("--figsize", type=float, default=12.0, help="Preview figure size")
    args = parser.parse_args()

    arm_counts = parse_arm_counts(args.arms)
    junction_filter = not args.no_junction_filter
    sign_spec = get_no_turn_sign_spec(args.pdd_code)
    if args.source is None:
        source_dir = (PDD_BENCH_DIR / "scenes" / sign_spec.catalog_subdir).resolve()
    else:
        source_dir = args.source.expanduser().resolve()
    scenes_base = args.scenes_base.expanduser().resolve()
    if args.dest is None:
        dest_root = local_core_scenes_root(scenes_base, sign_spec.pdd_code).resolve()
    else:
        dest_root = args.dest.expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    if not source_dir.is_dir():
        sys.exit(f"Source catalog not found: {source_dir}")

    run_sim = args.run_simulation and not args.no_simulation

    print(f"Sign:   {sign_spec.pdd_code} ({sign_spec.title})")
    print(f"Source: {source_dir}")
    print(f"Dest:   {dest_root}")
    if junction_filter:
        print(f"Arms:   {arm_counts_label(arm_counts)} (min lane {args.min_lane_length} m)")
    else:
        print("Filter: junction check disabled")
    print(f"Render previews: {not args.no_render}")
    print(f"Run simulation:  {run_sim}")
    print("Scanning catalog...")

    to_import = resolve_import_candidates(
        source_dir,
        dest_root,
        list(args.scenes),
        list(args.sign_ids),
        args.limit,
        skip_existing=not args.overwrite,
        arm_counts=arm_counts,
        min_lane_length_m=args.min_lane_length,
        junction_filter=junction_filter,
    )

    if not to_import:
        already = len(existing_dest_scene_names(dest_root))
        filter_note = (
            f" with {arm_counts_label(arm_counts)} junction"
            if junction_filter
            else ""
        )
        sys.exit(
            "No scenes to import. Pass scene names, --sign-ids, or --limit N.\n"
            f"Catalog: {len(discover_source_scenes(source_dir))} scene(s), "
            f"already in dest: {already}{filter_note}.\n"
            f"Available: {', '.join(p.name for p in discover_source_scenes(source_dir)[:8])}..."
        )

    print(f"Import: {', '.join(a.scene_name for a in to_import)}")

    imported = 0
    for analysis in to_import:
        scene_name = analysis.scene_name
        print(f"\n=== {scene_name} ===")
        if junction_filter and analysis.matched:
            print(
                f"  junction {analysis.junction_id} ({analysis.arm_count}-arm), "
                f"{analysis.total_lanes} lane(s)"
            )

        scene_dir = copy_scene(
            source_dir,
            dest_root,
            scene_name,
            overwrite=args.overwrite,
            analysis=analysis if junction_filter else None,
        )
        imported += 1

        if not args.no_render:
            try:
                preview = render_scene_preview(
                    scene_dir,
                    dpi=args.dpi,
                    figsize=args.figsize,
                )
                print(f"  preview: {preview}")
            except Exception as exc:
                print(f"  [render failed] {exc}")

        if run_sim:
            try:
                run_simulation(
                    scene_name,
                    policy=args.policy,
                    max_steps=args.max_steps,
                    model_path=args.model_path,
                    plant2_action_mode=args.plant2_action_mode,
                )
            except subprocess.CalledProcessError as exc:
                print(f"  [simulation failed] exit code {exc.returncode}")
            except Exception as exc:
                print(f"  [simulation failed] {exc}")

    print(f"\nDone: imported {imported} scene(s).")


if __name__ == "__main__":
    main()
