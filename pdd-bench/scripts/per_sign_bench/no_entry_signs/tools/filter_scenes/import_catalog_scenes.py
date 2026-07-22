#!/usr/bin/env python3
"""Import no-entry catalog scenes into scenes/<slug>/ for the 3.1 / 3.2 bench.

Catalog scenes under ``pdd-bench/scenes/3.1`` (and ``3.2``) are already local
OSM extracts around each sign. This tool copies them into the bench folder,
keeps catalog ``road_id`` / ``distance_from_start`` for exact sign placement,
filters out geometries that cannot host spawn-before / drive-past, and
optionally renders ``custom.png``.

Examples:
    python tools/filter_scenes/import_catalog_scenes.py --limit 10
    python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.2 --limit 20
    python tools/filter_scenes/import_catalog_scenes.py --sign-ids 82059 71972
    python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.1 --no-simulation
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
NO_ENTRY_SIGNS_DIR = TOOLS_DIR.parent
PDD_BENCH_DIR = NO_ENTRY_SIGNS_DIR.parent.parent.parent
SCENES_BASE_DEFAULT = NO_ENTRY_SIGNS_DIR / "scenes"

sys.path.insert(0, str(NO_ENTRY_SIGNS_DIR))

from lib.no_entry_route import scene_geometry_ok  # noqa: E402
from lib.no_entry_sign_spec import (  # noqa: E402
    DEFAULT_PDD_CODE,
    get_no_entry_sign_spec,
    local_scenes_root,
)
from lib.sumo_utils import load_scene_meta, resolve_net_file  # noqa: E402
from tools.render_map import parse_sumo_net, render_network  # noqa: E402


@dataclass
class SceneAnalysis:
    scene_name: str
    matched: bool
    reason: Optional[str] = None
    distance_from_start: Optional[float] = None
    road_id: Optional[str] = None


def _scene_name_from_sign_id(sign_id: int | str) -> str:
    return f"sign_{int(sign_id)}"


def discover_source_scenes(source_dir: Path) -> list[Path]:
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
    if not dest_root.is_dir():
        return set()
    return {
        entry.name
        for entry in dest_root.iterdir()
        if entry.is_dir() and (entry / "meta.json").is_file()
    }


def analyze_scene(scene_dir: Path) -> SceneAnalysis:
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

    road_id = str(meta.get("road_id") or "").strip()
    try:
        distance = float(meta.get("distance_from_start", 0.0) or 0.0)
    except (TypeError, ValueError):
        distance = 0.0

    ok, reason = scene_geometry_ok(net_path, road_id, distance)
    if not ok:
        return SceneAnalysis(
            scene_name=scene_name,
            matched=False,
            reason=reason,
            distance_from_start=distance,
            road_id=road_id or None,
        )
    return SceneAnalysis(
        scene_name=scene_name,
        matched=True,
        reason="ok",
        distance_from_start=distance,
        road_id=road_id,
    )


def resolve_import_candidates(
    source_dir: Path,
    dest_root: Path,
    names: list[str],
    sign_ids: list[int],
    limit: int | None,
    *,
    skip_existing: bool,
    geometry_filter: bool,
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

    candidate_names = requested if requested else sorted(available)

    if skip_existing:
        already = existing_dest_scene_names(dest_root)
        candidate_names = [name for name in candidate_names if name not in already]

    matched: list[SceneAnalysis] = []
    for name in candidate_names:
        if geometry_filter:
            analysis = analyze_scene(available[name])
            if not analysis.matched:
                print(f"  [skip]  {name}: {analysis.reason}")
                continue
        else:
            analysis = SceneAnalysis(scene_name=name, matched=True, reason="no geometry filter")
        matched.append(analysis)
        if limit is not None and not requested and len(matched) >= limit:
            break

    if requested and limit is not None:
        matched = matched[:limit]
    return matched


def normalize_meta(meta: dict, scene_name: str, analysis: SceneAnalysis | None = None) -> dict:
    """Preserve catalog placement fields; mark as a bench catalog scene."""
    out = dict(meta)
    out["scene_name"] = scene_name
    out["scene_kind"] = "catalog"
    if out.get("sign_type") and not out.get("pdd_code"):
        out["pdd_code"] = out["sign_type"]
    # Exact map placement: keep distance_from_start; mirror into sign_spawn_distance.
    if out.get("distance_from_start") is not None:
        try:
            dist = float(out["distance_from_start"])
            out["distance_from_start"] = dist
            out["sign_spawn_distance"] = dist
        except (TypeError, ValueError):
            pass
    if analysis is not None and analysis.road_id:
        out.setdefault("road_id", analysis.road_id)
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
    net_path = scene_dir / resolve_net_file(scene_dir, meta)
    edges, junctions = parse_sumo_net(net_path)
    out_path = scene_dir / "custom.png"
    render_network(edges, junctions, out_path, figsize=(figsize, figsize), dpi=dpi)
    # Review UI prefers custom_cropped.png when present.
    cropped = scene_dir / "custom_cropped.png"
    if not cropped.exists():
        shutil.copy2(out_path, cropped)
    return out_path


def maybe_run_simulation(scene_dir: Path, *, pdd_code: str) -> None:
    sim = NO_ENTRY_SIGNS_DIR / "tools" / "run_simulation.py"
    if not sim.is_file():
        return
    cmd = [
        sys.executable,
        str(sim),
        "--scene-dir",
        str(scene_dir),
        "--pdd-code",
        pdd_code,
    ]
    print(f"  simulation: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="*", help="Optional scene folder names or numeric ids")
    parser.add_argument("--sign-ids", nargs="+", type=int, default=[])
    parser.add_argument("--pdd-code", default=DEFAULT_PDD_CODE, help="3.1 or 3.2")
    parser.add_argument("--source", type=Path, default=None, help="Override catalog root")
    parser.add_argument("--dest", type=Path, default=None, help="Override destination root")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument(
        "--no-geometry-filter",
        action="store_true",
        help="Copy even if spawn-before / drive-past geometry checks fail",
    )
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--run-simulation", action="store_true", default=False)
    parser.add_argument("--no-simulation", action="store_true")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--figsize", type=float, default=8.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = get_no_entry_sign_spec(args.pdd_code)
    source_dir = Path(args.source) if args.source else (PDD_BENCH_DIR / "scenes" / spec.catalog_subdir)
    dest_root = Path(args.dest) if args.dest else local_scenes_root(SCENES_BASE_DEFAULT, spec.pdd_code)
    dest_root.mkdir(parents=True, exist_ok=True)

    print(f"PDD {spec.pdd_code} ({spec.title})")
    print(f"source: {source_dir}")
    print(f"dest:   {dest_root}")

    matched = resolve_import_candidates(
        source_dir,
        dest_root,
        list(args.scenes),
        list(args.sign_ids),
        args.limit,
        skip_existing=args.skip_existing and not args.overwrite,
        geometry_filter=not args.no_geometry_filter,
    )
    if not matched:
        print("No scenes to import.")
        return 1

    print(f"importing {len(matched)} scene(s)…")
    for analysis in matched:
        dst = copy_scene(
            source_dir,
            dest_root,
            analysis.scene_name,
            overwrite=args.overwrite,
            analysis=analysis,
        )
        if not args.no_render:
            try:
                render_scene_preview(dst, dpi=args.dpi, figsize=args.figsize)
                print(f"  preview -> {dst / 'custom.png'}")
            except Exception as exc:
                print(f"  [warn] render failed for {analysis.scene_name}: {exc}")
        if args.run_simulation and not args.no_simulation:
            maybe_run_simulation(dst, pdd_code=spec.pdd_code)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
