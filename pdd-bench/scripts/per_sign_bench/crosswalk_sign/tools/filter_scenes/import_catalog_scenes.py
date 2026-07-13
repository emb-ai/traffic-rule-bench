#!/usr/bin/env python3
"""Import pedestrian-crossing catalog scenes into crosswalk_sign/scenes/core.

Scans the catalog (default: pdd-bench/scenes/5.19), keeps only scenes whose SUMO
net contains at least one pedestrian crossing edge, then copies them into
crosswalk_sign/scenes/ and renders custom.png.

Examples:
    python tools/filter_scenes/import_catalog_scenes.py --limit 20
    python tools/filter_scenes/import_catalog_scenes.py --source ../../../scenes/5.16 --limit 5
    python tools/filter_scenes/import_catalog_scenes.py sign_71904 --no-simulation
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

FILTER_SCENES_DIR = Path(__file__).resolve().parent
CROSSWALK_SIGN_DIR = FILTER_SCENES_DIR.parent.parent
PDD_BENCH_DIR = CROSSWALK_SIGN_DIR.parent.parent.parent
DEFAULT_SOURCE = PDD_BENCH_DIR / "scenes" / "5.19"
DEFAULT_DEST = CROSSWALK_SIGN_DIR / "scenes" / "core"

sys.path.insert(0, str(CROSSWALK_SIGN_DIR))

from lib.crosswalk_layout import net_has_crossings  # noqa: E402
from lib.sumo_utils import load_scene_meta, resolve_net_file  # noqa: E402
from tools.render_map import parse_sumo_net, render_network  # noqa: E402


@dataclass
class SceneAnalysis:
    scene_name: str
    matched: bool
    crossing_count: int = 0
    reason: str | None = None


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
    meta = load_scene_meta(scene_dir)
    net_file = resolve_net_file(scene_dir, meta)
    net_path = scene_dir / net_file
    if not net_path.is_file():
        return SceneAnalysis(scene_dir.name, matched=False, reason="missing net.xml")

    if not net_has_crossings(net_path):
        return SceneAnalysis(scene_dir.name, matched=False, reason="no SUMO crossings")

    import xml.etree.ElementTree as ET

    root = ET.parse(net_path).getroot()
    crossing_count = sum(
        1 for edge in root.findall("edge") if edge.get("function") == "crossing"
    )
    return SceneAnalysis(scene_dir.name, matched=True, crossing_count=crossing_count)


def copy_scene(source_dir: Path, dest_root: Path) -> Path:
    dest_root.mkdir(parents=True, exist_ok=True)
    dest_dir = dest_root / source_dir.name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)
    return dest_dir


def render_custom_png(scene_dir: Path) -> None:
    meta = load_scene_meta(scene_dir)
    net_file = resolve_net_file(scene_dir, meta)
    net_path = scene_dir / net_file
    edges, junctions = parse_sumo_net(net_path)
    out_path = scene_dir / "custom.png"
    render_network(edges, junctions, out_path)


def run_simulation_gif(scene_dir: Path) -> None:
    sim_script = CROSSWALK_SIGN_DIR / "tools" / "run_simulation.py"
    if not sim_script.is_file():
        return
    subprocess.run(
        [sys.executable, str(sim_script), str(scene_dir), "--gif"],
        cwd=str(CROSSWALK_SIGN_DIR),
        check=False,
    )


def import_scenes(
    source_dir: Path,
    dest_dir: Path,
    *,
    limit: int | None,
    names: list[str],
    sign_ids: list[int],
    skip_existing: bool,
    no_simulation: bool,
) -> tuple[int, int]:
    if names:
        candidates = [source_dir / n for n in names]
    elif sign_ids:
        candidates = [source_dir / f"sign_{sid}" for sid in sign_ids]
    else:
        candidates = discover_source_scenes(source_dir)

    existing = existing_dest_scene_names(dest_dir) if skip_existing else set()
    imported = 0
    skipped = 0

    for scene_dir in candidates:
        if not scene_dir.is_dir():
            print(f"[skip] missing: {scene_dir}")
            continue
        if skip_existing and scene_dir.name in existing:
            print(f"[skip] already imported: {scene_dir.name}")
            skipped += 1
            continue

        analysis = analyze_scene(scene_dir)
        if not analysis.matched:
            print(f"[skip] {scene_dir.name}: {analysis.reason}")
            skipped += 1
            continue

        dest = copy_scene(scene_dir, dest_dir)
        try:
            render_custom_png(dest)
        except Exception as exc:
            print(f"[warn] render failed for {dest.name}: {exc}")

        meta_path = dest / "meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["pdd_code"] = "5.19"
            meta["sign_type"] = "crosswalk"
            meta["scene_name"] = dest.name
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        print(
            f"[import] {dest.name} ({analysis.crossing_count} crossing edge(s))"
        )
        imported += 1

        if not no_simulation:
            run_simulation_gif(dest)

        if limit is not None and imported >= limit:
            break

    return imported, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Import 5.19 crosswalk catalog scenes")
    parser.add_argument("names", nargs="*", help="Specific scene folder names")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sign-ids", type=int, nargs="*")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--no-simulation", action="store_true")
    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"Catalog not found: {args.source}")
        print("Create pdd-bench/scenes/5.19 or pass --source to another catalog with crossings.")
        sys.exit(1)

    imported, skipped = import_scenes(
        args.source,
        args.dest,
        limit=args.limit,
        names=list(args.names),
        sign_ids=list(args.sign_ids or []),
        skip_existing=not args.no_skip_existing,
        no_simulation=args.no_simulation,
    )
    print(f"\nDone: imported={imported}, skipped={skipped}, dest={args.dest}")


if __name__ == "__main__":
    main()
