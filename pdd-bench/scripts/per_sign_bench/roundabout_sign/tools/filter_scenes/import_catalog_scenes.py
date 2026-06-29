#!/usr/bin/env python3
"""Import roundabout (4.3) catalog scenes into roundabout_sign/scenes/core.

Scans pdd-bench/scenes/4.3, keeps only scenes whose SUMO net has a ``<roundabout>``
block reachable from the catalog sign road, then copies into scenes/core/.

Use crop_junction_scene.py afterward to emit cropped roundabout scenes
(e.g. sign_77277_rb_s00).

Examples:
    python tools/filter_scenes/import_catalog_scenes.py --limit 10
    python tools/filter_scenes/import_catalog_scenes.py sign_77277
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
ROUNDABOUT_SIGN_DIR = TOOLS_DIR.parent
PDD_BENCH_DIR = ROUNDABOUT_SIGN_DIR.parent.parent.parent
DEFAULT_SOURCE = PDD_BENCH_DIR / "scenes" / "4.3"
DEFAULT_DEST = ROUNDABOUT_SIGN_DIR / "scenes" / "core"

sys.path.insert(0, str(ROUNDABOUT_SIGN_DIR))

from lib.roundabout_topology import try_detect_roundabout, resolve_sumo_roundabout  # noqa: E402
from lib.roundabout_fingerprint import (  # noqa: E402
    RoundaboutFingerprintRegistry,
    fingerprint_from_sumo_roundabout,
    sumo_roundabout_record,
)
from lib.sumo_utils import load_scene_meta, resolve_net_file  # noqa: E402
from tools.render_map import parse_sumo_net, render_network  # noqa: E402


@dataclass
class SceneAnalysis:
    scene_name: str
    matched: bool
    entry_junction_id: Optional[str] = None
    ring_edge_count: Optional[int] = None
    spoke_edge_count: Optional[int] = None
    approach_edge_id: Optional[str] = None
    sumo_roundabout_fingerprint: Optional[str] = None
    reason: Optional[str] = None


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


def analyze_scene_roundabout(scene_dir: Path) -> SceneAnalysis:
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

    sign_edge = meta.get("road_id")
    pick = try_detect_roundabout(net_path, sign_edge_id=sign_edge)
    if pick is None:
        return SceneAnalysis(
            scene_name=scene_name,
            matched=False,
            reason="no SUMO <roundabout> reachable from catalog sign road",
        )

    try:
        rb = resolve_sumo_roundabout(net_path, sign_edge_id=sign_edge)
        fingerprint = fingerprint_from_sumo_roundabout(rb)
    except Exception:
        fingerprint = None

    return SceneAnalysis(
        scene_name=scene_name,
        matched=True,
        entry_junction_id=pick.entry_junction_id,
        ring_edge_count=pick.ring_edge_count,
        spoke_edge_count=len(pick.spoke_edge_ids),
        approach_edge_id=pick.approach_edge_id or sign_edge,
        sumo_roundabout_fingerprint=fingerprint,
    )


def normalize_meta(meta: dict, scene_name: str, analysis: SceneAnalysis | None = None) -> dict:
    out = dict(meta)
    out["scene_name"] = scene_name
    out["scene_kind"] = "core"
    out["pdd_code"] = out.get("pdd_code") or out.get("sign_type") or "4.3"
    out["sign_type"] = out.get("sign_type") or "4.3"
    if analysis is not None and analysis.matched:
        out["catalog_roundabout_entry"] = analysis.entry_junction_id
        out["catalog_roundabout_ring_edges"] = analysis.ring_edge_count
        out["catalog_roundabout_spokes"] = analysis.spoke_edge_count
        if analysis.approach_edge_id:
            out["catalog_roundabout_approach_edge"] = analysis.approach_edge_id
        if analysis.sumo_roundabout_fingerprint:
            out["sumo_roundabout_fingerprint"] = analysis.sumo_roundabout_fingerprint
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
    if analysis is not None and analysis.matched and analysis.sumo_roundabout_fingerprint:
        try:
            rb = resolve_sumo_roundabout(dst / resolve_net_file(dst, meta), sign_edge_id=meta.get("road_id"))
            meta.update(sumo_roundabout_record(rb))
        except Exception:
            pass
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
    subprocess.run(cmd, check=True, cwd=str(ROUNDABOUT_SIGN_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Import 4.3 catalog scenes with roundabout filter")
    parser.add_argument("scenes", nargs="*", help="Scene names or numeric sign ids")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--sign-ids", type=int, nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-duplicate-roundabout",
        action="store_true",
        help="Import even when scenes/roundabout_fingerprints.json already has this ring",
    )
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--run-simulation", action="store_true")
    parser.add_argument("--policy", default="idm")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--plant2-action-mode", default="pid", choices=["pid", "wps_pure_pursuit"])
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--figsize", type=float, default=12.0)
    args = parser.parse_args()

    source_dir = args.source.expanduser().resolve()
    dest_root = args.dest.expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    if not source_dir.is_dir():
        sys.exit(f"Source catalog not found: {source_dir}")

    available = {p.name: p for p in discover_source_scenes(source_dir)}
    if not available:
        sys.exit(f"No valid scenes under {source_dir}")

    requested: list[str] = []
    for raw in args.scenes:
        name = raw.strip()
        if name.isdigit():
            name = _scene_name_from_sign_id(int(name))
        if name not in available:
            sys.exit(f"Scene not found: {name!r}")
        requested.append(name)
    for sign_id in args.sign_ids:
        name = _scene_name_from_sign_id(sign_id)
        if name not in available:
            sys.exit(f"Sign id {sign_id} not found")
        requested.append(name)

    candidate_names = requested or sorted(available)
    if not args.overwrite:
        already = existing_dest_scene_names(dest_root)
        candidate_names = [n for n in candidate_names if n not in already]

    to_import: list[SceneAnalysis] = []
    scenes_root = dest_root.parent
    registry = RoundaboutFingerprintRegistry.for_scenes_root(scenes_root)
    skip_duplicate = not args.allow_duplicate_roundabout

    for name in candidate_names:
        analysis = analyze_scene_roundabout(available[name])
        if not analysis.matched:
            print(f"  [skip]  {name}: {analysis.reason}")
            continue
        if skip_duplicate and analysis.sumo_roundabout_fingerprint:
            duplicate = registry.duplicate_owner(
                analysis.sumo_roundabout_fingerprint,
                scene_name=name,
            )
            if duplicate is not None:
                print(
                    f"  [skip duplicate roundabout] {name}: "
                    f"same ring as {duplicate.get('scene_name')!r}"
                )
                continue
        to_import.append(analysis)
        if args.limit and not requested and len(to_import) >= args.limit:
            break

    if not to_import:
        sys.exit("No scenes to import.")

    print(f"Import: {', '.join(a.scene_name for a in to_import)}")
    imported = 0
    for analysis in to_import:
        scene_name = analysis.scene_name
        print(f"\n=== {scene_name} ===")
        if analysis.matched and analysis.entry_junction_id:
            print(
                f"  roundabout entry {analysis.entry_junction_id}, "
                f"{analysis.ring_edge_count} ring edge(s), "
                f"{analysis.spoke_edge_count} spoke(s)"
            )
        scene_dir = copy_scene(
            source_dir,
            dest_root,
            scene_name,
            overwrite=args.overwrite,
            analysis=analysis,
        )
        if analysis.sumo_roundabout_fingerprint:
            try:
                scene_meta = load_scene_meta(scene_dir)
                rb = resolve_sumo_roundabout(
                    scene_dir / resolve_net_file(scene_dir, scene_meta),
                    sign_edge_id=scene_meta.get("road_id"),
                )
                registry.upsert(
                    analysis.sumo_roundabout_fingerprint,
                    scene_name=scene_name,
                    core_scene_name=scene_name,
                    kind="core",
                    sign_id=scene_meta.get("sign_id"),
                    sumo_roundabout_nodes=rb.node_ids,
                    sumo_roundabout_ring_edges=rb.ring_edge_ids,
                )
                registry.save()
            except Exception as exc:
                print(f"  [fingerprint registry] {exc}")
        imported += 1
        if not args.no_render:
            try:
                preview = render_scene_preview(scene_dir, dpi=args.dpi, figsize=args.figsize)
                print(f"  preview: {preview}")
            except Exception as exc:
                print(f"  [render failed] {exc}")
        if args.run_simulation:
            try:
                run_simulation(
                    scene_name,
                    policy=args.policy,
                    max_steps=args.max_steps,
                    model_path=args.model_path,
                    plant2_action_mode=args.plant2_action_mode,
                )
            except Exception as exc:
                print(f"  [simulation failed] {exc}")

    print(f"\nDone: imported {imported} scene(s).")


if __name__ == "__main__":
    main()
