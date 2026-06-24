#!/usr/bin/env python3
"""Copy yield-sign (2.4) catalog scenes into yield_sign/scenes and preview them.

Source layout (pdd-bench/scenes/2.4/<scene>/):
  meta.json, <scene>.net.xml

For each imported scene this script:
  1. Copies the folder into yield_sign/scenes/
  2. Normalizes meta.json (adds scene_name)
  3. Renders a static map preview (custom.png)
  4. Optionally runs a short simulation GIF (simulation-<policy>.gif)

Examples:
    # Import two scenes by folder name
    python tools/filter_scenes/import_catalog_scenes.py sign_79054 sign_75605

    # Import next 5 catalog scenes not yet present in yield_sign/scenes/
    python tools/filter_scenes/import_catalog_scenes.py --limit 5

    # Import by numeric sign id
    python tools/filter_scenes/import_catalog_scenes.py --sign-ids 79054 75605

    # Preview only, no simulation
    python tools/filter_scenes/import_catalog_scenes.py sign_79054 --no-simulation

    # Import and run IDM simulation GIF
    python tools/filter_scenes/import_catalog_scenes.py sign_79054 --run-simulation --policy idm
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

FILTER_SCENES_DIR = Path(__file__).resolve().parent
TOOLS_DIR = FILTER_SCENES_DIR.parent
YIELD_SIGN_DIR = TOOLS_DIR.parent
PDD_BENCH_DIR = YIELD_SIGN_DIR.parent.parent.parent
DEFAULT_SOURCE = PDD_BENCH_DIR / "scenes" / "2.4_three_arm"
DEFAULT_DEST = YIELD_SIGN_DIR / "scenes"

sys.path.insert(0, str(YIELD_SIGN_DIR))

from lib.sumo_utils import load_scene_meta, resolve_net_file  # noqa: E402
from tools.render_map import parse_sumo_net, render_network  # noqa: E402


def _scene_name_from_sign_id(sign_id: int | str) -> str:
    return f"sign_{int(sign_id)}"


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


def resolve_scene_names(
    source_dir: Path,
    dest_root: Path,
    names: list[str],
    sign_ids: list[int],
    limit: int | None,
    *,
    skip_existing: bool,
) -> list[str]:
    available = {p.name for p in discover_source_scenes(source_dir)}
    if not available:
        raise SystemExit(f"No valid scenes found under {source_dir}")

    selected: list[str] = []

    for raw in names:
        name = raw.strip()
        if name.isdigit():
            name = _scene_name_from_sign_id(int(name))
        if name not in available:
            raise SystemExit(f"Scene not found in catalog: {name!r} (source: {source_dir})")
        selected.append(name)

    for sign_id in sign_ids:
        name = _scene_name_from_sign_id(sign_id)
        if name not in available:
            raise SystemExit(f"Sign id {sign_id} not found in catalog ({name!r})")
        selected.append(name)

    if not selected:
        ordered = sorted(available)
        if skip_existing:
            already = existing_dest_scene_names(dest_root)
            ordered = [name for name in ordered if name not in already]
        if limit is not None:
            ordered = ordered[:limit]
        selected = ordered

    if limit is not None and len(selected) > limit:
        selected = selected[:limit]

    # Preserve order, drop duplicates
    seen: set[str] = set()
    out: list[str] = []
    for name in selected:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def normalize_meta(meta: dict, scene_name: str) -> dict:
    """Ensure yield_sign-compatible meta fields."""
    out = dict(meta)
    out["scene_name"] = scene_name
    if out.get("sign_type") and not out.get("pdd_code"):
        out["pdd_code"] = out["sign_type"]
    return out


def copy_scene(source_dir: Path, dest_root: Path, scene_name: str, *, overwrite: bool) -> Path:
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
    meta = normalize_meta(meta, scene_name)
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
    subprocess.run(cmd, check=True, cwd=str(YIELD_SIGN_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import 2.4 catalog scenes into yield_sign/scenes with previews",
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
        default=DEFAULT_SOURCE,
        help=f"Catalog root (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination scenes root (default: {DEFAULT_DEST})",
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
        help="When no scenes are named, import up to N catalog scenes "
        "not already present in --dest (unless --overwrite)",
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

    source_dir = args.source.expanduser().resolve()
    dest_root = args.dest.expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    if not source_dir.is_dir():
        sys.exit(f"Source catalog not found: {source_dir}")

    scene_names = resolve_scene_names(
        source_dir,
        dest_root,
        list(args.scenes),
        list(args.sign_ids),
        args.limit,
        skip_existing=not args.overwrite,
    )
    if not scene_names:
        already = len(existing_dest_scene_names(dest_root))
        sys.exit(
            "No scenes selected. Pass scene names, --sign-ids, or --limit N.\n"
            f"Catalog: {len(discover_source_scenes(source_dir))} scene(s), "
            f"already in dest: {already}.\n"
            f"Available: {', '.join(p.name for p in discover_source_scenes(source_dir)[:8])}..."
        )

    run_sim = args.run_simulation and not args.no_simulation

    print(f"Source: {source_dir}")
    print(f"Dest:   {dest_root}")
    print(f"Scenes: {', '.join(scene_names)}")
    print(f"Render previews: {not args.no_render}")
    print(f"Run simulation:  {run_sim}")

    for scene_name in scene_names:
        print(f"\n=== {scene_name} ===")
        scene_dir = copy_scene(
            source_dir,
            dest_root,
            scene_name,
            overwrite=args.overwrite,
        )

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

    print("\nDone.")


if __name__ == "__main__":
    main()
