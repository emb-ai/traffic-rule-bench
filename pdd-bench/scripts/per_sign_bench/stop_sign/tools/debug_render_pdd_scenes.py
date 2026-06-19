#!/usr/bin/env python3
"""Batch-render cropped PNG previews of PDD 2.5 SUMO networks.

Reads ``meta.json`` + ``*.net.xml`` from ``pdd-bench/scenes/2.5/``, crops
around ``latitude`` / ``longitude`` from meta, and writes per-scene folders
to ``stop_sign/tools/debug_maps/2_5/<scene_name>/`` with ``map.png`` and
``center.json``.

Usage:
    python debug_render_pdd_scenes.py
    python debug_render_pdd_scenes.py --limit 10
    python debug_render_pdd_scenes.py --scene sign_224392
    python debug_render_pdd_scenes.py --radius-m 80 --dpi 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
STOP_SIGN_DIR = TOOLS_DIR.parent
PDD_BENCH_DIR = STOP_SIGN_DIR.parent.parent.parent
DEFAULT_SOURCE_DIR = PDD_BENCH_DIR / "scenes" / "2.5"
DEFAULT_OUTPUT_DIR = TOOLS_DIR / "debug_maps" / "2_5"
DEFAULT_RADIUS_M = 100.0

sys.path.insert(0, str(STOP_SIGN_DIR))
sys.path.insert(0, str(TOOLS_DIR))

from render_map import (  # noqa: E402
    filter_network_by_radius,
    latlon_to_net_xy,
    parse_net_location,
    parse_sumo_net,
    render_network,
)

from lib.sumo_utils import load_scene_meta, resolve_net_file  # noqa: E402


def write_center_json(scene_out_dir: Path, lat: float, lon: float) -> Path:
    scene_out_dir.mkdir(parents=True, exist_ok=True)
    center_path = scene_out_dir / "center.json"
    with open(center_path, "w", encoding="utf-8") as f:
        json.dump({"lat": lat, "lon": lon}, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return center_path


def discover_scene_dirs(source_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    scenes: list[Path] = []
    for entry in sorted(source_dir.iterdir()):
        if entry.is_dir() and list(entry.glob("*.net.xml")):
            scenes.append(entry)
    return scenes


def render_scene(
    scene_dir: Path,
    out_dir: Path,
    *,
    radius_m: float,
    figsize: float,
    dpi: int,
    overwrite: bool,
) -> Path | None:
    scene_name = scene_dir.name
    scene_out_dir = out_dir / scene_name
    out_path = scene_out_dir / "map.png"
    center_path = scene_out_dir / "center.json"

    try:
        meta = load_scene_meta(scene_dir)
        net_file = resolve_net_file(scene_dir, meta)
        net_path = scene_dir / net_file
    except FileNotFoundError as exc:
        print(f"  skip {scene_name}: {exc}")
        return None

    lat = meta.get("latitude")
    lon = meta.get("longitude")
    if lat is None or lon is None:
        print(f"  skip {scene_name}: meta.json missing latitude/longitude")
        return None

    lat_f, lon_f = float(lat), float(lon)
    if not center_path.exists() or overwrite:
        write_center_json(scene_out_dir, lat_f, lon_f)

    if out_path.exists() and not overwrite:
        print(f"  skip (exists): {scene_name}/map.png")
        return out_path

    location = parse_net_location(net_path)
    if location is None:
        print(f"  skip {scene_name}: net.xml has no location bounds")
        return None

    try:
        cx, cy = latlon_to_net_xy(lat_f, lon_f, location)
        edges, junctions = parse_sumo_net(net_path)
        edges, junctions = filter_network_by_radius(edges, junctions, cx, cy, radius_m)
        if not edges and not junctions:
            print(f"  skip {scene_name}: nothing within {radius_m:.0f}m of meta point")
            return None
        render_network(
            edges,
            junctions,
            out_path,
            figsize=(figsize, figsize),
            dpi=dpi,
            center=(cx, cy),
            radius_m=radius_m,
        )
    except Exception as exc:
        print(f"  FAIL {scene_name}: {exc}")
        return None

    sign_id = meta.get("sign_id", scene_name)
    print(
        f"  ok {scene_name} (sign_id={sign_id}, center=({cx:.1f},{cy:.1f}), "
        f"r={radius_m:.0f}m, lanes={len(edges)}) -> {scene_name}/"
    )
    return out_path


def write_index(
    out_dir: Path,
    rendered: list[tuple[Path, Path]],
    *,
    radius_m: float,
) -> None:
    index = []
    for scene_dir, png_path in rendered:
        meta = {}
        meta_path = scene_dir / "meta.json"
        if meta_path.is_file():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        scene_out_dir = png_path.parent
        index.append(
            {
                "scene_id": scene_dir.name,
                "sign_id": meta.get("sign_id"),
                "latitude": meta.get("latitude"),
                "longitude": meta.get("longitude"),
                "crop_radius_m": radius_m,
                "source_dir": str(scene_dir),
                "net_file": meta.get("net_file"),
                "output_dir": str(scene_out_dir.relative_to(STOP_SIGN_DIR)),
                "center_json": str((scene_out_dir / "center.json").relative_to(STOP_SIGN_DIR)),
                "png": str(png_path.relative_to(STOP_SIGN_DIR)),
            }
        )
    index_path = out_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"\nWrote index: {index_path} ({len(index)} entries)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render cropped PNG previews of PDD 2.5 scenes around meta.json lat/lon.",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scene", action="append", default=None, metavar="NAME")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--radius-m",
        type=float,
        default=DEFAULT_RADIUS_M,
        help=f"Crop radius in meters around meta lat/lon (default: {DEFAULT_RADIUS_M})",
    )
    parser.add_argument("--figsize", type=float, default=10)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scene:
        scene_dirs = [source_dir / name for name in args.scene if (source_dir / name).is_dir()]
    else:
        scene_dirs = discover_scene_dirs(source_dir)

    if args.limit is not None:
        scene_dirs = scene_dirs[: max(0, args.limit)]

    print(f"Source: {source_dir}")
    print(f"Output: {out_dir}")
    print(f"Crop radius: {args.radius_m:.0f} m")
    print(f"Scenes to render: {len(scene_dirs)}\n")

    rendered: list[tuple[Path, Path]] = []
    for i, scene_dir in enumerate(scene_dirs, start=1):
        print(f"[{i}/{len(scene_dirs)}] {scene_dir.name}")
        png = render_scene(
            scene_dir,
            out_dir,
            radius_m=args.radius_m,
            figsize=args.figsize,
            dpi=args.dpi,
            overwrite=args.overwrite,
        )
        if png is not None:
            rendered.append((scene_dir, png))

    if rendered:
        write_index(out_dir, rendered, radius_m=args.radius_m)

    print(f"\nDone. Rendered {len(rendered)}/{len(scene_dirs)} scenes.")


if __name__ == "__main__":
    main()
