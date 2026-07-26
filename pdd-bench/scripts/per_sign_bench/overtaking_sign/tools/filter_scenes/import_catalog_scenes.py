#!/usr/bin/env python3
"""Import 3.20 catalog scenes that are viable 1+1 straight approaches.

Copies from ``pdd-bench/scenes/3.20`` into ``overtaking_sign/scenes/3_20/core/``,
keeping only scenes whose ``meta.road_id`` is a 1-lane edge with a 1-lane opposite
and enough length / straightness.

Examples:
  python tools/filter_scenes/import_catalog_scenes.py --limit 20
  python tools/filter_scenes/import_catalog_scenes.py --overwrite --min-length 50
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

FILTER_SCENES_DIR = Path(__file__).resolve().parent
TOOLS_DIR = FILTER_SCENES_DIR.parent
SIGN_DIR = TOOLS_DIR.parent
PDD_BENCH_DIR = SIGN_DIR.parent.parent.parent
DEFAULT_SOURCE = PDD_BENCH_DIR / "scenes" / "3.20"
DEFAULT_DEST = SIGN_DIR / "scenes" / "3_20" / "core"

sys.path.insert(0, str(SIGN_DIR))
from lib.straight_pair import analyze_road_pair  # noqa: E402


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


def resolve_net(scene_dir: Path, meta: dict) -> Path | None:
    net_file = meta.get("net_file")
    if net_file and (scene_dir / net_file).is_file():
        return scene_dir / net_file
    nets = sorted(scene_dir.glob("*.net.xml"))
    return nets[0] if nets else None


def copy_scene(src: Path, dest_root: Path, *, overwrite: bool) -> Path:
    dst = dest_root / src.name
    if dst.exists():
        if not overwrite:
            return dst
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--min-length", type=float, default=60.0)
    ap.add_argument("--max-heading-std", type=float, default=12.0)
    ap.add_argument("--aux-frac", type=float, default=0.5)
    args = ap.parse_args()

    scenes = discover_source_scenes(args.source)
    if not scenes:
        sys.exit(f"ERROR: no scenes under {args.source}")

    args.dest.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    for scene_dir in scenes:
        if args.limit is not None and kept >= args.limit:
            break
        meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
        road_id = meta.get("road_id")
        net = resolve_net(scene_dir, meta)
        if not road_id or net is None:
            skipped += 1
            continue
        pair = analyze_road_pair(
            net,
            str(road_id),
            min_length_m=args.min_length,
            max_heading_std_deg=args.max_heading_std,
            aux_frac=args.aux_frac,
        )
        if pair is None:
            skipped += 1
            continue
        dst = copy_scene(scene_dir, args.dest, overwrite=args.overwrite)
        # Enrich meta for crop / manifest.
        meta = dict(meta)
        meta["sign_type"] = "3.20"
        meta["pdd_code"] = "3.20"
        meta["road_id"] = pair.ego_edge
        meta["opposite_edge_id"] = pair.opposite_edge
        meta["spawn_lane_num"] = 0
        meta["aux_long_m"] = pair.aux_long_m
        meta["destination_edge_id"] = pair.destination_edge
        meta["approach_length_m"] = pair.length_m
        meta["heading_std_deg"] = pair.heading_std_deg
        meta["net_path"] = f"{dst.name}/{net.name}"
        (dst / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        kept += 1
        print(
            f"[keep] {dst.name}  edge={pair.ego_edge}  "
            f"opp={pair.opposite_edge}  L={pair.length_m:.1f}m  "
            f"aux@{pair.aux_long_m:.1f}m"
        )

    print(f"\nKept {kept}, skipped {skipped}. Dest: {args.dest}")


if __name__ == "__main__":
    main()
