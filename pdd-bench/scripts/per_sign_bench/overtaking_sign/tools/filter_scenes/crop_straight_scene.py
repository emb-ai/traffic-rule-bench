#!/usr/bin/env python3
"""Promote core 3.20 scenes into runnable crops under scenes/3_20/.

For 3.20 we keep the full net (catalog scenes are already local) and emit a
sibling crop folder ``sign_*_s0`` with crop meta (aux / dest / spawn fields).
True geometric cropping is optional later; this step makes the layout match
other per_sign benches (core/ + crop siblings).

Examples:
  python tools/filter_scenes/crop_straight_scene.py --limit 20
  python tools/filter_scenes/crop_straight_scene.py --overwrite
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
CORE_DEFAULT = SIGN_DIR / "scenes" / "3_20" / "core"
OUT_DEFAULT = SIGN_DIR / "scenes" / "3_20"

sys.path.insert(0, str(SIGN_DIR))
from lib.straight_pair import analyze_road_pair  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--core", type=Path, default=CORE_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--min-length", type=float, default=60.0)
    ap.add_argument("--max-heading-std", type=float, default=12.0)
    ap.add_argument("--aux-frac", type=float, default=0.5)
    args = ap.parse_args()

    if not args.core.is_dir():
        sys.exit(f"ERROR: core dir missing: {args.core} (run import_catalog_scenes.py)")

    args.out.mkdir(parents=True, exist_ok=True)
    cores = sorted(
        p for p in args.core.iterdir()
        if p.is_dir() and (p / "meta.json").is_file()
    )
    n = 0
    for core in cores:
        if args.limit is not None and n >= args.limit:
            break
        meta = json.loads((core / "meta.json").read_text(encoding="utf-8"))
        road_id = meta.get("road_id")
        nets = sorted(core.glob("*.net.xml"))
        if not road_id or not nets:
            continue
        pair = analyze_road_pair(
            nets[0],
            str(road_id),
            min_length_m=args.min_length,
            max_heading_std_deg=args.max_heading_std,
            aux_frac=args.aux_frac,
        )
        if pair is None:
            continue
        crop_name = f"{core.name}_s0"
        dst = args.out / crop_name
        if dst.exists():
            if not args.overwrite:
                print(f"[skip] exists {dst.name}")
                n += 1
                continue
            shutil.rmtree(dst)
        shutil.copytree(core, dst)
        out_meta = dict(meta)
        out_meta.update(
            {
                "pdd_code": "3.20",
                "sign_type": "3.20",
                "road_id": pair.ego_edge,
                "opposite_edge_id": pair.opposite_edge,
                "spawn_lane_num": 0,
                "aux_long_m": pair.aux_long_m,
                "destination_edge_id": pair.destination_edge,
                "destination_lane_id": f"lane_{pair.destination_edge}_0",
                "approach_length_m": pair.length_m,
                "heading_std_deg": pair.heading_std_deg,
                "sign_distance_from_start": 2.0,
                "spawn_distance_from_start": 3.0,
                "net_path": f"{crop_name}/{nets[0].name}",
                "core_scene": core.name,
                "crop_id": 0,
            }
        )
        (dst / "meta.json").write_text(
            json.dumps(out_meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"[crop] {crop_name}  L={pair.length_m:.1f}m  "
            f"aux@{pair.aux_long_m:.1f}m  dest={pair.destination_edge}"
        )
        n += 1
    print(f"Wrote {n} crops under {args.out}")


if __name__ == "__main__":
    main()
