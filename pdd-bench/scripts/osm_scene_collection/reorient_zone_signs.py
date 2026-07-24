#!/usr/bin/env python3
"""Re-orient zone-entry signs (5.21) in ALREADY-COLLECTED scenes, in place.

Each scene's meta.json records `road_id` (a *directed* SUMO edge) and
`distance_from_start`. For residential-zone sign 5.21 the route must read
"big road -> sign -> courtyard"; for ~half the scenes the sign edge is oriented
the wrong way (see sign_edge_orientation.py). This pass re-reads the existing
net.xml (NO Overpass / netconvert), recomputes the correct directed edge, and
rewrites ONLY `road_id` + `distance_from_start`.

Dry-run by default — pass --apply to write. Idempotent: scenes already checked
are skipped unless --force.

Examples:
  # dry-run report over the residential scene sets
  python reorient_zone_signs.py --scenes-root scenes_resid scenes --report
  # apply the fix
  python reorient_zone_signs.py --scenes-root scenes_resid scenes --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sign_edge_orientation import (
    ORIENT_INTO_ZONE_CODES, orient_sign_edge, resnap_to_residential,
    resnap_off_service,
)

# Speed-limit signs: re-snap off `service` roads onto a real street; NO inbound
# orientation flip (3.24 faces approaching traffic on a normal road).
SPEED_LIMIT_CODES = {"3.24"}


def _iter_scene_dirs(scenes_root: Path, codes):
    for code in sorted(codes):
        code_dir = scenes_root / code
        if not code_dir.is_dir():
            continue
        for scene_dir in sorted(code_dir.iterdir()):
            if scene_dir.is_dir() and (scene_dir / "meta.json").exists():
                yield code, scene_dir


def process_scene(scene_dir: Path, sign_code: str, max_hops: int, apply: bool,
                  resnap: bool, max_resnap_dist: float):
    meta_path = scene_dir / "meta.json"
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return {"resnap": "meta_error", "orient": "meta_error", "scene": scene_dir.name}

    net_file = meta.get("net_file")
    if not net_file:
        return {"resnap": "no_net_file", "orient": "no_net_file", "scene": scene_dir.name}
    net_path = scene_dir / net_file
    if not net_path.exists():
        return {"resnap": "net_missing", "orient": "net_missing", "scene": scene_dir.name}

    road_id = str(meta.get("road_id", ""))
    dist = float(meta.get("distance_from_start", 0.0) or 0.0)

    # 1. Re-snap a sign off the wrong road type:
    #    - zone-entry (5.21): off through-road onto a residential/courtyard edge;
    #    - speed-limit (3.24): off a `service` alley onto the nearest real street.
    resnap_decision = "off"
    if resnap:
        if sign_code in SPEED_LIMIT_CODES:
            nr, nd, rinfo = resnap_off_service(
                str(net_path), road_id, dist, max_dist=max_resnap_dist)
        else:
            nr, nd, rinfo = resnap_to_residential(
                str(net_path), road_id, dist, sign_code, max_dist=max_resnap_dist)
        resnap_decision = rinfo["decision"]
        if resnap_decision == "resnap":
            road_id, dist = nr, nd
    # 2. Orient inbound ONLY for zone-entry signs (3.24 keeps its direction).
    if sign_code in ORIENT_INTO_ZONE_CODES:
        road_id, dist, oinfo = orient_sign_edge(
            str(net_path), road_id, dist, sign_code, max_hops=max_hops)
    else:
        oinfo = {"decision": "skip_non_zone", "dist_fwd": None, "dist_bwd": None}

    out = {
        "scene": scene_dir.name,
        "resnap": resnap_decision,
        "orient": oinfo["decision"],
        "dist_fwd": oinfo.get("dist_fwd"),
        "dist_bwd": oinfo.get("dist_bwd"),
        "new_road_id": road_id,
    }
    if apply:
        meta["road_id"] = road_id
        meta["distance_from_start"] = dist
        meta["orientation_checked"] = True
        meta["orientation_resnapped"] = (resnap_decision == "resnap")
        meta["orientation_resnap_reason"] = resnap_decision
        meta["orientation_fixed"] = (oinfo["decision"] == "flip")
        meta["orientation_reason"] = oinfo["decision"]
        with open(meta_path, "w") as f:
            f.write(json.dumps(meta, indent=2, ensure_ascii=False))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes-root", nargs="+", default=["scenes"],
                    help="one or more scene roots (each contains <code>/sign_*/)")
    ap.add_argument("--codes", nargs="+", default=sorted(ORIENT_INTO_ZONE_CODES),
                    help="sign codes to reorient (default: %(default)s)")
    ap.add_argument("--big-road-hops", type=int, default=6,
                    help="max BFS hops when locating the nearest big road")
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default: dry-run)")
    ap.add_argument("--report", action="store_true",
                    help="print a per-scene table")
    ap.add_argument("--no-resnap", action="store_true",
                    help="skip re-snapping through-road signs onto residential edges")
    ap.add_argument("--resnap-dist", type=float, default=40.0,
                    help="max metres to the nearest residential edge for re-snap")
    args = ap.parse_args()

    codes = set(args.codes)
    resnap_counts = Counter()
    orient_counts = Counter()
    rows = []
    for root in args.scenes_root:
        scenes_root = Path(root)
        if not scenes_root.is_dir():
            print(f"!! scenes root not found: {scenes_root}", file=sys.stderr)
            continue
        for code, scene_dir in _iter_scene_dirs(scenes_root, codes):
            info = process_scene(scene_dir, code, args.big_road_hops, args.apply,
                                 resnap=not args.no_resnap, max_resnap_dist=args.resnap_dist)
            resnap_counts[info["resnap"]] += 1
            orient_counts[info["orient"]] += 1
            rows.append((str(scenes_root), code, info))

    if args.report:
        print(f"{'root':<14} {'code':<5} {'scene':<16} {'fwd':>4} {'bwd':>4} "
              f"{'resnap':<22} {'orient':<10} {'new_road_id'}")
        for root, code, info in rows:
            df = info.get("dist_fwd"); db = info.get("dist_bwd")
            print(f"{root:<14} {code:<5} {info.get('scene',''):<16} {str(df):>4} {str(db):>4} "
                  f"{str(info.get('resnap')):<22} {str(info.get('orient')):<10} {info.get('new_road_id','')}")

    print("\n=== resnap summary ===")
    for d, n in sorted(resnap_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {d:<22} {n}")
    print("=== orient summary ===")
    for d, n in sorted(orient_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {d:<22} {n}")
    print(f"  {'TOTAL':<22} {sum(orient_counts.values())}")
    if not args.apply:
        print("\n(dry-run — re-run with --apply to write changes)")


if __name__ == "__main__":
    main()
