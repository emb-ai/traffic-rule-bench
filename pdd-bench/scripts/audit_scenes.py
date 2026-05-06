#!/usr/bin/env python3
"""Audit benchmark scenes against per-sign-type quality criteria.

Walks <root>/<sign>/sumo/sumo_manifest.jsonl (or <root>/<sign>/real_manifest.jsonl)
and reports per-sign-type counts of:
  - missing net file
  - short runway (sign_spawn_distance < 30 m)
  - single-lane road for lane-change-required signs (5.11.1/5.14.x/5.15.2/5.13.x)
  - missing destination

Usage:
  python3 pdd-bench/scripts/audit_scenes.py pdd-bench/scripts/per_sign_bench/benchmark_output/mini

Env-vars:
  SCENES_ROOT  : where net_path resolves (default: <repo>/pdd-bench/scenes)
  WRITE_REPORT : if set, write JSON details to this path
"""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


LANE_CHANGE_SIGNS = {
    "5_11_1", "5_11_2",
    "5_14_1", "5_14_2", "5_14_3",
    "5_15_2",
    "5_13_1", "5_13_2", "5_13_3", "5_13_4",
}


def lanes_of_edge(net_path: Path, edge_id: str) -> int:
    """Return number of <lane> children in <edge id=edge_id> in the SUMO net."""
    if not edge_id:
        return -1
    try:
        tree = ET.parse(net_path)
    except Exception:
        return -1
    for e in tree.getroot().findall("edge"):
        if e.get("id") == edge_id:
            return len(e.findall("lane"))
    return 0


def find_manifest(sign_dir: Path) -> Path | None:
    for cand in (sign_dir / "sumo" / "sumo_manifest.jsonl",
                 sign_dir / "real_manifest.jsonl"):
        if cand.exists():
            return cand
    return None


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <benchmark_output/mini path>", file=sys.stderr)
        sys.exit(1)
    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    # SCENES_ROOT defaults to pdd-bench/scenes relative to repo
    scenes_root_env = os.environ.get("SCENES_ROOT")
    if scenes_root_env:
        scenes_root = Path(scenes_root_env)
    else:
        # walk up from root to find pdd-bench/scenes
        scenes_root = root
        for parent in (root, *root.parents):
            if (parent / "pdd-bench" / "scenes").is_dir():
                scenes_root = parent / "pdd-bench" / "scenes"
                break
    print(f"Manifests root:  {root}")
    print(f"Scenes (.net.xml) root: {scenes_root}")
    print()

    stats = defaultdict(lambda: defaultdict(int))
    issues = defaultdict(list)
    by_scene = defaultdict(list)

    for sign_dir in sorted(root.iterdir()):
        if not sign_dir.is_dir():
            continue
        sign = sign_dir.name
        manifest = find_manifest(sign_dir)
        if manifest is None:
            stats[sign]["no_manifest"] += 1
            continue
        for line in manifest.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            scene_id = r.get("scene_id", "?")
            net = r.get("net_path") or ""
            road_id = r.get("road_id") or r.get("fingerprint", {}).get("start_lane", "")
            spawn_dist = float(r.get("sign_spawn_distance", 0) or
                               r.get("distance_from_start", 0) or 0)
            stats[sign]["total"] += 1

            # Resolve net path: absolute, or scenes_root-relative
            net_full = Path(net) if Path(net).is_absolute() else (scenes_root / net)
            if not net_full.is_file():
                # try sign-dir-relative as last resort
                alt = sign_dir / net
                if alt.is_file():
                    net_full = alt
                else:
                    stats[sign]["missing_net"] += 1
                    issues[sign].append(f"{scene_id}: net not found ({net})")
                    by_scene[(sign, scene_id)].append("missing_net")
                    continue

            # Check 1: spawn distance / runway
            if spawn_dist < 30:
                stats[sign]["short_runway"] += 1
                issues[sign].append(f"{scene_id}: spawn_dist={spawn_dist:.1f}m (<30)")
                by_scene[(sign, scene_id)].append(f"short_runway({spawn_dist:.0f}m)")

            # Check 2: lane count for lane-change-required signs
            if sign in LANE_CHANGE_SIGNS:
                n_lanes = lanes_of_edge(net_full, road_id)
                if n_lanes == 1:
                    stats[sign]["single_lane"] += 1
                    issues[sign].append(f"{scene_id}: road={road_id} has 1 lane — LC impossible")
                    by_scene[(sign, scene_id)].append("single_lane")
                elif n_lanes == 0:
                    stats[sign]["no_road_in_net"] += 1
                    issues[sign].append(f"{scene_id}: road={road_id} not found in net")
                    by_scene[(sign, scene_id)].append("no_road_in_net")
                else:
                    stats[sign][f"multi_lane_{n_lanes}"] += 1

            # Check 3: destination set
            dest = r.get("destination_lane_id") or r.get("fingerprint", {}).get("dest_node")
            if not dest:
                stats[sign]["no_destination"] += 1
                by_scene[(sign, scene_id)].append("no_destination")

    # ---- Print compact table ----
    cols = ("total", "missing_net", "short_runway",
            "single_lane", "no_road_in_net", "no_destination")
    hdr = "{:<10}".format("sign") + " ".join(f"{c:>14}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for sign in sorted(stats):
        row = "{:<10}".format(sign) + " ".join(f"{stats[sign].get(c, 0):>14d}" for c in cols)
        print(row)

    # ---- Lane-change subsection ----
    print()
    print("Lane-change-required signs — lane distribution:")
    for sign in sorted(LANE_CHANGE_SIGNS):
        if sign not in stats:
            continue
        s = stats[sign]
        total = s.get("total", 0)
        single = s.get("single_lane", 0)
        ml = {k: v for k, v in s.items() if k.startswith("multi_lane_")}
        print(f"  {sign:<10} total={total:>4} 1-lane={single:>4} "
              f"multi: {dict(sorted(ml.items()))}")

    # ---- Issue samples ----
    print()
    print("Sample issues per sign (first 3):")
    for sign in sorted(issues):
        for msg in issues[sign][:3]:
            print(f"  {sign:<10} {msg}")
        if len(issues[sign]) > 3:
            print(f"  {sign:<10} ... +{len(issues[sign]) - 3} more")

    # ---- Total counters ----
    total_scenes = sum(stats[s].get("total", 0) for s in stats)
    bad_scenes = len(by_scene)
    print()
    print(f"Total scenes audited: {total_scenes}")
    print(f"Scenes with at least one issue: {bad_scenes} "
          f"({100 * bad_scenes / max(1, total_scenes):.1f}%)")

    # ---- Optional JSON dump ----
    out_path = os.environ.get("WRITE_REPORT")
    if out_path:
        report = {
            "stats": {k: dict(v) for k, v in stats.items()},
            "issues_by_sign": {k: v for k, v in issues.items()},
            "scenes_with_issues": [
                {"sign": k[0], "scene": k[1], "issues": v}
                for k, v in by_scene.items()
            ],
        }
        Path(out_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nWrote report: {out_path}")


if __name__ == "__main__":
    main()
