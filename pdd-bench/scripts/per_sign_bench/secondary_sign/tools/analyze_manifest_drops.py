#!/usr/bin/env python3
"""Classify why cropped secondary-road scenes are dropped during manifest generation."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from lib.manifest_config import (  # noqa: E402
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_SPAWN_DISTANCE_BEFORE_END,
)
from lib.manifest_viability import check_scene_dir_viability  # noqa: E402


def diagnose_scene(scene_dir: Path) -> dict:
    result = check_scene_dir_viability(
        scene_dir,
        min_ego_lane_m=DEFAULT_SPAWN_DISTANCE_BEFORE_END,
        aux_distance_from_intersection=DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    )
    return {
        "scene": scene_dir.name,
        "outcome": "kept" if result.viable else "dropped",
        "reason": result.reason,
        "detail": result.detail,
        "spawn_lane_count": result.spawn_lane_count,
        "scenario_count": result.scenario_count,
    }


def main() -> None:
    scenes_dir = SCRIPT_DIR / "scenes"
    scene_dirs = sorted(
        p
        for p in scenes_dir.iterdir()
        if p.is_dir() and (p / "meta.json").exists() and (p / "map.net.xml").exists()
    )

    results = [diagnose_scene(scene_dir) for scene_dir in scene_dirs]

    kept = [r for r in results if r["outcome"] == "kept"]
    dropped = [r for r in results if r["outcome"] == "dropped"]

    print(f"Scenes scanned: {len(results)}")
    print(f"Kept: {len(kept)}")
    print(f"Dropped: {len(dropped)}")
    print()

    by_reason = Counter(r["reason"] for r in dropped)
    print("Drop reasons:")
    for reason, count in by_reason.most_common():
        print(f"  {count:3d}  {reason}")
    print()

    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in dropped:
        groups[row["reason"]].append(row)

    for reason, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"--- {reason} ({len(rows)}) ---")
        for row in sorted(rows, key=lambda r: r["scene"])[:8]:
            print(f"  {row['scene']}: {row['detail']}")
        if len(rows) > 8:
            print(f"  ... and {len(rows) - 8} more")
        print()


if __name__ == "__main__":
    main()
