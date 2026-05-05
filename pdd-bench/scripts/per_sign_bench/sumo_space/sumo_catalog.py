"""Build the SUMO scene catalog: enumerate all (scene, v_idx, var_idx) tuples
and assign deterministic seeds. Cheap — no env, no .net.xml parsing.

The catalog is the source of truth for "what scenes exist". Materialization
(running them through MetaDrive) is a separate, expensive step that reads from it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from .sumo_scene_enumerator import (
    SumoScene,
    enumerate_all_scenes,
    stratified_sample,
    filter_by_sign_codes,
    count_lanes_on_road,
)

# Legacy fallback. New pipelines draw spawn_velocity from nuPlan (see
# sumo_runner.py) — catalog rows no longer fix a velocity value. Kept only for
# old code paths that still iterate over this list.
DEFAULT_SPAWN_VELOCITIES = [0.0]

# Default number of nuPlan draws per (scene, spawn_lane_num, var_idx).
DEFAULT_N_VELOCITY_SAMPLES = 5


def stable_hash(*parts) -> int:
    """SHA-256-based deterministic hash → 32-bit int.

    Used so that the seed for a (scene_id, v_idx, var_idx) triple is reproducible
    across machines and Python versions.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest()[:4], "big")


def build_catalog(
    scenes_root: str | Path,
    n_per_category: int = 250,
    n_variations: int = 10,
    sign_categories: Optional[List[str]] = None,
    seed: int = 42,
    # Deprecated params — kept for backward compat with older callers.
    n_velocity_samples: int = 1,
    spawn_velocities: Optional[List[float]] = None,
) -> List[dict]:
    """Build the catalog of all SUMO-scene variants.

    Each row: {scene_id, sign_code, sign_id, road_id, net_path,
    sign_spawn_distance, distance_from_start, destination_lane_id, n_lanes,
    spawn_lane_num, var_idx, seed, spawn_velocity_ms=0.0, latitude,
    longitude, osm_way_id}.

    Enumeration axes for SUMO:
      - every scene contributes `n_lanes` rows (all spawn lanes on its road)
      - × `n_variations` (different NPC profile/traffic seeds)

    Ego spawn velocity is FIXED at 0 m/s (IDM accelerates ego from rest as in
    the original SUMO replay). No nuPlan sampling of ego velocity for SUMO —
    only NPC parameters vary through `n_variations`.
    """
    all_scenes = enumerate_all_scenes(scenes_root)
    if sign_categories:
        all_scenes = filter_by_sign_codes(all_scenes, sign_categories)
    sampled = stratified_sample(all_scenes, n_per_category=n_per_category, seed=seed)

    rows: List[dict] = []
    for scene in sampled:
        n_lanes = count_lanes_on_road(scene.net_path, scene.road_id)
        if n_lanes <= 0:
            lane_range = [0]
            n_lanes_field = 1
        else:
            lane_range = list(range(n_lanes))
            n_lanes_field = n_lanes

        for spawn_lane_num in lane_range:
            for var_idx in range(n_variations):
                row_seed = stable_hash(scene.scene_id, spawn_lane_num, var_idx)
                rows.append({
                    "scene_id": scene.scene_id,
                    "sign_code": scene.sign_code,
                    "sign_id": scene.sign_id,
                    "road_id": scene.road_id,
                    "net_path": scene.net_path,
                    "sign_spawn_distance": scene.distance_from_start,
                    "distance_from_start": scene.distance_from_start,
                    "destination_lane_id": scene.destination_lane_id,
                    "latitude": scene.latitude,
                    "longitude": scene.longitude,
                    "osm_way_id": scene.osm_way_id,
                    "n_lanes": n_lanes_field,
                    "spawn_lane_num": spawn_lane_num,
                    "var_idx": var_idx,
                    "seed": row_seed,
                    "spawn_velocity_ms": 0.0,
                })
    return rows


def save_catalog(catalog: List[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(catalog, f, indent=None, separators=(",", ":"))


def load_catalog(path: str | Path) -> List[dict]:
    """Load a catalog. Accepts both JSON array (.json) and JSONL (.jsonl)."""
    path = Path(path)
    with open(path) as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        # JSONL
        return [json.loads(line) for line in f if line.strip()]


def sample_catalog(
    catalog: List[dict],
    n_per_sign: int = 3,
    seed: int = 42,
) -> List[dict]:
    """Pick n_per_sign random rows from each sign_code in the catalog.

    Used by sumo_smoke_check.py for the 3-random-scenes-per-sign correctness gate.
    """
    rng = random.Random(seed)
    by_code: dict[str, List[dict]] = defaultdict(list)
    for row in catalog:
        by_code[row["sign_code"]].append(row)

    result: List[dict] = []
    for code in sorted(by_code.keys()):
        bucket = by_code[code]
        if n_per_sign >= len(bucket):
            picked = bucket
        else:
            picked = rng.sample(bucket, n_per_sign)
        result.extend(picked)
    return result


def main():
    parser = argparse.ArgumentParser(description="Build SUMO scene catalog")
    _default_scenes = str(Path(__file__).resolve().parent.parent.parent.parent / "scenes")
    parser.add_argument("--scenes-root", type=str,
                        default=_default_scenes)
    parser.add_argument("--n-per-category", type=int, default=250)
    parser.add_argument("--n-variations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write the catalog JSON")
    parser.add_argument("--sign-categories", type=str, default=None,
                        help="Comma-separated list of sign codes to restrict to")
    args = parser.parse_args()

    sign_codes = None
    if args.sign_categories:
        sign_codes = [c.strip() for c in args.sign_categories.split(",") if c.strip()]

    catalog = build_catalog(
        scenes_root=args.scenes_root,
        n_per_category=args.n_per_category,
        n_variations=args.n_variations,
        sign_categories=sign_codes,
        seed=args.seed,
    )
    save_catalog(catalog, args.output)
    print(f"Catalog written: {args.output}")
    print(f"  rows: {len(catalog):,}")
    by_code: dict[str, int] = defaultdict(int)
    for r in catalog:
        by_code[r["sign_code"]] += 1
    print(f"  sign categories: {len(by_code)}")
    print(f"  variants per category: min={min(by_code.values())}, "
          f"max={max(by_code.values())}, "
          f"median={sorted(by_code.values())[len(by_code)//2]}")


if __name__ == "__main__":
    main()
