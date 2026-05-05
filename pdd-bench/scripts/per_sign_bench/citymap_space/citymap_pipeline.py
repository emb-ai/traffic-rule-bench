"""End-to-end pipeline for the CityMap benchmark.

For each (map_seed, n_blocks):
1. Open env once, generate CityMap, extract graph
2. Enumerate base scenes (cached to JSON for reproducibility)
3. Materialize each base scene crossed with N profiles × M spawn velocities

Each (base_scene_id, profile_id, velocity_idx) gets a deterministic hash-based
seed so that materialization is byte-reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List

# Path bootstrap
SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_DIR = SCRIPT_PATH.parent.parent
SCRIPTS_DIR = BENCHMARK_DIR.parent
PDD_BENCH_DIR = SCRIPTS_DIR.parent
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"
for _p in (PDD_BENCH_DIR, METADRIVE_DIR, BENCHMARK_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from .citymap_env import CityMapTrafficSignEnv
from .citymap_scene_enumerator import (
    enumerate_scenes_for_map,
    scene_to_dict,
    dict_to_scene,
    CityMapScene,
)
from .citymap_runner import materialize_scene_in_env

from factorized_space.agent_profile_bank import sample_one_profile, sample_spawn_velocity
from factorized_space.space_definition import N_SPAWN_VELOCITY_SAMPLES


# Default spawn velocity bins (m/s) — realistic urban range
DEFAULT_SPAWN_VELOCITIES = [3.0, 5.0, 8.0, 11.0, 14.0]  # fallback only; per-scene nuPlan sampling is default


def stable_hash(*parts) -> int:
    """Deterministic hash for seeding."""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest()[:8], "big") % (2**31)


def graph_signature(graph) -> str:
    """A short hash of the road network graph for cache validation."""
    items = sorted(
        (f, sorted(graph[f].keys()))
        for f in sorted(graph.keys())
    )
    return hashlib.sha256(repr(items).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _cache_path(output_dir: Path, map_seed: int, n_blocks: int, max_per_sign: int,
                sign_categories: List[str] = None) -> Path:
    cat_tag = "all" if not sign_categories else "+".join(sorted(sign_categories))
    # Schema v2: prohibition scenes now store an alternative `route_path` that
    # bypasses the sign edge plus a `primary_path` for reference. Bumping the
    # filename suffix invalidates v1 caches without touching the loader.
    return output_dir / "maps" / f"seed{map_seed}_b{n_blocks}_per{max_per_sign}_{cat_tag}_v2.json"


def save_scene_cache(output_dir: Path, map_seed: int, n_blocks: int, max_per_sign: int,
                     scenes: List[CityMapScene], graph_sig: str,
                     sign_categories: List[str] = None):
    p = _cache_path(output_dir, map_seed, n_blocks, max_per_sign, sign_categories)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({
            "map_seed": map_seed,
            "n_blocks": n_blocks,
            "max_scenes_per_sign": max_per_sign,
            "sign_categories": sign_categories,
            "graph_signature": graph_sig,
            "scenes": [scene_to_dict(s) for s in scenes],
        }, f, indent=2)


def load_scene_cache(output_dir: Path, map_seed: int, n_blocks: int, max_per_sign: int,
                     graph_sig: str, sign_categories: List[str] = None) -> List[CityMapScene]:
    p = _cache_path(output_dir, map_seed, n_blocks, max_per_sign, sign_categories)
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    if data.get("graph_signature") != graph_sig:
        return None
    return [dict_to_scene(d) for d in data["scenes"]]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    map_seeds: List[int],
    n_blocks: int = 8,
    lane_num: int = 3,
    max_scenes_per_sign_per_map: int = 3,
    spawn_velocities: List[float] = None,
    n_velocity_samples: int = N_SPAWN_VELOCITY_SAMPLES,
    output_dir: str | Path = "citymap_output",
    sign_categories: List[str] = None,
    # Deprecated — kept for backward compat, no longer used
    n_profiles: int = 0,
    profile_seed: int = 42,
):
    """Run the CityMap materialization pipeline.

    Velocity handling: if `spawn_velocities` is None (default), each base
    scene produces `n_velocity_samples` rows with spawn_velocity drawn from
    nuPlan initial_speed (deterministic per (scene_id, v_idx)). If callers
    pass a non-empty fixed list, that list is used as-is (legacy behaviour).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_nuplan_velocity = spawn_velocities is None
    if not use_nuplan_velocity:
        print(f"Spawn velocities (fixed): {spawn_velocities} m/s")
    else:
        print(f"Spawn velocities: {n_velocity_samples} nuPlan samples per base-scene")

    print(f"Agent profiles: sampled per-scene from nuPlan (applied to NPC only; ego uses DEFAULT_EGO_PARAMS)")

    all_results = []
    failures = Counter()
    spawn_fallback_count = 0
    t0 = time.time()

    for map_seed in map_seeds:
        print(f"\n=== Map seed {map_seed} (n_blocks={n_blocks}) ===")
        env = CityMapTrafficSignEnv(dict(
            start_seed=map_seed,
            num_scenarios=1,
            use_render=False,
            use_mesh_terrain=False,
            log_level=logging.CRITICAL,
            traffic_density=0.0,
            map=n_blocks,
        ))
        try:
            env.reset(seed=map_seed)
            rn = env.current_map.road_network
            graph = rn.graph
            graph_sig = graph_signature(graph)
            print(f"  Map: {len(graph)} nodes, {len(env.current_map.blocks)} blocks, graph_sig={graph_sig}")

            # Try cache first
            cached = load_scene_cache(output_dir, map_seed, n_blocks,
                                       max_scenes_per_sign_per_map, graph_sig,
                                       sign_categories=sign_categories)
            if cached is not None:
                print(f"  Loaded {len(cached)} base scenes from cache")
                base_scenes = cached
            else:
                base_scenes = enumerate_scenes_for_map(
                    graph,
                    map_seed=map_seed,
                    n_blocks=n_blocks,
                    lane_num=lane_num,
                    max_scenes_per_sign=max_scenes_per_sign_per_map,
                    road_network=rn,
                    sign_categories=sign_categories,
                )
                save_scene_cache(output_dir, map_seed, n_blocks,
                                 max_scenes_per_sign_per_map, base_scenes, graph_sig,
                                 sign_categories=sign_categories)
                print(f"  Enumerated {len(base_scenes)} base scenes (cached)")

            by_sign = Counter(s.sign_placement.sign_key for s in base_scenes)
            prohib_count = sum(by_sign[k] for k in
                               ("no_entry", "no_traffic", "no_right_turn", "no_left_turn", "no_uturn"))
            print(f"  Base scene types: {len(by_sign)}, prohibition (with detour): {prohib_count}")

            n_valid_this_map = 0
            n_total_this_map = 0
            if use_nuplan_velocity:
                # nuPlan: one draw per (scene, v_idx); values are stochastic
                n_vel = n_velocity_samples
            else:
                n_vel = len(spawn_velocities)
            for base in base_scenes:
                for v_idx in range(n_vel):
                    seed = stable_hash(base.scene_id, v_idx)
                    profile = sample_one_profile(seed)
                    if use_nuplan_velocity:
                        vel = sample_spawn_velocity(stable_hash(base.scene_id, "spawn_vel", v_idx))
                    else:
                        vel = spawn_velocities[v_idx]
                    r = materialize_scene_in_env(
                        env, base, profile,
                        spawn_velocity_ms=vel,
                        deterministic_seed=seed,
                    )
                    r["base_scene_id"] = base.scene_id
                    r["spawn_velocity_idx"] = v_idx
                    all_results.append(r)
                    n_total_this_map += 1
                    if r["valid"]:
                        n_valid_this_map += 1
                    else:
                        reason = (r.get("failure_reason") or "?").split(":")[0]
                        failures[reason] += 1
                    if r.get("spawn_fallback"):
                        spawn_fallback_count += 1

            elapsed = time.time() - t0
            print(f"  Materialized {n_total_this_map} (valid: {n_valid_this_map}) "
                  f"[total elapsed {elapsed:.1f}s]")
        finally:
            try:
                env.close()
            except Exception:
                pass

    n_valid_total = sum(1 for r in all_results if r["valid"])
    print(f"\n=== Pipeline complete ===")
    print(f"Total scenes attempted: {len(all_results)}")
    print(f"Valid: {n_valid_total} ({100 * n_valid_total / max(len(all_results), 1):.1f}%)")
    print(f"Spawn fallbacks: {spawn_fallback_count}")
    print(f"Failures: {dict(failures.most_common())}")
    print(f"Wall time: {time.time() - t0:.1f}s")

    manifest_path = output_dir / "citymap_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Manifest saved to {manifest_path}")

    summary = {
        "n_maps": len(map_seeds),
        "n_blocks": n_blocks,
        "n_profiles": n_profiles,
        "spawn_velocities": spawn_velocities,
        "max_scenes_per_sign": max_scenes_per_sign_per_map,
        "total_attempted": len(all_results),
        "total_valid": n_valid_total,
        "valid_rate": round(n_valid_total / max(len(all_results), 1), 4),
        "spawn_fallback_count": spawn_fallback_count,
        "failures": dict(failures.most_common()),
        "wall_time_s": round(time.time() - t0, 1),
    }
    with open(output_dir / "citymap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_dir / 'citymap_summary.json'}")
    return all_results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-maps", type=int, default=5)
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument("--n-profiles", type=int, default=10)
    parser.add_argument("--max-per-sign", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="citymap_output")
    args = parser.parse_args()

    seeds = list(range(args.n_maps))
    run_pipeline(
        map_seeds=seeds,
        n_blocks=args.n_blocks,
        n_profiles=args.n_profiles,
        max_scenes_per_sign_per_map=args.max_per_sign,
        output_dir=args.output_dir,
    )
