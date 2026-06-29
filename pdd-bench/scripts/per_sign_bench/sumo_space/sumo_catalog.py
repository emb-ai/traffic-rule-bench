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
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from .sumo_scene_enumerator import (
    SumoScene,
    enumerate_all_scenes,
    stratified_sample,
    filter_by_sign_codes,
    count_lanes_on_road,
)

# Signs whose scenes get a braking-spawn (ego starts above the limit, placed far
# enough before the sign to slow down). Speed-limit 3.24 + speed-zone signs:
# 5.21 (residential zone, fixed 20 km/h) and 5.31 (zone speed limit).
BRAKING_SPAWN_CODES = {"3.24", "5.21", "5.31"}

# Minimum-speed sign 4.6: built like a braking-spawn scene (ego placed upstream
# with an approach run, route through the sign), but MIRRORED — the ego starts
# BELOW the minimum and must ACCELERATE up to it instead of braking down.
ACCEL_SPAWN_CODES = {"4.6"}
ACCEL_DEFICIT_KMH = 15.0        # ego spawns this far BELOW the minimum speed
ACCEL_V0_FLOOR_KMH = 5.0        # but never slower than this
# Short upstream approach: unlike braking (ego must reach v_target BY the sign),
# for a MIN-speed sign the ego accelerates IN the zone (after the sign). So we
# only need it established at v0 just before the sign — a small fixed approach,
# leaving the post-sign edge as the zone where it must reach the minimum.
ACCEL_APPROACH_M = 8.0
MIN_SPEED_FLOOR_KMH = 35.0      # drop 4.6 scenes whose realistic min (road-10, cap) is below
                                # this — a min <= base cruise (~30) isn't discriminative


def bucket_limit_kmh(raw_kmh: float, selector: Optional[int] = None):
    """Snap a raw OSM speed limit to {20, 30, 50} for 3.24.

    <20 → 20 ; 20–40 → 30 ; 40–80 → 30 or 50 (even split) ; >80 → None (dropped).

    Final targets are {20, 30}: low enough that an unaware ego can VIOLATE them at a
    MODERATE, lane-holdable speed (≤40 km/h) — high speeds (needed to violate a 40
    limit) made the ego fly off curvy OSM roads (OOR → low success). 50 is a MARKER:
    the caller redistributes every former-50 scene EVENLY into 20/30 (round-robin).
    """
    if raw_kmh > 80:
        return None
    if raw_kmh < 21:
        return 20
    if raw_kmh <= 40:
        return 30
    # was 60 → even 30/50 split keyed off the per-scene selector (50 = redistributed)
    if selector is None:
        return 50
    return 30 if (selector % 2 == 0) else 50

BRAKE_DECEL_MPS2_DEFAULT = 3.5     # assumed braking decel (firm but achievable;
                                   # idm brakes harder than its 2.5 comfort decel).
                                   # Higher -> shorter d_brake -> shorter approach.
BRAKE_DELAY_S_DEFAULT = 0.5        # small reaction buffer (room to settle → fewer OOR/crashes)
BRAKE_MARGIN_M_DEFAULT = 3.0       # small buffer: reach v_target ~3 m before the sign
V0_MIN_EXCESS_MPS_DEFAULT = 2.0    # min amount v0 must exceed the limit
# Cap how far above the limit ego may spawn (km/h).
V0_MAX_EXCESS_KMH = 30.0
# Full braking distance (=1.0): a SIGN-AWARE agent has exactly enough room to
# brake from v0 to v_target by the sign and comply; a SIGN-UNAWARE agent that
# doesn't brake enters the zone at v0 > v_target and VIOLATES. This is the whole
# point of the scene — discriminate aware vs unaware, not test braking strength.
BRAKE_DIST_FACTOR = 1.0
# Ego's IDM desired speed (= profile MAX_SPEED ~14.3 m/s ≈ 51 km/h). v0 above this
# is shed instantly at spawn, so cap v0 here to keep the approach short & the entry
# speed realistic. The zone limit (v_target, 20/40) sits BELOW this cruising speed,
# so an unaware agent cruising at ~51 violates it.
EGO_DESIRED_SPEED_MPS = 11.0   # ~40 km/h: spawn/approach speed matches ego cruise
                               # (no spawn spike) and holds the lane (less OOR)
SPAWN_VELOCITY_MAX_MPS = 22.0      # nuPlan clip upper bound


@lru_cache(maxsize=4096)
def edge_speed_mps(net_path: str, road_id: str) -> float:
    """Max lane `speed` (m/s) of the named edge in a SUMO .net.xml, or 0.0.

    This is the value SpeedLimitSign reads from lane.speed; ×3.6 → km/h limit.
    """
    try:
        root = ET.parse(net_path).getroot()
    except (ET.ParseError, OSError):
        return 0.0
    for edge in root.findall("edge"):
        if edge.get("id") == road_id:
            speeds = [float(l.get("speed")) for l in edge.findall("lane") if l.get("speed")]
            if speeds:
                return max(speeds)
            s = edge.get("speed")
            return float(s) if s else 0.0
    return 0.0

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
    # Braking-spawn (3.24): ego starts above the limit, placed before the sign.
    n_v0_samples: int = 1,
    brake_decel_mps2: float = BRAKE_DECEL_MPS2_DEFAULT,
    brake_delay_s: float = BRAKE_DELAY_S_DEFAULT,
    brake_margin_m: float = BRAKE_MARGIN_M_DEFAULT,
    v0_min_excess_mps: float = V0_MIN_EXCESS_MPS_DEFAULT,
    v0_max_kmh: float = 80.0,
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
    scenes_root = Path(scenes_root)
    all_scenes = enumerate_all_scenes(scenes_root)
    if sign_categories:
        all_scenes = filter_by_sign_codes(all_scenes, sign_categories)
    sampled = stratified_sample(all_scenes, n_per_category=n_per_category, seed=seed)

    # Lazy import: only needed for braking-spawn codes (loads nuPlan sampler).
    _v0_sampler = None
    _braking_dist = None

    rows: List[dict] = []
    insufficient_v0 = 0
    dropped_high_limit = 0
    dropped_slow_min = 0   # 4.6 scenes whose road is too slow for a meaningful min
    fifty_rr = 0   # round-robin counter: former-50 (3.24) scenes -> even 20/40
    for scene in sampled:
        n_lanes = count_lanes_on_road(scene.net_path, scene.road_id)
        if n_lanes <= 0:
            lane_range = [0]
            n_lanes_field = 1
        else:
            lane_range = list(range(n_lanes))
            n_lanes_field = n_lanes

        is_braking = scene.sign_code in BRAKING_SPAWN_CODES
        is_accel = scene.sign_code in ACCEL_SPAWN_CODES
        is_spawn = is_braking or is_accel
        # Enforced speed limit (km/h) ego must brake to before the sign.
        v_target_kmh = 0.0
        v_target_raw_kmh = 0.0
        if is_accel:
            # 4.6 minimum-speed: target = min(road_speed - 10, achievable cap),
            # cap split 40 / 60. The minimum must sit ABOVE the base policies'
            # natural cruise (~30 km/h) so an unaware agent UNDER-shoots it
            # (violates) while a compliant agent accelerates up to it (obeys) —
            # the mirror of 3.24. Roads whose realistic min (<= road_speed-10)
            # falls below MIN_SPEED_FLOOR_KMH are dropped (a 20 km/h min on a
            # 30 km/h road isn't discriminative — base already complies).
            net_abs = str(scenes_root / scene.net_path)
            v_target_raw_kmh = round(edge_speed_mps(net_abs, scene.road_id) * 3.6)
            cap = 40 if (stable_hash(scene.scene_id, "min2040") % 2 == 0) else 60
            v_target_kmh = min(v_target_raw_kmh - 10, cap)
            if v_target_kmh < MIN_SPEED_FLOOR_KMH:
                dropped_slow_min += 1
                continue
        elif is_braking:
            if scene.sign_code == "5.21":
                # Residential zone: fixed 20 km/h, independent of the road's speed.
                v_target_kmh = v_target_raw_kmh = 20.0
            else:
                net_abs = str(scenes_root / scene.net_path)
                v_target_raw_kmh = round(edge_speed_mps(net_abs, scene.road_id) * 3.6)
                if scene.sign_code == "3.24":
                    bucketed = bucket_limit_kmh(
                        v_target_raw_kmh,
                        selector=stable_hash(scene.scene_id, "limit40_50"))
                    if bucketed is None:    # raw limit > 80 km/h → drop the scene
                        dropped_high_limit += 1
                        continue
                    # 50 ≈ ego cruise → unaware agent wouldn't violate. Redistribute
                    # every former-50 scene EVENLY into 20/40 (round-robin → exact
                    # even split, deterministic in sampled-iteration order).
                    if bucketed == 50:
                        bucketed = 20 if (fifty_rr % 2 == 0) else 30
                        fifty_rr += 1
                    v_target_kmh = bucketed
                else:
                    # Zone-limit signs (5.31): brake to the real zone limit.
                    v_target_kmh = v_target_raw_kmh

        for spawn_lane_num in lane_range:
            for var_idx in range(n_variations):
                base = {
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
                }
                if not is_spawn:
                    base["seed"] = stable_hash(scene.scene_id, spawn_lane_num, var_idx)
                    base["spawn_velocity_ms"] = 0.0
                    rows.append(base)
                    continue

                if is_accel:
                    # Acceleration-spawn (4.6): ego starts BELOW the minimum and
                    # must speed up. One row; placed d_required upstream at v0 so a
                    # compliant policy reaches the minimum by the zone.
                    seed = stable_hash(scene.scene_id, spawn_lane_num, var_idx)
                    v_target_mps = v_target_kmh / 3.6
                    v0 = max(ACCEL_V0_FLOOR_KMH / 3.6,
                             (v_target_kmh - ACCEL_DEFICIT_KMH) / 3.6)
                    # Short approach: ego accelerates IN the post-sign zone, so it
                    # only needs to be at v0 just before the sign.
                    d_req = ACCEL_APPROACH_M
                    row = dict(base)
                    row["seed"] = seed
                    row["spawn_velocity_ms"] = round(v0, 4)
                    row["v_target_kmh"] = v_target_kmh
                    row["v_target_raw_kmh"] = v_target_raw_kmh
                    row["d_required_m"] = round(d_req, 3)
                    row["sign_s"] = scene.distance_from_start
                    row["brake_decel_mps2"] = brake_decel_mps2
                    row["brake_delay_s"] = brake_delay_s
                    row["brake_margin_m"] = brake_margin_m
                    row["braking_spawn"] = True
                    row["spawn_mode"] = "accel"
                    rows.append(row)
                    continue

                # Braking-spawn (3.24): one row per v0 sample; ego starts above
                # the limit and is placed `d_required` before the sign.
                if _v0_sampler is None:
                    from factorized_space.agent_profile_bank import (
                        sample_spawn_velocity_above_limit as _v0s,
                        braking_required_distance as _bd,
                    )
                    _v0_sampler, _braking_dist = _v0s, _bd
                v_target_mps = v_target_kmh / 3.6
                for v_idx in range(max(1, n_v0_samples)):
                    vseed = stable_hash(scene.scene_id, spawn_lane_num, var_idx, "v0", v_idx)
                    # Cap v0 at the ego's DESIRED speed (else it's shed instantly
                    # at spawn) AND near the limit (short approach). The ego holds
                    # this speed in, then must brake for the zone.
                    v0_cap_mps = min(float(v0_max_kmh) / 3.6,
                                     (v_target_kmh + V0_MAX_EXCESS_KMH) / 3.6,
                                     EGO_DESIRED_SPEED_MPS)
                    v0 = _v0_sampler(vseed, v_target_mps,
                                     min_excess=v0_min_excess_mps,
                                     max_v=v0_cap_mps)
                    if v0 <= v_target_mps + 1e-6:   # never with the floored sampler
                        insufficient_v0 += 1
                        continue
                    # Spawn at a FRACTION of the full braking distance: baseline
                    # (decel 2.5) can't reach v_target by the sign -> overshoots
                    # into the zone (real violation); only harder braking complies.
                    d_req = BRAKE_DIST_FACTOR * _braking_dist(
                        v0, v_target_mps, brake_decel_mps2,
                        brake_delay_s, brake_margin_m)
                    row = dict(base)
                    row["v_idx"] = v_idx
                    row["seed"] = vseed
                    row["spawn_velocity_ms"] = round(v0, 4)
                    row["v_target_kmh"] = v_target_kmh
                    row["v_target_raw_kmh"] = v_target_raw_kmh
                    row["d_required_m"] = round(d_req, 3)
                    row["sign_s"] = scene.distance_from_start
                    row["brake_decel_mps2"] = brake_decel_mps2
                    row["brake_delay_s"] = brake_delay_s
                    row["brake_margin_m"] = brake_margin_m
                    row["braking_spawn"] = True
                    row["spawn_mode"] = "brake"
                    rows.append(row)

    if dropped_high_limit:
        print(f"[build_catalog] dropped {dropped_high_limit} scenes (raw limit > 80 km/h)")
    if dropped_slow_min:
        print(f"[build_catalog] dropped {dropped_slow_min} 4.6 scenes (road too slow for a min)")
    if insufficient_v0:
        print(f"[build_catalog] dropped {insufficient_v0} braking rows "
              f"(v0 could not exceed limit; cap {SPAWN_VELOCITY_MAX_MPS} m/s)")
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
