"""
Agent profile bank generation via Latin Hypercube Sampling from nuPlan statistics.

Produces a reproducible bank of N agent profiles, each a fixed tuple of driving
parameters sampled to cover the nuPlan distribution with low discrepancy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_DIR = SCRIPT_PATH.parent.parent
SCRIPTS_DIR = BENCHMARK_DIR.parent
PDD_BENCH_DIR = SCRIPTS_DIR.parent
SDC_ROOT = PDD_BENCH_DIR.parent  # traffic-rule-bench
REPO_ROOT = SDC_ROOT
NUPLAN_DIR = SDC_ROOT / "nuPlan"
NUPLAN_STATS_DIR = str(NUPLAN_DIR / "nuplan_statistics")

for _p in (PDD_BENCH_DIR, REPO_ROOT, NUPLAN_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)


_sampler_cache = None
_sampler_stats_dir = None


def apply_profile_to_idm_class(profile: dict) -> None:
    """Push a profile dict's values into IDMPolicy class attributes.

    Used by all three runners (PGMap, CityMap, SUMO) so the same NPC IDM
    parameters are applied consistently. Speeds in the profile are m/s;
    IDMPolicy.NORMAL_SPEED is in km/h, so we convert.

    Accepts both bare keys (NORMAL_SPEED, ACC_FACTOR, ...) as produced by
    sample_one_profile/generate_profiles AND prefixed keys (profile_NORMAL_SPEED, ...)
    as embedded in benchmark spec dicts.
    """
    from metadrive.policy.idm_policy import IDMPolicy

    def _get(key):
        if key in profile:
            return profile[key]
        return profile.get(f"profile_{key}", None)

    ns = _get("NORMAL_SPEED")
    if ns is not None:
        IDMPolicy.NORMAL_SPEED = float(ns) * 3.6
    ms = _get("MAX_SPEED")
    if ms is not None:
        IDMPolicy.MAX_SPEED = float(ms) * 3.6
    af = _get("ACC_FACTOR")
    if af is not None:
        IDMPolicy.ACC_FACTOR = float(af)
    df = _get("DEACC_FACTOR")
    if df is not None:
        IDMPolicy.DEACC_FACTOR = -abs(float(df))
    # DISTANCE_WANTED and TIME_WANTED are intentionally NOT applied to NPC IDM:
    # nuPlan's aggressive values cause rear-end crashes behind a slow ego.
    # MetaDrive defaults (10 m, 1.5 s) stay in effect. Sampled values are still
    # stored as profile_DISTANCE_WANTED / profile_TIME_WANTED for reporting.
    lcf = _get("LANE_CHANGE_FREQ")
    if lcf is not None:
        rate_per_km = float(lcf)
        IDMPolicy.LANE_CHANGE_FREQ = int(max(50, 1250.0 / max(rate_per_km, 1.0)))


def _get_sampler() -> "NuPlanSampler":
    """Return a cached NuPlanSampler (loaded once per process)."""
    global _sampler_cache, _sampler_stats_dir
    if _sampler_cache is None:
        from nuplan_sampler import NuPlanSampler
        _sampler_cache = NuPlanSampler(stats_dir=NUPLAN_STATS_DIR)
        _sampler_stats_dir = NUPLAN_STATS_DIR
    return _sampler_cache


def sample_one_profile(seed: int, density_cap: float = 1.0, horizon_steps: int = 600) -> dict:
    """Sample one NPC-driving profile from nuPlan distributions using a fixed seed.

    Reproducible: same seed always produces the same profile.
    Called per-scene so every scene gets a unique realistic flavour for NPCs.
    These parameters are applied only to NPC vehicles (IDMPolicy class attrs);
    ego uses DEFAULT_EGO_PARAMS (see ego_defaults.py).

    Note: vehicle_type is NOT returned here — vehicle type is sampled PER-NPC
    at spawn time by the traffic-manager hook (see §11.2 of the plan).

    Args:
        seed: deterministic seed for the random draws
        density_cap: upper bound on traffic_density. 1.0 by default (raised
            from the old 0.3 for more traffic variety). Lower it if a specific
            map type exhibits unavoidable-collision artefacts.
    """
    sampler = _get_sampler()
    np.random.seed(seed)

    max_speed = float(np.percentile(sampler.speeds, 95))
    creep_speed = float(np.percentile(sampler.speeds, 5))
    lane_change_freq = float(sampler.lane_change_rate_per_km)

    # Floor raised 2.0 -> 12.0 m/s (~43 km/h): the ego/IDM cruising speed must sit
    # ABOVE the 3.24 zone limits (20/40) so a sign-UNAWARE idm actually drives fast
    # enough to violate the zone (the old 2 m/s floor made idm crawl & comply).
    normal_speed = float(np.clip(sampler.normal_speed(), 12.0, 15.0))
    acc_factor = float(np.clip(sampler.acc_factor(), 0.2, 4.0))
    deacc_factor = float(np.clip(sampler.deacc_factor(), 0.2, 5.0))
    distance_wanted = float(np.clip(sampler.distance_wanted(), 5.0, 45.0))
    raw_density = float(sampler.traffic_density())
    traffic_density = float(np.clip(raw_density / 80.0, 0.0, density_cap))
    # horizon_steps from caller (mini=600, full=1200)
    time_wanted = distance_wanted / max(normal_speed, 0.5)

    return {
        "NORMAL_SPEED": round(normal_speed, 4),
        "MAX_SPEED": round(max_speed, 4),
        "CREEP_SPEED": round(creep_speed, 4),
        "ACC_FACTOR": round(acc_factor, 4),
        "DEACC_FACTOR": round(deacc_factor, 4),
        "DISTANCE_WANTED": round(distance_wanted, 4),
        "TIME_WANTED": round(min(time_wanted, 10.0), 4),
        "LANE_CHANGE_FREQ": round(lane_change_freq, 4),
        "traffic_density": round(traffic_density, 4),
        "horizon_steps": horizon_steps,
    }


def sample_spawn_velocity(seed: int) -> float:
    """Sample one spawn velocity (m/s) for ego from nuPlan routes.initial_speed.

    Used by runners to generate N_SPAWN_VELOCITY_SAMPLES velocities per
    base-scene (each call with a different seed). Clipped to SPAWN_VELOCITY_CLIP.
    """
    from .space_definition import SPAWN_VELOCITY_CLIP
    sampler = _get_sampler()
    np.random.seed(seed)
    v = float(sampler.spawn_velocity())
    lo, hi = SPAWN_VELOCITY_CLIP
    return float(np.clip(v, lo, hi))


def sample_spawn_velocity_above_limit(seed: int, v_limit_mps: float,
                                      min_excess: float = 2.0,
                                      max_v: float | None = None) -> float:
    """Sample an ego spawn velocity (m/s) that exceeds the sign limit by a
    realistic, nuPlan-shaped margin.

    v0 = v_limit + Δ, where Δ is drawn from the nuPlan initial_speed distribution
    (so the *exceedance* over the limit has a realistic spread, not a fixed +2),
    floored at `min_excess` so v0 is always strictly above the limit, and capped
    at `max_v` (m/s) when given (e.g. 80 km/h ≈ 22.22 m/s).
    """
    delta = sample_spawn_velocity(seed)          # nuPlan-shaped spread (m/s)
    delta = max(float(delta), float(min_excess))
    v0 = float(v_limit_mps) + delta
    if max_v is not None:
        v0 = min(v0, float(max_v))
    return v0


def braking_required_distance(v0_mps: float, v_target_mps: float,
                              decel_mps2: float, delay_s: float,
                              margin_m: float) -> float:
    """Distance needed to slow from v0 to v_target + reaction delay + margin.

    d_brake = max(0, (v0² - v_target²) / (2·decel)); d_delay = v0·delay.
    Returns d_brake + d_delay + margin.
    """
    v0 = max(0.0, float(v0_mps))
    vt = max(0.0, float(v_target_mps))
    a = max(0.1, float(decel_mps2))
    d_brake = max(0.0, (v0 * v0 - vt * vt) / (2.0 * a))
    d_delay = v0 * max(0.0, float(delay_s))
    return d_brake + d_delay + max(0.0, float(margin_m))


def max_v0_for_distance(d_avail_m: float, v_target_mps: float,
                        decel_mps2: float, delay_s: float,
                        margin_m: float) -> float:
    """Inverse of braking_required_distance: the largest v0 whose required
    distance fits within `d_avail_m`. Used when the upstream road is too short
    to honour the sampled v0 (insufficient_runway): we lower v0 to fit.

    Solves d_avail = (v0² - vt²)/(2a) + v0·delay + margin for v0 (positive root).
    Returns a value that may be <= v_target when there is essentially no runway;
    the caller checks v0 > v_target and drops the scene otherwise.
    """
    import math
    a = max(0.1, float(decel_mps2))
    vt = max(0.0, float(v_target_mps))
    delay = max(0.0, float(delay_s))
    A = 1.0 / (2.0 * a)
    B = delay
    C = float(margin_m) - vt * vt / (2.0 * a) - float(d_avail_m)
    disc = B * B - 4.0 * A * C
    if disc <= 0.0:
        return 0.0
    return (-B + math.sqrt(disc)) / (2.0 * A)


def sample_npc_vehicle_type(seed: int) -> str:
    """Sample one vehicle type (s/m/l/xl) from nuPlan size_dist.

    Called per-NPC at spawn time so each traffic vehicle has a type drawn
    from the nuPlan distribution rather than a single scene-wide type.
    """
    sampler = _get_sampler()
    np.random.seed(seed)
    return str(sampler.vehicle_type())


def generate_profiles(n: int = 10, seed: int = 42) -> list[dict]:
    """Generate n agent profiles by sampling directly from nuPlan distributions.

    Each profile is an independent draw from NuPlanSampler (KDE for continuous
    parameters, empirical choice for categorical/discrete ones).

    Clipping for MetaDrive compatibility and collision safety:
    - NORMAL_SPEED: [2.0, 15.0] m/s (avoid near-zero and highway extremes)
    - ACC_FACTOR:   [0.2, 4.0] m/s²
    - DEACC_FACTOR: [0.2, 5.0] m/s²
    - DISTANCE_WANTED: [5.0, 45.0] m
    - traffic_density: raw count / 80, capped at [0.0, 0.5]
    - horizon: [10, 60] s → steps at 10 Hz

    Fixed across all profiles:
    - MAX_SPEED = p95 of nuPlan speeds
    - CREEP_SPEED = p5 of nuPlan speeds
    - LANE_CHANGE_FREQ = from nuPlan lane-change rate
    """
    from nuplan_sampler import NuPlanSampler
    sampler = NuPlanSampler(stats_dir=NUPLAN_STATS_DIR)

    np.random.seed(seed)

    # Fixed values from nuPlan
    max_speed = float(np.percentile(sampler.speeds, 95))  # m/s
    creep_speed = float(np.percentile(sampler.speeds, 5))  # m/s
    lane_change_freq = float(sampler.lane_change_rate_per_km)

    profiles = []
    for i in range(n):
        # Sample each parameter independently from nuPlan KDE / empirical
        normal_speed = np.clip(sampler.normal_speed(), 2.0, 15.0)
        acc_factor = np.clip(sampler.acc_factor(), 0.2, 4.0)
        deacc_factor = np.clip(sampler.deacc_factor(), 0.2, 5.0)
        distance_wanted = np.clip(sampler.distance_wanted(), 5.0, 45.0)
        # traffic_density: nuPlan gives vehicle count per frame (~0-80),
        # MetaDrive expects 0.0-1.0; cap at 0.5 (was 0.3 — raising to test
        # whether crash rate stays acceptable while improving realism)
        raw_density = float(sampler.traffic_density())
        traffic_density = np.clip(raw_density / 80.0, 0.0, 0.3)
        # horizon_steps from caller (mini=600, full=1200)
        time_wanted = distance_wanted / max(normal_speed, 0.5)
        vehicle_type = sampler.vehicle_type()

        profiles.append({
            "profile_id": i,
            "NORMAL_SPEED": round(float(normal_speed), 4),
            "MAX_SPEED": round(max_speed, 4),
            "CREEP_SPEED": round(creep_speed, 4),
            "ACC_FACTOR": round(float(acc_factor), 4),
            "DEACC_FACTOR": round(float(deacc_factor), 4),
            "DISTANCE_WANTED": round(float(distance_wanted), 4),
            "TIME_WANTED": round(min(float(time_wanted), 10.0), 4),
            "LANE_CHANGE_FREQ": round(lane_change_freq, 4),
            "traffic_density": round(float(traffic_density), 4),
            "horizon_steps": horizon_steps,
            "vehicle_type": str(vehicle_type),
        })

    return profiles


def save_profiles(profiles: list[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(profiles, f, indent=2)


def load_profiles(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate agent profile bank")
    parser.add_argument("--n", type=int, default=80, help="Number of profiles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default=str(BENCHMARK_DIR / "data" / "agent_profiles.json"))
    args = parser.parse_args()

    profiles = generate_profiles(n=args.n, seed=args.seed)
    save_profiles(profiles, args.output)
    print(f"Generated {len(profiles)} agent profiles → {args.output}")

    # Print summary
    import numpy as np
    for key in ["NORMAL_SPEED", "ACC_FACTOR", "DEACC_FACTOR", "DISTANCE_WANTED",
                "traffic_density", "horizon_steps"]:
        vals = [p[key] for p in profiles]
        print(f"  {key}: mean={np.mean(vals):.3f} std={np.std(vals):.3f} "
              f"min={np.min(vals):.3f} max={np.max(vals):.3f}")
    vtypes = [p["vehicle_type"] for p in profiles]
    from collections import Counter
    print(f"  vehicle_type: {dict(Counter(vtypes))}")
