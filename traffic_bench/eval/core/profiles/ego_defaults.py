"""
Ego-vehicle driving parameters that are NOT affected by the per-scene nuPlan
profile. The NPC profile (sampled by agent_profile_bank.sample_one_profile) is
pushed into IDMPolicy class attributes globally, which would normally bleed
into ego if ego's policy also reads those attributes. Applying these values on
the ego-policy instance AFTER reset isolates ego from NPC flavour so the
benchmark measures policy behaviour under a fixed ego driver style.

Speeds are in m/s here; IDMPolicy.NORMAL_SPEED is stored in km/h (× 3.6).

Vendored into ``priority_bench/core/profiles/`` from
``per_sign_bench/factorized_space/ego_defaults.py``.
"""

from __future__ import annotations

from typing import Any

DEFAULT_EGO_PARAMS = {
    "NORMAL_SPEED": 10.0,        # m/s (~36 km/h)
    "MAX_SPEED": 15.0,           # m/s (~54 km/h)
    "CREEP_SPEED": 1.0,          # m/s
    "ACC_FACTOR": 1.5,           # m/s^2
    "DEACC_FACTOR": 2.5,         # m/s^2 (stored as negative on IDMPolicy)
    "DISTANCE_WANTED": 10.0,     # m
    "TIME_WANTED": 1.5,          # s
    "LANE_CHANGE_FREQ": 200,     # cooldown in sim steps
}


_NUMPY_LEGACY_SEED_MOD = 2**32


def numpy_legacy_seed(seed: int) -> int:
    """Map any integer seed into the range accepted by ``np.random.seed``."""
    return int(seed) % _NUMPY_LEGACY_SEED_MOD


def apply_ego_defaults(ego_policy: Any) -> None:
    """Override ego-policy IDM parameters on the instance level.

    Call after env.reset() so the policy has been instantiated. Only touches
    the instance's attributes — does not change the class-level IDMPolicy
    attrs (those carry the NPC profile).

    If ego_policy is not IDM-based, this is a no-op.
    """
    for key, value in DEFAULT_EGO_PARAMS.items():
        if key == "NORMAL_SPEED" or key == "MAX_SPEED":
            setattr(ego_policy, key, float(value) * 3.6)
        elif key == "DEACC_FACTOR":
            setattr(ego_policy, key, -abs(float(value)))
        else:
            setattr(ego_policy, key, value)


EGO_SPEED_RANK_BANDS = {  # variant_k -> percentile band of moving-frame speeds
    1: (0.10, 0.35),   # slow
    2: (0.35, 0.60),   # medium
    3: (0.60, 0.85),   # brisk
    4: (0.85, 0.99),   # fast
}


def sample_ego_params(seed: int, mode: str | None = None,
                      variant_k: int | None = None) -> dict:
    """Sample ego IDM params from nuPlan statistics.

    Two modes (env `EGO_SAMPLER`, default "legacy"):

    "legacy" — the historical sampler: KDE over instantaneous values of ALL
    nuPlan frames. Faithful to the CSV, but frame states != driving styles:
    60% of egos get a cruise < 30 km/h (traffic-light standstills leak into
    the desired speed), DISTANCE_WANTED comes from in-motion following
    distances (while IDM's s0 is a standstill gap), and 21% of agents get
    ACC < 0.3 m/s^2 and cannot pull away.

    "styles" — semantic fixes (reports/idm_sampling_research.md, rec. 2;
    full per-track personas are a separate stage, plan 2c):
      NORMAL_SPEED    — percentile of moving frames (>2 m/s); with variant_k
                        set, drawn from that variant's quantile BAND so every
                        scene sees both slow and fast styles:
                          s1 [10,35]% ≈ 11–22 km/h   s2 [35,60]% ≈ 22–32 km/h
                          s3 [60,85]% ≈ 32–44 km/h   s4 [85,99]% ≈ 44–62 km/h
                        without variant_k — the full range [10,99]% (11–62 km/h);
      MAX_SPEED       — max(global p95, 1.2×NORMAL_SPEED), not a constant;
      ACC/DEACC       — floor 0.5 m/s^2 (removes half-dead agents), cap 4/5;
      DISTANCE_WANTED — nuPlan-distance rank via the empirical CDF → s0 ∈ [2,10] m;
      TIME_WANTED     — the same "caution" rank → headway ∈ [0.8,2.5] s
                        (correlated with s0, independent of how slow the agent is);
      LANE_CHANGE_FREQ— MetaDrive default 200 (the 62 changes/km estimate from
                        lane_changes.csv is a detector artifact).

    Reproducible via the seed argument. Returns the DEFAULT_EGO_PARAMS shape
    for apply_ego_sampled().
    """
    import os

    from .agent_profile_bank import _get_sampler
    import numpy as np

    if mode is None:
        mode = os.environ.get("EGO_SAMPLER", "legacy").strip().lower()

    sampler = _get_sampler()
    # Save/restore global numpy RNG so seeding here doesn't pollute the caller's
    # RNG state (run_one_episode sets np.random.seed(scene_seed) earlier and
    # downstream stochastic helpers — _spawn_cyclists_on_lane, the policy
    # itself — must keep deriving randomness from that scene_seed).
    saved_state = np.random.get_state()
    np.random.seed(numpy_legacy_seed(seed))
    normal_speed = float(sampler.normal_speed())
    distance_wanted = float(sampler.distance_wanted())
    safe_normal = max(normal_speed, 0.5)
    try:
        if mode != "styles":
            return {
                "NORMAL_SPEED": normal_speed,
                "MAX_SPEED": float(np.percentile(sampler.speeds, 95)),
                "CREEP_SPEED": float(np.percentile(sampler.speeds, 5)),
                "ACC_FACTOR": float(sampler.acc_factor()),
                "DEACC_FACTOR": float(sampler.deacc_factor()),
                "DISTANCE_WANTED": distance_wanted,
                "TIME_WANTED": float(min(distance_wanted / safe_normal, 10.0)),
                "LANE_CHANGE_FREQ": int(max(50, 1250.0 / max(float(sampler.lane_change_rate_per_km), 1.0))),
            }

        moving = sampler.speeds[sampler.speeds > 2.0]
        lo, hi = EGO_SPEED_RANK_BANDS.get(variant_k, (0.10, 0.99))
        speed_rank = float(np.random.uniform(lo, hi))
        normal_speed = float(np.percentile(moving, speed_rank * 100.0))
        # "Caution" rank from the nuPlan following distance: keep the shape of
        # the distribution via the empirical CDF while targeting IDM s0/headway
        # semantics.
        dw_raw = float(sampler.distance_wanted())
        caution_rank = float((sampler.following < dw_raw).mean())
        return {
            "NORMAL_SPEED": normal_speed,
            "MAX_SPEED": float(max(np.percentile(sampler.speeds, 95), 1.2 * normal_speed)),
            "CREEP_SPEED": float(np.percentile(sampler.speeds, 5)),
            "ACC_FACTOR": float(np.clip(sampler.acc_factor(), 0.5, 4.0)),
            "DEACC_FACTOR": float(np.clip(sampler.deacc_factor(), 0.5, 5.0)),
            "DISTANCE_WANTED": float(2.0 + 8.0 * caution_rank),
            "TIME_WANTED": float(0.8 + 1.7 * caution_rank),
            "LANE_CHANGE_FREQ": 200,
        }
    finally:
        np.random.set_state(saved_state)


def apply_ego_sampled(ego_policy: Any, params: dict) -> None:
    """Override ego-policy IDM params from a sampled dict (sister of apply_ego_defaults).

    Speed-like keys in the sampled dict are in m/s (consistent with
    sample_one_profile output) and are converted to km/h on the policy
    instance, matching IDMPolicy's km/h-based class attrs.
    """
    for key, value in params.items():
        if key in ("NORMAL_SPEED", "MAX_SPEED", "CREEP_SPEED"):
            setattr(ego_policy, key, float(value) * 3.6)
        elif key == "DEACC_FACTOR":
            setattr(ego_policy, key, -abs(float(value)))
        else:
            setattr(ego_policy, key, value)
