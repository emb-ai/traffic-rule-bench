"""
Ego-vehicle driving parameters that are NOT affected by the per-scene nuPlan
profile. The NPC profile (sampled by agent_profile_bank.sample_one_profile) is
pushed into IDMPolicy class attributes globally, which would normally bleed
into ego if ego's policy also reads those attributes. Applying these values on
the ego-policy instance AFTER reset isolates ego from NPC flavour so the
benchmark measures policy behaviour under a fixed ego driver style.

Speeds are in m/s here; IDMPolicy.NORMAL_SPEED is stored in km/h (× 3.6).
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


def sample_ego_params(seed: int) -> dict:
    """Sample ego IDM params from nuPlan distributions (no safety clipping).

    Returns a dict in the same shape as DEFAULT_EGO_PARAMS so it can be fed
    into apply_ego_sampled(). Reproducible via the seed argument.

    Unlike sample_one_profile() (which suppresses DISTANCE_WANTED/TIME_WANTED
    for NPCs to avoid rear-end crashes behind a slow ego), this sampler exposes
    the full nuPlan distribution for ego itself — used during trajectory
    recording for IDM diversity. Bad samples are filtered out by the oracle
    pass over multiple rollouts; we deliberately do NOT clip here.
    """
    from .agent_profile_bank import _get_sampler
    import numpy as np

    sampler = _get_sampler()
    # Save/restore global numpy RNG so seeding here doesn't pollute the caller's
    # RNG state (run_one_episode sets np.random.seed(scene_seed) earlier and
    # downstream stochastic helpers — _spawn_cyclists_on_lane, the policy
    # itself — must keep deriving randomness from that scene_seed).
    saved_state = np.random.get_state()
    try:
        np.random.seed(numpy_legacy_seed(seed))
        normal_speed = float(sampler.normal_speed())
        distance_wanted = float(sampler.distance_wanted())
        safe_normal = max(normal_speed, 0.5)
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
