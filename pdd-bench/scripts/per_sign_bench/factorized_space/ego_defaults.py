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


def sample_ego_params(seed: int, mode: str | None = None) -> dict:
    """Sample ego IDM params from nuPlan statistics.

    Два режима (env `EGO_SAMPLER`, default "legacy"):

    "legacy" — исторический сэмплер: KDE по мгновенным величинам ВСЕХ кадров
    nuPlan. Верен CSV, но кадровые состояния ≠ стили вождения: 60% ego получают
    крейсер < 30 км/ч (стояние на светофорах попадает в желаемую скорость),
    DISTANCE_WANTED берётся из дистанций следования на ходу (а s0 в IDM — зазор
    в остановке), 21% агентов получают ACC < 0.3 м/с² и не могут тронуться.

    "styles" — семантические фиксы (reports/idm_sampling_research.md, рек. 2;
    полноценные per-track персоны — отдельный этап, план 2c):
      NORMAL_SPEED    — перцентиль U[50,95] движущихся кадров (>2 м/с): желаемый
                        крейсер = верх наблюдаемых скоростей, а не простой;
      MAX_SPEED       — max(глобальный p95, 1.2×NORMAL_SPEED), не константа;
      ACC/DEACC       — floor 0.5 м/с² (убирает полумёртвых), кап 4/5;
      DISTANCE_WANTED — ранг nuPlan-дистанции через эмпирическую CDF → s0 ∈ [2,10] м;
      TIME_WANTED     — тот же ранг «осторожности» → headway ∈ [0.8,2.5] с
                        (коррелирован с s0 и не зависит от медленности агента);
      LANE_CHANGE_FREQ— дефолт MetaDrive 200 (оценка 62 смены/км из
                        lane_changes.csv — артефакт детектора).

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
    try:
        np.random.seed(seed)
        if mode != "styles":
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

        moving = sampler.speeds[sampler.speeds > 2.0]
        speed_rank = float(np.random.uniform(0.50, 0.95))
        normal_speed = float(np.percentile(moving, speed_rank * 100.0))
        # Ранг «осторожности» из nuPlan-дистанции следования: сохраняем форму
        # распределения через эмпирическую CDF, но целимся в IDM-семантику s0/headway.
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
