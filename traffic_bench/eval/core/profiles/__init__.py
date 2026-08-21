"""nuPlan-derived ego/NPC driving profiles for priority_bench.

Vendored from ``per_sign_bench/factorized_space/`` (agent_profile_bank +
ego_defaults) plus ``stable_hash`` from ``sumo_space.sumo_catalog``.
"""

from .agent_profile_bank import (  # noqa: F401
    apply_profile_to_idm_class,
    braking_required_distance,
    max_v0_for_distance,
    sample_one_profile,
    sample_spawn_velocity,
    sample_spawn_velocity_above_limit,
)
from .ego_defaults import (  # noqa: F401
    DEFAULT_EGO_PARAMS,
    apply_ego_defaults,
    apply_ego_sampled,
    numpy_legacy_seed,
    sample_ego_params,
)
from .stable_hash import stable_hash  # noqa: F401

__all__ = [
    "DEFAULT_EGO_PARAMS",
    "apply_ego_defaults",
    "apply_ego_sampled",
    "apply_profile_to_idm_class",
    "braking_required_distance",
    "max_v0_for_distance",
    "numpy_legacy_seed",
    "sample_ego_params",
    "sample_one_profile",
    "sample_spawn_velocity",
    "sample_spawn_velocity_above_limit",
    "stable_hash",
]
