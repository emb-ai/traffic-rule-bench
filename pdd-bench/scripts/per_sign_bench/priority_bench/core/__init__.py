"""Shared core for priority-junction benches (2.1 / 2.4)."""

from .manifest_config import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_AUX_LANES_OCCUPIED_MAX,
    DEFAULT_SPAWN_DISTANCE_BEFORE_END,
)
from .junction_priority_layout import (
    JunctionLayoutError,
    JunctionPriorityLayout,
    build_junction_priority_layout,
)
from .scene_augmentation import (
    SpawnScenario,
    ApproachSpawnLane,
    augment_layout_for_scene,
    enumerate_spawn_scenarios,
)

__all__ = [
    "DEFAULT_AUX_DISTANCE_FROM_INTERSECTION",
    "DEFAULT_AUX_LANES_OCCUPIED_MAX",
    "DEFAULT_SPAWN_DISTANCE_BEFORE_END",
    "JunctionLayoutError",
    "JunctionPriorityLayout",
    "build_junction_priority_layout",
    "SpawnScenario",
    "ApproachSpawnLane",
    "augment_layout_for_scene",
    "enumerate_spawn_scenarios",
]
