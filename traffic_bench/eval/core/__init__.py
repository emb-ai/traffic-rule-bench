"""Compatibility package. Engine lives in ``eval.lib``; family rules in ``eval.signs``.

See ``core/README.md``. Prefer ``traffic_bench.eval.lib.*`` in new code.
"""

from .manifest.manifest_config import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_AUX_LANES_OCCUPIED_MAX,
    DEFAULT_DESTINATION_MAX_ALONG_M,
    DEFAULT_SPAWN_DISTANCE_BEFORE_END,
)
from .layout.junction_priority_layout import (
    JunctionLayoutError,
    JunctionPriorityLayout,
    build_junction_priority_layout,
)
from .scenarios.scene_augmentation import (
    ApproachSpawnLane,
    SpawnScenario,
    augment_layout_for_scene,
    enumerate_spawn_scenarios,
)

__all__ = [
    "DEFAULT_AUX_DISTANCE_FROM_INTERSECTION",
    "DEFAULT_AUX_LANES_OCCUPIED_MAX",
    "DEFAULT_DESTINATION_MAX_ALONG_M",
    "DEFAULT_SPAWN_DISTANCE_BEFORE_END",
    "JunctionLayoutError",
    "JunctionPriorityLayout",
    "build_junction_priority_layout",
    "SpawnScenario",
    "ApproachSpawnLane",
    "augment_layout_for_scene",
    "enumerate_spawn_scenarios",
]
