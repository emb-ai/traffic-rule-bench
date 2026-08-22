"""Shared spawn types and runtime aux helpers."""

from .scene_augmentation import (
    SpawnScenario,
    SpawnStrategy,
    augment_layout_for_scene,
    enumerate_spawn_scenarios,
)

__all__ = [
    "SpawnScenario",
    "SpawnStrategy",
    "augment_layout_for_scene",
    "enumerate_spawn_scenarios",
]
