"""Predefined pedestrian crossing benchmark scenarios (PDD 5.19)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal

SpawnChainMode = Literal["time_gap", "after_previous"]

MAX_PEDESTRIAN_PRESETS = 5


@dataclass(frozen=True)
class PedestrianPreset:
    id: int
    name: str
    target_pedestrian_count: int
    ego_spawn_distance_m: float | None = None
    pedestrian_spawn_gap_s: float | None = None
    pedestrian_spawn_chain: SpawnChainMode = "time_gap"
    speed_mean: float | None = None
    speed_std: float | None = None

    def describe(self) -> str:
        lines = {
            1: "1 pedestrian spawns when ego is 1 m before crosswalk, walks at slow speed",
            2: "4 pedestrians chained with 2.5 s gap, slow speed",
            3: "3 pedestrians chained with 3.5 s gap, slow speed",
            4: "2 pedestrians; second spawns after the first finishes crossing",
            5: "3 pedestrians; each next spawns after the previous finishes crossing",
        }
        return lines.get(self.id, self.name)


def build_pedestrian_presets(
    *,
    default_ego_spawn_distance_m: float,
    default_speed_mean: float,
    default_speed_std: float,
    default_spawn_gap_s: float,
) -> Dict[int, PedestrianPreset]:
    slow_speed = max(0.3, float(default_speed_mean) / 4.0)
    slow_std = max(0.05, float(default_speed_std) / 2.0)
    return {
        1: PedestrianPreset(
            id=1,
            name="late_slow_single",
            target_pedestrian_count=1,
            ego_spawn_distance_m=10.0,
            speed_mean=slow_speed,
            speed_std=slow_std,
        ),
        2: PedestrianPreset(
            id=2,
            name="four_gap_35",
            target_pedestrian_count=4,
            ego_spawn_distance_m=default_ego_spawn_distance_m,
            pedestrian_spawn_gap_s=3.5,
            speed_mean=slow_speed,
            speed_std=slow_std,
        ),
        3: PedestrianPreset(
            id=3,
            name="three_gap_55",
            target_pedestrian_count=3,
            ego_spawn_distance_m=default_ego_spawn_distance_m,
            pedestrian_spawn_gap_s=5.5,
            speed_mean=slow_speed,
            speed_std=slow_std,
        ),
        4: PedestrianPreset(
            id=4,
            name="two_gap_75",
            target_pedestrian_count=2,
            ego_spawn_distance_m=default_ego_spawn_distance_m,
            pedestrian_spawn_gap_s=7.5,
            speed_mean=slow_speed,
            speed_std=slow_std,
        ),
        5: PedestrianPreset(
            id=5,
            name="three_after_previous",
            target_pedestrian_count=3,
            ego_spawn_distance_m=default_ego_spawn_distance_m,
            speed_mean=slow_speed,
            speed_std=slow_std,
            pedestrian_spawn_chain="after_previous",
        ),
    }


def list_pedestrian_presets(num_presets: int, **defaults) -> List[PedestrianPreset]:
    count = min(MAX_PEDESTRIAN_PRESETS, max(1, int(num_presets)))
    presets = build_pedestrian_presets(
        default_ego_spawn_distance_m=float(defaults.get("default_ego_spawn_distance_m", 15.0)),
        default_speed_mean=float(defaults.get("default_speed_mean", 1.2)),
        default_speed_std=float(defaults.get("default_speed_std", 0.2)),
        default_spawn_gap_s=float(defaults.get("default_spawn_gap_s", 2.5)),
    )
    return [presets[i] for i in range(1, count + 1)]


def pedestrian_manager_from_preset(
    preset: PedestrianPreset,
    *,
    default_ego_spawn_distance_m: float,
    default_speed_mean: float,
    default_speed_std: float,
    default_spawn_gap_s: float,
    yield_distance: float,
    no_stop_before_crosswalk_m: float,
) -> dict[str, Any]:
    count = max(1, int(preset.target_pedestrian_count))
    ego_spawn_distance_m = (
        float(preset.ego_spawn_distance_m)
        if preset.ego_spawn_distance_m is not None
        else float(default_ego_spawn_distance_m)
    )
    spawn_gap_s = (
        float(preset.pedestrian_spawn_gap_s)
        if preset.pedestrian_spawn_gap_s is not None
        else float(default_spawn_gap_s)
    )
    speed_mean = float(preset.speed_mean if preset.speed_mean is not None else default_speed_mean)
    speed_std = float(preset.speed_std if preset.speed_std is not None else default_speed_std)
    return {
        "enabled": True,
        "spawn_mode": "ego_proximity",
        "ego_spawn_distance_m": ego_spawn_distance_m,
        "target_pedestrian_count": count,
        "pedestrian_spawn_gap_s": spawn_gap_s,
        "pedestrian_spawn_chain": preset.pedestrian_spawn_chain,
        "initial_pedestrians": 0,
        "max_pedestrians": count,
        "spawn_by_interval": False,
        "spawn_probability": 0.0,
        "crossing_interval_range": [5.0, 10.0],
        "max_active_per_crosswalk": count,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "yield_distance": float(yield_distance),
        "no_stop_before_crosswalk_m": float(no_stop_before_crosswalk_m),
        "yield_to_vehicles": True,
        "yield_on_crosswalk": False,
    }
