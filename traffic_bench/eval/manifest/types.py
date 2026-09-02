"""Hydra-resolved knobs passed to ``signs.<group>.expand.generate``."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from traffic_bench.eval.engine.expand.manifest_config import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    DEFAULT_STOP_WAIT_STEPS,
)
from traffic_bench.eval.engine.expand.manifest_expansion import ExpansionConfig
from traffic_bench.eval.engine.spawn.auxiliary_agent import DEFAULT_CONVOY_GAP_M
from traffic_bench.eval.sign_registry import SignProfile


@dataclass
class PathsConfig:
    scenes_dir: Optional[str] = None
    output_base: Optional[str] = None
    experiment_name: Optional[str] = None
    split: str = "debug"


@dataclass
class ScenarioConfig:
    max_scenarios: Optional[int] = None
    max_total: Optional[int] = None
    min_dual_path_gain_m: float = 20.0


@dataclass
class AugmentationAxesConfig:
    enabled: bool = True
    layout: bool = False
    auxiliary: bool = False


@dataclass
class SimulationConfig:
    spawn_velocity_ms: float = 2.5
    traffic_density: float = 0.0
    horizon: int = 600
    sign_distance_before_end: float = 0.0
    spawn_distance_before_end: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END
    destination_max_along_m: Optional[float] = None
    sign_distance_from_start: float = 10.0
    n_variations: int = 3
    profile_density_cap: float = 1.0
    compliant_stop_success_seconds: float = 3.0
    compliant_stop_max_dist_m: float = 12.0
    compliant_stop_speed_mps: float = 0.5
    min_hops_after_depart: int = 0
    spawn_offset_from_start: float = 10.0
    max_path_length_m: float = 150.0
    max_path_length_levels: Tuple[float, ...] = (130.0, 150.0, 170.0)
    max_ego_lanes: int = 8
    zone_tail_m: float = 8.0
    zone_min_m: float = 20.0


@dataclass
class ExpertConfig:
    stop_wait_steps: int = DEFAULT_STOP_WAIT_STEPS


@dataclass
class AuxiliaryConfig:
    enabled: bool = True
    distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION
    convoy_size: int = 1
    convoy_gap_m: float = DEFAULT_CONVOY_GAP_M
    lanes_occupied: int = 1
    release_when_ego_within_m: float = 15.0


@dataclass
class GifConfig:
    enabled: bool = False
    policy: str = "idm"
    max_scenes: Optional[int] = None
    dry_run: bool = False
    hide_signs: bool = True
    dir: Optional[str] = None
    run_name: Optional[str] = None
    window_m: float = 80.0
    draw_path_conflict: bool = False
    model_path: Optional[str] = None


@dataclass
class ManifestConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    augmentation: AugmentationAxesConfig = field(default_factory=AugmentationAxesConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    auxiliary: AuxiliaryConfig = field(default_factory=AuxiliaryConfig)
    gif: GifConfig = field(default_factory=GifConfig)


@dataclass
class GenerateCfg:
    """Resolved job handed to ``signs.<group>.expand.generate``."""

    profile: SignProfile
    scenes_dir: Path
    output_dir: Path
    split: str
    scenario: ScenarioConfig
    simulation: SimulationConfig
    expansion: ExpansionConfig
    auxiliary: AuxiliaryConfig
    expert: ExpertConfig
    max_ego_lanes: int = 3
    max_pedestrian_presets: int = 3
    crosswalk_positions: Optional[List[str]] = None
    ped_cfg: Optional[Dict[str, Any]] = None
