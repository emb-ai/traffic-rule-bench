#!/usr/bin/env python3
"""Generate evaluation manifest from scenes.

Shared discover / write helpers live in ``eval.manifest``. Family row
builders live in ``eval.signs.<family>.expand``. This module is the Hydra
entry and the per-family ``generate_*_manifest`` shells.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

EVAL_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = EVAL_DIR
PACKAGE_DIR = EVAL_DIR.parent
REPO_ROOT = PACKAGE_DIR.parent
PDD_BENCH_DIR = PACKAGE_DIR
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

from traffic_bench.eval.lib.runtime.checkpoints import (
    DEFAULT_MODEL_PATHS,
    NN_NEED_CHECKPOINT,
    resolve_nn_checkpoint,
)

from traffic_bench.eval.lib.layout.junction_priority_layout import (
    JunctionLayoutError,
    allowed_shapes_for_mode,
    build_junction_priority_layout,
)
from traffic_bench.eval.lib.scenarios.auxiliary_agent import DEFAULT_CONVOY_GAP_M
from traffic_bench.eval.lib.manifest.manifest_config import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    DEFAULT_STOP_WAIT_STEPS,
)
from traffic_bench.eval.lib.manifest.manifest_expansion import (
    AuxiliaryParams,
    ExpansionConfig,
)
from traffic_bench.eval.signs.blocked.expand import (
    BlockedRoadExpansionConfig,
    BlockedRoadSimParams,
    build_blocked_road_manifest_entry,
    expand_blocked_road_scene_entries,
)
from traffic_bench.eval.signs.dual_path.expand import (
    DualPathExpansionConfig,
    DualPathSimParams,
    build_dual_path_manifest_entry,
    expand_dual_path_scene_entries,
)
from traffic_bench.eval.signs.crosswalk.expand import (
    CrosswalkExpansionConfig,
    CrosswalkSimParams,
    DEFAULT_POSITIONS,
    discover_segment_crosswalk_scenes,
    expand_crosswalk_scene_entries,
)
from traffic_bench.eval.signs.detour.expand import (
    DetourExpansionConfig,
    DetourSimParams,
    discover_segment_detour_scenes,
    expand_detour_scene_entries,
)
from traffic_bench.eval.signs.speed.expand import (
    SpeedExpansionConfig,
    SpeedSimParams,
    discover_segment_speed_scenes,
    expand_speed_scene_entries,
)
from traffic_bench.eval.signs.speed.spec import assign_limit_kmh
from traffic_bench.eval.signs.junction.expand import (
    build_manifest_entry,
    expand_scene_entries,
)
from traffic_bench.eval.lib.layout.roundabout_topology import build_roundabout_layout
from traffic_bench.scene_collection.sign_scenes.materialize.pool_index import (
    normalize_split,
)
from traffic_bench.eval.sign_registry import (
    STOP,
    SignProfile,
    get_profile,
    scenes_dir as profile_scenes_dir,
    output_dir as profile_output_dir,
)
from traffic_bench.eval.manifest.io import (
    append_scene_entries,
    apply_max_total,
    apply_split_filter,
    assert_rejected_scenes_applied,
    discover_scenes,
    load_scene_metadata,
    write_real_manifest,
    write_repro_artifacts,
)
from traffic_bench.eval.manifest.lanes import (
    SumoLaneInfo,
    filter_spawn_lanes_to_secondary,
    parse_sumo_net_for_spawn_lanes,
    select_random_spawn_lane,
)

RUN_BENCH_SCRIPT = SCRIPT_DIR / "run_benchmark.py"

# Set by main() from Hydra `sign=` before generation runs.
PROFILE: SignProfile | None = None


def _profile() -> SignProfile:
    if PROFILE is None:
        raise RuntimeError(
            "PROFILE is not set; call main() with sign=main|secondary|yield|stop|roundabout"
        )
    return PROFILE


PDD_CODE = "2.1"
SIGN_TYPE = "main"
SIGN_NAME = "Main road (equal priority)"


def _apply_profile(profile: SignProfile) -> None:
    global PROFILE, PDD_CODE, SIGN_TYPE, SIGN_NAME
    PROFILE = profile
    PDD_CODE = profile.pdd_code
    SIGN_TYPE = profile.sign_type
    SIGN_NAME = profile.sign_name



# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------
@dataclass
class PathsConfig:
    scenes_dir: Optional[str] = None
    output_base: Optional[str] = None
    experiment_name: Optional[str] = None
    split: str = "all"


@dataclass
class ScenarioConfig:
    max_scenarios: Optional[int] = None
    max_total: Optional[int] = None
    # One-way (5.7.x): min length gain compliant − baseline (m).
    min_dual_path_gain_m: float = 20.0
    # Shared travel budget (m) for truncating dual-path routes; None = no cut.
    dual_path_route_budget_m: Optional[float] = None


@dataclass
class AugmentationAxesConfig:
    """Which augmentation axes are active (see configs/sign/*.yaml)."""

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
    # Roundabout-only: set via configs/sign/roundabout.yaml.
    destination_max_along_m: Optional[float] = None
    # Blocked road (3.2) only:
    sign_distance_from_start: float = 10.0
    # Combined (lane/dest × n_variations) pool; max_scenarios caps the product.
    n_variations: int = 3
    profile_density_cap: float = 1.0
    compliant_stop_success_seconds: float = 3.0
    compliant_stop_max_dist_m: float = 12.0
    compliant_stop_speed_mps: float = 0.5
    min_hops_after_depart: int = 0
    # Detour (4.2.x) only:
    spawn_offset_from_start: float = 10.0
    max_path_length_m: float = 100.0
    # Speed family (3.24 / 4.6 / 5.21 / 5.31):
    max_ego_lanes: int = 8
    zone_tail_m: float = 8.0
    zone_min_m: float = 20.0


@dataclass
class ExpertConfig:
    """Rule-expert hyperparameters written into each manifest row."""

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
    # Meters of view on the 800px GIF. MetaDrive clamps zoom by film/map_bbox;
    # run_benchmark grows film_size so the window sticks on large maps.
    window_m: float = 80.0
    draw_path_conflict: bool = False
    # Optional override; carl/plant2* fall back to pretrained defaults.
    model_path: Optional[str] = None


def resolve_gif_model_path(policy: str, model_path: Optional[str]) -> Optional[str]:
    """Resolve checkpoint for GIF NN policies (carl / plant2 / plant2_ft / *_rule)."""
    resolved = resolve_nn_checkpoint(policy, model_path)
    if resolved:
        return resolved
    if policy in NN_NEED_CHECKPOINT:
        default = DEFAULT_MODEL_PATHS.get(policy)
        print(
            f"[GIF] Default checkpoint missing for {policy}: {default} "
            f"(pass gif.model_path=...)",
            file=sys.stderr,
        )
    return None


@dataclass
class ManifestConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    augmentation: AugmentationAxesConfig = field(default_factory=AugmentationAxesConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    auxiliary: AuxiliaryConfig = field(default_factory=AuxiliaryConfig)
    gif: GifConfig = field(default_factory=GifConfig)


# -----------------------------------------------------------------------------
# Junction layout utilities
# -----------------------------------------------------------------------------
def build_junction_layout_for_scene(
    net_path: Path,
    *,
    sign_lat: Optional[float] = None,
    sign_lon: Optional[float] = None,
    scene_meta: Optional[dict] = None,
) -> Optional[dict]:
    """Build junction layout using the active sign profile's layout mode."""
    try:
        if _profile().layout_mode == "roundabout":
            from traffic_bench.eval.signs.roundabout.spawn import (
                roundabout_meta_ring_kwargs,
            )

            meta = scene_meta or {}
            prefer_ego = meta.get("catalog_sign_road_id") or meta.get("road_id")
            layout = build_roundabout_layout(
                net_path,
                sign_edge_id=prefer_ego,
                **roundabout_meta_ring_kwargs(meta),
            )
        else:
            layout = build_junction_priority_layout(
                net_path,
                mode=_profile().layout_mode,
                sign_lat=sign_lat,
                sign_lon=sign_lon,
            )
    except JunctionLayoutError as exc:
        print(f"  [junction_layout] {net_path.parent.name}: {exc}")
        return None
    return layout.to_dict()


# -----------------------------------------------------------------------------
# Blocked road (3.2) manifest generation
# -----------------------------------------------------------------------------
def generate_blocked_road_manifest(
    scenes_dir: Path,
    output_dir: Path,
    scenario_cfg: ScenarioConfig,
    sim_cfg: SimulationConfig,
    expansion_cfg: ExpansionConfig,
    split: str = "all",
) -> List[Dict]:
    """Generate real_manifest.jsonl for PDD 3.2 (through-path × NPC profiles)."""
    split = normalize_split(split)
    assert_rejected_scenes_applied(scenes_dir)
    all_scenes = discover_scenes(scenes_dir)
    print(f"Scenes root: {scenes_dir.resolve()}")
    print(f"Discovered {len(all_scenes)} scene(s) on disk")
    scenes, split_by_id = apply_split_filter(
        all_scenes, scenes_dir=scenes_dir, split=split
    )
    n_variations = max(1, int(sim_cfg.n_variations))
    print(
        f"Augmentation axes: layout={expansion_cfg.layout_on}, "
        f"n_variations={n_variations} (combined with lane/dest; "
        f"max_scenarios caps the product)"
    )

    blocked_road_expansion = BlockedRoadExpansionConfig(
        layout=expansion_cfg.layout_on,
        max_scenarios=scenario_cfg.max_scenarios,
    )
    sim_params = BlockedRoadSimParams(
        sign_distance_from_start=sim_cfg.sign_distance_from_start,
        spawn_distance_before_end=sim_cfg.spawn_distance_before_end,
        spawn_velocity_ms=sim_cfg.spawn_velocity_ms,
        horizon=sim_cfg.horizon,
        compliant_stop_success_seconds=sim_cfg.compliant_stop_success_seconds,
        compliant_stop_max_dist_m=sim_cfg.compliant_stop_max_dist_m,
        compliant_stop_speed_mps=sim_cfg.compliant_stop_speed_mps,
        n_variations=n_variations,
        profile_density_cap=float(sim_cfg.profile_density_cap),
        destination_max_along_m=sim_cfg.destination_max_along_m,
    )

    print(
        f"[blocked_road] NPC world: sample_one_profile in shared pool with "
        f"lane/dest (n_variations={n_variations}, "
        f"density_cap={sim_cfg.profile_density_cap})"
    )

    build_entry = partial(
        build_blocked_road_manifest_entry,
        pdd_code=PDD_CODE,
        sign_type=SIGN_TYPE,
        sign_class="NoTrafficSign",
        sign_title="Movement prohibited",
    )

    entries: List[Dict] = []
    used_scene_ids: List[str] = []
    min_lane = min(float(sim_cfg.spawn_distance_before_end), 8.0)

    for scene_dir in scenes:
        meta = load_scene_metadata(scene_dir)
        scene_name = meta.get("scene_name", scene_dir.name)
        net_file = meta.get("net_file", "map.net.xml")
        net_full_path = scene_dir / net_file
        print(f"\n=== {scene_name} ===")

        spawn_lanes = parse_sumo_net_for_spawn_lanes(net_full_path, min_length=min_lane)
        print(f"  Found {len(spawn_lanes)} intersection-approaching lane(s)")

        sign_lat = meta.get("latitude") or meta.get("center_lat")
        sign_lon = meta.get("longitude") or meta.get("center_lon")
        junction_layout = build_junction_layout_for_scene(
            net_full_path,
            sign_lat=float(sign_lat) if sign_lat is not None else None,
            sign_lon=float(sign_lon) if sign_lon is not None else None,
            scene_meta=meta,
        )
        if junction_layout is None:
            print(f"  Skipping {scene_name}: no junction layout")
            continue
        shape = junction_layout.get("shape")
        allowed_shapes = allowed_shapes_for_mode(_profile().layout_mode)
        if shape not in allowed_shapes:
            print(
                f"  Skipping {scene_name}: junction shape {shape!r} "
                f"(need {sorted(allowed_shapes)})"
            )
            continue

        scene_entries = expand_blocked_road_scene_entries(
            scene_dir=scene_dir,
            scenes_root=scenes_dir,
            meta=meta,
            net_path=net_full_path,
            spawn_lanes=spawn_lanes,
            junction_layout=junction_layout,
            sim=sim_params,
            expansion=blocked_road_expansion,
            build_entry=build_entry,
        )
        if not scene_entries:
            print(f"  Skipping {scene_name}: no manifest entries after expansion")
            continue
        append_scene_entries(
            entries, used_scene_ids, scene_entries,
            scene_dir=scene_dir, meta=meta, split_by_id=split_by_id,
        )

    entries, used_scene_ids, pre_total = apply_max_total(
        entries, used_scene_ids,
        max_total=scenario_cfg.max_total, split=split, pdd_code=PDD_CODE,
    )
    write_real_manifest(
        output_dir=output_dir,
        scenes_dir=scenes_dir,
        entries=entries,
        used_scene_ids=used_scene_ids,
        split_by_id=split_by_id,
        split=split,
        pdd_code=PDD_CODE,
        summary={
            "pdd_code": PDD_CODE,
            "sign_type": SIGN_TYPE,
            "sign_name": SIGN_NAME,
            "sign_class": "NoTrafficSign",
            "sign_placement": (
                "artificial at start of forbidden (destination) lane "
                "(sign_distance_from_start); ego on approach "
                "(spawn_distance_before_end)"
            ),
            "total_scenes": len(used_scene_ids),
            "total_entries": len(entries),
            "total_entries_before_max_total": pre_total,
            "augmentation_layout": expansion_cfg.layout_on,
            "n_variations": n_variations,
            "npc_world": "core.profiles.agent_profile_bank.sample_one_profile",
            "profile_density_cap": sim_cfg.profile_density_cap,
            "max_scenarios": scenario_cfg.max_scenarios,
            "max_total": scenario_cfg.max_total,
            "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
            "horizon": sim_cfg.horizon,
            "sign_distance_from_start": sim_cfg.sign_distance_from_start,
            "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
            "destination_max_along_m": sim_cfg.destination_max_along_m,
            "compliant_stop_success_seconds": sim_cfg.compliant_stop_success_seconds,
            "compliant_stop_max_dist_m": sim_cfg.compliant_stop_max_dist_m,
            "compliant_stop_speed_mps": sim_cfg.compliant_stop_speed_mps,
            "auxiliary_agent": False,
        },
    )
    return entries


# -----------------------------------------------------------------------------
# Dual-path (4.1 / 5.7 / 3.18 / 3.1) manifest generation
# -----------------------------------------------------------------------------
_DUAL_PATH_PLACEMENT = {
    "one_way": (
        "OneWayEntrySign on ego approach "
        "(sign_distance_before_end); dual-path compliant nav installed at episode start"
    ),
    "direction": (
        "LaneAllowedDirectionSign on ego approach "
        "(sign_distance_before_end); dual-path compliant nav installed at episode start"
    ),
    "no_turn": (
        "NoLeft/RightTurnSign on ego approach "
        "(sign_distance_before_end); dual-path compliant nav installed at episode start"
    ),
    "no_entry": (
        "NoEntrySign on baseline first exit "
        "(sign_road_id / sign_distance_from_start≈5m); "
        "dual-path compliant nav installed at episode start"
    ),
}


def generate_dual_path_manifest(
    scenes_dir: Path,
    output_dir: Path,
    scenario_cfg: ScenarioConfig,
    sim_cfg: SimulationConfig,
    expansion_cfg: ExpansionConfig,
    split: str = "all",
) -> List[Dict]:
    """Generate real_manifest.jsonl for dual-path signs (dual-path × NPC profiles)."""
    split = normalize_split(split)
    assert_rejected_scenes_applied(scenes_dir)
    all_scenes = discover_scenes(scenes_dir)
    print(f"Scenes root: {scenes_dir.resolve()}")
    print(f"Discovered {len(all_scenes)} scene(s) on disk")
    scenes, split_by_id = apply_split_filter(
        all_scenes, scenes_dir=scenes_dir, split=split
    )
    n_variations = max(1, int(sim_cfg.n_variations))
    print(
        f"Augmentation axes: layout={expansion_cfg.layout_on}, "
        f"n_variations={n_variations} (combined with dual-path; "
        f"max_scenarios caps the product)"
    )

    expansion = DualPathExpansionConfig(
        layout=expansion_cfg.layout_on,
        max_scenarios=scenario_cfg.max_scenarios,
        arm_counts=(3, 4),
    )
    sim_params = DualPathSimParams(
        spawn_distance_before_end=sim_cfg.spawn_distance_before_end,
        sign_distance_before_end=sim_cfg.sign_distance_before_end,
        spawn_velocity_ms=sim_cfg.spawn_velocity_ms,
        horizon=sim_cfg.horizon,
        n_variations=n_variations,
        profile_density_cap=float(sim_cfg.profile_density_cap),
        min_dual_path_gain_m=float(scenario_cfg.min_dual_path_gain_m),
        min_ego_lane_m=min(float(sim_cfg.spawn_distance_before_end), 8.0),
        dual_path_route_budget_m=scenario_cfg.dual_path_route_budget_m,
    )

    family = _profile().sign_type
    print(
        f"[{family}] NPC world: sample_one_profile in shared pool with "
        f"dual-path (n_variations={n_variations}, "
        f"density_cap={sim_cfg.profile_density_cap}, "
        f"min_gain={scenario_cfg.min_dual_path_gain_m}m, "
        f"route_budget_m={scenario_cfg.dual_path_route_budget_m}, "
        f"horizon={sim_cfg.horizon})"
    )

    build_entry = partial(
        build_dual_path_manifest_entry,
        pdd_code=PDD_CODE,
        sign_type=SIGN_TYPE,
    )

    entries: List[Dict] = []
    used_scene_ids: List[str] = []
    min_lane = min(float(sim_cfg.spawn_distance_before_end), 8.0)

    for scene_dir in scenes:
        meta = load_scene_metadata(scene_dir)
        scene_name = meta.get("scene_name", scene_dir.name)
        net_file = meta.get("net_file", "map.net.xml")
        net_full_path = scene_dir / net_file
        print(f"\n=== {scene_name} ===")

        spawn_lanes = parse_sumo_net_for_spawn_lanes(net_full_path, min_length=min_lane)
        print(f"  Found {len(spawn_lanes)} intersection-approaching lane(s)")

        sign_lat = meta.get("latitude") or meta.get("center_lat")
        sign_lon = meta.get("longitude") or meta.get("center_lon")
        junction_layout = build_junction_layout_for_scene(
            net_full_path,
            sign_lat=float(sign_lat) if sign_lat is not None else None,
            sign_lon=float(sign_lon) if sign_lon is not None else None,
            scene_meta=meta,
        )
        if junction_layout is None:
            print(f"  Skipping {scene_name}: no junction layout")
            continue
        shape = junction_layout.get("shape")
        allowed_shapes = allowed_shapes_for_mode(_profile().layout_mode)
        if shape not in allowed_shapes:
            print(
                f"  Skipping {scene_name}: junction shape {shape!r} "
                f"(need {sorted(allowed_shapes)})"
            )
            continue

        scene_entries = expand_dual_path_scene_entries(
            scene_dir=scene_dir,
            scenes_root=scenes_dir,
            meta=meta,
            net_path=net_full_path,
            spawn_lanes=spawn_lanes,
            junction_layout=junction_layout,
            sim=sim_params,
            expansion=expansion,
            build_entry=build_entry,
            pdd_code=PDD_CODE,
        )
        if not scene_entries:
            print(f"  Skipping {scene_name}: no manifest entries after expansion")
            continue
        append_scene_entries(
            entries, used_scene_ids, scene_entries,
            scene_dir=scene_dir, meta=meta, split_by_id=split_by_id,
        )

    entries, used_scene_ids, pre_total = apply_max_total(
        entries, used_scene_ids,
        max_total=scenario_cfg.max_total, split=split, pdd_code=PDD_CODE,
    )
    write_real_manifest(
        output_dir=output_dir,
        scenes_dir=scenes_dir,
        entries=entries,
        used_scene_ids=used_scene_ids,
        split_by_id=split_by_id,
        split=split,
        pdd_code=PDD_CODE,
        summary={
            "pdd_code": PDD_CODE,
            "sign_type": SIGN_TYPE,
            "sign_name": SIGN_NAME,
            "sign_class": _profile().sign_type,
            "sign_placement": _DUAL_PATH_PLACEMENT.get(
                family, _DUAL_PATH_PLACEMENT["direction"]
            ),
            "total_scenes": len(used_scene_ids),
            "total_entries": len(entries),
            "total_entries_before_max_total": pre_total,
            "augmentation_layout": expansion_cfg.layout_on,
            "n_variations": n_variations,
            "npc_world": "core.profiles.agent_profile_bank.sample_one_profile",
            "profile_density_cap": sim_cfg.profile_density_cap,
            "min_dual_path_gain_m": scenario_cfg.min_dual_path_gain_m,
            "max_scenarios": scenario_cfg.max_scenarios,
            "max_total": scenario_cfg.max_total,
            "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
            "horizon": sim_cfg.horizon,
            "sign_distance_before_end": sim_cfg.sign_distance_before_end,
            "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
            "auxiliary_agent": False,
        },
    )
    return entries


generate_one_way_manifest = generate_dual_path_manifest
generate_direction_manifest = generate_dual_path_manifest
generate_no_turn_manifest = generate_dual_path_manifest
generate_no_entry_manifest = generate_dual_path_manifest


# -----------------------------------------------------------------------------
# Pedestrian crossing (5.19) manifest generation
# -----------------------------------------------------------------------------
def generate_crosswalk_manifest(
    scenes_dir: Path,
    output_dir: Path,
    scenario_cfg: ScenarioConfig,
    sim_cfg: SimulationConfig,
    expansion_cfg: ExpansionConfig,
    *,
    split: str = "all",
    max_ego_lanes: int = 3,
    max_density_levels: int = 3,
    max_pedestrian_presets: int = 3,
    crosswalk_positions: Optional[List[str]] = None,
    traffic_density_augment: bool = True,
    ped_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    """Generate real_manifest.jsonl for PDD 5.19 (segment_crosswalk maps)."""
    split = normalize_split(split)
    # Maps come from scene_collection harvest — no scene_selection / reject apply step.
    positions = tuple(crosswalk_positions or DEFAULT_POSITIONS)
    all_scenes = discover_segment_crosswalk_scenes(scenes_dir, positions=positions)
    print(f"Scenes root: {scenes_dir.resolve()}")
    print(f"Discovered {len(all_scenes)} segment_crosswalk scene(s) on disk")
    scenes, split_by_id = apply_split_filter(
        all_scenes, scenes_dir=scenes_dir, split=split
    )
    print(
        f"Augmentation axes (≤3 each): ego_lanes={max_ego_lanes}, "
        f"density={max_density_levels}, ped_presets={max_pedestrian_presets}, "
        f"positions={list(positions)}, layout={expansion_cfg.layout_on}"
    )

    ped = ped_cfg or {}
    sim_params = CrosswalkSimParams(
        spawn_distance_before_end=float(sim_cfg.spawn_distance_before_end),
        sign_distance_before_end=float(sim_cfg.sign_distance_before_end),
        spawn_velocity_ms=float(sim_cfg.spawn_velocity_ms),
        horizon=int(sim_cfg.horizon),
        traffic_density=float(sim_cfg.traffic_density),
        traffic_density_augment=bool(traffic_density_augment),
        min_hops_after_depart=int(getattr(sim_cfg, "min_hops_after_depart", 0) or 0),
        destination_max_along_m=float(
            sim_cfg.destination_max_along_m
            if sim_cfg.destination_max_along_m is not None
            else 40.0
        ),
        max_ego_lanes=int(max_ego_lanes),
        max_density_levels=int(max_density_levels),
        max_pedestrian_presets=int(max_pedestrian_presets),
        crosswalk_positions=positions,
        ped_ego_spawn_distance_m=float(ped.get("default_ego_spawn_distance_m", 50.0)),
        ped_speed_mean=float(ped.get("default_speed_mean", 1.2)),
        ped_speed_std=float(ped.get("default_speed_std", 0.2)),
        ped_spawn_gap_s=float(ped.get("default_spawn_gap_s", 2.5)),
        ped_yield_distance=float(ped.get("yield_distance", 12.0)),
        ped_no_stop_before_crosswalk_m=float(ped.get("no_stop_before_crosswalk_m", 3.0)),
    )
    cw_expansion = CrosswalkExpansionConfig(
        layout=expansion_cfg.layout_on,
        max_scenarios=scenario_cfg.max_scenarios,
    )

    entries: List[Dict] = []
    used_scene_ids: List[str] = []

    for scene_dir in scenes:
        meta = load_scene_metadata(scene_dir)
        scene_name = meta.get("scene_name", scene_dir.name)
        net_file = meta.get("net_file", "map.net.xml")
        net_full_path = scene_dir / net_file
        print(f"\n=== {scene_name} (pos={meta.get('crosswalk_position')}) ===")

        scene_entries = expand_crosswalk_scene_entries(
            scene_dir=scene_dir,
            scenes_root=scenes_dir,
            meta=meta,
            net_path=net_full_path,
            sim=sim_params,
            expansion=cw_expansion,
            pdd_code=PDD_CODE,
            sign_type=SIGN_TYPE,
        )
        if not scene_entries:
            print(f"  Skipping {scene_name}: no manifest entries after expansion")
            continue
        append_scene_entries(
            entries, used_scene_ids, scene_entries,
            scene_dir=scene_dir, meta=meta, split_by_id=split_by_id,
        )

    entries, used_scene_ids, pre_total = apply_max_total(
        entries, used_scene_ids,
        max_total=scenario_cfg.max_total, split=split, pdd_code=PDD_CODE,
        scene_id_key="scene_name",
    )
    write_real_manifest(
        output_dir=output_dir,
        scenes_dir=scenes_dir,
        entries=entries,
        used_scene_ids=used_scene_ids,
        split_by_id=split_by_id,
        split=split,
        pdd_code=PDD_CODE,
        summary={
            "pdd_code": PDD_CODE,
            "sign_type": SIGN_TYPE,
            "sign_name": SIGN_NAME,
            "sign_class": "PedestrianCrossingSign",
            "sign_placement": (
                "PedestrianCrossingSign (5.19 icon) on ego approach lane; "
                "yield enforced via PedestrianYieldRule + CrosswalkPedestrianManager"
            ),
            "total_scenes": len(used_scene_ids),
            "total_entries": len(entries),
            "total_entries_before_max_total": pre_total,
            "augmentation_layout": expansion_cfg.layout_on,
            "max_ego_lanes": max_ego_lanes,
            "max_density_levels": max_density_levels,
            "max_pedestrian_presets": max_pedestrian_presets,
            "crosswalk_positions": list(positions),
            "traffic_density_augment": bool(traffic_density_augment),
            "max_scenarios": scenario_cfg.max_scenarios,
            "max_total": scenario_cfg.max_total,
            "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
            "horizon": sim_cfg.horizon,
            "sign_distance_before_end": sim_cfg.sign_distance_before_end,
            "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
            "auxiliary_agent": False,
        },
    )
    return entries


# -----------------------------------------------------------------------------
# PDD 4.2.x detour manifest
# -----------------------------------------------------------------------------
def generate_detour_manifest(
    scenes_dir: Path,
    output_dir: Path,
    scenario_cfg: ScenarioConfig,
    sim_cfg: SimulationConfig,
    expansion_cfg: ExpansionConfig,
    split: str = "all",
    *,
    max_density_levels: int = 3,
    traffic_density_augment: bool = True,
) -> List[Dict]:
    """Generate real_manifest.jsonl for PDD 4.2.x (segment_detour maps)."""
    split = normalize_split(split)

    detour_code = PDD_CODE  # e.g. "4.2.1"
    all_scenes = discover_segment_detour_scenes(scenes_dir, detour_code=detour_code)
    print(f"Scenes root: {scenes_dir.resolve()}")
    print(f"Discovered {len(all_scenes)} segment_detour scene(s) for {detour_code}")
    scenes, split_by_id = apply_split_filter(
        all_scenes, scenes_dir=scenes_dir, split=split
    )
    print(
        f"Augmentation axes: density={max_density_levels} "
        f"(traffic_density_augment={bool(traffic_density_augment)})"
    )

    sim_params = DetourSimParams(
        spawn_offset_from_start=float(sim_cfg.spawn_offset_from_start),
        max_path_length_m=float(sim_cfg.max_path_length_m),
        sign_distance_before_end=float(sim_cfg.sign_distance_before_end),
        spawn_velocity_ms=float(sim_cfg.spawn_velocity_ms),
        horizon=int(sim_cfg.horizon),
        traffic_density=float(sim_cfg.traffic_density),
        traffic_density_augment=bool(traffic_density_augment),
        max_density_levels=int(max_density_levels),
    )
    det_expansion = DetourExpansionConfig(
        max_scenarios=scenario_cfg.max_scenarios,
    )

    entries: List[Dict] = []
    used_scene_ids: List[str] = []

    for scene_dir in scenes:
        meta = load_scene_metadata(scene_dir)
        scene_name = meta.get("scene_name", scene_dir.name)
        print(f"\n=== {scene_name} ===")

        scene_entries = expand_detour_scene_entries(
            scene_dir=scene_dir,
            scenes_root=scenes_dir,
            meta=meta,
            sim=sim_params,
            expansion=det_expansion,
            pdd_code=PDD_CODE,
            sign_type=SIGN_TYPE,
        )
        if not scene_entries:
            print(f"  Skipping {scene_name}: no manifest entries")
            continue
        append_scene_entries(
            entries, used_scene_ids, scene_entries,
            scene_dir=scene_dir, meta=meta, split_by_id=split_by_id,
        )

    entries, used_scene_ids, pre_total = apply_max_total(
        entries, used_scene_ids,
        max_total=scenario_cfg.max_total, split=split, pdd_code=PDD_CODE,
        scene_id_key="scene_name",
    )
    sign_class_map = {
        "4.2.1": "DetourRightSign",
        "4.2.2": "DetourLeftSign",
        "4.2.3": "DetourEitherSign",
    }
    write_real_manifest(
        output_dir=output_dir,
        scenes_dir=scenes_dir,
        entries=entries,
        used_scene_ids=used_scene_ids,
        split_by_id=split_by_id,
        split=split,
        pdd_code=PDD_CODE,
        summary={
            "pdd_code": PDD_CODE,
            "sign_type": SIGN_TYPE,
            "sign_name": SIGN_NAME,
            "sign_class": sign_class_map.get(PDD_CODE, "DetourRightSign"),
            "sign_placement": (
                f"DetourSign ({PDD_CODE}) placed on obstacle lane at sign_s from meta; "
                f"ego spawns on same lane and must change to adjacent lane"
            ),
            "total_scenes": len(used_scene_ids),
            "total_entries": len(entries),
            "total_entries_before_max_total": pre_total,
            "max_scenarios": scenario_cfg.max_scenarios,
            "max_total": scenario_cfg.max_total,
            "max_density_levels": max_density_levels,
            "traffic_density_augment": bool(traffic_density_augment),
            "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
            "horizon": sim_cfg.horizon,
            "sign_distance_before_end": sim_cfg.sign_distance_before_end,
            "spawn_offset_from_start": sim_cfg.spawn_offset_from_start,
            "max_path_length_m": sim_cfg.max_path_length_m,
            "auxiliary_agent": False,
        },
    )
    return entries


# -----------------------------------------------------------------------------
# PDD 3.24 / 4.6 / 5.21 / 5.31 speed-family manifest
# -----------------------------------------------------------------------------
def generate_speed_manifest(
    scenes_dir: Path,
    output_dir: Path,
    scenario_cfg: ScenarioConfig,
    sim_cfg: SimulationConfig,
    expansion_cfg: ExpansionConfig,
    split: str = "all",
    *,
    max_density_levels: int = 3,
    traffic_density_augment: bool = True,
) -> List[Dict]:
    """Generate real_manifest.jsonl for speed-family signs on segment maps."""
    split = normalize_split(split)
    pdd_code = PDD_CODE
    all_scenes = discover_segment_speed_scenes(scenes_dir)
    print(f"Scenes root: {scenes_dir.resolve()}")
    print(f"Discovered {len(all_scenes)} segment scene(s) for {pdd_code}")
    scenes, split_by_id = apply_split_filter(
        all_scenes, scenes_dir=scenes_dir, split=split
    )
    print(
        f"Augmentation axes: spawn_lane × density={max_density_levels} "
        f"(traffic_density_augment={bool(traffic_density_augment)})"
    )

    sim_params = SpeedSimParams(
        spawn_offset_from_start=float(sim_cfg.spawn_offset_from_start),
        max_path_length_m=float(sim_cfg.max_path_length_m),
        horizon=int(sim_cfg.horizon),
        traffic_density=float(sim_cfg.traffic_density),
        traffic_density_augment=bool(traffic_density_augment),
        max_density_levels=int(max_density_levels),
        max_ego_lanes=int(sim_cfg.max_ego_lanes),
        zone_tail_m=float(sim_cfg.zone_tail_m),
        zone_min_m=float(sim_cfg.zone_min_m),
    )
    speed_expansion = SpeedExpansionConfig(
        max_scenarios=scenario_cfg.max_scenarios,
    )

    entries: List[Dict] = []
    used_scene_ids: List[str] = []
    skipped_short = 0

    for scene_idx, scene_dir in enumerate(scenes):
        meta = load_scene_metadata(scene_dir)
        scene_name = meta.get("scene_name", scene_dir.name)
        v_target_kmh = assign_limit_kmh(pdd_code, scene_idx)
        print(f"\n=== {scene_name}  v_target={v_target_kmh:.0f} km/h ===")

        scene_entries = expand_speed_scene_entries(
            scene_dir=scene_dir,
            scenes_root=scenes_dir,
            meta=meta,
            sim=sim_params,
            expansion=speed_expansion,
            pdd_code=pdd_code,
            v_target_kmh=v_target_kmh,
        )
        if not scene_entries:
            skipped_short += 1
            print(f"  Skipping {scene_name}: no room for sign+zone before dest cap")
            continue
        append_scene_entries(
            entries, used_scene_ids, scene_entries,
            scene_dir=scene_dir, meta=meta, split_by_id=split_by_id,
        )

    entries, used_scene_ids, pre_total = apply_max_total(
        entries, used_scene_ids,
        max_total=scenario_cfg.max_total, split=split, pdd_code=pdd_code,
        scene_id_key="scene_name",
    )
    write_real_manifest(
        output_dir=output_dir,
        scenes_dir=scenes_dir,
        entries=entries,
        used_scene_ids=used_scene_ids,
        split_by_id=split_by_id,
        split=split,
        pdd_code=pdd_code,
        summary={
            "pdd_code": pdd_code,
            "sign_type": SIGN_TYPE,
            "sign_name": SIGN_NAME,
            "sign_class": {
                "3.24": "SpeedLimitSign",
                "4.6": "MinimumSpeedLimitSign",
                "5.21": "ResidentialZoneSign",
                "5.31": "ZoneSpeedLimitSign",
            }.get(pdd_code, "SpeedLimitSign"),
            "sign_placement": (
                f"ego at road start; start sign at spawn+approach; "
                f"dest min(edge_end, {sim_cfg.max_path_length_m}m)"
            ),
            "total_scenes": len(used_scene_ids),
            "total_entries": len(entries),
            "total_entries_before_max_total": pre_total,
            "skipped_short_scenes": skipped_short,
            "max_scenarios": scenario_cfg.max_scenarios,
            "max_total": scenario_cfg.max_total,
            "max_density_levels": max_density_levels,
            "traffic_density_augment": bool(traffic_density_augment),
            "spawn_offset_from_start": sim_cfg.spawn_offset_from_start,
            "max_path_length_m": sim_cfg.max_path_length_m,
            "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
            "horizon": sim_cfg.horizon,
            "auxiliary_agent": False,
        },
    )
    return entries


# -----------------------------------------------------------------------------
# Manifest generation
# -----------------------------------------------------------------------------
def generate_manifest(
    scenes_dir: Path,
    output_dir: Path,
    scenario_cfg: ScenarioConfig,
    sim_cfg: SimulationConfig,
    aux_cfg: AuxiliaryConfig,
    expansion_cfg: ExpansionConfig,
    split: str = "all",
    expert_cfg: Optional[ExpertConfig] = None,
) -> List[Dict]:
    """Generate real_manifest.jsonl from discovered scenes."""
    expert_cfg = expert_cfg or ExpertConfig()
    split = normalize_split(split)
    assert_rejected_scenes_applied(scenes_dir)
    all_scenes = discover_scenes(scenes_dir)
    print(f"Scenes root: {scenes_dir.resolve()}")
    print(f"Discovered {len(all_scenes)} scene(s) on disk")
    scenes, split_by_id = apply_split_filter(
        all_scenes, scenes_dir=scenes_dir, split=split
    )
    print(
        f"Augmentation axes: layout={expansion_cfg.layout_on}, "
        f"auxiliary={expansion_cfg.auxiliary_on}"
    )
    if not scenes:
        print(
            f"[warn] No scenes with meta.json + net found under {scenes_dir} "
            f"(after paths.split={split}). "
            "Check data/scenes/<sign> / paths.scenes_dir / moscow_pool.json."
        )
    entries = []

    aux_for_entry = aux_cfg
    if not expansion_cfg.auxiliary_on:
        aux_for_entry = AuxiliaryConfig(
            enabled=False,
            distance_from_intersection=aux_cfg.distance_from_intersection,
            convoy_size=aux_cfg.convoy_size,
            convoy_gap_m=aux_cfg.convoy_gap_m,
            lanes_occupied=aux_cfg.lanes_occupied,
            release_when_ego_within_m=aux_cfg.release_when_ego_within_m,
        )

    used_scene_ids: List[str] = []
    for scene_dir in scenes:
        meta = load_scene_metadata(scene_dir)
        scene_name = meta.get("scene_name", scene_dir.name)
        net_file = meta.get("net_file", "map.net.xml")
        net_full_path = scene_dir / net_file

        spawn_lanes = parse_sumo_net_for_spawn_lanes(net_full_path)
        print(f"  Found {len(spawn_lanes)} intersection-approaching lane(s)")

        sign_lat = meta.get("latitude") or meta.get("center_lat")
        sign_lon = meta.get("longitude") or meta.get("center_lon")

        junction_layout = build_junction_layout_for_scene(
            net_full_path,
            sign_lat=float(sign_lat) if sign_lat is not None else None,
            sign_lon=float(sign_lon) if sign_lon is not None else None,
            scene_meta=meta,
        )
        if junction_layout is None:
            print(f"  Skipping {scene_name}: no junction layout")
            continue
        shape = junction_layout.get("shape")
        allowed_shapes = allowed_shapes_for_mode(_profile().layout_mode)
        if shape not in allowed_shapes:
            print(
                f"  Skipping {scene_name}: junction shape {shape!r} "
                f"(need {sorted(allowed_shapes)})"
            )
            continue

        scene_entries = expand_scene_entries(
            scene_dir=scene_dir,
            scenes_root=scenes_dir,
            meta=meta,
            net_path=net_full_path,
            spawn_lanes=spawn_lanes,
            junction_layout=junction_layout,
            spawn_strategy=_profile().spawn_strategy,
            sim_cfg=sim_cfg,
            expansion=expansion_cfg,
            build_entry=partial(
                build_manifest_entry, expert_cfg=expert_cfg, profile=_profile()
            ),
            aux_cfg_for_entry=aux_for_entry,
        )
        if not scene_entries:
            print(f"  Skipping {scene_name}: no manifest entries after expansion")
            continue
        append_scene_entries(
            entries, used_scene_ids, scene_entries,
            scene_dir=scene_dir, meta=meta, split_by_id=split_by_id,
        )

    entries, used_scene_ids, pre_total = apply_max_total(
        entries, used_scene_ids,
        max_total=scenario_cfg.max_total, split=split, pdd_code=PDD_CODE,
        log_under_cap=True,
    )
    summary = {
        "pdd_code": PDD_CODE,
        "sign_type": SIGN_TYPE,
        "sign_name": SIGN_NAME,
        "total_scenes": len(used_scene_ids),
        "total_entries": len(entries),
        "total_entries_before_max_total": pre_total,
        "augmentation_layout": expansion_cfg.layout_on,
        "augmentation_auxiliary": expansion_cfg.auxiliary_on,
        "max_scenarios": scenario_cfg.max_scenarios,
        "max_total": scenario_cfg.max_total,
        "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
        "traffic_density": sim_cfg.traffic_density,
        "horizon": sim_cfg.horizon,
        "sign_distance_before_end": sim_cfg.sign_distance_before_end,
        "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
        "auxiliary_agent": aux_for_entry.enabled,
        "aux_distance_from_intersection": aux_cfg.distance_from_intersection,
        "aux_convoy_size_max": aux_cfg.convoy_size,
        "aux_convoy_gap_m": list(expansion_cfg.aux.convoy_gaps_m)
        if expansion_cfg.aux is not None
        else [aux_cfg.convoy_gap_m],
        "aux_lanes_occupied_max": aux_cfg.lanes_occupied,
    }
    if _profile().id == STOP.id:
        summary["stop_wait_steps"] = int(expert_cfg.stop_wait_steps)
    if (
        _profile().layout_mode == "roundabout"
        and sim_cfg.destination_max_along_m is not None
    ):
        summary["destination_max_along_m"] = float(sim_cfg.destination_max_along_m)
    write_real_manifest(
        output_dir=output_dir,
        scenes_dir=scenes_dir,
        entries=entries,
        used_scene_ids=used_scene_ids,
        split_by_id=split_by_id,
        split=split,
        pdd_code=PDD_CODE,
        summary=summary,
        announce=False,
    )
    return entries


# -----------------------------------------------------------------------------
# GIF rendering
# -----------------------------------------------------------------------------
def _iter_jsonl_rows(path: Path):
    """Iterate over JSONL file rows."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def render_gifs_from_manifest(
    manifest_path: Path,
    experiment_dir: Path,
    scenes_root: Path,
    gif_cfg: GifConfig,
    aux_cfg: AuxiliaryConfig,
) -> Tuple[int, int]:
    """Render GIFs for scenes from a manifest file."""
    if not RUN_BENCH_SCRIPT.is_file():
        print(f"[GIF] run_benchmark.py not found at {RUN_BENCH_SCRIPT}", file=sys.stderr)
        return 0, 1
    
    if not manifest_path.is_file():
        print(f"[GIF] Manifest not found: {manifest_path}", file=sys.stderr)
        return 0, 1
    
    gif_dir = Path(gif_cfg.dir) if gif_cfg.dir else experiment_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    run_name = gif_cfg.run_name or experiment_dir.name

    rows = []
    seen_keys = set()

    for row in _iter_jsonl_rows(manifest_path):
        if not row.get("valid", True):
            continue
        if row.get("pdd_code") != PDD_CODE:
            continue

        scene_id = row.get("scene_id")
        seed = row.get("seed")
        if scene_id is None or seed is None:
            continue
        row_key = (scene_id, seed)
        if row_key in seen_keys:
            continue

        seen_keys.add(row_key)
        rows.append(row)
        
        if gif_cfg.max_scenes is not None and len(rows) >= gif_cfg.max_scenes:
            break
    
    if not rows:
        print(f"[GIF] No valid scenes found in manifest for {PDD_CODE}.")
        return 0, 0
    
    print(f"\n[GIF] Rendering {len(rows)} scene(s)...")
    model_path = resolve_gif_model_path(gif_cfg.policy, gif_cfg.model_path)
    if gif_cfg.policy in NN_NEED_CHECKPOINT and not model_path:
        print(
            f"[GIF] No checkpoint for policy={gif_cfg.policy}; "
            f"set gif.model_path=... or place defaults under {CHECKPOINTS_DIR}",
            file=sys.stderr,
        )
        return 0, 1

    rendered = 0
    failed = 0
    for i, row in enumerate(rows, start=1):
        scene_uid = f"{row['scene_id']}:{row['pdd_code']}:{row['seed']}"
        cmd = [
            sys.executable,
            str(RUN_BENCH_SCRIPT),
            "--scene-uid", scene_uid,
            "--manifest", str(manifest_path),
            "--save-gifs",
            "--output-dir", str(experiment_dir),
            "--gif-dir", str(gif_dir),
            "--run-name", run_name,
            "--scenes-root", str(scenes_root),
            "--policy", gif_cfg.policy,
        ]
        if model_path:
            cmd.extend(["--model-path", model_path])
        if gif_cfg.hide_signs:
            cmd.append("--hide-signs")
        if gif_cfg.draw_path_conflict:
            cmd.append("--draw-path-conflict")
        if gif_cfg.window_m is not None and float(gif_cfg.window_m) > 0.0:
            cmd.extend(["--gif-window-m", str(float(gif_cfg.window_m))])
        row_horizon = int(row.get("horizon_steps") or row.get("horizon") or 600)
        cmd.extend(["--max-steps", str(max(1, row_horizon))])

        if aux_cfg.enabled:
            cmd.append("--auxiliary-agent")
            cmd.extend(["--aux-distance-from-intersection", str(aux_cfg.distance_from_intersection)])
            release_m = getattr(aux_cfg, "release_when_ego_within_m", None)
            if release_m is not None:
                cmd.extend(["--aux-release-when-ego-within-m", str(float(release_m))])
        
        print(f"\n[GIF {i}/{len(rows)}] {scene_uid}")
        print("  " + " ".join(cmd))
        
        if gif_cfg.dry_run:
            rendered += 1
            continue
        
        res = subprocess.run(cmd, cwd=str(RUN_BENCH_SCRIPT.parent))
        if res.returncode == 0:
            rendered += 1
        else:
            failed += 1
            print(f"[GIF] Command failed with code {res.returncode}")
    
    return rendered, failed


# -----------------------------------------------------------------------------
# Hydra entry point
# -----------------------------------------------------------------------------
def _resolve_max_scenarios(scenario_cfg) -> Optional[int]:
    """Prefer ``max_scenarios``; accept legacy ``max_scenarios_per_scene``."""
    raw = getattr(scenario_cfg, "max_scenarios", None)
    if raw is None:
        raw = getattr(scenario_cfg, "max_scenarios_per_scene", None)
    if raw is None:
        return None
    return int(raw)


def _resolve_max_total(scenario_cfg) -> Optional[int]:
    """Global cap on manifest rows after all scenes are expanded (debug)."""
    raw = getattr(scenario_cfg, "max_total", None)
    if raw is None:
        return None
    return int(raw)


def _resolve_convoy_gaps_m(raw) -> List[float]:
    """Accept a scalar or list for ``auxiliary.convoy_gap_m``."""
    if raw is None:
        return [float(DEFAULT_CONVOY_GAP_M)]
    if OmegaConf.is_list(raw) or isinstance(raw, (list, tuple)):
        gaps = [float(x) for x in raw]
        return gaps if gaps else [float(DEFAULT_CONVOY_GAP_M)]
    return [float(raw)]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    profile = get_profile(cfg.sign)
    _apply_profile(profile)
    print(f"Sign profile: {profile.id} ({profile.pdd_code} / {profile.sign_type})")
    expected_out = profile_output_dir(profile).resolve()
    configured_out = Path(str(cfg.paths.output_base))
    if not configured_out.is_absolute():
        configured_out = (REPO_ROOT / configured_out).resolve()
    if configured_out != expected_out and profile.data_subdir not in str(configured_out):
        print(
            f"[warn] paths.output_base={cfg.paths.output_base} may not match sign={profile.id}; "
            f"preferred: data/runs/{profile.data_subdir}"
        )


    if str(cfg.paths.scenes_dir) in {"scenes", "scenes/", ""}:
        scenes_dir = profile_scenes_dir(profile)
    else:
        scenes_dir = Path(cfg.paths.scenes_dir)
        if not scenes_dir.is_absolute():
            scenes_dir = (REPO_ROOT / scenes_dir).resolve()
    print(f"Using scenes_dir: {scenes_dir}")

    experiment_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_path = experiment_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    scenario_cfg = ScenarioConfig(
        max_scenarios=_resolve_max_scenarios(cfg.scenario),
        max_total=_resolve_max_total(cfg.scenario),
        min_dual_path_gain_m=float(
            getattr(cfg.scenario, "min_dual_path_gain_m", 20.0) or 20.0
        ),
        dual_path_route_budget_m=(
            float(cfg.scenario.dual_path_route_budget_m)
            if getattr(cfg.scenario, "dual_path_route_budget_m", None) is not None
            else None
        ),
    )
    sim_cfg = SimulationConfig(
        spawn_velocity_ms=cfg.simulation.spawn_velocity_ms,
        traffic_density=cfg.simulation.traffic_density,
        horizon=cfg.simulation.horizon,
        sign_distance_before_end=cfg.simulation.sign_distance_before_end,
        spawn_distance_before_end=cfg.simulation.spawn_distance_before_end,
        destination_max_along_m=(
            float(cfg.simulation.destination_max_along_m)
            if getattr(cfg.simulation, "destination_max_along_m", None) is not None
            else None
        ),
        sign_distance_from_start=float(
            getattr(cfg.simulation, "sign_distance_from_start", 10.0) or 10.0
        ),
        n_variations=int(getattr(cfg.simulation, "n_variations", 3) or 3),
        profile_density_cap=float(
            getattr(cfg.simulation, "profile_density_cap", 1.0) or 1.0
        ),
        compliant_stop_success_seconds=float(
            getattr(cfg.simulation, "compliant_stop_success_seconds", 3.0) or 3.0
        ),
        compliant_stop_max_dist_m=float(
            getattr(cfg.simulation, "compliant_stop_max_dist_m", 12.0) or 12.0
        ),
        compliant_stop_speed_mps=float(
            getattr(cfg.simulation, "compliant_stop_speed_mps", 0.5) or 0.5
        ),
        min_hops_after_depart=int(
            getattr(cfg.simulation, "min_hops_after_depart", 0) or 0
        ),
        spawn_offset_from_start=float(
            getattr(cfg.simulation, "spawn_offset_from_start", 10.0) or 10.0
        ),
        max_path_length_m=float(
            getattr(cfg.simulation, "max_path_length_m", 100.0) or 100.0
        ),
        max_ego_lanes=int(getattr(cfg.simulation, "max_ego_lanes", 8) or 8),
        zone_tail_m=float(getattr(cfg.simulation, "zone_tail_m", 8.0) or 8.0),
        zone_min_m=float(getattr(cfg.simulation, "zone_min_m", 20.0) or 20.0),
    )
    expert_cfg = ExpertConfig(
        stop_wait_steps=int(
            getattr(getattr(cfg, "expert", None), "stop_wait_steps", DEFAULT_STOP_WAIT_STEPS)
            or DEFAULT_STOP_WAIT_STEPS
        ),
    )
    convoy_gaps_m = _resolve_convoy_gaps_m(getattr(cfg.auxiliary, "convoy_gap_m", None))
    aux_cfg = AuxiliaryConfig(
        enabled=bool(cfg.auxiliary.enabled),
        distance_from_intersection=float(cfg.auxiliary.distance_from_intersection),
        convoy_size=int(cfg.auxiliary.convoy_size),
        convoy_gap_m=float(convoy_gaps_m[0]),
        lanes_occupied=int(cfg.auxiliary.lanes_occupied),
        release_when_ego_within_m=float(
            getattr(cfg.auxiliary, "release_when_ego_within_m", 15.0) or 15.0
        ),
    )
    aug_cfg = cfg.augmentation
    layout_flag = bool(getattr(aug_cfg, "layout", False))
    # Backward compat: scenario.augment → layout axis if still present in overrides.
    legacy_augment = getattr(cfg.scenario, "augment", None)
    if legacy_augment is not None and not layout_flag and bool(legacy_augment):
        layout_flag = True
    expansion_cfg = ExpansionConfig(
        enabled=bool(getattr(aug_cfg, "enabled", True)),
        layout=layout_flag,
        auxiliary=bool(getattr(aug_cfg, "auxiliary", False)),
        max_scenarios=scenario_cfg.max_scenarios,
        aux=AuxiliaryParams(
            enabled=aux_cfg.enabled,
            distance_from_intersection=aux_cfg.distance_from_intersection,
            convoy_size=aux_cfg.convoy_size,
            convoy_gaps_m=tuple(convoy_gaps_m),
            lanes_occupied=aux_cfg.lanes_occupied,
            release_when_ego_within_m=aux_cfg.release_when_ego_within_m,
        ),
    )
    gif_cfg = GifConfig(
        enabled=cfg.gif.enabled,
        policy=cfg.gif.policy,
        max_scenes=cfg.gif.max_scenes,
        dry_run=cfg.gif.dry_run,
        hide_signs=cfg.gif.hide_signs,
        dir=cfg.gif.dir,
        run_name=cfg.gif.run_name,
        window_m=float(getattr(cfg.gif, "window_m", 80.0) or 80.0),
        draw_path_conflict=bool(getattr(cfg.gif, "draw_path_conflict", False)),
        model_path=getattr(cfg.gif, "model_path", None) or None,
    )

    split = normalize_split(getattr(cfg.paths, "split", "all"))
    print(f"Using paths.split: {split}")

    if profile.family == "blocked":
        entries = generate_blocked_road_manifest(
            scenes_dir=scenes_dir,
            output_dir=experiment_dir,
            scenario_cfg=scenario_cfg,
            sim_cfg=sim_cfg,
            expansion_cfg=expansion_cfg,
            split=split,
        )
    elif profile.family == "dual_path":
        entries = generate_dual_path_manifest(
            scenes_dir=scenes_dir,
            output_dir=experiment_dir,
            scenario_cfg=scenario_cfg,
            sim_cfg=sim_cfg,
            expansion_cfg=expansion_cfg,
            split=split,
        )
    elif profile.family == "crosswalk":
        positions_raw = getattr(cfg.scenario, "crosswalk_positions", None)
        if positions_raw is None:
            positions_list = list(DEFAULT_POSITIONS)
        elif OmegaConf.is_list(positions_raw) or isinstance(positions_raw, (list, tuple)):
            positions_list = [str(x) for x in positions_raw]
        else:
            positions_list = [str(positions_raw)]
        ped_node = getattr(cfg, "pedestrian", None)
        ped_cfg = (
            OmegaConf.to_container(ped_node, resolve=True)
            if ped_node is not None
            else {}
        )
        if not isinstance(ped_cfg, dict):
            ped_cfg = {}
        entries = generate_crosswalk_manifest(
            scenes_dir=scenes_dir,
            output_dir=experiment_dir,
            scenario_cfg=scenario_cfg,
            sim_cfg=sim_cfg,
            expansion_cfg=expansion_cfg,
            split=split,
            max_ego_lanes=int(getattr(cfg.scenario, "max_ego_lanes", 3) or 3),
            max_density_levels=int(getattr(cfg.scenario, "max_density_levels", 3) or 3),
            max_pedestrian_presets=int(
                getattr(cfg.scenario, "max_pedestrian_presets", 3) or 3
            ),
            crosswalk_positions=positions_list,
            traffic_density_augment=bool(
                getattr(cfg.simulation, "traffic_density_augment", True)
            ),
            ped_cfg=ped_cfg,
        )
    elif profile.family == "detour":
        entries = generate_detour_manifest(
            scenes_dir=scenes_dir,
            output_dir=experiment_dir,
            scenario_cfg=scenario_cfg,
            sim_cfg=sim_cfg,
            expansion_cfg=expansion_cfg,
            split=split,
            max_density_levels=int(getattr(cfg.scenario, "max_density_levels", 3) or 3),
            traffic_density_augment=bool(
                getattr(cfg.simulation, "traffic_density_augment", True)
            ),
        )
    elif profile.family == "speed":
        entries = generate_speed_manifest(
            scenes_dir=scenes_dir,
            output_dir=experiment_dir,
            scenario_cfg=scenario_cfg,
            sim_cfg=sim_cfg,
            expansion_cfg=expansion_cfg,
            split=split,
            max_density_levels=int(getattr(cfg.scenario, "max_density_levels", 3) or 3),
            traffic_density_augment=bool(
                getattr(cfg.simulation, "traffic_density_augment", True)
            ),
        )
    else:
        entries = generate_manifest(
            scenes_dir=scenes_dir,
            output_dir=experiment_dir,
            scenario_cfg=scenario_cfg,
            sim_cfg=sim_cfg,
            aux_cfg=aux_cfg,
            expansion_cfg=expansion_cfg,
            split=split,
            expert_cfg=expert_cfg,
        )
    
    if gif_cfg.enabled and entries:
        manifest_path = experiment_dir / "real_manifest.jsonl"

        gif_rendered, gif_failed = render_gifs_from_manifest(
            manifest_path=manifest_path,
            experiment_dir=experiment_dir,
            scenes_root=scenes_dir,
            gif_cfg=gif_cfg,
            aux_cfg=aux_cfg,
        )

        resolved_gif_dir = Path(gif_cfg.dir) if gif_cfg.dir else (experiment_dir / "gifs")
        print(f"\n[GIF RESULTS]")
        print(f"  - GIFs rendered: {gif_rendered}")
        print(f"  - GIF failures: {gif_failed}")
        print(f"  - Experiment directory: {experiment_dir}")
        print(f"  - GIF directory: {resolved_gif_dir}")
    
    print(f"\nOutput files:")
    print(f"  - Manifest: {experiment_dir / 'real_manifest.jsonl'}")
    print(f"  - Repro: {experiment_dir / 'repro'}")
    print(f"  - Config: {config_path}")


if __name__ == "__main__":
    main()
