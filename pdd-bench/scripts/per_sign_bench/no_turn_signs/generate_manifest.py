#!/usr/bin/env python3
"""Generate evaluation manifest from scenes (no-turn signs 3.18.1 / 3.18.2).

Dual-path rows reuse crop-time spawn/dest from scene ``meta.json``: the same
destination is reachable by a shorter *forbidden* (baseline) first exit and a
longer *allowed* (compliant) path. Roles depend on ``sign.pdd_code``
(e.g. 3.18.1: baseline r + compliant s/l; 3.18.2: baseline l + compliant s/r).
``run_benchmark.py`` places the matching ``NoRightTurnSign`` / ``NoLeftTurnSign``.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from lib.junction_priority_layout import JunctionLayoutError, build_junction_priority_layout
from lib.lane_keys import make_lane_key
from lib.direction_dual_path import (
    DualPathScenario,
    dual_path_scenario_from_meta,
    straight_path_has_dead_end_uturn,
    straight_path_reenters_signed_junction,
)
from lib.no_turn_sign_spec import (
    DEFAULT_PDD_CODE,
    SIGN_FAMILY,
    NoTurnSignSpec,
    get_no_turn_sign_spec,
)
from lib.manifest_config import DEFAULT_SPAWN_DISTANCE_BEFORE_END
from lib.metadrive_route_check import filter_dual_paths_metadrive
from lib.scene_augmentation import SpawnScenario
from lib.traffic_density_levels import (
    MAX_TRAFFIC_DENSITY_LEVELS,
    TrafficDensityLevel,
    list_traffic_density_levels,
)
from lib.scene_selection import is_reserved_scene_dir, is_scene_rejected
from lib.sumo_utils import CORE_SCENES_SUBDIR


SCRIPT_DIR = Path(__file__).parent.resolve()
RUN_BENCH_SCRIPT = SCRIPT_DIR / "run_benchmark.py"
PDD_BENCH_DIR = SCRIPT_DIR.parents[2]

# Hydra: paths.output_base: benchmark_output/${pdd_slug:${sign.pdd_code}}
# so ``sign.pdd_code=3.18.1`` lands in ``benchmark_output/3_18_1/…`` without a
# manual paths.output_base override.
OmegaConf.register_new_resolver(
    "pdd_slug",
    lambda code: str(code).replace(".", "_"),
    replace=True,
)
DEFAULT_CARL_CKPT = (
    PDD_BENCH_DIR / "checkpoints" / "carl" / "nuplan_51479_1B" / "model_best.pth"
)
DEFAULT_NN_CHECKPOINTS = {
    "carl": DEFAULT_CARL_CKPT,
    "carl_rule": DEFAULT_CARL_CKPT,
    "plant2": PDD_BENCH_DIR / "checkpoints" / "plant2_finetuned" / "plant2_supervised_2nd_final.pt",
    "plant2_rule": PDD_BENCH_DIR
    / "checkpoints"
    / "plant2_finetuned"
    / "plant2_supervised_2nd_final.pt",
}

# Active sign for this process (set in ``main`` from Hydra). Module-level aliases
# keep the rest of the file close to other junction benches.
_ACTIVE_SIGN: NoTurnSignSpec = get_no_turn_sign_spec(DEFAULT_PDD_CODE)
PDD_CODE = _ACTIVE_SIGN.pdd_code
SIGN_TYPE = SIGN_FAMILY


def _set_active_sign(pdd_code: str | None) -> NoTurnSignSpec:
    global _ACTIVE_SIGN, PDD_CODE, SIGN_TYPE
    _ACTIVE_SIGN = get_no_turn_sign_spec(pdd_code)
    PDD_CODE = _ACTIVE_SIGN.pdd_code
    SIGN_TYPE = SIGN_FAMILY
    return _ACTIVE_SIGN


def dual_path_to_spawn_scenario(dp: DualPathScenario) -> SpawnScenario:
    """Adapt a dual-path pick to the junction SpawnScenario shape."""
    dest_key = make_lane_key(dp.dest_edge_id, dp.dest_lane_num)
    return SpawnScenario(
        ego_edge_id=dp.ego_edge_id,
        ego_lane_num=dp.ego_lane_num,
        ego_destination_edge_id=dp.dest_edge_id,
        ego_destination_lane_key=dest_key,
        scenario_id=(
            f"dual_{dp.junction_id}_{dp.ego_edge_id}_{dp.dest_edge_id}_{dp.turn_dir}"
        ),
    )


def resolve_dual_path_scenarios_for_scene(
    net_path: Path,
    meta: Dict[str, Any],
    *,
    max_scenarios: Optional[int] = None,
    min_gain_m: float = 20.0,
    min_lane_length_m: float = 8.0,
) -> List[DualPathScenario]:
    """Load the crop-time dual-path endpoints from ``meta.json``.

    Spawn/dest and both paths are fixed at crop time; manifest must not
    rediscover alternate destinations. Unused kwargs kept for call-site compat.
    """
    del max_scenarios, min_gain_m, min_lane_length_m  # meta is source of truth
    scenario = dual_path_scenario_from_meta(meta)
    if scenario is None:
        return []
    if straight_path_has_dead_end_uturn(net_path, scenario):
        print(
            f"    skip: straight path U-turns at a dead end "
            f"(junction {scenario.junction_id}, dest {scenario.dest_edge_id})"
        )
        return []
    if straight_path_reenters_signed_junction(net_path, scenario):
        print(
            f"    skip: compliant path revisits signed approach "
            f"{scenario.ego_edge_id} after the first exit "
            f"(junction {scenario.junction_id}, dest={scenario.dest_edge_id}) "
            f"— sign would still forbid the second pass"
        )
        return []
    return [scenario]


def filter_dual_paths_to_metadrive_routes(
    scenarios: List[DualPathScenario],
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    *,
    max_keep: Optional[int] = None,
) -> List[DualPathScenario]:
    """Sanity-check crop-time dests are still MetaDrive-routable on the cropped net."""
    if not scenarios:
        return []
    net_file = meta.get("net_file", "map.net.xml")
    net_path = (scene_dir / net_file).resolve()
    try:
        filtered, dropped = filter_dual_paths_metadrive(
            scenarios,
            net_path,
            one_per_ego=False,
            max_keep=max_keep,
            pdd_code=PDD_CODE,
        )
    except Exception as exc:
        print(f"  [dual-path] MetaDrive route validation failed: {exc}")
        return scenarios[: max_keep] if max_keep is not None else scenarios
    if dropped:
        print(
            f"  [dual-path] MetaDrive check: kept {len(filtered)}, "
            f"dropped {dropped} unroutable dest(s) from crop meta"
        )
    return filtered


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------
@dataclass
class PathsConfig:
    scenes_dir: Optional[str] = None
    output_base: Optional[str] = None
    experiment_name: Optional[str] = None


@dataclass
class ScenarioConfig:
    n_variants: int = 1
    augment: bool = True
    max_scenarios: Optional[int] = None
    max_scenarios_per_scene: Optional[int] = None
    respect_scene_selection: bool = True
    min_dual_path_gain_m: float = 20.0
    min_ego_lane_m: float = 8.0
    validate_metadrive_routes: bool = True


@dataclass
class SimulationConfig:
    spawn_velocity_ms: float = 2.5
    traffic_density: float = 0.0
    traffic_density_augment: bool = True
    horizon: int = 600
    sign_distance_before_end: float = 0.0
    spawn_distance_before_end: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END


@dataclass
class GifConfig:
    enabled: bool = False
    policy: str = "idm"
    max_scenes: Optional[int] = None
    dry_run: bool = False
    hide_signs: bool = True
    dir: Optional[str] = None
    run_name: Optional[str] = None
    model_path: Optional[str] = None  # Required for carl/plant2; default from checkpoints/


@dataclass
class ManifestConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    gif: GifConfig = field(default_factory=GifConfig)


# -----------------------------------------------------------------------------
# SUMO network parsing
# -----------------------------------------------------------------------------
@dataclass
class SumoLaneInfo:
    """Information about a SUMO lane suitable for spawning."""
    edge_id: str
    lane_num: int
    lane_id: str
    length: float
    to_junction: str
    junction_type: str


def parse_sumo_net_for_spawn_lanes(net_path: Path, min_length: float = 20.0) -> List[SumoLaneInfo]:
    """Parse SUMO .net.xml and find lanes that lead to intersections."""
    if not net_path.exists():
        return []
    
    tree = ET.parse(net_path)
    root = tree.getroot()
    
    junctions = {}
    for junction in root.findall("junction"):
        jid = junction.get("id")
        jtype = junction.get("type", "unknown")
        junctions[jid] = jtype
    
    intersection_types = {"priority", "right_before_left", "allway_stop", "traffic_light"}
    spawn_lanes = []
    
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        func = edge.get("function", "normal")
        
        if func == "internal" or edge_id.startswith(":"):
            continue
        
        to_junction = edge.get("to", "")
        junction_type = junctions.get(to_junction, "unknown")
        
        if junction_type not in intersection_types:
            continue
        
        for lane in edge.findall("lane"):
            lane_id = lane.get("id", "")
            length = float(lane.get("length", 0))
            
            if length == 0:
                shape_str = lane.get("shape", "")
                if shape_str:
                    points = shape_str.strip().split()
                    coords = [tuple(map(float, p.split(','))) for p in points if ',' in p]
                    if len(coords) >= 2:
                        length = sum(
                            ((coords[i+1][0] - coords[i][0])**2 + 
                             (coords[i+1][1] - coords[i][1])**2)**0.5
                            for i in range(len(coords) - 1)
                        )
            
            if length < min_length:
                continue
            
            try:
                lane_num = int(lane_id.rsplit("_", 1)[1])
            except (ValueError, IndexError):
                lane_num = 0
            
            spawn_lanes.append(SumoLaneInfo(
                edge_id=edge_id,
                lane_num=lane_num,
                lane_id=f"lane_{lane_id}",
                length=length,
                to_junction=to_junction,
                junction_type=junction_type,
            ))
    
    return spawn_lanes


# -----------------------------------------------------------------------------
# Junction layout utilities
# -----------------------------------------------------------------------------
def build_junction_layout_for_scene(
    net_path: Path,
    *,
    sign_lat: Optional[float] = None,
    sign_lon: Optional[float] = None,
    preferred_junction_id: Optional[str] = None,
) -> Optional[dict]:
    """Build junction layout from a scene net.xml (shared scaffold for no-turn signs).

    Prefer ``preferred_junction_id`` from dual-path crop meta — catalog lat/lon
    often points at a *different* OSM junction in the same crop and would make
    ``_discover_primary_junction`` pick the wrong arms for sign placement.
    """
    try:
        layout = build_junction_priority_layout(
            net_path,
            mode="main_main",
            sign_lat=sign_lat,
            sign_lon=sign_lon,
            preferred_junction_id=preferred_junction_id,
        )
    except JunctionLayoutError as exc:
        print(f"  [junction_layout] {net_path.parent.name}: {exc}")
        return None
    return layout.to_dict()


def select_random_spawn_lane(
    spawn_lanes: List[SumoLaneInfo],
    seed: int,
) -> Optional[SumoLaneInfo]:
    """Select a random lane from available spawn lanes."""
    if not spawn_lanes:
        return None
    rng = random.Random(seed)
    return rng.choice(spawn_lanes)


# -----------------------------------------------------------------------------
# Experiment directory management
# -----------------------------------------------------------------------------
# Experiment dir is created by Hydra (see config/config.yaml hydra.run.dir).

# -----------------------------------------------------------------------------
# Seed generation
# -----------------------------------------------------------------------------
def _stable_seed(
    scene_name: str,
    variant: int = 0,
    scenario_id: str = "",
) -> int:
    """Generate deterministic 32-bit seed from scene name, variant, and scenario."""
    h = hashlib.sha256()
    h.update(scene_name.encode("utf-8"))
    h.update(b"|")
    h.update(str(variant).encode("utf-8"))
    if scenario_id:
        h.update(b"|")
        h.update(scenario_id.encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")


# -----------------------------------------------------------------------------
# Scene discovery and metadata
# -----------------------------------------------------------------------------
def discover_scenes(
    scenes_dir: Path,
    *,
    respect_scene_selection: bool = True,
) -> List[Path]:
    """Find cropped scene directories (skip core/ reserved dirs and rejected scenes)."""
    scenes = []
    for entry in sorted(scenes_dir.iterdir()):
        if not entry.is_dir():
            continue
        if is_reserved_scene_dir(entry.name) or entry.name == CORE_SCENES_SUBDIR:
            continue
        if respect_scene_selection and is_scene_rejected(scenes_dir, entry.name):
            continue
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        net_file = meta.get("net_file", "map.net.xml")
        net_path = entry / net_file
        if net_path.exists():
            scenes.append(entry)
    return scenes


def load_scene_metadata(scene_dir: Path) -> Dict:
    """Load and parse scene metadata from meta.json."""
    meta_path = scene_dir / "meta.json"
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    center_path = scene_dir / "center.json"
    if center_path.exists():
        with open(center_path, "r", encoding="utf-8") as f:
            center = json.load(f)
            meta["center_lat"] = center.get("lat")
            meta["center_lon"] = center.get("lon")
    return meta


# -----------------------------------------------------------------------------
# Manifest entry builder
# -----------------------------------------------------------------------------
def build_manifest_entry(
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict,
    variant: int,
    sim_cfg: SimulationConfig,
    spawn_lanes_cache: Optional[List[SumoLaneInfo]] = None,
    junction_layout_cache: Optional[dict] = None,
    spawn_scenario: Optional[SpawnScenario] = None,
    dual_path: Optional[DualPathScenario] = None,
    density_level: Optional[TrafficDensityLevel] = None,
) -> Dict:
    """Build a single manifest entry for a scene."""
    scene_name = meta.get("scene_name", scene_dir.name)
    net_file = meta.get("net_file", "map.net.xml")
    
    net_path = scene_dir.relative_to(scenes_root) / net_file
    net_full_path = scene_dir / net_file
    
    scenario_id = spawn_scenario.scenario_id if spawn_scenario else ""
    traffic_density = (
        float(density_level.traffic_density)
        if density_level is not None
        else float(sim_cfg.traffic_density)
    )
    if density_level is not None:
        seed_key = f"{scenario_id}_td{density_level.id}"
        scene_id = f"{scene_name}_td{density_level.id}"
    else:
        seed_key = scenario_id
        scene_id = scene_name
    seed = _stable_seed(scene_name, variant, seed_key)

    if spawn_lanes_cache is None:
        spawn_lanes_cache = parse_sumo_net_for_spawn_lanes(
            net_full_path, min_length=min(8.0, sim_cfg.spawn_distance_before_end)
        )

    selected_lane = None
    if spawn_scenario is not None:
        for lane in spawn_lanes_cache:
            if (
                lane.edge_id == spawn_scenario.ego_edge_id
                and lane.lane_num == spawn_scenario.ego_lane_num
            ):
                selected_lane = lane
                break
        if selected_lane is None:
            for lane in spawn_lanes_cache:
                if lane.edge_id == spawn_scenario.ego_edge_id:
                    selected_lane = lane
                    break
    else:
        selected_lane = select_random_spawn_lane(spawn_lanes_cache, seed)
    
    entry = {
        "scene_id": scene_id,
        "scene_name": scene_name,
        "net_path": str(net_path),
        "seed": seed,
        "var_idx": variant,
        "pdd_code": PDD_CODE,
        "sign_code": PDD_CODE,
        "sign_type": SIGN_TYPE,
        "sign_family": SIGN_FAMILY,
        "allowed_dirs": sorted(_ACTIVE_SIGN.allowed_dirs),
        "forbidden_dir": _ACTIVE_SIGN.forbidden_dir,
        "sign_title": _ACTIVE_SIGN.title,
        "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
        "traffic_density": traffic_density,
        "traffic_density_level_id": (
            density_level.id if density_level is not None else None
        ),
        "traffic_density_level_name": (
            density_level.name if density_level is not None else None
        ),
        "nuplan_vehicles_per_frame": (
            density_level.nuplan_vehicles_per_frame
            if density_level is not None
            else None
        ),
        "horizon": sim_cfg.horizon,
        "sign_distance_before_end": sim_cfg.sign_distance_before_end,
        "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
        "valid": True,
        "latitude": meta.get("latitude"),
        "longitude": meta.get("longitude"),
        "crop_radius_m": meta.get("crop_radius_m") or meta.get("crop_margin_m"),
        "source_osm": meta.get("source_osm"),
        "osm_file": meta.get("osm_file"),
        "junction_id": meta.get("junction_id"),
    }
    
    if spawn_scenario is not None:
        entry.update(spawn_scenario.to_manifest_fields())
        if selected_lane is not None:
            entry["spawn_lane_length"] = selected_lane.length
            entry["spawn_to_junction"] = selected_lane.to_junction
    
    if spawn_scenario is None and meta.get("road_id"):
        entry["road_id"] = meta["road_id"]
        if meta.get("spawn_lane_num") is not None:
            entry["spawn_lane_num"] = meta["spawn_lane_num"]
    elif selected_lane is not None and spawn_scenario is None:
        entry["road_id"] = selected_lane.edge_id
        entry["spawn_lane_num"] = selected_lane.lane_num
        entry["spawn_lane_length"] = selected_lane.length
        entry["spawn_to_junction"] = selected_lane.to_junction
    elif spawn_scenario is None and junction_layout_cache is not None:
        entry["valid"] = False
        print(f"  [spawn] No incoming lanes available for {scene_name}")
    
    if meta.get("distance_from_start"):
        entry["distance_from_start"] = meta["distance_from_start"]
    if meta.get("sign_spawn_distance"):
        entry["sign_spawn_distance"] = meta["sign_spawn_distance"]
    if spawn_scenario is None and meta.get("destination_lane_id"):
        entry["destination_lane_id"] = meta["destination_lane_id"]

    if dual_path is not None:
        entry["dual_path"] = dual_path.to_meta_fields()["dual_path"]
        entry["baseline_turn_dir"] = dual_path.turn_dir
        entry["turn_length_m"] = dual_path.turn_length_m
        entry["straight_length_m"] = dual_path.straight_length_m
        entry["dual_path_gain_m"] = dual_path.gain_m
        # turn_* = baseline (short, forbidden); straight_* = compliant (long, allowed).
        entry["compliant_dir"] = dual_path.compliant_dir
        entry["baseline_dir"] = dual_path.turn_dir
        entry["compliant_first_exit"] = dual_path.straight_first_exit
        entry["baseline_first_exit"] = dual_path.turn_first_exit

    if junction_layout_cache is not None:
        entry["junction_layout"] = junction_layout_cache
    
    entry = {k: v for k, v in entry.items() if v is not None}
    
    return entry


# -----------------------------------------------------------------------------
# Manifest generation
# -----------------------------------------------------------------------------
def generate_manifest(
    scenes_dir: Path,
    output_dir: Path,
    scenario_cfg: ScenarioConfig,
    sim_cfg: SimulationConfig,
) -> List[Dict]:
    """Generate real_manifest.jsonl from dual-path direction-sign scenes."""
    scenes = discover_scenes(
        scenes_dir,
        respect_scene_selection=scenario_cfg.respect_scene_selection,
    )
    entries: List[Dict] = []
    print(
        f"[no_turn_signs] Generating manifest for {PDD_CODE} "
        f"({_ACTIVE_SIGN.title}); scenes={len(scenes)}"
    )

    density_levels: List[Optional[TrafficDensityLevel]]
    if sim_cfg.traffic_density_augment:
        density_levels = list(list_traffic_density_levels(MAX_TRAFFIC_DENSITY_LEVELS))
        print("[no_turn_signs] Traffic density levels (nuPlan):")
        for level in density_levels:
            print(f"  td{level.id} {level.describe()}")
    else:
        density_levels = [None]

    n_variants = max(1, int(scenario_cfg.n_variants))
    max_total = scenario_cfg.max_scenarios
    if max_total is not None:
        max_total = max(1, int(max_total))

    # max_scenarios caps dual-path geometry picks; density multiplies rows.
    n_density = max(1, len(density_levels))
    max_total_rows = None if max_total is None else max_total * n_density
    base_scenarios_kept = 0

    for scene_dir in scenes:
        if max_total is not None and base_scenarios_kept >= max_total:
            print(
                f"\n[cap] reached max_scenarios={max_total} "
                f"({len(entries)} row(s) with density×{n_density}); stopping"
            )
            break

        meta = load_scene_metadata(scene_dir)
        scene_name = meta.get("scene_name", scene_dir.name)
        net_file = meta.get("net_file", "map.net.xml")
        net_full_path = scene_dir / net_file
        print(f"\n=== {scene_name} ===")

        spawn_lanes = parse_sumo_net_for_spawn_lanes(
            net_full_path,
            min_length=min(scenario_cfg.min_ego_lane_m, sim_cfg.spawn_distance_before_end),
        )
        print(f"  Found {len(spawn_lanes)} approach lane(s)")

        sign_lat = meta.get("latitude") or meta.get("center_lat")
        sign_lon = meta.get("longitude") or meta.get("center_lon")
        preferred_jid = meta.get("junction_id")
        junction_layout = build_junction_layout_for_scene(
            net_full_path,
            sign_lat=float(sign_lat) if sign_lat is not None else None,
            sign_lon=float(sign_lon) if sign_lon is not None else None,
            preferred_junction_id=str(preferred_jid) if preferred_jid else None,
        )
        if junction_layout is not None:
            print(
                f"  Junction layout: {junction_layout['shape']} @ {junction_layout['junction_id']} "
                f"(arms={len(junction_layout.get('arms', []))})"
            )
            ego_road = meta.get("road_id")
            arm_ids = {a.get("edge_id") for a in junction_layout.get("arms", [])}
            if ego_road and ego_road not in arm_ids:
                print(
                    f"  [warn] meta.road_id={ego_road!r} is not an arm of "
                    f"junction {junction_layout['junction_id']} — check dual-path crop"
                )
        else:
            print("  Junction layout: unavailable (sign will use ego-lane placement)")

        scene_pdd = str(meta.get("pdd_code") or PDD_CODE)
        if scene_pdd != PDD_CODE:
            print(
                f"  [skip] scene pdd_code={scene_pdd!r} != active {PDD_CODE!r} "
                f"(re-crop with --pdd-code {PDD_CODE})"
            )
            continue

        dual_paths = resolve_dual_path_scenarios_for_scene(
            net_full_path,
            meta,
            max_scenarios=scenario_cfg.max_scenarios_per_scene,
            min_gain_m=scenario_cfg.min_dual_path_gain_m,
            min_lane_length_m=scenario_cfg.min_ego_lane_m,
        )
        if scenario_cfg.validate_metadrive_routes and dual_paths:
            dual_paths = filter_dual_paths_to_metadrive_routes(
                dual_paths,
                scene_dir=scene_dir,
                scenes_root=scenes_dir,
                meta=meta,
                max_keep=scenario_cfg.max_scenarios_per_scene,
            )
        elif (
            scenario_cfg.max_scenarios_per_scene is not None
            and len(dual_paths) > scenario_cfg.max_scenarios_per_scene
        ):
            dual_paths = dual_paths[: scenario_cfg.max_scenarios_per_scene]
        if not dual_paths:
            print(
                f"  [skip] no dual-path in meta for {scene_name} "
                f"(re-run crop_junction_scene.py)"
            )
            continue

        print(f"  Dual-path from meta: {len(dual_paths)}")
        for dp in dual_paths:
            print(
                f"    ego={dp.ego_edge_id} dest={dp.dest_edge_id} "
                f"baseline={dp.turn_dir} compliant={dp.compliant_dir} "
                f"Lb={dp.turn_length_m:.0f}m Lc={dp.straight_length_m:.0f}m "
                f"gain={dp.gain_m:.0f}m"
            )

        scene_entries: List[Dict] = []
        for variant, dual in enumerate(dual_paths):
            if max_total is not None and base_scenarios_kept >= max_total:
                break
            spawn_scenario = dual_path_to_spawn_scenario(dual)
            for density_level in density_levels:
                for _rep in range(n_variants):
                    entry = build_manifest_entry(
                        scene_dir=scene_dir,
                        scenes_root=scenes_dir,
                        meta=meta,
                        variant=variant,
                        sim_cfg=sim_cfg,
                        spawn_lanes_cache=spawn_lanes,
                        junction_layout_cache=junction_layout,
                        spawn_scenario=spawn_scenario,
                        dual_path=dual,
                        density_level=density_level,
                    )
                    scene_entries.append(entry)
            base_scenarios_kept += 1

        if not scene_entries:
            continue

        # Cap counts dual-path picks; density levels multiply rows on top.
        max_entries = (
            scenario_cfg.max_scenarios_per_scene * len(density_levels)
            if scenario_cfg.max_scenarios_per_scene is not None
            else None
        )
        if max_entries is not None and len(scene_entries) > max_entries:
            print(
                f"  Retained {max_entries} of "
                f"{len(scene_entries)} manifest entries for {scene_name}"
            )
            random.shuffle(scene_entries)
            scene_entries = scene_entries[:max_entries]

        if (
            max_total_rows is not None
            and len(entries) + len(scene_entries) > max_total_rows
        ):
            keep = max_total_rows - len(entries)
            print(
                f"  [cap] retaining {keep}/{len(scene_entries)} row(s) "
                f"for max_scenarios={max_total} × density"
            )
            scene_entries = scene_entries[:keep]

        print(f"  Manifest entries for {scene_name}: {len(scene_entries)}")
        entries.extend(scene_entries)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "real_manifest.jsonl"

    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, default=str) + "\n")

    summary = {
        "pdd_code": PDD_CODE,
        "sign_type": SIGN_TYPE,
        "sign_family": SIGN_FAMILY,
        "sign_name": _ACTIVE_SIGN.title,
        "allowed_dirs": sorted(_ACTIVE_SIGN.allowed_dirs),
        "forbidden_dir": _ACTIVE_SIGN.forbidden_dir,
        "direction_route_filter": (
            f"dual_path_baseline_shorter_compliant_longer"
            f"(baseline={_ACTIVE_SIGN.forbidden_dir}, "
            f"compliant={sorted(_ACTIVE_SIGN.allowed_dirs)})"
        ),
        "sign_placement": "No*TurnSign on ego approach (road_id)",
        "total_scenes": len(scenes),
        "total_entries": len(entries),
        "variants_per_scene": scenario_cfg.n_variants,
        "augment": scenario_cfg.augment,
        "max_scenarios": scenario_cfg.max_scenarios,
        "max_scenarios_per_scene": scenario_cfg.max_scenarios_per_scene,
        "min_dual_path_gain_m": scenario_cfg.min_dual_path_gain_m,
        "validate_metadrive_routes": scenario_cfg.validate_metadrive_routes,
        "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
        "traffic_density": sim_cfg.traffic_density,
        "traffic_density_augment": sim_cfg.traffic_density_augment,
        "traffic_density_levels": [
            {
                "id": level.id,
                "name": level.name,
                "percentile": level.percentile,
                "nuplan_vehicles_per_frame": level.nuplan_vehicles_per_frame,
                "traffic_density": level.traffic_density,
                "description": level.describe(),
            }
            for level in density_levels
            if level is not None
        ],
        "horizon": sim_cfg.horizon,
        "sign_distance_before_end": sim_cfg.sign_distance_before_end,
        "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
        "generated_at": datetime.now().isoformat(),
        "scenes": [s.name for s in scenes],
    }

    summary_path = output_dir / "real_manifest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    manifest_meta_path = output_dir / "manifest.json"
    manifest_meta = {"entries_file": "real_manifest.jsonl", **summary}
    with open(manifest_meta_path, "w", encoding="utf-8") as f:
        json.dump(manifest_meta, f, indent=2, ensure_ascii=False)

    print(f"\n[no_turn_signs] Wrote {len(entries)} entries -> {manifest_path}")
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

    model_path = gif_cfg.model_path
    if not model_path:
        default_ckpt = DEFAULT_NN_CHECKPOINTS.get(gif_cfg.policy)
        if default_ckpt is not None and Path(default_ckpt).is_file():
            model_path = str(default_ckpt)
    
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
        
        print(f"\n[GIF {i}/{len(rows)}] {scene_uid}")
        print("  " + " ".join(cmd))
        
        if gif_cfg.dry_run:
            rendered += 1
            continue

        seed_val = int(row.get("seed") or 0)
        var_idx = int(row.get("var_idx", 0) or 0)
        expected_gif = (
            gif_dir
            / f"{row['scene_id']}_v{var_idx}_s{seed_val}_{gif_cfg.policy}_default.gif"
        )
        if expected_gif.is_file():
            expected_gif.unlink()

        res = subprocess.run(cmd, cwd=str(RUN_BENCH_SCRIPT.parent))
        if res.returncode == 0 and expected_gif.is_file():
            rendered += 1
        else:
            failed += 1
            if res.returncode != 0:
                print(f"[GIF] Command failed with code {res.returncode}")
            else:
                print(
                    f"[GIF] Episode finished but GIF missing (likely bad route): "
                    f"{expected_gif.name}"
                )
    
    return rendered, failed


# -----------------------------------------------------------------------------
# Hydra entry point
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    sign_cfg = getattr(cfg, "sign", None)
    pdd_code = getattr(sign_cfg, "pdd_code", None) if sign_cfg is not None else None
    active = _set_active_sign(pdd_code)
    print(
        f"[no_turn_signs] Active sign {active.pdd_code} "
        f"({active.title}), forbidden={active.forbidden_dir}, "
        f"allowed_dirs={sorted(active.allowed_dirs)}"
    )

    scenes_dir_cfg = getattr(cfg.paths, "scenes_dir", None)
    scenes_base_cfg = getattr(cfg.paths, "scenes_base", "scenes") or "scenes"
    if scenes_dir_cfg in (None, "", "null"):
        scenes_dir = Path(scenes_base_cfg) / active.output_slug
    else:
        scenes_dir = Path(scenes_dir_cfg)
    if not scenes_dir.is_absolute():
        scenes_dir = (SCRIPT_DIR / scenes_dir).resolve()
    print(f"[no_turn_signs] Scenes dir: {scenes_dir}")

    experiment_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"[no_turn_signs] Output dir: {experiment_dir}")
    config_path = experiment_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    max_scenarios_raw = getattr(cfg.scenario, "max_scenarios", None)
    max_scenarios_per_scene_raw = getattr(
        cfg.scenario, "max_scenarios_per_scene", None
    )
    scenario_cfg = ScenarioConfig(
        n_variants=cfg.scenario.n_variants,
        augment=cfg.scenario.augment,
        max_scenarios=(
            None
            if max_scenarios_raw in (None, "", "null")
            else int(max_scenarios_raw)
        ),
        max_scenarios_per_scene=(
            None
            if max_scenarios_per_scene_raw in (None, "", "null")
            else int(max_scenarios_per_scene_raw)
        ),
        respect_scene_selection=bool(
            getattr(cfg.scenario, "respect_scene_selection", True)
        ),
        min_dual_path_gain_m=float(
            getattr(cfg.scenario, "min_dual_path_gain_m", 20.0)
        ),
        min_ego_lane_m=float(getattr(cfg.scenario, "min_ego_lane_m", 8.0)),
        validate_metadrive_routes=bool(
            getattr(cfg.scenario, "validate_metadrive_routes", True)
        ),
    )
    sim_cfg = SimulationConfig(
        spawn_velocity_ms=cfg.simulation.spawn_velocity_ms,
        traffic_density=cfg.simulation.traffic_density,
        traffic_density_augment=bool(
            getattr(cfg.simulation, "traffic_density_augment", True)
        ),
        horizon=cfg.simulation.horizon,
        sign_distance_before_end=cfg.simulation.sign_distance_before_end,
        spawn_distance_before_end=cfg.simulation.spawn_distance_before_end,
    )
    gif_cfg = GifConfig(
        enabled=cfg.gif.enabled,
        policy=cfg.gif.policy,
        max_scenes=cfg.gif.max_scenes,
        dry_run=cfg.gif.dry_run,
        hide_signs=cfg.gif.hide_signs,
        dir=cfg.gif.dir,
        run_name=cfg.gif.run_name,
        model_path=getattr(cfg.gif, "model_path", None),
    )
    
    entries = generate_manifest(
        scenes_dir=scenes_dir,
        output_dir=experiment_dir,
        scenario_cfg=scenario_cfg,
        sim_cfg=sim_cfg,
    )
    
    if gif_cfg.enabled and entries:
        manifest_path = experiment_dir / "real_manifest.jsonl"

        gif_rendered, gif_failed = render_gifs_from_manifest(
            manifest_path=manifest_path,
            experiment_dir=experiment_dir,
            scenes_root=scenes_dir,
            gif_cfg=gif_cfg,
        )

        resolved_gif_dir = Path(gif_cfg.dir) if gif_cfg.dir else (experiment_dir / "gifs")
        print(f"\n[GIF RESULTS]")
        print(f"  - GIFs rendered: {gif_rendered}")
        print(f"  - GIF failures: {gif_failed}")
        print(f"  - Experiment directory: {experiment_dir}")
        print(f"  - GIF directory: {resolved_gif_dir}")
    
    print(f"\nOutput files:")
    print(f"  - Manifest: {experiment_dir / 'real_manifest.jsonl'}")
    print(f"  - Config: {config_path}")


if __name__ == "__main__":
    main()
