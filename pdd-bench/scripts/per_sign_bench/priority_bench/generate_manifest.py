#!/usr/bin/env python3
"""Generate evaluation manifest from scenes."""
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

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from core.junction_priority_layout import (
    ALLOWED_PRIORITY_JUNCTION_SHAPES,
    JunctionLayoutError,
    build_junction_priority_layout,
)
from core.lane_keys import lane_edge_id, lane_num_from_key, make_lane_key
from core.auxiliary_agent import (
    DEFAULT_CONVOY_GAP_M,
    resolve_aux_destination_lane_key,
    right_lane_keys_for_aux,
    select_occupied_main_lanes,
    viable_aux_lane_keys,
    viable_right_aux_lane_keys,
)
from core.sumo_utils import load_vehicle_route_index
from core.manifest_config import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_SPAWN_DISTANCE_BEFORE_END,
)
from core.manifest_expansion import (
    AuxiliaryParams,
    ExpansionConfig,
    expand_scene_entries,
)
from core.scene_augmentation import (
    SpawnScenario,
    pick_default_main_spawn_meta_for_net,
    pick_default_yield_spawn_meta_for_net,
)
from core.scene_selection import is_reserved_scene_dir, unapplied_rejected_scenes
from core.sumo_utils import is_vehicle_drivable_lane
from signs import SignProfile, get_profile, scenes_dir as profile_scenes_dir, output_dir as profile_output_dir

RUN_BENCH_SCRIPT = SCRIPT_DIR / "run_benchmark.py"

# Set by main() from Hydra `sign=` before generation runs.
PROFILE: SignProfile | None = None


def _profile() -> SignProfile:
    if PROFILE is None:
        raise RuntimeError("PROFILE is not set; call main() with sign=main_road|yield")
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


@dataclass
class ScenarioConfig:
    max_scenarios: Optional[int] = None


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
    scaling: float = 24.0


@dataclass
class ManifestConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    augmentation: AugmentationAxesConfig = field(default_factory=AugmentationAxesConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    auxiliary: AuxiliaryConfig = field(default_factory=AuxiliaryConfig)
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
            if not is_vehicle_drivable_lane(lane):
                continue
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
) -> Optional[dict]:
    """Build junction layout using the active sign profile's layout mode."""
    try:
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


def filter_spawn_lanes_to_secondary(
    spawn_lanes: List[SumoLaneInfo],
    junction_layout: Optional[dict],
) -> List[SumoLaneInfo]:
    """Keep only lanes on secondary junction arms (yield ego pool)."""
    if not junction_layout:
        return spawn_lanes
    secondary_ids = set(junction_layout.get("secondary_edge_ids") or [])
    if not secondary_ids:
        return []
    return [lane for lane in spawn_lanes if lane.edge_id in secondary_ids]


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
    convoy_size: int = 0,
    lanes_occupied: int = 0,
    convoy_gap_m: float = 0.0,
) -> int:
    """Generate deterministic 32-bit seed from scene name, variant, scenario, and aux dims."""
    h = hashlib.sha256()
    h.update(scene_name.encode("utf-8"))
    h.update(b"|")
    h.update(str(variant).encode("utf-8"))
    if scenario_id:
        h.update(b"|")
        h.update(scenario_id.encode("utf-8"))
    if convoy_size > 0:
        h.update(b"|convoy")
        h.update(str(convoy_size).encode("utf-8"))
    if lanes_occupied > 0:
        h.update(b"|lanes")
        h.update(str(lanes_occupied).encode("utf-8"))
    # Gap only affects spawn when there are 2+ convoy vehicles.
    if convoy_size > 1 and convoy_gap_m > 0:
        h.update(b"|gap")
        h.update(f"{float(convoy_gap_m):.3f}".encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")


# -----------------------------------------------------------------------------
# Scene discovery and metadata
# -----------------------------------------------------------------------------
def discover_scenes(scenes_dir: Path) -> List[Path]:
    """Find all valid scene directories containing meta.json and map.net.xml."""
    scenes = []
    for entry in sorted(scenes_dir.iterdir()):
        if not entry.is_dir() or is_reserved_scene_dir(entry.name):
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


def assert_rejected_scenes_applied(scenes_dir: Path) -> None:
    """Fail if review rejects were not moved aside with ``--apply``."""
    pending = unapplied_rejected_scenes(scenes_dir)
    if not pending:
        return
    preview = ", ".join(pending[:8])
    more = f" (+{len(pending) - 8} more)" if len(pending) > 8 else ""
    raise SystemExit(
        f"[error] {len(pending)} scene(s) are marked reject in scene_selection.json "
        f"but still live under {scenes_dir.resolve()}.\n"
        f"  Run: python build_scenes/review_scenes.py --apply\n"
        f"  Pending: {preview}{more}"
    )


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
    aux_cfg: AuxiliaryConfig,
    aux_convoy_size: int,
    aux_lanes_occupied: int,
    spawn_lanes_cache: Optional[List[SumoLaneInfo]] = None,
    junction_layout_cache: Optional[dict] = None,
    spawn_scenario: Optional[SpawnScenario] = None,
) -> Dict:
    """Build a single manifest entry for a scene."""
    scene_name = meta.get("scene_name", scene_dir.name)
    net_file = meta.get("net_file", "map.net.xml")
    
    net_path = scene_dir.relative_to(scenes_root) / net_file
    net_full_path = scene_dir / net_file
    
    scenario_id = spawn_scenario.scenario_id if spawn_scenario else ""
    seed = _stable_seed(
        scene_name,
        variant,
        scenario_id,
        convoy_size=aux_convoy_size,
        lanes_occupied=aux_lanes_occupied if aux_cfg.enabled else 0,
        convoy_gap_m=float(aux_cfg.convoy_gap_m) if aux_cfg.enabled else 0.0,
    )

    if spawn_lanes_cache is None:
        spawn_lanes_cache = parse_sumo_net_for_spawn_lanes(net_full_path)

    spawn_candidates = spawn_lanes_cache
    if (
        _profile().ego_road_class == "secondary"
        and junction_layout_cache is not None
        and spawn_scenario is None
    ):
        spawn_candidates = filter_spawn_lanes_to_secondary(
            spawn_lanes_cache, junction_layout_cache
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
    else:
        selected_lane = select_random_spawn_lane(spawn_candidates, seed)
    
    entry = {
        "scene_id": scene_name,
        "net_path": str(net_path),
        "seed": seed,
        "var_idx": variant,
        "pdd_code": PDD_CODE,
        "sign_code": PDD_CODE,
        "sign_type": SIGN_TYPE,
        "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
        "traffic_density": sim_cfg.traffic_density,
        "horizon": sim_cfg.horizon,
        "sign_distance_before_end": sim_cfg.sign_distance_before_end,
        "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
        "valid": True,
        "latitude": meta.get("latitude"),
        "longitude": meta.get("longitude"),
        "crop_radius_m": meta.get("crop_radius_m"),
        "source_osm": meta.get("source_osm"),
        "osm_file": meta.get("osm_file"),
        "auxiliary_agent": aux_cfg.enabled,
        "aux_distance_from_intersection": aux_cfg.distance_from_intersection,
        "aux_convoy_size": aux_convoy_size,
        "aux_convoy_gap_m": aux_cfg.convoy_gap_m,
        "aux_lanes_occupied": aux_lanes_occupied,
    }
    
    if spawn_scenario is not None:
        entry.update(spawn_scenario.to_manifest_fields())
        if aux_cfg.enabled:
            suffix_parts = []
            if aux_lanes_occupied > 1:
                suffix_parts.append(f"lanes{aux_lanes_occupied}")
            if aux_convoy_size > 1:
                suffix_parts.append(f"convoy{aux_convoy_size}")
                gap = float(aux_cfg.convoy_gap_m)
                suffix_parts.append(f"gap{gap:g}")
            if suffix_parts:
                base_aug = entry.get("augmentation_id") or scenario_id
                entry["augmentation_id"] = f"{base_aug}_{'_'.join(suffix_parts)}"
        if selected_lane is not None:
            entry["spawn_lane_length"] = selected_lane.length
            entry["spawn_to_junction"] = selected_lane.to_junction
    
    if spawn_scenario is None and meta.get("road_id"):
        if _profile().ego_road_class == "secondary" and junction_layout_cache is not None:
            secondary_ids = set(junction_layout_cache.get("secondary_edge_ids") or [])
            spawn_edge_ids = {lane.edge_id for lane in (spawn_lanes_cache or [])}
            road_id = str(meta["road_id"])
            road_id_invalid = road_id not in spawn_edge_ids or (
                secondary_ids and road_id not in secondary_ids
            )
            if road_id_invalid:
                print(
                    f"  [spawn] meta road_id {road_id!r} is not a valid secondary approach lane; "
                    "picking from secondary arms"
                )
                selected_lane = select_random_spawn_lane(spawn_candidates, seed)
                if selected_lane is not None:
                    entry["road_id"] = selected_lane.edge_id
                    entry["spawn_lane_num"] = selected_lane.lane_num
                    entry["spawn_lane_length"] = selected_lane.length
                    entry["spawn_to_junction"] = selected_lane.to_junction
            else:
                entry["road_id"] = road_id
                if meta.get("spawn_lane_num") is not None:
                    entry["spawn_lane_num"] = meta["spawn_lane_num"]
        else:
            entry["road_id"] = meta["road_id"]
            if meta.get("spawn_lane_num") is not None:
                entry["spawn_lane_num"] = meta["spawn_lane_num"]
    elif selected_lane is not None:
        entry["road_id"] = selected_lane.edge_id
        entry["spawn_lane_num"] = selected_lane.lane_num
        entry["spawn_lane_length"] = selected_lane.length
        entry["spawn_to_junction"] = selected_lane.to_junction
    elif junction_layout_cache is not None:
        entry["valid"] = False
        print(f"  [spawn] No incoming lanes available for {scene_name}")
    
    if meta.get("distance_from_start"):
        distance = float(meta["distance_from_start"])
        spawn_len = entry.get("spawn_lane_length")
        if spawn_len is not None:
            distance = min(distance, max(float(spawn_len) - 5.0, 10.0))
        entry["distance_from_start"] = distance
    if meta.get("sign_spawn_distance"):
        entry["sign_spawn_distance"] = meta["sign_spawn_distance"]
    if spawn_scenario is None and meta.get("destination_lane_id"):
        entry["destination_lane_id"] = meta["destination_lane_id"]
        if meta.get("destination_edge_id"):
            entry["destination_edge_id"] = meta["destination_edge_id"]
    elif spawn_scenario is None and entry.get("road_id"):
        picker = (
            pick_default_yield_spawn_meta_for_net
            if _profile().spawn_strategy == "yield"
            else pick_default_main_spawn_meta_for_net
        )
        spawn_meta = picker(
            net_full_path,
            prefer_ego_edge_id=str(entry["road_id"]),
        )
        if spawn_meta:
            entry["destination_lane_id"] = spawn_meta["destination_lane_id"]
            entry["destination_edge_id"] = spawn_meta["destination_edge_id"]
            entry["road_id"] = spawn_meta["road_id"]
            entry["spawn_lane_num"] = spawn_meta["spawn_lane_num"]

    if junction_layout_cache is not None:
        entry["junction_layout"] = junction_layout_cache
        if _profile().spawn_strategy == "yield":
            entry["main_lane_keys"] = [
                lane_key
                for arm in junction_layout_cache.get("arms", [])
                if arm.get("road_class") == "main"
                for lane_key in arm.get("lane_keys", [])
            ]
            entry["secondary_lane_keys"] = [
                lane_key
                for arm in junction_layout_cache.get("arms", [])
                if arm.get("road_class") == "secondary"
                for lane_key in arm.get("lane_keys", [])
            ]
            if aux_cfg.enabled:
                ego_edge = entry.get("road_id") or (
                    spawn_scenario.ego_edge_id if spawn_scenario is not None else None
                )
                available_main = viable_aux_lane_keys(
                    junction_layout_cache,
                    aux_cfg.distance_from_intersection,
                    ego_edge,
                    convoy_size=aux_convoy_size,
                    convoy_gap_m=aux_cfg.convoy_gap_m,
                )
                if not available_main:
                    entry["valid"] = False
                prefer_aux = None
                if spawn_scenario is not None:
                    prefer_aux = make_lane_key(
                        spawn_scenario.aux_edge_id,
                        spawn_scenario.aux_lane_num,
                    )
                entry["aux_occupied_lane_keys"] = select_occupied_main_lanes(
                    available_main, aux_lanes_occupied, prefer_lane_key=prefer_aux
                )
                if entry["aux_occupied_lane_keys"]:
                    primary_aux = entry["aux_occupied_lane_keys"][0]
                    entry["aux_spawn_lane_index"] = primary_aux
                    entry["aux_road_id"] = lane_edge_id(primary_aux)
                    entry["aux_spawn_lane_num"] = lane_num_from_key(primary_aux)
                    # Keep layout-scenario turn destinations; reachable turn is
                    # only a fallback when no aux dest was set on the scenario.
                    if not entry.get("aux_destination_lane_id"):
                        route_index = load_vehicle_route_index(net_full_path)
                        aux_dest = resolve_aux_destination_lane_key(
                            junction_layout_cache,
                            primary_aux,
                            route_index=route_index,
                        )
                        if aux_dest:
                            entry["aux_destination_lane_id"] = aux_dest
                            entry["aux_destination_edge_id"] = lane_edge_id(aux_dest)
        else:
            entry["main_lane_keys"] = [
                lane_key
                for arm in junction_layout_cache.get("arms", [])
                for lane_key in arm.get("lane_keys", [])
            ]
            ego_edge = entry.get("road_id") or (
                spawn_scenario.ego_edge_id if spawn_scenario is not None else None
            )
            entry["right_lane_keys"] = right_lane_keys_for_aux(
                junction_layout_cache, ego_edge
            )
            if aux_cfg.enabled:
                available_right = viable_right_aux_lane_keys(
                    junction_layout_cache,
                    aux_cfg.distance_from_intersection,
                    ego_edge,
                    convoy_size=aux_convoy_size,
                    convoy_gap_m=aux_cfg.convoy_gap_m,
                )
                prefer_aux = None
                if spawn_scenario is not None:
                    prefer_aux = make_lane_key(
                        spawn_scenario.aux_edge_id,
                        spawn_scenario.aux_lane_num,
                    )
                entry["right_arm_edge_id"] = (
                    spawn_scenario.aux_edge_id if spawn_scenario is not None else None
                )
                entry["aux_occupied_lane_keys"] = select_occupied_main_lanes(
                    available_right, aux_lanes_occupied, prefer_lane_key=prefer_aux
                )
    
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
    aux_cfg: AuxiliaryConfig,
    expansion_cfg: ExpansionConfig,
) -> List[Dict]:
    """Generate real_manifest.jsonl from discovered scenes."""
    assert_rejected_scenes_applied(scenes_dir)
    scenes = discover_scenes(scenes_dir)
    print(f"Scenes root: {scenes_dir.resolve()}")
    print(f"Discovered {len(scenes)} scene(s)")
    print(
        f"Augmentation axes: layout={expansion_cfg.layout_on}, "
        f"auxiliary={expansion_cfg.auxiliary_on}"
    )
    if not scenes:
        print(
            f"[warn] No scenes with meta.json + net found under {scenes_dir}. "
            "Check data/<sign>/scenes symlink / paths.scenes_dir."
        )
    entries = []

    # When the auxiliary axis is off, force aux disabled in row fields.
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
        )
        if junction_layout is None:
            print(f"  Skipping {scene_name}: no junction layout")
            continue
        shape = junction_layout.get("shape")
        if shape not in ALLOWED_PRIORITY_JUNCTION_SHAPES:
            print(
                f"  Skipping {scene_name}: junction shape {shape!r} "
                f"(need {sorted(ALLOWED_PRIORITY_JUNCTION_SHAPES)})"
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
            build_entry=build_manifest_entry,
            aux_cfg_for_entry=aux_for_entry,
        )
        entries.extend(scene_entries)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "real_manifest.jsonl"

    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, default=str) + "\n")

    summary = {
        "pdd_code": PDD_CODE,
        "sign_type": SIGN_TYPE,
        "sign_name": SIGN_NAME,
        "total_scenes": len(scenes),
        "total_entries": len(entries),
        "augmentation_layout": expansion_cfg.layout_on,
        "augmentation_auxiliary": expansion_cfg.auxiliary_on,
        "max_scenarios": scenario_cfg.max_scenarios,
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
        if gif_cfg.hide_signs:
            cmd.append("--hide-signs")
        if gif_cfg.scaling:
            cmd.extend(["--gif-scaling", str(float(gif_cfg.scaling))])
        
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


def _resolve_convoy_gaps_m(raw) -> List[float]:
    """Accept a scalar or list for ``auxiliary.convoy_gap_m``."""
    if raw is None:
        return [float(DEFAULT_CONVOY_GAP_M)]
    if OmegaConf.is_list(raw) or isinstance(raw, (list, tuple)):
        gaps = [float(x) for x in raw]
        return gaps if gaps else [float(DEFAULT_CONVOY_GAP_M)]
    return [float(raw)]


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    profile = get_profile(cfg.sign)
    _apply_profile(profile)
    print(f"Sign profile: {profile.id} ({profile.pdd_code} / {profile.sign_type})")
    expected_out = profile_output_dir(profile).resolve()
    configured_out = Path(str(cfg.paths.output_base))
    if not configured_out.is_absolute():
        configured_out = (SCRIPT_DIR / configured_out).resolve()
    if configured_out != expected_out and profile.data_subdir not in str(configured_out):
        print(
            f"[warn] paths.output_base={cfg.paths.output_base} may not match sign={profile.id}; "
            f"preferred: data/{profile.data_subdir}/output"
        )


    if str(cfg.paths.scenes_dir) in {"scenes", "scenes/", ""}:
        scenes_dir = profile_scenes_dir(profile)
    else:
        scenes_dir = Path(cfg.paths.scenes_dir)
        if not scenes_dir.is_absolute():
            scenes_dir = (SCRIPT_DIR / scenes_dir).resolve()
    print(f"Using scenes_dir: {scenes_dir}")

    experiment_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_path = experiment_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    scenario_cfg = ScenarioConfig(
        max_scenarios=_resolve_max_scenarios(cfg.scenario),
    )
    sim_cfg = SimulationConfig(
        spawn_velocity_ms=cfg.simulation.spawn_velocity_ms,
        traffic_density=cfg.simulation.traffic_density,
        horizon=cfg.simulation.horizon,
        sign_distance_before_end=cfg.simulation.sign_distance_before_end,
        spawn_distance_before_end=cfg.simulation.spawn_distance_before_end,
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
        scaling=float(getattr(cfg.gif, "scaling", 24.0) or 24.0),
    )

    entries = generate_manifest(
        scenes_dir=scenes_dir,
        output_dir=experiment_dir,
        scenario_cfg=scenario_cfg,
        sim_cfg=sim_cfg,
        aux_cfg=aux_cfg,
        expansion_cfg=expansion_cfg,
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
    print(f"  - Config: {config_path}")


if __name__ == "__main__":
    main()
