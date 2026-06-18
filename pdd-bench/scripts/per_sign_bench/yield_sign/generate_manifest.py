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

from lib.junction_priority_layout import JunctionLayoutError, build_junction_priority_layout
from lib.lane_keys import make_lane_key
from lib.auxiliary_agent import (
    DEFAULT_CONVOY_GAP_M,
    DEFAULT_CONVOY_SIZE,
    main_lane_keys_for_aux,
    select_occupied_main_lanes,
)
from lib.manifest_config import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_AUX_LANES_OCCUPIED_MAX,
    DEFAULT_SPAWN_DISTANCE_BEFORE_END,
)
from lib.scene_augmentation import SpawnScenario, augment_layout_for_scene


SCRIPT_DIR = Path(__file__).parent.resolve()
RUN_BENCH_SCRIPT = SCRIPT_DIR / "run_benchmark.py"

PDD_CODE = "2.4"
SIGN_TYPE = "yield"


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
    max_scenarios_per_scene: Optional[int] = None


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


@dataclass
class GifConfig:
    enabled: bool = False
    policy: str = "idm"
    max_scenes: Optional[int] = None
    dry_run: bool = False
    hide_signs: bool = True
    dir: Optional[str] = None
    run_name: Optional[str] = None


@dataclass
class ManifestConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
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
def build_junction_layout_for_scene(net_path: Path) -> Optional[dict]:
    """Build main/secondary junction layout from a scene net.xml."""
    try:
        layout = build_junction_priority_layout(net_path)
    except JunctionLayoutError as exc:
        print(f"  [junction_layout] {net_path.parent.name}: {exc}")
        return None
    return layout.to_dict()


def filter_spawn_lanes_to_secondary(
    spawn_lanes: List[SumoLaneInfo],
    junction_layout: Optional[dict],
) -> List[SumoLaneInfo]:
    """Keep only lanes on secondary junction arms."""
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
    return int.from_bytes(h.digest()[:4], "big")


# -----------------------------------------------------------------------------
# Auxiliary agent dimension expansion
# -----------------------------------------------------------------------------
def sizes_up_to(
    max_value: int,
    *,
    auxiliary_enabled: bool = True,
    available: Optional[int] = None,
) -> List[int]:
    """Return values to materialize: {1, 2, ..., cap} for aux manifest expansion."""
    if not auxiliary_enabled:
        return [1]
    if available is not None and available <= 0:
        return [1]
    cap = max(1, int(max_value))
    if available is not None:
        cap = min(cap, int(available))
    return list(range(1, cap + 1))


# -----------------------------------------------------------------------------
# Scene discovery and metadata
# -----------------------------------------------------------------------------
def discover_scenes(scenes_dir: Path) -> List[Path]:
    """Find all valid scene directories containing meta.json and map.net.xml."""
    scenes = []
    for entry in sorted(scenes_dir.iterdir()):
        if not entry.is_dir():
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
    )

    if spawn_lanes_cache is None:
        spawn_lanes_cache = parse_sumo_net_for_spawn_lanes(net_full_path)

    spawn_candidates = spawn_lanes_cache
    if junction_layout_cache is not None and spawn_scenario is None:
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
            if suffix_parts:
                base_aug = entry.get("augmentation_id") or scenario_id
                entry["augmentation_id"] = f"{base_aug}_{'_'.join(suffix_parts)}"
        if selected_lane is not None:
            entry["spawn_lane_length"] = selected_lane.length
            entry["spawn_to_junction"] = selected_lane.to_junction
    
    if spawn_scenario is None and meta.get("road_id"):
        secondary_ids = set()
        if junction_layout_cache is not None:
            secondary_ids = set(junction_layout_cache.get("secondary_edge_ids") or [])
        if secondary_ids and meta["road_id"] not in secondary_ids:
            print(
                f"  [spawn] meta road_id {meta['road_id']!r} is not secondary; "
                "picking from secondary arms"
            )
            selected_lane = select_random_spawn_lane(spawn_candidates, seed)
            if selected_lane is not None:
                entry["road_id"] = selected_lane.edge_id
                entry["spawn_lane_num"] = selected_lane.lane_num
                entry["spawn_lane_length"] = selected_lane.length
                entry["spawn_to_junction"] = selected_lane.to_junction
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
        print(f"  [spawn] No secondary incoming lanes available for {scene_name}")
    
    if meta.get("distance_from_start"):
        entry["distance_from_start"] = meta["distance_from_start"]
    if meta.get("sign_spawn_distance"):
        entry["sign_spawn_distance"] = meta["sign_spawn_distance"]
    if spawn_scenario is None and meta.get("destination_lane_id"):
        entry["destination_lane_id"] = meta["destination_lane_id"]

    if junction_layout_cache is not None:
        entry["junction_layout"] = junction_layout_cache
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
            available_main = main_lane_keys_for_aux(junction_layout_cache, ego_edge)
            prefer_aux = None
            if spawn_scenario is not None:
                prefer_aux = make_lane_key(
                    spawn_scenario.aux_edge_id,
                    spawn_scenario.aux_lane_num,
                )
            entry["aux_occupied_lane_keys"] = select_occupied_main_lanes(
                available_main, aux_lanes_occupied, prefer_lane_key=prefer_aux
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
) -> List[Dict]:
    """Generate real_manifest.jsonl from discovered scenes."""
    scenes = discover_scenes(scenes_dir)
    entries = []
    
    for scene_dir in scenes:
        meta = load_scene_metadata(scene_dir)
        scene_name = meta.get("scene_name", scene_dir.name)
        net_file = meta.get("net_file", "map.net.xml")
        net_full_path = scene_dir / net_file
        
        spawn_lanes = parse_sumo_net_for_spawn_lanes(net_full_path)
        print(f"  Found {len(spawn_lanes)} intersection-approaching lane(s)")

        junction_layout = build_junction_layout_for_scene(net_full_path)
        assert junction_layout is not None, f"No junction layout found for {scene_name}"
        print(
            f"  Junction layout: {junction_layout['shape']} @ {junction_layout['junction_id']} "
            f"(main={len(junction_layout.get('main_edge_ids', []))}, "
            f"secondary={len(junction_layout.get('secondary_edge_ids', []))})"
        )
        
        available_main_lane_count = len(main_lane_keys_for_aux(junction_layout))
        print(f"  Main-road lane slots for aux: {available_main_lane_count}")

        scenarios: List[SpawnScenario] = []
        if scenario_cfg.augment:
            _, scenarios = augment_layout_for_scene(net_full_path, spawn_lanes)
            if not scenarios:
                print(f"  [augment] No valid scenarios for {scene_name}; skipping scene")
                continue
            if scenario_cfg.max_scenarios_per_scene is not None:
                print(f"Retained only {scenario_cfg.max_scenarios_per_scene} from {len(scenarios)} possible scenarios")
                random.shuffle(scenarios)
                scenarios = scenarios[:scenario_cfg.max_scenarios_per_scene]
            print(f"  Augmented scenarios: {len(scenarios)}")
        
        convoy_sizes = sizes_up_to(aux_cfg.convoy_size, auxiliary_enabled=aux_cfg.enabled)
        lanes_counts = sizes_up_to(
            aux_cfg.lanes_occupied,
            auxiliary_enabled=aux_cfg.enabled,
            available=available_main_lane_count,
        )

        for scenario in scenarios:
            ego_edge = scenario.ego_edge_id
            scene_main_lanes = main_lane_keys_for_aux(junction_layout, ego_edge)
            scene_lane_counts = sizes_up_to(
                aux_cfg.lanes_occupied,
                auxiliary_enabled=aux_cfg.enabled,
                available=len(scene_main_lanes),
            )
            for lanes_n in scene_lane_counts:
                for convoy_n in convoy_sizes:
                    entry = build_manifest_entry(
                        scene_dir=scene_dir,
                        scenes_root=scenes_dir,
                        meta=meta,
                        variant=variant,
                        sim_cfg=sim_cfg,
                        aux_cfg=aux_cfg,
                        aux_convoy_size=convoy_n,
                        aux_lanes_occupied=lanes_n,
                        spawn_lanes_cache=spawn_lanes,
                        junction_layout_cache=junction_layout,
                        spawn_scenario=scenario,
                    )
                    entries.append(entry)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "real_manifest.jsonl"
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, default=str) + "\n")
    
    summary = {
        "pdd_code": PDD_CODE,
        "sign_type": SIGN_TYPE,
        "sign_name": "Yield",
        "total_scenes": len(scenes),
        "total_entries": len(entries),
        "variants_per_scene": scenario_cfg.n_variants,
        "augment": scenario_cfg.augment,
        "max_scenarios_per_scene": scenario_cfg.max_scenarios_per_scene,
        "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
        "traffic_density": sim_cfg.traffic_density,
        "horizon": sim_cfg.horizon,
        "sign_distance_before_end": sim_cfg.sign_distance_before_end,
        "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
        "auxiliary_agent": aux_cfg.enabled,
        "aux_distance_from_intersection": aux_cfg.distance_from_intersection,
        "aux_convoy_size_max": aux_cfg.convoy_size,
        "aux_convoy_gap_m": aux_cfg.convoy_gap_m,
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
        
        if aux_cfg.enabled:
            cmd.append("--auxiliary-agent")
            cmd.extend(["--aux-distance-from-intersection", str(aux_cfg.distance_from_intersection)])
        
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
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    scenes_dir = Path(cfg.paths.scenes_dir)

    experiment_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_path = experiment_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    scenario_cfg = ScenarioConfig(
        n_variants=cfg.scenario.n_variants,
        augment=cfg.scenario.augment,
        max_scenarios_per_scene=cfg.scenario.max_scenarios_per_scene,
    )
    sim_cfg = SimulationConfig(
        spawn_velocity_ms=cfg.simulation.spawn_velocity_ms,
        traffic_density=cfg.simulation.traffic_density,
        horizon=cfg.simulation.horizon,
        sign_distance_before_end=cfg.simulation.sign_distance_before_end,
        spawn_distance_before_end=cfg.simulation.spawn_distance_before_end,
    )
    aux_cfg = AuxiliaryConfig(
        enabled=cfg.auxiliary.enabled,
        distance_from_intersection=cfg.auxiliary.distance_from_intersection,
        convoy_size=cfg.auxiliary.convoy_size,
        convoy_gap_m=cfg.auxiliary.convoy_gap_m,
        lanes_occupied=cfg.auxiliary.lanes_occupied,
    )
    gif_cfg = GifConfig(
        enabled=cfg.gif.enabled,
        policy=cfg.gif.policy,
        max_scenes=cfg.gif.max_scenes,
        dry_run=cfg.gif.dry_run,
        hide_signs=cfg.gif.hide_signs,
        dir=cfg.gif.dir,
        run_name=cfg.gif.run_name,
    )
    
    entries = generate_manifest(
        scenes_dir=scenes_dir,
        output_dir=experiment_dir,
        scenario_cfg=scenario_cfg,
        sim_cfg=sim_cfg,
        aux_cfg=aux_cfg,
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
