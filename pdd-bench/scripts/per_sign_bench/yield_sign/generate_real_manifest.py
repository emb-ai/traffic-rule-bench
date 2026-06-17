#!/usr/bin/env python3
"""
Real map manifest generator for sign 2.4 (Yield).

This script scans the scenes/ directory for manually collected SUMO scenes
and generates real_manifest.jsonl for policy evaluation.

Usage:
    python yield_sign/generate_real_manifest.py
    python yield_sign/generate_real_manifest.py --save-gifs --gif-policy idm
    python yield_sign/generate_real_manifest.py --save-gifs --gif-dry-run
    python yield_sign/generate_real_manifest.py --save-gifs --hide-signs

Each run writes to benchmark_output/2_4/<YYYY-MM-DD_HH-MM-SS>/ so previous
experiments are not overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from junction_priority_layout import JunctionLayoutError, build_junction_priority_layout
from manifest_config import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_SPAWN_DISTANCE_BEFORE_END,
)
from scene_augmentation import SpawnScenario, augment_layout_for_scene


SCRIPT_DIR = Path(__file__).parent
DEFAULT_SCENES_DIR = SCRIPT_DIR / "scenes"
DEFAULT_OUTPUT_BASE = SCRIPT_DIR / "benchmark_output" / "2_4"
RUN_BENCH_SCRIPT = SCRIPT_DIR / "run_benchmark_real.py"
EXPERIMENT_TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"

PDD_CODE = "2.4"
SIGN_TYPE = "yield"

@dataclass
class SumoLaneInfo:
    """Information about a SUMO lane suitable for spawning."""
    edge_id: str
    lane_num: int
    lane_id: str  # e.g., "lane_-123#0_0"
    length: float
    to_junction: str
    junction_type: str


def parse_sumo_net_for_spawn_lanes(net_path: Path, min_length: float = 20.0) -> List[SumoLaneInfo]:
    """
    Parse SUMO .net.xml and find lanes that lead to intersections.
    
    Args:
        net_path: Path to the .net.xml file
        min_length: Minimum lane length in meters
        
    Returns:
        List of SumoLaneInfo for lanes suitable for spawning
    """
    if not net_path.exists():
        return []
    
    tree = ET.parse(net_path)
    root = tree.getroot()
    
    # First, collect junction info
    junctions = {}
    for junction in root.findall("junction"):
        jid = junction.get("id")
        jtype = junction.get("type", "unknown")
        junctions[jid] = jtype
    
    # Intersection junction types (where yield signs matter)
    intersection_types = {"priority", "right_before_left", "allway_stop", "traffic_light"}
    
    spawn_lanes = []
    
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        func = edge.get("function", "normal")
        
        # Skip internal edges (junction connectors)
        if func == "internal" or edge_id.startswith(":"):
            continue
        
        to_junction = edge.get("to", "")
        junction_type = junctions.get(to_junction, "unknown")
        
        # Only consider edges leading to intersections
        if junction_type not in intersection_types:
            continue
        
        lanes = edge.findall("lane")
        for lane in lanes:
            lane_id = lane.get("id", "")
            
            # Parse lane length
            length = float(lane.get("length", 0))
            if length == 0:
                # Try to compute from shape
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
            
            # Extract lane number from lane_id (e.g., "-123#0_1" -> 1)
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
    """
    Select a random lane from available spawn lanes.
    
    Args:
        spawn_lanes: List of available lanes
        seed: Random seed for reproducibility
        
    Returns:
        Selected SumoLaneInfo or None if no lanes available
    """
    if not spawn_lanes:
        return None
    
    rng = random.Random(seed)
    return rng.choice(spawn_lanes)


def make_experiment_dir(
    output_base: Path,
    experiment_name: str | None = None,
) -> Path:
    """Create a dated experiment directory under benchmark_output/2_4."""
    name = experiment_name or datetime.now().strftime(EXPERIMENT_TIMESTAMP_FMT)
    experiment_dir = output_base / name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def _stable_seed(scene_name: str, variant: int = 0, scenario_id: str = "") -> int:
    """Generate deterministic 32-bit seed from scene name and variant/scenario."""
    h = hashlib.sha256()
    h.update(scene_name.encode("utf-8"))
    h.update(b"|")
    h.update(str(variant).encode("utf-8"))
    if scenario_id:
        h.update(b"|")
        h.update(scenario_id.encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")


def discover_scenes(scenes_dir: Path) -> List[Path]:
    """Find all valid scene directories containing meta.json and map.net.xml."""
    scenes = []
    for entry in sorted(scenes_dir.iterdir()):
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        net_file = meta.get("net_file", "map.net.xml")
        net_path = entry / net_file
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


def build_manifest_entry(
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict,
    variant: int = 0,
    spawn_velocity_ms: float = 5.0,
    traffic_density: float = 0.0,
    horizon: int = 600,
    sign_distance_before_end: float = 20.0,
    spawn_distance_before_end: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    auxiliary_agent: bool = False,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    spawn_lanes_cache: Optional[List[SumoLaneInfo]] = None,
    junction_layout_cache: Optional[dict] = None,
    spawn_scenario: Optional[SpawnScenario] = None,
) -> Dict:
    """Build a single manifest entry for a scene.
    
    Args:
        scene_dir: Directory containing the scene
        scenes_root: Root directory for all scenes
        meta: Scene metadata from meta.json
        variant: Variant index for seed generation
        spawn_velocity_ms: Initial vehicle velocity
        traffic_density: NPC traffic density
        horizon: Simulation steps
        sign_distance_before_end: Distance before lane end to place yield sign
        spawn_distance_before_end: Distance before lane end to spawn ego vehicle
        auxiliary_agent: Whether to spawn auxiliary agent
        aux_distance_from_intersection: Distance for auxiliary agent
        spawn_lanes_cache: Pre-parsed spawn lanes (to avoid re-parsing for each variant)
        junction_layout_cache: Pre-built junction layout dict for the scene net
    """
    scene_name = meta.get("scene_name", scene_dir.name)
    net_file = meta.get("net_file", "map.net.xml")
    
    net_path = scene_dir.relative_to(scenes_root) / net_file
    net_full_path = scene_dir / net_file
    
    scenario_id = spawn_scenario.scenario_id if spawn_scenario else ""
    seed = _stable_seed(scene_name, variant, scenario_id)

    # Parse spawn lanes if not cached
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
        "spawn_velocity_ms": spawn_velocity_ms,
        "traffic_density": traffic_density,
        "horizon": horizon,
        "sign_distance_before_end": sign_distance_before_end,
        "spawn_distance_before_end": spawn_distance_before_end,
        "valid": True,
        "latitude": meta.get("latitude"),
        "longitude": meta.get("longitude"),
        "crop_radius_m": meta.get("crop_radius_m"),
        "source_osm": meta.get("source_osm"),
        "osm_file": meta.get("osm_file"),
        # Auxiliary agent settings
        "auxiliary_agent": auxiliary_agent,
        "aux_distance_from_intersection": aux_distance_from_intersection,
    }
    
    if spawn_scenario is not None:
        entry.update(spawn_scenario.to_manifest_fields())
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
        # Use randomly selected lane from network
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
    
    entry = {k: v for k, v in entry.items() if v is not None}
    
    return entry


def generate_manifest(
    scenes_dir: Path,
    output_dir: Path,
    n_variants: int = 1,
    spawn_velocity_ms: float = 5.0,
    traffic_density: float = 0.0,
    horizon: int = 600,
    sign_distance_before_end: float = 20.0,
    spawn_distance_before_end: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    auxiliary_agent: bool = False,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    augment: bool = True,
    max_scenarios_per_scene: Optional[int] = None,
) -> List[Dict]:
    """Generate real_manifest.jsonl from discovered scenes."""
    scenes = discover_scenes(scenes_dir)
    print(f"\n[INFO] Found {len(scenes)} valid scene(s):")

    entries = []
    
    for scene_dir in scenes:
        meta = load_scene_metadata(scene_dir)
        scene_name = meta.get("scene_name", scene_dir.name)
        net_file = meta.get("net_file", "map.net.xml")
        net_full_path = scene_dir / net_file
        
        print(f"\n[SCENE] {scene_name}")
        print(f"  Location: ({meta.get('latitude')}, {meta.get('longitude')})")
        
        # Pre-parse spawn lanes for this scene (cache for all variants)
        spawn_lanes = parse_sumo_net_for_spawn_lanes(net_full_path)
        print(f"  Found {len(spawn_lanes)} intersection-approaching lane(s)")

        junction_layout = build_junction_layout_for_scene(net_full_path)
        if junction_layout is not None:
            print(
                f"  Junction layout: {junction_layout['shape']} @ {junction_layout['junction_id']} "
                f"(main={len(junction_layout.get('main_edge_ids', []))}, "
                f"secondary={len(junction_layout.get('secondary_edge_ids', []))})"
            )
            secondary_spawn = filter_spawn_lanes_to_secondary(spawn_lanes, junction_layout)
            secondary_edges = sorted({lane.edge_id for lane in secondary_spawn})
            print(
                f"  Secondary spawn pool: {len(secondary_spawn)} lane(s) "
                f"on edge(s) {secondary_edges}"
            )
        
        if spawn_lanes:
            # Show available lanes
            unique_edges = set(lane.edge_id for lane in spawn_lanes)
            print(f"  Available edges: {list(unique_edges)[:5]}{'...' if len(unique_edges) > 5 else ''}")

        scenarios: List[SpawnScenario] = []
        if augment and junction_layout is not None:
            _, scenarios, aug_stats = augment_layout_for_scene(net_full_path, spawn_lanes)
            if max_scenarios_per_scene is not None:
                scenarios = scenarios[:max_scenarios_per_scene]
            if aug_stats is not None:
                print(aug_stats.format_report(scene_name))
            print(f"  Augmented scenarios (conflict-valid): {len(scenarios)}")
            if not scenarios:
                print(f"  [augment] No valid scenarios for {scene_name}; skipping scene")
                continue

        if scenarios:
            for variant, scenario in enumerate(scenarios):
                entry = build_manifest_entry(
                    scene_dir=scene_dir,
                    scenes_root=scenes_dir,
                    meta=meta,
                    variant=variant,
                    spawn_velocity_ms=spawn_velocity_ms,
                    traffic_density=traffic_density,
                    horizon=horizon,
                    sign_distance_before_end=sign_distance_before_end,
                    spawn_distance_before_end=spawn_distance_before_end,
                    auxiliary_agent=auxiliary_agent,
                    aux_distance_from_intersection=aux_distance_from_intersection,
                    spawn_lanes_cache=spawn_lanes,
                    junction_layout_cache=junction_layout,
                    spawn_scenario=scenario,
                )
                entries.append(entry)
                if variant < 3 or variant == len(scenarios) - 1:
                    print(
                        f"  [{variant + 1}/{len(scenarios)}] {scenario.scenario_id} "
                        f"seed={entry['seed']}"
                    )
                elif variant == 3:
                    print(f"  ... ({len(scenarios) - 4} more)")
        else:
            for variant in range(n_variants):
                entry = build_manifest_entry(
                    scene_dir=scene_dir,
                    scenes_root=scenes_dir,
                    meta=meta,
                    variant=variant,
                    spawn_velocity_ms=spawn_velocity_ms,
                    traffic_density=traffic_density,
                    horizon=horizon,
                    sign_distance_before_end=sign_distance_before_end,
                    spawn_distance_before_end=spawn_distance_before_end,
                    auxiliary_agent=auxiliary_agent,
                    aux_distance_from_intersection=aux_distance_from_intersection,
                    spawn_lanes_cache=spawn_lanes,
                    junction_layout_cache=junction_layout,
                )
                entries.append(entry)

                road_id = entry.get("road_id", "N/A")
                lane_num = entry.get("spawn_lane_num", "N/A")
                lane_len = entry.get("spawn_lane_length", "N/A")
                if n_variants > 1:
                    print(f"  Variant {variant}: seed={entry['seed']}, road={road_id}, lane={lane_num}")
                else:
                    print(f"  Spawn lane: road_id={road_id}, lane_num={lane_num}, length={lane_len}m")
    
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
        "variants_per_scene": n_variants,
        "augment": augment,
        "max_scenarios_per_scene": max_scenarios_per_scene,
        "spawn_velocity_ms": spawn_velocity_ms,
        "traffic_density": traffic_density,
        "horizon": horizon,
        "sign_distance_before_end": sign_distance_before_end,
        "spawn_distance_before_end": spawn_distance_before_end,
        "auxiliary_agent": auxiliary_agent,
        "aux_distance_from_intersection": aux_distance_from_intersection,
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
    
    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  - Scenes processed: {len(scenes)}")
    print(f"  - Manifest entries: {len(entries)}")
    print(f"  - Sign distance before lane end: {sign_distance_before_end}m")
    print(f"  - Ego spawn distance before lane end: {spawn_distance_before_end}m")
    if auxiliary_agent:
        print(f"\nAuxiliary Agent: ENABLED (stationary, near intersection)")
        print(f"  - Distance from intersection: {aux_distance_from_intersection}m")
    print(f"\nOutput files:")
    print(f"  - Manifest: {manifest_path}")
    print(f"  - Experiment config: {manifest_meta_path}")
    print(f"  - Summary: {summary_path}")
    
    return entries


# =============================================================================
# GIF Rendering
# =============================================================================

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
    run_name: str,
    policy: str = "idm",
    max_scenes: Optional[int] = None,
    dry_run: bool = False,
    gif_dir: Optional[Path] = None,
    hide_signs: bool = False,
    auxiliary_agent: bool = False,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
) -> Tuple[int, int]:
    """Render GIFs for scenes from a manifest file.
    
    Args:
        manifest_path: Path to real_manifest.jsonl
        experiment_dir: Dated run directory under benchmark_output/2_4
        scenes_root: Root directory containing scene folders
        run_name: Name for the benchmark run
        gif_dir: Optional GIF subdirectory (default: <experiment_dir>/gifs)
        policy: Driving policy for rendering
        max_scenes: Limit number of scenes to render
        dry_run: Print commands without executing
        hide_signs: If True, hide traffic sign visual models in rendered GIFs
        auxiliary_agent: If True, spawn a stationary auxiliary agent near intersection
        aux_distance_from_intersection: Distance from intersection (meters)
        
    Returns:
        (rendered_count, failed_count)
    """
    if not RUN_BENCH_SCRIPT.is_file():
        print(f"[GIF] run_benchmark_real.py not found at {RUN_BENCH_SCRIPT}", file=sys.stderr)
        return 0, 1
    
    if not manifest_path.is_file():
        print(f"[GIF] Manifest not found: {manifest_path}", file=sys.stderr)
        return 0, 1
    
    if gif_dir is None:
        gif_dir = experiment_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

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
        
        if max_scenes is not None and len(rows) >= max_scenes:
            break
    
    if not rows:
        print(f"[GIF] No valid scenes found in manifest for {PDD_CODE}.")
        return 0, 0
    
    print(f"\n[GIF] Rendering {len(rows)} scene(s)...")
    
    rendered = 0
    failed = 0
    for i, row in enumerate(rows, start=1):
        # scene_uid format: scene_id:sign_type:seed (no backend prefix)
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
            "--policy", policy,
        ]
        if hide_signs:
            cmd.append("--hide-signs")
        
        # Auxiliary agent options
        if auxiliary_agent:
            cmd.append("--auxiliary-agent")
            cmd.extend(["--aux-distance-from-intersection", str(aux_distance_from_intersection)])
        
        print(f"\n[GIF {i}/{len(rows)}] {scene_uid}")
        print("  " + " ".join(cmd))
        
        if dry_run:
            rendered += 1
            continue
        
        res = subprocess.run(cmd, cwd=str(RUN_BENCH_SCRIPT.parent))
        if res.returncode == 0:
            rendered += 1
        else:
            failed += 1
            print(f"[GIF] Command failed with code {res.returncode}")
    
    return rendered, failed


def main():
    parser = argparse.ArgumentParser(
        description="Generate real_manifest.jsonl from manually collected SUMO scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenes-dir", type=str, default=str(DEFAULT_SCENES_DIR),
        help=f"Directory containing scene subdirectories (default: {DEFAULT_SCENES_DIR})"
    )
    parser.add_argument(
        "--output-base", type=str, default=str(DEFAULT_OUTPUT_BASE),
        help=f"Base directory for dated experiment folders (default: {DEFAULT_OUTPUT_BASE})"
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help="Experiment folder name under output-base "
             "(default: current timestamp YYYY-MM-DD_HH-MM-SS)"
    )
    parser.add_argument(
        "--n-variants", type=int, default=1,
        help="Random seed variants per scene when --no-augment (default: 1)"
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable spawn-point augmentation; use --n-variants random spawns instead"
    )
    parser.add_argument(
        "--max-scenarios-per-scene", type=int, default=None,
        help="Cap augmented scenarios per scene (default: all valid combinations)"
    )
    parser.add_argument(
        "--spawn-velocity", type=float, default=2.5,
        help="Spawn velocity in m/s (default: 5.0)"
    )
    parser.add_argument(
        "--traffic-density", type=float, default=0.0,
        help="Traffic density for NPCs (default: 0.0)"
    )
    parser.add_argument(
        "--horizon", type=int, default=600,
        help="Simulation horizon in steps (default: 600)"
    )
    parser.add_argument(
        "--sign-distance", type=float, default=0.0,
        help="Distance before lane end to place yield sign in meters (default: 5.0)"
    )
    parser.add_argument(
        "--spawn-distance", type=float, default=DEFAULT_SPAWN_DISTANCE_BEFORE_END,
        help=f"Distance before lane end to spawn ego vehicle in meters (default: {DEFAULT_SPAWN_DISTANCE_BEFORE_END})"
    )

    # GIF rendering options
    parser.add_argument(
        "--save-gifs", action="store_true",
        help="Render and save GIFs for scenes via run_benchmark_real.py"
    )
    parser.add_argument(
        "--gif-dir", type=str, default=None,
        help="GIF output directory (default: <experiment-dir>/gifs)"
    )
    parser.add_argument(
        "--gif-policy", type=str, default="idm",
        help="Policy for GIF rendering (default: idm)"
    )
    parser.add_argument(
        "--gif-max-scenes", type=int, default=None,
        help="Render GIFs for at most this many scenes (default: all)"
    )
    parser.add_argument(
        "--gif-dry-run", action="store_true",
        help="Print GIF rendering commands without executing"
    )
    parser.add_argument(
        "--hide-signs", action="store_true", default=True,
        help="Hide traffic sign visual models in GIFs (signs still affect behavior)"
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Run name for GIF rendering (default: real_<timestamp>)"
    )
    
    # Auxiliary agent options
    parser.add_argument(
        "--auxiliary-agent", action="store_true", default=True,
        help="Spawn a stationary auxiliary agent on the main road near intersection"
    )
    parser.add_argument(
        "--aux-distance-from-intersection", type=float,
        default=DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
        help=f"Distance from intersection to spawn aux agent (meters, "
             f"default: {DEFAULT_AUX_DISTANCE_FROM_INTERSECTION})"
    )
    
    args = parser.parse_args()

    scenes_dir = Path(args.scenes_dir)
    output_base = Path(args.output_base)
    experiment_dir = make_experiment_dir(output_base, args.experiment_name)
    print(f"\n[INFO] Experiment directory: {experiment_dir}")

    entries = generate_manifest(
        scenes_dir=scenes_dir,
        output_dir=experiment_dir,
        n_variants=args.n_variants,
        spawn_velocity_ms=args.spawn_velocity,
        traffic_density=args.traffic_density,
        horizon=args.horizon,
        sign_distance_before_end=args.sign_distance,
        spawn_distance_before_end=args.spawn_distance,
        auxiliary_agent=args.auxiliary_agent,
        aux_distance_from_intersection=args.aux_distance_from_intersection,
        augment=not args.no_augment,
        max_scenarios_per_scene=args.max_scenarios_per_scene,
    )
    
    if args.save_gifs and entries:
        manifest_path = experiment_dir / "real_manifest.jsonl"
        gif_dir = Path(args.gif_dir) if args.gif_dir else None
        run_name = args.run_name or experiment_dir.name

        print(f"\n[INFO] Rendering GIFs into experiment: {experiment_dir}")
        if args.auxiliary_agent:
            print(f"[INFO] Auxiliary agent: ENABLED (stationary, near intersection)")
            print(f"  - Distance from intersection: {args.aux_distance_from_intersection}m")
        gif_rendered, gif_failed = render_gifs_from_manifest(
            manifest_path=manifest_path,
            experiment_dir=experiment_dir,
            scenes_root=scenes_dir,
            run_name=run_name,
            policy=args.gif_policy,
            max_scenes=args.gif_max_scenes,
            dry_run=args.gif_dry_run,
            gif_dir=gif_dir,
            hide_signs=args.hide_signs,
            auxiliary_agent=args.auxiliary_agent,
            aux_distance_from_intersection=args.aux_distance_from_intersection,
        )

        resolved_gif_dir = gif_dir or (experiment_dir / "gifs")
        print(f"\n[GIF RESULTS]")
        print(f"  - GIFs rendered: {gif_rendered}")
        print(f"  - GIF failures: {gif_failed}")
        print(f"  - Experiment directory: {experiment_dir}")
        print(f"  - GIF directory: {resolved_gif_dir}")


if __name__ == "__main__":
    main()
