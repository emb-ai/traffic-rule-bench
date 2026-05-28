#!/usr/bin/env python3
"""
Fixed scene generator for sign 2.4 (Yield).

This script generates deterministic scenes with:
- Zero traffic density (no NPC vehicles)
- Controlled yield sign placement with detailed logging
- Output to benchmark_output/fixed/2_4 directory

Self-contained: no external space_definition.py dependency.

Usage:
    python yield_generate_fixed_scenes.py --n-scenes 10
    python yield_generate_fixed_scenes.py --dry-run
    python yield_generate_fixed_scenes.py --n-scenes 5 --save-gifs
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_DIR = SCRIPT_PATH.parent
SCRIPTS_DIR = BENCHMARK_DIR.parent
PDD_BENCH_DIR = SCRIPTS_DIR.parent
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"

DEFAULT_OUTPUT_BASE_DIR = (
    SDC_ROOT / "pdd-bench/scripts/per_sign_bench/benchmark_output/fixed/2_4"
)
RUN_BENCH_SCRIPT = BENCHMARK_DIR / "yield_run_benchmark_mini.py"

for _p in (PDD_BENCH_DIR, METADRIVE_DIR, BENCHMARK_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)


# =============================================================================
# Road Topology Definitions (Yield Sign Compatible)
# =============================================================================

@dataclass(frozen=True)
class TopologyTemplate:
    """Definition of a road topology template for scene generation."""
    block_id: str        # T or X (intersection block types)
    route_intent: str    # exit_0, exit_1, right, straight, left
    map_string: str      # Block sequence string for MetaDrive (e.g., "XS", "TS")
    socket_index: int    # Which socket the route_intent maps to (0-based)
    block_name: str      # Human-readable name


# Yield sign (2.4) requires intersection blocks: T-intersection or X-intersection.
YIELD_TOPOLOGY_TEMPLATES: List[TopologyTemplate] = [
    # T-intersection (2 exits)
    TopologyTemplate("T", "exit_0", "ST", 0, "T-Intersection exit 0"),
    TopologyTemplate("T", "exit_1", "ST", 1, "T-Intersection exit 1"),
    # X-intersection (3 exits: right, straight, left)
    TopologyTemplate("X", "right",    "SX", 0, "X-Intersection right"),
    TopologyTemplate("X", "straight", "SX", 1, "X-Intersection straight"),
    TopologyTemplate("X", "left",     "SX", 2, "X-Intersection left"),
]

INTERSECTION_BLOCK_IDS = {"T", "X"}


# =============================================================================
# Road Geometry Configuration
# =============================================================================

LANE_NUM_GRID = [2, 3, 4, 5]
LANE_WIDTH_GRID = [3.7]  # MetaDrive default lane width (meters)

SPAWN_LANE_SEMANTIC = ["left", "center", "right"]


def resolve_spawn_lane(semantic: str, lane_num: int) -> int:
    """Convert semantic spawn lane position to actual lane index.
    
    Args:
        semantic: "left", "center", or "right"
        lane_num: Total number of lanes
        
    Returns:
        Lane index (0-based, 0 = leftmost lane)
    """
    if semantic == "left":
        return 0
    elif semantic == "center":
        return lane_num // 2
    elif semantic == "right":
        return lane_num - 1
    raise ValueError(f"Unknown spawn lane semantic: {semantic}")


def effective_spawn_lanes(lane_num: int) -> List[str]:
    """Return deduplicated semantic spawn lanes for given lane count.
    
    For 2 lanes: left and right map to different indices, center = left.
    For 3+ lanes: all three are distinct.
    """
    seen = {}
    result = []
    for s in SPAWN_LANE_SEMANTIC:
        idx = resolve_spawn_lane(s, lane_num)
        if idx not in seen:
            seen[idx] = s
            result.append(s)
    return result


# =============================================================================
# Scene Configuration
# =============================================================================

@dataclass
class YieldSceneConfig:
    """Configuration for a single yield sign scene."""
    scene_id: str
    seed: int
    
    # Topology
    block_id: str           # T or X (intersection blocks)
    route_intent: str       # exit_0, exit_1, right, straight, left
    block_sequence: str     # e.g., "XS", "TS"
    socket_index: int
    block_name: str
    
    # Road geometry
    lane_num: int
    lane_width: float
    spawn_lane_semantic: str
    spawn_lane_index: int
    
    # Scene parameters
    spawn_velocity_ms: float = 0.0
    traffic_density: float = 0.0  # Zero traffic for fixed scenes
    horizon_steps: int = 600
    
    # Sign placement control (None = auto-placement by sign manager)
    sign_longitudinal_offset: Optional[float] = None  # Meters from lane end (negative = before end)
    sign_lateral_offset: Optional[float] = None       # Meters from lane center (positive = right)
    
    # Sign placement results (filled after generation)
    sign_placement_long: float = None
    sign_placement_lat: float = None
    sign_lane_index: tuple = None


# =============================================================================
# Seed & Config Generation
# =============================================================================

def _stable_seed(*parts) -> int:
    """Generate deterministic 32-bit seed from arbitrary parts using SHA-256."""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest()[:4], "big")


def build_yield_scene_configs(
    n_scenes: int,
    base_seed: int = 42,
    spawn_velocity_ms: float = 5.0,
    sign_longitudinal_offset: Optional[float] = None,
    sign_lateral_offset: Optional[float] = None,
    block_type_filter: Optional[str] = None,
    lane_num_filter: Optional[int] = None,
) -> List[YieldSceneConfig]:
    """Build configurations for yield sign scenes.
    
    Cycles through topology templates and lane configurations to generate
    diverse scene setups. All scenes have zero traffic density.
    
    Args:
        n_scenes: Number of scenes to generate
        base_seed: Base random seed for reproducibility
        spawn_velocity_ms: Initial vehicle velocity (m/s)
        sign_longitudinal_offset: Custom sign placement (meters from lane end)
        sign_lateral_offset: Custom sign lateral offset (meters from lane center)
        block_type_filter: Filter to specific block type ("T" or "X")
        lane_num_filter: Fixed number of lanes (2, 3, 4, or 5)
        
    Returns:
        List of YieldSceneConfig objects
    """
    # Filter templates by block type if specified
    templates = YIELD_TOPOLOGY_TEMPLATES
    if block_type_filter:
        templates = [t for t in templates if t.block_id == block_type_filter]
    
    if not templates:
        raise ValueError(f"No templates available for block_type={block_type_filter}")
    
    print(f"\n[INFO] Available intersection templates for yield sign:")
    for t in templates:
        print(f"  - {t.block_id} / {t.route_intent} / {t.block_name} (map: {t.map_string})")
    
    # Lane configurations
    lane_nums = [lane_num_filter] if lane_num_filter else LANE_NUM_GRID
    
    configs = []
    scene_idx = 0
    
    while len(configs) < n_scenes:
        # Cycle through templates
        tmpl = templates[scene_idx % len(templates)]
        
        # Cycle through lane configurations
        lane_num = lane_nums[scene_idx % len(lane_nums)]
        lane_width = LANE_WIDTH_GRID[0]
        
        # Get valid spawn lanes for this lane_num
        spawn_lanes = effective_spawn_lanes(lane_num)
        spawn_lane_semantic = spawn_lanes[scene_idx % len(spawn_lanes)]
        spawn_lane_index = resolve_spawn_lane(spawn_lane_semantic, lane_num)
        
        # Generate deterministic seed
        seed = _stable_seed(base_seed, "yield", scene_idx)
        
        config = YieldSceneConfig(
            scene_id=f"fixed_yield_{scene_idx:04d}",
            seed=seed,
            block_id=tmpl.block_id,
            route_intent=tmpl.route_intent,
            block_sequence=tmpl.map_string,
            socket_index=tmpl.socket_index,
            block_name=tmpl.block_name,
            lane_num=lane_num,
            lane_width=lane_width,
            spawn_lane_semantic=spawn_lane_semantic,
            spawn_lane_index=spawn_lane_index,
            spawn_velocity_ms=spawn_velocity_ms,
            traffic_density=0.0,
            horizon_steps=600,
            sign_longitudinal_offset=sign_longitudinal_offset,
            sign_lateral_offset=sign_lateral_offset,
        )
        
        configs.append(config)
        scene_idx += 1
    
    return configs


# =============================================================================
# Yield Sign Placement
# =============================================================================

def spawn_yield_sign(
    sign_mgr,
    lane,
    longitudinal_offset: Optional[float] = None,
    lateral_offset: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[object, Optional[str]]:
    """Spawn a yield sign on the specified lane.
    
    This is the central function for yield sign placement. All placement
    parameters are controlled here.
    
    Args:
        sign_mgr: TrafficSignManager instance
        lane: Lane object where sign should be placed
        longitudinal_offset: Meters from lane end (negative = before end).
                            None = use sign manager's default placement.
        lateral_offset: Meters from lane center (positive = right side).
                       None = use sign manager's default placement.
        verbose: Print placement details
        
    Returns:
        (sign_object, error_message) - error_message is None on success
    """
    from traffic_signs.priority_signs import YieldSign
    
    placement_kwargs = {"use_random_lane": False}
    
    if longitudinal_offset is not None:
        placement_kwargs["longitudinal_offset"] = longitudinal_offset
        if verbose:
            print(f"  [SIGN PLACEMENT] longitudinal_offset: {longitudinal_offset}m (from lane end)")
    
    if lateral_offset is not None:
        placement_kwargs["lateral_offset"] = lateral_offset
        if verbose:
            print(f"  [SIGN PLACEMENT] lateral_offset: {lateral_offset}m (from lane center)")
    
    try:
        sign = sign_mgr.add_sign(YieldSign, lane=lane, **placement_kwargs)
        return sign, None
    except Exception as exc:
        return None, f"sign_placement_error: {exc}"


def _record_sign_placement(sign, lane, result: dict, verbose: bool = True) -> None:
    """Record sign placement details into the result dictionary."""
    if sign is None:
        return
    
    sign_long = getattr(sign, "placement_long", None)
    sign_lat = getattr(sign, "_lateral_offset", None)
    lane_idx = getattr(lane, "index", None)
    lane_length = getattr(lane, "length", None)
    
    result["sign_placement"] = {
        "lane_index": str(lane_idx),
        "lane_length_m": float(lane_length) if lane_length else None,
        "longitudinal_offset_m": float(sign_long) if sign_long else None,
        "lateral_offset_m": float(sign_lat) if sign_lat else None,
        "zone_start_m": float(getattr(sign, "zone_start", 0)),
        "zone_end_m": float(getattr(sign, "zone_end", 0)),
    }
    
    # Get sign world position
    try:
        sign_pos = lane.position(sign_long, sign_lat)
        result["sign_placement"]["world_position"] = {
            "x": float(sign_pos[0]),
            "y": float(sign_pos[1]),
        }
        heading = lane.heading_theta_at(sign_long)
        result["sign_placement"]["heading_rad"] = float(heading)
    except Exception:
        pass
    
    if verbose:
        print(f"  [SIGN RESULT] Lane: {lane_idx}, Length: {lane_length:.2f}m" if lane_length else "  [SIGN RESULT] Lane: {lane_idx}")
        if sign_long:
            print(f"  [SIGN RESULT] Longitudinal: {sign_long:.2f}m")
        if sign_lat:
            print(f"  [SIGN RESULT] Lateral: {sign_lat:.2f}m")
        print(f"  [SIGN RESULT] Zone: [{getattr(sign, 'zone_start', 0):.2f}, {getattr(sign, 'zone_end', 0):.2f}]m")


# =============================================================================
# Environment & Route Utilities
# =============================================================================

def _get_route_lanes(env) -> List:
    """Get all lanes along the vehicle's navigation route."""
    vehicle = env.vehicle
    nav = getattr(vehicle, "navigation", None)
    if nav is None:
        return []
    
    checkpoints = getattr(nav, "checkpoints", None)
    if not checkpoints or len(checkpoints) < 2:
        return []
    
    road_network = env.current_map.road_network
    route_lanes = []
    
    for ckpt_start, ckpt_end in zip(checkpoints[:-1], checkpoints[1:]):
        try:
            lanes = road_network.graph[ckpt_start][ckpt_end]
            route_lanes.extend(lanes)
        except KeyError:
            continue
    
    return route_lanes


def _lane_has_continuation(lane, road_network) -> bool:
    """Check if lane has continuation (not a dead-end)."""
    idx = getattr(lane, "index", None)
    if idx is None or not (isinstance(idx, tuple) and len(idx) >= 2):
        return True
    
    graph = getattr(road_network, "graph", None)
    if graph is None:
        return True
    
    end_node = idx[1]
    outgoing = graph.get(end_node)
    
    if outgoing is None or (isinstance(outgoing, dict) and len(outgoing) == 0):
        return False
    return True


def _pick_route_lane(route_lanes: List, min_length: float = 10.0, road_network=None):
    """Pick a suitable lane from route lanes for sign placement."""
    # the function is only used for 2.4!
    for lane in route_lanes:
        if "S" in lane.index[1]:
            return lane
    return None


def _override_route_intent(env, config: YieldSceneConfig) -> bool:
    """Override navigation destination to match the route_intent."""
    route_intent = config.route_intent
    
    if route_intent in ("through", "merge"):
        return True
    
    socket_index = config.socket_index
    try:
        blocks = env.current_map.blocks
        target_block = blocks[-1]
        sockets = target_block.get_socket_list()
        
        if socket_index >= len(sockets):
            return False
        
        target_socket = sockets[socket_index]
        target_node = target_socket.positive_road.end_node
        
        vehicle = env.vehicle
        nav = vehicle.navigation
        current_lane_index = vehicle.lane.index
        nav.set_route(current_lane_index, target_node)
        return True
    except Exception:
        return False


# =============================================================================
# Scene Generation
# =============================================================================

def generate_yield_scene(
    config: YieldSceneConfig,
    render: bool = False,
    verbose: bool = True,
) -> Tuple[bool, Dict, Optional[str]]:
    """Generate a single yield scene and return placement information.
    
    Args:
        config: Scene configuration
        render: Enable visual rendering
        verbose: Print detailed information
        
    Returns:
        (success, result_dict, error_message)
    """
    from envs.traffic_sign_env import TrafficSignEnv
    from metadrive.component.pgblock.first_block import FirstPGBlock
    
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    result = {
        "scene_id": config.scene_id,
        "seed": config.seed,
        "block_id": config.block_id,
        "route_intent": config.route_intent,
        "block_sequence": config.block_sequence,
        "socket_index": config.socket_index,
        "block_name": config.block_name,
        "lane_num": config.lane_num,
        "lane_width": config.lane_width,
        "spawn_lane_semantic": config.spawn_lane_semantic,
        "spawn_lane_index": config.spawn_lane_index,
        "spawn_velocity_ms": config.spawn_velocity_ms,
        "traffic_density": config.traffic_density,
        "pdd_code": "2.4",
        "sign_type": "yield",
    }
    
    # Build spawn lane tuple
    spawn_lane_index_tuple = (
        FirstPGBlock.NODE_1,
        FirstPGBlock.NODE_2,
        config.spawn_lane_index,
    )
    
    vehicle_config = {"show_lidar": False}
    if config.spawn_velocity_ms > 0:
        vehicle_config["spawn_velocity"] = [config.spawn_velocity_ms, 0]
        vehicle_config["spawn_velocity_car_frame"] = True
    
    env_config = dict(
        start_seed=config.seed,
        use_render=render,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.WARNING if verbose else logging.CRITICAL,
        traffic_density=config.traffic_density,
        horizon=config.horizon_steps,
        vehicle_config=vehicle_config,
        map_config={
            "type": "block_sequence",
            "config": config.block_sequence,
            "lane_num": config.lane_num,
            "lane_width": config.lane_width,
        },
        random_spawn_lane_index=False,
        agent_configs={
            "default_agent": dict(
                use_special_color=True,
                spawn_lane_index=spawn_lane_index_tuple,
            ),
        },
    )
    
    env = None
    try:
        env = TrafficSignEnv(env_config)
        obs, info = env.reset(seed=config.seed)
    except Exception as exc:
        if env:
            try:
                env.close()
            except Exception:
                pass
        return False, result, f"env_creation: {exc}"
    
    try:
        # Override route intent for multi-exit blocks
        if not _override_route_intent(env, config):
            result["valid"] = False
            return False, result, "route_intent_override_failed"
        
        # Get route lanes for sign placement
        route_lanes = _get_route_lanes(env)
        if not route_lanes:
            return False, result, "no_route_lanes"
        
        # Pick a suitable lane for the yield sign
        road_network = env.current_map.road_network
        lane = _pick_route_lane(route_lanes, min_length=15.0, road_network=road_network)
        if lane is None:
            return False, result, "no_suitable_lane"
        
        # Place the yield sign
        sign_mgr = env.engine.traffic_sign_manager
        sign, sign_error = spawn_yield_sign(
            sign_mgr=sign_mgr,
            lane=lane,
            longitudinal_offset=config.sign_longitudinal_offset,
            lateral_offset=config.sign_lateral_offset,
            verbose=verbose,
        )
        
        if sign_error is not None:
            return False, result, sign_error
        
        # Record sign placement details
        _record_sign_placement(sign, lane, result, verbose=verbose)
        
        # Quick validation: run a few steps
        try:
            for _ in range(50):
                obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
                if terminated or truncated:
                    break
        except Exception as exc:
            return False, result, f"validation_step: {exc}"
        
        result["valid"] = True
        return True, result, None
        
    except Exception as exc:
        return False, result, f"unexpected: {exc}"
    finally:
        if env:
            try:
                env.close()
            except Exception:
                pass


# =============================================================================
# Output Directory Management
# =============================================================================

def _make_run_output_dir(base_output_dir: Path, run_name: Optional[str] = None) -> Path:
    """Create a new per-run output directory.
    
    If run_name is not provided, generates folder name from current datetime.
    Format: YYYY-MM-DD_HH-MM-SS
    """
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    if run_name:
        run_dir = base_output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    # Human-readable timestamp
    candidate = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_output_dir / candidate
    
    suffix = 1
    while run_dir.exists():
        run_dir = base_output_dir / f"{candidate}_{suffix:02d}"
        suffix += 1
    
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


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
    gif_dir: Path,
    run_name: str,
    backend: str = "pgmap",
    policy: str = "idm",
    max_scenes: Optional[int] = None,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Render GIFs for scenes from a manifest file.
    
    Args:
        manifest_path: Path to pgmap_materialized.jsonl
        gif_dir: Output directory for GIFs
        run_name: Name for the benchmark run
        backend: Simulation backend
        policy: Driving policy for rendering
        max_scenes: Limit number of scenes to render
        dry_run: Print commands without executing
        
    Returns:
        (rendered_count, failed_count)
    """
    if not RUN_BENCH_SCRIPT.is_file():
        print(f"[GIF] yield_run_benchmark_mini.py not found at {RUN_BENCH_SCRIPT}", file=sys.stderr)
        return 0, 1
    
    if not manifest_path.is_file():
        print(f"[GIF] Manifest not found: {manifest_path}", file=sys.stderr)
        return 0, 1
    
    gif_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect valid scenes
    rows = []
    seen_scene_ids = set()
    
    for row in _iter_jsonl_rows(manifest_path):
        if not row.get("valid", True):
            continue
        if row.get("pdd_code") != "2.4":
            continue
        
        scene_id = row.get("scene_id")
        seed = row.get("seed")
        if scene_id is None or seed is None:
            continue
        if scene_id in seen_scene_ids:
            continue
        
        seen_scene_ids.add(scene_id)
        rows.append(row)
        
        if max_scenes is not None and len(rows) >= max_scenes:
            break
    
    if not rows:
        print("[GIF] No valid scenes found in manifest for 2.4.")
        return 0, 0
    
    rendered = 0
    failed = 0
    
    for i, row in enumerate(rows, start=1):
        scene_uid = f"{backend}:{row['scene_id']}:{row['pdd_code']}:{row['seed']}"
        
        cmd = [
            sys.executable,
            str(RUN_BENCH_SCRIPT),
            "--scene-uid", scene_uid,
            "--manifest", str(manifest_path),
            "--save-gifs",
            "--gif-dir", str(gif_dir),
            "--run-name", run_name,
            "--backends", backend,
            "--policy", policy,
        ]
        
        print(f"[GIF {i}/{len(rows)}] {scene_uid}")
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


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate fixed yield sign (2.4) scenes with zero traffic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10 scenes with default settings
  python generate_fixed_yield_scenes.py --n-scenes 10
  
  # Dry run to see configurations
  python generate_fixed_yield_scenes.py --dry-run
  
  # Custom sign placement (10m before lane end, 2.5m to the right)
  python generate_fixed_yield_scenes.py --sign-long-offset -10.0 --sign-lat-offset 2.5
  
  # Generate scenes and render GIFs
  python generate_fixed_yield_scenes.py --n-scenes 5 --save-gifs
  
  # Generate single scene with specific block type
  python generate_fixed_yield_scenes.py --n-scenes 1 --block-type X

Sign placement coordinates:
  - longitudinal_offset: relative to lane END (negative = before end)
    Example: -10.0 places sign 10m before the lane ends
  - lateral_offset: distance from lane center (positive = right side)
    Example: 2.5 places sign 2.5m to the right of lane center
"""
    )
    
    # Scene generation options
    parser.add_argument(
        "--n-scenes", type=int, default=10,
        help="Number of scenes to generate (default: 10)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_BASE_DIR),
        help="Base output directory (default: .../benchmark_output/fixed/2_4)"
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Optional run folder name. Default: current datetime (YYYY-MM-DD_HH-MM-SS)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--spawn-velocity", type=float, default=5.0,
        help="Spawn velocity in m/s (default: 5.0)"
    )
    
    # Sign placement options
    parser.add_argument(
        "--sign-long-offset", type=float, default=None,
        help="Sign longitudinal offset from lane end in meters (e.g., -10.0)"
    )
    parser.add_argument(
        "--sign-lat-offset", type=float, default=None,
        help="Sign lateral offset from lane center in meters (e.g., 2.5)"
    )
    
    # Topology filters
    parser.add_argument(
        "--block-type", type=str, default=None, choices=["T", "X"],
        help="Filter to specific block type (T-intersection or X-intersection)"
    )
    parser.add_argument(
        "--lane-num", type=int, default=None, choices=[2, 3, 4, 5],
        help="Fixed number of lanes (default: varies)"
    )
    
    # Execution options
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print configurations without generating scenes"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable rendering (requires display)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print detailed placement information"
    )
    
    # GIF rendering options
    parser.add_argument(
        "--save-gifs", action="store_true",
        help="Render and save GIFs for generated scenes via yield_run_benchmark_mini.py"
    )
    parser.add_argument(
        "--gif-dir", type=str, default=None,
        help="GIF output directory (default: <run_dir>/gifs)"
    )
    parser.add_argument(
        "--gif-policy", type=str, default="idm",
        help="Policy for GIF rendering (default: idm)"
    )
    parser.add_argument(
        "--gif-backend", type=str, default="pgmap",
        choices=["pgmap", "sumo", "paired", "citymap"],
        help="Backend for GIF rendering (default: pgmap)"
    )
    parser.add_argument(
        "--gif-max-scenes", type=int, default=None,
        help="Render GIFs for at most this many unique scene_id entries (default: all)"
    )
    parser.add_argument(
        "--gif-dry-run", action="store_true",
        help="Print GIF rendering commands without executing"
    )
    
    args = parser.parse_args()
    
    # Print configuration header
    print("=" * 60)
    print("Fixed Yield Sign (2.4) Scene Generator")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Scenes to generate: {args.n_scenes}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Base seed: {args.seed}")
    print(f"  - Spawn velocity: {args.spawn_velocity} m/s")
    print(f"  - Traffic density: 0.0 (DISABLED)")
    
    if args.sign_long_offset is not None:
        print(f"  - Sign longitudinal offset: {args.sign_long_offset}m (from lane end)")
    if args.sign_lat_offset is not None:
        print(f"  - Sign lateral offset: {args.sign_lat_offset}m")
    if args.block_type:
        print(f"  - Block type filter: {args.block_type}")
    if args.lane_num:
        print(f"  - Lane number filter: {args.lane_num}")
    if args.save_gifs:
        print(f"  - GIF rendering: enabled (policy={args.gif_policy}, backend={args.gif_backend})")
        if args.gif_max_scenes is not None:
            print(f"  - GIF max scenes: {args.gif_max_scenes}")
    
    # Build scene configurations
    configs = build_yield_scene_configs(
        n_scenes=args.n_scenes,
        base_seed=args.seed,
        spawn_velocity_ms=args.spawn_velocity,
        sign_longitudinal_offset=args.sign_long_offset,
        sign_lateral_offset=args.sign_lat_offset,
        block_type_filter=args.block_type,
        lane_num_filter=args.lane_num,
    )
    
    print(f"\n[INFO] Built {len(configs)} scene configurations")
    
    # Dry run: print configurations and exit
    if args.dry_run:
        print("\n[DRY RUN] Scene configurations:")
        for i, cfg in enumerate(configs):
            print(f"\n  Scene {i}:")
            print(f"    ID: {cfg.scene_id}")
            print(f"    Block: {cfg.block_id} ({cfg.block_name})")
            print(f"    Map string: {cfg.block_sequence}")
            print(f"    Route intent: {cfg.route_intent}")
            print(f"    Lanes: {cfg.lane_num}, spawn lane: {cfg.spawn_lane_semantic} (idx={cfg.spawn_lane_index})")
            print(f"    Traffic density: {cfg.traffic_density}")
        print("\n[DRY RUN] No scenes generated. Remove --dry-run to generate.")
        return
    
    # Create per-run output directory
    output_base_dir = Path(args.output_dir)
    run_dir = _make_run_output_dir(output_base_dir, args.run_name)
    print(f"\n[INFO] Run directory: {run_dir}")
    
    # Generate scenes
    results = []
    valid_count = 0
    invalid_count = 0
    
    print(f"\n[INFO] Generating {len(configs)} scenes...")
    t0 = time.time()
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Generating {config.scene_id}...")
        print(f"  Block: {config.block_id} / {config.route_intent} (map: {config.block_sequence})")
        print(f"  Lanes: {config.lane_num}, spawn: {config.spawn_lane_semantic}")
        
        success, result, error = generate_yield_scene(
            config,
            render=args.render,
            verbose=args.verbose,
        )
        
        if success:
            valid_count += 1
            print(f"  [OK] Scene generated successfully")
        else:
            invalid_count += 1
            result["error"] = error
            print(f"  [FAIL] {error}")
        
        results.append(result)
    
    elapsed = time.time() - t0
    
    # Write manifest
    manifest_path = run_dir / "pgmap_materialized.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    
    # Write summary
    summary = {
        "pdd_code": "2.4",
        "sign_type": "yield",
        "sign_name": "Yield",
        "total_scenes": len(results),
        "valid_scenes": valid_count,
        "invalid_scenes": invalid_count,
        "traffic_density": 0.0,
        "spawn_velocity_ms": args.spawn_velocity,
        "base_seed": args.seed,
        "generation_time_s": round(elapsed, 2),
    }
    
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Render GIFs if requested
    gif_rendered = 0
    gif_failed = 0
    gif_dir = None
    
    if args.save_gifs:
        gif_dir = Path(args.gif_dir) if args.gif_dir else (run_dir / "gifs")
        print(f"\n[INFO] Rendering GIFs from manifest -> {gif_dir}")
        gif_rendered, gif_failed = render_gifs_from_manifest(
            manifest_path=manifest_path,
            gif_dir=gif_dir,
            run_name=run_dir.name,
            backend=args.gif_backend,
            policy=args.gif_policy,
            max_scenes=args.gif_max_scenes,
            dry_run=args.gif_dry_run,
        )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  - Valid scenes: {valid_count}")
    print(f"  - Invalid scenes: {invalid_count}")
    print(f"  - Time: {elapsed:.1f}s")
    print(f"\nOutput files:")
    print(f"  - Manifest: {manifest_path}")
    print(f"  - Summary: {summary_path}")
    
    if args.save_gifs:
        print(f"  - GIF directory: {gif_dir}")
        print(f"  - GIFs rendered: {gif_rendered}")
        print(f"  - GIF render failures: {gif_failed}")


if __name__ == "__main__":
    main()
