#!/usr/bin/env python3
"""
Real map manifest generator for sign 2.4 (Yield).

This script scans the scenes/ directory for manually collected SUMO scenes
and generates real_manifest.jsonl for policy evaluation.

Usage:
    python yield_sign/generate_real_manifest.py
    python yield_sign/generate_real_manifest.py --output-dir benchmark_output/2_4
    python yield_sign/generate_real_manifest.py --save-gifs --gif-policy idm
    python yield_sign/generate_real_manifest.py --save-gifs --gif-dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).parent
DEFAULT_SCENES_DIR = SCRIPT_DIR / "scenes"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "benchmark_output" / "2_4"
RUN_BENCH_SCRIPT = SCRIPT_DIR / "run_benchmark_real.py"

PDD_CODE = "2.4"
SIGN_TYPE = "yield"


def _stable_seed(scene_name: str, variant: int = 0) -> int:
    """Generate deterministic 32-bit seed from scene name and variant."""
    h = hashlib.sha256()
    h.update(scene_name.encode("utf-8"))
    h.update(b"|")
    h.update(str(variant).encode("utf-8"))
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
) -> Dict:
    """Build a single manifest entry for a scene."""
    scene_name = meta.get("scene_name", scene_dir.name)
    net_file = meta.get("net_file", "map.net.xml")
    
    net_path = scene_dir.relative_to(scenes_root) / net_file
    
    seed = _stable_seed(scene_name, variant)
    
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
        "valid": True,
        "latitude": meta.get("latitude"),
        "longitude": meta.get("longitude"),
        "crop_radius_m": meta.get("crop_radius_m"),
        "source_osm": meta.get("source_osm"),
        "osm_file": meta.get("osm_file"),
    }
    
    if meta.get("distance_from_start"):
        entry["distance_from_start"] = meta["distance_from_start"]
    if meta.get("sign_spawn_distance"):
        entry["sign_spawn_distance"] = meta["sign_spawn_distance"]
    if meta.get("road_id"):
        entry["road_id"] = meta["road_id"]
    if meta.get("spawn_lane_num") is not None:
        entry["spawn_lane_num"] = meta["spawn_lane_num"]
    if meta.get("destination_lane_id"):
        entry["destination_lane_id"] = meta["destination_lane_id"]
    
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
) -> List[Dict]:
    """Generate real_manifest.jsonl from discovered scenes."""
    scenes = discover_scenes(scenes_dir)
    print(f"\n[INFO] Found {len(scenes)} valid scene(s):")

    entries = []
    
    for scene_dir in scenes:
        meta = load_scene_metadata(scene_dir)
        scene_name = meta.get("scene_name", scene_dir.name)
        
        print(f"\n[SCENE] {scene_name}")
        print(f"  Location: ({meta.get('latitude')}, {meta.get('longitude')})")
        
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
            )
            entries.append(entry)
            
            if n_variants > 1:
                print(f"  Variant {variant}: seed={entry['seed']}")
    
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
        "spawn_velocity_ms": spawn_velocity_ms,
        "traffic_density": traffic_density,
        "horizon": horizon,
        "sign_distance_before_end": sign_distance_before_end,
        "generated_at": datetime.now().isoformat(),
        "scenes": [s.name for s in scenes],
    }
    
    summary_path = output_dir / "real_manifest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  - Scenes processed: {len(scenes)}")
    print(f"  - Manifest entries: {len(entries)}")
    print(f"  - Sign distance before lane end: {sign_distance_before_end}m")
    print(f"\nOutput files:")
    print(f"  - Manifest: {manifest_path}")
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
    gif_dir: Path,
    scenes_root: Path,
    run_name: str,
    policy: str = "idm",
    max_scenes: Optional[int] = None,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Render GIFs for scenes from a manifest file.
    
    Args:
        manifest_path: Path to real_manifest.jsonl
        gif_dir: Output directory for GIFs
        scenes_root: Root directory containing scene folders
        run_name: Name for the benchmark run
        policy: Driving policy for rendering
        max_scenes: Limit number of scenes to render
        dry_run: Print commands without executing
        
    Returns:
        (rendered_count, failed_count)
    """
    if not RUN_BENCH_SCRIPT.is_file():
        print(f"[GIF] run_benchmark_real.py not found at {RUN_BENCH_SCRIPT}", file=sys.stderr)
        return 0, 1
    
    if not manifest_path.is_file():
        print(f"[GIF] Manifest not found: {manifest_path}", file=sys.stderr)
        return 0, 1
    
    gif_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    seen_scene_ids = set()
    
    for row in _iter_jsonl_rows(manifest_path):
        if not row.get("valid", True):
            continue
        if row.get("pdd_code") != PDD_CODE:
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
            "--gif-dir", str(gif_dir),
            "--run-name", run_name,
            "--scenes-root", str(scenes_root),
            "--policy", policy,
        ]
        
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
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for real_manifest.jsonl (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--n-variants", type=int, default=1,
        help="Number of seed variants per scene (default: 1)"
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
        "--sign-distance", type=float, default=10.0,
        help="Distance before lane end to place yield sign in meters (default: 20.0)"
    )

    # GIF rendering options
    parser.add_argument(
        "--save-gifs", action="store_true",
        help="Render and save GIFs for scenes via run_benchmark.py"
    )
    parser.add_argument(
        "--gif-dir", type=str, default=None,
        help="GIF output directory (default: <output-dir>/gifs)"
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
        "--run-name", type=str, default=None,
        help="Run name for GIF rendering (default: real_<timestamp>)"
    )
    
    args = parser.parse_args()
    
    scenes_dir = Path(args.scenes_dir)
    output_dir = Path(args.output_dir)
    
    entries = generate_manifest(
        scenes_dir=scenes_dir,
        output_dir=output_dir,
        n_variants=args.n_variants,
        spawn_velocity_ms=args.spawn_velocity,
        traffic_density=args.traffic_density,
        horizon=args.horizon,
        sign_distance_before_end=args.sign_distance,
    )
    
    if args.save_gifs and entries:
        manifest_path = output_dir / "real_manifest.jsonl"
        gif_dir = Path(args.gif_dir) if args.gif_dir else (output_dir / "gifs")
        run_name = args.run_name or f"real_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        
        print(f"\n[INFO] Rendering GIFs -> {gif_dir}")
        gif_rendered, gif_failed = render_gifs_from_manifest(
            manifest_path=manifest_path,
            gif_dir=gif_dir,
            scenes_root=scenes_dir,
            run_name=run_name,
            policy=args.gif_policy,
            max_scenes=args.gif_max_scenes,
            dry_run=args.gif_dry_run,
        )
        
        print(f"\n[GIF RESULTS]")
        print(f"  - GIFs rendered: {gif_rendered}")
        print(f"  - GIF failures: {gif_failed}")
        print(f"  - Output directory: {gif_dir}")


if __name__ == "__main__":
    main()
