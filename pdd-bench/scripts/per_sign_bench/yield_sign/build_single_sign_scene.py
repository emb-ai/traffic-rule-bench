#!/usr/bin/env python3
"""Convert OSM map to SUMO net.xml (main roads only, no signs/lights/railways).

Usage:
    python build_single_sign_scene.py \
        --osm scenes/savvinskaya_3/map.osm \
        --name savvinskaya_3 \
        --scenes-dir ./scenes

    python build_single_sign_scene.py \
        --osm scenes/check/map.osm \
        --name check \
        --scenes-dir ./scenes
"""

import argparse
import json
import subprocess
import shutil
from pathlib import Path


def _find_netconvert() -> str:
    """Find netconvert executable."""
    candidates = [
        shutil.which("netconvert"),
        "/home/jovyan/.local/bin/netconvert",
        "/usr/local/bin/netconvert",
        "/usr/bin/netconvert",
    ]
    
    for path in candidates:
        if path and Path(path).exists():
            return path
    
    raise FileNotFoundError(
        "netconvert not found. Install SUMO or add netconvert to PATH.\n"
        "Try: pip install sumo-tools  OR  export PATH=$PATH:/home/jovyan/.local/bin"
    )


def convert_osm_to_sumo(osm_path, scene_name, scenes_dir):
    """Convert OSM file to SUMO net.xml (roads and lanes only, no signs/lights)."""
    osm_path = Path(osm_path)
    if not osm_path.exists():
        raise FileNotFoundError(f"OSM file not found: {osm_path}")
    
    print(f"\n{'=' * 60}")
    print(f"Converting OSM to SUMO network")
    print(f"  Input: {osm_path.name}")
    print(f"  Scene name: {scene_name}")
    print(f"{'=' * 60}")

    scene_dir = scenes_dir / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    net_file = f"{scene_name}.net.xml"
    osm_file = f"{scene_name}.osm"
    net_output_path = scene_dir / net_file

    print(f"\n1. Converting OSM to SUMO .net.xml (main roads only)...")
    print(f"   Input: {osm_path} ({osm_path.stat().st_size / 1024:.1f} KB)")
    
    # Road types to keep (main roads only, no service/footways/railways)
    keep_types = [
        "highway.motorway", "highway.motorway_link",
        "highway.trunk", "highway.trunk_link",
        "highway.primary", "highway.primary_link",
        "highway.secondary", "highway.secondary_link",
        "highway.tertiary", "highway.tertiary_link",
        "highway.unclassified",
        "highway.residential",
    ]
    
    # Types to explicitly remove (railways, subways, trams, etc.)
    remove_types = [
        "railway.subway", "railway.rail", "railway.light_rail",
        "railway.tram", "railway.monorail", "railway.preserved",
        "highway.service", "highway.footway", "highway.path",
        "highway.cycleway", "highway.pedestrian", "highway.steps",
        "highway.track", "highway.bridleway", "highway.corridor",
    ]
    
    subprocess.run([
        _find_netconvert(),
        "--osm-files", str(osm_path),
        "-o", str(net_output_path),
        "--tls.discard-loaded",
        "--tls.discard-simple",
        "--no-turnarounds",
        "--geometry.remove",
        "--junctions.join",
        "--keep-edges.by-type", ",".join(keep_types),
        "--remove-edges.by-type", ",".join(remove_types),
        "--no-internal-links",
    ], check=True)

    print(f"   Output: {net_output_path} ({net_output_path.stat().st_size / 1024:.1f} KB)")

    print("\n2. Saving scene files...")
    
    # Copy original OSM to scene dir
    (scene_dir / osm_file).write_bytes(osm_path.read_bytes())

    meta = {
        "scene_name": scene_name,
        "net_file": net_file,
        "osm_file": osm_file,
        "source_osm": str(osm_path),
    }
    with open(scene_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nScene saved to: {scene_dir}")
    print(f"  meta.json: {scene_dir / 'meta.json'}")
    print(f"  net.xml: {scene_dir / net_file}")
    print(f"  osm: {scene_dir / osm_file}")

    return scene_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert OSM map to SUMO net.xml (main roads only, no signs/lights/railways)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--osm", required=True, help="Path to OSM file")
    parser.add_argument("--name", required=True, help="Scene name")
    parser.add_argument("--scenes-dir", default="scenes", help="Output directory (default: scenes)")
    
    args = parser.parse_args()
    scenes_dir = Path(args.scenes_dir)
    
    scene_dir = convert_osm_to_sumo(
        osm_path=args.osm,
        scene_name=args.name,
        scenes_dir=scenes_dir
    )
    
    print(f"\n{'=' * 60}")
    print("SUCCESS!")
    print(f"Scene created at: {scene_dir}")
    print(f"\nTo run this scene:")
    print(f"  python run_single_sumo_scene.py \\")
    print(f"    --scene-dir {scene_dir} \\")
    print(f"    --traffic-density 0.0 \\")
    print(f"    --out {scene_dir / 'output.gif'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
