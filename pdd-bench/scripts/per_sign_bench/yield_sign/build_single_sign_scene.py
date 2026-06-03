#!/usr/bin/env python3
"""Convert OSM map to SUMO net.xml (main roads only, no signs/lights/railways).

Usage:
    # Full map (no cropping)
    python build_single_sign_scene.py \
        --osm scenes/check/map.osm \
        --name check \
        --scenes-dir ./scenes

    # Cropped to area around coordinates (~220m in each direction)
    python build_single_sign_scene.py \
        --osm scenes/check/map.osm \
        --name check \
        --lat 55.73937484 \
        --lon 37.57246089 \
        --delta 0.002 \
        --scenes-dir ./scenes
"""

import argparse
import json
import subprocess
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

DELTA_DEFAULT = 0.002  # ~200m in each direction


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


def crop_osm_to_bbox(input_osm_path, output_osm_path, lat, lon, delta):
    """Crop an OSM file to a bounding box around the given coordinates.
    
    Only keeps nodes within bbox and ways that have at least 2 nodes in bbox.
    Roads extending outside the bbox are truncated at the boundary.
    """
    tree = ET.parse(input_osm_path)
    root = tree.getroot()
    
    min_lat, max_lat = lat - delta, lat + delta
    min_lon, max_lon = lon - delta, lon + delta
    
    # Find all nodes within bbox
    nodes_in_bbox = {}
    for node in root.findall("node"):
        node_id = node.get("id")
        node_lat = float(node.get("lat"))
        node_lon = float(node.get("lon"))
        if min_lat <= node_lat <= max_lat and min_lon <= node_lon <= max_lon:
            nodes_in_bbox[node_id] = node
    
    print(f"   Found {len(nodes_in_bbox)} nodes within bbox")
    
    # Process ways - keep only nodes within bbox
    truncated_ways = []
    for way in root.findall("way"):
        node_refs = [nd.get("ref") for nd in way.findall("nd")]
        filtered_refs = [ref for ref in node_refs if ref in nodes_in_bbox]
        
        if len(filtered_refs) >= 2:
            new_way = ET.Element("way")
            new_way.set("id", way.get("id"))
            for attr in ["visible", "version", "changeset", "timestamp", "user", "uid"]:
                if way.get(attr):
                    new_way.set(attr, way.get(attr))
            
            for ref in filtered_refs:
                nd = ET.SubElement(new_way, "nd")
                nd.set("ref", ref)
            
            for tag in way.findall("tag"):
                new_tag = ET.SubElement(new_way, "tag")
                new_tag.set("k", tag.get("k"))
                new_tag.set("v", tag.get("v"))
            
            truncated_ways.append(new_way)
    
    print(f"   Keeping {len(truncated_ways)} ways (truncated to bbox)")
    
    # Build new OSM tree
    new_root = ET.Element("osm")
    new_root.set("version", root.get("version", "0.6"))
    new_root.set("generator", "crop_osm_to_bbox")
    
    new_bounds = ET.SubElement(new_root, "bounds")
    new_bounds.set("minlat", str(min_lat))
    new_bounds.set("minlon", str(min_lon))
    new_bounds.set("maxlat", str(max_lat))
    new_bounds.set("maxlon", str(max_lon))
    
    # Add only nodes referenced by kept ways
    used_nodes = set()
    for way in truncated_ways:
        for nd in way.findall("nd"):
            used_nodes.add(nd.get("ref"))
    
    for node_id in used_nodes:
        if node_id in nodes_in_bbox:
            new_root.append(nodes_in_bbox[node_id])
    
    for way in truncated_ways:
        new_root.append(way)
    
    new_tree = ET.ElementTree(new_root)
    ET.indent(new_tree, space="  ")
    new_tree.write(output_osm_path, encoding="unicode", xml_declaration=True)
    
    print(f"   Final: {len(used_nodes)} nodes, {len(truncated_ways)} ways")
    return output_osm_path


def convert_osm_to_sumo(osm_path, scene_name, scenes_dir, lat=None, lon=None, delta=None):
    """Convert OSM file to SUMO net.xml (roads and lanes only, no signs/lights).
    
    If lat/lon are provided, crops the map to a small area around that point.
    """
    osm_path = Path(osm_path)
    if not osm_path.exists():
        raise FileNotFoundError(f"OSM file not found: {osm_path}")
    
    do_crop = lat is not None and lon is not None
    if delta is None:
        delta = DELTA_DEFAULT
    
    print(f"\n{'=' * 60}")
    print(f"Converting OSM to SUMO network")
    print(f"  Input: {osm_path.name}")
    print(f"  Scene name: {scene_name}")
    if do_crop:
        print(f"  Crop center: ({lat:.6f}, {lon:.6f})")
        print(f"  Crop delta: ±{delta} degrees (~{delta * 111:.0f}m)")
    print(f"{'=' * 60}")

    scene_dir = scenes_dir / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    net_file = f"{scene_name}.net.xml"
    osm_file = f"{scene_name}.osm"
    net_output_path = scene_dir / net_file
    
    # Crop OSM if coordinates provided
    if do_crop:
        print(f"\n1. Cropping OSM to area around ({lat:.6f}, {lon:.6f})...")
        print(f"   Original: {osm_path} ({osm_path.stat().st_size / 1024:.1f} KB)")
        cropped_osm_path = scene_dir / f"{scene_name}_cropped.osm"
        crop_osm_to_bbox(osm_path, cropped_osm_path, lat, lon, delta)
        print(f"   Cropped: {cropped_osm_path} ({cropped_osm_path.stat().st_size / 1024:.1f} KB)")
        osm_to_convert = cropped_osm_path
        step = 2
    else:
        osm_to_convert = osm_path
        step = 1

    print(f"\n{step}. Converting OSM to SUMO .net.xml (main roads only)...")
    print(f"   Input: {osm_to_convert} ({osm_to_convert.stat().st_size / 1024:.1f} KB)")
    
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
    
    cmd = [
        _find_netconvert(),
        "--osm-files", str(osm_to_convert),
        "-o", str(net_output_path),
        "--tls.discard-loaded",
        "--tls.discard-simple",
        "--no-turnarounds",
        "--geometry.remove",
        "--junctions.join",
        "--keep-edges.by-type", ",".join(keep_types),
        "--remove-edges.by-type", ",".join(remove_types),
        "--no-internal-links",
    ]
    
    # Add geo-boundary filter if cropping
    if do_crop:
        geo_boundary = f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}"
        cmd.extend(["--keep-edges.in-geo-boundary", geo_boundary])
    
    subprocess.run(cmd, check=True)

    print(f"   Output: {net_output_path} ({net_output_path.stat().st_size / 1024:.1f} KB)")

    print(f"\n{step + 1}. Saving scene files...")
    
    # Copy OSM to scene dir (cropped if applicable)
    (scene_dir / osm_file).write_bytes(osm_to_convert.read_bytes())

    meta = {
        "scene_name": scene_name,
        "net_file": net_file,
        "osm_file": osm_file,
        "source_osm": str(osm_path),
    }
    if do_crop:
        meta["latitude"] = lat
        meta["longitude"] = lon
        meta["crop_delta"] = delta
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
    parser.add_argument("--lat", type=float, help="Latitude for cropping center (optional)")
    parser.add_argument("--lon", type=float, help="Longitude for cropping center (optional)")
    parser.add_argument("--delta", type=float, default=DELTA_DEFAULT,
                        help=f"Crop radius in degrees (default: {DELTA_DEFAULT}, ~{DELTA_DEFAULT*111:.0f}m)")
    
    args = parser.parse_args()
    scenes_dir = Path(args.scenes_dir)
    
    scene_dir = convert_osm_to_sumo(
        osm_path=args.osm,
        scene_name=args.name,
        scenes_dir=scenes_dir,
        lat=args.lat,
        lon=args.lon,
        delta=args.delta,
    )
    
    print(f"\n{'=' * 60}")
    print("SUCCESS!")
    print(f"Scene created at: {scene_dir}")
    print(f"\nTo render static map:")
    print(f"  python run_single_sumo_scene.py \\")
    print(f"    --scene-dir {scene_dir} \\")
    print(f"    --out {scene_dir / 'output.png'}")
    print(f"\nTo run IDM simulation:")
    print(f"  python run_idm_simulation.py \\")
    print(f"    --scene-dir {scene_dir} \\")
    print(f"    --out {scene_dir / 'simulation.gif'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
