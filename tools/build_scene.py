#!/usr/bin/env python3
"""Convert OSM map to SUMO net.xml for a scene."""

import argparse
import json
import math
import subprocess
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

RADIUS_DEFAULT = 200  # meters in each direction from center


def meters_to_degrees(meters: float, lat: float) -> tuple[float, float]:
    """Convert meters to degrees at a given latitude.

    Returns (delta_lat, delta_lon) in degrees.
    - 1 degree latitude ≈ 111,320 meters (constant)
    - 1 degree longitude ≈ 111,320 * cos(lat) meters (varies with latitude)
    """
    meters_per_degree_lat = 111_320
    meters_per_degree_lon = 111_320 * math.cos(math.radians(lat))

    delta_lat = meters / meters_per_degree_lat
    delta_lon = meters / meters_per_degree_lon

    return delta_lat, delta_lon


SCENES_DIR_DEFAULT = Path(__file__).resolve().parent / "scenes"

# Neutral filenames inside each scene folder (folder name identifies the scene)
NET_FILE = "map.net.xml"
CROPPED_OSM_FILE = "cropped.osm"
SOURCE_OSM_FILE = "map.osm"


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


def load_scene_inputs(scenes_dir: Path, scene_name: str) -> tuple[Path, float, float]:
    """Resolve map.osm and crop center from a scene folder."""
    scene_dir = scenes_dir / scene_name
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene folder not found: {scene_dir}")

    osm_path = scene_dir / SOURCE_OSM_FILE
    if not osm_path.exists():
        raise FileNotFoundError(f"OSM file not found: {osm_path}")

    lat = lon = None
    meta_path = scene_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        lat = meta.get("latitude", meta.get("lat"))
        lon = meta.get("longitude", meta.get("lon"))
    leftover = scene_dir / "center.json"
    if (lat is None or lon is None) and leftover.is_file():
        coords = json.loads(leftover.read_text(encoding="utf-8"))
        lat = coords.get("lat", coords.get("latitude"))
        lon = coords.get("lon", coords.get("longitude"))
    if lat is None or lon is None:
        raise FileNotFoundError(
            f"no lat/lon in {meta_path} (or leftover center.json) for {scene_dir}"
        )

    return osm_path, float(lat), float(lon)


def crop_osm_to_bbox(input_osm_path, output_osm_path, lat, lon, delta_lat, delta_lon):
    """Crop an OSM file to a bounding box around the given coordinates.

    Keeps entire roads (all nodes) if they have at least one node within the bbox.
    This prevents roads from being truncated at the boundary.
    """
    tree = ET.parse(input_osm_path)
    root = tree.getroot()

    min_lat, max_lat = lat - delta_lat, lat + delta_lat
    min_lon, max_lon = lon - delta_lon, lon + delta_lon

    # Build lookup of all nodes
    all_nodes = {}
    nodes_in_bbox = set()
    for node in root.findall("node"):
        node_id = node.get("id")
        node_lat = float(node.get("lat"))
        node_lon = float(node.get("lon"))
        all_nodes[node_id] = node
        if min_lat <= node_lat <= max_lat and min_lon <= node_lon <= max_lon:
            nodes_in_bbox.add(node_id)

    print(f"   Found {len(nodes_in_bbox)} nodes within bbox (total: {len(all_nodes)})")

    # Keep entire ways that have at least one node in the bbox
    kept_ways = []
    for way in root.findall("way"):
        node_refs = [nd.get("ref") for nd in way.findall("nd")]
        
        # Check if any node of this way is inside the bbox
        has_node_in_bbox = any(ref in nodes_in_bbox for ref in node_refs)
        
        if has_node_in_bbox:
            # Keep the entire way (all nodes), not just the truncated portion
            kept_ways.append(way)

    print(f"   Keeping {len(kept_ways)} ways that touch bbox (full roads, not truncated)")

    # Collect all nodes used by kept ways
    used_node_ids = set()
    for way in kept_ways:
        for nd in way.findall("nd"):
            used_node_ids.add(nd.get("ref"))

    new_root = ET.Element("osm")
    new_root.set("version", root.get("version", "0.6"))
    new_root.set("generator", "crop_osm_to_bbox")

    new_bounds = ET.SubElement(new_root, "bounds")
    new_bounds.set("minlat", str(min_lat))
    new_bounds.set("minlon", str(min_lon))
    new_bounds.set("maxlat", str(max_lat))
    new_bounds.set("maxlon", str(max_lon))

    # Add all nodes used by kept ways
    for node_id in used_node_ids:
        if node_id in all_nodes:
            new_root.append(all_nodes[node_id])

    # Add kept ways (full, not truncated)
    for way in kept_ways:
        new_root.append(way)

    new_tree = ET.ElementTree(new_root)
    ET.indent(new_tree, space="  ")
    new_tree.write(output_osm_path, encoding="unicode", xml_declaration=True)

    print(f"   Final: {len(used_node_ids)} nodes, {len(kept_ways)} ways")
    return output_osm_path


def convert_osm_to_sumo(osm_path, scene_name, scenes_dir, lat, lon, radius):
    """Crop OSM around (lat, lon) and convert to SUMO net.xml.
    
    Args:
        radius: Crop radius in meters from the center point.
    """
    osm_path = Path(osm_path)
    if not osm_path.exists():
        raise FileNotFoundError(f"OSM file not found: {osm_path}")

    delta_lat, delta_lon = meters_to_degrees(radius, lat)

    print(f"\n{'=' * 60}")
    print("Converting OSM to SUMO network")
    print(f"  Input: {osm_path.name}")
    print(f"  Scene name: {scene_name}")
    print(f"  Crop center: ({lat:.6f}, {lon:.6f})")
    print(f"  Crop radius: {radius}m (±{delta_lat:.6f}° lat, ±{delta_lon:.6f}° lon)")
    print(f"{'=' * 60}")

    scene_dir = scenes_dir / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    net_output_path = scene_dir / NET_FILE
    cropped_osm_path = scene_dir / CROPPED_OSM_FILE

    print(f"\n1. Cropping OSM to area around ({lat:.6f}, {lon:.6f})...")
    print(f"   Original: {osm_path} ({osm_path.stat().st_size / 1024:.1f} KB)")
    crop_osm_to_bbox(osm_path, cropped_osm_path, lat, lon, delta_lat, delta_lon)
    print(f"   Cropped: {cropped_osm_path} ({cropped_osm_path.stat().st_size / 1024:.1f} KB)")
    osm_to_convert = cropped_osm_path

    print("\n2. Converting OSM to SUMO .net.xml (main roads only)...")
    print(f"   Input: {osm_to_convert} ({osm_to_convert.stat().st_size / 1024:.1f} KB)")

    keep_types = [
        "highway.motorway", "highway.motorway_link",
        "highway.trunk", "highway.trunk_link",
        "highway.primary", "highway.primary_link",
        "highway.secondary", "highway.secondary_link",
        "highway.tertiary", "highway.tertiary_link",
        "highway.unclassified",
        "highway.residential",
    ]

    remove_types = [
        "railway.subway", "railway.rail", "railway.light_rail",
        "railway.tram", "railway.monorail", "railway.preserved",
        "highway.service", "highway.footway", "highway.path",
        "highway.cycleway", "highway.pedestrian", "highway.steps",
        "highway.track", "highway.bridleway", "highway.corridor",
    ]

    geo_boundary = f"{lon - delta_lon},{lat - delta_lat},{lon + delta_lon},{lat + delta_lat}"
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
        "--keep-edges.in-geo-boundary", geo_boundary,
    ]

    subprocess.run(cmd, check=True)

    print(f"   Output: {net_output_path} ({net_output_path.stat().st_size / 1024:.1f} KB)")

    print("\n3. Saving scene files...")

    try:
        source_osm = osm_path.relative_to(scenes_dir.parent)
    except ValueError:
        source_osm = osm_path

    meta = {
        "scene_name": scene_name,
        "net_file": NET_FILE,
        "osm_file": CROPPED_OSM_FILE,
        "source_osm": str(source_osm),
        "latitude": lat,
        "longitude": lon,
        "crop_radius_m": radius,
    }
    with open(scene_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nScene saved to: {scene_dir}")
    print(f"  meta.json: {scene_dir / 'meta.json'}")
    print(f"  net.xml: {scene_dir / NET_FILE}")
    print(f"  osm: {scene_dir / CROPPED_OSM_FILE}")

    return scene_dir


def main():
    parser = argparse.ArgumentParser(
        description="Build SUMO net from a scene folder (map.osm + meta.json lat/lon)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "scene",
        help="Scene folder name under scenes/ (e.g. savvinskaya_3)",
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=SCENES_DIR_DEFAULT,
        help=f"Scenes root directory (default: {SCENES_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=RADIUS_DEFAULT,
        help=f"Crop radius in meters from center (default: {RADIUS_DEFAULT}m)",
    )

    args = parser.parse_args()
    scenes_dir = Path(args.scenes_dir)

    osm_path, lat, lon = load_scene_inputs(scenes_dir, args.scene)

    scene_dir = convert_osm_to_sumo(
        osm_path=osm_path,
        scene_name=args.scene,
        scenes_dir=scenes_dir,
        lat=lat,
        lon=lon,
        radius=args.radius,
    )

    print(f"\n{'=' * 60}")
    print("SUCCESS!")
    print(f"Scene created at: {scene_dir}")
    print("\nTo render static map (debug):")
    print(f"  python -m tools.render_map {args.scene}")
    print("\nTo run simulation (debug):")
    print(f"  python -m tools.run_simulation {args.scene}")
    print(f"  python -m tools.run_simulation {args.scene} --policy carl")
    print("\nTo generate manifest for evaluation:")
    print(f"  python generate_manifest.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
