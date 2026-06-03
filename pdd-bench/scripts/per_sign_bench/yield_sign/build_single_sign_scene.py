#!/usr/bin/env python3
"""Convert OSM map to SUMO net.xml (main roads only, no signs/lights/railways).

Each scene folder under scenes/ must contain:
  - map.osm          — full OSM extract
  - coordinates.json — crop center: {"lat": ..., "lon": ...}

Usage:
    python build_single_sign_scene.py savvinskaya_3 --delta 0.001
    python build_single_sign_scene.py savvinskaya_3
"""

import argparse
import json
import subprocess
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

DELTA_DEFAULT = 0.002  # ~200m in each direction
SCENES_DIR_DEFAULT = Path(__file__).resolve().parent / "scenes"


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

    osm_path = scene_dir / "map.osm"
    if not osm_path.exists():
        raise FileNotFoundError(f"OSM file not found: {osm_path}")

    coords_path = scene_dir / "coordinates.json"
    if not coords_path.exists():
        raise FileNotFoundError(
            f"coordinates.json not found in {scene_dir}. "
            'Expected: {"lat": <float>, "lon": <float>}'
        )

    with open(coords_path, encoding="utf-8") as f:
        coords = json.load(f)

    lat = coords.get("lat", coords.get("latitude"))
    lon = coords.get("lon", coords.get("longitude"))
    if lat is None or lon is None:
        raise ValueError(
            f"{coords_path} must contain lat/lon (or latitude/longitude), got: {coords}"
        )

    return osm_path, float(lat), float(lon)


def crop_osm_to_bbox(input_osm_path, output_osm_path, lat, lon, delta):
    """Crop an OSM file to a bounding box around the given coordinates.

    Only keeps nodes within bbox and ways that have at least 2 nodes in bbox.
    Roads extending outside the bbox are truncated at the boundary.
    """
    tree = ET.parse(input_osm_path)
    root = tree.getroot()

    min_lat, max_lat = lat - delta, lat + delta
    min_lon, max_lon = lon - delta, lon + delta

    nodes_in_bbox = {}
    for node in root.findall("node"):
        node_id = node.get("id")
        node_lat = float(node.get("lat"))
        node_lon = float(node.get("lon"))
        if min_lat <= node_lat <= max_lat and min_lon <= node_lon <= max_lon:
            nodes_in_bbox[node_id] = node

    print(f"   Found {len(nodes_in_bbox)} nodes within bbox")

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

    new_root = ET.Element("osm")
    new_root.set("version", root.get("version", "0.6"))
    new_root.set("generator", "crop_osm_to_bbox")

    new_bounds = ET.SubElement(new_root, "bounds")
    new_bounds.set("minlat", str(min_lat))
    new_bounds.set("minlon", str(min_lon))
    new_bounds.set("maxlat", str(max_lat))
    new_bounds.set("maxlon", str(max_lon))

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


def convert_osm_to_sumo(osm_path, scene_name, scenes_dir, lat, lon, delta):
    """Crop OSM around (lat, lon) and convert to SUMO net.xml."""
    osm_path = Path(osm_path)
    if not osm_path.exists():
        raise FileNotFoundError(f"OSM file not found: {osm_path}")

    print(f"\n{'=' * 60}")
    print("Converting OSM to SUMO network")
    print(f"  Input: {osm_path.name}")
    print(f"  Scene name: {scene_name}")
    print(f"  Crop center: ({lat:.6f}, {lon:.6f})")
    print(f"  Crop delta: ±{delta} degrees (~{delta * 111:.0f}m)")
    print(f"{'=' * 60}")

    scene_dir = scenes_dir / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    net_file = f"{scene_name}.net.xml"
    osm_file = f"{scene_name}.osm"
    net_output_path = scene_dir / net_file

    print(f"\n1. Cropping OSM to area around ({lat:.6f}, {lon:.6f})...")
    print(f"   Original: {osm_path} ({osm_path.stat().st_size / 1024:.1f} KB)")
    cropped_osm_path = scene_dir / f"{scene_name}_cropped.osm"
    crop_osm_to_bbox(osm_path, cropped_osm_path, lat, lon, delta)
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

    geo_boundary = f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}"
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

    (scene_dir / osm_file).write_bytes(osm_to_convert.read_bytes())

    try:
        source_osm = osm_path.relative_to(scenes_dir.parent)
    except ValueError:
        source_osm = osm_path

    meta = {
        "scene_name": scene_name,
        "net_file": net_file,
        "osm_file": osm_file,
        "source_osm": str(source_osm),
        "latitude": lat,
        "longitude": lon,
        "crop_delta": delta,
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
        description="Build SUMO net from a scene folder (map.osm + coordinates.json)",
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
        "--delta",
        type=float,
        default=DELTA_DEFAULT,
        help=f"Crop radius in degrees (default: {DELTA_DEFAULT}, ~{DELTA_DEFAULT * 111:.0f}m)",
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
        delta=args.delta,
    )

    print(f"\n{'=' * 60}")
    print("SUCCESS!")
    print(f"Scene created at: {scene_dir}")
    print("\nTo render static map:")
    print(f"  python run_single_sumo_scene.py {args.scene}")
    print("\nTo run simulation (IDM/CARL/PLANT):")
    print(f"  python run_simulation.py {args.scene}")
    print(f"  python run_simulation.py {args.scene} --policy carl")
    print(f"  python run_simulation.py {args.scene} --policy plant2")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
