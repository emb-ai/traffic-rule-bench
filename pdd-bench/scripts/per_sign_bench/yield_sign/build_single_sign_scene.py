#!/usr/bin/env python3
"""Build a single scene for a specific sign from CSV or custom OSM file.

Mode 1: From CSV (requires sign coordinates in CSV)
    python build_single_sign_scene.py \
        --csv ../../../data/data-cleaned.csv \
        --sign-id 449674 \
        --scenes-dir ./scenes

Mode 2: From custom OSM file (no CSV needed, provide coordinates manually)
    python build_single_sign_scene.py \
        --osm scenes/savvinskaya_3/map.osm \
        --lat 55.739375 \
        --lon 37.572461 \
        --sign-type 2.4 \
        --name my_custom_scene \
        --scenes-dir ./scenes
"""

import argparse
import json
import subprocess
import sys
import time
import random
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import pyproj
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

DELTA = 0.002
OSM_FRAG_DIR = Path("osm_fragments")
MAPS_DIR = Path("./maps")

OVERPASS_URLS = [
    "https://overpass.nchc.org.tw/api/interpreter",
]


def create_overpass_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "TrafficSignBench/1.0 (research project; contact@example.com)",
        "Accept": "application/xml, */*",
    })
    retry_strategy = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST", "GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_osm_fragment(lat, lon, output_path, delta=DELTA):
    bbox = f"{lat - delta},{lon - delta},{lat + delta},{lon + delta}"
    query = f"""
    [out:xml][timeout:60];
    (
      node({bbox});
      way({bbox});
      relation({bbox});
    );
    out meta;
    """

    jitter = random.uniform(1.0, 3.0)
    print(f"Sleeping for {jitter:.1f}s before Overpass request...")
    time.sleep(jitter)

    session = create_overpass_session()

    last_exception = None
    for url in OVERPASS_URLS:
        try:
            print(f"Trying Overpass endpoint: {url}")
            response = session.post(url, data={"data": query}, timeout=60)
            response.raise_for_status()
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print("OSM fragment downloaded successfully.")
            return
        except Exception as e:
            print(f"Failed on {url}: {e}")
            last_exception = e
            time.sleep(3)

    # Fallback: try direct OSM API (limited to 0.25 degree bbox)
    print("Trying direct OSM API as fallback...")
    try:
        osm_bbox = f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}"
        osm_url = f"https://api.openstreetmap.org/api/0.6/map?bbox={osm_bbox}"
        response = session.get(osm_url, timeout=60)
        response.raise_for_status()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print("OSM fragment downloaded via direct OSM API.")
        return
    except Exception as e:
        print(f"Direct OSM API via requests failed: {e}")

    # Last resort: try curl (often works when Python requests are blocked)
    print("Trying curl as last resort...")
    try:
        osm_bbox = f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}"
        osm_url = f"https://api.openstreetmap.org/api/0.6/map?bbox={osm_bbox}"
        result = subprocess.run(
            ["curl", "-s", "-o", str(output_path), osm_url],
            timeout=90,
            capture_output=True
        )
        if result.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 1000:
            print("OSM fragment downloaded via curl.")
            return
        else:
            print(f"Curl failed: {result.stderr.decode()}")
    except Exception as e:
        print(f"Curl also failed: {e}")

    raise RuntimeError(f"All download methods failed. Last Overpass error: {last_exception}")


def crop_osm_to_bbox(input_osm_path, output_osm_path, lat, lon, delta=DELTA):
    """Crop an OSM file to a bounding box around the given coordinates.
    
    STRICT cropping:
    - Only keeps nodes that are within the bbox
    - Truncates ways to only include nodes within the bbox
    - Only keeps ways that have at least 2 remaining nodes
    - Discards relations (they often reference elements outside bbox)
    """
    tree = ET.parse(input_osm_path)
    root = tree.getroot()
    
    min_lat, max_lat = lat - delta, lat + delta
    min_lon, max_lon = lon - delta, lon + delta
    
    # Step 1: Find all nodes strictly within bbox
    nodes_in_bbox = {}
    for node in root.findall("node"):
        node_id = node.get("id")
        node_lat = float(node.get("lat"))
        node_lon = float(node.get("lon"))
        if min_lat <= node_lat <= max_lat and min_lon <= node_lon <= max_lon:
            nodes_in_bbox[node_id] = node
    
    print(f"   Found {len(nodes_in_bbox)} nodes within bbox")
    
    # Step 2: Process ways - truncate to only include nodes within bbox
    # Keep way if it has at least 2 consecutive nodes within bbox
    truncated_ways = []
    for way in root.findall("way"):
        way_id = way.get("id")
        node_refs = [nd.get("ref") for nd in way.findall("nd")]
        
        # Filter to only nodes in bbox
        filtered_refs = [ref for ref in node_refs if ref in nodes_in_bbox]
        
        # Need at least 2 nodes to form a way segment
        if len(filtered_refs) >= 2:
            # Create a new way element with only the filtered nodes
            new_way = ET.Element("way")
            new_way.set("id", way_id)
            # Copy other attributes
            for attr in ["visible", "version", "changeset", "timestamp", "user", "uid"]:
                if way.get(attr):
                    new_way.set(attr, way.get(attr))
            
            # Add filtered node refs
            for ref in filtered_refs:
                nd = ET.SubElement(new_way, "nd")
                nd.set("ref", ref)
            
            # Copy tags
            for tag in way.findall("tag"):
                new_tag = ET.SubElement(new_way, "tag")
                new_tag.set("k", tag.get("k"))
                new_tag.set("v", tag.get("v"))
            
            truncated_ways.append(new_way)
    
    print(f"   Keeping {len(truncated_ways)} ways (truncated to bbox)")
    
    # Step 3: Build new OSM tree
    new_root = ET.Element("osm")
    new_root.set("version", root.get("version", "0.6"))
    new_root.set("generator", "crop_osm_to_bbox")
    
    # Set bounds
    new_bounds = ET.SubElement(new_root, "bounds")
    new_bounds.set("minlat", str(min_lat))
    new_bounds.set("minlon", str(min_lon))
    new_bounds.set("maxlat", str(max_lat))
    new_bounds.set("maxlon", str(max_lon))
    
    # Add nodes (only those referenced by kept ways)
    used_nodes = set()
    for way in truncated_ways:
        for nd in way.findall("nd"):
            used_nodes.add(nd.get("ref"))
    
    for node_id in used_nodes:
        if node_id in nodes_in_bbox:
            new_root.append(nodes_in_bbox[node_id])
    
    # Add truncated ways
    for way in truncated_ways:
        new_root.append(way)
    
    # Skip relations - they usually reference elements outside the bbox
    
    # Write output
    new_tree = ET.ElementTree(new_root)
    ET.indent(new_tree, space="  ")
    new_tree.write(output_osm_path, encoding="unicode", xml_declaration=True)
    
    print(f"   Final: {len(used_nodes)} nodes, {len(truncated_ways)} ways")
    print(f"   Bbox: ({min_lat:.6f}, {min_lon:.6f}) to ({max_lat:.6f}, {max_lon:.6f})")
    
    return output_osm_path


def find_closest_way_and_distance(osm_path, sign_lat, sign_lon):
    """Returns (way_id, distance_from_start) for the nearest road in the OSM file."""
    tree = ET.parse(osm_path)
    root = tree.getroot()
    nodes = {n.get("id"): (float(n.get("lat")), float(n.get("lon"))) for n in root.findall("node")}
    if not nodes:
        raise ValueError("No nodes in OSM file")

    proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32637", always_xy=True)
    sign_x, sign_y = proj.transform(sign_lon, sign_lat)
    sign_point = Point(sign_x, sign_y)

    min_dist = float("inf")
    closest_way_id = None
    distance_from_start = None

    for way in root.findall("way"):
        if not any(tag.get("k") == "highway" for tag in way.findall("tag")):
            continue
        if any(tag.get("v") in ("footway", "steps") for tag in way.findall("tag")):
            continue

        coords = []
        for nd in way.findall("nd"):
            nid = nd.get("ref")
            if nid in nodes:
                lat, lon = nodes[nid]
                x, y = proj.transform(lon, lat)
                coords.append((x, y))
            else:
                coords = []
                break
        if len(coords) < 2:
            continue

        line = LineString(coords)
        dist = sign_point.distance(line)
        if dist < min_dist:
            min_dist = dist
            closest_way_id = way.get("id")
            nearest_on_road = nearest_points(line, sign_point)[0]
            dist_along = 0.0
            line_coords = list(line.coords)
            for i in range(len(line_coords) - 1):
                segment = LineString([line_coords[i], line_coords[i + 1]])
                if segment.distance(nearest_on_road) < 1e-6:
                    dist_along += Point(line_coords[i]).distance(nearest_on_road)
                    break
                else:
                    dist_along += segment.length
            distance_from_start = dist_along / line.length

    if closest_way_id is None:
        raise ValueError(f"No road found near ({sign_lat}, {sign_lon})")
    return closest_way_id, distance_from_start


def get_edge_length(edge):
    """Returns the edge length, computing it from shape (own or lane) if needed."""
    length = edge.get("length")
    if length is not None:
        return float(length)

    shape_str = None
    lane = edge.find("lane")
    if lane is not None:
        shape_str = lane.get("shape")
    if shape_str:
        points = shape_str.strip().split()
        coords = [tuple(map(float, p.split(','))) for p in points]
        if len(coords) >= 2:
            return sum(
                ((coords[i + 1][0] - coords[i][0]) ** 2 + (coords[i + 1][1] - coords[i][1]) ** 2) ** 0.5
                for i in range(len(coords) - 1)
            )
    return 0.0


def find_edge_and_offset_in_sumo_by_way_id(net_path, way_id, target_distance):
    tree = ET.parse(net_path)
    root = tree.getroot()

    edges = []

    for edge in root.findall("edge"):
        edge_id = edge.get("id")

        if edge_id == str(way_id):
            length = get_edge_length(edge)
            edges.append((0, edge_id, length))

        elif edge_id.startswith(f"{way_id}#"):
            suffix = edge_id[len(f"{way_id}#"):]
            try:
                idx = int(suffix)
                length = get_edge_length(edge)
                edges.append((idx, edge_id, length))
            except ValueError:
                pass

    if not edges:
        raise ValueError(f"No edges with way_id '{way_id}' found in {net_path}")

    edges.sort(key=lambda x: x[0])

    cumulative_sum = 0.0
    for idx, edge_id, length in edges:
        cumulative_sum += length

    cumulative = 0.0
    for idx, edge_id, length in edges:
        if (cumulative + length) / cumulative_sum >= target_distance:
            s_offset = target_distance * cumulative_sum - cumulative
            return edge_id, s_offset
        cumulative += length

    raise ValueError(f"Distance {target_distance} exceeds total length ({cumulative}) of way {way_id}")


def _find_netconvert() -> str:
    """Find netconvert executable."""
    import shutil
    
    # Check common locations
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


def build_scene_for_sign(sign_row, sign_type, scenes_dir):
    """Build a scene for a single sign."""
    sign_id = int(sign_row["ID"])
    lat = float(sign_row["Latitude_WGS84"])
    lon = float(sign_row["Longitude_WGS84"])

    print(f"\n{'=' * 60}")
    print(f"Building scene for sign ID: {sign_id}")
    print(f"  SignType: {sign_type}")
    print(f"  Location: ({lat:.6f}, {lon:.6f})")
    print(f"{'=' * 60}")

    OSM_FRAG_DIR.mkdir(exist_ok=True)
    MAPS_DIR.mkdir(exist_ok=True)

    sign_type_dir = scenes_dir / sign_type
    sign_type_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"sign_{sign_id}"
    osm_raw = OSM_FRAG_DIR / f"{base_name}.osm"
    net_file = f"{base_name}.net.xml"
    scene_dir = sign_type_dir / base_name

    print("   Downloading from Overpass API...")
    download_osm_fragment(lat, lon, osm_raw)

    print("\n2. Finding closest way and distance along it...")
    way_id, distance_from_start = find_closest_way_and_distance(osm_raw, lat, lon)
    print(f"   way_id: {way_id}")
    print(f"   distance_from_start (normalized): {distance_from_start:.4f}")

    net_output_path = MAPS_DIR / net_file
    print("\n3. Converting OSM to SUMO .net.xml...")
    if net_output_path.exists() and net_output_path.stat().st_size > 1000:
        print(f"   SUMO net file already exists: {net_output_path}")
    else:
        # Use geo-boundary filter to ensure network stays within bbox
        geo_boundary = f"{lon - DELTA},{lat - DELTA},{lon + DELTA},{lat + DELTA}"
        subprocess.run([
            _find_netconvert(),
            "--osm-files", str(osm_raw),
            "-o", str(net_output_path),
            "--osm.sidewalks",
            "--osm.crossings",
            "--crossings.guess",
            "--walkingareas",
            "--keep-edges.in-geo-boundary", geo_boundary,
        ], check=True)

    print("\n4. Finding edge and offset in SUMO network...")
    road_id, s_offset = find_edge_and_offset_in_sumo_by_way_id(
        net_output_path, way_id, distance_from_start
    )
    print(f"   road_id (SUMO edge): {road_id}")
    print(f"   s_offset: {s_offset:.2f}")

    print("\n5. Saving scene...")
    scene_dir.mkdir(exist_ok=True)
    
    # Save .net.xml
    (scene_dir / net_file).write_bytes(net_output_path.read_bytes())
    
    # Save .osm file to scene directory
    osm_file = f"{base_name}.osm"
    (scene_dir / osm_file).write_bytes(osm_raw.read_bytes())

    meta = {
        "sign_id": sign_id,
        "sign_type": sign_type,
        "latitude": lat,
        "longitude": lon,
        "osm_way_id": way_id,
        "road_id": road_id,
        "distance_from_start": s_offset,
        "net_file": net_file,
        "osm_file": osm_file
    }
    with open(scene_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nScene saved to: {scene_dir}")
    print(f"  meta.json: {scene_dir / 'meta.json'}")
    print(f"  net.xml: {scene_dir / net_file}")
    print(f"  osm: {scene_dir / osm_file}")

    return scene_dir


def build_scene_from_custom_osm(osm_path, lat, lon, sign_type, scene_name, scenes_dir):
    """Build a scene from a custom OSM file with manual coordinates.
    
    The OSM file is cropped to a small area (±DELTA degrees) around the sign
    coordinates before processing, similar to downloaded OSM fragments.
    """
    osm_path = Path(osm_path)
    if not osm_path.exists():
        raise FileNotFoundError(f"OSM file not found: {osm_path}")
    
    print(f"\n{'=' * 60}")
    print(f"Building scene from custom OSM: {osm_path.name}")
    print(f"  Scene name: {scene_name}")
    print(f"  SignType: {sign_type}")
    print(f"  Location: ({lat:.6f}, {lon:.6f})")
    print(f"{'=' * 60}")

    OSM_FRAG_DIR.mkdir(exist_ok=True)
    MAPS_DIR.mkdir(exist_ok=True)

    sign_type_dir = scenes_dir / sign_type
    sign_type_dir.mkdir(parents=True, exist_ok=True)

    scene_dir = sign_type_dir / scene_name
    net_file = f"{scene_name}.net.xml"
    osm_file = f"{scene_name}.osm"

    if scene_dir.exists():
        print(f"Scene already exists: {scene_dir}")
        print("Use --force to overwrite or choose a different --name")
        return scene_dir

    print("\n1. Cropping custom OSM file to area around sign...")
    print(f"   Original OSM file: {osm_path} ({osm_path.stat().st_size / 1024:.1f} KB)")
    
    # Crop OSM to small area around the sign
    cropped_osm_path = OSM_FRAG_DIR / f"{scene_name}_cropped.osm"
    crop_osm_to_bbox(osm_path, cropped_osm_path, lat, lon, delta=DELTA)
    print(f"   Cropped OSM file: {cropped_osm_path} ({cropped_osm_path.stat().st_size / 1024:.1f} KB)")

    print("\n2. Finding closest way and distance along it...")
    way_id, distance_from_start = find_closest_way_and_distance(cropped_osm_path, lat, lon)
    print(f"   way_id: {way_id}")
    print(f"   distance_from_start (normalized): {distance_from_start:.4f}")

    net_output_path = MAPS_DIR / net_file
    print("\n3. Converting cropped OSM to SUMO .net.xml...")
    
    # Use netconvert's geo-boundary filter to clip the network to the bbox
    # This is more reliable than pre-filtering OSM (which can keep long ways that extend outside)
    geo_boundary = f"{lon - DELTA},{lat - DELTA},{lon + DELTA},{lat + DELTA}"
    print(f"   Using geo-boundary filter: {geo_boundary}")
    
    subprocess.run([
        _find_netconvert(),
        "--osm-files", str(cropped_osm_path),
        "-o", str(net_output_path),
        "--osm.sidewalks",
        "--osm.crossings",
        "--crossings.guess",
        "--walkingareas",
        "--keep-edges.in-geo-boundary", geo_boundary,
    ], check=True)

    print("\n4. Finding edge and offset in SUMO network...")
    road_id, s_offset = find_edge_and_offset_in_sumo_by_way_id(
        net_output_path, way_id, distance_from_start
    )
    print(f"   road_id (SUMO edge): {road_id}")
    print(f"   s_offset: {s_offset:.2f}")

    print("\n5. Saving scene...")
    scene_dir.mkdir(exist_ok=True)
    
    # Save .net.xml
    (scene_dir / net_file).write_bytes(net_output_path.read_bytes())
    
    # Save cropped .osm file to scene directory
    (scene_dir / osm_file).write_bytes(cropped_osm_path.read_bytes())

    meta = {
        "scene_name": scene_name,
        "sign_type": sign_type,
        "latitude": lat,
        "longitude": lon,
        "osm_way_id": way_id,
        "road_id": road_id,
        "distance_from_start": s_offset,
        "net_file": net_file,
        "osm_file": osm_file,
        "source_osm": str(osm_path),
        "crop_delta": DELTA,
        "cropped": True
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
        description="Build a single scene for a sign (from CSV or custom OSM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From CSV:
  python build_single_sign_scene.py --csv data.csv --sign-id 449674

  # From custom OSM file:
  python build_single_sign_scene.py --osm map.osm --lat 55.739 --lon 37.572 --sign-type 2.4 --name my_scene
        """
    )
    
    # Mode 1: CSV-based
    parser.add_argument("--csv", help="Path to CSV with signs (Mode 1)")
    parser.add_argument("--sign-id", type=int, help="Sign ID to process (Mode 1)")
    
    # Mode 2: Custom OSM
    parser.add_argument("--osm", help="Path to custom OSM file (Mode 2)")
    parser.add_argument("--lat", type=float, help="Latitude of sign location (Mode 2)")
    parser.add_argument("--lon", type=float, help="Longitude of sign location (Mode 2)")
    parser.add_argument("--name", help="Scene name (Mode 2, default: derived from OSM filename)")
    
    # Common options
    parser.add_argument("--sign-type", default="2.4",
                        help='Sign type code (e.g. "2.4"). Auto-detected from CSV in Mode 1.')
    parser.add_argument("--scenes-dir", default="scenes", help="Path to scenes output directory")
    
    args = parser.parse_args()
    scenes_dir = Path(args.scenes_dir)

    # Determine mode
    if args.osm:
        # Mode 2: Custom OSM file
        if args.lat is None or args.lon is None:
            print("Error: --lat and --lon are required when using --osm")
            print("These specify where the sign is located within the OSM area.")
            sys.exit(1)
        
        scene_name = args.name
        if not scene_name:
            scene_name = Path(args.osm).stem  # Use OSM filename without extension
        
        scene_dir = build_scene_from_custom_osm(
            osm_path=args.osm,
            lat=args.lat,
            lon=args.lon,
            sign_type=args.sign_type,
            scene_name=scene_name,
            scenes_dir=scenes_dir
        )
    
    elif args.csv and args.sign_id:
        # Mode 1: CSV-based
        print(f"Loading CSV: {args.csv}")
        df = pd.read_csv(args.csv, sep=";")
        df["ID"] = df["ID"].astype(int)

        sign_row = df[df["ID"] == args.sign_id]
        if sign_row.empty:
            print(f"Error: Sign ID {args.sign_id} not found in CSV")
            sys.exit(1)

        sign_row = sign_row.iloc[0]

        if args.sign_type != "2.4":  # User explicitly provided sign type
            sign_type = args.sign_type
        else:
            full_sign_type = sign_row["SignType"]
            sign_type = full_sign_type.split()[0] if pd.notna(full_sign_type) else "unknown"

        scene_dir = build_scene_for_sign(sign_row, sign_type, scenes_dir)
    
    else:
        print("Error: Please specify either:")
        print("  Mode 1: --csv and --sign-id")
        print("  Mode 2: --osm, --lat, and --lon")
        parser.print_help()
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("SUCCESS!")
    print(f"Scene created at: {scene_dir}")
    print(f"\nTo run this scene:")
    print(f"  python run_single_sumo_scene.py \\")
    print(f"    --scene-dir {scene_dir.relative_to(Path.cwd()) if scene_dir.is_relative_to(Path.cwd()) else scene_dir} \\")
    print(f"    --traffic-density 0.0 \\")
    print(f"    --out {scene_dir / 'output.gif'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
