import argparse
import os
import sys
import subprocess
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import time

import pandas as pd
import pyproj
from shapely.geometry import Point, LineString
from tqdm import tqdm
from shapely.ops import nearest_points

import json


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DELTA = 0.002
OSM_FRAG_DIR = Path("osm_fragments")
MAPS_DIR = Path("./maps")
METADRIVE_ASSETS_DIR = Path("../metadrive/metadrive/assets/carla")

import requests
import time
import random
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

def create_overpass_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,  # 2s, 4s, 8s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
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
            response = session.post(
                url,
                data={"data": query},
                timeout=30
            )
            response.raise_for_status()
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print("OSM fragment downloaded successfully.")
            return
        except Exception as e:
            print(f"Failed on {url}: {e}")
            last_exception = e
            time.sleep(5)

    raise RuntimeError(f"All Overpass endpoints failed. Last error: {last_exception}")

def find_closest_way_and_distance(osm_path, sign_lat, sign_lon):
    """
    Returns (way_id, distance_from_start) for the nearest road in the OSM file.
    """
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
                segment = LineString([line_coords[i], line_coords[i+1]])
                if segment.distance(nearest_on_road) < 1e-6:
                    dist_along += Point(line_coords[i]).distance(nearest_on_road)
                    break
                else:
                    dist_along += segment.length
            distance_from_start = dist_along / line.length

    if closest_way_id is None:
        raise ValueError(f"No road found near ({sign_lat}, {sign_lon})")
    return closest_way_id, distance_from_start

def run_metadrive(road_id, sign_type, map_name):
    cmd = [
        sys.executable,
        "scripts/vis_env/vis_sumo_map_traffic_sign.py",
        "--road-id", str(road_id),
        "--sign-type", sign_type,
        "--map-name", map_name,
    ]
    print(f"▶️ Running MetaDrive: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
def collect_existing_sign_ids(scenes_dir):
    scenes_dir =Path(scenes_dir)
    existing_ids = set()
    existing_sign_ids = set()
    for sign_type_dir in scenes_dir.iterdir():
        if sign_type_dir.is_dir():
            for scene_dir in sign_type_dir.iterdir():
                if scene_dir.is_dir():
                    meta_file = scene_dir / "meta.json"
                    if meta_file.exists():
                        try:
                            with open(meta_file, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                                existing_sign_ids.add((str(meta.get("sign_id"))))
                                existing_ids.add((str(meta.get("sign_type")), str(meta.get("road_id"))))
                        except (json.JSONDecodeError, KeyError):
                            continue
    return existing_ids, existing_sign_ids

def get_edge_length(edge):
    """Returns the edge length, computing it from shape (own or lane) if needed."""
    length = edge.get("length")
    if length is not None:
        return float(length)
    
    # shape_str = edge.get("shape")
    # if not shape_str:
    shape_str = None
    lane = edge.find("lane")
    if lane is not None:
        shape_str = lane.get("shape")
    if shape_str:
        points = shape_str.strip().split()
        coords = [tuple(map(float, p.split(','))) for p in points]
        if len(coords) >= 2:
            return sum(
                ((coords[i+1][0] - coords[i][0])**2 + (coords[i+1][1] - coords[i][1])**2)**0.5
                for i in range(len(coords)-1)
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
        if (cumulative + length) / cumulative_sum  >= target_distance:
            s_offset = target_distance * cumulative_sum - cumulative
            return edge_id, s_offset
        cumulative += length

    raise ValueError(f"Distance {target_distance} exceeds total length ({cumulative}) of way {way_id}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV with signs")
    parser.add_argument("--sign-type", required=True, help='Sign type regex, e.g. "2.5"')
    parser.add_argument("--scenes_dir", default="scenes", help='Path to scenes dir')
    args = parser.parse_args()

    OSM_FRAG_DIR.mkdir(exist_ok=True)
    MAPS_DIR.mkdir(exist_ok=True)

    SCENES_DIR = Path(args.scenes_dir)
    SCENES_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(args.csv, sep=";")
    pattern = f"^{args.sign_type}"
    filtered = df[df["SignType"].str.contains(pattern, na=False, regex=True)]

    if filtered.empty:
        print(f"No signs of type '{args.sign_type}' found in {args.csv}")
        return

    print(f"Found {len(filtered)} signs of type '{args.sign_type}'")

    sign_type_dir = SCENES_DIR / args.sign_type
    sign_type_dir.mkdir(parents=True, exist_ok=True)
    existing_test_ids, existing_test_sign_ids  = collect_existing_sign_ids("./scenes")
    train_existing_ids, train_existing_sign_ids  = collect_existing_sign_ids("./train_scenes")
    filtered["ID"] = filtered["ID"].astype(int)
    filtered = filtered.sort_values(by="ID", ascending=False)
    counter = 1
    for idx, row in tqdm(filtered.iterrows()):
        if counter >= 250:
            break
        sign_id = row["ID"]
        # if sign_id != 1531530:
        #     continue
        # print("sign_id: ", sign_id)
        if sign_id in existing_test_sign_ids:
            print(f"\n--- Skipping existing sign id in test: {sign_id} ---")
            continue 
        if sign_id in train_existing_sign_ids:
            print(f"\n--- Skipping existing sign id in train: {sign_id} ---")
            continue 
        lat = float(row["Latitude_WGS84"])
        lon = float(row["Longitude_WGS84"])
        # lat = 55.708480
        # lon = 37.542097

        print(f"\n--- Processing sign ID: {sign_id} at ({lat:.6f}, {lon:.6f}) --- Count: {counter}/250")

        base_name = f"sign_{sign_id}"
        osm_raw = OSM_FRAG_DIR / f"{base_name}.osm"
        net_file = f"{base_name}.net.xml"
        scene_dir = sign_type_dir / base_name
        counter += 1   
        if scene_dir.exists():
            print(f"\n--- Skipping existing scene: {scene_dir} ---")
            continue

        try:
            print("Downloading OSM fragment...")
            download_osm_fragment(lat, lon, osm_raw)

            print("Finding closest way and distance along it...")
            way_id, distance_from_start = find_closest_way_and_distance(osm_raw, lat, lon)
            print("way_id, distance_from_start: ", way_id, distance_from_start)

            net_output_path = MAPS_DIR / net_file
            print("Converting OSM directly to SUMO .net.xml with crossings...")
            subprocess.run([
                "netconvert",
                "--osm-files", str(osm_raw),
                "-o", str(net_output_path),
                "--osm.sidewalks",
                "--osm.crossings",
                "--crossings.guess",
                "--walkingareas"
            ], check=True)

            print("Finding edge and offset in SUMO network...")
            road_id, s_offset = find_edge_and_offset_in_sumo_by_way_id(
                net_output_path, way_id, distance_from_start
            )
            
            print("road_id, s_offset:   ", road_id, s_offset)

            scene_dir.mkdir(exist_ok=True)
            (scene_dir / net_file).write_bytes(net_output_path.read_bytes())

            meta = {
                "sign_id": sign_id,
                "sign_type": args.sign_type,
                "latitude": lat,
                "longitude": lon,
                "osm_way_id": way_id,
                "road_id": road_id,         # SUMO edge id
                "distance_from_start": s_offset,
                "net_file": net_file
            }
            with open(scene_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            print(f"Saved benchmark scene to {scene_dir}")   

        except Exception as e:
            print(f"Error processing sign {sign_id}: {e}")
            continue


if __name__ == "__main__":
    main()