import argparse
import os
import sys
import subprocess
import shutil
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import time
import asyncio
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import random

import pandas as pd
import pyproj
from shapely.geometry import Point, LineString
from tqdm import tqdm
from shapely.ops import nearest_points
import json

# Pure-stdlib sign-orientation helper (lives beside this script). Ensure the
# script's own directory is importable whether run as a script or imported.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sign_edge_orientation import orient_sign_edge

# Overpass endpoints, tried in order. Override at runtime without editing code:
#   OVERPASS_URLS="https://overpass.openstreetmap.fr/api/interpreter,..." python ...
_DEFAULT_OVERPASS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OVERPASS_URLS = [u.strip() for u in os.environ.get("OVERPASS_URLS", "").split(",")
                 if u.strip()] or _DEFAULT_OVERPASS

# Per-sign-code highway-type filter for road matching. A residential/living zone
# (5.21 residential zone / 5.22 end of residential zone) is NEVER on a through-road, so
# restrict matching to courtyard/residential ways and PREFER an actual
# `living_street` when one is reasonably close. Codes not listed here keep the
# legacy behavior: nearest car-accessible way regardless of highway type.
THROUGH_ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary"}
RESIDENTIAL_WAYS = {"living_street", "service", "residential", "unclassified", "road"}
SIGN_HIGHWAY_FILTER = {
    "5.21": {"allowed": RESIDENTIAL_WAYS, "prefer": {"living_street"}},
    "5.22": {"allowed": RESIDENTIAL_WAYS, "prefer": {"living_street"}},
}


def _find_netconvert() -> str:
    """Resolve netconvert when it is not on PATH (e.g. pip install --user sumo-tools)."""
    candidates = [
        shutil.which("netconvert"),
        Path.home() / ".local" / "bin" / "netconvert",
        Path("/usr/local/bin/netconvert"),
        Path("/usr/bin/netconvert"),
    ]
    for path in candidates:
        if path and Path(path).exists():
            return str(path)
    raise FileNotFoundError(
        "netconvert not found. Install SUMO or add netconvert to PATH.\n"
        "Try: pip install sumo-tools  OR  export PATH=$PATH:$HOME/.local/bin"
    )

class AsyncOSMDownloader:
    def __init__(self, max_concurrent=5):
        self.max_concurrent = max_concurrent
        self.session = None
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        self.session = aiohttp.ClientSession(connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def download_osm_fragment(self, lat, lon, output_path, delta=0.002):
        """Asynchronous download of an OSM fragment"""
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
        
        await asyncio.sleep(random.uniform(0.5, 2.0))  # Jitter
        
        for url in OVERPASS_URLS:
            try:
                async with self.session.post(url, data={"data": query}, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Don't accept a 200 with no map data (regional/empty mirror) —
                        # it would write an empty fragment that netconvert rejects.
                        if "<node" not in content and "<way" not in content:
                            print(f"Empty data from {url}, trying next endpoint")
                            continue
                        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
                            await f.write(content)
                        return True
            except Exception as e:
                print(f"Failed on {url}: {e}")
                continue
        
        raise RuntimeError(f"All Overpass endpoints failed for ({lat}, {lon})")

class BatchSignProcessor:
    def __init__(self, csv_path, scenes_dir, max_concurrent_downloads=5, max_concurrent_netconvert=2,
                 dedup_osm_way=False, geo_spread=False, geo_spread_seed=42):
        self.csv_path = csv_path
        self.scenes_dir = Path(scenes_dir)
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_concurrent_netconvert = max_concurrent_netconvert
        self.df = None
        self.existing_ids = set()
        self.existing_counts = {}
        # Geometry-diversity options (opt-in; off = legacy behavior).
        self.dedup_osm_way = dedup_osm_way      # ≤1 scene per unique road (osm_way_id) per code
        self.geo_spread = geo_spread            # round-robin candidate order across districts
        self.geo_spread_seed = geo_spread_seed
        self.seen_ways = {}                     # {sign_type: set(normalized osm_way_id)}

    @staticmethod
    def _norm_way(way_id):
        """Normalize an OSM/SUMO way id to a road key: drop '#<seg>' and a leading '-'."""
        s = str(way_id).split("#", 1)[0]
        return s[1:] if s.startswith("-") else s
        
    def load_data(self):
        """Load the CSV with sign data"""
        
        self.df = pd.read_csv(self.csv_path, sep=";", low_memory=False)
        self.df["ID"] = self.df["ID"].astype(int)
        
    def filter_signs(self, sign_types, spread=False):
        """Filter signs by one or more types.

        spread=False: deterministic ID order (legacy).
        spread=True:  round-robin across District (fallback AdmArea) so candidates
                      are pulled from all over the city first — finds DIFFERENT roads
                      faster and avoids clustering many scenes on one street.
        """
        if isinstance(sign_types, str):
            sign_types = [sign_types]

        pattern = "|".join([f"^{st}" for st in sign_types])
        filtered = self.df[self.df["SignType"].str.contains(pattern, na=False, regex=True)]
        filtered = filtered.sort_values(by="ID", ascending=True)

        if not spread:
            return filtered

        group_col = "District" if "District" in filtered.columns else (
            "AdmArea" if "AdmArea" in filtered.columns else None)
        if group_col is None:
            return filtered

        groups = {}
        for _, row in filtered.iterrows():
            g = str(row.get(group_col) or "")
            groups.setdefault(g, []).append(row)
        order = list(groups.keys())
        random.Random(self.geo_spread_seed).shuffle(order)

        out_rows, i = [], 0
        while True:
            added = False
            for g in order:
                if i < len(groups[g]):
                    out_rows.append(groups[g][i])
                    added = True
            if not added:
                break
            i += 1
        return pd.DataFrame(out_rows)
    
    def collect_existing_counts(self):
        """
        Returns a dict: {sign_type: number of already existing scenes}
        and a set of existing sign_ids (to skip duplicates).
        """
        existing_counts = {}
        existing_ids = set()
        
        for sign_type_dir in self.scenes_dir.iterdir():
            if not sign_type_dir.is_dir():
                continue
            sign_type = sign_type_dir.name
            count = 0
            for scene_dir in sign_type_dir.iterdir():
                if scene_dir.is_dir():
                    meta_file = scene_dir / "meta.json"
                    if meta_file.exists():
                        try:
                            with open(meta_file, "r") as f:
                                meta = json.load(f)
                                sign_id = meta.get("sign_id")
                                if sign_id:
                                    existing_ids.add(sign_id)
                                    count += 1
                                # Seed per-road dedup set so a top-up run never re-uses
                                # a road already collected for this code.
                                way = meta.get("osm_way_id")
                                if way is not None:
                                    self.seen_ways.setdefault(sign_type, set()).add(self._norm_way(way))
                        except (json.JSONDecodeError, KeyError):
                            continue
            if count > 0:
                existing_counts[sign_type] = count
        
        return existing_counts, existing_ids
    
    def collect_existing_signs(self):
        """Collect already existing signs"""
        for sign_type_dir in self.scenes_dir.iterdir():
            if sign_type_dir.is_dir():
                for scene_dir in sign_type_dir.iterdir():
                    if scene_dir.is_dir():
                        meta_file = scene_dir / "meta.json"
                        if meta_file.exists():
                            try:
                                with open(meta_file, "r") as f:
                                    meta = json.load(f)
                                    self.existing_ids.add(meta.get("sign_id"))
                            except (json.JSONDecodeError, KeyError):
                                continue
    
    async def process_sign(self, sign_row, sign_type):
        """Asynchronous processing of a single sign"""
        sign_id = sign_row["ID"]
        lat = float(sign_row["Latitude_WGS84"])
        lon = float(sign_row["Longitude_WGS84"])
        
        if sign_id in self.existing_ids:
            print(f" Skipping existing sign: {sign_id} of sign type: {sign_type}")
            return None
        
        base_name = f"sign_{sign_id}"
        osm_raw = Path("osm_fragments") / f"{base_name}.osm"
        net_file = f"{base_name}.net.xml"
        scene_dir = self.scenes_dir / sign_type / base_name
        
        if scene_dir.exists():
            print(f" Scene already exists: {scene_dir}")
            return None
        
        try:
            if osm_raw.exists() and osm_raw.stat().st_size > 0:
                print(f" OSM file already exists: {osm_raw}, skipping download.")
            else:
                async with AsyncOSMDownloader(max_concurrent=self.max_concurrent_downloads) as downloader:
                    await downloader.download_osm_fragment(lat, lon, osm_raw)
            
            loop = asyncio.get_event_loop()
            net_output_path = Path("maps") / net_file
            
            await loop.run_in_executor(
                None,
                self.convert_osm_to_sumo,
                osm_raw, net_output_path
            )
            
            _hwf = SIGN_HIGHWAY_FILTER.get(sign_type, {})
            way_id, distance_from_start = await loop.run_in_executor(
                None,
                self.find_closest_way_and_distance,
                osm_raw, lat, lon, _hwf.get("allowed"), _hwf.get("prefer")
            )

            # Per-road dedup: ≤1 scene per unique road (osm_way_id) per code, so the
            # sign lands on a DIFFERENT road each time. Check-and-add here with NO
            # await in between => atomic under cooperative asyncio (no lock needed).
            if self.dedup_osm_way:
                norm_way = self._norm_way(way_id)
                seen = self.seen_ways.setdefault(sign_type, set())
                if norm_way in seen:
                    print(f" Skipping dup road {norm_way} for {sign_type} (sign {sign_id})")
                    return {"sign_id": sign_id, "sign_type": sign_type, "status": "skipped_dup_way"}
                seen.add(norm_way)

            road_id, s_offset = await loop.run_in_executor(
                None,
                self.find_edge_and_offset_in_sumo,
                net_output_path, way_id, distance_from_start
            )

            # Zone-entry signs (5.21): orient onto the directed edge whose
            # upstream side reaches the big road, so the braking-spawn route is
            # big-road -> sign -> courtyard (no-op for other codes).
            road_id, s_offset, _orient = await loop.run_in_executor(
                None,
                orient_sign_edge,
                str(net_output_path), road_id, s_offset, sign_type
            )

            scene_dir.mkdir(parents=True, exist_ok=True)
            (scene_dir / net_file).write_bytes(net_output_path.read_bytes())
            
            meta = {
                "sign_id": sign_id,
                "sign_type": sign_type,
                "latitude": lat,
                "longitude": lon,
                "osm_way_id": way_id,
                "road_id": road_id,
                "distance_from_start": s_offset,
                "net_file": net_file
            }
            
            async with aiofiles.open(scene_dir / "meta.json", "w") as f:
                await f.write(json.dumps(meta, indent=2, ensure_ascii=False))
            
            print(f" Processed sign {sign_id} ({sign_type})")
            return {"sign_id": sign_id, "sign_type": sign_type, "status": "success"}
            
        except Exception as e:
            print(f" Error processing sign {sign_id} / {sign_type}: {e}")
            return {"sign_id": sign_id, "sign_type": sign_type, "status": "failed", "error": str(e)}
    
    @staticmethod
    def convert_osm_to_sumo(osm_path, net_output_path):
        netconvert = _find_netconvert()
        subprocess.run([
            netconvert,
            "--osm-files", str(osm_path),
            "-o", str(net_output_path),
            "--osm.sidewalks",
            "--osm.crossings",
            "--crossings.guess",
            "--walkingareas"
        ], check=True, capture_output=True)
        
    @staticmethod
    def is_car_accessible(way):

        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
        highway = tags.get("highway")
        if highway is None:
            return False

        no_car_highways = {
            "footway", "steps", "bridleway", "cycleway", "path", "pedestrian",
            "track", "corridor", "elevator", "escalator", "bus_guideway"
        }
        if highway in no_car_highways:
            return False

        access = tags.get("access")
        if access == "no":
            return False
        motorcar = tags.get("motorcar")
        if motorcar == "no":
            return False
        if motorcar == "yes":
            return True
        vehicle = tags.get("vehicle")
        if vehicle == "no":
            return False
        if vehicle == "yes":
            return True

        if highway == "service":
            return True

        car_highways = {
            "motorway", "trunk", "primary", "secondary", "tertiary",
            "unclassified", "residential", "living_street", "road"
        }
        if highway in car_highways:
            return True

        return False
    
    @staticmethod
    def get_road_width(way):

        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
        highway = tags.get("highway", "")

        if "width" in tags:
            try:
                return float(tags["width"])
            except ValueError:
                pass

        if "width:lanes" in tags:
            widths_str = tags["width:lanes"]
            import re
            widths = re.split(r'[|; ]+', widths_str)
            total = 0.0
            for w in widths:
                try:
                    total += float(w)
                except ValueError:
                    pass
            if total > 0:
                return total

        if "lanes" in tags:
            try:
                lanes = int(tags["lanes"])
            except ValueError:
                lanes = 0
            if lanes > 0:
                lane_width_defaults = {
                    "motorway": 3.75,
                    "trunk": 3.75,
                    "primary": 3.5,
                    "secondary": 3.5,
                    "tertiary": 3.25,
                    "unclassified": 3.0,
                    "residential": 3.0,
                    "service": 3.0,
                    "secondary_link": 3.5,
                    "trunk_link": 3.5,
                    "primary_link": 3.5,
                }
                lane_width = lane_width_defaults.get(highway, 3.0)
                return lanes * lane_width
            else:
                pass

        default_widths = {
            "motorway": 11.25,  
            "trunk": 10.5,     
            "primary": 7.0,      
            "secondary": 7.0,
            "tertiary": 6.0,
            "unclassified": 5.0,
            "residential": 5.0,
            "service": 4.0,
            "secondary_link": 3.5,
            "trunk_link": 3.5,
            "primary_link": 3.5,
            "living_street": 4.0,
        }
        return default_widths.get(highway, 5.0)
    
    @staticmethod
    def find_closest_way_and_distance(osm_path, sign_lat, sign_lon,
                                      allowed_highways=None, prefer_highways=None,
                                      prefer_margin_m=20.0):
        """Find the OSM way nearest to the sign and the normalized offset along it.

        `allowed_highways` (set of highway= values): if given, only ways of these
        types are considered — used to keep zone signs (5.21/5.22) off through-
        roads. `prefer_highways`: if a way of one of these types lies within
        `prefer_margin_m` of the nearest allowed way, prefer it (so an actual
        `living_street` wins over an adjacent service road). None = legacy
        behavior (nearest car-accessible way of any type)."""
        tree = ET.parse(osm_path)
        root = tree.getroot()
        nodes = {n.get("id"): (float(n.get("lat")), float(n.get("lon")))
                for n in root.findall("node")}

        proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32637", always_xy=True)
        sign_x, sign_y = proj.transform(sign_lon, sign_lat)
        sign_point = Point(sign_x, sign_y)

        def _frac_along(line):
            nearest_on_road = nearest_points(line, sign_point)[0]
            dist_along = 0.0
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                segment = LineString([coords[i], coords[i + 1]])
                if segment.distance(nearest_on_road) < 1e-6:
                    dist_along += Point(coords[i]).distance(nearest_on_road)
                    break
                dist_along += segment.length
            return dist_along / line.length if line.length > 0 else 0.0

        # (dist, way_id, frac) for the nearest allowed way and the nearest preferred way.
        best_any = (float("inf"), None, None)
        best_pref = (float("inf"), None, None)

        for way in root.findall("way"):
            tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
            highway = tags.get("highway")
            if highway is None:
                continue
            if not BatchSignProcessor.is_car_accessible(way):
                continue
            if allowed_highways is not None and highway not in allowed_highways:
                continue

            blocks = []
            current_block = []
            for nd in way.findall("nd"):
                nid = nd.get("ref")
                if nid in nodes:
                    lat, lon = nodes[nid]
                    current_block.append(proj.transform(lon, lat))
                else:
                    if len(current_block) >= 2:
                        blocks.append(current_block)
                    current_block = []
            if len(current_block) >= 2:
                blocks.append(current_block)

            width = BatchSignProcessor.get_road_width(way)
            for block_coords in blocks:
                line = LineString(block_coords)
                dist = sign_point.distance(line.buffer(width / 2))
                if dist < best_any[0]:
                    best_any = (dist, way.get("id"), _frac_along(line))
                if prefer_highways and highway in prefer_highways and dist < best_pref[0]:
                    best_pref = (dist, way.get("id"), _frac_along(line))

        if best_any[1] is None:
            extra = (f" within allowed highways {sorted(allowed_highways)}"
                     if allowed_highways else "")
            raise ValueError(f"No road found near ({sign_lat}, {sign_lon}){extra}")
        # Prefer an actual living_street when it is about as close as the nearest allowed way.
        if (prefer_highways and best_pref[1] is not None
                and best_pref[0] <= best_any[0] + prefer_margin_m):
            return best_pref[1], best_pref[2]
        return best_any[1], best_any[2]
        
    @staticmethod
    def find_edge_and_offset_in_sumo(net_path, way_id, target_distance):

        tree = ET.parse(net_path)
        root = tree.getroot()
        
        def get_edge_length(edge):
            length = edge.get("length")
            if length is not None:
                return float(length)
            
            lane = edge.find("lane")
            if lane is not None:
                shape_str = lane.get("shape")
                if shape_str:
                    points = shape_str.strip().split()
                    coords = [tuple(map(float, p.split(','))) for p in points]
                    if len(coords) >= 2:
                        return sum(
                            ((coords[i+1][0] - coords[i][0])**2 + 
                             (coords[i+1][1] - coords[i][1])**2)**0.5
                            for i in range(len(coords)-1)
                        )
            return 0.0
        
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
        total_length = sum(length for _, _, length in edges)
        
        cumulative = 0.0
        for _, edge_id, length in edges:
            if (cumulative + length) / total_length >= target_distance:
                s_offset = target_distance * total_length - cumulative
                return edge_id, s_offset
            cumulative += length
        
        raise ValueError(f"Distance {target_distance} exceeds total length ({total_length})")
    
    async def process_batch(self, sign_types, max_signs_per_type=250):
        self.load_data()
        self.existing_counts, self.existing_ids = self.collect_existing_counts()
    
        targets = {}
        remaining_to_collect = {}
        base_existing = {}
        for st in sign_types:
            # Under --dedup-osm-way the target counts UNIQUE ROADS, not scenes
            # (a code may already have more scenes than roads). So "existing" =
            # distinct roads already collected; we top up to max_signs_per_type roads.
            if self.dedup_osm_way:
                existing = len(self.seen_ways.get(st, set()))
                unit = "roads"
            else:
                existing = self.existing_counts.get(st, 0)
                unit = "scenes"
            base_existing[st] = existing
            needed = max_signs_per_type - existing
            if needed <= 0:
                print(f"Type {st}: already have {existing} {unit} (limit {max_signs_per_type}), no new ones needed.")
            else:
                print(f"Type {st}: already have {existing} {unit}, need to create {needed} more (total {max_signs_per_type})")
            targets[st] = max_signs_per_type
            remaining_to_collect[st] = needed
    
        success_count = {st: 0 for st in sign_types}
        iterators = {}
        for sign_type in sign_types:
            filtered = self.filter_signs(sign_type, spread=self.geo_spread)
            iterators[sign_type] = filtered.iterrows()

        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)

        async def run_with_semaphore(sign_row, sign_type):
            async with semaphore:
                return await self.process_sign(sign_row, sign_type)

        tasks = []
        for sign_type in sign_types:
            if remaining_to_collect[sign_type] < 0:
                continue
            it = iterators[sign_type]
            for _ in range(self.max_concurrent_downloads):
                try:
                    idx, row = next(it)
                    task = asyncio.create_task(run_with_semaphore(row, sign_type))
                    tasks.append((task, sign_type, row))
                except StopIteration:
                    break

        results = []
        while tasks and any(success_count[st] < remaining_to_collect[st] for st in sign_types):
            done, pending = await asyncio.wait(
                [task for task, _, _ in tasks],
                return_when=asyncio.FIRST_COMPLETED
            )
            new_tasks = []
            for task, st, row in tasks:
                if task in done:
                    result = await task  
                    results.append(result)
                    if result and result.get("status") == "success":
                        success_count[st] += 1
                        collected = base_existing[st] + success_count[st]
                        target = targets[st]
                        print(f" {st}: progress {collected}/{target} ({target - collected} remaining)")
                    if success_count[st] < remaining_to_collect[st]:
                        try:
                            next_idx, next_row = next(iterators[st])
                            new_task = asyncio.create_task(run_with_semaphore(next_row, st))
                            new_tasks.append((new_task, st, next_row))
                        except StopIteration:
                            pass  
                else:
                    new_tasks.append((task, st, row))
            tasks = new_tasks

        successful = [r for r in results if r and r.get("status") == "success"]
        failed = [r for r in results if r and r.get("status") == "failed"]
        dup_roads = [r for r in results if r and r.get("status") == "skipped_dup_way"]

        print("\n" + "="*50)
        for st in sign_types:
            print(f" {st}: {success_count[st]} / {remaining_to_collect[st]} successful")
        print(f" Total successful: {len(successful)}, failed: {len(failed)}, "
              f"skipped dup-road: {len(dup_roads)}")

        if failed:
            print("\nFailed signs:")
            for f in failed:
                print(f"  - {f['sign_type']}:{f['sign_id']} - {f.get('error', 'Unknown error')}")

        return results

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV with signs")
    parser.add_argument("--sign-types", nargs="+", required=True, 
                       help='Sign types, e.g. "2.5" "3.27" "4.1.1"')
    parser.add_argument("--scenes_dir", default="scenes", help='Path to scenes dir')
    parser.add_argument("--max-concurrent", type=int, default=5, 
                       help="Maximum concurrent downloads")
    parser.add_argument("--max-signs-per-type", type=int, default=250,
                       help="Maximum signs per type")
    parser.add_argument("--dedup-osm-way", action="store_true",
                       help="Collect at most ONE scene per unique road (osm_way_id) per "
                            "code, so the sign lands on a different road each time.")
    parser.add_argument("--geo-spread", action="store_true",
                       help="Order candidates round-robin across districts (find diverse "
                            "roads faster, less clustering).")
    parser.add_argument("--geo-spread-seed", type=int, default=42,
                       help="Seed for --geo-spread district ordering.")

    args = parser.parse_args()

    Path("osm_fragments").mkdir(exist_ok=True)
    Path("maps").mkdir(exist_ok=True)
    Path(args.scenes_dir).mkdir(exist_ok=True)

    processor = BatchSignProcessor(
        args.csv,
        args.scenes_dir,
        max_concurrent_downloads=args.max_concurrent,
        dedup_osm_way=args.dedup_osm_way,
        geo_spread=args.geo_spread,
        geo_spread_seed=args.geo_spread_seed,
    )
    
    results = await processor.process_batch(
        args.sign_types,
        max_signs_per_type=args.max_signs_per_type
    )

if __name__ == "__main__":
    asyncio.run(main())