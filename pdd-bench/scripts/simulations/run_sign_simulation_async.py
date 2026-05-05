import argparse
import os
import sys
import subprocess
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

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

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
        """Асинхронная загрузка OSM фрагмента"""
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
                        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
                            await f.write(content)
                        return True
            except Exception as e:
                print(f"Failed on {url}: {e}")
                continue
        
        raise RuntimeError(f"All Overpass endpoints failed for ({lat}, {lon})")

class BatchSignProcessor:
    def __init__(self, csv_path, scenes_dir, max_concurrent_downloads=5, max_concurrent_netconvert=2):
        self.csv_path = csv_path
        self.scenes_dir = Path(scenes_dir)
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_concurrent_netconvert = max_concurrent_netconvert
        self.df = None
        self.existing_ids = set()
        self.existing_counts = {} 
        
    def load_data(self):
        """Загрузка CSV с данными знаков"""
        
        self.df = pd.read_csv(self.csv_path, sep=";", low_memory=False)
        self.df["ID"] = self.df["ID"].astype(int)
        
    def filter_signs(self, sign_types):
        """Фильтрация знаков по нескольким типам"""
        if isinstance(sign_types, str):
            sign_types = [sign_types]
        
        pattern = "|".join([f"^{st}" for st in sign_types])
        filtered = self.df[self.df["SignType"].str.contains(pattern, na=False, regex=True)]
        
        # Сортируем по ID и ограничиваем
        filtered = filtered.sort_values(by="ID", ascending=True)
        
        return filtered
    
    def collect_existing_counts(self):
        """
        Возвращает словарь: {sign_type: количество уже существующих сцен}
        и множество существующих sign_id (для пропуска дубликатов).
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
                        except (json.JSONDecodeError, KeyError):
                            continue
            if count > 0:
                existing_counts[sign_type] = count
        
        return existing_counts, existing_ids
    
    def collect_existing_signs(self):
        """Сбор уже существующих знаков"""
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
        """Асинхронная обработка одного знака"""
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
            
            way_id, distance_from_start = await loop.run_in_executor(
                None,
                self.find_closest_way_and_distance,
                osm_raw, lat, lon
            )
            
            road_id, s_offset = await loop.run_in_executor(
                None,
                self.find_edge_and_offset_in_sumo,
                net_output_path, way_id, distance_from_start
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
        subprocess.run([
            "netconvert",
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
    def find_closest_way_and_distance(osm_path, sign_lat, sign_lon):

        tree = ET.parse(osm_path)
        root = tree.getroot()
        nodes = {n.get("id"): (float(n.get("lat")), float(n.get("lon"))) 
                for n in root.findall("node")}
        
        proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32637", always_xy=True)
        sign_x, sign_y = proj.transform(sign_lon, sign_lat)
        sign_point = Point(sign_x, sign_y)
        
        min_dist = float("inf")
        closest_way_id = None
        distance_from_start = None
        
        for way in root.findall("way"):
            if not any(tag.get("k") == "highway" for tag in way.findall("tag")):
                continue
            if not BatchSignProcessor.is_car_accessible(way):
                continue
            
            blocks = []   
            current_block = []
            for nd in way.findall("nd"):
                nid = nd.get("ref")
                if nid in nodes:
                    lat, lon = nodes[nid]
                    x, y = proj.transform(lon, lat)
                    current_block.append((x, y))
                else:
                    if len(current_block) >= 2:
                        blocks.append(current_block)
                    current_block = []
            if len(current_block) >= 2:
                blocks.append(current_block)

            for block_coords in blocks:
                line = LineString(block_coords)
                width = BatchSignProcessor.get_road_width(way)
                polygon = line.buffer(width / 2)
                dist = sign_point.distance(polygon)
                if dist < min_dist:
                    min_dist = dist
                    closest_way_id = way.get("id")
                    nearest_on_road = nearest_points(line, sign_point)[0]
                    dist_along = 0.0
                    block_coords_list = list(line.coords)
                    for i in range(len(block_coords_list) - 1):
                        segment = LineString([block_coords_list[i], block_coords_list[i+1]])
                        if segment.distance(nearest_on_road) < 1e-6:
                            dist_along += Point(block_coords_list[i]).distance(nearest_on_road)
                            break
                        else:
                            dist_along += segment.length
                   distance_from_start = dist_along / line.length

        if closest_way_id is None:
            raise ValueError(f"No road found near ({sign_lat}, {sign_lon})")
        return closest_way_id, distance_from_start
        
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
        for st in sign_types:
            existing = self.existing_counts.get(st, 0)
            needed = max_signs_per_type - existing
            if needed <= 0:
                print(f"Тип {st}: уже есть {existing} сцен (лимит {max_signs_per_type}), новых не требуется.")
            else:
                print(f"Тип {st}: уже есть {existing}, нужно создать ещё {needed} (всего {max_signs_per_type})")
            targets[st] = max_signs_per_type
            remaining_to_collect[st] = needed
    
        success_count = {st: 0 for st in sign_types}
        iterators = {}
        for sign_type in sign_types:
            filtered = self.filter_signs(sign_type)
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
                    tasks.append((task, sign_type, row))  # храним задачу, тип и строку
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
                        collected = self.existing_counts.get(st, 0) + success_count[st]
                        target = targets[st]
                        print(f" {st}: прогресс {collected}/{target} (осталось {target - collected})")
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

        print("\n" + "="*50)
        for st in sign_types:
            print(f" {st}: {success_count[st]} / {remaining_to_collect[st]} successful")
        print(f" Total successful: {len(successful)}, failed: {len(failed)}")

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
    
    args = parser.parse_args()
    
    Path("osm_fragments").mkdir(exist_ok=True)
    Path("maps").mkdir(exist_ok=True)
    Path(args.scenes_dir).mkdir(exist_ok=True)
    
    processor = BatchSignProcessor(
        args.csv, 
        args.scenes_dir,
        max_concurrent_downloads=args.max_concurrent
    )
    
    results = await processor.process_batch(
        args.sign_types,
        max_signs_per_type=args.max_signs_per_type
    )

if __name__ == "__main__":
    asyncio.run(main())