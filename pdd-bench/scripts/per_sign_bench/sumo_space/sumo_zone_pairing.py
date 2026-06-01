"""Combine real SUMO start/end zone scenes into single paired scenes.

Background
----------
Per-sign SUMO scenes (`scenes/{3.24,3.25,5.31,5.32}/sign_*/`) are independent OSM
cut-outs: a 3.24 (speed-limit start) and its matching 3.25 (end) live in *separate*
scenes. A standalone end sign tests nothing — it only truncates a *preceding* start
sign that isn't present. This script finds real start/end pairs on the same road and
emits ONE combined scene per pair, so the env (see `_place_paired_end_sign` /
`build_zones` in envs/sumo_env.py) places both signs on one edge and the verifier can
measure both in-zone compliance and end-of-zone recognition.

Method (matches the approved plan, section 3)
---------------------------------------------
1. Group start (3.24/5.31) and end (3.25/5.32) scenes by normalized `osm_way_id`.
2. For each end, pick the nearest start by GEODESIC distance of the sign lat/lon
   (<= --max-dist m), same way preferred. `distance_from_start` from the two metas is
   NOT comparable across extractions, so it is never subtracted directly.
3. Re-project BOTH signs onto the START scene's net.xml using its <location>
   (netOffset + projParameter) and snap each to the nearest lane → (edge_id, s_offset)
   in ONE consistent frame.
4. Require both signs to snap to the SAME edge (so the manager's same-lane
   get_signs_before / build_zones truncation works) and s_end > s_start with room for
   approach + zone + tail. Otherwise drop the pair (logged).
5. Write a combined scene dir (augmented meta.json + symlink to the start net) and a
   manifest row. `sign_code` = start code so the existing single-sign placement path
   places the start; meta carries `sign_type_end`/`road_id_end`/`s_end`/`zone_length_m`
   for `_place_paired_end_sign`.

Run with an interpreter that has pyproj + shapely (e.g. the `ym2` env):
    ~/miniconda3/envs/ym2/bin/python sumo_zone_pairing.py \
        --scenes-root <.../pdd-bench/scenes> --out-root <.../scenes_paired> \
        --manifest <.../paired_manifest.jsonl>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pyproj
from shapely.geometry import LineString, Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sumo_scene_enumerator import enumerate_all_scenes, SumoScene  # noqa: E402

# Which end code terminates which start code (scope: speed + zone limits).
START_END_CODES = {
    "3.24": "3.25",   # speed limit  -> end of speed limit
    "5.31": "5.32",   # zone limit   -> end of zone limit
}

# Geometry margins (meters). Mirror factorized_space.sign_placement defaults.
APPROACH_MIN = 5.0     # min distance from edge start to the start sign
TAIL_MIN = 8.0         # min drivable length after the end sign
ZONE_MIN = 3.0         # reject degenerate zones


def _norm_way(way_id: str) -> str:
    return str(way_id).split("#")[0]


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Approximate great-circle distance in meters (fine for <1 km)."""
    dlat = (lat2 - lat1) * 111_000.0
    dlon = (lon2 - lon1) * 111_000.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.hypot(dlat, dlon)


def _parse_location(net_path: str):
    """Return (netOffsetX, netOffsetY, proj4) from the net's <location>, or None."""
    try:
        for _evt, elem in ET.iterparse(net_path, events=("end",)):
            if elem.tag == "location":
                off = (elem.get("netOffset") or "0,0").split(",")
                proj = elem.get("projParameter") or ""
                return float(off[0]), float(off[1]), proj
    except (ET.ParseError, OSError):
        return None
    return None


def _lane_lines(net_path: str):
    """Parse driving-lane shapes. Return list of (edge_id, lane_id, LineString, length)."""
    out = []
    try:
        tree = ET.parse(net_path)
    except (ET.ParseError, OSError):
        return out
    for edge in tree.getroot().findall("edge"):
        eid = edge.get("id")
        if not eid or eid.startswith(":") or edge.get("function") == "internal":
            continue
        for lane in edge.findall("lane"):
            shape = lane.get("shape")
            if not shape:
                continue
            try:
                pts = [tuple(map(float, p.split(","))) for p in shape.split()]
            except ValueError:
                continue
            if len(pts) < 2:
                continue
            line = LineString(pts)
            out.append((eid, lane.get("id"), line, float(line.length)))
    return out


def _project_sign(lat, lon, transformer, off_xy):
    xu, yu = transformer.transform(lon, lat)
    return Point(xu + off_xy[0], yu + off_xy[1])


def _edge_way(edge_id: str) -> str:
    """Normalized osm way of a SUMO edge id: strip leading '-' and '#seg'."""
    e = edge_id[1:] if edge_id.startswith("-") else edge_id
    return e.split("#")[0]


def _snap(point: Point, lane_lines, max_perp: float, way: Optional[str] = None):
    """Snap point to the nearest lane. Return (edge_id, lane_id, s_offset, perp, length).

    When `way` is given, only lanes whose edge belongs to that osm way are
    considered — this avoids snapping a sign to a crossing road at a junction or
    to the opposite-direction carriageway, both of which the unconstrained
    nearest-lane search picks wrongly.
    """
    best = None
    for eid, lid, line, length in lane_lines:
        if way is not None and _edge_way(eid) != way:
            continue
        perp = point.distance(line)
        if best is None or perp < best[3]:
            best = (eid, lid, float(line.project(point)), perp, length)
    if best is None or best[3] > max_perp:
        return None
    return best


def _forward_graph(net_path: str) -> dict:
    """{from_edge: [to_edge,...]} for non-internal SUMO connections (directed)."""
    g = defaultdict(list)
    try:
        root = ET.parse(net_path).getroot()
    except (ET.ParseError, OSError):
        return g
    for c in root.findall("connection"):
        frm, to = c.get("from"), c.get("to")
        if not frm or not to or frm.startswith(":") or to.startswith(":"):
            continue
        if to not in g[frm]:
            g[frm].append(to)
    return g


def _bfs_path(graph, start, goal, max_hops=12):
    """Shortest directed edge path start..goal inclusive (BFS). None if unreachable."""
    if start == goal:
        return [start]
    from collections import deque
    q = deque([[start]])
    seen = {start}
    while q:
        path = q.popleft()
        if len(path) > max_hops:
            continue
        for nxt in graph.get(path[-1], []):
            if nxt in seen:
                continue
            if nxt == goal:
                return path + [nxt]
            seen.add(nxt)
            q.append(path + [nxt])
    return None


def _edge_len_map(lane_lines) -> dict:
    m = {}
    for eid, _lid, _line, length in lane_lines:
        m[eid] = max(m.get(eid, 0.0), length)
    return m


ZONE_MAX = 400.0   # reject implausibly long zones (BFS wandering / wrong pairing)
MAX_HOPS = 12


def pair_scenes(scenes_root: Path, out_root: Path, manifest_path: Path,
                max_dist: float, max_perp: float) -> dict:
    scenes = enumerate_all_scenes(scenes_root)
    by_code: dict[str, list[SumoScene]] = defaultdict(list)
    for s in scenes:
        by_code[s.sign_code].append(s)

    stats = defaultdict(int)
    rows = []

    for start_code, end_code in START_END_CODES.items():
        starts = by_code.get(start_code, [])
        ends = by_code.get(end_code, [])
        starts_by_way: dict[str, list[SumoScene]] = defaultdict(list)
        for s in starts:
            starts_by_way[_norm_way(s.osm_way_id)].append(s)

        for end in ends:
            stats[f"{end_code}_total"] += 1
            way = _norm_way(end.osm_way_id)
            same_way = set(id(s) for s in starts_by_way.get(way, []))
            # Pick the nearest start by geodesic distance over ALL starts; break
            # ties toward a start on the same osm_way.
            best_start, best_key = None, None
            for st in starts:
                d = _haversine(st.latitude, st.longitude, end.latitude, end.longitude)
                # sort key: distance, then prefer same-way (False sorts before True)
                key = (d, id(st) not in same_way)
                if best_key is None or key < best_key:
                    best_start, best_key = st, key
            best_d = best_key[0] if best_key else float("inf")
            if best_start is None or best_d > max_dist:
                stats[f"{end_code}_no_start_within_{int(max_dist)}m"] += 1
                continue

            base_net = scenes_root / best_start.net_path
            loc = _parse_location(str(base_net))
            if loc is None:
                stats["drop_no_location"] += 1
                continue
            offx, offy, proj4 = loc
            try:
                transformer = pyproj.Transformer.from_crs(
                    "EPSG:4326", pyproj.CRS.from_proj4(proj4), always_xy=True)
            except Exception:
                stats["drop_bad_proj"] += 1
                continue

            lane_lines = _lane_lines(str(base_net))
            if not lane_lines:
                stats["drop_no_lanes"] += 1
                continue

            # Snap each sign to ITS OWN osm way (avoids latching onto a crossing
            # road / opposite carriageway). Start and end may be on different
            # connected edges; the route is built through both.
            p_start = _project_sign(best_start.latitude, best_start.longitude, transformer, (offx, offy))
            p_end = _project_sign(end.latitude, end.longitude, transformer, (offx, offy))
            snap_s = _snap(p_start, lane_lines, max_perp, way=_norm_way(best_start.osm_way_id))
            snap_e = _snap(p_end, lane_lines, max_perp, way=_norm_way(end.osm_way_id))
            if snap_s is None or snap_e is None:
                stats["drop_snap_failed"] += 1
                continue

            edge_s, _ls, s_start, _ps, len_s = snap_s
            edge_e, _le, s_end, _pe, len_e = snap_e

            # Forward edge-chain start_edge -> end_edge over connected edges (one
            # travel direction). Fails when the end is not reachable forward from
            # the start (e.g. reversed/opposite carriageway) → drop.
            graph = _forward_graph(str(base_net))
            zone_edges = _bfs_path(graph, edge_s, edge_e, max_hops=MAX_HOPS)
            if zone_edges is None:
                stats["drop_no_forward_route"] += 1
                continue
            # Reject U-turn routes: if an edge and its opposite carriageway both
            # appear, the "route" doubles back (start/end on opposite sides) —
            # not a real one-direction zone.
            def _opp(e):
                return e[1:] if e.startswith("-") else "-" + e
            if any(_opp(e) in zone_edges for e in zone_edges):
                stats["drop_uturn"] += 1
                continue

            elen = _edge_len_map(lane_lines)
            if len(zone_edges) == 1:
                if s_end <= s_start:
                    stats["drop_reversed_or_zero"] += 1
                    continue
                zone_length = s_end - s_start
            else:
                zone_length = ((len_s - s_start)
                               + sum(elen.get(e, 0.0) for e in zone_edges[1:-1])
                               + s_end)
            if zone_length < ZONE_MIN:
                stats["drop_zone_too_short"] += 1
                continue
            if zone_length > ZONE_MAX:
                stats["drop_zone_too_long"] += 1
                continue
            # Approach on the start edge, tail on the end edge.
            if s_start < APPROACH_MIN or (len_e - s_end) < TAIL_MIN:
                stats["drop_no_approach_or_tail"] += 1
                continue

            # ---- emit combined scene ----
            pair_dir = out_root / f"{start_code}_{end_code}" / f"pair_{best_start.sign_id}_{end.sign_id}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            net_name = Path(best_start.net_path).name
            link = pair_dir / net_name
            src_net = (scenes_root / best_start.net_path).resolve()
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(src_net, link)

            n_edges = len(zone_edges)
            dest_lane_id = f"lane_{edge_e}_0"   # route must pass through the end edge
            meta = {
                "sign_id": best_start.sign_id,
                "sign_type": start_code,          # env places this as the start sign
                "road_id": edge_s,                # start sign edge
                "distance_from_start": round(s_start, 3),
                "net_file": net_name,
                "latitude": best_start.latitude,
                "longitude": best_start.longitude,
                "osm_way_id": best_start.osm_way_id,
                # ---- paired fields read by sumo_env ----
                "is_paired": True,
                "sign_type_start": start_code,
                "sign_type_end": end_code,
                "road_id_end": edge_e,
                "s_start": round(s_start, 3),
                "s_end": round(s_end, 3),
                "zone_edges": zone_edges,         # ordered directed edge chain
                "zone_length_m": round(zone_length, 3),
                "destination_lane_id": dest_lane_id,
                "end_sign_id": end.sign_id,
                "pair_geo_dist_m": round(best_d, 2),
            }
            with open(pair_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            net_rel = str((pair_dir / net_name).relative_to(out_root))
            rows.append({
                "scene_id": f"sumo_pair_{start_code}_{end_code}_{best_start.sign_id}_{end.sign_id}",
                "sign_id": best_start.sign_id,
                "sign_code": start_code,
                "pdd_code_start": start_code,
                "pdd_code_end": end_code,
                "sign_type_start": start_code,
                "sign_type_end": end_code,
                "road_id": edge_s,
                "road_id_end": edge_e,
                "net_path": net_rel,
                "distance_from_start": round(s_start, 3),
                "s_start": round(s_start, 3),
                "s_end": round(s_end, 3),
                "zone_edges": zone_edges,
                "zone_length_m": round(zone_length, 3),
                "destination_lane_id": dest_lane_id,
                "spawn_lane_num": 0,
                "source": "sumo_pair",
                "is_paired": True,
                "n_zone_edges": n_edges,
            })
            stats[f"{end_code}_paired"] += 1
            stats[f"{end_code}_edges_{min(n_edges,5)}"] += 1
            stats["paired_total"] += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return dict(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes-root", required=True,
                    help="Dir containing {3.24,3.25,5.31,5.32}/sign_*/meta.json")
    ap.add_argument("--out-root", required=True,
                    help="Where to write combined paired scene dirs")
    ap.add_argument("--manifest", required=True, help="Output manifest .jsonl path")
    ap.add_argument("--max-dist", type=float, default=300.0,
                    help="Max geodesic start↔end distance to pair (m)")
    ap.add_argument("--max-perp", type=float, default=25.0,
                    help="Max perpendicular snap distance sign→lane (m)")
    args = ap.parse_args()

    stats = pair_scenes(Path(args.scenes_root), Path(args.out_root),
                        Path(args.manifest), args.max_dist, args.max_perp)
    print("Pairing stats:")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
