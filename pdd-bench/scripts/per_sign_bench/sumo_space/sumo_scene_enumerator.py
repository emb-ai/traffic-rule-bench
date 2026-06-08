"""Walks /scenes/{sign_code}/sign_*/meta.json and returns SumoScene records.

Metadata + lightweight .net.xml parsing (only the <edge id="road_id"> block
for lane count) — no MetaDrive env, no traci. Cheap.
"""
from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import List, Optional


@lru_cache(maxsize=4096)
def _forward_connections(net_path: str) -> dict:
    """Return {from_edge: [to_edge, ...]} for this .net.xml.

    Only non-internal edges are kept. Internal junction lanes (':' IDs)
    are used as bridges between real edges — `via` is resolved by matching
    the parent junction's non-internal exits.
    """
    try:
        tree = ET.parse(net_path)
    except (ET.ParseError, OSError):
        return {}
    root = tree.getroot()
    graph: dict = defaultdict(list)
    for c in root.findall("connection"):
        frm = c.get("from")
        to = c.get("to")
        if not frm or not to:
            continue
        if frm.startswith(":") or to.startswith(":"):
            # skip internal connections: :junction-lane hops are implicit
            continue
        graph[frm].append(to)
    # dedup while preserving order
    return {k: list(dict.fromkeys(v)) for k, v in graph.items()}


def _base_way(edge_id: str) -> str:
    """OSM base way id of a SUMO edge: strip a leading '-' and the '#<seg>' suffix.
    Edges '108888798#2' and '-108888798#0' share the base '108888798'."""
    return edge_id.lstrip("-").split("#")[0]


def _is_reverse(a: str, b: str) -> bool:
    """True if b is the OPPOSITE-direction edge of the same OSM way as a.
    SUMO marks reverse direction with a leading '-': '108888798#2' vs
    '-108888798#0'. A same-direction continuation ('108888798#2' -> '108888798#3')
    is NOT reverse and must stay allowed."""
    return (_base_way(a) == _base_way(b)
            and a.startswith("-") != b.startswith("-"))


def _walk_forward_edge(net_path: str, sign_edge: str, hops: int = 1) -> Optional[str]:
    """Walk `hops` edges forward from sign_edge via SUMO connections.

    Returns the edge_id reached, or None if unreachable (dead end). Picks the
    first outgoing edge at each hop (sorted for determinism), skipping only the
    REVERSE-direction segment of the current way (a U-turn that would put the
    destination behind the sign). A forward continuation of the same way is kept,
    so the spawn->dest route passes forward through the sign (sign stays mid-route).
    """
    graph = _forward_connections(net_path)
    current = sign_edge
    visited = {current}
    for _ in range(hops):
        outs = graph.get(current)
        if not outs:
            break
        # deterministic pick — alphabetical, skipping visited (cycles) and the
        # reverse direction of the current edge (U-turn back onto the same road).
        cand = sorted(e for e in outs
                      if e not in visited and not _is_reverse(current, e))
        if not cand:
            break
        current = cand[0]
        visited.add(current)
    return current if current != sign_edge else None


def _destination_lane_id(net_path: str, sign_road_id: str, hops: int = 1) -> Optional[str]:
    """Compute destination lane ID (the ID MetaDrive expects in
    nav.set_route / vehicle.config['destination']).

    MetaDrive wraps SUMO lane IDs as 'lane_<edge>_<laneNum>'. We walk forward
    `hops` edge past sign_road_id and pick lane 0 of the reached edge, so the
    route is spawn -> sign -> destination (sign in the MIDDLE, just inside the
    zone) and stays short. hops=1 keeps the route just past the sign.
    """
    dest_edge = _walk_forward_edge(net_path, sign_road_id, hops=hops)
    if dest_edge is None:
        return None
    return f"lane_{dest_edge}_0"


@lru_cache(maxsize=4096)
def count_lanes_on_road(net_path: str, road_id: str) -> int:
    """Return the number of driving lanes on the named edge in a SUMO .net.xml.

    Matches the edge whose `id` attribute is exactly `road_id`. Returns 0 when
    the edge is not found (caller should treat as unparsable). Cached so
    repeated calls for the same file are cheap.
    """
    try:
        tree = ET.parse(net_path)
    except (ET.ParseError, OSError):
        return 0
    root = tree.getroot()
    for edge in root.findall("edge"):
        if edge.get("id") == road_id:
            lanes = [l for l in edge.findall("lane")
                     if (l.get("disallow") or "").find("passenger") == -1
                     or (l.get("allow") or "").find("passenger") >= 0]
            # Fall back to all lanes if filter would exclude everything
            # (some networks don't set allow/disallow).
            return len(lanes) or len(edge.findall("lane"))
    return 0


@dataclass(frozen=True)
class SumoScene:
    sign_code: str            # GOST code, e.g. "2.5", "3.24"
    sign_id: int              # numeric sign ID
    road_id: str              # OSM way id (used for sign placement and spawn)
    net_path: str             # absolute path to .net.xml
    latitude: float
    longitude: float
    distance_from_start: float
    osm_way_id: str
    # Lane ID a few edges past the sign — used as an explicit destination
    # so navigation BFS returns a short deterministic route that passes
    # through the sign. None when no forward edge exists (dead end).
    destination_lane_id: Optional[str] = None

    @property
    def scene_id(self) -> str:
        return f"sumo_{self.sign_code}_{self.sign_id}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scene_id"] = self.scene_id
        return d


def enumerate_all_scenes(scenes_root: str | Path) -> List[SumoScene]:
    """Walk the scenes directory and parse every meta.json into a SumoScene.

    Returns scenes in deterministic sorted order (sign_code, sign_id).
    """
    scenes_root = Path(scenes_root)
    if not scenes_root.is_dir():
        raise FileNotFoundError(f"scenes_root not found: {scenes_root}")

    scenes: List[SumoScene] = []
    for sign_dir in sorted(scenes_root.iterdir()):
        if not sign_dir.is_dir():
            continue
        sign_code = sign_dir.name
        for scene_dir in sorted(sign_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            meta_path = scene_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                continue

            net_file = meta.get("net_file")
            if not net_file:
                continue
            net_path = scene_dir / net_file
            if not net_path.exists():
                continue

            try:
                road_id = str(meta.get("road_id", ""))
                dest_lane_id = _destination_lane_id(str(net_path), road_id, hops=2)
                scenes.append(SumoScene(
                    sign_code=sign_code,
                    sign_id=int(meta.get("sign_id", 0)),
                    road_id=road_id,
                    net_path=str(net_path.relative_to(scenes_root)),
                    latitude=float(meta.get("latitude", 0.0)),
                    longitude=float(meta.get("longitude", 0.0)),
                    distance_from_start=float(meta.get("distance_from_start", 0.0)),
                    osm_way_id=str(meta.get("osm_way_id", "")),
                    destination_lane_id=dest_lane_id,
                ))
            except (TypeError, ValueError):
                continue

    scenes.sort(key=lambda s: (s.sign_code, s.sign_id))
    return scenes


def stratified_sample(
    scenes: List[SumoScene],
    n_per_category: int,
    seed: int = 42,
) -> List[SumoScene]:
    """Pick `n_per_category` scenes from each sign code (or all if fewer available).

    Deterministic: same `seed` → same selection.
    """
    rng = random.Random(seed)
    by_code: dict[str, List[SumoScene]] = defaultdict(list)
    for s in scenes:
        by_code[s.sign_code].append(s)

    result: List[SumoScene] = []
    for code in sorted(by_code.keys()):
        bucket = list(by_code[code])
        if n_per_category >= len(bucket):
            picked = bucket
        else:
            picked = rng.sample(bucket, n_per_category)
        picked.sort(key=lambda s: s.sign_id)
        result.extend(picked)
    return result


def filter_by_sign_codes(
    scenes: List[SumoScene],
    codes: Optional[List[str]],
) -> List[SumoScene]:
    """Restrict to a subset of sign codes. None or empty list = no filter."""
    if not codes:
        return list(scenes)
    code_set = set(codes)
    return [s for s in scenes if s.sign_code in code_set]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-root", type=str, required=True)
    args = parser.parse_args()
    all_scenes = enumerate_all_scenes(args.scenes_root)
    print(f"Total: {len(all_scenes)} scenes across "
          f"{len(set(s.sign_code for s in all_scenes))} sign categories")
    by_code: dict[str, int] = defaultdict(int)
    for s in all_scenes:
        by_code[s.sign_code] += 1
    for code in sorted(by_code.keys()):
        print(f"  {code:8s}: {by_code[code]}")
