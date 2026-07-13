"""Parse SUMO networks for pedestrian crossings (PDD 5.19)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lib.lane_keys import lane_edge_id, make_lane_key
from lib.sumo_utils import VehicleRouteIndex, is_vehicle_drivable_lane, load_vehicle_route_index


@dataclass(frozen=True)
class CrosswalkApproach:
    """Ego approach lane toward a SUMO crossing edge."""

    crosswalk_id: str
    junction_id: str
    crossed_edge_ids: tuple[str, ...]
    approach_edge_id: str
    depart_edge_id: str
    approach_lane_num: int
    approach_lane_length: float
    destination_lane_id: str
    scenario_id: str


def parse_crossing_junction_id(crossing_edge_id: str) -> Optional[str]:
    """Extract junction id from a SUMO crossing edge like ``:6528538035_c0``."""
    if not crossing_edge_id.startswith(":"):
        return None
    body = crossing_edge_id[1:]
    if "_c" not in body:
        return None
    return body.split("_c", 1)[0]


def net_has_crossings(net_path: Path) -> bool:
    """Return True when the SUMO net defines at least one pedestrian crossing."""
    if not net_path.is_file():
        return False
    root = ET.parse(net_path).getroot()
    for edge in root.findall("edge"):
        if edge.get("function") == "crossing" and edge.get("crossingEdges"):
            return True
    return False


def _edge_lane_lengths(edge_el: ET.Element) -> list[tuple[int, float]]:
    lanes: list[tuple[int, float]] = []
    for lane in edge_el.findall("lane"):
        if not is_vehicle_drivable_lane(lane):
            continue
        lane_id = lane.get("id", "")
        try:
            lane_num = int(lane_id.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            lane_num = 0
        length = float(lane.get("length", 0) or 0)
        if length <= 0:
            shape = (lane.get("shape") or "").strip().split()
            coords = [tuple(map(float, p.split(","))) for p in shape if "," in p]
            if len(coords) >= 2:
                length = sum(
                    ((coords[i + 1][0] - coords[i][0]) ** 2 + (coords[i + 1][1] - coords[i][1]) ** 2) ** 0.5
                    for i in range(len(coords) - 1)
                )
        lanes.append((lane_num, length))
    return lanes


def _load_edge_endpoints(root: ET.Element) -> dict[str, tuple[str, str]]:
    endpoints: dict[str, tuple[str, str]] = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        if not edge_id or edge.get("function") in {"internal", "crossing", "walkingarea"}:
            continue
        if edge_id.startswith(":"):
            continue
        endpoints[edge_id] = (edge.get("from", ""), edge.get("to", ""))
    return endpoints


def _paired_depart_edge(
    approach_edge_id: str,
    depart_candidates: list[str],
    endpoints: dict[str, tuple[str, str]],
) -> Optional[str]:
    if not depart_candidates:
        return None
    if len(depart_candidates) == 1:
        return depart_candidates[0]

    approach_from, approach_to = endpoints.get(approach_edge_id, ("", ""))
    for depart_edge_id in depart_candidates:
        depart_from, depart_to = endpoints.get(depart_edge_id, ("", ""))
        if depart_from == approach_to and depart_to == approach_from:
            return depart_edge_id

    # Fallback: first depart edge on the same road name stem.
    stem = approach_edge_id.lstrip("-").split("#", 1)[0]
    for depart_edge_id in depart_candidates:
        if depart_edge_id.lstrip("-").split("#", 1)[0] == stem:
            return depart_edge_id
    return depart_candidates[0]


def resolve_destination_beyond_crosswalk(
    route_index: VehicleRouteIndex,
    approach_edge_id: str,
    approach_lane_num: int,
    depart_edge_id: str,
    depart_lane_num: int,
    *,
    min_hops_after_depart: int = 2,
    max_hops: int = 8,
) -> str:
    """Pick a navigation destination lane far enough past the crosswalk."""
    depart_lane_key = make_lane_key(depart_edge_id, depart_lane_num)
    if not route_index.can_reach_edge(approach_edge_id, approach_lane_num, depart_edge_id):
        return depart_lane_key

    for min_hops in range(min_hops_after_depart, 0, -1):
        farther = route_index.farthest_reachable_lane(
            depart_edge_id,
            depart_lane_num,
            min_hops=min_hops,
            max_hops=max_hops,
        )
        if farther is not None:
            edge_id, lane_num = farther
            if make_lane_key(edge_id, lane_num) != make_lane_key(approach_edge_id, approach_lane_num):
                return make_lane_key(edge_id, lane_num)

    return depart_lane_key


def pick_destination_from_road_network(
    road_network,
    spawn_lane_key: str,
    depart_lane_key: str,
    *,
    min_hops_after_depart: int = 2,
    max_hops: int = 12,
) -> Optional[str]:
    """Runtime fallback: pick a MetaDrive lane past the crosswalk using the road graph."""
    if spawn_lane_key not in road_network.graph:
        return None

    depths: dict[str, int] = {}
    queue: list[tuple[str, int]] = [(spawn_lane_key, 0)]
    visited = {spawn_lane_key}
    depths[spawn_lane_key] = 0

    while queue:
        lane_key, depth = queue.pop(0)
        if depth >= max_hops:
            continue
        info = road_network.graph.get(lane_key)
        if info is None:
            continue
        exit_lanes = sorted(set(getattr(info, "exit_lanes", None) or []))
        for nxt in exit_lanes:
            if nxt not in road_network.graph or nxt in visited:
                continue
            if str(nxt).startswith("lane_:"):
                continue
            visited.add(nxt)
            depths[nxt] = depth + 1
            queue.append((nxt, depth + 1))

    depart_depth = depths.get(depart_lane_key)
    if depart_depth is None:
        depart_edge = lane_edge_id(depart_lane_key)
        depart_depths = [d for key, d in depths.items() if lane_edge_id(key) == depart_edge]
        depart_depth = min(depart_depths) if depart_depths else None

    if depart_depth is None:
        candidates = [
            (key, depth)
            for key, depth in depths.items()
            if depth >= min_hops_after_depart and key != spawn_lane_key
        ]
    else:
        min_depth = depart_depth + min_hops_after_depart
        candidates = [
            (key, depth)
            for key, depth in depths.items()
            if depth >= min_depth and key != spawn_lane_key
        ]

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


def build_crosswalk_approaches(
    net_path: Path,
    *,
    min_approach_length: float = 15.0,
    min_hops_after_depart: int = 2,
    max_destination_hops: int = 8,
    route_index: Optional[VehicleRouteIndex] = None,
) -> list[CrosswalkApproach]:
    """Enumerate vehicle approach lanes that lead toward each SUMO crossing."""
    if not net_path.is_file():
        return []

    root = ET.parse(net_path).getroot()
    endpoints = _load_edge_endpoints(root)
    if route_index is None:
        route_index = load_vehicle_route_index(net_path)

    edge_lanes: dict[str, list[tuple[int, float]]] = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        if not edge_id or edge.get("function") not in {None, "normal", ""}:
            if edge.get("function") not in {None, "normal", ""}:
                continue
        if edge_id.startswith(":"):
            continue
        lanes = _edge_lane_lengths(edge)
        if lanes:
            edge_lanes[edge_id] = lanes

    approaches: list[CrosswalkApproach] = []
    seen: set[tuple[str, str, int]] = set()

    for edge in root.findall("edge"):
        if edge.get("function") != "crossing":
            continue
        crossing_id = edge.get("id", "")
        crossed_raw = (edge.get("crossingEdges") or "").strip()
        if not crossing_id or not crossed_raw:
            continue

        junction_id = parse_crossing_junction_id(crossing_id) or ""
        crossed_edge_ids = tuple(e for e in crossed_raw.split() if e)

        approach_edges = [
            eid
            for eid in crossed_edge_ids
            if endpoints.get(eid, ("", ""))[1] == junction_id
        ]
        depart_edges = [
            eid
            for eid in crossed_edge_ids
            if endpoints.get(eid, ("", ""))[0] == junction_id
        ]

        for approach_edge_id in approach_edges:
            depart_edge_id = _paired_depart_edge(approach_edge_id, depart_edges, endpoints)
            if depart_edge_id is None:
                continue

            for lane_num, lane_length in edge_lanes.get(approach_edge_id, []):
                if lane_length < min_approach_length:
                    continue

                if not route_index.can_reach_edge(approach_edge_id, lane_num, depart_edge_id):
                    continue

                dest_lane_id = resolve_destination_beyond_crosswalk(
                    route_index,
                    approach_edge_id,
                    lane_num,
                    depart_edge_id,
                    lane_num,
                    min_hops_after_depart=min_hops_after_depart,
                    max_hops=max_destination_hops,
                )
                key = (crossing_id, approach_edge_id, lane_num)
                if key in seen:
                    continue

                seen.add(key)
                scenario_id = (
                    f"cw_{crossing_id.replace(':', '')}_"
                    f"{approach_edge_id.replace('#', 'h')}_ln{lane_num}"
                )
                approaches.append(
                    CrosswalkApproach(
                        crosswalk_id=crossing_id,
                        junction_id=junction_id,
                        crossed_edge_ids=crossed_edge_ids,
                        approach_edge_id=approach_edge_id,
                        depart_edge_id=depart_edge_id,
                        approach_lane_num=lane_num,
                        approach_lane_length=lane_length,
                        destination_lane_id=dest_lane_id,
                        scenario_id=scenario_id,
                    )
                )

    return approaches
