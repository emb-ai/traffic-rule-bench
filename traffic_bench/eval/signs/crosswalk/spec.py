"""Parse SUMO networks for pedestrian crossings (PDD 5.19)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from traffic_bench.eval.engine.map.lane_keys import lane_edge_id, make_lane_key
from traffic_bench.eval.engine.map.sumo_utils import VehicleRouteIndex, is_vehicle_drivable_lane, load_vehicle_route_index


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


def count_net_crossings(net_path: Path) -> int:
    """Return the number of SUMO pedestrian crossing edges in a net."""
    if not net_path.is_file():
        return 0
    root = ET.parse(net_path).getroot()
    return sum(
        1
        for edge in root.findall("edge")
        if edge.get("function") == "crossing" and edge.get("crossingEdges")
    )


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


def _lane_depths_from_spawn(
    road_network,
    spawn_lane_key: str,
    *,
    max_hops: int = 12,
) -> dict[str, int]:
    """BFS lane depths from spawn; internal ``lane_:`` lanes are traversable but not destinations."""
    if spawn_lane_key not in road_network.graph:
        return {}

    depths: dict[str, int] = {spawn_lane_key: 0}
    queue: list[tuple[str, int]] = [(spawn_lane_key, 0)]
    visited = {spawn_lane_key}

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
            visited.add(nxt)
            depths[nxt] = depth + 1
            queue.append((nxt, depth + 1))
    return depths


def _is_internal_lane_key(lane_key: str) -> bool:
    return str(lane_key).startswith("lane_:")


def pick_destination_from_road_network(
    road_network,
    spawn_lane_key: str,
    depart_lane_key: str,
    *,
    min_hops_after_depart: int = 2,
    max_hops: int = 12,
) -> Optional[str]:
    """Runtime fallback: pick a MetaDrive lane past the crosswalk using the road graph."""
    depths = _lane_depths_from_spawn(road_network, spawn_lane_key, max_hops=max_hops)
    if not depths:
        return None

    depart_depth = depths.get(depart_lane_key)
    if depart_depth is None:
        depart_edge = lane_edge_id(depart_lane_key)
        depart_depths = [d for key, d in depths.items() if lane_edge_id(key) == depart_edge]
        depart_depth = min(depart_depths) if depart_depths else None

    for min_hops in range(min_hops_after_depart, 0, -1):
        if depart_depth is None:
            min_depth = min_hops
        else:
            min_depth = depart_depth + min_hops

        candidates = [
            (key, depth)
            for key, depth in depths.items()
            if depth >= min_depth
            and key != spawn_lane_key
            and not _is_internal_lane_key(key)
        ]
        if candidates:
            candidates.sort(key=lambda item: item[1], reverse=True)
            return candidates[0][0]

    return None


def enumerate_crosswalk_dest_candidates(
    road_network,
    spawn_lane_key: str,
    depart_lane_key: str,
    explicit_dest: str | None = None,
    *,
    min_hops_after_depart: int = 2,
    max_hops: int = 12,
) -> list[str]:
    """Ordered destination lane keys to try for crosswalk navigation."""
    candidates: list[str] = []
    seen: set[str] = set()

    def add(key: str | None) -> None:
        if not key or key in seen or _is_internal_lane_key(key):
            return
        seen.add(key)
        candidates.append(key)

    add(explicit_dest)
    picked = pick_destination_from_road_network(
        road_network,
        spawn_lane_key,
        depart_lane_key,
        min_hops_after_depart=min_hops_after_depart,
        max_hops=max_hops,
    )
    add(picked)

    depths = _lane_depths_from_spawn(road_network, spawn_lane_key, max_hops=max_hops)
    depart_depth = depths.get(depart_lane_key)
    if depart_depth is None:
        depart_edge = lane_edge_id(depart_lane_key)
        depart_depths = [d for d, key in ((depths[k], k) for k in depths) if lane_edge_id(key) == depart_edge]
        depart_depth = min(depart_depths) if depart_depths else None

    ranked = sorted(
        (
            (key, depth)
            for key, depth in depths.items()
            if key != spawn_lane_key and not _is_internal_lane_key(key)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if depart_depth is not None:
        for min_hops in range(min_hops_after_depart, 0, -1):
            min_depth = depart_depth + min_hops
            for key, depth in ranked:
                if depth >= min_depth:
                    add(key)
    else:
        for key, _depth in ranked:
            add(key)

    return candidates


def _collect_edge_lanes(root: ET.Element) -> dict[str, list[tuple[int, float]]]:
    edge_lanes: dict[str, list[tuple[int, float]]] = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        if not edge_id or edge_id.startswith(":"):
            continue
        if edge.get("function") not in {None, "normal", ""}:
            continue
        lanes = _edge_lane_lengths(edge)
        if lanes:
            edge_lanes[edge_id] = lanes
    return edge_lanes


def _approaches_for_crossing(
    *,
    crossing_id: str,
    junction_id: str,
    crossed_edge_ids: tuple[str, ...],
    endpoints: dict[str, tuple[str, str]],
    edge_lanes: dict[str, list[tuple[int, float]]],
    route_index: VehicleRouteIndex,
    min_approach_length: float,
    min_hops_after_depart: int,
    max_destination_hops: int,
    seen: set[tuple[str, str, int]],
) -> list[CrosswalkApproach]:
    approaches: list[CrosswalkApproach] = []
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
    # Meta / split-node fallback: crossed list may be incomplete — use all
    # edges that touch the zebra junction.
    if not approach_edges or not depart_edges:
        approach_edges = [
            eid for eid, (_frm, to) in endpoints.items() if to == junction_id
        ]
        depart_edges = [
            eid for eid, (frm, _to) in endpoints.items() if frm == junction_id
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
                    crossed_edge_ids=crossed_edge_ids or tuple(
                        sorted(set(approach_edges) | set(depart_edges))
                    ),
                    approach_edge_id=approach_edge_id,
                    depart_edge_id=depart_edge_id,
                    approach_lane_num=lane_num,
                    approach_lane_length=lane_length,
                    destination_lane_id=dest_lane_id,
                    scenario_id=scenario_id,
                )
            )
    return approaches


def _meta_crosswalk_node_id(meta: Mapping[str, Any] | None) -> str:
    if not meta:
        return ""
    for key in ("crosswalk_node_id", "junction_id"):
        raw = meta.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return ""


def _meta_crossed_edge_ids(meta: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not meta:
        return ()
    raw = meta.get("crossed_edge_ids")
    if isinstance(raw, (list, tuple)):
        return tuple(str(e) for e in raw if e)
    road_id = str(meta.get("road_id") or "").strip()
    if not road_id:
        return ()
    reverse = f"-{road_id}" if not road_id.startswith("-") else road_id[1:]
    return (road_id, reverse)


def _junction_exists(root: ET.Element, junction_id: str) -> bool:
    if not junction_id:
        return False
    for junction in root.findall("junction"):
        if junction.get("id") == junction_id:
            return True
    return False


def build_crosswalk_approaches(
    net_path: Path,
    *,
    min_approach_length: float = 15.0,
    min_hops_after_depart: int = 2,
    max_destination_hops: int = 8,
    route_index: Optional[VehicleRouteIndex] = None,
    meta: Mapping[str, Any] | None = None,
) -> list[CrosswalkApproach]:
    """Enumerate vehicle approach lanes toward a mid-block zebra.

    Prefers SUMO ``function=crossing`` edges. When netconvert dropped the
    crossing edge but prepare left ``crosswalk_node_id`` in meta, falls back to
    edges that enter/leave that split node.
    """
    if not net_path.is_file():
        return []

    root = ET.parse(net_path).getroot()
    endpoints = _load_edge_endpoints(root)
    if route_index is None:
        route_index = load_vehicle_route_index(net_path)
    edge_lanes = _collect_edge_lanes(root)

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
        approaches.extend(
            _approaches_for_crossing(
                crossing_id=crossing_id,
                junction_id=junction_id,
                crossed_edge_ids=crossed_edge_ids,
                endpoints=endpoints,
                edge_lanes=edge_lanes,
                route_index=route_index,
                min_approach_length=min_approach_length,
                min_hops_after_depart=min_hops_after_depart,
                max_destination_hops=max_destination_hops,
                seen=seen,
            )
        )

    if approaches:
        return approaches

    # Fallback: injected split node without a SUMO crossing edge.
    node_id = _meta_crosswalk_node_id(meta)
    if node_id and _junction_exists(root, node_id):
        crossed = _meta_crossed_edge_ids(meta)
        synthetic_id = str(
            meta.get("crosswalk_edge_id") or meta.get("crosswalk_id") or f":{node_id}_c0"
        )
        return _approaches_for_crossing(
            crossing_id=synthetic_id,
            junction_id=node_id,
            crossed_edge_ids=crossed,
            endpoints=endpoints,
            edge_lanes=edge_lanes,
            route_index=route_index,
            min_approach_length=min_approach_length,
            min_hops_after_depart=min_hops_after_depart,
            max_destination_hops=max_destination_hops,
            seen=seen,
        )

    # No-split prepare: continuous edge, zebra marked only in meta.
    return _approaches_from_meta_position(
        meta=meta,
        edge_lanes=edge_lanes,
        min_approach_length=min_approach_length,
        seen=seen,
    )


def _approaches_from_meta_position(
    *,
    meta: Mapping[str, Any] | None,
    edge_lanes: dict[str, list[tuple[int, float]]],
    min_approach_length: float,
    seen: set[tuple[str, str, int]],
) -> list[CrosswalkApproach]:
    """Build approaches on a continuous edge using ``crosswalk_position_m``.

    Approach and depart share the same edge — no mid-block SUMO split. Spawn /
    sign offsets and ``destination_max_along_m`` are applied relative to that
    along-edge mark in expand / place.
    """
    if not meta:
        return []
    try:
        pos_m = float(meta.get("crosswalk_position_m"))
    except (TypeError, ValueError):
        return []
    if pos_m <= 0.0:
        return []

    edge_id = str(
        meta.get("crosswalk_edge_id")
        or meta.get("road_id")
        or (list(meta.get("crossed_edge_ids") or [None])[0] or "")
    ).strip()
    if not edge_id or edge_id not in edge_lanes:
        return []

    crossed = tuple(
        str(e) for e in (meta.get("crossed_edge_ids") or (edge_id,)) if e
    ) or (edge_id,)
    approaches: list[CrosswalkApproach] = []
    for lane_num, lane_len in edge_lanes[edge_id]:
        if float(lane_len) < float(min_approach_length):
            continue
        if pos_m < float(min_approach_length) or pos_m >= float(lane_len) - 5.0:
            continue
        key = (f"meta_cw_{edge_id}", edge_id, int(lane_num))
        if key in seen:
            continue
        seen.add(key)
        approaches.append(
            CrosswalkApproach(
                crosswalk_id=f"meta_cw_{edge_id}",
                junction_id="",
                crossed_edge_ids=crossed,
                approach_edge_id=edge_id,
                depart_edge_id=edge_id,
                approach_lane_num=int(lane_num),
                # Treat the zebra mark as the approach "end" for spawn clamps.
                approach_lane_length=float(pos_m),
                destination_lane_id=make_lane_key(edge_id, int(lane_num)),
                scenario_id=f"{edge_id}_L{int(lane_num)}",
            )
        )
    return approaches
