"""Legal ego/aux combinations for a 4.3 roundabout (spoke in, ring aux)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from traffic_bench.eval.engine.map.junction_priority_layout import JunctionPriorityLayout
from traffic_bench.eval.engine.spawn.scene_augmentation import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    ApproachSpawnLane,
    SpawnScenario,
    _is_valid_departure,
    _lane_keys_lookup,
    _pick_outgoing_lane_key,
    build_spawn_lanes_by_edge,
    lane_lengths_from_spawn_lanes,
    parse_intersection_approach_lanes,
)
from traffic_bench.eval.engine.map.sumo_utils import VehicleRouteIndex, load_vehicle_route_index


def roundabout_meta_ring_kwargs(scene_meta: Optional[dict]) -> dict:
    """Map moscow / legacy roundabout meta keys into ``build_roundabout_layout`` kwargs."""
    meta = scene_meta or {}
    ring = meta.get("ring_edge_ids") or meta.get("roundabout_ring_edges")
    spokes = meta.get("spoke_edge_ids") or meta.get("roundabout_spoke_edges")
    entry = (
        meta.get("roundabout_entry_junction")
        or meta.get("entry_junction_id")
        or meta.get("junction_id")
    )
    kwargs: dict = {}
    if ring:
        kwargs["ring_edge_ids"] = list(ring)
    if spokes:
        kwargs["spoke_edge_ids"] = list(spokes)
    if entry:
        kwargs["entry_junction_id"] = str(entry)
    return kwargs


def _roundabout_exit_edges(
    layout: JunctionPriorityLayout,
    ego_edge_id: str,
) -> List[str]:
    """Outbound spoke edges whose lane end can be used as ego destination."""
    ring_nodes: set[str] = set()
    for arm in layout.arms:
        if arm.edge_id not in layout.main_edge_ids:
            continue
        if arm.from_node:
            ring_nodes.add(arm.from_node)
        if arm.to_node:
            ring_nodes.add(arm.to_node)

    exits: List[str] = []
    for arm in layout.arms:
        if arm.road_class != "secondary":
            continue
        if arm.edge_id == ego_edge_id:
            continue
        if arm.from_node in ring_nodes and arm.to_node not in ring_nodes:
            exits.append(arm.edge_id)
    return sorted(set(exits))


def _roundabout_ego_destination_edges(
    layout: JunctionPriorityLayout,
    ego_edge_id: str,
) -> List[str]:
    return _roundabout_exit_edges(layout, ego_edge_id)


def _roundabout_ego_spawn_edges(
    layout: JunctionPriorityLayout,
    spawn_by_edge: Dict[str, List[int]],
    *,
    prefer_ego_edge_id: Optional[str] = None,
) -> List[str]:
    """Spoke approaches (never ring edges)."""
    spokes = sorted(e for e in layout.secondary_edge_ids if e in spawn_by_edge)
    if prefer_ego_edge_id:
        prefix = prefer_ego_edge_id.split("#", 1)[0]
        chain = [
            e
            for e in spawn_by_edge
            if e not in layout.main_edge_ids
            and (e == prefer_ego_edge_id or e.startswith(prefix + "#"))
        ]
        spokes = sorted(set(spokes) | set(chain))
    if spokes:
        return spokes
    return sorted(e for e in spawn_by_edge if e not in layout.main_edge_ids)


def enumerate_spawn_scenarios_roundabout(
    layout: JunctionPriorityLayout,
    spawn_lanes_by_edge: Dict[str, List[int]],
    *,
    min_lane_length: float = 20.0,
    lane_lengths: Optional[Dict[Tuple[str, int], float]] = None,
    route_index: Optional[VehicleRouteIndex] = None,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
) -> List[SpawnScenario]:
    """Ego on spoke; aux on the left-hand conflict arc; aux dest = ego dest.

    Spawn: ring segment immediately upstream of ego's entry (left of the ego
    spoke). Conflict arcs shorter than ``MIN_CONFLICT_ARC_LENGTH_M`` are
    dropped; otherwise aux lead sits ``aux_distance`` before the entry,
    clamped to the segment when the arc is shorter than a full offset.
    Convoy followers may spill onto upstream ring hops.
    Destination: always the same exit edge/lane as ego (runtime may
    ring-circulate one hop at a time).
    """
    from traffic_bench.eval.engine.map.roundabout_yield_zone import entry_conflict_ring_edges
    from traffic_bench.eval.signs.roundabout.aux import (
        merge_lane_lengths_from_layout,
        resolve_aux_spawn_placement,
    )
    from traffic_bench.eval.engine.map.lane_keys import lane_num_from_key, pick_lane_key_on_edge

    layout_dict = layout.to_dict()
    lane_lengths = merge_lane_lengths_from_layout(layout_dict, lane_lengths or {})
    lane_keys_by_edge = _lane_keys_lookup(layout)

    def _aux_lane_nums(aux_edge: str) -> List[int]:
        nums = spawn_lanes_by_edge.get(aux_edge, [])
        if nums:
            return nums
        arm = layout.arm_for_edge(aux_edge)
        if arm is None:
            return []
        return sorted({lane_num_from_key(key) for key in arm.lane_keys})

    secondary_edges = _roundabout_ego_spawn_edges(
        layout, spawn_lanes_by_edge, prefer_ego_edge_id=None
    )
    scenarios: List[SpawnScenario] = []

    for ego_edge in secondary_edges:
        ego_lane_nums = spawn_lanes_by_edge.get(ego_edge, [])
        if not ego_lane_nums:
            continue

        ego_dest_edges = _roundabout_ego_destination_edges(layout, ego_edge)
        if not ego_dest_edges:
            continue

        left_conflict_edges = entry_conflict_ring_edges(layout_dict, ego_edge)
        if not left_conflict_edges:
            continue
        placement_allowed = set(left_conflict_edges)

        for aux_edge in sorted(left_conflict_edges):
            aux_lane_placements: List[tuple[int, object]] = []
            for lane_num in _aux_lane_nums(aux_edge):
                placement = resolve_aux_spawn_placement(
                    layout_dict,
                    aux_edge,
                    lane_num,
                    lane_lengths,
                    aux_distance_from_intersection,
                    allowed_ring_edges=placement_allowed,
                )
                if placement is not None:
                    aux_lane_placements.append((lane_num, placement))
            if not aux_lane_placements:
                continue

            for ego_lane in ego_lane_nums:
                if lane_lengths.get((ego_edge, ego_lane), min_lane_length) < min_lane_length:
                    continue

                reachable_destinations: List[tuple[str, str]] = []
                for ego_dest_edge in ego_dest_edges:
                    if ego_dest_edge == ego_edge:
                        continue
                    ego_dest_lane_key = _pick_outgoing_lane_key(
                        ego_dest_edge,
                        ego_lane,
                        lane_keys_by_edge,
                    )
                    if not _is_valid_departure(
                        ego_edge,
                        ego_lane,
                        ego_dest_edge,
                        ego_dest_lane_key,
                    ):
                        continue
                    if route_index is not None and not route_index.can_reach_edge(
                        ego_edge, ego_lane, ego_dest_edge
                    ):
                        continue
                    reachable_destinations.append((ego_dest_edge, ego_dest_lane_key))

                for ego_dest_edge, ego_dest_lane_key in reachable_destinations:
                    for _aux_lane, placement in aux_lane_placements:
                        aux_dest_edge = ego_dest_edge
                        aux_dest_lane_key = ego_dest_lane_key
                        if route_index is not None and route_index.can_reach_edge(
                            placement.spawn_edge_id,
                            placement.spawn_lane_num,
                            aux_dest_edge,
                        ):
                            allowed = route_index.reachable_lanes_on_edge(
                                placement.spawn_edge_id,
                                placement.spawn_lane_num,
                                aux_dest_edge,
                            )
                            ego_dest_ln = lane_num_from_key(ego_dest_lane_key)
                            if allowed and ego_dest_ln not in allowed:
                                remapped = pick_lane_key_on_edge(
                                    aux_dest_edge,
                                    placement.spawn_lane_num,
                                    lane_keys_by_edge,
                                    allowed_lane_nums=sorted(allowed),
                                )
                                if remapped:
                                    aux_dest_lane_key = remapped

                        scenario_id = (
                            f"ego_{ego_edge}_L{ego_lane}"
                            f"_to_{ego_dest_edge}"
                            f"_aux_{placement.spawn_edge_id}_L{placement.spawn_lane_num}"
                        )
                        scenarios.append(
                            SpawnScenario(
                                ego_edge_id=ego_edge,
                                ego_lane_num=ego_lane,
                                ego_destination_edge_id=ego_dest_edge,
                                ego_destination_lane_key=ego_dest_lane_key,
                                aux_edge_id=placement.spawn_edge_id,
                                aux_lane_num=placement.spawn_lane_num,
                                aux_destination_edge_id=aux_dest_edge,
                                aux_destination_lane_key=aux_dest_lane_key,
                                scenario_id=scenario_id,
                                aux_spawn_longitudinal=float(
                                    placement.spawn_longitudinal
                                ),
                            )
                        )

    return scenarios


def pick_default_roundabout_spawn_meta(
    layout: JunctionPriorityLayout,
    spawn_lanes: Iterable[ApproachSpawnLane],
    *,
    prefer_ego_edge_id: Optional[str] = None,
    min_lane_length: float = 20.0,
    route_index: Optional[VehicleRouteIndex] = None,
) -> Optional[dict]:
    """Pick ego spawn + exit-spoke destination on a roundabout spoke arm."""
    spawn_by_edge = build_spawn_lanes_by_edge(spawn_lanes)
    lane_lengths = lane_lengths_from_spawn_lanes(spawn_lanes)
    lane_keys_by_edge = _lane_keys_lookup(layout)

    ego_candidates = _roundabout_ego_spawn_edges(
        layout, spawn_by_edge, prefer_ego_edge_id=prefer_ego_edge_id
    )

    ego_edge: Optional[str] = None
    if prefer_ego_edge_id and prefer_ego_edge_id in ego_candidates:
        ego_edge = prefer_ego_edge_id
    elif prefer_ego_edge_id:
        prefix = prefer_ego_edge_id.split("#", 1)[0]
        chain = [
            e
            for e in ego_candidates
            if e == prefer_ego_edge_id or e.startswith(prefix + "#")
        ]
        if chain:
            best_length = -1.0
            for edge_id in chain:
                for lane_num in spawn_by_edge.get(edge_id, []):
                    length = lane_lengths.get((edge_id, lane_num), 0.0)
                    if length > best_length:
                        best_length = length
                        ego_edge = edge_id

    if ego_edge is None:
        best_length = -1.0
        for edge_id in ego_candidates:
            for lane_num in spawn_by_edge.get(edge_id, []):
                length = lane_lengths.get((edge_id, lane_num), 0.0)
                if length >= min_lane_length and length > best_length:
                    best_length = length
                    ego_edge = edge_id

    if ego_edge is None:
        return None

    ego_lane = 0
    best_length = -1.0
    for lane_num in spawn_by_edge.get(ego_edge, [0]):
        length = lane_lengths.get((ego_edge, lane_num), min_lane_length)
        if length < min_lane_length:
            continue
        if length > best_length:
            best_length = length
            ego_lane = lane_num

    dest_edges = _roundabout_ego_destination_edges(layout, ego_edge)
    if not dest_edges:
        return None

    dest_edge: Optional[str] = None
    dest_lane_key: Optional[str] = None
    for candidate_dest in dest_edges:
        if candidate_dest == ego_edge:
            continue
        lane_key = _pick_outgoing_lane_key(candidate_dest, ego_lane, lane_keys_by_edge)
        if not _is_valid_departure(ego_edge, ego_lane, candidate_dest, lane_key):
            continue
        if route_index is not None and not route_index.can_reach_edge(
            ego_edge, ego_lane, candidate_dest
        ):
            continue
        dest_edge = candidate_dest
        dest_lane_key = lane_key
        break

    if dest_edge is None or dest_lane_key is None:
        return None

    return {
        "road_id": ego_edge,
        "spawn_lane_num": ego_lane,
        "destination_edge_id": dest_edge,
        "destination_lane_id": dest_lane_key,
    }


def pick_default_roundabout_spawn_meta_for_net(
    net_path: Path,
    *,
    prefer_ego_edge_id: Optional[str] = None,
    min_lane_length: float = 20.0,
    scene_meta: Optional[dict] = None,
) -> Optional[dict]:
    from traffic_bench.eval.engine.map.roundabout_topology import build_roundabout_layout

    meta = scene_meta or {}
    layout = build_roundabout_layout(
        net_path,
        sign_edge_id=prefer_ego_edge_id
        or meta.get("catalog_sign_road_id")
        or meta.get("road_id"),
        **roundabout_meta_ring_kwargs(meta),
    )
    spawn_lanes = parse_intersection_approach_lanes(net_path, min_length=min_lane_length)
    route_index = load_vehicle_route_index(net_path)
    return pick_default_roundabout_spawn_meta(
        layout,
        spawn_lanes,
        prefer_ego_edge_id=prefer_ego_edge_id,
        min_lane_length=min_lane_length,
        route_index=route_index,
    )
