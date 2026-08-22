"""Legal ego/aux approach combinations for T/X priority junctions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from traffic_bench.eval.core.layout.junction_priority_layout import (
    JunctionPriorityLayout,
    build_junction_priority_layout,
    right_arm_for_layout,
)
from traffic_bench.eval.core.scenarios.scene_augmentation import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    ApproachSpawnLane,
    SpawnScenario,
    _append_valid_conflict_scenarios,
    _aux_side_relative_to_ego,
    _ego_destination_edges,
    _is_valid_departure,
    _lane_keys_lookup,
    _pick_outgoing_lane_key,
    build_spawn_lanes_by_edge,
    lane_lengths_from_spawn_lanes,
    parse_intersection_approach_lanes,
)
from traffic_bench.eval.core.sumo.sumo_utils import VehicleRouteIndex, load_vehicle_route_index


def enumerate_spawn_scenarios_equal_priority(
    layout: JunctionPriorityLayout,
    spawn_lanes_by_edge: Dict[str, List[int]],
    *,
    min_lane_length: float = 20.0,
    lane_lengths: Optional[Dict[Tuple[str, int], float]] = None,
    route_index: Optional[VehicleRouteIndex] = None,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
) -> List[SpawnScenario]:
    """Ego on any arm; aux only on the right-hand conflicting arm.

    Same ego/aux destination table as yield, but aux cannot sit on the left
    (so ego-right cases are dropped — no meaningful yield target).
    """
    from traffic_bench.eval.core.scenarios.auxiliary_agent import min_aux_spawn_lane_length

    lane_lengths = lane_lengths or {}
    min_aux_lane_length = min_aux_spawn_lane_length(aux_distance_from_intersection)
    lane_keys_by_edge = _lane_keys_lookup(layout)

    ego_edges = sorted({arm.edge_id for arm in layout.arms})
    scenarios: List[SpawnScenario] = []

    for ego_edge in ego_edges:
        right_arm = right_arm_for_layout(layout, ego_edge)
        if right_arm is None:
            continue

        aux_edge = right_arm.edge_id
        ego_lane_nums = spawn_lanes_by_edge.get(ego_edge, [])
        if not ego_lane_nums:
            continue

        ego_dest_edges = _ego_destination_edges(
            layout, ego_edge, aux_edge_id=aux_edge
        )
        if not ego_dest_edges:
            continue

        aux_lane_nums = [
            lane_num
            for lane_num in spawn_lanes_by_edge.get(aux_edge, [])
            if lane_lengths.get((aux_edge, lane_num), min_lane_length) >= min_aux_lane_length
        ]
        if not aux_lane_nums:
            continue

        for ego_lane in ego_lane_nums:
            if lane_lengths.get((ego_edge, ego_lane), min_lane_length) < min_lane_length:
                continue

            for ego_dest_edge in ego_dest_edges:
                ego_dest_lane_key = _pick_outgoing_lane_key(
                    ego_dest_edge, ego_lane, lane_keys_by_edge
                )
                if not _is_valid_departure(
                    ego_edge, ego_lane, ego_dest_edge, ego_dest_lane_key
                ):
                    continue
                if route_index is not None and not route_index.can_reach_edge(
                    ego_edge, ego_lane, ego_dest_edge
                ):
                    continue

                for aux_lane in aux_lane_nums:
                    _append_valid_conflict_scenarios(
                        scenarios,
                        layout=layout,
                        ego_edge=ego_edge,
                        ego_lane=ego_lane,
                        ego_dest_edge=ego_dest_edge,
                        ego_dest_lane_key=ego_dest_lane_key,
                        aux_edge=aux_edge,
                        aux_lane=aux_lane,
                        lane_keys_by_edge=lane_keys_by_edge,
                        route_index=route_index,
                    )

    return scenarios


def enumerate_spawn_scenarios_yield(
    layout: JunctionPriorityLayout,
    spawn_lanes_by_edge: Dict[str, List[int]],
    *,
    min_lane_length: float = 20.0,
    lane_lengths: Optional[Dict[Tuple[str, int], float]] = None,
    route_index: Optional[VehicleRouteIndex] = None,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
) -> List[SpawnScenario]:
    """Ego on secondary arms; aux on main-road left/right arms (not opposite)."""
    from traffic_bench.eval.core.scenarios.auxiliary_agent import min_aux_spawn_lane_length

    lane_lengths = lane_lengths or {}
    min_aux_lane_length = min_aux_spawn_lane_length(aux_distance_from_intersection)
    lane_keys_by_edge = _lane_keys_lookup(layout)

    secondary_edges = sorted(layout.secondary_edge_ids)
    main_edges = sorted(layout.main_edge_ids)
    scenarios: List[SpawnScenario] = []

    for ego_edge in secondary_edges:
        ego_lane_nums = spawn_lanes_by_edge.get(ego_edge, [])
        if not ego_lane_nums:
            continue

        for aux_edge in main_edges:
            if aux_edge == ego_edge:
                continue
            side = _aux_side_relative_to_ego(layout, ego_edge, aux_edge)
            if side in ("straight", "other"):
                continue

            ego_dest_edges = _ego_destination_edges(
                layout, ego_edge, aux_edge_id=aux_edge
            )
            if not ego_dest_edges:
                continue

            aux_lane_nums = [
                lane_num
                for lane_num in spawn_lanes_by_edge.get(aux_edge, [])
                if lane_lengths.get((aux_edge, lane_num), min_lane_length) >= min_aux_lane_length
            ]
            if not aux_lane_nums:
                continue

            for ego_lane in ego_lane_nums:
                if lane_lengths.get((ego_edge, ego_lane), min_lane_length) < min_lane_length:
                    continue

                for ego_dest_edge in ego_dest_edges:
                    if ego_dest_edge == ego_edge:
                        continue

                    ego_dest_lane_key = _pick_outgoing_lane_key(
                        ego_dest_edge, ego_lane, lane_keys_by_edge
                    )
                    if not _is_valid_departure(
                        ego_edge, ego_lane, ego_dest_edge, ego_dest_lane_key
                    ):
                        continue
                    if route_index is not None and not route_index.can_reach_edge(
                        ego_edge, ego_lane, ego_dest_edge
                    ):
                        continue

                    for aux_lane in aux_lane_nums:
                        _append_valid_conflict_scenarios(
                            scenarios,
                            layout=layout,
                            ego_edge=ego_edge,
                            ego_lane=ego_lane,
                            ego_dest_edge=ego_dest_edge,
                            ego_dest_lane_key=ego_dest_lane_key,
                            aux_edge=aux_edge,
                            aux_lane=aux_lane,
                            lane_keys_by_edge=lane_keys_by_edge,
                            route_index=route_index,
                        )

    return scenarios


def pick_default_main_spawn_meta(
    layout: JunctionPriorityLayout,
    spawn_lanes_by_edge: Dict[str, List[int]],
    lane_lengths: Dict[Tuple[str, int], float],
    *,
    prefer_ego_edge_id: Optional[str] = None,
    min_lane_length: float = 20.0,
    route_index: Optional[VehicleRouteIndex] = None,
) -> Optional[dict]:
    """Pick ego spawn + destination on any equal-priority arm."""
    lane_keys_by_edge = _lane_keys_lookup(layout)
    ego_edges = sorted({arm.edge_id for arm in layout.arms})

    ego_edge: Optional[str] = None
    if prefer_ego_edge_id and prefer_ego_edge_id in ego_edges:
        ego_edge = prefer_ego_edge_id

    if ego_edge is None:
        best_length = -1.0
        for edge_id in ego_edges:
            for lane_num in spawn_lanes_by_edge.get(edge_id, []):
                length = lane_lengths.get((edge_id, lane_num), 0.0)
                if length >= min_lane_length and length > best_length:
                    best_length = length
                    ego_edge = edge_id

    if ego_edge is None:
        return None

    ego_lane = 0
    best_length = -1.0
    for lane_num in spawn_lanes_by_edge.get(ego_edge, [0]):
        length = lane_lengths.get((ego_edge, lane_num), min_lane_length)
        if length < min_lane_length:
            continue
        if length > best_length:
            best_length = length
            ego_lane = lane_num

    dest_edges = _ego_destination_edges(layout, ego_edge)
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


def pick_default_yield_spawn_meta(
    layout: JunctionPriorityLayout,
    spawn_lanes: Iterable[ApproachSpawnLane],
    *,
    prefer_ego_edge_id: Optional[str] = None,
    min_lane_length: float = 20.0,
    route_index: Optional[VehicleRouteIndex] = None,
) -> Optional[dict]:
    """Pick ego spawn + destination on a secondary (yield) arm."""
    spawn_by_edge = build_spawn_lanes_by_edge(spawn_lanes)
    lane_lengths = lane_lengths_from_spawn_lanes(spawn_lanes)
    lane_keys_by_edge = _lane_keys_lookup(layout)

    ego_edge: Optional[str] = None
    if prefer_ego_edge_id and prefer_ego_edge_id in layout.secondary_edge_ids:
        ego_edge = prefer_ego_edge_id

    if ego_edge is None:
        best_length = -1.0
        for edge_id in sorted(layout.secondary_edge_ids):
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

    dest_edges = _ego_destination_edges(layout, ego_edge)
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


def pick_default_main_spawn_meta_for_net(
    net_path: Path,
    *,
    prefer_ego_edge_id: Optional[str] = None,
    min_lane_length: float = 20.0,
    sign_lat: Optional[float] = None,
    sign_lon: Optional[float] = None,
) -> Optional[dict]:
    layout = build_junction_priority_layout(
        net_path,
        mode="main_main",
        sign_lat=sign_lat,
        sign_lon=sign_lon,
    )
    spawn_lanes = parse_intersection_approach_lanes(net_path, min_length=min_lane_length)
    spawn_by_edge = build_spawn_lanes_by_edge(spawn_lanes)
    lane_lengths = lane_lengths_from_spawn_lanes(spawn_lanes)
    route_index = load_vehicle_route_index(net_path)
    return pick_default_main_spawn_meta(
        layout,
        spawn_by_edge,
        lane_lengths,
        prefer_ego_edge_id=prefer_ego_edge_id,
        min_lane_length=min_lane_length,
        route_index=route_index,
    )


def pick_default_yield_spawn_meta_for_net(
    net_path: Path,
    *,
    prefer_ego_edge_id: Optional[str] = None,
    min_lane_length: float = 20.0,
    sign_lat: Optional[float] = None,
    sign_lon: Optional[float] = None,
) -> Optional[dict]:
    layout = build_junction_priority_layout(
        net_path,
        mode="main_secondary",
        sign_lat=sign_lat,
        sign_lon=sign_lon,
    )
    spawn_lanes = parse_intersection_approach_lanes(net_path, min_length=min_lane_length)
    route_index = load_vehicle_route_index(net_path)
    return pick_default_yield_spawn_meta(
        layout,
        spawn_lanes,
        prefer_ego_edge_id=prefer_ego_edge_id,
        min_lane_length=min_lane_length,
        route_index=route_index,
    )
