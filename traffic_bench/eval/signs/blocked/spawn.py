"""Legal ego spawn × forbidden-exit combinations for 3.2."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from traffic_bench.eval.core.layout.junction_priority_layout import JunctionPriorityLayout
from traffic_bench.eval.core.scenarios.scene_augmentation import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    SpawnScenario,
    _filter_real_destination_edges,
    _is_valid_departure,
    _lane_keys_lookup,
    _pick_outgoing_lane_key,
)
from traffic_bench.eval.core.sumo.sumo_utils import VehicleRouteIndex


def _blocked_road_destination_edges(
    layout: JunctionPriorityLayout,
    ego_edge_id: str,
) -> List[str]:
    """All reachable exits from the ego approach (forbidden lane = dest edge).

    Spawn arm and lane are enumerated separately; here we list destination arms:
    typically **T → 2** exits and **X → 3** (left / right / straight), excluding
    U-turn back onto the same edge. Sign 3.2 is placed on this destination edge.
    """
    arm = layout.arm_for_edge(ego_edge_id)
    if arm is None:
        return []
    raw: List[str] = []
    for bucket in (arm.left_to, arm.right_to, arm.straight_to):
        for edge_id in bucket:
            if edge_id not in raw:
                raw.append(edge_id)
    if not raw:
        raw = list(arm.outgoing_to)
    return [
        edge_id
        for edge_id in _filter_real_destination_edges(raw)
        if edge_id != ego_edge_id
    ]


def enumerate_spawn_scenarios_blocked_road(
    layout: JunctionPriorityLayout,
    spawn_lanes_by_edge: Dict[str, List[int]],
    *,
    min_lane_length: float = 20.0,
    lane_lengths: Optional[Dict[Tuple[str, int], float]] = None,
    route_index: Optional[VehicleRouteIndex] = None,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
) -> List[SpawnScenario]:
    """Enumerate ego spawn arm × lane × destination arm (no aux).

    - spawn arm: every approach arm on the junction
    - lane: every spawnable lane on that arm
    - destination: every reachable other exit (T≈2, X≈3); 3.2 sits on that exit
    """
    del aux_distance_from_intersection  # no aux for blocked-road
    lane_lengths = lane_lengths or {}
    lane_keys_by_edge = _lane_keys_lookup(layout)
    ego_edges = sorted({arm.edge_id for arm in layout.arms})
    scenarios: List[SpawnScenario] = []

    for ego_edge in ego_edges:
        ego_lane_nums = spawn_lanes_by_edge.get(ego_edge, [])
        if not ego_lane_nums:
            continue
        ego_dest_edges = _blocked_road_destination_edges(layout, ego_edge)
        if not ego_dest_edges:
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
                scenario_id = f"ego_{ego_edge}_L{ego_lane}_to_{ego_dest_edge}"
                scenarios.append(
                    SpawnScenario(
                        ego_edge_id=ego_edge,
                        ego_lane_num=ego_lane,
                        ego_destination_edge_id=ego_dest_edge,
                        ego_destination_lane_key=ego_dest_lane_key,
                        aux_edge_id="",
                        aux_lane_num=0,
                        aux_destination_edge_id="",
                        aux_destination_lane_key="",
                        scenario_id=scenario_id,
                    )
                )
    return scenarios
