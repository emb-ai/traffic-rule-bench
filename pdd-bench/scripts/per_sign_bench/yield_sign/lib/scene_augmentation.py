"""Scenario augmentation for ego/aux spawn combinations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .junction_priority_layout import JunctionPriorityLayout, build_junction_priority_layout
from .lane_keys import lane_num_from_key, make_lane_key


@dataclass(frozen=True)
class SpawnScenario:
    """One ego/aux spawn + ego destination combination."""

    ego_edge_id: str
    ego_lane_num: int
    ego_destination_edge_id: str
    ego_destination_lane_key: str
    aux_edge_id: str
    aux_lane_num: int
    aux_destination_edge_id: str
    aux_destination_lane_key: str
    scenario_id: str

    def to_manifest_fields(self) -> dict:
        return {
            "road_id": self.ego_edge_id,
            "spawn_lane_num": self.ego_lane_num,
            "destination_lane_id": self.ego_destination_lane_key,
            "destination_edge_id": self.ego_destination_edge_id,
            "aux_road_id": self.aux_edge_id,
            "aux_spawn_lane_num": self.aux_lane_num,
            "aux_spawn_lane_index": _lane_key(self.aux_edge_id, self.aux_lane_num),
            "aux_destination_lane_id": self.aux_destination_lane_key,
            "aux_destination_edge_id": self.aux_destination_edge_id,
            "augmentation_id": self.scenario_id,
        }


def _lane_key(edge_id: str, lane_num: int) -> str:
    return make_lane_key(edge_id, lane_num)


def _pick_outgoing_lane_key(
    edge_id: str,
    lane_num: int,
    lane_keys_by_edge: Dict[str, List[str]],
) -> str:
    keys = lane_keys_by_edge.get(edge_id, [])
    if not keys:
        return _lane_key(edge_id, lane_num)
    for key in keys:
        if lane_num_from_key(key) == lane_num:
            return key
    return keys[min(lane_num, len(keys) - 1)]


def _ego_destination_edges(layout: JunctionPriorityLayout, ego_edge_id: str) -> List[str]:
    arm = layout.arm_for_edge(ego_edge_id)
    if arm is None or not arm.straight_to:
        return []
    return list(arm.straight_to)


def _aux_straight_destination(layout: JunctionPriorityLayout, aux_edge_id: str) -> Optional[str]:
    arm = layout.arm_for_edge(aux_edge_id)
    if arm is None or not arm.straight_to:
        return None
    return arm.straight_to[0]


def _is_valid_departure(
    spawn_edge: str,
    spawn_lane: int,
    dest_edge: str,
    dest_lane_key: str,
) -> bool:
    if spawn_edge == dest_edge:
        return False
    if _lane_key(spawn_edge, spawn_lane) == dest_lane_key:
        return False
    return True


def enumerate_spawn_scenarios(
    layout: JunctionPriorityLayout,
    spawn_lanes_by_edge: Dict[str, List[int]],
    *,
    min_lane_length: float = 20.0,
    lane_lengths: Optional[Dict[Tuple[str, int], float]] = None,
) -> List[SpawnScenario]:
    """Enumerate ego/aux spawn combinations from junction layout arms."""
    lane_lengths = lane_lengths or {}
    lane_keys_by_edge: Dict[str, List[str]] = {
        arm.edge_id: list(arm.lane_keys) for arm in layout.arms
    }

    secondary_edges = sorted(layout.secondary_edge_ids)
    main_edges = sorted(layout.main_edge_ids)
    scenarios: List[SpawnScenario] = []

    for ego_edge in secondary_edges:
        ego_lane_nums = spawn_lanes_by_edge.get(ego_edge, [])
        if not ego_lane_nums:
            continue

        ego_dest_edges = _ego_destination_edges(layout, ego_edge)
        if not ego_dest_edges:
            continue

        for aux_edge in main_edges:
            if aux_edge == ego_edge:
                continue

            aux_dest_edge = _aux_straight_destination(layout, aux_edge)
            if aux_dest_edge is None:
                continue

            aux_lane_nums = spawn_lanes_by_edge.get(aux_edge, [])
            if not aux_lane_nums:
                continue

            for ego_lane in ego_lane_nums:
                if lane_lengths.get((ego_edge, ego_lane), min_lane_length) < min_lane_length:
                    continue

                for ego_dest_edge in ego_dest_edges:
                    ego_dest_lane_key = _pick_outgoing_lane_key(
                        ego_dest_edge,
                        ego_lane,
                        lane_keys_by_edge,
                    )

                    for aux_lane in aux_lane_nums:
                        if lane_lengths.get((aux_edge, aux_lane), min_lane_length) < min_lane_length:
                            continue

                        if not _is_valid_departure(
                            ego_edge,
                            ego_lane,
                            ego_dest_edge,
                            ego_dest_lane_key,
                        ):
                            continue

                        aux_dest_lane_key = _pick_outgoing_lane_key(
                            aux_dest_edge,
                            aux_lane,
                            lane_keys_by_edge,
                        )

                        scenario_id = (
                            f"ego_{ego_edge}_L{ego_lane}"
                            f"_to_{ego_dest_edge}"
                            f"_aux_{aux_edge}_L{aux_lane}"
                        )
                        scenarios.append(
                            SpawnScenario(
                                ego_edge_id=ego_edge,
                                ego_lane_num=ego_lane,
                                ego_destination_edge_id=ego_dest_edge,
                                ego_destination_lane_key=ego_dest_lane_key,
                                aux_edge_id=aux_edge,
                                aux_lane_num=aux_lane,
                                aux_destination_edge_id=aux_dest_edge,
                                aux_destination_lane_key=aux_dest_lane_key,
                                scenario_id=scenario_id,
                            )
                        )

    return scenarios


def build_spawn_lanes_by_edge(
    spawn_lanes: Iterable,
) -> Dict[str, List[int]]:
    """Group parsed SumoLaneInfo rows by edge id."""
    by_edge: Dict[str, set[int]] = {}
    for lane in spawn_lanes:
        by_edge.setdefault(lane.edge_id, set()).add(lane.lane_num)
    return {edge: sorted(nums) for edge, nums in sorted(by_edge.items())}


def lane_lengths_from_spawn_lanes(spawn_lanes: Iterable) -> Dict[Tuple[str, int], float]:
    return {(lane.edge_id, lane.lane_num): float(lane.length) for lane in spawn_lanes}


def augment_layout_for_scene(
    net_path: Path,
    spawn_lanes: Iterable,
    *,
    min_lane_length: float = 20.0,
) -> Tuple[JunctionPriorityLayout, List[SpawnScenario]]:
    """Build layout and enumerate augmented scenarios for one scene."""
    layout = build_junction_priority_layout(net_path)
    spawn_by_edge = build_spawn_lanes_by_edge(spawn_lanes)
    lengths = lane_lengths_from_spawn_lanes(spawn_lanes)
    scenarios = enumerate_spawn_scenarios(
        layout,
        spawn_by_edge,
        min_lane_length=min_lane_length,
        lane_lengths=lengths,
    )
    return layout, scenarios
