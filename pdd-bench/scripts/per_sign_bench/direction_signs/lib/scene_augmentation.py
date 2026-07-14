"""Scenario augmentation for junction (direction signs 4.1.x scaffold) intersections."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .junction_priority_layout import (
    INTERSECTION_JUNCTION_TYPES,
    JunctionPriorityLayout,
    build_junction_priority_layout,
    right_arm_for_layout,
)
from .lane_keys import lane_num_from_key, make_lane_key
from .sumo_utils import VehicleRouteIndex, is_vehicle_drivable_lane, load_vehicle_route_index


@dataclass(frozen=True)
class ApproachSpawnLane:
    edge_id: str
    lane_num: int
    length: float


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
    if arm is None:
        return []
    if layout.shape == "T":
        return list(arm.left_to)
    if layout.shape == "X":
        return list(arm.straight_to)
    if layout.shape == "2":
        return list(arm.straight_to) or list(arm.outgoing_to)
    return list(arm.straight_to) or list(arm.left_to)


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
    """Enumerate ego on any arm; aux only on the right-hand conflicting arm."""
    lane_lengths = lane_lengths or {}
    lane_keys_by_edge: Dict[str, List[str]] = {
        arm.edge_id: list(arm.lane_keys) for arm in layout.arms
    }

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

        ego_dest_edges = _ego_destination_edges(layout, ego_edge)
        if not ego_dest_edges:
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
    sign_lat: Optional[float] = None,
    sign_lon: Optional[float] = None,
) -> Tuple[JunctionPriorityLayout, List[SpawnScenario]]:
    """Build equal-priority layout and enumerate right-hand aux scenarios."""
    layout = build_junction_priority_layout(
        net_path,
        mode="main_main",
        sign_lat=sign_lat,
        sign_lon=sign_lon,
    )
    spawn_by_edge = build_spawn_lanes_by_edge(spawn_lanes)
    lengths = lane_lengths_from_spawn_lanes(spawn_lanes)
    scenarios = enumerate_spawn_scenarios(
        layout,
        spawn_by_edge,
        min_lane_length=min_lane_length,
        lane_lengths=lengths,
    )
    return layout, scenarios


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
    lane_keys_by_edge = {arm.edge_id: list(arm.lane_keys) for arm in layout.arms}
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


def parse_intersection_approach_lanes(
    net_path: Path,
    *,
    min_length: float = 20.0,
) -> List[ApproachSpawnLane]:
    """Lanes on edges that approach an intersection junction."""
    if not net_path.is_file():
        return []

    root = ET.parse(net_path).getroot()
    junction_types = {
        junction.get("id"): junction.get("type", "unknown")
        for junction in root.findall("junction")
        if junction.get("id")
    }

    lanes: List[ApproachSpawnLane] = []
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id or edge_id.startswith(":"):
            continue
        if edge.get("function", "normal") == "internal":
            continue

        junction_type = junction_types.get(edge.get("to", ""), "unknown")
        if junction_type not in INTERSECTION_JUNCTION_TYPES:
            continue

        for lane in edge.findall("lane"):
            if not is_vehicle_drivable_lane(lane):
                continue
            lane_id = lane.get("id", "")
            length = float(lane.get("length", 0.0) or 0.0)
            if length <= 0.0:
                shape_str = lane.get("shape", "")
                if shape_str:
                    points = [
                        tuple(map(float, token.split(",")))
                        for token in shape_str.strip().split()
                        if "," in token
                    ]
                    if len(points) >= 2:
                        length = sum(
                            ((points[i + 1][0] - points[i][0]) ** 2
                             + (points[i + 1][1] - points[i][1]) ** 2) ** 0.5
                            for i in range(len(points) - 1)
                        )
            if length < min_length:
                continue

            try:
                lane_num = int(lane_id.rsplit("_", 1)[1])
            except (ValueError, IndexError):
                lane_num = 0

            lanes.append(
                ApproachSpawnLane(
                    edge_id=edge_id,
                    lane_num=lane_num,
                    length=length,
                )
            )
    return lanes


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
