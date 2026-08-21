"""Scenario augmentation for priority-junction benches (equal-priority + yield)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from ..layout.junction_priority_layout import (
    INTERSECTION_JUNCTION_TYPES,
    JunctionPriorityLayout,
    build_junction_priority_layout,
    left_arm_for_layout,
    right_arm_for_layout,
    straight_arm_for_layout,
)
from ..sumo.lane_keys import make_lane_key, pick_lane_key_on_edge
from ..sumo.sumo_utils import VehicleRouteIndex, is_vehicle_drivable_lane, load_vehicle_route_index

DEFAULT_AUX_DISTANCE_FROM_INTERSECTION = 20.0

SpawnStrategy = Literal[
    "equal_priority",
    "yield",
    "roundabout",
    "blocked_road",
    "one_way",
    "direction",
    "no_turn",
    "no_entry",
    "crosswalk",
    "detour",
    "speed_zone",
]
EgoManeuver = Literal["left", "right", "straight"]
AuxSide = Literal["left", "right", "straight", "other"]


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
    aux_spawn_longitudinal: Optional[float] = None

    def to_manifest_fields(self) -> dict:
        fields = {
            "road_id": self.ego_edge_id,
            "spawn_lane_num": self.ego_lane_num,
            "destination_lane_id": self.ego_destination_lane_key,
            "destination_edge_id": self.ego_destination_edge_id,
            "augmentation_id": self.scenario_id,
        }
        if self.aux_edge_id:
            fields.update(
                {
                    "aux_road_id": self.aux_edge_id,
                    "aux_spawn_lane_num": self.aux_lane_num,
                    "aux_spawn_lane_index": _lane_key(self.aux_edge_id, self.aux_lane_num),
                    "aux_destination_lane_id": self.aux_destination_lane_key,
                    "aux_destination_edge_id": self.aux_destination_edge_id,
                }
            )
            if self.aux_spawn_longitudinal is not None:
                fields["aux_spawn_longitudinal"] = float(self.aux_spawn_longitudinal)
        return fields


def _lane_key(edge_id: str, lane_num: int) -> str:
    return make_lane_key(edge_id, lane_num)


def _pick_outgoing_lane_key(
    edge_id: str,
    lane_num: int,
    lane_keys_by_edge: Dict[str, List[str]],
) -> str:
    return pick_lane_key_on_edge(edge_id, lane_num, lane_keys_by_edge)


def _lane_keys_lookup(layout: JunctionPriorityLayout) -> Dict[str, List[str]]:
    """Prefer full-net lane map; fall back to incoming arms only."""
    if layout.lane_keys_by_edge:
        return {edge_id: list(keys) for edge_id, keys in layout.lane_keys_by_edge.items()}
    return {arm.edge_id: list(arm.lane_keys) for arm in layout.arms}


def _is_real_edge_id(edge_id: str) -> bool:
    return bool(edge_id) and not str(edge_id).startswith(":")


def _filter_real_destination_edges(edge_ids: Iterable[str]) -> List[str]:
    return [edge_id for edge_id in edge_ids if _is_real_edge_id(edge_id)]


def _ego_destination_edges(
    layout: JunctionPriorityLayout,
    ego_edge_id: str,
    *,
    aux_edge_id: Optional[str] = None,
) -> List[str]:
    """Ego destinations allowed for this ego(+optional aux) conflict.

    Base set by junction shape; with ``aux_edge_id`` the allow-list follows the
    yield/main conflict table (no ego-right when aux is only on the right;
    no aux on the opposite/straight arm).
    """
    arm = layout.arm_for_edge(ego_edge_id)
    if arm is None:
        return []

    if aux_edge_id is None:
        # Defaults for pickers / viability without a chosen aux.
        if layout.shape == "T":
            candidates = _filter_real_destination_edges(arm.left_to)
        elif layout.shape == "X":
            candidates = _filter_real_destination_edges(
                list(arm.straight_to) + list(arm.left_to)
            )
        elif layout.shape == "2":
            candidates = _filter_real_destination_edges(
                list(arm.straight_to) or list(arm.outgoing_to)
            )
        else:
            candidates = _filter_real_destination_edges(
                list(arm.straight_to) or list(arm.left_to)
            )
        return [e for e in candidates if e != ego_edge_id]

    side = _aux_side_relative_to_ego(layout, ego_edge_id, aux_edge_id)
    if side == "straight" or side == "other":
        return []
    if side == "right":
        # Ego may not turn into the right arm when the only conflict is also
        # on the right (nobody to yield to on that path).
        candidates = _filter_real_destination_edges(arm.left_to)
        if layout.shape == "X":
            for e in _filter_real_destination_edges(arm.straight_to):
                if e not in candidates:
                    candidates.append(e)
    else:  # aux on left — ego may turn left, right, and (on X) go straight
        candidates = _filter_real_destination_edges(arm.left_to)
        for e in _filter_real_destination_edges(arm.right_to):
            if e not in candidates:
                candidates.append(e)
        if layout.shape == "X":
            for e in _filter_real_destination_edges(arm.straight_to):
                if e not in candidates:
                    candidates.append(e)
    return [e for e in candidates if e != ego_edge_id]


def _aux_side_relative_to_ego(
    layout: JunctionPriorityLayout,
    ego_edge_id: str,
    aux_edge_id: str,
) -> AuxSide:
    left = left_arm_for_layout(layout, ego_edge_id)
    right = right_arm_for_layout(layout, ego_edge_id)
    straight = straight_arm_for_layout(layout, ego_edge_id)
    if left is not None and left.edge_id == aux_edge_id:
        return "left"
    if right is not None and right.edge_id == aux_edge_id:
        return "right"
    if straight is not None and straight.edge_id == aux_edge_id:
        return "straight"
    return "other"


def _ego_maneuver_for_destination(
    layout: JunctionPriorityLayout,
    ego_edge_id: str,
    ego_dest_edge: str,
) -> Optional[EgoManeuver]:
    arm = layout.arm_for_edge(ego_edge_id)
    if arm is None:
        return None
    if ego_dest_edge in set(arm.right_to):
        return "right"
    if ego_dest_edge in set(arm.left_to):
        return "left"
    if ego_dest_edge in set(arm.straight_to):
        return "straight"
    return None


def _aux_turn_destination_edges(
    layout: JunctionPriorityLayout,
    aux_edge_id: str,
    turn: EgoManeuver,
) -> List[str]:
    arm = layout.arm_for_edge(aux_edge_id)
    if arm is None:
        return []
    if turn == "straight":
        raw = arm.straight_to
    elif turn == "left":
        raw = arm.left_to
    else:
        raw = arm.right_to
    return _filter_real_destination_edges(raw)


def _allowed_aux_destination_edges(
    layout: JunctionPriorityLayout,
    ego_edge_id: str,
    ego_dest_edge: str,
    aux_edge_id: str,
) -> List[str]:
    """Aux exit edges allowed for this (ego maneuver × aux side) conflict.

    T / X conflict table (aux never on the opposite/straight arm)::

    - Ego right → aux must be on left, aux goes straight only.
    - Ego left → aux on right: straight or left; aux on left: straight only.
    - Ego straight (X) → aux on right: straight/left/right; aux on left: straight/left.
    """
    side = _aux_side_relative_to_ego(layout, ego_edge_id, aux_edge_id)
    maneuver = _ego_maneuver_for_destination(layout, ego_edge_id, ego_dest_edge)
    if maneuver is None or side in ("straight", "other"):
        return []

    turns: List[EgoManeuver] = []
    if maneuver == "right":
        if side != "left":
            return []
        turns = ["straight"]
    elif maneuver == "left":
        if side == "right":
            turns = ["straight", "left"]
        elif side == "left":
            turns = ["straight"]
        else:
            return []
    elif maneuver == "straight":
        if side == "right":
            turns = ["straight", "left", "right"]
        elif side == "left":
            turns = ["straight", "left"]
        else:
            return []

    out: List[str] = []
    for turn in turns:
        for edge_id in _aux_turn_destination_edges(layout, aux_edge_id, turn):
            if edge_id != aux_edge_id and edge_id not in out:
                out.append(edge_id)
    return out


def _aux_straight_destination(layout: JunctionPriorityLayout, aux_edge_id: str) -> Optional[str]:
    """Legacy helper: first straight-through exit (used by viability diagnostics)."""
    dests = _aux_turn_destination_edges(layout, aux_edge_id, "straight")
    return dests[0] if dests else None


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


def build_spawn_lanes_by_edge(
    spawn_lanes: Iterable,
) -> Dict[str, List[int]]:
    by_edge: Dict[str, set[int]] = {}
    for lane in spawn_lanes:
        by_edge.setdefault(lane.edge_id, set()).add(lane.lane_num)
    return {edge: sorted(nums) for edge, nums in sorted(by_edge.items())}


def lane_lengths_from_spawn_lanes(spawn_lanes: Iterable) -> Dict[Tuple[str, int], float]:
    return {(lane.edge_id, lane.lane_num): float(lane.length) for lane in spawn_lanes}


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


def _append_scenario(
    scenarios: List[SpawnScenario],
    *,
    ego_edge: str,
    ego_lane: int,
    ego_dest_edge: str,
    ego_dest_lane_key: str,
    aux_edge: str,
    aux_lane: int,
    aux_dest_edge: str,
    lane_keys_by_edge: Dict[str, List[str]],
) -> None:
    aux_dest_lane_key = _pick_outgoing_lane_key(aux_dest_edge, aux_lane, lane_keys_by_edge)
    scenario_id = (
        f"ego_{ego_edge}_L{ego_lane}"
        f"_to_{ego_dest_edge}"
        f"_aux_{aux_edge}_L{aux_lane}"
        f"_to_{aux_dest_edge}"
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


def _append_valid_conflict_scenarios(
    scenarios: List[SpawnScenario],
    *,
    layout: JunctionPriorityLayout,
    ego_edge: str,
    ego_lane: int,
    ego_dest_edge: str,
    ego_dest_lane_key: str,
    aux_edge: str,
    aux_lane: int,
    lane_keys_by_edge: Dict[str, List[str]],
    route_index: Optional[VehicleRouteIndex],
) -> None:
    """Expand one ego/aux lane pair across allowed aux destinations."""
    for aux_dest_edge in _allowed_aux_destination_edges(
        layout, ego_edge, ego_dest_edge, aux_edge
    ):
        aux_dest_lane_key = _pick_outgoing_lane_key(
            aux_dest_edge, aux_lane, lane_keys_by_edge
        )
        if not _is_valid_departure(aux_edge, aux_lane, aux_dest_edge, aux_dest_lane_key):
            continue
        if route_index is not None and not route_index.can_reach_edge(
            aux_edge, aux_lane, aux_dest_edge
        ):
            continue
        _append_scenario(
            scenarios,
            ego_edge=ego_edge,
            ego_lane=ego_lane,
            ego_dest_edge=ego_dest_edge,
            ego_dest_lane_key=ego_dest_lane_key,
            aux_edge=aux_edge,
            aux_lane=aux_lane,
            aux_dest_edge=aux_dest_edge,
            lane_keys_by_edge=lane_keys_by_edge,
        )


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
    from .auxiliary_agent import min_aux_spawn_lane_length

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
    from .auxiliary_agent import min_aux_spawn_lane_length

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
            # Opposite (straight) main arm is never a conflict agent for these rules.
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
    from ..sumo.lane_keys import lane_num_from_key, pick_lane_key_on_edge
    from .roundabout_aux import (
        merge_lane_lengths_from_layout,
        resolve_aux_spawn_placement,
    )
    from ..layout.roundabout_yield_zone import entry_conflict_ring_edges

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

        # Left of ego spoke: ring edges ending at ego's entry junction.
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
                        # Prefer same exit as ego; if this lane cannot reach it,
                        # still keep the scenario — runtime ring-circulates.
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


def enumerate_spawn_scenarios(
    layout: JunctionPriorityLayout,
    spawn_lanes_by_edge: Dict[str, List[int]],
    *,
    strategy: SpawnStrategy = "equal_priority",
    min_lane_length: float = 20.0,
    lane_lengths: Optional[Dict[Tuple[str, int], float]] = None,
    route_index: Optional[VehicleRouteIndex] = None,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
) -> List[SpawnScenario]:
    if strategy == "roundabout":
        return enumerate_spawn_scenarios_roundabout(
            layout,
            spawn_lanes_by_edge,
            min_lane_length=min_lane_length,
            lane_lengths=lane_lengths,
            route_index=route_index,
            aux_distance_from_intersection=aux_distance_from_intersection,
        )
    if strategy == "yield":
        return enumerate_spawn_scenarios_yield(
            layout,
            spawn_lanes_by_edge,
            min_lane_length=min_lane_length,
            lane_lengths=lane_lengths,
            route_index=route_index,
            aux_distance_from_intersection=aux_distance_from_intersection,
        )
    if strategy == "blocked_road":
        return enumerate_spawn_scenarios_blocked_road(
            layout,
            spawn_lanes_by_edge,
            min_lane_length=min_lane_length,
            lane_lengths=lane_lengths,
            route_index=route_index,
        )
    if strategy == "one_way":
        # Dual-path discovery lives in one_way_expansion / one_way_bridge —
        # not through-path arm enumeration.
        return []
    if strategy == "direction":
        # Dual-path discovery lives in direction_expansion / direction_bridge.
        return []
    if strategy == "no_turn":
        # Dual-path discovery lives in no_turn_expansion / no_turn_bridge.
        return []
    if strategy == "no_entry":
        # Dual-path discovery lives in no_entry_expansion / no_entry_bridge.
        return []
    return enumerate_spawn_scenarios_equal_priority(
        layout,
        spawn_lanes_by_edge,
        min_lane_length=min_lane_length,
        lane_lengths=lane_lengths,
        route_index=route_index,
        aux_distance_from_intersection=aux_distance_from_intersection,
    )


def _roundabout_meta_ring_kwargs(scene_meta: Optional[dict]) -> dict:
    """Map moscow / legacy roundabout meta keys into build_roundabout_layout kwargs."""
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


def augment_layout_for_scene(
    net_path: Path,
    spawn_lanes: Iterable,
    *,
    strategy: SpawnStrategy = "equal_priority",
    min_lane_length: float = 20.0,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    sign_lat: Optional[float] = None,
    sign_lon: Optional[float] = None,
    scene_meta: Optional[dict] = None,
) -> Tuple[JunctionPriorityLayout, List[SpawnScenario]]:
    """Build layout and enumerate scenarios for one scene."""
    if strategy == "roundabout":
        from ..layout.junction_priority_layout import JunctionLayoutError
        from ..layout.roundabout_topology import build_roundabout_layout

        meta = scene_meta or {}
        prefer_ego = meta.get("catalog_sign_road_id") or meta.get("road_id")
        layout = build_roundabout_layout(
            net_path,
            sign_edge_id=prefer_ego,
            **_roundabout_meta_ring_kwargs(meta),
        )
        if layout.shape != "O" or layout.mode != "roundabout":
            raise JunctionLayoutError(
                f"Expected roundabout (O) layout, got shape={layout.shape!r} mode={layout.mode!r}"
            )
    else:
        mode = "main_secondary" if strategy == "yield" else "main_main"
        layout = build_junction_priority_layout(
            net_path,
            mode=mode,
            sign_lat=sign_lat,
            sign_lon=sign_lon,
        )
    spawn_by_edge = build_spawn_lanes_by_edge(spawn_lanes)
    lengths = lane_lengths_from_spawn_lanes(spawn_lanes)
    route_index = load_vehicle_route_index(net_path)
    scenarios = enumerate_spawn_scenarios(
        layout,
        spawn_by_edge,
        strategy=strategy,
        min_lane_length=min_lane_length,
        lane_lengths=lengths,
        route_index=route_index,
        aux_distance_from_intersection=aux_distance_from_intersection,
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
    from ..layout.roundabout_topology import build_roundabout_layout

    meta = scene_meta or {}
    layout = build_roundabout_layout(
        net_path,
        sign_edge_id=prefer_ego_edge_id
        or meta.get("catalog_sign_road_id")
        or meta.get("road_id"),
        **_roundabout_meta_ring_kwargs(meta),
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
