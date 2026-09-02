"""Shared spawn types and helpers. Family enumerators live in ``signs/*/spawn.py``."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from traffic_bench.eval.engine.map.junction_priority_layout import (
    INTERSECTION_JUNCTION_TYPES,
    JunctionPriorityLayout,
    build_junction_priority_layout,
    left_arm_for_layout,
    right_arm_for_layout,
    straight_arm_for_layout,
)
from traffic_bench.eval.engine.map.lane_keys import make_lane_key, pick_lane_key_on_edge
from traffic_bench.eval.engine.map.sumo_utils import VehicleRouteIndex, is_vehicle_drivable_lane, load_vehicle_route_index

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
    *,
    allowed_lane_nums: Optional[Sequence[int]] = None,
) -> Optional[str]:
    """Pick a lane on ``edge_id``, optionally restricted to SUMO-reachable lanes."""
    return pick_lane_key_on_edge(
        edge_id,
        lane_num,
        lane_keys_by_edge,
        allowed_lane_nums=allowed_lane_nums,
    )


def _pick_reachable_dest_lane_key(
    dest_edge: str,
    preferred_lane_num: int,
    lane_keys_by_edge: Dict[str, List[str]],
    *,
    route_index: Optional[Any],
    from_edge: str,
    from_lane: int,
) -> Optional[str]:
    """Dest lane that a vehicle can actually reach from ``from_edge``/``from_lane``.

    Prefer ``preferred_lane_num`` when it is reachable; otherwise the nearest
    reachable lane. Returns None when the dest edge is unreachable.
    """
    allowed = None
    if route_index is not None:
        allowed = sorted(
            route_index.reachable_lanes_on_edge(
                str(from_edge), int(from_lane), str(dest_edge)
            )
        )
        if not allowed:
            return None
    return _pick_outgoing_lane_key(
        dest_edge,
        preferred_lane_num,
        lane_keys_by_edge,
        allowed_lane_nums=allowed,
    )


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
    kwargs = dict(
        min_lane_length=min_lane_length,
        lane_lengths=lane_lengths,
        route_index=route_index,
        aux_distance_from_intersection=aux_distance_from_intersection,
    )
    if strategy == "roundabout":
        from traffic_bench.eval.signs.roundabout.spawn import (
            enumerate_spawn_scenarios_roundabout,
        )
        return enumerate_spawn_scenarios_roundabout(
            layout, spawn_lanes_by_edge, **kwargs
        )
    if strategy == "yield":
        from traffic_bench.eval.signs.junction.spawn import (
            enumerate_spawn_scenarios_yield,
        )
        return enumerate_spawn_scenarios_yield(layout, spawn_lanes_by_edge, **kwargs)
    if strategy == "blocked_road":
        from traffic_bench.eval.signs.blocked.spawn import (
            enumerate_spawn_scenarios_blocked_road,
        )
        return enumerate_spawn_scenarios_blocked_road(
            layout, spawn_lanes_by_edge, **kwargs
        )
    if strategy in ("one_way", "direction", "no_turn", "no_entry"):
        # Dual-path discovery lives in signs/dual_path, not through-path arms.
        return []
    from traffic_bench.eval.signs.junction.spawn import (
        enumerate_spawn_scenarios_equal_priority,
    )
    return enumerate_spawn_scenarios_equal_priority(
        layout, spawn_lanes_by_edge, **kwargs
    )


def _roundabout_meta_ring_kwargs(scene_meta: Optional[dict]) -> dict:
    from traffic_bench.eval.signs.roundabout.spawn import roundabout_meta_ring_kwargs

    return roundabout_meta_ring_kwargs(scene_meta)


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
        from traffic_bench.eval.engine.map.junction_priority_layout import JunctionLayoutError
        from traffic_bench.eval.engine.map.roundabout_topology import build_roundabout_layout

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


def enumerate_spawn_scenarios_equal_priority(*args, **kwargs):
    from traffic_bench.eval.signs.junction.spawn import (
        enumerate_spawn_scenarios_equal_priority as _impl,
    )
    return _impl(*args, **kwargs)


def enumerate_spawn_scenarios_yield(*args, **kwargs):
    from traffic_bench.eval.signs.junction.spawn import (
        enumerate_spawn_scenarios_yield as _impl,
    )
    return _impl(*args, **kwargs)


def enumerate_spawn_scenarios_roundabout(*args, **kwargs):
    from traffic_bench.eval.signs.roundabout.spawn import (
        enumerate_spawn_scenarios_roundabout as _impl,
    )
    return _impl(*args, **kwargs)


def enumerate_spawn_scenarios_blocked_road(*args, **kwargs):
    from traffic_bench.eval.signs.blocked.spawn import (
        enumerate_spawn_scenarios_blocked_road as _impl,
    )
    return _impl(*args, **kwargs)


def pick_default_main_spawn_meta(*args, **kwargs):
    from traffic_bench.eval.signs.junction.spawn import pick_default_main_spawn_meta as _impl
    return _impl(*args, **kwargs)


def pick_default_yield_spawn_meta(*args, **kwargs):
    from traffic_bench.eval.signs.junction.spawn import pick_default_yield_spawn_meta as _impl
    return _impl(*args, **kwargs)


def pick_default_main_spawn_meta_for_net(*args, **kwargs):
    from traffic_bench.eval.signs.junction.spawn import (
        pick_default_main_spawn_meta_for_net as _impl,
    )
    return _impl(*args, **kwargs)


def pick_default_yield_spawn_meta_for_net(*args, **kwargs):
    from traffic_bench.eval.signs.junction.spawn import (
        pick_default_yield_spawn_meta_for_net as _impl,
    )
    return _impl(*args, **kwargs)


def pick_default_roundabout_spawn_meta(*args, **kwargs):
    from traffic_bench.eval.signs.roundabout.spawn import (
        pick_default_roundabout_spawn_meta as _impl,
    )
    return _impl(*args, **kwargs)


def pick_default_roundabout_spawn_meta_for_net(*args, **kwargs):
    from traffic_bench.eval.signs.roundabout.spawn import (
        pick_default_roundabout_spawn_meta_for_net as _impl,
    )
    return _impl(*args, **kwargs)
