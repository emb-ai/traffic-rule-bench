"""Conflict-arc aux placement helpers for roundabout (4.3) layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .lane_keys import lane_num_from_key, make_lane_key

MIN_SPAWN_LONGITUDE_M = 3.0
MAX_UPSTREAM_HOPS = 8


def _layout_arms(junction_layout: Optional[dict]) -> List[dict]:
    if not junction_layout:
        return []
    return list(junction_layout.get("arms") or [])


def _arm_for_edge(junction_layout: Optional[dict], edge_id: str) -> Optional[dict]:
    for arm in _layout_arms(junction_layout):
        if arm.get("edge_id") == edge_id:
            return arm
    return None


def upstream_ring_arm(
    junction_layout: Optional[dict],
    edge_id: str,
) -> Optional[dict]:
    """Ring segment immediately upstream of ``edge_id`` (feeds into its ``from_node``)."""
    arm = _arm_for_edge(junction_layout, edge_id)
    if arm is None or arm.get("road_class") != "main":
        return None
    from_node = str(arm.get("from_node", ""))
    if not from_node:
        return None
    upstream: List[dict] = []
    for candidate in _layout_arms(junction_layout):
        if candidate.get("road_class") != "main":
            continue
        if str(candidate.get("to_node", "")) == from_node:
            upstream.append(candidate)
    if not upstream:
        return None
    return max(upstream, key=lambda item: float(item.get("min_lane_length", 0.0) or 0.0))


def lane_length_for_spawn(
    edge_id: str,
    lane_num: int,
    lane_lengths: Dict[Tuple[str, int], float],
    junction_layout: Optional[dict],
) -> float:
    length = float(lane_lengths.get((edge_id, lane_num), 0.0) or 0.0)
    if length > 0.0:
        return length
    arm = _arm_for_edge(junction_layout, edge_id)
    if arm is not None:
        return float(arm.get("min_lane_length", 0.0) or 0.0)
    return 0.0


@dataclass(frozen=True)
class AuxSpawnPlacement:
    """Resolved aux spawn lane and longitudinal offset along it."""

    spawn_edge_id: str
    spawn_lane_num: int
    spawn_longitudinal: float
    conflict_edge_id: str
    conflict_lane_num: int

    @property
    def spawn_lane_key(self) -> str:
        return make_lane_key(self.spawn_edge_id, self.spawn_lane_num)


def _upstream_chain(
    junction_layout: Optional[dict],
    edge_id: str,
    lane_num: int,
    lane_lengths: Dict[Tuple[str, int], float],
    *,
    max_hops: int = MAX_UPSTREAM_HOPS,
) -> List[Tuple[str, float]]:
    """Conflict edge plus upstream ring edges with positive length."""
    length0 = lane_length_for_spawn(edge_id, lane_num, lane_lengths, junction_layout)
    if length0 <= 0.0:
        return []
    chain: List[Tuple[str, float]] = [(edge_id, length0)]
    seen = {edge_id}
    current = edge_id
    for _ in range(max(0, int(max_hops))):
        upstream = upstream_ring_arm(junction_layout, current)
        if upstream is None:
            break
        up_edge = str(upstream.get("edge_id", ""))
        if not up_edge or up_edge in seen:
            break
        up_len = lane_length_for_spawn(up_edge, lane_num, lane_lengths, junction_layout)
        if up_len <= 0.0:
            break
        chain.append((up_edge, up_len))
        seen.add(up_edge)
        current = up_edge
    return chain


def resolve_aux_spawn_placement(
    junction_layout: Optional[dict],
    edge_id: str,
    lane_num: int,
    lane_lengths: Dict[Tuple[str, int], float],
    aux_distance_from_intersection: float,
    *,
    allowed_ring_edges: Optional[set[str]] = None,
) -> Optional[AuxSpawnPlacement]:
    """Place aux ``aux_distance`` before the entry along the ring.

    ``edge_id`` is the left-hand conflict segment at ego's entry. When that
    segment is shorter than ``aux_distance``, walk further upstream on the
    circle (multiple hops) until the offset fits. Never returns a longitudinal
    past the end of the spawn lane.
    """
    if allowed_ring_edges is not None and edge_id not in allowed_ring_edges:
        return None

    chain = _upstream_chain(junction_layout, edge_id, lane_num, lane_lengths)
    if not chain:
        return None

    total = sum(length for _, length in chain)
    if total < MIN_SPAWN_LONGITUDE_M + 0.5:
        return None

    # How far before the junction to sit; clamp to what the chain can support.
    target_back = min(
        float(aux_distance_from_intersection),
        total - MIN_SPAWN_LONGITUDE_M,
    )
    if target_back < 0.5:
        return None

    remaining = target_back
    for spawn_edge, spawn_len in chain:
        if remaining <= spawn_len - 0.1:
            spawn_long = spawn_len - remaining
            if spawn_long < MIN_SPAWN_LONGITUDE_M and spawn_len >= MIN_SPAWN_LONGITUDE_M + 0.5:
                spawn_long = MIN_SPAWN_LONGITUDE_M
            spawn_long = min(max(spawn_long, 0.5), spawn_len - 0.1)
            if spawn_long <= 0.0 or spawn_long >= spawn_len:
                remaining -= spawn_len
                continue
            return AuxSpawnPlacement(
                spawn_edge_id=spawn_edge,
                spawn_lane_num=lane_num,
                spawn_longitudinal=float(spawn_long),
                conflict_edge_id=edge_id,
                conflict_lane_num=lane_num,
            )
        remaining -= spawn_len

    # Exhausted chain: sit near the start of the farthest upstream edge.
    spawn_edge, spawn_len = chain[-1]
    spawn_long = min(max(MIN_SPAWN_LONGITUDE_M, 0.5), spawn_len - 0.1)
    if spawn_long <= 0.0:
        return None
    return AuxSpawnPlacement(
        spawn_edge_id=spawn_edge,
        spawn_lane_num=lane_num,
        spawn_longitudinal=float(spawn_long),
        conflict_edge_id=edge_id,
        conflict_lane_num=lane_num,
    )


def is_aux_lane_viable_with_ring_extension(
    junction_layout: Optional[dict],
    edge_id: str,
    lane_num: int,
    lane_lengths: Dict[Tuple[str, int], float],
    aux_distance_from_intersection: float,
    *,
    allowed_ring_edges: Optional[set[str]] = None,
) -> bool:
    return (
        resolve_aux_spawn_placement(
            junction_layout,
            edge_id,
            lane_num,
            lane_lengths,
            aux_distance_from_intersection,
            allowed_ring_edges=allowed_ring_edges,
        )
        is not None
    )


def merge_lane_lengths_from_layout(
    junction_layout: Optional[dict],
    lane_lengths: Dict[Tuple[str, int], float],
) -> Dict[Tuple[str, int], float]:
    """Fill missing (edge, lane) lengths from junction arm minima."""
    merged = dict(lane_lengths)
    for arm in _layout_arms(junction_layout):
        edge_id = str(arm.get("edge_id", ""))
        min_len = float(arm.get("min_lane_length", 0.0) or 0.0)
        if not edge_id or min_len <= 0.0:
            continue
        for lane_key in arm.get("lane_keys", []):
            lane_num = lane_num_from_key(str(lane_key))
            merged.setdefault((edge_id, lane_num), min_len)
    return merged
