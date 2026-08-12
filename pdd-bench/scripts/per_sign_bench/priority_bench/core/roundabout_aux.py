"""Conflict-arc aux placement helpers for roundabout (4.3) layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .lane_keys import lane_num_from_key, make_lane_key

MIN_SPAWN_LONGITUDE_M = 3.0
# Conflict arcs shorter than this are rejected at scene generation.
# On arcs in [MIN_CONFLICT_ARC_LENGTH_M, aux_distance+MIN_SPAWN) the offset
# is clamped to the edge (still no upstream walk for the *lead*).
MIN_CONFLICT_ARC_LENGTH_M = 15.0
# Lead never walks upstream. Followers may spill onto this many upstream
# ring hops when the conflict arc cannot hold the full convoy gap chain.
MAX_CONVOY_SPILLOVER_HOPS = 3
# Legacy alias kept for callers that still pass max_upstream_hops.
MAX_UPSTREAM_HOPS = 0


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


@dataclass(frozen=True)
class ConvoySpawnSlot:
    """One vehicle in an aux convoy (lead or spillover follower)."""

    spawn_edge_id: str
    spawn_lane_num: int
    spawn_longitudinal: float
    convoy_index: int
    conflict_edge_id: str
    conflict_lane_num: int

    @property
    def spawn_lane_key(self) -> str:
        return make_lane_key(self.spawn_edge_id, self.spawn_lane_num)

    @property
    def is_spillover(self) -> bool:
        return self.spawn_edge_id != self.conflict_edge_id


def _upstream_ring_chain(
    junction_layout: Optional[dict],
    edge_id: str,
    lane_num: int,
    lane_lengths: Dict[Tuple[str, int], float],
    *,
    max_hops: int,
) -> List[Tuple[str, float]]:
    """Conflict edge plus up to ``max_hops`` upstream ring edges."""
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


def _place_along_chain(
    chain: List[Tuple[str, float]],
    target_back: float,
    *,
    lane_num: int,
    conflict_edge_id: str,
    conflict_lane_num: int,
    convoy_index: int,
) -> Optional[ConvoySpawnSlot]:
    """Place a vehicle ``target_back`` meters before the chain start (entry).

    If the natural point on a segment would sit closer than
    ``MIN_SPAWN_LONGITUDE_M`` to the segment start, spill further upstream
    instead of clamping (keeps convoy gap along the ring).
    """
    if not chain or target_back < 0.5:
        return None
    remaining = float(target_back)
    for spawn_edge, spawn_len in chain:
        if remaining <= spawn_len - 0.1:
            spawn_long = spawn_len - remaining
            if spawn_long >= MIN_SPAWN_LONGITUDE_M - 1e-6:
                spawn_long = min(spawn_long, spawn_len - 0.1)
                if spawn_long > 0.0 and spawn_long < spawn_len:
                    return ConvoySpawnSlot(
                        spawn_edge_id=spawn_edge,
                        spawn_lane_num=lane_num,
                        spawn_longitudinal=float(spawn_long),
                        convoy_index=int(convoy_index),
                        conflict_edge_id=conflict_edge_id,
                        conflict_lane_num=conflict_lane_num,
                    )
            # Too close to segment start — keep walking upstream.
        remaining -= spawn_len

    # Exhausted chain: sit near the start of the farthest upstream edge.
    spawn_edge, spawn_len = chain[-1]
    spawn_long = min(max(MIN_SPAWN_LONGITUDE_M, 0.5), spawn_len - 0.1)
    if spawn_long <= 0.0:
        return None
    return ConvoySpawnSlot(
        spawn_edge_id=spawn_edge,
        spawn_lane_num=lane_num,
        spawn_longitudinal=float(spawn_long),
        convoy_index=int(convoy_index),
        conflict_edge_id=conflict_edge_id,
        conflict_lane_num=conflict_lane_num,
    )


def resolve_aux_spawn_placement(
    junction_layout: Optional[dict],
    edge_id: str,
    lane_num: int,
    lane_lengths: Dict[Tuple[str, int], float],
    aux_distance_from_intersection: float,
    *,
    allowed_ring_edges: Optional[set[str]] = None,
) -> Optional[AuxSpawnPlacement]:
    """Place aux *lead* on the left-hand conflict segment at ego's entry.

    Stay on ``edge_id`` only. Arcs shorter than ``MIN_CONFLICT_ARC_LENGTH_M``
    return ``None`` (scene rejected at generation). Longer arcs prefer the
    configured ``aux_distance``, but clamp the offset to the segment when the
    arc is shorter than ``aux_distance + MIN_SPAWN_LONGITUDE_M``.
    """
    if allowed_ring_edges is not None and edge_id not in allowed_ring_edges:
        return None

    aux_distance = float(aux_distance_from_intersection)
    if aux_distance < 0.5:
        return None

    length = lane_length_for_spawn(edge_id, lane_num, lane_lengths, junction_layout)
    if length < MIN_CONFLICT_ARC_LENGTH_M:
        return None

    # Prefer full aux_distance; clamp to what this edge can hold.
    target_back = min(aux_distance, length - MIN_SPAWN_LONGITUDE_M)
    if target_back < 0.5:
        return None

    spawn_long = length - target_back
    if spawn_long < MIN_SPAWN_LONGITUDE_M - 1e-6:
        return None
    spawn_long = min(max(spawn_long, MIN_SPAWN_LONGITUDE_M), length - 0.1)
    if spawn_long <= 0.0 or spawn_long >= length:
        return None

    return AuxSpawnPlacement(
        spawn_edge_id=edge_id,
        spawn_lane_num=lane_num,
        spawn_longitudinal=float(spawn_long),
        conflict_edge_id=edge_id,
        conflict_lane_num=lane_num,
    )


def resolve_convoy_spawn_slots(
    junction_layout: Optional[dict],
    edge_id: str,
    lane_num: int,
    lane_lengths: Dict[Tuple[str, int], float],
    aux_distance_from_intersection: float,
    convoy_size: int,
    convoy_gap_m: float,
    *,
    allowed_ring_edges: Optional[set[str]] = None,
    max_spillover_hops: int = MAX_CONVOY_SPILLOVER_HOPS,
) -> Optional[List[ConvoySpawnSlot]]:
    """Place a full aux convoy; spill followers onto upstream ring if needed.

    Lead always sits on the conflict edge (``edge_id``). Followers that do not
    fit on that segment are placed on the next upstream ring hop(s), spaced by
    ``convoy_gap_m`` along the approach path to the ego entry.
    """
    lead = resolve_aux_spawn_placement(
        junction_layout,
        edge_id,
        lane_num,
        lane_lengths,
        aux_distance_from_intersection,
        allowed_ring_edges=allowed_ring_edges,
    )
    if lead is None:
        return None

    n = max(1, int(convoy_size))
    gap = max(0.0, float(convoy_gap_m))
    conflict_len = lane_length_for_spawn(
        lead.spawn_edge_id, lead.spawn_lane_num, lane_lengths, junction_layout
    )
    lead_back = max(0.5, conflict_len - float(lead.spawn_longitudinal))

    if n == 1:
        return [
            ConvoySpawnSlot(
                spawn_edge_id=lead.spawn_edge_id,
                spawn_lane_num=lead.spawn_lane_num,
                spawn_longitudinal=float(lead.spawn_longitudinal),
                convoy_index=0,
                conflict_edge_id=lead.conflict_edge_id,
                conflict_lane_num=lead.conflict_lane_num,
            )
        ]

    chain = _upstream_ring_chain(
        junction_layout,
        lead.spawn_edge_id,
        lead.spawn_lane_num,
        lane_lengths,
        max_hops=max_spillover_hops,
    )
    if not chain:
        return None

    slots: List[ConvoySpawnSlot] = []
    for i in range(n):
        target_back = lead_back + i * gap
        slot = _place_along_chain(
            chain,
            target_back,
            lane_num=lead.spawn_lane_num,
            conflict_edge_id=lead.conflict_edge_id,
            conflict_lane_num=lead.conflict_lane_num,
            convoy_index=i,
        )
        if slot is None:
            return None
        # Lead must stay on the conflict edge.
        if i == 0 and slot.spawn_edge_id != lead.spawn_edge_id:
            return None
        slots.append(slot)

    # Same-edge non-spillover: last slot must still clear MIN_SPAWN.
    if all(s.spawn_edge_id == lead.spawn_edge_id for s in slots):
        if slots[-1].spawn_longitudinal < MIN_SPAWN_LONGITUDE_M - 1e-6:
            return None
    return slots


def placement_fits_convoy(
    placement: Optional[AuxSpawnPlacement],
    convoy_size: int,
    convoy_gap_m: float,
) -> bool:
    """True when lead + followers fit on the *same* placement edge (no spillover)."""
    if placement is None:
        return False
    n = max(1, int(convoy_size))
    gap = max(0.0, float(convoy_gap_m))
    last_long = float(placement.spawn_longitudinal) - (n - 1) * gap
    return last_long >= MIN_SPAWN_LONGITUDE_M - 1e-6


def convoy_fits_with_spillover(
    junction_layout: Optional[dict],
    edge_id: str,
    lane_num: int,
    lane_lengths: Dict[Tuple[str, int], float],
    aux_distance_from_intersection: float,
    convoy_size: int,
    convoy_gap_m: float,
    *,
    allowed_ring_edges: Optional[set[str]] = None,
) -> bool:
    """True when a full convoy fits on the conflict arc and/or upstream spillover."""
    return (
        resolve_convoy_spawn_slots(
            junction_layout,
            edge_id,
            lane_num,
            lane_lengths,
            aux_distance_from_intersection,
            convoy_size,
            convoy_gap_m,
            allowed_ring_edges=allowed_ring_edges,
        )
        is not None
    )


def max_convoy_for_placement(
    placement: Optional[AuxSpawnPlacement],
    convoy_gap_m: float,
    *,
    convoy_size_cap: int = 8,
) -> int:
    """Largest same-edge convoy that still fits on ``placement`` (no spillover)."""
    if placement is None:
        return 0
    gap = max(1e-3, float(convoy_gap_m))
    lead = float(placement.spawn_longitudinal)
    if lead < MIN_SPAWN_LONGITUDE_M - 1e-6:
        return 0
    extra = int((lead - MIN_SPAWN_LONGITUDE_M) // gap)
    return max(1, min(int(convoy_size_cap), 1 + max(0, extra)))


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
