"""Ego-entry conflict zones for PDD 4.3 roundabout yield verification."""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Sequence, Tuple

# When SUMO splits one physical entry across several nodes, ring pieces that
# end near the geometric spoke/ring meeting point still count as conflict.
DEFAULT_ENTRY_GEOM_RADIUS_M = 12.0
# Sample this much of the lane tail when deciding "approaches entry".
DEFAULT_ENTRY_GEOM_TAIL_M = 30.0


def _lane_index_key(lane) -> Optional[str]:
    idx = getattr(lane, "index", None)
    if idx is None:
        return None
    return str(idx)


def entry_xy_from_spoke_lane(lane) -> Optional[Tuple[float, float]]:
    """Map XY where the ego spoke meets the ring (spoke lane end)."""
    if lane is None:
        return None
    try:
        p = lane.position(float(lane.length), 0.0)
        return (float(p[0]), float(p[1]))
    except Exception:
        return None


def _point_xy(lane, long_m: float) -> Optional[Tuple[float, float]]:
    try:
        p = lane.position(float(long_m), 0.0)
        return (float(p[0]), float(p[1]))
    except Exception:
        return None


def _dist_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def closest_long_on_lane(
    lane,
    target_xy: Sequence[float],
    *,
    step_m: float = 2.0,
) -> Optional[Tuple[float, float]]:
    """Return ``(long_m, dist_m)`` for the sample on ``lane`` nearest ``target_xy``."""
    if lane is None or target_xy is None:
        return None
    length = float(getattr(lane, "length", 0.0) or 0.0)
    if length <= 1e-3:
        return None
    best_long = 0.0
    best_dist = float("inf")
    long_m = 0.0
    step = max(0.5, float(step_m))
    while long_m <= length + 1e-9:
        pt = _point_xy(lane, min(long_m, length))
        if pt is not None:
            dist = _dist_xy(pt, target_xy)
            if dist < best_dist:
                best_dist = dist
                best_long = min(long_m, length)
        long_m += step
    if not math.isfinite(best_dist):
        return None
    return (best_long, best_dist)


def mean_lane_end_xy(lanes: Sequence[Any]) -> Optional[Tuple[float, float]]:
    """Average of lane end XYs — ring-side conflict point for sticky release."""
    xs: List[float] = []
    ys: List[float] = []
    for lane in lanes or []:
        length = float(getattr(lane, "length", 0.0) or 0.0)
        pt = _point_xy(lane, length) if length > 1e-3 else None
        if pt is None:
            continue
        xs.append(pt[0])
        ys.append(pt[1])
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def lane_approaches_entry_xy(
    lane,
    entry_xy: Sequence[float],
    *,
    radius_m: float = DEFAULT_ENTRY_GEOM_RADIUS_M,
    tail_m: float = DEFAULT_ENTRY_GEOM_TAIL_M,
) -> bool:
    """True when the lane approaches ``entry_xy`` (not merely leaves it).

    Accepts:
    - downstream end / tail within ``radius_m``;
    - or closest point within ``radius_m`` when that point is in the downstream
      half (long circulating edges that pass a split SUMO node mid-lane).

    Outgoing pieces (closest near the start, end farther away) return False.
    """
    if lane is None or entry_xy is None:
        return False
    length = float(getattr(lane, "length", 0.0) or 0.0)
    if length <= 1e-3:
        return False
    end_xy = _point_xy(lane, length)
    start_xy = _point_xy(lane, 0.0)
    if end_xy is None:
        return False
    if _dist_xy(end_xy, entry_xy) <= float(radius_m):
        return True

    # Downstream tail near entry (SUMO ends a few metres early).
    start_long = max(0.0, length - max(0.0, float(tail_m)))
    step = 2.0
    long_m = start_long
    while long_m < length - 1e-6:
        pt = _point_xy(lane, long_m)
        if pt is not None and _dist_xy(pt, entry_xy) <= float(radius_m):
            if start_xy is None:
                return True
            return _dist_xy(end_xy, entry_xy) <= _dist_xy(start_xy, entry_xy) + 1.0
        long_m += step

    # Long ring edge: closest sample mid-lane, still approaching.
    closest = closest_long_on_lane(lane, entry_xy, step_m=step)
    if closest is None:
        return False
    closest_long, closest_dist = closest
    if closest_dist > float(radius_m):
        return False
    if closest_long < 0.5 * length:
        return False
    if start_xy is not None and _dist_xy(end_xy, entry_xy) > _dist_xy(start_xy, entry_xy) + 1.0:
        return False
    return True


def ring_lanes_near_entry_xy(
    ring_lanes: Sequence[Any],
    entry_xy: Sequence[float],
    *,
    radius_m: float = DEFAULT_ENTRY_GEOM_RADIUS_M,
    tail_m: float = DEFAULT_ENTRY_GEOM_TAIL_M,
) -> List[Any]:
    """Ring lanes whose approach (downstream end/tail/mid) is near the entry XY."""
    if entry_xy is None:
        return []
    out: List[Any] = []
    seen: set[str] = set()
    for lane in ring_lanes or []:
        if not lane_approaches_entry_xy(
            lane, entry_xy, radius_m=radius_m, tail_m=tail_m
        ):
            continue
        key = _lane_index_key(lane)
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        out.append(lane)
    return out


def merge_unique_lanes(*lane_groups: Sequence[Any]) -> List[Any]:
    """Concatenate lane lists, dropping duplicates by lane.index."""
    out: List[Any] = []
    seen: set[str] = set()
    for group in lane_groups:
        for lane in group or []:
            key = _lane_index_key(lane)
            if key is not None and key in seen:
                continue
            if key is not None:
                seen.add(key)
            out.append(lane)
    return out


def entry_conflict_ring_edges(
    layout: dict,
    ego_spoke_edge_id: str,
    *,
    entry_junction_id: Optional[str] = None,
) -> List[str]:
    """Ring edges with traffic approaching ego's entry junction from upstream on the circle.

  Ego must yield to vehicles on these segments (to the left / against ego's
  direction once on the ring), not to traffic ahead on the exit arc.
    """
    entry_j = entry_junction_id or ""
    if not entry_j:
        # Prefer the spoke's to-node (ego entry), not layout.junction_id —
        # the latter is often another entry where the catalog sign was placed.
        entry_j = ego_entry_junction_id(layout, ego_spoke_edge_id) or ""
    if not entry_j:
        entry_j = str(layout.get("junction_id") or "")
    if not entry_j:
        return []

    incoming_edges: List[str] = []
    for arm in layout.get("arms", []):
        if arm.get("road_class") != "main":
            continue
        eid = str(arm.get("edge_id", ""))
        if not eid:
            continue
        if str(arm.get("to_node", "")) == entry_j:
            incoming_edges.append(eid)
    return sorted(incoming_edges)


def entry_outgoing_ring_edges(
    layout: dict,
    ego_spoke_edge_id: str,
    *,
    entry_junction_id: Optional[str] = None,
) -> List[str]:
    """Ring edges leaving ego's entry junction onto the traffic circle."""
    entry_j = entry_junction_id or layout.get("junction_id") or ""
    if not entry_j:
        return []

    ego_arm = next(
        (arm for arm in layout.get("arms", []) if arm.get("edge_id") == ego_spoke_edge_id),
        None,
    )
    if entry_junction_id is None and ego_arm is not None:
        entry_j = str(ego_arm.get("to_node", "")) or entry_j

    outgoing_edges: List[str] = []
    for arm in layout.get("arms", []):
        if arm.get("road_class") != "main":
            continue
        eid = str(arm.get("edge_id", ""))
        if not eid:
            continue
        if str(arm.get("from_node", "")) == entry_j:
            outgoing_edges.append(eid)
    return sorted(outgoing_edges)


def all_entry_conflict_ring_edges(layout: dict) -> List[str]:
    """Ring edges immediately upstream of every roundabout entry spoke.

    For each secondary road entering the traffic circle, find the main/ring
    edge that ends at the same junction node. These are the left-side circle
    approaches whose last 20 m should be tracked for circulating traffic.
    """
    ring_nodes = set()
    for arm in layout.get("arms", []):
        if arm.get("road_class") != "main":
            continue
        if arm.get("from_node"):
            ring_nodes.add(str(arm.get("from_node")))
        if arm.get("to_node"):
            ring_nodes.add(str(arm.get("to_node")))

    entry_nodes = set()
    for arm in layout.get("arms", []):
        if arm.get("road_class") != "secondary":
            continue
        to_node = str(arm.get("to_node", ""))
        from_node = str(arm.get("from_node", ""))
        if to_node in ring_nodes and from_node not in ring_nodes:
            entry_nodes.add(to_node)

    incoming_edges: List[str] = []
    for arm in layout.get("arms", []):
        if arm.get("road_class") != "main":
            continue
        eid = str(arm.get("edge_id", ""))
        if eid and str(arm.get("to_node", "")) in entry_nodes:
            incoming_edges.append(eid)
    return sorted(set(incoming_edges))


def lane_keys_for_edges(layout: dict, edge_ids: Iterable[str]) -> List[str]:
    wanted = set(edge_ids)
    keys: List[str] = []
    for arm in layout.get("arms", []):
        if arm.get("edge_id") in wanted:
            keys.extend(arm.get("lane_keys", []))
    return sorted(keys)


def entry_conflict_aux_lane_keys(
    layout: dict,
    ego_spoke_edge_id: str,
    *,
    entry_junction_id: Optional[str] = None,
) -> List[str]:
    """Parallel main-road lane keys at ego entry conflict (excludes upstream hops)."""
    edge_ids = entry_conflict_ring_edges(
        layout,
        ego_spoke_edge_id,
        entry_junction_id=entry_junction_id,
    )
    return lane_keys_for_edges(layout, edge_ids)


def collect_lanes_for_edge_ids(env, layout: dict, edge_ids: Sequence[str]) -> List[Any]:
    from .junction_sign_placement import collect_lanes_for_keys

    return collect_lanes_for_keys(env, lane_keys_for_edges(layout, edge_ids))


def collect_entry_conflict_lanes(
    env,
    layout: dict,
    ego_spoke_edge_id: str,
    *,
    entry_junction_id: Optional[str] = None,
) -> List[Any]:
    """MetaDrive lanes on ring segments upstream of ego's entry junction."""
    edge_ids = entry_conflict_ring_edges(
        layout,
        ego_spoke_edge_id,
        entry_junction_id=entry_junction_id,
    )
    return collect_lanes_for_edge_ids(env, layout, edge_ids)


def collect_all_entry_conflict_lanes(env, layout: dict) -> List[Any]:
    """MetaDrive lanes on all ring segments upstream of roundabout entries."""
    return collect_lanes_for_edge_ids(env, layout, all_entry_conflict_ring_edges(layout))


def ego_entry_junction_id(
    layout: dict,
    ego_spoke_edge_id: str,
) -> Optional[str]:
    """Junction node where ego spoke meets the traffic circle."""
    for arm in layout.get("arms", []):
        if arm.get("edge_id") == ego_spoke_edge_id:
            return str(arm.get("to_node", "")) or None
    return layout.get("junction_id")


def _upstream_ring_arm(layout: dict, edge_id: str) -> Optional[dict]:
    arm = next((a for a in layout.get("arms", []) if a.get("edge_id") == edge_id), None)
    if arm is None or arm.get("road_class") != "main":
        return None
    from_node = str(arm.get("from_node", ""))
    if not from_node:
        return None
    upstream = [
        candidate
        for candidate in layout.get("arms", [])
        if candidate.get("road_class") == "main"
        and str(candidate.get("to_node", "")) == from_node
    ]
    if not upstream:
        return None
    return max(upstream, key=lambda item: float(item.get("min_lane_length", 0.0) or 0.0))


def conflict_aux_ring_edge_ids(
    layout: dict,
    ego_spoke_edge_id: str,
    *,
    entry_junction_id: Optional[str] = None,
    max_upstream_hops: int = 1,
) -> List[str]:
    """Ring edge IDs where aux may spawn for ego yield (conflict arc, not exit arc).

    Includes the ring segment(s) ending at ego's entry junction and at most
    ``max_upstream_hops`` upstream segments on the circle, excluding outgoing
    ring edges at the same entry.
    """
    entry_j = entry_junction_id or ego_entry_junction_id(layout, ego_spoke_edge_id) or ""
    if not entry_j:
        return []

    outgoing = set(
        entry_outgoing_ring_edges(
            layout,
            ego_spoke_edge_id,
            entry_junction_id=entry_j,
        )
    )
    conflict = entry_conflict_ring_edges(
        layout,
        ego_spoke_edge_id,
        entry_junction_id=entry_j,
    )
    allowed: List[str] = []
    seen: set[str] = set()
    for edge_id in conflict:
        if edge_id and edge_id not in seen:
            allowed.append(edge_id)
            seen.add(edge_id)

    frontier = list(conflict)
    for _ in range(max(0, int(max_upstream_hops))):
        next_frontier: List[str] = []
        for edge_id in frontier:
            upstream = _upstream_ring_arm(layout, edge_id)
            if upstream is None:
                continue
            up_id = str(upstream.get("edge_id", ""))
            if not up_id or up_id in seen or up_id in outgoing:
                continue
            allowed.append(up_id)
            seen.add(up_id)
            next_frontier.append(up_id)
        frontier = next_frontier

    return allowed


def compact_aux_ring_edges_for_ego(
    layout: dict,
    ego_spoke_edge_id: str,
    *,
    entry_junction_id: Optional[str] = None,
) -> List[str]:
    """Ring edges at ego's entry viable for compact-ring aux (blue zone + outgoing)."""
    entry_j = entry_junction_id or ego_entry_junction_id(layout, ego_spoke_edge_id) or ""
    if not entry_j:
        return []

    ordered: List[str] = []
    seen: set[str] = set()
    for edge_id in entry_conflict_ring_edges(
        layout, ego_spoke_edge_id, entry_junction_id=entry_j
    ):
        if edge_id not in seen:
            ordered.append(edge_id)
            seen.add(edge_id)
    for edge_id in entry_outgoing_ring_edges(
        layout, ego_spoke_edge_id, entry_junction_id=entry_j
    ):
        if edge_id not in seen:
            ordered.append(edge_id)
            seen.add(edge_id)
    return ordered


def _arm_min_lane_length(layout: dict, edge_id: str) -> float:
    for arm in layout.get("arms", []):
        if arm.get("edge_id") == edge_id:
            return float(arm.get("min_lane_length", 0.0) or 0.0)
    return 0.0
