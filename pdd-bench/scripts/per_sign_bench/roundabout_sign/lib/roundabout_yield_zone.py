"""Ego-entry conflict zones for PDD 4.3 roundabout yield verification."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence


def _lane_index_key(lane) -> Optional[str]:
    idx = getattr(lane, "index", None)
    if idx is None:
        return None
    return str(idx)


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
    entry_j = entry_junction_id or layout.get("junction_id") or ""
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
