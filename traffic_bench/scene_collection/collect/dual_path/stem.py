"""T-junction stem (bottom-arm) detection for dual-path / 5.7 ego spawn."""

from __future__ import annotations

from typing import TYPE_CHECKING, Set

if TYPE_CHECKING:
    from .dual_path import EdgeGraph


def incoming_edge_ids(graph: "EdgeGraph", junction_id: str) -> list[str]:
    return sorted(
        eid for eid, to_node in graph.edge_to_node.items() if to_node == junction_id
    )


def is_t_stem_approach(graph: "EdgeGraph", junction_id: str, ego_edge_id: str) -> bool:
    """True when ``ego`` is the bottom arm of a T (can turn both left and right).

    Crossbar approaches typically have a straight continuation; the stem has
    both ``l`` and ``r`` first exits onto the crossbar.
    """
    incoming = incoming_edge_ids(graph, junction_id)
    if len(incoming) != 3:
        return False
    if ego_edge_id not in incoming:
        return False
    exits = graph.first_exits.get(ego_edge_id) or {}
    dirs: Set[str] = {d for d, edges in exits.items() if edges and d in ("l", "r", "s")}
    return "l" in dirs and "r" in dirs
