"""Analyse a CityMap road network graph: find intersections, road segments,
shortest paths, and validate that prohibition signs have a real detour.

The graph is the NodeRoadNetwork.graph: dict[from_node][to_node] -> list[lane].
Block IDs are encoded in node names like "1X0_0_": block_idx=1, block_id="X".
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

NODE_RE = re.compile(r"-?(\d+)([A-Za-z])(\d+)_(\d+)_")


def parse_node(name: str) -> Optional[Tuple[int, str, int, int]]:
    """Parse a PG node name like '2X0_0_' into (block_idx, block_id, part_idx, road_idx).
    Returns None for special nodes like '>', '>>', '->', etc.
    """
    m = NODE_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))


@dataclass
class Edge:
    from_node: str
    to_node: str
    n_lanes: int
    length: float
    block_id: Optional[str]  # block_id of the to_node, if it has one


@dataclass
class IntersectionInfo:
    """A node where multiple roads diverge — candidate for turn-prohibition signs."""
    node: str
    block_id: str       # X or T or O
    out_edges: List[Edge]


@dataclass
class SegmentInfo:
    """A road segment (single edge with ≥1 lanes) — candidate for speed/stop/restricted_lane."""
    from_node: str
    to_node: str
    n_lanes: int
    length: float
    block_id: Optional[str]


def build_edges(graph) -> List[Edge]:
    """Convert the road_network graph into a list of Edge objects."""
    edges = []
    for from_node, outs in graph.items():
        for to_node, lanes in outs.items():
            if not lanes:
                continue
            length = float(lanes[0].length)
            parsed = parse_node(to_node)
            block_id = parsed[1] if parsed else None
            edges.append(Edge(
                from_node=from_node,
                to_node=to_node,
                n_lanes=len(lanes),
                length=length,
                block_id=block_id,
            ))
    return edges


def find_intersections(graph) -> List[IntersectionInfo]:
    """Find nodes where multiple roads diverge — these are intersection branch points."""
    intersections = []
    for node, outs in graph.items():
        if len(outs) < 2:
            continue
        parsed = parse_node(node)
        if parsed is None:
            continue
        # Branch points inside intersection blocks (X, T, O)
        block_id = parsed[1]
        if block_id not in ("X", "T", "O", "R", "r"):
            continue
        out_edges = []
        for to_node, lanes in outs.items():
            if not lanes:
                continue
            out_edges.append(Edge(
                from_node=node,
                to_node=to_node,
                n_lanes=len(lanes),
                length=float(lanes[0].length),
                block_id=parse_node(to_node)[1] if parse_node(to_node) else None,
            ))
        if len(out_edges) >= 2:
            intersections.append(IntersectionInfo(node=node, block_id=block_id, out_edges=out_edges))
    return intersections


def find_segments(graph, min_length: float = 15.0, min_lanes: int = 1) -> List[SegmentInfo]:
    """Find road segments suitable for placing universal signs (speed, stop, etc.)."""
    segs = []
    for from_node, outs in graph.items():
        for to_node, lanes in outs.items():
            if not lanes:
                continue
            length = float(lanes[0].length)
            if length < min_length:
                continue
            if len(lanes) < min_lanes:
                continue
            parsed = parse_node(to_node)
            block_id = parsed[1] if parsed else None
            segs.append(SegmentInfo(
                from_node=from_node,
                to_node=to_node,
                n_lanes=len(lanes),
                length=length,
                block_id=block_id,
            ))
    return segs


def _positive(name: str) -> bool:
    return not name.startswith("-")


def shortest_path_nodes(graph, start: str, dest: str) -> Optional[List[str]]:
    """BFS shortest path between two nodes (positive-direction only)."""
    if start == dest:
        return [start]
    visited = {start}
    queue = [(start, [start])]
    while queue:
        node, path = queue.pop(0)
        for to_node in graph.get(node, {}):
            if to_node in visited or not _positive(to_node):
                continue
            new_path = path + [to_node]
            if to_node == dest:
                return new_path
            visited.add(to_node)
            queue.append((to_node, new_path))
    return None


def path_length(graph, path: List[str]) -> float:
    """Total physical length of a route (sum of lane lengths)."""
    total = 0.0
    for i in range(len(path) - 1):
        lanes = graph.get(path[i], {}).get(path[i + 1])
        if lanes:
            total += float(lanes[0].length)
    return total


def shortest_path_avoiding_edge(
    graph, start: str, dest: str, forbidden_edge: Tuple[str, str]
) -> Optional[List[str]]:
    """BFS shortest path that does NOT traverse a specific edge (positive-direction only)."""
    if start == dest:
        return [start]
    visited = {start}
    queue = [(start, [start])]
    while queue:
        node, path = queue.pop(0)
        for to_node in graph.get(node, {}):
            if (node, to_node) == forbidden_edge:
                continue
            if to_node in visited or not _positive(to_node):
                continue
            new_path = path + [to_node]
            if to_node == dest:
                return new_path
            visited.add(to_node)
            queue.append((to_node, new_path))
    return None


def find_dead_ends(graph) -> Set[str]:
    """Nodes that have no outgoing edges (terminal points)."""
    return {n for n, outs in graph.items() if not outs}


def reachable_from(graph, start: str) -> Set[str]:
    """All nodes reachable from start via outgoing edges."""
    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for to_node in graph.get(node, {}):
            if to_node not in visited:
                visited.add(to_node)
                stack.append(to_node)
    return visited


def collect_meaningful_destinations(graph, start: str, min_length: float = 50.0,
                                     max_length: float = 400.0) -> List[Tuple[str, List[str], float]]:
    """Find candidate destinations reachable from start with route in [min_length, max_length] m.

    Returns list of (dest_node, path, length).
    """
    results = []
    visited = {start: ([start], 0.0)}
    queue = [(start, [start], 0.0)]
    while queue:
        node, path, length = queue.pop(0)
        # Destination must be a graph key (have outgoing edges) so MetaDrive's
        # shortest_path can find a route to it.
        if length >= min_length and node in graph:
            results.append((node, path, length))
        if length >= max_length:
            continue
        for to_node in graph.get(node, {}):
            if to_node in visited or not _positive(to_node):
                continue
            lanes = graph[node][to_node]
            if not lanes:
                continue
            new_length = length + float(lanes[0].length)
            if new_length > max_length * 1.5:
                continue
            new_path = path + [to_node]
            visited[to_node] = (new_path, new_length)
            queue.append((to_node, new_path, new_length))
    return results
