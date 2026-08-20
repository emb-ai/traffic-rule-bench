"""Enumerate semantically valid CityMap-based scenes with sign-centric routing.

For each candidate sign edge (A, B):
  - Pick a SAFE spawn node a few hops upstream of A
  - Pick a SAFE destination node a few hops downstream of B
  - Verify the natural shortest path from spawn to dest passes through (A, B)
  - For prohibition signs, verify an alternative path exists that AVOIDS (A, B)
    AND is not significantly longer (so the detour is short and meaningful)

Routes are short by construction:
  - Universal / restricted_lane: 80–180 m total
  - Prohibition (with detour):   120–250 m total

Enumeration is deterministic given (map_seed, n_blocks, max_scenes_per_sign).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple

from .citymap_analyzer import (
    parse_node,
    path_length,
    shortest_path_avoiding_edge,
)


# ---------------------------------------------------------------------------
# Sign families (CityMap subset of factorized_space.benchmark_runner.SIGN_CLASS_MAP)
# ---------------------------------------------------------------------------

UNIVERSAL_SIGN_KEYS = [
    "stop", "speed20", "speed30", "speed40", "speed60",
    "zone_speed20", "zone_speed30", "zone_speed40", "zone_speed60",
    "minspeed30", "minspeed40", "minspeed50",
    "no_stop", "bus_station", "only_auto", "no_overtaking",
]
PROHIBITION_SIGN_KEYS = [
    "no_entry", "no_traffic",
    "no_right_turn", "no_left_turn", "no_uturn",
]
RESTRICTED_LANE_SIGN_KEYS = [
    "bus_lane_road", "bike_lane_road", "bus_lane", "bike_lane",
]
DETOUR_SIGN_KEYS = ["detour_right", "detour_left", "detour_either"]


# ---------------------------------------------------------------------------
# Scene dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SignPlacement:
    sign_key: str
    edge_from: str
    edge_to: str
    lane_idx: int
    longitudinal: float


@dataclass
class CityMapScene:
    """A self-contained scene specification on a CityMap.

    For prohibition signs (with detour) the contract is:
      * ``primary_path`` — the natural shortest path from spawn to dest, the
        one that *would* hit the forbidden sign edge if the agent ignored the
        sign. Stored for evaluation reference only.
      * ``route_path`` — the path the agent is actually navigated along. For
        prohibition scenes this equals the alternative (detour) path that
        bypasses the sign edge ``(edge_from, edge_to)``. The first edge of
        ``route_path`` always equals ``(spawn_node_from, spawn_node_to)`` so
        the spawn lane is consistent with the navigation.
      * ``route_length`` — physical length of ``route_path`` (m).
      * ``has_alternative`` — True iff ``route_path`` differs from
        ``primary_path`` (i.e. it's a real detour). False for non-prohibition
        scenes, where ``route_path == primary_path``.
      * ``alternative_length`` — physical length of the alternative; equal to
        ``route_length`` when ``has_alternative`` is True, else None.
    """
    scene_id: str
    map_seed: int
    n_blocks: int
    lane_num: int
    spawn_node_from: str
    spawn_node_to: str
    spawn_lane_idx: int
    dest_node: str
    sign_placement: SignPlacement
    route_path: List[str]
    route_length: float
    has_alternative: bool
    alternative_length: Optional[float]
    primary_path: Optional[List[str]] = None
    primary_length: Optional[float] = None


# ---------------------------------------------------------------------------
# Safe-spawn heuristic
# ---------------------------------------------------------------------------

FIRST_BLOCK_NODES = {">", ">>", ">>>"}


def is_safe_spawn(node: str) -> bool:
    """Heuristic: would MetaDrive accept this node as a vehicle spawn lane source?

    Safe nodes are FROM-nodes of regular drivable road segments, NOT internal
    intersection / roundabout / ramp parts. Used to avoid env_creation crashes
    when overriding the spawn lane.
    """
    if node in FIRST_BLOCK_NODES:
        return True
    parsed = parse_node(node)
    if parsed is None:
        return False
    _, block_id, part_idx, _road_idx = parsed
    if block_id in ("S", "C"):
        return True
    if block_id in ("X", "T", "O") and part_idx == 0:
        return True
    if block_id in ("r", "R") and part_idx == 0:
        return True
    if block_id in ("y", "Y") and part_idx == 0:
        return True
    return False


def is_positive_node(name: str) -> bool:
    return not name.startswith("-")


# ---------------------------------------------------------------------------
# Reverse-graph helper
# ---------------------------------------------------------------------------

def build_reverse_graph(graph) -> Dict[str, List[str]]:
    rev: Dict[str, List[str]] = {}
    for f, outs in graph.items():
        for t in outs:
            rev.setdefault(t, []).append(f)
    return rev


# ---------------------------------------------------------------------------
# Sign-centric upstream / downstream walker
# ---------------------------------------------------------------------------

def walk_upstream(
    reverse: Dict[str, List[str]],
    start_node: str,
    min_hops: int,
    max_hops: int,
) -> List[Tuple[str, int]]:
    """BFS backward from start_node, return (node, hops) pairs in [min_hops, max_hops]
    that are positive-direction graph keys.
    """
    visited: Dict[str, int] = {start_node: 0}
    queue = [(start_node, 0)]
    results: List[Tuple[str, int]] = []
    while queue:
        node, depth = queue.pop(0)
        if depth >= min_hops and node != start_node and is_positive_node(node):
            results.append((node, depth))
        if depth >= max_hops:
            continue
        for prev in reverse.get(node, []):
            if prev in visited or not is_positive_node(prev):
                continue
            visited[prev] = depth + 1
            queue.append((prev, depth + 1))
    return results


def walk_downstream(
    graph,
    start_node: str,
    min_hops: int,
    max_hops: int,
) -> List[Tuple[str, int]]:
    """BFS forward from start_node, return (node, hops) pairs in [min_hops, max_hops]
    that are positive-direction graph keys.
    """
    visited: Dict[str, int] = {start_node: 0}
    queue = [(start_node, 0)]
    results: List[Tuple[str, int]] = []
    while queue:
        node, depth = queue.pop(0)
        if depth >= min_hops and node != start_node and is_positive_node(node) and node in graph:
            results.append((node, depth))
        if depth >= max_hops:
            continue
        for to_node in graph.get(node, {}):
            if to_node in visited or not is_positive_node(to_node):
                continue
            visited[to_node] = depth + 1
            queue.append((to_node, depth + 1))
    return results


# ---------------------------------------------------------------------------
# MetaDrive-compatible shortest path wrappers
# ---------------------------------------------------------------------------

def _md_shortest_path(road_network, graph, s_node: str, e_node: str) -> Optional[List[str]]:
    """Shortest path that exactly matches what MetaDrive's navigation would compute."""
    if road_network is not None:
        try:
            path = road_network.shortest_path((s_node, "_dummy_", 0), e_node)
            return path if path else None
        except Exception:
            return None
    # Fallback BFS that matches MetaDrive's bfs_paths semantics
    if s_node == e_node:
        return [s_node]
    queue = [(s_node, [s_node])]
    while queue:
        node, path = queue.pop(0)
        if node not in graph:
            continue
        for to_node in set(graph[node].keys()) - set(path):
            new_path = path + [to_node]
            if to_node == e_node:
                return new_path
            queue.append((to_node, new_path))
    return None


def _md_shortest_path_avoiding(graph, s_node: str, e_node: str,
                                forbidden: Tuple[str, str]) -> Optional[List[str]]:
    """BFS shortest path skipping a forbidden edge — same dict-iteration semantics
    as MetaDrive's bfs_paths but with edge exclusion.
    """
    if s_node == e_node:
        return [s_node]
    queue = [(s_node, [s_node])]
    while queue:
        node, path = queue.pop(0)
        if node not in graph:
            continue
        for to_node in set(graph[node].keys()) - set(path):
            if (node, to_node) == forbidden:
                continue
            new_path = path + [to_node]
            if to_node == e_node:
                return new_path
            queue.append((to_node, new_path))
    return None


def _alternative_with_fixed_first_edge(
    graph,
    spawn_from: str,
    spawn_to: str,
    dest: str,
    forbidden: Tuple[str, str],
) -> Optional[List[str]]:
    """Shortest path from spawn_to to dest avoiding ``forbidden``, prepended
    with ``spawn_from`` so the result starts with the edge ``(spawn_from,
    spawn_to)``.

    This is the alternative the runner installs as the agent's navigation
    when there is a prohibition sign on ``forbidden`` — it guarantees the
    spawn lane (which lives on the spawn edge) matches the first segment of
    the navigation route, and that the route does not use the forbidden edge.
    """
    if (spawn_from, spawn_to) == forbidden:
        return None
    if spawn_to == dest:
        return [spawn_from, spawn_to]
    queue = [(spawn_to, [spawn_from, spawn_to])]
    while queue:
        node, path = queue.pop(0)
        if node not in graph:
            continue
        for to_node in set(graph[node].keys()) - set(path):
            if (node, to_node) == forbidden:
                continue
            new_path = path + [to_node]
            if to_node == dest:
                return new_path
            queue.append((to_node, new_path))
    return None


def _edge_in_path(path: List[str], from_node: str, to_node: str) -> bool:
    for i in range(len(path) - 1):
        if path[i] == from_node and path[i + 1] == to_node:
            return True
    return False


# ---------------------------------------------------------------------------
# Sign edge candidates
# ---------------------------------------------------------------------------

def find_sign_edges(graph, min_length: float, min_lanes: int = 1) -> List[Tuple[str, str, int, float]]:
    """Return (from_node, to_node, n_lanes, length) edges suitable for placing a sign."""
    out = []
    for f, outs in graph.items():
        if not is_positive_node(f):
            continue
        for t, lanes in outs.items():
            if not lanes:
                continue
            if not is_positive_node(t):
                continue
            length = float(lanes[0].length)
            if length < min_length or len(lanes) < min_lanes:
                continue
            out.append((f, t, len(lanes), length))
    return out


# ---------------------------------------------------------------------------
# Main enumeration
# ---------------------------------------------------------------------------

def _build_short_scene(
    sign_key: str,
    sign_edge: Tuple[str, str, int, float],
    graph,
    reverse: Dict[str, List[str]],
    road_network,
    map_seed: int,
    n_blocks: int,
    lane_num: int,
    min_total_len: float,
    max_total_len: float,
    pre_hops_range: Tuple[int, int],
    post_hops_range: Tuple[int, int],
    require_alternative: bool,
    sign_lane_idx_fn,
    longitudinal_fn,
    scene_count: int,
) -> Optional[CityMapScene]:
    """Build a single short scene around a given sign edge.

    Searches over (spawn upstream, dest downstream) combinations until it finds
    one whose natural shortest path passes through the sign edge AND
    (if require_alternative) an alternative path exists.
    """
    f, t, n_lanes, edge_length = sign_edge

    upstream = walk_upstream(reverse, f, min_hops=pre_hops_range[0], max_hops=pre_hops_range[1])
    downstream = walk_downstream(graph, t, min_hops=post_hops_range[0], max_hops=post_hops_range[1])

    # Prefer safe spawns first
    upstream_sorted = sorted(upstream, key=lambda x: (0 if is_safe_spawn(x[0]) else 1, x[1]))
    downstream_sorted = sorted(downstream, key=lambda x: x[1])

    for spawn_node, _ in upstream_sorted:
        for dest_node, _ in downstream_sorted:
            if spawn_node == dest_node:
                continue
            primary = _md_shortest_path(road_network, graph, spawn_node, dest_node)
            if primary is None or len(primary) < 3:
                continue
            if not _edge_in_path(primary, f, t):
                continue
            primary_len = path_length(graph, primary)
            if primary_len < min_total_len or primary_len > max_total_len:
                continue

            spawn_to = primary[1]  # spawn lane edge = (spawn_node, primary[1])

            alt_path: Optional[List[str]] = None
            alt_len: Optional[float] = None
            if require_alternative:
                # The alternative MUST start with the same spawn edge as the
                # primary so the agent's spawn lane is consistent with the
                # navigation route, AND it must NOT use the forbidden sign
                # edge (f, t). The agent will be navigated along this path,
                # so its actual trajectory bypasses the sign.
                alt_path = _alternative_with_fixed_first_edge(
                    graph, spawn_node, spawn_to, dest_node, forbidden=(f, t)
                )
                if alt_path is None or len(alt_path) < 3:
                    continue
                if _edge_in_path(alt_path, f, t):
                    # Sanity check — should never happen by construction.
                    continue
                alt_len = path_length(graph, alt_path)
                # Detour must not be excessively long. 1.5× is the loose
                # setting used in benchmark_v3 and produces ~90-115 scenes
                # per prohibition sign at n_blocks=8. 1.3× was too tight
                # (0 scenes in mini tests). If you want tighter detours
                # for the final benchmark, pass a stricter cutoff via
                # enumerate_scenes_for_map(..., max_detour_ratio=1.3).
                if alt_len > max_total_len * 1.5:
                    continue

            placement = SignPlacement(
                sign_key=sign_key,
                edge_from=f,
                edge_to=t,
                lane_idx=sign_lane_idx_fn(n_lanes),
                longitudinal=longitudinal_fn(edge_length),
            )
            scene_id = f"map{map_seed}_{sign_key}_{f}_{t}_{scene_count}"

            # For prohibition scenes the agent's actual route is the alternative;
            # for everything else it's the natural shortest path.
            if require_alternative:
                actual_route = alt_path
                actual_route_len = alt_len
            else:
                actual_route = primary
                actual_route_len = primary_len

            return CityMapScene(
                scene_id=scene_id,
                map_seed=map_seed,
                n_blocks=n_blocks,
                lane_num=lane_num,
                spawn_node_from=spawn_node,
                spawn_node_to=spawn_to,
                spawn_lane_idx=0,
                dest_node=dest_node,
                sign_placement=placement,
                route_path=actual_route,
                route_length=actual_route_len,
                has_alternative=require_alternative,
                alternative_length=alt_len,
                primary_path=primary,
                primary_length=primary_len,
            )
    return None


def enumerate_scenes_for_map(
    graph,
    map_seed: int,
    n_blocks: int,
    lane_num: int,
    max_scenes_per_sign: int = 5,
    road_network=None,
    sign_categories: Optional[List[str]] = None,
) -> List[CityMapScene]:
    """Enumerate semantically valid SHORT scenes around each sign edge.

    Args:
        sign_categories: which sign families to enumerate. If None, enumerates
            all of {"universal", "prohibition", "restricted_lane"}. The unified
            benchmark passes ["prohibition"] to restrict CityMap to its only
            useful niche (signs that need a real physical detour).
    """
    if sign_categories is None:
        sign_categories = ["universal", "prohibition", "restricted_lane"]
    enabled = set(sign_categories)
    scenes: List[CityMapScene] = []
    reverse = build_reverse_graph(graph)

    # Universal signs: short routes 80–180 m, sign on lane 0, longitudinal mid-lane
    universal_edges = find_sign_edges(graph, min_length=15.0, min_lanes=1) if "universal" in enabled else []
    for sign_key in (UNIVERSAL_SIGN_KEYS if "universal" in enabled else []):
        count = 0
        for edge in universal_edges:
            if count >= max_scenes_per_sign:
                break
            scene = _build_short_scene(
                sign_key=sign_key,
                sign_edge=edge,
                graph=graph,
                reverse=reverse,
                road_network=road_network,
                map_seed=map_seed,
                n_blocks=n_blocks,
                lane_num=lane_num,
                min_total_len=80.0,
                max_total_len=180.0,
                pre_hops_range=(1, 3),
                post_hops_range=(1, 3),
                require_alternative=False,
                sign_lane_idx_fn=lambda n: 0,
                longitudinal_fn=lambda L: min(L / 2, 10.0),
                scene_count=count,
            )
            if scene is not None:
                scenes.append(scene)
                count += 1

    # Prohibition signs: 120–250 m, must have a real alternative
    prohibition_edges = find_sign_edges(graph, min_length=10.0, min_lanes=1) if "prohibition" in enabled else []
    for sign_key in (PROHIBITION_SIGN_KEYS if "prohibition" in enabled else []):
        count = 0
        for edge in prohibition_edges:
            if count >= max_scenes_per_sign:
                break
            scene = _build_short_scene(
                sign_key=sign_key,
                sign_edge=edge,
                graph=graph,
                reverse=reverse,
                road_network=road_network,
                map_seed=map_seed,
                n_blocks=n_blocks,
                lane_num=lane_num,
                min_total_len=80.0,
                max_total_len=250.0,
                pre_hops_range=(1, 3),
                post_hops_range=(1, 3),
                require_alternative=True,
                sign_lane_idx_fn=lambda n: 0,
                longitudinal_fn=lambda L: 2.0,
                scene_count=count,
            )
            if scene is not None:
                scenes.append(scene)
                count += 1

    # Restricted lane signs: need ≥2 lanes, sign on rightmost lane
    restricted_edges = (
        list(find_sign_edges(graph, min_length=20.0, min_lanes=2))
        if "restricted_lane" in enabled else []
    )
    for sign_key in (RESTRICTED_LANE_SIGN_KEYS if "restricted_lane" in enabled else []):
        count = 0
        for edge in restricted_edges:
            if count >= max_scenes_per_sign:
                break
            scene = _build_short_scene(
                sign_key=sign_key,
                sign_edge=edge,
                graph=graph,
                reverse=reverse,
                road_network=road_network,
                map_seed=map_seed,
                n_blocks=n_blocks,
                lane_num=lane_num,
                min_total_len=80.0,
                max_total_len=200.0,
                pre_hops_range=(1, 3),
                post_hops_range=(1, 3),
                require_alternative=False,
                sign_lane_idx_fn=lambda n: n - 1,
                longitudinal_fn=lambda L: 2.0,
                scene_count=count,
            )
            if scene is not None:
                scenes.append(scene)
                count += 1

    return scenes


# ---------------------------------------------------------------------------
# JSON (de)serialization
# ---------------------------------------------------------------------------

def scene_to_dict(scene: CityMapScene) -> dict:
    d = asdict(scene)
    return d


def dict_to_scene(d: dict) -> CityMapScene:
    sp = d["sign_placement"]
    return CityMapScene(
        scene_id=d["scene_id"],
        map_seed=d["map_seed"],
        n_blocks=d["n_blocks"],
        lane_num=d["lane_num"],
        spawn_node_from=d["spawn_node_from"],
        spawn_node_to=d["spawn_node_to"],
        spawn_lane_idx=d["spawn_lane_idx"],
        dest_node=d["dest_node"],
        sign_placement=SignPlacement(
            sign_key=sp["sign_key"],
            edge_from=sp["edge_from"],
            edge_to=sp["edge_to"],
            lane_idx=sp["lane_idx"],
            longitudinal=sp["longitudinal"],
        ),
        route_path=list(d["route_path"]),
        route_length=float(d["route_length"]),
        has_alternative=bool(d["has_alternative"]),
        alternative_length=(float(d["alternative_length"]) if d.get("alternative_length") is not None else None),
        primary_path=(list(d["primary_path"]) if d.get("primary_path") is not None else None),
        primary_length=(float(d["primary_length"]) if d.get("primary_length") is not None else None),
    )
