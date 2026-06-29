"""Detect SUMO roundabouts in net.xml for PDD 4.3 benchmarks.

Only nets with explicit ``<roundabout>`` blocks (from netconvert ``--roundabouts.guess``
or OSM import) qualify. Geometric loop detection is intentionally not used.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .junction_priority_layout import (
    ApproachArm,
    JunctionLayoutError,
    JunctionPriorityLayout,
    SumoEdge,
    _angle_of_point,
    _entry_point_for_lane,
    _left_targets,
    _load_net,
    _outgoing_targets,
    _straight_targets,
)


@dataclass(frozen=True)
class SumoRoundabout:
    """One ``<roundabout>`` entry from a SUMO net."""

    node_ids: frozenset[str]
    ring_edge_ids: frozenset[str]


@dataclass(frozen=True)
class RoundaboutPick:
    """One qualifying traffic circle near a sign approach road."""

    entry_junction_id: str
    approach_edge_id: str
    center_xy: Tuple[float, float]
    ring_junction_ids: Tuple[str, ...]
    ring_edge_ids: Tuple[str, ...]
    spoke_edge_ids: Tuple[str, ...]

    @property
    def ring_junction_count(self) -> int:
        return len(self.ring_junction_ids)

    @property
    def ring_edge_count(self) -> int:
        return len(self.ring_edge_ids)


def _dist_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _edge_max_length(edge: SumoEdge) -> float:
    if not edge.lanes:
        return 0.0
    return max(lane.length for lane in edge.lanes)


def _parse_sumo_roundabouts(net_path: Path) -> List[SumoRoundabout]:
    root = ET.parse(net_path).getroot()
    roundabouts: List[SumoRoundabout] = []
    for rb in root.findall("roundabout"):
        nodes = frozenset(token for token in rb.get("nodes", "").split() if token)
        edges = frozenset(token for token in rb.get("edges", "").split() if token)
        if edges:
            roundabouts.append(SumoRoundabout(node_ids=nodes, ring_edge_ids=edges))
    return roundabouts


def _adjacent_from_connections(edges: Dict[str, SumoEdge], connections: list) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = defaultdict(list)
    for conn in connections:
        src, dst = conn.get("from", ""), conn.get("to", "")
        if src and dst and dst in edges:
            adj[src].append(dst)
    return adj


def _bfs_edges_from_sign(
    start_edge: str,
    edges: Dict[str, SumoEdge],
    adj: Dict[str, List[str]],
    *,
    max_hops: int = 40,
) -> list[tuple[str, int, list[str]]]:
    """Return [(edge_id, hop_count, path_edges), ...] in BFS order from the sign road."""
    if start_edge not in edges:
        return []

    queue: deque[tuple[str, int, list[str]]] = deque([(start_edge, 0, [start_edge])])
    seen: Set[str] = {start_edge}
    ordered: list[tuple[str, int, list[str]]] = []

    while queue:
        edge_id, hops, path = queue.popleft()
        ordered.append((edge_id, hops, path))
        if hops >= max_hops:
            continue
        for nxt in adj.get(edge_id, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, hops + 1, path + [nxt]))
    return ordered


def _component_center(
    junction_ids: Iterable[str],
    junctions: dict,
) -> Tuple[float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for jid in junction_ids:
        cx, cy = junctions[jid]["center"]
        xs.append(cx)
        ys.append(cy)
    if not xs:
        return (0.0, 0.0)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _derive_spokes(
    ring_nodes: Set[str],
    ring_edges: Set[str],
    edges: Dict[str, SumoEdge],
    junctions: dict,
    hub_center: Tuple[float, float],
    *,
    min_spoke_m: float = 5.0,
) -> Set[str]:
    """Edges that leave the network and attach to the SUMO roundabout ring."""
    spokes: Set[str] = set()
    for edge_id, edge in edges.items():
        if edge_id in ring_edges:
            continue
        if _edge_max_length(edge) < min_spoke_m:
            continue
        from_in = edge.from_node in ring_nodes
        to_in = edge.to_node in ring_nodes
        if from_in and not to_in:
            inside, outside = edge.from_node, edge.to_node
        elif to_in and not from_in:
            inside, outside = edge.to_node, edge.from_node
        else:
            continue
        if outside not in junctions:
            continue
        outside_dist = _dist_xy(junctions[outside]["center"], hub_center)
        inside_dist = _dist_xy(junctions[inside]["center"], hub_center)
        if outside_dist > inside_dist + 5.0:
            spokes.add(edge_id)
    return spokes


def _hops_from_sign_to_roundabout(
    sign_edge_id: str,
    ring_edges: Set[str],
    spokes: Set[str],
    edges: Dict[str, SumoEdge],
    adj: Dict[str, List[str]],
) -> int:
    """Minimum BFS hops from the sign road to any ring or spoke edge."""
    target_edges = ring_edges | spokes
    best = 999
    for edge_id, hops, _ in _bfs_edges_from_sign(sign_edge_id, edges, adj):
        if edge_id in target_edges:
            best = min(best, hops)
    return best


def _pick_sumo_roundabout(
    roundabouts: List[SumoRoundabout],
    *,
    sign_edge_id: Optional[str],
    junctions: dict,
    edges: Dict[str, SumoEdge],
    adj: Dict[str, List[str]],
    min_ring_edges: int,
) -> SumoRoundabout:
    candidates: list[tuple[int, int, SumoRoundabout, Set[str]]] = []
    for rb in roundabouts:
        ring_edges = {eid for eid in rb.ring_edge_ids if eid in edges}
        if len(ring_edges) < min_ring_edges:
            continue
        ring_nodes = {
            node_id
            for node_id in rb.node_ids
            if node_id in junctions
        }
        if not ring_nodes:
            ring_nodes = {
                node_id
                for eid in ring_edges
                for node_id in (edges[eid].from_node, edges[eid].to_node)
                if node_id in junctions
            }
        center = _component_center(ring_nodes, junctions)
        spokes = _derive_spokes(ring_nodes, ring_edges, edges, junctions, center)
        if len(spokes) < 2:
            continue
        if sign_edge_id:
            hops = _hops_from_sign_to_roundabout(sign_edge_id, ring_edges, spokes, edges, adj)
            if hops >= 999:
                continue
        else:
            hops = 0
        candidates.append((hops, -len(ring_edges), rb, spokes))

    if not candidates:
        if sign_edge_id:
            raise JunctionLayoutError(
                f"No SUMO <roundabout> reachable from sign road {sign_edge_id!r}"
            )
        raise JunctionLayoutError("No qualifying SUMO <roundabout> blocks in net")

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _approach_edge_for_sign(
    sign_edge_id: str,
    spokes: Set[str],
    ring_nodes: Set[str],
    edges: Dict[str, SumoEdge],
    adj: Dict[str, List[str]],
) -> str:
    """Pick the spoke edge used for the 4.3 sign / ego approach."""
    for edge_id, _, _ in _bfs_edges_from_sign(sign_edge_id, edges, adj):
        if edge_id in spokes:
            return edge_id
    for edge_id, _, _ in _bfs_edges_from_sign(sign_edge_id, edges, adj):
        if edges[edge_id].to_node in ring_nodes:
            return edge_id
    return sign_edge_id


def _entry_junction_for_roundabout(
    ring_nodes: Set[str],
    junctions: dict,
    *,
    approach_edge_id: str,
    edges: Dict[str, SumoEdge],
) -> str:
    edge = edges.get(approach_edge_id)
    if edge is not None:
        if edge.to_node in ring_nodes and edge.to_node in junctions:
            return edge.to_node
        if edge.from_node in ring_nodes and edge.from_node in junctions:
            return edge.from_node
    for node_id in sorted(ring_nodes):
        if node_id in junctions:
            return node_id
    return next(iter(ring_nodes))


def _roundabout_pick_from_sumo(
    rb: SumoRoundabout,
    junctions: dict,
    edges: Dict[str, SumoEdge],
    adj: Dict[str, List[str]],
    *,
    sign_edge_id: Optional[str],
    ego_spoke_edge_id: Optional[str],
) -> RoundaboutPick:
    ring_edges = {eid for eid in rb.ring_edge_ids if eid in edges}
    ring_nodes = {
        node_id
        for node_id in rb.node_ids
        if node_id in junctions
    }
    if not ring_nodes:
        ring_nodes = {
            node_id
            for eid in ring_edges
            for node_id in (edges[eid].from_node, edges[eid].to_node)
            if node_id in junctions
        }
    center = _component_center(ring_nodes, junctions)
    spokes = _derive_spokes(ring_nodes, ring_edges, edges, junctions, center)

    if sign_edge_id:
        approach_edge = ego_spoke_edge_id or _approach_edge_for_sign(
            sign_edge_id, spokes, ring_nodes, edges, adj
        )
    else:
        approach_edge = ego_spoke_edge_id or (next(iter(sorted(spokes))) if spokes else "")

    if ego_spoke_edge_id and ego_spoke_edge_id not in spokes:
        prefix = ego_spoke_edge_id.split("#", 1)[0]
        spoke_match = any(
            eid == ego_spoke_edge_id or eid.startswith(prefix + "#")
            for eid in spokes
        )
        if not spoke_match:
            raise JunctionLayoutError(
                f"Spoke edge {ego_spoke_edge_id!r} is not attached to the SUMO roundabout"
            )
        approach_edge = ego_spoke_edge_id

    entry = _entry_junction_for_roundabout(
        ring_nodes,
        junctions,
        approach_edge_id=approach_edge,
        edges=edges,
    )
    return RoundaboutPick(
        entry_junction_id=entry,
        approach_edge_id=approach_edge,
        center_xy=center,
        ring_junction_ids=tuple(sorted(ring_nodes)),
        ring_edge_ids=tuple(sorted(ring_edges)),
        spoke_edge_ids=tuple(sorted(spokes)),
    )


def detect_roundabout(
    net_path: Path | str,
    *,
    sign_edge_id: Optional[str] = None,
    min_ring_junctions: int = 2,
    min_ring_edges: int = 3,
    ego_spoke_edge_id: Optional[str] = None,
) -> RoundaboutPick:
    """Find the SUMO roundabout nearest the catalog sign road."""
    net_path = Path(net_path)
    if not net_path.is_file():
        raise JunctionLayoutError(f"net.xml not found: {net_path}")

    roundabouts = _parse_sumo_roundabouts(net_path)
    if not roundabouts:
        raise JunctionLayoutError(f"No <roundabout> blocks in {net_path.name}")

    junctions, edges, _, connections = _load_net(net_path)
    adj = _adjacent_from_connections(edges, connections)

    if sign_edge_id and sign_edge_id not in edges:
        raise JunctionLayoutError(f"Sign approach edge {sign_edge_id!r} not in net")

    rb = _pick_sumo_roundabout(
        roundabouts,
        sign_edge_id=sign_edge_id,
        junctions=junctions,
        edges=edges,
        adj=adj,
        min_ring_edges=min_ring_edges,
    )
    pick = _roundabout_pick_from_sumo(
        rb,
        junctions,
        edges,
        adj,
        sign_edge_id=sign_edge_id,
        ego_spoke_edge_id=ego_spoke_edge_id,
    )

    if len(pick.ring_junction_ids) < min_ring_junctions or len(pick.ring_edge_ids) < min_ring_edges:
        raise JunctionLayoutError(
            f"SUMO roundabout too small: {len(pick.ring_junction_ids)} junction(s), "
            f"{len(pick.ring_edge_ids)} ring edge(s)"
        )
    return pick


def _pick_from_edge_lists(
    net_path: Path,
    *,
    ring_edge_ids: Iterable[str],
    spoke_edge_ids: Iterable[str],
    entry_junction_id: Optional[str],
    sign_edge_id: Optional[str],
) -> RoundaboutPick:
    """Rebuild a RoundaboutPick from stored metadata on a cropped net."""
    junctions, edges, _, connections = _load_net(net_path)
    ring_edges = {eid for eid in ring_edge_ids if eid in edges}
    spokes = {eid for eid in spoke_edge_ids if eid in edges}
    if not ring_edges:
        raise JunctionLayoutError("No ring edges from metadata are present in cropped net")

    ring_juncs: Set[str] = set()
    for eid in ring_edges:
        edge = edges[eid]
        ring_juncs.add(edge.from_node)
        ring_juncs.add(edge.to_node)

    entry = entry_junction_id or ""
    if entry not in junctions:
        entry = next(iter(ring_juncs))

    center = junctions[entry]["center"] if entry in junctions else _component_center(ring_juncs, junctions)
    adj = _adjacent_from_connections(edges, connections)
    approach = sign_edge_id or ""
    if approach and approach not in edges:
        approach = _approach_edge_for_sign(approach, spokes, ring_juncs, edges, adj)

    return RoundaboutPick(
        entry_junction_id=entry,
        approach_edge_id=approach,
        center_xy=center,
        ring_junction_ids=tuple(sorted(ring_juncs)),
        ring_edge_ids=tuple(sorted(ring_edges)),
        spoke_edge_ids=tuple(sorted(spokes)),
    )


def try_detect_roundabout(
    net_path: Path | str,
    *,
    sign_edge_id: Optional[str] = None,
) -> Optional[RoundaboutPick]:
    try:
        return detect_roundabout(net_path, sign_edge_id=sign_edge_id)
    except JunctionLayoutError:
        return None


def _spoke_arms_for_component(
    pick: RoundaboutPick,
    junctions: dict,
    edges: Dict[str, SumoEdge],
    straight_map: Dict[str, Set[str]],
    outgoing_map: Dict[str, Set[str]],
    left_map: Dict[str, Set[str]],
) -> List[ApproachArm]:
    """Build secondary (spoke) approach arms — one per spoke edge in the component."""
    arms: List[ApproachArm] = []
    center = pick.center_xy
    for edge_id in pick.spoke_edge_ids:
        edge = edges.get(edge_id)
        if edge is None or not edge.lanes:
            continue
        lane_keys = [lane.metadrive_key for lane in sorted(edge.lanes, key=lambda l: l.lane_num)]
        entry_lane = max(edge.lanes, key=lambda l: l.length)
        arms.append(
            ApproachArm(
                edge_id=edge_id,
                lane_keys=lane_keys,
                entry_point=_entry_point_for_lane(entry_lane, edge),
                entry_angle=_angle_of_point(center, _entry_point_for_lane(entry_lane, edge)),
                arm_index=-1,
                road_class="secondary",
                straight_to=sorted(straight_map.get(edge_id, set())),
                outgoing_to=sorted(outgoing_map.get(edge_id, set())),
                left_to=sorted(left_map.get(edge_id, set())),
                from_node=edge.from_node,
                min_lane_length=min((lane.length for lane in edge.lanes), default=0.0),
            )
        )
    arms.sort(key=lambda arm: arm.entry_angle)
    for idx, arm in enumerate(arms):
        arm.arm_index = idx
    return arms


def _ring_arms_for_component(
    pick: RoundaboutPick,
    edges: Dict[str, SumoEdge],
    straight_map: Dict[str, Set[str]],
    outgoing_map: Dict[str, Set[str]],
    left_map: Dict[str, Set[str]],
) -> List[ApproachArm]:
    """Build main (ring) arms — one per ring edge, entry at the downstream junction end."""
    arms: List[ApproachArm] = []
    center = pick.center_xy
    for edge_id in pick.ring_edge_ids:
        edge = edges.get(edge_id)
        if edge is None or not edge.lanes:
            continue
        lane_keys = [lane.metadrive_key for lane in sorted(edge.lanes, key=lambda l: l.lane_num)]
        entry_lane = max(edge.lanes, key=lambda l: l.length)
        entry_point = _entry_point_for_lane(entry_lane, edge)
        arms.append(
            ApproachArm(
                edge_id=edge_id,
                lane_keys=lane_keys,
                entry_point=entry_point,
                entry_angle=_angle_of_point(center, entry_point),
                arm_index=-1,
                road_class="main",
                straight_to=sorted(straight_map.get(edge_id, set())),
                outgoing_to=sorted(outgoing_map.get(edge_id, set())),
                left_to=sorted(left_map.get(edge_id, set())),
                from_node=edge.from_node,
                min_lane_length=min((lane.length for lane in edge.lanes), default=0.0),
            )
        )
    arms.sort(key=lambda arm: arm.entry_angle)
    for idx, arm in enumerate(arms):
        arm.arm_index = idx
    return arms


def build_roundabout_layout(
    net_path: Path | str,
    *,
    sign_edge_id: Optional[str] = None,
    require_ego_secondary: bool = False,
    ring_edge_ids: Optional[Iterable[str]] = None,
    spoke_edge_ids: Optional[Iterable[str]] = None,
    entry_junction_id: Optional[str] = None,
) -> JunctionPriorityLayout:
    """Layout with ring edges as main road and spoke approaches as secondary."""
    net_path = Path(net_path)
    if ring_edge_ids is not None:
        pick = _pick_from_edge_lists(
            net_path,
            ring_edge_ids=ring_edge_ids,
            spoke_edge_ids=spoke_edge_ids or (),
            entry_junction_id=entry_junction_id,
            sign_edge_id=sign_edge_id,
        )
    else:
        pick = detect_roundabout(net_path, sign_edge_id=sign_edge_id)
    junctions, edges, _, connections = _load_net(net_path)

    all_edge_ids = set(pick.ring_edge_ids) | set(pick.spoke_edge_ids)
    straight_map = {
        eid: _straight_targets(eid, connections) for eid in all_edge_ids if eid in edges
    }
    outgoing_map = {
        eid: _outgoing_targets(eid, connections) for eid in all_edge_ids if eid in edges
    }
    left_map = {eid: _left_targets(eid, connections) for eid in all_edge_ids if eid in edges}

    secondary_arms = _spoke_arms_for_component(
        pick, junctions, edges, straight_map, outgoing_map, left_map
    )
    main_arms = _ring_arms_for_component(
        pick, edges, straight_map, outgoing_map, left_map
    )
    arms = secondary_arms + main_arms

    main_ids = set(pick.ring_edge_ids)
    secondary_ids = set(pick.spoke_edge_ids)

    if sign_edge_id and require_ego_secondary:
        if sign_edge_id not in secondary_ids:
            if pick.approach_edge_id and pick.approach_edge_id in secondary_ids:
                pass
            else:
                raise JunctionLayoutError(
                    f"ego_edge_id {sign_edge_id!r} is not a spoke approach to the roundabout"
                )

    return JunctionPriorityLayout(
        junction_id=pick.entry_junction_id,
        junction_type=junctions[pick.entry_junction_id]["type"],
        shape="O",
        mode="roundabout",
        center=pick.center_xy,
        arms=arms,
        main_edge_ids=main_ids,
        secondary_edge_ids=secondary_ids,
    )
