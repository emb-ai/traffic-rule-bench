"""Lane-change dual-path scenarios for PDD 5.15.1.

Requirement: at least one junction approach has ≥2 lanes. Ego spawns on a lane
that has **no** first-exit route toward the destination; a peer lane on the same
approach does. Compliant behaviour = lane-change onto the peer, then follow its
exit to dest. Wrong path = stay on the spawn lane and take its natural exit
(does not reach dest — temptation / violation of the assigned task).

Meta field naming keeps historical dual_path keys:
  * ``turn_*`` / baseline = wrong path (no lane-change)
  * ``straight_*`` / compliant = correct path (after lane-change)
"""

from __future__ import annotations

import heapq
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .direction_sign_spec import DEFAULT_PDD_CODE, get_direction_sign_spec
from .junction_crop import JunctionPick, collect_intersection_junction_candidates
from .junction_priority_layout import INTERSECTION_JUNCTION_TYPES, JunctionLayoutError, _load_net
from .lane_keys import make_lane_key
from .sumo_utils import is_real_sumo_edge_id, is_vehicle_drivable_lane


TurnDir = str
_CARDINAL = frozenset({"s", "l", "r"})


@dataclass(frozen=True)
class DualPathScenario:
    """Spawn on wrong lane; dest reachable only from target peer lane."""

    junction_id: str
    junction_center_xy: Tuple[float, float]
    ego_edge_id: str
    ego_lane_num: int  # spawn / wrong lane
    dest_edge_id: str
    dest_lane_num: int
    turn_dir: TurnDir  # wrong first-exit dir
    turn_first_exit: str
    straight_first_exit: str
    turn_path: Tuple[str, ...]  # wrong edge path (no dest)
    straight_path: Tuple[str, ...]  # correct edge path to dest
    turn_length_m: float
    straight_length_m: float
    compliant_dir: str = "s"
    pdd_code: str = DEFAULT_PDD_CODE
    target_lane_num: int = 0  # correct peer lane
    approach_lane_dirs: Tuple[Tuple[int, Tuple[str, ...]], ...] = ()

    @property
    def gain_m(self) -> float:
        # Prefer longer wrong diversion vs short correct hop when ranking.
        return float(self.turn_length_m - self.straight_length_m)

    @property
    def baseline_dir(self) -> str:
        return self.turn_dir

    @property
    def path_edge_ids(self) -> Tuple[str, ...]:
        seen: Set[str] = set()
        out: List[str] = []
        for eid in (self.ego_edge_id, *self.turn_path, *self.straight_path):
            if eid not in seen:
                seen.add(eid)
                out.append(eid)
        return tuple(out)

    def to_meta_fields(self) -> dict:
        spec = get_direction_sign_spec(self.pdd_code)
        return {
            "road_id": self.ego_edge_id,
            "spawn_lane_num": self.ego_lane_num,
            "target_lane_num": self.target_lane_num,
            "destination_edge_id": self.dest_edge_id,
            "destination_lane_id": make_lane_key(self.dest_edge_id, self.dest_lane_num),
            "dual_path": {
                "kind": "lane_change",
                "turn_dir": self.turn_dir,
                "baseline_dir": self.turn_dir,
                "compliant_dir": self.compliant_dir,
                "spawn_lane_num": self.ego_lane_num,
                "target_lane_num": self.target_lane_num,
                "turn_first_exit": self.turn_first_exit,
                "straight_first_exit": self.straight_first_exit,
                "turn_path": list(self.turn_path),
                "straight_path": list(self.straight_path),
                "turn_length_m": self.turn_length_m,
                "straight_length_m": self.straight_length_m,
                "gain_m": self.gain_m,
                "approach_lane_dirs": {
                    str(ln): list(dirs) for ln, dirs in self.approach_lane_dirs
                },
            },
            "pdd_code": spec.pdd_code,
            "allowed_dirs": [],
        }


@dataclass
class _EdgeGraph:
    edge_length: Dict[str, float]
    edge_to_node: Dict[str, str]
    edge_from_node: Dict[str, str]
    lane_nums: Dict[str, List[int]]
    shapes: Dict[str, List[Tuple[float, float]]]
    lane_shapes: Dict[Tuple[str, int], List[Tuple[float, float]]]
    adj: Dict[str, List[Tuple[str, float]]]
    # edge -> lane_num -> dir -> set(to_edge)
    lane_first_exits: Dict[str, Dict[int, Dict[str, Set[str]]]]
    first_exits: Dict[str, Dict[str, Set[str]]]
    junctions: dict


def _parse_shape(shape_str: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for token in (shape_str or "").split():
        if "," not in token:
            continue
        x_s, y_s = token.split(",", 1)
        pts.append((float(x_s), float(y_s)))
    return pts


def _norm_dir(raw: str | None) -> str:
    d = (raw or "").strip()
    if d == "R":
        return "r"
    if d == "L":
        return "l"
    d = d.lower()
    return d if d in _CARDINAL or d == "t" else ""


def build_edge_graph(net_path: Path | str) -> _EdgeGraph:
    root = ET.parse(net_path).getroot()
    edge_length: Dict[str, float] = {}
    edge_to_node: Dict[str, str] = {}
    edge_from_node: Dict[str, str] = {}
    lane_nums: Dict[str, List[int]] = {}
    shapes: Dict[str, List[Tuple[float, float]]] = {}
    lane_shapes: Dict[Tuple[str, int], List[Tuple[float, float]]] = {}

    for edge_el in root.findall("edge"):
        eid = edge_el.get("id", "")
        if not is_real_sumo_edge_id(eid):
            continue
        if edge_el.get("function", "normal") == "internal":
            continue
        drivable_lanes = [
            lane_el
            for lane_el in edge_el.findall("lane")
            if is_vehicle_drivable_lane(lane_el)
        ]
        if not drivable_lanes:
            continue
        lengths = []
        nums: List[int] = []
        best_shape: List[Tuple[float, float]] = []
        for lane_el in drivable_lanes:
            lid = lane_el.get("id", "")
            try:
                num = int(lid.rsplit("_", 1)[1])
            except (ValueError, IndexError):
                num = 0
            nums.append(num)
            length = float(lane_el.get("length") or 0.0)
            shape = _parse_shape(lane_el.get("shape", ""))
            if length <= 0.0 and len(shape) >= 2:
                length = sum(
                    math.hypot(shape[i + 1][0] - shape[i][0], shape[i + 1][1] - shape[i][1])
                    for i in range(len(shape) - 1)
                )
            lengths.append(length)
            lane_shapes[(eid, num)] = shape
            if len(shape) > len(best_shape):
                best_shape = shape
        edge_length[eid] = max(lengths) if lengths else 0.0
        edge_to_node[eid] = edge_el.get("to", "")
        edge_from_node[eid] = edge_el.get("from", "")
        lane_nums[eid] = sorted(set(nums))
        shapes[eid] = best_shape or _parse_shape(edge_el.get("shape", ""))

    adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    first_exits: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    lane_first_exits: Dict[str, Dict[int, Dict[str, Set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    for conn in root.findall("connection"):
        fr = conn.get("from", "")
        to = conn.get("to", "")
        if fr not in edge_length or to not in edge_length:
            continue
        cost = edge_length[to]
        adj[fr].append((to, cost))
        d = _norm_dir(conn.get("dir"))
        if d in _CARDINAL:
            first_exits[fr][d].add(to)
            try:
                fl = int(conn.get("fromLane", "0"))
            except ValueError:
                fl = 0
            lane_first_exits[fr][fl][d].add(to)

    junctions, _, _, _ = _load_net(Path(net_path))
    return _EdgeGraph(
        edge_length=edge_length,
        edge_to_node=edge_to_node,
        edge_from_node=edge_from_node,
        lane_nums=lane_nums,
        shapes=shapes,
        lane_shapes=lane_shapes,
        adj=dict(adj),
        lane_first_exits={
            e: {ln: dict(dirs) for ln, dirs in lanes.items()}
            for e, lanes in lane_first_exits.items()
        },
        first_exits={k: dict(v) for k, v in first_exits.items()},
        junctions=junctions,
    )


def dual_path_role_dirs(pdd_code: str) -> Tuple[List[str], List[str]]:
    """Compatibility stub — 5.15.1 roles are per-lane, not class-wide."""
    del pdd_code
    return ["wrong"], ["correct"]


def _dijkstra_from(
    graph: _EdgeGraph,
    starts: Sequence[Tuple[str, float]],
    *,
    goal: Optional[str] = None,
    max_cost: float = 800.0,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    dist: Dict[str, float] = {}
    prev: Dict[str, str] = {}
    pq: List[Tuple[float, str]] = []
    for edge_id, cost0 in starts:
        if edge_id not in graph.edge_length:
            continue
        if cost0 < dist.get(edge_id, math.inf):
            dist[edge_id] = cost0
            heapq.heappush(pq, (cost0, edge_id))
    while pq:
        cost, u = heapq.heappop(pq)
        if cost != dist.get(u):
            continue
        if goal is not None and u == goal:
            break
        if cost > max_cost:
            continue
        for v, w in graph.adj.get(u, []):
            nd = cost + w
            if nd > max_cost:
                continue
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def _rebuild_path(prev: Dict[str, str], start_set: Set[str], goal: str) -> Optional[List[str]]:
    if goal not in prev and goal not in start_set:
        return None
    path = [goal]
    cur = goal
    guard = 0
    while cur not in start_set:
        parent = prev.get(cur)
        if parent is None:
            return None
        path.append(parent)
        cur = parent
        guard += 1
        if guard > 10_000:
            return None
    path.reverse()
    return path


def _default_lane_num(graph: _EdgeGraph, edge_id: str) -> int:
    nums = graph.lane_nums.get(edge_id) or [0]
    return int(nums[0])


def _path_bbox(
    graph: _EdgeGraph,
    edge_ids: Iterable[str],
    *,
    margin_m: float,
) -> Optional[Tuple[float, float, float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    for eid in edge_ids:
        for x, y in graph.shapes.get(eid, []):
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return (min(xs) - margin_m, min(ys) - margin_m, max(xs) + margin_m, max(ys) + margin_m)


def path_union_bbox(
    graph: _EdgeGraph,
    scenario: DualPathScenario,
    *,
    margin_m: float = 40.0,
) -> Optional[Tuple[float, float, float, float]]:
    return _path_bbox(graph, scenario.path_edge_ids, margin_m=margin_m)


def path_revisits_signed_approach(ego_edge_id: str, path: Sequence[str]) -> bool:
    return bool(ego_edge_id) and bool(path) and ego_edge_id in path


def straight_path_reenters_signed_junction(
    net_path: Path | str,
    scenario: DualPathScenario,
) -> bool:
    del net_path
    return path_revisits_signed_approach(scenario.ego_edge_id, scenario.straight_path)


def straight_path_has_dead_end_uturn(
    net_path: Path | str,
    scenario: DualPathScenario,
) -> bool:
    del net_path, scenario
    return False


def _lane_dirs(graph: _EdgeGraph, edge_id: str, lane_num: int) -> Set[str]:
    exits = (graph.lane_first_exits.get(edge_id) or {}).get(lane_num) or {}
    return {d for d, tos in exits.items() if tos and d in _CARDINAL}


def _lane_exit_edges(graph: _EdgeGraph, edge_id: str, lane_num: int) -> Dict[str, Set[str]]:
    return dict((graph.lane_first_exits.get(edge_id) or {}).get(lane_num) or {})


def _can_reach_from_exits(
    graph: _EdgeGraph,
    exit_edges: Set[str],
    dest: str,
    *,
    max_cost: float,
) -> Tuple[Optional[List[str]], float, Optional[str]]:
    if not exit_edges or dest not in graph.edge_length:
        return None, math.inf, None
    starts = [(e, graph.edge_length[e]) for e in exit_edges]
    dist, prev = _dijkstra_from(graph, starts, goal=dest, max_cost=max_cost)
    if dest not in dist:
        return None, math.inf, None
    path = _rebuild_path(prev, set(exit_edges), dest)
    if not path:
        return None, math.inf, None
    return path, float(dist[dest]), path[0]


def _wrong_spur_path(
    graph: _EdgeGraph,
    exit_edges: Set[str],
    *,
    avoid: Set[str],
    max_cost: float = 250.0,
    min_len: float = 25.0,
) -> Tuple[Optional[List[str]], float, Optional[str]]:
    """Short path along a wrong-lane exit that does not go toward ``avoid`` dests."""
    best: Optional[List[str]] = None
    best_len = -1.0
    best_first: Optional[str] = None
    for start in sorted(exit_edges):
        dist, prev = _dijkstra_from(
            graph, [(start, graph.edge_length[start])], max_cost=max_cost
        )
        for edge, cost in dist.items():
            if edge in avoid:
                continue
            if cost < min_len:
                continue
            path = _rebuild_path(prev, {start}, edge)
            if not path:
                continue
            if path_revisits_signed_approach(start, path[1:]):
                continue
            if cost > best_len:
                best_len = cost
                best = path
                best_first = start
    return best, best_len, best_first


def _approach_lane_dirs_tuple(
    graph: _EdgeGraph, edge_id: str
) -> Tuple[Tuple[int, Tuple[str, ...]], ...]:
    out: List[Tuple[int, Tuple[str, ...]]] = []
    for ln in graph.lane_nums.get(edge_id) or []:
        dirs = tuple(sorted(_lane_dirs(graph, edge_id, ln)))
        out.append((int(ln), dirs))
    return tuple(out)


def _find_multi_lane_junctions(
    graph: _EdgeGraph,
    *,
    min_arm_lane_m: float = 0.5,
    min_arms: int = 3,
) -> List[Tuple[str, Tuple[float, float], List[str]]]:
    """Junctions where ≥1 incoming edge has ≥2 lanes."""
    out: List[Tuple[str, Tuple[float, float], List[str]]] = []
    for jid, info in graph.junctions.items():
        if info.get("type") not in INTERSECTION_JUNCTION_TYPES:
            continue
        incoming = [
            eid for eid, to_node in graph.edge_to_node.items() if to_node == jid
        ]
        if len(incoming) < min_arms:
            continue
        if not all(graph.edge_length.get(eid, 0.0) > min_arm_lane_m for eid in incoming):
            continue
        multi = [eid for eid in incoming if len(graph.lane_nums.get(eid) or []) >= 2]
        if not multi:
            continue
        center = info["center"]
        out.append((jid, (float(center[0]), float(center[1])), sorted(incoming)))
    out.sort(key=lambda item: item[0])
    return out


def find_dual_path_scenarios(
    net_path: Path | str,
    *,
    pdd_code: str = DEFAULT_PDD_CODE,
    min_lane_length_m: float = 8.0,
    min_arm_lane_m: float = 0.5,
    min_gain_m: float = 0.0,
    max_turn_length_m: float = 350.0,
    max_straight_length_m: float = 700.0,
    max_scenarios: int = 20,
    dests_per_arm: int = 1,
    junction_ids: Optional[Sequence[str]] = None,
    require_uturn_continuation: bool = True,
    min_arms: int = 3,
) -> List[DualPathScenario]:
    """Find spawn-on-wrong-lane / dest-via-peer-lane scenarios."""
    del require_uturn_continuation
    pdd_code = get_direction_sign_spec(pdd_code).pdd_code
    net_path = Path(net_path)
    graph = build_edge_graph(net_path)

    if junction_ids is None:
        junctions = _find_multi_lane_junctions(
            graph, min_arm_lane_m=min_arm_lane_m, min_arms=min_arms
        )
    else:
        junctions = []
        for jid in junction_ids:
            info = graph.junctions.get(jid)
            if not info:
                continue
            incoming = [
                eid for eid, to_node in graph.edge_to_node.items() if to_node == jid
            ]
            multi = [eid for eid in incoming if len(graph.lane_nums.get(eid) or []) >= 2]
            if not multi:
                continue
            center = info["center"]
            junctions.append((jid, (float(center[0]), float(center[1])), sorted(incoming)))

    scenarios: List[DualPathScenario] = []
    for jid, center, incoming in junctions:
        for ego in incoming:
            if len(graph.lane_nums.get(ego) or []) < 2:
                continue
            if graph.edge_length.get(ego, 0.0) <= min_lane_length_m:
                continue

            lane_nums = list(graph.lane_nums.get(ego) or [])
            approach_dirs = _approach_lane_dirs_tuple(graph, ego)

            # Pair (spawn/wrong, target/correct): target must have a first-exit
            # edge that spawn does not (lane-level exclusive turn at the junction).
            for target_ln in lane_nums:
                target_exits = _lane_exit_edges(graph, ego, target_ln)
                target_exit_edges: Set[str] = set()
                target_exit_dir: Dict[str, str] = {}
                for d, tos in target_exits.items():
                    for t in tos:
                        target_exit_edges.add(t)
                        target_exit_dir.setdefault(t, d)
                if not target_exit_edges:
                    continue

                for spawn_ln in lane_nums:
                    if spawn_ln == target_ln:
                        continue
                    spawn_exits = _lane_exit_edges(graph, ego, spawn_ln)
                    spawn_exit_edges: Set[str] = set()
                    spawn_exit_dir: Dict[str, str] = {}
                    for d, tos in spawn_exits.items():
                        for t in tos:
                            spawn_exit_edges.add(t)
                            spawn_exit_dir.setdefault(t, d)
                    if not spawn_exit_edges:
                        continue

                    exclusive = target_exit_edges - spawn_exit_edges
                    if not exclusive:
                        continue

                    # Destinations via exclusive first exits only.
                    dist_from_excl, prev_from_excl = _dijkstra_from(
                        graph,
                        [(e, graph.edge_length[e]) for e in exclusive],
                        max_cost=max_straight_length_m,
                    )
                    dest_candidates = sorted(
                        ((cost, edge) for edge, cost in dist_from_excl.items() if edge != ego),
                        key=lambda x: x[0],
                    )
                    kept = 0
                    for dest_cost, dest in dest_candidates:
                        if kept >= max(1, dests_per_arm):
                            break
                        correct_path = _rebuild_path(prev_from_excl, exclusive, dest)
                        if not correct_path:
                            continue
                        if path_revisits_signed_approach(ego, correct_path):
                            continue
                        compliant_first = correct_path[0]
                        if compliant_first not in exclusive:
                            continue
                        # Lane-level: spawn has no first-hop onto the compliant exit.
                        if compliant_first in spawn_exit_edges:
                            continue

                        compliant_dir = target_exit_dir.get(compliant_first, "s")
                        wrong_path, wrong_len, wrong_first = _wrong_spur_path(
                            graph,
                            spawn_exit_edges,
                            avoid={dest, *exclusive},
                            max_cost=max_turn_length_m,
                        )
                        if not wrong_path or wrong_first is None:
                            wrong_first = sorted(spawn_exit_edges)[0]
                            wrong_path = [wrong_first]
                            wrong_len = float(graph.edge_length.get(wrong_first, 0.0))

                        scenarios.append(
                            DualPathScenario(
                                junction_id=jid,
                                junction_center_xy=center,
                                ego_edge_id=ego,
                                ego_lane_num=int(spawn_ln),
                                dest_edge_id=dest,
                                dest_lane_num=_default_lane_num(graph, dest),
                                turn_dir=spawn_exit_dir.get(wrong_first, "s"),
                                turn_first_exit=wrong_first,
                                straight_first_exit=compliant_first,
                                turn_path=tuple(wrong_path),
                                straight_path=tuple(correct_path),
                                turn_length_m=float(wrong_len),
                                straight_length_m=float(dest_cost),
                                compliant_dir=compliant_dir,
                                pdd_code=pdd_code,
                                target_lane_num=int(target_ln),
                                approach_lane_dirs=approach_dirs,
                            )
                        )
                        kept += 1
                        if len(scenarios) >= max_scenarios * 4:
                            break
                    if len(scenarios) >= max_scenarios * 4:
                        break
                if len(scenarios) >= max_scenarios * 4:
                    break
            if len(scenarios) >= max_scenarios * 4:
                break
        if len(scenarios) >= max_scenarios * 4:
            break

    # Prefer short correct path, longer wrong spur, multi-lane clarity.
    scenarios.sort(
        key=lambda s: (s.straight_length_m, -s.turn_length_m, s.ego_edge_id, s.ego_lane_num)
    )
    # Dedup by (junction, ego, spawn_lane, dest)
    uniq: List[DualPathScenario] = []
    seen: Set[Tuple[str, str, int, str]] = set()
    for sc in scenarios:
        key = (sc.junction_id, sc.ego_edge_id, sc.ego_lane_num, sc.dest_edge_id)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(sc)
        if len(uniq) >= max_scenarios:
            break
    return uniq



def dual_path_scenario_from_meta(meta: dict) -> Optional[DualPathScenario]:
    """Rebuild the crop-time lane-change pick from ``meta.json`` fields."""
    ego = meta.get("road_id")
    dest = meta.get("destination_edge_id")
    dp = meta.get("dual_path")
    if not ego or not dest or not isinstance(dp, dict):
        return None
    turn_path = tuple(dp.get("turn_path") or ())
    straight_path = tuple(dp.get("straight_path") or ())
    if not turn_path or not straight_path:
        return None
    center = meta.get("junction_center_xy") or (0.0, 0.0)
    try:
        center_xy = (float(center[0]), float(center[1]))
    except (TypeError, ValueError, IndexError):
        center_xy = (0.0, 0.0)
    dest_lane_id = meta.get("destination_lane_id")
    try:
        dest_lane_num = (
            int(str(dest_lane_id).rsplit("_", 1)[-1]) if dest_lane_id else 0
        )
    except ValueError:
        dest_lane_num = 0
    pdd_code = str(meta.get("pdd_code") or dp.get("pdd_code") or DEFAULT_PDD_CODE)
    try:
        pdd_code = get_direction_sign_spec(pdd_code).pdd_code
    except ValueError:
        pdd_code = DEFAULT_PDD_CODE

    raw_dirs = dp.get("approach_lane_dirs") or meta.get("approach_lane_dirs") or {}
    approach_dirs: List[Tuple[int, Tuple[str, ...]]] = []
    if isinstance(raw_dirs, dict):
        for ln, dirs in raw_dirs.items():
            try:
                approach_dirs.append((int(ln), tuple(str(d) for d in (dirs or ()))))
            except (TypeError, ValueError):
                continue
    approach_dirs.sort(key=lambda x: x[0])

    target_lane_num = int(
        meta.get("target_lane_num")
        or dp.get("target_lane_num")
        or meta.get("spawn_lane_num")
        or 0
    )
    return DualPathScenario(
        junction_id=str(meta.get("junction_id") or ""),
        junction_center_xy=center_xy,
        ego_edge_id=str(ego),
        ego_lane_num=int(meta.get("spawn_lane_num") or dp.get("spawn_lane_num") or 0),
        dest_edge_id=str(dest),
        dest_lane_num=dest_lane_num,
        turn_dir=str(dp.get("baseline_dir") or dp.get("turn_dir") or "r"),
        turn_first_exit=str(dp.get("turn_first_exit") or turn_path[0]),
        straight_first_exit=str(dp.get("straight_first_exit") or straight_path[0]),
        turn_path=turn_path,
        straight_path=straight_path,
        turn_length_m=float(dp.get("turn_length_m") or 0.0),
        straight_length_m=float(dp.get("straight_length_m") or 0.0),
        compliant_dir=str(dp.get("compliant_dir") or "s"),
        pdd_code=pdd_code,
        target_lane_num=target_lane_num,
        approach_lane_dirs=tuple(approach_dirs),
    )


def rebuild_dual_path_on_net(
    net_path: Path | str,
    scenario: DualPathScenario,
    *,
    max_turn_length_m: float = 350.0,
    max_straight_length_m: float = 700.0,
) -> Optional[DualPathScenario]:
    graph = build_edge_graph(net_path)
    ego = scenario.ego_edge_id
    dest = scenario.dest_edge_id
    if ego not in graph.edge_length or dest not in graph.edge_length:
        return None
    if scenario.target_lane_num not in (graph.lane_nums.get(ego) or []):
        return None
    if scenario.ego_lane_num not in (graph.lane_nums.get(ego) or []):
        return None

    target_exits = _lane_exit_edges(graph, ego, scenario.target_lane_num)
    target_exit_edges: Set[str] = set()
    target_exit_dir: Dict[str, str] = {}
    for d, tos in target_exits.items():
        for t in tos:
            target_exit_edges.add(t)
            target_exit_dir.setdefault(t, d)

    spawn_exits = _lane_exit_edges(graph, ego, scenario.ego_lane_num)
    spawn_exit_edges: Set[str] = set()
    spawn_exit_dir: Dict[str, str] = {}
    for d, tos in spawn_exits.items():
        for t in tos:
            spawn_exit_edges.add(t)
            spawn_exit_dir.setdefault(t, d)

    exclusive = target_exit_edges - spawn_exit_edges
    if not exclusive:
        return None
    # Prefer keeping the original compliant first exit when still exclusive.
    seed = exclusive
    if scenario.straight_first_exit in exclusive:
        seed = {scenario.straight_first_exit}

    correct_path, correct_len, compliant_first = _can_reach_from_exits(
        graph, seed, dest, max_cost=max_straight_length_m
    )
    if not correct_path or compliant_first is None:
        return None
    if compliant_first in spawn_exit_edges:
        return None

    wrong_path, wrong_len, wrong_first = _wrong_spur_path(
        graph,
        spawn_exit_edges,
        avoid={dest, *exclusive},
        max_cost=max_turn_length_m,
    )
    if not wrong_path or wrong_first is None:
        if not spawn_exit_edges:
            return None
        wrong_first = sorted(spawn_exit_edges)[0]
        wrong_path = [wrong_first]
        wrong_len = float(graph.edge_length.get(wrong_first, 0.0))

    info = graph.junctions.get(scenario.junction_id)
    if info and info.get("center") is not None:
        center = (float(info["center"][0]), float(info["center"][1]))
    else:
        center = scenario.junction_center_xy

    return DualPathScenario(
        junction_id=scenario.junction_id,
        junction_center_xy=center,
        ego_edge_id=ego,
        ego_lane_num=scenario.ego_lane_num,
        dest_edge_id=dest,
        dest_lane_num=_default_lane_num(graph, dest),
        turn_dir=spawn_exit_dir.get(wrong_first, scenario.turn_dir),
        turn_first_exit=wrong_first,
        straight_first_exit=compliant_first,
        turn_path=tuple(wrong_path),
        straight_path=tuple(correct_path),
        turn_length_m=float(wrong_len),
        straight_length_m=float(correct_len),
        compliant_dir=target_exit_dir.get(compliant_first, scenario.compliant_dir),
        pdd_code=scenario.pdd_code,
        target_lane_num=scenario.target_lane_num,
        approach_lane_dirs=_approach_lane_dirs_tuple(graph, ego),
    )


def pick_best_dual_path_scenario(net_path: Path | str, **kwargs) -> Optional[DualPathScenario]:
    found = find_dual_path_scenarios(net_path, max_scenarios=1, **kwargs)
    return found[0] if found else None


def find_ranked_dual_path_picks(
    net_path: Path | str,
    *,
    pdd_code: str = DEFAULT_PDD_CODE,
    min_lane_length_m: float = 10.0,
    min_gain_m: float = 0.0,
    max_scenarios: int = 5,
    **kwargs,
) -> List[Tuple[DualPathScenario, JunctionPick]]:
    net_path = Path(net_path)
    scenarios = find_dual_path_scenarios(
        net_path,
        pdd_code=pdd_code,
        min_lane_length_m=min_lane_length_m,
        min_gain_m=min_gain_m,
        max_scenarios=max_scenarios,
        **kwargs,
    )
    picks = collect_intersection_junction_candidates(
        net_path,
        min_lane_length_m=min_lane_length_m,
        arm_counts=(3, 4),
    )
    by_id = {p.junction_id: p for p in picks}
    out: List[Tuple[DualPathScenario, JunctionPick]] = []
    for sc in scenarios:
        pick = by_id.get(sc.junction_id)
        if pick is None:
            pick = JunctionPick(
                junction_id=sc.junction_id,
                center_xy=sc.junction_center_xy,
                total_lanes=0,
                incoming_edge_ids=(sc.ego_edge_id,),
                arm_count=4,
            )
        out.append((sc, pick))
    return out


def crop_scene_to_dual_path_scenario(
    scene_dir: Path,
    scenario: DualPathScenario,
    *,
    source_net: Path,
    margin_m: float = 40.0,
    output_dir: Optional[Path] = None,
    output_scene_name: Optional[str] = None,
    output_net_name: str = "map.net.xml",
    base_meta: Optional[dict] = None,
    junction_rank: Optional[int] = None,
    core_scene_name: Optional[str] = None,
) -> DualPathScenario:
    """Crop ``source_net`` to the XY bbox of wrong+correct paths and write meta."""
    from .junction_crop import (
        crop_net_to_xy_boundary,
        json_dumps,
        net_xy_to_latlon,
        parse_net_location,
    )
    from .sumo_utils import load_scene_meta

    scene_dir = scene_dir.resolve()
    output_dir = (output_dir or scene_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(base_meta if base_meta is not None else load_scene_meta(scene_dir))
    source_net = source_net.resolve()
    core_name = core_scene_name or meta.get("scene_name", scene_dir.name)

    graph = build_edge_graph(source_net)
    bbox = path_union_bbox(graph, scenario, margin_m=margin_m)
    if bbox is None:
        raise JunctionLayoutError(
            f"No geometry for lane-change scenario at junction {scenario.junction_id}"
        )

    out_net = output_dir / output_net_name
    last_error: Optional[Exception] = None
    cropped_scenario = scenario
    used_margin = margin_m
    used_bbox = bbox

    for attempt_margin in (margin_m, margin_m * 1.5, margin_m * 2.5):
        attempt_bbox = path_union_bbox(graph, scenario, margin_m=attempt_margin)
        if attempt_bbox is None:
            continue
        try:
            crop_net_to_xy_boundary(source_net, attempt_bbox, out_net)
        except JunctionLayoutError as exc:
            last_error = exc
            continue

        rebuilt = rebuild_dual_path_on_net(out_net, scenario)
        if rebuilt is None:
            last_error = JunctionLayoutError(
                f"Lane-change scenario did not survive crop "
                f"(junction {scenario.junction_id}, ego={scenario.ego_edge_id})"
            )
            continue
        if straight_path_reenters_signed_junction(out_net, rebuilt):
            last_error = JunctionLayoutError(
                f"Correct path revisits signed approach {scenario.ego_edge_id}"
            )
            continue
        cropped_scenario = rebuilt
        used_margin = attempt_margin
        used_bbox = attempt_bbox
        break
    else:
        raise JunctionLayoutError(str(last_error) if last_error else "crop failed")

    scenario = cropped_scenario
    bbox = used_bbox
    margin_m = used_margin

    conv, orig = parse_net_location(out_net if out_net.is_file() else source_net)
    try:
        center_lat, center_lon = net_xy_to_latlon(
            scenario.junction_center_xy[0],
            scenario.junction_center_xy[1],
            conv,
            orig,
        )
    except Exception:
        center_lat = meta.get("latitude")
        center_lon = meta.get("longitude")

    scene_name = output_scene_name or meta.get("scene_name", scene_dir.name)
    meta.update(
        {
            "scene_name": scene_name,
            "scene_kind": "lane_change_dual_path",
            "core_scene_name": core_name,
            "net_file": output_net_name,
            "latitude": center_lat,
            "longitude": center_lon,
            "crop_margin_m": margin_m,
            "crop_bbox_xy": [bbox[0], bbox[1], bbox[2], bbox[3]],
            "junction_id": scenario.junction_id,
            "junction_center_xy": [
                scenario.junction_center_xy[0],
                scenario.junction_center_xy[1],
            ],
            "sign_spawn_distance": 30.0,
            "crop_mode": "lane_change_dual_path_bbox",
            **scenario.to_meta_fields(),
        }
    )
    if junction_rank is not None:
        meta["junction_rank"] = junction_rank
    meta.pop("distance_from_start", None)

    center_path = output_dir / "center.json"
    if center_lat is not None and center_lon is not None:
        center_path.write_text(
            json_dumps({"lat": center_lat, "lon": center_lon}) + "\n",
            encoding="utf-8",
        )
    (output_dir / "meta.json").write_text(json_dumps(meta) + "\n", encoding="utf-8")
    return scenario
