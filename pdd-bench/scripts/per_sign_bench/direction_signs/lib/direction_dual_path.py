"""Select 4.1.1 scenes with one dest reachable via turn (shorter) and straight (longer).

Pipeline (variant 1):
  1. On the full core net, find an X junction + ego approach with both ``s`` and
     ``l``/``r`` first exits.
  2. Pick a destination edge reachable after *either* first exit, with
     ``len(turn) + min_gain < len(straight)``.
  3. Crop later by the XY bounding box of the two edge paths (+ margin).
"""

from __future__ import annotations

import heapq
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .junction_crop import JunctionPick, collect_intersection_junction_candidates
from .junction_priority_layout import INTERSECTION_JUNCTION_TYPES, JunctionLayoutError, _load_net
from .lane_keys import make_lane_key
from .sumo_utils import is_real_sumo_edge_id, is_vehicle_drivable_lane


TurnDir = str  # "l" | "r"


@dataclass(frozen=True)
class DualPathScenario:
    """One ego approach + shared dest with turn-shorter / straight-longer routes."""

    junction_id: str
    junction_center_xy: Tuple[float, float]
    ego_edge_id: str
    ego_lane_num: int
    dest_edge_id: str
    dest_lane_num: int
    turn_dir: TurnDir
    turn_first_exit: str
    straight_first_exit: str
    turn_path: Tuple[str, ...]
    straight_path: Tuple[str, ...]
    turn_length_m: float
    straight_length_m: float

    @property
    def gain_m(self) -> float:
        return float(self.straight_length_m - self.turn_length_m)

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
        return {
            "road_id": self.ego_edge_id,
            "spawn_lane_num": self.ego_lane_num,
            "destination_edge_id": self.dest_edge_id,
            "destination_lane_id": make_lane_key(self.dest_edge_id, self.dest_lane_num),
            "dual_path": {
                "turn_dir": self.turn_dir,
                "turn_first_exit": self.turn_first_exit,
                "straight_first_exit": self.straight_first_exit,
                "turn_path": list(self.turn_path),
                "straight_path": list(self.straight_path),
                "turn_length_m": self.turn_length_m,
                "straight_length_m": self.straight_length_m,
                "gain_m": self.gain_m,
            },
            "pdd_code": "4.1.1",
            "allowed_dirs": ["s"],
        }


@dataclass
class _EdgeGraph:
    edge_length: Dict[str, float]
    edge_to_node: Dict[str, str]
    edge_from_node: Dict[str, str]
    lane_nums: Dict[str, List[int]]
    shapes: Dict[str, List[Tuple[float, float]]]
    adj: Dict[str, List[Tuple[str, float]]]
    first_exits: Dict[str, Dict[str, Set[str]]]  # from_edge -> dir -> to_edges
    junctions: dict


def _parse_shape(shape_str: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for token in (shape_str or "").split():
        if "," not in token:
            continue
        x_s, y_s = token.split(",", 1)
        pts.append((float(x_s), float(y_s)))
    return pts


def build_edge_graph(net_path: Path | str) -> _EdgeGraph:
    """Build a length-weighted road-edge graph from a SUMO net."""
    root = ET.parse(net_path).getroot()
    edge_length: Dict[str, float] = {}
    edge_to_node: Dict[str, str] = {}
    edge_from_node: Dict[str, str] = {}
    lane_nums: Dict[str, List[int]] = {}
    shapes: Dict[str, List[Tuple[float, float]]] = {}

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
                nums.append(int(lid.rsplit("_", 1)[1]))
            except (ValueError, IndexError):
                nums.append(0)
            length = float(lane_el.get("length") or 0.0)
            shape = _parse_shape(lane_el.get("shape", ""))
            if length <= 0.0 and len(shape) >= 2:
                length = sum(
                    math.hypot(shape[i + 1][0] - shape[i][0], shape[i + 1][1] - shape[i][1])
                    for i in range(len(shape) - 1)
                )
            lengths.append(length)
            if len(shape) > len(best_shape):
                best_shape = shape
        edge_length[eid] = max(lengths) if lengths else 0.0
        edge_to_node[eid] = edge_el.get("to", "")
        edge_from_node[eid] = edge_el.get("from", "")
        lane_nums[eid] = sorted(set(nums))
        shapes[eid] = best_shape or _parse_shape(edge_el.get("shape", ""))

    adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    first_exits: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for conn in root.findall("connection"):
        fr = conn.get("from", "")
        to = conn.get("to", "")
        if fr not in edge_length or to not in edge_length:
            continue
        cost = edge_length[to]
        adj[fr].append((to, cost))
        d = (conn.get("dir") or "").strip().lower()
        if d == "R":
            d = "r"
        if d in ("s", "l", "r"):
            first_exits[fr][d].add(to)

    junctions, _, _, _ = _load_net(Path(net_path))
    return _EdgeGraph(
        edge_length=edge_length,
        edge_to_node=edge_to_node,
        edge_from_node=edge_from_node,
        lane_nums=lane_nums,
        shapes=shapes,
        adj=dict(adj),
        first_exits={k: dict(v) for k, v in first_exits.items()},
        junctions=junctions,
    )


def _dijkstra_from(
    graph: _EdgeGraph,
    starts: Sequence[Tuple[str, float]],
    *,
    goal: Optional[str] = None,
    max_cost: float = 800.0,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Shortest-path distances from seeded (edge, cost) starts.

    ``cost`` already includes that start edge's length (entered onto it).
    """
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
    return (
        min(xs) - margin_m,
        min(ys) - margin_m,
        max(xs) + margin_m,
        max(ys) + margin_m,
    )


def path_union_bbox(
    graph: _EdgeGraph,
    scenario: DualPathScenario,
    *,
    margin_m: float = 40.0,
) -> Optional[Tuple[float, float, float, float]]:
    """XY bbox (xmin, ymin, xmax, ymax) covering turn + straight paths + margin."""
    return _path_bbox(
        graph,
        scenario.path_edge_ids,
        margin_m=margin_m,
    )


def _find_x_junction_ids(
    net_path: Path,
    graph: _EdgeGraph,
    *,
    min_arm_lane_m: float = 3.0,
) -> List[Tuple[str, Tuple[float, float]]]:
    """Return (junction_id, center) for 4-arm intersection junctions."""
    out: List[Tuple[str, Tuple[float, float]]] = []
    for jid, info in graph.junctions.items():
        if info.get("type") not in INTERSECTION_JUNCTION_TYPES:
            continue
        incoming = [
            eid
            for eid, to_node in graph.edge_to_node.items()
            if to_node == jid
        ]
        if len(incoming) != 4:
            continue
        if not all(graph.edge_length.get(eid, 0.0) > min_arm_lane_m for eid in incoming):
            continue
        center = info["center"]
        out.append((jid, (float(center[0]), float(center[1]))))
    out.sort(key=lambda item: item[0])
    return out


def find_dual_path_scenarios(
    net_path: Path | str,
    *,
    min_lane_length_m: float = 8.0,
    min_arm_lane_m: float = 0.5,
    min_gain_m: float = 20.0,
    max_turn_length_m: float = 350.0,
    max_straight_length_m: float = 700.0,
    max_scenarios: int = 20,
    dests_per_arm: int = 1,
    junction_ids: Optional[Sequence[str]] = None,
) -> List[DualPathScenario]:
    """Find dual-path (turn shorter, straight longer) scenarios for 4.1.1.

    ``min_lane_length_m`` applies to the ego approach only (spawnable arm).
    Other X arms only need ``min_arm_lane_m`` so stubby OSM arms are allowed.
    Preference order: shorter turn path (MetaDrive ``shortest_path`` is capped
    at 10 hops and rejects far dual-path dests), then larger length gain.
    ``dests_per_arm`` keeps multiple dests per ego for MetaDrive filtering.
    """
    net_path = Path(net_path)
    graph = build_edge_graph(net_path)

    if junction_ids is None:
        x_junctions = _find_x_junction_ids(
            net_path, graph, min_arm_lane_m=min_arm_lane_m
        )
    else:
        x_junctions = []
        for jid in junction_ids:
            info = graph.junctions.get(jid)
            if not info:
                continue
            center = info["center"]
            x_junctions.append((jid, (float(center[0]), float(center[1]))))

    scenarios: List[DualPathScenario] = []
    for jid, center in x_junctions:
        info = graph.junctions.get(jid)
        if info is None:
            continue
        if info.get("type") not in INTERSECTION_JUNCTION_TYPES:
            continue
        incoming = [
            eid
            for eid, to_node in graph.edge_to_node.items()
            if to_node == jid
        ]
        if len(incoming) < 4:
            continue

        for ego in sorted(incoming):
            if graph.edge_length.get(ego, 0.0) <= min_lane_length_m:
                continue
            exits = graph.first_exits.get(ego) or {}
            s_exits = set(exits.get("s") or ())
            if not s_exits:
                continue
            turn_options: List[Tuple[TurnDir, Set[str]]] = []
            for d in ("r", "l"):
                ex = set(exits.get(d) or ())
                if ex:
                    turn_options.append((d, ex))
            if not turn_options:
                continue

            for turn_dir, turn_exits in turn_options:
                turn_starts = [(e, graph.edge_length[e]) for e in turn_exits]
                straight_starts = [(e, graph.edge_length[e]) for e in s_exits]
                turn_dist, turn_prev = _dijkstra_from(
                    graph, turn_starts, max_cost=max_turn_length_m
                )
                straight_dist, straight_prev = _dijkstra_from(
                    graph, straight_starts, max_cost=max_straight_length_m
                )

                shared = set(turn_dist) & set(straight_dist)
                shared -= {ego} | turn_exits | s_exits
                # Drop other approaches into the same junction (not beyond it).
                shared = {
                    dest for dest in shared if graph.edge_to_node.get(dest) != jid
                }

                # Rank by short turn first so dests stay within MetaDrive's
                # 10-hop navigation limit; secondary key is larger gain.
                arm_cands: List[DualPathScenario] = []
                for dest in shared:
                    lt = turn_dist[dest]
                    ls = straight_dist[dest]
                    if lt > max_turn_length_m or ls > max_straight_length_m:
                        continue
                    if ls - lt < min_gain_m:
                        continue
                    t_path = _rebuild_path(turn_prev, turn_exits, dest)
                    s_path = _rebuild_path(straight_prev, s_exits, dest)
                    if not t_path or not s_path:
                        continue
                    arm_cands.append(
                        DualPathScenario(
                            junction_id=jid,
                            junction_center_xy=center,
                            ego_edge_id=ego,
                            ego_lane_num=_default_lane_num(graph, ego),
                            dest_edge_id=dest,
                            dest_lane_num=_default_lane_num(graph, dest),
                            turn_dir=turn_dir,
                            turn_first_exit=t_path[0],
                            straight_first_exit=s_path[0],
                            turn_path=tuple(t_path),
                            straight_path=tuple(s_path),
                            turn_length_m=float(lt),
                            straight_length_m=float(ls),
                        )
                    )
                arm_cands.sort(key=lambda s: (s.turn_length_m, -s.gain_m))
                for cand in arm_cands[:max(1, dests_per_arm)]:
                    scenarios.append(cand)

    scenarios.sort(
        key=lambda s: (s.turn_length_m, -s.gain_m, s.junction_id, s.ego_edge_id)
    )
    # Deduplicate by (junction, ego, dest) keeping best ranking.
    seen_dest: Set[Tuple[str, str, str]] = set()
    # Cap distinct ego arms, but allow several dests per arm when requested.
    seen_arm_count: Dict[Tuple[str, str], int] = defaultdict(int)
    unique: List[DualPathScenario] = []
    for sc in scenarios:
        dest_key = (sc.junction_id, sc.ego_edge_id, sc.dest_edge_id)
        if dest_key in seen_dest:
            continue
        arm_key = (sc.junction_id, sc.ego_edge_id)
        if seen_arm_count[arm_key] >= max(1, dests_per_arm):
            continue
        seen_dest.add(dest_key)
        seen_arm_count[arm_key] += 1
        unique.append(sc)
        if len(unique) >= max_scenarios:
            break
    return unique


def dual_path_scenario_from_meta(meta: dict) -> Optional[DualPathScenario]:
    """Rebuild the crop-time dual-path pick from ``meta.json`` fields.

    Returns None if required spawn/dest/path fields are missing. Manifest
    generation should prefer this over rediscovering routes.
    """
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
    return DualPathScenario(
        junction_id=str(meta.get("junction_id") or ""),
        junction_center_xy=center_xy,
        ego_edge_id=str(ego),
        ego_lane_num=int(meta.get("spawn_lane_num") or 0),
        dest_edge_id=str(dest),
        dest_lane_num=dest_lane_num,
        turn_dir=str(dp.get("turn_dir") or "r"),
        turn_first_exit=str(dp.get("turn_first_exit") or turn_path[0]),
        straight_first_exit=str(dp.get("straight_first_exit") or straight_path[0]),
        turn_path=turn_path,
        straight_path=straight_path,
        turn_length_m=float(dp.get("turn_length_m") or 0.0),
        straight_length_m=float(dp.get("straight_length_m") or 0.0),
    )


def rebuild_dual_path_on_net(
    net_path: Path | str,
    scenario: DualPathScenario,
    *,
    max_turn_length_m: float = 350.0,
    max_straight_length_m: float = 700.0,
) -> Optional[DualPathScenario]:
    """Recompute turn/straight paths for the same ego→dest on ``net_path``.

    Preserves crop-time endpoints when only geometry/paths need refreshing after
    a crop. Returns None if either path is missing on the net.
    """
    graph = build_edge_graph(net_path)
    ego = scenario.ego_edge_id
    dest = scenario.dest_edge_id
    if ego not in graph.edge_length or dest not in graph.edge_length:
        return None
    exits = graph.first_exits.get(ego) or {}
    s_exits = set(exits.get("s") or ())
    turn_exits = set(exits.get(scenario.turn_dir) or ())
    if not s_exits or not turn_exits:
        return None
    turn_starts = [(e, graph.edge_length[e]) for e in turn_exits]
    straight_starts = [(e, graph.edge_length[e]) for e in s_exits]
    turn_dist, turn_prev = _dijkstra_from(
        graph, turn_starts, goal=dest, max_cost=max_turn_length_m
    )
    straight_dist, straight_prev = _dijkstra_from(
        graph, straight_starts, goal=dest, max_cost=max_straight_length_m
    )
    if dest not in turn_dist or dest not in straight_dist:
        return None
    t_path = _rebuild_path(turn_prev, turn_exits, dest)
    s_path = _rebuild_path(straight_prev, s_exits, dest)
    if not t_path or not s_path:
        return None
    info = graph.junctions.get(scenario.junction_id)
    if info and info.get("center") is not None:
        center = (float(info["center"][0]), float(info["center"][1]))
    else:
        center = scenario.junction_center_xy
    return DualPathScenario(
        junction_id=scenario.junction_id,
        junction_center_xy=center,
        ego_edge_id=ego,
        ego_lane_num=_default_lane_num(graph, ego),
        dest_edge_id=dest,
        dest_lane_num=_default_lane_num(graph, dest),
        turn_dir=scenario.turn_dir,
        turn_first_exit=t_path[0],
        straight_first_exit=s_path[0],
        turn_path=tuple(t_path),
        straight_path=tuple(s_path),
        turn_length_m=float(turn_dist[dest]),
        straight_length_m=float(straight_dist[dest]),
    )


def pick_best_dual_path_scenario(
    net_path: Path | str,
    **kwargs,
) -> Optional[DualPathScenario]:
    """Return the highest-gain dual-path scenario, or None."""
    found = find_dual_path_scenarios(net_path, max_scenarios=1, **kwargs)
    return found[0] if found else None


def find_ranked_dual_path_picks(
    net_path: Path | str,
    *,
    min_lane_length_m: float = 10.0,
    min_gain_m: float = 20.0,
    max_scenarios: int = 5,
    **kwargs,
) -> List[Tuple[DualPathScenario, JunctionPick]]:
    """Rank dual-path scenarios with a matching ``JunctionPick`` for crop tooling."""
    net_path = Path(net_path)
    scenarios = find_dual_path_scenarios(
        net_path,
        min_lane_length_m=min_lane_length_m,
        min_gain_m=min_gain_m,
        max_scenarios=max_scenarios,
        **kwargs,
    )
    picks = collect_intersection_junction_candidates(
        net_path,
        min_lane_length_m=min_lane_length_m,
        arm_counts=(4,),
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
    """Crop ``source_net`` to the XY bbox of turn+straight paths and write meta."""
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
            f"No geometry for dual-path scenario at junction {scenario.junction_id}"
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
        if rebuilt is not None and rebuilt.gain_m >= max(5.0, scenario.gain_m * 0.25):
            cropped_scenario = rebuilt
            used_margin = attempt_margin
            used_bbox = attempt_bbox
            last_error = None
            break

        still = find_dual_path_scenarios(
            out_net,
            junction_ids=[scenario.junction_id],
            min_gain_m=max(5.0, scenario.gain_m * 0.25),
            min_lane_length_m=5.0,
            max_scenarios=40,
            dests_per_arm=8,
        )
        matching = [
            s
            for s in still
            if s.ego_edge_id == scenario.ego_edge_id
            and s.dest_edge_id == scenario.dest_edge_id
        ]
        if not matching:
            # Keep crop viable even if dest ids shifted slightly: same ego + turn.
            matching = [
                s
                for s in still
                if s.ego_edge_id == scenario.ego_edge_id and s.turn_dir == scenario.turn_dir
            ]
        if matching:
            cropped_scenario = matching[0]
            used_margin = attempt_margin
            used_bbox = attempt_bbox
            last_error = None
            break
        last_error = JunctionLayoutError(
            f"Dual-path lost after crop for junction {scenario.junction_id} "
            f"ego={scenario.ego_edge_id} dest={scenario.dest_edge_id}"
        )
    else:
        if last_error is not None:
            raise last_error
        raise JunctionLayoutError(
            f"Dual-path lost after crop for junction {scenario.junction_id} "
            f"ego={scenario.ego_edge_id} dest={scenario.dest_edge_id}"
        )

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
            "scene_kind": "direction_dual_path",
            "core_scene_name": core_name,
            "net_file": output_net_name,
            "latitude": center_lat,
            "longitude": center_lon,
            "crop_margin_m": margin_m,
            "crop_bbox_xy": [bbox[0], bbox[1], bbox[2], bbox[3]],
            "junction_id": scenario.junction_id,
            "junction_arm_count": 4,
            "junction_center_xy": [
                scenario.junction_center_xy[0],
                scenario.junction_center_xy[1],
            ],
            "sign_spawn_distance": 30.0,
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

    meta_path = output_dir / "meta.json"
    meta_path.write_text(json_dumps(meta) + "\n", encoding="utf-8")
    return scenario
