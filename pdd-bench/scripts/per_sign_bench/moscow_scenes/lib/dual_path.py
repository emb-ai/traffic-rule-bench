"""Sign-free dual-path discovery + path-union crop for moscow_scenes.

Atom = one ``(baseline_dir, compliant_dir)`` slot with a shared destination
where the compliant route is at least ``min_gain_m`` longer than the baseline.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_PRIORITY = Path(__file__).resolve().parents[2] / "priority_bench"
if str(_PRIORITY) not in sys.path:
    sys.path.insert(0, str(_PRIORITY))

from core.layout.junction_crop import json_dumps, net_xy_to_latlon, parse_net_location  # noqa: E402
from core.layout.junction_priority_layout import (  # noqa: E402
    INTERSECTION_JUNCTION_TYPES,
    JunctionLayoutError,
    _load_net,
)

from .crop_xy import crop_net_to_xy_boundary
from .roles import SLOTS, parse_slot, slot_name, slots_from_iterable
from .stem import incoming_edge_ids, is_t_stem_approach

_NON_VEHICLE_ALLOW_ONLY = frozenset({"pedestrian"})
TurnDir = str
BBox = Tuple[float, float, float, float]


def _is_real_edge(edge_id: str) -> bool:
    return bool(edge_id) and not str(edge_id).startswith(":")


def _is_vehicle_lane(lane_el: ET.Element) -> bool:
    allow = (lane_el.get("allow") or "").strip()
    if not allow:
        return True
    return any(tok not in _NON_VEHICLE_ALLOW_ONLY for tok in allow.split())


def _osm_base(edge_id: str) -> str:
    e = edge_id[1:] if edge_id.startswith("-") else edge_id
    return e.split("#", 1)[0]


def _carriageway_key(edge_id: str) -> Tuple[str, bool]:
    return (_osm_base(edge_id), edge_id.startswith("-"))


def make_lane_key(edge_id: str, lane_num: int) -> str:
    return f"{edge_id}_{int(lane_num)}"


@dataclass(frozen=True)
class DualPathScenario:
    """One ego approach + shared dest: baseline shorter / compliant longer."""

    junction_id: str
    junction_center_xy: Tuple[float, float]
    ego_edge_id: str
    ego_lane_num: int
    dest_edge_id: str
    dest_lane_num: int
    baseline_dir: TurnDir
    compliant_dir: TurnDir
    baseline_first_exit: str
    compliant_first_exit: str
    baseline_path: Tuple[str, ...]
    compliant_path: Tuple[str, ...]
    baseline_length_m: float
    compliant_length_m: float
    ego_is_t_stem: bool = False
    carriageway_pair: bool = False
    wrong_dir_edges: Tuple[str, ...] = ()

    @property
    def gain_m(self) -> float:
        return float(self.compliant_length_m - self.baseline_length_m)

    @property
    def slot(self) -> str:
        return slot_name(self.baseline_dir, self.compliant_dir)

    @property
    def turn_dir(self) -> str:
        """Legacy alias for baseline_dir."""
        return self.baseline_dir

    @property
    def turn_path(self) -> Tuple[str, ...]:
        return self.baseline_path

    @property
    def straight_path(self) -> Tuple[str, ...]:
        return self.compliant_path

    @property
    def turn_length_m(self) -> float:
        return self.baseline_length_m

    @property
    def straight_length_m(self) -> float:
        return self.compliant_length_m

    @property
    def turn_first_exit(self) -> str:
        return self.baseline_first_exit

    @property
    def straight_first_exit(self) -> str:
        return self.compliant_first_exit

    def scene_id(self, shape: str) -> str:
        dest_hash = hashlib.sha1(
            f"{self.dest_edge_id}|{self.ego_edge_id}|{self.slot}".encode()
        ).hexdigest()[:8]
        jid = self.junction_id.replace(":", "_")
        return f"dual_{shape}_{jid}_{self.slot}_{dest_hash}"

    def to_meta_fields(self) -> dict:
        return {
            "road_id": self.ego_edge_id,
            "spawn_lane_num": self.ego_lane_num,
            "destination_edge_id": self.dest_edge_id,
            "destination_lane_id": make_lane_key(self.dest_edge_id, self.dest_lane_num),
            "baseline_dir": self.baseline_dir,
            "compliant_dir": self.compliant_dir,
            "slot": self.slot,
            "ego_is_t_stem": self.ego_is_t_stem,
            "carriageway_pair": self.carriageway_pair,
            "dual_path": {
                "baseline_dir": self.baseline_dir,
                "compliant_dir": self.compliant_dir,
                "turn_dir": self.baseline_dir,
                "baseline_first_exit": self.baseline_first_exit,
                "compliant_first_exit": self.compliant_first_exit,
                "turn_first_exit": self.baseline_first_exit,
                "straight_first_exit": self.compliant_first_exit,
                "baseline_path": list(self.baseline_path),
                "compliant_path": list(self.compliant_path),
                "turn_path": list(self.baseline_path),
                "straight_path": list(self.compliant_path),
                "baseline_length_m": self.baseline_length_m,
                "compliant_length_m": self.compliant_length_m,
                "turn_length_m": self.baseline_length_m,
                "straight_length_m": self.compliant_length_m,
                "gain_m": self.gain_m,
                "wrong_dir_edges": list(self.wrong_dir_edges),
            },
            "background_excluded_edges": list(self.wrong_dir_edges),
        }


@dataclass
class EdgeGraph:
    edge_length: Dict[str, float]
    edge_to_node: Dict[str, str]
    edge_from_node: Dict[str, str]
    lane_nums: Dict[str, List[int]]
    shapes: Dict[str, List[Tuple[float, float]]]
    adj: Dict[str, List[Tuple[str, float]]]
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


def build_edge_graph(net_path: Path | str) -> EdgeGraph:
    root = ET.parse(net_path).getroot()
    edge_length: Dict[str, float] = {}
    edge_to_node: Dict[str, str] = {}
    edge_from_node: Dict[str, str] = {}
    lane_nums: Dict[str, List[int]] = {}
    shapes: Dict[str, List[Tuple[float, float]]] = {}

    for edge_el in root.findall("edge"):
        eid = edge_el.get("id", "")
        if not _is_real_edge(eid) or edge_el.get("function", "normal") == "internal":
            continue
        drivable = [lane for lane in edge_el.findall("lane") if _is_vehicle_lane(lane)]
        if not drivable:
            continue
        lengths: List[float] = []
        nums: List[int] = []
        best_shape: List[Tuple[float, float]] = []
        for lane_el in drivable:
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
        adj[fr].append((to, edge_length[to]))
        d = (conn.get("dir") or "").strip().lower()
        if d == "R":
            d = "r"
        if d in ("s", "l", "r", "t"):
            first_exits[fr][d].add(to)

    junctions, _, _, _ = _load_net(Path(net_path))
    return EdgeGraph(
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
    graph: EdgeGraph,
    starts: Sequence[Tuple[str, float]],
    *,
    goal: Optional[str] = None,
    max_cost: float = 800.0,
    blocked: Optional[Set[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    blocked = blocked or set()
    dist: Dict[str, float] = {}
    prev: Dict[str, str] = {}
    pq: List[Tuple[float, str]] = []
    for edge_id, cost0 in starts:
        if edge_id not in graph.edge_length or edge_id in blocked:
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
            if v in blocked:
                continue
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
    for _ in range(10_000):
        if cur in start_set:
            path.reverse()
            return path
        parent = prev.get(cur)
        if parent is None:
            return None
        path.append(parent)
        cur = parent
    return None


def _default_lane_num(graph: EdgeGraph, edge_id: str) -> int:
    nums = graph.lane_nums.get(edge_id) or [0]
    return int(nums[0])


def _path_uturn_junctions(graph: EdgeGraph, edge_ids: Sequence[str]) -> Set[str]:
    """Junctions where consecutive edges reverse (from/to swap)."""
    out: Set[str] = set()
    for a, b in zip(edge_ids, edge_ids[1:]):
        if graph.edge_from_node.get(b) == graph.edge_to_node.get(a) and graph.edge_to_node.get(
            b
        ) == graph.edge_from_node.get(a):
            jid = graph.edge_to_node.get(a)
            if jid:
                out.add(jid)
    return out


def _path_revisits_approach(ego: str, path: Sequence[str]) -> bool:
    return bool(ego) and ego in path


def _junction_incident_edges(graph: EdgeGraph) -> Dict[str, Set[str]]:
    inc: Dict[str, Set[str]] = defaultdict(set)
    for eid, to_n in graph.edge_to_node.items():
        inc[to_n].add(eid)
    for eid, fr in graph.edge_from_node.items():
        inc[fr].add(eid)
    return dict(inc)


def _dead_end_uturn(
    graph: EdgeGraph,
    ego: str,
    path: Sequence[str],
    *,
    incident: Dict[str, Set[str]],
) -> bool:
    """True if compliant path U-turns at a junction with no continuation."""
    full = [ego, *path]
    for i in range(len(full) - 1):
        a, b = full[i], full[i + 1]
        if graph.edge_from_node.get(b) != graph.edge_to_node.get(a):
            continue
        if graph.edge_to_node.get(b) != graph.edge_from_node.get(a):
            continue
        jid = graph.edge_to_node.get(a)
        if not jid:
            continue
        # U-turn with only the two edges touching → dead end.
        if len(incident.get(jid, ())) <= 2:
            return True
    return False


def _carriageway_info(
    graph: EdgeGraph,
    baseline_exit: str,
    compliant_exit: str,
) -> Tuple[bool, Tuple[str, ...]]:
    """Detect opposite-carriageway stem entry; return (is_pair, wrong_dir_edges)."""
    if _osm_base(baseline_exit) != _osm_base(compliant_exit):
        return False, ()
    if _carriageway_key(baseline_exit) == _carriageway_key(compliant_exit):
        return False, ()
    forbidden_cw = {_carriageway_key(baseline_exit)}
    wrong = tuple(
        sorted(eid for eid in graph.edge_length if _carriageway_key(eid) in forbidden_cw)
    )
    return True, wrong


def path_union_bbox(
    graph: EdgeGraph,
    scenario: DualPathScenario,
    *,
    margin_m: float,
) -> Optional[BBox]:
    xs: List[float] = []
    ys: List[float] = []
    for eid in (
        scenario.ego_edge_id,
        *scenario.baseline_path,
        *scenario.compliant_path,
    ):
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


def find_dual_paths_for_slot(
    net_path: Path | str,
    *,
    baseline_dir: str,
    compliant_dir: str,
    junction_ids: Optional[Sequence[str]] = None,
    min_lane_length_m: float = 8.0,
    min_gain_m: float = 20.0,
    max_baseline_length_m: float = 350.0,
    max_compliant_length_m: float = 700.0,
    dests_per_arm: int = 1,
    arm_counts: Sequence[int] = (3, 4),
    require_uturn_continuation: bool = True,
    graph: Optional[EdgeGraph] = None,
) -> List[DualPathScenario]:
    """Find dual-path scenarios for one exact ``(baseline_dir, compliant_dir)`` slot."""
    b_dir, c_dir = parse_slot(slot_name(baseline_dir, compliant_dir))
    allowed_arms = {int(n) for n in arm_counts}
    net_path = Path(net_path)
    graph = graph or build_edge_graph(net_path)
    incident = _junction_incident_edges(graph) if require_uturn_continuation else {}

    if junction_ids is None:
        candidates = [
            (jid, (float(info["center"][0]), float(info["center"][1])))
            for jid, info in graph.junctions.items()
            if info.get("type") in INTERSECTION_JUNCTION_TYPES
        ]
    else:
        candidates = []
        for jid in junction_ids:
            info = graph.junctions.get(str(jid))
            if not info:
                continue
            center = info["center"]
            candidates.append((str(jid), (float(center[0]), float(center[1]))))

    scenarios: List[DualPathScenario] = []
    for jid, center in candidates:
        info = graph.junctions.get(jid)
        if info is None or info.get("type") not in INTERSECTION_JUNCTION_TYPES:
            continue
        incoming = incoming_edge_ids(graph, jid)
        if len(incoming) not in allowed_arms:
            continue

        for ego in incoming:
            if graph.edge_length.get(ego, 0.0) <= min_lane_length_m:
                continue
            exits = graph.first_exits.get(ego) or {}
            baseline_exits = set(exits.get(b_dir) or ())
            compliant_exits = set(exits.get(c_dir) or ())
            if not baseline_exits or not compliant_exits:
                continue

            # Prefer natural pair: if any (base,comp) share opposite carriageways,
            # restrict to those for wrong-way blocking; else use all exits.
            paired: List[Tuple[str, str]] = []
            for be in baseline_exits:
                for ce in compliant_exits:
                    ok, _ = _carriageway_info(graph, be, ce)
                    if ok:
                        paired.append((be, ce))

            if paired:
                baseline_use = {be for be, _ in paired}
                compliant_use = {ce for _, ce in paired}
                wrong_way = set()
                for be in baseline_use:
                    forbidden_cw = {_carriageway_key(be)}
                    wrong_way |= {
                        eid
                        for eid in graph.edge_length
                        if _carriageway_key(eid) in forbidden_cw
                    }
                is_cw = True
            else:
                baseline_use = baseline_exits
                compliant_use = compliant_exits
                wrong_way = set()
                is_cw = False

            baseline_starts = [(e, graph.edge_length[e]) for e in baseline_use]
            compliant_starts = [(e, graph.edge_length[e]) for e in compliant_use]
            baseline_dist, baseline_prev = _dijkstra_from(
                graph, baseline_starts, max_cost=max_baseline_length_m
            )
            compliant_dist, compliant_prev = _dijkstra_from(
                graph,
                compliant_starts,
                max_cost=max_compliant_length_m,
                blocked=wrong_way or None,
            )

            shared = set(baseline_dist) & set(compliant_dist)
            shared -= {ego} | baseline_use | compliant_use | wrong_way
            shared = {d for d in shared if graph.edge_to_node.get(d) != jid}

            arm_cands: List[DualPathScenario] = []
            stem = is_t_stem_approach(graph, jid, ego)
            for dest in shared:
                lt = baseline_dist[dest]
                ls = compliant_dist[dest]
                if lt > max_baseline_length_m or ls > max_compliant_length_m:
                    continue
                if ls - lt < min_gain_m:
                    continue
                t_path = _rebuild_path(baseline_prev, baseline_use, dest)
                s_path = _rebuild_path(compliant_prev, compliant_use, dest)
                if not t_path or not s_path:
                    continue
                if _path_uturn_junctions(graph, [ego, *t_path]):
                    continue
                if _path_revisits_approach(ego, t_path):
                    continue
                if require_uturn_continuation and _dead_end_uturn(
                    graph, ego, s_path, incident=incident
                ):
                    continue
                if _path_revisits_approach(ego, s_path):
                    continue

                cw_ok, wrong_tuple = _carriageway_info(graph, t_path[0], s_path[0])
                if is_cw and not cw_ok:
                    continue
                arm_cands.append(
                    DualPathScenario(
                        junction_id=jid,
                        junction_center_xy=center,
                        ego_edge_id=ego,
                        ego_lane_num=_default_lane_num(graph, ego),
                        dest_edge_id=dest,
                        dest_lane_num=_default_lane_num(graph, dest),
                        baseline_dir=b_dir,
                        compliant_dir=c_dir,
                        baseline_first_exit=t_path[0],
                        compliant_first_exit=s_path[0],
                        baseline_path=tuple(t_path),
                        compliant_path=tuple(s_path),
                        baseline_length_m=float(lt),
                        compliant_length_m=float(ls),
                        ego_is_t_stem=stem,
                        carriageway_pair=cw_ok,
                        wrong_dir_edges=wrong_tuple if cw_ok else tuple(sorted(wrong_way)),
                    )
                )
            arm_cands.sort(key=lambda s: (s.baseline_length_m, -s.gain_m))
            scenarios.extend(arm_cands[: max(1, dests_per_arm)])

    scenarios.sort(key=lambda s: (s.baseline_length_m, -s.gain_m, s.junction_id))
    return scenarios


def fill_slots_for_junctions(
    net_path: Path | str,
    *,
    junction_rows: Sequence[dict],
    slots: Optional[Sequence[str]] = None,
    n_per_junction_slot: int = 1,
    max_per_shape_slot: int = 500,
    min_gain_m: float = 20.0,
    min_lane_length_m: float = 8.0,
    seed: int = 42,
    already_filled: Optional[Dict[Tuple[str, str], int]] = None,
) -> List[Tuple[dict, DualPathScenario]]:
    """Fill dual-path atoms with a fixed pool size per ``(shape, slot)``.

    * ``n_per_junction_slot`` — at most this many atoms from one junction for a slot
      (default 1).
    * ``max_per_shape_slot`` — stop once this many atoms are collected for each
      ``(shape, slot)`` bucket (default **500**). This is a shared-pool inventory
      cap (many signs sample the same maps), not ``n_train+n_test`` for one sign.
      Dual_path crops are path-union netconvert-heavy, so we do not keep every
      Moscow hit like junction-only T/X/O; 500 is ~X-scale density per bucket.
    * Junctions are shuffled with ``seed`` so geography is not biased to index order.
    * ``already_filled`` counts existing on-disk scenes toward the cap (skip-existing).

    Returns list of ``(index_row, scenario)`` ready for cropping.
    """
    import random

    net_path = Path(net_path)
    graph = build_edge_graph(net_path)
    slot_list = slots_from_iterable(slots)
    out: List[Tuple[dict, DualPathScenario]] = []

    by_junc: Dict[str, dict] = {}
    for row in junction_rows:
        shape = str(row.get("shape") or "").upper()
        if shape not in ("T", "X"):
            continue
        jid = str(row.get("junction_id") or "")
        if not jid:
            continue
        by_junc[jid] = row

    order = list(by_junc.items())
    rng = random.Random(int(seed))
    rng.shuffle(order)

    counts: Dict[Tuple[str, str], int] = dict(already_filled or {})
    shapes_needed = {
        str(row.get("shape") or "").upper() for _, row in order if row.get("shape")
    }
    targets = {
        (shape, slot): int(max_per_shape_slot)
        for shape in shapes_needed
        for slot in slot_list
        if shape in ("T", "X")
    }

    def _bucket_full(shape: str, slot: str) -> bool:
        return counts.get((shape, slot), 0) >= targets.get((shape, slot), max_per_shape_slot)

    def _all_full() -> bool:
        return all(_bucket_full(shape, slot) for shape, slot in targets)

    for jid, row in order:
        if _all_full():
            break
        shape = str(row.get("shape") or "").upper()
        arm_counts = (3,) if shape == "T" else (4,)
        for slot in slot_list:
            if _bucket_full(shape, slot):
                continue
            b, c = parse_slot(slot)
            found = find_dual_paths_for_slot(
                net_path,
                baseline_dir=b,
                compliant_dir=c,
                junction_ids=[jid],
                min_gain_m=min_gain_m,
                min_lane_length_m=min_lane_length_m,
                dests_per_arm=max(1, n_per_junction_slot),
                arm_counts=arm_counts,
                graph=graph,
            )
            found.sort(
                key=lambda s: (
                    not s.ego_is_t_stem,
                    not s.carriageway_pair,
                    s.baseline_length_m,
                    -s.gain_m,
                )
            )
            kept: List[DualPathScenario] = []
            seen_ego: Set[str] = set()
            room = targets[(shape, slot)] - counts.get((shape, slot), 0)
            for sc in found:
                if len(kept) >= min(n_per_junction_slot, room):
                    break
                if sc.ego_edge_id in seen_ego and n_per_junction_slot <= 1:
                    continue
                kept.append(sc)
                seen_ego.add(sc.ego_edge_id)
            for sc in kept:
                out.append((row, sc))
                counts[(shape, slot)] = counts.get((shape, slot), 0) + 1
    return out


def dual_path_scenario_from_meta(meta: dict) -> Optional[DualPathScenario]:
    dp = meta.get("dual_path")
    ego = meta.get("road_id")
    dest = meta.get("destination_edge_id")
    if not ego or not dest or not isinstance(dp, dict):
        return None
    baseline_path = tuple(dp.get("baseline_path") or dp.get("turn_path") or ())
    compliant_path = tuple(dp.get("compliant_path") or dp.get("straight_path") or ())
    if not baseline_path or not compliant_path:
        return None
    center = meta.get("junction_center_xy") or (0.0, 0.0)
    try:
        center_xy = (float(center[0]), float(center[1]))
    except (TypeError, ValueError, IndexError):
        center_xy = (0.0, 0.0)
    dest_lane_id = meta.get("destination_lane_id")
    try:
        dest_lane_num = int(str(dest_lane_id).rsplit("_", 1)[-1]) if dest_lane_id else 0
    except ValueError:
        dest_lane_num = 0
    baseline_dir = str(dp.get("baseline_dir") or dp.get("turn_dir") or meta.get("baseline_dir") or "l")
    compliant_dir = str(dp.get("compliant_dir") or meta.get("compliant_dir") or "s")
    wrong = tuple(
        str(e)
        for e in (dp.get("wrong_dir_edges") or meta.get("background_excluded_edges") or ())
    )
    return DualPathScenario(
        junction_id=str(meta.get("junction_id") or ""),
        junction_center_xy=center_xy,
        ego_edge_id=str(ego),
        ego_lane_num=int(meta.get("spawn_lane_num") or 0),
        dest_edge_id=str(dest),
        dest_lane_num=dest_lane_num,
        baseline_dir=baseline_dir,
        compliant_dir=compliant_dir,
        baseline_first_exit=str(dp.get("baseline_first_exit") or dp.get("turn_first_exit") or baseline_path[0]),
        compliant_first_exit=str(
            dp.get("compliant_first_exit") or dp.get("straight_first_exit") or compliant_path[0]
        ),
        baseline_path=baseline_path,
        compliant_path=compliant_path,
        baseline_length_m=float(dp.get("baseline_length_m") or dp.get("turn_length_m") or 0.0),
        compliant_length_m=float(dp.get("compliant_length_m") or dp.get("straight_length_m") or 0.0),
        ego_is_t_stem=bool(meta.get("ego_is_t_stem")),
        carriageway_pair=bool(meta.get("carriageway_pair")),
        wrong_dir_edges=wrong,
    )


def rebuild_dual_path_on_net(
    net_path: Path | str,
    scenario: DualPathScenario,
    *,
    max_baseline_length_m: float = 350.0,
    max_compliant_length_m: float = 700.0,
) -> Optional[DualPathScenario]:
    found = find_dual_paths_for_slot(
        net_path,
        baseline_dir=scenario.baseline_dir,
        compliant_dir=scenario.compliant_dir,
        junction_ids=[scenario.junction_id],
        min_gain_m=max(5.0, scenario.gain_m * 0.25),
        min_lane_length_m=5.0,
        max_baseline_length_m=max_baseline_length_m,
        max_compliant_length_m=max_compliant_length_m,
        dests_per_arm=8,
        arm_counts=(3, 4),
    )
    matching = [
        s
        for s in found
        if s.ego_edge_id == scenario.ego_edge_id and s.dest_edge_id == scenario.dest_edge_id
    ]
    if not matching:
        matching = [s for s in found if s.ego_edge_id == scenario.ego_edge_id]
    return matching[0] if matching else None


def straight_path_has_dead_end_uturn(net_path: Path | str, scenario: DualPathScenario) -> bool:
    graph = build_edge_graph(net_path)
    incident = _junction_incident_edges(graph)
    return _dead_end_uturn(
        graph, scenario.ego_edge_id, scenario.compliant_path, incident=incident
    )


def straight_path_reenters_signed_junction(
    net_path: Path | str, scenario: DualPathScenario
) -> bool:
    del net_path
    return _path_revisits_approach(scenario.ego_edge_id, scenario.compliant_path)


def crop_to_dual_path(
    *,
    source_net: Path,
    scenario: DualPathScenario,
    output_dir: Path,
    shape: str,
    base_row: Optional[dict] = None,
    margin_m: float = 40.0,
) -> DualPathScenario:
    """Crop city net to path-union bbox and write scene folder + meta."""
    source_net = source_net.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = str(shape).upper()
    scene_name = scenario.scene_id(shape)
    scene_dir = output_dir / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    graph = build_edge_graph(source_net)
    out_net = scene_dir / "map.net.xml"
    last_error: Optional[Exception] = None
    cropped = scenario
    used_margin = margin_m
    used_bbox: Optional[BBox] = None

    for attempt_margin in (margin_m, margin_m * 1.5, margin_m * 2.5):
        bbox = path_union_bbox(graph, scenario, margin_m=attempt_margin)
        if bbox is None:
            continue
        try:
            crop_net_to_xy_boundary(source_net, bbox, out_net)
        except JunctionLayoutError as exc:
            last_error = exc
            continue
        rebuilt = rebuild_dual_path_on_net(out_net, scenario)
        if rebuilt is None or rebuilt.gain_m < max(5.0, scenario.gain_m * 0.25):
            last_error = JunctionLayoutError(
                f"Dual-path lost after crop for {scenario.junction_id} slot={scenario.slot}"
            )
            continue
        if straight_path_has_dead_end_uturn(out_net, rebuilt):
            last_error = JunctionLayoutError("Crop left dead-end U-turn on compliant path")
            continue
        if straight_path_reenters_signed_junction(out_net, rebuilt):
            last_error = JunctionLayoutError("Compliant path revisits signed approach")
            continue
        cropped = rebuilt
        used_margin = attempt_margin
        used_bbox = bbox
        last_error = None
        break
    else:
        if last_error is not None:
            raise last_error
        raise JunctionLayoutError(f"Failed to crop dual-path for {scenario.junction_id}")

    assert used_bbox is not None
    lat = float((base_row or {}).get("latitude") or 0.0)
    lon = float((base_row or {}).get("longitude") or 0.0)
    if (not lat or not lon) and base_row is None:
        try:
            conv, orig = parse_net_location(source_net)
            lat, lon = net_xy_to_latlon(
                cropped.junction_center_xy[0],
                cropped.junction_center_xy[1],
                conv,
                orig,
            )
        except Exception:
            pass

    meta = {
        "scene_kind": "dual_path",
        "crop_kind": "dual_path",
        "harvest": "sign_free_moscow_osm",
        "source_project": "moscow_scenes",
        "shape": shape,
        "scene_id": scene_name,
        "junction_id": cropped.junction_id,
        "junction_center_xy": list(cropped.junction_center_xy),
        "latitude": lat,
        "longitude": lon,
        "crop_bbox_xy": list(used_bbox),
        "crop_margin_m": used_margin,
        "source_net": source_net.name,
        **cropped.to_meta_fields(),
    }
    if base_row:
        for key in ("scene_id", "arm_count", "junction_type"):
            if key in base_row and key not in ("scene_id",):
                meta.setdefault(f"source_{key}", base_row[key])
        if "arm_count" in base_row:
            meta["arm_count"] = base_row["arm_count"]

    (scene_dir / "meta.json").write_text(json_dumps(meta) + "\n", encoding="utf-8")
    (scene_dir / "center.json").write_text(
        json_dumps({"lat": lat, "lon": lon}) + "\n", encoding="utf-8"
    )
    return cropped
