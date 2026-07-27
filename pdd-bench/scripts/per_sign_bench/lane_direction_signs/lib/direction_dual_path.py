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

import shutil
import subprocess
import tempfile

from .direction_sign_spec import DEFAULT_PDD_CODE, get_direction_sign_spec
from .junction_crop import JunctionPick, collect_intersection_junction_candidates
from .junction_priority_layout import INTERSECTION_JUNCTION_TYPES, JunctionLayoutError, _load_net
from .lane_keys import make_lane_key
from .sumo_utils import is_real_sumo_edge_id, is_vehicle_drivable_lane


TurnDir = str


# ---------------------------------------------------------------------------
# Forbidden-connector injection (baseline can physically reach dest)
# ---------------------------------------------------------------------------

def _find_netconvert() -> str:
    for path in (
        shutil.which("netconvert"),
        "/home/jovyan/.local/bin/netconvert",
        "/usr/bin/netconvert",
    ):
        if path and Path(path).exists():
            return path
    raise FileNotFoundError("netconvert not found on PATH")


def _parse_connections_from_edge(net_path: Path, from_edge: str) -> Dict[int, List[Tuple[str, int, str]]]:
    """Parse connections from ``from_edge`` → {fromLane: [(toEdge, toLane, dir), ...]}."""
    root = ET.parse(net_path).getroot()
    result: Dict[int, List[Tuple[str, int, str]]] = defaultdict(list)
    for conn in root.iter("connection"):
        if conn.get("from") != from_edge:
            continue
        try:
            from_lane = int(conn.get("fromLane", -1))
            to_edge = conn.get("to", "")
            to_lane = int(conn.get("toLane", 0))
            direction = conn.get("dir", "s")
            if from_lane >= 0 and to_edge:
                result[from_lane].append((to_edge, to_lane, direction))
        except (TypeError, ValueError):
            continue
    return dict(result)


def _connection_via(
    net_path: Path, from_edge: str, from_lane: int, to_edge: str
) -> Optional[str]:
    """Return the ``via`` internal lane id for a specific connection, if any."""
    root = ET.parse(net_path).getroot()
    for conn in root.iter("connection"):
        if conn.get("from") != from_edge or conn.get("to") != to_edge:
            continue
        try:
            if int(conn.get("fromLane", -1)) != int(from_lane):
                continue
        except (TypeError, ValueError):
            continue
        via = conn.get("via")
        if via:
            return via
    return None


def _lane_shape_points(net_path: Path, lane_id: str) -> List[Tuple[float, float]]:
    root = ET.parse(net_path).getroot()
    for lane in root.iter("lane"):
        if lane.get("id") != lane_id:
            continue
        shape = lane.get("shape") or ""
        pts: List[Tuple[float, float]] = []
        for token in shape.split():
            if "," not in token:
                continue
            xs, ys = token.split(",", 1)
            try:
                pts.append((float(xs), float(ys)))
            except ValueError:
                continue
        return pts
    return []


def _lane_end_point(net_path: Path, edge_id: str, lane_num: int) -> Optional[Tuple[float, float]]:
    pts = _lane_shape_points(net_path, f"{edge_id}_{lane_num}")
    return pts[-1] if pts else None


def _lane_start_point(net_path: Path, edge_id: str, lane_num: int) -> Optional[Tuple[float, float]]:
    pts = _lane_shape_points(net_path, f"{edge_id}_{lane_num}")
    return pts[0] if pts else None


def _polyline_length(pts: Sequence[Tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def _reshape_injected_via(
    net_path: Path,
    *,
    ego_edge: str,
    spawn_lane: int,
    target_lane: int,
    first_exit_edge: str,
    to_lane: int,
    injected_via: str,
    legal_via: Optional[str],
) -> bool:
    """Replace stub injected-via geometry with a smooth offset of the legal turn.

    ``netconvert`` often emits a 1–2 m chord for forbidden fromLane→toEdge links.
    IDM then cuts the corner. We rebuild the via as the legal connector warped
    so its endpoints match the spawn-lane end and exit-lane start (preserving
    relative bend), which IDM can track much more like a normal SUMO turn.
    """
    spawn_pts = _lane_shape_points(net_path, f"{ego_edge}_{spawn_lane}")
    exit_pts = _lane_shape_points(net_path, f"{first_exit_edge}_{to_lane}")
    spawn_end = spawn_pts[-1] if spawn_pts else None
    exit_start = exit_pts[0] if exit_pts else None
    if spawn_end is None or exit_start is None:
        return False

    legal_pts = _lane_shape_points(net_path, legal_via) if legal_via else []

    def _unit(dx: float, dy: float) -> Tuple[float, float]:
        n = math.hypot(dx, dy) or 1.0
        return dx / n, dy / n

    if len(spawn_pts) >= 2:
        shx, shy = _unit(
            spawn_pts[-1][0] - spawn_pts[-2][0],
            spawn_pts[-1][1] - spawn_pts[-2][1],
        )
    else:
        shx, shy = _unit(exit_start[0] - spawn_end[0], exit_start[1] - spawn_end[1])

    if len(exit_pts) >= 2:
        ehx, ehy = _unit(
            exit_pts[1][0] - exit_pts[0][0],
            exit_pts[1][1] - exit_pts[0][1],
        )
    else:
        ehx, ehy = _unit(exit_start[0] - spawn_end[0], exit_start[1] - spawn_end[1])

    if len(legal_pts) >= 3:
        # Warp legal polyline into new endpoints while keeping relative shape.
        old_s = legal_pts[0]
        old_e = legal_pts[-1]
        ox = old_e[0] - old_s[0]
        oy = old_e[1] - old_s[1]
        old_len = math.hypot(ox, oy) or 1.0
        nx = exit_start[0] - spawn_end[0]
        ny = exit_start[1] - spawn_end[1]
        new_len = math.hypot(nx, ny) or 1.0
        # Rotation+scale mapping old chord → new chord.
        # Complex multiply: (p-old_s) * (new_chord / old_chord).
        scale_re = (nx * ox + ny * oy) / (old_len * old_len)
        scale_im = (ny * ox - nx * oy) / (old_len * old_len)
        warped: List[Tuple[float, float]] = []
        for x, y in legal_pts:
            dx, dy = x - old_s[0], y - old_s[1]
            wx = spawn_end[0] + dx * scale_re - dy * scale_im
            wy = spawn_end[1] + dx * scale_im + dy * scale_re
            warped.append((wx, wy))
        warped[0] = spawn_end
        warped[-1] = exit_start
        # Soften ends so the path leaves along spawn heading / arrives along exit.
        extend = max(1.5, 0.25 * new_len)
        if len(warped) >= 4:
            warped[1] = (spawn_end[0] + shx * extend, spawn_end[1] + shy * extend)
            warped[-2] = (exit_start[0] - ehx * extend, exit_start[1] - ehy * extend)
        new_pts = warped
    else:
        extend = 3.0
        p1 = (spawn_end[0] + shx * extend, spawn_end[1] + shy * extend)
        p2 = (exit_start[0] - ehx * extend, exit_start[1] - ehy * extend)
        new_pts = [spawn_end, p1, p2, exit_start]

    if len(new_pts) < 2:
        return False
    shape_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in new_pts)
    length = _polyline_length(new_pts)

    tree = ET.parse(net_path)
    root = tree.getroot()
    touched = False
    for lane in root.iter("lane"):
        if lane.get("id") != injected_via:
            continue
        lane.set("shape", shape_str)
        lane.set("length", f"{length:.2f}")
        touched = True
        break
    if not touched:
        return False
    tree.write(net_path, encoding="utf-8", xml_declaration=True)
    return True


def _extract_original_allowed_exits(
    net_path: Path,
    ego_edge: str,
) -> Dict[int, Dict[str, List[str]]]:
    """Extract original allowed exits per lane: {lane_num: {dir: [to_edge, ...]}}."""
    conns = _parse_connections_from_edge(net_path, ego_edge)
    result: Dict[int, Dict[str, List[str]]] = {}
    for lane_num, targets in conns.items():
        by_dir: Dict[str, List[str]] = defaultdict(list)
        for to_edge, _, direction in targets:
            # Skip internal edges
            if not to_edge.startswith(":"):
                by_dir[direction].append(to_edge)
        result[lane_num] = dict(by_dir)
    return result


def inject_forbidden_connector(
    net_path: Path,
    scenario: "DualPathScenario",
    *,
    output_path: Optional[Path] = None,
) -> Tuple[Path, Dict[int, Dict[str, List[str]]], Dict[str, object]]:
    """Add a physical connector from spawn lane to the correct-path first exit.

    The baseline (sign-blind IDM) needs to physically reach dest. SUMO only
    builds connectors for allowed turns, so spawn_lane has no connector to the
    exclusive first exit. We inject one via netconvert, reshape the stub via so
    IDM can track it, then the sign detects the violation using the *original*
    allowed exits saved in meta.

    Returns ``(modified_net_path, original_allowed_exits_by_lane, inject_meta)``.
    """
    net_path = Path(net_path).resolve()
    output_path = (output_path or net_path).resolve()
    inject_meta: Dict[str, object] = {
        "injected_via_lane_ids": [],
        "legal_via_lane_ids": [],
        "injected_connection": None,
    }

    ego_edge = scenario.ego_edge_id
    spawn_lane = scenario.ego_lane_num
    target_lane = scenario.target_lane_num
    first_exit_edge = scenario.straight_first_exit  # the exclusive exit

    # 1) Extract original allowed exits BEFORE injection (for sign violation check)
    original_exits = _extract_original_allowed_exits(net_path, ego_edge)

    # 2) Check if spawn lane already has a connection to first_exit_edge
    spawn_exits = original_exits.get(spawn_lane, {})
    spawn_exit_edges = {e for edges in spawn_exits.values() for e in edges}
    if first_exit_edge in spawn_exit_edges:
        # Already connected — no injection needed (rare but possible)
        if output_path != net_path:
            shutil.copy2(net_path, output_path)
        return output_path, original_exits, inject_meta

    # 3) Find target lane's connection to first_exit_edge to get toLane + legal via
    target_exits = original_exits.get(target_lane, {})
    to_lane = 0
    legal_via = _connection_via(net_path, ego_edge, target_lane, first_exit_edge)
    for direction, edges in target_exits.items():
        if first_exit_edge in edges:
            conns = _parse_connections_from_edge(net_path, ego_edge)
            for to_edge, tl, _ in conns.get(target_lane, []):
                if to_edge == first_exit_edge:
                    to_lane = tl
                    break
            break
    if legal_via:
        inject_meta["legal_via_lane_ids"] = [legal_via]

    # 4) Create a temporary connection file
    conn_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<connections>
  <connection from="{ego_edge}" to="{first_exit_edge}" fromLane="{spawn_lane}" toLane="{to_lane}"/>
</connections>
'''
    with tempfile.NamedTemporaryFile(mode="w", suffix=".con.xml", delete=False) as f:
        f.write(conn_xml)
        conn_file = Path(f.name)

    # 5) Run netconvert to add the connection (rebuild junction corners)
    try:
        cmd = [
            _find_netconvert(),
            "--sumo-net-file", str(net_path),
            "--connection-files", str(conn_file),
            "--junctions.corner-detail", "5",
            "-o", str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise JunctionLayoutError(
                f"netconvert injection failed: {result.stderr or result.stdout}"
            )
        if not output_path.is_file():
            raise JunctionLayoutError(f"netconvert did not write {output_path}")
    finally:
        conn_file.unlink(missing_ok=True)

    injected_via = _connection_via(output_path, ego_edge, spawn_lane, first_exit_edge)
    # Legal via id may change after netconvert rebuild — re-resolve.
    legal_via = _connection_via(output_path, ego_edge, target_lane, first_exit_edge) or legal_via
    if legal_via:
        inject_meta["legal_via_lane_ids"] = [legal_via]
    if injected_via:
        inject_meta["injected_via_lane_ids"] = [injected_via]
        inject_meta["injected_connection"] = {
            "from": ego_edge,
            "to": first_exit_edge,
            "fromLane": int(spawn_lane),
            "toLane": int(to_lane),
            "via": injected_via,
        }
        reshaped = _reshape_injected_via(
            output_path,
            ego_edge=ego_edge,
            spawn_lane=int(spawn_lane),
            target_lane=int(target_lane),
            first_exit_edge=first_exit_edge,
            to_lane=int(to_lane),
            injected_via=injected_via,
            legal_via=legal_via,
        )
        inject_meta["injected_via_reshaped"] = bool(reshaped)
        if reshaped and injected_via:
            pts = _lane_shape_points(output_path, injected_via)
            inject_meta["injected_via_length_m"] = round(_polyline_length(pts), 2)

    return output_path, original_exits, inject_meta


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
    blocked: Optional[Set[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Shortest-path tree over edges. ``blocked`` edges are never expanded."""
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


def _sumo_edge_opposite(edge_id: str) -> str:
    """SUMO reverse of ``edge_id`` (``A`` ↔ ``-A``)."""
    if edge_id.startswith("-"):
        return edge_id[1:]
    return f"-{edge_id}"


def path_has_immediate_uturn(path: Sequence[str]) -> bool:
    """True if consecutive edges are a SUMO forward/back pair (tight U-turn)."""
    for a, b in zip(path, path[1:]):
        if not a or not b:
            continue
        if _sumo_edge_opposite(a) == b:
            return True
    return False


def path_rejoins_spawn_corridor(
    path: Sequence[str],
    dist_corridor: Dict[str, float],
    dist_via_turn: Dict[str, float],
    *,
    slack_m: float = 5.0,
) -> bool:
    """True if a hop after the exclusive exit is cheaper (or equal) via spawn corridor.

    That means the compliant route left the turn arm and rejoined the road the
    ego could already reach by staying in the spawn lane.
    """
    if len(path) < 2:
        return False
    for edge in path[1:]:
        c_turn = dist_via_turn.get(edge, math.inf)
        c_corr = dist_corridor.get(edge, math.inf)
        if c_corr <= c_turn + float(slack_m):
            return True
    return False


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
    del net_path
    return path_has_immediate_uturn(scenario.straight_path)


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
    min_arms: int = 2,
) -> List[Tuple[str, Tuple[float, float], List[str]]]:
    """Junctions where ≥1 incoming edge has ≥2 lanes.

    ``min_arms`` counts only non-stub incoming edges (length > ``min_arm_lane_m``).
    Short connector stubs must not disqualify an otherwise valid junction.
    """
    out: List[Tuple[str, Tuple[float, float], List[str]]] = []
    for jid, info in graph.junctions.items():
        if info.get("type") not in INTERSECTION_JUNCTION_TYPES:
            continue
        incoming = [
            eid for eid, to_node in graph.edge_to_node.items() if to_node == jid
        ]
        long_incoming = [
            eid
            for eid in incoming
            if graph.edge_length.get(eid, 0.0) > float(min_arm_lane_m)
        ]
        if len(long_incoming) < int(min_arms):
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
    min_lane_length_m: float = 21.0,
    min_arm_lane_m: float = 0.5,
    min_gain_m: float = 0.0,
    min_dest_after_exit_m: float = 30.0,
    max_turn_length_m: float = 350.0,
    max_straight_length_m: float = 700.0,
    max_scenarios: int = 20,
    dests_per_arm: int = 1,
    junction_ids: Optional[Sequence[str]] = None,
    require_uturn_continuation: bool = True,
    min_arms: int = 2,
) -> List[DualPathScenario]:
    """Find spawn-on-wrong-lane / dest-via-peer-lane scenarios.

    ``min_lane_length_m`` filters the *spawn approach* (must fit spawn ≥20 m
    before the junction). ``min_dest_after_exit_m`` skips stub destinations on
    the exclusive first-exit edge so rule-based and baseline share one dest
    marker past the turn.
    """
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
            # Spawn must sit ≥20 m before junction → approach longer than min.
            if graph.edge_length.get(ego, 0.0) < float(min_lane_length_m):
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
                    # Only left/right exclusive exits — dest must be on a turn
                    # arm, not a straight continuation of the approach.
                    turn_exclusive = {
                        e
                        for e in exclusive
                        if target_exit_dir.get(e) in ("l", "r")
                    }
                    if not turn_exclusive:
                        continue

                    # Corridor from spawn exits with the exclusive turn blocked.
                    # A dest is "on the turn" only if reaching it via the turn is
                    # strictly cheaper than via this corridor (no U-turn-back).
                    dist_corridor, _ = _dijkstra_from(
                        graph,
                        [(e, graph.edge_length[e]) for e in spawn_exit_edges],
                        max_cost=max_straight_length_m,
                        blocked=turn_exclusive,
                    )

                    # Destinations via exclusive turn exits only.
                    dist_from_excl, prev_from_excl = _dijkstra_from(
                        graph,
                        [(e, graph.edge_length[e]) for e in turn_exclusive],
                        max_cost=max_straight_length_m,
                    )
                    corridor_slack_m = 5.0
                    dest_candidates = sorted(
                        (
                            (cost, edge)
                            for edge, cost in dist_from_excl.items()
                            if edge != ego
                            and cost >= float(min_dest_after_exit_m)
                            and edge not in turn_exclusive
                            and dist_corridor.get(edge, math.inf) > cost + corridor_slack_m
                        ),
                        key=lambda x: x[0],
                    )
                    kept = 0
                    for dest_cost, dest in dest_candidates:
                        if kept >= max(1, dests_per_arm):
                            break
                        correct_path = _rebuild_path(prev_from_excl, turn_exclusive, dest)
                        if not correct_path:
                            continue
                        if path_revisits_signed_approach(ego, correct_path):
                            continue
                        if path_has_immediate_uturn(correct_path):
                            continue
                        if path_rejoins_spawn_corridor(
                            correct_path, dist_corridor, dist_from_excl, slack_m=corridor_slack_m
                        ):
                            continue
                        compliant_first = correct_path[0]
                        if compliant_first not in turn_exclusive:
                            continue
                        # Dest must be past the exclusive first exit (not the stub itself).
                        if len(correct_path) < 2:
                            continue
                        # Lane-level: spawn has no first-hop onto the compliant exit.
                        if compliant_first in spawn_exit_edges:
                            continue

                        compliant_dir = target_exit_dir.get(compliant_first, "s")
                        if compliant_dir not in ("l", "r"):
                            continue
                        wrong_path, wrong_len, wrong_first = _wrong_spur_path(
                            graph,
                            spawn_exit_edges,
                            avoid={dest, *turn_exclusive},
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

    # Prefer: short correct path, adjacent spawn↔target (LC must be 1 hop for
    # IDM/expert), longer wrong spur, stable ids. Adjacent preference matters:
    # spawn far from target (e.g. L0→L2) often never lane-changes because
    # MetaDrive ref_lanes only expose neighbors.
    scenarios.sort(
        key=lambda s: (
            s.straight_length_m,
            abs(int(s.ego_lane_num) - int(s.target_lane_num)),
            -s.turn_length_m,
            s.ego_edge_id,
            s.ego_lane_num,
        )
    )
    # Dedup by (junction, ego, dest, target): keep the best spawn for each
    # legal turn (adjacent wrong-lane preferred by the sort above).
    uniq: List[DualPathScenario] = []
    seen: Set[Tuple[str, str, str, int]] = set()
    for sc in scenarios:
        key = (sc.junction_id, sc.ego_edge_id, sc.dest_edge_id, int(sc.target_lane_num))
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

    # Lane index 0 is valid — do not use `or` (would fall through to spawn).
    def _lane_num(*candidates, default: int = 0) -> int:
        for c in candidates:
            if c is None:
                continue
            try:
                return int(c)
            except (TypeError, ValueError):
                continue
        return default

    target_lane_num = _lane_num(
        meta.get("target_lane_num"),
        dp.get("target_lane_num"),
        meta.get("spawn_lane_num"),
        default=0,
    )
    return DualPathScenario(
        junction_id=str(meta.get("junction_id") or ""),
        junction_center_xy=center_xy,
        ego_edge_id=str(ego),
        ego_lane_num=_lane_num(
            meta.get("spawn_lane_num"), dp.get("spawn_lane_num"), default=0
        ),
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
    # Dest must sit on a left/right turn arm, not a straight continuation.
    if target_exit_dir.get(next(iter(seed))) not in ("l", "r"):
        # If preferred seed isn't a turn, fall back to turn-only exclusive.
        seed = {e for e in exclusive if target_exit_dir.get(e) in ("l", "r")}
        if not seed:
            return None

    correct_path, correct_len, compliant_first = _can_reach_from_exits(
        graph, seed, dest, max_cost=max_straight_length_m
    )
    if not correct_path or compliant_first is None:
        return None
    if compliant_first in spawn_exit_edges:
        return None
    if target_exit_dir.get(compliant_first) not in ("l", "r"):
        return None
    if path_has_immediate_uturn(correct_path):
        return None

    dist_corridor, _ = _dijkstra_from(
        graph,
        [(e, graph.edge_length[e]) for e in spawn_exit_edges],
        max_cost=max_straight_length_m,
        blocked=set(seed),
    )
    dist_via_turn, _ = _dijkstra_from(
        graph,
        [(e, graph.edge_length[e]) for e in seed],
        max_cost=max_straight_length_m,
    )
    corridor_slack_m = 5.0
    if dist_corridor.get(dest, math.inf) <= correct_len + corridor_slack_m:
        return None
    if path_rejoins_spawn_corridor(
        correct_path, dist_corridor, dist_via_turn, slack_m=corridor_slack_m
    ):
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
    min_lane_length_m: float = 21.0,
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
    # Junction catalog: only spawn approach needs ≥20 m; other arms may be short.
    picks = collect_intersection_junction_candidates(
        net_path,
        min_lane_length_m=0.5,
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

    # Inject forbidden connector so baseline can physically reach dest.
    # This must happen BEFORE we read allowed exits for meta.
    original_allowed_exits: Dict[int, Dict[str, List[str]]] = {}
    inject_meta: Dict[str, object] = {}
    try:
        _, original_allowed_exits, inject_meta = inject_forbidden_connector(
            out_net, scenario, output_path=out_net
        )
    except Exception as exc:
        # Non-fatal: scene still usable, but baseline may not reach dest.
        print(f"  [connector-injection] warning: {exc}")

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
    # Save ORIGINAL allowed exits per lane (before injection) for sign violation check.
    if original_allowed_exits:
        meta["original_allowed_exits_by_lane"] = {
            str(ln): exits for ln, exits in original_allowed_exits.items()
        }
    if inject_meta:
        if inject_meta.get("injected_via_lane_ids"):
            meta["injected_via_lane_ids"] = list(inject_meta["injected_via_lane_ids"])
        if inject_meta.get("legal_via_lane_ids"):
            meta["legal_via_lane_ids"] = list(inject_meta["legal_via_lane_ids"])
        if inject_meta.get("injected_connection") is not None:
            meta["injected_connection"] = inject_meta["injected_connection"]
        if "injected_via_reshaped" in inject_meta:
            meta["injected_via_reshaped"] = inject_meta["injected_via_reshaped"]
        if inject_meta.get("injected_via_length_m") is not None:
            meta["injected_via_length_m"] = inject_meta["injected_via_length_m"]
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
