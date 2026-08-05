"""Junction main/secondary road layout analysis."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple


INTERSECTION_JUNCTION_TYPES = {
    "priority",
    "right_before_left",
    "allway_stop",
    "traffic_light",
}

RoadClass = Literal["main", "secondary"]
LayoutMode = Literal["main_secondary", "main_main"]
JunctionShape = Literal["2", "T", "X"]

# Priority signs 2.1 / 2.4 need a proper conflict table; 2-arm stubs are rejected.
ALLOWED_PRIORITY_JUNCTION_SHAPES: frozenset[str] = frozenset({"T", "X"})


@dataclass
class SumoLane:
    lane_id: str
    edge_id: str
    lane_num: int
    length: float
    shape: List[Tuple[float, float]]

    @property
    def metadrive_key(self) -> str:
        return f"lane_{self.lane_id}"


@dataclass
class SumoEdge:
    edge_id: str
    from_node: str
    to_node: str
    shape: List[Tuple[float, float]]
    lanes: List[SumoLane] = field(default_factory=list)


@dataclass
class ApproachArm:
    edge_id: str
    lane_keys: List[str]
    entry_point: Tuple[float, float]
    entry_angle: float
    arm_index: int
    road_class: RoadClass = "secondary"
    straight_to: List[str] = field(default_factory=list)
    outgoing_to: List[str] = field(default_factory=list)
    left_to: List[str] = field(default_factory=list)
    right_to: List[str] = field(default_factory=list)
    from_node: str = ""
    min_lane_length: float = 0.0

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "lane_keys": list(self.lane_keys),
            "entry_point": [float(self.entry_point[0]), float(self.entry_point[1])],
            "entry_angle": float(self.entry_angle),
            "arm_index": int(self.arm_index),
            "road_class": self.road_class,
            "straight_to": list(self.straight_to),
            "outgoing_to": list(self.outgoing_to),
            "left_to": list(self.left_to),
            "right_to": list(self.right_to),
            "from_node": self.from_node,
            "min_lane_length": float(self.min_lane_length),
        }


@dataclass
class JunctionPriorityLayout:
    junction_id: str
    junction_type: str
    shape: JunctionShape
    mode: LayoutMode
    center: Tuple[float, float]
    arms: List[ApproachArm]
    main_edge_ids: Set[str] = field(default_factory=set)
    secondary_edge_ids: Set[str] = field(default_factory=set)

    def arm_for_edge(self, edge_id: str) -> Optional[ApproachArm]:
        for arm in self.arms:
            if arm.edge_id == edge_id:
                return arm
        return None

    def lanes_for_class(self, road_class: RoadClass) -> List[str]:
        keys: List[str] = []
        for arm in self.arms:
            if arm.road_class == road_class:
                keys.extend(arm.lane_keys)
        return keys

    def to_dict(self) -> dict:
        return {
            "junction_id": self.junction_id,
            "junction_type": self.junction_type,
            "shape": self.shape,
            "mode": self.mode,
            "center": [float(self.center[0]), float(self.center[1])],
            "main_edge_ids": sorted(self.main_edge_ids),
            "secondary_edge_ids": sorted(self.secondary_edge_ids),
            "arms": [arm.to_dict() for arm in self.arms],
        }


class JunctionLayoutError(ValueError):
    pass


def _parse_shape(shape_str: str) -> List[Tuple[float, float]]:
    if not shape_str:
        return []
    points: List[Tuple[float, float]] = []
    for token in shape_str.strip().split():
        if "," not in token:
            continue
        x_str, y_str = token.split(",", 1)
        points.append((float(x_str), float(y_str)))
    return points


def _polyline_length(points: Iterable[Tuple[float, float]]) -> float:
    pts = list(points)
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        total += math.hypot(dx, dy)
    return total


def _entry_point_for_lane(lane: SumoLane, edge: SumoEdge) -> Tuple[float, float]:
    if lane.shape:
        return lane.shape[-1]
    if edge.shape:
        return edge.shape[-1]
    return (0.0, 0.0)


def _angle_of_point(center: Tuple[float, float], point: Tuple[float, float]) -> float:
    return math.atan2(point[1] - center[1], point[0] - center[0])


def _normalize_dir(raw: str) -> str:
    return str(raw or "").strip().lower()


def _load_net(net_path: Path) -> tuple[dict, dict, Dict[str, str], list]:
    tree = ET.parse(net_path)
    root = tree.getroot()

    junctions: Dict[str, dict] = {}
    for junction in root.findall("junction"):
        jid = junction.get("id")
        if not jid:
            continue
        shape = _parse_shape(junction.get("shape", ""))
        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]
        center = (
            (min(xs) + max(xs)) / 2.0 if xs else float(junction.get("x", 0.0)),
            (min(ys) + max(ys)) / 2.0 if ys else float(junction.get("y", 0.0)),
        )
        if not xs:
            center = (float(junction.get("x", 0.0)), float(junction.get("y", 0.0)))
        junctions[jid] = {
            "id": jid,
            "type": junction.get("type", "unknown"),
            "shape": shape,
            "center": center,
        }

    edges: Dict[str, SumoEdge] = {}
    edge_types: Dict[str, str] = {}
    for edge_el in root.findall("edge"):
        edge_id = edge_el.get("id")
        if not edge_id:
            continue
        func = edge_el.get("function", "normal")
        if func == "internal" or edge_id.startswith(":"):
            continue
        edge = SumoEdge(
            edge_id=edge_id,
            from_node=edge_el.get("from", ""),
            to_node=edge_el.get("to", ""),
            shape=_parse_shape(edge_el.get("shape", "")),
        )
        edge_types[edge_id] = edge_el.get("type", "")
        for lane_el in edge_el.findall("lane"):
            lane_id = lane_el.get("id", "")
            if not lane_id:
                continue
            length = float(lane_el.get("length", 0.0) or 0.0)
            shape = _parse_shape(lane_el.get("shape", ""))
            if length <= 0.0 and len(shape) >= 2:
                length = _polyline_length(shape)
            try:
                lane_num = int(lane_id.rsplit("_", 1)[1])
            except (ValueError, IndexError):
                lane_num = 0
            edge.lanes.append(
                SumoLane(
                    lane_id=lane_id,
                    edge_id=edge_id,
                    lane_num=lane_num,
                    length=length,
                    shape=shape,
                )
            )
        if edge.lanes:
            edges[edge_id] = edge

    connections = []
    for conn in root.findall("connection"):
        connections.append(
            {
                "from": conn.get("from", ""),
                "to": conn.get("to", ""),
                "dir": _normalize_dir(conn.get("dir", "")),
            }
        )

    return junctions, edges, edge_types, connections


def _is_service_road_type(road_type: str) -> bool:
    return road_type == "highway.service" or road_type.endswith(".service")


def _junction_is_mixed_main_service(
    junction_id: str,
    edges: Dict[str, SumoEdge],
    edge_types: Dict[str, str],
) -> bool:
    incoming = _incoming_edges_for_junction(junction_id, edges)
    has_service = any(_is_service_road_type(edge_types.get(edge.edge_id, "")) for edge in incoming)
    has_main = any(
        edge_types.get(edge.edge_id, "") and not _is_service_road_type(edge_types.get(edge.edge_id, ""))
        for edge in incoming
    )
    return has_service and has_main


def _junction_distance_to_point(
    junction_id: str,
    junctions: dict,
    point: Tuple[float, float],
) -> float:
    cx, cy = junctions[junction_id]["center"]
    return math.hypot(point[0] - cx, point[1] - cy)


def _incoming_edges_for_junction(junction_id: str, edges: Dict[str, SumoEdge]) -> List[SumoEdge]:
    incoming = [edge for edge in edges.values() if edge.to_node == junction_id]
    incoming.sort(key=lambda e: e.edge_id)
    return incoming


def _parse_net_location(root: ET.Element) -> Optional[dict]:
    location = root.find("location")
    if location is None:
        return None
    try:
        orig = tuple(float(v) for v in location.get("origBoundary", "").split(","))
        conv = tuple(float(v) for v in location.get("convBoundary", "").split(","))
    except ValueError:
        return None
    if len(orig) != 4 or len(conv) != 4:
        return None
    return {"orig": orig, "conv": conv}


def _latlon_to_net_xy(lat: float, lon: float, location: dict) -> Tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = location["orig"]
    min_x, min_y, max_x, max_y = location["conv"]
    x = min_x + (lon - min_lon) / (max_lon - min_lon) * (max_x - min_x)
    y = min_y + (lat - min_lat) / (max_lat - min_lat) * (max_y - min_y)
    return x, y


def _point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _min_dist_to_arm(
    point: Tuple[float, float],
    arm: ApproachArm,
    edges: Dict[str, SumoEdge],
) -> float:
    px, py = point
    edge = edges.get(arm.edge_id)
    if edge is None:
        return math.hypot(px - arm.entry_point[0], py - arm.entry_point[1])

    best = float("inf")
    for lane in edge.lanes:
        pts = lane.shape or edge.shape
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            best = min(best, _point_segment_distance(px, py, ax, ay, bx, by))
    if best == float("inf"):
        return math.hypot(px - arm.entry_point[0], py - arm.entry_point[1])
    return best


def _discover_primary_junction(
    junctions: dict,
    edges: Dict[str, SumoEdge],
    edge_types: Dict[str, str],
    sign_xy: Optional[Tuple[float, float]] = None,
) -> str:
    candidates: List[tuple[str, int]] = []
    for jid, info in junctions.items():
        if info["type"] not in INTERSECTION_JUNCTION_TYPES:
            continue
        arm_count = len(_incoming_edges_for_junction(jid, edges))
        if arm_count >= 2:
            candidates.append((jid, arm_count))

    if not candidates:
        raise JunctionLayoutError(
            "No intersection junction with at least 2 incoming arms found in net.xml"
        )

    if len(candidates) == 1:
        return candidates[0][0]

    mixed = [
        jid for jid, _ in candidates if _junction_is_mixed_main_service(jid, edges, edge_types)
    ]
    if mixed:
        pool = mixed
    else:
        pool = [jid for jid, _ in candidates]

    if sign_xy is not None:
        return min(pool, key=lambda jid: _junction_distance_to_point(jid, junctions, sign_xy))

    pool.sort(key=lambda jid: (-len(_incoming_edges_for_junction(jid, edges)), jid))
    return pool[0]


def _straight_targets(incoming_edge_id: str, connections: list) -> Set[str]:
    targets: Set[str] = set()
    for conn in connections:
        if conn["from"] != incoming_edge_id:
            continue
        if conn["dir"] != "s":
            continue
        if conn["to"]:
            targets.add(conn["to"])
    return targets


def _outgoing_targets(incoming_edge_id: str, connections: list) -> Set[str]:
    """All outgoing edge ids reachable from an incoming arm (any turn direction)."""
    targets: Set[str] = set()
    for conn in connections:
        if conn["from"] != incoming_edge_id:
            continue
        if conn["to"]:
            targets.add(conn["to"])
    return targets


def _left_targets(incoming_edge_id: str, connections: list) -> Set[str]:
    """Outgoing edges via a left turn (SUMO dir=l) from an incoming arm."""
    targets: Set[str] = set()
    for conn in connections:
        if conn["from"] != incoming_edge_id:
            continue
        if conn.get("dir") == "l" and conn.get("to"):
            targets.add(conn["to"])
    return targets


def _right_targets(incoming_edge_id: str, connections: list) -> Set[str]:
    """Outgoing edges via a right turn (SUMO dir=r) from an incoming arm."""
    targets: Set[str] = set()
    for conn in connections:
        if conn["from"] != incoming_edge_id:
            continue
        if conn.get("dir") == "r" and conn.get("to"):
            targets.add(conn["to"])
    return targets


def _are_through_partners(
    left: SumoEdge,
    right: SumoEdge,
    edges: Dict[str, SumoEdge],
    straight_map: Dict[str, Set[str]],
) -> bool:
    for out_id in straight_map.get(left.edge_id, set()):
        out_edge = edges.get(out_id)
        if out_edge is not None and out_edge.to_node == right.from_node:
            return True
    for out_id in straight_map.get(right.edge_id, set()):
        out_edge = edges.get(out_id)
        if out_edge is not None and out_edge.to_node == left.from_node:
            return True
    return False


def _find_through_pairs(
    incoming: List[SumoEdge],
    edges: Dict[str, SumoEdge],
    straight_map: Dict[str, Set[str]],
) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    for i, ei in enumerate(incoming):
        for ej in incoming[i + 1 :]:
            if _are_through_partners(ei, ej, edges, straight_map):
                pairs.append((ei.edge_id, ej.edge_id))
    return pairs


def _build_arms(
    junction_id: str,
    junctions: dict,
    incoming: List[SumoEdge],
    straight_map: Dict[str, Set[str]],
    outgoing_map: Dict[str, Set[str]],
    left_map: Dict[str, Set[str]],
    right_map: Dict[str, Set[str]],
) -> List[ApproachArm]:
    center = junctions[junction_id]["center"]
    arms: List[ApproachArm] = []
    for edge in incoming:
        lane_keys = [lane.metadrive_key for lane in sorted(edge.lanes, key=lambda l: l.lane_num)]
        entry_lane = max(edge.lanes, key=lambda l: l.length)
        entry_point = _entry_point_for_lane(entry_lane, edge)
        min_lane_length = min((lane.length for lane in edge.lanes), default=0.0)
        arms.append(
            ApproachArm(
                edge_id=edge.edge_id,
                lane_keys=lane_keys,
                entry_point=entry_point,
                entry_angle=_angle_of_point(center, entry_point),
                arm_index=-1,
                straight_to=sorted(straight_map.get(edge.edge_id, set())),
                outgoing_to=sorted(outgoing_map.get(edge.edge_id, set())),
                left_to=sorted(left_map.get(edge.edge_id, set())),
                right_to=sorted(right_map.get(edge.edge_id, set())),
                from_node=edge.from_node,
                min_lane_length=min_lane_length,
            )
        )

    arms.sort(key=lambda arm: arm.entry_angle)
    for idx, arm in enumerate(arms):
        arm.arm_index = idx
    return arms


def _infer_shape(num_arms: int) -> JunctionShape:
    if num_arms == 4:
        return "X"
    if num_arms == 3:
        return "T"
    if num_arms == 2:
        return "2"
    raise JunctionLayoutError(
        f"Unsupported arm count {num_arms}; expected 2, 3 (T), or 4 (X)"
    )


def _assign_main_secondary_x(
    arms: List[ApproachArm],
    incoming: List[SumoEdge],
    edges: Dict[str, SumoEdge],
    straight_map: Dict[str, Set[str]],
) -> tuple[Set[str], Set[str]]:
    through_pairs = _find_through_pairs(incoming, edges, straight_map)
    all_ids = {arm.edge_id for arm in arms}

    if through_pairs:
        main_ids = set(through_pairs[0])
        secondary_ids = all_ids - main_ids
        if len(secondary_ids) != 2:
            raise JunctionLayoutError(
                f"Expected 2 secondary arms for X junction, got {sorted(secondary_ids)}"
            )
        return main_ids, secondary_ids

    if len(arms) != 4:
        raise JunctionLayoutError("X junction expected 4 arms")

    # Circular fallback: opposite slots share class.
    main_ids = {arms[0].edge_id, arms[2].edge_id}
    secondary_ids = {arms[1].edge_id, arms[3].edge_id}
    return main_ids, secondary_ids


def _assign_main_secondary_t(
    arms: List[ApproachArm],
    incoming: List[SumoEdge],
    edges: Dict[str, SumoEdge],
    straight_map: Dict[str, Set[str]],
) -> tuple[Set[str], Set[str]]:
    through_pairs = _find_through_pairs(incoming, edges, straight_map)
    all_ids = {arm.edge_id for arm in arms}

    if through_pairs:
        main_ids = set(through_pairs[0])
        secondary_ids = all_ids - main_ids
        if len(secondary_ids) != 1:
            # Ambiguous T: prefer the arm with no straight exit as the stem.
            no_straight = [arm.edge_id for arm in arms if not arm.straight_to]
            if len(no_straight) == 1:
                secondary_ids = {no_straight[0]}
                main_ids = all_ids - secondary_ids
        return main_ids, secondary_ids

    no_straight = [arm.edge_id for arm in arms if not arm.straight_to]
    if len(no_straight) == 1:
        secondary_ids = {no_straight[0]}
        return all_ids - secondary_ids, secondary_ids

    # Last resort: shortest incoming arm is the stem (secondary).
    shortest = min(
        arms,
        key=lambda arm: min((lane.length for lane in edges[arm.edge_id].lanes), default=0.0),
    )
    secondary_ids = {shortest.edge_id}
    return all_ids - secondary_ids, secondary_ids


def _assign_main_secondary_2(
    arms: List[ApproachArm],
    edges: Dict[str, SumoEdge],
    edge_types: Dict[str, str],
    *,
    sign_xy: Optional[Tuple[float, float]] = None,
    secondary_edge_id: Optional[str] = None,
) -> tuple[Set[str], Set[str]]:
    all_ids = {arm.edge_id for arm in arms}
    if len(arms) != 2:
        raise JunctionLayoutError("2-way junction expected exactly 2 arms")

    if secondary_edge_id is not None:
        if secondary_edge_id not in all_ids:
            raise JunctionLayoutError(
                f"secondary_edge_id {secondary_edge_id!r} is not an incoming arm "
                f"(expected one of {sorted(all_ids)})"
            )
        secondary_ids = {secondary_edge_id}
        return all_ids - secondary_ids, secondary_ids

    service_arms = [
        arm for arm in arms if _is_service_road_type(edge_types.get(arm.edge_id, ""))
    ]
    main_arms = [
        arm for arm in arms if edge_types.get(arm.edge_id, "") and arm not in service_arms
    ]
    if len(service_arms) == 1 and len(main_arms) == 1:
        return {main_arms[0].edge_id}, {service_arms[0].edge_id}

    if sign_xy is not None:
        by_dist = sorted(
            arms,
            key=lambda arm: _min_dist_to_arm(sign_xy, arm, edges),
        )
        secondary_ids = {by_dist[0].edge_id}
        return all_ids - secondary_ids, secondary_ids

    shortest = min(
        arms,
        key=lambda arm: min((lane.length for lane in edges[arm.edge_id].lanes), default=0.0),
    )
    secondary_ids = {shortest.edge_id}
    return all_ids - secondary_ids, secondary_ids


def _apply_classes(arms: List[ApproachArm], main_ids: Set[str], secondary_ids: Set[str]) -> None:
    for arm in arms:
        if arm.edge_id in main_ids:
            arm.road_class = "main"
        elif arm.edge_id in secondary_ids:
            arm.road_class = "secondary"
        else:
            raise JunctionLayoutError(
                f"Arm {arm.edge_id} was not assigned main or secondary"
            )


def build_junction_priority_layout(
    net_path: Path | str,
    mode: LayoutMode = "main_secondary",
    ego_edge_id: Optional[str] = None,
    require_ego_secondary: bool = False,
    sign_lat: Optional[float] = None,
    sign_lon: Optional[float] = None,
    secondary_edge_id: Optional[str] = None,
) -> JunctionPriorityLayout:
    """
    Build main/secondary layout for the single intersection in a SUMO net.

    Args:
        net_path: Path to map.net.xml
        mode: ``main_secondary`` (yield/stop) or ``main_main`` (equal priority / 2.1).
        ego_edge_id: Optional spawn edge; when ``require_ego_secondary`` is set,
            raises if that arm is not secondary.
        require_ego_secondary: Validate ego spawn edge is secondary (stop scenarios).
        sign_lat: Optional WGS84 latitude of the stop sign (used for 2-arm junctions).
        sign_lon: Optional WGS84 longitude of the stop sign (used for 2-arm junctions).
        secondary_edge_id: Optional explicit secondary incoming edge id.

    Returns:
        JunctionPriorityLayout with arms sorted CCW by entry angle.
    """
    if mode not in ("main_secondary", "main_main"):
        raise JunctionLayoutError(f"Unsupported layout mode: {mode}")

    net_path = Path(net_path)
    if not net_path.is_file():
        raise JunctionLayoutError(f"net.xml not found: {net_path}")

    tree = ET.parse(net_path)
    root = tree.getroot()
    location = _parse_net_location(root)
    sign_xy: Optional[Tuple[float, float]] = None
    if sign_lat is not None and sign_lon is not None and location is not None:
        sign_xy = _latlon_to_net_xy(float(sign_lat), float(sign_lon), location)

    junctions, edges, edge_types, connections = _load_net(net_path)
    junction_id = _discover_primary_junction(junctions, edges, edge_types, sign_xy=sign_xy)
    incoming = _incoming_edges_for_junction(junction_id, edges)
    shape = _infer_shape(len(incoming))

    straight_map = {edge.edge_id: _straight_targets(edge.edge_id, connections) for edge in incoming}
    outgoing_map = {edge.edge_id: _outgoing_targets(edge.edge_id, connections) for edge in incoming}
    left_map = {edge.edge_id: _left_targets(edge.edge_id, connections) for edge in incoming}
    right_map = {edge.edge_id: _right_targets(edge.edge_id, connections) for edge in incoming}
    arms = _build_arms(
        junction_id, junctions, incoming, straight_map, outgoing_map, left_map, right_map
    )

    if mode == "main_main":
        all_ids = {arm.edge_id for arm in arms}
        main_ids = all_ids
        secondary_ids: Set[str] = set()
        for arm in arms:
            arm.road_class = "main"
    elif shape == "X":
        main_ids, secondary_ids = _assign_main_secondary_x(
            arms, incoming, edges, straight_map
        )
    elif shape == "T":
        main_ids, secondary_ids = _assign_main_secondary_t(
            arms, incoming, edges, straight_map
        )
    else:
        main_ids, secondary_ids = _assign_main_secondary_2(
            arms,
            edges,
            edge_types,
            sign_xy=sign_xy,
            secondary_edge_id=secondary_edge_id,
        )

    if mode != "main_main":
        _apply_classes(arms, main_ids, secondary_ids)

    if ego_edge_id is not None and require_ego_secondary:
        ego_arm = next((arm for arm in arms if arm.edge_id == ego_edge_id), None)
        if ego_arm is None:
            raise JunctionLayoutError(
                f"ego_edge_id {ego_edge_id!r} is not an incoming arm of junction {junction_id}"
            )
        if ego_arm.road_class != "secondary":
            raise JunctionLayoutError(
                f"ego_edge_id {ego_edge_id!r} is classified as {ego_arm.road_class}, "
                "expected secondary for stop benchmark"
            )

    return JunctionPriorityLayout(
        junction_id=junction_id,
        junction_type=junctions[junction_id]["type"],
        shape=shape,
        mode=mode,
        center=junctions[junction_id]["center"],
        arms=arms,
        main_edge_ids=main_ids,
        secondary_edge_ids=secondary_ids,
    )


def right_arm_edge_id(junction_layout: dict, ego_edge_id: str) -> Optional[str]:
    """Incoming edge on ego's right (right-hand traffic conflict arm)."""
    arms = junction_layout.get("arms", [])
    ego_arm = next((arm for arm in arms if arm.get("edge_id") == ego_edge_id), None)
    if ego_arm is None:
        return None

    ego_angle = float(ego_arm["entry_angle"])
    best_edge: Optional[str] = None
    best_diff = float("inf")
    for arm in arms:
        edge_id = arm.get("edge_id")
        if not edge_id or edge_id == ego_edge_id:
            continue
        diff = (float(arm["entry_angle"]) - ego_angle) % (2.0 * math.pi)
        if 0.3 < diff < math.pi and diff < best_diff:
            best_edge = edge_id
            best_diff = diff
    return best_edge


def left_arm_edge_id(junction_layout: dict, ego_edge_id: str) -> Optional[str]:
    """Incoming edge on ego's left (mirror of ``right_arm_edge_id``)."""
    arms = junction_layout.get("arms", [])
    ego_arm = next((arm for arm in arms if arm.get("edge_id") == ego_edge_id), None)
    if ego_arm is None:
        return None

    ego_angle = float(ego_arm["entry_angle"])
    best_edge: Optional[str] = None
    best_diff = float("inf")
    for arm in arms:
        edge_id = arm.get("edge_id")
        if not edge_id or edge_id == ego_edge_id:
            continue
        diff = (ego_angle - float(arm["entry_angle"])) % (2.0 * math.pi)
        if 0.3 < diff < math.pi and diff < best_diff:
            best_edge = edge_id
            best_diff = diff
    return best_edge


def right_arm_for_layout(
    layout: JunctionPriorityLayout,
    ego_edge_id: str,
) -> Optional[ApproachArm]:
    edge_id = right_arm_edge_id(layout.to_dict(), ego_edge_id)
    if edge_id is None:
        return None
    return layout.arm_for_edge(edge_id)


def left_arm_for_layout(
    layout: JunctionPriorityLayout,
    ego_edge_id: str,
) -> Optional[ApproachArm]:
    edge_id = left_arm_edge_id(layout.to_dict(), ego_edge_id)
    if edge_id is None:
        return None
    return layout.arm_for_edge(edge_id)


def straight_arm_edge_id(junction_layout: dict, ego_edge_id: str) -> Optional[str]:
    """Incoming edge roughly opposite ego (X junctions); None on T/2."""
    arms = junction_layout.get("arms", [])
    ego_arm = next((arm for arm in arms if arm.get("edge_id") == ego_edge_id), None)
    if ego_arm is None:
        return None

    left_id = left_arm_edge_id(junction_layout, ego_edge_id)
    right_id = right_arm_edge_id(junction_layout, ego_edge_id)
    ego_angle = float(ego_arm["entry_angle"])
    best_edge: Optional[str] = None
    best_opp = float("inf")
    for arm in arms:
        edge_id = arm.get("edge_id")
        if not edge_id or edge_id == ego_edge_id:
            continue
        if edge_id in {left_id, right_id}:
            continue
        delta = (float(arm["entry_angle"]) - ego_angle) % (2.0 * math.pi)
        opp = abs(delta - math.pi)
        # Within ~50° of opposite.
        if opp < best_opp and opp < 0.9:
            best_edge = edge_id
            best_opp = opp
    return best_edge


def straight_arm_for_layout(
    layout: JunctionPriorityLayout,
    ego_edge_id: str,
) -> Optional[ApproachArm]:
    edge_id = straight_arm_edge_id(layout.to_dict(), ego_edge_id)
    if edge_id is None:
        return None
    return layout.arm_for_edge(edge_id)


def load_junction_priority_layout(path: Path | str) -> JunctionPriorityLayout:
    """Load a serialized layout JSON written via ``layout.to_dict()``."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        import json

        data = json.load(f)

    arms = [
        ApproachArm(
            edge_id=arm["edge_id"],
            lane_keys=list(arm["lane_keys"]),
            entry_point=(float(arm["entry_point"][0]), float(arm["entry_point"][1])),
            entry_angle=float(arm["entry_angle"]),
            arm_index=int(arm["arm_index"]),
            road_class=arm["road_class"],
            straight_to=list(arm.get("straight_to", [])),
            outgoing_to=list(arm.get("outgoing_to", arm.get("straight_to", []))),
            left_to=list(arm.get("left_to", [])),
            right_to=list(arm.get("right_to", [])),
            from_node=arm.get("from_node", ""),
            min_lane_length=float(arm.get("min_lane_length", 0.0) or 0.0),
        )
        for arm in data["arms"]
    ]
    return JunctionPriorityLayout(
        junction_id=data["junction_id"],
        junction_type=data.get("junction_type", "unknown"),
        shape=data["shape"],
        mode=data.get("mode", "main_secondary"),
        center=(float(data["center"][0]), float(data["center"][1])),
        arms=arms,
        main_edge_ids=set(data.get("main_edge_ids", [])),
        secondary_edge_ids=set(data.get("secondary_edge_ids", [])),
    )


def _format_layout(layout: JunctionPriorityLayout) -> str:
    lines = [
        f"Junction: {layout.junction_id} ({layout.junction_type}, shape={layout.shape}, mode={layout.mode})",
        f"Center: ({layout.center[0]:.1f}, {layout.center[1]:.1f})",
        "",
    ]
    for arm in layout.arms:
        lines.append(
            f"  [{arm.arm_index}] {arm.edge_id:<20} class={arm.road_class:<9} "
            f"angle={math.degrees(arm.entry_angle):6.1f}°  lanes={len(arm.lane_keys)}  "
            f"straight_to={arm.straight_to}"
        )
    lines.append("")
    lines.append(f"Main edges: {sorted(layout.main_edge_ids)}")
    lines.append(f"Secondary edges: {sorted(layout.secondary_edge_ids)}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Analyze junction main/secondary layout")
    parser.add_argument("net_path", type=Path, help="Path to map.net.xml")
    parser.add_argument("--ego-edge-id", type=str, default=None, help="Ego spawn edge (info only)")
    parser.add_argument(
        "--require-ego-secondary",
        action="store_true",
        help="Fail if --ego-edge-id is not classified as secondary",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write layout JSON to path")
    args = parser.parse_args(argv)

    layout = build_junction_priority_layout(
        args.net_path,
        ego_edge_id=args.ego_edge_id,
        require_ego_secondary=args.require_ego_secondary,
    )
    print(_format_layout(layout))
    if args.ego_edge_id:
        ego_arm = layout.arm_for_edge(args.ego_edge_id)
        if ego_arm is not None:
            print(f"Ego edge {args.ego_edge_id!r} -> {ego_arm.road_class}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(layout.to_dict(), f, indent=2)
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
