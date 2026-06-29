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
LayoutMode = Literal["roundabout"]
JunctionShape = Literal["O"]


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


def _load_net(net_path: Path) -> tuple[dict, dict, dict, list]:
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

    return junctions, edges, {}, connections


def _incoming_edges_for_junction(junction_id: str, edges: Dict[str, SumoEdge]) -> List[SumoEdge]:
    incoming = [edge for edge in edges.values() if edge.to_node == junction_id]
    incoming.sort(key=lambda e: e.edge_id)
    return incoming


def build_junction_priority_layout(
    net_path: Path | str,
    mode: LayoutMode = "roundabout",
    ego_edge_id: Optional[str] = None,
    require_ego_secondary: bool = False,
    ring_edge_ids: Optional[Iterable[str]] = None,
    spoke_edge_ids: Optional[Iterable[str]] = None,
    entry_junction_id: Optional[str] = None,
) -> JunctionPriorityLayout:
    """Build roundabout (O) layout: ring edges are main, spokes are secondary."""
    if mode != "roundabout":
        raise JunctionLayoutError(
            f"PDD 4.3 roundabout benchmark only supports traffic-circle (O) layouts; "
            f"got mode={mode!r}"
        )

    net_path = Path(net_path)
    from .roundabout_topology import build_roundabout_layout

    return build_roundabout_layout(
        net_path,
        sign_edge_id=ego_edge_id,
        require_ego_secondary=require_ego_secondary,
        ring_edge_ids=ring_edge_ids,
        spoke_edge_ids=spoke_edge_ids,
        entry_junction_id=entry_junction_id,
    )


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
            from_node=arm.get("from_node", ""),
        )
        for arm in data["arms"]
    ]
    return JunctionPriorityLayout(
        junction_id=data["junction_id"],
        junction_type=data.get("junction_type", "unknown"),
        shape=data["shape"],
        mode=data.get("mode", "roundabout"),
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

    parser = argparse.ArgumentParser(description="Analyze roundabout (O) traffic-circle layout")
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
