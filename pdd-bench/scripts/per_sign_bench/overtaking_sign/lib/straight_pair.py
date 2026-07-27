"""Discover 1+1 opposite-direction straight edges for PDD 3.20 scenes."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def base_way(edge_id: str) -> str:
    """OSM base way id: strip leading '-' and '#<seg>'."""
    return str(edge_id).lstrip("-").split("#")[0]


def is_reverse(a: str, b: str) -> bool:
    """True if ``b`` is the opposite-direction edge of the same OSM way as ``a``."""
    return base_way(a) == base_way(b) and a.startswith("-") != b.startswith("-")


def _parse_shape(shape: str) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for token in (shape or "").split():
        if "," not in token:
            continue
        xs, ys = token.split(",", 1)
        try:
            pts.append((float(xs), float(ys)))
        except ValueError:
            continue
    return pts


def _heading_std_deg(pts: list[tuple[float, float]]) -> float:
    if len(pts) < 2:
        return 0.0
    headings: list[float] = []
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        if abs(dx) + abs(dy) < 1e-6:
            continue
        headings.append(math.atan2(dy, dx))
    if len(headings) < 2:
        return 0.0
    # Circular std via mean unit vector.
    mx = sum(math.cos(h) for h in headings) / len(headings)
    my = sum(math.sin(h) for h in headings) / len(headings)
    r = max(1e-9, math.hypot(mx, my))
    # Approximate angular std (rad) → degrees.
    std_rad = math.sqrt(max(0.0, -2.0 * math.log(r)))
    return math.degrees(std_rad)


@dataclass
class EdgeInfo:
    edge_id: str
    n_lanes: int
    length_m: float
    shape: str
    heading_std_deg: float


@dataclass
class StraightPair:
    ego_edge: str
    opposite_edge: str
    length_m: float
    heading_std_deg: float
    aux_long_m: float
    destination_edge: str


def load_edge_infos(net_path: Path | str) -> dict[str, EdgeInfo]:
    """Non-internal edges from a SUMO .net.xml."""
    tree = ET.parse(str(net_path))
    out: dict[str, EdgeInfo] = {}
    for e in tree.getroot().findall("edge"):
        eid = e.get("id")
        if not eid or eid.startswith(":") or e.get("function") == "internal":
            continue
        lanes = e.findall("lane")
        if not lanes:
            continue
        length = float(lanes[0].get("length") or 0.0)
        shape = lanes[0].get("shape") or ""
        out[eid] = EdgeInfo(
            edge_id=eid,
            n_lanes=len(lanes),
            length_m=length,
            shape=shape,
            heading_std_deg=_heading_std_deg(_parse_shape(shape)),
        )
    return out


def forward_connections(net_path: Path | str) -> dict[str, list[str]]:
    """``{from_edge: [to_edge, ...]}`` for non-internal connections."""
    tree = ET.parse(str(net_path))
    graph: dict[str, list[str]] = {}
    for c in tree.getroot().findall("connection"):
        frm = c.get("from")
        to = c.get("to")
        if not frm or not to or frm.startswith(":") or to.startswith(":"):
            continue
        graph.setdefault(frm, [])
        if to not in graph[frm]:
            graph[frm].append(to)
    return graph


def find_opposite_1lane(
    edges: dict[str, EdgeInfo], ego_edge: str
) -> Optional[str]:
    """Best opposite 1-lane edge for ``ego_edge`` (prefer longest)."""
    cands = [
        e
        for e, info in edges.items()
        if is_reverse(ego_edge, e) and info.n_lanes == 1
    ]
    if not cands:
        return None
    cands.sort(key=lambda e: (-edges[e].length_m, e))
    return cands[0]


def pick_destination_edge(
    net_path: Path | str,
    ego_edge: str,
    *,
    prefer_same_edge: bool = True,
) -> str:
    """Destination for nav: stay on ego edge end, else first forward hop (not reverse)."""
    if prefer_same_edge:
        return ego_edge
    graph = forward_connections(net_path)
    outs = [
        e
        for e in sorted(graph.get(ego_edge) or [])
        if not is_reverse(ego_edge, e)
    ]
    return outs[0] if outs else ego_edge


def analyze_road_pair(
    net_path: Path | str,
    road_id: str,
    *,
    min_length_m: float = 60.0,
    max_heading_std_deg: float = 12.0,
    aux_frac: float = 0.5,
) -> Optional[StraightPair]:
    """Return a StraightPair if ``road_id`` is a viable 1+1 straight approach."""
    edges = load_edge_infos(net_path)
    info = edges.get(str(road_id))
    if info is None:
        return None
    if info.n_lanes != 1:
        return None
    if info.length_m < float(min_length_m):
        return None
    if info.heading_std_deg > float(max_heading_std_deg):
        return None
    opp = find_opposite_1lane(edges, info.edge_id)
    if opp is None:
        return None
    aux_long = max(8.0, min(info.length_m - 8.0, info.length_m * float(aux_frac)))
    dest = pick_destination_edge(net_path, info.edge_id, prefer_same_edge=True)
    return StraightPair(
        ego_edge=info.edge_id,
        opposite_edge=opp,
        length_m=info.length_m,
        heading_std_deg=info.heading_std_deg,
        aux_long_m=aux_long,
        destination_edge=dest,
    )
