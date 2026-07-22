"""Route helpers for no-entry benches (3.1 / 3.2).

Ego spawns on the signed road *before* the catalog sign position and is
routed a few edges past the sign so baselines drive through the forbidden
segment. Compliant experts stop before the sign when no legal detour exists.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from .lane_keys import make_lane_key
from .sumo_utils import is_real_sumo_edge_id, is_vehicle_drivable_lane


# Minimum clearance (m) between ego spawn and the sign line.
DEFAULT_SPAWN_MARGIN_BEFORE_SIGN_M = 15.0
# Minimum catalog distance_from_start to keep a scene.
MIN_SIGN_DISTANCE_FROM_START_M = 8.0
# Minimum remaining length past the sign on the signed edge.
MIN_LENGTH_PAST_SIGN_M = 3.0


def edge_length_m(net_path: Path | str, edge_id: str) -> Optional[float]:
    """Return length of lane 0 on ``edge_id``, or None if missing."""
    try:
        root = ET.parse(str(net_path)).getroot()
    except (ET.ParseError, OSError):
        return None
    for edge in root.findall("edge"):
        if edge.get("id") != edge_id:
            continue
        for lane in edge.findall("lane"):
            if not is_vehicle_drivable_lane(lane):
                continue
            try:
                return float(lane.get("length", 0.0) or 0.0)
            except (TypeError, ValueError):
                return None
    return None


def count_drivable_lanes(net_path: Path | str, edge_id: str) -> int:
    try:
        root = ET.parse(str(net_path)).getroot()
    except (ET.ParseError, OSError):
        return 0
    for edge in root.findall("edge"):
        if edge.get("id") != edge_id:
            continue
        return sum(1 for lane in edge.findall("lane") if is_vehicle_drivable_lane(lane))
    return 0


def _walk_forward_edge(net_path: Path | str, sign_edge: str, hops: int = 2) -> Optional[str]:
    """Walk ``hops`` vehicle-drivable edges forward from ``sign_edge``."""
    try:
        root = ET.parse(str(net_path)).getroot()
    except (ET.ParseError, OSError):
        return None

    edge_fn = {
        edge.get("id", ""): edge.get("function", "normal")
        for edge in root.findall("edge")
        if edge.get("id")
    }
    adj: dict[str, list[str]] = {}
    for conn in root.findall("connection"):
        frm = conn.get("from")
        to = conn.get("to")
        if not frm or not to:
            continue
        if edge_fn.get(to) in ("walkingarea", "crossing"):
            continue
        if not is_real_sumo_edge_id(to):
            continue
        adj.setdefault(frm, [])
        if to not in adj[frm]:
            adj[frm].append(to)

    current = sign_edge
    visited = {sign_edge}
    for _ in range(max(1, hops)):
        cand = [e for e in adj.get(current, []) if e not in visited]
        if not cand:
            break
        current = cand[0]
        visited.add(current)
    return current if current != sign_edge else None


def destination_lane_id(
    net_path: Path | str,
    sign_road_id: str,
    *,
    hops: int = 2,
) -> Optional[str]:
    """MetaDrive destination lane a few edges past the signed road."""
    dest_edge = _walk_forward_edge(net_path, sign_road_id, hops=hops)
    if dest_edge is None:
        return None
    return make_lane_key(dest_edge, 0)


def spawn_longitude_before_sign(
    sign_distance_from_start: float,
    lane_length: float,
    *,
    margin_m: float = DEFAULT_SPAWN_MARGIN_BEFORE_SIGN_M,
) -> float:
    """Longitudinal position (from lane start) for ego spawn before the sign."""
    target = float(sign_distance_from_start) - float(margin_m)
    return max(1.0, min(target, max(lane_length - 1.0, 1.0)))


def scene_geometry_ok(
    net_path: Path | str,
    road_id: str,
    distance_from_start: float,
    *,
    min_sign_dist: float = MIN_SIGN_DISTANCE_FROM_START_M,
    min_past: float = MIN_LENGTH_PAST_SIGN_M,
) -> tuple[bool, str]:
    """Validate that the signed edge can host spawn-before / drive-past."""
    if not road_id:
        return False, "missing road_id"
    length = edge_length_m(net_path, road_id)
    if length is None or length <= 0:
        return False, f"edge {road_id!r} missing or empty"
    dist = float(distance_from_start)
    if dist < min_sign_dist:
        return False, f"distance_from_start={dist:.2f} < {min_sign_dist}"
    if dist > length - min_past:
        return False, f"sign too close to edge end ({dist:.2f}/{length:.2f})"
    if spawn_longitude_before_sign(dist, length) >= dist - 1.0:
        return False, "not enough room to spawn before the sign"
    if destination_lane_id(net_path, road_id) is None:
        return False, "no forward destination past the signed edge"
    return True, "ok"
