"""Route helpers for blocked-road bench (PDD 3.2)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from ..sumo.sumo_utils import is_vehicle_drivable_lane


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


def forbidden_edge_geometry_ok(
    net_path: Path | str,
    edge_id: str,
    *,
    sign_distance_from_start: float,
    destination_max_along_m: float,
) -> tuple[bool, str]:
    """Check destination/forbidden edge is long enough for sign + finish mark."""
    if not edge_id:
        return False, "missing forbidden edge_id"
    length = edge_length_m(net_path, edge_id)
    if length is None or length <= 0:
        return False, f"edge {edge_id!r} missing or empty"
    # Need room for the sign and for the capped finish (leave 5 m like MetaDrive).
    needed = max(
        float(sign_distance_from_start) + 1.0,
        float(destination_max_along_m) + 5.0,
    )
    if length <= needed:
        return (
            False,
            f"forbidden edge {edge_id!r} length {length:.2f}m <= "
            f"needed {needed:.2f}m (sign/dest cap)",
        )
    return True, "ok"
