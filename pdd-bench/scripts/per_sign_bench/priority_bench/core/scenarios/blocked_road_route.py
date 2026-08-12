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
    destination_past_sign_m: float,
) -> tuple[bool, str]:
    """Check destination/forbidden edge is long enough for sign + short route end."""
    if not edge_id:
        return False, "missing forbidden edge_id"
    length = edge_length_m(net_path, edge_id)
    if length is None or length <= 0:
        return False, f"edge {edge_id!r} missing or empty"
    needed = float(sign_distance_from_start) + float(destination_past_sign_m)
    if length <= needed:
        return (
            False,
            f"forbidden edge {edge_id!r} length {length:.2f}m <= "
            f"sign_from_start+past {needed:.2f}m",
        )
    return True, "ok"
