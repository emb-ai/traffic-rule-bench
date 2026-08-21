"""Straight road segment discovery for speed/pedestrian/detour sign scenes.

A segment is a single edge (incoming to a junction) that meets length and
straightness criteria. Unlike junction scenes, segment scenes contain no
intersection — just a straight road for testing sign compliance.

Straightness thresholds (from physical analysis):
- straight (>= 0.99): for speed signs — curve-aware IDM won't brake
- curved (>= 0.97): for 5.19 pedestrian, 4.2.x detour — slight curves allowed
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Straightness thresholds
STRAIGHT_THRESHOLD = 0.99  # for speed signs (3.24, 3.25)
CURVED_THRESHOLD = 0.97    # for 5.19, 4.2.x (allows slight curves)

# Minimum segment length (meters) — derived from braking physics:
# v0=60 km/h, v_target=20 km/h, decel=3.5 m/s² → d_brake ≈ 47m
# + compliance zone 60m + buffers 15m ≈ 120m minimum, 150m comfortable
MIN_SEGMENT_LENGTH_M = 150.0


@dataclass(frozen=True)
class SegmentCandidate:
    """A road segment candidate for sign placement scenes."""

    edge_id: str
    junction_id: str           # source junction (for provenance)
    osm_way_id: str            # for train/test split (prevents data leakage)
    length_m: float
    straightness: float        # chord/arc ratio, 1.0 = perfectly straight
    lane_count: int
    center_xy: Tuple[float, float]
    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    to_junction_xy: Tuple[float, float]  # junction position (segment ends here)

    @property
    def segment_type(self) -> str:
        """Classify segment by straightness for sign family compatibility."""
        if self.straightness >= STRAIGHT_THRESHOLD:
            return "straight"
        elif self.straightness >= CURVED_THRESHOLD:
            return "curved"
        return "too_curved"

    def scene_id(self) -> str:
        """Unique scene identifier safe for filesystem paths."""
        safe = self.edge_id.replace(":", "_").replace("#", "_").replace("-", "m")
        return f"seg_{safe}"


def osm_way_id_from_edge(edge_id: str) -> str:
    """Extract OSM way ID from a SUMO edge ID.

    SUMO edges: '108888798#2' or '-108888798#0' share base '108888798'.
    The leading '-' indicates reverse direction; '#N' is segment index.
    """
    return edge_id.lstrip("-").split("#")[0]


def calculate_straightness(shape_points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    """Calculate arc length, chord length, and straightness ratio.

    Returns:
        (arc_length_m, chord_length_m, straightness)
        straightness = chord / arc, where 1.0 = perfectly straight
    """
    if len(shape_points) < 2:
        return 0.0, 0.0, 0.0

    arc_length = 0.0
    for i in range(1, len(shape_points)):
        dx = shape_points[i][0] - shape_points[i - 1][0]
        dy = shape_points[i][1] - shape_points[i - 1][1]
        arc_length += math.sqrt(dx * dx + dy * dy)

    chord_length = math.sqrt(
        (shape_points[-1][0] - shape_points[0][0]) ** 2 +
        (shape_points[-1][1] - shape_points[0][1]) ** 2
    )

    straightness = chord_length / arc_length if arc_length > 0 else 0.0
    return arc_length, chord_length, straightness


def parse_shape_string(shape_str: str) -> List[Tuple[float, float]]:
    """Parse SUMO shape string 'x1,y1 x2,y2 ...' into coordinate list."""
    points: List[Tuple[float, float]] = []
    for token in (shape_str or "").split():
        if "," not in token:
            continue
        x_str, y_str = token.split(",", 1)
        try:
            points.append((float(x_str), float(y_str)))
        except ValueError:
            continue
    return points


def get_edge_metrics(
    net_path: Path,
    edge_id: str,
) -> Optional[Dict]:
    """Get length, straightness, lane count, and geometry for an edge."""
    try:
        root = ET.parse(net_path).getroot()
    except (ET.ParseError, OSError):
        return None

    for edge_el in root.findall("edge"):
        if edge_el.get("id") != edge_id:
            continue
        if edge_el.get("function") == "internal":
            continue

        lanes = edge_el.findall("lane")
        vehicle_lanes = [
            lane for lane in lanes
            if not _is_pedestrian_only(lane)
        ]
        if not vehicle_lanes:
            return None

        # Use first vehicle lane for geometry
        lane = vehicle_lanes[0]
        shape_str = lane.get("shape", "")
        points = parse_shape_string(shape_str)
        if len(points) < 2:
            return None

        arc_len, chord_len, straightness = calculate_straightness(points)
        if arc_len < 1.0:
            return None

        # Center and endpoints
        mid_idx = len(points) // 2
        center_xy = points[mid_idx]
        start_xy = points[0]
        end_xy = points[-1]

        return {
            "edge_id": edge_id,
            "length_m": arc_len,
            "straightness": straightness,
            "lane_count": len(vehicle_lanes),
            "center_xy": center_xy,
            "start_xy": start_xy,
            "end_xy": end_xy,
            "shape_points": points,
        }

    return None


def _is_pedestrian_only(lane_el: ET.Element) -> bool:
    """True if lane allows only pedestrians."""
    allow = (lane_el.get("allow") or "").strip()
    if not allow:
        return False
    return all(tok == "pedestrian" for tok in allow.split())


def build_edge_metrics_cache(net_path: Path) -> Dict[str, Dict]:
    """Build a cache of edge metrics for fast lookup."""
    cache: Dict[str, Dict] = {}
    try:
        root = ET.parse(net_path).getroot()
    except (ET.ParseError, OSError):
        return cache

    for edge_el in root.findall("edge"):
        edge_id = edge_el.get("id", "")
        if not edge_id or edge_id.startswith(":"):
            continue
        if edge_el.get("function") == "internal":
            continue

        lanes = edge_el.findall("lane")
        vehicle_lanes = [l for l in lanes if not _is_pedestrian_only(l)]
        if not vehicle_lanes:
            continue

        lane = vehicle_lanes[0]
        shape_str = lane.get("shape", "")
        points = parse_shape_string(shape_str)
        if len(points) < 2:
            continue

        arc_len, chord_len, straightness = calculate_straightness(points)
        if arc_len < 1.0:
            continue

        mid_idx = len(points) // 2
        cache[edge_id] = {
            "edge_id": edge_id,
            "length_m": arc_len,
            "straightness": straightness,
            "lane_count": len(vehicle_lanes),
            "center_xy": points[mid_idx],
            "start_xy": points[0],
            "end_xy": points[-1],
        }

    return cache


def get_junction_position(net_path: Path, junction_id: str) -> Optional[Tuple[float, float]]:
    """Get XY position of a junction."""
    try:
        root = ET.parse(net_path).getroot()
    except (ET.ParseError, OSError):
        return None

    for j in root.findall("junction"):
        if j.get("id") == junction_id:
            try:
                return (float(j.get("x", 0)), float(j.get("y", 0)))
            except ValueError:
                return None
    return None


def build_junction_positions_cache(net_path: Path) -> Dict[str, Tuple[float, float]]:
    """Build cache of junction positions."""
    cache: Dict[str, Tuple[float, float]] = {}
    try:
        root = ET.parse(net_path).getroot()
    except (ET.ParseError, OSError):
        return cache

    for j in root.findall("junction"):
        jid = j.get("id", "")
        if not jid:
            continue
        try:
            cache[jid] = (float(j.get("x", 0)), float(j.get("y", 0)))
        except ValueError:
            continue
    return cache
