"""Corridor metrics for segment harvest (speed / detour / crosswalk maps).

A segment is a contiguous window along a vehicle edge (not a junction approach).
Straightness = chord/arc; gates match sign families in signs.yaml.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

STRAIGHT_THRESHOLD = 0.99
CURVED_THRESHOLD = 0.97
MIN_SEGMENT_LENGTH_M = 150.0

# Preferred crop window along a long edge (physics floor is still MIN_SEGMENT_LENGTH_M).
TARGET_WINDOW_M = 250.0

LENGTH_SHORT_LT_M = 250.0
LENGTH_MID_LT_M = 400.0
GEO_CELL_M = 1500.0


@dataclass(frozen=True)
class SegmentCandidate:
    """One corridor candidate (at most one kept per osm_way_id downstream)."""

    edge_id: str
    osm_way_id: str
    length_m: float
    straightness: float
    lane_count: int
    center_xy: Tuple[float, float]
    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    window_shape: Tuple[Tuple[float, float], ...]
    vehicle_lane_indices: Tuple[int, ...] = ()
    pass_right_ok: bool = False
    pass_left_ok: bool = False
    junction_id: str = ""

    @property
    def segment_type(self) -> str:
        if self.straightness >= STRAIGHT_THRESHOLD:
            return "straight"
        if self.straightness >= CURVED_THRESHOLD:
            return "curved"
        return "too_curved"

    @property
    def lane_bucket(self) -> str:
        return lane_bucket(self.lane_count)

    @property
    def length_bucket(self) -> str:
        return length_bucket(self.length_m)

    @property
    def subtype(self) -> str:
        return f"{self.segment_type}|{self.lane_bucket}"

    @property
    def geo_cell(self) -> str:
        return geo_cell_id(self.center_xy)

    def scene_id(self) -> str:
        safe = self.edge_id.replace(":", "_").replace("#", "_").replace("-", "m")
        return f"seg_{safe}"


def lane_bucket(lane_count: int) -> str:
    n = int(lane_count)
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    return "3plus"


def length_bucket(length_m: float) -> str:
    if length_m < LENGTH_SHORT_LT_M:
        return "short"
    if length_m < LENGTH_MID_LT_M:
        return "mid"
    return "long"


def geo_cell_id(xy: Tuple[float, float], cell_m: float = GEO_CELL_M) -> str:
    x, y = float(xy[0]), float(xy[1])
    return f"cell{int(math.floor(x / cell_m))}_{int(math.floor(y / cell_m))}"


def pass_ok_from_indices(indices: Sequence[int]) -> Tuple[bool, bool]:
    """SUMO lane 0 is rightmost; pass_right if a lower-index neighbor exists."""
    lanes = sorted({int(i) for i in indices})
    if len(lanes) < 2:
        return False, False
    pass_right = any(any(j < i for j in lanes) for i in lanes)
    pass_left = any(any(j > i for j in lanes) for i in lanes)
    return pass_right, pass_left


def enrich_lane_fields(meta: dict) -> dict:
    out = dict(meta)
    raw = out.get("vehicle_lane_indices")
    if raw:
        indices = tuple(int(i) for i in raw)
    else:
        n = int(out.get("lane_count") or 0)
        indices = tuple(range(n))
    pass_right, pass_left = pass_ok_from_indices(indices)
    out["vehicle_lane_indices"] = list(indices)
    out["pass_right_ok"] = pass_right
    out["pass_left_ok"] = pass_left
    if "lane_bucket" not in out:
        out["lane_bucket"] = lane_bucket(int(out.get("lane_count") or 0))
    if "length_bucket" not in out and out.get("length_m") is not None:
        out["length_bucket"] = length_bucket(float(out["length_m"]))
    return out


def osm_way_id_from_edge(edge_id: str) -> str:
    return edge_id.lstrip("-").split("#")[0]


def calculate_straightness(
    shape_points: List[Tuple[float, float]],
) -> Tuple[float, float, float]:
    if len(shape_points) < 2:
        return 0.0, 0.0, 0.0

    arc_length = 0.0
    for i in range(1, len(shape_points)):
        dx = shape_points[i][0] - shape_points[i - 1][0]
        dy = shape_points[i][1] - shape_points[i - 1][1]
        arc_length += math.sqrt(dx * dx + dy * dy)

    chord_length = math.sqrt(
        (shape_points[-1][0] - shape_points[0][0]) ** 2
        + (shape_points[-1][1] - shape_points[0][1]) ** 2
    )
    straightness = chord_length / arc_length if arc_length > 0 else 0.0
    return arc_length, chord_length, straightness


def parse_shape_string(shape_str: str) -> List[Tuple[float, float]]:
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


def _cumulative_lengths(points: Sequence[Tuple[float, float]]) -> List[float]:
    cum = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        cum.append(cum[-1] + math.sqrt(dx * dx + dy * dy))
    return cum


def _point_at_s(
    points: Sequence[Tuple[float, float]],
    cum: Sequence[float],
    s: float,
) -> Tuple[float, float]:
    if s <= 0:
        return points[0]
    total = cum[-1]
    if s >= total:
        return points[-1]
    for i in range(1, len(cum)):
        if cum[i] >= s:
            t0, t1 = cum[i - 1], cum[i]
            if t1 <= t0:
                return points[i]
            u = (s - t0) / (t1 - t0)
            x0, y0 = points[i - 1]
            x1, y1 = points[i]
            return (x0 + u * (x1 - x0), y0 + u * (y1 - y0))
    return points[-1]


def extract_window(
    shape_points: List[Tuple[float, float]],
    *,
    min_length_m: float = MIN_SEGMENT_LENGTH_M,
    target_length_m: float = TARGET_WINDOW_M,
) -> Optional[Tuple[List[Tuple[float, float]], float, float, float]]:
    """Take a mid-corridor window; return (points, arc_m, chord_m, straightness)."""
    if len(shape_points) < 2:
        return None
    cum = _cumulative_lengths(shape_points)
    total = cum[-1]
    if total < min_length_m:
        return None

    window_m = min(total, max(min_length_m, target_length_m))
    if total <= window_m + 1e-6:
        start_s, end_s = 0.0, total
        window = list(shape_points)
    else:
        start_s = max(0.0, (total - window_m) * 0.5)
        end_s = start_s + window_m
        window = [_point_at_s(shape_points, cum, start_s)]
        for i, s in enumerate(cum):
            if start_s < s < end_s:
                window.append(shape_points[i])
        window.append(_point_at_s(shape_points, cum, end_s))
        # Drop near-duplicates
        cleaned: List[Tuple[float, float]] = [window[0]]
        for p in window[1:]:
            if math.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) > 0.05:
                cleaned.append(p)
        window = cleaned

    arc, chord, straightness = calculate_straightness(window)
    if arc < min_length_m:
        return None
    return window, arc, chord, straightness


def _is_pedestrian_only(lane_el: ET.Element) -> bool:
    allow = (lane_el.get("allow") or "").strip()
    if not allow:
        return False
    return all(tok == "pedestrian" for tok in allow.split())


def _vehicle_lane_indices(vehicle_lanes: List[ET.Element]) -> Tuple[int, ...]:
    out: List[int] = []
    for i, lane in enumerate(vehicle_lanes):
        raw = lane.get("index")
        try:
            out.append(int(raw) if raw is not None else i)
        except ValueError:
            out.append(i)
    return tuple(out)


def candidate_from_edge_element(
    edge_el: ET.Element,
    *,
    min_length_m: float = MIN_SEGMENT_LENGTH_M,
    min_straightness: float = CURVED_THRESHOLD,
    target_window_m: float = TARGET_WINDOW_M,
) -> Optional[SegmentCandidate]:
    edge_id = edge_el.get("id") or ""
    if not edge_id or edge_id.startswith(":"):
        return None
    if edge_el.get("function") == "internal":
        return None

    lanes = edge_el.findall("lane")
    vehicle_lanes = [lane for lane in lanes if not _is_pedestrian_only(lane)]
    if not vehicle_lanes:
        return None

    indices = _vehicle_lane_indices(vehicle_lanes)
    pass_right, pass_left = pass_ok_from_indices(indices)
    points = parse_shape_string(vehicle_lanes[0].get("shape", ""))
    extracted = extract_window(
        points,
        min_length_m=min_length_m,
        target_length_m=target_window_m,
    )
    if extracted is None:
        return None
    window, arc_len, _chord, straightness = extracted
    if straightness < min_straightness:
        return None

    mid = window[len(window) // 2]
    return SegmentCandidate(
        edge_id=edge_id,
        osm_way_id=osm_way_id_from_edge(edge_id),
        length_m=arc_len,
        straightness=straightness,
        lane_count=len(vehicle_lanes),
        center_xy=(mid[0], mid[1]),
        start_xy=window[0],
        end_xy=window[-1],
        window_shape=tuple(window),
        vehicle_lane_indices=indices,
        pass_right_ok=pass_right,
        pass_left_ok=pass_left,
        junction_id="",
    )


def enumerate_edge_candidates(
    net_path: Path,
    *,
    min_length_m: float = MIN_SEGMENT_LENGTH_M,
    min_straightness: float = CURVED_THRESHOLD,
    target_window_m: float = TARGET_WINDOW_M,
) -> List[SegmentCandidate]:
    """All qualifying edge windows from the city net (may have multiple per way)."""
    try:
        root = ET.parse(net_path).getroot()
    except (ET.ParseError, OSError):
        return []

    out: List[SegmentCandidate] = []
    for edge_el in root.findall("edge"):
        cand = candidate_from_edge_element(
            edge_el,
            min_length_m=min_length_m,
            min_straightness=min_straightness,
            target_window_m=target_window_m,
        )
        if cand is not None:
            out.append(cand)
    return out


def dedupe_by_osm_way(candidates: Sequence[SegmentCandidate]) -> List[SegmentCandidate]:
    """Keep one candidate per osm_way_id: prefer longer, then straighter."""
    best: Dict[str, SegmentCandidate] = {}
    for cand in candidates:
        way = cand.osm_way_id
        prev = best.get(way)
        if prev is None:
            best[way] = cand
            continue
        if cand.length_m > prev.length_m + 1e-6:
            best[way] = cand
        elif abs(cand.length_m - prev.length_m) <= 1e-6 and cand.straightness > prev.straightness:
            best[way] = cand
    return list(best.values())


# Backward-compatible helpers used elsewhere in the repo
def get_edge_metrics(net_path: Path, edge_id: str) -> Optional[Dict]:
    try:
        root = ET.parse(net_path).getroot()
    except (ET.ParseError, OSError):
        return None
    for edge_el in root.findall("edge"):
        if edge_el.get("id") != edge_id:
            continue
        cand = candidate_from_edge_element(edge_el)
        if cand is None:
            return None
        return {
            "edge_id": cand.edge_id,
            "length_m": cand.length_m,
            "straightness": cand.straightness,
            "lane_count": cand.lane_count,
            "vehicle_lane_indices": list(cand.vehicle_lane_indices),
            "pass_right_ok": cand.pass_right_ok,
            "pass_left_ok": cand.pass_left_ok,
            "center_xy": cand.center_xy,
            "start_xy": cand.start_xy,
            "end_xy": cand.end_xy,
            "shape_points": list(cand.window_shape),
        }
    return None


def build_edge_metrics_cache(net_path: Path) -> Dict[str, Dict]:
    cache: Dict[str, Dict] = {}
    for cand in enumerate_edge_candidates(net_path):
        cache[cand.edge_id] = {
            "edge_id": cand.edge_id,
            "length_m": cand.length_m,
            "straightness": cand.straightness,
            "lane_count": cand.lane_count,
            "vehicle_lane_indices": list(cand.vehicle_lane_indices),
            "pass_right_ok": cand.pass_right_ok,
            "pass_left_ok": cand.pass_left_ok,
            "center_xy": cand.center_xy,
            "start_xy": cand.start_xy,
            "end_xy": cand.end_xy,
        }
    return cache


def get_junction_position(net_path: Path, junction_id: str) -> Optional[Tuple[float, float]]:
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
