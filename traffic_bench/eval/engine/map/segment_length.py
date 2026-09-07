"""Segment corridor length: harvest window vs SUMO net edge.

Enumerate writes ``length_m`` for a mid-corridor *window* (~250 m). XY crop keeps
whole SUMO edges that intersect the bbox, so ``map.net.xml`` may contain a much
longer ``road_id``. Eval anchors spawn/sign/dest to the harvest window projected
onto that edge (``corridor_s0..corridor_s1``), not to the far junction at the
edge end.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from traffic_bench.eval.signs.blocked.spec import edge_length_m

# Absolute and relative gap between harvest window and net edge length.
SEGMENT_LENGTH_ABS_TOL_M = 20.0
SEGMENT_LENGTH_REL_TOL = 0.15


@dataclass(frozen=True)
class SegmentCorridor:
    """Drivable frame for speed/detour on a (possibly longer) SUMO edge."""

    road_id: str
    net_length_m: float
    corridor_s0: float
    corridor_s1: float
    window_length_m: float
    from_window: bool

    @property
    def usable_length_m(self) -> float:
        return max(0.0, float(self.corridor_s1) - float(self.corridor_s0))

    def to_absolute(self, local_along_m: float) -> float:
        """Map a mark measured inside the harvest window onto edge along-metres."""
        return float(self.corridor_s0) + float(local_along_m)


def window_length_from_meta(meta: Optional[dict[str, Any]]) -> Optional[float]:
    """Harvest corridor window length (enumerate), not necessarily net edge length."""
    meta = meta or {}
    for key in ("window_length_m", "length_m"):
        raw = meta.get(key)
        if raw is None or raw == "":
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val > 0.0:
            return val
    return None


def net_edge_length_m(
    net_path: Path | str,
    road_id: str,
    *,
    meta: Optional[dict[str, Any]] = None,
) -> Optional[float]:
    """Prefer measured SUMO lane length; fall back to meta.net_length_m."""
    road = str(road_id or "").strip()
    if road:
        measured = edge_length_m(net_path, road)
        if measured is not None and measured > 0.0:
            return float(measured)
    meta = meta or {}
    raw = meta.get("net_length_m")
    if raw is None or raw == "":
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if val > 0.0 else None


def resolve_segment_edge_length_m(
    net_path: Path | str,
    meta: Optional[dict[str, Any]],
    *,
    road_id: Optional[str] = None,
) -> Optional[float]:
    """Full SUMO edge length (net), falling back to window meta."""
    meta = meta or {}
    rid = str(road_id or meta.get("road_id") or "").strip()
    net_len = net_edge_length_m(net_path, rid, meta=meta)
    if net_len is not None:
        return net_len
    return window_length_from_meta(meta)


def segment_window_net_mismatch(
    net_path: Path | str,
    meta: Optional[dict[str, Any]],
    *,
    road_id: Optional[str] = None,
    abs_tol_m: float = SEGMENT_LENGTH_ABS_TOL_M,
    rel_tol: float = SEGMENT_LENGTH_REL_TOL,
) -> Optional[Tuple[str, str]]:
    """Return ``(reason, detail)`` when harvest window and net edge disagree.

    Informational for logging; geometry should use :func:`resolve_segment_corridor`
    instead of rejecting the scene.
    """
    meta = meta or {}
    rid = str(road_id or meta.get("road_id") or "").strip()
    window = window_length_from_meta(meta)
    net_len = net_edge_length_m(net_path, rid, meta=meta)
    if window is None or net_len is None or window <= 0.0 or net_len <= 0.0:
        return None
    delta = abs(float(net_len) - float(window))
    if delta <= float(abs_tol_m):
        return None
    if delta / float(window) <= float(rel_tol):
        return None
    return (
        "segment_length_mismatch",
        (
            f"road_id={rid!r} net_length_m={net_len:.1f} "
            f"window_length_m={window:.1f} Δ={delta:.1f}m "
            f"(tol abs={abs_tol_m:.0f}m rel={rel_tol:.0%}); "
            f"XY crop kept a longer edge than the harvest window"
        ),
    )


def _parse_lane_shape(net_path: Path | str, road_id: str) -> List[Tuple[float, float]]:
    try:
        root = ET.parse(str(net_path)).getroot()
    except (ET.ParseError, OSError):
        return []
    for edge in root.findall("edge"):
        if edge.get("id") != road_id:
            continue
        for lane in edge.findall("lane"):
            raw = (lane.get("shape") or "").strip()
            if not raw:
                continue
            pts: List[Tuple[float, float]] = []
            for token in raw.split():
                try:
                    x_s, y_s = token.split(",")
                    pts.append((float(x_s), float(y_s)))
                except (TypeError, ValueError):
                    continue
            if len(pts) >= 2:
                return pts
    return []


def _cum_lengths(pts: Sequence[Tuple[float, float]]) -> List[float]:
    out = [0.0]
    for i in range(1, len(pts)):
        out.append(
            out[-1]
            + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        )
    return out


def _nearest_along(
    pts: Sequence[Tuple[float, float]],
    cum: Sequence[float],
    xy: Sequence[float],
) -> float:
    """Along-metres of the polyline vertex nearest to ``xy``."""
    tx, ty = float(xy[0]), float(xy[1])
    best_i = 0
    best_d = float("inf")
    for i, (x, y) in enumerate(pts):
        d = (x - tx) * (x - tx) + (y - ty) * (y - ty)
        if d < best_d:
            best_d = d
            best_i = i
    return float(cum[best_i])


def _crop_window_xy(
    meta: Optional[dict[str, Any]],
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    meta = meta or {}
    cw = meta.get("crop_window") or {}
    start = cw.get("start_xy") or meta.get("start_xy")
    end = cw.get("end_xy") or meta.get("end_xy")
    try:
        s = (float(start[0]), float(start[1]))
        e = (float(end[0]), float(end[1]))
    except (TypeError, ValueError, IndexError):
        return None
    return s, e


def resolve_segment_corridor(
    net_path: Path | str,
    meta: Optional[dict[str, Any]],
    *,
    road_id: Optional[str] = None,
    abs_tol_m: float = SEGMENT_LENGTH_ABS_TOL_M,
    rel_tol: float = SEGMENT_LENGTH_REL_TOL,
) -> Optional[SegmentCorridor]:
    """Resolve the along-edge interval used for spawn / plate / destination.

    When the cropped edge is only slightly longer than the harvest window, the
    corridor is the whole edge. When XY crop kept a much longer edge, project
    ``crop_window`` onto the lane shape and use that mid-edge span.
    """
    meta = meta or {}
    rid = str(road_id or meta.get("road_id") or "").strip()
    if not rid:
        return None
    net_len = net_edge_length_m(net_path, rid, meta=meta)
    window = window_length_from_meta(meta)
    if net_len is None or net_len <= 0.0:
        if window is None or window <= 0.0:
            return None
        return SegmentCorridor(
            road_id=rid,
            net_length_m=float(window),
            corridor_s0=0.0,
            corridor_s1=float(window),
            window_length_m=float(window),
            from_window=False,
        )

    matched = True
    if window is not None and window > 0.0:
        delta = abs(float(net_len) - float(window))
        matched = delta <= float(abs_tol_m) or delta / float(window) <= float(rel_tol)

    if matched or window is None:
        return SegmentCorridor(
            road_id=rid,
            net_length_m=float(net_len),
            corridor_s0=0.0,
            corridor_s1=float(net_len),
            window_length_m=float(window or net_len),
            from_window=False,
        )

    xy = _crop_window_xy(meta)
    pts = _parse_lane_shape(net_path, rid)
    if xy is None or len(pts) < 2:
        # Cannot project — fall back to whole edge (caller may still reject).
        return SegmentCorridor(
            road_id=rid,
            net_length_m=float(net_len),
            corridor_s0=0.0,
            corridor_s1=float(net_len),
            window_length_m=float(window),
            from_window=False,
        )

    cum = _cum_lengths(pts)
    s0 = _nearest_along(pts, cum, xy[0])
    s1 = _nearest_along(pts, cum, xy[1])
    if s1 < s0:
        s0, s1 = s1, s0
    # Keep a usable span; pad slightly inward from ambiguous endpoints.
    span = s1 - s0
    if span < max(40.0, 0.5 * float(window)):
        # Projection collapsed — centre a window-sized slice on the midpoint.
        mid = 0.5 * (s0 + s1)
        half = 0.5 * float(window)
        s0 = max(0.0, mid - half)
        s1 = min(float(net_len), mid + half)
    s0 = max(0.0, min(s0, float(net_len) - 1.0))
    s1 = max(s0 + 1.0, min(s1, float(net_len)))
    return SegmentCorridor(
        road_id=rid,
        net_length_m=float(net_len),
        corridor_s0=float(s0),
        corridor_s1=float(s1),
        window_length_m=float(window),
        from_window=True,
    )
