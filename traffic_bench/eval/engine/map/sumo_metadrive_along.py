"""Remap SUMO-edge along marks onto MetaDrive lanes.

Manifest geometry for segment families (detour / speed / crosswalk) is authored
against ``meta.length_m`` (the SUMO edge). MetaDrive often stitches upstream
edges into one longer lane that still *ends* at the same place. Spawn placed
via distance-before-end already aligns to that shared end; absolute-from-start
marks (``sign_s``, ``destination_max_along_m``) must shift by the stitched
prefix length, otherwise the ego can spawn past the dest cap and the episode
ends on step 1.
"""

from __future__ import annotations

from typing import Optional


def remap_sumo_along_to_metadrive(
    sumo_along_m: float,
    *,
    sumo_edge_length_m: Optional[float],
    metadrive_lane_length_m: float,
    tol_m: float = 1.0,
) -> float:
    """Map a SUMO-edge along mark onto a (possibly longer) MetaDrive lane.

    Shared-end model: ``md = md_len - sumo_len + sumo_along``.
    When lengths already match, returns the clamped SUMO mark unchanged.
    """
    try:
        md_len = float(metadrive_lane_length_m)
        along = float(sumo_along_m)
    except (TypeError, ValueError):
        return float(sumo_along_m or 0.0)

    def _clamp(x: float) -> float:
        if md_len <= 0.0:
            return max(0.0, x)
        return max(0.0, min(x, max(0.0, md_len - 1e-3)))

    if sumo_edge_length_m is None:
        return _clamp(along)
    try:
        sumo_len = float(sumo_edge_length_m)
    except (TypeError, ValueError):
        return _clamp(along)
    if sumo_len <= 0.0 or md_len <= 0.0:
        return _clamp(along)
    if abs(md_len - sumo_len) <= float(tol_m):
        return _clamp(along)
    return _clamp(md_len - sumo_len + along)


def row_sumo_edge_length_m(row: dict) -> Optional[float]:
    """Best-effort SUMO edge length from a manifest row.

    Prefer explicit edge length. Do **not** treat no-split crosswalk
    ``approach_lane_length_m`` (often equal to ``crosswalk_position_m``, the
    zebra mark) as the edge length — that falsely triggers stitched-prefix
    remapping and collapses spawn/dest onto the MetaDrive lane end.
    """
    for key in ("edge_length_m", "length_m"):
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val > 0.0:
            return val

    approach = row.get("approach_lane_length_m")
    if approach is None or approach == "":
        return None
    try:
        approach_m = float(approach)
    except (TypeError, ValueError):
        return None
    if approach_m <= 0.0:
        return None

    cw = row.get("crosswalk_position_m")
    if cw is not None and cw != "":
        try:
            if abs(approach_m - float(cw)) <= 1.0:
                return None
        except (TypeError, ValueError):
            pass
    return approach_m
