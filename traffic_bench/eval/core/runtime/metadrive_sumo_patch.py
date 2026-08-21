"""Runtime patch for MetaDrive SUMO via-chain wiring."""

from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger(__name__)
_APPLIED = False


def _is_internal_lane_node(lane: Any) -> bool:
    name = getattr(lane, "name", None) or ""
    if str(name).startswith(":"):
        return True
    fn = getattr(lane, "function", None)
    return fn == "internal"


def _is_internal_road_node(road: Any) -> bool:
    name = getattr(road, "name", None) or ""
    if str(name).startswith(":"):
        return True
    edge = getattr(road, "sumolib_obj", None)
    try:
        return bool(edge is not None and edge.getFunction() == "internal")
    except Exception:
        return str(name).startswith(":")


def strip_split_via_shortcuts(graph: Any) -> int:
    """Remove via→dest shortcuts when a via→via continuation exists. Returns count."""
    removed = 0
    lanes = getattr(graph, "lanes", None) or {}
    for lane in list(lanes.values()):
        if not _is_internal_lane_node(lane):
            continue
        outgoing = list(getattr(lane, "outgoing", []) or [])
        internal_outs = [o for o in outgoing if _is_internal_lane_node(o)]
        external_outs = [o for o in outgoing if not _is_internal_lane_node(o)]
        if not internal_outs or not external_outs:
            continue
        for dest in external_outs:
            try:
                lane.outgoing.remove(dest)
            except ValueError:
                continue
            incoming = getattr(dest, "incoming", None)
            if incoming is not None and lane in incoming:
                try:
                    incoming.remove(lane)
                except ValueError:
                    pass
            removed += 1

    roads = getattr(graph, "roads", None) or {}
    for road in list(roads.values()):
        if not _is_internal_road_node(road):
            continue
        outgoing = list(getattr(road, "outgoing", []) or [])
        internal_outs = [o for o in outgoing if _is_internal_road_node(o)]
        external_outs = [o for o in outgoing if not _is_internal_road_node(o)]
        if not internal_outs or not external_outs:
            continue
        for dest in external_outs:
            try:
                road.outgoing.remove(dest)
            except ValueError:
                continue
            incoming = getattr(dest, "incoming", None)
            if incoming is not None and road in incoming:
                try:
                    incoming.remove(road)
                except ValueError:
                    pass
            removed += 1

    return removed


def apply_metadrive_sumo_via_patch() -> bool:
    """Monkeypatch ``RoadLaneJunctionGraph.__init__`` once. Safe to call repeatedly."""
    global _APPLIED
    if _APPLIED:
        return True
    try:
        from metadrive.utils.sumo import map_utils as sumo_map_utils
    except ImportError as exc:
        _logger.warning("MetaDrive SUMO via patch skipped (import failed): %s", exc)
        return False

    cls = sumo_map_utils.RoadLaneJunctionGraph
    if getattr(cls, "_priority_bench_via_patch", False):
        _APPLIED = True
        return True

    original_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        n = strip_split_via_shortcuts(self)
        if n:
            _logger.debug(
                "Stripped %d split-via shortcut link(s) from RoadLaneJunctionGraph",
                n,
            )

    cls.__init__ = patched_init  # type: ignore[method-assign]
    cls._priority_bench_via_patch = True
    _APPLIED = True
    _logger.info(
        "Applied priority_bench MetaDrive SUMO via-chain patch "
        "(split-turn shortcuts stripped after graph build)"
    )
    return True
