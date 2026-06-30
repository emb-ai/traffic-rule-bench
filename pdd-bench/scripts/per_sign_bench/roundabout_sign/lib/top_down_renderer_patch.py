"""Roundabout benchmark top-down render tweaks without patching the metadrive package."""

from __future__ import annotations

from typing import Any, Optional


def road_key_from_lane_index(lane_idx) -> Optional[object]:
    """Collapse parallel lanes and SUMO split-edge fragments to one road key."""
    if lane_idx is None:
        return None
    if isinstance(lane_idx, str):
        raw = lane_idx[5:] if lane_idx.startswith("lane_") else lane_idx
        edge_id = raw.rsplit("_", 1)[0]
        return edge_id.split("#", 1)[0]
    if isinstance(lane_idx, (tuple, list)) and len(lane_idx) >= 2:
        return tuple(lane_idx[:2])
    return lane_idx


def dedupe_roundabout_sign_mgr(sign_mgr) -> int:
    """Keep one visible RoundaboutSign per physical approach road."""
    if sign_mgr is None:
        return 0
    seen_roads: set[object] = set()
    kept: list[Any] = []
    removed = 0
    for sign in sign_mgr.signs:
        if sign is None:
            continue
        if type(sign).__name__ != "RoundaboutSign":
            kept.append(sign)
            continue
        road_key = road_key_from_lane_index(getattr(getattr(sign, "lane", None), "index", None))
        if road_key is not None and road_key in seen_roads:
            removed += 1
            continue
        if road_key is not None:
            seen_roads.add(road_key)
        kept.append(sign)
    sign_mgr.signs = kept
    return removed


def apply_roundabout_top_down_renderer_patch() -> None:
    """Disable lane-level priority diamonds on roundabout benchmark GIFs."""
    from metadrive.engine.top_down_renderer import TopDownRenderer

    if getattr(TopDownRenderer, "_roundabout_bench_patch_applied", False):
        return

    _orig_init = TopDownRenderer.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._draw_priority_signs = False

    TopDownRenderer.__init__ = _patched_init
    TopDownRenderer._roundabout_bench_patch_applied = True
