"""Center the top-down film on ego so huge dual-path maps stay zoomed in.

MetaDrive clamps ``scaling`` to ``film / map_bbox - 0.1``. A kilometre-scale
net then shows hundreds of metres on screen. For follow-cam GIFs we temporarily
replace the map bbox with a box around the ego so the requested window (80 m)
is honored without a 24k-px film.
"""

from __future__ import annotations

from typing import Optional

_APPLIED = False
_CENTER: Optional[tuple[float, float]] = None
_HALF_M = 200.0


def set_top_down_local_film(
    center_xy: Optional[tuple[float, float]],
    *,
    half_m: float = 200.0,
) -> None:
    global _CENTER, _HALF_M
    _CENTER = None if center_xy is None else (float(center_xy[0]), float(center_xy[1]))
    _HALF_M = float(half_m)


def apply_top_down_local_film_patch() -> None:
    global _APPLIED
    if _APPLIED:
        return

    import metadrive.engine.top_down_renderer as td

    _orig = td.draw_top_down_map_native

    def _patched(map, *args, **kwargs):
        if _CENTER is None or map is None:
            return _orig(map, *args, **kwargs)
        network = getattr(map, "road_network", None)
        if network is None or not hasattr(network, "get_bounding_box"):
            return _orig(map, *args, **kwargs)
        orig_bb = network.get_bounding_box
        cx, cy = _CENTER
        half = max(40.0, _HALF_M)

        def _local_bb(*_a, **_k):
            return (cx - half, cx + half, cy - half, cy + half)

        network.get_bounding_box = _local_bb
        try:
            return _orig(map, *args, **kwargs)
        finally:
            network.get_bounding_box = orig_bb

    td.draw_top_down_map_native = _patched
    _APPLIED = True
