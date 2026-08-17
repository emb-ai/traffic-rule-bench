"""Draw ego/aux route polylines + path-conflict points on top-down GIFs.

Monkey-patches ``TopDownRenderer._draw`` (does not edit the metadrive package).

Overlays are painted on ``_frame_canvas`` in *map* pixel space (via ``pos2pix``),
the same place MetaDrive draws green navigation routes — *before* the heading-up
crop/rotate. Drawing on ``_screen_canvas`` after rotate misaligned paths off-road.

Injection trick: ``pygame.Surface.fill``/``blit`` are read-only, so we temporarily
wrap ``_screen_canvas`` and draw when ``get_size()`` is called at the start of the
camera section in ``TopDownRenderer._draw``.

Also patches the red destination marker so roundabout ``destination_max_along_m``
caps land the mark at the truncated exit longitude (MetaDrive always used
``final_lane.length``).
"""

from __future__ import annotations

_APPLIED = False
_ENABLED = False
_DEST_CAP_APPLIED = False

# pygame RGB
EGO_PATH_COLOR = (0, 220, 255)       # cyan
FOE_BLOCKING_COLOR = (255, 64, 220)  # magenta
FOE_OTHER_COLOR = (160, 160, 160)    # gray
CONFLICT_COLOR = (255, 220, 0)       # yellow
ZONE_COLOR = (255, 180, 40)          # amber


def _xy_list(points) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in points or []:
        try:
            out.append((float(p[0]), float(p[1])))
        except Exception:
            continue
    return out


def _draw_polyline_frame(canvas, points, color, width: int) -> None:
    import pygame

    pts = _xy_list(points)
    if len(pts) < 2:
        if len(pts) == 1:
            pix = canvas.pos2pix(pts[0][0], pts[0][1])
            pygame.draw.circle(canvas, color, pix, max(2, width))
        return
    pix_pts = [canvas.pos2pix(x, y) for x, y in pts]
    for i in range(len(pix_pts) - 1):
        pygame.draw.line(canvas, color, pix_pts[i], pix_pts[i + 1], width)


def _draw_conflict_marker_frame(canvas, point, color) -> None:
    import pygame

    try:
        cx, cy = canvas.pos2pix(float(point[0]), float(point[1]))
    except Exception:
        return
    r = 10
    pygame.draw.circle(canvas, color, (cx, cy), r, 0)
    pygame.draw.circle(canvas, (20, 20, 20), (cx, cy), r, 2)
    pygame.draw.line(canvas, (20, 20, 20), (cx - r - 3, cy), (cx + r + 3, cy), 2)
    pygame.draw.line(canvas, (20, 20, 20), (cx, cy - r - 3), (cx, cy + r + 3), 2)


def _zone_polyline(zone) -> list[tuple[float, float]]:
    lane = zone.get("lane")
    if lane is None:
        return []
    try:
        long_start = float(zone.get("long_start", 0.0))
        long_end = float(zone.get("long_end", getattr(lane, "length", 0.0)))
        length = float(getattr(lane, "length", 0.0))
    except Exception:
        return []
    if length <= 1e-3:
        return []
    s0 = max(0.0, min(long_start, length - 1e-3))
    s1 = max(s0 + 1e-3, min(long_end, length))
    pts: list[tuple[float, float]] = []
    step = 2.0
    s = s0
    while s <= s1 + 1e-6:
        try:
            p = lane.position(min(s, length - 1e-3), 0.0)
            pts.append((float(p[0]), float(p[1])))
        except Exception:
            break
        s += step
    return pts


def _draw_path_conflict_overlays_on_frame(renderer) -> None:
    """Draw overlays onto ``_frame_canvas`` (pre-camera map pixels)."""
    engine = getattr(renderer, "engine", None)
    if engine is None or not hasattr(engine, "traffic_sign_manager"):
        return
    sign_mgr = engine.traffic_sign_manager
    if sign_mgr is None:
        return
    ego = getattr(renderer, "current_track_agent", None)
    canvas = getattr(renderer, "_frame_canvas", None)
    if ego is None or canvas is None:
        return

    from traffic_signs.priority_signs import YieldSign

    for sign in getattr(sign_mgr, "signs", []) or []:
        if not isinstance(sign, YieldSign):
            continue
        try:
            overlay = sign.get_top_down_path_conflict_overlay(ego)
        except Exception:
            continue
        if not overlay:
            continue

        for zone in overlay.get("zones") or []:
            _draw_polyline_frame(canvas, _zone_polyline(zone), ZONE_COLOR, 5)

        _draw_polyline_frame(canvas, overlay.get("ego_path"), EGO_PATH_COLOR, 5)

        for foe in overlay.get("foes") or []:
            # Always magenta for aux routes so short ring-circulate / non-blocking
            # paths stay visible on GIFs (gray was easy to miss).
            color = FOE_BLOCKING_COLOR
            width = 5 if foe.get("blocking") else 3
            _draw_polyline_frame(canvas, foe.get("path"), color, width)
            pt = foe.get("conflict_point")
            if pt is not None:
                _draw_conflict_marker_frame(canvas, pt, CONFLICT_COLOR)


def set_path_conflict_overlay_enabled(enabled: bool) -> None:
    """Enable/disable path-conflict GIF overlays at runtime."""
    global _ENABLED
    _ENABLED = bool(enabled)


def is_path_conflict_overlay_enabled() -> bool:
    return bool(_ENABLED)


class _ScreenCanvasHook:
    """Proxy that draws path overlays on first ``get_size()`` (camera section)."""

    __slots__ = ("_real", "_renderer", "_drawn")

    def __init__(self, real, renderer):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_renderer", renderer)
        object.__setattr__(self, "_drawn", False)

    def get_size(self):
        if not object.__getattribute__(self, "_drawn"):
            object.__setattr__(self, "_drawn", True)
            try:
                _draw_path_conflict_overlays_on_frame(
                    object.__getattribute__(self, "_renderer")
                )
            except Exception:
                pass
        return object.__getattribute__(self, "_real").get_size()

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)


def _dest_along_for_agent(agent):
    """Return capped destination longitude when priority_bench set one."""
    if agent is None:
        return None
    for obj in (agent, getattr(agent, "navigation", None)):
        if obj is None:
            continue
        raw = getattr(obj, "_priority_bench_dest_along_m", None)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val > 0.0:
            return val
    return None


def apply_top_down_destination_cap_patch() -> None:
    """Make the red top-down dest mark honor ``_priority_bench_dest_along_m``."""
    global _DEST_CAP_APPLIED
    if _DEST_CAP_APPLIED:
        return

    from metadrive.engine.top_down_renderer import TopDownRenderer

    _orig_draw = TopDownRenderer._draw

    def _patched_draw(self, *args, **kwargs):
        agent = getattr(self, "current_track_agent", None)
        nav = getattr(agent, "navigation", None) if agent is not None else None
        final = getattr(nav, "final_lane", None) if nav is not None else None
        along = _dest_along_for_agent(agent)
        if final is None or along is None:
            return _orig_draw(self, *args, **kwargs)

        orig_position = final.position

        def _capped_position(longitudinal, lateral=0.0, *rest, **kw):
            try:
                length = float(getattr(final, "length", 0.0) or 0.0)
            except Exception:
                length = 0.0
            try:
                req = float(longitudinal)
            except (TypeError, ValueError):
                return orig_position(longitudinal, lateral, *rest, **kw)
            # Prefer the priority_bench cap whenever MetaDrive asks near lane end
            # (or any request past the cap).
            cap = min(float(along), max(0.5, length - 1e-3)) if length > 1e-3 else float(along)
            if length > 1e-3 and (abs(req - length) < 1e-2 or req > cap):
                req = cap
            return orig_position(req, lateral, *rest, **kw)

        final.position = _capped_position  # type: ignore[method-assign]
        try:
            return _orig_draw(self, *args, **kwargs)
        finally:
            final.position = orig_position  # type: ignore[method-assign]

    TopDownRenderer._draw = _patched_draw
    _DEST_CAP_APPLIED = True


def apply_top_down_path_conflict_overlay_patch() -> None:
    """Monkey-patch ``TopDownRenderer._draw`` once (idempotent).

    Drawing only runs when ``set_path_conflict_overlay_enabled(True)``.
    Also installs the destination-cap patch for the red finish marker.
    """
    global _APPLIED
    apply_top_down_destination_cap_patch()
    if _APPLIED:
        return

    from metadrive.engine.top_down_renderer import TopDownRenderer

    # Destination-cap patch already wrapped _draw; wrap again for overlays.
    _orig_draw = TopDownRenderer._draw

    def _patched_draw(self, *args, **kwargs):
        if not _ENABLED:
            return _orig_draw(self, *args, **kwargs)
        real_screen = self._screen_canvas
        self._screen_canvas = _ScreenCanvasHook(real_screen, self)
        try:
            return _orig_draw(self, *args, **kwargs)
        finally:
            self._screen_canvas = real_screen

    TopDownRenderer._draw = _patched_draw
    _APPLIED = True
