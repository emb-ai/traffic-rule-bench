"""Larger HUD font for Violations lines — without editing the metadrive package."""

from __future__ import annotations

_APPLIED = False

DEFAULT_FONT_SIZE = 25
VIOLATIONS_FONT_SIZE = 34
VIOLATIONS_LINE_INTERVAL = 30


def apply_top_down_violations_text_patch(
    *,
    default_size: int = DEFAULT_FONT_SIZE,
    violations_size: int = VIOLATIONS_FONT_SIZE,
    violations_interval: int = VIOLATIONS_LINE_INTERVAL,
) -> None:
    """Monkey-patch ``TopDownRenderer._add_text`` once (idempotent)."""
    global _APPLIED
    if _APPLIED:
        return

    import pygame
    from metadrive.engine.top_down_renderer import TopDownRenderer

    def _patched_add_text(self, text: dict):
        if not text:
            return
        if not pygame.get_init():
            pygame.init()
        font_default = pygame.font.SysFont("didot.ttc", int(default_size))
        font_violations = pygame.font.SysFont("didot.ttc", int(violations_size))
        x, y = self._text_render_pos
        for key, value in text.items():
            is_violations = "violation" in str(key).lower()
            font = font_violations if is_violations else font_default
            interval = (
                int(violations_interval)
                if is_violations
                else self._text_render_interval
            )
            line = (
                f"{key}: {value}"
                if not str(key).rstrip().endswith(":")
                else f"{key}{value}"
            )
            img2 = font.render(line, True, (0, 0, 0))
            self._screen_canvas.blit(img2, (x, y))
            y += interval

    TopDownRenderer._add_text = _patched_add_text
    _APPLIED = True
