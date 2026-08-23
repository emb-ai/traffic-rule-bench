"""Top-down GIF film size / scaling."""
from __future__ import annotations

from traffic_bench.eval.engine.sim.top_down_local_film_patch import set_top_down_local_film

# pygame segfaults around 24k; 4800 covers ~480 m at 10 px/m.
_SAFE_FILM_PX = 4800


def _topdown_gif_film_and_scaling(
    env,
    *,
    screen_size: tuple[int, int] = (800, 800),
    window_m: float = 80.0,
) -> tuple[tuple[int, int], float]:
    """Choose film_size + MetaDrive scaling for a follow-cam GIF.

    Dual-path crops can be kilometres across. We keep a 4800 px film and tell
    MetaDrive the map bbox is a box around the ego so zoom stays at ``window_m``.
    """
    screen = int(max(screen_size[0], screen_size[1]))
    win = float(window_m) if window_m and float(window_m) > 0.0 else 80.0
    scaling_req = float(screen) / win
    film = _SAFE_FILM_PX
    # MetaDrive: scaling = min(requested, film/max_len - 0.1)
    max_len = film / (scaling_req + 0.1) - 4.0
    half_m = max(win, 0.5 * max_len)
    vehicle = getattr(env, "vehicle", None)
    pos = getattr(vehicle, "position", None) if vehicle is not None else None
    if pos is not None:
        set_top_down_local_film((float(pos[0]), float(pos[1])), half_m=half_m)
    else:
        set_top_down_local_film(None)
    return (film, film), float(scaling_req)
