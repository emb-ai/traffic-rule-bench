"""Top-down GIF film size / scaling."""
from __future__ import annotations

def _topdown_gif_film_and_scaling(
    env,
    *,
    screen_size: tuple[int, int] = (800, 800),
    window_m: float = 80.0,
) -> tuple[tuple[int, int], float]:
    """Choose film_size + MetaDrive scaling for a fixed visible window (meters).

    Dual-path crops are often >1 km; a fixed ``film_size=(4800,4800)`` would
    clamp zoom. Grow film with map bbox so ``window_m`` is honored without
    editing MetaDrive.
    """
    screen = int(max(screen_size[0], screen_size[1]))
    win = float(window_m) if window_m and float(window_m) > 0.0 else 80.0
    scaling_req = float(screen) / win

    max_len = 400.0
    try:
        b_box = env.engine.current_map.road_network.get_bounding_box()
        max_len = max(
            float(b_box[1] - b_box[0]),
            float(b_box[3] - b_box[2]),
            1.0,
        )
    except Exception:
        pass

    # MetaDrive: scaling = min(requested, film/max_len - 0.1)
    need = int((scaling_req + 0.1) * max_len + 64)
    film = max(4800, need)
    film = min(film, 24000)  # soft cap for RAM
    return (film, film), float(scaling_req)


