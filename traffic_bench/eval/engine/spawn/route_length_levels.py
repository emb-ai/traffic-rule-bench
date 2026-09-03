"""Route-length augmentation levels for manifest expand.

Single source of truth: ``simulation.max_path_length_levels``.
One value (e.g. ``[150]``) → no route-length axis; several → cartesian expand.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from traffic_bench.eval.engine.expand.manifest_config import (
    DEFAULT_MAX_PATH_LENGTH_LEVELS,
    DEFAULT_MAX_PATH_LENGTH_M,
)


def list_route_length_levels(sim: Any) -> List[float]:
    """Return path-length budget values to materialize for one base scenario."""
    default = float(
        getattr(sim, "max_path_length_m", DEFAULT_MAX_PATH_LENGTH_M)
        or DEFAULT_MAX_PATH_LENGTH_M
    )
    raw: Sequence[float] = getattr(sim, "max_path_length_levels", None) or ()
    if not raw:
        raw = DEFAULT_MAX_PATH_LENGTH_LEVELS
    levels = sorted({float(x) for x in raw if float(x) > 0.0})
    return levels if levels else [default]


def select_route_length_levels(
    levels: Sequence[float],
    available_m: float | None,
    *,
    eps: float = 1.0,
) -> tuple[List[float], bool]:
    """Pick route budgets that actually shorten the natural path.

    A level ``L`` is kept only when ``L + eps < available`` (it truncates).
    If the natural path is shorter than every configured level, return a single
    budget equal to the available length with ``augment=False`` so expand does
    not invent fake ``rl130`` / ``rl150`` / ``rl170`` clones of the same route.
    """
    cleaned = sorted({float(x) for x in levels if float(x) > 0.0})
    if not cleaned:
        cleaned = [float(DEFAULT_MAX_PATH_LENGTH_M)]
    if available_m is None:
        return cleaned, len(cleaned) > 1
    avail = max(5.0, float(available_m))
    usable = [L for L in cleaned if L + float(eps) < avail]
    if usable:
        return usable, len(usable) > 1
    return [round(avail, 1)], False


def route_length_scene_suffix(path_len_m: float) -> str:
    return f"rl{int(round(float(path_len_m)))}"


def tag_entry_route_length(
    entry: Dict[str, Any],
    path_len_m: float,
    *,
    augment: bool,
) -> Dict[str, Any]:
    """Stamp route budget on a manifest row; suffix ``scene_id`` when augmenting."""
    out = dict(entry)
    plen = float(path_len_m)
    out["max_path_length_m"] = plen
    out["route_length_level_m"] = plen
    if not augment:
        return out
    suffix = route_length_scene_suffix(plen)
    base_id = str(out.get("scene_id") or out.get("scene_name") or "scene")
    if suffix not in base_id:
        out["scene_id"] = f"{base_id}_{suffix}"
    aug = out.get("augmentation_id")
    if aug and suffix not in str(aug):
        out["augmentation_id"] = f"{aug}_{suffix}"
    return out
