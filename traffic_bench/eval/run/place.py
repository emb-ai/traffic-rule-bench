"""Dispatch plate placement to ``signs/<family>/place.py``."""

from __future__ import annotations

from pathlib import Path

from traffic_bench.eval.signs.blocked.place import (
    place_blocked_road_sign,
    row_is_blocked_road,
)
from traffic_bench.eval.signs.crosswalk.place import (
    place_crosswalk_signs,
    row_is_crosswalk,
)
from traffic_bench.eval.signs.detour.place import place_detour_signs, row_is_detour
from traffic_bench.eval.signs.dual_path.place import (
    place_dual_path_signs,
    row_uses_dual_path_nav,
)
from traffic_bench.eval.signs.junction.place import place_junction_signs
from traffic_bench.eval.signs.roundabout.place import (
    place_roundabout_signs,
    row_is_roundabout,
)
from traffic_bench.eval.signs.restricted_lane.place import (
    place_restricted_lane_signs,
    row_is_restricted_lane,
)
from traffic_bench.eval.signs.speed.place import place_speed_signs, row_is_speed


def place_signs_for_row(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Put the family's plates in the live world for this manifest row."""
    if row_is_detour(row):
        return place_detour_signs(env, row, show_model=show_model)
    if row_is_speed(row):
        return place_speed_signs(env, row, show_model=show_model)
    if row_is_restricted_lane(row):
        return place_restricted_lane_signs(env, row, show_model=show_model)
    if row_is_crosswalk(row):
        return place_crosswalk_signs(
            env,
            row,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )
    if row_uses_dual_path_nav(row):
        return place_dual_path_signs(
            env,
            row,
            scenes_root=scenes_root,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )
    if row_is_blocked_road(row):
        return place_blocked_road_sign(
            env,
            row,
            scenes_root=scenes_root,
            show_model=show_model,
        )
    if row_is_roundabout(row):
        return place_roundabout_signs(
            env,
            row,
            scenes_root=scenes_root,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )
    return place_junction_signs(
        env,
        row,
        scenes_root=scenes_root,
        distance_before_end=distance_before_end,
        show_model=show_model,
    )
