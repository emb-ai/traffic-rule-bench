"""Bridge to one_way_signs dual-path discovery (read-only import).

Moscow / priority_bench scenes do not store crop-time ``dual_path`` meta.
Manifest expansion rediscovers short-forbidden + long-compliant routes via
``one_way_signs.lib.direction_dual_path`` without modifying that package.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

_ONE_WAY_ROOT = Path(__file__).resolve().parents[3] / "one_way_signs"
if str(_ONE_WAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_ONE_WAY_ROOT))

from lib.direction_dual_path import (  # noqa: E402
    DualPathScenario,
    find_dual_path_scenarios,
    straight_path_has_dead_end_uturn,
    straight_path_reenters_signed_junction,
)
from lib.one_way_sign_spec import (  # noqa: E402
    get_one_way_sign_spec,
    resolve_sign_class,
)

from ..sumo.lane_keys import make_lane_key
from .scene_augmentation import SpawnScenario


def dual_path_to_spawn_scenario(
    dp: DualPathScenario,
    *,
    ego_lane_num: Optional[int] = None,
) -> SpawnScenario:
    """Map a dual-path pick onto priority_bench ``SpawnScenario`` (no aux).

    ``ego_lane_num`` overrides the discovery default so manifest expansion can
    multiply rows across all approach lanes on the ego edge.
    """
    lane = int(dp.ego_lane_num if ego_lane_num is None else ego_lane_num)
    dest_key = make_lane_key(dp.dest_edge_id, dp.dest_lane_num)
    return SpawnScenario(
        ego_edge_id=dp.ego_edge_id,
        ego_lane_num=lane,
        ego_destination_edge_id=dp.dest_edge_id,
        ego_destination_lane_key=dest_key,
        aux_edge_id="",
        aux_lane_num=0,
        aux_destination_edge_id="",
        aux_destination_lane_key="",
        scenario_id=(
            f"dual_{dp.junction_id}_{dp.ego_edge_id}_{dp.dest_edge_id}"
            f"_{dp.turn_dir}_L{lane}"
        ),
    )


def ego_spawn_lane_nums_for_dual(
    dp: DualPathScenario,
    spawn_lanes: Sequence[Any],
    *,
    min_lane_length_m: float,
) -> List[int]:
    """Approach lane indices on the dual-path ego edge (length-filtered)."""
    nums = sorted(
        {
            int(lane.lane_num)
            for lane in spawn_lanes
            if str(getattr(lane, "edge_id", "")) == str(dp.ego_edge_id)
            and float(getattr(lane, "length", 0.0) or 0.0) >= float(min_lane_length_m)
        }
    )
    if nums:
        return nums
    return [int(dp.ego_lane_num)]


def discover_one_way_dual_paths(
    net_path: Path,
    *,
    pdd_code: str,
    min_gain_m: float = 20.0,
    min_lane_length_m: float = 8.0,
    max_scenarios: int = 20,
    junction_ids: Optional[Sequence[str]] = None,
    arm_counts: Sequence[int] = (3, 4),
) -> List[DualPathScenario]:
    """Find dual-path scenarios on a cropped net; drop dead-end / revisit paths."""
    raw = find_dual_path_scenarios(
        net_path,
        pdd_code=pdd_code,
        min_lane_length_m=min_lane_length_m,
        min_gain_m=min_gain_m,
        max_scenarios=max_scenarios,
        junction_ids=list(junction_ids) if junction_ids else None,
        arm_counts=tuple(arm_counts),
    )
    kept: List[DualPathScenario] = []
    for scenario in raw:
        if straight_path_has_dead_end_uturn(net_path, scenario):
            continue
        if straight_path_reenters_signed_junction(net_path, scenario):
            continue
        kept.append(scenario)
    return kept


__all__ = [
    "DualPathScenario",
    "discover_one_way_dual_paths",
    "dual_path_to_spawn_scenario",
    "ego_spawn_lane_nums_for_dual",
    "get_one_way_sign_spec",
    "resolve_sign_class",
]
