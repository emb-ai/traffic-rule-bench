"""Shared expansion-axis types and shuffle/cap helpers.

Junction / roundabout cartesian product lives in
``traffic_bench.eval.signs.junction.expand``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AuxiliaryParams:
    """Aux spawn parameters (ignored when the auxiliary axis is off)."""

    enabled: bool = True
    distance_from_intersection: float = 20.0
    convoy_size: int = 1
    convoy_gaps_m: Tuple[float, ...] = (10.0,)
    lanes_occupied: int = 1
    release_when_ego_within_m: float = 15.0

    @property
    def convoy_gap_m(self) -> float:
        """Default / first gap (used when the gap axis is collapsed)."""
        return float(self.convoy_gaps_m[0]) if self.convoy_gaps_m else 10.0


@dataclass(frozen=True)
class ExpansionConfig:
    """Which augmentation axes to run for the current sign."""

    enabled: bool = True
    layout: bool = False
    auxiliary: bool = False
    max_scenarios: Optional[int] = None
    aux: Optional[AuxiliaryParams] = None

    @property
    def layout_on(self) -> bool:
        return bool(self.enabled) and bool(self.layout)

    @property
    def auxiliary_on(self) -> bool:
        """True when the auxiliary axis is enabled and aux agents are on."""
        if not (self.enabled and self.auxiliary and self.aux is not None):
            return False
        return bool(self.aux.enabled)


def shuffle_cap(items: List, cap: Optional[int], *, seed_key: tuple) -> List:
    """Keep ``cap`` items after a deterministic shuffle.

    Used by every sign after the full augmentation product: shuffle only when
    ``len(items) > cap``. ``cap is None`` or ``cap < 0`` leaves the list as-is.
    """
    if cap is None:
        return items
    try:
        cap_i = int(cap)
    except (TypeError, ValueError):
        return items
    if cap_i < 0 or len(items) <= cap_i:
        return items
    out = list(items)
    rng = random.Random(hash(tuple(seed_key)) & 0xFFFFFFFF)
    rng.shuffle(out)
    return out[:cap_i]


def sizes_up_to(
    max_value: int,
    *,
    auxiliary_enabled: bool = True,
    available: Optional[int] = None,
) -> List[int]:
    """Return values to materialize: {1, 2, ..., cap} for aux manifest expansion."""
    if not auxiliary_enabled:
        return [1]
    if available is not None and available <= 0:
        return [1]
    cap = max(1, int(max_value))
    if available is not None:
        cap = min(cap, int(available))
    return list(range(1, cap + 1))


def entry_geometry_key(entry: Dict) -> Tuple:
    """Identity of what appears on the map (order of aux lanes ignored).

    When several aux lanes are occupied, runtime fills *all* of them; the
    layout-scenario "primary" aux spawn/dest only picks prefer-order and the
    written ``aux_destination_*``. Two rows with the same occupied set then
    look identical (each non-primary lane still gets its straight-through
    route), so primary dest is omitted from the key for multi-lane occupy.
    """
    occupied = frozenset(entry.get("aux_occupied_lane_keys") or [])
    if len(occupied) <= 1:
        dest_key = (
            entry.get("aux_destination_lane_id")
            or entry.get("aux_destination_edge_id")
        )
        spawn_key = entry.get("aux_spawn_lane_index") or entry.get("aux_road_id")
    else:
        dest_key = None
        spawn_key = None
    convoy_n = int(entry.get("aux_convoy_size") or 1)
    gap_key = (
        round(float(entry.get("aux_convoy_gap_m") or 0.0), 3) if convoy_n > 1 else 0.0
    )
    return (
        entry.get("road_id"),
        entry.get("spawn_lane_num"),
        entry.get("destination_lane_id"),
        occupied,
        spawn_key,
        dest_key,
        convoy_n,
        gap_key,
    )


def expand_scene_entries(*args, **kwargs):
    """Shim → ``signs.junction.expand.expand_scene_entries``."""
    from traffic_bench.eval.signs.junction.expand import expand_scene_entries as _impl

    return _impl(*args, **kwargs)
