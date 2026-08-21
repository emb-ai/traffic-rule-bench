"""Cartesian expansion of layout/aux axes into manifest rows."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..scenarios.auxiliary_agent import (
    main_lane_keys_for_aux,
    min_aux_spawn_lane_length,
    right_lane_keys_for_aux,
    viable_aux_lane_keys,
    viable_right_aux_lane_keys,
)
from ..sumo.lane_keys import make_lane_key
from ..scenarios.scene_augmentation import SpawnScenario, SpawnStrategy, augment_layout_for_scene


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


BuildEntryFn = Callable[..., Dict]


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
    # Gap only separates convoy members; size=1 → collapse gap in the key.
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


def _fit_aux_lane_keys(
    *,
    junction_layout: dict,
    spawn_strategy: str,
    aux: AuxiliaryParams,
    ego_edge: Optional[str],
    convoy_size: int,
    convoy_gap_m: Optional[float] = None,
) -> List[str]:
    gap = float(aux.convoy_gap_m if convoy_gap_m is None else convoy_gap_m)
    if spawn_strategy in ("yield", "roundabout"):
        return viable_aux_lane_keys(
            junction_layout,
            aux.distance_from_intersection,
            ego_edge,
            convoy_size=convoy_size,
            convoy_gap_m=gap,
        )
    return viable_right_aux_lane_keys(
        junction_layout,
        aux.distance_from_intersection,
        ego_edge,
        convoy_size=convoy_size,
        convoy_gap_m=gap,
    )


def _scene_aux_lane_keys_for_lane_axis(
    *,
    junction_layout: dict,
    spawn_strategy: str,
    auxiliary_on: bool,
    aux: Optional[AuxiliaryParams],
    ego_edge: Optional[str],
) -> List[str]:
    """Lane pool for the lanes-occupied axis (lead-only length when aux on)."""
    if not auxiliary_on or aux is None:
        if spawn_strategy in ("yield", "roundabout"):
            return main_lane_keys_for_aux(junction_layout, ego_edge)
        return right_lane_keys_for_aux(junction_layout, ego_edge)
    return _fit_aux_lane_keys(
        junction_layout=junction_layout,
        spawn_strategy=spawn_strategy,
        aux=aux,
        ego_edge=ego_edge,
        convoy_size=1,
    )


def _print_aux_lane_availability(
    *,
    scene_name: str,
    junction_layout: dict,
    spawn_strategy: str,
    expansion: ExpansionConfig,
) -> bool:
    """Log aux slot counts. Return False if the scene should be skipped."""
    aux = expansion.aux
    if not expansion.auxiliary_on or aux is None:
        if spawn_strategy in ("yield", "roundabout"):
            print(
                f"  Main-road lane slots for aux: "
                f"{len(main_lane_keys_for_aux(junction_layout))} (aux axis off)"
            )
        return True

    min_lane_for_lead = min_aux_spawn_lane_length(
        aux.distance_from_intersection,
        convoy_size=1,
        convoy_gap_m=min(aux.convoy_gaps_m) if aux.convoy_gaps_m else 10.0,
    )
    if spawn_strategy == "roundabout":
        from ..scenarios.roundabout_aux import MIN_CONFLICT_ARC_LENGTH_M

        min_lane_for_lead = float(MIN_CONFLICT_ARC_LENGTH_M)
    if spawn_strategy in ("yield", "roundabout"):
        available_keys = viable_aux_lane_keys(
            junction_layout, aux.distance_from_intersection
        )
        available = len(available_keys)
        label = "Conflict-arc ring" if spawn_strategy == "roundabout" else "Main-road"
        print(f"  {label} lane slots for aux: {available}")
        if available <= 0:
            print(
                f"  [aux] No {label.lower()} lanes viable for aux spawning "
                f"(need >={min_lane_for_lead:.0f}m); "
                f"skipping {scene_name}"
            )
            return False
        return True

    available = 0
    if junction_layout.get("arms"):
        from ..layout.junction_priority_layout import right_arm_edge_id

        sample_ego = junction_layout["arms"][0].get("edge_id")
        if sample_ego:
            right_edge = right_arm_edge_id(junction_layout, sample_ego)
            if right_edge:
                available = sum(
                    len(arm.get("lane_keys", []))
                    for arm in junction_layout["arms"]
                    if arm.get("edge_id") == right_edge
                )
    print(f"  Right-arm lane slots for aux (example): {available}")
    return True


def expand_scene_entries(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict,
    net_path: Path,
    spawn_lanes: Sequence[Any],
    junction_layout: dict,
    spawn_strategy: SpawnStrategy,
    sim_cfg: Any,
    expansion: ExpansionConfig,
    build_entry: BuildEntryFn,
    aux_cfg_for_entry: Any,
) -> List[Dict]:
    """Expand one scene into manifest rows (layout × aux axes + filters).

    ``aux_cfg_for_entry`` is the dataclass passed through to ``build_entry``
    (typically ``AuxiliaryConfig``); when the auxiliary axis is off the caller
    should pass a copy with ``enabled=False``.
    """
    scene_name = meta.get("scene_name", scene_dir.name)
    if spawn_strategy == "roundabout":
        print(
            f"  Junction layout: {junction_layout['shape']} @ {junction_layout['junction_id']} "
            f"(ring={len(junction_layout.get('main_edge_ids', []))}, "
            f"spokes={len(junction_layout.get('secondary_edge_ids', []))})"
        )
    elif spawn_strategy == "yield":
        print(
            f"  Junction layout: {junction_layout['shape']} @ {junction_layout['junction_id']} "
            f"(main={len(junction_layout.get('main_edge_ids', []))}, "
            f"secondary={len(junction_layout.get('secondary_edge_ids', []))})"
        )
    else:
        print(
            f"  Junction layout: {junction_layout['shape']} @ {junction_layout['junction_id']} "
            f"(equal-priority arms={len(junction_layout.get('main_edge_ids', []))})"
        )

    if not _print_aux_lane_availability(
        scene_name=scene_name,
        junction_layout=junction_layout,
        spawn_strategy=spawn_strategy,
        expansion=expansion,
    ):
        return []

    sign_lat = meta.get("latitude") or meta.get("center_lat")
    sign_lon = meta.get("longitude") or meta.get("center_lon")
    aux = expansion.aux
    aux_distance = (
        float(aux.distance_from_intersection) if aux is not None else 20.0
    )

    scenarios: List[Optional[SpawnScenario]] = []
    if expansion.layout_on:
        _, layout_scenarios = augment_layout_for_scene(
            net_path,
            list(spawn_lanes),
            strategy=spawn_strategy,
            aux_distance_from_intersection=aux_distance,
            sign_lat=float(sign_lat) if sign_lat is not None else None,
            sign_lon=float(sign_lon) if sign_lon is not None else None,
            scene_meta=meta,
        )
        if not layout_scenarios:
            print(f"  [augment] No valid scenarios for {scene_name}; skipping scene")
            return []
        scenarios = list(layout_scenarios)
        print(f"  Augmented spawn scenarios: {len(scenarios)}")
    else:
        # Single default row (no layout cartesian product).
        scenarios = [None]
        print("  Layout axis off: one default spawn per scene")

    auxiliary_on = expansion.auxiliary_on
    convoy_sizes = sizes_up_to(
        aux.convoy_size if aux is not None else 1,
        auxiliary_enabled=auxiliary_on,
    )
    # Gap is part of the aux cartesian product when the auxiliary axis is on;
    # otherwise keep a single configured gap on the row.
    if auxiliary_on and aux is not None and aux.convoy_gaps_m:
        gap_values = [float(g) for g in aux.convoy_gaps_m]
    elif aux is not None and aux.convoy_gaps_m:
        gap_values = [float(aux.convoy_gaps_m[0])]
    else:
        gap_values = [10.0]

    scene_entries: List[Dict] = []
    skipped_short_aux = 0
    skipped_dup_geometry = 0
    seen_geometries: set = set()

    for variant, scenario in enumerate(scenarios):
        ego_edge = scenario.ego_edge_id if scenario is not None else None
        prefer_aux = (
            make_lane_key(scenario.aux_edge_id, scenario.aux_lane_num)
            if scenario is not None
            else None
        )
        scene_aux_lanes = _scene_aux_lane_keys_for_lane_axis(
            junction_layout=junction_layout,
            spawn_strategy=spawn_strategy,
            auxiliary_on=auxiliary_on,
            aux=aux,
            ego_edge=ego_edge,
        )
        scene_lane_counts = sizes_up_to(
            aux.lanes_occupied if aux is not None else 1,
            auxiliary_enabled=auxiliary_on,
            available=len(scene_aux_lanes),
        )
        for lanes_n in scene_lane_counts:
            for convoy_n in convoy_sizes:
                # Gap only spaces convoy members; with size=1 it is a no-op and
                # must not duplicate otherwise-identical scenarios.
                gaps_for_n = gap_values if convoy_n > 1 else gap_values[:1]
                for gap_m in gaps_for_n:
                    if auxiliary_on and aux is not None:
                        fit_lanes = _fit_aux_lane_keys(
                            junction_layout=junction_layout,
                            spawn_strategy=spawn_strategy,
                            aux=aux,
                            ego_edge=ego_edge,
                            convoy_size=convoy_n,
                            convoy_gap_m=gap_m,
                        )
                        # Skip when the scenario aux lane (or enough lanes) cannot
                        # hold the full convoy — otherwise convoy=N collapses to
                        # fewer cars and duplicates a smaller convoy row.
                        if prefer_aux is not None and prefer_aux not in fit_lanes:
                            skipped_short_aux += 1
                            continue
                        if len(fit_lanes) < lanes_n:
                            skipped_short_aux += 1
                            continue
                    aux_cfg_gap = replace(aux_cfg_for_entry, convoy_gap_m=gap_m)
                    entry = build_entry(
                        scene_dir=scene_dir,
                        scenes_root=scenes_root,
                        meta=meta,
                        variant=variant,
                        sim_cfg=sim_cfg,
                        aux_cfg=aux_cfg_gap,
                        aux_convoy_size=convoy_n,
                        aux_lanes_occupied=lanes_n,
                        spawn_lanes_cache=list(spawn_lanes),
                        junction_layout_cache=junction_layout,
                        spawn_scenario=scenario,
                    )
                    geom_key = entry_geometry_key(entry)
                    if geom_key in seen_geometries:
                        skipped_dup_geometry += 1
                        continue
                    seen_geometries.add(geom_key)
                    scene_entries.append(entry)

    if skipped_short_aux:
        print(
            f"  [aux] Skipped {skipped_short_aux} convoy×lanes×gap combo(s) "
            f"(aux approach too short for full convoy)"
        )
    if skipped_dup_geometry:
        print(
            f"  [aux] Skipped {skipped_dup_geometry} duplicate combo(s) "
            f"(same ego path + occupied aux lanes + convoy + gap)"
        )

    # After the full cartesian product (+ filters), shuffle then cap per scene.
    cap = expansion.max_scenarios
    pre_cap = len(scene_entries)
    scene_entries = shuffle_cap(
        scene_entries,
        cap,
        seed_key=(scene_name, "max_scenarios_shuffle", int(cap) if cap is not None else 0),
    )
    if cap is not None and pre_cap > cap:
        print(
            f"  Retained {len(scene_entries)} of {pre_cap} manifest entries "
            f"for {scene_name} (shuffled, cap={cap})"
        )
    elif cap is not None:
        print(
            f"  Manifest entries for {scene_name}: {len(scene_entries)} "
            f"(under cap={cap})"
        )
    else:
        print(f"  Manifest entries for {scene_name}: {len(scene_entries)}")

    return scene_entries
