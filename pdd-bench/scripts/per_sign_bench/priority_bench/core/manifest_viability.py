"""Shared checks for whether a cropped scene can enter generate_manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from .auxiliary_agent import (
    has_viable_aux_lanes,
    min_aux_spawn_lane_length,
    viable_right_aux_lane_keys,
)
from .junction_priority_layout import (
    ALLOWED_PRIORITY_JUNCTION_SHAPES,
    JunctionLayoutError,
    build_junction_priority_layout,
    right_arm_for_layout,
)
from .manifest_config import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_SPAWN_DISTANCE_BEFORE_END,
)
from .scene_augmentation import (
    SpawnStrategy,
    _aux_straight_destination,
    _ego_destination_edges,
    _is_valid_departure,
    _pick_outgoing_lane_key,
    augment_layout_for_scene,
    build_spawn_lanes_by_edge,
    lane_lengths_from_spawn_lanes,
    parse_intersection_approach_lanes,
)
from .sumo_utils import load_vehicle_route_index, load_scene_meta, resolve_net_file


@dataclass
class ManifestViabilityResult:
    viable: bool
    reason: str = ""
    detail: str = ""
    spawn_lane_count: int = 0
    scenario_count: int = 0


def parse_spawn_lanes_for_viability(net_path: Path, min_length: float) -> list:
    return parse_intersection_approach_lanes(net_path, min_length=min_length)


def _layout_kwargs_from_meta(
    meta: Optional[dict[str, Any]],
    *,
    mode: str = "main_main",
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"mode": mode}
    if not meta:
        return kwargs
    sign_lat = meta.get("latitude") or meta.get("center_lat")
    sign_lon = meta.get("longitude") or meta.get("center_lon")
    if sign_lat is not None and sign_lon is not None:
        kwargs["sign_lat"] = float(sign_lat)
        kwargs["sign_lon"] = float(sign_lon)
    return kwargs


def has_any_viable_right_aux_arm(
    junction_layout: dict,
    aux_distance_from_intersection: float,
) -> bool:
    """True if some ego arm has a right-hand aux arm with lanes long enough to spawn."""
    for arm in junction_layout.get("arms", []):
        ego_edge = arm.get("edge_id")
        if not ego_edge:
            continue
        if viable_right_aux_lane_keys(
            junction_layout, aux_distance_from_intersection, ego_edge
        ):
            return True
    return False


def _explain_no_scenarios_equal_priority(
    layout,
    spawn_lanes,
    net_path: Path,
    *,
    min_ego_lane_m: float,
    aux_distance_from_intersection: float,
) -> Tuple[str, str]:
    spawn_by_edge = build_spawn_lanes_by_edge(spawn_lanes)
    lengths = lane_lengths_from_spawn_lanes(spawn_lanes)
    route_index = load_vehicle_route_index(net_path)
    lane_keys_by_edge = {arm.edge_id: list(arm.lane_keys) for arm in layout.arms}
    min_aux_lane = min_aux_spawn_lane_length(aux_distance_from_intersection)

    ego_edges = sorted({arm.edge_id for arm in layout.arms})
    ego_edges_with_spawn = [e for e in ego_edges if spawn_by_edge.get(e)]
    if not ego_edges_with_spawn:
        return (
            "no_ego_spawn_lanes",
            f"no arm with vehicle approach lane >= {min_ego_lane_m:.0f}m",
        )

    no_right_arm: list[str] = []
    right_aux_too_short: list[str] = []
    ego_no_dest: list[str] = []
    ego_no_route = 0
    ego_no_valid_departure = 0
    aux_no_straight_dest = 0

    for ego_edge in ego_edges_with_spawn:
        right_arm = right_arm_for_layout(layout, ego_edge)
        if right_arm is None:
            no_right_arm.append(ego_edge)
            continue

        if not viable_right_aux_lane_keys(
            layout.to_dict(), aux_distance_from_intersection, ego_edge
        ):
            right_aux_too_short.append(ego_edge)
            continue

        aux_edge = right_arm.edge_id
        aux_dest_edge = _aux_straight_destination(layout, aux_edge)
        if aux_dest_edge is None:
            aux_no_straight_dest += 1
            continue

        ego_dest_edges = _ego_destination_edges(layout, ego_edge)
        if not ego_dest_edges:
            ego_no_dest.append(ego_edge)
            continue

        ego_lane_nums = [
            ln
            for ln in spawn_by_edge.get(ego_edge, [])
            if lengths.get((ego_edge, ln), 0) >= min_ego_lane_m
        ]
        if not ego_lane_nums:
            continue

        for ego_lane in ego_lane_nums:
            for ego_dest_edge in ego_dest_edges:
                if ego_dest_edge == ego_edge:
                    continue
                ego_dest_lane_key = _pick_outgoing_lane_key(
                    ego_dest_edge, ego_lane, lane_keys_by_edge
                )
                if not _is_valid_departure(
                    ego_edge, ego_lane, ego_dest_edge, ego_dest_lane_key
                ):
                    ego_no_valid_departure += 1
                    continue
                if route_index is not None and not route_index.can_reach_edge(
                    ego_edge, ego_lane, ego_dest_edge
                ):
                    ego_no_route += 1
                    continue

    if no_right_arm and len(no_right_arm) == len(ego_edges_with_spawn):
        return (
            "no_right_arm",
            f"no conflicting arm to the right for ego arms: {no_right_arm}",
        )

    if right_aux_too_short and len(right_aux_too_short) == len(ego_edges_with_spawn):
        return (
            "right_aux_lane_too_short",
            f"right-hand aux arm lane < {min_aux_lane:.0f}m for all ego arms: "
            f"{right_aux_too_short}",
        )

    if ego_no_dest and len(ego_no_dest) == len(ego_edges_with_spawn):
        return (
            "no_ego_destination",
            f"T left-turn / X straight destination missing for ego arms: {ego_no_dest}",
        )

    if ego_no_route > 0:
        return (
            "no_routable_ego_path",
            f"SUMO routing failed for ego ({ego_no_route} checks)",
        )

    if ego_no_valid_departure > 0:
        return (
            "invalid_ego_departure",
            f"ego destination equals spawn lane/edge ({ego_no_valid_departure} checks)",
        )

    if aux_no_straight_dest > 0:
        return (
            "right_aux_no_straight_dest",
            f"right-hand aux arm has no straight-through destination ({aux_no_straight_dest})",
        )

    return (
        "no_valid_scenario_combo",
        f"ego_arms={ego_edges_with_spawn}, no_right={no_right_arm}, "
        f"right_aux_short={right_aux_too_short}, ego_no_dest={ego_no_dest}, "
        f"ego_no_route={ego_no_route}",
    )


def _explain_no_scenarios_yield(
    layout,
    spawn_lanes,
    net_path: Path,
    *,
    min_ego_lane_m: float,
    aux_distance_from_intersection: float,
) -> Tuple[str, str]:
    spawn_by_edge = build_spawn_lanes_by_edge(spawn_lanes)
    lengths = lane_lengths_from_spawn_lanes(spawn_lanes)
    route_index = load_vehicle_route_index(net_path)
    lane_keys_by_edge = {arm.edge_id: list(arm.lane_keys) for arm in layout.arms}
    min_aux_lane = min_aux_spawn_lane_length(aux_distance_from_intersection)

    secondary = sorted(layout.secondary_edge_ids)
    main_edges = sorted(layout.main_edge_ids)
    ego_edges_with_spawn = [e for e in secondary if spawn_by_edge.get(e)]
    if not ego_edges_with_spawn:
        return (
            "no_ego_spawn_lanes",
            f"no secondary arm with vehicle approach lane >= {min_ego_lane_m:.0f}m "
            f"(secondary={secondary})",
        )

    ego_no_dest: list[str] = []
    ego_no_route = 0
    ego_no_valid_departure = 0
    aux_no_straight_dest = 0
    aux_lane_too_short = 0
    aux_lane_not_in_spawn_set = 0

    for ego_edge in ego_edges_with_spawn:
        ego_dest_edges = _ego_destination_edges(layout, ego_edge)
        if not ego_dest_edges:
            ego_no_dest.append(ego_edge)
            continue

        ego_lane_nums = [
            ln
            for ln in spawn_by_edge.get(ego_edge, [])
            if lengths.get((ego_edge, ln), 0) >= min_ego_lane_m
        ]
        if not ego_lane_nums:
            continue

        any_ego_route = False
        for ego_lane in ego_lane_nums:
            for ego_dest_edge in ego_dest_edges:
                if ego_dest_edge == ego_edge:
                    continue
                ego_dest_lane_key = _pick_outgoing_lane_key(
                    ego_dest_edge, ego_lane, lane_keys_by_edge
                )
                if not _is_valid_departure(
                    ego_edge, ego_lane, ego_dest_edge, ego_dest_lane_key
                ):
                    ego_no_valid_departure += 1
                    continue
                if route_index is not None and not route_index.can_reach_edge(
                    ego_edge, ego_lane, ego_dest_edge
                ):
                    ego_no_route += 1
                    continue
                any_ego_route = True

                for aux_edge in main_edges:
                    if aux_edge == ego_edge:
                        continue
                    aux_dest = _aux_straight_destination(layout, aux_edge)
                    if aux_dest is None:
                        aux_no_straight_dest += 1
                        continue
                    aux_lanes = spawn_by_edge.get(aux_edge, [])
                    if not aux_lanes:
                        aux_lane_not_in_spawn_set += 1
                        continue
                    long_enough = [
                        ln
                        for ln in aux_lanes
                        if lengths.get((aux_edge, ln), 0) >= min_aux_lane
                    ]
                    if not long_enough:
                        aux_lane_too_short += 1

        if not any_ego_route and ego_edge not in ego_no_dest:
            pass

    if ego_no_dest and len(ego_no_dest) == len(ego_edges_with_spawn):
        return (
            "no_ego_destination",
            f"T left-turn / X straight destination missing for secondary arms: {ego_no_dest}",
        )
    if ego_no_route > 0:
        return (
            "no_routable_ego_path",
            f"SUMO routing failed for ego ({ego_no_route} checks)",
        )
    if aux_lane_too_short and not aux_lane_not_in_spawn_set:
        return (
            "aux_lane_too_short",
            f"main-road aux lanes shorter than {min_aux_lane:.0f}m",
        )
    if aux_lane_not_in_spawn_set:
        return (
            "aux_lane_not_in_spawn_set",
            "main-road arms have no approach lanes in spawn set",
        )
    if aux_no_straight_dest > 0:
        return (
            "aux_no_straight_dest",
            f"main-road aux arm has no straight-through destination ({aux_no_straight_dest})",
        )
    if ego_no_valid_departure > 0:
        return (
            "invalid_ego_departure",
            f"ego destination equals spawn lane/edge ({ego_no_valid_departure} checks)",
        )

    return (
        "no_valid_scenario_combo",
        f"secondary={ego_edges_with_spawn}, main={main_edges}, "
        f"ego_no_dest={ego_no_dest}, ego_no_route={ego_no_route}, "
        f"aux_constraints=(short={aux_lane_too_short}, missing={aux_lane_not_in_spawn_set})",
    )


def check_manifest_viability(
    net_path: Path,
    *,
    meta: Optional[dict[str, Any]] = None,
    strategy: SpawnStrategy = "equal_priority",
    min_ego_lane_m: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    auxiliary_enabled: bool = True,
) -> ManifestViabilityResult:
    """Return whether a cropped scene would survive generate_manifest filters."""
    result = ManifestViabilityResult(viable=True, reason="", detail="")
    layout_mode = "main_secondary" if strategy == "yield" else "main_main"

    try:
        layout = build_junction_priority_layout(
            net_path, **_layout_kwargs_from_meta(meta, mode=layout_mode)
        )
    except JunctionLayoutError as exc:
        return ManifestViabilityResult(
            viable=False,
            reason="no_junction_layout",
            detail=str(exc),
        )

    if layout.shape not in ALLOWED_PRIORITY_JUNCTION_SHAPES:
        return ManifestViabilityResult(
            viable=False,
            reason="unsupported_junction_shape",
            detail=(
                f"shape={layout.shape!r}; priority signs 2.1/2.4 require "
                f"{sorted(ALLOWED_PRIORITY_JUNCTION_SHAPES)}"
            ),
        )

    spawn_lanes = parse_spawn_lanes_for_viability(net_path, min_ego_lane_m)
    result.spawn_lane_count = len(spawn_lanes)

    junction_layout = layout.to_dict()
    if auxiliary_enabled:
        if strategy == "yield":
            if not has_viable_aux_lanes(junction_layout, aux_distance_from_intersection):
                min_aux_lane = min_aux_spawn_lane_length(aux_distance_from_intersection)
                return ManifestViabilityResult(
                    viable=False,
                    reason="no_viable_aux_arm",
                    detail=f"no main-road aux arm with lane length >= {min_aux_lane:.0f}m",
                    spawn_lane_count=result.spawn_lane_count,
                )
        else:
            if not has_any_viable_right_aux_arm(
                junction_layout, aux_distance_from_intersection
            ):
                min_aux_lane = min_aux_spawn_lane_length(aux_distance_from_intersection)
                return ManifestViabilityResult(
                    viable=False,
                    reason="no_viable_right_aux_arm",
                    detail=f"no right-hand aux arm with lane length >= {min_aux_lane:.0f}m",
                    spawn_lane_count=result.spawn_lane_count,
                )

    sign_lat = meta.get("latitude") or meta.get("center_lat") if meta else None
    sign_lon = meta.get("longitude") or meta.get("center_lon") if meta else None
    _, scenarios = augment_layout_for_scene(
        net_path,
        spawn_lanes,
        strategy=strategy,
        min_lane_length=min_ego_lane_m,
        aux_distance_from_intersection=aux_distance_from_intersection,
        sign_lat=float(sign_lat) if sign_lat is not None else None,
        sign_lon=float(sign_lon) if sign_lon is not None else None,
    )
    result.scenario_count = len(scenarios)
    if scenarios:
        return result

    explain = (
        _explain_no_scenarios_yield
        if strategy == "yield"
        else _explain_no_scenarios_equal_priority
    )
    reason, detail = explain(
        layout,
        spawn_lanes,
        net_path,
        min_ego_lane_m=min_ego_lane_m,
        aux_distance_from_intersection=aux_distance_from_intersection,
    )
    return ManifestViabilityResult(
        viable=False,
        reason=reason,
        detail=detail,
        spawn_lane_count=result.spawn_lane_count,
        scenario_count=0,
    )


def check_scene_dir_viability(
    scene_dir: Path,
    *,
    strategy: SpawnStrategy = "equal_priority",
    min_ego_lane_m: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    auxiliary_enabled: bool = True,
) -> ManifestViabilityResult:
    """Check viability for a scene folder (meta.json + net.xml)."""
    meta_path = scene_dir / "meta.json"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = load_scene_meta(scene_dir)

    net_file = resolve_net_file(scene_dir, meta)
    net_path = scene_dir / net_file
    if not net_path.is_file():
        return ManifestViabilityResult(
            viable=False,
            reason="missing_net",
            detail=f"{scene_dir.name}: {net_file} not found",
        )

    return check_manifest_viability(
        net_path,
        meta=meta,
        strategy=strategy,
        min_ego_lane_m=min_ego_lane_m,
        aux_distance_from_intersection=aux_distance_from_intersection,
        auxiliary_enabled=auxiliary_enabled,
    )
