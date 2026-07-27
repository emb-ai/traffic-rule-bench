"""Shared checks for whether a cropped scene can enter generate_manifest (direction signs / junction scaffold)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from lib.junction_priority_layout import (
    JunctionLayoutError,
    build_junction_priority_layout,
)
from lib.manifest_config import (
    DEFAULT_DESTINATION_PAST_SIGN_M,
    DEFAULT_SIGN_DISTANCE_FROM_START,
    DEFAULT_SPAWN_DISTANCE_BEFORE_END,
)
from lib.no_entry_route import forbidden_edge_geometry_ok
from lib.scene_augmentation import (
    _ego_destination_edges,
    _is_valid_departure,
    _pick_outgoing_lane_key,
    augment_layout_for_scene,
    build_spawn_lanes_by_edge,
    lane_lengths_from_spawn_lanes,
    parse_intersection_approach_lanes,
)
from lib.sumo_utils import load_vehicle_route_index, load_scene_meta, resolve_net_file


@dataclass
class ManifestViabilityResult:
    viable: bool
    reason: str = ""
    detail: str = ""
    spawn_lane_count: int = 0
    scenario_count: int = 0


def parse_spawn_lanes_for_viability(net_path: Path, min_length: float) -> list:
    return parse_intersection_approach_lanes(net_path, min_length=min_length)


def _layout_kwargs_from_meta(meta: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not meta:
        return {}
    sign_lat = meta.get("latitude") or meta.get("center_lat")
    sign_lon = meta.get("longitude") or meta.get("center_lon")
    kwargs: dict[str, Any] = {"mode": "main_main"}
    if sign_lat is not None and sign_lon is not None:
        kwargs["sign_lat"] = float(sign_lat)
        kwargs["sign_lon"] = float(sign_lon)
    return kwargs


def _explain_no_scenarios(
    layout,
    spawn_lanes,
    net_path: Path,
    *,
    min_ego_lane_m: float,
) -> Tuple[str, str]:
    spawn_by_edge = build_spawn_lanes_by_edge(spawn_lanes)
    lengths = lane_lengths_from_spawn_lanes(spawn_lanes)
    route_index = load_vehicle_route_index(net_path)
    lane_keys_by_edge = {arm.edge_id: list(arm.lane_keys) for arm in layout.arms}

    ego_edges = sorted({arm.edge_id for arm in layout.arms})
    ego_edges_with_spawn = [e for e in ego_edges if spawn_by_edge.get(e)]
    if not ego_edges_with_spawn:
        return (
            "no_ego_spawn_lanes",
            f"no arm with vehicle approach lane >= {min_ego_lane_m:.0f}m",
        )

    ego_no_dest: list[str] = []
    ego_no_route = 0
    ego_no_valid_departure = 0

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

    return (
        "no_valid_scenario_combo",
        f"ego_arms={ego_edges_with_spawn}, ego_no_dest={ego_no_dest}, "
        f"ego_no_route={ego_no_route}",
    )


def check_manifest_viability(
    net_path: Path,
    *,
    meta: Optional[dict[str, Any]] = None,
    min_ego_lane_m: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    sign_distance_from_start: float = DEFAULT_SIGN_DISTANCE_FROM_START,
    destination_past_sign_m: float = DEFAULT_DESTINATION_PAST_SIGN_M,
) -> ManifestViabilityResult:
    """Return whether a cropped scene would survive generate_manifest filters."""
    result = ManifestViabilityResult(viable=True, reason="", detail="")

    try:
        layout = build_junction_priority_layout(net_path, **_layout_kwargs_from_meta(meta))
    except JunctionLayoutError as exc:
        return ManifestViabilityResult(
            viable=False,
            reason="no_junction_layout",
            detail=str(exc),
        )

    spawn_lanes = parse_spawn_lanes_for_viability(net_path, min_ego_lane_m)
    result.spawn_lane_count = len(spawn_lanes)

    sign_lat = meta.get("latitude") or meta.get("center_lat") if meta else None
    sign_lon = meta.get("longitude") or meta.get("center_lon") if meta else None
    _, scenarios = augment_layout_for_scene(
        net_path,
        spawn_lanes,
        min_lane_length=min_ego_lane_m,
        sign_lat=float(sign_lat) if sign_lat is not None else None,
        sign_lon=float(sign_lon) if sign_lon is not None else None,
    )
    long_enough = [
        sc
        for sc in scenarios
        if forbidden_edge_geometry_ok(
            net_path,
            sc.ego_destination_edge_id,
            sign_distance_from_start=sign_distance_from_start,
            destination_past_sign_m=destination_past_sign_m,
        )[0]
    ]
    result.scenario_count = len(long_enough)
    if long_enough:
        return result

    if scenarios:
        needed = float(sign_distance_from_start) + float(destination_past_sign_m)
        return ManifestViabilityResult(
            viable=False,
            reason="forbidden_lane_too_short",
            detail=(
                f"{len(scenarios)} through-path scenario(s) but all forbidden "
                f"edges <= sign_from_start+past ({needed:.1f}m)"
            ),
            spawn_lane_count=result.spawn_lane_count,
            scenario_count=0,
        )

    reason, detail = _explain_no_scenarios(
        layout,
        spawn_lanes,
        net_path,
        min_ego_lane_m=min_ego_lane_m,
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
    min_ego_lane_m: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    sign_distance_from_start: float = DEFAULT_SIGN_DISTANCE_FROM_START,
    destination_past_sign_m: float = DEFAULT_DESTINATION_PAST_SIGN_M,
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
        min_ego_lane_m=min_ego_lane_m,
        sign_distance_from_start=sign_distance_from_start,
        destination_past_sign_m=destination_past_sign_m,
    )
