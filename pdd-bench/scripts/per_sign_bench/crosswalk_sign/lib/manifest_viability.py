"""Shared checks for whether a cropped scene can enter generate_manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from lib.auxiliary_agent import has_viable_aux_lanes, min_aux_spawn_lane_length
from lib.junction_priority_layout import JunctionLayoutError, build_junction_priority_layout
from lib.manifest_config import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_SPAWN_DISTANCE_BEFORE_END,
)
from lib.scene_augmentation import (
    _aux_straight_destination,
    _ego_destination_edges,
    _is_valid_departure,
    _pick_outgoing_lane_key,
    augment_layout_for_scene,
    build_spawn_lanes_by_edge,
    lane_lengths_from_spawn_lanes,
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
    """Match generate_manifest.parse_sumo_net_for_spawn_lanes."""
    from generate_manifest import parse_sumo_net_for_spawn_lanes

    return parse_sumo_net_for_spawn_lanes(net_path, min_length=min_length)


def _explain_no_scenarios(
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

    ego_no_dest = []
    ego_no_route = 0
    ego_no_valid_departure = 0
    aux_no_viable_lane = 0
    aux_no_approach_lane = 0
    aux_no_straight_dest = 0

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
                    aux_dest_edge = _aux_straight_destination(layout, aux_edge)
                    if aux_dest_edge is None:
                        aux_no_straight_dest += 1
                        continue
                    aux_lane_nums = spawn_by_edge.get(aux_edge, [])
                    if not aux_lane_nums:
                        aux_no_approach_lane += 1
                        continue
                    viable_aux = [
                        ln
                        for ln in aux_lane_nums
                        if lengths.get((aux_edge, ln), 0) >= min_aux_lane
                    ]
                    if not viable_aux:
                        aux_no_viable_lane += 1
                        continue

        if not any_ego_route and ego_dest_edges:
            pass

    if ego_no_dest and len(ego_no_dest) == len(ego_edges_with_spawn):
        return (
            "no_ego_destination",
            f"T-junction left-turn / X straight destination missing for ego arms: {ego_no_dest}",
        )

    if ego_no_route > 0 and aux_no_viable_lane == 0 and aux_no_approach_lane == 0:
        if ego_no_route >= len(ego_edges_with_spawn):
            return (
                "no_routable_ego_path",
                f"ego cannot reach left/straight destination ({ego_no_route} blocked route checks)",
            )

    if aux_no_viable_lane > 0 and ego_no_route == 0:
        return (
            "aux_lane_too_short",
            f"main arms exist but no aux approach lane >= {min_aux_lane:.0f}m "
            f"({aux_no_viable_lane} arm checks)",
        )

    if aux_no_approach_lane > 0 and aux_no_viable_lane == 0:
        return (
            "aux_lane_not_in_spawn_set",
            f"main arm lanes < {min_ego_lane_m:.0f}m in approach-lane parse "
            f"({aux_no_approach_lane} checks)",
        )

    if ego_no_valid_departure > 0:
        return (
            "invalid_ego_departure",
            f"ego destination equals spawn lane/edge ({ego_no_valid_departure} checks)",
        )

    if ego_no_route > 0:
        return (
            "no_routable_ego_path",
            f"SUMO routing failed for ego left/straight ({ego_no_route} checks)",
        )

    if aux_no_straight_dest > 0 and aux_no_viable_lane > 0:
        return (
            "aux_constraints",
            f"aux straight dest missing ({aux_no_straight_dest}), "
            f"short aux lanes ({aux_no_viable_lane})",
        )

    return (
        "no_valid_scenario_combo",
        f"ego_arms={ego_edges_with_spawn}, main={main_edges}, "
        f"ego_no_dest={ego_no_dest}, ego_no_route={ego_no_route}, "
        f"aux_short={aux_no_viable_lane}, aux_no_spawn={aux_no_approach_lane}",
    )


def check_manifest_viability(
    net_path: Path,
    *,
    meta: Optional[dict[str, Any]] = None,
    min_ego_lane_m: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    auxiliary_enabled: bool = True,
) -> ManifestViabilityResult:
    """Return whether a cropped scene would survive generate_manifest filters."""
    result = ManifestViabilityResult(viable=True, reason="", detail="")

    try:
        layout = build_junction_priority_layout(net_path)
    except JunctionLayoutError as exc:
        return ManifestViabilityResult(
            viable=False,
            reason="no_junction_layout",
            detail=str(exc),
        )

    spawn_lanes = parse_spawn_lanes_for_viability(net_path, min_ego_lane_m)
    result.spawn_lane_count = len(spawn_lanes)

    junction_layout = layout.to_dict()
    if auxiliary_enabled and not has_viable_aux_lanes(
        junction_layout, aux_distance_from_intersection
    ):
        min_aux_lane = min_aux_spawn_lane_length(aux_distance_from_intersection)
        return ManifestViabilityResult(
            viable=False,
            reason="no_viable_aux_arm",
            detail=f"no main arm with lane length >= {min_aux_lane:.0f}m",
            spawn_lane_count=result.spawn_lane_count,
        )

    _, scenarios = augment_layout_for_scene(
        net_path,
        spawn_lanes,
        min_lane_length=min_ego_lane_m,
        aux_distance_from_intersection=aux_distance_from_intersection,
    )
    result.scenario_count = len(scenarios)
    if scenarios:
        return result

    reason, detail = _explain_no_scenarios(
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
        min_ego_lane_m=min_ego_lane_m,
        aux_distance_from_intersection=aux_distance_from_intersection,
        auxiliary_enabled=auxiliary_enabled,
    )
