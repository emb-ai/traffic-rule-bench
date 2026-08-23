"""Place RoundaboutSign (4.3) on the ego spoke + yield tracker."""

from __future__ import annotations

import json
from pathlib import Path

from traffic_bench.eval.engine.map.junction_priority_layout import JunctionLayoutError
from traffic_bench.eval.engine.map.junction_sign_placement import (
    SIGN_SHOULDER_OFFSET_M,
    arms_for_road_class,
    collect_lanes_for_keys,
    lateral_offset_beside_lane,
    resolve_sign_lane_for_edge,
    sign_longitudinal_offset,
    sign_placement_long,
)
from traffic_bench.eval.engine.map.roundabout_topology import build_roundabout_layout
from traffic_bench.eval.engine.map.roundabout_yield_zone import (
    all_entry_conflict_ring_edges,
    collect_all_entry_conflict_lanes,
    collect_entry_conflict_lanes,
    collect_lanes_for_edge_ids,
    conflict_aux_ring_edge_ids,
    entry_conflict_ring_edges,
)
from traffic_bench.eval.signs.roundabout.spawn import roundabout_meta_ring_kwargs
from traffic_bench.signs.junction import RoundaboutSign, RoundaboutYieldSign


def row_is_roundabout(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "").replace("_", ".")
    sign_type = str(row.get("sign_type") or row.get("sign_family") or "")
    return code == "4.3" or sign_type == "roundabout"


def _clear_signs(sign_mgr) -> None:
    sign_mgr.signs.clear()


def layout_from_row(row: dict, scenes_root: Path) -> dict | None:
    """Load a roundabout layout from the row or rebuild from the scene net."""
    if row.get("junction_layout"):
        layout = row["junction_layout"]
        if layout.get("shape") == "O" and layout.get("mode") == "roundabout":
            return layout

    net_path = row.get("net_path")
    if not net_path:
        return None
    net_file = Path(str(net_path))
    full_path = net_file if net_file.is_absolute() else scenes_root / net_file

    scene_meta: dict | None = None
    meta_path = full_path.parent / "meta.json"
    if meta_path.is_file():
        try:
            scene_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            scene_meta = None
    ego_edge = row.get("road_id")
    if scene_meta:
        ego_edge = (
            scene_meta.get("catalog_sign_road_id")
            or scene_meta.get("road_id")
            or ego_edge
        )
    try:
        layout_obj = build_roundabout_layout(
            full_path,
            sign_edge_id=ego_edge,
            **roundabout_meta_ring_kwargs(scene_meta),
        )
    except JunctionLayoutError as exc:
        print(f"[JunctionLayout] Failed to build roundabout layout: {exc}")
        return None
    if layout_obj.shape != "O" or layout_obj.mode != "roundabout":
        print(
            f"[JunctionLayout] Rejecting non-roundabout layout "
            f"(shape={layout_obj.shape}, mode={layout_obj.mode})"
        )
        return None
    return layout_obj.to_dict()


def _place_roundabout_sign_on_spawn_lane(
    env, distance_before_end: float = 20.0, show_model: bool = True
) -> bool:
    """Fallback: place a single RoundaboutSign beside the ego approach road."""
    try:
        vehicle = env.agent
        if vehicle is None or vehicle.lane is None:
            return False

        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False

        _clear_signs(sign_mgr)
        lane = vehicle.lane
        placement_long = sign_placement_long(lane, distance_before_end)
        sign = sign_mgr.add_sign(
            RoundaboutSign,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=show_model,
            use_random_lane=False,
            auto_detect_main_roads=False,
        )
        if sign is not None:
            sign.is_priority_sign = False
        return sign is not None
    except Exception as e:
        print(f"[RoundaboutSign] Failed to place sign: {e}")
        return False


def place_roundabout_signs(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Place RoundaboutSign (4.3) on ego spoke + invisible RoundaboutYieldSign tracker."""
    layout = layout_from_row(row, scenes_root)
    if layout is None:
        print("[RoundaboutSigns] No layout available, falling back to ego-only sign")
        return _place_roundabout_sign_on_spawn_lane(
            env, distance_before_end=distance_before_end, show_model=show_model
        )

    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return False

    _clear_signs(sign_mgr)

    main_arms = arms_for_road_class(layout, "main")
    secondary_arms = arms_for_road_class(layout, "secondary")
    if not main_arms:
        print("[RoundaboutSigns] No ring arms in layout, falling back to ego-only sign")
        return _place_roundabout_sign_on_spawn_lane(
            env, distance_before_end=distance_before_end, show_model=show_model
        )

    ring_lanes = []
    for arm in main_arms:
        ring_lanes.extend(collect_lanes_for_keys(env, arm.get("lane_keys", [])))

    if not ring_lanes:
        print("[RoundaboutSigns] Could not resolve ring lanes, falling back to ego-only sign")
        return _place_roundabout_sign_on_spawn_lane(
            env, distance_before_end=distance_before_end, show_model=show_model
        )

    ego_edge = str(row.get("road_id") or "")
    ego_arm = next((a for a in secondary_arms if a.get("edge_id") == ego_edge), None)
    entry_junction = (ego_arm or {}).get("to_node") or row.get("roundabout_entry_junction")
    incoming_edges = entry_conflict_ring_edges(
        layout,
        ego_edge,
        entry_junction_id=entry_junction,
    )
    entry_incoming_lanes = collect_entry_conflict_lanes(
        env,
        layout,
        ego_edge,
        entry_junction_id=entry_junction,
    )
    ego_main_traffic_edges = conflict_aux_ring_edge_ids(
        layout,
        ego_edge,
        entry_junction_id=entry_junction,
    )
    ego_main_traffic_lanes = collect_lanes_for_edge_ids(
        env,
        layout,
        ego_main_traffic_edges,
    )
    all_incoming_edges = all_entry_conflict_ring_edges(layout)
    all_entry_incoming_lanes = collect_all_entry_conflict_lanes(env, layout)
    print(
        f"[RoundaboutSigns] Yield conflict zone: "
        f"ego_entry={len(incoming_edges)} edge(s) "
        f"({', '.join(incoming_edges) or 'none'}), "
        f"ego_main_traffic={len(ego_main_traffic_edges)} edge(s) "
        f"({', '.join(ego_main_traffic_edges) or 'none'}), "
        f"{len(ego_main_traffic_lanes)} lane(s), "
        f"all_entries={len(all_incoming_edges)} edge(s), "
        f"{len(all_entry_incoming_lanes)} lane(s)"
    )

    junction_id = layout.get("junction_id", "")
    placed_plate = 0

    plate_arms = secondary_arms
    if ego_edge:
        if ego_arm is not None:
            plate_arms = [ego_arm]
        elif ego_edge not in layout.get("main_edge_ids", []):
            plate_arms = [{"edge_id": ego_edge, "lane_keys": []}]

    for arm in plate_arms:
        edge_id = arm.get("edge_id", "")
        lane = resolve_sign_lane_for_edge(env, edge_id, arm.get("lane_keys", []))
        if lane is None:
            print(f"[RoundaboutSigns] Skipping plate, lane not found for edge: {edge_id}")
            continue
        placement_long = sign_placement_long(lane, distance_before_end)
        try:
            sign = sign_mgr.add_sign(
                RoundaboutSign,
                lane=lane,
                longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
                lateral_offset=lateral_offset_beside_lane(lane, placement_long),
                show_model=show_model,
                use_random_lane=False,
                intersection_name=junction_id,
            )
            if sign is not None:
                sign.is_priority_sign = False
            placed_plate += 1
        except Exception as exc:
            print(f"[RoundaboutSigns] Failed RoundaboutSign on edge {edge_id}: {exc}")

    ego_lane = getattr(env.agent, "lane", None)
    if ego_lane is not None:
        placement_long = sign_placement_long(ego_lane, distance_before_end)
        try:
            tracker = sign_mgr.add_sign(
                RoundaboutYieldSign,
                lane=ego_lane,
                longitudinal_offset=sign_longitudinal_offset(ego_lane, distance_before_end),
                lateral_offset=lateral_offset_beside_lane(ego_lane, placement_long),
                show_model=False,
                use_random_lane=False,
                intersection_name=junction_id,
                ring_road_lanes=ring_lanes,
                entry_incoming_lanes=ego_main_traffic_lanes or entry_incoming_lanes,
                entry_junction_xy=None,
            )
            if tracker is not None:
                tracker.is_priority_sign = False
        except Exception as exc:
            print(f"[RoundaboutSigns] Failed yield tracker on ego lane: {exc}")

    print(
        f"[RoundaboutSigns] Placed {placed_plate} RoundaboutSign(s) + yield tracker "
        f"at entry {junction_id} (shape={layout.get('shape')}), "
        f"ring_lanes={len(ring_lanes)}, shoulder offset={SIGN_SHOULDER_OFFSET_M}m"
    )
    return placed_plate > 0

