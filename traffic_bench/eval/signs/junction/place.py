"""Place 2.1 / 2.3 / 2.4 / 2.5 plates on T/X junction arms."""

from __future__ import annotations

from pathlib import Path

from traffic_bench.eval.engine.map.junction_priority_layout import (
    JunctionLayoutError,
    build_junction_priority_layout,
    right_arm_edge_id,
    secondary_side_from_main_arm,
    straight_arm_edge_id,
)
from traffic_bench.eval.engine.map.junction_sign_placement import (
    SIGN_SHOULDER_OFFSET_M,
    arms_for_road_class,
    collect_lanes_for_keys,
    lateral_offset_beside_lane,
    resolve_sign_lane_for_edge,
    sign_longitudinal_offset,
    sign_placement_long,
)
from traffic_bench.signs.priority_signs import (
    MainRoadSign,
    RightHandYieldSign,
    SecondaryRoadLeftSign,
    SecondaryRoadRightSign,
    SecondaryRoadSign,
    StopSign,
    YieldSign,
)


def _code(row: dict) -> str:
    return str(row.get("pdd_code") or row.get("sign_code") or "")


def _sign_type(row: dict) -> str:
    return str(row.get("sign_type") or row.get("sign_family") or "")


def row_is_yield(row: dict) -> bool:
    code = _code(row).replace("_", ".")
    return code == "2.4" or _sign_type(row) == "yield"


def row_is_stop(row: dict) -> bool:
    code = _code(row).replace("_", ".")
    return code == "2.5" or _sign_type(row) in {"stop", "stop_sign"}


def row_is_secondary_road(row: dict) -> bool:
    code = _code(row).replace("_", ".")
    return code.startswith("2.3") or _sign_type(row) in {"secondary", "secondary_road"}


def row_is_main(row: dict) -> bool:
    code = _code(row).replace("_", ".")
    return code == "2.1" or _sign_type(row) in {"main", "main_road"}


def row_is_junction(row: dict) -> bool:
    return row_is_yield(row) or row_is_stop(row) or row_is_secondary_road(row) or row_is_main(row)


def _clear_signs(sign_mgr) -> None:
    sign_mgr.signs.clear()


def _layout_from_row(row: dict, scenes_root: Path) -> dict | None:
    layout = row.get("junction_layout")
    if layout:
        return layout
    net_path = row.get("net_path")
    if not net_path:
        return None
    net_file = Path(str(net_path))
    full_path = net_file if net_file.is_absolute() else scenes_root / net_file
    mode = "main_secondary" if (
        row_is_yield(row) or row_is_stop(row) or row_is_secondary_road(row)
    ) else "main_main"
    try:
        return build_junction_priority_layout(full_path, mode=mode).to_dict()
    except JunctionLayoutError as exc:
        print(f"[JunctionLayout] Failed to build layout: {exc}")
        return None


def _place_main_sign_on_spawn_lane(
    env, distance_before_end: float = 20.0, show_model: bool = True
) -> bool:
    """Fallback: place a single MainRoadSign beside the ego approach road."""
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
            MainRoadSign,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=show_model,
            use_random_lane=False,
        )
        if sign is not None:
            sign.is_priority_sign = False
        return sign is not None
    except Exception as e:
        print(f"[MainRoadSign] Failed to place sign: {e}")
        return False


def _place_equal_priority_main_signs(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Place MainRoadSign (2.1) on every incoming arm — equal priority intersection."""
    layout = _layout_from_row(row, scenes_root)
    if layout is None:
        print("[JunctionSigns] No layout available, falling back to ego-only main sign")
        return _place_main_sign_on_spawn_lane(
            env, distance_before_end=distance_before_end, show_model=show_model
        )

    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return False

    _clear_signs(sign_mgr)

    all_arms = layout.get("arms", [])
    if not all_arms:
        print("[JunctionSigns] No arms in layout, falling back to ego-only main sign")
        return _place_main_sign_on_spawn_lane(
            env, distance_before_end=distance_before_end, show_model=show_model
        )

    junction_id = layout.get("junction_id", "")
    placed_main = 0

    for arm in all_arms:
        edge_id = arm.get("edge_id", "")
        lane = resolve_sign_lane_for_edge(env, edge_id, arm.get("lane_keys", []))
        if lane is None:
            print(f"[JunctionSigns] Skipping main sign, lane not found for edge: {edge_id}")
            continue
        placement_long = sign_placement_long(lane, distance_before_end)
        try:
            sign = sign_mgr.add_sign(
                MainRoadSign,
                lane=lane,
                longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
                lateral_offset=lateral_offset_beside_lane(lane, placement_long),
                show_model=show_model,
                use_random_lane=False,
                intersection_name=junction_id,
            )
            if sign is not None:
                sign.is_priority_sign = False
            placed_main += 1
        except Exception as exc:
            print(f"[JunctionSigns] Failed MainRoadSign on edge {edge_id}: {exc}")

    print(
        f"[JunctionSigns] Placed {placed_main} MainRoadSign(s) "
        f"at junction {junction_id} ({layout.get('shape')}), "
        f"shoulder offset={SIGN_SHOULDER_OFFSET_M}m"
    )
    return placed_main > 0


def _place_right_hand_yield_tracker(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
) -> bool:
    """Invisible RightHandYieldSign on ego lane for violation metrics."""
    layout = _layout_from_row(row, scenes_root)
    if layout is None:
        return False

    ego_edge = row.get("road_id")
    if not ego_edge:
        return False

    right_edge = right_arm_edge_id(layout, str(ego_edge))
    if not right_edge:
        print(f"[RightHandRule] No right arm for ego edge {ego_edge!r}")
        return False

    right_lane_keys = []
    for arm in layout.get("arms", []):
        if arm.get("edge_id") == right_edge:
            right_lane_keys = list(arm.get("lane_keys", []))
            break

    right_lanes = collect_lanes_for_keys(env, right_lane_keys)
    if not right_lanes:
        print(f"[RightHandRule] Could not resolve lanes for right arm {right_edge}")
        return False

    vehicle = env.agent
    if vehicle is None or vehicle.lane is None:
        return False

    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return False

    lane = vehicle.lane
    placement_long = sign_placement_long(lane, distance_before_end)
    try:
        sign = sign_mgr.add_sign(
            RightHandYieldSign,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=False,
            use_random_lane=False,
            intersection_name=layout.get("junction_id", ""),
            right_road_lanes=right_lanes,
        )
        if sign is not None:
            sign.is_priority_sign = False
        print(
            f"[RightHandRule] Tracker on ego {ego_edge}, "
            f"yield-to-right arm {right_edge} ({len(right_lanes)} lane(s))"
        )
        return sign is not None
    except Exception as exc:
        print(f"[RightHandRule] Failed to place tracker: {exc}")
        return False


def _place_secondary_sign_on_spawn_lane(
    env,
    secondary_sign_cls,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Fallback: place a single secondary-arm plate beside the ego approach."""
    label = secondary_sign_cls.__name__
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
            secondary_sign_cls,
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
        print(f"[{label}] Failed to place sign: {e}")
        return False


def _place_main_secondary_junction_signs(
    env,
    row: dict,
    scenes_root: Path,
    secondary_sign_cls,
    distance_before_end: float = 20.0,
    show_model: bool = True,
    secondary_sign_for_arm=None,
) -> bool:
    """Place MainRoadSign on main arms and secondary plates on secondary arms.

    Shared by yield (2.4 / YieldSign) and stop (2.5 / StopSign). Keeps priority_bench
    outgoing_edge_ids exclusions that the standalone stop_sign bench lacked.

    ``secondary_sign_for_arm(arm) -> sign_cls`` optionally picks a different plate
    per secondary arm (stop X: ego arm StopSign, opposite YieldSign).
    """
    label = secondary_sign_cls.__name__
    layout = _layout_from_row(row, scenes_root)
    if layout is None:
        print(f"[JunctionSigns] No layout available, falling back to ego-only {label}")
        return _place_secondary_sign_on_spawn_lane(
            env,
            secondary_sign_cls,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )

    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return False

    _clear_signs(sign_mgr)

    main_arms = arms_for_road_class(layout, "main")
    secondary_arms = arms_for_road_class(layout, "secondary")
    if not main_arms or not secondary_arms:
        print(
            f"[JunctionSigns] Missing main/secondary arms, falling back to ego-only {label}"
        )
        return _place_secondary_sign_on_spawn_lane(
            env,
            secondary_sign_cls,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )

    main_lanes = []
    outgoing_edge_ids: set[str] = set()
    for arm in main_arms:
        main_lanes.extend(collect_lanes_for_keys(env, arm.get("lane_keys", [])))
        for out_edge in arm.get("outgoing_to") or arm.get("straight_to") or []:
            outgoing_edge_ids.add(str(out_edge))
    for arm in secondary_arms:
        for out_edge in arm.get("outgoing_to") or arm.get("straight_to") or []:
            outgoing_edge_ids.add(str(out_edge))
    # Never treat a monitored main approach as outgoing.
    main_approach_edges = {str(arm.get("edge_id")) for arm in main_arms if arm.get("edge_id")}
    outgoing_edge_ids -= main_approach_edges

    if not main_lanes:
        print(
            f"[JunctionSigns] Could not resolve main lanes, falling back to ego-only {label}"
        )
        return _place_secondary_sign_on_spawn_lane(
            env,
            secondary_sign_cls,
            distance_before_end=distance_before_end,
            show_model=show_model,
        )

    junction_id = layout.get("junction_id", "")
    placed_main = 0
    placed_by_cls: dict[str, int] = {}

    for arm in main_arms:
        edge_id = arm.get("edge_id", "")
        lane = resolve_sign_lane_for_edge(env, edge_id, arm.get("lane_keys", []))
        if lane is None:
            print(f"[JunctionSigns] Skipping main sign, lane not found for edge: {edge_id}")
            continue
        placement_long = sign_placement_long(lane, distance_before_end)
        try:
            sign = sign_mgr.add_sign(
                MainRoadSign,
                lane=lane,
                longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
                lateral_offset=lateral_offset_beside_lane(lane, placement_long),
                show_model=show_model,
                use_random_lane=False,
                intersection_name=junction_id,
            )
            if sign is not None:
                sign.is_priority_sign = False
            placed_main += 1
        except Exception as exc:
            print(f"[JunctionSigns] Failed MainRoadSign on edge {edge_id}: {exc}")

    for arm in secondary_arms:
        edge_id = arm.get("edge_id", "")
        lane = resolve_sign_lane_for_edge(env, edge_id, arm.get("lane_keys", []))
        if lane is None:
            print(f"[JunctionSigns] Skipping secondary sign, lane not found for edge: {edge_id}")
            continue
        arm_cls = secondary_sign_cls
        if secondary_sign_for_arm is not None:
            arm_cls = secondary_sign_for_arm(arm) or secondary_sign_cls
        arm_label = arm_cls.__name__
        placement_long = sign_placement_long(lane, distance_before_end)
        try:
            sign = sign_mgr.add_sign(
                arm_cls,
                lane=lane,
                longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
                lateral_offset=lateral_offset_beside_lane(lane, placement_long),
                show_model=show_model,
                use_random_lane=False,
                intersection_name=junction_id,
                main_road_lanes=main_lanes,
                outgoing_edge_ids=sorted(outgoing_edge_ids),
                auto_detect_main_roads=False,
            )
            if sign is not None:
                sign.is_priority_sign = False
            placed_by_cls[arm_label] = placed_by_cls.get(arm_label, 0) + 1
        except Exception as exc:
            print(f"[JunctionSigns] Failed {arm_label} on edge {edge_id}: {exc}")

    placed_secondary = sum(placed_by_cls.values())
    secondary_summary = ", ".join(
        f"{n} {name}(s)" for name, n in sorted(placed_by_cls.items())
    ) or f"0 {label}(s)"
    print(
        f"[JunctionSigns] Placed {placed_main} MainRoadSign(s) and "
        f"{secondary_summary} "
        f"at junction {junction_id} ({layout.get('shape')}), "
        f"shoulder offset={SIGN_SHOULDER_OFFSET_M}m"
    )
    return placed_main > 0 or placed_secondary > 0


def _place_yield_sign_on_spawn_lane(
    env, distance_before_end: float = 20.0, show_model: bool = True
) -> bool:
    return _place_secondary_sign_on_spawn_lane(
        env, YieldSign, distance_before_end=distance_before_end, show_model=show_model
    )


def _place_yield_junction_signs(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Place MainRoadSign on main arms and YieldSign on secondary arms."""
    return _place_main_secondary_junction_signs(
        env,
        row,
        scenes_root,
        YieldSign,
        distance_before_end=distance_before_end,
        show_model=show_model,
    )


def _place_stop_sign_on_spawn_lane(
    env, distance_before_end: float = 20.0, show_model: bool = True
) -> bool:
    return _place_secondary_sign_on_spawn_lane(
        env, StopSign, distance_before_end=distance_before_end, show_model=show_model
    )


def _stop_secondary_sign_for_arm(layout: dict, row: dict, arm: dict):
    """On X stop scenes: ego secondary → StopSign (2.5), opposite → YieldSign (2.4).

    T / single-secondary keep StopSign on every secondary arm. Priority logic is
    unchanged (main vs secondary); only the opposite plate differs visually.
    """
    if str(layout.get("shape") or "") != "X":
        return StopSign
    ego_edge = str(row.get("road_id") or "")
    edge_id = str(arm.get("edge_id") or "")
    if not ego_edge or not edge_id:
        return StopSign
    if edge_id == ego_edge:
        return StopSign
    opposite = straight_arm_edge_id(layout, ego_edge)
    if opposite is not None and edge_id == str(opposite):
        return YieldSign
    # Other non-ego secondary on X (should be the opposite in the usual layout).
    secondary_ids = {str(e) for e in (layout.get("secondary_edge_ids") or [])}
    if edge_id in secondary_ids and edge_id != ego_edge:
        return YieldSign
    return StopSign


def _place_stop_junction_signs(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Place MainRoadSign on main; StopSign on ego secondary; YieldSign opposite on X."""
    layout = _layout_from_row(row, scenes_root)

    def _for_arm(arm: dict):
        if layout is None:
            return StopSign
        return _stop_secondary_sign_for_arm(layout, row, arm)

    return _place_main_secondary_junction_signs(
        env,
        row,
        scenes_root,
        StopSign,
        distance_before_end=distance_before_end,
        show_model=show_model,
        secondary_sign_for_arm=_for_arm,
    )


def _secondary_road_sign_cls_for_main_arm(layout: dict, main_edge_id: str):
    """X: 2.3.1 on all main arms. T: 2.3.2 (stem on right) / 2.3.3 (stem on left)."""
    if str(layout.get("shape") or "") == "X":
        return SecondaryRoadSign

    secondary_ids = list(layout.get("secondary_edge_ids") or [])
    if len(secondary_ids) != 1:
        return SecondaryRoadSign

    side = secondary_side_from_main_arm(layout, main_edge_id, secondary_ids[0])
    if side == "right":
        return SecondaryRoadRightSign  # 2.3.2
    if side == "left":
        return SecondaryRoadLeftSign  # 2.3.3
    return SecondaryRoadSign


def _secondary_road_plate_label(sign_cls) -> str:
    if sign_cls is SecondaryRoadRightSign:
        return "2.3.2"
    if sign_cls is SecondaryRoadLeftSign:
        return "2.3.3"
    return "2.3.1"


def _place_secondary_road_junction_signs(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Place 2.3.x on main arms and YieldSign on secondary arms.

    X → SecondaryRoadSign (2.3.1) on every main approach.
    T → SecondaryRoadRightSign (2.3.2) and SecondaryRoadLeftSign (2.3.3) on the
    two main approaches (stem on right / left of that approach).
    Secondary approaches always get YieldSign (2.4), with priority_bench
    outgoing_edge_ids exclusions.
    """
    layout = _layout_from_row(row, scenes_root)
    if layout is None:
        print("[JunctionSigns] No layout available, falling back to ego-only YieldSign")
        return _place_yield_sign_on_spawn_lane(
            env, distance_before_end=distance_before_end, show_model=show_model
        )

    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return False

    _clear_signs(sign_mgr)

    main_arms = arms_for_road_class(layout, "main")
    secondary_arms = arms_for_road_class(layout, "secondary")
    if not main_arms or not secondary_arms:
        print(
            "[JunctionSigns] Missing main/secondary arms, falling back to ego-only YieldSign"
        )
        return _place_yield_sign_on_spawn_lane(
            env, distance_before_end=distance_before_end, show_model=show_model
        )

    main_lanes = []
    outgoing_edge_ids: set[str] = set()
    for arm in main_arms:
        main_lanes.extend(collect_lanes_for_keys(env, arm.get("lane_keys", [])))
        for out_edge in arm.get("outgoing_to") or arm.get("straight_to") or []:
            outgoing_edge_ids.add(str(out_edge))
    for arm in secondary_arms:
        for out_edge in arm.get("outgoing_to") or arm.get("straight_to") or []:
            outgoing_edge_ids.add(str(out_edge))
    main_approach_edges = {
        str(arm.get("edge_id")) for arm in main_arms if arm.get("edge_id")
    }
    outgoing_edge_ids -= main_approach_edges

    if not main_lanes:
        print(
            "[JunctionSigns] Could not resolve main lanes, falling back to ego-only YieldSign"
        )
        return _place_yield_sign_on_spawn_lane(
            env, distance_before_end=distance_before_end, show_model=show_model
        )

    junction_id = layout.get("junction_id", "")
    placed_by_plate: dict[str, int] = {}
    placed_yield = 0

    for arm in main_arms:
        edge_id = arm.get("edge_id", "")
        lane = resolve_sign_lane_for_edge(env, edge_id, arm.get("lane_keys", []))
        if lane is None:
            print(f"[JunctionSigns] Skipping 2.3 plate, lane not found for edge: {edge_id}")
            continue
        sign_cls = _secondary_road_sign_cls_for_main_arm(layout, edge_id)
        plate = _secondary_road_plate_label(sign_cls)
        placement_long = sign_placement_long(lane, distance_before_end)
        try:
            sign = sign_mgr.add_sign(
                sign_cls,
                lane=lane,
                longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
                lateral_offset=lateral_offset_beside_lane(lane, placement_long),
                show_model=show_model,
                use_random_lane=False,
                intersection_name=junction_id,
            )
            if sign is not None:
                sign.is_priority_sign = False
            placed_by_plate[plate] = placed_by_plate.get(plate, 0) + 1
            print(f"[JunctionSigns] Placed {plate} on main edge {edge_id}")
        except Exception as exc:
            print(f"[JunctionSigns] Failed {plate} on edge {edge_id}: {exc}")

    for arm in secondary_arms:
        edge_id = arm.get("edge_id", "")
        lane = resolve_sign_lane_for_edge(env, edge_id, arm.get("lane_keys", []))
        if lane is None:
            print(f"[JunctionSigns] Skipping yield sign, lane not found for edge: {edge_id}")
            continue
        placement_long = sign_placement_long(lane, distance_before_end)
        try:
            sign = sign_mgr.add_sign(
                YieldSign,
                lane=lane,
                longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
                lateral_offset=lateral_offset_beside_lane(lane, placement_long),
                show_model=show_model,
                use_random_lane=False,
                intersection_name=junction_id,
                main_road_lanes=main_lanes,
                outgoing_edge_ids=sorted(outgoing_edge_ids),
                auto_detect_main_roads=False,
            )
            if sign is not None:
                sign.is_priority_sign = False
            placed_yield += 1
        except Exception as exc:
            print(f"[JunctionSigns] Failed YieldSign on edge {edge_id}: {exc}")

    plate_summary = ", ".join(
        f"{n}×{code}" for code, n in sorted(placed_by_plate.items())
    ) or "0×2.3"
    print(
        f"[JunctionSigns] Placed {plate_summary} on main and {placed_yield} YieldSign(s) "
        f"at junction {junction_id} ({layout.get('shape')}), "
        f"shoulder offset={SIGN_SHOULDER_OFFSET_M}m"
    )
    return bool(placed_by_plate) or placed_yield > 0


def place_junction_signs(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Dispatch 2.1 / 2.3 / 2.4 / 2.5 placement."""
    if row_is_stop(row):
        return _place_stop_junction_signs(
            env, row, scenes_root,
            distance_before_end=distance_before_end, show_model=show_model,
        )
    if row_is_secondary_road(row):
        return _place_secondary_road_junction_signs(
            env, row, scenes_root,
            distance_before_end=distance_before_end, show_model=show_model,
        )
    if row_is_yield(row):
        return _place_yield_junction_signs(
            env, row, scenes_root,
            distance_before_end=distance_before_end, show_model=show_model,
        )
    return _place_equal_priority_main_signs(
        env, row, scenes_root,
        distance_before_end=distance_before_end, show_model=show_model,
    )


place_right_hand_yield_tracker = _place_right_hand_yield_tracker
