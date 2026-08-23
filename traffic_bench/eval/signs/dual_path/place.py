"""Place 4.1 / 5.7 / 3.18 / 3.1 plates in a live episode."""

from __future__ import annotations

from pathlib import Path

from traffic_bench.eval.engine.map.junction_priority_layout import (
    JunctionLayoutError,
    build_junction_priority_layout,
)
from traffic_bench.eval.engine.map.junction_sign_placement import (
    SIGN_SHOULDER_OFFSET_M,
    lateral_offset_beside_lane,
    resolve_sign_lane_for_edge,
    sign_longitudinal_offset,
    sign_longitudinal_offset_from_start,
    sign_placement_long,
    sign_placement_long_from_start,
)
from traffic_bench.eval.engine.expand.manifest_config import (
    DEFAULT_DESTINATION_MAX_ALONG_M,
    DEFAULT_SIGN_DISTANCE_FROM_START,
)
from traffic_bench.eval.signs.dual_path.nav import (
    resolve_row_background_excluded_edges,
)
from traffic_bench.eval.engine.map.lane_keys import lane_edge_id
from traffic_bench.eval.signs.dual_path.spec import get_spec, resolve_sign_class

_ONE_WAY_TYPES = frozenset({"one_way", "one_way_right", "one_way_left"})
_DIRECTION_TYPES = frozenset({
    "direction",
    "direction_straight",
    "direction_right",
    "direction_left",
    "direction_straight_right",
    "direction_straight_left",
    "direction_left_right",
})
_NO_TURN_TYPES = frozenset({"no_turn", "no_turn_right", "no_turn_left"})
_NO_ENTRY_TYPES = frozenset({"no_entry"})

_LOG = {
    "one_way": "OneWaySign",
    "direction": "LaneAllowedDirectionSign",
    "no_turn": "NoTurnSign",
    "no_entry": "NoEntrySign",
}


def _row_codes(row: dict) -> tuple[str, str]:
    code = str(row.get("pdd_code") or row.get("sign_code") or "")
    sign_type = str(row.get("sign_type") or row.get("sign_family") or "")
    return code, sign_type


def _code_in_family(code: str, family: str) -> bool:
    if not code:
        return False
    try:
        return get_spec(code).family == family
    except ValueError:
        return False


def row_is_one_way(row: dict) -> bool:
    code, sign_type = _row_codes(row)
    return _code_in_family(code, "one_way") or sign_type in _ONE_WAY_TYPES


def row_is_direction(row: dict) -> bool:
    code, sign_type = _row_codes(row)
    return _code_in_family(code, "direction") or sign_type in _DIRECTION_TYPES


def row_is_no_turn(row: dict) -> bool:
    code, sign_type = _row_codes(row)
    return _code_in_family(code, "no_turn") or sign_type in _NO_TURN_TYPES


def row_is_no_entry(row: dict) -> bool:
    code, sign_type = _row_codes(row)
    return _code_in_family(code, "no_entry") or sign_type in _NO_ENTRY_TYPES


COMPLIANT_NAV_POLICIES = frozenset({
    "comprehensive_rule_expert",
    "rule_compliant",
    "carl_rule",
    "plant2_rule",
})


def resolve_row_for_policy(row: dict, policy_type: str) -> dict:
    """Pick baseline vs compliant dest (and along-cap) for the active policy."""
    if not row_uses_dual_path_nav(row):
        return row
    out = dict(row)
    use_compliant = policy_type in COMPLIANT_NAV_POLICIES
    if use_compliant:
        dest = row.get("compliant_destination_lane_id") or row.get("destination_lane_id")
        along = row.get("compliant_destination_max_along_m")
        if along is None:
            along = row.get("destination_max_along_m")
    else:
        dest = row.get("baseline_destination_lane_id") or row.get("destination_lane_id")
        along = row.get("baseline_destination_max_along_m")
        if along is None:
            along = row.get("destination_max_along_m")
    if dest:
        out["destination_lane_id"] = dest
        edge = lane_edge_id(str(dest))
        if edge:
            out["destination_edge_id"] = edge
    if along is not None:
        out["destination_max_along_m"] = float(along)
    elif out.get("destination_max_along_m") is None:
        out["destination_max_along_m"] = 1e9
    return out


def row_uses_dual_path_nav(row: dict) -> bool:
    """5.7 / 4.1 / 3.18 / 3.1: truncated dests; only rule-compliant policies replan."""
    family = str(row.get("sign_family") or "")
    if family in {"one_way", "direction", "no_turn", "no_entry"}:
        return True
    return (
        row_is_one_way(row)
        or row_is_direction(row)
        or row_is_no_turn(row)
        or row_is_no_entry(row)
    )


def resolve_row_sign_code(row: dict, *, default: str | None = None) -> str:
    family = str(row.get("sign_family") or row.get("sign_type") or "") or None
    for key in ("_sign_code", "sign_code", "pdd_code"):
        raw = row.get(key)
        if not raw:
            continue
        try:
            return get_spec(str(raw).strip()).sign_code
        except ValueError:
            continue
    if family:
        try:
            return get_spec(None, family=family).sign_code
        except ValueError:
            pass
    if default:
        return default
    return get_spec(None, family="direction").sign_code


def _clear_sign_manager(sign_mgr) -> None:
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
    try:
        return build_junction_priority_layout(full_path, mode="main_main").to_dict()
    except JunctionLayoutError as exc:
        print(f"[JunctionLayout] Failed to build layout: {exc}")
        return None


def _place_on_spawn_lane(
    env,
    pdd_code: str,
    *,
    distance_before_end: float,
    show_model: bool,
    log_tag: str,
) -> bool:
    try:
        vehicle = env.agent
        if vehicle is None or vehicle.lane is None:
            return False
        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False
        _clear_sign_manager(sign_mgr)
        lane = vehicle.lane
        sign_cls = resolve_sign_class(pdd_code)
        placement_long = sign_placement_long(lane, distance_before_end)
        sign = sign_mgr.add_sign(
            sign_cls,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=show_model,
            use_random_lane=False,
        )
        return sign is not None
    except Exception as e:
        print(f"[{log_tag}] Failed to place {pdd_code}: {e}")
        return False


def _place_on_ego_approach(
    env,
    row: dict,
    scenes_root: Path,
    *,
    distance_before_end: float,
    show_model: bool,
) -> bool:
    spec = get_spec(resolve_row_sign_code(row))
    pdd_code = spec.sign_code
    log_tag = _LOG[spec.family]
    layout = _layout_from_row(row, scenes_root)
    if layout is None:
        print(f"[{log_tag}] No layout; ego-only {pdd_code}")
        return _place_on_spawn_lane(
            env,
            pdd_code,
            distance_before_end=distance_before_end,
            show_model=show_model,
            log_tag=log_tag,
        )

    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return False
    _clear_sign_manager(sign_mgr)

    ego_edge = row.get("road_id")
    if not ego_edge:
        return _place_on_spawn_lane(
            env,
            pdd_code,
            distance_before_end=distance_before_end,
            show_model=show_model,
            log_tag=log_tag,
        )

    arm = next(
        (a for a in layout.get("arms", []) if a.get("edge_id") == ego_edge),
        None,
    )
    lane = resolve_sign_lane_for_edge(
        env, str(ego_edge), (arm or {}).get("lane_keys", [])
    )
    if lane is None:
        print(f"[{log_tag}] Lane not found for ego edge {ego_edge}; ego-only fallback")
        return _place_on_spawn_lane(
            env,
            pdd_code,
            distance_before_end=distance_before_end,
            show_model=show_model,
            log_tag=log_tag,
        )

    sign_cls = resolve_sign_class(pdd_code)
    placement_long = sign_placement_long(lane, distance_before_end)
    try:
        sign = sign_mgr.add_sign(
            sign_cls,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=show_model,
            use_random_lane=False,
            intersection_name=str(
                row.get("junction_id") or layout.get("junction_id") or ""
            ),
        )
        if spec.family == "one_way":
            net_path = env.config.get("map_name") or ""
            forbidden_edges = resolve_row_background_excluded_edges(row, net_path)
            if sign is not None and forbidden_edges:
                try:
                    sign.one_way_forbidden_edges = frozenset(str(e) for e in forbidden_edges)
                except Exception:
                    pass
        extra = ""
        if spec.forbidden_dir:
            extra = f"forbidden={spec.forbidden_dir}, "
        print(
            f"[{log_tag}] Placed {pdd_code} ({spec.title}) on {ego_edge}, "
            f"junction={row.get('junction_id') or layout.get('junction_id')}, "
            f"{extra}allowed={sorted(spec.allowed_dirs)}, "
            f"shoulder offset={SIGN_SHOULDER_OFFSET_M}m"
        )
        return sign is not None
    except Exception as exc:
        print(f"[{log_tag}] Failed {pdd_code} on {ego_edge}: {exc}")
        return False


def _place_no_entry_on_forbidden_exit(
    env,
    row: dict,
    *,
    show_model: bool,
) -> bool:
    """Place NoEntrySign (3.1) at the start of the short forbidden exit."""
    pdd_code = resolve_row_sign_code(row, default="3.1")
    try:
        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False

        sign_road_id = (
            row.get("sign_road_id")
            or row.get("baseline_first_exit")
            or row.get("destination_edge_id")
        )
        if not sign_road_id and row.get("destination_lane_id"):
            sign_road_id = lane_edge_id(str(row["destination_lane_id"]))
        if not sign_road_id:
            print("[NoEntrySign] Missing sign_road_id / baseline_first_exit")
            return False

        lane_keys: list = []
        layout = row.get("junction_layout") or {}
        for arm in layout.get("arms", []) or []:
            if arm.get("edge_id") == sign_road_id:
                lane_keys = list(arm.get("lane_keys") or [])
                break

        lane = resolve_sign_lane_for_edge(env, str(sign_road_id), lane_keys)
        if lane is None:
            print(f"[NoEntrySign] Lane not found for forbidden edge {sign_road_id}")
            return False

        distance_from_start = float(
            row.get("sign_distance_from_start", DEFAULT_SIGN_DISTANCE_FROM_START)
            or DEFAULT_SIGN_DISTANCE_FROM_START
        )
        raw_cap = row.get("destination_max_along_m")
        try:
            dest_cap = float(
                DEFAULT_DESTINATION_MAX_ALONG_M if raw_cap is None else raw_cap
            )
        except (TypeError, ValueError):
            dest_cap = float(DEFAULT_DESTINATION_MAX_ALONG_M)
        needed = max(distance_from_start + 1.0, dest_cap + 5.0)
        lane_len = float(getattr(lane, "length", 0.0) or 0.0)
        if lane_len <= needed:
            print(
                f"[NoEntrySign] Forbidden lane too short on {sign_road_id}: "
                f"{lane_len:.2f}m <= needed {needed:.2f}m (sign/dest cap)"
            )
            return False

        _clear_sign_manager(sign_mgr)
        sign_cls = resolve_sign_class(pdd_code)
        placement_long = sign_placement_long_from_start(lane, distance_from_start)
        longitudinal_offset = sign_longitudinal_offset_from_start(lane, distance_from_start)
        lateral = lateral_offset_beside_lane(lane, placement_long)

        sign = sign_mgr.add_sign(
            sign_cls,
            lane=lane,
            longitudinal_offset=longitudinal_offset,
            lateral_offset=lateral,
            show_model=show_model,
            use_random_lane=False,
        )
        spec = get_spec(pdd_code)
        print(
            f"[NoEntrySign] Placed {pdd_code} ({spec.title}) on forbidden edge "
            f"{sign_road_id} at {distance_from_start:.2f}m from lane start "
            f"(long_offset={longitudinal_offset:.2f})"
        )
        return sign is not None
    except Exception as e:
        print(f"[NoEntrySign] Failed to place: {e}")
        return False


def place_dual_path_signs(
    env,
    row: dict,
    scenes_root: Path,
    distance_before_end: float = 20.0,
    show_model: bool = True,
) -> bool:
    """Place the plate for this dual-path row (approach or forbidden-exit)."""
    if row_is_no_entry(row):
        return _place_no_entry_on_forbidden_exit(env, row, show_model=show_model)
    return _place_on_ego_approach(
        env,
        row,
        scenes_root,
        distance_before_end=distance_before_end,
        show_model=show_model,
    )
