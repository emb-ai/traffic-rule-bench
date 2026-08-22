"""Place NoTrafficSign (3.2) at the start of the forbidden lane."""

from __future__ import annotations

from pathlib import Path

from traffic_bench.eval.core.layout.junction_sign_placement import (
    lateral_offset_beside_lane,
    resolve_sign_lane_for_edge,
    sign_longitudinal_offset_from_start,
    sign_placement_long_from_start,
)
from traffic_bench.eval.core.manifest.manifest_config import (
    DEFAULT_DESTINATION_MAX_ALONG_M,
    DEFAULT_SIGN_DISTANCE_FROM_START,
)
from traffic_bench.eval.core.sumo.lane_keys import lane_edge_id
from traffic_bench.signs.no_traffic_sign import NoTrafficSign


def row_is_blocked_road(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "")
    sign_type = str(row.get("sign_type") or row.get("sign_family") or "")
    if code.replace("_", ".") == "3.2":
        return True
    return sign_type in {"blocked_road", "blocked"}


def place_blocked_road_sign(
    env,
    row: dict,
    scenes_root: Path,
    show_model: bool = True,
) -> bool:
    """Place NoTrafficSign (3.2) at the start of the forbidden lane."""
    del scenes_root
    try:
        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False

        sign_road_id = row.get("sign_road_id") or row.get("destination_edge_id")
        if not sign_road_id and row.get("destination_lane_id"):
            sign_road_id = lane_edge_id(str(row["destination_lane_id"]))
        if not sign_road_id:
            print("[NoTrafficSign] Missing sign_road_id / destination edge")
            return False

        lane_keys: list = []
        layout = row.get("junction_layout") or {}
        for arm in layout.get("arms", []) or []:
            if arm.get("edge_id") == sign_road_id:
                lane_keys = list(arm.get("lane_keys") or [])
                break

        lane = resolve_sign_lane_for_edge(env, str(sign_road_id), lane_keys)
        if lane is None:
            print(f"[NoTrafficSign] Lane not found for forbidden edge {sign_road_id}")
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
                f"[NoTrafficSign] Forbidden lane too short on {sign_road_id}: "
                f"{lane_len:.2f}m <= needed {needed:.2f}m (sign/dest cap)"
            )
            return False

        placement_long = sign_placement_long_from_start(lane, distance_from_start)
        longitudinal_offset = sign_longitudinal_offset_from_start(lane, distance_from_start)
        lateral = lateral_offset_beside_lane(lane, placement_long)

        sign_mgr.signs.clear()
        sign = sign_mgr.add_sign(
            NoTrafficSign,
            lane=lane,
            longitudinal_offset=longitudinal_offset,
            lateral_offset=lateral,
            show_model=show_model,
            use_random_lane=False,
        )
        print(
            f"[NoTrafficSign] Placed 3.2 on forbidden edge {sign_road_id} at "
            f"{distance_from_start:.2f}m from lane start "
            f"(long_offset={longitudinal_offset:.2f})"
        )
        return sign is not None
    except Exception as e:
        print(f"[NoTrafficSign] Failed to place: {e}")
        return False


def ego_compliant_stop_before_blocked_road(
    env,
    vehicle,
    *,
    max_dist_before_sign_m: float,
    speed_max_mps: float,
) -> bool:
    """True when ego is nearly stopped just before a 3.2 sign line."""
    if vehicle is None:
        return False
    try:
        speed = float(getattr(vehicle, "speed", 0.0) or 0.0)
    except Exception:
        return False
    if speed > float(speed_max_mps):
        return False

    sign_mgr = getattr(getattr(env, "engine", None), "traffic_sign_manager", None)
    if sign_mgr is None:
        return False

    max_dist = float(max_dist_before_sign_m)
    for sign in list(getattr(sign_mgr, "signs", None) or []):
        if not isinstance(sign, NoTrafficSign):
            continue
        sign_lane = getattr(sign, "lane", None)
        if sign_lane is None:
            continue
        sign_long = float(
            getattr(sign, "sign_line_position", getattr(sign, "placement_long", 0.0))
            or 0.0
        )
        try:
            veh_long = float(sign_lane.local_coordinates(vehicle.position)[0])
        except Exception:
            continue
        dist_to_line = sign_long - veh_long
        if -0.25 < dist_to_line <= max_dist:
            return True
    return False
