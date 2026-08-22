"""Place DetourSign (4.2.x) on the obstacle lane."""

from __future__ import annotations

from traffic_bench.eval.engine.map.junction_sign_placement import resolve_layout_lane
from traffic_bench.signs.detour_sign import DetourEitherSign, DetourLeftSign, DetourRightSign


def row_is_detour(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "").replace("_", ".")
    sign_type = str(row.get("sign_type") or row.get("sign_family") or "")
    if bool(row.get("place_detour_sign")):
        return True
    return code.startswith("4.2") or sign_type == "detour"


def place_detour_signs(env, row: dict, show_model: bool = True) -> bool:
    """Place a DetourSign (4.2.x) on the obstacle lane at sign_s from the row."""
    try:
        vehicle = env.agent
        if vehicle is None or vehicle.lane is None:
            return False

        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False

        sign_mgr.signs.clear()

        pdd_code = str(row.get("pdd_code") or row.get("detour_code") or "4.2.1")
        sign_cls_map = {
            "4.2.1": DetourRightSign,
            "4.2.2": DetourLeftSign,
            "4.2.3": DetourEitherSign,
        }
        sign_cls = sign_cls_map.get(pdd_code, DetourRightSign)

        road_id = str(row.get("road_id") or "")
        sign_lane_index = int(row.get("sign_lane_index", 0))
        sign_s = float(row.get("sign_s", 60.0))

        lane = None
        lane_key = ""
        if road_id:
            lane_key = f"{road_id}_{sign_lane_index}"
            lane = resolve_layout_lane(env, lane_key)

        if lane is None:
            lane = vehicle.lane

        placement_long = max(0.0, min(sign_s, lane.length - 1.0))
        longitudinal_offset = placement_long - lane.length

        sign = sign_mgr.add_sign(
            sign_cls,
            lane=lane,
            longitudinal_offset=longitudinal_offset,
            lateral_offset=0,
            show_model=show_model,
            use_random_lane=False,
        )
        if sign is not None:
            sign.is_priority_sign = False
            print(
                f"[DetourSign] Placed {pdd_code} on lane "
                f"{getattr(lane, 'index', lane_key)} "
                f"at s={sign_s:.1f}m (zone [{sign.zone_start:.1f}, {sign.zone_end:.1f}])"
            )
        return sign is not None
    except Exception as e:
        print(f"[DetourSign] Failed to place sign: {e}")
        import traceback
        traceback.print_exc()
        return False
