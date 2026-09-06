"""Place the reserved-lane plate (5.14.1/2, 5.11.1/2) on the rightmost lane."""

from __future__ import annotations

from traffic_bench.eval.engine.map.junction_sign_placement import resolve_layout_lane
from traffic_bench.signs.extra.restricted_lane import (
    BikeLaneRoadSign,
    BikeLaneSign,
    BusLaneRoadSign,
    BusLaneSign,
)

_RESTRICTED_LANE_CODES = {"5.14.1", "5.14.2", "5.11.1", "5.11.2"}
_SIGN_CLS_BY_CODE = {
    "5.14.1": BusLaneSign,
    "5.14.2": BikeLaneSign,
    "5.11.1": BusLaneRoadSign,
    "5.11.2": BikeLaneRoadSign,
}


def row_is_restricted_lane(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "").replace("_", ".")
    sign_type = str(row.get("sign_type") or row.get("sign_family") or "")
    if bool(row.get("place_restricted_lane_sign")):
        return True
    return code in _RESTRICTED_LANE_CODES or sign_type == "restricted_lane"


def place_restricted_lane_signs(env, row: dict, show_model: bool = True) -> bool:
    """Plate on ``road_id`` lane ``sign_lane_index`` at ``sign_s``; zone ``zone_length_m`` after it."""
    try:
        vehicle = env.agent
        if vehicle is None or vehicle.lane is None:
            return False
        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False
        sign_mgr.signs.clear()

        pdd_code = str(row.get("pdd_code") or row.get("sign_code") or "5.14.1")
        sign_cls = _SIGN_CLS_BY_CODE.get(pdd_code, BusLaneSign)
        road_id = str(row.get("road_id") or "")
        lane_index = int(row.get("sign_lane_index", row.get("restricted_lane_index", 0)) or 0)
        sign_s = float(row.get("sign_s", 60.0))
        zone_m = row.get("zone_length_m")

        lane = None
        lane_key = ""
        if road_id:
            lane_key = f"{road_id}_{lane_index}"
            lane = resolve_layout_lane(env, lane_key)
        if lane is None:
            lane = vehicle.lane

        placement_long = max(0.1, min(sign_s, float(lane.length) - 1.0))
        kwargs = dict(
            lane=lane,
            # RestrictedLaneSign reads the offset from the lane END (like 4.6).
            longitudinal_offset=placement_long - float(lane.length),
            lateral_offset=0,
            show_model=show_model,
            use_random_lane=False,
        )
        if zone_m is not None:
            kwargs["zone_length"] = float(zone_m)
        sign = sign_mgr.add_sign(sign_cls, **kwargs)
        if sign is None:
            print(f"[RestrictedLaneSign] Failed to place {pdd_code}")
            return False
        sign.is_priority_sign = False
        print(
            f"[RestrictedLaneSign] Placed {pdd_code} ({sign_cls.__name__}) on lane "
            f"{getattr(lane, 'index', lane_key)} at s={sign_s:.1f}m "
            f"(zone [{sign.zone_start:.1f}, {sign.zone_end:.1f}])"
        )
        return True
    except Exception as e:
        print(f"[RestrictedLaneSign] Failed to place sign: {e}")
        import traceback

        traceback.print_exc()
        return False
