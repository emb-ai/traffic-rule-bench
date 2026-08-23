"""Place start (and paired end) speed signs at sign_s / s_end."""

from __future__ import annotations

from traffic_bench.eval.engine.map.junction_sign_placement import resolve_layout_lane
from traffic_bench.signs.speed.end_of_zone import EndOfSpeedLimitSign, EndOfZoneSpeedLimitSign
from traffic_bench.signs.speed.min_speed import MinimumSpeedLimitSign
from traffic_bench.signs.speed.residential import (
    EndOfResidentialZoneSign,
    ResidentialZoneSign,
)
from traffic_bench.signs.speed.limit import SpeedLimitSign
from traffic_bench.signs.speed.zone import ZoneSpeedLimitSign

_SPEED_CODES = {"3.24", "4.6", "5.21", "5.31"}


def row_is_speed(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "").replace("_", ".")
    sign_type = str(row.get("sign_type") or row.get("sign_family") or "")
    if bool(row.get("place_speed_sign")):
        return True
    return code in _SPEED_CODES or sign_type == "speed"


def place_speed_signs(env, row: dict, show_model: bool = True) -> bool:
    """Place start (and paired end) speed signs at sign_s / s_end from the row."""
    try:
        vehicle = env.agent
        if vehicle is None or vehicle.lane is None:
            return False
        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False
        sign_mgr.signs.clear()

        pdd_code = str(row.get("pdd_code") or row.get("sign_code") or "3.24")
        start_cls_map = {
            "3.24": SpeedLimitSign,
            "4.6": MinimumSpeedLimitSign,
            "5.21": ResidentialZoneSign,
            "5.31": ZoneSpeedLimitSign,
        }
        end_cls_map = {
            "3.25": EndOfSpeedLimitSign,
            "5.22": EndOfResidentialZoneSign,
            "5.32": EndOfZoneSpeedLimitSign,
        }
        start_cls = start_cls_map.get(pdd_code, SpeedLimitSign)
        v_target = float(row.get("v_target_kmh") or 0.0)
        road_id = str(row.get("road_id") or "")
        lane_num = int(row.get("sign_lane_index", row.get("spawn_lane_num", 0)) or 0)
        sign_s = float(row.get("sign_s", 60.0))

        lane = None
        if road_id:
            lane = resolve_layout_lane(env, f"{road_id}_{lane_num}")
        if lane is None:
            lane = vehicle.lane

        placement_long = max(0.1, min(sign_s, float(lane.length) - 1.0))
        start_kwargs = dict(
            lane=lane,
            longitudinal_offset=placement_long,
            lateral_offset=0,
            show_model=show_model,
            use_random_lane=False,
        )
        if start_cls is SpeedLimitSign or start_cls is ZoneSpeedLimitSign:
            if v_target > 0:
                start_kwargs["speed_limit_override"] = v_target
        elif start_cls is MinimumSpeedLimitSign:
            if v_target > 0:
                start_kwargs["min_speed_override"] = v_target

        start_sign = sign_mgr.add_sign(start_cls, **start_kwargs)
        if start_sign is None:
            print(f"[SpeedSign] Failed to place start {pdd_code}")
            return False
        start_sign.is_priority_sign = False

        end_code = str(row.get("sign_type_end") or "")
        s_end = row.get("s_end")
        if end_code and s_end is not None:
            end_cls = end_cls_map.get(end_code)
            if end_cls is not None:
                end_long = max(placement_long + 1.0, min(float(s_end), float(lane.length) - 0.5))
                end_kwargs = dict(
                    lane=lane,
                    longitudinal_offset=end_long,
                    lateral_offset=0,
                    show_model=show_model,
                    use_random_lane=False,
                )
                if end_cls is EndOfSpeedLimitSign or end_cls is EndOfZoneSpeedLimitSign:
                    if v_target > 0:
                        end_kwargs["speed_limit"] = v_target
                end_sign = sign_mgr.add_sign(end_cls, **end_kwargs)
                if end_sign is not None:
                    end_sign.is_priority_sign = False
            try:
                sign_mgr.build_zones()
            except Exception as exc:
                print(f"[SpeedSign] build_zones failed: {exc}")

        print(
            f"[SpeedSign] Placed {pdd_code}@{placement_long:.1f}m "
            f"v_target={v_target:.0f} end={end_code or '-'} "
            f"s_end={float(s_end) if s_end is not None else float('nan'):.1f}"
        )
        return True
    except Exception as e:
        print(f"[SpeedSign] Failed to place sign: {e}")
        import traceback
        traceback.print_exc()
        return False
