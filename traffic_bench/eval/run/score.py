"""Episode scoring: TTC, smoothness, route completion, aggregates."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from traffic_bench.eval.engine.sim.sign_eval import _ego_in_sign_zone
from traffic_bench.signs.priority_signs import MainRoadSign, YieldSign

def _is_ego_in_yield_zone(sign_mgr, vehicle) -> bool:
    """True when ego is in a YieldSign / RightHandYieldSign approach zone."""
    if sign_mgr is None or vehicle is None:
        return False
    for sign in getattr(sign_mgr, "signs", []) or []:
        if not isinstance(sign, YieldSign):
            continue
        if isinstance(sign, MainRoadSign):
            continue
        if _ego_in_sign_zone(sign, vehicle):
            return True
    return False


def _is_aux_in_main_zone(sign_mgr, aux_vehicles, ego_vehicle=None) -> bool:
    """True when any aux is in the main conflict zone (GIF / debug).

    Uses geometric main-zone presence so gated (not-yet-released) aux still
    count — matching what the camera shows. Yield decisions continue to ignore
    gated aux via ``_is_waiting_gated_aux``.
    """
    if sign_mgr is None or not aux_vehicles:
        return False
    yield_signs = [
        sign
        for sign in (getattr(sign_mgr, "signs", []) or [])
        if isinstance(sign, YieldSign) and not isinstance(sign, MainRoadSign)
    ]
    if not yield_signs:
        return False
    for aux in aux_vehicles:
        if aux is None:
            continue
        for sign in yield_signs:
            try:
                if sign.is_vehicle_on_main_road(aux):
                    return True
            except Exception:
                continue
    return False


def _unwrap_base_env(env):
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    return base_env


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _route_completion_percent(info: dict, reached_dest: bool) -> float:
    candidates = (
        "route_completion",
        "route_completion_rate",
        "route_completion_ratio",
        "route_completion_percentage",
    )
    for k in candidates:
        if k in info:
            v = _safe_float(info.get(k), 0.0)
            if v <= 1.0:
                v *= 100.0
            return max(0.0, min(100.0, v))
    return 100.0 if reached_dest else 0.0


def _route_length_m(info: dict) -> float | None:
    candidates = (
        "route_length_m",
        "route_length",
        "route_total_length",
        "route_distance_m",
        "episode_route_length",
    )
    for k in candidates:
        if k in info:
            try:
                v = float(info.get(k))
            except Exception:
                continue
            if math.isfinite(v) and v >= 0.0:
                return v
    return None


def _infraction_penalty(crashed: bool, out_of_road: bool, violations: int) -> float:
    p = 1.0
    if crashed:
        p *= 0.5
    if out_of_road:
        p *= 0.7
    if violations > 0:
        p *= (0.9 ** int(violations))
    return max(0.0, min(1.0, p))


def _nearby_speed_percentage(vehicle) -> float | None:
    try:
        nearby = vehicle.lidar.get_surrounding_objects(vehicle)
    except Exception:
        return None

    speeds = []
    for obj in nearby:
        if obj is vehicle:
            continue
        s = None
        if hasattr(obj, "speed_km_h"):
            s = _safe_float(getattr(obj, "speed_km_h"), 0.0)
        elif hasattr(obj, "speed"):
            s = _safe_float(getattr(obj, "speed"), 0.0) * 3.6
        if s is not None and s > 0.5:
            speeds.append(s)

    if not speeds:
        return None
    avg = float(np.mean(speeds))
    if avg <= 1e-3:
        return None

    ego = _safe_float(getattr(vehicle, "speed_km_h", 0.0), 0.0)
    pct = 100.0 * ego / avg
    if pct > 1000.0:
        return None
    return float(pct)


def _min_ttc_seconds(vehicle) -> float | None:
    try:
        nearby = vehicle.lidar.get_surrounding_objects(vehicle)
        ego_pos = np.asarray(vehicle.position, dtype=np.float64)
        ego_speed = _safe_float(getattr(vehicle, "speed", 0.0), 0.0)
        ego_heading = _safe_float(getattr(vehicle, "heading_theta", 0.0), 0.0)
    except Exception:
        return None

    ego_dir = np.array([math.cos(ego_heading), math.sin(ego_heading)], dtype=np.float64)
    ego_vel = ego_dir * ego_speed

    best = None
    for obj in nearby:
        if obj is vehicle:
            continue
        try:
            rel = np.asarray(obj.position, dtype=np.float64) - ego_pos
        except Exception:
            continue
        dist = float(np.linalg.norm(rel))
        if dist < 1e-3 or dist > 60.0:
            continue

        rel_along = float(np.dot(rel, ego_dir))
        if rel_along <= 0.0:
            continue

        obj_speed = _safe_float(getattr(obj, "speed", 0.0), 0.0)
        obj_heading = _safe_float(getattr(obj, "heading_theta", 0.0), ego_heading)
        obj_vel = np.array([math.cos(obj_heading), math.sin(obj_heading)], dtype=np.float64) * obj_speed
        rel_vel = ego_vel - obj_vel
        closing = float(np.dot(rel_vel, ego_dir))
        if closing <= 1e-3:
            continue
        ttc = rel_along / closing
        if ttc < 0.0:
            continue
        if best is None or ttc < best:
            best = ttc
    return best


def _compute_smoothness(step_vars: list[dict], segment_len: int = 20) -> dict:
    if not step_vars:
        return {
            "smoothness_ratio": 0.0,
            "smooth_segments": 0,
            "total_segments": 0,
            "frame_smooth_ratio": 0.0,
        }

    def _frame_ok(v: dict) -> bool:
        return (
            -4.05 <= v["long_acc"] <= 2.40
            and abs(v["lat_acc"]) <= 4.89
            and abs(v["yaw_rate"]) <= 0.95
            and abs(v["yaw_acc"]) <= 1.93
            and abs(v["long_jerk"]) <= 4.13
            and abs(v["jerk_mag"]) <= 8.37
        )

    frame_flags = [_frame_ok(v) for v in step_vars]
    frame_smooth_ratio = float(np.mean(frame_flags)) if frame_flags else 0.0

    total_segments = len(step_vars) // segment_len
    if total_segments <= 0:
        return {
            "smoothness_ratio": frame_smooth_ratio,
            "smooth_segments": int(sum(frame_flags)),
            "total_segments": len(frame_flags),
            "frame_smooth_ratio": frame_smooth_ratio,
        }

    smooth_segments = 0
    for i in range(total_segments):
        seg = frame_flags[i * segment_len : (i + 1) * segment_len]
        if seg and all(seg):
            smooth_segments += 1

    return {
        "smoothness_ratio": float(smooth_segments / total_segments),
        "smooth_segments": int(smooth_segments),
        "total_segments": int(total_segments),
        "frame_smooth_ratio": frame_smooth_ratio,
    }


def aggregate_results(results: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if not r.get("ok"):
            continue
        key = str(r.get("sign_type"))
        grouped[key].append(r)

    summary: dict[str, dict] = {}
    for sign, runs in sorted(grouped.items()):
        success_rate = float(np.mean([x["success"] for x in runs])) if runs else 0.0
        crash_rate = float(np.mean([x["crashed"] for x in runs])) if runs else 0.0
        avg_violations = float(np.mean([x["violations"] for x in runs])) if runs else 0.0
        avg_sign_viol = float(np.mean([x.get("sign_violations", 0) for x in runs])) if runs else 0.0
        avg_tl_viol = float(np.mean([x.get("traffic_light_violations", 0) for x in runs])) if runs else 0.0
        avg_cw_viol = float(np.mean([x.get("crosswalk_violations", 0) for x in runs])) if runs else 0.0
        avg_violations_event = float(np.mean([x.get("violations_event_count", 0) for x in runs])) if runs else 0.0
        violations_by_class_event_total: dict[str, int] = {}
        for x in runs:
            for cls, cnt in (x.get("violations_by_class_event") or {}).items():
                violations_by_class_event_total[cls] = (
                    violations_by_class_event_total.get(cls, 0) + int(cnt))
        violations_by_class_step_total: dict[str, int] = {}
        for x in runs:
            for cls, cnt in (x.get("violations_by_class_step") or {}).items():
                violations_by_class_step_total[cls] = (
                    violations_by_class_step_total.get(cls, 0) + int(cnt))
        avg_in_zone_steps = float(np.mean([x.get("in_zone_total_steps", 0) for x in runs])) if runs else 0.0
        in_zone_by_class_step_total: dict[str, int] = {}
        for x in runs:
            for cls, cnt in (x.get("in_zone_by_class_step") or {}).items():
                in_zone_by_class_step_total[cls] = (
                    in_zone_by_class_step_total.get(cls, 0) + int(cnt))
        avg_reward = float(np.mean([x["total_reward"] for x in runs])) if runs else 0.0
        avg_ds = float(np.mean([x.get("driving_score", 0.0) for x in runs])) if runs else 0.0
        avg_route = float(np.mean([x.get("route_completion_pct", 0.0) for x in runs])) if runs else 0.0
        avg_eff = float(np.mean([x.get("driving_efficiency", 0.0) for x in runs])) if runs else 0.0
        avg_smooth = float(np.mean([x.get("smoothness", 0.0) for x in runs])) if runs else 0.0
        avg_frame_smooth = float(np.mean([x.get("smoothness_frame_ratio", 0.0) for x in runs])) if runs else 0.0
        avg_hb = float(np.mean([x.get("hard_brake_count", 0) for x in runs])) if runs else 0.0
        avg_ha = float(np.mean([x.get("hard_accel_count", 0) for x in runs])) if runs else 0.0
        lane_offsets = [x.get("mean_abs_lane_offset") for x in runs if x.get("mean_abs_lane_offset") is not None]
        steer_deltas = [x.get("mean_abs_steer_delta") for x in runs if x.get("mean_abs_steer_delta") is not None]
        min_ttc_vals = [x.get("min_ttc_sec") for x in runs if x.get("min_ttc_sec") is not None]
        route_len_vals = [x.get("route_length_m") for x in runs if x.get("route_length_m") is not None]
        dist_vals = [x.get("distance_travelled_m") for x in runs if x.get("distance_travelled_m") is not None]
        summary[sign] = {
            "backend": "sumo",
            "sign_type": sign,
            "total_runs": len(runs),
            "success_rate": success_rate,
            "crash_rate": crash_rate,
            "average_violations": avg_violations,
            "average_sign_violations": avg_sign_viol,
            "average_traffic_light_violations": avg_tl_viol,
            "average_crosswalk_violations": avg_cw_viol,
            "average_violations_event_count": avg_violations_event,
            "violations_by_class_event_total": violations_by_class_event_total,
            "violations_by_class_step_total": violations_by_class_step_total,
            "average_in_zone_steps": avg_in_zone_steps,
            "in_zone_by_class_step_total": in_zone_by_class_step_total,
            "average_reward": avg_reward,
            "average_route_completion_pct": avg_route,
            "average_infraction_penalty": float(np.mean([x.get("infraction_penalty", 1.0) for x in runs])) if runs else 1.0,
            "average_driving_score": avg_ds,
            "average_driving_efficiency": avg_eff,
            "average_smoothness": avg_smooth,
            "average_smoothness_frame_ratio": avg_frame_smooth,
            "average_hard_brake_count": avg_hb,
            "average_hard_accel_count": avg_ha,
            "average_mean_abs_lane_offset": float(np.mean(lane_offsets)) if lane_offsets else None,
            "average_mean_abs_steer_delta": float(np.mean(steer_deltas)) if steer_deltas else None,
            "average_min_ttc_sec": float(np.mean(min_ttc_vals)) if min_ttc_vals else None,
            "average_route_length_m": float(np.mean(route_len_vals)) if route_len_vals else None,
            "average_distance_travelled_m": float(np.mean(dist_vals)) if dist_vals else None,
        }
    return summary


