"""Pure per-episode metric helpers (efficiency, TTC, smoothness, ...)."""
from __future__ import annotations

import math

import numpy as np


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
    # Bench2Drive/CARLA-like multiplicative penalty, with project-specific proxies.
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
    # Bench2Drive filters extreme spikes.
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
