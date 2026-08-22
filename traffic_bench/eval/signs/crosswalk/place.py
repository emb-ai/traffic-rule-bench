"""Place 5.19 plate and reconstruct zebra geometry on segment maps."""

from __future__ import annotations

import math

import numpy as np

from traffic_bench.eval.core.layout.junction_sign_placement import (
    lateral_offset_beside_lane,
    resolve_sign_lane_for_edge,
    sign_longitudinal_offset,
    sign_placement_long,
)
from traffic_bench.eval.core.sumo.lane_keys import clamp_lane_key_to_graph, make_lane_key
from traffic_bench.signs.pedestrian_crossing_sign import PedestrianCrossingSign
from traffic_bench.signs.pedestrian_yield_rule import PedestrianYieldRule


def row_is_crosswalk(row: dict) -> bool:
    code = str(row.get("pdd_code") or row.get("sign_code") or "").replace("_", ".")
    sign_type = str(row.get("sign_type") or row.get("sign_family") or "")
    if bool(row.get("place_crosswalk_sign")):
        return True
    return code.startswith("5.19") or sign_type == "crosswalk"


def _clear_signs(sign_mgr) -> None:
    sign_mgr.signs.clear()


def ensure_pedestrian_yield_rule(env) -> None:
    """Re-register PedestrianYieldRule if sign placement left the manager empty."""
    engine = getattr(env, "engine", None)
    sign_mgr = getattr(engine, "traffic_sign_manager", None) if engine is not None else None
    if sign_mgr is None:
        return
    if any(type(rule).__name__ == "PedestrianYieldRule" for rule in getattr(sign_mgr, "rules", []) or []):
        return
    ped_cfg = engine.global_config.get("pedestrian_manager", {})
    if hasattr(ped_cfg, "get_dict"):
        ped_cfg = ped_cfg.get_dict()
    ped_cfg = dict(ped_cfg or {})
    sign_mgr.add_rule(
        PedestrianYieldRule(
            yield_distance=float(ped_cfg.get("yield_distance", 12.0)),
            yield_speed_kmh=float(ped_cfg.get("yield_speed_kmh", 8.0)),
            no_stop_before_m=float(ped_cfg.get("no_stop_before_crosswalk_m", 0.0)),
            no_stop_speed_kmh=float(ped_cfg.get("no_stop_speed_kmh", 1.0)),
            no_stop_min_duration_s=float(ped_cfg.get("no_stop_min_duration_s", 1.0)),
        )
    )
    print("[PedestrianYieldRule] re-registered after 5.19 placement")


def _iter_sumo_graph_lanes(graph) -> list:
    if not isinstance(graph, dict):
        return []
    out = []
    for key, val in graph.items():
        lane = val
        if isinstance(val, dict):
            lane = val.get("lane", val)
        if hasattr(lane, "position") and hasattr(lane, "length"):
            out.append((str(key), lane))
    return out


def install_segment_crosswalk_geometry(env, row: dict) -> bool:
    """Build an OSM-style zebra from driving lanes at the injected split."""
    if not row_is_crosswalk(row):
        return False
    current_map = getattr(getattr(env, "engine", None), "current_map", None)
    if current_map is None:
        return False
    graph = getattr(getattr(current_map, "road_network", None), "graph", {}) or {}
    road_network = getattr(current_map, "road_network", None)

    edge_id = str(row.get("road_id") or "")
    lane_num = int(row.get("spawn_lane_num", 0) or 0)
    approach_key = make_lane_key(edge_id, lane_num) if edge_id else ""
    approach_key = clamp_lane_key_to_graph(approach_key, graph) if approach_key else None
    approach = None
    if approach_key and road_network is not None and hasattr(road_network, "get_lane"):
        try:
            approach = road_network.get_lane(approach_key)
        except Exception:
            approach = None
    if approach is None and approach_key:
        approach = graph.get(approach_key)
    if approach is None:
        print(f"[CrosswalkGeom] approach lane missing: {approach_key}")
        return False

    try:
        lane_len = float(getattr(approach, "length", 0.0) or 0.0)
        sample_s = max(0.5, lane_len - 0.5)
        heading = float(approach.heading_theta_at(sample_s))
        center = np.asarray(approach.position(sample_s, 0.0), dtype=np.float64)[:2]
    except Exception as exc:
        print(f"[CrosswalkGeom] Could not sample approach lane: {exc}")
        return False

    forward = np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)
    lateral = np.array([-forward[1], forward[0]], dtype=np.float64)

    lat_hits: list[float] = []
    for key, lane in _iter_sumo_graph_lanes(graph):
        raw = key[5:] if key.startswith("lane_") else key
        if raw.startswith(":"):
            continue
        try:
            length = float(lane.length)
            s_end = max(0.5, length - 0.5)
            s_start = min(0.5, max(0.1, length * 0.05))
            p_end = np.asarray(lane.position(s_end, 0.0), dtype=np.float64)[:2]
            p_start = np.asarray(lane.position(s_start, 0.0), dtype=np.float64)[:2]
            d_end = float(np.linalg.norm(p_end - center))
            d_start = float(np.linalg.norm(p_start - center))
            if d_end <= 14.0:
                s, pt = s_end, p_end
            elif d_start <= 14.0:
                s, pt = s_start, p_start
            else:
                continue
            width = float(lane.width_at(s))
        except Exception:
            continue
        lat0 = float(np.dot(pt - center, lateral))
        lat_hits.append(lat0 - width / 2.0)
        lat_hits.append(lat0 + width / 2.0)

    if lat_hits:
        min_lat = min(lat_hits) - 0.6
        max_lat = max(lat_hits) + 0.6
    else:
        min_lat, max_lat = -5.0, 5.0

    half_thick = max(1.75, float(row.get("crosswalk_width_m") or 4.0) / 2.0)
    corners = [
        center - forward * half_thick + lateral * min_lat,
        center - forward * half_thick + lateral * max_lat,
        center + forward * half_thick + lateral * max_lat,
        center + forward * half_thick + lateral * min_lat,
    ]
    polygon_pts = []
    for i, corner in enumerate(corners):
        nxt = corners[(i + 1) % 4]
        polygon_pts.append(corner)
        polygon_pts.append(0.5 * (corner + nxt))
    polygon = np.asarray(polygon_pts, dtype=np.float64)

    existing = dict(getattr(current_map, "crosswalks", {}) or {})
    cleaned = {}
    for key, feat in existing.items():
        poly = np.asarray((feat or {}).get("polygon", []), dtype=np.float64)
        if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] < 2:
            continue
        span = float(np.linalg.norm(poly.max(axis=0)[:2] - poly.min(axis=0)[:2]))
        if span >= 2.0:
            cleaned[key] = feat
    cleaned["segment_cw_5_19"] = {
        "type": "CROSSWALK",
        "polygon": polygon,
        "walk_direction": lateral.tolist(),
    }
    current_map.crosswalks = cleaned

    ped_mgr = getattr(env.engine, "pedestrian_manager", None)
    n_specs = 0
    if ped_mgr is not None and hasattr(ped_mgr, "_collect_crosswalk_specs"):
        ped_mgr._crosswalks = ped_mgr._collect_crosswalk_specs()
        n_specs = len(getattr(ped_mgr, "_crosswalks", {}) or {})
        if (
            n_specs > 0
            and str(getattr(ped_mgr, "spawn_mode", "") or "").lower() == "ego_proximity"
            and int(getattr(ped_mgr, "_ego_spawns_scheduled", 0) or 0) == 0
            and hasattr(ped_mgr, "_schedule_track")
        ):
            preferred = [k for k in ped_mgr._crosswalks if k == "segment_cw_5_19"]
            rest = [k for k in ped_mgr._crosswalks if k != "segment_cw_5_19"]
            for cw_id in preferred + rest:
                if not ped_mgr._schedule_track(cw_id, on_crosswalk=True, immediate=True):
                    continue
                ped_mgr._ego_spawns_scheduled = 1
                ped_mgr._ego_trigger_crosswalk_id = cw_id
                if hasattr(ped_mgr, "_spawn_due_tracks"):
                    ped_mgr._spawn_due_tracks()
                print(f"[CrosswalkGeom] primed pedestrian on {cw_id}")
                break
    span_m = float(max_lat - min_lat)
    print(
        f"[CrosswalkGeom] zebra span={span_m:.1f}m thick={half_thick * 2:.1f}m "
        f"ped_specs={n_specs}"
    )
    return True


def place_crosswalk_signs(
    env,
    row: dict,
    distance_before_end: float = 15.0,
    show_model: bool = True,
) -> bool:
    """Place PedestrianCrossingSign (5.19 icon) beside the ego approach lane."""
    try:
        vehicle = env.agent
        if vehicle is None or vehicle.lane is None:
            return False

        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return False

        _clear_signs(sign_mgr)
        ensure_pedestrian_yield_rule(env)

        edge_id = row.get("road_id") or row.get("depart_edge_id")
        lane = None
        if edge_id:
            lane = resolve_sign_lane_for_edge(env, str(edge_id), [])
        if lane is None:
            lane = vehicle.lane

        placement_long = sign_placement_long(lane, distance_before_end)
        sign = sign_mgr.add_sign(
            PedestrianCrossingSign,
            lane=lane,
            longitudinal_offset=sign_longitudinal_offset(lane, distance_before_end),
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=show_model,
            use_random_lane=False,
        )
        if sign is not None:
            sign.is_priority_sign = False
            print(
                f"[PedestrianCrossingSign] Placed 5.19 on edge "
                f"{getattr(lane, 'index', edge_id)} "
                f"({distance_before_end:.1f}m before end), "
                f"yield_rules={sum(type(r).__name__ == 'PedestrianYieldRule' for r in sign_mgr.rules)}"
            )
        return sign is not None
    except Exception as e:
        print(f"[PedestrianCrossingSign] Failed to place sign: {e}")
        return False
