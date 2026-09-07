"""Place 5.19 plate and reconstruct zebra geometry on segment maps."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from traffic_bench.eval.engine.map.junction_sign_placement import (
    lateral_offset_beside_lane,
    resolve_sign_lane_for_edge,
    sign_longitudinal_offset,
    sign_placement_long,
)
from traffic_bench.eval.engine.map.lane_keys import (
    clamp_lane_key_to_graph,
    lane_edge_id,
    make_lane_key,
)
from traffic_bench.signs.crosswalk.plate import PedestrianCrossingSign
from traffic_bench.signs.crosswalk.yield_rule import PedestrianYieldRule

# Small pad past the outermost driving-lane edges. Full sidewalk width is NOT
# added: MetaDrive does not draw sidewalks as road gray, so a 2 m pad reads as
# the zebra spilling into empty space.
_DEFAULT_CURB_PAD_M = 0.4
# Max empty lateral gap (m) between ego carriageway and another parallel road
# before we refuse to extend the zebra across it. Narrow painted medians /
# back-to-back curbs stay in; divided dual carriageways stay out.
_DEFAULT_OPPOSITE_GAP_M = 3.0


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


def _resolve_graph_lane(road_network, graph, key: str):
    """Return a lane object with ``position``/``length`` for a graph key."""
    if road_network is not None and hasattr(road_network, "get_lane"):
        try:
            lane = road_network.get_lane(key)
            if lane is not None and hasattr(lane, "position") and hasattr(lane, "length"):
                return lane
        except Exception:
            pass
    raw = graph.get(key) if isinstance(graph, dict) else None
    if isinstance(raw, dict):
        raw = raw.get("lane", raw)
    if raw is not None and hasattr(raw, "position") and hasattr(raw, "length"):
        return raw
    return None


def _same_edge_lane_keys(graph, edge_id: str) -> list[str]:
    """Keys whose parsed edge id equals ``edge_id`` (no substring / reverse false hits)."""
    if not edge_id or not isinstance(graph, dict):
        return []
    out: list[str] = []
    for key in graph.keys():
        sk = str(key)
        if not sk.startswith("lane_"):
            continue
        if sk[5:].startswith(":"):
            continue
        if lane_edge_id(sk) == edge_id:
            out.append(sk)
    return out


def _peer_lane_keys(road_network, graph, seed_key: str) -> list[str]:
    """Walk left/right neighbors from ``seed_key`` (carriageway peers only)."""
    if not seed_key or seed_key not in graph:
        return [seed_key] if seed_key else []
    if road_network is not None and hasattr(road_network, "get_peer_lanes_from_index"):
        try:
            peers = road_network.get_peer_lanes_from_index(seed_key)
            keys = []
            for lane in peers or []:
                idx = getattr(lane, "index", None)
                if isinstance(idx, str) and idx in graph:
                    keys.append(idx)
            if keys:
                return keys
        except Exception:
            pass

    seen: set[str] = set()
    order: list[str] = []
    q: deque[str] = deque([seed_key])
    while q:
        key = q.popleft()
        if key in seen or key not in graph:
            continue
        seen.add(key)
        order.append(key)
        info = graph.get(key)
        for nb in list(getattr(info, "left_lanes", None) or []) + list(
            getattr(info, "right_lanes", None) or []
        ):
            if isinstance(nb, str) and nb not in seen:
                q.append(nb)
    return order


def _is_crosswalk_type(feat_type) -> bool:
    if feat_type is None:
        return False
    text = str(feat_type)
    return text == "CROSSWALK" or text.endswith("CROSSWALK")


def _clip_poly_to_lateral_band(
    polygon,
    *,
    origin: np.ndarray,
    forward: np.ndarray,
    lateral: np.ndarray,
    min_lat: float,
    max_lat: float,
) -> np.ndarray:
    """Clamp polygon vertices to ``[min_lat, max_lat]`` in the road frame."""
    poly = np.asarray(polygon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] < 2:
        return poly
    out = np.empty((poly.shape[0], 2), dtype=np.float64)
    for i, raw in enumerate(poly[:, :2]):
        delta = raw - origin
        along = float(np.dot(delta, forward))
        lat = float(np.dot(delta, lateral))
        lat = min(max(lat, min_lat), max_lat)
        out[i] = origin + forward * along + lateral * lat
    return out


def _clip_cw_junction_fills(
    current_map,
    *,
    origin: np.ndarray,
    forward: np.ndarray,
    lateral: np.ndarray,
    min_lat: float,
    max_lat: float,
    crosswalk_node_id: str | None,
) -> int:
    """Clip split-junction fill polygons to the carriageway — no curb overhang.

    These fills come from netconvert after the mid-block split. Leaving them
    unclipped paints a gray stub past the curb; deleting them leaves holes in
    top-down. Clipping is visual-only (``map_data``); the road network is untouched.
    """
    node_token = str(crosswalk_node_id or "").strip()
    tokens = {"cw_node"}
    if node_token:
        tokens.add(node_token)

    def _is_cw_fill(feat_id: str) -> bool:
        fid = str(feat_id)
        if not any(tok and tok in fid for tok in tokens):
            return False
        return fid.startswith("junction_") or fid.startswith("lane_:")

    n_clipped = 0
    datasets = []
    for block in getattr(current_map, "blocks", None) or []:
        md = getattr(block, "map_data", None)
        if isinstance(md, dict):
            datasets.append(md)
    md = getattr(current_map, "map_data", None)
    if isinstance(md, dict):
        datasets.append(md)

    for map_data in datasets:
        for key, data in list(map_data.items()):
            if not _is_cw_fill(key):
                continue
            poly = (data or {}).get("polygon")
            if poly is None:
                continue
            clipped = _clip_poly_to_lateral_band(
                poly,
                origin=origin,
                forward=forward,
                lateral=lateral,
                min_lat=min_lat,
                max_lat=max_lat,
            )
            data = dict(data)
            data["polygon"] = clipped
            map_data[key] = data
            n_clipped += 1
    return n_clipped


def _replace_map_crosswalks(current_map, feat: dict, *, crosswalk_node_id: str | None = None) -> None:
    """Keep only our synthetic curb-to-curb zebra in ``map.crosswalks``.

    Strip SUMO ``CROSSWALK`` entries from ``map_data`` so they are not drawn as
    a second solid underlay. Junction fills stay (clipped separately) so the
    split does not leave white holes in top-down.
    """
    del crosswalk_node_id  # junction clip uses the same id via a dedicated helper
    current_map.crosswalks = {"segment_cw_5_19": feat}

    for block in getattr(current_map, "blocks", None) or []:
        if hasattr(block, "crosswalks"):
            block.crosswalks = {"segment_cw_5_19": feat}
        map_data = getattr(block, "map_data", None)
        if isinstance(map_data, dict):
            stale = [
                key
                for key, data in map_data.items()
                if _is_crosswalk_type((data or {}).get("type"))
            ]
            for key in stale:
                map_data.pop(key, None)

    map_data = getattr(current_map, "map_data", None)
    if isinstance(map_data, dict):
        stale = [
            key
            for key, data in map_data.items()
            if _is_crosswalk_type((data or {}).get("type"))
        ]
        for key in stale:
            map_data.pop(key, None)


def _sample_lane_lat_extent(
    lane,
    *,
    approach_center: np.ndarray,
    forward: np.ndarray,
    lateral: np.ndarray,
    max_along_m: float,
) -> tuple[float, float, float, float, float] | None:
    """Return ``(along, center_lat, lat_min, lat_max, align)`` at the zebra sample, or None."""
    try:
        length = float(lane.length)
        if length <= 1.0:
            return None
        if hasattr(lane, "local_coordinates"):
            long, _lat = lane.local_coordinates(approach_center)
            s = float(np.clip(long, 0.5, max(0.5, length - 0.5)))
        else:
            s = max(0.5, min(length - 0.5, length - 2.0))
        pt = np.asarray(lane.position(s, 0.0), dtype=np.float64)[:2]
        heading = float(lane.heading_theta_at(s))
        width = float(lane.width_at(s))
        if width <= 0.3:
            return None
        left = np.asarray(lane.position(s, -width / 2.0), dtype=np.float64)[:2]
        right = np.asarray(lane.position(s, width / 2.0), dtype=np.float64)[:2]
    except Exception:
        return None
    delta = pt - approach_center
    along = float(np.dot(delta, forward))
    if abs(along) > max_along_m:
        return None
    align = math.cos(heading) * float(forward[0]) + math.sin(heading) * float(forward[1])
    center_lat = float(np.dot(delta, lateral))
    lat_a = float(np.dot(left - approach_center, lateral))
    lat_b = float(np.dot(right - approach_center, lateral))
    return along, center_lat, min(lat_a, lat_b), max(lat_a, lat_b), align


def _lateral_gap(band_min: float, band_max: float, lat_min: float, lat_max: float) -> float:
    """Empty gap between two lateral intervals (0 if they touch/overlap)."""
    if lat_max < band_min:
        return band_min - lat_max
    if lat_min > band_max:
        return lat_min - band_max
    return 0.0


def _nearby_carriageway_lane_keys(
    road_network,
    graph,
    *,
    approach_center: np.ndarray,
    forward: np.ndarray,
    seed_keys: list[str],
    max_along_m: float = 18.0,
    max_gap_m: float = _DEFAULT_OPPOSITE_GAP_M,
    min_align: float = 0.7,
) -> list[str]:
    """Grow the ego carriageway to adjacent parallel / opposite lanes only.

    Opposite carriageways often use a different OSM edge id (not just ``-edge``).
    We still discover them geometrically, but only when the empty lateral gap to
    the current band is small — divided dual carriageways / distant parallel
    roads are left out so the zebra stays on the ego roadway.
    """
    lateral = np.array([-forward[1], forward[0]], dtype=np.float64)
    # Always keep seed keys (ego carriageway); geometric growth is additive.
    out: list[str] = []
    seen: set[str] = set()
    for key in seed_keys:
        sk = str(key)
        if sk and sk not in seen:
            seen.add(sk)
            out.append(sk)

    band_min: float | None = None
    band_max: float | None = None
    for sk in list(out):
        lane = _resolve_graph_lane(road_network, graph, sk)
        if lane is None:
            continue
        sample = _sample_lane_lat_extent(
            lane,
            approach_center=approach_center,
            forward=forward,
            lateral=lateral,
            max_along_m=max_along_m,
        )
        if sample is None:
            continue
        _along, _clat, lat_min, lat_max, align = sample
        if abs(align) < min_align:
            continue
        if band_min is None or band_max is None:
            band_min, band_max = lat_min, lat_max
        else:
            band_min = min(band_min, lat_min)
            band_max = max(band_max, lat_max)

    if band_min is None or band_max is None:
        return out

    def _try_add(sk: str) -> bool:
        nonlocal band_min, band_max
        if sk in seen or not sk.startswith("lane_") or sk[5:].startswith(":"):
            return False
        lane = _resolve_graph_lane(road_network, graph, sk)
        if lane is None:
            return False
        sample = _sample_lane_lat_extent(
            lane,
            approach_center=approach_center,
            forward=forward,
            lateral=lateral,
            max_along_m=max_along_m,
        )
        if sample is None:
            return False
        _along, _clat, lat_min, lat_max, align = sample
        if abs(align) < min_align:
            return False
        assert band_min is not None and band_max is not None
        if _lateral_gap(band_min, band_max, lat_min, lat_max) > max_gap_m:
            return False
        seen.add(sk)
        out.append(sk)
        band_min = min(band_min, lat_min)
        band_max = max(band_max, lat_max)
        return True

    # Grow until no adjacent parallel/anti-parallel lane remains outside the band.
    changed = True
    while changed:
        changed = False
        for key in list(graph.keys()) if isinstance(graph, dict) else []:
            if _try_add(str(key)):
                changed = True
    return out


def _invalidate_topdown_background(env) -> None:
    """Force the next top-down render to rebake the map (crosswalks changed)."""
    # Recreate TopDownRenderer on next render() so the baked background picks up
    # the replaced zebra (background is otherwise immutable after first paint).
    if hasattr(env, "top_down_renderer"):
        try:
            env.top_down_renderer = None
        except Exception:
            pass


def install_segment_crosswalk_geometry(env, row: dict) -> bool:
    """Build a curb-to-curb zebra perpendicular to the road at the inject split."""
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
    approach = _resolve_graph_lane(road_network, graph, approach_key) if approach_key else None
    if approach is None:
        print(f"[CrosswalkGeom] approach lane missing: {approach_key}")
        return False

    try:
        lane_len = float(getattr(approach, "length", 0.0) or 0.0)
        # No-split: zebra at meta mark along the continuous edge.
        # Legacy inject: a few metres before the split (lanes stay parallel).
        try:
            zebra_s = float(row.get("crosswalk_position_m") or 0.0)
        except (TypeError, ValueError):
            zebra_s = 0.0
        if zebra_s > 0.0 and not row.get("crosswalk_node_id"):
            sample_s = max(0.5, min(lane_len - 0.5, zebra_s))
        else:
            sample_s = max(0.5, min(lane_len - 0.5, lane_len - 2.0))
        heading = float(approach.heading_theta_at(sample_s))
        approach_center = np.asarray(approach.position(sample_s, 0.0), dtype=np.float64)[:2]
        approach_width = float(approach.width_at(sample_s))
    except Exception as exc:
        print(f"[CrosswalkGeom] Could not sample approach lane: {exc}")
        return False

    forward = np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)
    # Left-hand normal of travel; MetaDrive lane.position lateral+ is right-handed,
    # so we project true left/right edge points rather than assuming ±width/2 sign.
    lateral = np.array([-forward[1], forward[0]], dtype=np.float64)

    # Carriageway peers on the approach stub (exact edge id), falling back to
    # neighbor walk if the edge listing is incomplete.
    approach_keys = _same_edge_lane_keys(graph, edge_id)
    if approach_key:
        peers = _peer_lane_keys(road_network, graph, approach_key)
        for key in peers:
            if key not in approach_keys:
                approach_keys.append(key)
    if not approach_keys and approach_key:
        approach_keys = [approach_key]

    # Also include same-stub lanes from crossed_edge_ids (exact match only).
    for raw in row.get("crossed_edge_ids") or ():
        eid = str(raw or "").strip()
        if eid and eid != edge_id:
            for key in _same_edge_lane_keys(graph, eid):
                if key not in approach_keys:
                    approach_keys.append(key)

    # Adjacent opposite / parallel carriageway only (gap-gated; not across medians).
    max_gap = float(row.get("crosswalk_opposite_gap_m") or _DEFAULT_OPPOSITE_GAP_M)
    approach_keys = _nearby_carriageway_lane_keys(
        road_network,
        graph,
        approach_center=approach_center,
        forward=forward,
        seed_keys=approach_keys,
        max_gap_m=max_gap,
    )

    lat_hits: list[float] = []
    used = 0
    for key in approach_keys:
        lane = _resolve_graph_lane(road_network, graph, key)
        if lane is None:
            continue
        sample = _sample_lane_lat_extent(
            lane,
            approach_center=approach_center,
            forward=forward,
            lateral=lateral,
            max_along_m=25.0,
        )
        if sample is None:
            continue
        _along, _clat, lat_min, lat_max, _align = sample
        lat_hits.append(lat_min)
        lat_hits.append(lat_max)
        used += 1

    curb_pad = float(row.get("crosswalk_curb_pad_m") or _DEFAULT_CURB_PAD_M)
    if lat_hits:
        band_min = min(lat_hits) - curb_pad
        band_max = max(lat_hits) + curb_pad
    else:
        half = max(1.5, approach_width / 2.0) + curb_pad
        band_min, band_max = -half, half
        used = 1

    # Re-center on the carriageway midline (not the ego-lane centerline).
    mid_lat = 0.5 * (band_min + band_max)
    center = approach_center + lateral * mid_lat
    min_lat = band_min - mid_lat
    max_lat = band_max - mid_lat

    half_thick = max(1.75, float(row.get("crosswalk_width_m") or 4.0) / 2.0)
    # Quad order expected by MetaDrive zebra renderer when returned as 4 points:
    # a→b along the short (along-road) axis, a→d along the long (across-road) axis.
    a = center - forward * half_thick + lateral * min_lat
    b = center + forward * half_thick + lateral * min_lat
    c = center + forward * half_thick + lateral * max_lat
    d = center - forward * half_thick + lateral * max_lat
    polygon = np.asarray([a, b, c, d], dtype=np.float64)

    # Drop SUMO-imported crossing polygons — they are often degenerate (narrow
    # shoulder stubs) and draw beside our curb-to-curb zebra. Clear every place
    # MetaDrive may still read them from (map.crosswalks, block.crosswalks,
    # block.map_data / map.map_data).
    feat = {
        "type": "CROSSWALK",
        "polygon": polygon,
        "walk_direction": lateral.tolist(),
    }
    node_id = str(row.get("crosswalk_node_id") or "") or None
    _replace_map_crosswalks(current_map, feat, crosswalk_node_id=node_id)
    n_clipped = 0
    if node_id:
        n_clipped = _clip_cw_junction_fills(
            current_map,
            origin=approach_center,
            forward=forward,
            lateral=lateral,
            min_lat=band_min,
            max_lat=band_max,
            crosswalk_node_id=node_id,
        )
    _invalidate_topdown_background(env)

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
        f"lanes={used} clipped_fills={n_clipped} ped_specs={n_specs}"
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
        from traffic_bench.eval.engine.map.junction_sign_placement import (
            sign_longitudinal_offset_from_start,
            sign_placement_long_from_start,
        )

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

        # No-split: place relative to zebra mark (from lane start).
        # Legacy inject: place relative to approach stub end.
        from_start = row.get("sign_distance_from_start")
        if from_start is None and row.get("crosswalk_position_m") and not row.get(
            "crosswalk_node_id"
        ):
            try:
                from_start = max(
                    1.0,
                    float(row["crosswalk_position_m"]) - float(distance_before_end),
                )
            except (TypeError, ValueError):
                from_start = None

        if from_start is not None:
            placement_long = sign_placement_long_from_start(lane, float(from_start))
            long_offset = sign_longitudinal_offset_from_start(lane, float(from_start))
            where = f"{float(from_start):.1f}m from start"
        else:
            placement_long = sign_placement_long(lane, distance_before_end)
            long_offset = sign_longitudinal_offset(lane, distance_before_end)
            where = f"{distance_before_end:.1f}m before end"

        sign = sign_mgr.add_sign(
            PedestrianCrossingSign,
            lane=lane,
            longitudinal_offset=long_offset,
            lateral_offset=lateral_offset_beside_lane(lane, placement_long),
            show_model=show_model,
            use_random_lane=False,
        )
        if sign is not None:
            sign.is_priority_sign = False
            print(
                f"[PedestrianCrossingSign] Placed 5.19 on edge "
                f"{getattr(lane, 'index', edge_id)} ({where}), "
                f"yield_rules={sum(type(r).__name__ == 'PedestrianYieldRule' for r in sign_mgr.rules)}"
            )
        return sign is not None
    except Exception as e:
        print(f"[PedestrianCrossingSign] Failed to place sign: {e}")
        return False
