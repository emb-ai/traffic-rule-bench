"""Sign violation / zone-of-effect / crash-attribution helpers."""
from __future__ import annotations


def _format_violation(sign, vehicle):
    sign_name = type(sign).__name__
    lane = getattr(sign, "lane", None)
    lane_idx = getattr(lane, "index", None)
    intersection = getattr(sign, "intersection_name", None)
    parts = [f"{sign_name}"]
    if intersection:
        parts.append(f"J:{intersection}")
    if lane_idx is not None:
        parts.append(f"L:{lane_idx}")
    try:
        if lane is not None:
            veh_long = float(lane.local_coordinates(vehicle.position)[0])
            dist = float(lane.length - veh_long)
            parts.append(f"d={dist:.1f}m")
    except Exception:
        pass
    return " | ".join(parts)


def _violation_bucket(sign_obj) -> str:
    name = type(sign_obj).__name__.lower()
    if "trafficlight" in name or "traffic_light" in name or "light" in name:
        return "traffic_light"
    if "crosswalk" in name or "pedestrian" in name or "zebra" in name:
        return "crosswalk"
    return "sign"


def _on_same_road(lane_a, lane_b) -> bool:
    """Cheap lane-equality check (PG tuple or SUMO `lane_<edge>_<num>`)."""
    idx_a = getattr(lane_a, "index", None)
    idx_b = getattr(lane_b, "index", None)
    if idx_a is None or idx_b is None:
        return False
    if isinstance(idx_a, str) and isinstance(idx_b, str):
        return idx_a.rsplit("_", 1)[0] == idx_b.rsplit("_", 1)[0]
    try:
        return idx_a[0] == idx_b[0] and idx_a[1] == idx_b[1]
    except (IndexError, TypeError):
        return False


# Lookahead for proximity-style zones (StopSign, TrafficLightSign etc.) — same
# value as the rule mixin's SPEED_SIGN_LOOKAHEAD so "in zone" agrees with
# "policy starts reacting".
_IN_ZONE_LOOKAHEAD_M = 50.0


def _ego_in_sign_zone(sign, vehicle) -> bool:
    """Heuristic: is ego inside (or approaching) this sign's zone of effect?

    Returns True if ego is on the same road segment as the sign AND within the
    sign's longitudinal zone (zone_start..zone_end) — or for proximity-style
    signs (Stop/TrafficLight) within `_IN_ZONE_LOOKAHEAD_M` metres before the
    stop line. False for cross-edge cases (cheap heuristic; matches the
    same-road branch of SignComplianceMixin handlers).

    Multi-edge zones (5.21/5.31 etc. configured with a `zone_edges` chain that
    spans several connected edges) ADD coverage via the sign's canonical
    `_in_multi_edge_zone`, so the metric counts the ego on downstream edges of
    the zone instead of clipping to the sign's own edge. This is monotonic: a
    True from the multi-edge check short-circuits to True, but a False/None
    falls through to the single-lane heuristic below (which also handles the
    approach lookahead on the sign's own edge) — so the multi-edge zone can only
    add in-zone steps, never remove the single-edge ones.
    """
    lane = getattr(sign, "lane", None)
    if lane is None:
        return False
    _multi = getattr(sign, "_in_multi_edge_zone", None)
    if callable(_multi):
        try:
            if _multi(vehicle) is True:
                return True
        except Exception:
            pass
    veh_lane = getattr(vehicle, "lane", None)
    if veh_lane is None:
        return False
    if not _on_same_road(veh_lane, lane):
        return False
    try:
        veh_long = float(lane.local_coordinates(vehicle.position)[0])
    except Exception:
        return False

    zone_start = getattr(sign, "zone_start", None)
    zone_end = getattr(sign, "zone_end", None)
    if zone_start is not None and zone_end is not None:
        if float(zone_start) <= veh_long <= float(zone_end):
            return True
        if veh_long < float(zone_start) and (float(zone_start) - veh_long) < _IN_ZONE_LOOKAHEAD_M:
            return True
        return False

    # Proximity-style sign (Stop, TrafficLight): use stop_line_position or
    # placement_long; "in zone" if approaching within lookahead, or just past it.
    anchor = (getattr(sign, "stop_line_position", None)
              or getattr(sign, "placement_long", None))
    if anchor is not None:
        anchor = float(anchor)
        dist = anchor - veh_long
        if 0 <= dist < _IN_ZONE_LOOKAHEAD_M:
            return True
        if -5.0 < dist < 0:   # just past sign — still in effect
            return True
        return False

    return False


def _extract_sign_info(env) -> list[dict]:
    """Snapshot signs currently placed in the env (after reset + sign placement).

    Mirrors expert_replay._extract_sign_info so replay.json sidecar carries the
    same `signs` field in both pipelines.
    """
    signs = []
    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return signs
    for s in sign_mgr.signs:
        lane = getattr(s, "lane", None)
        lane_index = list(getattr(lane, "index", ())) if lane is not None else None
        pos = None
        try:
            pos = [float(s.position[0]), float(s.position[1])]
        except Exception:
            pass
        signs.append({
            "sign_class": type(s).__name__,
            "lane_index": lane_index,
            "longitudinal_offset": float(getattr(s, "longitudinal_offset", 0.0)),
            "lateral_offset": float(getattr(s, "lateral_offset", 0.0)),
            "position_world": pos,
        })
    return signs


def _ego_at_fault_for_crash(ego, engine, contact_dist: float = 4.0) -> bool:
    """Heuristic ego-vs-NPC crash attribution (port of expert_replay.py:483).

    Returns True iff a colliding NPC is in ego's forward half AND ego speed > 0.5
    km/h — meaning ego drove into it. Otherwise NPC fault.
    """
    import math as _m
    try:
        ego_pos = ego.position
        ego_heading = ego.heading_theta
        ego_speed_kmh = float(getattr(ego, "speed_km_h", 0.0))
    except Exception:
        return True
    cos_h, sin_h = _m.cos(ego_heading), _m.sin(ego_heading)
    try:
        objs = engine.get_objects(lambda o: o is not ego).values()
    except Exception:
        return True
    from metadrive.component.vehicle.base_vehicle import BaseVehicle
    for obj in objs:
        if not isinstance(obj, BaseVehicle):
            continue
        try:
            dx = obj.position[0] - ego_pos[0]
            dy = obj.position[1] - ego_pos[1]
            dist = _m.hypot(dx, dy)
        except Exception:
            continue
        if dist > contact_dist:
            continue
        rel_x = cos_h * dx + sin_h * dy
        if rel_x > 0.0 and ego_speed_kmh > 0.5:
            return True
    return False
