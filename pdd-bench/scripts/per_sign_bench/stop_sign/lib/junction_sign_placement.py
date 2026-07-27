"""Helpers for placing stop / main-road signs at junction approaches."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .lane_keys import lane_num_from_key, make_lane_key

# Distance from lane centerline edge to sign anchor (meters beyond pavement).
SIGN_SHOULDER_OFFSET_M = 1.8


def pick_rightmost_lane_key(lane_keys: List[str]) -> Optional[str]:
    """Pick the rightmost SUMO lane key on an edge (lane index 0 = rightmost)."""
    if not lane_keys:
        return None
    return min(lane_keys, key=lane_num_from_key)


def arms_for_road_class(layout: dict, road_class: str) -> List[dict]:
    return [arm for arm in layout.get("arms", []) if arm.get("road_class") == road_class]


def sign_placement_long(lane, distance_before_end: float) -> float:
    lane_length = lane.length
    if lane_length < distance_before_end + 5.0:
        return max(lane_length - 5.0, 0.1)
    return max(lane_length - distance_before_end, 0.1)


def sign_longitudinal_offset(lane, distance_before_end: float) -> float:
    """MetaDrive offset from lane end (negative = before intersection)."""
    return sign_placement_long(lane, distance_before_end) - lane.length


def lateral_offset_beside_lane(
    lane,
    placement_long: float,
    shoulder_m: float = SIGN_SHOULDER_OFFSET_M,
) -> float:
    """Positive lateral offset placing the sign on the right shoulder, off the pavement."""
    return lane.width_at(placement_long) / 2.0 + shoulder_m


def resolve_layout_lane(env, lane_key: str):
    road_network = env.engine.current_map.road_network
    try:
        return road_network.get_lane(lane_key)
    except Exception:
        lane_info = getattr(road_network, "graph", {}).get(lane_key)
        if lane_info is not None and hasattr(lane_info, "lane"):
            return lane_info.lane
    return None


def resolve_sign_lane_for_edge(env, edge_id: str, lane_keys: List[str]):
    """Resolve the rightmost driving lane on a road edge for sign placement."""
    road_network = env.engine.current_map.road_network
    try:
        lane_key = road_network.find_rightmost_lane_by_road_id(str(edge_id))
        lane = road_network.get_lane(lane_key)
        if lane is not None:
            return lane
    except Exception:
        pass

    lane_key = pick_rightmost_lane_key(lane_keys)
    if lane_key is None:
        return None
    return resolve_layout_lane(env, lane_key)


def collect_lanes_for_keys(env, lane_keys: List[str]) -> List[Any]:
    lanes = []
    for lane_key in lane_keys:
        lane = resolve_layout_lane(env, lane_key)
        if lane is not None:
            lanes.append(lane)
    return lanes
