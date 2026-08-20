"""Sign placement helpers for overtaking_sign (3.20)."""

from __future__ import annotations

from typing import List, Optional

from .lane_keys import lane_num_from_key, make_lane_key

SIGN_SHOULDER_OFFSET_M = 1.8


def sign_placement_long_from_start(lane, distance_from_start: float) -> float:
    lane_length = float(getattr(lane, "length", 0.0) or 0.0)
    dist = max(0.1, float(distance_from_start))
    if lane_length <= 0.2:
        return 0.1
    return max(0.1, min(dist, lane_length - 0.1))


def sign_longitudinal_offset_from_start(lane, distance_from_start: float) -> float:
    return sign_placement_long_from_start(lane, distance_from_start) - lane.length


def lateral_offset_beside_lane(
    lane,
    placement_long: float,
    shoulder_m: float = SIGN_SHOULDER_OFFSET_M,
) -> float:
    return lane.width_at(placement_long) / 2.0 + shoulder_m


def resolve_sign_lane_for_edge(env, edge_id: str, lane_keys: Optional[List[str]] = None):
    road_network = env.engine.current_map.road_network
    try:
        lane_key = road_network.find_rightmost_lane_by_road_id(str(edge_id))
        # None means the road id is not in this network. Fall through to the
        # scene's own lane list below instead of leaning on get_lane(None) to
        # raise: that worked, but only because the except clause is broad.
        if lane_key is not None:
            lane = road_network.get_lane(lane_key)
            if lane is not None:
                return lane
    except Exception:
        pass
    for key in lane_keys or [make_lane_key(str(edge_id), 0)]:
        try:
            lane = road_network.get_lane(key)
            if lane is not None:
                return lane
        except Exception:
            continue
    return None
