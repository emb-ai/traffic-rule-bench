"""Parse and build MetaDrive/SUMO lane keys (handles '#' in edge ids)."""

from __future__ import annotations


def parse_lane_key(lane_key: str) -> tuple[str, int]:
    """Split 'lane_<edge_id>_<lane_num>' into edge id and lane index."""
    raw = lane_key[5:] if lane_key.startswith("lane_") else lane_key
    if "_" not in raw:
        return raw, 0
    edge_id, lane_num_s = raw.rsplit("_", 1)
    try:
        return edge_id, int(lane_num_s)
    except ValueError:
        return raw, 0


def make_lane_key(edge_id: str, lane_num: int) -> str:
    return f"lane_{edge_id}_{lane_num}"


def lane_edge_id(lane_key: str) -> str:
    return parse_lane_key(lane_key)[0]


def lane_num_from_key(lane_key: str) -> int:
    return parse_lane_key(lane_key)[1]
