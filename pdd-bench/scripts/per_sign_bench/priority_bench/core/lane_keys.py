"""Parse and build MetaDrive/SUMO lane keys (handles '#' in edge ids)."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence


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


def pick_lane_key_on_edge(
    edge_id: str,
    preferred_lane_num: int,
    lane_keys_by_edge: Optional[Mapping[str, Sequence[str]]] = None,
) -> str:
    """Choose a lane key on ``edge_id``, clamping to lanes that actually exist.

    Outgoing destination edges are often absent from junction *arms* (arms are
    incoming only). Callers must pass ``lane_keys_by_edge`` covering all net
    edges; otherwise this falls back to an unclamped synthetic key.
    """
    keys = list((lane_keys_by_edge or {}).get(edge_id) or [])
    if not keys:
        return make_lane_key(edge_id, preferred_lane_num)
    for key in keys:
        if lane_num_from_key(key) == preferred_lane_num:
            return key
    idx = min(max(int(preferred_lane_num), 0), len(keys) - 1)
    return keys[idx]


def clamp_lane_key_to_graph(lane_key: Optional[str], graph) -> Optional[str]:
    """If ``lane_key`` is missing from MetaDrive ``graph``, clamp lane index."""
    if not lane_key or graph is None:
        return lane_key
    if lane_key in graph:
        return lane_key

    edge_id, lane_num = parse_lane_key(lane_key)
    for n in range(int(lane_num), -1, -1):
        candidate = make_lane_key(edge_id, n)
        if candidate in graph:
            return candidate

    prefix = f"lane_{edge_id}_"
    matches = [
        key
        for key in graph
        if isinstance(key, str) and key.startswith(prefix)
    ]
    if not matches:
        return lane_key
    matches.sort(key=lane_num_from_key)
    return matches[0]
