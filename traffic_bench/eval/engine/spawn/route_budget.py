"""Cap ego route length for junction-family manifest rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from traffic_bench.eval.engine.map.lane_keys import lane_num_from_key
from traffic_bench.eval.engine.map.sumo_utils import load_vehicle_route_index
from traffic_bench.eval.signs.dual_path.budget import (
    load_sumo_edge_lengths,
    truncate_edge_path,
)


def apply_route_budget(
    entry: Dict[str, Any],
    *,
    net_path: Path,
    max_path_length_m: float,
    spawn_distance_before_end: float,
) -> Dict[str, Any]:
    """Trim ego route so total travel from spawn ≤ ``max_path_length_m``.

    Sets ``destination_edge_id``, ``destination_lane_id``, and
    ``destination_max_along_m`` on the trimmed final edge (same mechanism as
    roundabout / dual-path budgets).
    """
    ego_edge = entry.get("road_id") or entry.get("ego_edge_id")
    ego_lane = entry.get("spawn_lane_num")
    dest_edge = entry.get("destination_edge_id")
    dest_lane_key = entry.get("destination_lane_id")
    if not ego_edge or ego_lane is None or not dest_edge:
        return entry

    route_index = load_vehicle_route_index(net_path)
    real_path = route_index.find_real_edge_path(
        str(ego_edge), int(ego_lane), str(dest_edge)
    )
    if not real_path:
        return entry

    post = list(real_path)
    if post and post[0] == str(ego_edge):
        post = post[1:]
    if not post:
        post = [str(dest_edge)]

    edge_lengths = load_sumo_edge_lengths(net_path)
    ego_rem = max(0.0, float(spawn_distance_before_end))
    spawn_len = entry.get("spawn_lane_length") or entry.get("approach_lane_length_m")
    if spawn_len is not None:
        try:
            ego_rem = min(ego_rem, max(0.0, float(spawn_len) - 1.0))
        except (TypeError, ValueError):
            pass

    budget = max(5.0, float(max_path_length_m))
    budget_after_ego = max(5.0, budget - ego_rem)

    dest_lane_num = int(ego_lane)
    if dest_lane_key:
        try:
            dest_lane_num = int(lane_num_from_key(str(dest_lane_key)))
        except (TypeError, ValueError):
            pass

    trimmed = truncate_edge_path(
        post,
        edge_lengths=edge_lengths,
        budget_after_ego_m=budget_after_ego,
        dest_lane_num=dest_lane_num,
    )
    if trimmed is None:
        return entry

    out = dict(entry)
    out["destination_edge_id"] = trimmed.dest_edge_id
    out["destination_lane_id"] = trimmed.dest_lane_id
    out["destination_max_along_m"] = trimmed.destination_max_along_m
    out["max_path_length_m"] = budget
    out["route_length_m"] = float(ego_rem + trimmed.length_m)
    out["route_truncated"] = trimmed.truncated
    if trimmed.truncated:
        out["full_destination_edge_id"] = str(dest_edge)
        if dest_lane_key:
            out["full_destination_lane_id"] = str(dest_lane_key)
    return out


# Back-compat alias (junction expand and older call sites).
apply_junction_route_budget = apply_route_budget
