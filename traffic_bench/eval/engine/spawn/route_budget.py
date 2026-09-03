"""Cap / measure ego route length from spawn to destination."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from traffic_bench.eval.engine.map.lane_keys import lane_num_from_key, make_lane_key
from traffic_bench.eval.engine.map.sumo_utils import load_vehicle_route_index
from traffic_bench.eval.signs.dual_path.budget import (
    load_sumo_edge_lengths,
    truncate_edge_path,
)


def ego_remaining_on_approach_m(
    spawn_distance_before_end: float,
    spawn_lane_length: Optional[float] = None,
) -> float:
    """Meters left on the approach after ego spawn (not from lane start)."""
    ego_rem = max(0.0, float(spawn_distance_before_end))
    if spawn_lane_length is None:
        return ego_rem
    try:
        return min(ego_rem, max(0.0, float(spawn_lane_length) - 1.0))
    except (TypeError, ValueError):
        return ego_rem


def _resolve_spawn_along_m(
    *,
    spawn_edge_len: float,
    spawn_along_m: Optional[float],
    spawn_distance_before_end: Optional[float],
) -> Optional[float]:
    """Absolute longitude of ego on the spawn edge (from lane start)."""
    if spawn_along_m is not None:
        return float(max(0.0, min(float(spawn_along_m), max(0.0, spawn_edge_len - 1e-3))))
    if spawn_distance_before_end is not None:
        rem = ego_remaining_on_approach_m(spawn_distance_before_end, spawn_edge_len)
        return float(max(0.0, spawn_edge_len - rem))
    return None


def measure_spawn_to_dest_length_m(
    *,
    net_path: Path,
    spawn_edge: str,
    spawn_lane: int,
    dest_edge: str,
    spawn_along_m: Optional[float] = None,
    spawn_distance_before_end: Optional[float] = None,
    dest_end_margin_m: float = 0.0,
    edge_lengths: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Unified travel distance: ego spawn → end of the chosen destination edge.

    Works for every sign family:
      - junction / blocked / crosswalk: multi-edge SUMO path,
        spawn via ``spawn_distance_before_end`` (ego near approach end)
      - speed / detour: usually same edge, spawn via ``spawn_along_m``
        (offset from lane start)
      - dual_path: call once per branch dest

    Length = remaining on spawn edge + full intermediate edges +
    (dest edge length − ``dest_end_margin_m``). Same-edge routes reduce to
    ``dest_end − spawn_along``.
    """
    if not spawn_edge or spawn_lane is None or not dest_edge:
        return None
    if spawn_along_m is None and spawn_distance_before_end is None:
        return None

    lengths = edge_lengths if edge_lengths is not None else load_sumo_edge_lengths(net_path)
    route_index = load_vehicle_route_index(net_path)
    real_path = route_index.find_real_edge_path(
        str(spawn_edge), int(spawn_lane), str(dest_edge)
    )
    if not real_path:
        # Same-edge fallback (speed/detour often stay on one road).
        if str(spawn_edge) == str(dest_edge):
            real_path = [str(spawn_edge)]
        else:
            return None

    edges = [str(e) for e in real_path]
    spawn_len = float(lengths.get(edges[0], 0.0) or 0.0)
    spawn_along = _resolve_spawn_along_m(
        spawn_edge_len=spawn_len,
        spawn_along_m=spawn_along_m,
        spawn_distance_before_end=spawn_distance_before_end,
    )
    if spawn_along is None:
        return None

    margin = max(0.0, float(dest_end_margin_m))
    if len(edges) == 1:
        dest_end = max(0.0, spawn_len - margin)
        return float(max(0.0, dest_end - spawn_along))

    total = max(0.0, spawn_len - spawn_along)
    for eid in edges[1:-1]:
        total += float(lengths.get(eid, 0.0) or 0.0)
    dest_len = float(lengths.get(edges[-1], 0.0) or 0.0)
    total += max(0.0, dest_len - margin)
    return float(total)


def measure_available_route_length_m(
    *,
    net_path: Path,
    ego_edge: str,
    ego_lane: int,
    dest_edge: str,
    spawn_distance_before_end: float,
    spawn_lane_length: Optional[float] = None,
    edge_lengths: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Back-compat wrapper → :func:`measure_spawn_to_dest_length_m`."""
    del spawn_lane_length  # length taken from net edge table
    return measure_spawn_to_dest_length_m(
        net_path=net_path,
        spawn_edge=ego_edge,
        spawn_lane=ego_lane,
        dest_edge=dest_edge,
        spawn_distance_before_end=spawn_distance_before_end,
        edge_lengths=edge_lengths,
    )


def ensure_reachable_ego_destination(
    entry: Dict[str, Any],
    *,
    net_path: Path,
) -> Dict[str, Any]:
    """Rewrite ``destination_lane_id`` to a SUMO-reachable lane, or mark invalid."""
    ego_edge = entry.get("road_id") or entry.get("ego_edge_id")
    ego_lane = entry.get("spawn_lane_num")
    dest_edge = entry.get("destination_edge_id")
    dest_lane_key = entry.get("destination_lane_id")
    if not ego_edge or ego_lane is None or not dest_edge:
        return entry

    route_index = load_vehicle_route_index(net_path)
    reachable = sorted(
        route_index.reachable_lanes_on_edge(
            str(ego_edge), int(ego_lane), str(dest_edge)
        )
    )
    out = dict(entry)
    if not reachable:
        out["valid"] = False
        out["invalid_route_reason"] = "dest_edge_unreachable_from_spawn_lane"
        return out

    pref = None
    if dest_lane_key:
        try:
            pref = int(lane_num_from_key(str(dest_lane_key)))
        except (TypeError, ValueError):
            pref = None
    lane_num = pref if pref in reachable else int(reachable[0])
    out["destination_lane_id"] = make_lane_key(str(dest_edge), lane_num)
    return out


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
    spawn_len = entry.get("spawn_lane_length") or entry.get("approach_lane_length_m")
    try:
        spawn_len_f = float(spawn_len) if spawn_len is not None else None
    except (TypeError, ValueError):
        spawn_len_f = None
    if spawn_len_f is None:
        spawn_len_f = float(edge_lengths.get(str(ego_edge), 0.0) or 0.0)
    ego_rem = ego_remaining_on_approach_m(spawn_distance_before_end, spawn_len_f)

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

    # Snap dest lane to one SUMO can actually reach from spawn. Truncation /
    # preferred lane often keep spawn's lane index (0) even when connections
    # only enter dest lanes 1/2 — MetaDrive then builds a degenerate route.
    reachable = sorted(
        route_index.reachable_lanes_on_edge(
            str(ego_edge), int(ego_lane), str(trimmed.dest_edge_id)
        )
    )
    if not reachable:
        out = dict(entry)
        out["valid"] = False
        out["invalid_route_reason"] = "dest_edge_unreachable_from_spawn_lane"
        return out
    if int(dest_lane_num) in reachable:
        final_lane_num = int(dest_lane_num)
    else:
        final_lane_num = int(reachable[0])
    final_dest_lane_id = make_lane_key(trimmed.dest_edge_id, final_lane_num)

    out = dict(entry)
    out["destination_edge_id"] = trimmed.dest_edge_id
    out["destination_lane_id"] = final_dest_lane_id
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
