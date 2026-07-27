"""MetaDrive shortest-path reachability helpers for dual-path scenes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .lane_keys import make_lane_key


# MetaDrive EdgeNetworkNavigation.shortest_path -> find_path(..., max_len=10)
METADRIVE_MAX_ROUTE_HOPS = 10


def is_metadrive_path_ok(path: Sequence[str] | None, *, spawn: str, dest: str) -> bool:
    return bool(
        path
        and len(path) > 1
        and path[0] != path[-1]
        and path[-1] == dest
        and spawn in path[:1]
    )


def filter_dual_paths_with_road_network(
    scenarios: Sequence[Any],
    road_network: Any,
    *,
    one_per_ego: bool = True,
    max_keep: Optional[int] = None,
) -> Tuple[List[Any], int]:
    """Keep scenarios whose spawn→dest is MetaDrive ``shortest_path``-reachable.

    Returns ``(kept, n_dropped)``.
    """
    filtered: List[Any] = []
    seen_arm: set[str] = set()
    dropped = 0
    for scenario in scenarios:
        start_lane = make_lane_key(scenario.ego_edge_id, scenario.ego_lane_num)
        dest_lane = make_lane_key(scenario.dest_edge_id, scenario.dest_lane_num)
        if start_lane not in road_network.graph or dest_lane not in road_network.graph:
            dropped += 1
            continue
        path = road_network.shortest_path(start_lane, dest_lane)
        if not is_metadrive_path_ok(path, spawn=start_lane, dest=dest_lane):
            dropped += 1
            continue
        if one_per_ego and scenario.ego_edge_id in seen_arm:
            continue
        if one_per_ego:
            seen_arm.add(scenario.ego_edge_id)
        filtered.append(scenario)
        if max_keep is not None and len(filtered) >= max_keep:
            break
    return filtered, dropped


def probe_road_network_for_net(
    net_path: Path,
    *,
    spawn_edge_id: str,
    spawn_lane_num: int = 0,
    destination_lane_id: Optional[str] = None,
    pdd_code: str = "4.1.1",
):
    """Open a short-lived MetaDrive env on ``net_path`` and return ``(env, road_network)``."""
    from run_benchmark import _build_sumo_env

    net_path = Path(net_path).resolve()
    probe_row = {
        "net_path": str(net_path),
        "sign_code": pdd_code,
        "sign_type": "direction_signs",
        "pdd_code": pdd_code,
        "traffic_density": 0.0,
        "horizon": 5,
        "road_id": spawn_edge_id,
        "spawn_lane_num": spawn_lane_num,
    }
    if destination_lane_id:
        probe_row["destination_lane_id"] = destination_lane_id

    # scenes_root unused when net_path is absolute
    env = _build_sumo_env(probe_row, net_path.parent, max_steps=5)
    env.reset(seed=0)
    return env, env.engine.current_map.road_network


def filter_dual_paths_metadrive(
    scenarios: Sequence[Any],
    net_path: Path,
    *,
    one_per_ego: bool = True,
    max_keep: Optional[int] = None,
    pdd_code: str = "4.1.1",
) -> Tuple[List[Any], int]:
    """Open MetaDrive once on ``net_path`` and filter scenarios."""
    if not scenarios:
        return [], 0
    probe = scenarios[0]
    env = None
    try:
        env, road_network = probe_road_network_for_net(
            net_path,
            spawn_edge_id=probe.ego_edge_id,
            spawn_lane_num=probe.ego_lane_num,
            destination_lane_id=make_lane_key(probe.dest_edge_id, probe.dest_lane_num),
            pdd_code=pdd_code,
        )
        return filter_dual_paths_with_road_network(
            scenarios,
            road_network,
            one_per_ego=one_per_ego,
            max_keep=max_keep,
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
