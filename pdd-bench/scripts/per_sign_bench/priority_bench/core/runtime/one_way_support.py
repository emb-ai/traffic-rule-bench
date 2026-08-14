"""One-way / direction dual-path runtime: NPC exclusion + compliant nav.

Wrong-way (5.7) / forbidden first exit (4.1) stay in the net so ego baselines
can violate via ordinary MetaDrive routing. Background traffic (5.7) and
rule-compliant nav never use those edges.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from envs.sumo_traffic_manager import SumoTrafficManager

from ..sumo.lane_keys import lane_edge_id, make_lane_key


class OneWaySumoTrafficManager(SumoTrafficManager):
    """Background traffic that respects the one-way crossing road."""

    def _excluded_edges(self) -> set:
        raw = self.engine.global_config.get("background_excluded_edges") or ()
        return {str(e) for e in raw}

    def _get_spawnable_lanes(self):
        lanes = super()._get_spawnable_lanes()
        excluded = self._excluded_edges()
        if not excluded:
            return lanes
        return [ln for ln in lanes if lane_edge_id(str(ln.index)) not in excluded]

    def _build_forward_route(self, spawn_lane_index):
        route = super()._build_forward_route(spawn_lane_index)
        excluded = self._excluded_edges()
        if not excluded:
            return route
        trimmed = []
        for lane_idx in route:
            if lane_edge_id(str(lane_idx)) in excluded:
                break
            trimmed.append(lane_idx)
        return trimmed or route


def scene_background_excluded_edges(net_path: Path) -> list:
    """Wrong-way carriageway edges from scene meta.json (if any)."""
    meta_path = Path(net_path).resolve().parent / "meta.json"
    if not meta_path.is_file():
        return []
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return []
    edges = meta.get("background_excluded_edges")
    if not edges:
        dp = meta.get("dual_path")
        edges = (dp or {}).get("wrong_dir_edges") if isinstance(dp, dict) else None
    return [str(e) for e in (edges or [])]


def _lanes_on_edges(road_network, edges) -> set[str]:
    want = {str(e) for e in (edges or ())}
    if not want:
        return set()
    graph = getattr(road_network, "graph", None) or {}
    out: set[str] = set()
    for lid in graph:
        if not isinstance(lid, str):
            continue
        if lane_edge_id(lid) in want:
            out.add(lid)
    return out


def _bfs_lane_path(
    road_network,
    start_lane_id: str,
    goal_lane_id: str,
    blocked_lanes,
    *,
    max_len: int = 120,
) -> list[str] | None:
    graph = getattr(road_network, "graph", None)
    if graph is None or start_lane_id not in graph:
        return None
    blocked = set(blocked_lanes or ())
    if start_lane_id in blocked:
        return None
    queue = deque([(start_lane_id, [start_lane_id])])
    seen = {start_lane_id}
    while queue:
        lane_id, path = queue.popleft()
        if lane_id == goal_lane_id:
            return path
        lane_data = graph.get(lane_id)
        if lane_data is None:
            continue
        for nxt in sorted(set(getattr(lane_data, "exit_lanes", None) or [])):
            if nxt in seen or nxt in blocked or nxt not in graph:
                continue
            new_path = path + [nxt]
            if len(new_path) > max_len:
                continue
            seen.add(nxt)
            queue.append((nxt, new_path))
    return None


def _expand_edge_seq_to_lane_path(
    road_network,
    edge_seq: list[str],
    spawn_lane_id: str,
    dest_lane_id: str,
    blocked_lanes,
    *,
    max_seg: int = 25,
) -> list[str] | None:
    graph = getattr(road_network, "graph", None) or {}
    if spawn_lane_id not in graph:
        return None
    blocked = set(blocked_lanes or ())
    path: list[str] = [spawn_lane_id]

    def _lane_exits(lid: str) -> list[str]:
        data = graph.get(lid)
        if data is None:
            return []
        return list(set(getattr(data, "exit_lanes", None) or []))

    def _prefer_live_peer(lid: str) -> str:
        edge = lane_edge_id(lid)
        peers = [p for p in _lanes_on_edges(road_network, [edge]) if p not in blocked]
        if not peers:
            return lid
        if _lane_exits(lid):
            return lid
        peers_sorted = sorted(
            peers,
            key=lambda p: (0 if p.endswith("_0") else 1, -len(_lane_exits(p)), p),
        )
        return peers_sorted[0] if peers_sorted else lid

    for next_edge in edge_seq[1:]:
        cur = path[-1]
        if lane_edge_id(cur) == next_edge:
            continue
        targets = _lanes_on_edges(road_network, [next_edge]) - blocked
        if not targets:
            return None
        queue = deque([(cur, [cur])])
        seen = {cur}
        found: list[str] | None = None
        while queue:
            lid, seg = queue.popleft()
            if lid in targets and lid != cur:
                found = seg
                break
            if len(seg) > max_seg:
                continue
            exits = _lane_exits(lid)
            exits.sort(
                key=lambda x: (
                    0 if lane_edge_id(x) == next_edge else 1,
                    0 if x.endswith("_0") else 1,
                    x,
                )
            )
            for nxt in exits:
                if nxt in seen or nxt in blocked or nxt not in graph:
                    continue
                seen.add(nxt)
                queue.append((nxt, seg + [nxt]))
        if not found:
            return None
        # Drop the start of the segment (already on path).
        path.extend(found[1:])
        path[-1] = _prefer_live_peer(path[-1])

    if path[-1] != dest_lane_id:
        tail = _bfs_lane_path(
            road_network, path[-1], dest_lane_id, blocked, max_len=max_seg
        )
        if not tail:
            return None
        path.extend(tail[1:])
    return path


def _apply_nav_lane_path(env, path: list[str]) -> bool:
    vehicle = env.agent
    nav = getattr(vehicle, "navigation", None)
    if nav is None or not path:
        return False
    road_network = env.engine.current_map.road_network
    try:
        nav.checkpoints = list(path)
        nav._target_checkpoints_index = [0, 1]
        nav.final_lane = road_network.get_lane(path[-1])
        if getattr(nav, "_navi_info", None) is not None:
            nav._navi_info.fill(0.0)
        nav.current_ref_lanes = road_network.get_peer_lanes_from_index(path[0])
        nav.next_ref_lanes = road_network.get_peer_lanes_from_index(path[1])
        try:
            vehicle.config["destination"] = path[-1]
        except Exception:
            pass
        try:
            nav.update_localization(vehicle)
        except Exception:
            pass
        return True
    except Exception as exc:
        print(f"[OneWayNav] apply path failed: {exc}")
        return False


def forbidden_edges_for_compliant_nav(row: dict) -> list[str]:
    """Edges rule-compliant nav must not use.

    5.7.x: the opposite carriageway (``wrong_dir_edges``). Blocking those
    is safe — the compliant detour never needs them.

    4.1.x / 3.18.x / 3.1: do not permanently block the harvested forbidden
    branch. A long compliant detour may legally re-enter that turn later.
    Pin ``straight_path`` only; an empty blocked set (no wrong-dir edges)
    is correct.
    """
    dual = row.get("dual_path") or {}
    one_way = [str(e) for e in (row.get("background_excluded_edges") or []) if e]
    if not one_way:
        one_way = [str(e) for e in (dual.get("wrong_dir_edges") or []) if e]
    return one_way


def install_one_way_compliant_nav_route(env, row: dict) -> bool:
    """Rebuild ego nav on the allowed route with the forbidden path blocked."""
    dual = row.get("dual_path") or {}
    dest = (
        row.get("compliant_destination_lane_id")
        or row.get("destination_lane_id")
    )
    road_id = row.get("road_id")
    straight_edges = [str(e) for e in (dual.get("straight_path") or [])]
    if not dest or not road_id or not straight_edges:
        return False
    dest = str(dest)
    if not dest.startswith("lane_"):
        dest = f"lane_{dest}"
    spawn_num = int(row.get("spawn_lane_num", 0) or 0)
    spawn_key = make_lane_key(str(road_id), spawn_num)

    forbidden_edges = forbidden_edges_for_compliant_nav(row)

    try:
        vehicle = env.agent
        nav = getattr(vehicle, "navigation", None)
        if nav is None:
            return False
        road_network = env.engine.current_map.road_network
        graph = getattr(road_network, "graph", None) or {}
        if spawn_key not in graph or dest not in graph:
            print(f"[OneWayNav] missing lanes spawn={spawn_key} dest={dest}")
            return False

        blocked = _lanes_on_edges(road_network, forbidden_edges)
        edge_seq = [str(road_id), *straight_edges]
        path = _expand_edge_seq_to_lane_path(
            road_network, edge_seq, spawn_key, dest, blocked
        )
        if not path:
            path = _bfs_lane_path(
                road_network, spawn_key, dest, blocked, max_len=120
            )
            method = "bfs"
        else:
            method = "edge_expand"

        if not path:
            print(
                f"[OneWayNav] no compliant path {spawn_key} → {dest} "
                f"(blocked {len(blocked)} forbidden lanes; "
                f"straight_edges={len(straight_edges)})"
            )
            return False

        if any(ck in blocked for ck in path):
            print("[OneWayNav] path still touches wrong-dir lanes; refusing")
            return False

        if not _apply_nav_lane_path(env, path):
            return False

        first_road = None
        for ck in path[1:]:
            e = lane_edge_id(ck)
            if not str(e).startswith(":"):
                first_road = e
                break
        expected = dual.get("straight_first_exit") or row.get("compliant_first_exit")
        note = ""
        if expected and first_road and str(expected) != str(first_road):
            note = f" (first_road={first_road}, manifest={expected})"
        print(
            f"[OneWayNav] installed compliant {len(path)}-hop route via {method} "
            f"{spawn_key} → {dest}{note}"
        )
        return True
    except Exception as exc:
        print(f"[OneWayNav] failed: {exc}")
        return False


def resolve_row_background_excluded_edges(row: dict, net_path: Path | str) -> list:
    edges = row.get("background_excluded_edges")
    if edges:
        return list(edges)
    dual = row.get("dual_path") or {}
    if isinstance(dual, dict) and dual.get("wrong_dir_edges"):
        return list(dual["wrong_dir_edges"])
    return scene_background_excluded_edges(Path(net_path))
