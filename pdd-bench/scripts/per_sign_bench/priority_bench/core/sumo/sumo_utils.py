"""Shared SUMO utilities."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

DEFAULT_NET_FILE = "map.net.xml"
CORE_SCENES_SUBDIR = "core"

# SUMO allow tokens that do not permit motor-vehicle travel on their own.
_NON_VEHICLE_ALLOW_ONLY = frozenset({"pedestrian"})


def is_real_sumo_edge_id(edge_id: str) -> bool:
    """True for normal road edges (not internal ``:`` edges)."""
    return bool(edge_id) and not str(edge_id).startswith(":")


def is_vehicle_drivable_lane(lane_el: ET.Element) -> bool:
    """True when a SUMO lane can carry ego/aux vehicles (not pedestrian-only)."""
    allow = (lane_el.get("allow") or "").strip()
    if not allow:
        return True
    drivable = [token for token in allow.split() if token not in _NON_VEHICLE_ALLOW_ONLY]
    return bool(drivable)


class VehicleRouteIndex:
    """SUMO connection graph for checking vehicle reachability between lanes."""

    def __init__(self, net_root: ET.Element):
        self._edge_fn = {
            edge.get("id", ""): edge.get("function", "normal")
            for edge in net_root.findall("edge")
            if edge.get("id")
        }
        self._adj: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
        for conn in net_root.findall("connection"):
            from_edge = conn.get("from")
            to_edge = conn.get("to")
            if not from_edge or not to_edge:
                continue
            if self._edge_fn.get(to_edge) == "walkingarea":
                continue
            from_lane = int(conn.get("fromLane", 0) or 0)
            to_lane = int(conn.get("toLane", 0) or 0)
            self._adj[(from_edge, from_lane)].append((to_edge, to_lane))

    def can_reach_edge(
        self,
        from_edge: str,
        from_lane: int,
        to_edge: str,
        *,
        max_hops: int = 8,
    ) -> bool:
        """Return True if a vehicle can route from spawn lane to any lane on ``to_edge``."""
        if not is_real_sumo_edge_id(to_edge) or from_edge == to_edge:
            return False

        start = (from_edge, from_lane)
        queue: deque[tuple[tuple[str, int], int]] = deque([(start, 0)])
        visited = {start}

        while queue:
            (edge, lane), depth = queue.popleft()
            if depth > max_hops:
                continue
            for next_edge, next_lane in self._adj.get((edge, lane), []):
                if self._edge_fn.get(next_edge) == "walkingarea":
                    continue
                if is_real_sumo_edge_id(next_edge) and next_edge == to_edge:
                    return True
                state = (next_edge, next_lane)
                if state in visited:
                    continue
                visited.add(state)
                queue.append((state, depth + 1))
        return False

    def has_exit(self, from_edge: str, from_lane: int) -> bool:
        """True if this lane has at least one SUMO connection leaving it."""
        return bool(self._adj.get((from_edge, int(from_lane))))

    def reachable_lanes_on_edge(
        self,
        from_edge: str,
        from_lane: int,
        to_edge: str,
        *,
        max_hops: int = 8,
    ) -> set[int]:
        """Lane indices on ``to_edge`` reachable from the spawn lane."""
        if not is_real_sumo_edge_id(to_edge) or from_edge == to_edge:
            return set()

        start = (from_edge, int(from_lane))
        queue: deque[tuple[tuple[str, int], int]] = deque([(start, 0)])
        visited = {start}
        hits: set[int] = set()

        while queue:
            (edge, lane), depth = queue.popleft()
            if depth > max_hops:
                continue
            for next_edge, next_lane in self._adj.get((edge, lane), []):
                if self._edge_fn.get(next_edge) == "walkingarea":
                    continue
                if is_real_sumo_edge_id(next_edge) and next_edge == to_edge:
                    hits.add(int(next_lane))
                state = (next_edge, next_lane)
                if state in visited:
                    continue
                visited.add(state)
                queue.append((state, depth + 1))
        return hits

    def reachable_real_edges(
        self,
        from_edge: str,
        from_lane: int,
        *,
        max_hops: int = 8,
    ) -> list[str]:
        """Real (non-internal) edges reachable from this lane, BFS order."""
        return [edge for edge, _hops in self.reachable_real_edges_with_hops(
            from_edge, from_lane, max_hops=max_hops
        )]

    def reachable_real_edges_with_hops(
        self,
        from_edge: str,
        from_lane: int,
        *,
        max_hops: int = 8,
    ) -> list[tuple[str, int]]:
        """Real edges reachable from this lane as ``(edge_id, min_hops)`` BFS order."""
        start = (from_edge, int(from_lane))
        queue: deque[tuple[tuple[str, int], int]] = deque([(start, 0)])
        visited = {start}
        found: list[tuple[str, int]] = []
        seen_edges: set[str] = set()

        while queue:
            (edge, lane), depth = queue.popleft()
            if depth > max_hops:
                continue
            for next_edge, next_lane in self._adj.get((edge, lane), []):
                if self._edge_fn.get(next_edge) == "walkingarea":
                    continue
                if is_real_sumo_edge_id(next_edge) and next_edge not in seen_edges:
                    seen_edges.add(next_edge)
                    found.append((next_edge, depth + 1))
                state = (next_edge, next_lane)
                if state in visited:
                    continue
                visited.add(state)
                queue.append((state, depth + 1))
        return found


def load_vehicle_route_index(net_path: Path) -> VehicleRouteIndex:
    root = ET.parse(net_path).getroot()
    return VehicleRouteIndex(root)


def junction_scene_name(core_scene_name: str, rank: int) -> str:
    """Build a junction crop folder name, e.g. sign_72424 + 0 -> sign_72424_j0."""
    return f"{core_scene_name}_j{rank}"


def is_junction_scene_name(name: str) -> bool:
    """True for junction crop folders like sign_72424_j0."""
    if "_j" not in name:
        return False
    base, suffix = name.rsplit("_j", 1)
    return base.startswith("sign_") and suffix.isdigit()


def is_core_scene_name(name: str) -> bool:
    """True for imported catalog scenes like sign_72424 (not junction crops)."""
    return name.startswith("sign_") and name[5:].isdigit()


def resolve_net_file(scene_dir: Path, meta: dict) -> str:
    """Resolve SUMO net filename (neutral map.net.xml, with legacy fallback).
    
    Args:
        scene_dir: Path to the scene directory.
        meta: Scene metadata dict (from meta.json).
        
    Returns:
        Name of the .net.xml file in the scene directory.
        
    Raises:
        FileNotFoundError: If no .net.xml file is found.
    """
    net_file = meta.get("net_file", DEFAULT_NET_FILE)
    if (scene_dir / net_file).exists():
        return net_file

    net_files = sorted(scene_dir.glob("*.net.xml"))
    if net_files:
        return net_files[0].name

    raise FileNotFoundError(
        f"No .net.xml file found in {scene_dir} "
        f"(expected {DEFAULT_NET_FILE} or net_file in meta.json)"
    )


def load_scene_meta(scene_dir: Path) -> dict:
    """Load meta.json from a scene directory.
    
    Args:
        scene_dir: Path to the scene directory.
        
    Returns:
        Dictionary with scene metadata.
        
    Raises:
        FileNotFoundError: If meta.json is not found.
    """
    meta_path = scene_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {scene_dir}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def resolve_scene_dir(scenes_dir: Path, scene_name: str) -> Path:
    """Resolve and validate a scene directory path.
    
    Args:
        scenes_dir: Root directory containing scenes.
        scene_name: Name of the scene subdirectory.
        
    Returns:
        Path to the scene directory.
        
    Raises:
        FileNotFoundError: If the scene directory doesn't exist.
    """
    scene_dir = scenes_dir / scene_name
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene folder not found: {scene_dir}")
    return scene_dir


def find_first_edge_id(net_path: Path) -> Optional[str]:
    """Find the first non-internal edge ID from a SUMO net.xml file.
    
    Args:
        net_path: Path to the .net.xml file.
        
    Returns:
        First non-internal edge ID, or None if not found.
    """
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(net_path)
        root = tree.getroot()
        for edge in root.findall("edge"):
            edge_id = edge.get("id", "")
            if not edge_id.startswith(":"):
                return edge_id
    except Exception:
        pass
    return None
