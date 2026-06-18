"""Scenario augmentation for ego/aux spawn combinations."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .junction_priority_layout import JunctionPriorityLayout, build_junction_priority_layout


@dataclass(frozen=True)
class SpawnScenario:
    """One valid ego/aux spawn + ego destination combination."""

    ego_edge_id: str
    ego_lane_num: int
    ego_destination_edge_id: str
    ego_destination_lane_key: str
    aux_edge_id: str
    aux_lane_num: int
    aux_destination_edge_id: str
    aux_destination_lane_key: str
    scenario_id: str

    def to_manifest_fields(self) -> dict:
        return {
            "road_id": self.ego_edge_id,
            "spawn_lane_num": self.ego_lane_num,
            "destination_lane_id": self.ego_destination_lane_key,
            "destination_edge_id": self.ego_destination_edge_id,
            "aux_road_id": self.aux_edge_id,
            "aux_spawn_lane_num": self.aux_lane_num,
            "aux_spawn_lane_index": _lane_key(self.aux_edge_id, self.aux_lane_num),
            "aux_destination_lane_id": self.aux_destination_lane_key,
            "aux_destination_edge_id": self.aux_destination_edge_id,
            "augmentation_id": self.scenario_id,
        }


def _lane_key(edge_id: str, lane_num: int) -> str:
    return f"lane_{edge_id}_{lane_num}"


def _edge_base_id(edge_id: str) -> str:
    return edge_id.lstrip("-").split("#", 1)[0]


def _reverse_edge_id(edge_id: str) -> str:
    return edge_id[1:] if edge_id.startswith("-") else f"-{edge_id}"


def _lane_num_from_key(lane_key: str) -> int:
    raw = lane_key[5:] if lane_key.startswith("lane_") else lane_key
    try:
        return int(raw.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return 0


@dataclass
class AugmentationStats:
    """Counts explaining how augmented scenario totals are obtained."""

    secondary_edges: int = 0
    main_edges: int = 0
    ego_lane_slots: int = 0
    aux_lane_slots: int = 0
    ego_dest_edges: int = 0
    raw_combos: int = 0
    skipped_no_ego_connection: int = 0
    skipped_invalid_departure: int = 0
    skipped_no_conflict: int = 0
    valid_scenarios: int = 0
    by_ego_edge: Dict[str, int] = field(default_factory=dict)
    by_aux_edge: Dict[str, int] = field(default_factory=dict)

    def format_report(self, scene_name: str = "") -> str:
        header = f"Augmentation breakdown{f' ({scene_name})' if scene_name else ''}:"
        lines = [
            header,
            f"  secondary arms: {self.secondary_edges}, main arms: {self.main_edges}",
            f"  ego lane slots: {self.ego_lane_slots}, aux lane slots: {self.aux_lane_slots}",
            f"  distinct ego destinations (edges): {self.ego_dest_edges}",
            f"  raw combos (ego_lane x ego_dest x aux_lane x aux_edge): {self.raw_combos}",
            f"  skipped (no SUMO link for ego lane+dest): {self.skipped_no_ego_connection}",
            f"  skipped (invalid departure / same-lane merge): {self.skipped_invalid_departure}",
            f"  skipped (no junction conflict with aux straight): {self.skipped_no_conflict}",
            f"  valid scenarios: {self.valid_scenarios}",
        ]
        if self.by_ego_edge:
            lines.append(f"  by ego edge: {dict(sorted(self.by_ego_edge.items()))}")
        if self.by_aux_edge:
            lines.append(f"  by aux edge: {dict(sorted(self.by_aux_edge.items()))}")
        return "\n".join(lines)


class JunctionConflictModel:
    """SUMO junction request/foes model for maneuver conflict checks."""

    def __init__(self, net_path: Path, junction_id: str, incoming_edge_ids: Set[str]):
        root = ET.parse(net_path).getroot()
        self._conn_links: Dict[Tuple[str, str, str], int] = {}
        self._conn_to_lane: Dict[Tuple[str, str, str], str] = {}
        self._conn_dir: Dict[Tuple[str, str, str], str] = {}
        self._lane_adjacency: Dict[str, List[str]] = {}
        self._outgoing_edges: Set[str] = set()
        auto_idx = 0
        for edge_el in root.findall("edge"):
            edge_id = edge_el.get("id", "")
            if edge_id and edge_el.get("from", "") == junction_id:
                self._outgoing_edges.add(edge_id)

        for conn in root.findall("connection"):
            from_edge = conn.get("from", "")
            if from_edge not in incoming_edge_ids:
                continue
            to_edge = conn.get("to", "")
            from_lane = conn.get("fromLane", "0")
            to_lane = conn.get("toLane", "0")
            conn_dir = (conn.get("dir", "") or "").strip().lower()
            if conn.get("linkIndex") is not None:
                link_idx = int(conn.get("linkIndex"))
            else:
                link_idx = auto_idx
                auto_idx += 1
            key = (from_edge, to_edge, from_lane)
            self._conn_links[key] = link_idx
            self._conn_to_lane[key] = to_lane
            self._conn_dir[key] = conn_dir
            from_key = _lane_key(from_edge, int(from_lane))
            to_key = _lane_key(to_edge, int(to_lane))
            self._lane_adjacency.setdefault(from_key, []).append(to_key)

        self._foes: Dict[int, Set[int]] = {}
        for junction in root.findall("junction"):
            if junction.get("id") != junction_id:
                continue
            for req in junction.findall("request"):
                idx = int(req.get("index"))
                foes_str = req.get("foes", "")
                self._foes[idx] = {i for i, ch in enumerate(foes_str) if ch == "1"}

        self._outgoing_by_incoming: Dict[str, Set[str]] = {}
        for from_edge, to_edge, _ in self._conn_links:
            self._outgoing_by_incoming.setdefault(from_edge, set()).add(to_edge)

    def outgoing_edges(self, from_edge: str) -> Set[str]:
        return set(self._outgoing_by_incoming.get(from_edge, set()))

    def link_indices(
        self,
        from_edge: str,
        to_edge: str,
        from_lane: Optional[int] = None,
    ) -> Set[int]:
        indices: Set[int] = set()
        for (fr, to, fl), link_idx in self._conn_links.items():
            if fr != from_edge or to != to_edge or link_idx < 0:
                continue
            if from_lane is not None and int(fl) != int(from_lane):
                continue
            indices.add(link_idx)
        return indices

    def maneuvers_conflict(self, links_a: Set[int], links_b: Set[int]) -> bool:
        if not links_a or not links_b:
            return False
        for la in links_a:
            if self._foes.get(la, set()) & links_b:
                return True
        for lb in links_b:
            if self._foes.get(lb, set()) & links_a:
                return True
        return False

    def connection_dir(self, from_edge: str, to_edge: str, from_lane: int) -> str:
        return self._conn_dir.get((from_edge, to_edge, str(from_lane)), "")

    def is_outgoing_edge(self, edge_id: str) -> bool:
        return edge_id in self._outgoing_edges

    def lane_path_exists(self, start_lane_key: str, goal_lane_key: str) -> bool:
        if start_lane_key == goal_lane_key:
            return False
        visited = {start_lane_key}
        queue = [start_lane_key]
        while queue:
            cur = queue.pop(0)
            for nxt in self._lane_adjacency.get(cur, []):
                if nxt == goal_lane_key:
                    return True
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def is_corridor_merge_right_turn(
        self,
        layout: JunctionPriorityLayout,
        spawn_edge: str,
        dest_edge: str,
        conn_dir: str,
    ) -> bool:
        """Right turn that only continues the adjacent main road (not a real crossing)."""
        if conn_dir != "r":
            return False
        spawn_arm = layout.arm_for_edge(spawn_edge)
        if spawn_arm is None or layout.shape != "X":
            return False
        n_arms = len(layout.arms)
        right_idx = (spawn_arm.arm_index + 1) % n_arms
        right_arm = layout.arms[right_idx]
        if right_arm.road_class != "main":
            return False
        if _edge_base_id(dest_edge) != _edge_base_id(right_arm.edge_id):
            return False
        if dest_edge in right_arm.straight_to:
            return True
        return _edge_base_id(dest_edge) == _edge_base_id(_reverse_edge_id(right_arm.edge_id))

    def is_valid_ego_departure(
        self,
        layout: JunctionPriorityLayout,
        spawn_edge: str,
        spawn_lane: int,
        dest_edge: str,
        dest_lane_key: str,
    ) -> bool:
        spawn_lane_key = _lane_key(spawn_edge, spawn_lane)
        if spawn_lane_key == dest_lane_key:
            return False
        if spawn_edge == dest_edge:
            return False
        if not self.is_outgoing_edge(dest_edge):
            return False
        if not self.link_indices(spawn_edge, dest_edge, spawn_lane):
            return False
        if not self.lane_path_exists(spawn_lane_key, dest_lane_key):
            return False
        conn_dir = self.connection_dir(spawn_edge, dest_edge, spawn_lane)
        if self.is_corridor_merge_right_turn(layout, spawn_edge, dest_edge, conn_dir):
            return False
        return True

    def ego_aux_conflict(
        self,
        ego_edge: str,
        ego_lane: int,
        ego_dest_edge: str,
        aux_edge: str,
        aux_lane: int,
        aux_dest_edge: str,
    ) -> bool:
        ego_links = self.link_indices(ego_edge, ego_dest_edge, ego_lane)
        aux_links = self.link_indices(aux_edge, aux_dest_edge, aux_lane)
        return self.maneuvers_conflict(ego_links, aux_links)

    def destination_lane_key(
        self,
        from_edge: str,
        from_lane: int,
        to_edge: str,
        lane_keys_by_edge: Dict[str, List[str]],
    ) -> Optional[str]:
        """Resolve manifest destination lane key from SUMO fromLane/toLane."""
        to_lane_num: Optional[int] = None
        for (fr, to, fl), to_lane in self._conn_to_lane.items():
            if fr == from_edge and to == to_edge and int(fl) == int(from_lane):
                try:
                    to_lane_num = int(to_lane)
                except ValueError:
                    continue
                break
        if to_lane_num is not None:
            return _pick_outgoing_lane_key(to_edge, to_lane_num, lane_keys_by_edge)
        if self.link_indices(from_edge, to_edge, from_lane):
            return _pick_outgoing_lane_key(to_edge, 0, lane_keys_by_edge)
        return None


def _pick_outgoing_lane_key(
    edge_id: str,
    lane_num: int,
    lane_keys_by_edge: Dict[str, List[str]],
) -> str:
    keys = lane_keys_by_edge.get(edge_id, [])
    if not keys:
        return _lane_key(edge_id, lane_num)
    for key in keys:
        if _lane_num_from_key(key) == lane_num:
            return key
    return keys[min(lane_num, len(keys) - 1)]


def _aux_straight_destination(
    layout: JunctionPriorityLayout,
    aux_edge_id: str,
    conflict_model: JunctionConflictModel,
) -> Optional[str]:
    arm = layout.arm_for_edge(aux_edge_id)
    if arm is not None and arm.straight_to:
        return arm.straight_to[0]
    outgoing = conflict_model.outgoing_edges(aux_edge_id)
    if not outgoing:
        return None
    # Prefer the longest outgoing edge name as a stable straight-ish default.
    return sorted(outgoing)[0]


def enumerate_spawn_scenarios(
    net_path: Path,
    layout: JunctionPriorityLayout,
    spawn_lanes_by_edge: Dict[str, List[int]],
    *,
    min_lane_length: float = 20.0,
    lane_lengths: Optional[Dict[Tuple[str, int], float]] = None,
    collect_stats: bool = False,
) -> List[SpawnScenario] | Tuple[List[SpawnScenario], AugmentationStats]:
    """
    Enumerate valid ego/aux spawn combinations with conflicting paths.

    Args:
        net_path: Scene map.net.xml path.
        layout: Junction priority layout for the scene.
        spawn_lanes_by_edge: edge_id -> sorted lane indices available for spawn.
        min_lane_length: Skip lanes shorter than this (metres).
        lane_lengths: Optional (edge_id, lane_num) -> length map for filtering.
    """
    lane_lengths = lane_lengths or {}
    incoming_edges = {arm.edge_id for arm in layout.arms}
    conflict_model = JunctionConflictModel(net_path, layout.junction_id, incoming_edges)

    lane_keys_by_edge: Dict[str, List[str]] = {
        arm.edge_id: list(arm.lane_keys) for arm in layout.arms
    }

    secondary_edges = sorted(layout.secondary_edge_ids)
    main_edges = sorted(layout.main_edge_ids)

    scenarios: List[SpawnScenario] = []
    stats = AugmentationStats(
        secondary_edges=len(secondary_edges),
        main_edges=len(main_edges),
        ego_lane_slots=sum(len(spawn_lanes_by_edge.get(e, [])) for e in secondary_edges),
        aux_lane_slots=sum(len(spawn_lanes_by_edge.get(e, [])) for e in main_edges),
    )
    all_dest_edges: Set[str] = set()

    for ego_edge in secondary_edges:
        ego_lane_nums = spawn_lanes_by_edge.get(ego_edge, [])
        if not ego_lane_nums:
            continue

        ego_dest_edges = sorted(conflict_model.outgoing_edges(ego_edge))
        if not ego_dest_edges:
            continue
        all_dest_edges.update(ego_dest_edges)

        for aux_edge in main_edges:
            if aux_edge == ego_edge:
                continue

            aux_dest_edge = _aux_straight_destination(layout, aux_edge, conflict_model)
            if aux_dest_edge is None:
                continue

            aux_lane_nums = spawn_lanes_by_edge.get(aux_edge, [])
            if not aux_lane_nums:
                continue

            for ego_lane in ego_lane_nums:
                if lane_lengths.get((ego_edge, ego_lane), min_lane_length) < min_lane_length:
                    continue

                for ego_dest_edge in ego_dest_edges:
                    ego_dest_lane_key = conflict_model.destination_lane_key(
                        ego_edge,
                        ego_lane,
                        ego_dest_edge,
                        lane_keys_by_edge,
                    )
                    if ego_dest_lane_key is None:
                        stats.raw_combos += len(aux_lane_nums)
                        stats.skipped_no_ego_connection += len(aux_lane_nums)
                        continue

                    for aux_lane in aux_lane_nums:
                        stats.raw_combos += 1
                        if lane_lengths.get((aux_edge, aux_lane), min_lane_length) < min_lane_length:
                            continue

                        if not conflict_model.is_valid_ego_departure(
                            layout,
                            ego_edge,
                            ego_lane,
                            ego_dest_edge,
                            ego_dest_lane_key,
                        ):
                            stats.skipped_invalid_departure += 1
                            continue

                        if not conflict_model.ego_aux_conflict(
                            ego_edge,
                            ego_lane,
                            ego_dest_edge,
                            aux_edge,
                            aux_lane,
                            aux_dest_edge,
                        ):
                            stats.skipped_no_conflict += 1
                            continue

                        aux_dest_lane_key = _pick_outgoing_lane_key(
                            aux_dest_edge,
                            aux_lane,
                            lane_keys_by_edge,
                        )

                        scenario_id = (
                            f"ego_{ego_edge}_L{ego_lane}"
                            f"_to_{ego_dest_edge}"
                            f"_aux_{aux_edge}_L{aux_lane}"
                        )
                        scenarios.append(
                            SpawnScenario(
                                ego_edge_id=ego_edge,
                                ego_lane_num=ego_lane,
                                ego_destination_edge_id=ego_dest_edge,
                                ego_destination_lane_key=ego_dest_lane_key,
                                aux_edge_id=aux_edge,
                                aux_lane_num=aux_lane,
                                aux_destination_edge_id=aux_dest_edge,
                                aux_destination_lane_key=aux_dest_lane_key,
                                scenario_id=scenario_id,
                            )
                        )
                        stats.by_ego_edge[ego_edge] = stats.by_ego_edge.get(ego_edge, 0) + 1
                        stats.by_aux_edge[aux_edge] = stats.by_aux_edge.get(aux_edge, 0) + 1

    stats.ego_dest_edges = len(all_dest_edges)
    stats.valid_scenarios = len(scenarios)
    if collect_stats:
        return scenarios, stats
    return scenarios


def build_spawn_lanes_by_edge(
    spawn_lanes: Iterable,
) -> Dict[str, List[int]]:
    """Group parsed SumoLaneInfo rows by edge id."""
    by_edge: Dict[str, Set[int]] = {}
    for lane in spawn_lanes:
        by_edge.setdefault(lane.edge_id, set()).add(lane.lane_num)
    return {edge: sorted(nums) for edge, nums in sorted(by_edge.items())}


def lane_lengths_from_spawn_lanes(spawn_lanes: Iterable) -> Dict[Tuple[str, int], float]:
    return {(lane.edge_id, lane.lane_num): float(lane.length) for lane in spawn_lanes}


def augment_layout_for_scene(
    net_path: Path,
    spawn_lanes: Iterable,
    *,
    min_lane_length: float = 20.0,
) -> Tuple[Optional[JunctionPriorityLayout], List[SpawnScenario], Optional[AugmentationStats]]:
    """Build layout and enumerate augmented scenarios for one scene."""
    layout = build_junction_priority_layout(net_path)

    spawn_by_edge = build_spawn_lanes_by_edge(spawn_lanes)
    lengths = lane_lengths_from_spawn_lanes(spawn_lanes)

    scenarios, stats = enumerate_spawn_scenarios(
        net_path,
        layout,
        spawn_by_edge,
        min_lane_length=min_lane_length,
        lane_lengths=lengths,
        collect_stats=True,
    )
    return layout, scenarios, stats
