"""Junction background traffic: spawn only on outgoing (departure) edges."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from traffic_bench.envs.traffic import SumoTrafficManager
from traffic_bench.eval.engine.map.lane_keys import lane_edge_id

_OUTGOING_ARM_KEYS = ("straight_to", "left_to", "right_to", "outgoing_to")


def outgoing_edges_from_junction_layout(layout: dict) -> List[str]:
    """Normal edges that depart the junction (connection targets, not approaches)."""
    outgoing: set[str] = set()
    incoming: set[str] = set()
    for arm in layout.get("arms") or []:
        edge_id = arm.get("edge_id")
        if edge_id:
            incoming.add(str(edge_id))
        for key in _OUTGOING_ARM_KEYS:
            for raw in arm.get(key) or []:
                eid = str(raw).strip()
                if eid and not eid.startswith(":"):
                    outgoing.add(eid)
    outgoing -= incoming
    return sorted(outgoing)


def resolve_row_background_spawn_edges(row: dict, net_path: Path | str) -> List[str]:
    """Outgoing-edge whitelist for junction background traffic."""
    stored = row.get("background_spawn_edges")
    if stored:
        return [str(e) for e in stored if e]
    layout = row.get("junction_layout")
    if isinstance(layout, dict):
        edges = outgoing_edges_from_junction_layout(layout)
        if edges:
            return edges
    return []


class JunctionOutgoingTrafficManager(SumoTrafficManager):
    """Background traffic restricted to outgoing roads from the junction."""

    def _allowed_edges(self) -> set[str]:
        raw = self.engine.global_config.get("background_spawn_edges") or ()
        return {str(e) for e in raw}

    def _filter_spawn_lanes(self, lanes: Sequence) -> list:
        allowed = self._allowed_edges()
        if not allowed:
            return []
        return [ln for ln in lanes if lane_edge_id(str(ln.index)) in allowed]

    def _get_spawnable_lanes(self):
        return self._filter_spawn_lanes(super()._get_spawnable_lanes())
