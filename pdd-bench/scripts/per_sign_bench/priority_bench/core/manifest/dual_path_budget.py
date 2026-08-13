"""Truncate dual-path routes to a shared travel budget (meters).

Both baseline (turn) and compliant (straight) paths are cut to the same
``L_budget`` so a fixed episode horizon can finish. Destinations differ per
route; each dest stays past the critical junction exit.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..sumo.lane_keys import make_lane_key


def load_sumo_edge_lengths(net_path: Path) -> Dict[str, float]:
    """Return ``edge_id -> length_m`` from a SUMO ``*.net.xml``."""
    root = ET.parse(net_path).getroot()
    out: Dict[str, float] = {}
    for edge in root.findall("edge"):
        eid = edge.get("id")
        if not eid or eid.startswith(":"):
            continue
        lengths: List[float] = []
        for lane in edge.findall("lane"):
            try:
                lengths.append(float(lane.get("length") or 0.0))
            except (TypeError, ValueError):
                continue
        if lengths:
            out[str(eid)] = float(max(lengths))
        else:
            try:
                out[str(eid)] = float(edge.get("length") or 0.0)
            except (TypeError, ValueError):
                out[str(eid)] = 0.0
    return out


@dataclass(frozen=True)
class TruncatedRoute:
    """One dual-path branch cut to ``budget_m``."""

    edges: Tuple[str, ...]
    dest_edge_id: str
    dest_lane_id: str
    length_m: float
    destination_max_along_m: Optional[float]
    truncated: bool


def truncate_edge_path(
    path_edges: Sequence[str],
    *,
    edge_lengths: Dict[str, float],
    budget_after_ego_m: float,
    dest_lane_num: int = 0,
) -> Optional[TruncatedRoute]:
    """Cut ``path_edges`` (post-ego exits) so cumulative length ≤ budget.

    Always keeps at least the first exit so the signed junction decision remains
    in the episode. If the cut lands mid-edge, ``destination_max_along_m`` is set.
    """
    edges = [str(e) for e in path_edges if e]
    if not edges:
        return None

    budget = max(5.0, float(budget_after_ego_m))
    truncated: List[str] = []
    accum = 0.0
    along_cap: Optional[float] = None
    was_truncated = False

    for i, eid in enumerate(edges):
        length = float(edge_lengths.get(eid, 0.0) or 0.0)
        # Always include the first exit (junction decision).
        if i == 0:
            truncated.append(eid)
            if length <= budget:
                accum = length
            else:
                along_cap = max(5.0, budget)
                accum = along_cap
                was_truncated = True
                break
            continue

        if accum >= budget:
            was_truncated = True
            break

        room = budget - accum
        if length <= room + 1e-6:
            truncated.append(eid)
            accum += length
            continue

        truncated.append(eid)
        along_cap = max(5.0, room)
        accum += along_cap
        was_truncated = True
        break
    else:
        was_truncated = False

    if not truncated:
        return None

    dest_edge = truncated[-1]
    return TruncatedRoute(
        edges=tuple(truncated),
        dest_edge_id=dest_edge,
        dest_lane_id=make_lane_key(dest_edge, int(dest_lane_num)),
        length_m=float(accum),
        destination_max_along_m=(
            float(along_cap) if along_cap is not None else None
        ),
        truncated=bool(was_truncated or len(truncated) < len(edges)),
    )


def apply_dual_path_route_budget(
    dual_meta: dict,
    *,
    ego_edge_id: str,
    edge_lengths: Dict[str, float],
    budget_m: float,
    spawn_remaining_on_ego_m: float,
    dest_lane_num: int = 0,
) -> dict:
    """Return updated ``dual_path`` dict + baseline/compliant dest fields.

    ``budget_m`` is total travel from spawn. Ego remainder is subtracted so the
    post-junction branches share the leftover budget.
    """
    budget = float(budget_m)
    ego_rem = max(0.0, float(spawn_remaining_on_ego_m))
    after_ego = max(5.0, budget - ego_rem)

    turn_path = list(dual_meta.get("turn_path") or [])
    straight_path = list(dual_meta.get("straight_path") or [])

    base = truncate_edge_path(
        turn_path,
        edge_lengths=edge_lengths,
        budget_after_ego_m=after_ego,
        dest_lane_num=dest_lane_num,
    )
    comp = truncate_edge_path(
        straight_path,
        edge_lengths=edge_lengths,
        budget_after_ego_m=after_ego,
        dest_lane_num=dest_lane_num,
    )
    if base is None or comp is None:
        raise ValueError("dual_path truncation failed: empty turn/straight path")

    out = dict(dual_meta)
    out["full_turn_path"] = list(turn_path)
    out["full_straight_path"] = list(straight_path)
    out["full_turn_length_m"] = float(dual_meta.get("turn_length_m") or 0.0)
    out["full_straight_length_m"] = float(dual_meta.get("straight_length_m") or 0.0)
    out["turn_path"] = list(base.edges)
    out["straight_path"] = list(comp.edges)
    out["turn_length_m"] = float(ego_rem + base.length_m)
    out["straight_length_m"] = float(ego_rem + comp.length_m)
    out["gain_m"] = float(out["straight_length_m"] - out["turn_length_m"])
    out["route_budget_m"] = budget
    out["ego_edge_id"] = str(ego_edge_id)

    return {
        "dual_path": out,
        "baseline_destination_lane_id": base.dest_lane_id,
        "baseline_destination_edge_id": base.dest_edge_id,
        "baseline_destination_max_along_m": base.destination_max_along_m,
        "compliant_destination_lane_id": comp.dest_lane_id,
        "compliant_destination_edge_id": comp.dest_edge_id,
        "compliant_destination_max_along_m": comp.destination_max_along_m,
        "destination_lane_id": comp.dest_lane_id,
        "destination_edge_id": comp.dest_edge_id,
        "turn_length_m": out["turn_length_m"],
        "straight_length_m": out["straight_length_m"],
        "dual_path_gain_m": out["gain_m"],
        "dual_path_route_budget_m": budget,
        "baseline_truncated": base.truncated,
        "compliant_truncated": comp.truncated,
    }
