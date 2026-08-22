"""Incoming-lane parse used by junction / blocked / dual-path generators."""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from traffic_bench.eval.core.sumo.sumo_utils import is_vehicle_drivable_lane


@dataclass
class SumoLaneInfo:
    """Information about a SUMO lane suitable for spawning."""

    edge_id: str
    lane_num: int
    lane_id: str
    length: float
    to_junction: str
    junction_type: str


def parse_sumo_net_for_spawn_lanes(
    net_path: Path, min_length: float = 20.0
) -> List[SumoLaneInfo]:
    """Parse SUMO ``.net.xml`` and find lanes that lead to intersections."""
    if not net_path.exists():
        return []

    tree = ET.parse(net_path)
    root = tree.getroot()

    junctions = {}
    for junction in root.findall("junction"):
        jid = junction.get("id")
        jtype = junction.get("type", "unknown")
        junctions[jid] = jtype

    intersection_types = {"priority", "right_before_left", "allway_stop", "traffic_light"}
    spawn_lanes = []

    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        func = edge.get("function", "normal")

        if func == "internal" or edge_id.startswith(":"):
            continue

        to_junction = edge.get("to", "")
        junction_type = junctions.get(to_junction, "unknown")

        if junction_type not in intersection_types:
            continue

        for lane in edge.findall("lane"):
            if not is_vehicle_drivable_lane(lane):
                continue
            lane_id = lane.get("id", "")
            length = float(lane.get("length", 0))

            if length == 0:
                shape_str = lane.get("shape", "")
                if shape_str:
                    points = shape_str.strip().split()
                    coords = [tuple(map(float, p.split(","))) for p in points if "," in p]
                    if len(coords) >= 2:
                        length = sum(
                            (
                                (coords[i + 1][0] - coords[i][0]) ** 2
                                + (coords[i + 1][1] - coords[i][1]) ** 2
                            )
                            ** 0.5
                            for i in range(len(coords) - 1)
                        )

            if length < min_length:
                continue

            try:
                lane_num = int(lane_id.rsplit("_", 1)[1])
            except (ValueError, IndexError):
                lane_num = 0

            spawn_lanes.append(
                SumoLaneInfo(
                    edge_id=edge_id,
                    lane_num=lane_num,
                    lane_id=f"lane_{lane_id}",
                    length=length,
                    to_junction=to_junction,
                    junction_type=junction_type,
                )
            )

    return spawn_lanes


def filter_spawn_lanes_to_secondary(
    spawn_lanes: List[SumoLaneInfo],
    junction_layout: Optional[dict],
) -> List[SumoLaneInfo]:
    """Keep only lanes on secondary junction arms (yield ego pool)."""
    if not junction_layout:
        return spawn_lanes
    secondary_ids = set(junction_layout.get("secondary_edge_ids") or [])
    if not secondary_ids:
        return []
    return [lane for lane in spawn_lanes if lane.edge_id in secondary_ids]


def select_random_spawn_lane(
    spawn_lanes: List[SumoLaneInfo],
    seed: int,
) -> Optional[SumoLaneInfo]:
    """Select a random lane from available spawn lanes."""
    if not spawn_lanes:
        return None
    rng = random.Random(seed)
    return rng.choice(spawn_lanes)
