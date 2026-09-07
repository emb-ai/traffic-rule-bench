"""Mark mid-block zebra position on copied segment scenes (PDD 5.19), in place.

Does **not** split the SUMO edge. A netconvert split left a white gap at the
junction and a separate depart stub after the zebra; MetaDrive top-down then
looked like the road was cut and a new piece inserted. The continuous edge is
kept; eval places the zebra geometrically and the manifest truncates the ego
path (``destination_max_along_m`` / route budget) like other signs.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from traffic_bench.eval.signs.crosswalk.spec import net_has_crossings
from traffic_bench.scene_collection.sign_scenes.filter.selection import is_reserved_scene_dir
from traffic_bench.scene_collection.sign_scenes.prepare.crosswalk.inject import (
    MIN_POSITION_FROM_END_M,
    MIN_POSITION_FROM_START_M,
    calculate_crosswalk_positions,
    find_paired_edges,
    identify_main_edges,
)

# Keep enough approach for eval spawn_distance_before_end (crosswalk.yaml: 50).
_MIN_APPROACH_FOR_SPAWN_M = 60.0


def _load_meta(scene_dir: Path) -> Dict[str, Any]:
    return json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))


def net_has_crosswalk_inject(net_path: Path) -> bool:
    """True if this net already has a mid-block zebra inject (legacy split)."""
    if net_has_crossings(net_path):
        return True
    if not net_path.is_file():
        return False
    root = ET.parse(net_path).getroot()
    return any(
        (j.get("id") or "").startswith("cw_node") for j in root.findall("junction")
    )


def _way_prefix(road_id: str, osm_way_id: Any) -> str:
    if osm_way_id not in (None, ""):
        return str(osm_way_id)
    rid = str(road_id or "").lstrip("-")
    return rid.split("#", 1)[0] if rid else ""


def _edge_ids_and_length(scene_dir: Path, meta: Dict[str, Any]) -> tuple[tuple[str, ...], float]:
    """Pick a long enough same-way edge to host the mid-corridor zebra mark."""
    source_net = scene_dir / str(meta.get("net_file") or "map.net.xml")
    edges = identify_main_edges(source_net)
    if not edges:
        raise RuntimeError(f"{scene_dir.name}: no main edges")
    road_id = str(meta.get("road_id") or "")
    prefix = _way_prefix(road_id, meta.get("osm_way_id"))
    min_len = _MIN_APPROACH_FOR_SPAWN_M + float(MIN_POSITION_FROM_END_M)

    same_way = [
        e
        for e in edges
        if prefix and e["edge_id"].lstrip("-").split("#", 1)[0] == prefix
    ]
    pool = same_way or list(edges)
    pool.sort(key=lambda e: float(e["length_m"]), reverse=True)

    target: Optional[Dict] = next(
        (e for e in pool if e["edge_id"] in {road_id, f"-{road_id}"}), None
    )
    if target is None or float(target["length_m"]) < min_len:
        long_enough = [e for e in pool if float(e["length_m"]) >= min_len]
        target = long_enough[0] if long_enough else (pool[0] if pool else None)

    if target is not None:
        edge_id = str(target["edge_id"])
        length = float(target["length_m"])
        reverse_id = f"-{edge_id}" if not edge_id.startswith("-") else edge_id[1:]
        if any(e["edge_id"] == reverse_id for e in edges):
            return (edge_id, reverse_id), length
        return (edge_id,), length

    pairs = find_paired_edges(edges)
    if pairs:
        edge_ids = pairs[0]
        length = next(
            (e["length_m"] for e in edges if e["edge_id"] == edge_ids[0]),
            max(e["length_m"] for e in edges),
        )
        return edge_ids, float(length)
    edges.sort(key=lambda e: e["length_m"], reverse=True)
    return (edges[0]["edge_id"],), float(edges[0]["length_m"])


def add_zebra_in_place(scene_dir: Path) -> str:
    """Record mid-corridor zebra meta without splitting the SUMO net."""
    meta_path = scene_dir / "meta.json"
    net_path = scene_dir / "map.net.xml"
    if not meta_path.is_file() or not net_path.is_file():
        return "skip"
    meta = _load_meta(scene_dir)
    # Already marked (no-split) or legacy inject present → leave net alone.
    if meta.get("crosswalk_position_m") is not None and not net_has_crosswalk_inject(net_path):
        if str(meta.get("scene_kind") or "") == "segment_crosswalk":
            _refresh_preview(scene_dir)
            return "skip"
    if net_has_crosswalk_inject(net_path):
        # Legacy split nets: do not stack another inject; just refresh preview.
        _refresh_preview(scene_dir)
        return "skip"

    edge_ids, edge_length = _edge_ids_and_length(scene_dir, meta)
    positions = calculate_crosswalk_positions(edge_length, ["middle"])
    if "middle" not in positions:
        raise RuntimeError(
            f"{scene_dir.name}: edge too short for middle zebra "
            f"(len={edge_length:.1f}m, need >{MIN_POSITION_FROM_START_M + MIN_POSITION_FROM_END_M:.0f}m)"
        )
    pos_m = float(positions["middle"])
    try:
        corridor_m = float(meta.get("length_m") or 0.0)
    except (TypeError, ValueError):
        corridor_m = 0.0
    if corridor_m > 0.0:
        pos_m = min(pos_m, corridor_m / 2.0)
    pos_m = max(pos_m, min(edge_length - MIN_POSITION_FROM_END_M, _MIN_APPROACH_FOR_SPAWN_M))
    pos_m = max(MIN_POSITION_FROM_START_M, min(edge_length - MIN_POSITION_FROM_END_M, pos_m))

    meta["scene_kind"] = "segment_crosswalk"
    meta["crosswalk_position"] = "middle"
    meta["crosswalk_position_m"] = pos_m
    meta["crosswalk_edge_id"] = edge_ids[0]
    meta["crossed_edge_ids"] = list(edge_ids)
    meta["road_id"] = edge_ids[0]
    # Explicitly clear legacy inject fields so expand uses the no-split path.
    meta.pop("crosswalk_node_id", None)
    meta["pdd_code"] = "5.19"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _refresh_preview(scene_dir)
    return "ok"


def _refresh_preview(scene_dir: Path) -> None:
    from traffic_bench.scene_collection.preview import render_scene_preview

    try:
        render_scene_preview(scene_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"  [preview-fail] {scene_dir.name}: {exc}")


def add_zebra_to_scenes_dir(scenes_dir: Path) -> Dict[str, int]:
    """Walk live scene dirs under a sign folder and mark a middle zebra."""
    stats = {"ok": 0, "skip": 0, "fail": 0}
    if not scenes_dir.is_dir():
        raise FileNotFoundError(scenes_dir)
    for child in sorted(scenes_dir.iterdir()):
        if not child.is_dir() or is_reserved_scene_dir(child.name):
            continue
        try:
            status = add_zebra_in_place(child)
        except Exception as exc:  # noqa: BLE001
            print(f"  [fail] {child.name}: {exc}")
            stats["fail"] += 1
            continue
        stats[status] = stats.get(status, 0) + 1
        print(f"  [{status}] {child.name}")
    return stats
