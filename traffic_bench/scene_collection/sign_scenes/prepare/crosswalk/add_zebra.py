"""Inject a mid-block zebra into copied segment scenes (PDD 5.19), in place."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from traffic_bench.eval.core.layout.crosswalk_layout import net_has_crossings
from traffic_bench.scene_collection.sign_scenes.filter.selection import is_reserved_scene_dir
from traffic_bench.scene_collection.sign_scenes.prepare.crosswalk.inject import (
    CrosswalkInjection,
    calculate_crosswalk_positions,
    find_paired_edges,
    identify_main_edges,
    inject_crosswalk,
)


def _load_meta(scene_dir: Path) -> Dict[str, Any]:
    return json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))


def _edge_ids_and_length(scene_dir: Path, meta: Dict[str, Any]) -> tuple[tuple[str, ...], float]:
    source_net = scene_dir / str(meta.get("net_file") or "map.net.xml")
    edges = identify_main_edges(source_net)
    if not edges:
        raise RuntimeError(f"{scene_dir.name}: no main edges")
    road_id = str(meta.get("road_id") or "")
    target = next((e for e in edges if e["edge_id"] in {road_id, f"-{road_id}"}), None)
    if target is None:
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
    reverse_id = f"-{road_id}" if not road_id.startswith("-") else road_id[1:]
    if any(e["edge_id"] == reverse_id for e in edges):
        return (road_id, reverse_id), float(target["length_m"])
    return (road_id,), float(target["length_m"])


def add_zebra_in_place(scene_dir: Path) -> str:
    """Write a middle-of-segment zebra into ``scene_dir/map.net.xml``. Returns status."""
    meta_path = scene_dir / "meta.json"
    net_path = scene_dir / "map.net.xml"
    if not meta_path.is_file() or not net_path.is_file():
        return "skip"
    if net_has_crossings(net_path):
        _refresh_preview(scene_dir)
        return "skip"
    meta = _load_meta(scene_dir)
    edge_ids, edge_length = _edge_ids_and_length(scene_dir, meta)
    pos_m = float(calculate_crosswalk_positions(edge_length, ["middle"])["middle"])
    injection = CrosswalkInjection(
        source_net=net_path,
        crosswalk_position_m=pos_m,
        edge_ids=edge_ids,
        priority=True,
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_net = Path(tmp) / "map.net.xml"
        result = inject_crosswalk(injection, tmp_net)
        if not result.success:
            raise RuntimeError(result.error or f"zebra inject failed for {scene_dir.name}")
        shutil.copy2(tmp_net, net_path)
    meta["scene_kind"] = "segment_crosswalk"
    meta["crosswalk_position"] = "middle"
    meta["crosswalk_position_m"] = pos_m
    meta["crosswalk_node_id"] = result.crosswalk_node_id
    meta["crosswalk_edge_id"] = result.crosswalk_edge_id
    meta["crossed_edge_ids"] = list(result.crossed_edge_ids or ())
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
    """Walk live scene dirs under a sign folder and inject a middle zebra."""
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
