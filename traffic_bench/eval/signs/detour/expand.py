"""Expand corridor segment scenes into manifest rows for PDD 4.2.x.

Harvested ``scene_kind: segment`` maps have ``road_id``, ``vehicle_lane_indices``,
and ``pass_right_ok`` / ``pass_left_ok``. Eval picks the obstacle lane from
those fields (4.2.1 = pass right, 4.2.2 = pass left, 4.2.3 = either).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from traffic_bench.eval.core.manifest.manifest_expansion import shuffle_cap
from traffic_bench.eval.core.manifest.traffic_density_levels import TrafficDensityLevel, list_traffic_density_levels
from traffic_bench.scene_collection.sign_scenes.filter.selection import is_reserved_scene_dir

MAX_AXIS = 3


@dataclass(frozen=True)
class DetourSimParams:
    spawn_offset_from_start: float = 10.0
    max_path_length_m: float = 100.0
    sign_distance_before_end: float = 12.0
    spawn_velocity_ms: float = 5.0
    horizon: int = 400
    traffic_density: float = 0.0
    traffic_density_augment: bool = True
    max_density_levels: int = MAX_AXIS


@dataclass(frozen=True)
class DetourExpansionConfig:
    max_scenarios: Optional[int] = None


def _load_meta(scene_dir: Path) -> Dict[str, Any]:
    return json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))


def _is_segment_detour_meta(meta: Dict[str, Any]) -> bool:
    kind = str(meta.get("scene_kind") or "")
    return kind in {"segment", "segment_detour"}


def _obstacle_lane_index(meta: Dict[str, Any], pdd_code: str) -> int:
    """Pick the blocked lane. SUMO index 0 is the rightmost vehicle lane."""
    raw = meta.get("sign_lane_index")
    if raw is not None and str(raw).strip() != "":
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    indices = sorted(int(i) for i in (meta.get("vehicle_lane_indices") or []))
    if not indices:
        return 0
    has_right = [i for i in indices if any(j < i for j in indices)]
    has_left = [i for i in indices if any(j > i for j in indices)]
    if pdd_code == "4.2.1":
        return max(has_right) if has_right else indices[-1]
    if pdd_code == "4.2.2":
        return min(has_left) if has_left else indices[0]
    both = [i for i in indices if i in has_right and i in has_left]
    if both:
        return both[len(both) // 2]
    if has_right:
        return max(has_right)
    if has_left:
        return min(has_left)
    return indices[0]


def discover_segment_detour_scenes(
    scenes_root: Path,
    *,
    detour_code: Optional[str] = None,
) -> List[Path]:
    """Find corridor scene dirs (flat, or leftover straight/curved nesting)."""
    scenes: List[Path] = []
    if not scenes_root.is_dir():
        return scenes

    def _maybe_add(scene_dir: Path) -> None:
        if not (scene_dir / "meta.json").is_file():
            return
        if not (scene_dir / "map.net.xml").is_file():
            return
        meta = _load_meta(scene_dir)
        if not _is_segment_detour_meta(meta):
            return
        tagged = str(meta.get("detour_code") or "")
        if detour_code is not None and tagged and tagged != detour_code:
            return
        scenes.append(scene_dir)

    for child in sorted(scenes_root.iterdir()):
        if not child.is_dir() or is_reserved_scene_dir(child.name):
            continue
        if child.name in {"straight", "curved"}:
            for scene_dir in sorted(child.iterdir()):
                if scene_dir.is_dir():
                    _maybe_add(scene_dir)
            continue
        _maybe_add(child)
    return scenes


def _stable_seed(scene_name: str, variant: int, key: str = "") -> int:
    h = hashlib.sha256()
    h.update(scene_name.encode("utf-8"))
    h.update(b"|")
    h.update(str(variant).encode("utf-8"))
    h.update(b"|")
    h.update(key.encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")


def build_detour_manifest_entry(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    sim: DetourSimParams,
    pdd_code: str,
    sign_type: str = "detour",
    variant: int = 0,
    density_level: Optional[TrafficDensityLevel] = None,
) -> Dict[str, Any]:
    """Build one manifest row for a detour scene."""
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    net_file = str(meta.get("net_file") or "map.net.xml")
    net_path = scene_dir.relative_to(scenes_root) / net_file

    road_id = str(meta.get("road_id") or "")
    sign_lane_index = _obstacle_lane_index(meta, pdd_code)
    edge_length = float(meta.get("length_m", 200.0))
    if meta.get("sign_s") is not None:
        sign_s = float(meta["sign_s"])
    else:
        sign_s = max(20.0, edge_length - float(sim.sign_distance_before_end))

    spawn_lane_id = f"{road_id}_{sign_lane_index}"
    spawn_offset = float(sim.spawn_offset_from_start)
    spawn_before_end = max(20.0, edge_length - spawn_offset)
    dest_along = min(
        spawn_offset + float(sim.max_path_length_m),
        max(spawn_offset + 1.0, edge_length - 5.0),
    )

    traffic_density = (
        float(density_level.traffic_density)
        if density_level is not None
        else float(sim.traffic_density)
    )
    if density_level is not None:
        seed_key = f"td{density_level.id}"
        scene_id = f"{scene_name}_td{density_level.id}_v{variant}"
    else:
        seed_key = "td0"
        scene_id = f"{scene_name}_v{variant}"
    seed = _stable_seed(scene_name, variant, seed_key)

    sign_class_map = {
        "4.2.1": "DetourRightSign",
        "4.2.2": "DetourLeftSign",
        "4.2.3": "DetourEitherSign",
    }
    sign_class = sign_class_map.get(pdd_code, "DetourRightSign")

    return {
        "valid": True,
        "scene_id": scene_id,
        "scene_name": scene_name,
        "pdd_code": pdd_code,
        "sign_code": pdd_code,
        "sign_type": sign_type,
        "sign_class": sign_class,
        "place_detour_sign": True,
        "net_path": str(net_path),
        "seed": seed,
        "deterministic_seed": seed,
        "var_idx": variant,
        "road_id": road_id,
        "spawn_lane_id": spawn_lane_id,
        "destination_lane_id": spawn_lane_id,
        "destination_edge_id": road_id,
        "spawn_lane_num": sign_lane_index,
        "sign_lane_index": sign_lane_index,
        "sign_s": sign_s,
        "detour_code": pdd_code,
        "spawn_distance_before_end": spawn_before_end,
        "destination_max_along_m": dest_along,
        "max_path_length_m": float(sim.max_path_length_m),
        "sign_distance_before_end": float(sim.sign_distance_before_end),
        "spawn_velocity_ms": float(sim.spawn_velocity_ms),
        "spawn_offset_from_start": spawn_offset,
        "traffic_density": traffic_density,
        "traffic_density_level_id": density_level.id if density_level is not None else None,
        "traffic_density_level_name": density_level.name if density_level is not None else None,
        "nuplan_vehicles_per_frame": (
            density_level.nuplan_vehicles_per_frame if density_level is not None else None
        ),
        "horizon": int(sim.horizon),
        "horizon_steps": int(sim.horizon),
        "auxiliary_agent": False,
        "latitude": meta.get("latitude"),
        "longitude": meta.get("longitude"),
        "source_segment_scene": meta.get("source_segment_scene"),
        "segment_type": meta.get("segment_type"),
        "osm_way_id": meta.get("osm_way_id"),
        "junction_id": meta.get("junction_id"),
        "lane_count": meta.get("lane_count"),
        "edge_length_m": edge_length,
        "valid_obstacle_lanes": meta.get("valid_obstacle_lanes"),
    }


def expand_detour_scene_entries(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    sim: DetourSimParams,
    expansion: DetourExpansionConfig,
    pdd_code: str,
    sign_type: str = "detour",
) -> List[Dict[str, Any]]:
    """Expand one segment_detour scene into manifest rows (density axis)."""
    density_levels: List[Optional[TrafficDensityLevel]]
    if sim.traffic_density_augment:
        density_levels = list(list_traffic_density_levels(int(sim.max_density_levels)))
    else:
        density_levels = [None]

    entries: List[Dict[str, Any]] = []
    for density in density_levels:
        entries.append(
            build_detour_manifest_entry(
                scene_dir=scene_dir,
                scenes_root=scenes_root,
                meta=meta,
                sim=sim,
                pdd_code=pdd_code,
                sign_type=sign_type,
                variant=0,
                density_level=density,
            )
        )

    max_sc = expansion.max_scenarios
    pre_cap = len(entries)
    entries = shuffle_cap(
        entries,
        max_sc,
        seed_key=(
            str(scene_dir.name),
            "detour_density_cap",
            int(max_sc) if max_sc is not None else 0,
        ),
    )
    if max_sc is not None and pre_cap > max_sc:
        print(
            f"  Retained {len(entries)} of {pre_cap} density variants "
            f"(shuffled, cap={max_sc})"
        )
    return entries
