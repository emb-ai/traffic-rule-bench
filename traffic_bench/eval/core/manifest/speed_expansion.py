"""Expand straight segment scenes into speed-family manifest rows.

Layout on one edge (user constraints):
  - ego at the start of the road (spawn_offset_from_start)
  - start sign at spawn + braking/accel approach
  - destination at min(edge_end, max_path_length_m)  (default cap 150 m)
  - paired end-of-limit sign just before dest (3.24/5.21/5.31)

Axes: spawn_lane × traffic_density. Limit is one per map (round-robin).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from traffic_bench.eval.core.scenarios.speed_scene_design import (
    BRAKE_DECEL_MPS2_DEFAULT,
    BRAKE_DELAY_S_DEFAULT,
    BRAKE_MARGIN_M_DEFAULT,
    ZONE_MIN_M,
    ZONE_TAIL_M,
    accel_v0_mps,
    approach_m,
    assign_limit_kmh,
    braking_v0_mps,
    paired_end_code,
    spawn_mode_for,
)
from .manifest_expansion import shuffle_cap
from .traffic_density_levels import TrafficDensityLevel, list_traffic_density_levels

MAX_AXIS = 3

SIGN_CLASS = {
    "3.24": "SpeedLimitSign",
    "4.6": "MinimumSpeedLimitSign",
    "5.21": "ResidentialZoneSign",
    "5.31": "ZoneSpeedLimitSign",
}
END_SIGN_CLASS = {
    "3.25": "EndOfSpeedLimitSign",
    "5.22": "EndOfResidentialZoneSign",
    "5.32": "EndOfZoneSpeedLimitSign",
}


@dataclass(frozen=True)
class SpeedSimParams:
    spawn_offset_from_start: float = 5.0
    max_path_length_m: float = 150.0
    horizon: int = 400
    traffic_density: float = 0.0
    traffic_density_augment: bool = True
    max_density_levels: int = MAX_AXIS
    max_ego_lanes: int = 8
    zone_tail_m: float = ZONE_TAIL_M
    zone_min_m: float = ZONE_MIN_M


@dataclass(frozen=True)
class SpeedExpansionConfig:
    max_scenarios: Optional[int] = None


def _load_meta(scene_dir: Path) -> Dict[str, Any]:
    return json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))


def _is_segment_meta(meta: Dict[str, Any]) -> bool:
    return str(meta.get("scene_kind") or "") == "segment"


def discover_segment_speed_scenes(scenes_root: Path) -> List[Path]:
    """Find segment scene dirs (flat or under straight/curved)."""
    scenes: List[Path] = []
    if not scenes_root.is_dir():
        return scenes

    def _maybe_add(scene_dir: Path) -> None:
        if not (scene_dir / "meta.json").is_file():
            return
        if not (scene_dir / "map.net.xml").is_file():
            return
        if _is_segment_meta(_load_meta(scene_dir)):
            scenes.append(scene_dir)

    for child in sorted(scenes_root.iterdir()):
        if not child.is_dir():
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


def _lane_range(meta: Dict[str, Any], max_ego_lanes: int) -> List[int]:
    n = int(meta.get("lane_count") or 1)
    n = max(1, n)
    cap = max(1, int(max_ego_lanes))
    return list(range(min(n, cap)))


def build_speed_manifest_entry(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    sim: SpeedSimParams,
    pdd_code: str,
    v_target_kmh: float,
    spawn_lane_num: int,
    variant: int = 0,
    density_level: Optional[TrafficDensityLevel] = None,
) -> Optional[Dict[str, Any]]:
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    net_file = str(meta.get("net_file") or "map.net.xml")
    net_path = scene_dir.relative_to(scenes_root) / net_file

    road_id = str(meta.get("road_id") or "")
    edge_length = float(meta.get("length_m") or 0.0)
    if edge_length <= 0.0:
        return None

    spawn_offset = float(sim.spawn_offset_from_start)
    dest_along = min(
        float(sim.max_path_length_m),
        max(spawn_offset + 1.0, edge_length - 5.0),
    )
    spawn_before_end = max(20.0, edge_length - spawn_offset)

    spawn_mode = spawn_mode_for(pdd_code)
    seed_key = f"l{spawn_lane_num}"
    if density_level is not None:
        seed_key += f"_td{density_level.id}"
        scene_id = f"{scene_name}_l{spawn_lane_num}_td{density_level.id}_v{variant}"
    else:
        scene_id = f"{scene_name}_l{spawn_lane_num}_v{variant}"
    seed = _stable_seed(scene_name, variant, seed_key)

    if spawn_mode == "accel":
        v0 = accel_v0_mps(v_target_kmh)
    else:
        v0 = braking_v0_mps(seed, v_target_kmh)
    d_req = approach_m(pdd_code, v0, v_target_kmh)
    sign_s = spawn_offset + d_req

    end_code = paired_end_code(pdd_code)
    if end_code:
        s_end = dest_along - float(sim.zone_tail_m)
        if s_end - sign_s < float(sim.zone_min_m):
            return None
    else:
        s_end = dest_along
        if dest_along - sign_s < float(sim.zone_min_m):
            return None

    if sign_s <= spawn_offset + 0.5:
        return None

    spawn_lane_id = f"{road_id}_{spawn_lane_num}"
    traffic_density = (
        float(density_level.traffic_density)
        if density_level is not None
        else float(sim.traffic_density)
    )

    row: Dict[str, Any] = {
        "valid": True,
        "scene_id": scene_id,
        "scene_name": scene_name,
        "pdd_code": pdd_code,
        "sign_code": pdd_code,
        "sign_type": "speed",
        "sign_class": SIGN_CLASS.get(pdd_code, "SpeedLimitSign"),
        "place_speed_sign": True,
        "net_path": str(net_path),
        "seed": seed,
        "deterministic_seed": seed,
        "var_idx": variant,
        "road_id": road_id,
        "spawn_lane_id": spawn_lane_id,
        "destination_lane_id": spawn_lane_id,
        "destination_edge_id": road_id,
        "spawn_lane_num": int(spawn_lane_num),
        "sign_lane_index": int(spawn_lane_num),
        "sign_s": round(float(sign_s), 3),
        "spawn_distance_before_end": spawn_before_end,
        "destination_max_along_m": dest_along,
        "max_path_length_m": float(sim.max_path_length_m),
        "spawn_offset_from_start": spawn_offset,
        "spawn_velocity_ms": round(float(v0), 4),
        "v_target_kmh": float(v_target_kmh),
        "d_required_m": round(float(d_req), 3),
        "braking_spawn": True,
        "spawn_mode": spawn_mode,
        "brake_decel_mps2": BRAKE_DECEL_MPS2_DEFAULT,
        "brake_delay_s": BRAKE_DELAY_S_DEFAULT,
        "brake_margin_m": BRAKE_MARGIN_M_DEFAULT,
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
        "segment_type": meta.get("segment_type"),
        "osm_way_id": meta.get("osm_way_id"),
        "junction_id": meta.get("junction_id"),
        "lane_count": meta.get("lane_count"),
        "edge_length_m": edge_length,
    }
    if end_code:
        row["is_paired"] = True
        row["sign_type_end"] = end_code
        row["sign_class_end"] = END_SIGN_CLASS.get(end_code, "EndOfSpeedLimitSign")
        row["s_end"] = round(float(s_end), 3)
        row["zone_length_m"] = round(float(s_end) - float(sign_s), 3)
    else:
        row["is_paired"] = False
        row["s_end"] = round(float(s_end), 3)
        row["zone_length_m"] = round(max(0.0, float(s_end) - float(sign_s)), 3)
    return row


def expand_speed_scene_entries(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    sim: SpeedSimParams,
    expansion: SpeedExpansionConfig,
    pdd_code: str,
    v_target_kmh: float,
) -> List[Dict[str, Any]]:
    density_levels: List[Optional[TrafficDensityLevel]]
    if sim.traffic_density_augment:
        density_levels = list(list_traffic_density_levels(int(sim.max_density_levels)))
    else:
        density_levels = [None]

    entries: List[Dict[str, Any]] = []
    for lane_num in _lane_range(meta, sim.max_ego_lanes):
        for density in density_levels:
            row = build_speed_manifest_entry(
                scene_dir=scene_dir,
                scenes_root=scenes_root,
                meta=meta,
                sim=sim,
                pdd_code=pdd_code,
                v_target_kmh=v_target_kmh,
                spawn_lane_num=lane_num,
                variant=0,
                density_level=density,
            )
            if row is not None:
                entries.append(row)

    max_sc = expansion.max_scenarios
    pre_cap = len(entries)
    entries = shuffle_cap(
        entries,
        max_sc,
        seed_key=(
            str(scene_dir.name),
            "speed_lane_density_cap",
            int(max_sc) if max_sc is not None else 0,
        ),
    )
    if max_sc is not None and pre_cap > max_sc:
        print(
            f"  Retained {len(entries)} of {pre_cap} lane×density variants "
            f"(shuffled, cap={max_sc})"
        )
    return entries
