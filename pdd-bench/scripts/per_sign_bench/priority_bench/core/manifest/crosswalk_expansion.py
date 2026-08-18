"""Expand segment_crosswalk scenes into PDD 5.19 manifest rows.

Augmentation axes (each capped at ``max_*`` ≤ 3 by default):
  1. ego approach lane (from SUMO crossing approaches)
  2. traffic density (nuPlan low/medium/high via crosswalk_sign)
  3. crosswalk position (near_start / middle / near_end — separate maps)
  4. pedestrian presets (from crosswalk_sign/lib/pedestrian_presets.py)

Maps are prepared by moscow_scenes/scripts/prepare_segment_crosswalk.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PER_SIGN_BENCH = Path(__file__).resolve().parents[3]
_CROSSWALK_SIGN = _PER_SIGN_BENCH / "crosswalk_sign"
if str(_CROSSWALK_SIGN) not in sys.path:
    sys.path.insert(0, str(_CROSSWALK_SIGN))

from lib.crosswalk_layout import CrosswalkApproach, build_crosswalk_approaches  # noqa: E402
from lib.lane_keys import lane_edge_id, make_lane_key  # noqa: E402
from lib.pedestrian_presets import (  # noqa: E402
    PedestrianPreset,
    list_pedestrian_presets,
    pedestrian_manager_from_preset,
)
from lib.traffic_density_levels import (  # noqa: E402
    TrafficDensityLevel,
    list_traffic_density_levels,
)

from .manifest_expansion import shuffle_cap

MAX_AXIS = 3
DEFAULT_POSITIONS = ("near_start", "middle", "near_end")


@dataclass(frozen=True)
class CrosswalkSimParams:
    spawn_distance_before_end: float = 55.0
    sign_distance_before_end: float = 12.0
    spawn_velocity_ms: float = 2.5
    horizon: int = 600
    traffic_density: float = 0.0
    traffic_density_augment: bool = True
    min_hops_after_depart: int = 0
    destination_max_along_m: float = 40.0
    max_ego_lanes: int = MAX_AXIS
    max_density_levels: int = MAX_AXIS
    max_pedestrian_presets: int = MAX_AXIS
    crosswalk_positions: Tuple[str, ...] = DEFAULT_POSITIONS
    # Pedestrian defaults (mirrors crosswalk_sign)
    ped_ego_spawn_distance_m: float = 50.0
    ped_speed_mean: float = 1.2
    ped_speed_std: float = 0.2
    ped_spawn_gap_s: float = 2.5
    ped_yield_distance: float = 12.0
    ped_no_stop_before_crosswalk_m: float = 3.0


@dataclass(frozen=True)
class CrosswalkExpansionConfig:
    layout: bool = True
    max_scenarios: Optional[int] = None


def discover_segment_crosswalk_scenes(
    scenes_root: Path,
    *,
    positions: Sequence[str] = DEFAULT_POSITIONS,
) -> List[Path]:
    """Find segment_crosswalk scene dirs under straight/ and curved/."""
    want_pos = {str(p).strip() for p in positions if str(p).strip()}
    scenes: List[Path] = []
    if not scenes_root.is_dir():
        return scenes

    for type_dir in sorted(scenes_root.iterdir()):
        if not type_dir.is_dir():
            continue
        if type_dir.name not in {"straight", "curved"}:
            # Also allow flat layout (scene dirs directly under root)
            if (type_dir / "meta.json").is_file() and (type_dir / "map.net.xml").is_file():
                meta = _load_meta(type_dir)
                if _meta_matches_position(meta, type_dir.name, want_pos):
                    scenes.append(type_dir)
            continue
        for scene_dir in sorted(type_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if not (scene_dir / "meta.json").is_file():
                continue
            if not (scene_dir / "map.net.xml").is_file():
                continue
            meta = _load_meta(scene_dir)
            if not _is_segment_crosswalk_meta(meta):
                continue
            if not _meta_matches_position(meta, scene_dir.name, want_pos):
                continue
            scenes.append(scene_dir)
    return scenes


def _load_meta(scene_dir: Path) -> Dict[str, Any]:
    return json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))


def _is_segment_crosswalk_meta(meta: Dict[str, Any]) -> bool:
    kind = str(meta.get("scene_kind") or "")
    if kind in {"segment_crosswalk", "crosswalk"}:
        return True
    return bool(meta.get("crosswalk_edge_id") or meta.get("crosswalk_id") or meta.get("crosswalk_node_id"))


def _meta_matches_position(
    meta: Dict[str, Any],
    scene_name: str,
    want_pos: set[str],
) -> bool:
    if not want_pos:
        return True
    pos = str(meta.get("crosswalk_position") or "").strip()
    if pos and pos in want_pos:
        return True
    for p in want_pos:
        if scene_name.endswith(f"_cw_{p}"):
            return True
    # If meta has no position tag, keep the scene (legacy / single-position maps).
    if not pos and "_cw_" not in scene_name:
        return True
    return False


def _resolve_crosswalk_id(meta: Dict[str, Any]) -> Optional[str]:
    for key in ("crosswalk_edge_id", "crosswalk_id"):
        val = meta.get(key)
        if val:
            return str(val)
    node = meta.get("crosswalk_node_id")
    if node:
        return f":{node}_c0"
    return None


def _stable_seed(scene_name: str, variant: int, key: str) -> int:
    h = hashlib.sha256()
    h.update(scene_name.encode("utf-8"))
    h.update(b"|")
    h.update(str(variant).encode("utf-8"))
    h.update(b"|")
    h.update(key.encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")


def _limit_approaches(
    approaches: List[CrosswalkApproach],
    max_lanes: int,
) -> List[CrosswalkApproach]:
    """Keep up to ``max_lanes`` ego lanes, preferring longer approaches."""
    if max_lanes < 1 or len(approaches) <= max_lanes:
        return approaches
    ranked = sorted(
        approaches,
        key=lambda a: (-float(a.approach_lane_length), a.approach_edge_id, a.approach_lane_num),
    )
    return ranked[:max_lanes]


def build_crosswalk_manifest_entry(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    approach: CrosswalkApproach,
    preset: PedestrianPreset,
    density_level: Optional[TrafficDensityLevel],
    sim: CrosswalkSimParams,
    pdd_code: str = "5.19",
    sign_type: str = "crosswalk",
    variant: int = 0,
) -> Dict[str, Any]:
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    net_file = str(meta.get("net_file") or "map.net.xml")
    net_path = scene_dir.relative_to(scenes_root) / net_file

    ped_count = max(1, int(preset.target_pedestrian_count))
    traffic_density = (
        float(density_level.traffic_density)
        if density_level is not None
        else float(sim.traffic_density)
    )
    if density_level is not None:
        seed_key = f"{approach.scenario_id}_s{preset.id}_td{density_level.id}"
        scene_id = (
            f"{scene_name}_{approach.scenario_id}_s{preset.id}"
            f"_td{density_level.id}_v{variant}"
        )
    else:
        seed_key = f"{approach.scenario_id}_s{preset.id}"
        scene_id = f"{scene_name}_{approach.scenario_id}_s{preset.id}_v{variant}"

    seed = _stable_seed(scene_name, variant, seed_key)
    crosswalk_id = approach.crosswalk_id or _resolve_crosswalk_id(meta) or ""
    # Finish on the first edge past the crossing, not several hops downstream.
    dest_lane_id = make_lane_key(approach.depart_edge_id, approach.approach_lane_num)
    if not dest_lane_id:
        dest_lane_id = approach.destination_lane_id

    # Spawn far enough before the zebra that the crossing sits ahead of ego,
    # but never off the approach stub (near_start is ~40 m).
    approach_len = float(approach.approach_lane_length)
    spawn_before_end = min(
        float(sim.spawn_distance_before_end),
        max(20.0, approach_len - 8.0),
    )

    ped_mgr = pedestrian_manager_from_preset(
        preset,
        default_ego_spawn_distance_m=sim.ped_ego_spawn_distance_m,
        default_speed_mean=sim.ped_speed_mean,
        default_speed_std=sim.ped_speed_std,
        default_spawn_gap_s=sim.ped_spawn_gap_s,
        yield_distance=sim.ped_yield_distance,
        no_stop_before_crosswalk_m=sim.ped_no_stop_before_crosswalk_m,
    )
    # Preset 1 hardcodes 10 m; that puts the ped on the zebra as ego arrives.
    ped_mgr["ego_spawn_distance_m"] = max(
        float(ped_mgr.get("ego_spawn_distance_m") or 0.0),
        float(sim.ped_ego_spawn_distance_m),
    )

    return {
        "valid": True,
        "scene_id": scene_id,
        "scene_name": scene_name,
        "pdd_code": pdd_code,
        "sign_code": pdd_code,
        "sign_type": sign_type,
        "sign_class": "PedestrianCrossingSign",
        "place_crosswalk_sign": True,
        "net_path": str(net_path),
        "seed": seed,
        "deterministic_seed": seed,
        "var_idx": variant,
        "pedestrian_preset_id": preset.id,
        "pedestrian_preset_name": preset.name,
        "pedestrian_count": ped_count,
        "traffic_density_level_id": density_level.id if density_level is not None else None,
        "traffic_density_level_name": density_level.name if density_level is not None else None,
        "nuplan_vehicles_per_frame": (
            density_level.nuplan_vehicles_per_frame if density_level is not None else None
        ),
        "scenario_id": approach.scenario_id,
        "crosswalk_id": crosswalk_id,
        "crosswalk_position": meta.get("crosswalk_position"),
        "crosswalk_position_m": meta.get("crosswalk_position_m"),
        "crosswalk_node_id": meta.get("crosswalk_node_id"),
        "junction_id": approach.junction_id or meta.get("crosswalk_node_id"),
        "road_id": approach.approach_edge_id,
        "spawn_lane_num": approach.approach_lane_num,
        "depart_edge_id": approach.depart_edge_id,
        "destination_lane_id": dest_lane_id,
        "destination_edge_id": lane_edge_id(dest_lane_id),
        "min_hops_after_depart": sim.min_hops_after_depart,
        "spawn_distance_before_end": spawn_before_end,
        "sign_distance_before_end": sim.sign_distance_before_end,
        "destination_max_along_m": float(sim.destination_max_along_m),
        "spawn_velocity_ms": sim.spawn_velocity_ms,
        "traffic_density": traffic_density,
        "horizon": int(sim.horizon),
        "horizon_steps": int(sim.horizon),
        "use_pedestrian_manager": True,
        "use_pedestrian_yield_rule": True,
        "pedestrian_manager": ped_mgr,
        "auxiliary_agent": False,
        "approach_lane_length_m": approach.approach_lane_length,
        "crossed_edge_ids": list(approach.crossed_edge_ids),
        "latitude": meta.get("latitude"),
        "longitude": meta.get("longitude"),
        "source_segment_scene": meta.get("source_segment_scene"),
        "segment_type": meta.get("segment_type"),
        "osm_way_id": meta.get("osm_way_id"),
        "crosswalk_width_m": meta.get("crosswalk_width_m", 4.0),
    }


def expand_crosswalk_scene_entries(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    net_path: Path,
    sim: CrosswalkSimParams,
    expansion: CrosswalkExpansionConfig,
    pdd_code: str = "5.19",
    sign_type: str = "crosswalk",
) -> List[Dict[str, Any]]:
    """Expand one segment_crosswalk scene into manifest rows."""
    if not expansion.layout:
        return []

    approaches = build_crosswalk_approaches(
        net_path,
        # Approach stubs after mid-block split (esp. near_start) are ~40m;
        # require slightly less than spawn distance so those maps stay usable.
        min_approach_length=20.0,
        min_hops_after_depart=int(sim.min_hops_after_depart),
    )
    # Prefer approaches that match the injected crosswalk when available.
    target_cw = _resolve_crosswalk_id(meta)
    if target_cw:
        matched = [a for a in approaches if a.crosswalk_id == target_cw]
        if matched:
            approaches = matched
        else:
            # Match by junction/node id if edge id drifted after split.
            node = str(meta.get("crosswalk_node_id") or "")
            if node:
                matched = [a for a in approaches if node in a.crosswalk_id or a.junction_id == node]
                if matched:
                    approaches = matched

    approaches = _limit_approaches(approaches, int(sim.max_ego_lanes))
    if not approaches:
        print(f"  [crosswalk] no viable approaches in {scene_dir.name}")
        return []

    presets = list_pedestrian_presets(
        int(sim.max_pedestrian_presets),
        default_ego_spawn_distance_m=sim.ped_ego_spawn_distance_m,
        default_speed_mean=sim.ped_speed_mean,
        default_speed_std=sim.ped_speed_std,
        default_spawn_gap_s=sim.ped_spawn_gap_s,
    )
    density_levels: List[Optional[TrafficDensityLevel]]
    if sim.traffic_density_augment:
        density_levels = list(list_traffic_density_levels(int(sim.max_density_levels)))  # type: ignore[arg-type]
    else:
        density_levels = [None]

    print(
        f"  approaches={len(approaches)} presets={len(presets)} "
        f"density={len(density_levels)} pos={meta.get('crosswalk_position')}"
    )

    entries: List[Dict[str, Any]] = []
    for approach in approaches:
        for preset in presets:
            for density in density_levels:
                entries.append(
                    build_crosswalk_manifest_entry(
                        scene_dir=scene_dir,
                        scenes_root=scenes_root,
                        meta=meta,
                        approach=approach,
                        preset=preset,
                        density_level=density,
                        sim=sim,
                        pdd_code=pdd_code,
                        sign_type=sign_type,
                        variant=0,
                    )
                )

    max_sc = expansion.max_scenarios
    pre_cap = len(entries)
    entries = shuffle_cap(
        entries,
        max_sc,
        seed_key=(
            str(scene_dir.name),
            "crosswalk_combo_cap",
            int(max_sc) if max_sc is not None else 0,
        ),
    )
    if max_sc is not None and pre_cap > max_sc:
        print(
            f"  Retained {len(entries)} of {pre_cap} manifest entries "
            f"(shuffled, cap={max_sc})"
        )

    return entries
