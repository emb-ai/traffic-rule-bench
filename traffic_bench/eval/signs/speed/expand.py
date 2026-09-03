"""Expand straight segment scenes into speed-family manifest rows.

Layout on one edge (user constraints):
  - ego at the start of the road (spawn_offset_from_start)
  - start sign at spawn + braking/accel approach
  - destination at min(edge_end, max_path_length_m)  (default cap 150 m)
  - paired end-of-limit sign just before dest (3.24/5.21/5.31)

Axes: spawn_lane × profile. Each row samples its own traffic density and its
own sign offset; the plate value stays one per map (round-robin).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from traffic_bench.eval.signs.speed.spec import (
    BRAKE_DECEL_MPS2_DEFAULT,
    BRAKE_DELAY_S_DEFAULT,
    BRAKE_MARGIN_M_DEFAULT,
    ZONE_MIN_M,
    ZONE_TAIL_M,
    accel_v0_mps,
    approach_m,
    assign_limit_kmh,
    braking_v0_mps,
    edge_speed_mps,
    new_limit_state,
    paired_end_code,
    spawn_mode_for,
)
from traffic_bench.eval.engine.expand.manifest_expansion import shuffle_cap
from traffic_bench.eval.engine.traffic.traffic_density_levels import (
    density_quantiles,
    sample_traffic_density,
)

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
    # Extra metres the plate may slide DOWNSTREAM of its earliest legal position.
    # Never upstream: the approach before it is the braking (or accel) distance
    # the scene is built on, and shortening it would make the scene unsatisfiable
    # rather than varied.
    sign_jitter_max_m: float = 25.0
    # NPC/approach profiles per lane x density cell. The seed folds the variant
    # in, so each one draws its own nuPlan spawn velocity -- and for the braking
    # families that moves the approach distance and the sign with it.
    n_variations: int = 10


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
    scene_id = f"{scene_name}_l{spawn_lane_num}_v{variant}"
    seed = _stable_seed(scene_name, variant, seed_key)
    traffic_density = sample_traffic_density(seed)

    if spawn_mode == "accel":
        v0 = accel_v0_mps(v_target_kmh, seed=seed)
    else:
        v0 = braking_v0_mps(seed, v_target_kmh)
    d_req = approach_m(pdd_code, v0, v_target_kmh)
    sign_s = spawn_offset + d_req

    end_code = paired_end_code(pdd_code)
    s_end = dest_along - float(sim.zone_tail_m) if end_code else dest_along
    if s_end - sign_s < float(sim.zone_min_m):
        return None

    # Slide the plate downstream within whatever room the zone can spare, so the
    # profiles of one cell do not all put the sign at the same metre.
    room = (s_end - float(sim.zone_min_m)) - sign_s
    if room > 0.0:
        jitter = random.Random(seed ^ 0x516E4A).random() * min(
            float(sim.sign_jitter_max_m), room)
        sign_s += jitter

    if sign_s <= spawn_offset + 0.5:
        return None

    spawn_lane_id = f"{road_id}_{spawn_lane_num}"

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
        "traffic_density": float(traffic_density),
        # Kept as None so readers that group by them collapse to one group
        # instead of raising: a scene samples its density, it has no tier.
        "traffic_density_level_id": None,
        "traffic_density_level_name": None,
        "nuplan_vehicles_per_frame": None,
        "sign_s_earliest": round(float(spawn_offset + d_req), 3),
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
    n_variations = max(1, int(sim.n_variations))

    entries: List[Dict[str, Any]] = []
    for lane_num in _lane_range(meta, sim.max_ego_lanes):
        for variant in range(n_variations):
            row = build_speed_manifest_entry(
                scene_dir=scene_dir,
                scenes_root=scenes_root,
                meta=meta,
                sim=sim,
                pdd_code=pdd_code,
                v_target_kmh=v_target_kmh,
                spawn_lane_num=lane_num,
                variant=variant,
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
            "speed_lane_density_profile_cap",
            int(max_sc) if max_sc is not None else 0,
        ),
    )
    if max_sc is not None and pre_cap > max_sc:
        print(
            f"  Retained {len(entries)} of {pre_cap} lane×profile "
            f"variants (shuffled, cap={max_sc})"
        )
    return entries

from traffic_bench.eval.manifest.io import (
    append_scene_entries,
    apply_max_total,
    apply_split_filter,
    load_scene_metadata,
    write_real_manifest,
)
from traffic_bench.scene_collection.sign_scenes.materialize.pool_index import normalize_split


def generate(cfg, scenes=None):
    """Speed-family rows from segment scenes."""

    profile = cfg.profile
    PDD_CODE = profile.pdd_code
    SIGN_TYPE = profile.sign_type
    SIGN_NAME = profile.sign_name
    def _profile():
        return profile
    scenes_dir = cfg.scenes_dir
    output_dir = cfg.output_dir
    scenario_cfg = cfg.scenario
    sim_cfg = cfg.simulation
    expansion_cfg = cfg.expansion
    aux_cfg = cfg.auxiliary
    expert_cfg = cfg.expert
    split = cfg.split

    split = normalize_split(split)
    max_density_levels = int(cfg.max_density_levels)
    traffic_density_augment = bool(cfg.traffic_density_augment)
    pdd_code = PDD_CODE
    all_scenes = discover_segment_speed_scenes(scenes_dir)
    print(f"Scenes root: {scenes_dir.resolve()}")
    print(f"Discovered {len(all_scenes)} segment scene(s) for {pdd_code}")
    scenes, split_by_id = apply_split_filter(
        all_scenes, scenes_dir=scenes_dir, split=split
    )
    print(
        f"Augmentation axes: spawn_lane × profiles="
        f"{int(getattr(sim_cfg, 'n_variations', 10) or 10)}; each row samples its "
        f"own traffic density and sign offset "
        f"(density quantiles {density_quantiles()})"
    )

    sim_params = SpeedSimParams(
        spawn_offset_from_start=float(sim_cfg.spawn_offset_from_start),
        max_path_length_m=float(sim_cfg.max_path_length_m),
        horizon=int(sim_cfg.horizon),
        traffic_density=float(sim_cfg.traffic_density),
        traffic_density_augment=bool(traffic_density_augment),
        max_density_levels=int(max_density_levels),
        max_ego_lanes=int(sim_cfg.max_ego_lanes),
        zone_tail_m=float(sim_cfg.zone_tail_m),
        zone_min_m=float(sim_cfg.zone_min_m),
        n_variations=int(getattr(sim_cfg, "n_variations", 10) or 10),
    )
    speed_expansion = SpeedExpansionConfig(
        max_scenarios=scenario_cfg.max_scenarios,
    )

    entries: List[Dict] = []
    used_scene_ids: List[str] = []
    skipped_short = 0
    skipped_road = 0
    limit_state = new_limit_state()

    for scene_dir in scenes:
        meta = load_scene_metadata(scene_dir)
        scene_name = meta.get("scene_name", scene_dir.name)
        net_abs = scene_dir / str(meta.get("net_file") or "map.net.xml")
        road_kmh = edge_speed_mps(str(net_abs), str(meta.get("road_id") or "")) * 3.6
        v_target_kmh = assign_limit_kmh(
            pdd_code, road_speed_kmh=road_kmh, state=limit_state
        )
        if v_target_kmh is None:
            skipped_road += 1
            print(f"\n=== {scene_name}  skipped: road {road_kmh:.0f} km/h "
                  f"cannot carry a {pdd_code} plate ===")
            continue
        print(f"\n=== {scene_name}  v_target={v_target_kmh:.0f} km/h "
              f"(road {road_kmh:.0f} km/h) ===")

        scene_entries = expand_speed_scene_entries(
            scene_dir=scene_dir,
            scenes_root=scenes_dir,
            meta=meta,
            sim=sim_params,
            expansion=speed_expansion,
            pdd_code=pdd_code,
            v_target_kmh=v_target_kmh,
        )
        if not scene_entries:
            skipped_short += 1
            print(f"  Skipping {scene_name}: no room for sign+zone before dest cap")
            continue
        append_scene_entries(
            entries, used_scene_ids, scene_entries,
            scene_dir=scene_dir, meta=meta, split_by_id=split_by_id,
        )

    print(f"\nDropped for road speed: {skipped_road}; for no room before the dest cap: {skipped_short}")
    if pdd_code == "4.6":
        buckets = ", ".join(f"{int(k)}:{v}" for k, v in
                            sorted(limit_state["min_counts"].items()))
        print(f"4.6 minimum split over reachable buckets: {buckets}")

    entries, used_scene_ids, pre_total = apply_max_total(
        entries, used_scene_ids,
        max_total=scenario_cfg.max_total, split=split, pdd_code=pdd_code,
        scene_id_key="scene_name",
    )
    write_real_manifest(
        output_dir=output_dir,
        scenes_dir=scenes_dir,
        entries=entries,
        used_scene_ids=used_scene_ids,
        split_by_id=split_by_id,
        split=split,
        pdd_code=pdd_code,
        summary={
            "pdd_code": pdd_code,
            "sign_type": SIGN_TYPE,
            "sign_name": SIGN_NAME,
            "sign_class": {
                "3.24": "SpeedLimitSign",
                "4.6": "MinimumSpeedLimitSign",
                "5.21": "ResidentialZoneSign",
                "5.31": "ZoneSpeedLimitSign",
            }.get(pdd_code, "SpeedLimitSign"),
            "sign_placement": (
                f"ego at road start; start sign at spawn+approach; "
                f"dest min(edge_end, {sim_cfg.max_path_length_m}m)"
            ),
            "total_scenes": len(used_scene_ids),
            "total_entries": len(entries),
            "total_entries_before_max_total": pre_total,
            "skipped_short_scenes": skipped_short,
            "skipped_road_speed_scenes": skipped_road,
            "max_scenarios": scenario_cfg.max_scenarios,
            "max_total": scenario_cfg.max_total,
            "max_density_levels": max_density_levels,
            "traffic_density_augment": bool(traffic_density_augment),
            "spawn_offset_from_start": sim_cfg.spawn_offset_from_start,
            "max_path_length_m": sim_cfg.max_path_length_m,
            "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
            "horizon": sim_cfg.horizon,
            "auxiliary_agent": False,
        },
    )
    return entries

