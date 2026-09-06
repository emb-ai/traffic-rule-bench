"""Expand segment scenes into manifest rows for the reserved-lane plates.

5.14.1 / 5.14.2 (bus / bicycle lane) and 5.11.1 / 5.11.2 (road with a bus /
bicycle lane). The same multi-lane segment crops as the detour family: the
rightmost vehicle lane (SUMO index 0) becomes the reserved lane from the plate
on, the ego spawns on that lane before the plate and has to move to a
neighbouring lane before the zone starts. A baseline that stays in its lane
collects a violation on every zone step; the rule expert changes lane ahead of
the plate (``SignComplianceMixin._handle_restricted_lane``).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from traffic_bench.eval.engine.expand.manifest_expansion import shuffle_cap
from traffic_bench.eval.engine.spawn.route_budget import measure_spawn_to_dest_length_m
from traffic_bench.eval.engine.spawn.route_length_levels import (
    list_route_length_levels,
    select_route_length_levels,
)
from traffic_bench.eval.engine.traffic.agent_profile_bank import sample_one_profile
from traffic_bench.eval.engine.traffic.npc_profile import embed_npc_profile
from traffic_bench.eval.engine.traffic.stable_hash import stable_hash
from traffic_bench.eval.engine.traffic.traffic_density_levels import (
    density_quantiles,
    sample_traffic_density,
)
from traffic_bench.eval.manifest.io import (
    append_scene_entries,
    apply_max_total,
    apply_split_filter,
    load_scene_metadata,
    write_real_manifest,
)
from traffic_bench.scene_collection.sign_scenes.filter.selection import is_reserved_scene_dir
from traffic_bench.scene_collection.sign_scenes.materialize.pool_index import normalize_split

MAX_AXIS = 3

# PDD code → runtime plate class (traffic_bench.signs.extra.restricted_lane).
SIGN_CLASS_BY_CODE = {
    "5.14.1": "BusLaneSign",
    "5.14.2": "BikeLaneSign",
    "5.11.1": "BusLaneRoadSign",
    "5.11.2": "BikeLaneRoadSign",
}
RESTRICTED_LANE_CODES = frozenset(SIGN_CLASS_BY_CODE)
# The checker (RestrictedLaneSign._is_violating) only fires on SUMO lane 0,
# the rightmost lane, so that is the reserved lane by construction.
RESERVED_LANE_INDEX = 0


@dataclass(frozen=True)
class RestrictedLaneSimParams:
    spawn_offset_from_start: float = 10.0
    max_path_length_m: float = 150.0
    max_path_length_levels: Tuple[float, ...] = (150.0,)
    # Length of the reserved-lane zone after the plate. The violation is
    # judged inside it, so the whole zone has to fit on the edge and the
    # destination has to lie past its end.
    zone_m: float = 60.0
    # Room past the zone end kept for the finish line.
    tail_after_zone_m: float = 10.0
    # How far before the plate the ego starts (the expert pre-empts at 50 m,
    # so the run-up must be at least that long to make the manoeuvre legal).
    approach_before_sign_m: float = 60.0
    spawn_velocity_ms: float = 5.0
    horizon: int = 600
    traffic_density: float = 0.0
    n_variations: int = MAX_AXIS
    profile_density_cap: float = 1.0
    # Upstream slide of the plate on the sampled variants (never downstream:
    # the nominal position is the last metre that leaves room for the zone).
    sign_jitter_m: float = 15.0
    default_first_variant: bool = False


@dataclass(frozen=True)
class RestrictedLaneExpansionConfig:
    max_scenarios: Optional[int] = None


def _load_meta(scene_dir: Path) -> Dict[str, Any]:
    return json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))


def _is_segment_meta(meta: Dict[str, Any]) -> bool:
    return str(meta.get("scene_kind") or "") in {"segment", "segment_detour"}


def _vehicle_lane_indices(meta: Dict[str, Any]) -> List[int]:
    raw = meta.get("vehicle_lane_indices")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw.replace("'", '"'))
        except json.JSONDecodeError:
            raw = []
    try:
        return sorted(int(i) for i in (raw or []))
    except (TypeError, ValueError):
        return []


def scene_supports_restricted_lane(meta: Dict[str, Any]) -> bool:
    """Two or more vehicle lanes, and the rightmost one is SUMO lane 0."""
    lanes = _vehicle_lane_indices(meta)
    return len(lanes) >= 2 and lanes[0] == RESERVED_LANE_INDEX


def discover_restricted_lane_scenes(scenes_root: Path) -> List[Path]:
    scenes: List[Path] = []
    if not scenes_root.is_dir():
        return scenes
    for child in sorted(scenes_root.iterdir()):
        if not child.is_dir() or is_reserved_scene_dir(child.name):
            continue
        if not (child / "meta.json").is_file() or not (child / "map.net.xml").is_file():
            continue
        meta = _load_meta(child)
        if not _is_segment_meta(meta):
            continue
        if not scene_supports_restricted_lane(meta):
            print(f"  [skip] {child.name}: needs >=2 vehicle lanes with lane 0 drivable")
            continue
        scenes.append(child)
    return scenes


def _stable_seed(scene_name: str, variant: int, key: str = "") -> int:
    h = hashlib.sha256()
    h.update(scene_name.encode("utf-8"))
    h.update(b"|")
    h.update(str(variant).encode("utf-8"))
    h.update(b"|")
    h.update(key.encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")


def build_restricted_lane_entry(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    sim: RestrictedLaneSimParams,
    pdd_code: str,
    sign_type: str = "restricted_lane",
    variant: int = 0,
    npc_profile: Optional[Dict[str, Any]] = None,
    max_path_length_m: Optional[float] = None,
    route_length_augment: bool = False,
    default_variant: bool = False,
) -> Dict[str, Any]:
    """One manifest row: plate at ``sign_s`` on lane 0, zone ``[sign_s, sign_s+zone]``."""
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    net_file = str(meta.get("net_file") or "map.net.xml")
    net_path = scene_dir.relative_to(scenes_root) / net_file

    road_id = str(meta.get("road_id") or "")
    lane_index = RESERVED_LANE_INDEX
    edge_length = float(meta.get("length_m", 200.0))
    zone_m = float(sim.zone_m)

    # Nominal plate: as late as the edge allows while the whole zone plus the
    # finish room still fit; never earlier than the run-up needs.
    tail = zone_m + float(sim.tail_after_zone_m)
    sign_s = max(20.0, edge_length - tail)
    jitter = 0.0 if default_variant else float(sim.sign_jitter_m)
    if jitter > 0.0:
        room = min(jitter, max(0.0, sign_s - 20.0))
        if room > 0.0:
            jitter_seed = _stable_seed(scene_name, variant, "sign_jitter")
            sign_s -= random.Random(jitter_seed ^ 0x52534C).random() * room
    zone_end = min(edge_length - 1.0, sign_s + zone_m)

    spawn_lane_id = f"{road_id}_{lane_index}"
    spawn_offset = float(sim.spawn_offset_from_start)
    approach = float(sim.approach_before_sign_m)
    if sign_s - approach > spawn_offset:
        spawn_offset = sign_s - approach
    path_budget_m = float(
        max_path_length_m if max_path_length_m is not None else sim.max_path_length_m
    )
    spawn_before_end = max(20.0, edge_length - spawn_offset)
    # Finish past the zone end (the verdict needs the whole zone), within the
    # route budget when the budget allows it, never past the edge.
    dest_along = min(spawn_offset + path_budget_m, max(spawn_offset + 1.0, edge_length - 5.0))
    dest_along = max(dest_along, min(zone_end + float(sim.tail_after_zone_m) * 0.5, edge_length - 5.0))

    seed_key = f"npc{variant}"
    scene_id = f"{scene_name}_v{variant}"
    if route_length_augment:
        seed_key += f"_rl{int(round(path_budget_m))}"
        scene_id = f"{scene_id}_rl{int(round(path_budget_m))}"
    seed = _stable_seed(scene_name, variant, seed_key)
    traffic_density = 0.0 if default_variant else sample_traffic_density(seed)

    row: Dict[str, Any] = {
        "valid": True,
        "scene_id": scene_id,
        "scene_name": scene_name,
        "pdd_code": pdd_code,
        "sign_code": pdd_code,
        "sign_type": sign_type,
        "sign_class": SIGN_CLASS_BY_CODE.get(pdd_code, "BusLaneSign"),
        "place_restricted_lane_sign": True,
        "net_path": str(net_path),
        "seed": seed,
        "deterministic_seed": seed,
        "var_idx": variant,
        "road_id": road_id,
        "spawn_lane_id": spawn_lane_id,
        "destination_lane_id": spawn_lane_id,
        "destination_edge_id": road_id,
        "spawn_lane_num": lane_index,
        "sign_lane_index": lane_index,
        "restricted_lane_index": lane_index,
        "sign_s": sign_s,
        "zone_length_m": zone_m,
        "zone_end_s": zone_end,
        "spawn_distance_before_end": spawn_before_end,
        "destination_max_along_m": dest_along,
        "max_path_length_m": path_budget_m,
        "route_length_level_m": path_budget_m,
        "spawn_velocity_ms": float(sim.spawn_velocity_ms),
        "spawn_offset_from_start": spawn_offset,
        "traffic_density": traffic_density,
        "horizon": int(sim.horizon),
        "horizon_steps": int(sim.horizon),
        "auxiliary_agent": False,
        "latitude": meta.get("latitude"),
        "longitude": meta.get("longitude"),
        "segment_type": meta.get("segment_type"),
        "osm_way_id": meta.get("osm_way_id"),
        "junction_id": meta.get("junction_id"),
        "lane_count": meta.get("lane_count"),
        "vehicle_lane_indices": _vehicle_lane_indices(meta),
        "edge_length_m": edge_length,
    }
    if npc_profile is not None:
        row = embed_npc_profile(row, npc_profile, density_cap=float(sim.profile_density_cap))
    return row


def expand_restricted_lane_scene_entries(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    sim: RestrictedLaneSimParams,
    expansion: RestrictedLaneExpansionConfig,
    pdd_code: str,
    sign_type: str = "restricted_lane",
) -> List[Dict[str, Any]]:
    n_variations = max(1, int(sim.n_variations))
    entries: List[Dict[str, Any]] = []
    configured_route_levels = list_route_length_levels(sim)
    road_id = str(meta.get("road_id") or "")
    net_full = scene_dir / str(meta.get("net_file") or "map.net.xml")
    available_route_m = None
    if road_id:
        available_route_m = measure_spawn_to_dest_length_m(
            net_path=net_full,
            spawn_edge=road_id,
            spawn_lane=RESERVED_LANE_INDEX,
            dest_edge=road_id,
            spawn_along_m=float(sim.spawn_offset_from_start),
        )
    route_levels, route_augment = select_route_length_levels(
        configured_route_levels, available_route_m
    )
    for npc_var in range(n_variations):
        nominal = bool(sim.default_first_variant) and npc_var == 0
        levels = [float(sim.max_path_length_m)] if nominal else route_levels
        for path_len_m in levels:
            seed = stable_hash(
                str(meta.get("scene_name") or scene_dir.name),
                npc_var,
                int(round(float(path_len_m))),
            )
            npc_profile = None if nominal else sample_one_profile(
                int(seed),
                density_cap=float(sim.profile_density_cap),
                horizon_steps=int(sim.horizon),
            )
            entries.append(
                build_restricted_lane_entry(
                    default_variant=nominal,
                    scene_dir=scene_dir,
                    scenes_root=scenes_root,
                    meta=meta,
                    sim=sim,
                    pdd_code=pdd_code,
                    sign_type=sign_type,
                    variant=npc_var,
                    npc_profile=npc_profile,
                    max_path_length_m=float(path_len_m),
                    route_length_augment=route_augment,
                )
            )

    max_sc = expansion.max_scenarios
    pre_cap = len(entries)
    entries = shuffle_cap(
        entries,
        max_sc,
        seed_key=(str(scene_dir.name), "restricted_lane_npc_cap", int(max_sc) if max_sc is not None else 0),
    )
    if max_sc is not None and pre_cap > max_sc:
        print(f"  Retained {len(entries)} of {pre_cap} NPC variants (shuffled, cap={max_sc})")
    return entries


def generate(cfg, scenes=None):
    """Reserved-lane rows (5.14.1/2, 5.11.1/2) from multi-lane segment scenes."""
    profile = cfg.profile
    PDD_CODE = profile.pdd_code
    SIGN_TYPE = profile.sign_type
    SIGN_NAME = profile.sign_name
    scenes_dir = cfg.scenes_dir
    output_dir = cfg.output_dir
    scenario_cfg = cfg.scenario
    sim_cfg = cfg.simulation
    split = normalize_split(cfg.split)
    n_variations = max(1, int(getattr(sim_cfg, "n_variations", 3) or 3))

    all_scenes = discover_restricted_lane_scenes(scenes_dir)
    print(f"Scenes root: {scenes_dir.resolve()}")
    print(f"Discovered {len(all_scenes)} multi-lane segment scene(s) for {PDD_CODE}")
    scenes, split_by_id = apply_split_filter(all_scenes, scenes_dir=scenes_dir, split=split)
    zone_m = float(getattr(sim_cfg, "restricted_zone_m", 60.0) or 60.0)
    print(
        f"Augmentation axes: n_variations={n_variations} "
        f"× route_length={list(getattr(sim_cfg, 'max_path_length_levels', (150.0,)))}; "
        f"reserved lane {RESERVED_LANE_INDEX}, zone {zone_m:.0f} m after the plate; "
        f"each row samples its own traffic density and plate offset "
        f"(density quantiles {density_quantiles()})"
    )

    sim_params = RestrictedLaneSimParams(
        spawn_offset_from_start=float(sim_cfg.spawn_offset_from_start),
        max_path_length_m=float(sim_cfg.max_path_length_m),
        max_path_length_levels=tuple(
            float(x) for x in getattr(sim_cfg, "max_path_length_levels", (150.0,))
        ),
        zone_m=zone_m,
        tail_after_zone_m=float(getattr(sim_cfg, "tail_after_sign_m", 10.0) or 10.0),
        approach_before_sign_m=float(getattr(sim_cfg, "approach_before_sign_m", 60.0) or 60.0),
        spawn_velocity_ms=float(sim_cfg.spawn_velocity_ms),
        horizon=int(sim_cfg.horizon),
        traffic_density=float(sim_cfg.traffic_density),
        n_variations=n_variations,
        profile_density_cap=float(getattr(sim_cfg, "profile_density_cap", 1.0) or 1.0),
        sign_jitter_m=float(getattr(sim_cfg, "sign_jitter_m", 15.0) or 0.0),
        default_first_variant=bool(getattr(sim_cfg, "default_first_variant", False)),
    )
    expansion = RestrictedLaneExpansionConfig(max_scenarios=scenario_cfg.max_scenarios)

    entries: List[Dict] = []
    used_scene_ids: List[str] = []
    for scene_dir in scenes:
        meta = load_scene_metadata(scene_dir)
        scene_name = meta.get("scene_name", scene_dir.name)
        print(f"\n=== {scene_name} ===")
        scene_entries = expand_restricted_lane_scene_entries(
            scene_dir=scene_dir,
            scenes_root=scenes_dir,
            meta=meta,
            sim=sim_params,
            expansion=expansion,
            pdd_code=PDD_CODE,
            sign_type=SIGN_TYPE,
        )
        if not scene_entries:
            print(f"  Skipping {scene_name}: no manifest entries")
            continue
        append_scene_entries(
            entries, used_scene_ids, scene_entries,
            scene_dir=scene_dir, meta=meta, split_by_id=split_by_id,
        )

    entries, used_scene_ids, pre_total = apply_max_total(
        entries, used_scene_ids,
        max_total=scenario_cfg.max_total, split=split, pdd_code=PDD_CODE,
        scene_id_key="scene_name",
    )
    write_real_manifest(
        output_dir=output_dir,
        scenes_dir=scenes_dir,
        entries=entries,
        used_scene_ids=used_scene_ids,
        split_by_id=split_by_id,
        split=split,
        pdd_code=PDD_CODE,
        summary={
            "pdd_code": PDD_CODE,
            "sign_type": SIGN_TYPE,
            "sign_name": SIGN_NAME,
            "sign_class": SIGN_CLASS_BY_CODE.get(PDD_CODE, "BusLaneSign"),
            "sign_placement": (
                f"{SIGN_CLASS_BY_CODE.get(PDD_CODE)} ({PDD_CODE}) on the rightmost lane at sign_s; "
                f"zone {zone_m:.0f} m after the plate; ego spawns on that lane "
                f"{sim_params.approach_before_sign_m:.0f} m before the plate and must move to a neighbouring lane"
            ),
            "total_scenes": len(used_scene_ids),
            "total_entries": len(entries),
            "total_entries_before_max_total": pre_total,
            "max_scenarios": scenario_cfg.max_scenarios,
            "max_total": scenario_cfg.max_total,
            "n_variations": n_variations,
            "profile_density_cap": sim_params.profile_density_cap,
            "npc_world": "engine.traffic.agent_profile_bank.sample_one_profile",
            "spawn_velocity_ms": sim_params.spawn_velocity_ms,
            "horizon": sim_params.horizon,
            "spawn_offset_from_start": sim_params.spawn_offset_from_start,
            "max_path_length_m": sim_params.max_path_length_m,
            "max_path_length_levels": list(sim_params.max_path_length_levels),
            "approach_before_sign_m": sim_params.approach_before_sign_m,
            "restricted_zone_m": zone_m,
            "tail_after_zone_m": sim_params.tail_after_zone_m,
            "auxiliary_agent": False,
        },
    )
    return entries
