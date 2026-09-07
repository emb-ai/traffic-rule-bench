"""Expand segment scenes into manifest rows for the reserved-lane plates.

5.14.1 / 5.14.2 (bus / bicycle lane) and 5.11.1 / 5.11.2 (road with a bus /
bicycle lane). Multi-lane segment crops: one lane becomes the reserved lane
from the plate on (5.14.x: the rightmost lane, with the flow; 5.11.x: the
leftmost lane of a one-way street, counter-flow), the ego spawns on that lane
before the plate and has to move to a neighbouring lane before the zone starts.
A baseline that stays in its lane collects a violation on every zone step; the
rule expert changes lane ahead of the plate
(``SignComplianceMixin._handle_restricted_lane``).

Sampling follows the shared world grid (``engine.expand.world_axes``: route
budget x traffic density probe x ego spawn speed x NPC profile variant) times
the family axes (zone length x approach length x number of lane users). All
combinations that fit the edge are enumerated, shuffled with a stable seed and
the first ``max_scenarios`` rows per map are built; the nominal row (no
background traffic, nominal geometry, one lane user, 5 m/s) is always first.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from traffic_bench.eval.engine.expand.manifest_config import (
    DEFAULT_SPAWN_VELOCITY_LEVELS_MS,
    DEFAULT_TRAFFIC_DENSITY_LEVELS,
)
from traffic_bench.eval.engine.expand.manifest_expansion import mark_nominal_row, shuffle_cap
from traffic_bench.eval.engine.expand.world_axes import (
    WorldAxisCell,
    describe_world_axes,
    iter_world_axis_cells,
    sample_profile_for_cell,
    stamp_world_axis_fields,
)
from traffic_bench.eval.engine.map.segment_length import SegmentCorridor, resolve_segment_corridor
from traffic_bench.eval.engine.spawn.route_budget import measure_spawn_to_dest_length_m
from traffic_bench.eval.engine.spawn.route_length_levels import (
    list_route_length_levels,
    select_route_length_levels,
)
from traffic_bench.eval.engine.traffic.npc_profile import embed_npc_profile
from traffic_bench.eval.engine.traffic.stable_hash import stable_hash
from traffic_bench.eval.manifest.io import (
    append_scene_entries,
    apply_max_total,
    apply_split_filter,
    load_scene_metadata,
    write_real_manifest,
)
from traffic_bench.scene_collection.sign_scenes.filter.selection import is_reserved_scene_dir
from traffic_bench.scene_collection.sign_scenes.materialize.pool_index import normalize_split

# PDD code -> runtime plate class (traffic_bench.signs.extra.restricted_lane).
SIGN_CLASS_BY_CODE = {
    "5.14.1": "BusLaneSign",
    "5.14.2": "BikeLaneSign",
    "5.11.1": "BusLaneRoadSign",
    "5.11.2": "BikeLaneRoadSign",
}
RESTRICTED_LANE_CODES = frozenset(SIGN_CLASS_BY_CODE)
# 5.14.x: the reserved lane runs WITH the ego on the rightmost lane (SUMO 0).
# 5.11.x: a counter-flow lane on a one-way street, i.e. the leftmost lane; the
# buses / cyclists on it come towards the ego.
RESERVED_LANE_INDEX = 0
# Family axis defaults (per lane user); the nominal row always has one user.
DEFAULT_AGENTS_N_LEVELS = {"bus": (1, 2), "bicycle": (1, 2, 3)}
NOMINAL_AGENTS_N = 1
# The plate never sits closer than this to the edge start.
MIN_SIGN_S_M = 20.0


def flow_for(pdd_code: str) -> str:
    return "opposite" if str(pdd_code).startswith("5.11") else "same"


def lane_user_for(pdd_code: str) -> str:
    return "bus" if str(pdd_code).endswith(".1") else "bicycle"


def reserved_lane_for(meta: Dict[str, Any], pdd_code: str) -> int:
    lanes = _vehicle_lane_indices(meta)
    if not lanes:
        return RESERVED_LANE_INDEX
    return lanes[-1] if flow_for(pdd_code) == "opposite" else lanes[0]


@dataclass(frozen=True)
class RestrictedLaneSimParams:
    spawn_offset_from_start: float = 10.0
    # Single route level: it has to cover approach + zone + tail/2 for the
    # largest family levels, so the shared [90, 120] m budgets do not apply.
    max_path_length_m: float = 220.0
    max_path_length_levels: Tuple[float, ...] = (220.0,)
    # Length of the reserved-lane zone after the plate (nominal + levels). The
    # violation is judged inside it, so the whole zone has to fit on the edge
    # and the destination has to lie past its end.
    zone_m: float = 60.0
    zone_levels_m: Tuple[float, ...] = (60.0,)
    # Room past the zone end kept for the finish line.
    tail_after_zone_m: float = 10.0
    # How far before the plate the ego starts (nominal + levels). The expert
    # pre-empts 40-60 m ahead of the zone, so the run-up must be at least that.
    approach_before_sign_m: float = 60.0
    approach_levels_m: Tuple[float, ...] = (60.0,)
    # Number of buses / cyclists sharing the lane (family axis).
    agents_n_levels: Tuple[int, ...] = (NOMINAL_AGENTS_N,)
    # Ego spawn speed: nominal value + the shared world-grid levels.
    spawn_velocity_ms: float = 5.0
    spawn_velocity_levels_ms: Tuple[float, ...] = DEFAULT_SPAWN_VELOCITY_LEVELS_MS
    traffic_density_levels: Tuple[float, ...] = DEFAULT_TRAFFIC_DENSITY_LEVELS
    horizon: int = 600
    traffic_density: float = 0.0
    n_variations: int = 3
    profile_density_cap: float = 1.0
    # Upstream slide of the plate on the sampled variants (never downstream:
    # the nominal position is the last metre that leaves room for the zone).
    sign_jitter_m: float = 15.0
    default_first_variant: bool = True
    # A (spawn speed, approach) pair is admissible only when the approach gives
    # the ego at least this long to plan and execute the lane change.
    lc_planning_time_s: float = 6.0


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


def geometry_fits(
    usable_length_m: float,
    *,
    approach_m: float,
    zone_m: float,
    tail_m: float,
    spawn_offset_from_start: float,
) -> bool:
    """Plate at ``L - (zone + tail)``, ego ``approach`` before it: both on the edge."""
    sign_s = float(usable_length_m) - (float(zone_m) + float(tail_m))
    return sign_s >= MIN_SIGN_S_M and sign_s - float(approach_m) >= float(spawn_offset_from_start)


def build_restricted_lane_entry(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    sim: RestrictedLaneSimParams,
    pdd_code: str,
    corridor: SegmentCorridor,
    sign_type: str = "restricted_lane",
    variant: int = 0,
    npc_profile: Optional[Dict[str, Any]] = None,
    zone_m: Optional[float] = None,
    approach_m: Optional[float] = None,
    reserved_agents_n: int = NOMINAL_AGENTS_N,
    max_path_length_m: Optional[float] = None,
    route_length_augment: bool = False,
    default_variant: bool = False,
    spawn_velocity_ms: Optional[float] = None,
    traffic_density: Optional[float] = None,
    scene_id_suffix: str = "",
    seed: Optional[int] = None,
    level_ids: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """One manifest row: plate at ``sign_s`` on the reserved lane, zone ``[sign_s, sign_s+zone]``.

    Geometry is authored in corridor-local metres and lifted with
    ``corridor.to_absolute`` (XY crops may keep a longer SUMO edge than the
    harvest window). Returns ``{}`` when the requested zone / approach do not
    fit the edge.
    """
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    net_file = str(meta.get("net_file") or "map.net.xml")
    net_path = scene_dir.relative_to(scenes_root) / net_file

    road_id = str(meta.get("road_id") or "")
    lane_index = reserved_lane_for(meta, pdd_code)
    usable = float(corridor.usable_length_m)
    net_length = float(corridor.net_length_m)
    zone = float(zone_m if zone_m is not None else sim.zone_m)
    approach = float(approach_m if approach_m is not None else sim.approach_before_sign_m)
    tail = float(sim.tail_after_zone_m)
    spawn_min = float(sim.spawn_offset_from_start)
    if not geometry_fits(usable, approach_m=approach, zone_m=zone, tail_m=tail, spawn_offset_from_start=spawn_min):
        return {}

    # Nominal plate: as late as the edge allows while the whole zone plus the
    # finish room still fit. Sampled rows slide it upstream (never downstream)
    # as far as the run-up keeps the ego on the edge.
    sign_s_nominal = usable - (zone + tail)
    sign_s = sign_s_nominal
    jitter = 0.0 if default_variant else float(sim.sign_jitter_m)
    if jitter > 0.0:
        room = min(jitter, sign_s - MIN_SIGN_S_M, sign_s - approach - spawn_min)
        if room > 0.0:
            sign_s -= random.Random(stable_hash(scene_name, int(variant), "sign_jitter")).random() * room
    zone_end = min(usable - 1.0, sign_s + zone)
    spawn_offset = sign_s - approach

    path_budget_m = float(
        max_path_length_m if max_path_length_m is not None else sim.max_path_length_m
    )
    # The verdict needs the whole zone: the budget has to reach past its end.
    if path_budget_m + 1e-6 < approach + zone + tail / 2.0:
        return {}
    dest_along = min(spawn_offset + path_budget_m, max(spawn_offset + 1.0, usable - 5.0))

    sign_s_abs = corridor.to_absolute(sign_s)
    zone_end_abs = corridor.to_absolute(zone_end)
    spawn_offset_abs = corridor.to_absolute(spawn_offset)
    dest_along_abs = corridor.to_absolute(dest_along)
    spawn_before_end = max(20.0, net_length - spawn_offset_abs)

    scene_id = f"{scene_name}_v{int(variant)}"
    if scene_id_suffix and scene_id_suffix not in scene_id:
        scene_id = f"{scene_id}_{scene_id_suffix}"
    elif route_length_augment and not scene_id_suffix:
        scene_id = f"{scene_id}_rl{int(round(path_budget_m))}"
    row_seed = int(seed) if seed is not None else int(
        stable_hash(scene_name, int(variant), scene_id_suffix or "nominal")
    )
    if default_variant:
        row_density = 0.0
    elif traffic_density is not None:
        row_density = float(traffic_density)
    else:
        row_density = float(sim.traffic_density)
    v0 = float(spawn_velocity_ms if spawn_velocity_ms is not None else sim.spawn_velocity_ms)

    spawn_lane_id = f"{road_id}_{lane_index}"
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
        "seed": row_seed,
        "deterministic_seed": row_seed,
        "var_idx": int(variant),
        "road_id": road_id,
        "spawn_lane_id": spawn_lane_id,
        "destination_lane_id": spawn_lane_id,
        "destination_edge_id": road_id,
        "spawn_lane_num": lane_index,
        "sign_lane_index": lane_index,
        "restricted_lane_index": lane_index,
        "reserved_lane_index": lane_index,
        "flow": flow_for(pdd_code),
        "lane_user": lane_user_for(pdd_code),
        "reserved_agents_n": int(reserved_agents_n),
        "sign_s": sign_s_abs,
        "sign_s_nominal": corridor.to_absolute(sign_s_nominal),
        "zone_length_m": zone,
        "zone_end_s": zone_end_abs,
        "approach_before_sign_m": approach,
        "spawn_distance_before_end": spawn_before_end,
        "destination_max_along_m": dest_along_abs,
        "max_path_length_m": path_budget_m,
        "route_length_level_m": path_budget_m,
        "spawn_velocity_ms": v0,
        "spawn_offset_from_start": spawn_offset_abs,
        "traffic_density": row_density,
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
        # Full edge length for MD<->SUMO remap; usable window is corridor_*.
        "edge_length_m": net_length,
        "window_length_m": float(corridor.window_length_m),
        "corridor_s0": float(corridor.corridor_s0),
        "corridor_s1": float(corridor.corridor_s1),
        "corridor_from_window": bool(corridor.from_window),
    }
    for key, val in (level_ids or {}).items():
        row[key] = int(val)
    if npc_profile is not None:
        row = embed_npc_profile(row, npc_profile, density_cap=float(sim.profile_density_cap))
    return row


def _family_candidates(
    *,
    usable_length_m: float,
    sim: RestrictedLaneSimParams,
    route_levels: List[float],
) -> List[Tuple[int, float, int, float, int, int, WorldAxisCell]]:
    """All (zone, approach, n_users, world cell) tuples that fit this edge."""
    tail = float(sim.tail_after_zone_m)
    fits = [
        (zi, float(z), ai, float(a))
        for zi, z in enumerate(sim.zone_levels_m)
        for ai, a in enumerate(sim.approach_levels_m)
        if geometry_fits(
            usable_length_m,
            approach_m=float(a),
            zone_m=float(z),
            tail_m=tail,
            spawn_offset_from_start=float(sim.spawn_offset_from_start),
        )
    ]
    cells = list(
        iter_world_axis_cells(route_levels=route_levels, sim=sim, task_conditioned_spawn=False)
    )
    out = []
    for zi, z, ai, a in fits:
        for ni, n in enumerate(sim.agents_n_levels):
            for cell in cells:
                v0 = cell.spawn_velocity_ms
                if v0 is not None and a + 1e-6 < float(sim.lc_planning_time_s) * float(v0):
                    continue
                out.append((zi, z, ai, a, ni, int(n), cell))
    return out


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
    """Nominal row + ``max_scenarios - 1`` rows sampled from the full grid."""
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    road_id = str(meta.get("road_id") or "")
    net_full = scene_dir / str(meta.get("net_file") or "map.net.xml")
    corridor = resolve_segment_corridor(net_full, meta, road_id=road_id)
    if corridor is None or corridor.usable_length_m <= 0.0:
        print(f"  Skipping {scene_dir.name}: no usable corridor on {road_id!r}")
        return []
    if corridor.from_window:
        print(
            f"  corridor window on edge: s=[{corridor.corridor_s0:.1f}, "
            f"{corridor.corridor_s1:.1f}]m of net={corridor.net_length_m:.1f}m"
        )
    usable = float(corridor.usable_length_m)
    tail = float(sim.tail_after_zone_m)
    if not geometry_fits(
        usable,
        approach_m=float(sim.approach_before_sign_m),
        zone_m=float(sim.zone_m),
        tail_m=tail,
        spawn_offset_from_start=float(sim.spawn_offset_from_start),
    ):
        print(
            f"  Skipping {scene_dir.name}: usable {usable:.0f} m < approach "
            f"{sim.approach_before_sign_m:.0f} + zone {sim.zone_m:.0f} + tail {tail:.0f} + "
            f"{sim.spawn_offset_from_start:.0f}"
        )
        return []

    available_route_m = None
    if road_id:
        available_route_m = measure_spawn_to_dest_length_m(
            net_path=net_full,
            spawn_edge=road_id,
            spawn_lane=reserved_lane_for(meta, pdd_code),
            dest_edge=road_id,
            spawn_along_m=corridor.to_absolute(float(sim.spawn_offset_from_start)),
        )
    route_levels, route_augment = select_route_length_levels(
        list_route_length_levels(sim), available_route_m
    )

    common = dict(
        scene_dir=scene_dir,
        scenes_root=scenes_root,
        meta=meta,
        sim=sim,
        pdd_code=pdd_code,
        corridor=corridor,
        sign_type=sign_type,
    )
    entries: List[Dict[str, Any]] = []
    if bool(sim.default_first_variant):
        nominal = build_restricted_lane_entry(
            default_variant=True,
            variant=0,
            npc_profile=None,
            zone_m=float(sim.zone_m),
            approach_m=float(sim.approach_before_sign_m),
            reserved_agents_n=NOMINAL_AGENTS_N,
            max_path_length_m=float(sim.max_path_length_m),
            route_length_augment=False,
            spawn_velocity_ms=float(sim.spawn_velocity_ms),
            traffic_density=0.0,
            level_ids={"zone_level_id": -1, "approach_level_id": -1, "agents_level_id": -1},
            **common,
        )
        if nominal:
            entries.append(mark_nominal_row(nominal))

    candidates = _family_candidates(usable_length_m=usable, sim=sim, route_levels=route_levels)
    n_candidates = len(candidates)
    cap = expansion.max_scenarios
    cap_i: Optional[int] = None
    if cap is not None:
        try:
            cap_i = max(0, int(cap))
        except (TypeError, ValueError):
            cap_i = None
    if cap_i is not None and n_candidates > max(0, cap_i - len(entries)):
        # Sample before build: shuffle the cheap tuples, build only what the cap
        # keeps. The nominal row is already in place and never dropped.
        random.Random(stable_hash(str(scene_dir.name), "restricted_lane_world_cap", cap_i)).shuffle(candidates)

    built = 0
    for zi, zone, ai, approach, ni, n_users, cell in candidates:
        if cap_i is not None and len(entries) >= cap_i:
            break
        suffix = f"z{int(round(zone))}a{int(round(approach))}n{n_users}_{cell.scene_suffix(route_augment=route_augment)}"
        seed = stable_hash(scene_name, int(round(zone)), int(round(approach)), int(n_users), *cell.seed_tags())
        npc_profile = sample_profile_for_cell(cell, seed=int(seed), sim=sim)
        row = build_restricted_lane_entry(
            default_variant=False,
            variant=int(cell.npc_var),
            npc_profile=npc_profile,
            zone_m=zone,
            approach_m=approach,
            reserved_agents_n=n_users,
            max_path_length_m=float(cell.route_length_m),
            route_length_augment=route_augment,
            spawn_velocity_ms=cell.spawn_velocity_ms,
            traffic_density=float(cell.density.traffic_density),
            scene_id_suffix=suffix,
            seed=int(seed),
            level_ids={"zone_level_id": zi, "approach_level_id": ai, "agents_level_id": ni},
            **common,
        )
        if row:
            entries.append(stamp_world_axis_fields(row, cell))
            built += 1

    entries = shuffle_cap(
        entries,
        cap_i,
        seed_key=(str(scene_dir.name), "restricted_lane_world_cap", cap_i if cap_i is not None else 0),
    )
    print(
        f"  Retained {len(entries)} of {n_candidates + (1 if bool(sim.default_first_variant) else 0)} "
        f"candidates (sampled before build, cap={cap_i}; nominal first; built {built} sampled row(s))"
    )
    return entries


def _levels(raw, cast, fallback: Tuple) -> Tuple:
    vals = tuple(cast(x) for x in (raw or ()))
    return vals if vals else tuple(fallback)


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
    user = lane_user_for(PDD_CODE)

    all_scenes = discover_restricted_lane_scenes(scenes_dir)
    print(f"Scenes root: {scenes_dir.resolve()}")
    print(f"Discovered {len(all_scenes)} multi-lane segment scene(s) for {PDD_CODE}")
    scenes, split_by_id = apply_split_filter(all_scenes, scenes_dir=scenes_dir, split=split)

    zone_m = float(getattr(sim_cfg, "restricted_zone_m", 60.0) or 60.0)
    approach_m = float(getattr(sim_cfg, "approach_before_sign_m", 60.0) or 60.0)
    sim_params = RestrictedLaneSimParams(
        spawn_offset_from_start=float(sim_cfg.spawn_offset_from_start),
        max_path_length_m=float(sim_cfg.max_path_length_m),
        max_path_length_levels=_levels(
            getattr(sim_cfg, "max_path_length_levels", None), float, (float(sim_cfg.max_path_length_m),)
        ),
        zone_m=zone_m,
        zone_levels_m=_levels(getattr(sim_cfg, "restricted_zone_levels_m", None), float, (zone_m,)),
        tail_after_zone_m=float(getattr(sim_cfg, "tail_after_sign_m", 10.0) or 10.0),
        approach_before_sign_m=approach_m,
        approach_levels_m=_levels(getattr(sim_cfg, "approach_levels_m", None), float, (approach_m,)),
        agents_n_levels=_levels(
            getattr(sim_cfg, "reserved_agents_n_levels", None), int, DEFAULT_AGENTS_N_LEVELS[user]
        ),
        spawn_velocity_ms=float(sim_cfg.spawn_velocity_ms),
        spawn_velocity_levels_ms=_levels(
            getattr(sim_cfg, "spawn_velocity_levels_ms", None), float, DEFAULT_SPAWN_VELOCITY_LEVELS_MS
        ),
        traffic_density_levels=_levels(
            getattr(sim_cfg, "traffic_density_levels", None), float, DEFAULT_TRAFFIC_DENSITY_LEVELS
        ),
        horizon=int(sim_cfg.horizon),
        traffic_density=float(sim_cfg.traffic_density),
        n_variations=max(1, int(getattr(sim_cfg, "n_variations", 3) or 3)),
        profile_density_cap=float(getattr(sim_cfg, "profile_density_cap", 1.0) or 1.0),
        sign_jitter_m=float(getattr(sim_cfg, "sign_jitter_m", 15.0) or 0.0),
        default_first_variant=bool(getattr(sim_cfg, "default_first_variant", True)),
        lc_planning_time_s=float(getattr(sim_cfg, "lc_planning_time_s", 6.0) or 0.0),
    )
    expansion = RestrictedLaneExpansionConfig(max_scenarios=scenario_cfg.max_scenarios)
    family_axes = (
        f"zone={list(sim_params.zone_levels_m)} approach={list(sim_params.approach_levels_m)} "
        f"n_users={list(sim_params.agents_n_levels)} ({user}) plate_jitter<={sim_params.sign_jitter_m:.0f} m "
        f"lc_planning>={sim_params.lc_planning_time_s:.0f} s x spawn_v"
    )
    print(f"Augmentation axes: {describe_world_axes(sim_params)} | family: {family_axes}")
    print(
        f"Reserved lane: {'leftmost, counter-flow' if flow_for(PDD_CODE) == 'opposite' else 'rightmost, with the flow'}; "
        f"nominal zone {zone_m:.0f} m, approach {approach_m:.0f} m, 1 {user}, {sim_params.spawn_velocity_ms:.1f} m/s"
    )

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
                f"{SIGN_CLASS_BY_CODE.get(PDD_CODE)} ({PDD_CODE}) on the reserved lane at sign_s; "
                f"zone after the plate; ego spawns on that lane `approach` m before the plate "
                f"and must move to a neighbouring lane"
            ),
            "total_scenes": len(used_scene_ids),
            "total_entries": len(entries),
            "total_entries_before_max_total": pre_total,
            "max_scenarios": scenario_cfg.max_scenarios,
            "max_total": scenario_cfg.max_total,
            "sampling": "world grid x family axes, all combinations shuffled, first max_scenarios built; nominal first",
            "world_axes": describe_world_axes(sim_params),
            "family_axes": family_axes,
            "n_variations": sim_params.n_variations,
            "traffic_density_levels": list(sim_params.traffic_density_levels),
            "spawn_velocity_levels_ms": list(sim_params.spawn_velocity_levels_ms),
            "restricted_zone_levels_m": list(sim_params.zone_levels_m),
            "approach_levels_m": list(sim_params.approach_levels_m),
            "reserved_agents_n_levels": list(sim_params.agents_n_levels),
            "lc_planning_time_s": sim_params.lc_planning_time_s,
            "sign_jitter_m": sim_params.sign_jitter_m,
            "profile_density_cap": sim_params.profile_density_cap,
            "npc_world": "world_axes: density probe x agent_profile_bank.sample_one_profile",
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
