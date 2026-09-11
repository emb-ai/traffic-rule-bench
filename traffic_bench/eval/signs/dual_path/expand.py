"""Dual-path × ego spawn lane × NPC profile → manifest rows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from traffic_bench.eval.engine.expand.manifest_config import (
    DEFAULT_SPAWN_VELOCITY_LEVELS_MS,
    DEFAULT_TRAFFIC_DENSITY_LEVELS,
)
from traffic_bench.eval.engine.expand.manifest_expansion import (
    mark_nominal_row,
    non_nominal_budget,
    shuffle_cap,
    shuffled_copy,
)
from traffic_bench.eval.engine.expand.world_axes import (
    DEFAULT_HORIZON_STEPS,
    DEFAULT_MAX_PATH_LENGTH_M,
    DEFAULT_ROUTE_LENGTH_LEVELS_M,
    describe_world_axes,
    iter_world_axis_cells,
    sample_profile_for_cell,
    stamp_world_axis_fields,
)
from traffic_bench.eval.engine.traffic.npc_profile import embed_npc_profile
from traffic_bench.eval.engine.traffic.stable_hash import stable_hash
from traffic_bench.eval.engine.spawn.scene_augmentation import SpawnScenario
from traffic_bench.eval.engine.spawn.route_length_levels import (
    list_route_length_levels,
    select_route_length_levels,
    tag_entry_route_length,
)
from traffic_bench.eval.engine.spawn.route_budget import measure_spawn_to_dest_length_m
from traffic_bench.eval.signs.dual_path.budget import (
    apply_dual_path_route_budget,
    load_sumo_edge_lengths,
)
from traffic_bench.eval.signs.dual_path.scene import (
    DualPathScenario,
    dual_path_to_spawn_scenario,
    ego_spawn_lane_nums_for_dual,
)
from traffic_bench.eval.signs.dual_path.spec import discover_dual_paths, get_spec

_CARDINAL = frozenset({"s", "r", "l"})


@dataclass(frozen=True)
class DualPathSimParams:
    spawn_distance_before_end: float
    sign_distance_before_end: float
    spawn_velocity_ms: float
    horizon: int = DEFAULT_HORIZON_STEPS
    n_variations: int = 3
    profile_density_cap: float = 1.0
    traffic_density_levels: Tuple[float, ...] = DEFAULT_TRAFFIC_DENSITY_LEVELS
    spawn_velocity_levels_ms: Tuple[float, ...] = DEFAULT_SPAWN_VELOCITY_LEVELS_MS
    min_dual_path_gain_m: float = 20.0
    min_ego_lane_m: float = 8.0
    max_path_length_m: float = DEFAULT_MAX_PATH_LENGTH_M
    max_path_length_levels: Tuple[float, ...] = DEFAULT_ROUTE_LENGTH_LEVELS_M
    default_first_variant: bool = True


@dataclass(frozen=True)
class DualPathExpansionConfig:
    layout: bool = True
    max_scenarios: Optional[int] = None
    max_dual_paths: int = 20
    arm_counts: Tuple[int, ...] = (3, 4)


BuildEntryFn = Callable[..., Dict[str, Any]]


def dual_path_geometry_key(entry: Dict[str, Any]) -> Tuple:
    return (
        entry.get("road_id"),
        entry.get("spawn_lane_num"),
        entry.get("destination_lane_id"),
        entry.get("baseline_dir") or entry.get("baseline_turn_dir"),
        entry.get("var_idx"),
        entry.get("density_level_id"),
        entry.get("spawn_velocity_level_id"),
        round(float(entry.get("route_length_level_m") or entry.get("max_path_length_m") or 0.0), 1),
    )


def _manifest_entry_routable(entry: Dict[str, Any], net_path: Path) -> bool:
    """True when both baseline and compliant MetaDrive dests are non-looping."""
    from traffic_bench.eval.signs.dual_path.route_probe import probe_manifest_entry

    result = probe_manifest_entry(entry, net_path=net_path)
    if result.ok:
        return True
    print(
        f"  [route-probe] skip {entry.get('scene_id')}: {result.reason}",
        flush=True,
    )
    return False


def expand_dual_path_scene_entries(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    net_path: Path,
    spawn_lanes: Sequence[Any],
    junction_layout: dict,
    sim: DualPathSimParams,
    expansion: DualPathExpansionConfig,
    build_entry: BuildEntryFn,
    pdd_code: str,
) -> List[Dict[str, Any]]:
    spec = get_spec(pdd_code)
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    print(
        f"  Junction layout: {junction_layout.get('shape')} @ "
        f"{junction_layout.get('junction_id')} "
        f"(arms={len(junction_layout.get('arms', []))})"
    )

    if not expansion.layout:
        print("  Layout axis off: dual-path requires layout; skipping")
        return []

    preferred_jid = str(
        junction_layout.get("junction_id") or meta.get("junction_id") or ""
    ).strip()
    junction_ids = [preferred_jid] if preferred_jid else None

    dual_paths = discover_dual_paths(
        net_path,
        pdd_code=pdd_code,
        min_gain_m=float(sim.min_dual_path_gain_m),
        min_lane_length_m=float(sim.min_ego_lane_m),
        max_scenarios=int(expansion.max_dual_paths),
        junction_ids=junction_ids,
        arm_counts=expansion.arm_counts,
        scene_meta=meta,
    )
    if not dual_paths and junction_ids is not None:
        print(
            f"  [dual-path] No picks at junction {preferred_jid}; "
            f"retrying without junction filter"
        )
        dual_paths = discover_dual_paths(
            net_path,
            pdd_code=pdd_code,
            min_gain_m=float(sim.min_dual_path_gain_m),
            min_lane_length_m=float(sim.min_ego_lane_m),
            max_scenarios=int(expansion.max_dual_paths),
            junction_ids=None,
            arm_counts=expansion.arm_counts,
            scene_meta=meta,
        )

    if not dual_paths:
        print(f"  [dual-path] No scenarios for {scene_name}; skipping scene")
        return []

    print(f"  Dual-path scenarios: {len(dual_paths)}")
    for dp in dual_paths:
        lanes = ego_spawn_lane_nums_for_dual(
            dp,
            spawn_lanes,
            min_lane_length_m=float(sim.min_ego_lane_m),
        )
        print(
            f"    junc={dp.junction_id} ego={dp.ego_edge_id} "
            f"lanes={lanes} "
            f"base={dp.turn_dir}->{dp.turn_first_exit} "
            f"comp={dp.compliant_dir}->{dp.straight_first_exit} "
            f"gain={dp.gain_m:.1f}m"
        )

    geometries: List[Tuple[int, DualPathScenario, int]] = [
        (dual_i, dp, lane_num)
        for dual_i, dp in enumerate(dual_paths)
        for lane_num in ego_spawn_lane_nums_for_dual(
            dp,
            spawn_lanes,
            min_lane_length_m=float(sim.min_ego_lane_m),
        )
    ]
    n_lane_combos = len(geometries)
    print(
        f"  Dual×lane geometries: {n_lane_combos} "
        f"({len(dual_paths)} dual × lane slots)"
    )

    scene_entries: List[Dict[str, Any]] = []
    seen: set = set()
    configured_route_levels = list_route_length_levels(sim)
    spawn_before_end = float(sim.spawn_distance_before_end)
    net_full = scene_dir / str(meta.get("net_file") or "map.net.xml")

    # One no-NPC reference row per map (first dual×lane geometry).
    if bool(sim.default_first_variant) and geometries:
        dual_i, dual, lane_num = geometries[0]
        spawn_scenario = dual_path_to_spawn_scenario(dual, ego_lane_num=lane_num)
        seed = stable_hash(
            scene_name, spawn_scenario.scenario_id, lane_num, "nominal"
        )
        nominal = build_entry(
            scene_dir=scene_dir,
            scenes_root=scenes_root,
            meta=meta,
            layout_variant=dual_i,
            var_idx=0,
            seed=seed,
            sim=sim,
            spawn_scenario=spawn_scenario,
            dual_path=dual,
            spawn_lanes_cache=list(spawn_lanes),
            junction_layout_cache=junction_layout,
            npc_profile=None,
            max_path_length_m=float(sim.max_path_length_m),
            route_length_augment=False,
            spawn_velocity_ms=float(sim.spawn_velocity_ms),
            traffic_density=0.0,
        )
        if _manifest_entry_routable(nominal, net_full):
            scene_entries.append(mark_nominal_row(nominal))

    combos = []
    for dual_i, dual, lane_num in geometries:
        spawn_scenario = dual_path_to_spawn_scenario(dual, ego_lane_num=lane_num)
        net_full = scene_dir / str(meta.get("net_file") or "map.net.xml")
        branch_lens = []
        for path in (dual.turn_path, dual.straight_path):
            if not path:
                continue
            measured = measure_spawn_to_dest_length_m(
                net_path=net_full,
                spawn_edge=str(dual.ego_edge_id),
                spawn_lane=int(lane_num),
                dest_edge=str(path[-1]),
                spawn_distance_before_end=spawn_before_end,
            )
            if measured is not None:
                branch_lens.append(measured)
        available_route_m = max(branch_lens) if branch_lens else None
        route_levels, route_augment = select_route_length_levels(
            configured_route_levels, available_route_m
        )
        for cell in iter_world_axis_cells(
            route_levels=route_levels,
            sim=sim,
            task_conditioned_spawn=False,
        ):
            combos.append((dual_i, dual, lane_num, spawn_scenario, cell, bool(route_augment)))

    cap = expansion.max_scenarios
    budget = non_nominal_budget(cap, len(scene_entries))
    ordered = shuffled_copy(
        combos,
        seed_key=(
            scene_name,
            f"{spec.family}_world_cap",
            int(cap) if cap is not None else 0,
        ),
    )
    pre_pool = len(ordered)
    built = 0
    for dual_i, dual, lane_num, spawn_scenario, cell, route_augment in ordered:
        if budget is not None and built >= budget:
            break
        suffix = cell.scene_suffix(route_augment=route_augment)
        seed = stable_hash(
            scene_name,
            spawn_scenario.scenario_id,
            lane_num,
            *cell.seed_tags(),
        )
        profile = sample_profile_for_cell(cell, seed=int(seed), sim=sim)
        entry = build_entry(
            scene_dir=scene_dir,
            scenes_root=scenes_root,
            meta=meta,
            layout_variant=dual_i,
            var_idx=cell.npc_var,
            seed=seed,
            sim=sim,
            spawn_scenario=spawn_scenario,
            dual_path=dual,
            spawn_lanes_cache=list(spawn_lanes),
            junction_layout_cache=junction_layout,
            npc_profile=profile,
            max_path_length_m=float(cell.route_length_m),
            route_length_augment=route_augment,
            spawn_velocity_ms=cell.spawn_velocity_ms,
            traffic_density=float(cell.density.traffic_density),
            scene_id_suffix=suffix,
        )
        entry = stamp_world_axis_fields(entry, cell)
        if not _manifest_entry_routable(entry, net_full):
            continue
        key = dual_path_geometry_key(entry)
        if key in seen:
            continue
        seen.add(key)
        scene_entries.append(entry)
        built += 1

    pre_cap = len(scene_entries)
    scene_entries = shuffle_cap(
        scene_entries,
        cap,
        seed_key=(
            scene_name,
            f"{spec.family}_world_cap",
            int(cap) if cap is not None else 0,
        ),
    )
    if cap is not None and pre_pool > (budget if budget is not None else pre_pool):
        print(
            f"  Early-sampled {built} of {pre_pool} combos "
            f"(cap={cap}; nominal preserved; "
            f"{n_lane_combos} dual×lane geometries)"
        )
    elif cap is not None and pre_cap > cap:
        print(
            f"  Retained {len(scene_entries)} of {pre_cap} world-grid variants "
            f"(shuffled, cap={cap}; nominal preserved; "
            f"{n_lane_combos} dual×lane geometries)"
        )
    else:
        print(f"  Manifest entries for {scene_name}: {len(scene_entries)}")
    return scene_entries




def build_dual_path_manifest_entry(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    layout_variant: int,
    var_idx: int,
    seed: int,
    sim: DualPathSimParams,
    spawn_scenario: Optional[SpawnScenario],
    dual_path: Optional[DualPathScenario],
    spawn_lanes_cache: Optional[List[Any]],
    junction_layout_cache: Optional[dict],
    npc_profile: Optional[Dict[str, Any]],
    pdd_code: str,
    sign_type: str = "",
    max_path_length_m: Optional[float] = None,
    route_length_augment: bool = False,
    spawn_velocity_ms: Optional[float] = None,
    traffic_density: Optional[float] = None,
    scene_id_suffix: str = "",
) -> Dict[str, Any]:
    del layout_variant
    spec = get_spec(pdd_code)
    sign_type = sign_type or spec.family
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    net_file = meta.get("net_file", "map.net.xml")
    net_rel = scene_dir.relative_to(scenes_root) / net_file

    if traffic_density is not None:
        row_density = float(traffic_density)
    elif npc_profile is not None:
        row_density = float(npc_profile["traffic_density"])
    else:
        row_density = 0.0
    horizon = int(
        npc_profile.get("horizon_steps", sim.horizon) if npc_profile else sim.horizon
    )
    scene_id = scene_name
    if scene_id_suffix and scene_id_suffix not in scene_id:
        scene_id = f"{scene_id}_{scene_id_suffix}"
    v0 = float(
        spawn_velocity_ms if spawn_velocity_ms is not None else sim.spawn_velocity_ms
    )

    selected_lane = None
    if spawn_scenario is not None and spawn_lanes_cache:
        for lane in spawn_lanes_cache:
            if (
                lane.edge_id == spawn_scenario.ego_edge_id
                and lane.lane_num == spawn_scenario.ego_lane_num
            ):
                selected_lane = lane
                break
        if selected_lane is None:
            for lane in spawn_lanes_cache:
                if lane.edge_id == spawn_scenario.ego_edge_id:
                    selected_lane = lane
                    break

    entry: Dict[str, Any] = {
        "scene_id": scene_id,
        "scene_name": scene_name,
        "net_path": str(net_rel),
        "seed": int(seed),
        "var_idx": int(var_idx),
        "pdd_code": spec.sign_code,
        "sign_code": spec.sign_code,
        "sign_type": sign_type,
        "sign_family": spec.family,
        "sign_title": spec.title,
        "sign_class": spec.class_name,
        "allowed_dirs": sorted(spec.allowed_dirs),
        "spawn_velocity_ms": v0,
        "traffic_density": row_density,
        "horizon": horizon,
        "sign_distance_before_end": float(sim.sign_distance_before_end),
        "spawn_distance_before_end": float(sim.spawn_distance_before_end),
        "valid": True,
        "auxiliary_agent": False,
        "latitude": meta.get("latitude"),
        "longitude": meta.get("longitude"),
        "crop_radius_m": meta.get("crop_radius_m"),
        "source_osm": meta.get("source_osm"),
        "osm_file": meta.get("osm_file"),
    }
    if spec.family == "one_way":
        entry["forbidden_dir"] = spec.forbidden_dir
    elif spec.family == "direction":
        entry["forbidden_dirs"] = sorted(_CARDINAL - set(spec.allowed_dirs))
    elif spec.family == "no_turn":
        entry["forbidden_dir"] = spec.forbidden_dir
        entry["forbidden_dirs"] = [spec.forbidden_dir]
    else:
        entry["forbidden_dirs"] = []

    if npc_profile is not None:
        entry = embed_npc_profile(
            entry,
            npc_profile,
            density_cap=float(sim.profile_density_cap),
        )

    if spawn_scenario is not None:
        entry.update(spawn_scenario.to_manifest_fields())

    if selected_lane is not None:
        entry["spawn_lane_length"] = selected_lane.length
        entry["spawn_to_junction"] = selected_lane.to_junction

    if dual_path is not None:
        meta_fields = dual_path.to_meta_fields()
        dual_meta = dict(meta_fields["dual_path"])
        if spawn_scenario is not None:
            dual_meta["spawn_lane_num"] = int(spawn_scenario.ego_lane_num)
        entry["dual_path"] = dual_meta
        entry["baseline_turn_dir"] = dual_path.turn_dir
        entry["turn_length_m"] = dual_path.turn_length_m
        entry["straight_length_m"] = dual_path.straight_length_m
        entry["dual_path_gain_m"] = dual_path.gain_m
        entry["compliant_dir"] = dual_path.compliant_dir
        entry["baseline_dir"] = dual_path.turn_dir
        entry["compliant_first_exit"] = dual_path.straight_first_exit
        entry["baseline_first_exit"] = dual_path.turn_first_exit
        entry["junction_id"] = dual_path.junction_id
        if spec.family == "one_way":
            entry["background_excluded_edges"] = list(dual_path.wrong_dir_edges)
            entry["forbidden_dir"] = spec.forbidden_dir
            dual_meta["forbidden_dir"] = spec.forbidden_dir

        budget = float(
            max_path_length_m if max_path_length_m is not None else sim.max_path_length_m
        )
        if budget > 0.0:
            net_full = scene_dir / str(meta.get("net_file", "map.net.xml"))
            edge_lengths = load_sumo_edge_lengths(net_full)
            spawn_rem = float(sim.spawn_distance_before_end)
            if selected_lane is not None:
                spawn_rem = min(spawn_rem, float(selected_lane.length))
            trimmed = apply_dual_path_route_budget(
                dual_meta,
                ego_edge_id=str(dual_path.ego_edge_id),
                edge_lengths=edge_lengths,
                budget_m=float(budget),
                spawn_remaining_on_ego_m=spawn_rem,
                dest_lane_num=int(dual_path.dest_lane_num),
            )
            entry.update(trimmed)
            for key in (
                "destination_lane_id",
                "destination_edge_id",
                "baseline_destination_lane_id",
                "compliant_destination_lane_id",
                "destination_max_along_m",
                "baseline_destination_max_along_m",
                "compliant_destination_max_along_m",
            ):
                if trimmed.get(key) is None:
                    entry.pop(key, None)
            entry = tag_entry_route_length(entry, budget, augment=route_length_augment)
            print(
                f"  [dual-path budget] L={float(budget):.0f}m "
                f"(both branches capped; shorter keeps full geometry)\n"
                f"    baseline: {entry['turn_length_m']:.0f}m"
                f"{' CUT' if trimmed.get('baseline_truncated') else ' ok'} "
                f"→ {entry.get('baseline_destination_lane_id')} "
                f"@ along={entry.get('baseline_destination_max_along_m')}\n"
                f"    compliant: {entry['straight_length_m']:.0f}m"
                f"{' CUT' if trimmed.get('compliant_truncated') else ' ok'} "
                f"→ {entry.get('compliant_destination_lane_id')} "
                f"@ along={entry.get('compliant_destination_max_along_m')}"
            )

    if junction_layout_cache is not None:
        layout = dict(junction_layout_cache)
        if dual_path is not None and dual_path.junction_id:
            if str(layout.get("junction_id")) != str(dual_path.junction_id):
                layout["junction_id"] = dual_path.junction_id
        entry["junction_layout"] = layout

    return {k: v for k, v in entry.items() if v is not None}


# Aliases kept for older expand callers.
OneWaySimParams = DualPathSimParams
DirectionSimParams = DualPathSimParams
NoTurnSimParams = DualPathSimParams
NoEntrySimParams = DualPathSimParams
OneWayExpansionConfig = DualPathExpansionConfig
DirectionExpansionConfig = DualPathExpansionConfig
NoTurnExpansionConfig = DualPathExpansionConfig
NoEntryExpansionConfig = DualPathExpansionConfig
one_way_geometry_key = dual_path_geometry_key
direction_geometry_key = dual_path_geometry_key
no_turn_geometry_key = dual_path_geometry_key
no_entry_geometry_key = dual_path_geometry_key
expand_one_way_scene_entries = expand_dual_path_scene_entries
expand_direction_scene_entries = expand_dual_path_scene_entries
expand_no_turn_scene_entries = expand_dual_path_scene_entries
expand_no_entry_scene_entries = expand_dual_path_scene_entries
build_one_way_manifest_entry = build_dual_path_manifest_entry
build_direction_manifest_entry = build_dual_path_manifest_entry
build_no_turn_manifest_entry = build_dual_path_manifest_entry
build_no_entry_manifest_entry = build_dual_path_manifest_entry

from functools import partial

from traffic_bench.eval.engine.map.junction_priority_layout import allowed_shapes_for_mode
from traffic_bench.eval.manifest.io import (
    append_scene_entries,
    apply_pending_scene_rejects,
    apply_scene_row_quota,
    apply_split_filter,
    assert_rejected_scenes_applied,
    discover_scenes,
    load_scene_metadata,
    sign_split_quota,
    write_real_manifest,
)
from traffic_bench.eval.manifest.lanes import parse_sumo_net_for_spawn_lanes
from traffic_bench.eval.signs.junction.expand import build_junction_layout_for_scene
from traffic_bench.scene_collection.sign_scenes.materialize.pool_index import normalize_split
_DUAL_PATH_PLACEMENT = {
    "one_way": (
        "OneWayEntrySign on ego approach "
        "(sign_distance_before_end); dual-path compliant nav installed at episode start"
    ),
    "direction": (
        "LaneAllowedDirectionSign on ego approach "
        "(sign_distance_before_end); dual-path compliant nav installed at episode start"
    ),
    "no_turn": (
        "NoLeft/RightTurnSign on ego approach "
        "(sign_distance_before_end); dual-path compliant nav installed at episode start"
    ),
    "no_entry": (
        "NoEntrySign on baseline first exit "
        "(sign_road_id / sign_distance_from_start≈5m); "
        "dual-path compliant nav installed at episode start"
    ),
}



def generate(cfg, scenes=None):
    """Dual-path (direction / one-way / no-turn / no-entry) rows."""

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
    apply_pending_scene_rejects(scenes_dir)
    assert_rejected_scenes_applied(scenes_dir)
    all_scenes = discover_scenes(scenes_dir)
    print(f"Scenes root: {scenes_dir.resolve()}")
    print(f"Discovered {len(all_scenes)} scene(s) on disk")
    scenes, split_by_id = apply_split_filter(
        all_scenes, scenes_dir=scenes_dir, split=split
    )
    n_variations = max(1, int(sim_cfg.n_variations))
    print(
        f"Augmentation axes: layout={expansion_cfg.layout_on}, "
        f"{describe_world_axes(sim_cfg)}"
    )

    expansion = DualPathExpansionConfig(
        layout=expansion_cfg.layout_on,
        max_scenarios=scenario_cfg.max_scenarios,
        arm_counts=(3, 4),
    )
    sim_params = DualPathSimParams(
        spawn_distance_before_end=sim_cfg.spawn_distance_before_end,
        sign_distance_before_end=sim_cfg.sign_distance_before_end,
        spawn_velocity_ms=sim_cfg.spawn_velocity_ms,
        horizon=sim_cfg.horizon,
        n_variations=n_variations,
        profile_density_cap=float(sim_cfg.profile_density_cap),
        traffic_density_levels=tuple(
            float(x)
            for x in (
                getattr(sim_cfg, "traffic_density_levels", None)
                or DEFAULT_TRAFFIC_DENSITY_LEVELS
            )
        ),
        spawn_velocity_levels_ms=tuple(
            float(x)
            for x in (
                getattr(sim_cfg, "spawn_velocity_levels_ms", None)
                or DEFAULT_SPAWN_VELOCITY_LEVELS_MS
            )
        ),
        min_dual_path_gain_m=float(scenario_cfg.min_dual_path_gain_m),
        min_ego_lane_m=min(float(sim_cfg.spawn_distance_before_end), 8.0),
        max_path_length_m=float(sim_cfg.max_path_length_m),
        max_path_length_levels=tuple(
            float(x)
            for x in getattr(sim_cfg, "max_path_length_levels", DEFAULT_ROUTE_LENGTH_LEVELS_M)
        ),
        default_first_variant=bool(getattr(sim_cfg, "default_first_variant", True)),
    )

    family = profile.sign_type
    print(
        f"[{family}] world grid: {describe_world_axes(sim_params)} "
        f"(density_cap={sim_cfg.profile_density_cap}, "
        f"min_gain={scenario_cfg.min_dual_path_gain_m}m)"
    )

    build_entry = partial(
        build_dual_path_manifest_entry,
        pdd_code=PDD_CODE,
        sign_type=SIGN_TYPE,
    )

    entries: List[Dict] = []
    used_scene_ids: List[str] = []
    min_lane = min(float(sim_cfg.spawn_distance_before_end), 8.0)
    max_total = scenario_cfg.max_total
    scene_cache: Dict[str, Dict[str, Any]] = {}
    n_maps_quota = sign_split_quota(PDD_CODE, split)

    from traffic_bench.eval.sign_registry import hydra_sign_override
    from traffic_bench.eval.signs.dual_path.route_probe import (
        _run_materialize_refill,
        close_probe_envs,
    )
    from traffic_bench.scene_collection.sign_scenes.filter.selection import (
        apply_rejected_scenes,
        set_scene_reject,
    )

    def _expand_scene(scene_dir: Path, cap: Optional[int]) -> List[Dict]:
        meta = load_scene_metadata(scene_dir)
        scene_name = meta.get("scene_name", scene_dir.name)
        net_file = meta.get("net_file", "map.net.xml")
        net_full_path = scene_dir / net_file
        cached = scene_cache.get(scene_dir.name)
        if cached is None:
            print(f"\n=== {scene_name} ===")
            spawn_lanes = parse_sumo_net_for_spawn_lanes(net_full_path, min_length=min_lane)
            print(f"  Found {len(spawn_lanes)} intersection-approaching lane(s)")
            sign_lat = meta.get("latitude") or meta.get("center_lat")
            sign_lon = meta.get("longitude") or meta.get("center_lon")
            junction_layout = build_junction_layout_for_scene(
                net_full_path,
                profile=profile,
                sign_lat=float(sign_lat) if sign_lat is not None else None,
                sign_lon=float(sign_lon) if sign_lon is not None else None,
                scene_meta=meta,
            )
            cached = {
                "meta": meta,
                "net_full_path": net_full_path,
                "spawn_lanes": spawn_lanes,
                "junction_layout": junction_layout,
            }
            scene_cache[scene_dir.name] = cached
        else:
            meta = cached["meta"]
            net_full_path = cached["net_full_path"]
            spawn_lanes = cached["spawn_lanes"]
            junction_layout = cached["junction_layout"]
        if junction_layout is None:
            print(f"  Skipping {scene_name}: no junction layout")
            return []
        shape = junction_layout.get("shape")
        allowed_shapes = allowed_shapes_for_mode(profile.layout_mode)
        if shape not in allowed_shapes:
            print(
                f"  Skipping {scene_name}: junction shape {shape!r} "
                f"(need {sorted(allowed_shapes)})"
            )
            return []
        local_expansion = DualPathExpansionConfig(
            layout=expansion.layout,
            max_scenarios=cap,
            max_dual_paths=expansion.max_dual_paths,
            arm_counts=expansion.arm_counts,
        )
        return expand_dual_path_scene_entries(
            scene_dir=scene_dir,
            scenes_root=scenes_dir,
            meta=meta,
            net_path=net_full_path,
            spawn_lanes=spawn_lanes,
            junction_layout=junction_layout,
            sim=sim_params,
            expansion=local_expansion,
            build_entry=build_entry,
            pdd_code=PDD_CODE,
        )

    empty_names: List[str] = []

    def _reject_unexpandable(scene_dir: Path, why: str) -> None:
        meta = scene_cache.get(scene_dir.name, {}).get("meta") or load_scene_metadata(
            scene_dir
        )
        scene_name = meta.get("scene_name", scene_dir.name)
        print(f"  Skipping {scene_name}: {why}")
        set_scene_reject(
            scenes_dir,
            scene_dir.name,
            reason="no_expand_dests",
            detail=why,
        )
        empty_names.append(scene_dir.name)
        apply_rejected_scenes(scenes_dir, only=[scene_dir.name])

    def _expand_and_append(scene_dir: Path) -> bool:
        scene_entries = _expand_scene(scene_dir, expansion.max_scenarios)
        if not scene_entries:
            _reject_unexpandable(scene_dir, "no manifest entries after expansion")
            return False
        append_scene_entries(
            entries, used_scene_ids, scene_entries,
            scene_dir=scene_dir, meta=scene_cache[scene_dir.name]["meta"],
            split_by_id=split_by_id,
        )
        return True

    def _flush_empty() -> None:
        if not empty_names:
            return
        moved, n = apply_rejected_scenes(scenes_dir, only=list(empty_names))
        print(
            f"[expand] moved {moved}/{n} unexpandable scene(s) to _rejected/",
            flush=True,
        )
        empty_names.clear()

    try:
        for scene_dir in scenes:
            _expand_and_append(scene_dir)
        _flush_empty()

        sign_token = hydra_sign_override(profile)
        for _map_i in range(10):
            n_unique = len(set(used_scene_ids))
            if n_maps_quota is None or n_unique >= int(n_maps_quota):
                break
            print(
                f"[expand] unique maps {n_unique}/{n_maps_quota}; "
                f"materialize --refill and expand replacements",
                flush=True,
            )
            close_probe_envs()
            rc = _run_materialize_refill(sign_token)
            if rc:
                print(f"[expand] warn: refill exited {rc}", flush=True)
            all_scenes = discover_scenes(scenes_dir)
            scenes, split_by_id = apply_split_filter(
                all_scenes, scenes_dir=scenes_dir, split=split
            )
            used = set(used_scene_ids)
            new_dirs = [s for s in scenes if s.name not in used]
            if not new_dirs:
                print(
                    "[expand] refill added no new maps in this split; "
                    "crop pool may be exhausted",
                    flush=True,
                )
                break
            for scene_dir in new_dirs:
                _expand_and_append(scene_dir)
            _flush_empty()

        n_unique = len(set(used_scene_ids))
        at_map_quota = n_maps_quota is None or n_unique >= int(n_maps_quota)
        if (
            max_total is not None
            and int(max_total) > 0
            and len(entries) < int(max_total)
            and at_map_quota
        ):
            print(
                f"[max_total] {len(entries)}/{max_total} after {n_unique} maps; "
                f"refilling leftover combos (unique-map quota already met)",
                flush=True,
            )
            have = {dual_path_geometry_key(e) for e in entries}
            used = set(used_scene_ids)

            def _merge_leftover(cap: Optional[int]) -> int:
                added_n = 0
                for scene_dir in scenes:
                    if scene_dir.name not in used:
                        continue
                    extra = _expand_scene(scene_dir, cap)
                    added = []
                    for entry in extra:
                        key = dual_path_geometry_key(entry)
                        if key in have:
                            continue
                        have.add(key)
                        added.append(entry)
                    if not added:
                        continue
                    meta = scene_cache[scene_dir.name]["meta"]
                    scene_name = meta.get("scene_name", scene_dir.name)
                    scene_split = split_by_id.get(scene_name) or split_by_id.get(
                        scene_dir.name
                    )
                    for entry in added:
                        entry["split"] = scene_split
                    entries.extend(added)
                    added_n += len(added)
                    print(
                        f"  [max_total] {scene_dir.name}: +{len(added)} leftover combos",
                        flush=True,
                    )
                return added_n

            base_cap = expansion.max_scenarios
            if base_cap is not None:
                shortfall = int(max_total) - len(entries)
                n_ok = max(len(used), 1)
                raised = int(base_cap) + math.ceil(shortfall / n_ok)
                _merge_leftover(raised)
            if len(entries) < int(max_total):
                _merge_leftover(None)
        elif (
            max_total is not None
            and int(max_total) > 0
            and len(entries) < int(max_total)
            and not at_map_quota
        ):
            print(
                f"[expand] skip leftover combo padding: "
                f"{n_unique}/{n_maps_quota} unique maps (want rows from new maps)",
                flush=True,
            )
    finally:
        close_probe_envs()

    entries, used_scene_ids, pre_total = apply_scene_row_quota(
        entries, used_scene_ids,
        n_maps=n_maps_quota,
        max_scenarios=expansion.max_scenarios,
        max_total=max_total,
        split=split,
        pdd_code=PDD_CODE,
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
            "sign_class": profile.sign_type,
            "sign_placement": _DUAL_PATH_PLACEMENT.get(
                family, _DUAL_PATH_PLACEMENT["direction"]
            ),
            "total_scenes": len(used_scene_ids),
            "total_entries": len(entries),
            "total_entries_before_max_total": pre_total,
            "augmentation_layout": expansion_cfg.layout_on,
            "n_variations": n_variations,
            "npc_world": "engine.traffic.agent_profile_bank.sample_one_profile",
            "profile_density_cap": sim_cfg.profile_density_cap,
            "min_dual_path_gain_m": scenario_cfg.min_dual_path_gain_m,
            "max_scenarios": scenario_cfg.max_scenarios,
            "max_total": scenario_cfg.max_total,
            "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
            "horizon": sim_cfg.horizon,
            "sign_distance_before_end": sim_cfg.sign_distance_before_end,
            "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
            "max_path_length_m": float(sim_cfg.max_path_length_m),
            "max_path_length_levels": list(
                getattr(sim_cfg, "max_path_length_levels", DEFAULT_ROUTE_LENGTH_LEVELS_M)
            ),
            "auxiliary_agent": False,
        },
    )
    return entries

