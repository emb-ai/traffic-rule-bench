"""Expand blocked-road (3.2) scenes into manifest rows (layout × world grid).

NPC world params come from ``sample_profile_for_cell`` over the shared
route × density × spawn-velocity × NPC axes. Geometry expansion stays here.

``max_scenarios`` caps the combined (layout × world-grid) pool after shuffle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from traffic_bench.eval.engine.expand.manifest_config import (
    DEFAULT_SPAWN_VELOCITY_LEVELS_MS,
    DEFAULT_TRAFFIC_DENSITY_LEVELS,
)
from traffic_bench.eval.engine.expand.manifest_expansion import (
    mark_nominal_row,
    shuffle_cap,
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
from traffic_bench.eval.engine.spawn.route_budget import (
    apply_route_budget,
    measure_spawn_to_dest_length_m,
)
from traffic_bench.eval.engine.spawn.route_length_levels import (
    list_route_length_levels,
    select_route_length_levels,
    tag_entry_route_length,
)
from traffic_bench.eval.engine.traffic.npc_profile import embed_npc_profile
from traffic_bench.eval.engine.traffic.stable_hash import stable_hash
from traffic_bench.eval.engine.spawn.scene_augmentation import (
    SpawnScenario,
    augment_layout_for_scene,
)
from traffic_bench.eval.signs.blocked.spec import forbidden_edge_geometry_ok


@dataclass(frozen=True)
class BlockedRoadSimParams:
    sign_distance_from_start: float
    spawn_distance_before_end: float
    spawn_velocity_ms: float
    horizon: int = DEFAULT_HORIZON_STEPS
    compliant_stop_success_seconds: float = 3.0
    compliant_stop_max_dist_m: float = 12.0
    compliant_stop_speed_mps: float = 0.5
    n_variations: int = 3
    profile_density_cap: float = 1.0
    traffic_density_levels: Tuple[float, ...] = DEFAULT_TRAFFIC_DENSITY_LEVELS
    spawn_velocity_levels_ms: Tuple[float, ...] = DEFAULT_SPAWN_VELOCITY_LEVELS_MS
    max_path_length_m: float = DEFAULT_MAX_PATH_LENGTH_M
    max_path_length_levels: Tuple[float, ...] = DEFAULT_ROUTE_LENGTH_LEVELS_M
    default_first_variant: bool = True


@dataclass(frozen=True)
class BlockedRoadExpansionConfig:
    layout: bool = True
    max_scenarios: Optional[int] = None
    validate_metadrive_routes: bool = False


BuildBlockedRoadEntryFn = Callable[..., Dict[str, Any]]


def blocked_road_geometry_key(entry: Dict[str, Any]) -> Tuple:
    return (
        entry.get("road_id"),
        entry.get("spawn_lane_num"),
        entry.get("destination_lane_id"),
        entry.get("var_idx"),
        entry.get("density_level_id"),
        entry.get("spawn_velocity_level_id"),
        round(float(entry.get("route_length_level_m") or entry.get("max_path_length_m") or 0.0), 1),
    )


def expand_blocked_road_scene_entries(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    net_path: Path,
    spawn_lanes: Sequence[Any],
    junction_layout: dict,
    sim: BlockedRoadSimParams,
    expansion: BlockedRoadExpansionConfig,
    build_entry: BuildBlockedRoadEntryFn,
) -> List[Dict[str, Any]]:
    """Expand one scene: layout through-paths × shared world grid."""
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    print(
        f"  Junction layout: {junction_layout['shape']} @ {junction_layout['junction_id']} "
        f"(arms={len(junction_layout.get('arms', []))})"
    )

    sign_lat = meta.get("latitude") or meta.get("center_lat")
    sign_lon = meta.get("longitude") or meta.get("center_lon")

    scenarios: List[Optional[SpawnScenario]] = []
    if expansion.layout:
        _, layout_scenarios = augment_layout_for_scene(
            net_path,
            list(spawn_lanes),
            strategy="blocked_road",
            min_lane_length=float(sim.spawn_distance_before_end),
            sign_lat=float(sign_lat) if sign_lat is not None else None,
            sign_lon=float(sign_lon) if sign_lon is not None else None,
            scene_meta=meta,
        )
        if not layout_scenarios:
            print(f"  [augment] No spawn×exit scenarios for {scene_name}; skipping scene")
            return []
        scenarios = list(layout_scenarios)
        print(f"  Spawn×exit scenarios: {len(scenarios)}")
    else:
        scenarios = [None]
        print("  Layout axis off: one default spawn per scene")

    layout_kept: List[Tuple[int, Optional[SpawnScenario]]] = []
    skipped_geometry = 0
    for layout_i, scenario in enumerate(scenarios):
        if scenario is not None:
            geom_ok, geom_reason = forbidden_edge_geometry_ok(
                net_path,
                scenario.ego_destination_edge_id,
                sign_distance_from_start=sim.sign_distance_from_start,
            )
            if not geom_ok:
                skipped_geometry += 1
                print(
                    f"  [skip] forbidden-lane geometry {geom_reason} "
                    f"({scenario.scenario_id})"
                )
                continue
        layout_kept.append((layout_i, scenario))

    if skipped_geometry:
        print(f"  [geometry] Skipped {skipped_geometry} layout(s) (forbidden edge too short)")

    configured_route_levels = list_route_length_levels(sim)
    spawn_before_end = float(sim.spawn_distance_before_end)
    scene_entries: List[Dict[str, Any]] = []
    seen: set = set()

    # One no-NPC reference row per map (first kept layout).
    if bool(sim.default_first_variant) and layout_kept:
        layout_i, scenario = layout_kept[0]
        scenario_id = scenario.scenario_id if scenario is not None else ""
        seed = stable_hash(scene_name, scenario_id, "nominal")
        nominal = build_entry(
            scene_dir=scene_dir,
            scenes_root=scenes_root,
            meta=meta,
            layout_variant=layout_i,
            var_idx=0,
            seed=seed,
            sim=sim,
            spawn_scenario=scenario,
            spawn_lanes_cache=list(spawn_lanes),
            junction_layout_cache=junction_layout,
            npc_profile=None,
            max_path_length_m=float(sim.max_path_length_m),
            route_length_augment=False,
            spawn_velocity_ms=float(sim.spawn_velocity_ms),
            traffic_density=0.0,
        )
        scene_entries.append(mark_nominal_row(nominal))

    for layout_i, scenario in layout_kept:
        scenario_id = scenario.scenario_id if scenario is not None else ""
        available_route_m = None
        if scenario is not None:
            available_route_m = measure_spawn_to_dest_length_m(
                net_path=net_path,
                spawn_edge=str(scenario.ego_edge_id),
                spawn_lane=int(scenario.ego_lane_num),
                dest_edge=str(scenario.ego_destination_edge_id),
                spawn_distance_before_end=spawn_before_end,
            )
        route_levels, route_augment = select_route_length_levels(
            configured_route_levels, available_route_m
        )
        for cell in iter_world_axis_cells(
            route_levels=route_levels,
            sim=sim,
            task_conditioned_spawn=False,
        ):
            suffix = cell.scene_suffix(route_augment=route_augment)
            seed = stable_hash(scene_name, scenario_id, *cell.seed_tags())
            profile = sample_profile_for_cell(cell, seed=int(seed), sim=sim)
            entry = build_entry(
                scene_dir=scene_dir,
                scenes_root=scenes_root,
                meta=meta,
                layout_variant=layout_i,
                var_idx=cell.npc_var,
                seed=seed,
                sim=sim,
                spawn_scenario=scenario,
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
            key = blocked_road_geometry_key(entry)
            if key in seen:
                continue
            seen.add(key)
            scene_entries.append(entry)

    cap = expansion.max_scenarios
    pre_cap = len(scene_entries)
    scene_entries = shuffle_cap(
        scene_entries,
        cap,
        seed_key=(scene_name, "blocked_road_world_cap", int(cap) if cap is not None else 0),
    )
    if cap is not None and pre_cap > cap:
        print(
            f"  Retained {len(scene_entries)} of {pre_cap} world-grid variants "
            f"(shuffled, cap={cap}; nominal preserved; {len(layout_kept)} layouts)"
        )
    else:
        print(
            f"  Manifest entries for {scene_name}: {len(scene_entries)} "
            f"({len(layout_kept)} layouts × world grid)"
        )
    return scene_entries


def build_blocked_road_manifest_entry(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    layout_variant: int,
    var_idx: int,
    seed: int,
    sim: BlockedRoadSimParams,
    spawn_scenario: Optional[SpawnScenario],
    spawn_lanes_cache: Optional[List[Any]],
    junction_layout_cache: Optional[dict],
    npc_profile: Optional[Dict[str, Any]],
    pdd_code: str,
    sign_type: str,
    sign_class: str = "NoTrafficSign",
    sign_title: str = "Movement prohibited",
    max_path_length_m: Optional[float] = None,
    route_length_augment: bool = False,
    spawn_velocity_ms: Optional[float] = None,
    traffic_density: Optional[float] = None,
    scene_id_suffix: str = "",
) -> Dict[str, Any]:
    """Build one manifest row for a through-path + world-grid cell."""
    del layout_variant  # encoded in augmentation_id / spawn fields
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

    sign_road_id = ""
    if spawn_scenario is not None:
        sign_road_id = str(spawn_scenario.ego_destination_edge_id or "").strip()

    entry: Dict[str, Any] = {
        "scene_id": scene_id,
        "scene_name": scene_name,
        "net_path": str(net_rel),
        "seed": int(seed),
        "var_idx": int(var_idx),
        "pdd_code": pdd_code,
        "sign_code": pdd_code,
        "sign_type": sign_type,
        "sign_family": "blocked_road",
        "sign_title": sign_title,
        "sign_class": sign_class,
        "spawn_velocity_ms": v0,
        "traffic_density": row_density,
        "horizon": horizon,
        "sign_road_id": sign_road_id,
        "sign_distance_from_start": sim.sign_distance_from_start,
        "spawn_distance_before_end": sim.spawn_distance_before_end,
        "compliant_stop_success_seconds": sim.compliant_stop_success_seconds,
        "compliant_stop_max_dist_m": sim.compliant_stop_max_dist_m,
        "compliant_stop_speed_mps": sim.compliant_stop_speed_mps,
        "valid": True,
        "auxiliary_agent": False,
        "latitude": meta.get("latitude"),
        "longitude": meta.get("longitude"),
        "crop_radius_m": meta.get("crop_radius_m"),
        "source_osm": meta.get("source_osm"),
        "osm_file": meta.get("osm_file"),
    }

    # Same profile_* embedding as sumo_runner.materialize_sumo_scene.
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

    if junction_layout_cache is not None:
        entry["junction_layout"] = junction_layout_cache

    path_budget_m = float(
        max_path_length_m if max_path_length_m is not None else sim.max_path_length_m
    )

    max_path_m = path_budget_m
    if max_path_m > 0.0 and entry.get("destination_edge_id"):
        entry = apply_route_budget(
            entry,
            net_path=scene_dir / net_file,
            max_path_length_m=max_path_m,
            spawn_distance_before_end=float(sim.spawn_distance_before_end),
        )
    entry = tag_entry_route_length(entry, path_budget_m, augment=route_length_augment)

    return {k: v for k, v in entry.items() if v is not None}

from functools import partial

from traffic_bench.eval.engine.map.junction_priority_layout import allowed_shapes_for_mode
from traffic_bench.eval.manifest.io import (
    append_scene_entries,
    apply_max_total,
    apply_split_filter,
    assert_rejected_scenes_applied,
    discover_scenes,
    load_scene_metadata,
    write_real_manifest,
)
from traffic_bench.eval.manifest.lanes import parse_sumo_net_for_spawn_lanes
from traffic_bench.eval.signs.junction.expand import build_junction_layout_for_scene
from traffic_bench.scene_collection.sign_scenes.materialize.pool_index import normalize_split


def generate(cfg, scenes=None):
    """Blocked-road (3.2) rows from harvested scenes."""

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

    blocked_road_expansion = BlockedRoadExpansionConfig(
        layout=expansion_cfg.layout_on,
        max_scenarios=scenario_cfg.max_scenarios,
    )
    sim_params = BlockedRoadSimParams(
        sign_distance_from_start=sim_cfg.sign_distance_from_start,
        spawn_distance_before_end=sim_cfg.spawn_distance_before_end,
        spawn_velocity_ms=sim_cfg.spawn_velocity_ms,
        horizon=sim_cfg.horizon,
        compliant_stop_success_seconds=sim_cfg.compliant_stop_success_seconds,
        compliant_stop_max_dist_m=sim_cfg.compliant_stop_max_dist_m,
        compliant_stop_speed_mps=sim_cfg.compliant_stop_speed_mps,
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
        max_path_length_m=float(sim_cfg.max_path_length_m),
        max_path_length_levels=tuple(
            float(x)
            for x in getattr(sim_cfg, "max_path_length_levels", DEFAULT_ROUTE_LENGTH_LEVELS_M)
        ),
        default_first_variant=bool(getattr(sim_cfg, "default_first_variant", True)),
    )

    print(
        f"[blocked_road] world grid: {describe_world_axes(sim_params)} "
        f"(density_cap={sim_cfg.profile_density_cap})"
    )

    build_entry = partial(
        build_blocked_road_manifest_entry,
        pdd_code=PDD_CODE,
        sign_type=SIGN_TYPE,
        sign_class="NoTrafficSign",
        sign_title="Movement prohibited",
    )

    entries: List[Dict] = []
    used_scene_ids: List[str] = []
    min_lane = min(float(sim_cfg.spawn_distance_before_end), 8.0)

    for scene_dir in scenes:
        meta = load_scene_metadata(scene_dir)
        scene_name = meta.get("scene_name", scene_dir.name)
        net_file = meta.get("net_file", "map.net.xml")
        net_full_path = scene_dir / net_file
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
        if junction_layout is None:
            print(f"  Skipping {scene_name}: no junction layout")
            continue
        shape = junction_layout.get("shape")
        allowed_shapes = allowed_shapes_for_mode(profile.layout_mode)
        if shape not in allowed_shapes:
            print(
                f"  Skipping {scene_name}: junction shape {shape!r} "
                f"(need {sorted(allowed_shapes)})"
            )
            continue

        scene_entries = expand_blocked_road_scene_entries(
            scene_dir=scene_dir,
            scenes_root=scenes_dir,
            meta=meta,
            net_path=net_full_path,
            spawn_lanes=spawn_lanes,
            junction_layout=junction_layout,
            sim=sim_params,
            expansion=blocked_road_expansion,
            build_entry=build_entry,
        )
        if not scene_entries:
            print(f"  Skipping {scene_name}: no manifest entries after expansion")
            continue
        append_scene_entries(
            entries, used_scene_ids, scene_entries,
            scene_dir=scene_dir, meta=meta, split_by_id=split_by_id,
        )

    entries, used_scene_ids, pre_total = apply_max_total(
        entries, used_scene_ids,
        max_total=scenario_cfg.max_total, split=split, pdd_code=PDD_CODE,
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
            "sign_class": "NoTrafficSign",
            "sign_placement": (
                "artificial at start of forbidden (destination) lane "
                "(sign_distance_from_start); ego on approach "
                "(spawn_distance_before_end)"
            ),
            "total_scenes": len(used_scene_ids),
            "total_entries": len(entries),
            "total_entries_before_max_total": pre_total,
            "augmentation_layout": expansion_cfg.layout_on,
            "n_variations": n_variations,
            "npc_world": "engine.traffic.agent_profile_bank.sample_one_profile",
            "profile_density_cap": sim_cfg.profile_density_cap,
            "max_scenarios": scenario_cfg.max_scenarios,
            "max_total": scenario_cfg.max_total,
            "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
            "horizon": sim_cfg.horizon,
            "sign_distance_from_start": sim_cfg.sign_distance_from_start,
            "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
            "max_path_length_m": float(sim_cfg.max_path_length_m),
            "max_path_length_levels": list(
                getattr(sim_cfg, "max_path_length_levels", DEFAULT_ROUTE_LENGTH_LEVELS_M)
            ),
            "compliant_stop_success_seconds": sim_cfg.compliant_stop_success_seconds,
            "compliant_stop_max_dist_m": sim_cfg.compliant_stop_max_dist_m,
            "compliant_stop_speed_mps": sim_cfg.compliant_stop_speed_mps,
            "auxiliary_agent": False,
        },
    )
    return entries

