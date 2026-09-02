"""Expand junction / roundabout scenes into manifest rows (layout × aux × density).

Roundabout (4.3) still uses this cartesian product; its plates live in
``signs/roundabout/place.py``. Spawn combinatorics: ``signs/junction/spawn.py``
and ``signs/roundabout/spawn.py``.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from traffic_bench.eval.engine.map.junction_priority_layout import right_arm_edge_id
from traffic_bench.eval.engine.expand.manifest_config import DEFAULT_STOP_WAIT_STEPS
from traffic_bench.eval.engine.expand.manifest_expansion import (
    AuxiliaryParams,
    ExpansionConfig,
    entry_geometry_key,
    shuffle_cap,
    sizes_up_to,
)
from traffic_bench.eval.engine.spawn.auxiliary_agent import (
    main_lane_keys_for_aux,
    min_aux_spawn_lane_length,
    resolve_aux_destination_lane_key,
    right_lane_keys_for_aux,
    select_occupied_main_lanes,
    viable_aux_lane_keys,
    viable_right_aux_lane_keys,
)
from traffic_bench.eval.engine.spawn.route_budget import (
    apply_route_budget,
    ensure_reachable_ego_destination,
    measure_spawn_to_dest_length_m,
)
from traffic_bench.eval.engine.spawn.route_length_levels import (
    list_route_length_levels,
    select_route_length_levels,
    tag_entry_route_length,
)
from traffic_bench.eval.engine.traffic.agent_profile_bank import sample_one_profile
from traffic_bench.eval.engine.traffic.npc_profile import embed_npc_profile
from traffic_bench.eval.engine.traffic.stable_hash import stable_hash
from traffic_bench.eval.signs.junction.nav import outgoing_edges_from_junction_layout
from traffic_bench.eval.engine.spawn.scene_augmentation import (
    SpawnScenario,
    SpawnStrategy,
    augment_layout_for_scene,
)
from traffic_bench.eval.signs.junction.spawn import (
    pick_default_main_spawn_meta_for_net,
    pick_default_yield_spawn_meta_for_net,
)
from traffic_bench.eval.signs.roundabout.spawn import (
    pick_default_roundabout_spawn_meta_for_net,
)
from traffic_bench.eval.engine.map.lane_keys import lane_edge_id, lane_num_from_key, make_lane_key
from traffic_bench.eval.engine.map.sumo_utils import load_vehicle_route_index
from traffic_bench.eval.manifest.lanes import (
    SumoLaneInfo,
    filter_spawn_lanes_to_secondary,
    parse_sumo_net_for_spawn_lanes,
    select_random_spawn_lane,
)
from traffic_bench.eval.sign_registry import STOP, SignProfile


BuildEntryFn = Callable[..., Dict]


def _stable_seed(
    scene_name: str,
    variant: int = 0,
    scenario_id: str = "",
    convoy_size: int = 0,
    lanes_occupied: int = 0,
    convoy_gap_m: float = 0.0,
    path_length_m: float = 0.0,
    npc_var_idx: int = 0,
) -> int:
    """Deterministic 32-bit seed from scene name, variant, scenario, and aux dims."""
    h = hashlib.sha256()
    h.update(scene_name.encode("utf-8"))
    h.update(b"|")
    h.update(str(variant).encode("utf-8"))
    if scenario_id:
        h.update(b"|")
        h.update(scenario_id.encode("utf-8"))
    if convoy_size > 0:
        h.update(b"|convoy")
        h.update(str(convoy_size).encode("utf-8"))
    if lanes_occupied > 0:
        h.update(b"|lanes")
        h.update(str(lanes_occupied).encode("utf-8"))
    if convoy_size > 1 and convoy_gap_m > 0:
        h.update(b"|gap")
        h.update(f"{float(convoy_gap_m):.3f}".encode("utf-8"))
    if path_length_m > 0:
        h.update(b"|rl")
        h.update(f"{float(path_length_m):.1f}".encode("utf-8"))
    if npc_var_idx > 0:
        h.update(b"|npc")
        h.update(str(npc_var_idx).encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")


def _fit_aux_lane_keys(
    *,
    junction_layout: dict,
    spawn_strategy: str,
    aux: AuxiliaryParams,
    ego_edge: Optional[str],
    convoy_size: int,
    convoy_gap_m: Optional[float] = None,
) -> List[str]:
    gap = float(aux.convoy_gap_m if convoy_gap_m is None else convoy_gap_m)
    if spawn_strategy in ("yield", "roundabout"):
        return viable_aux_lane_keys(
            junction_layout,
            aux.distance_from_intersection,
            ego_edge,
            convoy_size=convoy_size,
            convoy_gap_m=gap,
        )
    return viable_right_aux_lane_keys(
        junction_layout,
        aux.distance_from_intersection,
        ego_edge,
        convoy_size=convoy_size,
        convoy_gap_m=gap,
    )


def _scene_aux_lane_keys_for_lane_axis(
    *,
    junction_layout: dict,
    spawn_strategy: str,
    auxiliary_on: bool,
    aux: Optional[AuxiliaryParams],
    ego_edge: Optional[str],
) -> List[str]:
    """Lane pool for the lanes-occupied axis (lead-only length when aux on)."""
    if not auxiliary_on or aux is None:
        if spawn_strategy in ("yield", "roundabout"):
            return main_lane_keys_for_aux(junction_layout, ego_edge)
        return right_lane_keys_for_aux(junction_layout, ego_edge)
    return _fit_aux_lane_keys(
        junction_layout=junction_layout,
        spawn_strategy=spawn_strategy,
        aux=aux,
        ego_edge=ego_edge,
        convoy_size=1,
    )


def _print_aux_lane_availability(
    *,
    scene_name: str,
    junction_layout: dict,
    spawn_strategy: str,
    expansion: ExpansionConfig,
) -> bool:
    """Log aux slot counts. Return False if the scene should be skipped."""
    aux = expansion.aux
    if not expansion.auxiliary_on or aux is None:
        if spawn_strategy in ("yield", "roundabout"):
            print(
                f"  Main-road lane slots for aux: "
                f"{len(main_lane_keys_for_aux(junction_layout))} (aux axis off)"
            )
        return True

    min_lane_for_lead = min_aux_spawn_lane_length(
        aux.distance_from_intersection,
        convoy_size=1,
        convoy_gap_m=min(aux.convoy_gaps_m) if aux.convoy_gaps_m else 10.0,
    )
    if spawn_strategy == "roundabout":
        from traffic_bench.eval.signs.roundabout.aux import MIN_CONFLICT_ARC_LENGTH_M

        min_lane_for_lead = float(MIN_CONFLICT_ARC_LENGTH_M)
    if spawn_strategy in ("yield", "roundabout"):
        available_keys = viable_aux_lane_keys(
            junction_layout, aux.distance_from_intersection
        )
        available = len(available_keys)
        label = "Conflict-arc ring" if spawn_strategy == "roundabout" else "Main-road"
        print(f"  {label} lane slots for aux: {available}")
        if available <= 0:
            print(
                f"  [aux] No {label.lower()} lanes viable for aux spawning "
                f"(need >={min_lane_for_lead:.0f}m); "
                f"skipping {scene_name}"
            )
            return False
        return True

    available = 0
    if junction_layout.get("arms"):
        sample_ego = junction_layout["arms"][0].get("edge_id")
        if sample_ego:
            right_edge = right_arm_edge_id(junction_layout, sample_ego)
            if right_edge:
                available = sum(
                    len(arm.get("lane_keys", []))
                    for arm in junction_layout["arms"]
                    if arm.get("edge_id") == right_edge
                )
    print(f"  Right-arm lane slots for aux (example): {available}")
    return True


def expand_scene_entries(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict,
    net_path: Path,
    spawn_lanes: Sequence[Any],
    junction_layout: dict,
    spawn_strategy: SpawnStrategy,
    sim_cfg: Any,
    expansion: ExpansionConfig,
    build_entry: BuildEntryFn,
    aux_cfg_for_entry: Any,
) -> List[Dict]:
    """Expand one scene into manifest rows (layout × aux × route × NPC profile).

    ``aux_cfg_for_entry`` is the dataclass passed through to ``build_entry``
    (typically ``AuxiliaryConfig``); when the auxiliary axis is off the caller
    should pass a copy with ``enabled=False``.
    """
    scene_name = meta.get("scene_name", scene_dir.name)
    if spawn_strategy == "roundabout":
        print(
            f"  Junction layout: {junction_layout['shape']} @ {junction_layout['junction_id']} "
            f"(ring={len(junction_layout.get('main_edge_ids', []))}, "
            f"spokes={len(junction_layout.get('secondary_edge_ids', []))})"
        )
    elif spawn_strategy == "yield":
        print(
            f"  Junction layout: {junction_layout['shape']} @ {junction_layout['junction_id']} "
            f"(main={len(junction_layout.get('main_edge_ids', []))}, "
            f"secondary={len(junction_layout.get('secondary_edge_ids', []))})"
        )
    else:
        print(
            f"  Junction layout: {junction_layout['shape']} @ {junction_layout['junction_id']} "
            f"(equal-priority arms={len(junction_layout.get('main_edge_ids', []))})"
        )

    if not _print_aux_lane_availability(
        scene_name=scene_name,
        junction_layout=junction_layout,
        spawn_strategy=spawn_strategy,
        expansion=expansion,
    ):
        return []

    sign_lat = meta.get("latitude") or meta.get("center_lat")
    sign_lon = meta.get("longitude") or meta.get("center_lon")
    aux = expansion.aux
    aux_distance = (
        float(aux.distance_from_intersection) if aux is not None else 20.0
    )

    if expansion.layout_on:
        _, layout_scenarios = augment_layout_for_scene(
            net_path,
            list(spawn_lanes),
            strategy=spawn_strategy,
            aux_distance_from_intersection=aux_distance,
            sign_lat=float(sign_lat) if sign_lat is not None else None,
            sign_lon=float(sign_lon) if sign_lon is not None else None,
            scene_meta=meta,
        )
        if not layout_scenarios:
            print(f"  [augment] No valid scenarios for {scene_name}; skipping scene")
            return []
        scenarios: List[Optional[SpawnScenario]] = list(layout_scenarios)
        print(f"  Augmented spawn scenarios: {len(scenarios)}")
    else:
        scenarios = [None]
        print("  Layout axis off: one default spawn per scene")

    auxiliary_on = expansion.auxiliary_on
    convoy_sizes = sizes_up_to(
        aux.convoy_size if aux is not None else 1,
        auxiliary_enabled=auxiliary_on,
    )
    if auxiliary_on and aux is not None and aux.convoy_gaps_m:
        gap_values = [float(g) for g in aux.convoy_gaps_m]
    elif aux is not None and aux.convoy_gaps_m:
        gap_values = [float(aux.convoy_gaps_m[0])]
    else:
        gap_values = [10.0]

    configured_route_levels = list_route_length_levels(sim_cfg)
    n_variations = max(1, int(getattr(sim_cfg, "n_variations", 3) or 3))
    density_cap = float(getattr(sim_cfg, "profile_density_cap", 1.0) or 1.0)
    spawn_before_end = float(getattr(sim_cfg, "spawn_distance_before_end", 15.0) or 15.0)

    scene_entries: List[Dict] = []
    skipped_short_aux = 0
    skipped_dup_geometry = 0
    skipped_short_route_levels = 0
    skipped_invalid_route = 0
    seen_geometries: set = set()

    for variant, scenario in enumerate(scenarios):
        ego_edge = scenario.ego_edge_id if scenario is not None else None
        prefer_aux = (
            make_lane_key(scenario.aux_edge_id, scenario.aux_lane_num)
            if scenario is not None
            else None
        )
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
        if (
            available_route_m is not None
            and len(configured_route_levels) > 1
            and not route_augment
        ):
            skipped_short_route_levels += len(configured_route_levels) - 1
        scene_aux_lanes = _scene_aux_lane_keys_for_lane_axis(
            junction_layout=junction_layout,
            spawn_strategy=spawn_strategy,
            auxiliary_on=auxiliary_on,
            aux=aux,
            ego_edge=ego_edge,
        )
        scene_lane_counts = sizes_up_to(
            aux.lanes_occupied if aux is not None else 1,
            auxiliary_enabled=auxiliary_on,
            available=len(scene_aux_lanes),
        )
        for lanes_n in scene_lane_counts:
            for convoy_n in convoy_sizes:
                gaps_for_n = gap_values if convoy_n > 1 else gap_values[:1]
                for gap_m in gaps_for_n:
                    for path_len_m in route_levels:
                        for npc_var in range(n_variations):
                            if auxiliary_on and aux is not None:
                                fit_lanes = _fit_aux_lane_keys(
                                    junction_layout=junction_layout,
                                    spawn_strategy=spawn_strategy,
                                    aux=aux,
                                    ego_edge=ego_edge,
                                    convoy_size=convoy_n,
                                    convoy_gap_m=gap_m,
                                )
                                if prefer_aux is not None and prefer_aux not in fit_lanes:
                                    skipped_short_aux += 1
                                    continue
                                if len(fit_lanes) < lanes_n:
                                    skipped_short_aux += 1
                                    continue
                            aux_cfg_gap = replace(aux_cfg_for_entry, convoy_gap_m=gap_m)
                            scenario_id = scenario.scenario_id if scenario else ""
                            seed = stable_hash(
                                scene_name,
                                scenario_id,
                                variant,
                                convoy_n,
                                lanes_n,
                                round(float(gap_m), 3),
                                int(round(float(path_len_m))),
                                npc_var,
                            )
                            npc_profile = sample_one_profile(
                                int(seed),
                                density_cap=density_cap,
                                horizon_steps=int(sim_cfg.horizon),
                            )
                            entry = build_entry(
                                scene_dir=scene_dir,
                                scenes_root=scenes_root,
                                meta=meta,
                                variant=variant,
                                sim_cfg=sim_cfg,
                                aux_cfg=aux_cfg_gap,
                                aux_convoy_size=convoy_n,
                                aux_lanes_occupied=lanes_n,
                                spawn_lanes_cache=list(spawn_lanes),
                                junction_layout_cache=junction_layout,
                                spawn_scenario=scenario,
                                max_path_length_m=float(path_len_m),
                                route_length_augment=route_augment,
                                npc_profile=npc_profile,
                                npc_var_idx=npc_var,
                                seed_override=int(seed),
                            )
                            if entry.get("valid") is False:
                                skipped_invalid_route += 1
                                continue
                            geom_key = entry_geometry_key(entry)
                            if geom_key in seen_geometries:
                                skipped_dup_geometry += 1
                                continue
                            seen_geometries.add(geom_key)
                            scene_entries.append(entry)

    if skipped_short_aux:
        print(
            f"  [aux] Skipped {skipped_short_aux} convoy×lanes×gap combo(s) "
            f"(aux approach too short for full convoy)"
        )
    if skipped_short_route_levels:
        print(
            f"  [route] Collapsed {skipped_short_route_levels} route-length "
            f"level(s) (natural path shorter than configured budgets)"
        )
    if skipped_invalid_route:
        print(
            f"  [route] Skipped {skipped_invalid_route} combo(s) "
            f"(dest lane unreachable from ego spawn lane)"
        )
    if skipped_dup_geometry:
        print(
            f"  [aux] Skipped {skipped_dup_geometry} duplicate combo(s) "
            f"(same ego path + occupied aux lanes + convoy + gap)"
        )

    cap = expansion.max_scenarios
    pre_cap = len(scene_entries)
    scene_entries = shuffle_cap(
        scene_entries,
        cap,
        seed_key=(scene_name, "max_scenarios_shuffle", int(cap) if cap is not None else 0),
    )
    if cap is not None and pre_cap > cap:
        print(
            f"  Retained {len(scene_entries)} of {pre_cap} manifest entries "
            f"for {scene_name} (shuffled, cap={cap})"
        )
    elif cap is not None:
        print(
            f"  Manifest entries for {scene_name}: {len(scene_entries)} "
            f"(under cap={cap})"
        )
    else:
        print(f"  Manifest entries for {scene_name}: {len(scene_entries)}")

    return scene_entries


def build_manifest_entry(
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict,
    variant: int,
    sim_cfg: Any,
    aux_cfg: Any,
    aux_convoy_size: int,
    aux_lanes_occupied: int,
    spawn_lanes_cache: Optional[List[SumoLaneInfo]] = None,
    junction_layout_cache: Optional[dict] = None,
    spawn_scenario: Optional[SpawnScenario] = None,
    expert_cfg: Optional[Any] = None,
    max_path_length_m: Optional[float] = None,
    route_length_augment: bool = False,
    npc_profile: Optional[Dict] = None,
    npc_var_idx: int = 0,
    seed_override: Optional[int] = None,
    *,
    profile: SignProfile,
) -> Dict:
    """Build a single junction / roundabout manifest entry."""
    scene_name = meta.get("scene_name", scene_dir.name)
    net_file = meta.get("net_file", "map.net.xml")

    net_path = scene_dir.relative_to(scenes_root) / net_file
    net_full_path = scene_dir / net_file

    scenario_id = spawn_scenario.scenario_id if spawn_scenario else ""
    path_budget_m = float(
        max_path_length_m
        if max_path_length_m is not None
        else getattr(sim_cfg, "max_path_length_m", 150.0)
    )
    seed = (
        int(seed_override)
        if seed_override is not None
        else _stable_seed(
            scene_name,
            variant,
            scenario_id,
            convoy_size=aux_convoy_size,
            lanes_occupied=aux_lanes_occupied if aux_cfg.enabled else 0,
            convoy_gap_m=float(aux_cfg.convoy_gap_m) if aux_cfg.enabled else 0.0,
            path_length_m=path_budget_m if route_length_augment else 0.0,
            npc_var_idx=npc_var_idx,
        )
    )

    if spawn_lanes_cache is None:
        spawn_lanes_cache = parse_sumo_net_for_spawn_lanes(net_full_path)

    spawn_candidates = spawn_lanes_cache
    if (
        profile.ego_road_class == "secondary"
        and junction_layout_cache is not None
        and spawn_scenario is None
    ):
        spawn_candidates = filter_spawn_lanes_to_secondary(
            spawn_lanes_cache, junction_layout_cache
        )

    selected_lane = None
    if spawn_scenario is not None:
        for lane in spawn_lanes_cache:
            if (
                lane.edge_id == spawn_scenario.ego_edge_id
                and lane.lane_num == spawn_scenario.ego_lane_num
            ):
                selected_lane = lane
                break
    else:
        selected_lane = select_random_spawn_lane(spawn_candidates, seed)

    pdd_code = profile.pdd_code
    entry = {
        "scene_id": scene_name,
        "net_path": str(net_path),
        "seed": seed,
        "var_idx": int(npc_var_idx),
        "pdd_code": pdd_code,
        "sign_code": pdd_code,
        "sign_type": profile.sign_type,
        "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
        "horizon": sim_cfg.horizon,
        "sign_distance_before_end": sim_cfg.sign_distance_before_end,
        "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
        "valid": True,
        "latitude": meta.get("latitude"),
        "longitude": meta.get("longitude"),
        "crop_radius_m": meta.get("crop_radius_m"),
        "source_osm": meta.get("source_osm"),
        "osm_file": meta.get("osm_file"),
        "auxiliary_agent": aux_cfg.enabled,
        "aux_distance_from_intersection": aux_cfg.distance_from_intersection,
        "aux_convoy_size": aux_convoy_size,
        "aux_convoy_gap_m": aux_cfg.convoy_gap_m,
        "aux_lanes_occupied": aux_lanes_occupied,
    }
    if profile.id == STOP.id:
        entry["stop_wait_steps"] = int(
            getattr(expert_cfg, "stop_wait_steps", DEFAULT_STOP_WAIT_STEPS)
            if expert_cfg is not None
            else DEFAULT_STOP_WAIT_STEPS
        )

    if spawn_scenario is not None:
        entry.update(spawn_scenario.to_manifest_fields())
        if aux_cfg.enabled:
            suffix_parts = []
            if aux_lanes_occupied > 1:
                suffix_parts.append(f"lanes{aux_lanes_occupied}")
            if aux_convoy_size > 1:
                suffix_parts.append(f"convoy{aux_convoy_size}")
                gap = float(aux_cfg.convoy_gap_m)
                suffix_parts.append(f"gap{gap:g}")
            if suffix_parts:
                base_aug = entry.get("augmentation_id") or scenario_id
                entry["augmentation_id"] = f"{base_aug}_{'_'.join(suffix_parts)}"
        if selected_lane is not None:
            entry["spawn_lane_length"] = selected_lane.length
            entry["spawn_to_junction"] = selected_lane.to_junction

    if spawn_scenario is None and meta.get("road_id"):
        if profile.ego_road_class == "secondary" and junction_layout_cache is not None:
            secondary_ids = set(junction_layout_cache.get("secondary_edge_ids") or [])
            spawn_edge_ids = {lane.edge_id for lane in (spawn_lanes_cache or [])}
            road_id = str(meta["road_id"])
            road_id_invalid = road_id not in spawn_edge_ids or (
                secondary_ids and road_id not in secondary_ids
            )
            if road_id_invalid:
                print(
                    f"  [spawn] meta road_id {road_id!r} is not a valid secondary approach lane; "
                    "picking from secondary arms"
                )
                selected_lane = select_random_spawn_lane(spawn_candidates, seed)
                if selected_lane is not None:
                    entry["road_id"] = selected_lane.edge_id
                    entry["spawn_lane_num"] = selected_lane.lane_num
                    entry["spawn_lane_length"] = selected_lane.length
                    entry["spawn_to_junction"] = selected_lane.to_junction
            else:
                entry["road_id"] = road_id
                if meta.get("spawn_lane_num") is not None:
                    entry["spawn_lane_num"] = meta["spawn_lane_num"]
        else:
            entry["road_id"] = meta["road_id"]
            if meta.get("spawn_lane_num") is not None:
                entry["spawn_lane_num"] = meta["spawn_lane_num"]
    elif selected_lane is not None:
        entry["road_id"] = selected_lane.edge_id
        entry["spawn_lane_num"] = selected_lane.lane_num
        entry["spawn_lane_length"] = selected_lane.length
        entry["spawn_to_junction"] = selected_lane.to_junction
    elif junction_layout_cache is not None:
        entry["valid"] = False
        print(f"  [spawn] No incoming lanes available for {scene_name}")

    if meta.get("distance_from_start"):
        distance = float(meta["distance_from_start"])
        spawn_len = entry.get("spawn_lane_length")
        if spawn_len is not None:
            distance = min(distance, max(float(spawn_len) - 5.0, 10.0))
        entry["distance_from_start"] = distance
    if meta.get("sign_spawn_distance"):
        entry["sign_spawn_distance"] = meta["sign_spawn_distance"]
    if spawn_scenario is None and meta.get("destination_lane_id"):
        entry["destination_lane_id"] = meta["destination_lane_id"]
        if meta.get("destination_edge_id"):
            entry["destination_edge_id"] = meta["destination_edge_id"]
    elif spawn_scenario is None and entry.get("road_id"):
        strategy = profile.spawn_strategy
        if strategy == "roundabout":
            spawn_meta = pick_default_roundabout_spawn_meta_for_net(
                net_full_path,
                prefer_ego_edge_id=str(entry["road_id"]),
                scene_meta=meta,
            )
        else:
            picker = (
                pick_default_yield_spawn_meta_for_net
                if strategy == "yield"
                else pick_default_main_spawn_meta_for_net
            )
            spawn_meta = picker(
                net_full_path,
                prefer_ego_edge_id=str(entry["road_id"]),
            )
        if spawn_meta:
            entry["destination_lane_id"] = spawn_meta["destination_lane_id"]
            entry["destination_edge_id"] = spawn_meta["destination_edge_id"]
            entry["road_id"] = spawn_meta["road_id"]
            entry["spawn_lane_num"] = spawn_meta["spawn_lane_num"]

    if junction_layout_cache is not None:
        entry["junction_layout"] = junction_layout_cache
        spawn_edges = outgoing_edges_from_junction_layout(junction_layout_cache)
        if spawn_edges:
            entry["background_spawn_edges"] = spawn_edges
        if profile.spawn_strategy in ("yield", "roundabout"):
            entry["main_lane_keys"] = [
                lane_key
                for arm in junction_layout_cache.get("arms", [])
                if arm.get("road_class") == "main"
                for lane_key in arm.get("lane_keys", [])
            ]
            entry["secondary_lane_keys"] = [
                lane_key
                for arm in junction_layout_cache.get("arms", [])
                if arm.get("road_class") == "secondary"
                for lane_key in arm.get("lane_keys", [])
            ]
            if aux_cfg.enabled:
                ego_edge = entry.get("road_id") or (
                    spawn_scenario.ego_edge_id if spawn_scenario is not None else None
                )
                available_main = viable_aux_lane_keys(
                    junction_layout_cache,
                    aux_cfg.distance_from_intersection,
                    ego_edge,
                    convoy_size=aux_convoy_size,
                    convoy_gap_m=aux_cfg.convoy_gap_m,
                )
                if not available_main:
                    entry["valid"] = False
                prefer_aux = None
                if spawn_scenario is not None:
                    prefer_aux = make_lane_key(
                        spawn_scenario.aux_edge_id,
                        spawn_scenario.aux_lane_num,
                    )
                entry["aux_occupied_lane_keys"] = select_occupied_main_lanes(
                    available_main, aux_lanes_occupied, prefer_lane_key=prefer_aux
                )
                if entry["aux_occupied_lane_keys"]:
                    primary_aux = entry["aux_occupied_lane_keys"][0]
                    entry["aux_spawn_lane_index"] = primary_aux
                    entry["aux_road_id"] = lane_edge_id(primary_aux)
                    entry["aux_spawn_lane_num"] = lane_num_from_key(primary_aux)
                    if not entry.get("aux_destination_lane_id"):
                        route_index = load_vehicle_route_index(net_full_path)
                        aux_dest = resolve_aux_destination_lane_key(
                            junction_layout_cache,
                            primary_aux,
                            route_index=route_index,
                        )
                        if aux_dest:
                            entry["aux_destination_lane_id"] = aux_dest
                            entry["aux_destination_edge_id"] = lane_edge_id(aux_dest)
        else:
            entry["main_lane_keys"] = [
                lane_key
                for arm in junction_layout_cache.get("arms", [])
                for lane_key in arm.get("lane_keys", [])
            ]
            ego_edge = entry.get("road_id") or (
                spawn_scenario.ego_edge_id if spawn_scenario is not None else None
            )
            entry["right_lane_keys"] = right_lane_keys_for_aux(
                junction_layout_cache, ego_edge
            )
            if aux_cfg.enabled:
                available_right = viable_right_aux_lane_keys(
                    junction_layout_cache,
                    aux_cfg.distance_from_intersection,
                    ego_edge,
                    convoy_size=aux_convoy_size,
                    convoy_gap_m=aux_cfg.convoy_gap_m,
                )
                prefer_aux = None
                if spawn_scenario is not None:
                    prefer_aux = make_lane_key(
                        spawn_scenario.aux_edge_id,
                        spawn_scenario.aux_lane_num,
                    )
                entry["right_arm_edge_id"] = (
                    spawn_scenario.aux_edge_id if spawn_scenario is not None else None
                )
                entry["aux_occupied_lane_keys"] = select_occupied_main_lanes(
                    available_right, aux_lanes_occupied, prefer_lane_key=prefer_aux
                )

    if entry.get("destination_edge_id") and entry.get("spawn_lane_num") is not None:
        if path_budget_m > 0.0:
            entry = apply_route_budget(
                entry,
                net_path=net_full_path,
                max_path_length_m=path_budget_m,
                spawn_distance_before_end=float(entry.get("spawn_distance_before_end") or sim_cfg.spawn_distance_before_end),
            )
        else:
            entry = ensure_reachable_ego_destination(entry, net_path=net_full_path)
    entry = tag_entry_route_length(
        entry,
        path_budget_m,
        augment=route_length_augment,
    )
    if npc_profile is not None:
        entry = embed_npc_profile(
            entry,
            npc_profile,
            apply_aux_credit=bool(aux_cfg.enabled),
            density_cap=float(getattr(sim_cfg, "profile_density_cap", 1.0) or 1.0),
        )

    return {k: v for k, v in entry.items() if v is not None}

from functools import partial
from typing import Optional

from traffic_bench.eval.engine.map.junction_priority_layout import (
    JunctionLayoutError,
    allowed_shapes_for_mode,
    build_junction_priority_layout,
)
from traffic_bench.eval.engine.map.roundabout_topology import build_roundabout_layout
from traffic_bench.eval.manifest.io import (
    append_scene_entries,
    apply_max_total,
    apply_split_filter,
    assert_rejected_scenes_applied,
    discover_scenes,
    load_scene_metadata,
    write_real_manifest,
)
from traffic_bench.eval.manifest.types import AuxiliaryConfig, ExpertConfig
from traffic_bench.scene_collection.sign_scenes.materialize.pool_index import normalize_split
def build_junction_layout_for_scene(
    net_path: Path,
    *,
    profile,
    sign_lat: Optional[float] = None,
    sign_lon: Optional[float] = None,
    scene_meta: Optional[dict] = None,
) -> Optional[dict]:
    """Build junction / roundabout layout for one scene."""
    try:
        if profile.layout_mode == "roundabout":
            from traffic_bench.eval.signs.roundabout.spawn import (
                roundabout_meta_ring_kwargs,
            )

            meta = scene_meta or {}
            prefer_ego = meta.get("catalog_sign_road_id") or meta.get("road_id")
            layout = build_roundabout_layout(
                net_path,
                sign_edge_id=prefer_ego,
                **roundabout_meta_ring_kwargs(meta),
            )
        else:
            layout = build_junction_priority_layout(
                net_path,
                mode=profile.layout_mode,
                sign_lat=sign_lat,
                sign_lon=sign_lon,
            )
    except JunctionLayoutError as exc:
        print(f"  [junction_layout] {net_path.parent.name}: {exc}")
        return None
    return layout.to_dict()



def generate(cfg, scenes=None):
    """Junction and roundabout rows. Roundabout still uses this generator."""

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

    expert_cfg = expert_cfg or ExpertConfig()
    split = normalize_split(split)
    assert_rejected_scenes_applied(scenes_dir)
    all_scenes = discover_scenes(scenes_dir)
    print(f"Scenes root: {scenes_dir.resolve()}")
    print(f"Discovered {len(all_scenes)} scene(s) on disk")
    scenes, split_by_id = apply_split_filter(
        all_scenes, scenes_dir=scenes_dir, split=split
    )
    n_variations = max(1, int(getattr(sim_cfg, "n_variations", 3) or 3))
    print(
        f"Augmentation axes: layout={expansion_cfg.layout_on}, "
        f"auxiliary={expansion_cfg.auxiliary_on}, "
        f"n_variations={n_variations}"
    )
    if not scenes:
        print(
            f"[warn] No scenes with meta.json + net found under {scenes_dir} "
            f"(after paths.split={split}). "
            "Check data/scenes/<sign> / paths.scenes_dir / moscow_pool.json."
        )
    entries = []

    aux_for_entry = aux_cfg
    if not expansion_cfg.auxiliary_on:
        aux_for_entry = AuxiliaryConfig(
            enabled=False,
            distance_from_intersection=aux_cfg.distance_from_intersection,
            convoy_size=aux_cfg.convoy_size,
            convoy_gap_m=aux_cfg.convoy_gap_m,
            lanes_occupied=aux_cfg.lanes_occupied,
            release_when_ego_within_m=aux_cfg.release_when_ego_within_m,
        )

    used_scene_ids: List[str] = []
    for scene_dir in scenes:
        meta = load_scene_metadata(scene_dir)
        scene_name = meta.get("scene_name", scene_dir.name)
        net_file = meta.get("net_file", "map.net.xml")
        net_full_path = scene_dir / net_file

        spawn_lanes = parse_sumo_net_for_spawn_lanes(net_full_path)
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

        scene_entries = expand_scene_entries(
            scene_dir=scene_dir,
            scenes_root=scenes_dir,
            meta=meta,
            net_path=net_full_path,
            spawn_lanes=spawn_lanes,
            junction_layout=junction_layout,
            spawn_strategy=profile.spawn_strategy,
            sim_cfg=sim_cfg,
            expansion=expansion_cfg,
            build_entry=partial(
                build_manifest_entry, expert_cfg=expert_cfg, profile=profile
            ),
            aux_cfg_for_entry=aux_for_entry,
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
        log_under_cap=True,
    )
    summary = {
        "pdd_code": PDD_CODE,
        "sign_type": SIGN_TYPE,
        "sign_name": SIGN_NAME,
        "total_scenes": len(used_scene_ids),
        "total_entries": len(entries),
        "total_entries_before_max_total": pre_total,
        "augmentation_layout": expansion_cfg.layout_on,
        "augmentation_auxiliary": expansion_cfg.auxiliary_on,
        "max_scenarios": scenario_cfg.max_scenarios,
        "max_total": scenario_cfg.max_total,
        "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
        "horizon": sim_cfg.horizon,
        "sign_distance_before_end": sim_cfg.sign_distance_before_end,
        "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
        "max_path_length_m": float(sim_cfg.max_path_length_m),
        "max_path_length_levels": list(
            getattr(sim_cfg, "max_path_length_levels", (130.0, 150.0, 170.0))
        ),
        "auxiliary_agent": aux_for_entry.enabled,
        "aux_distance_from_intersection": aux_cfg.distance_from_intersection,
        "aux_convoy_size_max": aux_cfg.convoy_size,
        "aux_convoy_gap_m": list(expansion_cfg.aux.convoy_gaps_m)
        if expansion_cfg.aux is not None
        else [aux_cfg.convoy_gap_m],
        "aux_lanes_occupied_max": aux_cfg.lanes_occupied,
        "n_variations": n_variations,
        "profile_density_cap": float(getattr(sim_cfg, "profile_density_cap", 1.0) or 1.0),
        "npc_world": "engine.traffic.agent_profile_bank.sample_one_profile",
    }
    if profile.id == STOP.id:
        summary["stop_wait_steps"] = int(expert_cfg.stop_wait_steps)
    write_real_manifest(
        output_dir=output_dir,
        scenes_dir=scenes_dir,
        entries=entries,
        used_scene_ids=used_scene_ids,
        split_by_id=split_by_id,
        split=split,
        pdd_code=PDD_CODE,
        summary=summary,
        announce=False,
    )
    return entries

