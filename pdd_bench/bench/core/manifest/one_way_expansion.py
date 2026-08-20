"""Expand one-way (5.7.1 / 5.7.2) scenes: dual-path × ego spawn lane × NPC.

Dual-path geometry comes from crop ``meta.json`` (moscow dual_path harvest),
then is multiplied across approach lanes on the ego edge
(length ≥ ``min_ego_lane_m``) and ``n_variations`` NPC profiles.

``max_scenarios`` caps the combined (dual × lane × n_variations) pool after
shuffle.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..scenarios.one_way_bridge import (
    DualPathScenario,
    discover_one_way_dual_paths,
    dual_path_to_spawn_scenario,
    ego_spawn_lane_nums_for_dual,
    get_one_way_sign_spec,
)
from ..scenarios.scene_augmentation import SpawnScenario
from .dual_path_budget import (
    apply_dual_path_route_budget,
    load_sumo_edge_lengths,
)
from .manifest_expansion import shuffle_cap

from pdd_bench.bench.core.profiles.agent_profile_bank import sample_one_profile
from pdd_bench.bench.core.profiles.stable_hash import stable_hash


@dataclass(frozen=True)
class OneWaySimParams:
    spawn_distance_before_end: float
    sign_distance_before_end: float
    spawn_velocity_ms: float
    horizon: int
    n_variations: int = 3
    profile_density_cap: float = 1.0
    min_dual_path_gain_m: float = 20.0
    min_ego_lane_m: float = 8.0
    # Shared travel budget (m) for baseline + compliant truncations; None = no cut.
    dual_path_route_budget_m: Optional[float] = None


@dataclass(frozen=True)
class OneWayExpansionConfig:
    layout: bool = True
    max_scenarios: Optional[int] = None
    max_dual_paths: int = 20
    arm_counts: Tuple[int, ...] = (3, 4)


BuildOneWayEntryFn = Callable[..., Dict[str, Any]]


def one_way_geometry_key(entry: Dict[str, Any]) -> Tuple:
    return (
        entry.get("road_id"),
        entry.get("spawn_lane_num"),
        entry.get("destination_lane_id"),
        entry.get("baseline_dir") or entry.get("baseline_turn_dir"),
        entry.get("var_idx"),
    )


def expand_one_way_scene_entries(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    net_path: Path,
    spawn_lanes: Sequence[Any],
    junction_layout: dict,
    sim: OneWaySimParams,
    expansion: OneWayExpansionConfig,
    build_entry: BuildOneWayEntryFn,
    pdd_code: str,
) -> List[Dict[str, Any]]:
    """Expand one scene: dual-path × ego spawn lanes × n_variations NPC profiles."""
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    n_variations = max(1, int(sim.n_variations))
    print(
        f"  Junction layout: {junction_layout.get('shape')} @ "
        f"{junction_layout.get('junction_id')} "
        f"(arms={len(junction_layout.get('arms', []))})"
    )

    if not expansion.layout:
        print("  Layout axis off: one-way requires dual-path layout; skipping")
        return []

    preferred_jid = str(
        junction_layout.get("junction_id") or meta.get("junction_id") or ""
    ).strip()
    junction_ids = [preferred_jid] if preferred_jid else None

    dual_paths = discover_one_way_dual_paths(
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
        dual_paths = discover_one_way_dual_paths(
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

    candidates: List[Tuple[int, DualPathScenario, int, int]] = [
        (dual_i, dp, lane_num, var_idx)
        for dual_i, dp in enumerate(dual_paths)
        for lane_num in ego_spawn_lane_nums_for_dual(
            dp,
            spawn_lanes,
            min_lane_length_m=float(sim.min_ego_lane_m),
        )
        for var_idx in range(n_variations)
    ]
    pre_cap = len(candidates)
    n_lane_combos = sum(
        len(
            ego_spawn_lane_nums_for_dual(
                dp,
                spawn_lanes,
                min_lane_length_m=float(sim.min_ego_lane_m),
            )
        )
        for dp in dual_paths
    )
    cap = expansion.max_scenarios
    candidates = shuffle_cap(
        candidates,
        cap,
        seed_key=(scene_name, "one_way_combo_cap", int(cap) if cap is not None else 0),
    )
    if cap is not None and pre_cap > cap:
        print(
            f"  Retained {len(candidates)} of {pre_cap} "
            f"(dual×lane×NPC) scenario(s) (shuffled, cap={cap}; "
            f"{len(dual_paths)} dual, {n_lane_combos} lane-slots, "
            f"{n_variations} profiles)"
        )
    else:
        print(
            f"  Combined scenarios: {pre_cap} "
            f"({len(dual_paths)} dual, {n_lane_combos} lane-slots, "
            f"{n_variations} NPC profiles)"
        )

    scene_entries: List[Dict[str, Any]] = []
    seen: set = set()

    for dual_i, dual, lane_num, var_idx in candidates:
        spawn_scenario = dual_path_to_spawn_scenario(dual, ego_lane_num=lane_num)
        seed = stable_hash(
            scene_name, spawn_scenario.scenario_id, lane_num, var_idx
        )
        profile = sample_one_profile(
            int(seed),
            density_cap=float(sim.profile_density_cap),
            horizon_steps=int(sim.horizon),
        )
        entry = build_entry(
            scene_dir=scene_dir,
            scenes_root=scenes_root,
            meta=meta,
            layout_variant=dual_i,
            var_idx=var_idx,
            seed=seed,
            sim=sim,
            spawn_scenario=spawn_scenario,
            dual_path=dual,
            spawn_lanes_cache=list(spawn_lanes),
            junction_layout_cache=junction_layout,
            npc_profile=profile,
        )
        key = one_way_geometry_key(entry)
        if key in seen:
            continue
        seen.add(key)
        scene_entries.append(entry)

    print(f"  Manifest entries for {scene_name}: {len(scene_entries)}")
    return scene_entries


def build_one_way_manifest_entry(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    layout_variant: int,
    var_idx: int,
    seed: int,
    sim: OneWaySimParams,
    spawn_scenario: Optional[SpawnScenario],
    dual_path: Optional[DualPathScenario],
    spawn_lanes_cache: Optional[List[Any]],
    junction_layout_cache: Optional[dict],
    npc_profile: Dict[str, Any],
    pdd_code: str,
    sign_type: str = "one_way",
) -> Dict[str, Any]:
    """Build one manifest row for a dual-path + spawn-lane + NPC variation."""
    del layout_variant
    spec = get_one_way_sign_spec(pdd_code)
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    net_file = meta.get("net_file", "map.net.xml")
    net_rel = scene_dir.relative_to(scenes_root) / net_file

    traffic_density = float(npc_profile["traffic_density"])
    horizon = int(npc_profile.get("horizon_steps", sim.horizon))
    scene_id = scene_name

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
        "pdd_code": spec.pdd_code,
        "sign_code": spec.pdd_code,
        "sign_type": sign_type,
        "sign_family": "one_way",
        "sign_title": spec.title,
        "sign_class": spec.class_name,
        "allowed_dirs": sorted(spec.allowed_dirs),
        "forbidden_dir": spec.forbidden_dir,
        "spawn_velocity_ms": sim.spawn_velocity_ms,
        "traffic_density": traffic_density,
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

    for key, value in npc_profile.items():
        entry[f"profile_{key}"] = value

    if spawn_scenario is not None:
        entry.update(spawn_scenario.to_manifest_fields())

    if selected_lane is not None:
        entry["spawn_lane_length"] = selected_lane.length
        entry["spawn_to_junction"] = selected_lane.to_junction

    if dual_path is not None:
        meta_fields = dual_path.to_meta_fields()
        dual_meta = dict(meta_fields["dual_path"])
        # Keep spawn lane consistent with the lane-axis row (discovery defaults
        # to lane 0 only).
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
        entry["background_excluded_edges"] = list(dual_path.wrong_dir_edges)
        entry["junction_id"] = dual_path.junction_id
        entry["forbidden_dir"] = spec.forbidden_dir
        dual_meta["forbidden_dir"] = spec.forbidden_dir

        budget = sim.dual_path_route_budget_m
        if budget is not None and float(budget) > 0.0:
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
            # Prefer truncated dest over full-path spawn_scenario fields.
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
        # Prefer dual-path junction when it differs from lat/lon layout pick.
        layout = dict(junction_layout_cache)
        if dual_path is not None and dual_path.junction_id:
            if str(layout.get("junction_id")) != str(dual_path.junction_id):
                layout["junction_id"] = dual_path.junction_id
        entry["junction_layout"] = layout

    return {k: v for k, v in entry.items() if v is not None}
