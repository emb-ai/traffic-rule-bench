"""Expand one-way (5.7.1 / 5.7.2) scenes: dual-path × NPC profile.

Dual-path geometry comes from ``one_way_signs`` discovery (temptation +
compliant detour). NPC world params use priority_bench shared helpers
(``sample_one_profile`` / ``stable_hash``), same as blocked_road — not
density-tier multiplication from one_way_signs.

``max_scenarios`` caps the combined (dual-path × n_variations) pool after shuffle.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..scenarios.one_way_bridge import (
    DualPathScenario,
    discover_one_way_dual_paths,
    dual_path_to_spawn_scenario,
    get_one_way_sign_spec,
)
from ..scenarios.scene_augmentation import SpawnScenario

_PER_SIGN_BENCH = Path(__file__).resolve().parents[3]
if str(_PER_SIGN_BENCH) not in sys.path:
    sys.path.insert(0, str(_PER_SIGN_BENCH))

from factorized_space.agent_profile_bank import sample_one_profile  # noqa: E402
from sumo_space.sumo_catalog import stable_hash  # noqa: E402


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
    """Expand one scene: dual-path picks × n_variations NPC profiles."""
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
    )
    # If preferred junction yields nothing, search the whole crop (multi-junc).
    if not dual_paths and junction_ids is not None:
        print(
            f"  [dual-path] No picks at junction {preferred_jid}; "
            f"retrying full crop"
        )
        dual_paths = discover_one_way_dual_paths(
            net_path,
            pdd_code=pdd_code,
            min_gain_m=float(sim.min_dual_path_gain_m),
            min_lane_length_m=float(sim.min_ego_lane_m),
            max_scenarios=int(expansion.max_dual_paths),
            junction_ids=None,
            arm_counts=expansion.arm_counts,
        )

    if not dual_paths:
        print(f"  [dual-path] No scenarios for {scene_name}; skipping scene")
        return []

    print(f"  Dual-path scenarios: {len(dual_paths)}")
    for dp in dual_paths:
        print(
            f"    junc={dp.junction_id} ego={dp.ego_edge_id} "
            f"base={dp.turn_dir}->{dp.turn_first_exit} "
            f"comp={dp.compliant_dir}->{dp.straight_first_exit} "
            f"gain={dp.gain_m:.1f}m"
        )

    candidates: List[Tuple[int, DualPathScenario, int]] = [
        (dual_i, dp, var_idx)
        for dual_i, dp in enumerate(dual_paths)
        for var_idx in range(n_variations)
    ]
    pre_cap = len(candidates)
    cap = expansion.max_scenarios
    if cap is not None and pre_cap > cap:
        rng = random.Random(
            hash((scene_name, "one_way_combo_cap", int(cap))) & 0xFFFFFFFF
        )
        rng.shuffle(candidates)
        candidates = candidates[:cap]
        print(
            f"  Retained {len(candidates)} of {pre_cap} "
            f"(dual×NPC) scenario(s) (cap={cap}; "
            f"{len(dual_paths)} dual × {n_variations} profiles)"
        )
    else:
        print(
            f"  Combined scenarios: {pre_cap} "
            f"({len(dual_paths)} dual × {n_variations} NPC profiles)"
        )

    scene_entries: List[Dict[str, Any]] = []
    seen: set = set()

    for dual_i, dual, var_idx in candidates:
        spawn_scenario = dual_path_to_spawn_scenario(dual)
        seed = stable_hash(scene_name, spawn_scenario.scenario_id, var_idx)
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
    """Build one manifest row for a dual-path + NPC-profile variation."""
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
        entry["dual_path"] = meta_fields["dual_path"]
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
        entry["forbidden_dir"] = meta_fields["dual_path"]["forbidden_dir"]

    if junction_layout_cache is not None:
        # Prefer dual-path junction when it differs from lat/lon layout pick.
        layout = dict(junction_layout_cache)
        if dual_path is not None and dual_path.junction_id:
            if str(layout.get("junction_id")) != str(dual_path.junction_id):
                layout["junction_id"] = dual_path.junction_id
        entry["junction_layout"] = layout

    return {k: v for k, v in entry.items() if v is not None}
