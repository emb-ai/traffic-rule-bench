"""Expand blocked-road (3.2) scenes into manifest rows (layout × NPC profile).

NPC world params come from ``core.profiles``
(``sample_one_profile``, ``stable_hash``). Geometry expansion stays here.

``max_scenarios`` caps the combined (lane/dest × n_variations) pool after shuffle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..scenarios.blocked_road_route import forbidden_edge_geometry_ok
from ..scenarios.scene_augmentation import SpawnScenario, augment_layout_for_scene
from .manifest_expansion import shuffle_cap

from traffic_bench.eval.core.profiles.agent_profile_bank import sample_one_profile
from traffic_bench.eval.core.profiles.stable_hash import stable_hash


@dataclass(frozen=True)
class BlockedRoadSimParams:
    sign_distance_from_start: float
    spawn_distance_before_end: float
    spawn_velocity_ms: float
    horizon: int
    compliant_stop_success_seconds: float
    compliant_stop_max_dist_m: float
    compliant_stop_speed_mps: float
    n_variations: int = 3
    profile_density_cap: float = 1.0
    destination_max_along_m: Optional[float] = 50.0


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
    """Expand one scene: layout through-paths × n_variations NPC profiles."""
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    n_variations = max(1, int(sim.n_variations))
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

    # Filter layouts by forbidden-lane geometry, then expand with NPC profiles.
    # max_scenarios caps the *combined* (lane/dest × var_idx) pool — not layouts
    # first and then × n_variations.
    layout_kept: List[Tuple[int, Optional[SpawnScenario]]] = []
    skipped_geometry = 0
    for layout_i, scenario in enumerate(scenarios):
        if scenario is not None:
            geom_ok, geom_reason = forbidden_edge_geometry_ok(
                net_path,
                scenario.ego_destination_edge_id,
                sign_distance_from_start=sim.sign_distance_from_start,
                destination_max_along_m=float(sim.destination_max_along_m or 50.0),
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

    candidates: List[Tuple[int, Optional[SpawnScenario], int]] = [
        (layout_i, scenario, var_idx)
        for layout_i, scenario in layout_kept
        for var_idx in range(n_variations)
    ]
    pre_cap = len(candidates)
    cap = expansion.max_scenarios
    candidates = shuffle_cap(
        candidates,
        cap,
        seed_key=(scene_name, "blocked_road_combo_cap", int(cap) if cap is not None else 0),
    )
    if cap is not None and pre_cap > cap:
        print(
            f"  Retained {len(candidates)} of {pre_cap} "
            f"(layout×NPC) scenario(s) (shuffled, cap={cap}; "
            f"{len(layout_kept)} layouts × {n_variations} profiles)"
        )
    else:
        print(
            f"  Combined scenarios: {pre_cap} "
            f"({len(layout_kept)} layouts × {n_variations} NPC profiles)"
        )

    scene_entries: List[Dict[str, Any]] = []
    seen: set = set()

    for layout_i, scenario, var_idx in candidates:
        scenario_id = scenario.scenario_id if scenario is not None else ""
        seed = stable_hash(scene_name, scenario_id, var_idx)
        profile = sample_one_profile(
            int(seed),
            density_cap=float(sim.profile_density_cap),
            horizon_steps=int(sim.horizon),
        )

        entry = build_entry(
            scene_dir=scene_dir,
            scenes_root=scenes_root,
            meta=meta,
            layout_variant=layout_i,
            var_idx=var_idx,
            seed=seed,
            sim=sim,
            spawn_scenario=scenario,
            spawn_lanes_cache=list(spawn_lanes),
            junction_layout_cache=junction_layout,
            npc_profile=profile,
        )
        key = blocked_road_geometry_key(entry)
        if key in seen:
            continue
        seen.add(key)
        scene_entries.append(entry)

    print(f"  Manifest entries for {scene_name}: {len(scene_entries)}")
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
    npc_profile: Dict[str, Any],
    pdd_code: str,
    sign_type: str,
    sign_class: str = "NoTrafficSign",
    sign_title: str = "Movement prohibited",
) -> Dict[str, Any]:
    """Build one manifest row for a through-path + NPC-profile variation."""
    del layout_variant  # encoded in augmentation_id / spawn fields
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    net_file = meta.get("net_file", "map.net.xml")
    net_rel = scene_dir.relative_to(scenes_root) / net_file

    traffic_density = float(npc_profile["traffic_density"])
    horizon = int(npc_profile.get("horizon_steps", sim.horizon))
    # Same scene_id across NPC variations (sumo_catalog); distinguish via var_idx.
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
        "spawn_velocity_ms": sim.spawn_velocity_ms,
        "traffic_density": traffic_density,
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
    if sim.destination_max_along_m is not None:
        entry["destination_max_along_m"] = float(sim.destination_max_along_m)

    # Same profile_* embedding as sumo_runner.materialize_sumo_scene.
    for key, value in npc_profile.items():
        entry[f"profile_{key}"] = value

    if spawn_scenario is not None:
        entry.update(spawn_scenario.to_manifest_fields())

    if selected_lane is not None:
        entry["spawn_lane_length"] = selected_lane.length
        entry["spawn_to_junction"] = selected_lane.to_junction

    if junction_layout_cache is not None:
        entry["junction_layout"] = junction_layout_cache

    return {k: v for k, v in entry.items() if v is not None}
