"""Expand blocked-road (3.2) scenes into manifest rows (layout × traffic density)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..scenarios.blocked_road_route import forbidden_edge_geometry_ok
from ..scenarios.scene_augmentation import SpawnScenario, augment_layout_for_scene
from .traffic_density_levels import (
    MAX_TRAFFIC_DENSITY_LEVELS,
    TrafficDensityLevel,
    list_traffic_density_levels,
)


@dataclass(frozen=True)
class BlockedRoadSimParams:
    sign_distance_from_start: float
    destination_past_sign_m: float
    spawn_distance_before_end: float
    spawn_velocity_ms: float
    traffic_density: float
    horizon: int
    compliant_stop_success_seconds: float
    compliant_stop_max_dist_m: float
    compliant_stop_speed_mps: float
    sign_distance_before_end: float = 0.0


@dataclass(frozen=True)
class BlockedRoadExpansionConfig:
    layout: bool = True
    max_scenarios: Optional[int] = None
    traffic_density_augment: bool = True
    validate_metadrive_routes: bool = False


BuildBlockedRoadEntryFn = Callable[..., Dict[str, Any]]


def blocked_road_geometry_key(entry: Dict[str, Any]) -> Tuple:
    return (
        entry.get("road_id"),
        entry.get("spawn_lane_num"),
        entry.get("destination_lane_id"),
        entry.get("traffic_density_level_id"),
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
    density_levels: Optional[List[Optional[TrafficDensityLevel]]] = None,
) -> List[Dict[str, Any]]:
    """Expand one scene into blocked-road manifest rows (no auxiliary axis)."""
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    print(
        f"  Junction layout: {junction_layout['shape']} @ {junction_layout['junction_id']} "
        f"(through-path arms={len(junction_layout.get('main_edge_ids', []))})"
    )

    sign_lat = meta.get("latitude") or meta.get("center_lat")
    sign_lon = meta.get("longitude") or meta.get("center_lon")

    scenarios: List[SpawnScenario] = []
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
            print(f"  [augment] No through-path scenarios for {scene_name}; skipping scene")
            return []
        scenarios = list(layout_scenarios)
        print(f"  Through-path scenarios: {len(scenarios)}")
    else:
        scenarios = [None]  # type: ignore[list-item]
        print("  Layout axis off: one default spawn per scene")

    if density_levels is None:
        if expansion.traffic_density_augment:
            density_levels = list(list_traffic_density_levels(MAX_TRAFFIC_DENSITY_LEVELS))
        else:
            density_levels = [None]

    scene_entries: List[Dict[str, Any]] = []
    skipped_geometry = 0
    seen: set = set()

    for variant, scenario in enumerate(scenarios):
        if scenario is not None:
            geom_ok, geom_reason = forbidden_edge_geometry_ok(
                net_path,
                scenario.ego_destination_edge_id,
                sign_distance_from_start=sim.sign_distance_from_start,
                destination_past_sign_m=sim.destination_past_sign_m,
            )
            if not geom_ok:
                skipped_geometry += 1
                print(
                    f"  [skip] forbidden-lane geometry {geom_reason} "
                    f"({scenario.scenario_id})"
                )
                continue

        for density_level in density_levels:
            entry = build_entry(
                scene_dir=scene_dir,
                scenes_root=scenes_root,
                meta=meta,
                variant=variant,
                sim=sim,
                spawn_scenario=scenario,
                spawn_lanes_cache=list(spawn_lanes),
                junction_layout_cache=junction_layout,
                density_level=density_level,
            )
            key = blocked_road_geometry_key(entry)
            if key in seen:
                continue
            seen.add(key)
            scene_entries.append(entry)

    if skipped_geometry:
        print(f"  [geometry] Skipped {skipped_geometry} scenario(s) (forbidden edge too short)")

    cap = expansion.max_scenarios
    if cap is not None and len(scene_entries) > cap:
        rng = random.Random(
            hash((scene_name, "blocked_road_max_scenarios", int(cap))) & 0xFFFFFFFF
        )
        rng.shuffle(scene_entries)
        print(
            f"  Retained {cap} of {len(scene_entries)} manifest entries "
            f"for {scene_name} (shuffled)"
        )
        scene_entries = scene_entries[:cap]
    else:
        print(f"  Manifest entries for {scene_name}: {len(scene_entries)}")

    return scene_entries


def build_blocked_road_manifest_entry(
    *,
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    variant: int,
    sim: BlockedRoadSimParams,
    spawn_scenario: Optional[SpawnScenario],
    spawn_lanes_cache: Optional[List[Any]],
    junction_layout_cache: Optional[dict],
    density_level: Optional[TrafficDensityLevel],
    stable_seed_fn: Callable[..., int],
    pdd_code: str,
    sign_type: str,
    sign_class: str = "NoTrafficSign",
    sign_title: str = "Movement prohibited",
) -> Dict[str, Any]:
    """Build one manifest row for a through-path blocked-road scenario."""
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    net_file = meta.get("net_file", "map.net.xml")
    net_rel = scene_dir.relative_to(scenes_root) / net_file

    scenario_id = spawn_scenario.scenario_id if spawn_scenario else ""
    traffic_density = (
        float(density_level.traffic_density)
        if density_level is not None
        else float(sim.traffic_density)
    )
    if density_level is not None:
        seed_key = f"{scenario_id}_td{density_level.id}"
        scene_id = f"{scene_name}_td{density_level.id}"
    else:
        seed_key = scenario_id
        scene_id = scene_name
    seed = stable_seed_fn(scene_name, variant, scenario_id=seed_key)

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
        "seed": seed,
        "var_idx": variant,
        "pdd_code": pdd_code,
        "sign_code": pdd_code,
        "sign_type": sign_type,
        "sign_family": "blocked_road",
        "sign_title": sign_title,
        "sign_class": sign_class,
        "spawn_velocity_ms": sim.spawn_velocity_ms,
        "traffic_density": traffic_density,
        "traffic_density_level_id": (
            density_level.id if density_level is not None else None
        ),
        "traffic_density_level_name": (
            density_level.name if density_level is not None else None
        ),
        "nuplan_vehicles_per_frame": (
            density_level.nuplan_vehicles_per_frame
            if density_level is not None
            else None
        ),
        "horizon": sim.horizon,
        "sign_road_id": sign_road_id,
        "sign_distance_from_start": sim.sign_distance_from_start,
        "destination_past_sign_m": sim.destination_past_sign_m,
        "spawn_distance_before_end": sim.spawn_distance_before_end,
        "sign_distance_before_end": sim.sign_distance_before_end,
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

    if spawn_scenario is not None:
        entry.update(spawn_scenario.to_manifest_fields())

    if selected_lane is not None:
        entry["spawn_lane_length"] = selected_lane.length
        entry["spawn_to_junction"] = selected_lane.to_junction

    if junction_layout_cache is not None:
        entry["junction_layout"] = junction_layout_cache

    return {k: v for k, v in entry.items() if v is not None}
