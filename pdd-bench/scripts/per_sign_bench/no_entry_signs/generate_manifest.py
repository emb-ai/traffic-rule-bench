#!/usr/bin/env python3
"""Generate evaluation manifest from junction crops (no-entry signs 3.1 / 3.2).

Each top-level crop under ``scenes/<slug>/`` places the active sign at the
*start* of the forbidden (destination) lane (``sign_distance_from_start``).
Ego spawns on the approach arm (``spawn_distance_before_end`` before the
junction) and is routed onto that destination so baselines enter past the
sign while experts stop before entering. ``run_benchmark.py`` places
``NoEntrySign`` (3.1) or ``NoTrafficSign`` (3.2).
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from lib.junction_priority_layout import JunctionLayoutError, build_junction_priority_layout
from lib.lane_keys import make_lane_key
from lib.manifest_config import DEFAULT_SPAWN_DISTANCE_BEFORE_END
from lib.no_entry_route import forbidden_edge_long_enough
from lib.no_entry_sign_spec import (
    DEFAULT_PDD_CODE,
    SIGN_FAMILY,
    NoEntrySignSpec,
    get_no_entry_sign_spec,
    local_scenes_root,
)
from lib.scene_augmentation import (
    SpawnScenario,
    augment_layout_for_scene,
    pick_default_main_spawn_meta_for_net,
)
from lib.scene_selection import is_reserved_scene_dir, is_scene_rejected
from lib.sumo_utils import CORE_SCENES_SUBDIR, is_vehicle_drivable_lane


SCRIPT_DIR = Path(__file__).parent.resolve()
RUN_BENCH_SCRIPT = SCRIPT_DIR / "run_benchmark.py"
PDD_BENCH_DIR = SCRIPT_DIR.parents[2]

# Hydra: paths.output_base: benchmark_output/${pdd_slug:${sign.pdd_code}}
# so ``sign.pdd_code=3.1`` lands in ``benchmark_output/3_1/…`` without a
# manual paths.output_base override.
OmegaConf.register_new_resolver(
    "pdd_slug",
    lambda code: str(code).replace(".", "_"),
    replace=True,
)
DEFAULT_CARL_CKPT = (
    PDD_BENCH_DIR / "checkpoints" / "carl" / "nuplan_51479_1B" / "model_best.pth"
)
DEFAULT_NN_CHECKPOINTS = {
    "carl": DEFAULT_CARL_CKPT,
    "carl_rule": DEFAULT_CARL_CKPT,
    "plant2": PDD_BENCH_DIR / "checkpoints" / "plant2_finetuned" / "plant2_supervised_2nd_final.pt",
    "plant2_rule": PDD_BENCH_DIR
    / "checkpoints"
    / "plant2_finetuned"
    / "plant2_supervised_2nd_final.pt",
}

# Soft env hint for MetaDrive map config (placement uses before_end offset).
DEFAULT_SIGN_SPAWN_DISTANCE = 30.0

# Active sign for this process (set in ``main`` from Hydra).
_ACTIVE_SIGN: NoEntrySignSpec = get_no_entry_sign_spec(DEFAULT_PDD_CODE)
PDD_CODE = _ACTIVE_SIGN.pdd_code
SIGN_TYPE = SIGN_FAMILY


def _set_active_sign(pdd_code: str | None) -> NoEntrySignSpec:
    global _ACTIVE_SIGN, PDD_CODE, SIGN_TYPE
    _ACTIVE_SIGN = get_no_entry_sign_spec(pdd_code)
    PDD_CODE = _ACTIVE_SIGN.pdd_code
    SIGN_TYPE = SIGN_FAMILY
    return _ACTIVE_SIGN


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------
@dataclass
class PathsConfig:
    scenes_dir: Optional[str] = None
    output_base: Optional[str] = None
    experiment_name: Optional[str] = None


@dataclass
class ScenarioConfig:
    n_variants: int = 1
    augment: bool = True
    max_scenarios_per_scene: Optional[int] = None
    respect_scene_selection: bool = True
    min_ego_lane_m: float = 8.0
    validate_metadrive_routes: bool = True


@dataclass
class SimulationConfig:
    spawn_velocity_ms: float = 2.5
    traffic_density: float = 0.0
    horizon: int = 600
    sign_distance_from_start: float = 10.0
    destination_past_sign_m: float = 8.0
    spawn_distance_before_end: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END
    # Legacy / unused for placement (kept for Hydra compatibility).
    sign_distance_before_end: float = 0.0
    compliant_stop_success_seconds: float = 3.0
    compliant_stop_max_dist_m: float = 12.0
    compliant_stop_speed_mps: float = 0.5


@dataclass
class GifConfig:
    enabled: bool = False
    policy: str = "idm"
    max_scenes: Optional[int] = None
    dry_run: bool = False
    hide_signs: bool = True
    dir: Optional[str] = None
    run_name: Optional[str] = None
    model_path: Optional[str] = None


@dataclass
class ManifestConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    gif: GifConfig = field(default_factory=GifConfig)


# -----------------------------------------------------------------------------
# SUMO network parsing
# -----------------------------------------------------------------------------
@dataclass
class SumoLaneInfo:
    """Information about a SUMO lane suitable for spawning."""

    edge_id: str
    lane_num: int
    lane_id: str
    length: float
    to_junction: str
    junction_type: str


def parse_sumo_net_for_spawn_lanes(net_path: Path, min_length: float = 20.0) -> List[SumoLaneInfo]:
    """Parse SUMO .net.xml and find lanes that lead to intersections."""
    if not net_path.exists():
        return []

    tree = ET.parse(net_path)
    root = tree.getroot()

    junctions = {}
    for junction in root.findall("junction"):
        jid = junction.get("id")
        jtype = junction.get("type", "unknown")
        junctions[jid] = jtype

    intersection_types = {"priority", "right_before_left", "allway_stop", "traffic_light"}
    spawn_lanes: List[SumoLaneInfo] = []

    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        func = edge.get("function", "normal")

        if func == "internal" or (edge_id or "").startswith(":"):
            continue

        to_junction = edge.get("to", "")
        junction_type = junctions.get(to_junction, "unknown")

        if junction_type not in intersection_types:
            continue

        for lane in edge.findall("lane"):
            if not is_vehicle_drivable_lane(lane):
                continue
            lane_id = lane.get("id", "")
            length = float(lane.get("length", 0) or 0)

            if length == 0:
                shape_str = lane.get("shape", "")
                if shape_str:
                    points = shape_str.strip().split()
                    coords = [tuple(map(float, p.split(","))) for p in points if "," in p]
                    if len(coords) >= 2:
                        length = sum(
                            (
                                (coords[i + 1][0] - coords[i][0]) ** 2
                                + (coords[i + 1][1] - coords[i][1]) ** 2
                            )
                            ** 0.5
                            for i in range(len(coords) - 1)
                        )

            if length < min_length:
                continue

            try:
                lane_num = int(lane_id.rsplit("_", 1)[1])
            except (ValueError, IndexError):
                lane_num = 0

            spawn_lanes.append(
                SumoLaneInfo(
                    edge_id=edge_id,
                    lane_num=lane_num,
                    lane_id=f"lane_{lane_id}",
                    length=length,
                    to_junction=to_junction,
                    junction_type=junction_type,
                )
            )

    return spawn_lanes


def build_junction_layout_for_scene(
    net_path: Path,
    *,
    sign_lat: Optional[float] = None,
    sign_lon: Optional[float] = None,
    preferred_junction_id: Optional[str] = None,
) -> Optional[dict]:
    """Build equal-priority junction layout from a cropped scene net.xml."""
    try:
        layout = build_junction_priority_layout(
            net_path,
            mode="main_main",
            sign_lat=sign_lat,
            sign_lon=sign_lon,
            preferred_junction_id=preferred_junction_id,
        )
    except JunctionLayoutError as exc:
        print(f"  [junction_layout] {net_path.parent.name}: {exc}")
        return None
    return layout.to_dict()


# -----------------------------------------------------------------------------
# Seed / scene helpers
# -----------------------------------------------------------------------------
def _stable_seed(scene_name: str, variant: int = 0, scenario_id: str = "") -> int:
    """Generate deterministic 32-bit seed from scene name, variant, and scenario."""
    h = hashlib.sha256()
    h.update(scene_name.encode("utf-8"))
    h.update(b"|")
    h.update(str(variant).encode("utf-8"))
    if scenario_id:
        h.update(b"|")
        h.update(scenario_id.encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")


def discover_scenes(
    scenes_dir: Path,
    *,
    respect_scene_selection: bool = True,
) -> List[Path]:
    """Find junction crop directories (skip core/ reserved dirs and rejected scenes)."""
    scenes: List[Path] = []
    if not scenes_dir.is_dir():
        return scenes
    for entry in sorted(scenes_dir.iterdir()):
        if not entry.is_dir():
            continue
        if is_reserved_scene_dir(entry.name) or entry.name == CORE_SCENES_SUBDIR:
            continue
        if respect_scene_selection and is_scene_rejected(scenes_dir, entry.name):
            continue
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        net_file = meta.get("net_file", "map.net.xml")
        if (entry / net_file).exists():
            scenes.append(entry)
    return scenes


def load_scene_metadata(scene_dir: Path) -> Dict[str, Any]:
    """Load scene metadata from meta.json (and optional center.json)."""
    meta_path = scene_dir / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    center_path = scene_dir / "center.json"
    if center_path.exists():
        with open(center_path, "r", encoding="utf-8") as f:
            center = json.load(f)
            meta["center_lat"] = center.get("lat")
            meta["center_lon"] = center.get("lon")
    return meta


def _scene_id_from_meta(meta: Dict[str, Any], scene_dir: Path) -> str:
    """Prefer ``sign_<id>`` when catalog sign_id is present; else folder name."""
    sign_id = meta.get("sign_id")
    if sign_id is not None and str(sign_id).strip() != "":
        try:
            return f"sign_{int(sign_id)}"
        except (TypeError, ValueError):
            return f"sign_{sign_id}"
    return str(meta.get("scene_name") or scene_dir.name)


def _spawn_scenario_from_meta(meta: Dict[str, Any]) -> Optional[SpawnScenario]:
    """Build a through-path SpawnScenario from crop meta when spawn/dest exist."""
    road_id = str(meta.get("road_id") or "").strip()
    dest_lane = meta.get("destination_lane_id")
    if not road_id or not dest_lane:
        return None
    dest_edge = meta.get("destination_edge_id")
    if not dest_edge:
        # MetaDrive lane keys are ``lane_<edge>_<num>``; strip prefix carefully.
        raw = str(dest_lane)
        if raw.startswith("lane_"):
            raw = raw[len("lane_"):]
        dest_edge = raw.rsplit("_", 1)[0]
    # Reject SUMO internal / walking-area destinations.
    if not dest_edge or str(dest_edge).startswith(":"):
        return None
    if str(dest_lane).startswith("lane_:"):
        return None
    lane_num = int(meta.get("spawn_lane_num", 0) or 0)
    return SpawnScenario(
        ego_edge_id=road_id,
        ego_lane_num=lane_num,
        ego_destination_edge_id=str(dest_edge),
        ego_destination_lane_key=str(dest_lane),
        scenario_id=f"meta_{road_id}_to_{dest_edge}",
    )


def _optional_metadrive_route_ok(
    net_path: Path,
    *,
    road_id: str,
    spawn_lane_num: int,
    dest_lane_id: str,
    pdd_code: str,
) -> Tuple[bool, str]:
    """Best-effort MetaDrive spawn→dest check; keep scene on import/runtime failure."""
    try:
        from lib.metadrive_route_check import (
            is_metadrive_path_ok,
            probe_road_network_for_net,
        )
    except Exception as exc:
        return True, f"metadrive check unavailable ({exc}); keeping scene"

    env = None
    try:
        env, road_network = probe_road_network_for_net(
            net_path,
            spawn_edge_id=road_id,
            spawn_lane_num=spawn_lane_num,
            destination_lane_id=dest_lane_id,
            pdd_code=pdd_code,
        )
        start_lane = make_lane_key(road_id, spawn_lane_num)
        if start_lane not in road_network.graph or dest_lane_id not in road_network.graph:
            return False, f"lanes missing in MetaDrive graph ({start_lane} -> {dest_lane_id})"
        path = road_network.shortest_path(start_lane, dest_lane_id)
        if not is_metadrive_path_ok(path, spawn=start_lane, dest=dest_lane_id):
            return False, f"unroutable {start_lane} -> {dest_lane_id}"
        return True, "ok"
    except Exception as exc:
        return True, f"metadrive check failed ({exc}); keeping scene"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _filter_scenarios_forbidden_length(
    net_path: Path,
    scenarios: List[SpawnScenario],
    *,
    sign_distance_from_start: float,
    destination_past_sign_m: float,
) -> List[SpawnScenario]:
    """Drop scenarios whose destination (sign) edge is too short for sign+dest."""
    kept: List[SpawnScenario] = []
    for sc in scenarios:
        edge_id = str(sc.ego_destination_edge_id or "").strip()
        ok, reason = forbidden_edge_long_enough(
            net_path,
            edge_id,
            sign_distance_from_start=sign_distance_from_start,
            destination_past_sign_m=destination_past_sign_m,
        )
        if not ok:
            print(f"  [skip] {reason} ({sc.scenario_id})")
            continue
        kept.append(sc)
    return kept


def resolve_through_path_scenarios(
    net_path: Path,
    meta: Dict[str, Any],
    spawn_lanes: List[SumoLaneInfo],
    *,
    augment: bool,
    max_scenarios: Optional[int],
    min_ego_lane_m: float,
    sign_distance_from_start: float = 10.0,
    destination_past_sign_m: float = 8.0,
) -> List[SpawnScenario]:
    """Resolve one or more through-path spawn/dest scenarios for a crop.

    Prefer crop-time meta spawn/dest (already validated at crop) when present,
    then optional layout augmentation, then default main-arm pick.

    Dest (forbidden) edges must be strictly longer than
    ``sign_distance_from_start + destination_past_sign_m``.
    """
    meta_scenario = _spawn_scenario_from_meta(meta)

    scenarios: List[SpawnScenario] = []
    if augment and spawn_lanes:
        try:
            _, scenarios = augment_layout_for_scene(
                net_path,
                spawn_lanes,
                min_lane_length=min_ego_lane_m,
                sign_lat=float(meta["latitude"]) if meta.get("latitude") is not None else None,
                sign_lon=float(meta["longitude"]) if meta.get("longitude") is not None else None,
            )
        except Exception as exc:
            print(f"  [augment] failed ({exc}); falling back to meta/default")
            scenarios = []

        prefer_edge = str(meta.get("road_id") or "").strip()
        if prefer_edge and scenarios:
            preferred = [s for s in scenarios if s.ego_edge_id == prefer_edge]
            if preferred:
                scenarios = preferred

    # Crop meta first (stable, real-edge dest), then augmented variants.
    ordered: List[SpawnScenario] = []
    seen: set[str] = set()
    for sc in ([meta_scenario] if meta_scenario is not None else []) + list(scenarios):
        if sc is None:
            continue
        key = f"{sc.ego_edge_id}|{sc.ego_lane_num}|{sc.ego_destination_lane_key}"
        if key in seen:
            continue
        if str(sc.ego_destination_edge_id).startswith(":"):
            continue
        if str(sc.ego_destination_lane_key).startswith("lane_:"):
            continue
        seen.add(key)
        ordered.append(sc)

    if not ordered:
        spawn_meta = pick_default_main_spawn_meta_for_net(
            net_path,
            prefer_ego_edge_id=str(meta.get("road_id") or "") or None,
            min_lane_length=min_ego_lane_m,
            sign_lat=float(meta["latitude"]) if meta.get("latitude") is not None else None,
            sign_lon=float(meta["longitude"]) if meta.get("longitude") is not None else None,
        )
        if spawn_meta:
            ordered = [
                SpawnScenario(
                    ego_edge_id=str(spawn_meta["road_id"]),
                    ego_lane_num=int(spawn_meta.get("spawn_lane_num", 0) or 0),
                    ego_destination_edge_id=str(spawn_meta["destination_edge_id"]),
                    ego_destination_lane_key=str(spawn_meta["destination_lane_id"]),
                    scenario_id=(
                        f"default_{spawn_meta['road_id']}_to_"
                        f"{spawn_meta['destination_edge_id']}"
                    ),
                )
            ]

    ordered = _filter_scenarios_forbidden_length(
        net_path,
        ordered,
        sign_distance_from_start=sign_distance_from_start,
        destination_past_sign_m=destination_past_sign_m,
    )
    if not ordered:
        return []

    if max_scenarios is not None and len(ordered) > max_scenarios:
        head = ordered[:1]
        tail = ordered[1:]
        random.shuffle(tail)
        ordered = (head + tail)[:max_scenarios]
    return ordered


# -----------------------------------------------------------------------------
# Manifest entry builder
# -----------------------------------------------------------------------------
def build_manifest_entry(
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    variant: int,
    sim_cfg: SimulationConfig,
    spawn_scenario: SpawnScenario,
    spawn_lanes_cache: Optional[List[SumoLaneInfo]] = None,
    junction_layout_cache: Optional[dict] = None,
) -> Dict[str, Any]:
    """Build one manifest row for a junction through-path scenario."""
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    net_file = meta.get("net_file", "map.net.xml")
    net_rel = scene_dir.relative_to(scenes_root) / net_file

    scenario_id = spawn_scenario.scenario_id
    seed = _stable_seed(scene_name, variant, scenario_id=scenario_id)
    scene_id = _scene_id_from_meta(meta, scene_dir)

    selected_lane = None
    if spawn_lanes_cache:
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

    # Soft env hint only; placement uses sign_distance_from_start on the
    # forbidden (destination) edge.
    if meta.get("sign_spawn_distance") is not None:
        try:
            sign_spawn_distance = float(meta["sign_spawn_distance"])
        except (TypeError, ValueError):
            sign_spawn_distance = DEFAULT_SIGN_SPAWN_DISTANCE
    else:
        sign_spawn_distance = DEFAULT_SIGN_SPAWN_DISTANCE

    sign_road_id = str(spawn_scenario.ego_destination_edge_id or "").strip()
    if not sign_road_id and spawn_scenario.ego_destination_lane_key:
        from lib.lane_keys import lane_edge_id

        sign_road_id = lane_edge_id(str(spawn_scenario.ego_destination_lane_key))

    entry: Dict[str, Any] = {
        "scene_id": scene_id,
        "scene_name": scene_name,
        "net_path": str(net_rel),
        "seed": seed,
        "var_idx": variant,
        "pdd_code": PDD_CODE,
        "sign_code": PDD_CODE,
        "sign_type": SIGN_TYPE,
        "sign_family": SIGN_FAMILY,
        "sign_title": _ACTIVE_SIGN.title,
        "sign_class": _ACTIVE_SIGN.class_name,
        "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
        "traffic_density": sim_cfg.traffic_density,
        "horizon": sim_cfg.horizon,
        "sign_road_id": sign_road_id,
        "sign_distance_from_start": sim_cfg.sign_distance_from_start,
        "destination_past_sign_m": sim_cfg.destination_past_sign_m,
        "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
        "sign_distance_before_end": sim_cfg.sign_distance_before_end,
        "compliant_stop_success_seconds": sim_cfg.compliant_stop_success_seconds,
        "compliant_stop_max_dist_m": sim_cfg.compliant_stop_max_dist_m,
        "compliant_stop_speed_mps": sim_cfg.compliant_stop_speed_mps,
        "sign_spawn_distance": sign_spawn_distance,
        "valid": True,
        "latitude": meta.get("latitude"),
        "longitude": meta.get("longitude"),
        "crop_radius_m": meta.get("crop_radius_m") or meta.get("crop_margin_m"),
        "source_osm": meta.get("source_osm"),
        "osm_file": meta.get("osm_file"),
        "sign_id": meta.get("sign_id"),
        "net_file": net_file,
        "junction_id": meta.get("junction_id"),
    }
    entry.update(spawn_scenario.to_manifest_fields())

    if selected_lane is not None:
        entry["spawn_lane_length"] = selected_lane.length
        entry["spawn_to_junction"] = selected_lane.to_junction

    if junction_layout_cache is not None:
        entry["junction_layout"] = junction_layout_cache

    return {k: v for k, v in entry.items() if v is not None}


# -----------------------------------------------------------------------------
# Manifest generation
# -----------------------------------------------------------------------------
def generate_manifest(
    scenes_dir: Path,
    output_dir: Path,
    scenario_cfg: ScenarioConfig,
    sim_cfg: SimulationConfig,
) -> List[Dict[str, Any]]:
    """Generate real_manifest.jsonl from junction no-entry crops."""
    scenes = discover_scenes(
        scenes_dir,
        respect_scene_selection=scenario_cfg.respect_scene_selection,
    )
    entries: List[Dict[str, Any]] = []
    print(
        f"[no_entry_signs] Generating manifest for {PDD_CODE} "
        f"({_ACTIVE_SIGN.title}); scenes={len(scenes)}"
    )

    n_variants = max(1, int(scenario_cfg.n_variants))
    for scene_dir in scenes:
        meta = load_scene_metadata(scene_dir)
        scene_name = str(meta.get("scene_name") or scene_dir.name)
        net_file = meta.get("net_file", "map.net.xml")
        net_full_path = scene_dir / net_file
        print(f"\n=== {scene_name} ===")

        scene_pdd = str(meta.get("pdd_code") or meta.get("sign_type") or PDD_CODE)
        if scene_pdd != PDD_CODE:
            print(
                f"  [skip] scene pdd_code={scene_pdd!r} != active {PDD_CODE!r}"
            )
            continue

        min_lane = min(scenario_cfg.min_ego_lane_m, sim_cfg.spawn_distance_before_end)
        spawn_lanes = parse_sumo_net_for_spawn_lanes(net_full_path, min_length=min_lane)
        print(f"  Found {len(spawn_lanes)} approach lane(s)")

        sign_lat = meta.get("latitude") or meta.get("center_lat")
        sign_lon = meta.get("longitude") or meta.get("center_lon")
        preferred_jid = meta.get("junction_id")
        junction_layout = build_junction_layout_for_scene(
            net_full_path,
            sign_lat=float(sign_lat) if sign_lat is not None else None,
            sign_lon=float(sign_lon) if sign_lon is not None else None,
            preferred_junction_id=str(preferred_jid) if preferred_jid else None,
        )
        if junction_layout is not None:
            print(
                f"  Junction layout: {junction_layout['shape']} @ "
                f"{junction_layout['junction_id']} "
                f"(arms={len(junction_layout.get('arms', []))})"
            )
        else:
            print("  Junction layout: unavailable (sign will use ego-lane placement)")

        scenarios = resolve_through_path_scenarios(
            net_full_path,
            meta,
            spawn_lanes,
            augment=scenario_cfg.augment,
            max_scenarios=scenario_cfg.max_scenarios_per_scene,
            min_ego_lane_m=scenario_cfg.min_ego_lane_m,
            sign_distance_from_start=sim_cfg.sign_distance_from_start,
            destination_past_sign_m=sim_cfg.destination_past_sign_m,
        )
        if not scenarios:
            print(
                f"  [skip] no through-path spawn/dest for {scene_name} "
                f"(need forbidden edge > "
                f"{sim_cfg.sign_distance_from_start + sim_cfg.destination_past_sign_m:.1f}m)"
            )
            continue

        print(f"  Through-path scenarios: {len(scenarios)}")
        scene_entries: List[Dict[str, Any]] = []
        for variant, scenario in enumerate(scenarios):
            if scenario_cfg.validate_metadrive_routes:
                md_ok, md_reason = _optional_metadrive_route_ok(
                    net_full_path,
                    road_id=scenario.ego_edge_id,
                    spawn_lane_num=scenario.ego_lane_num,
                    dest_lane_id=scenario.ego_destination_lane_key,
                    pdd_code=PDD_CODE,
                )
                if not md_ok:
                    print(
                        f"  [skip] MetaDrive route {md_reason} "
                        f"({scenario.scenario_id})"
                    )
                    continue
                if md_reason != "ok":
                    print(f"  [metadrive] {scene_name}: {md_reason}")

            for _rep in range(n_variants):
                entry = build_manifest_entry(
                    scene_dir=scene_dir,
                    scenes_root=scenes_dir,
                    meta=meta,
                    variant=variant,
                    sim_cfg=sim_cfg,
                    spawn_scenario=scenario,
                    spawn_lanes_cache=spawn_lanes,
                    junction_layout_cache=junction_layout,
                )
                scene_entries.append(entry)

        if not scene_entries:
            continue

        sample = scene_entries[0]
        print(
            f"  ok: road={sample['road_id']} "
            f"dest={sample['destination_lane_id']} "
            f"sign_road={sample.get('sign_road_id')} "
            f"sign_from_start={sample.get('sign_distance_from_start')}m "
            f"spawn_before_end={sample['spawn_distance_before_end']}m "
            f"rows={len(scene_entries)}"
        )
        entries.extend(scene_entries)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "real_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, default=str) + "\n")

    summary = {
        "pdd_code": PDD_CODE,
        "sign_type": SIGN_TYPE,
        "sign_family": SIGN_FAMILY,
        "sign_name": _ACTIVE_SIGN.title,
        "sign_class": _ACTIVE_SIGN.class_name,
        "sign_placement": (
            "artificial at start of forbidden (destination) lane "
            "(sign_distance_from_start); ego on approach "
            "(spawn_distance_before_end)"
        ),
        "total_scenes": len(scenes),
        "total_entries": len(entries),
        "variants_per_scene": n_variants,
        "augment": scenario_cfg.augment,
        "max_scenarios_per_scene": scenario_cfg.max_scenarios_per_scene,
        "min_ego_lane_m": scenario_cfg.min_ego_lane_m,
        "validate_metadrive_routes": scenario_cfg.validate_metadrive_routes,
        "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
        "traffic_density": sim_cfg.traffic_density,
        "horizon": sim_cfg.horizon,
        "sign_distance_from_start": sim_cfg.sign_distance_from_start,
        "destination_past_sign_m": sim_cfg.destination_past_sign_m,
        "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
        "sign_distance_before_end": sim_cfg.sign_distance_before_end,
        "generated_at": datetime.now().isoformat(),
        "scenes": [s.name for s in scenes],
    }

    summary_path = output_dir / "real_manifest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    manifest_meta_path = output_dir / "manifest.json"
    with open(manifest_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"entries_file": "real_manifest.jsonl", **summary},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[no_entry_signs] Wrote {len(entries)} entries -> {manifest_path}")
    return entries


# -----------------------------------------------------------------------------
# GIF rendering
# -----------------------------------------------------------------------------
def _iter_jsonl_rows(path: Path):
    """Iterate over JSONL file rows."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def render_gifs_from_manifest(
    manifest_path: Path,
    experiment_dir: Path,
    scenes_root: Path,
    gif_cfg: GifConfig,
) -> Tuple[int, int]:
    """Render GIFs for scenes from a manifest file via run_benchmark.py."""
    if not RUN_BENCH_SCRIPT.is_file():
        print(f"[GIF] run_benchmark.py not found at {RUN_BENCH_SCRIPT}", file=sys.stderr)
        return 0, 1

    if not manifest_path.is_file():
        print(f"[GIF] Manifest not found: {manifest_path}", file=sys.stderr)
        return 0, 1

    gif_dir = Path(gif_cfg.dir) if gif_cfg.dir else experiment_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    run_name = gif_cfg.run_name or experiment_dir.name

    rows: List[Dict[str, Any]] = []
    seen_keys: set[tuple] = set()

    for row in _iter_jsonl_rows(manifest_path):
        if not row.get("valid", True):
            continue
        if row.get("pdd_code") != PDD_CODE:
            continue

        scene_id = row.get("scene_id")
        seed = row.get("seed")
        if scene_id is None or seed is None:
            continue
        row_key = (scene_id, seed)
        if row_key in seen_keys:
            continue

        seen_keys.add(row_key)
        rows.append(row)

        if gif_cfg.max_scenes is not None and len(rows) >= gif_cfg.max_scenes:
            break

    if not rows:
        print(f"[GIF] No valid scenes found in manifest for {PDD_CODE}.")
        return 0, 0

    print(f"\n[GIF] Rendering {len(rows)} scene(s)...")

    model_path = gif_cfg.model_path
    if not model_path:
        default_ckpt = DEFAULT_NN_CHECKPOINTS.get(gif_cfg.policy)
        if default_ckpt is not None and Path(default_ckpt).is_file():
            model_path = str(default_ckpt)

    rendered = 0
    failed = 0
    for i, row in enumerate(rows, start=1):
        scene_uid = f"{row['scene_id']}:{row['pdd_code']}:{row['seed']}"
        cmd = [
            sys.executable,
            str(RUN_BENCH_SCRIPT),
            "--scene-uid", scene_uid,
            "--manifest", str(manifest_path),
            "--save-gifs",
            "--output-dir", str(experiment_dir),
            "--gif-dir", str(gif_dir),
            "--run-name", run_name,
            "--scenes-root", str(scenes_root),
            "--policy", gif_cfg.policy,
        ]
        if model_path:
            cmd.extend(["--model-path", model_path])
        if gif_cfg.hide_signs:
            cmd.append("--hide-signs")

        print(f"\n[GIF {i}/{len(rows)}] {scene_uid}")
        print("  " + " ".join(cmd))

        if gif_cfg.dry_run:
            rendered += 1
            continue

        seed_val = int(row.get("seed") or 0)
        var_idx = int(row.get("var_idx", 0) or 0)
        expected_gif = (
            gif_dir
            / f"{row['scene_id']}_v{var_idx}_s{seed_val}_{gif_cfg.policy}_default.gif"
        )
        if expected_gif.is_file():
            expected_gif.unlink()

        res = subprocess.run(cmd, cwd=str(RUN_BENCH_SCRIPT.parent))
        if res.returncode == 0 and expected_gif.is_file():
            rendered += 1
        else:
            failed += 1
            if res.returncode != 0:
                print(f"[GIF] Command failed with code {res.returncode}")
            else:
                print(
                    f"[GIF] Episode finished but GIF missing (likely bad route): "
                    f"{expected_gif.name}"
                )

    return rendered, failed


# -----------------------------------------------------------------------------
# Hydra entry point
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    sign_cfg = getattr(cfg, "sign", None)
    pdd_code = getattr(sign_cfg, "pdd_code", None) if sign_cfg is not None else None
    active = _set_active_sign(pdd_code)
    print(
        f"[no_entry_signs] Active sign {active.pdd_code} "
        f"({active.title}), class={active.class_name}"
    )

    scenes_dir_cfg = getattr(cfg.paths, "scenes_dir", None)
    scenes_base_cfg = getattr(cfg.paths, "scenes_base", "scenes") or "scenes"
    if scenes_dir_cfg in (None, "", "null"):
        scenes_dir = local_scenes_root(scenes_base_cfg, active.pdd_code)
    else:
        scenes_dir = Path(scenes_dir_cfg)
    if not scenes_dir.is_absolute():
        scenes_dir = (SCRIPT_DIR / scenes_dir).resolve()
    print(f"[no_entry_signs] Scenes dir: {scenes_dir}")

    experiment_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"[no_entry_signs] Output dir: {experiment_dir}")
    config_path = experiment_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    scenario_cfg = ScenarioConfig(
        n_variants=int(cfg.scenario.n_variants),
        augment=bool(getattr(cfg.scenario, "augment", True)),
        max_scenarios_per_scene=getattr(cfg.scenario, "max_scenarios_per_scene", None),
        respect_scene_selection=bool(
            getattr(cfg.scenario, "respect_scene_selection", True)
        ),
        min_ego_lane_m=float(getattr(cfg.scenario, "min_ego_lane_m", 8.0)),
        validate_metadrive_routes=bool(
            getattr(cfg.scenario, "validate_metadrive_routes", True)
        ),
    )
    sim_cfg = SimulationConfig(
        spawn_velocity_ms=float(cfg.simulation.spawn_velocity_ms),
        traffic_density=float(cfg.simulation.traffic_density),
        horizon=int(cfg.simulation.horizon),
        sign_distance_from_start=float(
            getattr(cfg.simulation, "sign_distance_from_start", 10.0)
        ),
        destination_past_sign_m=float(
            getattr(cfg.simulation, "destination_past_sign_m", 8.0)
        ),
        spawn_distance_before_end=float(
            getattr(
                cfg.simulation,
                "spawn_distance_before_end",
                DEFAULT_SPAWN_DISTANCE_BEFORE_END,
            )
        ),
        sign_distance_before_end=float(
            getattr(cfg.simulation, "sign_distance_before_end", 0.0)
        ),
        compliant_stop_success_seconds=float(
            getattr(cfg.simulation, "compliant_stop_success_seconds", 3.0)
        ),
        compliant_stop_max_dist_m=float(
            getattr(cfg.simulation, "compliant_stop_max_dist_m", 12.0)
        ),
        compliant_stop_speed_mps=float(
            getattr(cfg.simulation, "compliant_stop_speed_mps", 0.5)
        ),
    )
    gif_cfg = GifConfig(
        enabled=bool(cfg.gif.enabled),
        policy=str(cfg.gif.policy),
        max_scenes=cfg.gif.max_scenes,
        dry_run=bool(cfg.gif.dry_run),
        hide_signs=bool(cfg.gif.hide_signs),
        dir=cfg.gif.dir,
        run_name=cfg.gif.run_name,
        model_path=getattr(cfg.gif, "model_path", None),
    )

    entries = generate_manifest(
        scenes_dir=scenes_dir,
        output_dir=experiment_dir,
        scenario_cfg=scenario_cfg,
        sim_cfg=sim_cfg,
    )

    if gif_cfg.enabled and entries:
        manifest_path = experiment_dir / "real_manifest.jsonl"
        gif_rendered, gif_failed = render_gifs_from_manifest(
            manifest_path=manifest_path,
            experiment_dir=experiment_dir,
            scenes_root=scenes_dir,
            gif_cfg=gif_cfg,
        )
        resolved_gif_dir = Path(gif_cfg.dir) if gif_cfg.dir else (experiment_dir / "gifs")
        print("\n[GIF RESULTS]")
        print(f"  - GIFs rendered: {gif_rendered}")
        print(f"  - GIF failures: {gif_failed}")
        print(f"  - Experiment directory: {experiment_dir}")
        print(f"  - GIF directory: {resolved_gif_dir}")

    print("\nOutput files:")
    print(f"  - Manifest: {experiment_dir / 'real_manifest.jsonl'}")
    print(f"  - Config: {config_path}")


if __name__ == "__main__":
    main()
