#!/usr/bin/env python3
"""Generate evaluation manifest for pedestrian crossing (PDD 5.19) scenes."""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from lib.crosswalk_layout import CrosswalkApproach, build_crosswalk_approaches, net_has_crossings
from lib.lane_keys import lane_edge_id
from lib.manifest_config import DEFAULT_SPAWN_DISTANCE_BEFORE_END
from lib.sumo_utils import resolve_net_file


SCRIPT_DIR = Path(__file__).parent.resolve()
RUN_BENCH_SCRIPT = SCRIPT_DIR / "run_benchmark.py"

PDD_CODE = "5.19"
SIGN_TYPE = "crosswalk"


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


@dataclass
class SimulationConfig:
    spawn_velocity_ms: float = 2.5
    traffic_density: float = 0.0
    horizon: int = 600
    spawn_distance_before_end: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END
    min_hops_after_depart: int = 2
    max_destination_hops: int = 8


@dataclass
class PedestrianConfig:
    initial_pedestrians: int = 2
    max_pedestrians: int = 6
    spawn_probability: float = 0.12
    crossing_interval_range: Tuple[float, float] = (5.0, 10.0)
    max_active_per_crosswalk: int = 1
    speed_mean: float = 1.2
    speed_std: float = 0.2
    yield_distance: float = 12.0
    no_stop_before_crosswalk_m: float = 3.0


@dataclass
class GifConfig:
    enabled: bool = False
    policy: str = "idm"
    max_scenes: Optional[int] = None
    dry_run: bool = False
    hide_signs: bool = True
    dir: Optional[str] = None
    run_name: Optional[str] = None


@dataclass
class ManifestConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    pedestrian: PedestrianConfig = field(default_factory=PedestrianConfig)
    gif: GifConfig = field(default_factory=GifConfig)


def _stable_seed(scene_name: str, variant: int, scenario_id: str = "") -> int:
    h = hashlib.sha256()
    h.update(scene_name.encode("utf-8"))
    h.update(b"|")
    h.update(str(variant).encode("utf-8"))
    if scenario_id:
        h.update(b"|")
        h.update(scenario_id.encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")


def discover_scenes(scenes_dir: Path, *, respect_scene_selection: bool = True) -> List[Path]:
    """Find cropped crosswalk scene directories (skips core/ and rejected scenes)."""
    from lib.crosswalk_crop import is_crosswalk_scene_meta
    from lib.scene_selection import is_reserved_scene_dir, is_scene_rejected

    scenes: List[Path] = []
    if not scenes_dir.is_dir():
        return scenes

    for entry in sorted(scenes_dir.iterdir()):
        if not entry.is_dir():
            continue
        if is_reserved_scene_dir(entry.name):
            continue
        if respect_scene_selection and is_scene_rejected(scenes_dir, entry.name):
            print(f"  [skip] {entry.name}: rejected in scene_selection.json")
            continue

        meta_path = entry / "meta.json"
        if not meta_path.is_file():
            continue

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        if not is_crosswalk_scene_meta(meta):
            print(f"  [skip] {entry.name}: not a cropped crosswalk scene (run crop_crosswalk_scene.py)")
            continue

        net_file = resolve_net_file(entry, meta)
        if (entry / net_file).is_file():
            scenes.append(entry)
    return scenes


def load_scene_metadata(scene_dir: Path) -> Dict[str, Any]:
    meta_path = scene_dir / "meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    center_path = scene_dir / "center.json"
    if center_path.is_file():
        with open(center_path, encoding="utf-8") as f:
            center = json.load(f)
            meta["center_lat"] = center.get("lat")
            meta["center_lon"] = center.get("lon")
    return meta


def _pedestrian_manager_dict(ped_cfg: PedestrianConfig) -> dict[str, Any]:
    return {
        "enabled": True,
        "initial_pedestrians": ped_cfg.initial_pedestrians,
        "max_pedestrians": ped_cfg.max_pedestrians,
        "spawn_by_interval": True,
        "spawn_probability": ped_cfg.spawn_probability,
        "crossing_interval_range": list(ped_cfg.crossing_interval_range),
        "max_active_per_crosswalk": ped_cfg.max_active_per_crosswalk,
        "speed_mean": ped_cfg.speed_mean,
        "speed_std": ped_cfg.speed_std,
        "yield_distance": ped_cfg.yield_distance,
        "no_stop_before_crosswalk_m": ped_cfg.no_stop_before_crosswalk_m,
        "yield_to_vehicles": True,
        "yield_on_crosswalk": False,
    }


def build_manifest_entry(
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    approach: CrosswalkApproach,
    variant: int,
    sim_cfg: SimulationConfig,
    ped_cfg: PedestrianConfig,
) -> Dict[str, Any]:
    scene_name = meta.get("scene_name", scene_dir.name)
    net_file = meta.get("net_file", resolve_net_file(scene_dir, meta))
    net_path = scene_dir.relative_to(scenes_root) / net_file
    seed = _stable_seed(scene_name, variant, approach.scenario_id)
    scene_id = f"{scene_name}_{approach.scenario_id}_v{variant}"

    return {
        "valid": True,
        "scene_id": scene_id,
        "scene_name": scene_name,
        "sign_id": meta.get("sign_id", scene_name.replace("sign_", "")),
        "pdd_code": PDD_CODE,
        "sign_type": SIGN_TYPE,
        "sign_code": PDD_CODE,
        "net_path": str(net_path),
        "seed": seed,
        "deterministic_seed": seed,
        "var_idx": variant,
        "scenario_id": approach.scenario_id,
        "crosswalk_id": approach.crosswalk_id,
        "junction_id": approach.junction_id,
        "road_id": approach.approach_edge_id,
        "spawn_lane_num": approach.approach_lane_num,
        "depart_edge_id": approach.depart_edge_id,
        "destination_lane_id": approach.destination_lane_id,
        "destination_edge_id": lane_edge_id(approach.destination_lane_id),
        "min_hops_after_depart": sim_cfg.min_hops_after_depart,
        "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
        "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
        "traffic_density": sim_cfg.traffic_density,
        "horizon": sim_cfg.horizon,
        "use_pedestrian_manager": True,
        "use_pedestrian_yield_rule": True,
        "pedestrian_manager": _pedestrian_manager_dict(ped_cfg),
        "auxiliary_agent": False,
        "center_lat": meta.get("center_lat"),
        "center_lon": meta.get("center_lon"),
        "approach_lane_length_m": approach.approach_lane_length,
        "crossed_edge_ids": list(approach.crossed_edge_ids),
    }


def generate_manifest(
    scenes_dir: Path,
    output_dir: Path,
    scenario_cfg: ScenarioConfig,
    sim_cfg: SimulationConfig,
    ped_cfg: PedestrianConfig,
) -> List[Dict[str, Any]]:
    scenes = discover_scenes(scenes_dir, respect_scene_selection=scenario_cfg.respect_scene_selection)
    print(f"Found {len(scenes)} scene(s) in {scenes_dir}")

    entries: List[Dict[str, Any]] = []
    for scene_dir in scenes:
        scene_name = scene_dir.name
        meta = load_scene_metadata(scene_dir)
        net_file = resolve_net_file(scene_dir, meta)
        net_full_path = scene_dir / net_file

        if not net_has_crossings(net_full_path):
            print(f"  [skip] {scene_name}: no SUMO crossings")
            continue

        approaches = build_crosswalk_approaches(
            net_full_path,
            min_approach_length=sim_cfg.spawn_distance_before_end,
            min_hops_after_depart=sim_cfg.min_hops_after_depart,
            max_destination_hops=sim_cfg.max_destination_hops,
        )
        target_crosswalk = meta.get("crosswalk_id")
        if target_crosswalk:
            approaches = [a for a in approaches if a.crosswalk_id == target_crosswalk]
        print(f"  {scene_name}: {len(approaches)} crosswalk approach(es)")

        if not approaches:
            print(f"  [skip] {scene_name}: no viable approach lanes")
            continue

        if scenario_cfg.max_scenarios_per_scene is not None:
            random.shuffle(approaches)
            approaches = approaches[: scenario_cfg.max_scenarios_per_scene]

        n_variants = max(1, scenario_cfg.n_variants) if scenario_cfg.augment else 1
        for approach in approaches:
            for variant in range(n_variants):
                entries.append(
                    build_manifest_entry(
                        scene_dir=scene_dir,
                        scenes_root=scenes_dir,
                        meta=meta,
                        approach=approach,
                        variant=variant,
                        sim_cfg=sim_cfg,
                        ped_cfg=ped_cfg,
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "real_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, default=str) + "\n")

    summary = {
        "pdd_code": PDD_CODE,
        "sign_type": SIGN_TYPE,
        "sign_name": "Pedestrian crossing",
        "total_scenes": len(scenes),
        "total_entries": len(entries),
        "variants_per_scene": scenario_cfg.n_variants,
        "augment": scenario_cfg.augment,
        "max_scenarios_per_scene": scenario_cfg.max_scenarios_per_scene,
        "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
        "traffic_density": sim_cfg.traffic_density,
        "horizon": sim_cfg.horizon,
        "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
        "pedestrian_manager": _pedestrian_manager_dict(ped_cfg),
        "auxiliary_agent": False,
        "generated_at": datetime.now().isoformat(),
        "scenes": [s.name for s in scenes],
    }

    summary_path = output_dir / "real_manifest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    manifest_meta_path = output_dir / "manifest.json"
    with open(manifest_meta_path, "w", encoding="utf-8") as f:
        json.dump({"entries_file": "real_manifest.jsonl", **summary}, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(entries)} manifest entries -> {manifest_path}")
    return entries


def _iter_jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def render_gifs_from_manifest(
    manifest_path: Path,
    experiment_dir: Path,
    scenes_root: Path,
    gif_cfg: GifConfig,
) -> Tuple[int, int]:
    if not RUN_BENCH_SCRIPT.is_file():
        print(f"[GIF] run_benchmark.py not found at {RUN_BENCH_SCRIPT}", file=sys.stderr)
        return 0, 1
    if not manifest_path.is_file():
        print(f"[GIF] Manifest not found: {manifest_path}", file=sys.stderr)
        return 0, 1

    gif_dir = Path(gif_cfg.dir) if gif_cfg.dir else experiment_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    run_name = gif_cfg.run_name or experiment_dir.name

    rows = []
    seen_keys = set()
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
    rendered = 0
    failed = 0
    for i, row in enumerate(rows, start=1):
        scene_uid = f"{row['scene_id']}:{row['pdd_code']}:{row['seed']}"
        cmd = [
            sys.executable,
            str(RUN_BENCH_SCRIPT),
            "--scene-uid",
            scene_uid,
            "--manifest",
            str(manifest_path),
            "--save-gifs",
            "--output-dir",
            str(experiment_dir),
            "--gif-dir",
            str(gif_dir),
            "--run-name",
            run_name,
            "--scenes-root",
            str(scenes_root),
            "--policy",
            gif_cfg.policy,
            "--no-auxiliary-agent",
        ]
        if gif_cfg.hide_signs:
            cmd.append("--hide-signs")

        print(f"\n[GIF {i}/{len(rows)}] {scene_uid}")
        print("  " + " ".join(cmd))
        if gif_cfg.dry_run:
            rendered += 1
            continue
        res = subprocess.run(cmd, cwd=str(RUN_BENCH_SCRIPT.parent))
        if res.returncode == 0:
            rendered += 1
        else:
            failed += 1
            print(f"[GIF] Command failed with code {res.returncode}")
    return rendered, failed


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    scenes_dir = Path(cfg.paths.scenes_dir)
    experiment_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_path = experiment_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    scenario_cfg = ScenarioConfig(
        n_variants=cfg.scenario.n_variants,
        augment=cfg.scenario.augment,
        max_scenarios_per_scene=cfg.scenario.max_scenarios_per_scene,
        respect_scene_selection=cfg.scenario.get("respect_scene_selection", True),
    )
    sim_cfg = SimulationConfig(
        spawn_velocity_ms=cfg.simulation.spawn_velocity_ms,
        traffic_density=cfg.simulation.traffic_density,
        horizon=cfg.simulation.horizon,
        spawn_distance_before_end=cfg.simulation.spawn_distance_before_end,
        min_hops_after_depart=int(cfg.simulation.get("min_hops_after_depart", 2)),
        max_destination_hops=int(cfg.simulation.get("max_destination_hops", 8)),
    )
    ped_interval = cfg.pedestrian.crossing_interval_range
    ped_cfg = PedestrianConfig(
        initial_pedestrians=cfg.pedestrian.initial_pedestrians,
        max_pedestrians=cfg.pedestrian.max_pedestrians,
        spawn_probability=cfg.pedestrian.spawn_probability,
        crossing_interval_range=(float(ped_interval[0]), float(ped_interval[1])),
        max_active_per_crosswalk=cfg.pedestrian.max_active_per_crosswalk,
        speed_mean=cfg.pedestrian.speed_mean,
        speed_std=cfg.pedestrian.speed_std,
        yield_distance=cfg.pedestrian.yield_distance,
        no_stop_before_crosswalk_m=cfg.pedestrian.no_stop_before_crosswalk_m,
    )
    gif_cfg = GifConfig(
        enabled=cfg.gif.enabled,
        policy=cfg.gif.policy,
        max_scenes=cfg.gif.max_scenes,
        dry_run=cfg.gif.dry_run,
        hide_signs=cfg.gif.hide_signs,
        dir=cfg.gif.dir,
        run_name=cfg.gif.run_name,
    )

    entries = generate_manifest(
        scenes_dir=scenes_dir,
        output_dir=experiment_dir,
        scenario_cfg=scenario_cfg,
        sim_cfg=sim_cfg,
        ped_cfg=ped_cfg,
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
        print(f"  - GIF directory: {resolved_gif_dir}")

    print("\nOutput files:")
    print(f"  - Manifest: {experiment_dir / 'real_manifest.jsonl'}")
    print(f"  - Config: {config_path}")


if __name__ == "__main__":
    main()
