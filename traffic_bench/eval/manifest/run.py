#!/usr/bin/env python3
"""Hydra entry: discover scenes, dispatch ``signs.<group>.expand.generate``, write."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

EVAL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EVAL_DIR.parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

from traffic_bench.eval.engine.expand.manifest_config import (
    DEFAULT_STOP_WAIT_STEPS,
    enrich_manifest_row,
    load_manifest_config,
)
from traffic_bench.eval.engine.expand.manifest_expansion import (
    AuxiliaryParams,
    ExpansionConfig,
)
from traffic_bench.eval.engine.sim.checkpoints import (
    DEFAULT_MODEL_PATHS,
    NN_NEED_CHECKPOINT,
    resolve_nn_checkpoint,
)
from traffic_bench.eval.engine.spawn.auxiliary_agent import DEFAULT_CONVOY_GAP_M
from traffic_bench.eval.manifest.types import (
    AuxiliaryConfig,
    ExpertConfig,
    GenerateCfg,
    GifConfig,
    ScenarioConfig,
    SimulationConfig,
)
from traffic_bench.eval.run_layout import (
    point_debug_latest,
    register_path_resolvers,
    snapshot_output_base_and_name,
)
from traffic_bench.eval.sign_registry import (
    get_profile,
    output_dir as profile_output_dir,
    scenes_dir as profile_scenes_dir,
)
from traffic_bench.scene_collection.sign_scenes.materialize.pool_index import (
    normalize_split,
)


def eval_scene_split(raw) -> str:
    """Map ``paths.split`` to a moscow-pool filter. ``debug`` uses the test pool."""
    value = str(raw or "debug").strip().lower()
    if value == "debug":
        return normalize_split("test")
    return normalize_split(value)


def _nonempty_dir(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def write_run_config(
    cfg: DictConfig,
    *,
    path: Path,
    scenes_dir: Path,
    experiment_dir: Path,
    scene_split: str,
) -> None:
    """Snapshot the resolved Hydra cfg plus the train/test scene filter actually used."""
    payload = OmegaConf.to_container(cfg, resolve=True)
    paths = payload.setdefault("paths", {})
    output_base, experiment_name = snapshot_output_base_and_name(experiment_dir)
    paths["scenes_dir"] = _repo_rel(scenes_dir)
    paths["output_base"] = output_base
    paths["experiment_name"] = experiment_name
    paths["split"] = str(getattr(cfg.paths, "split", "debug"))
    paths["scene_split"] = scene_split
    path.write_text(OmegaConf.to_yaml(payload), encoding="utf-8")


def assert_manifest_dir_clean(experiment_dir: Path, split: str) -> None:
    if str(split).strip().lower() == "debug":
        return
    leftover = [
        name
        for name in ("eval_out", "gifs")
        if _nonempty_dir(experiment_dir / name)
    ]
    has_jsonl = (experiment_dir / "real_manifest.jsonl").is_file()
    if not leftover and not has_jsonl:
        return
    try:
        shown = experiment_dir.relative_to(REPO_ROOT)
    except ValueError:
        shown = experiment_dir
    why = []
    if has_jsonl:
        why.append("real_manifest.jsonl")
    why.extend(f"{name}/" for name in leftover)
    print(
        f"Refusing to overwrite {shown}: already has {', '.join(why)}.\n"
        f"Remove the folder first:\n"
        f"  rm -rf {shown}",
        file=sys.stderr,
    )
    raise SystemExit(1)


register_path_resolvers()


def resolve_gif_model_path(policy: str, model_path: Optional[str]) -> Optional[str]:
    resolved = resolve_nn_checkpoint(policy, model_path)
    if resolved:
        return resolved
    if policy in NN_NEED_CHECKPOINT:
        default = DEFAULT_MODEL_PATHS.get(policy)
        print(
            f"[GIF] Default checkpoint missing for {policy}: {default} "
            f"(pass gif.model_path=...)",
        )
    return None


def _iter_jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def render_gifs_from_manifest(
    manifest_path: Path,
    experiment_dir: Path,
    scenes_root: Path,
    gif_cfg: GifConfig,
    sign_code: str,
) -> tuple[int, int]:
    """Render GIFs by calling the closed-loop runner in process."""
    if not manifest_path.is_file():
        print(f"[GIF] Manifest not found: {manifest_path}")
        return 0, 1

    gif_dir = Path(gif_cfg.dir) if gif_cfg.dir else experiment_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seen = set()
    manifest_defaults = load_manifest_config(manifest_path)
    for row in _iter_jsonl_rows(manifest_path):
        if not row.get("valid", True):
            continue
        if row.get("pdd_code") != sign_code:
            continue
        scene_id = row.get("scene_id")
        seed = row.get("seed")
        if scene_id is None or seed is None:
            continue
        key = (scene_id, seed)
        if key in seen:
            continue
        seen.add(key)
        row["_backend"] = "sumo"
        row["_sign_code"] = row.get("sign_code") or row.get("pdd_code") or ""
        row = enrich_manifest_row(row, manifest_defaults)
        rows.append(row)
        if gif_cfg.max_scenes is not None and len(rows) >= gif_cfg.max_scenes:
            break

    if not rows:
        print(f"[GIF] No valid scenes found in manifest for {sign_code}.")
        return 0, 0

    print(f"\n[GIF] Rendering {len(rows)} scene(s)...")
    model_path = resolve_gif_model_path(gif_cfg.policy, gif_cfg.model_path)
    if gif_cfg.policy in NN_NEED_CHECKPOINT and not model_path:
        print(
            f"[GIF] No checkpoint for policy={gif_cfg.policy}; "
            f"set gif.model_path=... or place defaults under {CHECKPOINTS_DIR}"
        )
        return 0, 1

    if gif_cfg.dry_run:
        for row in rows:
            print(f"  [dry-run] {row.get('scene_id')}:{row.get('pdd_code')}:{row.get('seed')}")
        return len(rows), 0

    from traffic_bench.eval.run.main import run_episodes

    run_episodes(
        policy=gif_cfg.policy,
        rows=rows,
        scenes_root=scenes_root,
        out_dir=experiment_dir,
        model_path=model_path,
        save_gifs=True,
        gif_dir=gif_dir,
        gif_window_m=float(gif_cfg.window_m or 80.0),
        hide_signs=bool(gif_cfg.hide_signs),
        draw_path_conflict=bool(gif_cfg.draw_path_conflict),
        run_name=gif_cfg.run_name or experiment_dir.name,
    )
    return len(rows), 0


def _resolve_max_scenarios(scenario_cfg) -> Optional[int]:
    raw = getattr(scenario_cfg, "max_scenarios", None)
    if raw is None:
        raw = getattr(scenario_cfg, "max_scenarios_per_scene", None)
    if raw is None:
        return None
    return int(raw)


def _resolve_max_total(scenario_cfg) -> Optional[int]:
    raw = getattr(scenario_cfg, "max_total", None)
    if raw is None:
        return None
    return int(raw)


def _resolve_convoy_gaps_m(raw) -> List[float]:
    if raw is None:
        return [float(DEFAULT_CONVOY_GAP_M)]
    if OmegaConf.is_list(raw) or isinstance(raw, (list, tuple)):
        gaps = [float(x) for x in raw]
        return gaps if gaps else [float(DEFAULT_CONVOY_GAP_M)]
    return [float(raw)]


def _job_from_hydra(cfg: DictConfig, profile, scenes_dir: Path, output_dir: Path) -> GenerateCfg:
    scenario_cfg = ScenarioConfig(
        max_scenarios=_resolve_max_scenarios(cfg.scenario),
        max_total=_resolve_max_total(cfg.scenario),
        min_dual_path_gain_m=float(
            getattr(cfg.scenario, "min_dual_path_gain_m", 20.0) or 20.0
        ),
    )
    sim_cfg = SimulationConfig(
        spawn_velocity_ms=cfg.simulation.spawn_velocity_ms,
        traffic_density=cfg.simulation.traffic_density,
        horizon=cfg.simulation.horizon,
        sign_distance_before_end=cfg.simulation.sign_distance_before_end,
        spawn_distance_before_end=cfg.simulation.spawn_distance_before_end,
        sign_distance_from_start=float(
            getattr(cfg.simulation, "sign_distance_from_start", 10.0) or 10.0
        ),
        n_variations=int(getattr(cfg.simulation, "n_variations", 3) or 3),
        default_first_variant=bool(
            getattr(cfg.simulation, "default_first_variant", False)
        ),
        profile_density_cap=float(
            getattr(cfg.simulation, "profile_density_cap", 1.0) or 1.0
        ),
        compliant_stop_success_seconds=float(
            getattr(cfg.simulation, "compliant_stop_success_seconds", 3.0) or 3.0
        ),
        compliant_stop_max_dist_m=float(
            getattr(cfg.simulation, "compliant_stop_max_dist_m", 12.0) or 12.0
        ),
        compliant_stop_speed_mps=float(
            getattr(cfg.simulation, "compliant_stop_speed_mps", 0.5) or 0.5
        ),
        min_hops_after_depart=int(
            getattr(cfg.simulation, "min_hops_after_depart", 0) or 0
        ),
        spawn_offset_from_start=float(
            getattr(cfg.simulation, "spawn_offset_from_start", 10.0) or 10.0
        ),
        max_path_length_m=float(
            getattr(cfg.simulation, "max_path_length_m", 150.0) or 150.0
        ),
        max_path_length_levels=tuple(
            float(x)
            for x in (
                getattr(cfg.simulation, "max_path_length_levels", None)
                or (130.0, 150.0, 170.0)
            )
        ),
        approach_before_sign_m=float(
            getattr(cfg.simulation, "approach_before_sign_m", 50.0) or 50.0
        ),
        tail_after_sign_m=float(
            getattr(cfg.simulation, "tail_after_sign_m", 30.0) or 30.0
        ),
        max_ego_lanes=int(getattr(cfg.simulation, "max_ego_lanes", 8) or 8),
        zone_tail_m=float(getattr(cfg.simulation, "zone_tail_m", 8.0) or 8.0),
        zone_min_m=float(getattr(cfg.simulation, "zone_min_m", 20.0) or 20.0),
    )
    expert_cfg = ExpertConfig(
        stop_wait_steps=int(
            getattr(getattr(cfg, "expert", None), "stop_wait_steps", DEFAULT_STOP_WAIT_STEPS)
            or DEFAULT_STOP_WAIT_STEPS
        ),
    )
    convoy_gaps_m = _resolve_convoy_gaps_m(getattr(cfg.auxiliary, "convoy_gap_m", None))
    aux_cfg = AuxiliaryConfig(
        enabled=bool(cfg.auxiliary.enabled),
        distance_from_intersection=float(cfg.auxiliary.distance_from_intersection),
        convoy_size=int(cfg.auxiliary.convoy_size),
        convoy_gap_m=float(convoy_gaps_m[0]),
        lanes_occupied=int(cfg.auxiliary.lanes_occupied),
        release_when_ego_within_m=float(
            getattr(cfg.auxiliary, "release_when_ego_within_m", 15.0) or 15.0
        ),
    )
    aug_cfg = cfg.augmentation
    layout_flag = bool(getattr(aug_cfg, "layout", False))
    legacy_augment = getattr(cfg.scenario, "augment", None)
    if legacy_augment is not None and not layout_flag and bool(legacy_augment):
        layout_flag = True
    expansion_cfg = ExpansionConfig(
        enabled=bool(getattr(aug_cfg, "enabled", True)),
        layout=layout_flag,
        auxiliary=bool(getattr(aug_cfg, "auxiliary", False)),
        max_scenarios=scenario_cfg.max_scenarios,
        aux=AuxiliaryParams(
            enabled=aux_cfg.enabled,
            distance_from_intersection=aux_cfg.distance_from_intersection,
            convoy_size=aux_cfg.convoy_size,
            convoy_gaps_m=tuple(convoy_gaps_m),
            lanes_occupied=aux_cfg.lanes_occupied,
            release_when_ego_within_m=aux_cfg.release_when_ego_within_m,
        ),
    )
    positions_raw = getattr(cfg.scenario, "crosswalk_positions", None)
    if positions_raw is None:
        positions_list = None
    elif OmegaConf.is_list(positions_raw) or isinstance(positions_raw, (list, tuple)):
        positions_list = [str(x) for x in positions_raw]
    else:
        positions_list = [str(positions_raw)]
    ped_node = getattr(cfg, "pedestrian", None)
    ped_cfg = (
        OmegaConf.to_container(ped_node, resolve=True) if ped_node is not None else {}
    )
    if not isinstance(ped_cfg, dict):
        ped_cfg = {}
    return GenerateCfg(
        profile=profile,
        scenes_dir=scenes_dir,
        output_dir=output_dir,
        split=eval_scene_split(getattr(cfg.paths, "split", "debug")),
        scenario=scenario_cfg,
        simulation=sim_cfg,
        expansion=expansion_cfg,
        auxiliary=aux_cfg,
        expert=expert_cfg,
        max_ego_lanes=int(getattr(cfg.scenario, "max_ego_lanes", 3) or 3),
        max_pedestrian_presets=int(
            getattr(cfg.scenario, "max_pedestrian_presets", 3) or 3
        ),
        crosswalk_positions=positions_list,
        ped_cfg=ped_cfg,
    )


def _generate_for_family(job: GenerateCfg):
    family = job.profile.family
    if family == "blocked":
        from traffic_bench.eval.signs.blocked.expand import generate
    elif family == "dual_path":
        from traffic_bench.eval.signs.dual_path.expand import generate
    elif family == "crosswalk":
        from traffic_bench.eval.signs.crosswalk.expand import generate
    elif family == "detour":
        from traffic_bench.eval.signs.detour.expand import generate
    elif family == "speed":
        from traffic_bench.eval.signs.speed.expand import generate
    else:
        from traffic_bench.eval.signs.junction.expand import generate
    return generate(job)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    profile = get_profile(cfg.sign)
    print(f"Sign profile: {profile.id} ({profile.pdd_code} / {profile.sign_type})")
    expected_out = profile_output_dir(profile).resolve()
    configured_out = Path(str(cfg.paths.output_base))
    if not configured_out.is_absolute():
        configured_out = (REPO_ROOT / configured_out).resolve()
    if configured_out != expected_out and profile.data_subdir not in str(configured_out):
        print(
            f"[warn] paths.output_base={cfg.paths.output_base} may not match sign={profile.id}; "
            f"preferred: data/runs/{profile.data_subdir}"
        )

    if str(cfg.paths.scenes_dir) in {"scenes", "scenes/", ""}:
        scenes_dir = profile_scenes_dir(profile)
    else:
        scenes_dir = Path(cfg.paths.scenes_dir)
        if not scenes_dir.is_absolute():
            scenes_dir = (REPO_ROOT / scenes_dir).resolve()
    print(f"Using scenes_dir: {scenes_dir}")

    experiment_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    assert_manifest_dir_clean(experiment_dir, str(getattr(cfg.paths, "split", "debug")))
    job = _job_from_hydra(cfg, profile, scenes_dir, experiment_dir)
    config_path = experiment_dir / "config.yaml"
    write_run_config(
        cfg,
        path=config_path,
        scenes_dir=scenes_dir,
        experiment_dir=experiment_dir,
        scene_split=job.split,
    )
    print(f"Using paths.split: {cfg.paths.split} (scene filter {job.split})")
    entries = _generate_for_family(job)

    gif_cfg = GifConfig(
        enabled=cfg.gif.enabled,
        policy=cfg.gif.policy,
        max_scenes=cfg.gif.max_scenes,
        dry_run=cfg.gif.dry_run,
        hide_signs=cfg.gif.hide_signs,
        dir=cfg.gif.dir,
        run_name=cfg.gif.run_name,
        window_m=float(getattr(cfg.gif, "window_m", 80.0) or 80.0),
        draw_path_conflict=bool(getattr(cfg.gif, "draw_path_conflict", False)),
        model_path=getattr(cfg.gif, "model_path", None) or None,
    )
    if gif_cfg.enabled and entries:
        gif_rendered, gif_failed = render_gifs_from_manifest(
            manifest_path=experiment_dir / "real_manifest.jsonl",
            experiment_dir=experiment_dir,
            scenes_root=scenes_dir,
            gif_cfg=gif_cfg,
            sign_code=profile.pdd_code,
        )
        resolved_gif_dir = Path(gif_cfg.dir) if gif_cfg.dir else (experiment_dir / "gifs")
        print("\n[GIF RESULTS]")
        print(f"  - GIFs rendered: {gif_rendered}")
        print(f"  - GIF failures: {gif_failed}")
        print(f"  - Experiment directory: {experiment_dir}")
        print(f"  - GIF directory: {resolved_gif_dir}")

    latest = point_debug_latest(experiment_dir)
    print("\nOutput files:")
    print(f"  - Manifest: {experiment_dir / 'real_manifest.jsonl'}")
    print(f"  - Repro: {experiment_dir / 'repro'}")
    print(f"  - Config: {config_path}")
    if latest is not None:
        print(f"  - Latest: {latest}")


if __name__ == "__main__":
    main()
