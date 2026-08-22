"""Hydra entry for one policy (or a list via ``policies.py``)."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from traffic_bench.eval.engine.sim.checkpoints import (
    DEFAULT_MODEL_PATHS,
    NN_NEED_CHECKPOINT,
)
from traffic_bench.eval.engine.traffic.ego_defaults import sample_ego_params
from traffic_bench.eval.run.episode import (
    _episode_key_from_result,
    _episode_key_from_row,
    _load_enriched_manifest_rows,
    _load_existing_results,
    _load_policy_models,
    aggregate_results,
    resolve_model_path,
    run_one_episode,
)

EVAL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EVAL_DIR.parent.parent


def _as_list(value) -> list | None:
    if value is None:
        return None
    if OmegaConf.is_list(value) or isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    text = str(value).strip()
    if not text:
        return None
    return [p.strip() for p in text.split(",") if p.strip()]


def _bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def resolve_manifest_file(raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    if path.is_dir():
        candidate = path / "real_manifest.jsonl"
        if not candidate.is_file():
            raise FileNotFoundError(f"No real_manifest.jsonl in {path}")
        return candidate
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    return path


def run_episodes(
    *,
    policy: str,
    rows: list[dict],
    scenes_root: Path,
    out_dir: Path,
    ego_variant: str = "default",
    ego_sample_seed_base: int = 42,
    max_steps: int = 1500,
    model_path: str | None = None,
    plant2_action_mode: str = "pid",
    force_rerun: bool = False,
    rerun_failed: bool = False,
    skip_error_episodes: bool = False,
    emit_replay_sidecar: bool = False,
    replay_root: Path | None = None,
    save_gifs: bool = False,
    gif_dir: Path | None = None,
    gif_window_m: float = 80.0,
    hide_signs: bool = False,
    draw_path_conflict: bool = False,
    run_name: str | None = None,
) -> Path:
    """Run closed-loop episodes and write ``episodes_<policy>.jsonl``."""
    if ego_variant != "default":
        seed = ego_sample_seed_base + 12345
        first = sample_ego_params(seed)
        second = sample_ego_params(seed)
        for key in first:
            assert math.isclose(float(first[key]), float(second[key]), abs_tol=1e-9)

    logging.getLogger().setLevel(logging.CRITICAL)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / f"episodes_{policy}.jsonl"

    model_path = resolve_model_path(policy, model_path)
    if policy in NN_NEED_CHECKPOINT and not model_path:
        default = DEFAULT_MODEL_PATHS.get(policy)
        raise ValueError(
            f"model_path is required for policy {policy}"
            + (f" (default missing: {default})" if default else "")
        )
    models = _load_policy_models(policy, model_path, plant2_action_mode=plant2_action_mode)

    replay_dir: Path | None = None
    if emit_replay_sidecar:
        replay_dir = Path(replay_root) if replay_root else (out_dir / "replays")
        replay_dir.mkdir(parents=True, exist_ok=True)

    gifs_dir: Path | None = None
    if save_gifs:
        gifs_dir = Path(gif_dir) if gif_dir else (out_dir / "gifs")
        gifs_dir.mkdir(parents=True, exist_ok=True)

    existing_results = _load_existing_results(episodes_path)
    existing_by_key: dict[tuple[str, str, int], dict] = {
        _episode_key_from_result(r): r for r in existing_results
    }

    rows_to_run: list[dict] = []
    skipped = 0
    for row in rows:
        key = _episode_key_from_row(row)
        old = existing_by_key.get(key)
        if force_rerun or old is None:
            rows_to_run.append(row)
            continue
        if skip_error_episodes and not bool(old.get("ok", False)):
            skipped += 1
            continue
        if rerun_failed and not bool(old.get("ok", False)):
            rows_to_run.append(row)
            continue
        skipped += 1

    print(
        f"Resume: loaded {len(existing_results)} existing episodes, "
        f"skip {skipped}, run {len(rows_to_run)}"
        + (" (force_rerun)" if force_rerun else "")
    )

    results_by_key: dict[tuple[str, str, int], dict] = dict(existing_by_key)
    write_mode = "a" if episodes_path.exists() else "w"
    variant = ego_variant
    with open(episodes_path, write_mode, encoding="utf-8") as handle:
        for idx, row in enumerate(rows_to_run, start=1):
            scene_id = row.get("scene_id")
            sign_code = row.get("_sign_code")
            print(f"[{idx}/{len(rows_to_run)}] sign={sign_code} scene={scene_id}")
            gif_path = None
            if gifs_dir is not None:
                seed_val = int(row.get("seed") or row.get("deterministic_seed") or 0)
                var_idx = int(row.get("var_idx", 0) or 0)
                uid = f"{scene_id or 'scene'}_v{var_idx}_s{seed_val}"
                gif_path = gifs_dir / f"{uid}_{policy}_{variant}.gif"
            t0 = time.time()
            result = run_one_episode(
                row=row,
                policy_type=policy,
                models=models,
                scenes_root=scenes_root,
                max_steps=max_steps,
                ego_variant=variant,
                ego_sample_seed_base=ego_sample_seed_base,
                replay_root=replay_dir,
                save_gif=gif_path,
                gif_window_m=gif_window_m,
                hide_signs=hide_signs,
                draw_path_conflict=draw_path_conflict,
            )
            print(f"{policy}  elapsed_s={time.time() - t0:.3f}")
            key = _episode_key_from_row(row)
            results_by_key[key] = result
            handle.write(json.dumps(result, default=str) + "\n")
            handle.flush()

    summary = aggregate_results(list(results_by_key.values()))
    summary_path = out_dir / f"summary_{policy}.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    ok_runs = sum(1 for r in results_by_key.values() if r.get("ok"))
    print("\n=== Done ===")
    print(f"Episodes OK: {ok_runs}/{len(results_by_key)}")
    print(f"Episodes: {episodes_path}")
    print(f"Summary:  {summary_path}")
    if run_name:
        print(f"Run name: {run_name}")
    return episodes_path


def _load_rows(cfg: DictConfig) -> tuple[list[dict], Path]:
    if not cfg.manifest:
        raise ValueError("manifest= is required")
    manifest_path = resolve_manifest_file(cfg.manifest)
    rows: list[dict] = []
    for row in _load_enriched_manifest_rows(manifest_path):
        if "valid" in row and not row["valid"]:
            continue
        row["_backend"] = "sumo"
        if not row.get("_sign_code"):
            row["_sign_code"] = (
                row.get("sign_code") or row.get("pdd_code") or row.get("sign_type") or ""
            )
        rows.append(row)
    scene_id = cfg.get("scene_id")
    scene_uid = cfg.get("scene_uid")
    if scene_id and scene_uid:
        raise ValueError("scene_id and scene_uid are mutually exclusive")
    if scene_id:
        rows = [r for r in rows if str(r.get("scene_id")) == str(scene_id)]
    if scene_uid:
        rows = [
            r for r in rows
            if ":".join(str(x) for x in _episode_key_from_row(r)) == str(scene_uid)
        ]
    scene_line = cfg.get("scene_line")
    if scene_line is not None:
        idx = int(scene_line)
        if not (1 <= idx <= len(rows)):
            raise ValueError(f"scene_line {idx} out of range [1, {len(rows)}]")
        rows = [rows[idx - 1]]
    max_scenes = cfg.get("max_scenes")
    if max_scenes is not None and int(max_scenes) < len(rows):
        rows = rows[: int(max_scenes)]
    if not rows:
        raise RuntimeError("No scenes selected. Check manifest=/scene_id=/scene_uid=")
    return rows, manifest_path


def _scenes_root(cfg: DictConfig, manifest_path: Path) -> Path:
    raw = cfg.get("scenes_root")
    if raw:
        path = Path(str(raw))
        return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    saved = manifest_path.parent / "config.yaml"
    if saved.is_file():
        from omegaconf import OmegaConf as OC

        saved_cfg = OC.load(saved)
        scenes = saved_cfg.get("paths", {}).get("scenes_dir")
        if scenes:
            path = Path(str(scenes))
            return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    return (REPO_ROOT / "data" / "scenes").resolve()


def run_one_policy(cfg: DictConfig) -> Path:
    rows, manifest_path = _load_rows(cfg)
    scenes_root = _scenes_root(cfg, manifest_path)
    policy = str(cfg.policy)
    run_name = str(cfg.run_name or policy)
    if cfg.output_dir:
        out_dir = Path(str(cfg.output_dir)).resolve()
    elif manifest_path.name == "real_manifest.jsonl":
        out_dir = manifest_path.parent / "eval_out" / run_name
    else:
        out_dir = Path("eval_out").resolve() / run_name
    gif_cfg = cfg.get("gif") or {}
    print(f"Policy: {policy}")
    print(f"Selected scenes: {len(rows)}")
    print(f"Scenes root: {scenes_root}")
    return run_episodes(
        policy=policy,
        rows=rows,
        scenes_root=scenes_root,
        out_dir=out_dir,
        ego_variant=str(cfg.ego_variant),
        ego_sample_seed_base=int(cfg.ego_sample_seed_base),
        max_steps=int(cfg.max_steps),
        model_path=cfg.model_path,
        plant2_action_mode=str(cfg.plant2_action_mode),
        force_rerun=_bool(cfg.force_rerun),
        rerun_failed=_bool(cfg.rerun_failed),
        skip_error_episodes=_bool(cfg.skip_error_episodes),
        emit_replay_sidecar=_bool(cfg.emit_replay_sidecar),
        replay_root=Path(str(cfg.replay_root)) if cfg.replay_root else None,
        save_gifs=_bool(gif_cfg.get("enabled")),
        gif_dir=Path(str(gif_cfg.dir)) if gif_cfg.get("dir") else None,
        gif_window_m=float(gif_cfg.get("window_m") or 80.0),
        hide_signs=_bool(cfg.hide_signs),
        draw_path_conflict=_bool(cfg.draw_path_conflict) or _bool(gif_cfg.get("draw_path_conflict")),
        run_name=run_name,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="run")
def main(cfg: DictConfig) -> None:
    policies = _as_list(cfg.get("policies"))
    if policies:
        from traffic_bench.eval.run.policies import run_policy_list

        run_policy_list(cfg, policies)
        return
    run_one_policy(cfg)


if __name__ == "__main__":
    main()
