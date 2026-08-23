"""Run several policies on one manifest, then write the metrics table."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from traffic_bench.eval.engine.expand.manifest_config import (
    DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    enrich_manifest_row,
    load_manifest_config,
)
from traffic_bench.eval.engine.sim.checkpoints import (
    DEFAULT_MODEL_PATHS,
    NN_NEED_CHECKPOINT,
    resolve_nn_checkpoint,
)
from traffic_bench.eval.run.episode import _load_enriched_manifest_rows
from traffic_bench.eval.run.main import (
    REPO_ROOT,
    _bool,
    resolve_manifest_file,
    run_episodes,
)

IDM_FAMILY = {"idm", "comprehensive_rule_expert"}
NN_NO_CHECKPOINT = {"rule_compliant", "ppo_lidar"}
ALL_POLICIES = IDM_FAMILY | NN_NEED_CHECKPOINT | NN_NO_CHECKPOINT
EGO_VARIANTS = ["default", "s1", "s2", "s3", "s4"]
RUN_EVAL_OUT = "eval_out"


def plan_baselines(
    policies: list[str],
    variants: list[str] | None = None,
) -> list[tuple[str, str]]:
    idm_variants = list(variants) if variants is not None else list(EGO_VARIANTS)
    out: list[tuple[str, str]] = []
    for policy in policies:
        if policy in IDM_FAMILY:
            out.extend((policy, variant) for variant in idm_variants)
        else:
            out.append((policy, "default"))
    return out


def resolve_ego_variants(cfg: DictConfig) -> list[str] | None:
    """``null`` / ``all`` → default,s1–s4 for IDM-family. Else a subset."""
    raw = cfg.get("ego_variants")
    if raw is None:
        return None
    if OmegaConf.is_list(raw) or isinstance(raw, (list, tuple)):
        names = [str(x).strip() for x in raw if str(x).strip()]
    else:
        text = str(raw).strip()
        if not text or text.lower() in {"all", "*"}:
            return None
        names = [p.strip() for p in text.split(",") if p.strip()]
    bad = [name for name in names if name not in EGO_VARIANTS]
    if bad:
        raise ValueError(f"Unknown ego_variants: {bad}. Supported: {EGO_VARIANTS}")
    return names


def _model_paths(cfg: DictConfig, policies: list[str]) -> dict[str, str]:
    raw = cfg.get("model_paths") or {}
    if isinstance(raw, str):
        parsed: dict[str, str] = {}
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"model_paths: bad item {item!r}; expected policy:path")
            key, value = item.split(":", 1)
            parsed[key.strip()] = value.strip()
        raw = parsed
    elif OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"model_paths must be a mapping, got {type(raw).__name__}")
    paths = {str(k): str(v) for k, v in raw.items()}
    if cfg.get("model_path") and len(policies) == 1:
        paths.setdefault(policies[0], str(cfg.model_path))
    for policy in policies:
        if policy not in NN_NEED_CHECKPOINT:
            continue
        resolved = resolve_nn_checkpoint(policy, paths.get(policy))
        if not resolved:
            default = DEFAULT_MODEL_PATHS.get(policy)
            raise FileNotFoundError(
                f"No model_paths entry for {policy!r} and default missing: {default}"
            )
        paths[policy] = resolved
    return paths


def _assemble_rows(manifest_path: Path, cfg: DictConfig) -> list[dict]:
    config = load_manifest_config(manifest_path)
    rows: list[dict] = []
    for row in _load_enriched_manifest_rows(manifest_path):
        if "valid" in row and not row["valid"]:
            continue
        row = enrich_manifest_row(row, config)
        row["_backend"] = "sumo"
        if not row.get("_sign_code"):
            row["_sign_code"] = (
                row.get("sign_code") or row.get("pdd_code") or row.get("sign_type") or ""
            )
        rows.append(row)
    scene_line = cfg.get("scene_line")
    if scene_line is not None:
        idx = int(scene_line)
        rows = [rows[idx - 1]]
    max_scenes = cfg.get("max_scenes")
    if max_scenes is not None and int(max_scenes) < len(rows):
        rows = rows[: int(max_scenes)]
    if not rows:
        raise RuntimeError("No scenes left after filtering")
    return rows


def _run_metrics(out_dir: Path) -> None:
    from traffic_bench.eval.cli import _run_module_main
    from traffic_bench.eval.metrics import aggregate as aggregate_mod
    from traffic_bench.eval.metrics import csv as csv_mod
    from traffic_bench.eval.metrics import report as report_mod

    csv_code = _run_module_main(
        csv_mod,
        [
            "--episodes-root",
            str(out_dir / "benchmark" / "full" / "policy_eval"),
            "--out",
            str(out_dir / "metrics_per_episode.csv"),
        ],
    )
    if csv_code != 0:
        raise RuntimeError(f"metrics csv failed (exit {csv_code})")
    agg_code = _run_module_main(
        aggregate_mod,
        ["--csv", str(out_dir / "metrics_per_episode.csv"), "--out-dir", str(out_dir)],
    )
    if agg_code != 0:
        raise RuntimeError(f"metrics aggregate failed (exit {agg_code})")
    report_code = _run_module_main(
        report_mod,
        [
            "--run-root",
            str(out_dir),
            "--cumulative",
            str(out_dir / "reports" / "cumulative.json"),
        ],
    )
    if report_code != 0:
        raise RuntimeError(f"metrics report failed (exit {report_code})")


def _expand_policy_names(policies: list[str]) -> list[str]:
    if len(policies) == 1 and policies[0] in {"all", "*"}:
        return sorted(ALL_POLICIES)
    return policies


def print_run_plan(
    *,
    manifest_path: Path,
    rows: list[dict],
    policies: list[str],
    out_dir: Path,
    expand_idm: bool = True,
    variants: list[str] | None = None,
) -> None:
    if expand_idm:
        baselines = plan_baselines(policies, variants)
    else:
        baselines = [(p, "default") for p in policies]
    n_rows = len(rows)
    n_ep = n_rows * len(baselines)
    print("======== eval run ========")
    print(f"  manifest:  {manifest_path}")
    print(f"  rows:      {n_rows}")
    print(
        f"  policies:  {len(policies)} names → {len(baselines)} baselines "
        f"({', '.join(f'{p}_{v}' for p, v in baselines)})"
    )
    print(f"  episodes:  {n_rows} × {len(baselines)} = {n_ep}")
    print(f"  output:    {out_dir}")
    print("==========================")


def run_policy_list(cfg: DictConfig, policies: list[str]) -> None:
    policies = _expand_policy_names(policies)
    bad = [p for p in policies if p not in ALL_POLICIES]
    if bad:
        raise ValueError(f"Unknown policies: {bad}. Supported: {sorted(ALL_POLICIES)}")
    manifest_path = resolve_manifest_file(cfg.manifest)
    rows = _assemble_rows(manifest_path, cfg)
    scenes_root = cfg.get("scenes_root")
    if scenes_root:
        scenes_root = Path(str(scenes_root))
        scenes_root = scenes_root.resolve() if scenes_root.is_absolute() else (REPO_ROOT / scenes_root).resolve()
    else:
        from traffic_bench.eval.run.main import _scenes_root

        scenes_root = _scenes_root(cfg, manifest_path)

    if cfg.output_dir:
        out_dir = Path(str(cfg.output_dir)).resolve()
    elif manifest_path.name == "real_manifest.jsonl":
        out_dir = manifest_path.parent / RUN_EVAL_OUT
    else:
        out_dir = Path("./eval_out").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = resolve_ego_variants(cfg)
    print_run_plan(
        manifest_path=manifest_path,
        rows=rows,
        policies=policies,
        out_dir=out_dir,
        variants=variants,
    )

    model_paths = _model_paths(cfg, policies)
    baselines = plan_baselines(policies, variants)

    input_manifest = out_dir / "input_manifest.jsonl"
    input_manifest.write_text(
        "\n".join(json.dumps(row, default=str) for row in rows) + "\n",
        encoding="utf-8",
    )
    gif_cfg = cfg.get("gif") or {}
    save_gifs = _bool(gif_cfg.get("enabled"))
    jobs = max(1, int(cfg.get("jobs") or 1))
    if save_gifs and jobs > 1:
        print("gif.enabled=true: jobs=1 (Panda3D ShowBase is not thread-safe)")
        jobs = 1
    bench_root = out_dir / "benchmark" / "full" / "policy_eval"

    def _one(policy: str, variant: str, shard_rows: list[dict], shard_out: Path) -> None:
        run_episodes(
            policy=policy,
            rows=shard_rows,
            scenes_root=scenes_root,
            out_dir=shard_out,
            ego_variant=variant,
            ego_sample_seed_base=int(cfg.ego_sample_seed_base),
            max_steps=int(cfg.max_steps),
            model_path=model_paths.get(policy),
            plant2_action_mode=str(cfg.plant2_action_mode),
            force_rerun=_bool(cfg.force_rerun),
            rerun_failed=_bool(cfg.rerun_failed),
            skip_error_episodes=_bool(cfg.skip_error_episodes),
            emit_replay_sidecar=True,
            replay_root=out_dir / "runs" / "var_0" / f"{policy}_{variant}" / "replays",
            save_gifs=save_gifs,
            gif_dir=Path(str(gif_cfg.dir)) if gif_cfg.get("dir") else None,
            gif_window_m=float(gif_cfg.get("window_m") or 80.0),
            hide_signs=_bool(cfg.hide_signs),
            draw_path_conflict=_bool(cfg.draw_path_conflict)
            or _bool(gif_cfg.get("draw_path_conflict")),
            run_name=f"{policy}_{variant}",
        )

    for policy, variant in baselines:
        run_name = f"{policy}_{variant}"
        final_dir = bench_root / run_name
        if jobs <= 1 or len(rows) <= 1:
            _one(policy, variant, rows, final_dir)
            continue
        workers = min(jobs, len(rows))
        shards_root = out_dir / "_scene_shards" / run_name
        shards_root.mkdir(parents=True, exist_ok=True)
        tasks = []
        for idx, row in enumerate(rows):
            shard_out = final_dir / "_shards" / f"{idx:04d}"
            tasks.append((idx, [row], shard_out))
        print(f"\n[{run_name}] parallelizing {len(tasks)} scene(s) with jobs={workers}")
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_one, policy, variant, shard_rows, shard_out): idx
                for idx, shard_rows, shard_out in tasks
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append(f"row {idx}: {exc}")
        if failures:
            raise RuntimeError(f"[{run_name}] scene failures:\n  - " + "\n  - ".join(failures))
        merged = final_dir / f"episodes_{policy}.jsonl"
        lines: list[str] = []
        for _, _, shard_out in tasks:
            ep = shard_out / f"episodes_{policy}.jsonl"
            if ep.is_file():
                lines.extend(ln for ln in ep.read_text(encoding="utf-8").splitlines() if ln.strip())
        merged.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"[{run_name}] merged episodes → {merged}")

    _run_metrics(out_dir)
    report = out_dir / "reports" / "report_cumulative.md"
    print("\n" + "=" * 60)
    print("DONE.")
    print(f"  Baselines: {len(baselines)}")
    print(f"  CSV:    {out_dir}/metrics_per_episode.csv")
    print(f"  Report: {report}")
    print("=" * 60)
