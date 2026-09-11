"""Run several policies on one manifest, then write the metrics table."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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

from traffic_bench.agents.policy_names import canonical_policy_name

IDM_FAMILY = {"idm", "idm_rule"}
NN_NO_CHECKPOINT = {"ppo_rule", "ppo_lidar"}
ALL_POLICIES = IDM_FAMILY | NN_NEED_CHECKPOINT | NN_NO_CHECKPOINT
EGO_VARIANTS = ["default", "s1", "s2", "s3", "s4"]
RUN_EVAL_OUT = "eval_out"


def _is_nn_policy(policy: str) -> bool:
    return policy in NN_NEED_CHECKPOINT


def _cuda_devices_csv(raw) -> str | None:
    """Normalize Hydra list / string ``cuda_devices`` to ``0,1,2``."""
    if raw is None:
        return None
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    if isinstance(raw, (list, tuple)):
        parts = [str(x).strip() for x in raw if str(x).strip()]
        return ",".join(parts) or None
    text = str(raw).strip()
    if not text or text.lower() in {"null", "none", "~"}:
        return None
    # Hydra list printed as ``[0, 1, 2]``
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].replace(" ", "")
        return inner or None
    return text


def _apply_cuda_devices(raw) -> None:
    """Set CUDA_VISIBLE_DEVICES before NN checkpoints load (leave GPU 0 free, etc.)."""
    text = _cuda_devices_csv(raw)
    if not text:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = text
    print(f"[run] CUDA_VISIBLE_DEVICES={text}", flush=True)


def _parse_cuda_device_list(raw) -> list[str]:
    """Physical GPU ids from cfg or existing CUDA_VISIBLE_DEVICES."""
    text = _cuda_devices_csv(raw)
    if not text:
        text = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not text:
        return []
    return [p.strip() for p in text.split(",") if p.strip()]


def _nn_gpu_targets(cuda_device_list: list[str]) -> list[str]:
    """GPU ids for NN workers: numeric CUDA_VISIBLE_DEVICES only (collect.sh style).

    On MLSpace, overriding ``NVIDIA_VISIBLE_DEVICES`` to a single UUID is ignored by
    the container runtime; pairing that with ``CUDA_VISIBLE_DEVICES=0`` dumps every
    worker onto physical GPU0. Keep the cluster-injected NVD list and pin via CVD.
    """
    cvd = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    # Parent (run_signs_parallel) already pinned us to one card.
    if cvd and "," not in cvd and cvd.replace(".", "").isdigit():
        print(f"[run] honor existing CUDA_VISIBLE_DEVICES={cvd}", flush=True)
        return [cvd]
    if not cuda_device_list:
        return []
    # Drop accidental UUID tokens — numeric indices only.
    out = [x for x in cuda_device_list if x.replace(".", "").isdigit()]
    if len(out) != len(cuda_device_list):
        print(
            "[run] warn: ignoring non-numeric cuda_devices entries; "
            "use nvidia-smi indices (collect.sh style)",
            flush=True,
        )
    return out


def _completed_row_indices(final_dir: Path, policy: str, rows: list[dict]) -> set[int]:
    """Rows already present in a serial/merged episodes jsonl (for resume → parallel)."""
    path = final_dir / f"episodes_{policy}.jsonl"
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    from traffic_bench.eval.run.episode import _episode_key_from_result, _episode_key_from_row

    done: set[tuple] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(_episode_key_from_result(json.loads(line)))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
    return {i for i, row in enumerate(rows) if _episode_key_from_row(row) in done}


def _pool_initializer(gpu_ids: list[str], counter, lock) -> None:
    """Pin each process-pool worker to one GPU before tasks run (CPU policies)."""
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PULSE_SERVER", "none")
    if not gpu_ids:
        return
    with lock:
        slot = int(counter.value)
        counter.value = slot + 1
    _pin_job_gpu(gpu_ids[slot % len(gpu_ids)])
    print(
        f"[worker pid={os.getpid()}] CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
        f"NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES')!r}",
        flush=True,
    )


def _pin_job_gpu(gpu: str | None) -> None:
    """Pin to one GPU via CUDA_VISIBLE_DEVICES; leave NVIDIA_VISIBLE_DEVICES alone."""
    if gpu is None:
        return
    text = str(gpu).strip()
    if not text or text.lower() in {"null", "none", "~"}:
        return
    # Never replace cluster NVD with a single UUID — that remaps everyone to GPU0.
    os.environ["CUDA_VISIBLE_DEVICES"] = text
    os.environ["_TRAFFIC_BENCH_GPU_PINNED"] = "1"


def _run_scene_shard(job: dict[str, Any]) -> int:
    """Process/subprocess worker: one shard (possibly many scenes)."""
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PULSE_SERVER", "none")
    if os.environ.get("_TRAFFIC_BENCH_GPU_PINNED") != "1":
        _pin_job_gpu(job.get("cuda_devices"))
    run_episodes(
        policy=str(job["policy"]),
        rows=list(job["rows"]),
        scenes_root=Path(job["scenes_root"]),
        out_dir=Path(job["out_dir"]),
        ego_variant=str(job["ego_variant"]),
        ego_sample_seed_base=int(job["ego_sample_seed_base"]),
        max_steps=int(job["max_steps"]),
        model_path=Path(job["model_path"]) if job.get("model_path") else None,
        plant2_action_mode=str(job["plant2_action_mode"]),
        force_rerun=bool(job["force_rerun"]),
        rerun_failed=bool(job["rerun_failed"]),
        skip_error_episodes=bool(job["skip_error_episodes"]),
        emit_replay_sidecar=bool(job["emit_replay_sidecar"]),
        replay_root=Path(job["replay_root"]) if job.get("replay_root") else None,
        save_gifs=bool(job["save_gifs"]),
        gif_dir=Path(job["gif_dir"]) if job.get("gif_dir") else None,
        gif_window_m=float(job["gif_window_m"]),
        hide_signs=bool(job["hide_signs"]),
        draw_path_conflict=bool(job["draw_path_conflict"]),
        run_name=str(job["run_name"]),
    )
    return int(job["idx"])


def _run_nn_chunks_subprocess(
    tasks: list[tuple[int, Path, dict[str, Any]]],
    *,
    run_name: str,
) -> list[str]:
    """Launch one OS process per chunk with GPU UUID env (collect.sh pattern)."""
    import subprocess

    failures: list[str] = []
    procs: list[tuple[int, Path, subprocess.Popen]] = []
    for idx, shard_out, job in tasks:
        shard_out.mkdir(parents=True, exist_ok=True)
        job_path = shard_out / "job.json"
        job_path.write_text(json.dumps(job, default=str), encoding="utf-8")
        env = os.environ.copy()
        gpu = job.get("cuda_devices")
        if gpu:
            # Numeric CVD only; preserve inherited NVIDIA_VISIBLE_DEVICES.
            env["CUDA_VISIBLE_DEVICES"] = str(gpu).strip()
        env.setdefault("SDL_AUDIODRIVER", "dummy")
        env.setdefault("PULSE_SERVER", "none")
        env["_TRAFFIC_BENCH_GPU_PINNED"] = "1"
        print(
            f"[{run_name}] spawn chunk={shard_out.name} "
            f"scenes={len(job['rows'])} "
            f"CVD={env.get('CUDA_VISIBLE_DEVICES')}",
            flush=True,
        )
        proc = subprocess.Popen(
            [sys.executable, "-m", "traffic_bench.eval.run.shard_worker", str(job_path)],
            env=env,
        )
        procs.append((idx, shard_out, proc))

    for idx, shard_out, proc in procs:
        code = proc.wait()
        if code != 0:
            failures.append(f"chunk {idx} ({shard_out.name}): exit {code}")
        else:
            print(f"[{run_name}] finished {shard_out.name}", flush=True)
    return failures

def _jobs_for_policy(policy: str, *, jobs: int, jobs_nn: int, save_gifs: bool) -> int:
    if save_gifs:
        return 1
    if _is_nn_policy(policy):
        return max(1, int(jobs_nn))
    return max(1, int(jobs))


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
    # Legacy spellings (comprehensive_rule_expert / rule_compliant) → canonical.
    return [canonical_policy_name(p) for p in policies]


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

    # Before torch/CARL touch CUDA — hide GPUs the user wants free (e.g. physical 0).
    _apply_cuda_devices(cfg.get("cuda_devices"))

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
    jobs_nn = max(1, int(cfg.get("jobs_nn") or 1))
    if save_gifs and (jobs > 1 or jobs_nn > 1):
        print("gif.enabled=true: jobs=jobs_nn=1 (Panda3D ShowBase is not process/thread-safe for GIFs)")
        jobs = 1
        jobs_nn = 1
    print(
        f"[run] parallelism: jobs={jobs} (CPU, process pool)  "
        f"jobs_nn={jobs_nn} (carl/plant2)"
    )
    bench_root = out_dir / "benchmark" / "full" / "policy_eval"
    cuda_device_list = _parse_cuda_device_list(cfg.get("cuda_devices"))
    if cuda_device_list:
        print(f"[run] NN GPU pool ({len(cuda_device_list)}): {','.join(cuda_device_list)}")

    def _shard_job(
        *,
        idx: int,
        policy: str,
        variant: str,
        shard_rows: list[dict],
        shard_out: Path,
        worker_slot: int = 0,
    ) -> dict[str, Any]:
        model_path = model_paths.get(policy)
        # Pin each NN worker to one GPU so jobs_nn>1 actually uses the pool.
        if _is_nn_policy(policy) and cuda_device_list:
            cuda_for_job: str | None = cuda_device_list[worker_slot % len(cuda_device_list)]
        elif cuda_device_list:
            cuda_for_job = ",".join(cuda_device_list)
        else:
            cuda_for_job = None
        return {
            "idx": idx,
            "policy": policy,
            "rows": shard_rows,
            "scenes_root": str(scenes_root),
            "out_dir": str(shard_out),
            "ego_variant": variant,
            "ego_sample_seed_base": int(cfg.ego_sample_seed_base),
            "max_steps": int(cfg.max_steps),
            "model_path": str(model_path) if model_path else None,
            "plant2_action_mode": str(cfg.plant2_action_mode),
            "force_rerun": _bool(cfg.force_rerun),
            "rerun_failed": _bool(cfg.rerun_failed),
            "skip_error_episodes": _bool(cfg.skip_error_episodes),
            "emit_replay_sidecar": True,
            "replay_root": str(out_dir / "runs" / "var_0" / f"{policy}_{variant}" / "replays"),
            "save_gifs": save_gifs,
            "gif_dir": str(gif_cfg.dir) if gif_cfg.get("dir") else None,
            "gif_window_m": float(gif_cfg.get("window_m") or 80.0),
            "hide_signs": _bool(cfg.hide_signs),
            "draw_path_conflict": _bool(cfg.draw_path_conflict)
            or _bool(gif_cfg.get("draw_path_conflict")),
            "run_name": f"{policy}_{variant}",
            "cuda_devices": cuda_for_job,
        }

    def _one(policy: str, variant: str, shard_rows: list[dict], shard_out: Path) -> None:
        _run_scene_shard(
            _shard_job(
                idx=0,
                policy=policy,
                variant=variant,
                shard_rows=shard_rows,
                shard_out=shard_out,
                worker_slot=0,
            )
        )

    for policy, variant in baselines:
        run_name = f"{policy}_{variant}"
        final_dir = bench_root / run_name
        policy_jobs = _jobs_for_policy(
            policy, jobs=jobs, jobs_nn=jobs_nn, save_gifs=save_gifs
        )
        if policy_jobs <= 1 or len(rows) <= 1:
            print(f"\n[{run_name}] jobs={policy_jobs} (serial)")
            _one(policy, variant, rows, final_dir)
            continue
        workers = min(policy_jobs, len(rows))
        (out_dir / "_scene_shards" / run_name).mkdir(parents=True, exist_ok=True)
        skip_idx = set()
        if not _bool(cfg.force_rerun):
            skip_idx = _completed_row_indices(final_dir, policy, rows)
            if skip_idx:
                print(
                    f"[{run_name}] resume: skip {len(skip_idx)} episode(s) "
                    f"already in {final_dir / f'episodes_{policy}.jsonl'}"
                )
        pending = [(idx, row) for idx, row in enumerate(rows) if idx not in skip_idx]
        if not pending:
            print(f"\n[{run_name}] nothing to run (all {len(rows)} already done)")
            continue
        is_nn = _is_nn_policy(policy)
        gpu_targets = _nn_gpu_targets(cuda_device_list) if is_nn else []
        workers = min(workers, len(pending))
        # Collect-style: few long-lived workers, each runs many scenes with one
        # model load — not one ProcessPool job per scene.
        chunks: list[list[tuple[int, dict]]] = [[] for _ in range(workers)]
        for i, item in enumerate(pending):
            chunks[i % workers].append(item)
        tasks: list[tuple[int, Path, dict[str, Any]]] = []
        for slot, chunk in enumerate(chunks):
            if not chunk:
                continue
            shard_rows = [row for _, row in chunk]
            shard_id = chunk[0][0]
            shard_out = final_dir / "_shards" / f"w{slot:02d}"
            job = _shard_job(
                idx=shard_id,
                policy=policy,
                variant=variant,
                shard_rows=shard_rows,
                shard_out=shard_out,
                worker_slot=slot,
            )
            if gpu_targets:
                job["cuda_devices"] = gpu_targets[slot % len(gpu_targets)]
            tasks.append((shard_id, shard_out, job))
        print(
            f"\n[{run_name}] parallelizing {len(pending)} scene(s) in "
            f"{len(tasks)} chunk(s), jobs={workers}"
            + (
                f" (subprocess+CVD, GPUs={len(gpu_targets)}: "
                f"{','.join(gpu_targets)})"
                if is_nn and gpu_targets
                else " (ProcessPool spawn)"
            ),
            flush=True,
        )
        if is_nn and gpu_targets:
            failures = _run_nn_chunks_subprocess(tasks, run_name=run_name)
        else:
            failures = []
            done = 0
            ctx = mp.get_context("spawn")
            counter = ctx.Value("i", 0)
            lock = ctx.Lock()
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_pool_initializer,
                initargs=([], counter, lock),
            ) as pool:
                futures = {
                    pool.submit(_run_scene_shard, job): idx for idx, _, job in tasks
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        failures.append(f"chunk {idx}: {exc}")
                    done += 1
                    print(
                        f"[{run_name}] finished chunk {done}/{len(tasks)}",
                        flush=True,
                    )
        if failures:
            raise RuntimeError(f"[{run_name}] scene failures:\n  - " + "\n  - ".join(failures))
        merged = final_dir / f"episodes_{policy}.jsonl"
        lines: list[str] = []
        # Keep previously completed serial/merged rows, then append new shards.
        if merged.is_file() and skip_idx:
            lines.extend(
                ln for ln in merged.read_text(encoding="utf-8").splitlines() if ln.strip()
            )
        for _, shard_out, _ in tasks:
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
