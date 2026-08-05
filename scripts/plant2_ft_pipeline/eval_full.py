#!/usr/bin/env python3
"""Full plant2-ft eval orchestrator.

Subcommands:
  fv           FV-fast eval for one checkpoint
  queue        Parallel eval queue (signs + fv_fast + fv_detour)
  spatial      7-GPU spatial FT eval waves (signs + fv ×2 per LR×epoch slot)
"""
from __future__ import annotations

import argparse
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from _env import metrics_root, plan_t, shepelev, setup_eval_thread_env
from _eval import (
    DEFAULT_MANIFEST,
    DEFAULT_MANIFEST_DETOUR,
    DEFAULT_SCENES,
    DEFAULT_SCENES_DETOUR,
    SignsEvalConfig,
    fv_done,
    make_spatial_tag,
    resolve_ckpt_spatial,
    run_fv_fast,
    run_signs_eval,
    setup_metrics_tag,
    signs_done,
    tag_from_ckpt,
)

SPATIAL_LRS = ["1e6", "5e6", "1e5", "3e5", "5e5", "7e5", "1e4"]
DEFAULT_SLOTS = ["best", "ep029", "ep024", "ep019", "ep014", "ep009", "ep004"]


def _build_queue_ckpts(ckpt_root: Path) -> list[Path]:
    ckpts: list[Path] = []
    for lr in ("1e4", "1e5", "3e5", "7e5"):
        for addon_suffix, files in (
            (f"fvexp30_sign_lr{lr}", ["best_002", "last_ft"]),
            (f"fvexp30_lr{lr}", ["best_000", f"epoch=029"]),
        ):
            d = ckpt_root / addon_suffix
            for stem in files:
                if stem.startswith("epoch"):
                    hits = list(d.glob(f"{stem}_*.ckpt"))
                else:
                    hits = list(d.glob(f"{stem}_*.ckpt"))
                ckpts.extend(hits)
    return ckpts


def cmd_fv(args: argparse.Namespace) -> int:
    return run_fv_fast(
        ckpt=args.ckpt,
        out=args.out,
        manifest=args.manifest,
        scenes=args.scenes,
        gpus=args.gpus.split(),
        nshards=args.nshards,
        concurrency=args.concurrency,
        exclude_codes=args.exclude_codes.split() if args.exclude_codes else None,
    )


def _eval_one_ckpt_full(
    ckpt: Path,
    tag: str,
    metrics: Path,
    *,
    gpu: str,
    signs_jobs: int,
    scenes_per_job: int,
    fv_gpus: list[str],
    fv_nshards: int,
    fv_concurrency: int,
    manifest_v61: Path,
    scenes_v61: Path,
    manifest_detour: Path,
    scenes_detour: Path,
    skip_fv: bool,
) -> int:
    setup_metrics_tag(metrics, tag, ckpt)
    rc = 0
    if not signs_done(tag):
        rc |= run_signs_eval(SignsEvalConfig(
            ckpt=ckpt, tag=tag, gpu=gpu,
            jobs=signs_jobs, scenes_per_job=scenes_per_job,
            metrics_root=metrics, only_signs=None,
        ))
    if skip_fv:
        return rc
    fv_out = metrics / tag / "fv_fast"
    if not fv_done(fv_out):
        rc |= run_fv_fast(
            ckpt=ckpt, out=fv_out, manifest=manifest_v61, scenes=scenes_v61,
            gpus=fv_gpus, nshards=fv_nshards, concurrency=fv_concurrency,
        )
    det_out = metrics / tag / "fv_fast_detour"
    if not fv_done(det_out):
        rc |= run_fv_fast(
            ckpt=ckpt, out=det_out, manifest=manifest_detour, scenes=scenes_detour,
            gpus=fv_gpus, nshards=fv_nshards, concurrency=fv_concurrency,
        )
    return rc


def cmd_queue(args: argparse.Namespace) -> int:
    ckpt_root = plan_t() / "checkpoints_ft"
    metrics = args.metrics_root
    metrics.mkdir(parents=True, exist_ok=True)
    ckpts = _build_queue_ckpts(ckpt_root)
    print(f"queue: {len(ckpts)} checkpoints")

    indexed: list[tuple[str, Path]] = []
    for ckpt in ckpts:
        tag = tag_from_ckpt(ckpt)
        if tag:
            indexed.append((tag, ckpt))

    wave1 = indexed[:8]
    wave2 = indexed[8:]
    fail = 0
    gpus = args.gpus.split()

    for wave_i, wave in enumerate((wave1, wave2), 1):
        print(f"=== SIGNS WAVE {wave_i} ({len(wave)} ckpts) ===")
        with ProcessPoolExecutor(max_workers=args.signs_parallel) as ex:
            futs = []
            for i, (tag, ckpt) in enumerate(wave):
                gpu = gpus[i % len(gpus)]
                futs.append(ex.submit(
                    run_signs_eval,
                    SignsEvalConfig(
                        ckpt=ckpt, tag=tag, gpu=gpu,
                        jobs=args.signs_jobs, scenes_per_job=args.scenes_per_job,
                        metrics_root=metrics,
                    ),
                ))
            for fut in as_completed(futs):
                if fut.result() != 0:
                    fail += 1

        print(f"=== FV WAVE {wave_i} ===")

        def _fv_only(ckpt: Path, tag: str) -> int:
            setup_metrics_tag(metrics, tag, ckpt)
            rc = 0
            fv_out = metrics / tag / "fv_fast"
            if not fv_done(fv_out):
                rc |= run_fv_fast(
                    ckpt=ckpt, out=fv_out, manifest=args.manifest, scenes=args.scenes,
                    gpus=gpus, nshards=args.fv_nshards, concurrency=args.fv_concurrency,
                )
            det_out = metrics / tag / "fv_fast_detour"
            if not fv_done(det_out):
                rc |= run_fv_fast(
                    ckpt=ckpt, out=det_out, manifest=args.manifest_detour, scenes=args.scenes_detour,
                    gpus=gpus, nshards=args.fv_nshards, concurrency=args.fv_concurrency,
                )
            return rc

        with ProcessPoolExecutor(max_workers=args.fv_parallel) as ex:
            futs = [ex.submit(_fv_only, ckpt, tag) for tag, ckpt in wave]
            for fut in as_completed(futs):
                if fut.result() != 0:
                    fail += 1

    print(f"QUEUE DONE fail={fail}")
    return fail


def _wait_spatial_ft(lrs: list[str]) -> None:
    ckpt_root = plan_t() / "checkpoints_ft"
    while True:
        missing = sum(
            1 for lr in lrs
            if not (ckpt_root / f"fvexp30_spatial_lr{lr}" / f"epoch=029_fvexp30_spatial_lr{lr}_1.ckpt").is_file()
        )
        tmux = subprocess.run(["tmux", "ls"], capture_output=True, text=True)
        alive = sum(1 for ln in (tmux.stdout or "").splitlines() if "arbelyaev-ft-spatial-lr" in ln)
        print(f"wait FT: missing_ep029={missing} tmux={alive}")
        if missing == 0 and alive == 0:
            return
        time.sleep(120)


def cmd_spatial(args: argparse.Namespace) -> int:
    setup_eval_thread_env()
    metrics = args.metrics_root
    metrics.mkdir(parents=True, exist_ok=True)
    slots = args.slots.split() if args.slots else DEFAULT_SLOTS
    lrs = args.lrs.split() if args.lrs else SPATIAL_LRS

    if not args.skip_wait_ft:
        _wait_spatial_ft(lrs)

    fail = 0
    for slot in slots:
        print(f"======== WAVE slot={slot} ========")
        jobs: list[tuple[int, str, str, Path, str]] = []
        for i, lr in enumerate(lrs):
            ckpt = resolve_ckpt_spatial(lr, slot)
            if ckpt is None:
                print(f"SKIP lr={lr} slot={slot}")
                continue
            tag = make_spatial_tag(lr, slot, ckpt, suffix=args.tag_suffix)
            jobs.append((i, lr, slot, ckpt, tag))

        with ProcessPoolExecutor(max_workers=len(jobs) or 1) as ex:
            futs = [
                ex.submit(
                    _eval_one_ckpt_full, ckpt, tag, metrics,
                    gpu=str(gpu_i), signs_jobs=args.signs_jobs, scenes_per_job=args.scenes_per_job,
                    fv_gpus=[str(gpu_i)], fv_nshards=args.fv_nshards_per_gpu,
                    fv_concurrency=args.fv_concurrency_per_gpu,
                    manifest_v61=args.manifest, scenes_v61=args.scenes,
                    manifest_detour=args.manifest_detour, scenes_detour=args.scenes_detour,
                    skip_fv=False,
                )
                for gpu_i, lr, slot, ckpt, tag in jobs
            ]
            for fut in as_completed(futs):
                if fut.result() != 0:
                    fail += 1
    print(f"SPATIAL DONE fail={fail}")
    return fail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    fv = sub.add_parser("fv", help="FV-fast for one ckpt")
    fv.add_argument("--ckpt", type=Path, required=True)
    fv.add_argument("--out", type=Path, required=True)
    fv.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    fv.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    fv.add_argument("--gpus", default="0 1 2 3 4 5 6")
    fv.add_argument("--nshards", type=int, default=28)
    fv.add_argument("--concurrency", type=int, default=28)
    fv.add_argument("--exclude-codes", default="3.25 5.22 5.32")

    q = sub.add_parser("queue", help="Parallel eval queue")
    q.add_argument("--metrics-root", type=Path, default=shepelev() / "plant2_ft_metrics")
    q.add_argument("--gpus", default="0 1 2 3 4 5 6")
    q.add_argument("--signs-parallel", type=int, default=8)
    q.add_argument("--signs-jobs", type=int, default=20)
    q.add_argument("--scenes-per-job", type=int, default=32)
    q.add_argument("--fv-parallel", type=int, default=4)
    q.add_argument("--fv-nshards", type=int, default=28)
    q.add_argument("--fv-concurrency", type=int, default=28)
    q.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    q.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    q.add_argument("--manifest-detour", type=Path, default=DEFAULT_MANIFEST_DETOUR)
    q.add_argument("--scenes-detour", type=Path, default=DEFAULT_SCENES_DETOUR)

    sp = sub.add_parser("spatial", help="7-GPU spatial FT eval waves")
    sp.add_argument("--metrics-root", type=Path, default=shepelev() / "plant2_ft_metrics/spatial_signs_eval")
    sp.add_argument("--tag-suffix", default="")
    sp.add_argument("--slots", default=None, help="Space-separated, default: best ep029 ...")
    sp.add_argument("--lrs", default=None)
    sp.add_argument("--skip-wait-ft", action="store_true")
    sp.add_argument("--signs-jobs", type=int, default=20)
    sp.add_argument("--scenes-per-job", type=int, default=32)
    sp.add_argument("--fv-nshards-per-gpu", type=int, default=8)
    sp.add_argument("--fv-concurrency-per-gpu", type=int, default=8)
    sp.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    sp.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    sp.add_argument("--manifest-detour", type=Path, default=DEFAULT_MANIFEST_DETOUR)
    sp.add_argument("--scenes-detour", type=Path, default=DEFAULT_SCENES_DETOUR)

    args = parser.parse_args()
    handlers = {"fv": cmd_fv, "queue": cmd_queue, "spatial": cmd_spatial}
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
