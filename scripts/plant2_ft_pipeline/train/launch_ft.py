#!/usr/bin/env python3
"""Unified PlanT2 FT launcher.

Presets (multi-GPU sweeps):
  spatial-lr     7× LR on GPUs 0-6 via tmux (full spatial split)
  2p5-tsfix      2× LR background jobs (2.5 tsfix cache)
  2p5-stopw      6× stop_weight × LR
  2p5-hyp        7× H1/H2/H1+H2/H5 matrix

Single job: use run_plant2_finetune.py directly, or --job flags below.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lib.env import default_ckpt0, plan_t, pipeline_dir, resolve_python, shepelev
from lib.finetune import FinetuneConfig, lr_tag, run_finetune

SPLIT_SPATIAL = lambda: shepelev() / "plant2_l1_fv_experts_split_signs"
SPLIT_2P5 = lambda: shepelev() / "plant2_l1_fv_experts_split_signs_2.5"


@dataclass
class SweepJob:
    gpu: str
    lr: str
    addon: str
    extra_hydra: list[str] | None = None
    hydra_run_dir: Path | None = None
    stop_weight: float | None = None


def _iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _base_2p5_cfg(job: SweepJob, args: argparse.Namespace) -> FinetuneConfig:
    return FinetuneConfig(
        split=args.split or SPLIT_2P5(),
        learning_rate=job.lr,
        checkpoint_addon=job.addon,
        cuda_device=job.gpu,
        ds_local=args.ds_local or Path("/tmp/plant2_ds_cache_2p5_tsfix"),
        cache_size_gb=args.cache_size_gb,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        augment=True,
        augment_parked=False,
        filter_routes=False,
        stop_speed_loss_weight=job.stop_weight,
        extra_hydra=job.extra_hydra or [],
        hydra_run_dir=job.hydra_run_dir,
        resume_ckpt=args.resume_ckpt or default_ckpt0(),
        python=resolve_python(args.python_exe),
    )


def _run_bg(cfg: FinetuneConfig, log: Path, logdir: Path | None) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(pipeline_dir() / "train" / "run_plant2_finetune.py")]
    cmd.extend([
        "--split", str(cfg.split),
        "--learning-rate", cfg.learning_rate,
        "--checkpoint-addon", cfg.checkpoint_addon,
        "--cuda-device", cfg.cuda_device,
        "--ds-local", str(cfg.ds_local),
        "--cache-size-gb", str(cfg.cache_size_gb),
        "--batch-size", str(cfg.batch_size),
        "--num-workers", str(cfg.num_workers),
        "--max-epochs", str(cfg.max_epochs),
        "--resume-ckpt", str(cfg.resume_ckpt),
        "--log", str(log),
    ])
    if cfg.augment:
        cmd.append("--augment")
    else:
        cmd.append("--no-augment")
    if not cfg.filter_routes:
        cmd.append("--no-filter-routes")
    if cfg.stop_speed_loss_weight is not None:
        cmd.extend(["--stop-speed-loss-weight", str(cfg.stop_speed_loss_weight)])
    for o in cfg.extra_hydra:
        cmd.extend(["--hydra-override", o])
    if cfg.hydra_run_dir:
        cmd.extend(["--hydra-run-dir", str(cfg.hydra_run_dir)])

    with log.open("a") as f:
        f.write(f"FT_START {_iso()} gpu={cfg.cuda_device} lr={cfg.learning_rate} addon={cfg.checkpoint_addon}\n")
    return subprocess.Popen(cmd, stdout=log.open("a"), stderr=subprocess.STDOUT)


def _tmux_session(name: str, inner: str) -> int:
    if subprocess.run(["tmux", "has-session", "-t", name], capture_output=True).returncode == 0:
        print(f"WARN: session {name} exists — skip")
        return 0
    return subprocess.run(["tmux", "new-session", "-d", "-s", name, "bash", "-lc", inner]).returncode


def preset_spatial_lr(args: argparse.Namespace) -> int:
    split = args.split or SPLIT_SPATIAL()
    ds_local = args.ds_local or Path("/tmp/plant2_ds_cache_spatial_aug")
    jobs = [
        SweepJob("0", "1e-6", "fvexp30_spatial_lr1e6"),
        SweepJob("1", "5e-6", "fvexp30_spatial_lr5e6"),
        SweepJob("2", "1e-5", "fvexp30_spatial_lr1e5"),
        SweepJob("3", "3e-5", "fvexp30_spatial_lr3e5"),
        SweepJob("4", "5e-5", "fvexp30_spatial_lr5e5"),
        SweepJob("5", "7e-5", "fvexp30_spatial_lr7e5"),
        SweepJob("6", "1e-4", "fvexp30_spatial_lr1e4"),
    ]
    run_sh = pipeline_dir() / "train" / "run_plant2_finetune.py"
    py = resolve_python(args.python_exe)
    pt = plan_t()
    fail = 0
    for job in jobs:
        tag = lr_tag(job.lr)
        session = f"arbelyaev-ft-spatial-lr{tag}"
        log = Path(f"/tmp/plant2_ft_spatial_lr{tag}.log")
        inner = f"""
set -euo pipefail
cd '{pt}'
export CUDA_VISIBLE_DEVICES={job.gpu}
echo "FT_START $(date -Is)" | tee -a '{log}'
'{py}' -u '{run_sh}' \\
  --split '{split}' \\
  --learning-rate {job.lr} \\
  --checkpoint-addon {job.addon} \\
  --cuda-device {job.gpu} \\
  --ds-local '{ds_local}' \\
  --cache-size-gb {args.cache_size_gb} \\
  --batch-size {args.batch_size} \\
  --max-epochs {args.max_epochs} \\
  --num-workers {args.num_workers} \\
  --augment --no-filter-routes \\
  --log '{log}'
echo "FT_EXIT=$? $(date -Is)" | tee -a '{log}'
exec bash
"""
        fail += _tmux_session(session, inner) != 0
        print(f"started {session} gpu={job.gpu} lr={job.lr}")
    return fail


def preset_2p5_tsfix(args: argparse.Namespace) -> int:
    jobs = [
        SweepJob("0", "1e-4", "fvexp30_spatial_2p5_tsfix_lr1e4"),
        SweepJob("1", "1e-5", "fvexp30_spatial_2p5_tsfix_lr1e5"),
    ]
    logdir = pipeline_dir() / "logs_pipeline_2p5_tsfix"
    logdir.mkdir(parents=True, exist_ok=True)
    procs = []
    for job in jobs:
        cfg = _base_2p5_cfg(job, args)
        log = Path(f"/tmp/plant2_ft_2p5_tsfix_lr{lr_tag(job.lr)}.log")
        procs.append(_run_bg(cfg, log, logdir))
        print(f"started gpu={job.gpu} lr={job.lr} pid={procs[-1].pid}")
    if args.wait:
        return sum(p.wait() for p in procs)
    return 0


def preset_2p5_stopw(args: argparse.Namespace) -> int:
    jobs = [
        SweepJob("0", "1e-4", "fvexp30_2p5_stopw5_lr1e4", stop_weight=5),
        SweepJob("1", "1e-5", "fvexp30_2p5_stopw5_lr1e5", stop_weight=5),
        SweepJob("2", "1e-4", "fvexp30_2p5_stopw10_lr1e4", stop_weight=10),
        SweepJob("3", "1e-5", "fvexp30_2p5_stopw10_lr1e5", stop_weight=10),
        SweepJob("4", "1e-4", "fvexp30_2p5_stopw20_lr1e4", stop_weight=20),
        SweepJob("5", "1e-5", "fvexp30_2p5_stopw20_lr1e5", stop_weight=20),
    ]
    logdir = pipeline_dir() / "logs_pipeline_2p5_stopw"
    procs = []
    for job in jobs:
        cfg = _base_2p5_cfg(job, args)
        sw = job.stop_weight
        log = logdir / f"ft_stopw{sw}_lr{lr_tag(job.lr)}.log"
        procs.append(_run_bg(cfg, log, logdir))
    if args.wait:
        return sum(p.wait() for p in procs)
    return 0


def preset_2p5_hyp(args: argparse.Namespace) -> int:
    h1 = [
        "model.waypoints.path_weight=0",
        "model.pre_training.forecastLoss_weight=0",
        "model.waypoints.speed_weight=5",
        "model.training.augment=True",
    ]
    h2 = ["model.training.speed_class_weights=[15,1,1,1,1,1,1,1]", "model.training.augment=True"]
    h1h2 = h1[:-1] + [h2[0], h2[1]]
    jobs = [
        SweepJob("0", "1e-4", "fvexp30_2p5_h1_path0_sw5_lr1e4", extra_hydra=h1),
        SweepJob("1", "1e-5", "fvexp30_2p5_h1_path0_sw5_lr1e5", extra_hydra=h1),
        SweepJob("2", "1e-4", "fvexp30_2p5_h2_cw15_lr1e4", extra_hydra=h2),
        SweepJob("3", "1e-5", "fvexp30_2p5_h2_cw15_lr1e5", extra_hydra=h2),
        SweepJob("4", "1e-4", "fvexp30_2p5_h1h2_path0_sw5_cw15_lr1e4", extra_hydra=h1h2),
        SweepJob("5", "1e-5", "fvexp30_2p5_h1h2_path0_sw5_cw15_lr1e5", extra_hydra=h1h2),
        SweepJob("6", "1e-5", "fvexp30_2p5_h5_noaug_lr1e5", extra_hydra=["model.training.augment=False"]),
    ]
    logdir = pipeline_dir() / "logs_pipeline_2p5_hyp"
    procs = []
    for job in jobs:
        cfg = _base_2p5_cfg(job, args)
        if job.extra_hydra and "model.training.augment=False" in job.extra_hydra:
            cfg.augment = False
        run_dir = plan_t() / "outputs/PlanT2_train" / job.addon
        cfg.hydra_run_dir = run_dir
        log = logdir / f"ft_{job.addon}.log"
        procs.append(_run_bg(cfg, log, logdir))
    if args.wait:
        return sum(p.wait() for p in procs)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "preset",
        choices=("spatial-lr", "2p5-tsfix", "2p5-stopw", "2p5-hyp"),
        help="FT sweep preset",
    )
    p.add_argument("--split", type=Path, default=None)
    p.add_argument("--ds-local", type=Path, default=None)
    p.add_argument("--cache-size-gb", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=1344)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--resume-ckpt", type=Path, default=None)
    p.add_argument("--python", dest="python_exe", default=None)
    p.add_argument("--wait", action="store_true", help="Wait for all background jobs")
    args = p.parse_args()

    if args.preset == "spatial-lr":
        if args.ds_local is None:
            args.ds_local = Path("/tmp/plant2_ds_cache_spatial_aug")
        if args.cache_size_gb == 400:  # default for 2p5 presets
            args.cache_size_gb = 1800
        return preset_spatial_lr(args)

    if args.preset == "2p5-tsfix":
        if args.ds_local is None:
            args.ds_local = Path("/tmp/plant2_ds_cache_2p5_tsfix")
        return preset_2p5_tsfix(args)
    if args.preset == "2p5-stopw":
        if args.ds_local is None:
            args.ds_local = Path("/tmp/plant2_ds_cache_2p5_tsfix")
        return preset_2p5_stopw(args)
    if args.preset == "2p5-hyp":
        if args.ds_local is None:
            args.ds_local = Path("/tmp/plant2_ds_cache_2p5_tsfix")
        return preset_2p5_hyp(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
