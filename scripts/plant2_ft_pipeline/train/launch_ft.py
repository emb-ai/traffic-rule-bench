#!/usr/bin/env python3
"""Sweep-spec-driven PlanT2 FT launcher.

One job = one FT run (one GPU, one LR, optional Hydra overrides). A sweep is a
YAML/JSON file with a `defaults` dict applied to every job and a `jobs` list.
See train/sweeps/*.yaml for the sweeps this replaced (spatial-lr, 2p5-tsfix,
2p5-stopw, 2p5-hyp) as data instead of one hardcoded Python function each.

Usage::

  python launch_ft.py --spec train/sweeps/2p5_tsfix.yaml --wait
  python launch_ft.py --spec train/sweeps/spatial_lr.yaml   # tmux, doesn't block

Spec format::

  defaults:
    split: ${SHEPELEV}/plant2_l1_fv_experts_split_signs_2.5
    ds_local: /tmp/plant2_ds_cache_2p5_tsfix
    cache_size_gb: 400
    launch: bg          # bg (background subprocess) | tmux
  jobs:
    - {gpu: "0", lr: "1e-4", addon: my_experiment_lr1e4}
    - {gpu: "1", lr: "1e-5", addon: my_experiment_lr1e5, stop_weight: 10}
    - {gpu: "2", lr: "1e-5", addon: my_experiment_h1, extra_hydra: [model.waypoints.path_weight=0]}

`${SHEPELEV}` / `${TRB_ROOT}` in string values are expanded. Any FinetuneConfig
field can be set per-job or in `defaults` (gpu/lr/addon are per-job only).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import subprocess
from datetime import datetime, timezone

import yaml

from lib.env import default_ckpt0, plan_t, pipeline_dir, resolve_python, shepelev, trb_root
from lib.finetune import FinetuneConfig, lr_tag


def _iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _expand(value: str) -> str:
    return str(value).replace("${SHEPELEV}", str(shepelev())).replace("${TRB_ROOT}", str(trb_root()))


def _load_spec(path: Path) -> dict:
    text = path.read_text()
    return yaml.safe_load(text) if path.suffix in (".yaml", ".yml") else json.loads(text)


def _pick(merged: dict, key: str, args_val, fallback):
    if merged.get(key) is not None:
        return merged[key]
    if args_val is not None:
        return args_val
    return fallback


def _job_cfg(job: dict, defaults: dict, args: argparse.Namespace) -> FinetuneConfig:
    merged = {**defaults, **job}
    for required in ("gpu", "lr", "addon", "split"):
        if not merged.get(required):
            raise SystemExit(f"ERROR: job missing required field {required!r}: {job}")

    resume_ckpt = merged.get("resume_ckpt")
    hydra_run_dir = merged.get("hydra_run_dir")
    return FinetuneConfig(
        split=Path(_expand(merged["split"])),
        learning_rate=str(merged["lr"]),
        checkpoint_addon=merged["addon"],
        cuda_device=str(merged["gpu"]),
        ds_local=Path(_expand(merged["ds_local"])) if merged.get("ds_local") else None,
        cache_size_gb=int(_pick(merged, "cache_size_gb", args.cache_size_gb, 1800)),
        batch_size=int(_pick(merged, "batch_size", args.batch_size, 1344)),
        num_workers=int(_pick(merged, "num_workers", args.num_workers, 4)),
        max_epochs=int(_pick(merged, "max_epochs", args.max_epochs, 30)),
        augment=bool(merged.get("augment", True)),
        augment_parked=bool(merged.get("augment_parked", False)),
        filter_routes=bool(merged.get("filter_routes", True)),
        stop_speed_loss_weight=merged.get("stop_weight"),
        extra_hydra=list(merged.get("extra_hydra") or []),
        hydra_run_dir=Path(_expand(hydra_run_dir)) if hydra_run_dir else None,
        resume_ckpt=Path(_expand(resume_ckpt)) if resume_ckpt else (args.resume_ckpt or default_ckpt0()),
        python=resolve_python(args.python_exe),
    )


def _run_sh_args(cfg: FinetuneConfig, log: Path) -> list[str]:
    argv = [
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
        "--augment" if cfg.augment else "--no-augment",
        "--filter-routes" if cfg.filter_routes else "--no-filter-routes",
        "--log", str(log),
    ]
    if cfg.stop_speed_loss_weight is not None:
        argv.extend(["--stop-speed-loss-weight", str(cfg.stop_speed_loss_weight)])
    for o in cfg.extra_hydra:
        argv.extend(["--hydra-override", o])
    if cfg.hydra_run_dir:
        argv.extend(["--hydra-run-dir", str(cfg.hydra_run_dir)])
    return argv


def _run_bg(cfg: FinetuneConfig, log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    run_py = pipeline_dir() / "train" / "run_plant2_finetune.py"
    cmd = [str(cfg.python), "-u", str(run_py), *_run_sh_args(cfg, log)]
    with log.open("a") as f:
        f.write(f"FT_START {_iso()} gpu={cfg.cuda_device} lr={cfg.learning_rate} addon={cfg.checkpoint_addon}\n")
    return subprocess.Popen(cmd, stdout=log.open("a"), stderr=subprocess.STDOUT)


def _run_tmux(cfg: FinetuneConfig, session: str, log: Path) -> int:
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        print(f"WARN: tmux session {session} exists — skip")
        return 0
    log.parent.mkdir(parents=True, exist_ok=True)
    run_py = pipeline_dir() / "train" / "run_plant2_finetune.py"
    cmd_line = " ".join(f"'{a}'" for a in [str(cfg.python), "-u", str(run_py), *_run_sh_args(cfg, log)])
    inner = f"""
set -euo pipefail
cd '{plan_t()}'
export CUDA_VISIBLE_DEVICES={cfg.cuda_device}
echo "FT_START {_iso()}" | tee -a '{log}'
{cmd_line}
echo "FT_EXIT=$? $(date -Is)" | tee -a '{log}'
exec bash
"""
    return subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash", "-lc", inner]).returncode


def cmd_sweep(args: argparse.Namespace) -> int:
    spec = _load_spec(args.spec)
    defaults = spec.get("defaults", {})
    jobs = spec.get("jobs") or []
    if not jobs:
        raise SystemExit(f"ERROR: no jobs in {args.spec}")

    log_dir = Path(_expand(defaults.get("log_dir") or f"/tmp/plant2_ft_{args.spec.stem}"))
    procs: list[subprocess.Popen] = []
    fail = 0

    for job in jobs:
        cfg = _job_cfg(job, defaults, args)
        merged = {**defaults, **job}
        log = log_dir / f"ft_{cfg.checkpoint_addon}.log"
        if merged.get("launch", "bg") == "tmux":
            prefix = merged.get("tmux_prefix", "ft")
            session = f"{prefix}-{cfg.checkpoint_addon}"
            fail += _run_tmux(cfg, session, log) != 0
            print(f"[tmux] {session} gpu={cfg.cuda_device} lr={cfg.learning_rate} addon={cfg.checkpoint_addon}")
        else:
            procs.append(_run_bg(cfg, log))
            print(f"[bg] pid={procs[-1].pid} gpu={cfg.cuda_device} lr={cfg.learning_rate} addon={cfg.checkpoint_addon}")

    if args.wait and procs:
        fail += sum(1 for p in procs if p.wait() != 0)
    return fail


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", type=Path, required=True, help="Sweep YAML/JSON (see train/sweeps/*.yaml)")
    p.add_argument("--cache-size-gb", type=int, default=None, help="Fallback if spec job/defaults omit it")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--resume-ckpt", type=Path, default=None)
    p.add_argument("--python", dest="python_exe", default=None)
    p.add_argument("--wait", action="store_true", help="Wait for all background (non-tmux) jobs")
    args = p.parse_args()
    return cmd_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
