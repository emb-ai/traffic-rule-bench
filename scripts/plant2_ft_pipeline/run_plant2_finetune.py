#!/usr/bin/env python3
"""Fine-tune PlanT2 on an explicit train/val SPLIT."""
from __future__ import annotations

import argparse
from pathlib import Path

from _env import plan_t, resolve_python, shepelev
from _finetune import FinetuneConfig, run_finetune


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, required=True, help="Split root (must contain split_meta.json)")
    p.add_argument("--learning-rate", default="1e-4")
    p.add_argument("--checkpoint-addon", default="arbelyaev_ft5")
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--ds", type=Path, default=None)
    p.add_argument("--ds-val", type=Path, default=None)
    p.add_argument("--ds-local", type=Path, default=None)
    p.add_argument("--cache-size-gb", type=int, default=641)
    p.add_argument("--batch-size", type=int, default=1536)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--ckpt-every-n-epochs", type=int, default=5)
    p.add_argument("--lr-scheduler", default="cosine_warmup", choices=("multistep", "cosine_warmup"))
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume-ckpt", type=Path, default=None)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--augment-parked", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--filter-routes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stop-speed-loss-weight", type=float, default=None)
    p.add_argument("--python", dest="python_exe", default=None)
    p.add_argument("--wandb-mode", default="offline")
    p.add_argument("--log", type=Path, default=None, help="Append stdout to this log file")
    p.add_argument(
        "--hydra-override",
        action="append",
        default=[],
        help="Extra Hydra overrides (repeatable)",
    )
    p.add_argument("--hydra-run-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.split:
        raise SystemExit(
            f"ERROR: --split required (e.g. {shepelev() / 'plant2_l1_fv_experts_split'})"
        )
    cfg = FinetuneConfig(
        split=args.split,
        learning_rate=args.learning_rate,
        checkpoint_addon=args.checkpoint_addon,
        cuda_device=args.cuda_device,
        ds=args.ds,
        ds_val=args.ds_val,
        ds_local=args.ds_local,
        cache_size_gb=args.cache_size_gb,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        ckpt_every_n_epochs=args.ckpt_every_n_epochs,
        lr_scheduler=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        resume_ckpt=args.resume_ckpt,
        augment=args.augment,
        augment_parked=args.augment_parked,
        filter_routes=args.filter_routes,
        stop_speed_loss_weight=args.stop_speed_loss_weight,
        extra_hydra=args.hydra_override,
        hydra_run_dir=args.hydra_run_dir,
        python=resolve_python(args.python_exe),
        wandb_mode=args.wandb_mode,
    )
    raise SystemExit(run_finetune(cfg, cwd=plan_t(), log_path=args.log))


if __name__ == "__main__":
    main()
