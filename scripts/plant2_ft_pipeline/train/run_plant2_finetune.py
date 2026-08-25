#!/usr/bin/env python3
"""Fine-tune PlanT2 on an explicit train/val SPLIT."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import os
from pathlib import Path

from lib.env import plan_t, resolve_python, shepelev
from lib.finetune import FinetuneConfig, run_finetune


def _env(name: str, default):
    """Env fallback for a flag.

    The pipeline runners (run_fix_pipeline.sh, run_sign_pair_experiment.sh)
    drive a whole campaign through the environment; keeping that contract lets
    them stay as they are while the flags cover interactive use.
    """
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else raw


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip() in ("1", "true", "True", "yes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, default=_env("SPLIT", None),
                   help="Split root (must contain split_meta.json)")
    p.add_argument("--learning-rate", default=_env("LEARNING_RATE", "1e-4"))
    p.add_argument("--checkpoint-addon", default=_env("CHECKPOINT_ADDON", "arbelyaev_ft5"))
    p.add_argument("--cuda-device", default=_env("CUDA_VISIBLE_DEVICES", "0"))
    p.add_argument("--ds", type=Path, default=_env("DS", None))
    p.add_argument("--ds-val", type=Path, default=_env("DS_VAL", None))
    p.add_argument("--ds-local", type=Path, default=_env("DS_LOCAL", None))
    p.add_argument("--cache-size-gb", type=int, default=int(_env("CACHE_SIZE_GB", 641)))
    p.add_argument("--batch-size", type=int, default=int(_env("BATCH_SIZE", 1536)))
    p.add_argument("--num-workers", type=int, default=int(_env("NUM_WORKERS", 4)))
    p.add_argument("--max-epochs", type=int, default=int(_env("MAX_EPOCHS", 30)))
    p.add_argument("--ckpt-every-n-epochs", type=int, default=int(_env("CKPT_EVERY_N_EPOCHS", 5)))
    p.add_argument("--lr-scheduler", default=_env("LR_SCHEDULER", "cosine_warmup"),
                   choices=("multistep", "cosine_warmup"))
    p.add_argument("--warmup-ratio", type=float, default=float(_env("WARMUP_RATIO", 0.1)))
    p.add_argument("--seed", type=int, default=int(_env("SEED", 1)))
    p.add_argument("--resume-ckpt", type=Path, default=None)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--augment-parked", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--filter-routes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stop-speed-loss-weight", type=float,
                   default=float(_env("STOP_SPEED_LOSS_WEIGHT", 1.0)))
    p.add_argument("--gpus", type=int, default=int(_env("GPUS", 1)))
    p.add_argument("--ddp-strategy", default=_env("DDP_STRATEGY", "ddp_find_unused_parameters_true"))
    p.add_argument("--custom-sampler", action=argparse.BooleanOptionalAction,
                   default=_env_bool("CUSTOM_SAMPLER", False),
                   help="Draw frames by sample_weights.json instead of uniformly")
    p.add_argument("--init-sign-from-stop", action=argparse.BooleanOptionalAction,
                   default=_env_bool("INIT_SIGN_FROM_STOP", False),
                   help="Seed the PDD sign tok_emb from the trained stop_sign layer")
    p.add_argument("--new-param-lr-mult", type=float, default=float(_env("NEW_PARAM_LR_MULT", 1.0)),
                   help="LR multiplier for the parameters the checkpoint does not carry")
    p.add_argument("--trunk-lr-mult", type=float, default=float(_env("TRUNK_LR_MULT", 1.0)),
                   help="0 freezes the pretrained trunk")
    p.add_argument("--working-dir", type=Path, default=None,
                   help="user.working_dir for Hydra (default: this checkout's plant2)")
    p.add_argument("--python", dest="python_exe", default=None)
    p.add_argument("--wandb-mode", default="offline")
    p.add_argument("--log", type=Path, default=None, help="Append stdout to this log file")
    p.add_argument(
        "--hydra-override",
        action="append",
        default=[o for o in os.environ.get("HYDRA_OVERRIDES", "").split(";") if o],
        help="Extra Hydra overrides (repeatable); HYDRA_OVERRIDES seeds it, "
             "semicolon-separated, so a campaign script can pass them like "
             "every other setting",
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
        gpus=args.gpus,
        ddp_strategy=args.ddp_strategy,
        custom_sampler=args.custom_sampler,
        init_sign_from_stop=args.init_sign_from_stop,
        new_param_lr_mult=args.new_param_lr_mult,
        trunk_lr_mult=args.trunk_lr_mult,
        working_dir=args.working_dir,
        extra_hydra=args.hydra_override,
        hydra_run_dir=args.hydra_run_dir,
        python=resolve_python(args.python_exe),
        wandb_mode=args.wandb_mode,
    )
    raise SystemExit(run_finetune(cfg, cwd=plan_t(), log_path=args.log))


if __name__ == "__main__":
    main()
