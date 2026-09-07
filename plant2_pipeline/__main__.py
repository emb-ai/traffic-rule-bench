"""Command line for the dump -> split -> train -> eval loop.

    python -m plant2_pipeline all --name joint --families stop detour speed --gpu 0

Stage order: plan, dump, split, prefill, train, eval. Each can be run alone and
in any order once its inputs exist:

    python -m plant2_pipeline plan    --name joint --families stop detour speed
    python -m plant2_pipeline dump    --name joint
    python -m plant2_pipeline split   --name joint
    python -m plant2_pipeline prefill --name joint
    python -m plant2_pipeline train   --name joint --gpu 0
    python -m plant2_pipeline eval    --name joint --gpu 0
    python -m plant2_pipeline report  --name joint

`--families` takes group names (stop / detour / speed) or individual family
names; groups expand to their families.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import config as cfg
from . import metrics, stages

STAGES = ("plan", "dump", "split", "prefill", "train", "eval", "report", "all")
IN_ORDER = ("plan", "dump", "split", "prefill", "train", "eval")


def build_experiment(args: argparse.Namespace) -> cfg.Experiment:
    window = None if args.no_oversample else (args.oversample_from, args.oversample_to)
    return cfg.Experiment(
        name=args.name,
        families=cfg.resolve_families(args.families),
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        split_seed=args.seed,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        augment=not args.no_augment,
        oversample_cone_m=window,
        oversample_factor=args.oversample_factor,
        hydra_overrides=tuple(args.override),
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="plant2_pipeline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--name", required=True, help="Experiment name; also the run directory")
    p.add_argument("--gpu", default="0")
    p.add_argument("--families", nargs="+", default=["stop", "detour", "speed"])
    p.add_argument("--test-fraction", type=float, default=0.2,
                   help="Share of scenes held out of training for the metrics")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", default="3e-4")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--no-augment", action="store_true",
                   help="Disable pose jitter. On by default: it cut off-road "
                        "runs from 0.75 to 0.16.")
    p.add_argument("--no-oversample", action="store_true")
    p.add_argument("--oversample-from", type=float, default=40.0)
    p.add_argument("--oversample-to", type=float, default=70.0)
    p.add_argument("--oversample-factor", type=int, default=8)
    p.add_argument("--override", action="append", default=[],
                   help="Extra Hydra override, repeatable")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Evaluate this checkpoint instead of the run's own last_ft_*")
    p.add_argument("--dump-workers", type=int, default=24,
                   help="Parallel dump shards per family")
    p.add_argument("--prefill-workers", type=int, default=32)
    p.add_argument("--eval-shards", type=int, default=4)
    p.add_argument("--no-wait", action="store_true",
                   help="Start training detached and return immediately")
    args = p.parse_args()

    exp = build_experiment(args)
    cfg.check_inputs(exp)
    print(cfg.describe(exp))

    if args.stage == "report":
        print(metrics.report(exp))
        return

    todo = IN_ORDER if args.stage == "all" else (args.stage,)
    for stage in todo:
        if stage == "plan":
            stages.plan(exp)
        elif stage == "dump":
            stages.dump(exp, workers=args.dump_workers)
        elif stage == "split":
            stages.split(exp)
        elif stage == "prefill":
            stages.prefill(exp, workers=args.prefill_workers)
        elif stage == "train":
            stages.train(exp, gpu=args.gpu, wait=not args.no_wait)
        elif stage == "eval":
            stages.evaluate(exp, checkpoint=args.checkpoint, gpu=args.gpu,
                            shards=args.eval_shards)


if __name__ == "__main__":
    main()
