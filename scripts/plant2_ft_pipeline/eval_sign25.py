#!/usr/bin/env python3
"""Sign SR eval for plant2-ft checkpoints (2.5-only by default, extensible).

Examples:
  python eval_sign25.py --ckpt /path/to.ckpt --tag my_run --gpu 0
  python eval_sign25.py --addon fvexp30_spatial_2p5_tsfix_lr1e5 --slot best --gpu 1
  python eval_sign25.py --ckpt ... --tag ... --only 2.5 --jobs 8 --metrics-root ...
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from _env import metrics_root, plan_t, shepelev
from _eval import SignsEvalConfig, run_signs_eval, setup_metrics_tag, signs_done


def pick_ckpt(addon_dir: Path, slot: str) -> Path | None:
    if slot == "best":
        hits = sorted(addon_dir.glob("best_*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return hits[0] if hits else None
    hits = sorted(addon_dir.glob("epoch=*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def make_tag_from_ckpt(addon: str, slot: str, ckpt: Path, *, suffix: str = "_sign25") -> str:
    bn = ckpt.name
    lr_tag = addon.split("_lr")[-1] if "_lr" in addon else ""
    if slot == "best":
        m = re.search(r"best_(\d+)_", bn)
        n = m.group(1) if m else slot
        if lr_tag and "spatial_2p5_tsfix" in addon:
            return f"fvexp30_spatial_2p5_tsfix_lr{lr_tag}_best{n}{suffix}"
        return f"{addon}_best{n}{suffix}"
    m = re.search(r"epoch=(\d+)_", bn)
    ep = m.group(1) if m else slot
    if lr_tag and "spatial_2p5_tsfix" in addon:
        return f"fvexp30_spatial_2p5_tsfix_lr{lr_tag}_ep{ep}{suffix}"
    return f"{addon}_ep{ep}{suffix}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", type=Path, help="Path to .ckpt")
    g.add_argument("--addon", help="Checkpoint addon name under checkpoints_ft/")
    p.add_argument("--slot", choices=("best", "last"), default="best", help="With --addon")
    p.add_argument("--tag", default=None, help="Run name / output tag (default: derived from ckpt)")
    p.add_argument("--gpu", default="0")
    p.add_argument("--only", dest="only_signs", default="2.5", help="Sign filter (eval_checkpoint --only)")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--scenes-per-job", type=int, default=20)
    p.add_argument("--metrics-root", type=Path, default=shepelev() / "plant2_ft_metrics/spatial_2p5_tsfix_eval_sign25")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--skip-if-done", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.ckpt:
        ckpt = args.ckpt
        tag = args.tag or ckpt.stem
    else:
        addon_dir = plan_t() / "checkpoints_ft" / args.addon
        slot = "last" if args.slot == "last" else "best"
        ckpt = pick_ckpt(addon_dir, slot)
        if ckpt is None:
            raise SystemExit(f"ERROR: no ckpt for addon={args.addon} slot={slot}")
        tag = args.tag or make_tag_from_ckpt(args.addon, slot, ckpt)

    if args.skip_if_done and signs_done(tag):
        print(f"SKIP done: {tag}")
        return 0

    cfg = SignsEvalConfig(
        ckpt=ckpt,
        tag=tag,
        gpu=args.gpu,
        only_signs=args.only_signs,
        jobs=args.jobs,
        scenes_per_job=args.scenes_per_job,
        metrics_root=args.metrics_root,
        max_retries=args.max_retries,
    )
    setup_metrics_tag(args.metrics_root, tag, ckpt)
    return run_signs_eval(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
