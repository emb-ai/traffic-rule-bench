#!/usr/bin/env python3
"""Sign SR eval for plant2-ft checkpoints.

Examples:
  python eval_sign.py --ckpt /path/to.ckpt --tag my_run --gpu 0 --only 2.5
  python eval_sign.py --addon fvexp30_spatial_2p5_tsfix_lr1e5 --slot best --gpu 1 --only 2.5
  python eval_sign.py --ckpt ... --tag ... --only 4.2.1,4.2.2,4.2.3 --jobs 8 --metrics-root ...
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import re

from lib.env import metrics_root, plan_t
from lib.eval_core import (
    SignsEvalConfig,
    run_signs_eval,
    setup_metrics_tag,
    signs_done,
    trajectory_done,
)


def pick_ckpt(addon_dir: Path, slot: str) -> Path | None:
    if slot == "best":
        hits = sorted(addon_dir.glob("best_*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return hits[0] if hits else None
    if slot == "last":
        hits = sorted(addon_dir.glob("last_*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return hits[0] if hits else None
    hits = sorted(addon_dir.glob("epoch=*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def make_tag_from_ckpt(addon: str, slot: str, ckpt: Path, *, suffix: str) -> str:
    bn = ckpt.name
    if slot == "best":
        m = re.search(r"best_(\d+)_", bn)
        n = m.group(1) if m else slot
        return f"{addon}_best{n}{suffix}"
    m = re.search(r"epoch=(\d+)_", bn)
    ep = m.group(1) if m else slot
    return f"{addon}_ep{ep}{suffix}"


def tag_suffix(only_signs: str) -> str:
    parts = [re.sub(r"[^0-9a-zA-Z]+", "", s) for s in only_signs.split(",")]
    return "_sign" + "_".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", type=Path, help="Path to .ckpt")
    g.add_argument("--addon", help="Checkpoint addon name under checkpoints_ft/")
    p.add_argument("--slot", choices=("best", "last"), default="best", help="With --addon")
    p.add_argument("--tag", default=None, help="Run name / output tag (default: derived from ckpt + --only)")
    p.add_argument("--gpu", default="0")
    p.add_argument("--only", dest="only_signs", required=True, help="Sign filter, e.g. 2.5 or 4.2.1,4.2.2,4.2.3")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--scenes-per-job", type=int, default=20)
    p.add_argument("--metrics-root", type=Path, default=None, help="default: $METRICS_ROOT/<derived from --only>")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--skip-if-done", action="store_true", default=True)
    p.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run eval even if trajectory report already exists",
    )
    p.add_argument(
        "--trajectory",
        default=None,
        help="Single train trajectory / scene_uid (e.g. sign_100062_j0_lane0_seed1974118946_v0_default)",
    )
    p.add_argument("--save-gifs", action="store_true", help="Write eval GIF (stop_sign eval_pipeline)")
    p.add_argument(
        "--save-predictions",
        action="store_true",
        help="Log ego-speed preds, dist to sign, and x_objs per step (JSONL in metrics tag dir)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    suffix = tag_suffix(args.only_signs)
    metrics_root_arg = args.metrics_root or metrics_root() / f"eval{suffix}"

    if args.ckpt:
        ckpt = args.ckpt
        tag = args.tag or f"{ckpt.stem}{suffix}"
    else:
        addon_dir = plan_t() / "checkpoints_ft" / args.addon
        slot = "last" if args.slot == "last" else "best"
        ckpt = pick_ckpt(addon_dir, slot)
        if ckpt is None:
            raise SystemExit(f"ERROR: no ckpt for addon={args.addon} slot={slot}")
        tag = args.tag or make_tag_from_ckpt(args.addon, slot, ckpt, suffix=suffix)

    predictions_path = None
    if args.trajectory and args.save_predictions:
        predictions_path = metrics_root_arg / tag / f"{args.trajectory}_predictions.jsonl"
    if args.skip_if_done and not args.force_rerun:
        if args.trajectory:
            if trajectory_done(tag, predictions_path=predictions_path if args.save_predictions else None):
                print(f"SKIP done: {tag}")
                return 0
        elif signs_done(tag):
            print(f"SKIP done: {tag}")
            return 0

    cfg = SignsEvalConfig(
        ckpt=ckpt,
        tag=tag,
        gpu=args.gpu,
        only_signs=args.only_signs,
        jobs=args.jobs,
        scenes_per_job=args.scenes_per_job,
        metrics_root=metrics_root_arg,
        max_retries=args.max_retries,
        trajectory=args.trajectory,
        save_gifs=args.save_gifs,
        save_predictions=args.save_predictions,
    )
    setup_metrics_tag(metrics_root_arg, tag, ckpt)
    return run_signs_eval(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
