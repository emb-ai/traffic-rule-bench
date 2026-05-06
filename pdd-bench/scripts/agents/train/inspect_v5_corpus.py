#!/usr/bin/env python3
"""Fast inventory of a v5-style trajectory corpus.

Each .pt file is ~50–100 MB (it stores the full road_network + a 200-600
step plant2_batch sequence), so loading every file just to read scalar
metadata is wasteful and competes with live collection workers for IO.

This script does the cheap thing instead:
  * for every sign-prefix (e.g. "2_1", "3_27", "5_13_1"),
    - count completed episodes from the file list,
    - load **one** sample episode and print its return / steps / outcome,
  * and aggregate failed-attempt counts from the _failed/ subdirectory.

Use the slower _stats_trajectories_v4.py script when you need the full
return / steps distribution after collection has finished.

Usage:
    python inspect_v5_corpus.py [<corpus_dir>] [--sample-per-sign N]
                                [--target N] [--full-stats]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict


def _slug_to_code(slug: str) -> str:
    """Inverse of `pdd_code.replace('.', '_')`."""
    return slug.replace("_", ".")


def _sort_key(slug: str):
    """Natural sort: '2_1' < '2_2' < '3_18_2' < '5_13_2'."""
    return [int(x) if x.isdigit() else x for x in slug.split("_")]


def _load_sample(fp: str):
    """Load a single .pt and return a dict with the scalar fields we care
    about. Returns None on failure."""
    try:
        import torch
    except ImportError:
        return None
    try:
        ep = torch.load(fp, weights_only=False, map_location="cpu")
    except Exception as e:
        return {"error": f"load failed: {e}"}
    return {
        "return": float(ep.get("return", 0.0)),
        "steps": int(ep.get("num_steps", 0)),
        "arrived": bool(ep.get("arrived", False)),
        "crash": bool(ep.get("crash", False)),
        "out_of_road": bool(ep.get("out_of_road", False)),
        "truncated_horizon": bool(ep.get("truncated_horizon", False)),
        "sign_type": ep.get("sign_type"),
        "pdd_code": ep.get("pdd_code"),
        "scene_id": ep.get("scene_id"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("corpus_dir", nargs="?",
                   default="pdd-bench/outputs/benchmark_sign_trajectories_v5")
    p.add_argument("--sample-per-sign", type=int, default=1,
                   help="how many sample .pt to load per sign (loaded "
                        "sequentially, IO-friendly)")
    p.add_argument("--target", type=int, default=300,
                   help="success target per sign (for the progress bar)")
    p.add_argument("--full-stats", action="store_true",
                   help="ALSO load every .pt file (slow, ~minutes)")
    args = p.parse_args()

    corpus = args.corpus_dir
    if not os.path.isdir(corpus):
        print(f"ERROR: not a directory: {corpus}", file=sys.stderr)
        return 1

    print(f"Corpus: {corpus}")

    pt_files = sorted(glob.glob(os.path.join(corpus, "*.pt")))
    pat = re.compile(r"^(?P<slug>.+?)_ep(?P<idx>\d+)\.pt$")
    by_slug: dict[str, list[str]] = defaultdict(list)
    for fp in pt_files:
        m = pat.match(os.path.basename(fp))
        if not m:
            continue
        by_slug[m["slug"]].append(fp)

    total = sum(len(v) for v in by_slug.values())
    print(f"Total saved episodes (success): {total}")

    failed_dir = os.path.join(corpus, "_failed")
    failed = sorted(glob.glob(os.path.join(failed_dir, "*.pt"))) \
        if os.path.isdir(failed_dir) else []
    failed_by_slug: dict[str, int] = defaultdict(int)
    fail_pat = re.compile(r"^(?P<slug>.+?)_attempt\d+\.pt$")
    for fp in failed:
        m = fail_pat.match(os.path.basename(fp))
        if m:
            failed_by_slug[m["slug"]] += 1
    print(f"Total failed-attempt .pt:       {len(failed)}\n")

    slugs = sorted(by_slug.keys(), key=_sort_key)

    # -- per-sign inventory + sample-load -----------------------------
    hdr = (f"{'pdd_code':<10} {'success':>8}/{'tgt':<5} "
           f"{'pct':>5}  {'failed':>6}  "
           f"{'sample.return':>13}  {'sample.steps':>12}  "
           f"{'arrived':>7}  sign_type")
    print(hdr)
    print("-" * len(hdr))

    for slug in slugs:
        n = len(by_slug[slug])
        nf = failed_by_slug.get(slug, 0)
        pct = 100 * n / max(args.target, 1)
        sample = _load_sample(by_slug[slug][0]) if args.sample_per_sign > 0 else None
        if sample is None:
            sample_ret, sample_steps, sample_arr, sign_type = "—", "—", "—", "—"
        elif "error" in sample:
            sample_ret, sample_steps, sample_arr, sign_type = (
                f"ERR({sample['error'][:25]})", "—", "—", "—")
        else:
            sample_ret = f"{sample['return']:13.1f}"
            sample_steps = f"{sample['steps']:12d}"
            sample_arr = "yes" if sample["arrived"] else "NO"
            sign_type = sample["sign_type"] or "?"
        print(f"{_slug_to_code(slug):<10} {n:>8}/{args.target:<5} "
              f"{pct:>4.0f}%  {nf:>6}  "
              f"{sample_ret:>13}  {sample_steps:>12}  "
              f"{sample_arr:>7}  {sign_type}")

    # -- optional global stats over every file -----------------------
    if args.full_stats:
        print("\n=== --full-stats: loading every .pt (slow!) ===")
        import statistics
        rets, steps, n_arr = [], [], 0
        for slug in slugs:
            for fp in by_slug[slug]:
                d = _load_sample(fp)
                if d and "error" not in d:
                    rets.append(d["return"])
                    steps.append(d["steps"])
                    if d["arrived"]:
                        n_arr += 1
        if rets:
            print(f"  episodes loaded: {len(rets)}")
            print(f"  arrived=True:    {n_arr}/{len(rets)} "
                  f"({100*n_arr/len(rets):.1f}%)")
            print(f"  return:          mean={statistics.mean(rets):.1f}  "
                  f"std={statistics.pstdev(rets):.1f}  "
                  f"min={min(rets):.1f}  max={max(rets):.1f}")
            print(f"  steps:           mean={statistics.mean(steps):.1f}  "
                  f"std={statistics.pstdev(steps):.1f}  "
                  f"min={min(steps)}  max={max(steps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
