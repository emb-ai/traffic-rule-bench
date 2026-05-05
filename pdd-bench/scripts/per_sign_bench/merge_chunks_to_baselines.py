#!/usr/bin/env python3
"""Merge chunked runs back into flat per-baseline dirs.

The chunked orchestrator writes episodes to:
   $RUNS_DIR/var_<i>/chunk_<k>/<baseline>/episodes_<policy>.jsonl

If you killed the run before the post-merge step finished, you can run this
helper to consolidate whatever chunks exist into:
   $RUNS_DIR/var_<i>/<baseline>/episodes_<policy>.jsonl

Then re-run aggregate_per_var_baselines.py to populate cumulative.

Usage:
   python3 merge_chunks_to_baselines.py \
       --runs-dir /path/to/benchmark_2node_eval/runs/var_0
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def merge_chunks(runs_var: Path) -> dict:
    chunk_dirs = sorted(d for d in runs_var.iterdir()
                        if d.is_dir() and d.name.startswith("chunk_"))
    if not chunk_dirs:
        print(f"[warn] no chunk_* dirs under {runs_var}", file=sys.stderr)
        return {"n_baselines": 0, "n_chunks": 0}

    print(f"[scan] {len(chunk_dirs)} chunk dirs under {runs_var}")

    # Discover all baseline names across chunks.
    baselines = set()
    for cd in chunk_dirs:
        for bd in cd.iterdir():
            if bd.is_dir():
                baselines.add(bd.name)

    n_merged = 0
    for baseline in sorted(baselines):
        out_dir = runs_var / baseline
        out_dir.mkdir(exist_ok=True)

        # Merge episodes_<policy>.jsonl (concatenate from each chunk).
        by_policy: dict[str, list[Path]] = {}
        for cd in chunk_dirs:
            bdir = cd / baseline
            if not bdir.exists():
                continue
            for ep in bdir.glob("episodes_*.jsonl"):
                by_policy.setdefault(ep.name, []).append(ep)

        for fname, srcs in by_policy.items():
            merged = out_dir / fname
            with merged.open("w", encoding="utf-8") as fh:
                for src in sorted(srcs):
                    fh.write(src.read_text(encoding="utf-8"))
            n_lines = sum(1 for _ in merged.open(encoding="utf-8"))
            print(f"  [merge] {baseline}/{fname}: {len(srcs)} chunks → {n_lines} rows")

        # Copy replays/ from each chunk (no overwrites — first-writer wins).
        merged_replays = out_dir / "replays"
        for cd in chunk_dirs:
            rep = cd / baseline / "replays"
            if not rep.exists():
                continue
            for src in rep.rglob("*"):
                if src.is_file():
                    rel = src.relative_to(rep)
                    dst = merged_replays / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        shutil.copy2(src, dst)
        n_merged += 1

    return {"n_baselines": n_merged, "n_chunks": len(chunk_dirs)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True,
                   help="Path to runs/var_<i> directory containing chunk_*/ subdirs.")
    args = p.parse_args()

    runs_var = Path(args.runs_dir).resolve()
    if not runs_var.is_dir():
        print(f"ERROR: not a dir: {runs_var}", file=sys.stderr)
        sys.exit(2)

    result = merge_chunks(runs_var)
    print()
    print(f"Merged {result['n_baselines']} baselines from {result['n_chunks']} chunks → {runs_var}/<baseline>/")
    print()
    print("Next step: run aggregate to populate cumulative:")
    print(f"  python3 aggregate_per_var_baselines.py \\")
    print(f"    --runs-dir {runs_var} \\")
    print(f"    --var-name {runs_var.name} \\")
    print(f"    --cumulative-path {runs_var.parent.parent}/cumulative_node1.json \\")
    print(f"    --node node1")


if __name__ == "__main__":
    main()
