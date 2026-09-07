#!/usr/bin/env python3
"""Filter a detour (4.2.1/4.2.2/4.2.3) split down to routes that actually show
the obstacle.

~20-26% of routes in $SM/plant2_fix/detour_split/{train,val} never have a
'static' (cone) object in any frame despite results.json.gz reporting a
completed/perfect-score route -- verified against both an independent scan of
boxes/*.json.gz AND the split's own sample_weights.json ("cones": [] for the
same routes). Training on those gives zero avoidance signal for a "detour"
label. This drops them, preserving the original train/val assignment
(exclude, don't reshuffle) via symlinks into a new split dir.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import os
from collections import Counter


def link_route(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and os.readlink(dst) == str(src):
            return
        dst.unlink()
    os.symlink(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="e.g. $SM/plant2_fix/detour_split")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    split_meta = json.loads((args.src / "split_meta.json").read_text())
    out_meta = {
        "seed": split_meta.get("seed"),
        "parent_split": str(args.src),
        "filter": "drop routes with empty sample_weights.json 'cones' (obstacle never visible)",
        "per_sign": {},
        "train_counts": {},
        "val": {},
        "train": {},
    }

    for split in ("train", "val"):
        weights_path = args.src / split / "sample_weights.json"
        weights = json.loads(weights_path.read_text())
        data_src = args.src / split / "data"
        data_out = args.out / split / "data"
        data_out.mkdir(parents=True, exist_ok=True)

        kept: list[str] = []
        dropped_by_sign: Counter[str] = Counter()
        for name, meta in weights.items():
            route_src = data_src / name
            if not route_src.is_dir():
                raise SystemExit(f"missing route dir referenced by sample_weights.json: {route_src}")
            if meta.get("cones"):
                link_route(route_src, data_out / name)
                kept.append(name)
            else:
                dropped_by_sign[meta.get("code", "?")] += 1

        slurm_src = args.src / split / "slurm"
        if slurm_src.is_dir():
            link_route(slurm_src, args.out / split / "slurm")

        by_sign: Counter[str] = Counter()
        for name in kept:
            for code in ("4.2.1", "4.2.2", "4.2.3"):
                if name.startswith(f"sumo_{code}_"):
                    by_sign[code] += 1
                    break
        if split == "val":
            out_meta["val"] = {code: [n for n in kept if n.startswith(f"sumo_{code}_")] for code in by_sign}
        else:
            out_meta["train_counts"] = dict(by_sign)
            out_meta["train"] = {code: [n for n in kept if n.startswith(f"sumo_{code}_")] for code in by_sign}

        print(f"{split}: kept={len(kept)} dropped={sum(dropped_by_sign.values())} dropped_by_sign={dict(dropped_by_sign)} kept_by_sign={dict(by_sign)}")

    for code in ("4.2.1", "4.2.2", "4.2.3"):
        n_train = len(out_meta["train"].get(code, []))
        n_val = len(out_meta["val"].get(code, []))
        out_meta["per_sign"][code] = {"N": n_train + n_val, "n_train": n_train, "n_val": n_val, "mode": "scene_holdout_filtered"}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "split_meta.json").write_text(json.dumps(out_meta, indent=2) + "\n")
    print(f"wrote {args.out / 'split_meta.json'}")


if __name__ == "__main__":
    main()
