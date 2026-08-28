#!/usr/bin/env python3
"""Read a finetune run's CSVLogger and print the curves that matter.

Groups the columns by what question they answer: did it fit, does it imitate,
does it obey the plate. Also reports the denominator behind the compliance
rate -- a rate on a collapsed denominator looks like progress and is not.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

GROUPS = [
    ("fit", ["train/loss_all", "val/loss_all"]),
    ("imitation", ["train/loss_wp", "val/loss_wp", "val/loss_path"]),
    ("speed head", ["train/loss_egospeed", "val/loss_egospeed", "val/acc_egospeed"]),
    ("sign rule", ["val/sign_compliance_speed", "val/sign_zone_frames"]),
    ("sampling", ["train/transient_share"]),
]


def load(run_dir: Path):
    versions = sorted(run_dir.glob("CSVLogger/version_*"))
    if not versions:
        return {}
    per = defaultdict(dict)
    with open(versions[-1] / "metrics.csv") as fh:
        for row in csv.DictReader(fh):
            try:
                epoch = int(float(row["epoch"]))
            except (KeyError, TypeError, ValueError):
                continue
            for key, raw in row.items():
                if raw in ("", None) or key in ("epoch", "step"):
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if not math.isnan(value):
                    per[epoch][key] = value
    return per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="log dirs, e.g. .../log/ft_big_uniform_1")
    ap.add_argument("--every", type=int, default=5)
    args = ap.parse_args()

    loaded = {}
    for run in args.runs:
        path = Path(run)
        per = load(path)
        if not per:
            print(f"{path.name}: no metrics", file=sys.stderr)
            continue
        loaded[path.name] = per

    for name, per in loaded.items():
        epochs = sorted(per)
        print(f"\n===== {name}  ({len(epochs)} epochs, last={epochs[-1]}) =====")
        for title, cols in GROUPS:
            cols = [c for c in cols if any(c in per[e] for e in epochs)]
            if not cols:
                continue
            print(f"\n  [{title}]")
            print("    ep " + " ".join(f"{c.split('/')[0][0]}.{c.split('/')[1][:13]:>13}" for c in cols))
            show = [e for e in epochs if e % args.every == 0]
            if epochs[-1] not in show:
                show.append(epochs[-1])
            for e in show:
                line = f"    {e:>3} "
                for c in cols:
                    v = per[e].get(c)
                    line += f"{v:>15.4f} " if v is not None else f"{'-':>15} "
                print(line)

    if len(loaded) == 2:
        (na, pa), (nb, pb) = loaded.items()
        ea, eb = max(pa), max(pb)
        print(f"\n===== {na} (ep {ea})  vs  {nb} (ep {eb}) =====")
        keys = sorted(set(pa[ea]) & set(pb[eb]) &
                      {c for _, cols in GROUPS for c in cols})
        print(f"  {'metric':>28} {'A':>10} {'B':>10} {'B-A':>10}")
        for k in keys:
            a, b = pa[ea][k], pb[eb][k]
            print(f"  {k:>28} {a:>10.4f} {b:>10.4f} {b - a:>+10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
