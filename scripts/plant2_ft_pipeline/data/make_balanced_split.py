#!/usr/bin/env python3
"""Rebalance an existing split by capping how many routes each sign contributes.

A finetune on the full mixture drowns the junction signs: 2.5 and 4.3 are ~10%
of the routes, the rest is fast free-flow driving, and the model settles on
"always drive" -- stop compliance stays at zero no matter how the stop bin is
weighted in the loss. Capping the over-represented signs raises the junction
share without touching the frames themselves.

Routes are symlinked to the parent split, so the tree costs no disk and the
frames stay bit-identical.

  python3 make_balanced_split.py --src <split> --out <split_bal> --cap 350
  python3 make_balanced_split.py --src <split> --out <split_bal> --cap 350 \
      --keep-all 2.5 --keep-all 4.3
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

SEED = 42


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() and not dst.is_symlink():
        os.symlink(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="parent split (has split_meta.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=350,
                    help="max train routes per sign (val is never capped)")
    ap.add_argument("--keep-all", action="append", default=[],
                    help="sign that keeps every route regardless of --cap")
    ap.add_argument("--only", action="append", default=[],
                    help="restrict the split to these signs (repeatable). A "
                         "narrow split answers whether a signal is learnable at "
                         "all, which a full mixture cannot: with two detour "
                         "codes and nothing else, the side is the only thing "
                         "there is to learn. Read such a run as a plumbing "
                         "check, never as compliance evidence — a narrow "
                         "finetune degenerates on everything left out.")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    meta = json.loads((src / "split_meta.json").read_text())
    train_by_sign: dict[str, list[str]] = meta.get("train") or {}
    if not train_by_sign:
        # The mixture split records only counts, so recover the per-sign lists
        # the same way the pair split does: by name, falling back to the PDD
        # code stored in the route's own boxes.
        print("split_meta has no per-sign train lists — resolving from the routes",
              flush=True)
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from make_train_val_split_fv_experts_signs import (  # noqa: E402
            load_priority_uid_sign, route_sign, sniff_sign)

        uid2sign = load_priority_uid_sign()
        train_by_sign = defaultdict(list)
        unknown = 0
        for p in sorted((src / "train" / "data").iterdir()):
            if not (p.is_dir() or p.is_symlink()):
                continue
            sign = route_sign(p.name, uid2sign) or sniff_sign(p)
            if sign is None:
                unknown += 1
                continue
            train_by_sign[sign].append(p.name)
        print(f"resolved {sum(len(v) for v in train_by_sign.values())} routes, "
              f"unknown={unknown}", flush=True)

    rng = random.Random(SEED)
    keep_all = set(args.keep_all)
    only = set(args.only)
    if only:
        missing = only - set(train_by_sign)
        if missing:
            raise SystemExit(f"--only names signs absent from the parent split: {sorted(missing)}")
        train_by_sign = {k: v for k, v in train_by_sign.items() if k in only}
        print(f"restricted to {sorted(only)}", flush=True)
    kept: dict[str, list[str]] = {}
    for sign, routes in sorted(train_by_sign.items()):
        routes = sorted(routes)
        if sign not in keep_all and len(routes) > args.cap:
            rng.shuffle(routes)
            routes = sorted(routes[: args.cap])
        kept[sign] = routes

    for sign, routes in kept.items():
        for name in routes:
            link(src / "train" / "data" / name, out / "train" / "data" / name)
    val_names = [p.name for p in (src / "val" / "data").iterdir() if p.is_dir() or p.is_symlink()]
    if only:
        # Validation must hold the same signs as training: a val loss averaged
        # over signs the run never saw says nothing about the run.
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from make_train_val_split_fv_experts_signs import sign_of  # noqa: E402
        val_names = [n for n in val_names if sign_of(n) in only]
        print(f"val restricted to {len(val_names)} routes", flush=True)
    for name in val_names:
        link(src / "val" / "data" / name, out / "val" / "data" / name)

    for half in ("train", "val"):
        log_dir = out / half / "slurm" / "run_files" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = log_dir / "qsub_out2025_07.log"
        if not stamp.exists():
            stamp.write_text("# dummy log\n", encoding="utf-8")

    total = sum(len(v) for v in kept.values())
    per_sign = {s: {"N": len(v), "n_train": len(v), "n_val": 0, "mode": "balanced_symlink"}
                for s, v in kept.items()}
    (out / "split_meta.json").write_text(json.dumps({
        "seed": SEED, "parent_split": str(src), "cap": args.cap,
        "keep_all": sorted(keep_all), "per_sign": per_sign,
        "train_counts": {s: len(v) for s, v in kept.items()},
        "train": kept, "val": meta.get("val", {}),
    }, indent=2) + "\n")

    print("train routes per sign:", {s: len(v) for s, v in sorted(kept.items())})
    for s in sorted(keep_all):
        share = len(kept.get(s, [])) / total if total else 0
        print(f"  {s}: {share:.1%} of the training routes")
    print(f"total train routes: {total}  ->  {out}")


if __name__ == "__main__":
    main()
