#!/usr/bin/env python3
"""Build a matched pair of single-sign splits: same scenes, different frames.

The question is whether auxiliary traffic in the training frames teaches the
model to stop. Answering it needs two finetunes that differ in exactly one
thing, so both splits here carry the *same route names* in the *same* train/val
halves — one pointing at the re-dumped frames (convoy present), one at the
frames the baseline was trained on (convoy missing). A route absent from either
dump is dropped from both, so the pair never drifts apart.

Routes are symlinked, so neither tree costs disk.

  python3 make_sign_pair_splits.py --sign 2.5 \
      --new-dump <fix>/plant2_l1_from_experts_signs \
      --old-dump <shep>/plant2_l1_from_experts_signs \
      --out-new <fix>/split_2.5_new --out-old <fix>/split_2.5_old
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_train_val_split_fv_experts_signs import (  # noqa: E402
    load_priority_uid_sign,
    route_is_ok,
    route_sign,
    split_counts,
)

SEED = 42


def ensure_slurm(split_root: Path) -> None:
    """filter_routes reads a slurm log; a dummy keeps it from refusing to run."""
    for half in ("train", "val"):
        log_dir = split_root / half / "slurm" / "run_files" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = log_dir / "qsub_out2025_07.log"
        if not stamp.exists():
            stamp.write_text("# dummy log\n", encoding="utf-8")


def link_all(dump: Path, names: dict[str, list[str]], out: Path, sign: str) -> dict:
    for half, routes in names.items():
        data = out / half / "data"
        data.mkdir(parents=True, exist_ok=True)
        for name in routes:
            dst = data / name
            if dst.is_symlink() or dst.exists():
                continue
            os.symlink(dump / "data" / name, dst)
    ensure_slurm(out)
    meta = {
        "seed": SEED,
        "source_dump": str(dump),
        "sign_filter": [sign],
        "per_sign": {sign: {
            "N": len(names["train"]) + len(names["val"]),
            "n_train": len(names["train"]),
            "n_val": len(names["val"]),
            "mode": "pair_symlink",
        }},
        "train_counts": {sign: len(names["train"])},
        "val": {sign: names["val"]},
        "train": {sign: names["train"]},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sign", default="2.5")
    ap.add_argument("--new-dump", required=True)
    ap.add_argument("--old-dump", required=True)
    ap.add_argument("--out-new", required=True)
    ap.add_argument("--out-old", required=True)
    args = ap.parse_args()

    new_dump, old_dump = Path(args.new_dump), Path(args.old_dump)
    uid2sign = load_priority_uid_sign()

    def routes_of(dump: Path) -> set[str]:
        data = dump / "data"
        if not data.is_dir():
            raise SystemExit(f"missing {data}")
        found = set()
        for p in sorted(data.iterdir()):
            if not p.is_dir():
                continue
            if route_sign(p.name, uid2sign) != args.sign:
                continue
            if not route_is_ok(p):
                continue
            found.add(p.name)
        return found

    fresh, base = routes_of(new_dump), routes_of(old_dump)
    common = sorted(fresh & base)
    print(f"{args.sign}: new={len(fresh)} old={len(base)} usable in both={len(common)}")
    if not common:
        raise SystemExit("no routes present in both dumps — nothing comparable to train on")
    dropped = (fresh | base) - set(common)
    if dropped:
        print(f"dropped {len(dropped)} route(s) present in only one dump")

    rng = random.Random(SEED)
    routes = list(common)
    rng.shuffle(routes)
    n_train, n_val, mode = split_counts(len(routes))
    names = {"val": sorted(routes[:n_val]), "train": sorted(routes[n_val:n_val + n_train])}
    print(f"split: train={len(names['train'])} val={len(names['val'])} mode={mode}")

    for dump, out in ((new_dump, Path(args.out_new)), (old_dump, Path(args.out_old))):
        meta = link_all(dump, names, out, args.sign)
        print(f"wrote {out} (train={meta['train_counts'][args.sign]}, "
              f"val={len(meta['val'][args.sign])})")


if __name__ == "__main__":
    main()
