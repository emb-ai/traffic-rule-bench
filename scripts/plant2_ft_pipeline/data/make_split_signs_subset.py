#!/usr/bin/env python3
"""Build a sign-only symlink subset of a full sign split.

Keeps route directory targets under the original split so on-disk dumps are
shared. Diskcache keys follow the *new* absolute paths, so the first FT epoch
will lazily re-fill entries into DS_LOCAL (no full re-prefill).
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

from lib.env import shepelev

SHEPELEV = shepelev()

sys.path.insert(0, str(SHEPELEV / "traffic-rule-bench/plant2/PlanT"))
from util.sign_id import load_uid2sign, resolve_route_sign  # noqa: E402


def link_route(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and os.readlink(dst) == str(src):
            return
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            raise SystemExit(f"refusing to replace non-symlink path: {dst}")
    os.symlink(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signs", nargs="+", required=True, help="e.g. --signs 2.5  or  --signs 4.2.1 4.2.2 4.2.3")
    ap.add_argument("--src", type=Path, default=SHEPELEV / "plant2_l1_fv_experts_split_signs")
    ap.add_argument("--out", type=Path, default=None, help="default: <src>_<sign1>_<sign2>...")
    args = ap.parse_args()

    signs = set(args.signs)
    slug = "_".join(s.replace(".", "_") for s in args.signs)
    src = args.src
    out = args.out or src.parent / f"{src.name}_{'_'.join(args.signs)}"

    if not (src / "train/data").is_dir() or not (src / "val/data").is_dir():
        raise SystemExit(f"missing source split at {src}")

    uid2 = load_uid2sign()
    meta_src = json.loads((src / "split_meta.json").read_text())
    val_routes: list[str] = []
    for sign in args.signs:
        val_routes.extend((meta_src.get("val") or {}).get(sign) or [])
    if not val_routes:
        raise SystemExit(f"no val routes for signs={args.signs} in split_meta.json")

    train_routes: list[str] = []
    with os.scandir(src / "train/data") as it:
        for e in it:
            if e.is_dir() and resolve_route_sign(e.name, uid2) in signs:
                train_routes.append(e.name)
    train_routes.sort()
    if not train_routes:
        raise SystemExit(f"no train routes resolved as signs={args.signs}")

    for split, routes in (("train", train_routes), ("val", val_routes)):
        data_out = out / split / "data"
        data_out.mkdir(parents=True, exist_ok=True)
        src_data = src / split / "data"
        for name in routes:
            route_src = src_data / name
            if not route_src.is_dir():
                raise SystemExit(f"missing route dir: {route_src}")
            link_route(route_src, data_out / name)
        # Optional slurm tree (not required when filter_routes=False).
        slurm_src = src / split / "slurm"
        if slurm_src.is_dir():
            link_route(slurm_src, out / split / "slurm")

    split_meta = {
        "seed": meta_src.get("seed"),
        "parent_split": str(src),
        "sign_filter": args.signs,
        "sources": meta_src.get("sources"),
        "per_sign": {
            slug: {
                "N": len(train_routes) + len(val_routes),
                "n_train": len(train_routes),
                "n_val": len(val_routes),
                "mode": "subset_symlink",
            }
        },
        "train_counts": {slug: len(train_routes)},
        "val": {slug: val_routes},
        "train": {slug: train_routes},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_meta.json").write_text(json.dumps(split_meta, indent=2) + "\n")

    n_train = sum(1 for p in (out / "train/data").iterdir() if p.is_dir() or p.is_symlink())
    n_val = sum(1 for p in (out / "val/data").iterdir() if p.is_dir() or p.is_symlink())
    print(f"OUT={out}")
    print(f"train_{slug}={n_train} val_{slug}={n_val}")
    print(f"wrote {out / 'split_meta.json'}")


if __name__ == "__main__":
    main()
