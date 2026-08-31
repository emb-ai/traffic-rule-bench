#!/usr/bin/env python3
"""Build a 2.5-only symlink subset of plant2_l1_fv_experts_split_signs.

Keeps route directory targets under the original split so on-disk dumps are
shared. Diskcache keys follow the *new* absolute paths, so the first FT epoch
will lazily re-fill ~2.5 entries into DS_LOCAL (no full re-prefill).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import os
import sys
from pathlib import Path

from lib.paths import plan_t, shepelev

SHEPELEV = shepelev()
SIGN = os.environ.get("SUBSET_SIGN", "2.5")
SRC = Path(os.environ.get("SUBSET_SRC", SHEPELEV / "plant2_l1_fv_experts_split_signs"))
OUT = Path(os.environ.get("SUBSET_OUT", f"{SRC}_{SIGN}"))

# PlanT lives next to whichever tree we are splitting, not always under
# SHEPELEV -- with SHEPELEV redirected at a private root the hardcoded
# join points at a path that does not exist.
sys.path.insert(0, str(plan_t()))
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
    if not (SRC / "train/data").is_dir() or not (SRC / "val/data").is_dir():
        raise SystemExit(f"missing source split at {SRC}")

    uid2 = load_uid2sign()
    meta_src = json.loads((SRC / "split_meta.json").read_text())
    val_routes = list((meta_src.get("val") or {}).get(SIGN) or [])
    if not val_routes:
        raise SystemExit("no val routes for 2.5 in split_meta.json")

    train_routes: list[str] = []
    with os.scandir(SRC / "train/data") as it:
        for e in it:
            if e.is_dir() and resolve_route_sign(e.name, uid2) == SIGN:
                train_routes.append(e.name)
    train_routes.sort()
    if not train_routes:
        raise SystemExit("no train routes resolved as 2.5")

    for split, routes in (("train", train_routes), ("val", val_routes)):
        data_out = OUT / split / "data"
        data_out.mkdir(parents=True, exist_ok=True)
        src_data = SRC / split / "data"
        for name in routes:
            src = src_data / name
            if not src.is_dir():
                raise SystemExit(f"missing route dir: {src}")
            link_route(src, data_out / name)
        # Optional slurm tree (not required when filter_routes=False).
        slurm_src = SRC / split / "slurm"
        if slurm_src.is_dir():
            link_route(slurm_src, OUT / split / "slurm")

    split_meta = {
        "seed": meta_src.get("seed"),
        "parent_split": str(SRC),
        "sign_filter": [SIGN],
        "sources": meta_src.get("sources"),
        "per_sign": {
            SIGN: {
                "N": len(train_routes) + len(val_routes),
                "n_train": len(train_routes),
                "n_val": len(val_routes),
                "mode": "subset_symlink",
            }
        },
        "train_counts": {SIGN: len(train_routes)},
        "val": {SIGN: val_routes},
        "train": {SIGN: train_routes},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split_meta.json").write_text(json.dumps(split_meta, indent=2) + "\n")

    n_train = sum(1 for p in (OUT / "train/data").iterdir() if p.is_dir() or p.is_symlink())
    n_val = sum(1 for p in (OUT / "val/data").iterdir() if p.is_dir() or p.is_symlink())
    print(f"OUT={OUT}")
    print(f"train_2.5={n_train} val_2.5={n_val}")
    print(f"wrote {OUT / 'split_meta.json'}")


if __name__ == "__main__":
    main()
