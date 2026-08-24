#!/usr/bin/env python3
"""
Histogram presence of PDD sign "2.5" in PlanTDataset x_objs.

For each frame index within trajectory (current-frame boxes/*.json.gz):
  fraction[t] = (# trajectories with sign 2.5 present at t) / (# trajectories that have t)

Sign "2.5" corresponds to internal token id=12.0 in x_objs[...,0] (see print_plant_batch.py).

This script follows the same "sign survives PlanTDataset filters" logic as
`boxes_has_class()` inside `print_plant_batch.py`:
  - sign must have affects_ego=True
  - must be within 30m in xy and abs(z) <= 30m
  - PlanTDataset skips boxes frames 0–4, which is already encoded in the dataset
    sampling window during PlanTDataset init.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable
import sys


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PT = _REPO_ROOT / "plant2" / "PlanT"
if str(_PT) not in sys.path:
    sys.path.insert(0, str(_PT))


def _boxes_has_class(path: str | Path, class_name: str) -> bool:
    """True if class_name would survive PlanTDataset filters for *signs*."""
    from util.sign_id import SIGN_CODES  # local import for multiprocessing safety

    _SIGN_RANGE_M = 30.0
    sign_like = set(SIGN_CODES) | {"stop_sign"}

    with gzip.open(path, "rt", encoding="utf-8") as f:
        boxes = json.load(f)

    if not isinstance(boxes, list) or len(boxes) < 2:
        return False

    want = str(class_name)
    if want not in sign_like:
        # For non-sign classes, this helper is intentionally not implemented.
        return False

    for obj in boxes[1:]:  # skip ego
        if str(obj.get("class")) != want:
            continue

        pos = obj.get("position") or [0.0, 0.0, 0.0]
        px, py = float(pos[0]), float(pos[1])
        pz = float(pos[2]) if len(pos) > 2 else 0.0

        if px * px + py * py > _SIGN_RANGE_M**2 or abs(pz) > _SIGN_RANGE_M:
            continue
        if not obj.get("affects_ego"):
            continue
        return True

    return False


def _parse_frame_idx_from_label0(label0_path: str | Path) -> int:
    # ".../boxes/0005.json.gz" → 5
    name = Path(label0_path).name
    m = re.match(r"(\d+)\.", name)
    if not m:
        raise ValueError(f"Unexpected boxes filename: {name!r}")
    return int(m.group(1))


def _parse_route_from_label0(label0_path: str | Path) -> str:
    # ".../<route>/boxes/0005.json.gz" → "<route>"
    p = Path(label0_path)
    return p.parent.parent.name


def _load_cfg_and_dataset(*, data_root: Path, augment: bool) -> tuple[object, dict]:
    """Load PlanT cfg and instantiate PlanTDataset (only for metadata/labels list)."""
    from omegaconf import OmegaConf, open_dict

    from dataset import PlanTDataset

    cfg = OmegaConf.load(_PT / "config" / "config.yaml")
    cfg = OmegaConf.merge(
        cfg,
        {
            "user": OmegaConf.load(_PT / "config" / "user" / "arbelyaev.yaml"),
            "model": OmegaConf.load(_PT / "config" / "model" / "PlanT.yaml"),
        },
    )

    with open_dict(cfg):
        # Keep dataset sampling logic intact, but avoid heavy extra work.
        cfg.use_caching = False
        cfg.model.training.augment = bool(augment)
        cfg.model.training.augment_parked = False
        cfg.model.training.input_bev = False

    ds = PlanTDataset(str(data_root), cfg, shared_dict=None)
    return ds, cfg


def _iter_label0_paths(ds: object) -> tuple[list[str], list[int], list[str]]:
    """Extract current-frame boxes label0 path, t, route for each sample."""
    labels = getattr(ds, "labels")
    if labels is None:
        raise RuntimeError("Dataset has no .labels")

    # ds.labels is a numpy array of bytes, shape: (N, seq_len+wps_len)
    label0_paths: list[str] = []
    frame_idxs: list[int] = []
    routes: list[str] = []

    for i in range(len(ds)):
        lab0 = labels[i][0]
        path = lab0.decode() if isinstance(lab0, (bytes, bytearray)) else str(lab0)
        label0_paths.append(path)
        frame_idxs.append(_parse_frame_idx_from_label0(path))
        routes.append(_parse_route_from_label0(path))

    return label0_paths, frame_idxs, routes


def _write_tsv(path: Path, rows: Iterable[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("frame_idx\tfraction\n")
        for t, frac in rows:
            f.write(f"{t}\t{frac:.8f}\n")


def _worker_boxes_has_class(packed: tuple[str, str]) -> bool:
    path, sign = packed
    return _boxes_has_class(path, sign)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ds-root",
        type=Path,
        required=True,
        help="PlanT dump root or .../data/ directory with route dirs containing boxes/",
    )
    ap.add_argument("--sign", type=str, default="2.5", help="Sign code, e.g. 2.5")
    ap.add_argument("--expected-token-id", type=int, default=12, help="x_objs type id for sign")
    ap.add_argument("--augment", action="store_true", help="Enable geometric augment (default: off)")
    ap.add_argument("--workers", type=int, default=4, help="Process workers for gzip/json scanning")
    ap.add_argument("--chunksize", type=int, default=64, help="Worker chunk size")
    ap.add_argument("--bin-size", type=int, default=0, help="If >0, output binned histogram every N frames")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = ap.parse_args(argv)

    print(f"ds-root={args.ds_root}")
    print(f"sign={args.sign}  expected-token-id={args.expected_token_id}")
    print(f"augment={args.augment}  workers={args.workers}  chunksize={args.chunksize}")
    if args.bin_size:
        print(f"bin-size={args.bin_size}")

    ds, _cfg = _load_cfg_and_dataset(data_root=args.ds_root, augment=args.augment)

    # Verify mapping (sanity check for the "token id=12" constraint).
    if hasattr(ds, "type_nums") and str(args.sign) in ds.type_nums:
        token_id = int(ds.type_nums[str(args.sign)])
        print(f"token_id(type_nums[{args.sign!r}])={token_id}")
        if token_id != int(args.expected_token_id):
            print(
                f"WARNING: expected-token-id={args.expected_token_id} but dataset has token_id={token_id}",
                flush=True,
            )
    else:
        print("WARNING: could not verify token id via ds.type_nums", flush=True)

    label0_paths, frame_idxs, _routes = _iter_label0_paths(ds)

    denom = Counter(frame_idxs)  # per-frame count == per-route count (seq_len=1)

    # Parallel scan for sign presence in boxes/frames for each sample.
    num = Counter()

    # Instead, we use a top-level argument-packed worker.
    # (Using `map` with a single iterable allows fork-based passing of constants too.)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        # args: iterable of (path, sign) tuples
        packed = [(p, args.sign) for p in label0_paths]
        for i, present in enumerate(
            ex.map(_worker_boxes_has_class, packed, chunksize=args.chunksize)
        ):
            if present:
                num[frame_idxs[i]] += 1
            if i and i % 5000 == 0:
                print(f"scanned {i}/{len(label0_paths)} samples ...", flush=True)

    # Build exact fractions.
    fractions: dict[int, float] = {}
    for t, d in denom.items():
        fractions[int(t)] = float(num[t]) / float(d) if d else 0.0

    # Sort by increasing frame index for the main table.
    all_rows = sorted(((t, fractions[t]) for t in fractions.keys()), key=lambda x: x[0])

    out_dir = args.out_dir
    raw_path = out_dir / f"hist_sign{args.sign}_raw.tsv"
    _write_tsv(raw_path, all_rows)
    print(f"wrote: {raw_path}")

    # Top-N frames by fraction.
    top_n = 20
    top_rows = sorted(((t, fractions[t], denom[t], num[t]) for t in fractions.keys()), key=lambda x: (-x[1], x[0]))
    print(f"\nTop {top_n} frame indices by fraction:")
    for t, frac, d, n in top_rows[:top_n]:
        print(f"  t={t:6d}  fraction={frac:.6f}  {n}/{d}")

    # Compact output in stdout (optional binning, otherwise top only).
    if args.bin_size and args.bin_size > 0:
        b = int(args.bin_size)
        bin_denom = Counter((t // b) for t in denom.keys())
        bin_num = Counter((t // b) for t in num.keys())
        b_rows = sorted(((k * b, float(bin_num[k]) / float(bin_denom[k])) for k in bin_denom.keys()), key=lambda x: x[0])
        b_path = out_dir / f"hist_sign{args.sign}_binned_{b}.tsv"
        _write_tsv(b_path, b_rows)
        print(f"\nBinned (every {b} frames) wrote: {b_path}")

        print("\nBinned table preview:")
        for t0, frac in b_rows[: min(20, len(b_rows))]:
            print(f"  [{t0:6d}..{t0+b-1:6d}]  fraction={frac:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

