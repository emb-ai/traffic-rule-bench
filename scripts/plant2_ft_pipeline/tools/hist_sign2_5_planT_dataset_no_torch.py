#!/usr/bin/env python3
"""
Histogram presence of PDD sign "2.5" in PlanTDataset x_objs (no torch dependency).

Why "no torch":
  - `plant2/PlanT/dataset.py` imports torch + beartype, which may be missing.
  - For this histogram we only need to know whether a sign survives the
    PlanTDataset *sign-related filters*, which depend only on `boxes/*.json.gz`.

Algorithm:
  1) Reproduce PlanTDataset route filtering (filter_routes=True by default)
     using `results.json.gz` and slurm logs.
  2) For each surviving route_dir, enumerate seq frames with the same window:
       seq in range(5, num_seq - wps_len - seq_len - 2)
     and since cfg.model.training.seq_len=1:
       current-frame index t == seq
  3) For each boxes/XXXX.json.gz at current frame t, check "sign 2.5 present"
     using the same logic as `boxes_has_class()` in `print_plant_batch.py`.
  4) fraction[t] = (# routes with sign present at t) / (# routes that have t).

Output:
  - raw TSV: frame_idx, fraction
  - top20 frames by fraction
  - optional binned histogram
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


def _boxes_has_class(path: str | Path, class_name: str) -> bool:
    """True if class_name would survive PlanTDataset filters for *signs*."""
    # Import inside function: safe for multiprocessing.
    repo_root = Path(__file__).resolve().parents[3]
    pt = repo_root / "plant2" / "PlanT"
    if str(pt) not in sys.path:
        sys.path.insert(0, str(pt))
    from util.sign_id import SIGN_CODES

    _SIGN_RANGE_M = 30.0
    sign_like = set(SIGN_CODES) | {"stop_sign"}

    with gzip.open(path, "rt", encoding="utf-8") as f:
        boxes = json.load(f)

    if not isinstance(boxes, list) or len(boxes) < 2:
        return False

    want = str(class_name)
    if want not in sign_like:
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


def _write_tsv(path: Path, rows: Iterable[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("frame_idx\tfraction\n")
        for t, frac in rows:
            f.write(f"{t}\t{frac:.8f}\n")


def _parse_wps_len_seq_len_from_planT_yaml() -> tuple[int, int]:
    """Hardcoded based on PlanT/config/model/PlanT.yaml shipped in this repo."""
    # waypoints.wps_len: 8
    # training.seq_len: 1
    return 8, 1


def _is_route_trainable(
    *,
    route_dir: str,
    root: str,
) -> bool:
    """
    Copy of PlanTDataset init filtering for cfg.model.training.filter_routes=True.
    """
    route = os.path.basename(route_dir)
    if route.startswith("FAILED_") or not os.path.isfile(route_dir + "/results.json.gz"):
        return False

    # Filter by results.json.gz content.
    with gzip.open(route_dir + "/results.json.gz", "rt", encoding="utf-8") as f:
        results_route = json.load(f)

    condition1 = (
        results_route["scores"]["score_composed"] < 100.0
        and not (
            results_route["num_infractions"]
            == len(results_route["infractions"]["min_speed_infractions"])
        )
    )
    condition2 = results_route["status"] == "Failed - Agent couldn't be set up"
    condition3 = results_route["status"] == "Failed"
    condition4 = results_route["status"] == "Failed - Simulation crashed"
    condition5 = results_route["status"] == "Failed - Agent crashed"
    if condition1 or condition2 or condition3 or condition4 or condition5:
        return False

    # Silent crash filter from slurm logs.
    if results_route["timestamp"][:4] == "Town":
        log_file = "qsub_out" + "_".join(results_route["timestamp"].split("_")[:3]) + ".log"
    else:
        log_file = "qsub_out" + "_".join(results_route["timestamp"].split("_")[:2]) + ".log"

    log_file = root.rstrip("/")[:-4] + "/slurm/run_files/logs/" + log_file

    silentcrash = False
    with open(log_file, "r", encoding="utf8") as f:
        lines = f.readlines()
    for line in lines:
        if "SKIPPED" in line:
            vehicle = line.split(" ")[-1].strip()
            # Keep same exception list as PlanTDataset.
            if vehicle[:6] != "walker" and vehicle not in [
                "vehicle.bh.crossbike",
                "vehicle.diamondback.century",
                "vehicle.gazelle.omafiets",
            ]:
                silentcrash = True
                break
    if silentcrash:
        return False

    return True


def _worker_boxes_has_class_packed(packed: tuple[str, str]) -> bool:
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
    ap.add_argument("--workers", type=int, default=6, help="Process workers for gzip/json scanning")
    ap.add_argument("--chunksize", type=int, default=64, help="Worker chunk size")
    ap.add_argument("--bin-size", type=int, default=0, help="If >0, output binned histogram every N frames")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = ap.parse_args(argv)

    root = str(args.ds_root)
    root = root.rstrip("/")

    wps_len, seq_len = _parse_wps_len_seq_len_from_planT_yaml()
    # current frame index t == seq because seq_len == 1
    assert seq_len == 1

    # Find route dirs with boxes.
    import glob

    label_raw_path_all = glob.glob(os.path.join(root, "**/boxes"), recursive=True)
    route_dirs = [p[:-5] for p in label_raw_path_all]  # strip "/boxes"

    print(f"Found route dirs with boxes: {len(route_dirs)}")

    # Build tasks: boxes path for each (route, t).
    label0_paths: list[str] = []
    frame_idxs: list[int] = []

    trainable = 0
    skipped = 0

    for route_dir in route_dirs:
        if not _is_route_trainable(route_dir=route_dir, root=root):
            skipped += 1
            continue
        trainable += 1

        num_seq = len(os.listdir(os.path.join(route_dir, "boxes")))
        # Match PlanTDataset init:
        # for seq in range(5, num_seq - wps_len - seq_len - 2)
        start = 5
        end = num_seq - wps_len - seq_len - 2
        if end <= start:
            continue
        for seq in range(start, end):
            label0_paths.append(os.path.join(route_dir, "boxes", f"{seq:04d}.json.gz"))
            frame_idxs.append(seq)

    print(f"trainable routes={trainable}  skipped routes={skipped}")
    print(f"samples(tasks)={len(label0_paths)}")

    denom = Counter(frame_idxs)  # per-frame count == per-route count (seq_len=1)
    num = Counter()

    # Parallel scan for sign presence in boxes/frames for each sample.
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        packed = [(p, args.sign) for p in label0_paths]
        for i, present in enumerate(ex.map(_worker_boxes_has_class_packed, packed, chunksize=args.chunksize)):
            if present:
                num[frame_idxs[i]] += 1
            if i and i % 5000 == 0:
                print(f"scanned {i}/{len(label0_paths)} samples ...", flush=True)

    fractions: dict[int, float] = {}
    for t, d in denom.items():
        fractions[int(t)] = float(num[t]) / float(d) if d else 0.0

    all_rows = sorted(((t, fractions[t]) for t in fractions.keys()), key=lambda x: x[0])

    out_dir = args.out_dir
    raw_path = out_dir / f"hist_sign{args.sign}_raw.tsv"
    _write_tsv(raw_path, all_rows)
    print(f"wrote: {raw_path}")

    top_n = 20
    top_rows = sorted(((t, fractions[t], denom[t], num[t]) for t in fractions.keys()), key=lambda x: (-x[1], x[0]))
    print(f"\nTop {top_n} frame indices by fraction:")
    for t, frac, d, n in top_rows[:top_n]:
        print(f"  t={t:6d}  fraction={frac:.6f}  {n}/{d}")

    if args.bin_size and args.bin_size > 0:
        b = int(args.bin_size)
        # Weighted binning: fraction_bin = sum_t_in_bin num[t] / sum_t_in_bin denom[t]
        bin_denom: Counter[int] = Counter()
        bin_num: Counter[int] = Counter()
        for t, d in denom.items():
            k = int(t // b)
            bin_denom[k] += d
            bin_num[k] += num[t]

        b_rows = sorted(
            ((k * b, float(bin_num[k]) / float(bin_denom[k])) for k in bin_denom.keys()),
            key=lambda x: x[0],
        )
        b_path = out_dir / f"hist_sign{args.sign}_binned_{b}.tsv"
        _write_tsv(b_path, b_rows)
        print(f"\nBinned (every {b} frames) wrote: {b_path}")

        print("\nBinned preview:")
        for t0, frac in b_rows[: min(20, len(b_rows))]:
            print(f"  [{t0:6d}..{t0 + b - 1:6d}]  fraction={frac:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

