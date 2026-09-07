#!/usr/bin/env python3
"""
Histogram presence of a PDD sign (default "2.5") in PlanTDataset x_objs.

For each frame index within trajectory (current-frame boxes/*.json.gz):
  fraction[t] = (# trajectories with the sign present at t) / (# trajectories that have t)

Sign presence uses the same "sign survives PlanTDataset filters" rule as
tools/print_plant_batch.py's boxes_has_class(): affects_ego=True, within 30m
xy, |z| <= 30m (see lib_tools.sign_survives_filter).

Two ways to enumerate (route, frame) samples, selected by --no-torch:

  default (torch):
    Instantiate PlanTDataset (needs torch + beartype) and read its .labels
    for the exact set of samples PlanTDataset would train on -- including
    frames 0-4 being skipped, which falls out of the dataset's own sampling
    window.

  --no-torch:
    `plant2/PlanT/dataset.py` imports torch + beartype, which may be missing
    in some envs. This path reproduces PlanTDataset's route filtering
    (cfg.model.training.filter_routes=True) and its per-route frame window
    (seq in range(5, num_seq - wps_len - seq_len - 2), current-frame index
    t == seq since seq_len=1) using only results.json.gz + slurm logs --
    no torch import required.

Output (either mode):
  - raw TSV: frame_idx, fraction
  - top-20 frames by fraction
  - optional binned histogram (weighted by actual sample counts per bin)
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PT = _REPO_ROOT / "plant2" / "PlanT"
if str(_PT) not in sys.path:
    sys.path.insert(0, str(_PT))

import lib_tools

# --------------------------------------------------------------------------
# shared: sign-presence scan (uses lib_tools, identical in both modes)
# --------------------------------------------------------------------------


def _worker_has_sign(packed: tuple[str, str]) -> bool:
    path, sign = packed
    return lib_tools.boxes_has_sign(path, sign)


def scan_sign_presence(
    label0_paths: list[str],
    frame_idxs: list[int],
    *,
    sign: str,
    workers: int,
    chunksize: int,
) -> Counter:
    num: Counter = Counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        packed = [(p, sign) for p in label0_paths]
        for i, present in enumerate(ex.map(_worker_has_sign, packed, chunksize=chunksize)):
            if present:
                num[frame_idxs[i]] += 1
            if i and i % 5000 == 0:
                print(f"scanned {i}/{len(label0_paths)} samples ...", flush=True)
    return num


def compute_fractions(denom: Counter, num: Counter) -> dict[int, float]:
    return {int(t): (float(num[t]) / float(d) if d else 0.0) for t, d in denom.items()}


# --------------------------------------------------------------------------
# shared: TSV writing, top-N, binning
# --------------------------------------------------------------------------


def write_tsv(path: Path, rows: Iterable[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("frame_idx\tfraction\n")
        for t, frac in rows:
            f.write(f"{t}\t{frac:.8f}\n")


def report_top_n(fractions: dict[int, float], denom: Counter, num: Counter, *, top_n: int = 20) -> None:
    top_rows = sorted(
        ((t, fractions[t], denom[t], num[t]) for t in fractions.keys()),
        key=lambda x: (-x[1], x[0]),
    )
    print(f"\nTop {top_n} frame indices by fraction:")
    for t, frac, d, n in top_rows[:top_n]:
        print(f"  t={t:6d}  fraction={frac:.6f}  {n}/{d}")


def write_binned_histogram(denom: Counter, num: Counter, *, bin_size: int, out_dir: Path, sign: str) -> None:
    """Weighted binning: fraction_bin = sum_t_in_bin num[t] / sum_t_in_bin denom[t]."""
    b = int(bin_size)
    bin_denom: Counter = Counter()
    bin_num: Counter = Counter()
    for t, d in denom.items():
        k = t // b
        bin_denom[k] += d
        bin_num[k] += num[t]

    b_rows = sorted(
        ((k * b, float(bin_num[k]) / float(bin_denom[k])) for k in bin_denom.keys()),
        key=lambda x: x[0],
    )
    b_path = out_dir / f"hist_sign{sign}_binned_{b}.tsv"
    write_tsv(b_path, b_rows)
    print(f"\nBinned (every {b} frames) wrote: {b_path}")

    print("\nBinned table preview:")
    for t0, frac in b_rows[: min(20, len(b_rows))]:
        print(f"  [{t0:6d}..{t0 + b - 1:6d}]  fraction={frac:.6f}")


# --------------------------------------------------------------------------
# torch-mode sample collection (PlanTDataset)
# --------------------------------------------------------------------------


def _parse_frame_idx_from_label0(label0_path: str | Path) -> int:
    # ".../boxes/0005.json.gz" -> 5
    name = Path(label0_path).name
    m = re.match(r"(\d+)\.", name)
    if not m:
        raise ValueError(f"Unexpected boxes filename: {name!r}")
    return int(m.group(1))


def collect_samples_torch(
    *, ds_root: Path, sign: str, expected_token_id: int, augment: bool,
) -> tuple[list[str], list[int]]:
    """Instantiate PlanTDataset and read its .labels (needs torch + beartype)."""
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

    ds = PlanTDataset(str(ds_root), cfg, shared_dict=None)

    # Verify mapping (sanity check for the "token id" constraint).
    if hasattr(ds, "type_nums") and str(sign) in ds.type_nums:
        token_id = int(ds.type_nums[str(sign)])
        print(f"token_id(type_nums[{sign!r}])={token_id}")
        if token_id != int(expected_token_id):
            print(
                f"WARNING: expected-token-id={expected_token_id} but dataset has token_id={token_id}",
                flush=True,
            )
    else:
        print("WARNING: could not verify token id via ds.type_nums", flush=True)

    labels = getattr(ds, "labels")
    if labels is None:
        raise RuntimeError("Dataset has no .labels")

    label0_paths: list[str] = []
    frame_idxs: list[int] = []
    for i in range(len(ds)):
        lab0 = labels[i][0]
        path = lab0.decode() if isinstance(lab0, (bytes, bytearray)) else str(lab0)
        label0_paths.append(path)
        frame_idxs.append(_parse_frame_idx_from_label0(path))
    return label0_paths, frame_idxs


# --------------------------------------------------------------------------
# no-torch sample collection (hand-rolled route filtering)
# --------------------------------------------------------------------------


def _parse_wps_len_seq_len_from_planT_yaml() -> tuple[int, int]:
    """Hardcoded based on PlanT/config/model/PlanT.yaml shipped in this repo."""
    # waypoints.wps_len: 8
    # training.seq_len: 1
    return 8, 1


def _is_route_trainable(*, route_dir: str, root: str) -> bool:
    """Copy of PlanTDataset init filtering for cfg.model.training.filter_routes=True."""
    route = os.path.basename(route_dir)
    if route.startswith("FAILED_") or not os.path.isfile(route_dir + "/results.json.gz"):
        return False

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


def collect_samples_no_torch(*, ds_root: Path) -> tuple[list[str], list[int]]:
    root = str(ds_root).rstrip("/")
    wps_len, seq_len = _parse_wps_len_seq_len_from_planT_yaml()
    # current frame index t == seq because seq_len == 1
    assert seq_len == 1

    import glob

    label_raw_path_all = glob.glob(os.path.join(root, "**/boxes"), recursive=True)
    route_dirs = [p[:-5] for p in label_raw_path_all]  # strip "/boxes"
    print(f"Found route dirs with boxes: {len(route_dirs)}")

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
        # Match PlanTDataset init: for seq in range(5, num_seq - wps_len - seq_len - 2)
        start = 5
        end = num_seq - wps_len - seq_len - 2
        if end <= start:
            continue
        for seq in range(start, end):
            label0_paths.append(os.path.join(route_dir, "boxes", f"{seq:04d}.json.gz"))
            frame_idxs.append(seq)

    print(f"trainable routes={trainable}  skipped routes={skipped}")
    print(f"samples(tasks)={len(label0_paths)}")
    return label0_paths, frame_idxs


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--ds-root",
        type=Path,
        required=True,
        help="PlanT dump root or .../data/ directory with route dirs containing boxes/",
    )
    ap.add_argument("--sign", type=str, default="2.5", help="Sign code, e.g. 2.5")
    ap.add_argument(
        "--no-torch",
        action="store_true",
        help=(
            "Reimplement PlanTDataset route filtering by hand instead of "
            "instantiating it (use when torch/beartype aren't installed)."
        ),
    )
    ap.add_argument(
        "--expected-token-id",
        type=int,
        default=12,
        help="x_objs type id for --sign, sanity-checked against ds.type_nums (torch mode only)",
    )
    ap.add_argument(
        "--augment",
        action="store_true",
        help="Enable geometric augment (torch mode only; default off)",
    )
    ap.add_argument("--workers", type=int, default=4, help="Process workers for gzip/json scanning")
    ap.add_argument("--chunksize", type=int, default=64, help="Worker chunk size")
    ap.add_argument("--bin-size", type=int, default=0, help="If >0, output binned histogram every N frames")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"ds-root={args.ds_root}")
    print(f"sign={args.sign}  no-torch={args.no_torch}")
    print(f"workers={args.workers}  chunksize={args.chunksize}")
    if not args.no_torch:
        print(f"expected-token-id={args.expected_token_id}  augment={args.augment}")
    if args.bin_size:
        print(f"bin-size={args.bin_size}")

    if args.no_torch:
        label0_paths, frame_idxs = collect_samples_no_torch(ds_root=args.ds_root)
    else:
        label0_paths, frame_idxs = collect_samples_torch(
            ds_root=args.ds_root,
            sign=args.sign,
            expected_token_id=args.expected_token_id,
            augment=args.augment,
        )

    denom = Counter(frame_idxs)  # per-frame count == per-route count (seq_len=1)
    num = scan_sign_presence(label0_paths, frame_idxs, sign=args.sign, workers=args.workers, chunksize=args.chunksize)
    fractions = compute_fractions(denom, num)

    all_rows = sorted(fractions.items(), key=lambda x: x[0])
    out_dir = args.out_dir
    raw_path = out_dir / f"hist_sign{args.sign}_raw.tsv"
    write_tsv(raw_path, all_rows)
    print(f"wrote: {raw_path}")

    report_top_n(fractions, denom, num)

    if args.bin_size and args.bin_size > 0:
        write_binned_histogram(denom, num, bin_size=args.bin_size, out_dir=out_dir, sign=args.sign)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
