#!/usr/bin/env python3
"""Write ``measurements_all.json.gz`` (frame stem -> measurement) into every route.

    consolidate_measurements.py <data_dir> [<data_dir> ...] [--workers N]

PlanTDataset reads seq_len + future_frames measurement files per sample; with
the realised-future path target that is 41 files, and on a network share the
loader spends ~250 ms per sample on latency alone. The consolidated file is
read once per route per worker (dataset.py keeps an LRU), which brings a sample
down to three reads. Existing consolidated files are left alone unless they are
older than the newest measurement frame.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def consolidate(route: str) -> tuple[str, int]:
    r = Path(route)
    meas = r / "measurements"
    if not meas.is_dir():
        return route, -1
    out = r / "measurements_all.json.gz"
    frames = sorted(meas.glob("*.json.gz"))
    if not frames:
        return route, 0
    if out.is_file() and out.stat().st_mtime >= max(f.stat().st_mtime for f in frames):
        return route, 0
    table = {}
    for f in frames:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            table[f.name.split(".")[0]] = json.load(fh)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=5) as fh:
        json.dump(table, fh)
    os.replace(tmp, out)
    return route, len(table)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    args = ap.parse_args()
    routes = []
    for d in args.dirs:
        for p in sorted(Path(d).iterdir()):
            if p.is_dir() or p.is_symlink():
                routes.append(str(p.resolve()))
    routes = sorted(set(routes))
    print(f"{len(routes)} routes, {args.workers} workers", flush=True)
    done = written = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for route, n in pool.map(consolidate, routes, chunksize=4):
            done += 1
            if n > 0:
                written += 1
            if done % 500 == 0:
                print(f"  {done}/{len(routes)} written={written}", flush=True)
    print(f"done: {done} routes, {written} consolidated files written", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
