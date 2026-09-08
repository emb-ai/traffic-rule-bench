#!/usr/bin/env python3
"""Pack every frame file of a route into one ``pack.bin`` + ``pack.json`` index.

    pack_routes.py <data_dir> [<data_dir> ...] [--workers N]

PlanTDataset reads three to forty small files per sample; on a network share
the latency of those reads, not the bytes, is what starves the GPU. With a
pack per route the dataset maps the file once (``mmap``) and slices frames out
of it, so a split of a few thousand routes costs a few thousand opens per
worker instead of millions of reads, and the operating system's page cache
keeps the packs in memory for every worker and every run on the node.

Packed: boxes/*.json.gz, measurements/*.json.gz, bev_no_car_semantics/*.png,
bev_no_car_semantics_augmented/*.png (if present). Index entries are the path
relative to the route directory -> [offset, length]. A pack older than the
newest frame is rebuilt; others are left alone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SUBDIRS = ("boxes", "measurements", "bev_no_car_semantics", "bev_no_car_semantics_augmented")


def pack(route: str) -> tuple[str, int]:
    r = Path(route)
    files = []
    for sub in SUBDIRS:
        d = r / sub
        if d.is_dir():
            files += sorted(p for p in d.iterdir() if p.is_file())
    if not files:
        return route, -1
    out_bin, out_idx = r / "pack.bin", r / "pack.json"
    if out_bin.is_file() and out_idx.is_file():
        newest = max(f.stat().st_mtime for f in files)
        if out_bin.stat().st_mtime >= newest:
            return route, 0
    index = {}
    tmp = out_bin.with_suffix(".tmp")
    off = 0
    with open(tmp, "wb") as fh:
        for f in files:
            data = f.read_bytes()
            index[f"{f.parent.name}/{f.name}"] = [off, len(data)]
            fh.write(data)
            off += len(data)
    os.replace(tmp, out_bin)
    out_idx.write_text(json.dumps(index, separators=(",", ":")))
    return route, len(index)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    args = ap.parse_args()
    routes = sorted({str(p.resolve()) for d in args.dirs for p in Path(d).iterdir() if p.is_dir() or p.is_symlink()})
    print(f"{len(routes)} routes, {args.workers} workers", flush=True)
    done = written = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for _route, n in pool.map(pack, routes, chunksize=2):
            done += 1
            written += n > 0
            if done % 500 == 0:
                print(f"  {done}/{len(routes)} packed={written}", flush=True)
    print(f"done: {done} routes, {written} packs written", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
