#!/usr/bin/env python3
"""Synthesize the no-traffic half of the pair from the re-dumped frames.

The baseline dump and the re-dump replay the SAME recorded episodes; the only
difference the auxiliary convoy makes in the dumped data is extra vehicle
entries in boxes/*.json.gz. Dropping every vehicle except the ego row (always
boxes[0], id 0, at the origin) therefore reproduces the baseline condition
exactly — same ego trace, same labels, same signs — which is all the 'old'
half of the experiment needs. Useful on nodes that cannot see the original
baseline dump at all.

Everything except boxes/ is hardlinked, so the tree costs no disk.

  python3 make_old_half_from_new.py --new-dump <dump> --out <old_dump> [--jobs 16]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def is_ego(entry: dict) -> bool:
    if entry.get("id") == 0:
        return True
    pos = entry.get("position") or []
    return len(pos) >= 2 and abs(pos[0]) < 1e-6 and abs(pos[1]) < 1e-6


def strip_route(src: Path, dst: Path) -> tuple[int, int]:
    """Return (frames, vehicles_dropped)."""
    dropped = 0
    frames = 0
    for item in src.iterdir():
        out = dst / item.name
        if item.name == "boxes":
            out.mkdir(parents=True, exist_ok=True)
            for f in item.iterdir():
                frames += 1
                entries = json.load(gzip.open(f))
                kept = []
                for e in entries:
                    if e.get("class") == "car" and not is_ego(e):
                        dropped += 1
                        continue
                    kept.append(e)
                with gzip.open(out / f.name, "wt", encoding="utf-8") as fh:
                    json.dump(kept, fh)
        elif item.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            for f in item.iterdir():
                if not (out / f.name).exists():
                    os.link(f, out / f.name)
        else:
            if not out.exists():
                os.link(item, out)
    return frames, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    src_data = Path(args.new_dump) / "data"
    out_data = Path(args.out) / "data"
    if not src_data.is_dir():
        raise SystemExit(f"missing {src_data}")
    out_data.mkdir(parents=True, exist_ok=True)

    routes = sorted(p for p in src_data.iterdir() if p.is_dir())
    print(f"stripping vehicles from {len(routes)} route(s) -> {out_data}", flush=True)

    total_frames = total_dropped = done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(strip_route, r, out_data / r.name): r for r in routes}
        for fut in as_completed(futs):
            frames, dropped = fut.result()
            total_frames += frames
            total_dropped += dropped
            done += 1
            if done % 100 == 0 or done == len(routes):
                print(f"  {done}/{len(routes)} routes, frames={total_frames}, "
                      f"vehicle entries dropped={total_dropped}", flush=True)

    if total_dropped == 0:
        print("!! no vehicle entries were dropped — the source dump has no "
              "traffic, so this 'old' half would be identical to 'new'")
    print(f"DONE routes={len(routes)} frames={total_frames} dropped={total_dropped}")


if __name__ == "__main__":
    main()
