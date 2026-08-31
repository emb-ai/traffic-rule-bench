#!/usr/bin/env python3
"""Keep the speed routes where the sign actually constrains the ego.

Two conditions, both measured from the dump itself:

1. the plate is readable -- the sign reached the frames at all. Measured on the
   150-route r120 dump, 54% of routes pass; on the older 30 m dump, none do.
2. the ego goes above 0.8 x plate at some point after passing the sign. Where
   it never does, the limit never binds and the frames teach nothing about the
   sign: 3.24-at-40 scenes score 0.0% on this and are pure noise.

Writes a report and, with --link-into, a directory of symlinks ready for
make_train_val_split_fv_experts_signs.py.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SPEED_CODES = ("3.24", "5.31", "4.6")


def _code_of(name: str) -> str | None:
    for code in SPEED_CODES:
        if name.startswith(f"sumo_{code}_") or f"_{code}_" in name:
            return code
    return None


def _read_gz(path: Path):
    try:
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    except Exception:
        return None


def judge(args):
    route_dir, ratio = args
    route = Path(route_dir)
    code = _code_of(route.name)
    if code is None:
        return route.name, "not_a_speed_route", None, 0.0

    boxes_dir, meas_dir = route / "boxes", route / "measurements"
    frames = sorted(p.name for p in boxes_dir.glob("*.json.gz"))
    if not frames:
        return route.name, "no_frames", None, 0.0

    plate = None
    peak_after = 0.0
    seen_sign = False
    for fname in frames:
        boxes = _read_gz(boxes_dir / fname)
        if boxes is None:
            continue
        sign = next((b for b in boxes if b.get("class") == code), None)
        if sign is None:
            continue
        value = sign.get("sign_value_kmh")
        if value is None:
            continue
        plate = float(value)
        seen_sign = True
        if float(sign["position"][0]) < 0:
            meas = _read_gz(meas_dir / fname)
            if meas is not None:
                peak_after = max(peak_after, float(meas.get("speed", 0.0)) * 3.6)

    if not seen_sign:
        return route.name, "plate_unreadable", None, 0.0
    if peak_after <= ratio * plate:
        return route.name, "limit_never_binds", plate, peak_after
    return route.name, "keep", plate, peak_after


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="dump root holding data/")
    ap.add_argument("--ratio", type=float, default=0.8,
                    help="ego must exceed ratio*plate after the sign")
    ap.add_argument("--link-into", help="directory to fill with symlinks to the kept routes")
    ap.add_argument("--report", help="write the full per-route verdict here as JSON")
    ap.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 8)))
    args = ap.parse_args()

    data = Path(args.dump) / "data"
    if not data.is_dir():
        print(f"no {data}", file=sys.stderr)
        return 2
    routes = sorted(p for p in data.iterdir() if p.is_dir() or p.is_symlink())
    print(f"judging {len(routes)} routes with {args.workers} workers …", flush=True)

    verdicts = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, verdict, plate, peak in pool.map(
                judge, [(str(r.resolve()), args.ratio) for r in routes], chunksize=8):
            verdicts[name] = {"verdict": verdict, "plate": plate, "peak_after_kmh": round(peak, 1)}

    tally = Counter(v["verdict"] for v in verdicts.values())
    total = max(1, len(verdicts))
    print()
    for verdict, n in tally.most_common():
        print(f"  {verdict:20} {n:5d}  ({100 * n / total:.1f}%)")

    kept = [n for n, v in verdicts.items() if v["verdict"] == "keep"]
    by_plate = Counter(v["plate"] for v in verdicts.values() if v["verdict"] == "keep")
    print(f"\nkept {len(kept)} routes; by plate: "
          f"{ {int(k): c for k, c in sorted(by_plate.items()) if k} }")

    if args.report:
        Path(args.report).write_text(json.dumps(verdicts, indent=1))
        print(f"report -> {args.report}")

    if args.link_into:
        out = Path(args.link_into)
        out.mkdir(parents=True, exist_ok=True)
        made = 0
        for name in kept:
            link = out / name
            if link.is_symlink() or link.exists():
                continue
            link.symlink_to((data / name).resolve())
            made += 1
        print(f"linked {made} routes -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
