#!/usr/bin/env python3
"""Per-frame sampling weights and metadata for a PlanT2 split.

Training expands each route into one sample per frame and draws them
uniformly. For a speed-limit sign that is the wrong distribution: measured on
good_split_bal, the braking transient runs 7-10 frames out of episodes of
177-1361, so ~6% of frames carry the sign->speed signal. Detour scenes need no
correction -- their cones are visible in 95% of frames and the manoeuvre spans
32% of them.

Writes one ``sample_weights.json`` per split root:

    {"<route>": {"code": "3.24", "plate": 20.0, "side": null,
                 "w": 8.0, "transient": [start, end],
                 "in_zone": [frame, ...], "cones": [frame, ...]}}

``in_zone`` is a proxy for the privileged is_vehicle_in_zone(): the plate is
visible in the frame and sits behind the ego. The zone outlives the 120 m
sign radius, so late frames drop out of it -- report the count, never assume
the zone ended.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SPEED_CODES = ("3.24", "5.31")
# From traffic_signs/detour_sign.py: 4.2.1 passes on the right, 4.2.2 on the
# left, 4.2.3 either. Reading this off the sign number instead of the class is
# how a mirrored metric gets written, so it is copied from there deliberately.
DETOUR_SIDE = {"4.2.1": "right", "4.2.2": "left", "4.2.3": None}


def _code_of(route_name: str) -> str | None:
    for code in list(SPEED_CODES) + list(DETOUR_SIDE):
        if f"_{code}_" in route_name or route_name.startswith(f"sumo_{code}_"):
            return code
    return None


def _read_gz(path: Path):
    try:
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    except Exception:
        return None


def scan_route(args):
    route_dir, settle_pad = args
    route_dir = Path(route_dir)
    name = route_dir.name
    code = _code_of(name)
    if code is None:
        return name, None

    boxes_dir, meas_dir = route_dir / "boxes", route_dir / "measurements"
    frames = sorted(p.name for p in boxes_dir.glob("*.json.gz"))
    if not frames:
        return name, None

    out = {"code": code, "plate": None, "side": DETOUR_SIDE.get(code),
           "w": 1.0, "transient": None, "in_zone": [], "cones": []}

    plate = None
    first_seen = None
    settle = None

    for fname in frames:
        seq = int(fname.split(".")[0])
        boxes = _read_gz(boxes_dir / fname)
        if boxes is None:
            continue

        if code in DETOUR_SIDE:
            if any((b.get("type_id") or "").endswith("constructioncone")
                   and float(b.get("position", [0])[0]) > 0 for b in boxes):
                out["cones"].append(seq)
            continue

        sign = next((b for b in boxes if b.get("class") == code), None)
        if sign is None:
            continue
        value = sign.get("sign_value_kmh")
        if value is None:
            continue
        plate = float(value)
        if first_seen is None:
            first_seen = seq
        if float(sign["position"][0]) < 0:
            out["in_zone"].append(seq)

        if settle is None and first_seen is not None:
            meas = _read_gz(meas_dir / fname)
            if meas is not None and float(meas.get("speed", 0.0)) * 3.6 <= plate:
                settle = seq

    out["plate"] = plate
    if code in SPEED_CODES and first_seen is not None:
        # Braking runs from the frame the plate becomes readable to the frame
        # the ego reaches it, plus a pad. When the ego never drops to the plate
        # the window would otherwise swallow the whole episode and upweight the
        # crawl we are trying to dilute, so anchor it on first_seen instead.
        out["transient"] = [first_seen, (settle if settle is not None else first_seen) + settle_pad]
    return name, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="split root holding train/ and val/")
    ap.add_argument("--weight", type=float, default=8.0,
                    help="multiplier for frames inside the braking transient")
    ap.add_argument("--settle-pad", type=int, default=20,
                    help="frames kept after the ego reaches the plate (measured settle: 7-10)")
    ap.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 8)))
    args = ap.parse_args()

    split = Path(args.split)
    for part in ("train", "val"):
        data = split / part / "data"
        if not data.is_dir():
            print(f"{part}: no {data}, skipping")
            continue
        routes = sorted(p for p in data.iterdir() if p.is_dir() or p.is_symlink())
        print(f"{part}: scanning {len(routes)} routes with {args.workers} workers …", flush=True)

        result = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for name, info in pool.map(scan_route,
                                       [(str(r.resolve()), args.settle_pad) for r in routes],
                                       chunksize=8):
                if info is not None:
                    result[name] = info

        n_speed = n_transient = n_zone = n_cones = 0
        for info in result.values():
            if info["code"] in SPEED_CODES:
                n_speed += 1
                if info["transient"]:
                    info["w"] = args.weight
                    n_transient += 1
                n_zone += len(info["in_zone"])
            else:
                n_cones += len(info["cones"])

        out_path = split / part / "sample_weights.json"
        out_path.write_text(json.dumps(result))
        print(f"{part}: {len(result)} routes  speed={n_speed} with_transient={n_transient} "
              f"in_zone_frames={n_zone} cone_frames={n_cones}  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
