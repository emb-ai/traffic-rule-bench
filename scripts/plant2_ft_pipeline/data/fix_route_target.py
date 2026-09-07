#!/usr/bin/env python3
"""Rebuild a PlanT2 split whose path TARGET is the ego's driven trajectory.

Why
---
The dump writes the SAME array into both fields
(``bench/plant2_frames.py``: ``"route": route_pts`` and
``"route_original": route_pts``, both from ``get_route(vehicle)``), so the
model's route INPUT is byte-identical to its path TARGET. Copying the input
is then *exactly* optimal, and that is the identity mapping the model learns
-- measured: shifting the input route by +3 m shifts the prediction by
+2.73 m, and zeroing it collapses the prediction to 0.21 m. In the simulator
the route is the real SUMO plan, which runs straight through the obstacle,
so the model copies it and drives in.

Upstream PlanT 2.0 keeps these distinct: ``route_original`` is the planner's
untouched route while ``route`` is the (possibly deviated) route the expert
follows, and it even records a ``changed_route`` flag marking the deviation
(``Bench2Drive/leaderboard/team_code/autopilot.py``).

What this does
--------------
``route_original`` is already correct -- ``get_route()`` returns the
navigation route, i.e. the lane centreline ahead, which is obstacle-unaware.
Only the target is wrong, and it can be rebuilt offline: every frame already
stores ``pos_global`` and ``theta``, so the driven trajectory is recoverable
without re-running the simulator.

For frame i we take the future ego positions, express them in frame i's ego
frame (x=forward, y=left -- the convention ``get_route()`` writes, verified
empirically: recomputed trajectory matches ``route_original`` to 0.054 m on
frames far from any obstacle, vs 0.080 m under y=right), resample by arc
length every ``step_m`` and keep ``num_points``.

Heavy files (boxes / BEV / results) are symlinked, so the shared source
dumps are never modified and the copy costs only the measurements.

Usage:
  fix_route_target.py --src <split_dir> --out <new_split_dir> [--jobs N]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

LINK_ENTRIES = ("boxes", "bev_no_car_semantics", "bev_no_car_semantics_augmented",
                "results.json.gz")


def trajectory_ego_frame(pos, theta, i, num_points=20, step_m=1.0):
    """Future ego path from frame i, in frame i's ego frame (x=fwd, y=left)."""
    p0, th = pos[i], theta[i]
    fwd = np.array([math.cos(th), math.sin(th)])
    left = np.array([-math.sin(th), math.cos(th)])
    rel = pos[i:] - p0
    loc = np.stack([rel @ fwd, rel @ left], axis=1)
    seg = np.linalg.norm(np.diff(loc, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])

    out = []
    for k in range(1, num_points + 1):
        t = k * step_m
        if t > arc[-1]:
            # Route ran out (end of episode): hold the last known point rather
            # than extrapolating a trajectory we never observed.
            out.append(out[-1] if out else [0.0, 0.0])
            continue
        j = int(np.searchsorted(arc, t))
        span = arc[j] - arc[j - 1]
        a = (t - arc[j - 1]) / span if span > 1e-9 else 0.0
        out.append((loc[j - 1] + a * (loc[j] - loc[j - 1])).tolist())
    return out


def convert_route(src_route: Path, out_route: Path, num_points: int, step_m: float) -> int:
    out_route.mkdir(parents=True, exist_ok=True)
    for name in LINK_ENTRIES:
        s, d = src_route / name, out_route / name
        if s.exists() and not d.exists():
            d.symlink_to(s.resolve())

    files = sorted((src_route / "measurements").glob("*.json.gz"))
    if not files:
        return 0
    meas = []
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            meas.append(json.load(fh))
    pos = np.array([m["pos_global"] for m in meas], dtype=np.float64)
    theta = np.array([m["theta"] for m in meas], dtype=np.float64)

    dst = out_route / "measurements"
    dst.mkdir(exist_ok=True)
    for i, (f, m) in enumerate(zip(files, meas)):
        m["route"] = trajectory_ego_frame(pos, theta, i, num_points, step_m)
        # route_original is left exactly as dumped (navigation route).
        with gzip.open(dst / f.name, "wt", encoding="utf-8") as fh:
            json.dump(m, fh)
    return len(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-points", type=int, default=20)
    ap.add_argument("--step-m", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    # run_plant2_finetune.py requires this at the split root.
    for meta in ("split_meta.json", "sample_weights.json"):
        s, d = args.src / meta, args.out / meta
        if s.exists() and not d.exists():
            d.symlink_to(s.resolve())

    tasks = []
    for part in ("train", "val"):
        src_data = args.src / part / "data"
        if not src_data.is_dir():
            continue
        for extra in ("slurm",):
            s, d = args.src / part / extra, args.out / part / extra
            if s.exists() and not d.exists():
                d.parent.mkdir(parents=True, exist_ok=True)
                d.symlink_to(s.resolve())
        for route in sorted(src_data.iterdir()):
            if (route / "boxes").is_dir():
                tasks.append((route, args.out / part / "data" / route.name))

    print(f"routes to convert: {len(tasks)}")
    done = frames = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(convert_route, s, d, args.num_points, args.step_m): s
                for s, d in tasks}
        for fut in as_completed(futs):
            frames += fut.result()
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(tasks)} routes, {frames} frames", flush=True)
    print(f"DONE: {done} routes, {frames} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
