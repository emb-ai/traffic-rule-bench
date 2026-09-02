#!/usr/bin/env python3
"""Calibrate MetaDrive's traffic_density against nuPlan.

nuPlan gives a distribution of "moving cars within 50 m of the ego"
(densities.csv, column count_moving_r50; targets p25/50/75 = 5/11/18).
MetaDrive takes traffic_density in [0,1], the share of spawn slots it fills.
The relation between the two is not known analytically -- the old /80 divisor
was fitted against a different definition, every annotated car in the frame.

So run MetaDrive over a grid of traffic_density, measure the same quantity IN
THE SIMULATOR (moving NPCs within 50 m of an IDM-driven ego), build the
density -> count curve, and interpolate the density levels at which the
simulator reproduces nuPlan's p25/50/75.

Run:
  python3 calibrate_density_metadrive.py \
      --out $SM/nuplan/density_calibration.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

DENSITIES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
SEEDS = [0, 1, 2]              # different PG maps
WARMUP_STEPS = 50
SAMPLE_STEPS = 400             # steps after warm-up
SAMPLE_EVERY = 5               # count every 0.5 s (the bench runs physics at 10 Hz)
RADIUS = 50.0                  # m, matching nuPlan's count_moving_r50
MOVING_KMH = 1.8               # 0.5 m/s, the same "is moving" threshold
NUPLAN_TARGETS = {"p25": 5.0, "p50": 11.0, "p75": 18.0}   # count_moving_r50


def run_episode(density: float, seed: int) -> list[int]:
    """One MetaDrive episode: the counts of moving NPCs within 50 m of the ego."""
    from metadrive.envs import MetaDriveEnv
    from metadrive.policy.idm_policy import IDMPolicy
    from metadrive.component.vehicle.base_vehicle import BaseVehicle

    env = MetaDriveEnv(dict(
        use_render=False,
        map=4,                          # four random blocks per seed
        start_seed=seed,
        traffic_density=density,
        agent_policy=IDMPolicy,         # the ego drives itself, like a bench NPC
        horizon=WARMUP_STEPS + SAMPLE_STEPS + 100,
        log_level=50,
    ))
    counts = []
    try:
        env.reset(seed=seed)
        ego = env.agent
        for step in range(WARMUP_STEPS + SAMPLE_STEPS):
            _, _, term, trunc, _ = env.step([0.0, 0.0])   # IDM ignores the action
            if term or trunc:
                break
            if step < WARMUP_STEPS or step % SAMPLE_EVERY:
                continue
            ex, ey = ego.position
            n = 0
            for v in env.engine.get_objects(
                    lambda o: isinstance(o, BaseVehicle)).values():
                if v is ego:
                    continue
                dx, dy = v.position[0] - ex, v.position[1] - ey
                if dx * dx + dy * dy <= RADIUS * RADIUS and v.speed_km_h > MOVING_KMH:
                    n += 1
            counts.append(n)
    finally:
        env.close()
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    curve = {}
    for d in DENSITIES:
        allc = []
        for s in SEEDS:
            t0 = time.time()
            c = run_episode(d, s)
            allc += c
            print(f"density={d:.2f} seed={s}: n={len(c)} "
                  f"median={np.median(c) if c else float('nan'):.1f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        allc = np.asarray(allc)
        curve[d] = {"n": int(len(allc)),
                    "p25": float(np.percentile(allc, 25)),
                    "p50": float(np.percentile(allc, 50)),
                    "p75": float(np.percentile(allc, 75)),
                    "mean": float(allc.mean())}

    # interpolate: which density yields nuPlan's count, along the median curve
    ds = np.array(sorted(curve))
    med = np.array([curve[d]["p50"] for d in ds])
    levels = {}
    for name, target in NUPLAN_TARGETS.items():
        levels[name] = float(np.interp(target, med, ds))
    # the effective divisor in raw/K at each point
    divisors = {name: NUPLAN_TARGETS[name] / levels[name] if levels[name] > 0
                else float("nan") for name in levels}

    result = {
        "sim_curve": {str(d): curve[d] for d in ds},
        "nuplan_targets_count_moving_r50": NUPLAN_TARGETS,
        "calibrated_density_levels": levels,
        "effective_divisor_raw_over_density": divisors,
        "method": {
            "metric": "moving (>0.5 m/s) NPCs within 50 m of the ego",
            "env": "MetaDriveEnv, map=4 blocks, default traffic_mode (trigger), "
                   "ego = IDMPolicy, 3 seeds, 400 steps after 50 of warm-up, counted every 5 steps",
            "caveat": "NPC IDM parameters are the defaults, not nuPlan profiles; "
                      "other bench map types (SUMO/CityMap) may shift the curve",
        },
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result["calibrated_density_levels"], indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
