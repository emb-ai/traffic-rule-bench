#!/usr/bin/env python3
"""Calibrate `traffic_density` against nuPlan, measured in SumoTrafficManager.

`calibrate_density_metadrive.py` measured PG maps under MetaDrive's own trigger
manager. That is not what the benchmark runs: its scenes are SUMO networks and
its NPCs come from `SumoTrafficManager`, which fills slots per lane as
`n = clamp(int(density * lane_len / 12), 1, 6)`. The per-lane cap saturates, so
a density read off the MetaDrive curve does not produce the traffic it promises
here -- and the three tiers it produced (0.2 / 0.3125 / 0.425) all landed
between nuPlan's p35 and p70, which is why the benchmark's scenes all carried
roughly the same traffic.

So sweep `traffic_density` over the benchmark's own scenes, count in the
simulator the quantity nuPlan reports (`count_moving_r150_per_lane`: moving NPCs
within 150 m of the ego, divided by the lanes of the ego's road), and invert the
measured curve by quantile matching. The result is a sampling table `u ->
traffic_density` that `traffic_bench/eval/engine/traffic/traffic_density_levels.py`
reads at scene-expansion time.

Two steps, because the first one needs a GPU node and takes hours while the
second is arithmetic:

    # 1. measure -- writes probe_<density>.jsonl.* into --work
    python3 tools/nuplan_resample/calibrate_density_sumo.py sweep \
        --manifest data/runs_v2/speed_limit/test --work /tmp/dens_sweep

    # 2. fit -- writes the calibration the benchmark reads, and the figure
    python3 tools/nuplan_resample/calibrate_density_sumo.py fit \
        --work /tmp/dens_sweep \
        --densities-csv /path/to/stats_r150/densities.csv \
        --out traffic_bench/eval/engine/traffic/nuplan_statistics/density_calibration_sumo.json \
        --plot reports/density_calibration_sumo.png

The counting itself lives in the benchmark, not here: `TrafficManager` writes one
JSON line per step when `TRB_DENSITY_PROBE` names an output path (see
`traffic_bench/envs/traffic.py`). `sweep` only sets that variable and drives the
runs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

DEFAULT_DENSITIES = (0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0)
DEFAULT_ROWS = 12
DEFAULT_JOBS = 8
NUPLAN_COLUMN = "count_moving_r150_per_lane"
MANIFEST_FILES = ("config.yaml", "manifest.json", "real_manifest_summary.json")


# --------------------------------------------------------------------------
# step 1: measure the simulator's response
# --------------------------------------------------------------------------

def _write_manifest_at_density(src: Path, dst: Path, density: float, rows: int) -> int:
    """Copy the first `rows` manifest rows with `traffic_density` overridden."""
    out = []
    with (src / "real_manifest.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["traffic_density"] = density
            # The tiers these named are gone; leave the fields so a row written
            # by this script stays loadable by the same manifest reader.
            r["traffic_density_level_id"] = None
            r["traffic_density_level_name"] = None
            out.append(r)
            if len(out) >= rows:
                break
    dst.mkdir(parents=True, exist_ok=True)
    for name in MANIFEST_FILES:
        if (src / name).is_file():
            shutil.copy(src / name, dst / name)
    with (dst / "real_manifest.jsonl").open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    return len(out)


def cmd_sweep(args) -> int:
    src = Path(args.manifest).resolve()
    work = Path(args.work).resolve()
    if not (src / "real_manifest.jsonl").is_file():
        print(f"no real_manifest.jsonl under {src}", file=sys.stderr)
        return 2
    work.mkdir(parents=True, exist_ok=True)

    for d in args.densities:
        cell = work / f"d{d}"
        n = _write_manifest_at_density(src, cell, d, args.rows)
        probe = work / f"probe_{d}.jsonl"
        env = dict(os.environ, TRB_DENSITY_PROBE=str(probe))
        cmd = [sys.executable, "-m", "traffic_bench.eval", "run",
               f"manifest={cell}", f"policy={args.policy}", f"jobs={args.jobs}",
               f"run_name={args.run_name}"]
        print(f"[density={d}] {n} rows -> {' '.join(cmd)}", flush=True)
        with (work / f"eval_{d}.log").open("w") as log:
            rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        got = sum(1 for _ in _probe_lines(work, only=d))
        print(f"[density={d}] rc={rc} samples={got}", flush=True)
    return 0


# --------------------------------------------------------------------------
# step 2: fit the sampling table
# --------------------------------------------------------------------------

def _probe_lines(work: Path, only: float | None = None):
    """Every probe record the sweep left behind.

    Each worker appends its own suffix to the path the env var names, so one
    density leaves several files.
    """
    pattern = f"probe_{only}.jsonl*" if only is not None else "probe_*.jsonl*"
    for path in sorted(glob.glob(str(work / pattern))):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _sim_curve(work: Path) -> dict[float, np.ndarray]:
    per_density: dict[float, list[float]] = {}
    for r in _probe_lines(work):
        per_density.setdefault(round(float(r["density"]), 4), []).append(
            float(r["per_lane"]))
    return {k: np.asarray(v, dtype=float) for k, v in sorted(per_density.items())}


def _nuplan_target(csv_path: Path) -> np.ndarray:
    import pandas as pd

    col = pd.read_csv(csv_path)[NUPLAN_COLUMN].to_numpy(dtype=float)
    return col[np.isfinite(col)]


def cmd_fit(args) -> int:
    work = Path(args.work).resolve()
    curve = _sim_curve(work)
    if not curve:
        print(f"no probe records under {work}; run `sweep` first", file=sys.stderr)
        return 2
    target = _nuplan_target(Path(args.densities_csv))

    ds = np.array(list(curve))
    med = np.array([np.percentile(curve[d], 50) for d in ds])
    q25 = np.array([np.percentile(curve[d], 25) for d in ds])
    q75 = np.array([np.percentile(curve[d], 75) for d in ds])

    # Quantile matching: for a uniform draw u take nuPlan's u-quantile, then
    # invert the measured curve at it. np.interp clamps outside the range the
    # simulator can reach, which the calibration records as reachable_fraction.
    order = np.argsort(med)
    med_s, ds_s = med[order], ds[order]
    us = np.linspace(0.01, 0.99, 99)
    target_q = np.percentile(target, us * 100)
    density_q = np.interp(target_q, med_s, ds_s)
    reachable = (target_q >= med_s.min()) & (target_q <= med_s.max())

    calibration = {
        "nuplan_per_lane": {str(q): float(np.percentile(target, q))
                            for q in (5, 10, 25, 50, 75, 90, 95)},
        "sim_curve": {str(float(d)): {"n": int(curve[d].size),
                                      "p25": float(np.percentile(curve[d], 25)),
                                      "p50": float(np.percentile(curve[d], 50)),
                                      "p75": float(np.percentile(curve[d], 75)),
                                      "mean": float(curve[d].mean())} for d in ds},
        "sampling_table": {"u": [round(float(x), 4) for x in us],
                           "density": [round(float(x), 4) for x in density_q]},
        "reachable_fraction": float(reachable.mean()),
        "method": {
            "metric": f"moving (>0.5 m/s) NPCs within 150 m of the ego, divided "
                      f"by the lanes on the ego's road ({NUPLAN_COLUMN})",
            "env": "the benchmark's own SUMO scenes via traffic_bench.eval run, "
                   "policy=idm, SumoTrafficManager -- not PG maps under "
                   "MetaDrive's trigger manager, which the previous "
                   "calibration used",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calibration, indent=1) + "\n")
    print("wrote", out)

    print(f"reachable fraction: {reachable.mean():.2f}")
    for q in (10, 25, 50, 75, 90):
        t = float(np.percentile(target, q))
        if t > med_s.max():
            print(f"  nuPlan p{q:<3d} = {t:5.2f} cars/lane -> above what the "
                  f"simulator reaches ({med_s.max():.2f})")
        else:
            print(f"  nuPlan p{q:<3d} = {t:5.2f} cars/lane -> density "
                  f"{float(np.interp(t, med_s, ds_s)):.3f}")

    if args.plot:
        _plot(target, ds, med, q25, q75, us, density_q, reachable, Path(args.plot))
        print("wrote", args.plot)
    return 0


def _plot(target, ds, med, q25, q75, us, density_q, reachable, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    ax[0].hist(target, bins=np.arange(0, 16, 0.5), color="#4C72B0",
               edgecolor="white")
    for q, c in [(25, "#DD8452"), (50, "#C44E52"), (75, "#DD8452")]:
        ax[0].axvline(np.percentile(target, q), color=c, ls="--", lw=1.4)
    ax[0].set_xlim(0, 15)
    ax[0].set_title("nuPlan: moving cars per lane within 150 m\n"
                    "p25=%.2f  p50=%.2f  p75=%.2f"
                    % tuple(np.percentile(target, [25, 50, 75])))
    ax[0].set_xlabel("cars per lane")
    ax[0].set_ylabel("frames")

    ax[1].fill_between(ds, q25, q75, alpha=0.25, color="#4C72B0", label="p25-p75")
    ax[1].plot(ds, med, "o-", color="#4C72B0", label="median")
    for q, lab in [(25, "nuPlan p25"), (50, "nuPlan p50"), (75, "nuPlan p75")]:
        ax[1].axhline(np.percentile(target, q), color="#C44E52", ls="--", lw=1.0)
        ax[1].text(ds.max(), np.percentile(target, q), " " + lab, va="center",
                   fontsize=8, color="#C44E52")
    ax[1].set_title("SumoTrafficManager response\n"
                    "(measured on the benchmark's own scenes)")
    ax[1].set_xlabel("traffic_density")
    ax[1].set_ylabel("cars per lane within 150 m")
    ax[1].legend(fontsize=8)

    ax[2].plot(us, density_q, "-", color="#55A868", lw=2)
    ax[2].fill_between(us, 0, density_q, where=~reachable, color="#C44E52",
                       alpha=0.15, label="outside the reachable range (clamped)")
    ax[2].set_xlabel("uniform draw u")
    ax[2].set_ylabel("traffic_density")
    ax[2].set_title("Sampling table: u -> traffic_density\n"
                    "%d%% of the target range is reachable"
                    % int(100 * reachable.mean()))
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("sweep", help="measure the simulator's density response")
    sw.add_argument("--manifest", required=True,
                    help="an expanded manifest directory to reuse the scenes of")
    sw.add_argument("--work", required=True, help="where probe records are written")
    sw.add_argument("--densities", type=float, nargs="+", default=list(DEFAULT_DENSITIES))
    sw.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                    help="manifest rows per density (default %(default)s)")
    sw.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    sw.add_argument("--policy", default="idm")
    sw.add_argument("--run-name", default="densprobe")
    sw.set_defaults(func=cmd_sweep)

    ft = sub.add_parser("fit", help="quantile-match against nuPlan and write the table")
    ft.add_argument("--work", required=True, help="the sweep's output directory")
    ft.add_argument("--densities-csv", required=True,
                    help=f"densities.csv carrying {NUPLAN_COLUMN}")
    ft.add_argument("--out", required=True, help="calibration json to write")
    ft.add_argument("--plot", default=None, help="optional figure to write")
    ft.set_defaults(func=cmd_fit)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
