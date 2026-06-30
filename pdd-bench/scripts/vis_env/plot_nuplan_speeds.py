"""Plot the nuPlan speed distributions used for sampling.

Produces a PNG with:
  - nuPlan observed speeds (sampler.speeds) in km/h
  - nuPlan initial/spawn speed draws (sample_spawn_velocity) in km/h
Marks the canonical 3.24 limits {20,40,50} for reference.

Run:
  ~/miniconda3/envs/metadrive_sdc/bin/python scripts/vis_env/plot_nuplan_speeds.py \
      --out /Users/victoria_s/sdc_new_signs/braking_3_24_preview/stats/nuplan_speeds.png [--n 8000]
"""
import argparse
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PDD = SCRIPT.parent.parent.parent
for p in (str(PDD), str(PDD / "scripts" / "per_sign_bench")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from factorized_space.agent_profile_bank import _get_sampler, sample_spawn_velocity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--n", type=int, default=8000, help="number of spawn-velocity draws")
    args = ap.parse_args()

    sampler = _get_sampler()
    speeds_kmh = np.asarray(sampler.speeds, dtype=float) * 3.6          # observed speeds
    spawn_kmh = np.array([sample_spawn_velocity(i) for i in range(args.n)]) * 3.6  # initial-speed draws

    def stat(a):
        return (f"n={len(a)} med={np.median(a):.1f} mean={a.mean():.1f} "
                f"p95={np.percentile(a,95):.1f} max={a.max():.1f} km/h")

    print("nuPlan observed speeds:   ", stat(speeds_kmh))
    print("nuPlan initial/spawn draw:", stat(spawn_kmh))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, data, title, color in (
        (axes[0], speeds_kmh, "nuPlan observed speeds", "#3b76af"),
        (axes[1], spawn_kmh, "nuPlan initial/spawn speed (sampled)", "#4f9d69"),
    ):
        ax.hist(data, bins=40, color=color, edgecolor="white")
        for lim in (20, 40, 50):
            ax.axvline(lim, color="#c0504d", ls="--", lw=1, alpha=0.7)
        ax.set_title(f"{title}\nmed={np.median(data):.1f} mean={data.mean():.1f} "
                     f"p95={np.percentile(data,95):.1f} km/h")
        ax.set_xlabel("speed, km/h")
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("nuPlan speed distributions (dashed = 3.24 limits 20/40/50)")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
