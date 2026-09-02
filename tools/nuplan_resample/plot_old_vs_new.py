#!/usr/bin/env python3
"""Overlaid old (April 2026) vs new (full-mini recompute) statistics plots."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OLD = Path("/Users/victoria_s/sdc_new_signs/nuplan_statistics")
NEW = Path("/Users/victoria_s/sdc_new_signs/nuplan_statistics_v2")
OUT = NEW / "plots"
OUT.mkdir(exist_ok=True)

C_OLD = "#8a8f98"       # old — neutral gray (reference)
C_NEW = "#2563eb"       # new — accent blue
C_EGO = "#0e9f6e"
INK = "#1f2430"
MUT = "#6b7280"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d7dade", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#eceef1", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "text.color": INK, "axes.labelcolor": MUT,
    "xtick.color": MUT, "ytick.color": MUT,
    "legend.frameon": False,
})


def dens_hist(ax, old, new, bins, xlabel):
    """Two density-normalized step profiles + median markers."""
    ho, eo = np.histogram(old, bins=bins, density=True)
    hn, en = np.histogram(new, bins=bins, density=True)
    ax.fill_between(eo[:-1], ho, step="post", color=C_OLD, alpha=0.28, lw=0)
    ax.step(eo[:-1], ho, where="post", color=C_OLD, lw=1.6)
    ax.fill_between(en[:-1], hn, step="post", color=C_NEW, alpha=0.22, lw=0)
    ax.step(en[:-1], hn, where="post", color=C_NEW, lw=1.8)
    ymax = max(ho.max(), hn.max())
    ax.axvline(np.median(old), color=C_OLD, lw=1.0, ls=(0, (3, 2)))
    ax.axvline(np.median(new), color=C_NEW, lw=1.0, ls=(0, (3, 2)))
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.set_ylim(0, ymax * 1.12)
    ax.set_xlim(bins[0], bins[-1])


def col(d, f, c):
    return pd.read_csv(d / f)[c].dropna().to_numpy()


fig, axes = plt.subplots(3, 3, figsize=(13.2, 10.4))
fig.subplots_adjust(hspace=0.52, wspace=0.18, top=0.875, bottom=0.06,
                    left=0.045, right=0.985)

# 1. speed
ax = axes[0, 0]
dens_hist(ax, col(OLD, "speeds.csv", "speed"), col(NEW, "speeds.csv", "speed"),
          np.linspace(0, 22, 60), "m/s")
ax.set_title("Vehicle speed")

# 2. acceleration
ax = axes[0, 1]
dens_hist(ax, col(OLD, "acc_pos.csv", "acceleration"),
          col(NEW, "acc_pos.csv", "acceleration"),
          np.linspace(0, 4.5, 60), "m/s²")
ax.set_title("Acceleration")

# 3. deceleration
ax = axes[0, 2]
dens_hist(ax, col(OLD, "acc_neg.csv", "deceleration"),
          col(NEW, "acc_neg.csv", "deceleration"),
          np.linspace(0, 4.5, 60), "m/s²")
ax.set_title("Deceleration")

# 4. following distance
ax = axes[1, 0]
dens_hist(ax, col(OLD, "following.csv", "following_distance"),
          col(NEW, "following.csv", "following_distance"),
          np.linspace(0, 90, 60), "m")
ax.set_title("Following distance")

# 5. traffic density: old-compatible count + corrected variant
ax = axes[1, 1]
dn = pd.read_csv(NEW / "densities.csv")
bins = np.arange(0, 105, 2)
dens_hist(ax, col(OLD, "densities.csv", "count"), dn["count"].to_numpy(),
          bins, "vehicles/frame")
hm, em = np.histogram(dn["count_moving_r50"].dropna(), bins=bins, density=True)
ax.step(em[:-1], hm, where="post", color=C_NEW, lw=1.4, ls=(0, (4, 2)),
        label="new: moving, ≤50 m of ego")
ax.set_ylim(0, max(ax.get_ylim()[1], hm.max() * 1.1))
ax.legend(loc="upper right", fontsize=8)
ax.set_title("Traffic density")

# 6. route length: tracks (annotation window) + ego per scene
ax = axes[1, 2]
ro = col(OLD, "routes.csv", "distance")
rn = col(NEW, "routes.csv", "distance")
bins = np.linspace(0, 400, 60)
dens_hist(ax, ro, rn, bins, "m")
ego = pd.read_csv(NEW / "ego_routes.csv")
egos = ego[ego["kind"] == "scene"]["distance"].to_numpy()
he, ee = np.histogram(egos, bins=bins, density=True)
ax.step(ee[:-1], he, where="post", color=C_EGO, lw=1.6,
        label="ego route per scene")
ax.legend(loc="upper right", fontsize=8)
ax.set_title("Route length (tracks — annotation window)")

# 7. track duration
ax = axes[2, 0]
dens_hist(ax, col(OLD, "routes.csv", "duration"),
          col(NEW, "routes.csv", "duration"),
          np.linspace(0, 60, 60), "s")
ax.set_title("Track duration in window")

# 8. initial speed (spawn)
ax = axes[2, 1]
dens_hist(ax, col(OLD, "routes.csv", "initial_speed"),
          col(NEW, "routes.csv", "initial_speed"),
          np.linspace(0, 20, 60), "m/s")
ax.set_title("Track initial speed (spawn_velocity)")

# 9. lane-change rate
ax = axes[2, 2]
rate_old = len(pd.read_csv(OLD / "lane_changes.csv")) / (ro.sum() / 1000)
rate_new = len(pd.read_csv(NEW / "lane_changes.csv")) / (rn.sum() / 1000)
ax.barh([1, 0], [rate_old, rate_new], height=0.55,
        color=[C_OLD, C_NEW], alpha=0.9)
ax.set_yticks([1, 0], ["old", "new"])
ax.set_xscale("log")
ax.set_xlim(0.5, 130)
ax.text(rate_old * 0.9, 1, f"{rate_old:.1f}/km  ", va="center", ha="right",
        fontsize=9, color="white", fontweight="bold")
ax.text(rate_new, 0, f"  {rate_new:.2f}/km", va="center", fontsize=9, color=INK)
ax.set_xlabel("lane changes per km (log scale)")
ax.set_title("Lane-change rate")
ax.grid(axis="y", visible=False)

fig.suptitle("nuPlan statistics for per_sign_bench: old (April 2026, ~31 min of logs) "
             "vs new (full mini split, 7.2 h)", fontsize=13, fontweight="bold",
             y=0.985)
handles = [plt.Rectangle((0, 0), 1, 1, fc=C_OLD, alpha=0.5),
           plt.Rectangle((0, 0), 1, 1, fc=C_NEW, alpha=0.5)]
fig.legend(handles, ["old", "new"], loc="upper center",
           bbox_to_anchor=(0.5, 0.955), ncol=2, fontsize=10.5)

fig.savefig(OUT / "old_vs_new_grid.png", dpi=170)
print("saved:", OUT / "old_vs_new_grid.png")
