"""Rebuild reviewer-evidence figures in the site's dark palette.

Data source (single source of truth):
  pdd-bench/scripts/per_sign_bench/benchmark_output/ready_test_summary/reviewer_evidence/tables/
Outputs:
  docs/static/images/figures/corr_heatmap_dark.png
  docs/static/images/figures/rcr_ci_forest_dark.png
  docs/static/images/figures/twin_gap_dark.png
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable

DOCS = Path(__file__).resolve().parent
TABLES = (DOCS / "../pdd-bench/scripts/per_sign_bench/benchmark_output/"
          "ready_test_summary/reviewer_evidence/tables").resolve()
OUT = DOCS / "static/images/figures"

# --- site palette ---
BG = "#131a2e"
TEXT = "#e8ecf8"
DIM = "#9aa5c4"
GRID = "#2a3454"
RED = "#f87171"
GREEN = "#34d399"
TEAL = "#2dd4bf"
ROSE = "#fb7185"
SKY = "#38bdf8"


def mix(c, t, base=BG):
    """Blend color c toward base; t=1 keeps c, t=0 gives base."""
    c, b = np.array(to_rgb(c)), np.array(to_rgb(base))
    return tuple(b + t * (c - b))


RED_SOFT = mix(ROSE, 0.72)
GREEN_SOFT = mix(TEAL, 0.72)

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": TEXT,
    "axes.edgecolor": GRID,
    "axes.labelcolor": DIM,
    "xtick.color": DIM,
    "ytick.color": DIM,
    "font.family": "DejaVu Sans",
})


def style_axes(ax, keep=()):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(length=0)


def corr_heatmap():
    with open(TABLES / "correlation_spearman.csv") as f:
        rows = list(csv.DictReader(f))

    row_labels = {
        "efficiency": "Efficiency ↑",
        "comfort": "Comfort ↑",
        "route_completion": "Route Completion ↑",
        "collision": "Collision ↓",
    }
    col_keys = ["scr", "gcr_priority", "gcr_prohibitory", "gcr_mandatory", "gcr_special"]
    col_labels = ["SCR", "GCR\nPriority", "GCR\nProhibitory", "GCR\nMandatory", "GCR\nSpecial"]

    data = np.array([[float(r[k]) for k in col_keys] for r in rows])
    ylabels = [row_labels[r["conventional"]] for r in rows]

    # Richer diverging palette: rose ↔ slate ↔ teal
    cmap = LinearSegmentedColormap.from_list(
        "site_div",
        [ROSE, mix(ROSE, 0.55), "#1a2340", mix(TEAL, 0.55), TEAL],
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.35), dpi=200)
    im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=9.5)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=9.5)
    style_axes(ax)

    ax.set_xticks(np.arange(-0.5, len(col_labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels)), minor=True)
    ax.grid(which="minor", color=BG, linewidth=2.4)
    ax.tick_params(which="minor", length=0)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            # Dark text on strong cells, light text on near-zero cells
            tc = "#0a0e1a" if abs(v) >= 0.55 else TEXT
            ax.text(
                j, i, f"{v:+.2f}".replace("-", "\u2212"),
                ha="center", va="center", fontsize=10, fontweight="600", color=tc,
            )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.18)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_label("Spearman ρ", color=DIM, fontsize=9, labelpad=8)
    cbar.ax.tick_params(colors=DIM, labelsize=8, length=0)
    cbar.outline.set_edgecolor(GRID)
    cbar.outline.set_linewidth(0.6)

    fig.tight_layout()
    fig.savefig(OUT / "corr_heatmap_dark.png", bbox_inches="tight")
    plt.close(fig)


def rcr_forest():
    with open(TABLES / "bootstrap_rcr_ci.csv") as f:
        rows = list(csv.DictReader(f))

    def pretty(name):
        for k in ("IDM", "PPO", "CaRL", "PlanT-2"):
            name = name.replace(f"{k}^e", f"{k}$^e$")
        return name.replace(" (default)", "")

    labels = [pretty(r["display"]) for r in rows]
    vals = np.array([float(r["rcr"]) for r in rows]) * 100
    lo = np.array([float(r["ci95_lo"]) for r in rows]) * 100
    hi = np.array([float(r["ci95_hi"]) for r in rows]) * 100
    expert = np.array([r["is_expert"] == "1" for r in rows])

    n = len(rows)
    y = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=200)

    for i in range(n):
        c = TEAL if expert[i] else ROSE
        ax.plot(
            [lo[i], hi[i]], [y[i], y[i]],
            color=mix(c, 0.85), linewidth=2.4, solid_capstyle="round", zorder=2,
        )
        ax.scatter(vals[i], y[i], s=36, color=c, zorder=3, edgecolors=BG, linewidths=0.8)
        ax.text(hi[i] + 1.8, y[i], f"{vals[i]:.1f}", va="center", fontsize=8.5, color=DIM)

    split = y[np.argmax(expert)] + 0.5
    ax.axhline(split, color=GRID, linewidth=0.8)

    ax.text(2, y[~expert].mean(), "Base planners", color=mix(ROSE, 0.95),
            fontsize=10, va="center", fontweight="600")
    ax.text(2, y[expert].mean(), "Rule-compliant\nexperts", color=mix(TEAL, 0.95),
            fontsize=10, va="center", fontweight="600")

    ax.set_yticks(y, labels, fontsize=9.5)
    ax.set_xlim(0, 104)
    ax.set_xlabel("Overall SCR (%) — 95% episode-bootstrap CI (1,000 replicates)", fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(OUT / "rcr_ci_forest_dark.png", bbox_inches="tight")
    plt.close(fig)


def twin_gap():
    with open(TABLES / "joint_metrics_by_baseline.csv") as f:
        rows = {r["baseline"]: r for r in csv.DictReader(f)}

    # Overall compliance column in the CSV is named `rcr`; the page labels it SCR
    # to match the OpenReview response wording.
    twins = [
        ("IDM", "idm_default", "comprehensive_rule_expert_default"),
        ("IDM-s1", "idm_s1", "comprehensive_rule_expert_s1"),
        ("IDM-s2", "idm_s2", "comprehensive_rule_expert_s2"),
        ("IDM-s3", "idm_s3", "comprehensive_rule_expert_s3"),
        ("IDM-s4", "idm_s4", "comprehensive_rule_expert_s4"),
        ("PPO", "ppo_lidar_default", "rule_compliant_default"),
        ("CaRL", "carl_default", "carl_rule_default"),
        ("PlanT-2", "plant2_default", "plant2_rule_default"),
    ]

    def series(metric):
        base = [float(rows[b][metric]) * 100 for _, b, _ in twins]
        exp = [float(rows[e][metric]) * 100 for _, _, e in twins]
        return base, exp

    names = [t[0] for t in twins]
    x = np.arange(len(twins))
    w = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), dpi=200, sharey=True)

    panels = [
        ("route_completion", "Route Completion (%) — base ≈ expert"),
        ("rcr", "SCR (%) — expert ≫ base"),
    ]

    for ax, (metric, title) in zip(axes, panels):
        base, exp = series(metric)
        bars_b = ax.bar(x - w / 2, base, w, color=RED_SOFT, label="Base",
                        edgecolor=mix(ROSE, 0.9), linewidth=0.6)
        bars_e = ax.bar(x + w / 2, exp, w, color=GREEN_SOFT, label="Rule expert",
                        edgecolor=mix(TEAL, 0.9), linewidth=0.6)

        for bar, v in zip(bars_b, base):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.8, f"{v:.0f}",
                    ha="center", fontsize=7.5, color=DIM)
        for bar, v in zip(bars_e, exp):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.8, f"{v:.0f}",
                    ha="center", fontsize=7.5, color=DIM)

        ax.set_xticks(x, names, fontsize=8.5, rotation=18, ha="right")
        ax.set_ylim(0, 110)
        ax.set_title(title, fontsize=11, color=TEXT, pad=10, fontweight="600")
        ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        style_axes(ax)

    leg = axes[0].legend(loc="upper left", frameon=False, fontsize=9,
                         handlelength=1.2, handleheight=1.0)
    for t in leg.get_texts():
        t.set_color(DIM)

    fig.tight_layout()
    fig.savefig(OUT / "twin_gap_dark.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    corr_heatmap()
    rcr_forest()
    twin_gap()
    print("written:", *(p.name for p in sorted(OUT.glob("*_dark.png"))))
