#!/usr/bin/env python3
"""Render markdown tables + figures from collected dataset statistics.

Reads ``output/raw/{overview,sign_summary}.json`` and writes:
  - output/tables/*.md|*.csv
  - output/figures/*.png
  - output/DATASET_OVERVIEW.md   (reviewer-ready response draft)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fmt(x: Any, digits: int = 1) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _pct(n: float, total: float) -> str:
    if not total:
        return "—"
    return f"{100.0 * n / total:.1f}%"


def write_distribution_table(summaries: list[dict], overview: dict, out: Path) -> None:
    total_scen = overview["n_scenarios"] or 1
    rows = []
    for s in summaries:
        rows.append(
            {
                "sign": s["pdd_code"],
                "category": s["category"],
                "scenario_pct": round(100.0 * s["n_scenarios"] / total_scen, 2)
                if overview["n_scenarios"]
                else 0.0,
                "scenarios": s["n_scenarios"],
            }
        )

    csv_path = out / "sign_distribution.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["sign", "category", "scenario_pct", "scenarios"]
        )
        w.writeheader()
        w.writerows(rows)

    # Horizontal (transposed): columns = signs; rows = sign / Category / Scenario %
    signs = [r["sign"] for r in rows]
    cats = [r["category"] for r in rows]
    pcts = [f"{r['scenario_pct']:.1f}%" for r in rows]

    def _md_row(label: str, values: list[str]) -> str:
        return "| " + " | ".join([label, *values]) + " |"

    sep = "|---|" + "|".join(["---:" for _ in signs]) + "|"
    md = [
        "# Sign distribution",
        "",
        _md_row("sign", signs),
        sep,
        _md_row("Category", cats),
        _md_row("Scenario %", pcts),
        "",
        f"**Total scenarios:** {overview['n_scenarios']}.",
        "",
    ]
    (out / "sign_distribution.md").write_text("\n".join(md))


DURATION_BASELINES = ("idm", "idm_rule", "plant2", "plant_rule", "carl", "carl_rule")
DURATION_BASELINE_LABELS = {
    "idm": "IDM",
    "idm_rule": "IDM rule",
    "plant2": "Plant2",
    "plant_rule": "Plant rule",
    "carl": "CARL",
    "carl_rule": "CARL rule",
}


def _realized_by_baseline(rd: dict | None) -> dict[str, float | None]:
    rd = rd or {}
    by = rd.get("by_baseline") or {}
    out: dict[str, float | None] = {}
    for fam in DURATION_BASELINES:
        payload = by.get(fam) or {}
        out[fam] = payload.get("avg_seconds")
    # Backward compat if only top-level avg_seconds exists
    if out["idm"] is None and rd.get("avg_seconds") is not None:
        out["idm"] = rd.get("avg_seconds")
    return out


def write_duration_by_planner_table(summaries: list[dict], overview: dict, out: Path) -> None:
    """Compact table: sign × average realized duration per planner (seconds)."""
    header_labels = [DURATION_BASELINE_LABELS[f] for f in DURATION_BASELINES]
    rows = []
    for s in summaries:
        by = _realized_by_baseline(s.get("realized_duration"))
        row = {"sign": s["pdd_code"]}
        for fam in DURATION_BASELINES:
            row[fam] = by.get(fam)
        rows.append(row)

    csv_path = out / "duration_by_planner.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sign", *DURATION_BASELINES])
        w.writeheader()
        w.writerows(rows)

    md = [
        "# Average realized duration by planner (seconds)",
        "",
        "| Sign | " + " | ".join(f"{lab} (s)" for lab in header_labels) + " |",
        "|---|" + "|".join(["---:"] * len(DURATION_BASELINES)) + "|",
    ]
    for r in rows:
        cells = " | ".join(_fmt(r[fam], 1) for fam in DURATION_BASELINES)
        md.append(f"| {r['sign']} | {cells} |")

    by_mean = overview.get("mean_realized_duration_by_baseline_s") or {}
    if by_mean:
        mean_cells = " | ".join(
            _fmt(by_mean.get(fam), 1) for fam in DURATION_BASELINES
        )
        md.append(f"| **mean** | {mean_cells} |")

    md += [
        "",
        "Seconds = weighted `avg_steps` / `final_step` × 0.1. "
        "Families: `idm_*`, `idm_rule` (`modified_idm_*` / `comprehensive_rule_expert_*`), "
        "`plant2_default`, `plant2_rule_default`, `carl_default`, `carl_rule_default`.",
        "",
    ]
    (out / "duration_by_planner.md").write_text("\n".join(md))


def write_agents_duration_table(summaries: list[dict], overview: dict, out: Path) -> None:
    rows = []
    for s in summaries:
        a = s["agents"]
        h = s["horizon_seconds"]
        rd = s.get("realized_duration") or {}
        by = _realized_by_baseline(rd)
        row = {
            "sign": s["pdd_code"],
            "agent_mode": s["agent_mode"],
            "n_scenarios": s["n_scenarios"],
            "mean_agents": a.get("mean"),
            "std_agents": a.get("std"),
            "median_agents": a.get("median"),
            "min_agents": a.get("min"),
            "max_agents": a.get("max"),
            "mean_horizon_s": h.get("mean"),
        }
        for fam in DURATION_BASELINES:
            row[f"realized_{fam}_s"] = by.get(fam)
        rows.append(row)

    csv_path = out / "agents_and_duration.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    planner_hdr = " | ".join(
        f"{DURATION_BASELINE_LABELS[f]} (s)" for f in DURATION_BASELINES
    )
    planner_sep = "|".join(["---:"] * len(DURATION_BASELINES))
    md = [
        "# Agents and duration per sign",
        "",
        f"| Sign | Mode | N scenarios | Mean agents | Std | Median | Min–Max | Horizon (s) | {planner_hdr} |",
        f"|---|---|---:|---:|---:|---:|---:|---:|{planner_sep}|",
    ]
    for r in rows:
        mm = (
            f"{_fmt(r['min_agents'],0)}–{_fmt(r['max_agents'],0)}"
            if r["min_agents"] is not None
            else "—"
        )
        planner_cells = " | ".join(
            _fmt(r[f"realized_{fam}_s"], 1) for fam in DURATION_BASELINES
        )
        md.append(
            f"| {r['sign']} | {r['agent_mode']} | {r['n_scenarios']} | "
            f"{_fmt(r['mean_agents'],2)} | {_fmt(r['std_agents'],2)} | {_fmt(r['median_agents'],1)} | "
            f"{mm} | {_fmt(r['mean_horizon_s'],0)} | {planner_cells} |"
        )
    md += [
        "",
        f"**Configured horizon (all):** {_fmt(overview.get('mean_configured_horizon_s'), 0)} s.",
    ]
    by_mean = overview.get("mean_realized_duration_by_baseline_s") or {}
    if by_mean:
        parts = [
            f"{DURATION_BASELINE_LABELS.get(k, k)} {_fmt(by_mean.get(k), 1)} s"
            for k in DURATION_BASELINES
            if by_mean.get(k) is not None
        ]
        if parts:
            md.append(f"**Mean realized (weighted):** {'; '.join(parts)}.")
    else:
        md.append(
            f"**Realized duration (IDM, where measured):** "
            f"{_fmt(overview.get('mean_realized_duration_s'), 1)} s."
        )
    md.append("")
    mode_means = overview.get("mean_agents_by_mode") or {}
    if mode_means:
        md += [
            "| Traffic mode | Mean agents | N scenarios |",
            "|---|---:|---:|",
        ]
        for mode, payload in mode_means.items():
            md.append(
                f"| `{mode}` | {_fmt(payload.get('mean'), 2)} | {payload.get('n_scenarios')} |"
            )
        md.append("")
    md += [
        "**Agent definition.** `aux_convoy`: 1 ego + `aux_convoy_size × aux_lanes_occupied`. "
        "`density`: 1 ego + nuPlan vehicles/frame (fallback: `traffic_density × 80`). "
        "`density_ped`: density agents + `pedestrian_count`. "
        "`speed_ego` / `detour_ego`: 1 ego + `sample_one_profile(seed).traffic_density × 80` "
        "(catalog-direct eval samples density from the row seed; not stored in catalog.jsonl).",
        "",
        "**Duration.** Configured horizon = `horizon_steps × 0.1 s` "
        "(speed: 1500/150 s; detour: 1200/120 s; others: 600/60 s). "
        "Realized = weighted `avg_steps`/`final_step` × 0.1 s for "
        "`idm` (`idm_*`), `idm_rule` (`modified_idm_*` / `comprehensive_rule_expert_*`), "
        "`plant2` (`plant2_default`), `plant_rule` (`plant2_rule_default`), "
        "`carl` (`carl_default`), `carl_rule` (`carl_rule_default`). "
        "Compact planner-only table: `duration_by_planner.md`.",
        "",
    ]
    (out / "agents_and_duration.md").write_text("\n".join(md))


def write_agents_by_category_table(summaries: list[dict], out: Path) -> None:
    from collections import defaultdict

    by_cat: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"wsum": 0.0, "n": 0, "signs": []}
    )
    for s in summaries:
        a = s.get("agents") or {}
        mean, n = a.get("mean"), a.get("n") or s.get("n_scenarios") or 0
        if mean is None or not n:
            continue
        cat = s.get("category", "?")
        by_cat[cat]["wsum"] += float(mean) * float(n)
        by_cat[cat]["n"] += int(n)
        by_cat[cat]["signs"].append(s["pdd_code"])

    order = ["Priority", "Prohibitory", "Mandatory", "Special"]
    md = [
        "# Mean agents per scenario by sign category",
        "",
        "| Category | Mean agents | N scenarios | Signs |",
        "|---|---:|---:|---|",
    ]
    total_w, total_n = 0.0, 0
    for cat in order:
        if cat not in by_cat:
            continue
        v = by_cat[cat]
        mean = v["wsum"] / v["n"]
        md.append(
            f"| {cat} | {mean:.1f} | {v['n']} | {', '.join(v['signs'])} |"
        )
        total_w += v["wsum"]
        total_n += v["n"]
    if total_n:
        md.append(f"| **All** | **{total_w / total_n:.1f}** | **{total_n}** |  |")
    md += [
        "",
        "Weighted by number of scenarios per sign.",
        "",
    ]
    (out / "agents_by_category.md").write_text("\n".join(md))


def write_map_complexity_table(summaries: list[dict], out: Path) -> None:
    rows = []
    for s in summaries:
        lanes = s["map_lanes"]
        edges = s["map_edges"]
        arms = s["junction_arms"]
        rows.append(
            {
                "sign": s["pdd_code"],
                "mean_lanes": lanes.get("mean"),
                "mean_edges": edges.get("mean"),
                "mean_junction_arms": arms.get("mean"),
                "n_maps_parsed": lanes.get("n", 0),
            }
        )
    with (out / "map_complexity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = [
        "# Map complexity (SUMO net.xml)",
        "",
        "| Sign | Mean lanes | Mean edges | Mean junction arms | Maps parsed |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['sign']} | {_fmt(r['mean_lanes'],1)} | {_fmt(r['mean_edges'],1)} | "
            f"{_fmt(r['mean_junction_arms'],2)} | {r['n_maps_parsed']} |"
        )
    md.append("")
    (out / "map_complexity.md").write_text("\n".join(md))


def write_category_table(overview: dict, out: Path) -> None:
    cats = overview.get("category_distribution") or {}
    total = sum(v["n_scenarios"] for v in cats.values()) or 1
    md = [
        "# Category distribution (scenarios)",
        "",
        "| Category | Scenarios | % | Catalog scenes |",
        "|---|---:|---:|---:|",
    ]
    for cat, v in sorted(cats.items(), key=lambda kv: -kv[1]["n_scenarios"]):
        md.append(
            f"| {cat} | {int(v['n_scenarios'])} | {_pct(v['n_scenarios'], total)} | {int(v['n_catalog'])} |"
        )
    md.append("")
    (out / "category_distribution.md").write_text("\n".join(md))


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def fig_sign_distribution(summaries: list[dict], out: Path) -> None:
    _style()
    codes = [s["pdd_code"] for s in summaries]
    scen = np.array([s["n_scenarios"] for s in summaries], dtype=float)
    cat = np.array([s["n_catalog_scenes"] for s in summaries], dtype=float)
    pkg = np.array([s["n_package_scenes"] for s in summaries], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(codes))
    w = 0.38
    ax.bar(x - w / 2, cat, width=w, label="Catalog OSM scenes", color="#2F6F8F")
    ax.bar(x + w / 2, pkg, width=w, label="Package scenes", color="#C47A2C")
    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=55, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Scenes per sign")
    ax.legend(frameon=False)

    ax = axes[1]
    colors = plt.cm.tab20(np.linspace(0, 1, len(codes)))
    # Hide zero-scenario signs in pie to avoid clutter; keep legend.
    nonzero = [(c, n, col) for c, n, col in zip(codes, scen, colors) if n > 0]
    if nonzero:
        labels = [f"{c} ({int(n)})" for c, n, _ in nonzero]
        ax.pie(
            [n for _, n, _ in nonzero],
            labels=None,
            colors=[col for _, _, col in nonzero],
            startangle=90,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        )
        ax.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
    ax.set_title("Scenario share (final_metrics_v1)")

    fig.savefig(out / "fig_sign_distribution.png", dpi=160)
    plt.close(fig)


def fig_agents(summaries: list[dict], out: Path) -> None:
    _style()
    codes = [s["pdd_code"] for s in summaries if s["agents"].get("mean") is not None]
    means = [s["agents"]["mean"] for s in summaries if s["agents"].get("mean") is not None]
    stds = [s["agents"].get("std") or 0 for s in summaries if s["agents"].get("mean") is not None]

    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    x = np.arange(len(codes))
    ax.bar(x, means, yerr=stds, capsize=3, color="#3D7A5A", ecolor="#333333", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=55, ha="right")
    ax.set_ylabel("Mean estimated agents / scenario")
    ax.set_title("Average agents per scenario (ego + NPCs / pedestrians)")
    fig.savefig(out / "fig_agents_per_sign.png", dpi=160)
    plt.close(fig)


def fig_duration(summaries: list[dict], out: Path) -> None:
    _style()
    codes = []
    series: dict[str, list[float]] = {fam: [] for fam in DURATION_BASELINES}
    for s in summaries:
        if s["n_scenarios"] == 0:
            continue
        codes.append(s["pdd_code"])
        by = _realized_by_baseline(s.get("realized_duration"))
        for fam in DURATION_BASELINES:
            series[fam].append(by.get(fam) or 0)

    fig, ax = plt.subplots(figsize=(12.5, 5.0), constrained_layout=True)
    x = np.arange(len(codes))
    n_bars = len(DURATION_BASELINES)
    w = 0.13
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * w
    colors = {
        "idm": "#B84A3E",
        "idm_rule": "#D4896A",
        "plant2": "#1F6F8B",
        "plant_rule": "#3A9B7A",
        "carl": "#5B4B8A",
        "carl_rule": "#8B7BB8",
    }
    for i, fam in enumerate(DURATION_BASELINES):
        ax.bar(
            x + offsets[i],
            series[fam],
            width=w,
            label=DURATION_BASELINE_LABELS[fam],
            color=colors[fam],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=55, ha="right")
    ax.set_ylabel("Seconds")
    ax.set_title("Average realized duration by policy")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(out / "fig_duration_per_sign.png", dpi=160)
    plt.close(fig)


def fig_map_complexity(summaries: list[dict], out: Path) -> None:
    _style()
    codes = [s["pdd_code"] for s in summaries if s["map_lanes"].get("mean") is not None]
    lanes = [s["map_lanes"]["mean"] for s in summaries if s["map_lanes"].get("mean") is not None]
    edges = [s["map_edges"].get("mean") or 0 for s in summaries if s["map_lanes"].get("mean") is not None]

    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    x = np.arange(len(codes))
    w = 0.38
    ax.bar(x - w / 2, lanes, width=w, label="Mean lanes", color="#1F6F8B")
    ax.bar(x + w / 2, edges, width=w, label="Mean edges", color="#8B5E34")
    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=55, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Map complexity from SUMO net.xml")
    ax.legend(frameon=False)
    fig.savefig(out / "fig_map_complexity.png", dpi=160)
    plt.close(fig)


def fig_geo_scatter(scenario_jsonl: Path, out: Path) -> None:
    _style()
    if not scenario_jsonl.exists():
        return
    by_sign: dict[str, list[tuple[float, float]]] = {}
    with scenario_jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            lat, lon = r.get("latitude"), r.get("longitude")
            if lat is None or lon is None:
                continue
            by_sign.setdefault(r["pdd_code"], []).append((float(lon), float(lat)))

    if not by_sign:
        return

    fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    cmap = plt.cm.tab20
    for i, (code, pts) in enumerate(sorted(by_sign.items())):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=8, alpha=0.55, color=cmap(i % 20), label=f"{code} (n={len(pts)})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic footprint of scenarios (Moscow OSM)")
    ax.legend(frameon=False, fontsize=7, loc="best", markerscale=2)
    ax.set_aspect("equal", adjustable="datalim")
    fig.savefig(out / "fig_geo_footprint.png", dpi=160)
    plt.close(fig)


def write_reviewer_report(
    summaries: list[dict],
    overview: dict,
    tables_dir: Path,
    figures_dir: Path,
    out_path: Path,
) -> None:
    # Build a persuasive, self-contained reviewer response.
    lines = [
        "# Dataset statistical overview (reviewer response)",
        "",
        "We thank the reviewer for requesting a high-level statistical overview of the",
        "generated scenes. Below we report the **actual contents** of the released subset",
        "covering priority / prohibitory / mandatory / special signs",
        "(2.1, 2.3.1–2.3.3, 2.4, 2.5, 3.1–3.2, 3.24, 4.2.1–4.2.3, 4.3, 4.6,",
        "5.7.1–5.7.2, 5.15.1, 5.19, 5.21, 5.31).",
        "",
        "Speed / zone signs (3.24, 4.6, 5.21, 5.31) use the map-trimmed",
        "`catalog_balanced_1k2.jsonl`. Detour signs (4.2.1–4.2.3) use",
        "`detour_v1/catalog.jsonl`.",
        "",
        "## Headline numbers",
        "",
        f"| Quantity | Value |",
        f"|---|---:|",
        f"| Signs in this overview | {overview['n_signs']} |",
        f"| OSM catalog scenes | {overview['n_catalog_scenes']} |",
        f"| Filtered package scenes | {overview['n_package_scenes']} |",
        f"| Augmented scenarios (`final_metrics_v1`) | **{overview['n_scenarios']}** |",
        f"| Configured horizon (all scenarios) | **{_fmt(overview.get('mean_configured_horizon_s'), 0)} s** |",
    ]
    by_mean = overview.get("mean_realized_duration_by_baseline_s") or {}
    if by_mean:
        for fam in DURATION_BASELINES:
            if by_mean.get(fam) is None:
                continue
            lines.append(
                f"| Mean realized ({DURATION_BASELINE_LABELS[fam]}) | "
                f"**{_fmt(by_mean[fam], 1)} s** |"
            )
    else:
        lines.append(
            f"| Mean realized duration (IDM) | "
            f"**{_fmt(overview.get('mean_realized_duration_s'), 1)} s** |"
        )
    lines += [
        "",
    ]

    mode_means = overview.get("mean_agents_by_mode") or {}
    if mode_means:
        lines += [
            "### Agents by traffic design",
            "",
            "| Traffic mode | Mean agents / scenario | N scenarios |",
            "|---|---:|---:|",
        ]
        mode_labels = {
            "aux_convoy": "Local auxiliary convoy (priority / roundabout)",
            "density": "nuPlan density tiers (prohibitory / one-way / lane dirs)",
            "density_ped": "nuPlan density + pedestrians (crosswalk)",
            "speed_ego": "Ego-centric speed / zone (3.24, 4.6, 5.21, 5.31)",
            "detour_ego": "Ego-centric detour (4.2.1–4.2.3)",
        }
        for mode, payload in mode_means.items():
            lines.append(
                f"| {mode_labels.get(mode, mode)} | **{_fmt(payload.get('mean'), 2)}** | {payload.get('n_scenarios')} |"
            )
        lines += [
            "",
            "These are intentionally different designs: priority signs stress a small number of",
            "interacting vehicles on the conflicting arm; density-augmented signs replay",
            "nuPlan-calibrated traffic levels (low / medium / high ≈ 21 / 31 / 43 vehicles/frame).",
            "",
        ]

    lines += [
        "We distinguish three nested units (all counted below):",
        "",
        "1. **Catalog scenes** — OSM road crops centered on a real traffic sign.",
        "2. **Package scenes** — geometrically validated / manually curated crops used for benchmarking.",
        "3. **Scenarios** — evaluation units obtained by spawning / density / pedestrian augmentation",
        "   of package scenes (what agents actually roll out).",
        "",
        "## Distribution by sign",
        "",
        "See `tables/sign_distribution.md` and `figures/fig_sign_distribution.png`.",
        "",
    ]

    # Horizontal (transposed) distribution: Sign / Category / Scenario %
    total = overview["n_scenarios"] or 1
    signs = [s["pdd_code"] for s in summaries]
    cats = [s["category"] for s in summaries]
    pcts = [_pct(s["n_scenarios"], total) for s in summaries]

    def _hrow(label: str, values: list[str]) -> str:
        return "| " + " | ".join([label, *values]) + " |"

    lines += [
        _hrow("sign", signs),
        "|---|" + "|".join(["---:" for _ in signs]) + "|",
        _hrow("Category", cats),
        _hrow("Scenario %", pcts),
        "",
        f"**Total scenarios:** {overview['n_scenarios']}.",
        "",
        "![Sign distribution](figures/fig_sign_distribution.png)",
        "",
        "## Agents per scenario",
        "",
        "See `tables/agents_and_duration.md` and `figures/fig_agents_per_sign.png`.",
        "",
        "| Sign | Mean agents | Median | Range | Traffic mode |",
        "|---|---:|---:|---:|---|",
    ]
    for s in summaries:
        a = s["agents"]
        if not a.get("n"):
            lines.append(f"| {s['pdd_code']} | — | — | — | {s['agent_mode']} |")
            continue
        lines.append(
            f"| {s['pdd_code']} | {_fmt(a.get('mean'),2)} | {_fmt(a.get('median'),1)} | "
            f"{_fmt(a.get('min'),0)}–{_fmt(a.get('max'),0)} | {s['agent_mode']} |"
        )
    lines += [
        "",
        "Do not average across modes naively: convoy signs and density signs answer different",
        "interaction questions. Prefer the per-mode means in the headline table.",
        "",
        "![Agents per sign](figures/fig_agents_per_sign.png)",
        "",
        "## Duration",
        "",
        "Priority / density scenarios use a **600-step / 60 s** MetaDrive horizon;",
        "speed / zone signs use **1500-step / 150 s**; detour signs use **1200-step / 120 s**.",
        "Realized episode length (until arrival / crash / timeout) is shorter on average.",
        "",
        "| Sign | Horizon (s) | "
        + " | ".join(f"{DURATION_BASELINE_LABELS[f]} (s)" for f in DURATION_BASELINES)
        + " |",
        "|---|---:|" + "|".join(["---:"] * len(DURATION_BASELINES)) + "|",
    ]
    for s in summaries:
        h = s["horizon_seconds"].get("mean")
        by = _realized_by_baseline(s.get("realized_duration"))
        cells = " | ".join(_fmt(by.get(fam), 1) for fam in DURATION_BASELINES)
        lines.append(f"| {s['pdd_code']} | {_fmt(h,0)} | {cells} |")
    lines += [
        "",
        overview.get("realized_duration_coverage_note")
        or "Realized duration is reported where eval aggregations exist.",
        "",
        "Compact planner-only table: `tables/duration_by_planner.md`.",
        "",
        "![Duration](figures/fig_duration_per_sign.png)",
        "",
        "## Map complexity & geography",
        "",
        "Cropped SUMO maps typically contain hundreds of lanes/edges; junction crops are",
        "multi-arm (T / X / roundabout). See `tables/map_complexity.md`,",
        "`figures/fig_map_complexity.png`, and `figures/fig_geo_footprint.png`.",
        "",
        "![Map complexity](figures/fig_map_complexity.png)",
        "",
        "![Geographic footprint](figures/fig_geo_footprint.png)",
        "",
        "## Category mix",
        "",
    ]
    # inline category
    cats = overview.get("category_distribution") or {}
    ctot = sum(v["n_scenarios"] for v in cats.values()) or 1
    lines += [
        "| Category | Scenarios | % |",
        "|---|---:|---:|",
    ]
    for cat, v in sorted(cats.items(), key=lambda kv: -kv[1]["n_scenarios"]):
        lines.append(f"| {cat} | {int(v['n_scenarios'])} | {_pct(v['n_scenarios'], ctot)} |")

    # Notes / caveats
    caveats = []
    for s in summaries:
        for n in s.get("notes") or []:
            caveats.append(f"- **{s['pdd_code']}:** {n}")
    if caveats:
        lines += ["", "## Coverage notes", "", *caveats]

    lines += [
        "",
        "## Reproducibility",
        "",
        "```bash",
        "cd pdd-bench/scripts/dataset_stats",
        "python collect_stats.py",
        "python plot_and_report.py",
        "```",
        "",
        "Artifacts: `output/raw/`, `output/tables/`, `output/figures/`.",
        "",
    ]
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    args = parser.parse_args()

    raw = args.out_dir / "raw"
    tables = args.out_dir / "tables"
    figures = args.out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    overview = json.loads((raw / "overview.json").read_text())
    summaries = json.loads((raw / "sign_summary.json").read_text())

    write_distribution_table(summaries, overview, tables)
    write_agents_duration_table(summaries, overview, tables)
    write_duration_by_planner_table(summaries, overview, tables)
    write_agents_by_category_table(summaries, tables)
    write_map_complexity_table(summaries, tables)
    write_category_table(overview, tables)

    fig_sign_distribution(summaries, figures)
    fig_agents(summaries, figures)
    fig_duration(summaries, figures)
    fig_map_complexity(summaries, figures)
    fig_geo_scatter(raw / "scenario_stats.jsonl", figures)

    write_reviewer_report(
        summaries,
        overview,
        tables,
        figures,
        args.out_dir / "DATASET_OVERVIEW.md",
    )
    print(f"Wrote tables → {tables}")
    print(f"Wrote figures → {figures}")
    print(f"Wrote report → {args.out_dir / 'DATASET_OVERVIEW.md'}")


if __name__ == "__main__":
    main()
