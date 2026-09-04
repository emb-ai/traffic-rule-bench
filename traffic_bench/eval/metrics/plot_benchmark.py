#!/usr/bin/env python3
"""Baseline comparison charts from eval ``reports/cumulative.json`` files.

Scans per-sign train eval outputs, merges ``per_sign`` slices, and writes:

- one PNG per sign (three metrics, base vs rule-expert bars side by side);
- optional family overview PNGs when several signs share a semantic group;
- macro-average summary + short markdown analysis.

Usage::

    python -m traffic_bench.eval metrics plot
    python -m traffic_bench.eval metrics plot --split train --agg map
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from traffic_bench.eval.metrics.report import _display
from traffic_bench.eval.sign_registry import (
    REPO_ROOT,
    SignProfile,
    list_profiles,
    resolve_repo_path,
)

AggMode = Literal["episode", "map"]

ALL_RUNS_PLOTS = "data/runs/_all/train/plots/benchmark"

METRICS: tuple[tuple[str, str, str], ...] = (
    (
        "sr_and_dest",
        "SR&Dest",
        "Target sign obeyed AND destination reached",
    ),
    (
        "dest_rate",
        "Destination rate",
        "Episodes that reach the goal",
    ),
    (
        "sign_compliance_x",
        "Sign compliance (in-zone)",
        "No sign violation while ego is in the sign zone",
    ),
)

DEFAULT_POLICY_ORDER: tuple[str, ...] = (
    "idm_default",
    "idm_rule_default",
    "ppo_lidar_default",
    "ppo_rule_default",
    "carl_default",
    "carl_rule_default",
)

# (family label, base policy, rule-expert policy)
POLICY_TRIPLETS: tuple[tuple[str, str, str], ...] = (
    ("IDM", "idm_default", "idm_rule_default"),
    ("PPO", "ppo_lidar_default", "ppo_rule_default"),
    ("CaRL", "carl_default", "carl_rule_default"),
)

POLICY_COLORS: dict[str, str] = {
    "idm_default": "#94a3b8",
    "idm_rule_default": "#2563eb",
    "ppo_lidar_default": "#cbd5e1",
    "ppo_rule_default": "#7c3aed",
    "carl_default": "#fdba74",
    "carl_rule_default": "#ea580c",
}

# Semantic groups for family / all-signs charts (bench sign_code → plate family).
# Sub-plates (2.3.1–2.3.3, 5.19.1–5.19.2, …) collapse to one eval sign each.
PLOT_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "priority",
        "Приоритетность движения",
        ("2.1", "2.3", "2.4", "2.5", "4.3", "5.19"),
    ),
    (
        "speed",
        "Скоростной режим",
        ("3.24", "4.6", "5.21", "5.31"),
    ),
    (
        "lane_obstacle",
        "Запрещенная линия внутри полосы (препятствие)",
        ("3.2", "4.2.1", "4.2.2", "4.2.3"),
    ),
    (
        "lane_rerouting",
        "Запрещенная полоса (re-routing)",
        (
            "3.1",
            "3.18.1",
            "3.18.2",
            "4.1.1",
            "4.1.2",
            "4.1.3",
            "4.1.4",
            "4.1.5",
            "4.1.6",
            "5.7.1",
            "5.7.2",
        ),
    ),
)

FAMILY_METRIC_KEY = "sign_compliance_x"
FAMILY_METRIC_TITLE = "Sign compliance (in-zone)"

INK = "#1f2937"
MUT = "#6b7280"
GRID = "#e5e7eb"
GROUP_GAP = 0.55
PAIR_GAP = 0.12

_PROFILE_BY_CODE = {p.sign_code: p for p in list_profiles()}
_PROFILE_BY_SUBDIR = {p.data_subdir: p for p in list_profiles()}


@dataclass
class SignMetrics:
    profile: SignProfile
    n: int
    by_policy: dict[str, dict[str, float]]


@dataclass
class BenchmarkTable:
    signs: list[SignMetrics] = field(default_factory=list)
    policies: list[str] = field(default_factory=list)
    sources: list[Path] = field(default_factory=list)

    def value(self, sign: SignMetrics, policy: str, metric: str) -> float | None:
        raw = sign.by_policy.get(policy, {}).get(metric)
        return None if raw is None else float(raw)

    def weighted_overall(self, metric: str) -> dict[str, float]:
        totals: dict[str, float] = {}
        weights: dict[str, int] = {}
        for sign in self.signs:
            for policy, metrics in sign.by_policy.items():
                value = metrics.get(metric)
                if value is None or sign.n <= 0:
                    continue
                totals[policy] = totals.get(policy, 0.0) + float(value) * sign.n
                weights[policy] = weights.get(policy, 0) + sign.n
        return {
            p: totals[p] / weights[p]
            for p in self.policies
            if weights.get(p, 0) > 0
        }


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "text.color": INK,
            "axes.labelcolor": MUT,
            "xtick.color": MUT,
            "ytick.color": MUT,
            "legend.frameon": False,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def _policy_label(policy: str) -> str:
    name = _display(policy)
    return name.replace("_default", "").replace("_", " ")


def _sign_slug(code: str) -> str:
    return re.sub(r"[^\w.]+", "_", code)


def _active_triplets(policies: list[str]) -> list[tuple[str, str, str]]:
    active: list[tuple[str, str, str]] = []
    for family, base, expert in POLICY_TRIPLETS:
        pair = [p for p in (base, expert) if p in policies]
        if pair:
            active.append((family, base, expert))
    return active


def _bar_positions(n_groups: int, bars_per_group: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return x positions and (start, end) index ranges for each group."""
    step = bars_per_group + PAIR_GAP
    group_stride = bars_per_group * step + GROUP_GAP
    positions: list[float] = []
    spans: list[tuple[int, int]] = []
    cursor = 0.0
    for _ in range(n_groups):
        start = len(positions)
        for j in range(bars_per_group):
            positions.append(cursor + j * step)
        spans.append((start, len(positions)))
        cursor = positions[-1] + step + GROUP_GAP
    return np.array(positions, dtype=float), spans


def _format_pct_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:.0%}"))


def _annotate_bars(ax: plt.Axes, bars, values: list[float | None]) -> None:
    for bar, val in zip(bars, values):
        if val is None or np.isnan(val):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(float(val) + 0.02, 1.01),
            f"{val:.0%}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=INK,
        )


def _legend_handles(policies: list[str]) -> list[Patch]:
    handles: list[Patch] = []
    for _family, base, expert in POLICY_TRIPLETS:
        for policy, suffix in ((base, "base"), (expert, "rule")):
            if policy not in policies:
                continue
            handles.append(
                Patch(
                    facecolor=POLICY_COLORS.get(policy, "#64748b"),
                    edgecolor="white",
                    label=f"{_policy_label(policy)} ({suffix})",
                )
            )
    return handles


def _plot_metric_bars(
    ax: plt.Axes,
    *,
    group_labels: list[str],
    values_by_group: list[list[float | None]],
    policies: list[str],
    title: str,
    show_legend: bool,
) -> None:
    if not values_by_group:
        return

    bars_per_group = max(len(row) for row in values_by_group)
    n_groups = len(group_labels)
    positions, spans = _bar_positions(n_groups, bars_per_group)
    width = 0.78

    for g_idx, group_values in enumerate(values_by_group):
        start, end = spans[g_idx]
        xs = positions[start:end]
        vals = group_values[: len(xs)]
        policy_order: list[str] = []
        for _family, base, expert in _active_triplets(policies):
            if len(policy_order) >= len(vals):
                break
            if base in policies:
                policy_order.append(base)
            if len(policy_order) >= len(vals):
                break
            if expert in policies:
                policy_order.append(expert)
        colors = [POLICY_COLORS.get(p, "#64748b") for p in policy_order[: len(vals)]]
        bars = ax.bar(xs, vals, width=width, color=colors, edgecolor="white", linewidth=0.6)
        _annotate_bars(ax, bars, vals)

    centers = [(positions[a] + positions[b - 1]) / 2 for a, b in spans]
    ax.set_xticks(centers)
    ax.set_xticklabels(group_labels, rotation=0 if n_groups <= 5 else 28, ha="center")
    ax.set_title(title)
    _format_pct_axis(ax)

    for a, b in spans[:-1]:
        boundary = (positions[b - 1] + positions[b]) / 2
        ax.axvline(boundary, color=GRID, linewidth=0.9, zorder=0)

    if show_legend:
        ax.legend(handles=_legend_handles(policies), loc="upper right", fontsize=7.5, ncol=2)


def _triplet_values(sign: SignMetrics, policies: list[str], metric_key: str) -> tuple[list[str], list[list[float | None]]]:
    labels: list[str] = []
    rows: list[list[float | None]] = []
    for family, base, expert in _active_triplets(policies):
        labels.append(family)
        rows.append([
            sign.by_policy.get(base, {}).get(metric_key),
            sign.by_policy.get(expert, {}).get(metric_key),
        ])
    return labels, rows


def _plot_per_sign(sign: SignMetrics, policies: list[str], out_path: Path) -> None:
    _apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharey=True)
    fig.subplots_adjust(wspace=0.14, top=0.78, bottom=0.16, left=0.06, right=0.98)

    profile = sign.profile
    for ax, (key, title, _desc) in zip(axes, METRICS):
        group_labels, values_by_group = _triplet_values(sign, policies, key)
        _plot_metric_bars(
            ax,
            group_labels=group_labels,
            values_by_group=values_by_group,
            policies=policies,
            title=title,
            show_legend=False,
        )

    fig.suptitle(
        f"{profile.sign_code} — {profile.sign_name}",
        fontsize=12,
        fontweight="bold",
        y=0.96,
    )
    fig.legend(handles=_legend_handles(policies), loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _signs_by_codes(table: BenchmarkTable, codes: tuple[str, ...]) -> list[SignMetrics]:
    by_code = {s.profile.sign_code: s for s in table.signs}
    return [by_code[code] for code in codes if code in by_code]


def _plot_sign_compliance_bars(
    *,
    panel_title: str,
    signs: list[SignMetrics],
    policies: list[str],
    out_path: Path,
    group_boundaries: list[float] | None = None,
    fig_height: float = 4.8,
) -> None:
    """Grouped bars: each sign on x-axis, six baselines per sign (no averaging)."""
    if not signs:
        return

    _apply_plot_style()
    n_signs = len(signs)
    n_policies = len(policies)
    x = np.arange(n_signs, dtype=float)
    cluster_width = 0.82
    bar_width = cluster_width / max(n_policies, 1)

    fig_w = max(8.5, 0.55 * n_signs + 4.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_height))

    for i, policy in enumerate(policies):
        offset = (i - (n_policies - 1) / 2) * bar_width
        values = [
            sign.by_policy.get(policy, {}).get(FAMILY_METRIC_KEY)
            for sign in signs
        ]
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width * 0.92,
            color=POLICY_COLORS.get(policy, "#64748b"),
            edgecolor="white",
            linewidth=0.5,
            label=_policy_label(policy),
            zorder=3,
        )
        _annotate_bars(ax, bars, values)

    if group_boundaries:
        for pos in group_boundaries:
            ax.axvline(pos, color=GRID, linewidth=1.2, linestyle="--", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.profile.sign_code for s in signs],
        rotation=45 if n_signs > 6 else 0,
        ha="right" if n_signs > 6 else "center",
        fontsize=8 if n_signs > 12 else 9,
    )
    ax.set_ylabel(FAMILY_METRIC_TITLE)
    ax.set_title(FAMILY_METRIC_TITLE)
    _format_pct_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=8)
    fig.suptitle(panel_title, fontsize=12, fontweight="bold", y=1.06)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _family_groups(table: BenchmarkTable) -> list[tuple[str, str, list[SignMetrics]]]:
    panels: list[tuple[str, str, list[SignMetrics]]] = []
    for slug, title, codes in PLOT_GROUPS:
        signs = _signs_by_codes(table, codes)
        if signs:
            panels.append((slug, title, signs))
    return panels


def _all_signs_ordered(table: BenchmarkTable) -> tuple[list[SignMetrics], list[float]]:
    """All signs in semantic-group order; boundary x-positions between groups."""
    signs: list[SignMetrics] = []
    boundaries: list[float] = []
    cursor = 0
    for _slug, _title, codes in PLOT_GROUPS:
        group = _signs_by_codes(table, codes)
        if not group:
            continue
        if signs:
            boundaries.append(cursor - 0.5)
        signs.extend(group)
        cursor += len(group)

    seen = {s.profile.sign_code for s in signs}
    for profile in list_profiles():
        if profile.sign_code in seen:
            continue
        extra = next((s for s in table.signs if s.profile.sign_code == profile.sign_code), None)
        if extra is not None:
            if signs:
                boundaries.append(cursor - 0.5)
            signs.append(extra)
            cursor += 1
    return signs, boundaries


def _plot_overall_summary(table: BenchmarkTable, out_path: Path) -> None:
    _apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    fig.subplots_adjust(wspace=0.12, top=0.82, bottom=0.14, left=0.06, right=0.98)

    group_labels = ["macro avg"]
    for ax, (key, title, _desc) in zip(axes, METRICS):
        overall = table.weighted_overall(key)
        labels: list[str] = []
        rows: list[list[float | None]] = []
        for family, base, expert in _active_triplets(table.policies):
            labels.append(family)
            rows.append([overall.get(base), overall.get(expert)])
        _plot_metric_bars(
            ax,
            group_labels=labels,
            values_by_group=rows,
            policies=table.policies,
            title=title,
            show_legend=False,
        )

    fig.suptitle(
        "Macro average across signs (episode-weighted)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.legend(handles=_legend_handles(table.policies), loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _resolve_policies(raw: Iterable[str] | None, seen: set[str], *, all_policies: bool) -> list[str]:
    if raw is not None:
        return [p for p in raw if p in seen]
    pool = seen if all_policies else set(DEFAULT_POLICY_ORDER)
    ordered = [p for p in DEFAULT_POLICY_ORDER if p in seen and p in pool]
    if all_policies:
        extras = sorted(seen - set(ordered) - set(DEFAULT_POLICY_ORDER))
        return ordered + extras
    return ordered


def _per_sign_block(data: dict, agg: AggMode) -> dict[str, dict[str, dict]]:
    if agg == "map":
        block = data.get("per_sign_map") or {}
        if block:
            return block
    return data.get("per_sign") or {}


def _load_cumulative(path: Path, agg: AggMode) -> dict[str, dict[str, dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return _per_sign_block(data, agg)


def _profile_for_sign_code(code: str) -> SignProfile | None:
    return _PROFILE_BY_CODE.get(code)


def _profile_for_run_dir(run_subdir: str) -> SignProfile | None:
    return _PROFILE_BY_SUBDIR.get(run_subdir)


def _merge_sign_slice(
    table: BenchmarkTable,
    profile: SignProfile,
    policy: str,
    metrics: dict,
    metric_keys: set[str],
) -> None:
    n = int(metrics.get("n") or 0)
    values = {
        key: float(metrics[key])
        for key in metric_keys
        if metrics.get(key) not in (None, "")
    }
    if not values:
        return
    for sign in table.signs:
        if sign.profile.sign_code == profile.sign_code:
            sign.by_policy.setdefault(policy, {}).update(values)
            sign.n = max(sign.n, n)
            return
    table.signs.append(SignMetrics(profile=profile, n=n, by_policy={policy: values}))


def load_from_cumulative(
    path: Path,
    *,
    agg: AggMode,
    policies: list[str] | None = None,
    all_policies: bool = False,
) -> BenchmarkTable:
    per_sign = _load_cumulative(path, agg)
    metric_keys = {m[0] for m in METRICS}
    seen_policies: set[str] = set()
    table = BenchmarkTable(sources=[path.resolve()])

    for policy, sign_map in per_sign.items():
        if policies and policy not in policies:
            continue
        if not isinstance(sign_map, dict):
            continue
        seen_policies.add(policy)
        for sign_code, metrics in sign_map.items():
            if not isinstance(metrics, dict):
                continue
            profile = _profile_for_sign_code(str(sign_code))
            if profile is None:
                continue
            _merge_sign_slice(table, profile, policy, metrics, metric_keys)

    table.policies = _resolve_policies(policies, seen_policies, all_policies=all_policies)
    _sort_signs(table)
    return table


def discover_cumulative_paths(runs_root: Path, split: str) -> list[Path]:
    pattern = f"*/{split}/eval_out/reports/cumulative.json"
    return sorted(runs_root.glob(pattern))


def load_from_runs(
    runs_root: Path,
    *,
    split: str,
    agg: AggMode,
    policies: list[str] | None = None,
    all_policies: bool = False,
) -> BenchmarkTable:
    paths = discover_cumulative_paths(runs_root, split)
    if not paths:
        raise FileNotFoundError(
            f"No cumulative.json under {runs_root}/*/{split}/eval_out/reports/"
        )

    metric_keys = {m[0] for m in METRICS}
    seen_policies: set[str] = set()
    table = BenchmarkTable()

    for path in paths:
        run_subdir = path.parts[-5]
        profile = _profile_for_run_dir(run_subdir)
        if profile is None:
            continue
        per_sign = _load_cumulative(path, agg)
        table.sources.append(path.resolve())
        for policy, sign_map in per_sign.items():
            if policies and policy not in policies:
                continue
            if not isinstance(sign_map, dict):
                continue
            seen_policies.add(policy)
            metrics = sign_map.get(profile.sign_code)
            if not isinstance(metrics, dict):
                if len(sign_map) == 1:
                    metrics = next(iter(sign_map.values()))
                else:
                    continue
            if not isinstance(metrics, dict):
                continue
            _merge_sign_slice(table, profile, policy, metrics, metric_keys)

    table.policies = _resolve_policies(policies, seen_policies, all_policies=all_policies)
    _sort_signs(table)
    return table


def _sort_signs(table: BenchmarkTable) -> None:
    order = {p.sign_code: i for i, p in enumerate(list_profiles())}
    table.signs.sort(key=lambda s: order.get(s.profile.sign_code, 999))


def _best_policy(table: BenchmarkTable, metric: str) -> tuple[str, float]:
    overall = table.weighted_overall(metric)
    if not overall:
        return ("—", float("nan"))
    policy = max(overall, key=overall.get)
    return policy, overall[policy]


def _expert_vs_base_lines(table: BenchmarkTable, metric: str) -> list[str]:
    lines: list[str] = []
    overall = table.weighted_overall(metric)
    for family, base, expert in _active_triplets(table.policies):
        b = overall.get(base)
        e = overall.get(expert)
        if b is None or e is None:
            continue
        lines.append(f"- **{family}**: base {b:.1%}, rule expert {e:.1%}")
    return lines


def write_analysis(table: BenchmarkTable, out_path: Path, *, agg: AggMode) -> None:
    lines: list[str] = []
    lines.append("# Benchmark comparison analysis")
    lines.append("")
    lines.append(f"Sources: {len(table.sources)} cumulative report(s)")
    lines.append(f"Signs: {len(table.signs)}   Policies: {len(table.policies)}")
    lines.append(f"Aggregation: **{agg}**")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `overall_summary.png` — macro average (base vs rule expert per family)")
    lines.append("- `by_sign/<code>.png` — one chart per sign, three metrics")
    lines.append("- `by_family/<group>.png` — sign compliance by semantic group (4 groups)")
    lines.append("- `all_signs_sign_compliance.png` — every sign, six baselines, no averaging")
    lines.append("")
    lines.append("Within every chart, each policy family shows **base** then **rule expert** side by side.")
    lines.append("")

    lines.append("## Macro average")
    lines.append("")
    for key, title, _desc in METRICS:
        lines.append(f"### {title}")
        lines.append("")
        for line in _expert_vs_base_lines(table, key):
            lines.append(line)
        policy, rate = _best_policy(table, key)
        lines.append(f"- Best overall: `{_policy_label(policy)}` ({rate:.1%})")
        lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append("- **SR&Dest** — target sign obeyed and destination reached")
    lines.append("- **Destination rate** — navigation success only")
    lines.append("- **Sign compliance (in-zone)** — no violation while in the sign zone")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _remove_stale(path: Path) -> None:
    if path.is_file():
        path.unlink()


def render_benchmark(table: BenchmarkTable, out_dir: Path, *, agg: AggMode) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Drop legacy figure types from earlier versions of this script.
    for stale in out_dir.glob("heatmap_*.png"):
        _remove_stale(stale)
    _remove_stale(out_dir / "rule_lift_sr_dest.png")

    overall = out_dir / "overall_summary.png"
    _plot_overall_summary(table, overall)
    written.append(overall)

    by_sign_dir = out_dir / "by_sign"
    for sign in table.signs:
        path = by_sign_dir / f"{_sign_slug(sign.profile.sign_code)}.png"
        _plot_per_sign(sign, table.policies, path)
        written.append(path)

    by_family_dir = out_dir / "by_family"
    by_family_dir.mkdir(parents=True, exist_ok=True)
    for stale in by_family_dir.glob("*.png"):
        _remove_stale(stale)

    for slug, panel_title, signs in _family_groups(table):
        path = by_family_dir / f"{slug}.png"
        _plot_sign_compliance_bars(
            panel_title=panel_title,
            signs=signs,
            policies=table.policies,
            out_path=path,
        )
        written.append(path)

    all_signs, boundaries = _all_signs_ordered(table)
    all_path = out_dir / "all_signs_sign_compliance.png"
    _plot_sign_compliance_bars(
        panel_title="All signs — sign compliance (in-zone)",
        signs=all_signs,
        policies=table.policies,
        out_path=all_path,
        group_boundaries=boundaries,
        fig_height=5.6,
    )
    written.append(all_path)

    analysis = out_dir / "analysis.md"
    write_analysis(table, analysis, agg=agg)
    written.append(analysis)
    return written


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Plot baseline comparison charts from cumulative.json reports",
    )
    ap.add_argument("--runs-root", default=str(REPO_ROOT / "data" / "runs"))
    ap.add_argument("--split", default="train", choices=("train", "test", "debug"))
    ap.add_argument("--cumulative", default=None)
    ap.add_argument("--out-dir", default=None, help=f"default: {ALL_RUNS_PLOTS}")
    ap.add_argument("--agg", choices=("episode", "map"), default="episode")
    ap.add_argument("--policies", default=None)
    ap.add_argument("--all-policies", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    policies = None
    if args.policies:
        policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    agg: AggMode = args.agg
    if args.cumulative:
        table = load_from_cumulative(
            Path(args.cumulative),
            agg=agg,
            policies=policies,
            all_policies=args.all_policies,
        )
    else:
        runs_root = resolve_repo_path(args.runs_root)
        table = load_from_runs(
            runs_root,
            split=args.split,
            agg=agg,
            policies=policies,
            all_policies=args.all_policies,
        )

    if not table.signs:
        print("ERROR: no sign slices loaded.", file=sys.stderr)
        raise SystemExit(1)
    if not table.policies:
        print("ERROR: no policies found in reports.", file=sys.stderr)
        raise SystemExit(1)

    out_dir = resolve_repo_path(args.out_dir or ALL_RUNS_PLOTS)
    written = render_benchmark(table, out_dir, agg=agg)

    n_sign = len([p for p in written if p.parent.name == "by_sign"])
    n_family = len([p for p in written if p.parent.name == "by_family"])
    print(f"[plot] signs={len(table.signs)} policies={len(table.policies)} agg={agg}")
    print(f"[plot] out: {out_dir}")
    print(f"[plot] wrote {n_sign} per-sign, {n_family} group charts, all-signs overview, plus summary + analysis")


if __name__ == "__main__":
    main()
