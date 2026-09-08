#!/usr/bin/env python3
"""cumulative.json → markdown.

Every numeric cell shows both aggregations side by side:

    <per-episode> / <per-map>

* per-episode — every episode weighs the same (the original aggregation;
  ``per_baseline`` / ``per_sign`` blocks of cumulative.json);
* per-map — a map's episodes (its augmented variants) are collapsed first,
  then the mean is taken over maps, so every map contributes exactly one
  number (``per_baseline_map`` / ``per_sign_map`` blocks; see
  ``aggregate.aggregate_by_map``).

Each table is followed by a dispersion table over the per-map values:

    <mean> ± <std over maps> [<ci_lo>, <ci_hi>]

with the bootstrap CI of the mean (``per_baseline_map_ci`` / ``per_sign_map_ci``
blocks, parameters in ``ci``).

Old cumulative.json files without the ``*_map`` blocks render only the
per-episode value; files without the ``*_map_ci`` blocks skip the
dispersion tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(x) -> str:
    if x is None or x == "":
        return "—"
    return f"{float(x):.3f}"


def _cell(m_ep: dict, m_map: dict | None, key: str) -> str:
    """``episode / map`` for one metric; only ``episode`` when no map block."""
    ep = _fmt(m_ep.get(key))
    if m_map is None:
        return ep
    return f"{ep} / {_fmt(m_map.get(key))}"


# Display-only renames for the markdown report. cumulative.json keeps raw names.
# Legacy spellings come from cumulative.json files written before the rename.
POLICY_DISPLAY_NAME: dict[str, str] = {
    "comprehensive_rule_expert_default": "idm_rule_default",
    "comprehensive_rule_expert_s1": "idm_rule_s1",
    "comprehensive_rule_expert_s2": "idm_rule_s2",
    "comprehensive_rule_expert_s3": "idm_rule_s3",
    "comprehensive_rule_expert_s4": "idm_rule_s4",
    "ppo_expert": "ppo",
    "rule_compliant": "ppo_rule",
}


def _display(policy: str) -> str:
    return POLICY_DISPLAY_NAME.get(policy, policy)


# (header, metric key) — the order of the numeric columns in every table.
METRIC_COLUMNS: list[tuple[str, str]] = [
    ("Success rate", "success_rate"),
    ("Dest rate", "dest_rate"),
    ("Target SR", "target_compliance_rate_event"),
    ("SR&Dest", "sr_and_dest"),
    ("TL SR", "traffic_light_sr"),
    ("CW SR", "crosswalk_sr"),
    ("Sign compliance SR", "sign_compliance_sr"),
    ("Sign compliance (in-zone)", "sign_compliance_x"),
    ("Avg driving score", "avg_driving_score"),
    ("Avg efficiency", "avg_efficiency"),
    ("Avg smoothness", "avg_smoothness"),
    ("Avg route len (m)", "avg_route_length_m"),
    ("Avg distance (m)", "avg_distance_travelled_m"),
]


def _table_header(with_map: bool) -> list[str]:
    cols = ["Policy", "Runs", "In-zone runs"]
    if with_map:
        cols.append("Maps")
    cols += [h for h, _ in METRIC_COLUMNS]
    return [
        "| " + " | ".join(cols) + " |",
        "|---|" + "|".join(["---:"] * (len(cols) - 1)) + "|",
    ]


def _table_row(policy: str, m_ep: dict, m_map: dict | None, with_map: bool) -> str:
    n = int(m_ep.get("n", 0) or 0)
    nz = int(m_ep.get("n_in_zone", 0) or 0)
    cells = [f"`{_display(policy)}`", str(n), str(nz)]
    if with_map:
        n_maps = (m_map or {}).get("n_maps")
        cells.append(str(int(n_maps)) if n_maps not in (None, "") else "—")
    cells += [_cell(m_ep, m_map, key) for _, key in METRIC_COLUMNS]
    return "| " + " | ".join(cells) + " |"


def _ci_cell(block: dict | None, key: str) -> str:
    """``mean ± std [lo, hi]`` for one metric of a ``*_map_ci`` block."""
    d = (block or {}).get(key)
    if not d:
        return "—"
    s = _fmt(d.get("mean"))
    std = d.get("std")
    if std not in (None, ""):
        s += f" ± {float(std):.3f}"
    lo, hi = d.get("ci_lo"), d.get("ci_hi")
    if lo not in (None, "") and hi not in (None, ""):
        s += f" [{float(lo):.3f}, {float(hi):.3f}]"
    return s


def _ci_table_header() -> list[str]:
    cols = ["Policy", "Maps", "Episodes / map"] + [h for h, _ in METRIC_COLUMNS]
    return [
        "| " + " | ".join(cols) + " |",
        "|---|" + "|".join(["---:"] * (len(cols) - 1)) + "|",
    ]


def _ci_table_row(policy: str, m_map: dict, m_ci: dict | None) -> str:
    n_maps = m_map.get("n_maps")
    lo, hi = m_map.get("episodes_per_map_min"), m_map.get("episodes_per_map_max")
    if lo in (None, ""):
        per_map = "—"
    elif hi in (None, "") or int(lo) == int(hi):
        per_map = str(int(lo))
    else:
        per_map = f"{int(lo)}–{int(hi)}"
    cells = [f"`{_display(policy)}`",
             str(int(n_maps)) if n_maps not in (None, "") else "—",
             per_map]
    cells += [_ci_cell(m_ci, key) for _, key in METRIC_COLUMNS]
    return "| " + " | ".join(cells) + " |"


def _ci_caption(ci_meta: dict) -> str:
    level = ci_meta.get("level")
    level_s = f"{float(level):.0%}" if level not in (None, "") else "bootstrap"
    n_boot = ci_meta.get("n_boot")
    boot_s = f"{int(n_boot)} resamples" if n_boot not in (None, "") else "bootstrap"
    return f"per-map mean ± std, {level_s} CI ({boot_s})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate markdown table from cumulative.json")
    ap.add_argument(
        "--run-root",
        required=True,
        help="Benchmark root containing reports/cumulative.json",
    )
    ap.add_argument(
        "--cumulative",
        default=None,
        help="Path to cumulative JSON (default: <run-root>/reports/cumulative.json). "
             "Use this to render per-var reports built with --cumulative-out.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output markdown path (default: <run-root>/reports/report_cumulative.md)",
    )
    args = ap.parse_args()

    run_root = Path(args.run_root)
    cumulative_path = (Path(args.cumulative) if args.cumulative
                       else run_root / "reports" / "cumulative.json")
    if not cumulative_path.exists():
        raise FileNotFoundError(f"Not found: {cumulative_path}")

    data = json.loads(cumulative_path.read_text(encoding="utf-8"))
    per_baseline = data.get("per_baseline", {})
    per_sign_by_baseline = data.get("per_sign", {})
    # Map-level aggregation blocks (absent in cumulative.json written before
    # aggregate_by_map existed → per-episode values only).
    per_baseline_map = data.get("per_baseline_map") or {}
    per_sign_by_baseline_map = data.get("per_sign_map") or {}
    with_map = bool(per_baseline_map)
    # Dispersion over maps (absent in cumulative.json written before the
    # bootstrap CI existed → no dispersion tables).
    per_baseline_map_ci = data.get("per_baseline_map_ci") or {}
    per_sign_by_baseline_map_ci = data.get("per_sign_map_ci") or {}
    ci_meta = data.get("ci") or {}
    with_ci = with_map and bool(per_baseline_map_ci)

    lines: list[str] = []
    lines.append("# Cumulative Benchmark Results")
    lines.append("")
    lines.append(f"Source: `{cumulative_path}`")
    lines.append("")
    chunks = data.get("chunks", [])
    lines.append(f"Aggregated chunks: {len(chunks)}")
    if chunks:
        lines.append("")
        lines.append("`" + ", ".join(chunks) + "`")
    lines.append("")
    if with_map:
        lines.append(
            "Each numeric cell is `per-episode / per-map`: per-episode weighs every "
            "episode the same; per-map collapses each map's episodes first and then "
            "averages over maps (`Maps` = number of maps). `Target SR` = compliance "
            "with the target sign (`target_compliant_event`); `SR&Dest` = target "
            "sign obeyed AND destination reached, over all scored episodes."
        )
        lines.append("")
    if with_ci:
        level = ci_meta.get("level")
        level_s = f"{float(level):.0%}" if level not in (None, "") else "bootstrap"
        n_boot = ci_meta.get("n_boot")
        lines.append(
            "Dispersion tables (`mean ± std [lo, hi]`): every map is collapsed to one "
            "value per metric (the mean over its augmented variants); `mean` is the "
            "mean of those per-map values, `std` their sample standard deviation, "
            f"`[lo, hi]` the {level_s} percentile-bootstrap CI of the mean"
            + (f" ({int(n_boot)} resamples of the maps"
               + (f", seed {int(ci_meta['seed'])}" if ci_meta.get("seed") not in (None, "") else "")
               + ")" if n_boot not in (None, "") else "")
            + ". `Episodes / map` = augmented variants per map (min–max when unbalanced)."
        )
        lines.append("")
    lines.append("## Overall (weighted by total_runs across chunks)")
    lines.append("")
    lines.extend(_table_header(with_map))

    for policy, m in sorted(per_baseline.items(), key=lambda kv: _display(kv[0])):
        lines.append(_table_row(policy, m, per_baseline_map.get(policy) if with_map else None,
                                with_map))
    if with_ci:
        lines.append("")
        lines.append(f"### Overall — {_ci_caption(ci_meta)}")
        lines.append("")
        lines.extend(_ci_table_header())
        for policy, m_map in sorted(per_baseline_map.items(), key=lambda kv: _display(kv[0])):
            lines.append(_ci_table_row(policy, m_map, per_baseline_map_ci.get(policy)))

    # Aggregate per-sign across baselines for a sign-centric table
    by_sign: dict[str, dict[str, dict]] = {}
    for policy, sign_map in per_sign_by_baseline.items():
        if not isinstance(sign_map, dict):
            continue
        for sign, m in sign_map.items():
            if sign not in by_sign:
                by_sign[sign] = {}
            by_sign[sign][policy] = m
    by_sign_map: dict[str, dict[str, dict]] = {}
    for policy, sign_map in per_sign_by_baseline_map.items():
        if not isinstance(sign_map, dict):
            continue
        for sign, m in sign_map.items():
            by_sign_map.setdefault(sign, {})[policy] = m
    by_sign_map_ci: dict[str, dict[str, dict]] = {}
    for policy, sign_map in per_sign_by_baseline_map_ci.items():
        if not isinstance(sign_map, dict):
            continue
        for sign, m in sign_map.items():
            by_sign_map_ci.setdefault(sign, {})[policy] = m

    lines.append("")
    lines.append("## Per Sign (aggregated across chunks)")
    lines.append("")
    for sign in sorted(by_sign.keys()):
        lines.append(f"### Sign `{sign}`")
        lines.append("")
        lines.extend(_table_header(with_map))
        for policy, m in sorted(by_sign[sign].items(), key=lambda kv: _display(kv[0])):
            m_map = by_sign_map.get(sign, {}).get(policy) if with_map else None
            lines.append(_table_row(policy, m, m_map, with_map))
        lines.append("")
        if with_ci and by_sign_map.get(sign):
            lines.append(f"#### Sign `{sign}` — {_ci_caption(ci_meta)}")
            lines.append("")
            lines.extend(_ci_table_header())
            for policy, m_map in sorted(by_sign_map[sign].items(),
                                        key=lambda kv: _display(kv[0])):
                lines.append(_ci_table_row(policy, m_map,
                                           by_sign_map_ci.get(sign, {}).get(policy)))
            lines.append("")

    out_path = Path(args.out) if args.out else (run_root / "reports" / "report_cumulative.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
