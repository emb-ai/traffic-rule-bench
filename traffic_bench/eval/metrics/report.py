#!/usr/bin/env python3
"""cumulative.json → markdown.

Every numeric cell shows both aggregations side by side:

    <per-episode> / <per-map>

* per-episode — every episode weighs the same (the original aggregation;
  ``per_baseline`` / ``per_sign`` blocks of cumulative.json);
* per-map — a map's episodes are collapsed first, then the mean is taken over
  maps, so every map contributes exactly one number (``per_baseline_map`` /
  ``per_sign_map`` blocks; see ``aggregate.aggregate_by_map``).

Old cumulative.json files without the ``*_map`` blocks render only the
per-episode value.
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
    lines.append("## Overall (weighted by total_runs across chunks)")
    lines.append("")
    lines.extend(_table_header(with_map))

    for policy, m in sorted(per_baseline.items(), key=lambda kv: _display(kv[0])):
        lines.append(_table_row(policy, m, per_baseline_map.get(policy) if with_map else None,
                                with_map))

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

    out_path = Path(args.out) if args.out else (run_root / "reports" / "report_cumulative.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
