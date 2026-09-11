#!/usr/bin/env python3
"""cumulative.json → markdown.

Every numeric cell shows both aggregations side by side:

    <per-episode> / <per-map>

* per-episode — every episode weighs the same (the original aggregation;
  ``per_baseline`` / ``per_sign`` blocks of cumulative.json);
* per-map — a map's episodes are collapsed first, then the mean is taken over
  maps, so every map contributes exactly one number (``per_baseline_map`` /
  ``per_sign_map`` blocks; see ``aggregate.aggregate_by_map``).

Each table is followed by ``mean ± std [ci_lo, ci_hi]`` over maps. A
cumulative.json without manifest maps (``ci.map_id``) raises.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_bench.eval.metrics.map_id import MAP_ID_SOURCE

REQUIRED_BLOCKS: tuple[str, ...] = (
    "chunks", "per_baseline", "per_sign", "per_baseline_map", "per_sign_map",
    "per_baseline_map_ci", "per_sign_map_ci", "ci",
)


def load_cumulative(path: Path) -> dict:
    """cumulative.json of the current `metrics aggregate`; anything else raises."""
    if not path.is_file():
        raise FileNotFoundError(f"Not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_BLOCKS if k not in data]
    if missing:
        raise ValueError(
            f"{path} lacks {', '.join(missing)}: it was not written by the current "
            "`metrics aggregate`. Rebuild it with `metrics csv --manifest` and "
            "`metrics aggregate`.")
    source = data["ci"].get("map_id")
    if source != MAP_ID_SOURCE:
        raise ValueError(
            f"{path}: its per-map blocks were not keyed on manifest maps "
            f"(ci.map_id = {source!r}, expected {MAP_ID_SOURCE!r}). Rebuild it with "
            "`metrics csv --manifest` and `metrics aggregate`.")
    return data


def _fmt(x) -> str:
    if x is None or x == "":
        return "—"
    return f"{float(x):.3f}"


def _cell(m_ep: dict, m_map: dict, key: str) -> str:
    """``episode / map`` for one metric."""
    return f"{_fmt(m_ep[key])} / {_fmt(m_map[key])}"


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


def _table_header() -> list[str]:
    cols = ["Policy", "Runs", "In-zone runs", "Maps"] + [h for h, _ in METRIC_COLUMNS]
    return [
        "| " + " | ".join(cols) + " |",
        "|---|" + "|".join(["---:"] * (len(cols) - 1)) + "|",
    ]


def _table_row(policy: str, m_ep: dict, m_map: dict) -> str:
    cells = [f"`{_display(policy)}`", str(int(m_ep["n"])), str(int(m_ep["n_in_zone"])),
             str(int(m_map["n_maps"]))]
    cells += [_cell(m_ep, m_map, key) for _, key in METRIC_COLUMNS]
    return "| " + " | ".join(cells) + " |"


def _ci_cell(block: dict, key: str) -> str:
    """``mean ± std [lo, hi]``; "—" when the metric is undefined on every map."""
    d = block.get(key)
    if d is None:
        return "—"
    s = _fmt(d["mean"])
    if d["std"] is not None:
        s += f" ± {float(d['std']):.3f}"
    if d["ci_lo"] is not None and d["ci_hi"] is not None:
        s += f" [{float(d['ci_lo']):.3f}, {float(d['ci_hi']):.3f}]"
    return s


def _ci_table_header() -> list[str]:
    cols = ["Policy", "Maps", "Episodes / map"] + [h for h, _ in METRIC_COLUMNS]
    return [
        "| " + " | ".join(cols) + " |",
        "|---|" + "|".join(["---:"] * (len(cols) - 1)) + "|",
    ]


def _ci_table_row(policy: str, m_map: dict, m_ci: dict) -> str:
    lo, hi = int(m_map["episodes_per_map_min"]), int(m_map["episodes_per_map_max"])
    per_map = str(lo) if lo == hi else f"{lo}–{hi}"
    cells = [f"`{_display(policy)}`", str(int(m_map["n_maps"])), per_map]
    cells += [_ci_cell(m_ci, key) for _, key in METRIC_COLUMNS]
    return "| " + " | ".join(cells) + " |"


def _ci_caption(ci_meta: dict) -> str:
    return (f"per-map mean ± std, {float(ci_meta['level']):.0%} CI "
            f"({int(ci_meta['n_boot'])} resamples)")


def _by_sign(block: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    """``{policy: {sign: m}}`` → ``{sign: {policy: m}}``."""
    out: dict[str, dict[str, dict]] = {}
    for policy, sign_map in block.items():
        for sign, m in sign_map.items():
            out.setdefault(sign, {})[policy] = m
    return out


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
    data = load_cumulative(cumulative_path)
    per_baseline = data["per_baseline"]
    per_baseline_map = data["per_baseline_map"]
    per_baseline_map_ci = data["per_baseline_map_ci"]
    ci_meta = data["ci"]
    by_sign = _by_sign(data["per_sign"])
    by_sign_map = _by_sign(data["per_sign_map"])
    by_sign_map_ci = _by_sign(data["per_sign_map_ci"])

    lines: list[str] = []
    lines.append("# Cumulative Benchmark Results")
    lines.append("")
    lines.append(f"Source: `{cumulative_path}`")
    lines.append("")
    chunks = data["chunks"]
    lines.append(f"Aggregated chunks: {len(chunks)}")
    if chunks:
        lines.append("")
        lines.append("`" + ", ".join(chunks) + "`")
    lines.append("")
    lines.append(
        "Each numeric cell is `per-episode / per-map`: per-episode weighs every "
        "episode the same; per-map collapses each map's episodes first and then "
        "averages over maps (`Maps` = number of maps; a map is the scene net named "
        "by the manifest's `net_path`). `Target SR` = compliance with the target "
        "sign (`target_compliant_event`); `SR&Dest` = target sign obeyed AND "
        "destination reached, over all scored episodes."
    )
    lines.append("")
    lines.append(
        "Dispersion tables (`mean ± std [lo, hi]`): every map is collapsed to one "
        "value per metric (the mean over its augmented variants); `mean` is the "
        "mean of those per-map values, `std` their sample standard deviation, "
        f"`[lo, hi]` the {float(ci_meta['level']):.0%} percentile-bootstrap CI of the "
        f"mean ({int(ci_meta['n_boot'])} resamples of the maps, seed "
        f"{int(ci_meta['seed'])}). `Episodes / map` = augmented variants per map "
        "(min–max when unbalanced)."
    )
    lines.append("")
    lines.append("## Overall (weighted by total_runs across chunks)")
    lines.append("")
    lines.extend(_table_header())
    for policy, m in sorted(per_baseline.items(), key=lambda kv: _display(kv[0])):
        lines.append(_table_row(policy, m, per_baseline_map[policy]))
    lines.append("")
    lines.append(f"### Overall — {_ci_caption(ci_meta)}")
    lines.append("")
    lines.extend(_ci_table_header())
    for policy, m_map in sorted(per_baseline_map.items(), key=lambda kv: _display(kv[0])):
        lines.append(_ci_table_row(policy, m_map, per_baseline_map_ci[policy]))

    lines.append("")
    lines.append("## Per Sign (aggregated across chunks)")
    lines.append("")
    for sign in sorted(by_sign):
        lines.append(f"### Sign `{sign}`")
        lines.append("")
        lines.extend(_table_header())
        for policy, m in sorted(by_sign[sign].items(), key=lambda kv: _display(kv[0])):
            lines.append(_table_row(policy, m, by_sign_map[sign][policy]))
        lines.append("")
        lines.append(f"#### Sign `{sign}` — {_ci_caption(ci_meta)}")
        lines.append("")
        lines.extend(_ci_table_header())
        for policy, m_map in sorted(by_sign_map[sign].items(), key=lambda kv: _display(kv[0])):
            lines.append(_ci_table_row(policy, m_map, by_sign_map_ci[sign][policy]))
        lines.append("")

    out_path = Path(args.out) if args.out else (run_root / "reports" / "report_cumulative.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
