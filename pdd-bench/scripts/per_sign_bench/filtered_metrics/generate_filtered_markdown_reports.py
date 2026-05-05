#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


SIGN_TO_CATEGORY = {
    # Priority
    "2.1": "priority",
    "2.3.1": "priority",
    "2.3.2": "priority",
    "2.3.3": "priority",
    "2.4": "priority",
    "2.5": "priority",
    # Prohibitory
    "3.1": "prohibitory",
    "3.2": "prohibitory",
    "3.18.1": "prohibitory",
    "3.18.2": "prohibitory",
    "3.19": "prohibitory",
    "3.20": "prohibitory",
    "3.24": "prohibitory",
    "3.27": "prohibitory",
    "3.31": "prohibitory",
    # Mandatory
    "4.1.1": "mandatory",
    "4.1.2": "mandatory",
    "4.1.3": "mandatory",
    "4.1.4": "mandatory",
    "4.1.5": "mandatory",
    "4.1.6": "mandatory",
    "4.2.1": "mandatory",
    "4.2.2": "mandatory",
    "4.2.3": "mandatory",
    "4.6": "mandatory",
    # Special regulation
    "5.3": "special",
    "5.5": "special",
    "5.7.1": "special",
    "5.7.2": "special",
    "5.11.1": "special",
    "5.11.2": "special",
    "5.12.2": "special",
    "5.13.1": "special",
    "5.13.2": "special",
    "5.13.3": "special",
    "5.13.4": "special",
    "5.14.1": "special",
    "5.14.2": "special",
    "5.15.2": "special",
    "5.16": "special",
    "5.31": "special",
    "5.32": "special",
}


DEFAULT_EXCLUDED_SIGNS = (
    "2.1",
    "2.2",
    "2.3.1",
    "2.3.2",
    "2.3.3",

    "3.18.1",
    "3.18.2",
    "3.19",
    "3.20",
    "3.21",
    "3.24",
    "3.25",
    "3.27",
    "3.31",
    
    "4.1.1",
    "4.1.2",
    "4.1.3",
    "4.1.4",
    "4.1.5",
    "4.1.6",
    "4.2.1",
    "4.2.2",
    "4.2.3",
    "4.6",
    
    "5.3",
    "5.4",
    "5.5",
    "5.6",
    "5.7.1",
    "5.7.2",
    "5.12.1",
    "5.12.2",
    "5.13.1",
    "5.13.2",
    "5.13.3",
    "5.13.4",
    "5.14.1",
    "5.14.2",
    "5.14.3",
    "5.16",
    "5.19"
    "5.32",
)

METRIC_KEYS = (
    "success_rate",
    "dest_rate",
    "sign_compliance_sr",
    "sign_compliance_x",
    "traffic_light_sr",
    "crosswalk_sr",
    "avg_driving_score",
    "avg_efficiency",
    "avg_smoothness",
    "avg_steps",
    "avg_route_length_m",
    "avg_distance_travelled_m",
)


def _normalize_sign(sign: str) -> str:
    out = str(sign).strip()
    while out.endswith("."):
        out = out[:-1]
    return out


def _fmt(x: float) -> str:
    return f"{float(x):.3f}"


def _weighted_merge(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "n_in_zone": 0}
    n = sum(int(r.get("n", 0) or 0) for r in rows)
    n_in_zone = sum(int(r.get("n_in_zone", 0) or 0) for r in rows)
    if n <= 0:
        return {"n": 0, "n_in_zone": n_in_zone}
    merged: dict[str, float | int] = {"n": n, "n_in_zone": n_in_zone}
    for key in METRIC_KEYS:
        weight_key = "n_in_zone" if key == "sign_compliance_x" else "n"
        denom = n_in_zone if key == "sign_compliance_x" else n
        if denom <= 0:
            merged[key] = 0.0
            continue
        merged[key] = sum(
            float(r.get(key, 0.0)) * int(r.get(weight_key, 0) or 0)
            for r in rows
        ) / denom
    return merged


def _build_filtered_per_sign(
    per_sign_by_policy: dict[str, dict],
    excluded_signs: set[str],
) -> tuple[dict[str, dict], dict[str, dict]]:
    filtered_per_policy: dict[str, dict] = {}
    by_sign: dict[str, dict] = {}
    for policy, sign_map in per_sign_by_policy.items():
        if not isinstance(sign_map, dict):
            continue
        filtered_sign_map: dict[str, dict] = {}
        for sign, metrics in sign_map.items():
            if _normalize_sign(sign) in excluded_signs:
                continue
            filtered_sign_map[sign] = metrics
            by_sign.setdefault(sign, {})[policy] = metrics
        filtered_per_policy[policy] = filtered_sign_map
    return filtered_per_policy, by_sign


def _aggregate_overall(filtered_per_policy: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for policy, sign_map in filtered_per_policy.items():
        out[policy] = _weighted_merge(list(sign_map.values()))
    return out


def _aggregate_categories(filtered_per_policy: dict[str, dict]) -> dict[str, dict]:
    categories = ["priority", "prohibitory", "mandatory", "special"]
    buckets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for policy, sign_map in filtered_per_policy.items():
        for sign, metrics in sign_map.items():
            category = SIGN_TO_CATEGORY.get(_normalize_sign(sign))
            if category is None:
                continue
            buckets[policy][category].append(metrics)

    out: dict[str, dict] = defaultdict(dict)
    for policy, category_rows in buckets.items():
        for category in categories:
            out[policy][category] = _weighted_merge(category_rows.get(category, []))
    return out


def _render_cumulative_markdown(
    cumulative_path: Path,
    chunks: list[str],
    excluded_signs: set[str],
    overall: dict[str, dict],
    by_sign: dict[str, dict],
) -> str:
    lines: list[str] = []
    lines.append("# Filtered Cumulative Benchmark Results")
    lines.append("")
    lines.append(f"Source: `{cumulative_path}`")
    lines.append("")
    lines.append(
        "Excluded signs: `{}`".format(", ".join(sorted(excluded_signs)))
    )
    lines.append("")
    lines.append(f"Aggregated chunks: {len(chunks)}")
    if chunks:
        lines.append("")
        lines.append("`" + ", ".join(chunks) + "`")
    lines.append("")
    lines.append("## Overall (weighted by total_runs across included signs)")
    lines.append("")
    lines.append(
        "| Policy | Runs | In-zone runs | Success rate | Dest rate | TL SR | CW SR | "
        "Sign compliance SR | Sign compliance (in-zone) | "
        "Avg driving score | Avg efficiency | Avg smoothness | Avg route len (m) | Avg distance (m) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for policy, metrics in sorted(overall.items()):
        n = int(metrics.get("n", 0) or 0)
        nz = int(metrics.get("n_in_zone", 0) or 0)
        lines.append(
            f"| `{policy}` | {n} | {nz} | {_fmt(metrics.get('success_rate', 0.0))} | {_fmt(metrics.get('dest_rate', 0.0))} | "
            f"{_fmt(metrics.get('traffic_light_sr', 0.0))} | {_fmt(metrics.get('crosswalk_sr', 0.0))} | "
            f"{_fmt(metrics.get('sign_compliance_sr', 0.0))} | {_fmt(metrics.get('sign_compliance_x', 0.0))} | "
            f"{_fmt(metrics.get('avg_driving_score', 0.0))} | {_fmt(metrics.get('avg_efficiency', 0.0))} | {_fmt(metrics.get('avg_smoothness', 0.0))} | "
            f"{_fmt(metrics.get('avg_route_length_m', 0.0))} | {_fmt(metrics.get('avg_distance_travelled_m', 0.0))} |"
        )

    lines.append("")
    lines.append("## Per Sign (aggregated across chunks, excluded signs removed)")
    lines.append("")
    for sign in sorted(by_sign.keys()):
        lines.append(f"### Sign `{sign}`")
        lines.append("")
        lines.append(
            "| Policy | Runs | In-zone runs | Success rate | Dest rate | TL SR | CW SR | "
            "Sign compliance SR | Sign compliance (in-zone) | "
            "Avg driving score | Avg efficiency | Avg smoothness | Avg route len (m) | Avg distance (m) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for policy, metrics in sorted(by_sign[sign].items()):
            n = int(metrics.get("n", 0) or 0)
            nz = int(metrics.get("n_in_zone", 0) or 0)
            lines.append(
                f"| `{policy}` | {n} | {nz} | {_fmt(metrics.get('success_rate', 0.0))} | {_fmt(metrics.get('dest_rate', 0.0))} | "
                f"{_fmt(metrics.get('traffic_light_sr', 0.0))} | {_fmt(metrics.get('crosswalk_sr', 0.0))} | "
                f"{_fmt(metrics.get('sign_compliance_sr', 0.0))} | {_fmt(metrics.get('sign_compliance_x', 0.0))} | "
                f"{_fmt(metrics.get('avg_driving_score', 0.0))} | {_fmt(metrics.get('avg_efficiency', 0.0))} | {_fmt(metrics.get('avg_smoothness', 0.0))} | "
                f"{_fmt(metrics.get('avg_route_length_m', 0.0))} | {_fmt(metrics.get('avg_distance_travelled_m', 0.0))} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _render_category_markdown(
    cumulative_path: Path,
    excluded_signs: set[str],
    category_agg: dict[str, dict],
) -> str:
    lines = [
        "# Filtered Category-level Cumulative Results",
        "",
        f"Source: `{cumulative_path}`",
        "",
        "Excluded signs: `{}`".format(", ".join(sorted(excluded_signs))),
        "",
        "## Success Rate by Category",
        "",
        "| Policy | Priority | Prohibitory | Mandatory | Special |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in sorted(category_agg.keys()):
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                policy,
                _fmt(category_agg[policy]["priority"].get("success_rate", 0.0)),
                _fmt(category_agg[policy]["prohibitory"].get("success_rate", 0.0)),
                _fmt(category_agg[policy]["mandatory"].get("success_rate", 0.0)),
                _fmt(category_agg[policy]["special"].get("success_rate", 0.0)),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate filtered cumulative markdown reports from "
            "<run-root>/reports/cumulative_var_0.json in a separate output folder."
        )
    )
    ap.add_argument("--run-root", required=True, help="Benchmark root that contains reports/cumulative_var_0.json")
    ap.add_argument(
        "--out-cumulative-md",
        default=None,
        help=(
            "Output path for filtered cumulative markdown "
            "(default: <run-root>/filtered_metrics/reports/report_cumulative.md)"
        ),
    )
    ap.add_argument(
        "--out-categories-md",
        default=None,
        help=(
            "Output path for filtered categories markdown "
            "(default: <run-root>/filtered_metrics/reports/report_cumulative_categories.md)"
        ),
    )
    ap.add_argument(
        "--exclude-sign",
        action="append",
        default=[],
        help="Additional sign code to exclude. Can be passed multiple times.",
    )
    args = ap.parse_args()

    run_root = Path(args.run_root)
    cumulative_path = run_root / "reports" / "cumulative_var_0.json"
    if not cumulative_path.exists():
        raise FileNotFoundError(f"Not found: {cumulative_path}")

    excluded_signs = {_normalize_sign(sign) for sign in DEFAULT_EXCLUDED_SIGNS}
    excluded_signs.update(_normalize_sign(sign) for sign in args.exclude_sign)

    data = json.loads(cumulative_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    per_sign_by_policy = data.get("per_sign", {})

    filtered_per_policy, by_sign = _build_filtered_per_sign(per_sign_by_policy, excluded_signs)
    overall = _aggregate_overall(filtered_per_policy)
    category_agg = _aggregate_categories(filtered_per_policy)

    default_out_dir = run_root / "filtered_metrics" / "reports"
    cumulative_out = (
        Path(args.out_cumulative_md)
        if args.out_cumulative_md
        else default_out_dir / "report_cumulative.md"
    )
    categories_out = (
        Path(args.out_categories_md)
        if args.out_categories_md
        else default_out_dir / "report_cumulative_categories.md"
    )

    cumulative_out.parent.mkdir(parents=True, exist_ok=True)
    categories_out.parent.mkdir(parents=True, exist_ok=True)

    cumulative_text = _render_cumulative_markdown(
        cumulative_path=cumulative_path,
        chunks=chunks,
        excluded_signs=excluded_signs,
        overall=overall,
        by_sign=by_sign,
    )
    categories_text = _render_category_markdown(
        cumulative_path=cumulative_path,
        excluded_signs=excluded_signs,
        category_agg=category_agg,
    )

    cumulative_out.write_text(cumulative_text, encoding="utf-8")
    categories_out.write_text(categories_text, encoding="utf-8")

    print(cumulative_out)
    print(categories_out)


if __name__ == "__main__":
    main()
