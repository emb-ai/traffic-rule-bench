#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(x: float) -> str:
    return f"{float(x):.3f}"


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
    lines.append("## Overall (weighted by total_runs across chunks)")
    lines.append("")
    lines.append(
        "| Policy | Runs | In-zone runs | Success rate | Dest rate | TL SR | CW SR | "
        "Sign compliance SR | Sign compliance (in-zone) | "
        "Avg driving score | Avg efficiency | Avg smoothness | Avg route len (m) | Avg distance (m) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for policy, m in sorted(per_baseline.items(), key=lambda kv: _display(kv[0])):
        n = int(m.get("n", 0) or 0)
        nz = int(m.get("n_in_zone", 0) or 0)
        lines.append(
            f"| `{_display(policy)}` | {n} | {nz} | {_fmt(m.get('success_rate', 0.0))} | {_fmt(m.get('dest_rate', 0.0))} | "
            f"{_fmt(m.get('traffic_light_sr', 0.0))} | {_fmt(m.get('crosswalk_sr', 0.0))} | "
            f"{_fmt(m.get('sign_compliance_sr', 0.0))} | {_fmt(m.get('sign_compliance_x', 0.0))} | "
            f"{_fmt(m.get('avg_driving_score', 0.0))} | {_fmt(m.get('avg_efficiency', 0.0))} | {_fmt(m.get('avg_smoothness', 0.0))} | "
            f"{_fmt(m.get('avg_route_length_m', 0.0))} | {_fmt(m.get('avg_distance_travelled_m', 0.0))} |"
        )

    # Aggregate per-sign across baselines for a sign-centric table
    by_sign: dict[str, dict[str, dict]] = {}
    for policy, sign_map in per_sign_by_baseline.items():
        if not isinstance(sign_map, dict):
            continue
        for sign, m in sign_map.items():
            if sign not in by_sign:
                by_sign[sign] = {}
            by_sign[sign][policy] = m

    lines.append("")
    lines.append("## Per Sign (aggregated across chunks)")
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
        for policy, m in sorted(by_sign[sign].items(), key=lambda kv: _display(kv[0])):
            n = int(m.get("n", 0) or 0)
            nz = int(m.get("n_in_zone", 0) or 0)
            lines.append(
                f"| `{_display(policy)}` | {n} | {nz} | {_fmt(m.get('success_rate', 0.0))} | {_fmt(m.get('dest_rate', 0.0))} | "
                f"{_fmt(m.get('traffic_light_sr', 0.0))} | {_fmt(m.get('crosswalk_sr', 0.0))} | "
                f"{_fmt(m.get('sign_compliance_sr', 0.0))} | {_fmt(m.get('sign_compliance_x', 0.0))} | "
                f"{_fmt(m.get('avg_driving_score', 0.0))} | {_fmt(m.get('avg_efficiency', 0.0))} | {_fmt(m.get('avg_smoothness', 0.0))} | "
                f"{_fmt(m.get('avg_route_length_m', 0.0))} | {_fmt(m.get('avg_distance_travelled_m', 0.0))} |"
            )
        lines.append("")

    out_path = Path(args.out) if args.out else (run_root / "reports" / "report_cumulative.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
