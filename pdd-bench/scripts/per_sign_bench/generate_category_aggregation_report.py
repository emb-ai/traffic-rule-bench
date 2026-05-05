#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


# SIGN_TO_CATEGORY = {
#     # Priority
#     "2.1": "priority",
#     # "2.2": "priority",
#     "2.3.1": "priority",
#     "2.3.2": "priority",
#     "2.3.3": "priority",
#     "2.4": "priority",
#     "2.5": "priority",
#     # Prohibitory
#     "3.1": "prohibitory",
#     "3.2": "prohibitory",
#     "3.18.1": "prohibitory",
#     "3.18.2": "prohibitory",
#     "3.19": "prohibitory",
#     "3.20": "prohibitory",
#     "3.24": "prohibitory",
#     "3.27": "prohibitory",
#     # "3.31": "prohibitory",
#     # Mandatory
#     "4.1.1": "mandatory",
#     "4.1.2": "mandatory",
#     "4.1.3": "mandatory",
#     "4.1.4": "mandatory",
#     "4.1.5": "mandatory",
#     "4.1.6": "mandatory",
#     "4.2.1": "mandatory",
#     "4.2.2": "mandatory",
#     "4.2.3": "mandatory",
#     "4.6": "mandatory",
#     # Special regulation
#     "5.3": "special",
#     "5.5": "special",
#     "5.7.1": "special",
#     "5.7.2": "special",
#     "5.11.1": "special",
#     "5.11.2": "special",
#     "5.12.2": "special",
#     "5.13.1": "special",
#     "5.13.2": "special",
#     "5.13.3": "special",
#     "5.13.4": "special",
#     "5.14.1": "special",
#     "5.14.2": "special",
#     "5.15.2": "special",
#     "5.16": "special",
#     "5.31": "special",
#     "5.32": "special",
# }
SIGN_TO_CATEGORY = {
    "2.1": "Maneuvr",
    "2.2": "exclude",
    "2.3.1": "exclude",
    "2.3.2": "exclude",
    "2.3.3": "Maneuvr",
    "2.4": "Maneuvr",
    "2.5": "exclude",
    "3.1": "Navigation",
    "3.18.1": "Navigation",
    "3.18.2": "Navigation",
    "3.19": "Navigation",
    "3.2": "Navigation",
    "3.20": "Maneuvr",
    "3.21": "exclude",
    "3.24": "Maneuvr",
    "3.25": "exclude",
    "3.27": "Maneuvr",
    "3.31": "exclude",
    "4.1.1": "Navigation",
    "4.1.2": "Navigation",
    "4.1.3": "Navigation",
    "4.1.4": "Navigation",
    "4.1.5": "Navigation",
    "4.1.6": "Navigation",
    "4.2.1": "Maneuvr",
    "4.2.2": "Maneuvr",
    "4.2.3": "Maneuvr",
    "4.6": "Maneuvr",
    "5.11.1": "Maneuvr",
    "5.11.2": "Maneuvr",
    "5.12.1": "exclude",
    "5.12.2": "exclude",
    "5.13.1": "Maneuvr",
    "5.13.2": "Maneuvr",
    "5.13.3": "Maneuvr",
    "5.13.4": "Maneuvr",
    "5.14.1": "exclude",
    "5.14.2": "Maneuvr",
    "5.14.3": "exclude",
    "5.15.2": "Maneuvr",
    "5.16": "Maneuvr",
    "5.3": "exclude",
    "5.4": "exclude",
    "5.31": "Maneuvr",
    "5.32": "exclude",
    "5.5": "Navigation",
    "5.7.1": "Navigation",
    "5.7.2": "Navigation",
 }

def _fmt(x: float) -> str:
    return f"{float(x):.3f}"


# Display-only policy renames (kept in sync with generate_cumulative_markdown_report.py).
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


def _weighted_merge(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "n_in_zone": 0, "sign_compliance_x": 0.0}
    n = sum(int(r.get("n", 0) or 0) for r in rows)
    if n <= 0:
        return {"n": 0, "n_in_zone": 0, "sign_compliance_x": 0.0}
    out = {"n": n}
    keys = (
        "success_rate", "dest_rate", "sign_compliance_sr", "traffic_light_sr", "crosswalk_sr",
        "avg_driving_score", "avg_efficiency", "avg_smoothness", "avg_steps",
        "avg_route_length_m", "avg_distance_travelled_m",
    )
    for k in keys:
        out[k] = sum(float(r.get(k, 0.0)) * int(r.get("n", 0) or 0) for r in rows) / n
    # n_in_zone: sum across signs.
    nz = sum(int(r.get("n_in_zone", 0) or 0) for r in rows)
    out["n_in_zone"] = nz
    # sign_compliance_x: weighted-mean by n_in_zone.
    if nz > 0:
        out["sign_compliance_x"] = (
            sum(float(r.get("sign_compliance_x", 0.0)) * int(r.get("n_in_zone", 0) or 0)
                for r in rows) / nz
        )
    else:
        out["sign_compliance_x"] = 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate cumulative per-sign metrics by sign category")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--cumulative", default=None,
                    help="Path to cumulative JSON (default: <run-root>/reports/cumulative.json). "
                         "Use this for per-var cumulatives produced by aggregate_legacy_format --cumulative-out.")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-tex", default=None)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    cumulative = (Path(args.cumulative) if args.cumulative
                  else run_root / "reports" / "cumulative.json")
    data = json.loads(cumulative.read_text(encoding="utf-8"))
    per_sign = data.get("per_sign", {})

    # policy -> category -> list[metric dict]
    buckets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for policy, sign_map in per_sign.items():
        if not isinstance(sign_map, dict):
            continue
        for sign, m in sign_map.items():
            cat = SIGN_TO_CATEGORY.get(str(sign))
            if cat is None:
                continue
            buckets[policy][cat].append(m)

    #cats = ["priority", "prohibitory", "mandatory", "special"]
    cats = ["Maneuvr", "Navigation", "None"]
    agg: dict[str, dict[str, dict]] = defaultdict(dict)
    for policy, by_cat in buckets.items():
        for cat in cats:
            agg[policy][cat] = _weighted_merge(by_cat.get(cat, []))

    sorted_policies = sorted(agg.keys(), key=_display)

    def _table(title: str, field: str, fmt_field: str | None = None) -> list[str]:
        out = [
            f"## {title}",
            "",
            #"| Policy | Priority | Prohibitory | Mandatory | Special |",
            "| Policy | Maneuvr | Navigation | None |",
            "|---|---:|---:|---:|---:|",
        ]
        for p in sorted_policies:
            fmt = (lambda v: str(int(v))) if fmt_field == "int" else _fmt
            out.append(
                "| `{}` | {} | {} | {} | {} |".format(
                    _display(p),
                    fmt(agg[p]["Maneuvr"].get(field, 0)),
                    fmt(agg[p]["Navigation"].get(field, 0)),
                    fmt(agg[p]["None"].get(field, 0)),
                    #fmt(agg[p]["special"].get(field, 0)),
                )
            )
        out.append("")
        return out

    md_lines: list[str] = [
        "# Category-level Cumulative Results",
        "",
        f"Source: `{cumulative}`",
        "",
    ]
    md_lines += _table("Success Rate by Category", "success_rate")
    md_lines += _table("Sign Compliance SR by Category (all episodes)", "sign_compliance_sr")
    md_lines += _table("Sign Compliance (in-zone) by Category", "sign_compliance_x")
    md_lines += _table("Runs by Category", "n", fmt_field="int")
    md_lines += _table("In-zone Runs by Category", "n_in_zone", fmt_field="int")

    md_out = Path(args.out_md) if args.out_md else (run_root / "reports" / "report_cumulative_categories.md")
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # latex
    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Planner} & \\textbf{Priority} & \\textbf{Prohibitory} & \\textbf{Mandatory} & \\textbf{Special} \\\\",
        "\\midrule",
    ]
    for policy in sorted_policies:
        tex_lines.append(
            "{} & {} & {} & {} & {} \\\\".format(
                _display(policy).replace("_", "\\_"),
                _fmt(agg[policy]["priority"].get("success_rate", 0.0)),
                _fmt(agg[policy]["prohibitory"].get("success_rate", 0.0)),
                _fmt(agg[policy]["mandatory"].get("success_rate", 0.0)),
                _fmt(agg[policy]["special"].get("success_rate", 0.0)),
            )
        )
    tex_lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Category-level success rate aggregated from cumulative per-sign results.}",
        "\\label{tab:category_success}",
        "\\end{table}",
        "",
    ]
    tex_out = Path(args.out_tex) if args.out_tex else (run_root / "reports" / "report_cumulative_categories.tex")
    tex_out.write_text("\n".join(tex_lines), encoding="utf-8")

    print(md_out)
    print(tex_out)


if __name__ == "__main__":
    main()
