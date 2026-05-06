#!/usr/bin/env python3
"""compute_carl_metrics.py — carl vs carl_rule metrics from sidecars.

Uses the same functions as select_from_replays.py:
  - parse_sidecar_to_row from select_from_replays
  - passes_filter, recompute_dest, comfort, f1_score, time_eff from select_experts
  - SIGN_CLASS_MAP

Unlike select_from_replays.py, this does NOT require the full set of 8 experts
and does no selection — it just aggregates raw metrics across the two policies.

Usage:
  python3 compute_carl_metrics.py --root <ROOT> [--output report.md]
  python3 compute_carl_metrics.py --root <ROOT> --signs 2.1 3.1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "expert_selection_exps"))

from select_experts import (    # noqa: E402
    SIGN_CLASS_MAP, HORIZON_DEFAULT, BETA_DEFAULT,
    MIN_ROUTE_COMPLETION, MIN_FINAL_STEP,
    passes_filter, recompute_dest, comfort, f1_score, time_eff,
)
from select_from_replays import parse_sidecar_to_row    # noqa: E402


POLICIES = ("carl", "carl_rule")


def collect_rows(root: Path, policies=POLICIES):
    """Scan replays/<sign>/by_sign/<sign>/by_scene/<uid>/<expert>/replay.json
    under $ROOT/runs/chunk_*/{policy}/replays/ for each policy in the list.
    Returns dict[policy] -> list of rows.
    """
    by_policy = {p: [] for p in policies}
    for policy in policies:
        for sc in root.glob(
            f"runs/chunk_*/{policy}/replays/*/by_sign/*/by_scene/*/*/replay.json"
        ):
            row = parse_sidecar_to_row(sc)
            if row is None:
                continue
            if row.get("policy") != policy:
                continue
            by_policy[policy].append(row)
    return by_policy


def metrics_for_rows(rows, args, scene_minmax=None):
    """Same as compute_oracle_metrics::metrics_for_rows from select_from_replays,
    but without depending on the full oracle pipeline."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    n_passing = n_arrived = n_crashed = n_oor = 0
    sum_viols = n_compliant = n_arrived_compliant = 0
    sum_total_viols = n_compliant_total = 0
    sum_comfort = sum_te = sum_f1 = 0.0
    for r in rows:
        sc = r.get("sign_code")
        tc = SIGN_CLASS_MAP.get(sc) if sc else None
        arrived = (recompute_dest(r, sc, tc, args.horizon)
                   if tc else bool(r.get("arrived_dest")))
        crashed = bool(r.get("crashed"))
        oor = bool(r.get("out_of_road"))
        vbc = r.get("violations_by_class") or {}
        target_viols = int(vbc.get(tc, 0) or 0) if tc else 0
        total_viols = sum(int(v or 0) for v in vbc.values())
        compliant = (target_viols == 0)
        compliant_total = (total_viols == 0)
        if arrived: n_arrived += 1
        if crashed: n_crashed += 1
        if oor: n_oor += 1
        if compliant: n_compliant += 1
        if compliant_total: n_compliant_total += 1
        if arrived and compliant: n_arrived_compliant += 1
        sum_viols += target_viols
        sum_total_viols += total_viols
        sum_comfort += comfort(r)
        if tc and passes_filter(r, sc, tc, args.horizon,
                                args.min_route_completion, args.min_final_step):
            n_passing += 1
            if scene_minmax is not None:
                mm = scene_minmax.get(r.get("scene_id"), [1, 1])
                t = time_eff(r, mm[0], mm[1], args.time_eff_formula)
                c = comfort(r)
                sum_te += t
                sum_f1 += f1_score(t, c, args.beta)
    return {
        "n": n,
        "n_passing": n_passing,
        "dest_rate": n_arrived / n,
        "crash_rate": n_crashed / n,
        "out_of_road_rate": n_oor / n,
        "avg_violations": sum_viols / n,
        "avg_total_violations": sum_total_viols / n,
        "compliance_rate": n_compliant / n,
        "compliance_rate_total": n_compliant_total / n,
        "compliance_arrived_n": n_arrived,
        "compliance_arrived_pct": (n_arrived_compliant / n_arrived) if n_arrived else 0,
        "avg_comfort": sum_comfort / n,
        "avg_time_eff_passing": (sum_te / n_passing) if n_passing else 0,
        "avg_f1_passing": (sum_f1 / n_passing) if n_passing else 0,
        "pass_rate": n_passing / n,
    }


def compute_scene_minmax(by_policy, args):
    """Compute min/max final_step across passing trajectories per scene_id —
    used for time_eff normalization."""
    mm = defaultdict(lambda: [10**9, 0])
    for rows in by_policy.values():
        for r in rows:
            sc = r.get("sign_code")
            tc = SIGN_CLASS_MAP.get(sc) if sc else None
            if not tc:
                continue
            if passes_filter(r, sc, tc, args.horizon,
                             args.min_route_completion, args.min_final_step):
                fs = max(1, int(r.get("final_step") or 1))
                cell = mm[r.get("scene_id")]
                cell[0] = min(cell[0], fs)
                cell[1] = max(cell[1], fs)
    return dict(mm)


def render_markdown(by_policy, args, signs_filter=None):
    out = []
    scene_minmax = compute_scene_minmax(by_policy, args)

    def filt(rows):
        if signs_filter:
            return [r for r in rows if r.get("sign_code") in signs_filter]
        return rows

    out.append("
    out.append(f"_horizon={args.horizon}, beta={args.beta}, "
               f"min_route_completion={args.min_route_completion}, "
               f"min_final_step={args.min_final_step}, "
               f"time_eff_formula={args.time_eff_formula}_\n")

    # === ALL SIGNS ===
    out.append("## All signs\n")
    out.append("| metric | carl | carl_rule | Δ (rule − carl) |")
    out.append("| --- | --- | --- | --- |")
    metrics_all = {p: metrics_for_rows(filt(by_policy[p]), args, scene_minmax)
                   for p in POLICIES}
    metric_defs = [
        ("n (episodes)",                   "n",                       "{:,}",  False),
        ("n_passing",                      "n_passing",               "{:,}",  False),
        ("dest_rate (recomputed)",         "dest_rate",               "{:.2f}", True),
        ("crash_rate",                     "crash_rate",              "{:.2f}", True),
        ("out_of_road_rate",               "out_of_road_rate",        "{:.2f}", True),
        ("avg_violations (target)",        "avg_violations",          "{:.2f}", True),
        ("avg_total_violations",           "avg_total_violations",    "{:.2f}", True),
        ("compliance_rate (target)",       "compliance_rate",         "{:.2f}", True),
        ("compliance_rate_total",          "compliance_rate_total",   "{:.2f}", True),
        ("compliance among arrived",       "compliance_arrived_pct",  "{:.2f}", True),
        ("avg_comfort",                    "avg_comfort",             "{:.3f}", True),
        ("avg_time_eff (passing)",         "avg_time_eff_passing",    "{:.3f}", True),
        (f"F1 beta={args.beta} (passing)", "avg_f1_passing",          "{:.3f}", True),
        ("pass_rate",                      "pass_rate",               "{:.2f}", True),
    ]
    for label, key, fmt, has_delta in metric_defs:
        c = metrics_all["carl"].get(key, 0)
        r = metrics_all["carl_rule"].get(key, 0)
        if has_delta:
            d = r - c
            sign = "+" if d > 0 else ""
            d_str = f"{sign}{fmt.format(d)}"
        else:
            d_str = "—"
        out.append(f"| {label} | {fmt.format(c)} | {fmt.format(r)} | {d_str} |")
    out.append("")

    # === PER-SIGN ===
    out.append("## Per-sign breakdown\n")
    all_signs = sorted({r.get("sign_code") for p in POLICIES for r in by_policy[p]
                        if r.get("sign_code")})
    if signs_filter:
        all_signs = [s for s in all_signs if s in signs_filter]

    out.append("| sign | n_carl | dest_carl | viol_carl | F1_carl | "
               "n_rule | dest_rule | viol_rule | F1_rule |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for sign in all_signs:
        c_rows = [r for r in by_policy["carl"]      if r.get("sign_code") == sign]
        r_rows = [r for r in by_policy["carl_rule"] if r.get("sign_code") == sign]
        mc = metrics_for_rows(c_rows, args, scene_minmax)
        mr = metrics_for_rows(r_rows, args, scene_minmax)
        out.append(
            f"| {sign} | "
            f"{mc.get('n', 0)} | {mc.get('dest_rate', 0):.2f} | "
            f"{mc.get('avg_violations', 0):.2f} | {mc.get('avg_f1_passing', 0):.3f} | "
            f"{mr.get('n', 0)} | {mr.get('dest_rate', 0):.2f} | "
            f"{mr.get('avg_violations', 0):.2f} | {mr.get('avg_f1_passing', 0):.3f} |"
        )
    out.append("")

    # === HEAD-TO-HEAD on common scenes ===
    def key(r): return (r.get("sign_code"), r.get("scene_id"))
    c_by_key = {key(r): r for r in by_policy["carl"]}
    r_by_key = {key(r): r for r in by_policy["carl_rule"]}
    common = c_by_key.keys() & r_by_key.keys()
    if signs_filter:
        common = {k for k in common if k[0] in signs_filter}

    if common:
        out.append("
        out.append(f"_common (sign_code, scene_id): **{len(common):,}**_\n")

        c_common = [c_by_key[k] for k in common]
        r_common = [r_by_key[k] for k in common]
        mc = metrics_for_rows(c_common, args, scene_minmax)
        mr = metrics_for_rows(r_common, args, scene_minmax)

        out.append("| metric | carl | carl_rule | Δ |")
        out.append("| --- | --- | --- | --- |")
        for label, key_, fmt, has_delta in metric_defs[2:]:
            c = mc.get(key_, 0)
            r = mr.get(key_, 0)
            d = r - c
            sign = "+" if d > 0 else ""
            out.append(f"| {label} | {fmt.format(c)} | {fmt.format(r)} | {sign}{fmt.format(d)} |")
        out.append("")

        # win/lose summary
        rule_wins_dest = sum(1 for k in common
            if recompute_dest(r_by_key[k], k[0], SIGN_CLASS_MAP.get(k[0]), args.horizon)
               and not recompute_dest(c_by_key[k], k[0], SIGN_CLASS_MAP.get(k[0]), args.horizon))
        carl_wins_dest = sum(1 for k in common
            if recompute_dest(c_by_key[k], k[0], SIGN_CLASS_MAP.get(k[0]), args.horizon)
               and not recompute_dest(r_by_key[k], k[0], SIGN_CLASS_MAP.get(k[0]), args.horizon))
        rule_better_viol = sum(1 for k in common
            if (r_by_key[k].get("violations_by_class") or {}).get(SIGN_CLASS_MAP.get(k[0]), 0)
            < (c_by_key[k].get("violations_by_class") or {}).get(SIGN_CLASS_MAP.get(k[0]), 0))
        carl_better_viol = sum(1 for k in common
            if (c_by_key[k].get("violations_by_class") or {}).get(SIGN_CLASS_MAP.get(k[0]), 0)
            < (r_by_key[k].get("violations_by_class") or {}).get(SIGN_CLASS_MAP.get(k[0]), 0))

        out.append("### Wins / losses\n")
        out.append("| Scenario | Count |")
        out.append("| --- | --- |")
        out.append(f"| rule arrived, carl did not | {rule_wins_dest} ({100*rule_wins_dest/len(common):.1f}%) |")
        out.append(f"| carl arrived, rule did not | {carl_wins_dest} ({100*carl_wins_dest/len(common):.1f}%) |")
        out.append(f"| rule fewer target violations | {rule_better_viol} ({100*rule_better_viol/len(common):.1f}%) |")
        out.append(f"| carl fewer target violations | {carl_better_viol} ({100*carl_better_viol/len(common):.1f}%) |")
        out.append("")

    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="Directory containing runs/ (e.g. benchmark_carl_only)")
    p.add_argument("--output", default=None,
                   help="Where to write the markdown report. Default: stdout.")
    p.add_argument("--signs", nargs="*", default=None,
                   help="Filter by sign codes. Empty = all signs.")
    p.add_argument("--horizon", type=int, default=HORIZON_DEFAULT)
    p.add_argument("--beta", type=float, default=BETA_DEFAULT)
    p.add_argument("--min-route-completion", type=float, default=MIN_ROUTE_COMPLETION)
    p.add_argument("--min-final-step", type=int, default=MIN_FINAL_STEP)
    p.add_argument("--time-eff-formula",
                   choices=["min_over_final", "inv_final_max", "linear", "absolute"],
                   default="min_over_final")
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"=== Scanning {root}/runs/chunk_*/ ===", file=sys.stderr)
    by_policy = collect_rows(root)
    for pol, rows in by_policy.items():
        print(f"  {pol}: {len(rows):,} sidecars", file=sys.stderr)

    signs_filter = set(args.signs) if args.signs else None
    md = render_markdown(by_policy, args, signs_filter)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(md)
        print(f"\n=== Written → {args.output} ===", file=sys.stderr)
    else:
        print(md)


if __name__ == "__main__":
    main()
