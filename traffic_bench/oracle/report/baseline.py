#!/usr/bin/env python3
"""Add an `oracle_rule` baseline to metrics_per_episode.csv.

Semantics: for each unique (var_idx, scene_uid), oracle = the BEST compliant
rule-expert run if at least one was compliant; else "failed scene" (the best
non-compliant run is kept, but success/arrived flags are forced to False so
the scene contributes to denominators but NOT to numerators of success rates).

Picking among rule-experts uses lex sort (descending):
  1. target_compliant_event   (in-zone sign compliance — primary)
  2. arrived_dest             (reached destination)
  3. route_completion         (further along the route)
  4. -total_violations        (fewer total violations)

For scenes where ALL rule experts failed compliance, the picked row's bool
indicators are overridden to mark oracle as "didn't manage":
  arrived_dest=False, success=False, dest_recomputed=False
This way:
  • The scene IS in oracle's row count (proper denominator).
  • dest_rate / sign_compliance_x / dest_x_comp don't credit oracle for it.
  • route_completion / distance_travelled / smoothness / etc. stay as-is —
    they reflect what the best-attempt expert physically did.

Rule experts (default list — override with --rule-experts):
  • carl_rule, plant2_rule, plant2_smirnova_rule, plant2_artem_rule
  • comprehensive_rule_expert_default / s1 / s2 / s3 / s4
  • rule_compliant  (a.k.a. ppo_rule)

Usage:
  python -m traffic_bench.oracle.report.baseline --csv path/to/metrics_per_episode.csv
  python -m traffic_bench.oracle.report.baseline --csv ... --baseline-name oracle_rule \\
      --rule-experts "carl_rule,plant2_rule,..."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


DEFAULT_RULE_EXPERTS = [
    "carl_rule",
    "plant2_rule",
    "plant2_smirnova_rule",
    "plant2_artem_rule",
    "comprehensive_rule_expert_default",
    "comprehensive_rule_expert_s1",
    "comprehensive_rule_expert_s2",
    "comprehensive_rule_expert_s3",
    "comprehensive_rule_expert_s4",
    "rule_compliant",
]


def _to_bool(s: pd.Series) -> pd.Series:
    """Coerce mixed-type bool/int/str series to int 0/1."""
    if s.dtype == bool:
        return s.astype(int)
    return s.fillna(False).map(
        lambda v: 1 if str(v).strip().lower() in ("true", "1", "yes") else 0
    ).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="metrics_per_episode.csv (modified in-place by default)")
    ap.add_argument("--out", default=None,
                    help="Output path (default: overwrite --csv)")
    ap.add_argument("--baseline-name", default="oracle_rule",
                    help="Name for the synthetic baseline (default: oracle_rule)")
    ap.add_argument("--rule-experts", default=",".join(DEFAULT_RULE_EXPERTS),
                    help="Comma-separated baselines to consider for oracle")
    args = ap.parse_args()

    rule_experts = {b.strip() for b in args.rule_experts.split(",") if b.strip()}
    src = Path(args.csv).resolve()
    dst = Path(args.out).resolve() if args.out else src

    df = pd.read_csv(src, low_memory=False)
    print(f"[oracle] loaded {len(df)} rows from {src}")
    print(f"[oracle] rule experts to consider: {sorted(rule_experts)}")

    # Drop existing oracle rows (idempotent)
    if (df["baseline"] == args.baseline_name).any():
        n = (df["baseline"] == args.baseline_name).sum()
        df = df[df["baseline"] != args.baseline_name]
        print(f"[oracle] removed {n} existing {args.baseline_name} rows")

    rule_df = df[df["baseline"].isin(rule_experts)].copy()
    if rule_df.empty:
        print(f"WARN: no rule-expert rows found for {sorted(rule_experts)}",
              file=sys.stderr)
        sys.exit(1)
    print(f"[oracle] {len(rule_df)} rule-expert rows across "
          f"{rule_df['baseline'].nunique()} baselines: "
          f"{sorted(rule_df['baseline'].unique())}")

    # Build sort keys (all numeric, descending)
    rule_df["_compliant"] = _to_bool(rule_df["target_compliant_event"])
    rule_df["_arrived"]   = _to_bool(rule_df["arrived_dest"])
    rule_df["_rc"]        = pd.to_numeric(rule_df["route_completion"],
                                           errors="coerce").fillna(0.0)
    rule_df["_neg_viol"]  = -pd.to_numeric(rule_df["total_violations"],
                                            errors="coerce").fillna(0.0)

    sort_keys = ["_compliant", "_arrived", "_rc", "_neg_viol"]
    rule_df = rule_df.sort_values(sort_keys, ascending=[False] * len(sort_keys))

    dedup_keys = ["var_idx", "scene_id", "scene_uid"]
    n_in_set_check = rule_df[dedup_keys].drop_duplicates().shape[0]
    n_unique_scene_uid = rule_df["scene_uid"].nunique()
    if n_in_set_check != n_unique_scene_uid:
        print(f"[oracle] note: {n_in_set_check} unique (var_idx,scene_id,scene_uid) "
              f"vs {n_unique_scene_uid} unique scene_uid alone — using full key")
    oracle = rule_df.drop_duplicates(subset=dedup_keys, keep="first").copy()
    # Mark scenes where NO rule expert was compliant as oracle-failures.
    # We force success/arrived flags to False so the scene contributes to
    # denominators (n, n_in_zone) but NOT to dest_rate / sign_compliance_x /
    # dest_x_comp_* numerators.
    failed_mask = oracle["_compliant"] == 0
    n_failed = int(failed_mask.sum())
    n_or = len(oracle)

    if n_failed:
        # Bool/int outcome columns
        for col, default in [
            ("arrived_dest",          False),
            ("dest_recomputed",       False),
            ("success",               False),
            ("sign_compliant_high",   False),
            ("target_compliant_event", False),
            ("target_compliant_step",  False),
            ("passes_filter",         False),
        ]:
            if col in oracle.columns:
                oracle.loc[failed_mask, col] = default
        # Set target violations to a sentinel non-zero so target_compliant_*
        # remains False after re-derivation (some downstream tools recompute).
        for col in ("target_violations_event", "target_violations_step"):
            if col in oracle.columns:
                # max of (existing, 1) to mark "at least 1 violation"
                cur = pd.to_numeric(oracle.loc[failed_mask, col],
                                     errors="coerce").fillna(0).astype(int)
                oracle.loc[failed_mask, col] = cur.where(cur > 0, 1)

    n_compl = int((oracle["_compliant"] == 1).sum())
    n_arr_after = int(_to_bool(oracle["arrived_dest"]).sum())
    print(f"[oracle] picked {n_or} rows | compliant={n_compl} | "
          f"failed_overridden={n_failed} | arrived(after)={n_arr_after}")

    # Per-source baseline distribution of picks (BEFORE relabel)
    # Show only compliant picks — for failed scenes "the best non-compliant"
    # source label is mostly noise.
    print("[oracle] picks per rule-expert (compliant scenes only):")
    compl_picks = oracle.loc[oracle["_compliant"] == 1, "baseline"]
    if len(compl_picks):
        for base, n in compl_picks.value_counts().sort_index().items():
            print(f"    {base:<40s} {n:>6d}  ({100.0*n/len(compl_picks):5.1f}%)")
    else:
        print("    (none — no rule expert was compliant on any scene)")

    oracle = oracle.drop(columns=sort_keys)

    # Relabel as oracle_rule
    oracle["baseline"] = args.baseline_name
    if "display_policy" in oracle.columns:
        oracle["display_policy"] = args.baseline_name
    if "policy" in oracle.columns:
        oracle["policy"] = args.baseline_name

    out_df = pd.concat([df, oracle], ignore_index=True)

    dedup_keys = ["var_idx", "baseline", "scene_id", "scene_uid"]
    before = len(out_df)
    out_df = out_df.drop_duplicates(subset=dedup_keys, keep="last")
    n_dropped = before - len(out_df)
    if n_dropped:
        print(f"[oracle] final dedup dropped {n_dropped} duplicate rows "
              f"(key: {dedup_keys})")

    # Sanity per-baseline
    per_baseline_check = out_df.groupby("baseline").apply(
        lambda g: len(g) - g[["var_idx", "scene_id", "scene_uid"]].drop_duplicates().shape[0],
        include_groups=False,
    )
    bad = per_baseline_check[per_baseline_check > 0]
    if len(bad):
        print("[oracle] WARN: some baselines still have duplicate scenes:")
        for b, n in bad.items():
            print(f"    {b}: {n} duplicates")
    else:
        print(f"[oracle] sanity OK: each (var_idx, scene_id, scene_uid) "
              f"appears at most once per baseline")

    out_df.to_csv(dst, index=False)
    print(f"[oracle] wrote {len(out_df)} rows -> {dst}  "
          f"(+{n_or} {args.baseline_name})")


if __name__ == "__main__":
    main()
