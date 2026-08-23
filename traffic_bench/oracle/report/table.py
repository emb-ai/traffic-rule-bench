#!/usr/bin/env python3
"""Build policy-vs-oracle metric tables for oracle expert selection.

The table uses the same target-sign compliance and recomputed destination
rules as select.filter, so the reported pass/oracle numbers match the
selection logic.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from traffic_bench.oracle.select.filter import (
    BETA_DEFAULT,
    HORIZON_DEFAULT,
    SIGN_CLASS_MAP,
    comfort,
    f1_score,
    is_compliant,
    normalize_sign,
    passes_filter,
    recompute_dest,
)


# Every row except "n" and "compliance among arrived" shares ONE denominator:
# the number of unique scenes of the sign (the scene universe observed in the
# source rows). A missing or filter-failing episode counts as a failure for
# its policy on that scene; for avg_* rows such a scene contributes 0 to the
# numerator. This makes every column (policies and ORACLE) a per-scene
# expectation over the SAME task set — directly comparable, with no
# survivor-selection bias in the avg rows. "compliance among arrived" is the
# only intentionally conditional row (its own n is shown in parentheses).
METRICS = [
    "n (episodes)",
    "n_scenes (common denominator)",
    "dest_rate (of scenes)",
    "crash_rate (of scenes)",
    "out_of_road_rate (of scenes)",
    "avg_violations (target, per scene)",
    "compliance_rate (of scenes)",
    "compliance among arrived",
    "avg_comfort (per scene, 0 if missing)",
    "avg_time_eff (per scene, 0 if not passed)",
    "F1 beta={beta} (per scene, 0 if not passed)",
    "pass_rate (of scenes)",
    "oracle picks (of {n})",
]

POLICY_ALIASES = {
    # RuleCompliantExpertPolicy is the PPO+signs expert in replay_mini_new.py.
    "rule_compliant": "ppo_signs",
    "carl": "carl_signs",
    "plant2": "plant2_signs",
}


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"[error] not found: {p}", file=sys.stderr)
        sys.exit(1)
    rows = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            print(f"[warn] bad json in {p}", file=sys.stderr)
    return rows


def row_sign(row: dict) -> str | None:
    sign = row.get("sign_code") or row.get("sign_slug") or row.get("sign")
    return normalize_sign(sign) if isinstance(sign, str) else None


def scene_key(row: dict) -> str | None:
    """Sign coverage key: one physical scene may appear under several signs.

    We keep sign outside of this key and always aggregate by (sign, scene_key).
    scene_uid is preferred because it preserves lane/seed/var_idx variants;
    scene_id is only a fallback for old files.
    """
    key = row.get("scene_uid") or row.get("scene_key") or row.get("scene_id")
    return str(key) if key is not None else None


def policy_label(policy: str | None, variant: str | None) -> str:
    policy = policy or "unknown"
    variant = variant or "default"
    # The IDM-family expert is reported as "idm_rule" in the tables.
    if policy in {"comprehensive", "comprehensive_rule_expert"}:
        return f"idm_rule_{variant}"
    if policy in {"ppo_signs", "carl_signs", "plant2_signs"}:
        return policy
    if policy in POLICY_ALIASES:
        return POLICY_ALIASES[policy]
    if variant and variant != "default":
        return f"{policy}_{variant}"
    return policy


def row_policy_label(row: dict) -> str:
    return policy_label(row.get("policy"), row.get("variant"))


def pick_policy_label(pick: dict) -> str:
    return policy_label(pick.get("winner_policy"), pick.get("winner_variant"))


def column_sort_key(label: str) -> tuple:
    if label == "idm_rule_default":
        return (0, 0, "")
    if label.startswith("idm_rule_s"):
        suffix = label[10:]
        if suffix.isdigit():
            return (0, int(suffix), "")
        return (0, 1000, suffix)
    fixed = {
        "ppo_signs": 1,
        "carl_signs": 2,
        "plant2_signs": 3,
        "ORACLE_new": 99,
    }
    return (fixed.get(label, 50), 0, label)


def fmt_rate(v: float | None) -> str:
    return "" if v is None else f"{v:.2f}"


def fmt_avg(v: float | None) -> str:
    return "" if v is None else f"{v:.3f}"


def fmt_count(v: int | None) -> str:
    return "" if v is None else str(v)


def fmt_compliance_among_arrived(rate: float | None, n: int) -> str:
    if rate is None:
        return ""
    return f"{rate:.2f} ({n})"


def build_records(rows: list[dict], signs: list[str] | None,
                  horizon: int, beta: float) -> tuple[list[dict], Counter]:
    sign_set = {normalize_sign(s) for s in signs} if signs else {
        row_sign(r) for r in rows if row_sign(r)
    }
    records = []
    skipped_unknown_class: Counter = Counter()

    for row in rows:
        if not row.get("valid"):
            continue
        sign = row_sign(row)
        if sign not in sign_set:
            continue
        target_class = SIGN_CLASS_MAP.get(sign)
        if target_class is None:
            skipped_unknown_class[sign] += 1
            continue
        skey = scene_key(row)
        if not skey:
            continue

        vbc = row.get("violations_by_class_event")
        if vbc is None:  # legacy rows: event counts under violations_by_class
            vbc = row.get("violations_by_class") or {}
        target_viol = int(vbc.get(target_class, 0) or 0)
        # min_final_step=0: anti-bug filter is for expert_selection, NOT metric compute.
        passed = passes_filter(row, sign, target_class, horizon, min_final_step=0)
        records.append({
            "row": row,
            "sign": sign,
            "scene_key": skey,
            "label": row_policy_label(row),
            "target_class": target_class,
            "target_violations": target_viol,
            "compliant": is_compliant(row, target_class),
            "dest_new": recompute_dest(row, sign, target_class, horizon),
            "passing": passed,
            "final_step": max(1, int(row.get("final_step") or 1)),
            "comfort": comfort(row),
            "time_eff": None,
            "f1": None,
        })

    min_step_by_scene: dict[tuple[str, str], int] = {}
    for rec in records:
        if not rec["passing"]:
            continue
        key = (rec["sign"], rec["scene_key"])
        min_step_by_scene[key] = min(
            rec["final_step"],
            min_step_by_scene.get(key, rec["final_step"]),
        )

    for rec in records:
        if not rec["passing"]:
            continue
        min_step = min_step_by_scene[(rec["sign"], rec["scene_key"])]
        t = min_step / rec["final_step"]
        rec["time_eff"] = t
        rec["f1"] = f1_score(t, rec["comfort"], beta)

    return records, skipped_unknown_class


_EMPTY_STATS = {
    "n": 0,
    "n_scenes": 0,
    "dest_rate": None,
    "crash_rate": None,
    "out_of_road_rate": None,
    "avg_violations": None,
    "compliance_rate": None,
    "compliance_arrived_rate": None,
    "arrived_n": 0,
    "avg_comfort": None,
    "avg_time_eff": None,
    "avg_f1": None,
    "pass_rate": None,
}


def aggregate_records(records: list[dict], n_scenes: int) -> dict:
    """Column stats for one policy_variant.

    All *_rate values share the common denominator n_scenes (the sign's scene
    universe): a scene where this policy has no valid episode, or fails the
    condition, counts against the rate. avg_* values are means over the
    episodes where the quantity is defined.
    """
    n = len(records)
    if n == 0 or n_scenes == 0:
        return dict(_EMPTY_STATS, n_scenes=n_scenes)

    def scene_count(pred) -> int:
        return len({(r["sign"], r["scene_key"]) for r in records if pred(r)})

    passing = [r for r in records if r["passing"]]
    arrived = [r for r in records if r["dest_new"]]
    passing_time = [r["time_eff"] for r in passing if r["time_eff"] is not None]
    passing_f1 = [r["f1"] for r in passing if r["f1"] is not None]

    return {
        "n": n,
        "n_scenes": n_scenes,
        "dest_rate": scene_count(lambda r: r["dest_new"]) / n_scenes,
        "crash_rate": scene_count(lambda r: r["row"].get("crashed")) / n_scenes,
        "out_of_road_rate": (
            scene_count(lambda r: r["row"].get("out_of_road")) / n_scenes
        ),
        "avg_violations": sum(r["target_violations"] for r in records) / n_scenes,
        "compliance_rate": scene_count(lambda r: r["compliant"]) / n_scenes,
        "compliance_arrived_rate": (
            sum(1 for r in arrived if r["compliant"]) / len(arrived)
            if arrived else None
        ),
        "arrived_n": len(arrived),
        "avg_comfort": sum(r["comfort"] for r in records) / n_scenes,
        "avg_time_eff": sum(passing_time) / n_scenes,
        "avg_f1": sum(passing_f1) / n_scenes,
        "pass_rate": scene_count(lambda r: r["passing"]) / n_scenes,
    }


def aggregate_picks(picks: list[dict], n_scenes: int) -> dict:
    """ORACLE_new column stats over the SAME scene denominator.

    dest/compliance/pass rates = covered scenes / n_scenes (scene coverage);
    an uncovered scene counts as an oracle failure, mirroring the policy
    columns. avg_* values are per-scene: the best (rank-1) pick represents its
    scene (top-2 lists hold two picks per scene) and an uncovered scene
    contributes 0 — the same convention as the policy columns.
    """
    n = len(picks)
    if n == 0 or n_scenes == 0:
        return dict(_EMPTY_STATS, n_scenes=n_scenes)
    best_by_scene: dict[tuple, dict] = {}
    for p in picks:
        key = (normalize_sign(p.get("sign")), scene_key(p))
        cur = best_by_scene.get(key)
        if cur is None or (float(p.get("f1_score") or 0.0)
                           > float(cur.get("f1_score") or 0.0)):
            best_by_scene[key] = p
    best = list(best_by_scene.values())
    covered = len(best) / n_scenes
    return {
        "n": n,
        "n_scenes": n_scenes,
        "dest_rate": covered,
        "crash_rate": 0.0,
        "out_of_road_rate": 0.0,
        "avg_violations": 0.0,
        "compliance_rate": covered,
        "compliance_arrived_rate": 1.0,
        "arrived_n": n,
        "avg_comfort": sum(float(p.get("comfort") or 0.0) for p in best) / n_scenes,
        "avg_time_eff": sum(float(p.get("time_eff") or 0.0) for p in best) / n_scenes,
        "avg_f1": sum(float(p.get("f1_score") or 0.0) for p in best) / n_scenes,
        "pass_rate": covered,
    }


def values_for_stats(stats: dict, oracle_total: int,
                     beta: float) -> dict[str, str]:
    metric_names = [m.format(beta=beta, n=oracle_total) for m in METRICS]
    return {
        metric_names[0]: fmt_count(stats["n"]),
        metric_names[1]: fmt_count(stats["n_scenes"]),
        metric_names[2]: fmt_rate(stats["dest_rate"]),
        metric_names[3]: fmt_rate(stats["crash_rate"]),
        metric_names[4]: fmt_rate(stats["out_of_road_rate"]),
        metric_names[5]: fmt_rate(stats["avg_violations"]),
        metric_names[6]: fmt_rate(stats["compliance_rate"]),
        metric_names[7]: fmt_compliance_among_arrived(
            stats["compliance_arrived_rate"],
            stats["arrived_n"],
        ),
        metric_names[8]: fmt_avg(stats["avg_comfort"]),
        metric_names[9]: fmt_avg(stats["avg_time_eff"]),
        metric_names[10]: fmt_avg(stats["avg_f1"]),
        metric_names[11]: fmt_rate(stats["pass_rate"]),
        metric_names[12]: "",
    }


def build_table(records: list[dict], picks: list[dict],
                columns: list[str], beta: float) -> list[dict[str, str]]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_label[rec["label"]].append(rec)

    # The single rate denominator: unique scenes observed in the source rows.
    n_scenes = len({(rec["sign"], rec["scene_key"]) for rec in records})

    pick_counts = Counter(pick_policy_label(p) for p in picks)
    oracle_stats = aggregate_picks(picks, n_scenes)
    oracle_total = len(picks)
    metric_names = [m.format(beta=beta, n=oracle_total) for m in METRICS]

    values_by_column: dict[str, dict[str, str]] = {}
    for col in columns:
        if col == "ORACLE_new":
            values_by_column[col] = values_for_stats(
                oracle_stats, oracle_total, beta,
            )
            values_by_column[col][metric_names[-1]] = str(oracle_total)
        else:
            stats = aggregate_records(by_label.get(col, []), n_scenes)
            values_by_column[col] = values_for_stats(
                stats, oracle_total, beta,
            )
            values_by_column[col][metric_names[-1]] = str(pick_counts.get(col, 0))

    table = []
    for metric in metric_names:
        row = {"metric": metric}
        for col in columns:
            row[col] = values_by_column[col].get(metric, "")
        table.append(row)
    return table


def write_tsv(path: Path, columns: list[str],
              rows: list[dict[str, str]], first_columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = first_columns + columns
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(k, "")) for k in header) + "\n")


def markdown_table(title: str, columns: list[str],
                   rows: list[dict[str, str]],
                   first_columns: list[str]) -> str:
    header = first_columns + columns
    lines = [f"## {title}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in header) + " |")
    lines.append("")
    return "\n".join(lines)


def build_coverage_rows(records: list[dict], picks: list[dict],
                        signs: list[str]) -> list[dict[str, str]]:
    by_sign_scenes: dict[str, set[str]] = defaultdict(set)
    passing_scenes: dict[str, set[str]] = defaultdict(set)
    episodes: Counter = Counter()
    for rec in records:
        sign = rec["sign"]
        by_sign_scenes[sign].add(rec["scene_key"])
        episodes[sign] += 1
        if rec["passing"]:
            passing_scenes[sign].add(rec["scene_key"])

    picked_scenes: dict[str, set[str]] = defaultdict(set)
    for pick in picks:
        sign = normalize_sign(pick.get("sign"))
        skey = scene_key(pick)
        if sign and skey:
            picked_scenes[sign].add(skey)

    rows = []
    for sign in signs:
        total = len(by_sign_scenes[sign])
        picked = len(picked_scenes[sign])
        rows.append({
            "sign": sign,
            "episodes": str(episodes[sign]),
            "sign_scenes": str(total),
            "passing_sign_scenes": str(len(passing_scenes[sign])),
            "oracle_picked_scenes": str(picked),
            "oracle_coverage_rate": f"{(picked / total):.2f}" if total else "",
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--jsonl", action="append", default=[],
                   help="source all_runs.jsonl; can be repeated")
    p.add_argument("--picks", required=True,
                   help="experts_selected.jsonl from select.filter")
    p.add_argument("--signs", nargs="*", default=None,
                   help="signs to include; default: infer from source rows")
    p.add_argument("--beta", type=float, default=BETA_DEFAULT)
    p.add_argument("--horizon", type=int, default=HORIZON_DEFAULT)
    p.add_argument("--output-dir", default=".",
                   help="where to write oracle_metrics_*.tsv/md")
    p.add_argument("--universe", choices=["all", "covered"], default="all",
                   help="scene universe for the rate denominator: 'all' = "
                        "every scene seen in the source rows; 'covered' = "
                        "only scenes that received an oracle pick, i.e. the "
                        "unified solvable set (the oracle is 1.00 there by "
                        "construction; policy rates show their true share "
                        "of solvable scenes)")
    args = p.parse_args()

    if not args.jsonl:
        print("ERROR: provide at least one --jsonl", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    for path in args.jsonl:
        loaded = load_jsonl(path)
        print(f"  {path}: {len(loaded)} source rows")
        rows.extend(loaded)
    picks = load_jsonl(args.picks)
    print(f"  {args.picks}: {len(picks)} oracle picks")

    signs = [normalize_sign(s) for s in args.signs] if args.signs else sorted({
        row_sign(r) for r in rows if row_sign(r)
    })
    records, skipped_unknown = build_records(rows, signs, args.horizon, args.beta)
    if skipped_unknown:
        for sign, n in sorted(skipped_unknown.items()):
            print(f"[warn] skipped {n} rows for unknown sign class: {sign}",
                  file=sys.stderr)

    if args.universe == "covered":
        covered_keys = {(normalize_sign(p_.get("sign")), scene_key(p_))
                        for p_ in picks}
        n_full = len({(r["sign"], r["scene_key"]) for r in records})
        records = [r for r in records
                   if (r["sign"], r["scene_key"]) in covered_keys]
        n_kept = len({(r["sign"], r["scene_key"]) for r in records})
        print(f"universe=covered: kept {n_kept} of {n_full} scenes "
              f"({n_full - n_kept} uncovered dropped)")

    observed_columns = {r["label"] for r in records}
    observed_columns.update(pick_policy_label(p) for p in picks)
    columns = sorted(observed_columns, key=column_sort_key)
    columns.append("ORACLE_new")

    out_dir = Path(args.output_dir)
    summary_rows = build_table(records, picks, columns, args.beta)
    write_tsv(out_dir / "oracle_metrics_summary.tsv", columns, summary_rows,
              ["metric"])

    by_sign_rows = []
    md_parts = [
        markdown_table("all signs", columns, summary_rows, ["metric"])
    ]
    for sign in signs:
        sign_records = [r for r in records if r["sign"] == sign]
        sign_picks = [p for p in picks if normalize_sign(p.get("sign")) == sign]
        table = build_table(sign_records, sign_picks, columns, args.beta)
        for row in table:
            by_sign_rows.append({"sign": sign, **row})
        md_parts.append(markdown_table(sign, columns, table, ["metric"]))

    write_tsv(out_dir / "oracle_metrics_by_sign.tsv", columns, by_sign_rows,
              ["sign", "metric"])

    coverage_rows = build_coverage_rows(records, picks, signs)
    write_tsv(
        out_dir / "sign_coverage.tsv",
        ["episodes", "sign_scenes", "passing_sign_scenes",
         "oracle_picked_scenes", "oracle_coverage_rate"],
        coverage_rows,
        ["sign"],
    )

    (out_dir / "oracle_metrics_summary.md").write_text("\n".join(md_parts))

    print("\nWrote metric tables:")
    print(f"  {out_dir / 'oracle_metrics_summary.tsv'}")
    print(f"  {out_dir / 'oracle_metrics_by_sign.tsv'}")
    print(f"  {out_dir / 'sign_coverage.tsv'}")
    print(f"  {out_dir / 'oracle_metrics_summary.md'}")


if __name__ == "__main__":
    main()
