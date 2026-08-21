#!/usr/bin/env python3
"""Select oracle experts only on scenes covered by all 8 expert agents.

This is a wrapper around select_experts.py: it first keeps only sign-scenes
where every required expert has a recorded run, then applies the same
per-scene oracle selection logic.

Default required agents:
  comp_default, comp_s1, comp_s2, comp_s3, comp_s4,
  ppo_signs, carl_signs, plant2_signs

Here ppo_signs is the stored policy="rule_compliant" expert.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from traffic_bench.oracle.select_experts import (
    BETA_DEFAULT,
    HORIZON_DEFAULT,
    SIGN_CLASS_MAP,
    load_runs,
    normalize_sign,
    print_summary,
    select_expert_per_scene,
)


DEFAULT_REQUIRED_AGENTS = (
    "comp_default",
    "comp_s1",
    "comp_s2",
    "comp_s3",
    "comp_s4",
    "ppo_signs",
    "carl_signs",
    "plant2_signs",
)


def row_sign(row: dict) -> str | None:
    sign = row.get("sign_code") or row.get("sign_slug") or row.get("sign")
    return normalize_sign(sign) if isinstance(sign, str) else None


def scene_key(row: dict) -> str | None:
    key = row.get("scene_uid") or row.get("scene_key") or row.get("scene_id")
    return str(key) if key is not None else None


def normalize_agent_label(label: str) -> str:
    label = label.strip()
    aliases = {
        "comprehensive_default": "comp_default",
        "rule_compliant": "ppo_signs",
        "rule_compliant_default": "ppo_signs",
        "ppo": "ppo_signs",
        "ppo_signs_default": "ppo_signs",
        "carl": "carl_signs",
        "carl_default": "carl_signs",
        "carl_signs_default": "carl_signs",
        "plant2": "plant2_signs",
        "plant2_default": "plant2_signs",
        "plant2_signs_default": "plant2_signs",
    }
    if label in aliases:
        return aliases[label]
    if label.startswith("comprehensive_s"):
        return "comp_" + label.rsplit("_", 1)[-1]
    return label


def row_agent_label(row: dict) -> str:
    policy = row.get("policy") or "unknown"
    variant = row.get("variant") or "default"
    if policy == "comprehensive":
        return normalize_agent_label(f"comprehensive_{variant}")
    return normalize_agent_label(f"{policy}_{variant}")


def infer_signs(rows: list[dict]) -> list[str]:
    signs = sorted({
        s for s in (row_sign(row) for row in rows)
        if s in SIGN_CLASS_MAP
    })
    return signs


def build_groups(rows: list[dict], signs: list[str],
                 count_invalid_runs: bool) -> dict[tuple[str, str], list[dict]]:
    sign_set = {normalize_sign(s) for s in signs}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if not count_invalid_runs and not row.get("valid"):
            continue
        sign = row_sign(row)
        if sign not in sign_set or sign not in SIGN_CLASS_MAP:
            continue
        skey = scene_key(row)
        if not skey:
            continue
        groups[(sign, skey)].append(row)
    return groups


def filter_complete_scene_rows(
    rows: list[dict],
    signs: list[str],
    required_agents: set[str],
    count_invalid_runs: bool,
) -> tuple[list[dict], dict, list[dict]]:
    """Return rows whose (sign, scene_key) has all required agent labels."""
    groups = build_groups(rows, signs, count_invalid_runs)
    complete_keys = set()
    diagnostics = []
    missing_counter: Counter[str] = Counter()

    for (sign, skey), items in groups.items():
        present = {row_agent_label(row) for row in items}
        missing = sorted(required_agents - present)
        if not missing:
            complete_keys.add((sign, skey))
        else:
            for label in missing:
                missing_counter[label] += 1
        diagnostics.append({
            "sign": sign,
            "scene_key": skey,
            "present": sorted(present),
            "missing": missing,
            "complete": not missing,
        })

    filtered = []
    for row in rows:
        sign = row_sign(row)
        skey = scene_key(row)
        if (sign, skey) in complete_keys:
            filtered.append(row)

    stats = {
        "groups_total": len(groups),
        "groups_complete": len(complete_keys),
        "rows_before": len(rows),
        "rows_after": len(filtered),
        "missing_counter": missing_counter,
        "complete_keys": complete_keys,
    }
    return filtered, stats, diagnostics


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def write_scene_reports(output_path: Path, diagnostics: list[dict]) -> None:
    complete_path = output_path.with_suffix(output_path.suffix + ".complete_scenes.tsv")
    missing_path = output_path.with_suffix(output_path.suffix + ".missing_scenes.tsv")

    with open(complete_path, "w") as f:
        f.write("sign\tscene_key\tpresent_agents\n")
        for d in sorted(diagnostics, key=lambda x: (x["sign"], x["scene_key"])):
            if not d["complete"]:
                continue
            f.write(
                f"{d['sign']}\t{d['scene_key']}\t{','.join(d['present'])}\n"
            )

    with open(missing_path, "w") as f:
        f.write("sign\tscene_key\tpresent_agents\tmissing_agents\n")
        for d in sorted(diagnostics, key=lambda x: (x["sign"], x["scene_key"])):
            if d["complete"]:
                continue
            f.write(
                f"{d['sign']}\t{d['scene_key']}\t"
                f"{','.join(d['present'])}\t{','.join(d['missing'])}\n"
            )

    print("\nCoverage reports:")
    print(f"  complete scenes: {complete_path}")
    print(f"  missing scenes:  {missing_path}")


def print_complete_summary(diagnostics: list[dict], stats: dict) -> None:
    by_sign = defaultdict(lambda: {"total": 0, "complete": 0})
    for d in diagnostics:
        by_sign[d["sign"]]["total"] += 1
        by_sign[d["sign"]]["complete"] += int(d["complete"])

    print("\n" + "=" * 70)
    print("Complete-scene coverage before oracle selection")
    print("=" * 70)
    print(f"{'sign':<10} {'sign_scenes':>11} {'complete8':>10} {'rate':>7}")
    print("-" * 44)
    for sign in sorted(by_sign):
        total = by_sign[sign]["total"]
        complete = by_sign[sign]["complete"]
        rate = complete / total if total else 0.0
        print(f"{sign:<10} {total:>11} {complete:>10} {rate:>6.1%}")

    print("\nMissing required agents across incomplete sign-scenes:")
    if stats["missing_counter"]:
        for label, n in sorted(stats["missing_counter"].items(),
                               key=lambda kv: (-kv[1], kv[0])):
            print(f"  {label:<14} {n}")
    else:
        print("  none")


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--run-dir", action="append", default=[],
                   help="directory with all_runs.jsonl (may be repeated)")
    p.add_argument("--jsonl", action="append", default=[],
                   help="direct path to an all_runs.jsonl (may be repeated)")
    p.add_argument("--signs", nargs="*", default=None,
                   help="signs to select; default: inferred from the data")
    p.add_argument("--required-agents", nargs="+",
                   default=list(DEFAULT_REQUIRED_AGENTS),
                   help="required agent labels for a complete scene")
    p.add_argument("--count-invalid-runs", action="store_true",
                   help="count invalid rows as a successful pass for coverage")
    p.add_argument("--beta", type=float, default=BETA_DEFAULT)
    p.add_argument("--horizon", type=int, default=HORIZON_DEFAULT)
    p.add_argument("--max-per-sign", type=int, default=None,
                   help="keep only the top-N scenes per sign by f1_score")
    p.add_argument("--output", default="experts_selected_complete8.jsonl")
    args = p.parse_args()

    if not (args.run_dir or args.jsonl):
        print("ERROR: specify --run-dir or --jsonl", file=sys.stderr)
        sys.exit(1)

    print("Inputs:")
    rows = load_runs(args.run_dir, args.jsonl)
    signs = ([normalize_sign(s) for s in args.signs]
             if args.signs is not None else infer_signs(rows))
    required_agents = {
        normalize_agent_label(label) for label in args.required_agents
    }

    print(f"\nTotal rows:        {len(rows)}")
    print(f"Signs:             {signs}")
    print(f"Required agents:   {sorted(required_agents)}")
    print(f"Coverage rows:     {'all rows' if args.count_invalid_runs else 'valid rows only'}")
    print(f"Beta:              {args.beta}")
    print(f"Horizon:           {args.horizon}")

    filtered_rows, stats, diagnostics = filter_complete_scene_rows(
        rows,
        signs,
        required_agents,
        count_invalid_runs=args.count_invalid_runs,
    )

    print("\nComplete-scene filter:")
    print(f"  sign-scenes total:     {stats['groups_total']}")
    print(f"  sign-scenes complete:  {stats['groups_complete']}")
    print(f"  rows before:           {stats['rows_before']}")
    print(f"  rows after:            {stats['rows_after']}")
    print_complete_summary(diagnostics, stats)

    picks, scene_groups, filter_records = select_expert_per_scene(
        filtered_rows,
        signs,
        beta=args.beta,
        horizon=args.horizon,
        max_per_sign=args.max_per_sign,
    )

    out_path = Path(args.output)
    write_jsonl(out_path, picks)
    print(f"\nWrote {len(picks)} complete-scene oracle picks -> {out_path}")
    write_scene_reports(out_path, diagnostics)

    print_summary(picks, scene_groups, filter_records,
                  signs, args.beta, args.horizon)


if __name__ == "__main__":
    main()
