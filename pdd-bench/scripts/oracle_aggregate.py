#!/usr/bin/env python3
"""Cross-policy oracle aggregator.

Reads per-policy outcomes from BOTH replay_mini_new.py output
(pdd-bench/replay_results/mini_new_*) AND expert_replay.py output
(per-sign-bench/benchmark_output/oracle_run/<sign>/*/all_runs.jsonl),
then for each (sign, scene) picks the best across:
  - IDM default + IDM-sampled variants
  - comprehensive (IDM + mixin)
  - rule_compliant (PPO + mixin)
  - carl (CaRL + mixin)
  - plant2 (PlanT2 + mixin) [if checkpoint available]

Usage:
  python3 pdd-bench/scripts/oracle_aggregate.py
  python3 pdd-bench/scripts/oracle_aggregate.py \
      --replay-results pdd-bench/replay_results \
      --oracle-runs pdd-bench/scripts/per_sign_bench/benchmark_output/oracle_run

Score: dest=+1000, violation=-10 each, ego_crash=-500, out_of_road=-200.
Higher is better.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def score_outcome(r: dict) -> float:
    return (
        int(bool(r.get("reached_dest", r.get("dest", False)))) * 1000
        - int(r.get("violations", 0)) * 10
        - int(bool(r.get("crashed", False))) * 500
        - int(bool(r.get("out_of_road", False))) * 200
    )


def _read_replay_mini_new(root: Path) -> dict:
    """Return {(sign, scene_id): {policy_label: best-of-variants outcome}}.

    For each (run_dir, scene), pick the per-policy oracle = the variant
    with the best score. This collapses N+1 variants per scene into one
    candidate per policy, so the cross-policy oracle compares apples to
    apples instead of inflating IDM by counting all its samples.
    """
    out: dict = defaultdict(dict)
    if not root.exists():
        return out
    for run_dir in sorted(root.glob("mini_new_*")):
        summary_p = run_dir / "summary.json"
        episodes_p = run_dir / "episode_results.jsonl"
        if not summary_p.exists() or not episodes_p.exists():
            continue
        try:
            s = json.loads(summary_p.read_text())
            policy = s.get("policy", run_dir.name)
            sign = s.get("sign_code", "?")
        except Exception:
            continue
        # group episodes by scene_id, keep best variant per scene
        per_scene: dict = {}
        for line in episodes_p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
            except Exception:
                continue
            scene = ep.get("scene_id", "?")
            cur = per_scene.get(scene)
            if cur is None or score_outcome(ep) > score_outcome(cur):
                per_scene[scene] = ep
        for scene, ep in per_scene.items():
            out[(sign, scene)][policy] = ep
    return out


def _read_oracle_runs(root: Path) -> dict:
    """Return {(sign, scene_id): {variant_label: outcome}} from expert_replay.py.

    expert_replay writes `all_runs.jsonl` (one row per scene+variant) under
    its --output-dir. Each row has at least `scene_uid`, `variant`, and the
    standard outcome keys.
    """
    out: dict = defaultdict(dict)
    if not root.exists():
        return out
    for jsonl in sorted(root.rglob("all_runs.jsonl")):
        # sign code is one of the path components (mini/<sign>/...) — try
        # to recover it from path; fall back to row's `sign_code`.
        sign_from_path = None
        for part in jsonl.parts:
            # match like "5_11_1" or "2_5"
            if part.replace("_", "").replace(".", "").isdigit():
                sign_from_path = part.replace("_", ".")
        for line in jsonl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            sign = row.get("sign_code") or sign_from_path or "?"
            scene = row.get("scene_uid") or row.get("scene_id") or "?"
            variant = row.get("variant", "default")
            label = f"idm_{variant}"
            out[(sign, scene)][label] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-results", type=str,
                    default="pdd-bench/replay_results")
    ap.add_argument("--oracle-runs", type=str,
                    default="pdd-bench/scripts/per_sign_bench/benchmark_output/oracle_run")
    ap.add_argument("--out", type=str, default=None,
                    help="Write JSON summary to this file as well as printing")
    args = ap.parse_args()

    wrapped = _read_replay_mini_new(Path(args.replay_results))
    idm = _read_oracle_runs(Path(args.oracle_runs))

    # Merge: for each scene, combine wrapped policies (already collapsed
    # to per-policy oracle in _read_replay_mini_new) + IDM variants.
    all_keys = set(wrapped) | set(idm)
    rows = []
    for key in sorted(all_keys):
        merged = {**wrapped.get(key, {}), **idm.get(key, {})}
        if not merged:
            continue
        scored = [(p, r, score_outcome(r)) for p, r in merged.items()]
        scored.sort(key=lambda x: -x[2])
        best_p, best_r, best_score = scored[0]
        # Variant identity if available (helps spot when IDM oracle is
        # actually a sampled run vs default).
        best_variant = best_r.get("variant", "default")
        rows.append({
            "sign": key[0],
            "scene": key[1],
            "best_policy": best_p,
            "best_variant": best_variant,
            "best_score": best_score,
            "best_dest": bool(best_r.get("reached_dest", best_r.get("dest", False))),
            "best_violations": int(best_r.get("violations", 0)),
            "best_crashed": bool(best_r.get("crashed", False)),
            "all": {
                p: {
                    "score": sc,
                    "variant": r.get("variant", "default"),
                    "dest": bool(r.get("reached_dest", r.get("dest", False))),
                    "violations": int(r.get("violations", 0)),
                    "crashed": bool(r.get("crashed", False)),
                } for p, r, sc in scored
            },
        })

    # Print table
    print(f"\n{'sign':<8} {'scene':<32} {'best_policy':<18} {'var':<8} "
          f"{'score':>6} {'dest':>5} {'viol':>4} {'crash':>5}")
    print("-" * 110)
    for r in rows:
        print(f"{r['sign']:<8} {r['scene']:<32} {r['best_policy']:<18} "
              f"{r['best_variant']:<8} {r['best_score']:>6.0f} "
              f"{str(r['best_dest']):>5} {r['best_violations']:>4} {str(r['best_crashed']):>5}")

    print(f"\nTotal scenes: {len(rows)}")

    # Pick distribution
    dist = defaultdict(int)
    for r in rows:
        dist[r["best_policy"]] += 1
    print("\nOracle pick distribution (per-policy oracle aggregated cross-policy):")
    for p, n in sorted(dist.items(), key=lambda x: -x[1]):
        share = 100 * n / max(1, len(rows))
        print(f"  {p:<22}: {n:4d}  ({share:.1f}%)")

    # Per-policy aggregates: success rate = both dest=True AND violations=0
    # AND not crashed, mean violations per scene, etc.
    per_pol_scores = defaultdict(list)
    per_pol_dest = defaultdict(list)
    per_pol_viol = defaultdict(list)
    per_pol_crash = defaultdict(list)
    per_pol_clean = defaultdict(list)  # success: dest & no viol & no crash
    for r in rows:
        for p, info in r["all"].items():
            per_pol_scores[p].append(info["score"])
            per_pol_dest[p].append(int(info["dest"]))
            per_pol_viol[p].append(int(info["violations"]))
            per_pol_crash[p].append(int(info["crashed"]))
            clean = int(info["dest"]) and (info["violations"] == 0) and not info["crashed"]
            per_pol_clean[p].append(int(bool(clean)))

    print(
        f"\n{'policy':<22} {'mean_score':>11} {'dest_rate':>10} {'viol_avg':>9} "
        f"{'crash_rate':>11} {'clean_rate':>11} {'n':>5}"
    )
    print("-" * 90)
    for p in sorted(per_pol_scores.keys()):
        n = len(per_pol_scores[p]) or 1
        ms = sum(per_pol_scores[p]) / n
        dr = sum(per_pol_dest[p]) / n
        va = sum(per_pol_viol[p]) / n
        cr = sum(per_pol_crash[p]) / n
        cl = sum(per_pol_clean[p]) / n
        print(
            f"  {p:<20} {ms:>11.1f} {dr*100:>9.1f}% {va:>9.2f} "
            f"{cr*100:>10.1f}% {cl*100:>10.1f}% {n:>5d}"
        )

    # Build summary obj for --out
    summary_obj = {
        "n_scenes": len(rows),
        "oracle_pick_distribution": dict(dist),
        "per_policy": {
            p: {
                "n": len(per_pol_scores[p]),
                "mean_score": sum(per_pol_scores[p]) / max(1, len(per_pol_scores[p])),
                "dest_rate": sum(per_pol_dest[p]) / max(1, len(per_pol_dest[p])),
                "avg_violations": sum(per_pol_viol[p]) / max(1, len(per_pol_viol[p])),
                "crash_rate": sum(per_pol_crash[p]) / max(1, len(per_pol_crash[p])),
                "clean_rate": sum(per_pol_clean[p]) / max(1, len(per_pol_clean[p])),
            }
            for p in sorted(per_pol_scores.keys())
        },
        "scenes": rows,
    }

    if args.out:
        Path(args.out).write_text(json.dumps(summary_obj, indent=2, ensure_ascii=False))
        print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
