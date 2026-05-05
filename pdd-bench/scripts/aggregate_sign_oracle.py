"""Aggregate sign-expert trajectories into oracle picks (new strategy).

Reads <run_root>/all_runs.jsonl produced by expert_replay.py
run_batch_multi_variant, applies the new oracle ranking strategy:

  HARD FILTER:  exclude episodes with crashed=True or out_of_road=True
  RANK BY:
    1. meaningful_compliance: viol=0 AND (arrived_dest OR steps>=MIN_STEPS)
    2. arrived_dest
    3. -total_violations
    4. smoothness_ratio
    5. steps  (longer drive at equal everything else)

Usage:
    python aggregate_sign_oracle.py --run-dir <run_root> --out oracle.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

MIN_ENGAGED_STEPS = 200


def is_disqualified(ep: dict) -> bool:
    return bool(ep.get("crashed")) or bool(ep.get("out_of_road"))


def meaningful_compliant(ep: dict) -> bool:
    if is_disqualified(ep):
        return False
    if int(ep.get("total_violations", 0) or 0) > 0:
        return False
    if ep.get("arrived_dest"):
        return True
    return int(ep.get("final_step", 0) or 0) >= MIN_ENGAGED_STEPS


def rank_key(ep: dict) -> tuple:
    return (
        1 if meaningful_compliant(ep) else 0,
        1 if ep.get("arrived_dest") else 0,
        -int(ep.get("total_violations", 0) or 0),
        float(ep.get("smoothness_ratio", 0.0) or 0.0),
        int(ep.get("final_step", 0) or 0),
    )


def load_runs(run_root: Path) -> list[dict]:
    path = run_root / "all_runs.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"all_runs.jsonl not found at {path}")
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def aggregate(eps: list[dict]) -> dict:
    n = len(eps)
    if n == 0:
        return {"n": 0}
    arrived = [e for e in eps if e.get("arrived_dest")]
    return {
        "n": n,
        "dest_rate": sum(1 for e in eps if e.get("arrived_dest")) / n,
        "crash_rate": sum(1 for e in eps if e.get("crashed")) / n,
        "crash_ego_fault_rate": sum(1 for e in eps if e.get("crashed_ego_fault")) / n,
        "crash_npc_fault_rate": sum(1 for e in eps if e.get("crashed_npc_fault")) / n,
        "out_of_road_rate": sum(1 for e in eps if e.get("out_of_road")) / n,
        "avg_violations": sum(int(e.get("total_violations", 0) or 0) for e in eps) / n,
        "compliance_rate": sum(1 for e in eps
                                if int(e.get("total_violations", 0) or 0) == 0) / n,
        "compliance_among_arrived": (
            sum(1 for e in arrived if int(e.get("total_violations", 0) or 0) == 0)
            / len(arrived)
        ) if arrived else None,
        "n_arrived": len(arrived),
        "meaningful_compliance_rate": sum(1 for e in eps if meaningful_compliant(e)) / n,
        "avg_smoothness_ratio": sum(float(e.get("smoothness_ratio", 0.0) or 0.0)
                                     for e in eps) / n,
        "avg_frame_smooth_ratio": sum(float(e.get("frame_smooth_ratio", 0.0) or 0.0)
                                       for e in eps) / n,
        "mean_steps": sum(int(e.get("final_step", 0) or 0) for e in eps) / n,
        "mean_total_reward": sum(float(e.get("total_reward", 0.0) or 0.0)
                                  for e in eps) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True,
                         help="Run root containing all_runs.jsonl")
    parser.add_argument("--out", type=str, default=None,
                         help="Output JSON path (default: <run-dir>/oracle_summary.json)")
    parser.add_argument("--min-engaged-steps", type=int, default=MIN_ENGAGED_STEPS,
                         help=f"Threshold for meaningful 0-viol (default: {MIN_ENGAGED_STEPS})")
    args = parser.parse_args()

    global MIN_ENGAGED_STEPS
    MIN_ENGAGED_STEPS = args.min_engaged_steps

    run_root = Path(args.run_dir)
    rows = load_runs(run_root)
    print(f"Loaded {len(rows)} episode rows from {run_root}/all_runs.jsonl")

    # Group by (sign_code, scene_id) — pick best within group from non-disqualified
    by_scene = defaultdict(list)
    for r in rows:
        key = (r.get("sign_code") or r.get("pdd_code"), r.get("scene_id"))
        by_scene[key].append(r)

    oracle_picks = []
    skipped_scenes = []
    for (sign, scene), eps in sorted(by_scene.items()):
        valid = [e for e in eps if not is_disqualified(e)]
        if not valid:
            skipped_scenes.append({"sign": sign, "scene": scene,
                                     "n_candidates": len(eps)})
            continue
        best = max(valid, key=rank_key)
        oracle_picks.append({
            "sign": sign,
            "scene": scene,
            "best_policy": best.get("policy"),
            "best_variant": best.get("variant"),
            "best_seed": best.get("seed"),
            "scene_uid": best.get("scene_uid"),
            "arrived_dest": best.get("arrived_dest"),
            "violations": int(best.get("total_violations", 0) or 0),
            "smoothness_ratio": float(best.get("smoothness_ratio", 0.0) or 0.0),
            "frame_smooth_ratio": float(best.get("frame_smooth_ratio", 0.0) or 0.0),
            "final_step": int(best.get("final_step", 0) or 0),
            "pkl_path": best.get("pkl_path"),
            "sidecar_path": best.get("sidecar_path"),
            "meaningful_compliant": meaningful_compliant(best),
        })

    # Aggregate metrics
    print("\n=== Per-policy metrics (across all episodes) ===")
    by_policy = defaultdict(list)
    for r in rows:
        by_policy[r.get("policy")].append(r)
    per_policy = {pol: aggregate(eps) for pol, eps in by_policy.items()}
    for pol in sorted(per_policy):
        m = per_policy[pol]
        print(f"  {pol:<16} n={m['n']:>3}  dest={m['dest_rate']:.2f}  "
              f"crash={m['crash_rate']:.2f}(ego={m['crash_ego_fault_rate']:.2f}/"
              f"npc={m['crash_npc_fault_rate']:.2f})  "
              f"oor={m['out_of_road_rate']:.2f}  viol={m['avg_violations']:.2f}  "
              f"smooth={m['avg_smoothness_ratio']:.2f}")

    # Oracle aggregate
    oracle_eps = []
    for pick in oracle_picks:
        # Find original ep for full metrics
        for r in rows:
            if (r.get("sign_code") == pick["sign"]
                    and r.get("scene_id") == pick["scene"]
                    and r.get("policy") == pick["best_policy"]
                    and r.get("variant") == pick["best_variant"]):
                oracle_eps.append(r)
                break

    oracle_metrics = aggregate(oracle_eps)
    pick_dist = Counter((p["best_policy"], p["best_variant"])
                         for p in oracle_picks)

    print(f"\n=== Oracle ({len(oracle_picks)} scenes; "
          f"{len(skipped_scenes)} skipped) ===")
    print(f"  dest_rate:                    {oracle_metrics.get('dest_rate', 0):.2f}")
    print(f"  crash_rate:                   {oracle_metrics.get('crash_rate', 0):.2f} (forced 0)")
    print(f"  out_of_road_rate:             {oracle_metrics.get('out_of_road_rate', 0):.2f} (forced 0)")
    print(f"  avg_violations:               {oracle_metrics.get('avg_violations', 0):.2f}")
    print(f"  compliance_rate:              {oracle_metrics.get('compliance_rate', 0):.2f}")
    print(f"  compliance_among_arrived:     "
          f"{oracle_metrics.get('compliance_among_arrived') or 0:.2f}"
          f" (n={oracle_metrics.get('n_arrived', 0)})")
    print(f"  meaningful_compliance_rate:   {oracle_metrics.get('meaningful_compliance_rate', 0):.2f}")
    print(f"  avg_smoothness_ratio:         {oracle_metrics.get('avg_smoothness_ratio', 0):.2f}")
    print(f"  mean_steps:                   {oracle_metrics.get('mean_steps', 0):.0f}")

    print(f"\n=== Oracle pick distribution ===")
    for (pol, var), cnt in pick_dist.most_common():
        print(f"  {pol}/{var:<10} {cnt}")

    # Per-sign breakdown
    by_sign_oracle = defaultdict(list)
    for ep in oracle_eps:
        by_sign_oracle[ep.get("sign_code") or ep.get("pdd_code")].append(ep)
    per_sign = {sign: aggregate(eps) for sign, eps in by_sign_oracle.items()}

    summary = {
        "min_engaged_steps": args.min_engaged_steps,
        "n_total_episodes": len(rows),
        "n_scenes": len(by_scene),
        "n_oracle_picks": len(oracle_picks),
        "n_skipped_scenes": len(skipped_scenes),
        "skipped_scenes": skipped_scenes,
        "oracle_metrics": oracle_metrics,
        "per_policy_metrics": per_policy,
        "per_sign_oracle_metrics": per_sign,
        "pick_distribution": {f"{pol}/{var}": cnt
                                for (pol, var), cnt in pick_dist.items()},
        "scenes": oracle_picks,
    }

    out_path = Path(args.out) if args.out else (run_root / "oracle_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
