"""Analyze metrics from all_runs.jsonl produced by expert_replay.py.

Prints multiple tables:
  1. Per-policy aggregate (dest_rate, crash, oor, viol, smoothness, ...)
  2. Per-sign × per-policy dest_rate / crash_rate / viol
  3. idm_signs (comprehensive) variant breakdown — default vs sampled
  4. Crash attribution split (ego_fault vs npc_fault)
  5. Oracle picks under new strategy (compliance > dest with hard filter)

Usage:
    python analyze_all_runs.py /Users/victoria_s/sdc_new_signs/all_runs.jsonl
    python analyze_all_runs.py /path/to/all_runs.jsonl --policies comprehensive,carl
    python analyze_all_runs.py /path/to/all_runs.jsonl --out summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


MIN_ENGAGED_STEPS = 200
HORIZON = 600   # used to compute time_efficiency = 1 - final_step / HORIZON


def load_runs(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def normalize_sign(s: str) -> str:
    return s.replace("_", ".") if s else s


def safe_get(r: dict, *keys, default=None):
    for k in keys:
        v = r.get(k)
        if v is not None:
            return v
    return default


def policy_label(r: dict) -> str:
    p = r.get("policy", "?")
    v = r.get("variant", "?")
    if p == "comprehensive":
        return f"{p}/{v}"
    return p


def is_disqualified(r: dict) -> bool:
    return bool(r.get("crashed")) or bool(r.get("out_of_road"))


def meaningful_compliant(r: dict) -> bool:
    if is_disqualified(r):
        return False
    if int(r.get("total_violations", 0) or 0) > 0:
        return False
    return r.get("arrived_dest") or int(r.get("final_step", 0) or 0) >= MIN_ENGAGED_STEPS


def oracle_rank(r: dict) -> tuple:
    return (
        1 if meaningful_compliant(r) else 0,
        1 if r.get("arrived_dest") else 0,
        -int(r.get("total_violations", 0) or 0),
        float(r.get("smoothness_ratio", 0.0) or 0.0),
        int(r.get("final_step", 0) or 0),
    )


def time_efficiency(r: dict, horizon: int = None) -> float:
    """1 - final_step / horizon. Higher = finished faster (less time used)."""
    H = horizon if horizon is not None else HORIZON
    if H <= 0:
        return 0.0
    steps = int(r.get("final_step", 0) or 0)
    return max(0.0, 1.0 - steps / H)


def harmonic_mean(*vals: float) -> float:
    """Harmonic mean of non-negative values. Returns 0.0 if any value <= 0
    (a single zero pulls HM to 0 — useful for combined-metric semantics:
    'good only if both components are good')."""
    vals = [float(v) for v in vals]
    if not vals or any(v <= 0 for v in vals):
        return 0.0
    n = len(vals)
    b = 0.25
    return (b*b + 1) * vals[0] * vals[1] / ((b*b) * vals[0] + vals[1]) 


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    valid = [r for r in rows if r.get("valid")]
    arrived = [r for r in rows if r.get("arrived_dest")]

    # Per-episode time_efficiency and comfort×time hmean.
    time_eff_per_row = [time_efficiency(r) for r in rows]
    comfort_per_row = [float(r.get("smoothness_ratio", 0.0) or 0.0) for r in rows]
    hmean_per_row = [harmonic_mean(t, c) for t, c in zip(time_eff_per_row, comfort_per_row)]

    return {
        "n": n,
        "n_valid": len(valid),
        "dest_rate": sum(1 for r in rows if r.get("arrived_dest")) / n,
        "crash_rate": sum(1 for r in rows if r.get("crashed")) / n,
        "ego_fault_rate": sum(1 for r in rows if r.get("crashed_ego_fault")) / n,
        "npc_fault_rate": sum(1 for r in rows if r.get("crashed_npc_fault")) / n,
        "oor_rate": sum(1 for r in rows if r.get("out_of_road")) / n,
        "avg_viol": mean(int(r.get("total_violations", 0) or 0) for r in rows),
        "viol0_rate": sum(1 for r in rows
                          if int(r.get("total_violations", 0) or 0) == 0) / n,
        "viol0_among_arrived": (
            sum(1 for r in arrived
                if int(r.get("total_violations", 0) or 0) == 0) / len(arrived)
        ) if arrived else None,
        "n_arrived": len(arrived),
        "meaningful_compl": sum(1 for r in rows if meaningful_compliant(r)) / n,
        "avg_smoothness": mean(float(r.get("smoothness_ratio", 0.0) or 0.0) for r in rows),
        "avg_frame_smooth": mean(float(r.get("frame_smooth_ratio", 0.0) or 0.0) for r in rows),
        "avg_steps": mean(int(r.get("final_step", 0) or 0) for r in rows),
        "avg_reward": mean(float(r.get("total_reward", 0.0) or 0.0) for r in rows),

        # New: 1 - steps/horizon (per-episode mean) and harmonic mean with smoothness.
        "avg_time_eff": mean(time_eff_per_row),
        "avg_comfort_time_hmean": mean(hmean_per_row),
        # Same metrics computed over arrived-only (non-arrived gives misleading
        # high time_eff for early crashes — restricting to arrived is honest).
        "time_eff_arrived": (
            mean([time_efficiency(r) for r in arrived]) if arrived else None
        ),
        "comfort_time_hmean_arrived": (
            mean([
                harmonic_mean(time_efficiency(r),
                              float(r.get("smoothness_ratio", 0.0) or 0.0))
                for r in arrived
            ]) if arrived else None
        ),
    }


def print_per_policy(rows: list[dict]) -> dict:
    by_policy = defaultdict(list)
    for r in rows:
        by_policy[r.get("policy", "?")].append(r)

    print("=" * 160)
    print("1. Per-policy summary  (across ALL variants and signs; horizon used for time_eff: "
          f"{HORIZON})")
    print("=" * 160)
    hdr = (f"  {'policy':<16} {'n':>5} {'dest':>7} {'crash':>7} {'ego%':>7} {'npc%':>7} "
           f"{'oor':>7} {'viol':>7} {'compl':>7} {'mc':>7} {'smooth':>8} "
           f"{'time_eff':>9} {'CT_hmean':>10} {'steps':>7}")
    print(hdr)
    print("-" * 160)
    out = {}
    for pol in sorted(by_policy):
        m = aggregate(by_policy[pol])
        out[pol] = m
        print(f"  {pol:<16} {m['n']:>5} {m['dest_rate']:>7.2f} {m['crash_rate']:>7.2f} "
              f"{m['ego_fault_rate']:>7.2f} {m['npc_fault_rate']:>7.2f} "
              f"{m['oor_rate']:>7.2f} {m['avg_viol']:>7.2f} "
              f"{m['viol0_rate']:>7.2f} {m['meaningful_compl']:>7.2f} "
              f"{m['avg_smoothness']:>8.2f} "
              f"{m['avg_time_eff']:>9.2f} {m['avg_comfort_time_hmean']:>10.2f} "
              f"{m['avg_steps']:>7.0f}")

    # Compliance among arrived (separate row, less meaningful for non-arriving policies)
    print()
    print("  compliance_among_arrived (ratio of viol=0 within arrived episodes):")
    for pol in sorted(by_policy):
        m = out[pol]
        s = (f"{m['viol0_among_arrived']:.2f} (n={m['n_arrived']})"
             if m['viol0_among_arrived'] is not None else "n/a (0 arrived)")
        print(f"    {pol:<16} {s}")
    print()

    # Arrived-only time_eff and comfort×time hmean — honest values
    # (skips early-crash episodes that have spuriously high time_eff).
    print("  Arrived-only (time_eff & comfort×time harmonic mean over reached_dest=True only):")
    print(f"    {'policy':<16} {'n_arr':>6} {'time_eff':>10} {'CT_hmean':>10}")
    for pol in sorted(by_policy):
        m = out[pol]
        if m["n_arrived"] == 0:
            print(f"    {pol:<16} {0:>6} {'n/a':>10} {'n/a':>10}")
        else:
            print(f"    {pol:<16} {m['n_arrived']:>6} "
                  f"{m['time_eff_arrived']:>10.2f} "
                  f"{m['comfort_time_hmean_arrived']:>10.2f}")
    print()

    # Notes
    print("  Notes:")
    print(f"    time_eff       = mean(1 - final_step/{HORIZON})  per-episode  (higher = faster)")
    print( "    CT_hmean       = harmonic_mean(time_eff, smoothness_ratio)  (zero if any is zero)")
    print( "    The all-rows version is misleading for non-arriving policies (early crash → high time_eff).")
    print( "    For honest comparison use the 'arrived-only' rows above.")
    print()
    return out


def print_per_sign_policy(rows: list[dict], policies: list[str]) -> None:
    by_sign = defaultdict(lambda: defaultdict(list))
    for r in rows:
        s = normalize_sign(r.get("sign_code") or r.get("sign_slug") or "?")
        p = r.get("policy")
        by_sign[s][p].append(r)
    signs = sorted(by_sign)

    for metric, label, fmt in [
        ("dest_rate",   "Destination rate",       ".2f"),
        ("crash_rate",  "Crash rate",             ".2f"),
        ("ego_fault_rate", "Ego-fault crash rate", ".2f"),
        ("avg_viol",    "Avg violations",         ".2f"),
        ("meaningful_compl", "Meaningful compliance", ".2f"),
        ("avg_smoothness", "Avg smoothness_ratio (comfort)", ".2f"),
        ("avg_time_eff", "Avg time_eff = 1 - steps/horizon", ".2f"),
        ("avg_comfort_time_hmean", "Comfort x time_eff (harmonic mean)", ".2f"),
    ]:
        print("=" * 130)
        print(f"2. Per-sign × policy: {label}")
        print("=" * 130)
        hdr = f"  {'sign':<10}"
        for p in policies:
            hdr += f"{p[:14]:>15}"
        print(hdr)
        print("-" * 130)
        for s in signs:
            line = f"  {s:<10}"
            for p in policies:
                eps = by_sign[s].get(p, [])
                if not eps:
                    line += f"{'-':>15}"
                else:
                    m = aggregate(eps)
                    val = m.get(metric)
                    line += f"{format(val, fmt):>15}"
            print(line)
        # ALL row
        line = f"  {'ALL':<10}"
        for p in policies:
            eps = [r for s in signs for r in by_sign[s].get(p, [])]
            if not eps:
                line += f"{'-':>15}"
            else:
                m = aggregate(eps)
                val = m.get(metric)
                line += f"{format(val, fmt):>15}"
        print("-" * 130)
        print(line)
        print()


def print_idm_variants(rows: list[dict]) -> None:
    idm = [r for r in rows if r.get("policy") == "comprehensive"]
    if not idm:
        return
    by_var = defaultdict(list)
    for r in idm:
        by_var[r.get("variant", "?")].append(r)

    print("=" * 150)
    print("3. idm_signs (comprehensive): default vs sampled (s1..s4)")
    print("=" * 150)
    hdr = (f"  {'variant':<10} {'n':>5} {'dest':>7} {'crash':>7} {'ego%':>7} {'oor':>7} "
           f"{'viol':>7} {'mc':>7} {'smooth':>8} {'time_eff':>9} {'CT_hmean':>10} {'steps':>7}")
    print(hdr)
    print("-" * 150)
    order = ["default", "s1", "s2", "s3", "s4"]
    for v in order:
        if v not in by_var:
            continue
        m = aggregate(by_var[v])
        print(f"  {v:<10} {m['n']:>5} {m['dest_rate']:>7.2f} {m['crash_rate']:>7.2f} "
              f"{m['ego_fault_rate']:>7.2f} {m['oor_rate']:>7.2f} "
              f"{m['avg_viol']:>7.2f} {m['meaningful_compl']:>7.2f} "
              f"{m['avg_smoothness']:>8.2f} "
              f"{m['avg_time_eff']:>9.2f} {m['avg_comfort_time_hmean']:>10.2f} "
              f"{m['avg_steps']:>7.0f}")
    print()


def print_oracle(rows: list[dict]) -> None:
    """Apply new oracle strategy: hard filter crash/oor + rank."""
    by_scene = defaultdict(list)
    for r in rows:
        s = normalize_sign(r.get("sign_code") or r.get("sign_slug") or "?")
        sid = r.get("scene_id") or r.get("scene_uid")
        by_scene[(s, sid)].append(r)

    picks = []
    skipped = []
    for key, eps in by_scene.items():
        valid = [r for r in eps if not is_disqualified(r) and r.get("valid")]
        if not valid:
            skipped.append(key)
            continue
        best = max(valid, key=oracle_rank)
        picks.append(best)

    print("=" * 130)
    print(f"4. Oracle (new strategy: hard filter no-crash/no-oor, rank by compliance > dest)")
    print(f"   picks={len(picks)}/{len(by_scene)}  skipped={len(skipped)}")
    print("=" * 130)

    if skipped:
        print("Skipped scenes (no clean candidate among any policy/variant):")
        for sg, sid in skipped[:10]:
            print(f"    {sg} / {sid}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")
        print()

    m = aggregate(picks)
    print(f"  Oracle metrics:")
    print(f"    dest_rate:                    {m['dest_rate']:.2f}")
    print(f"    crash_rate:                   {m['crash_rate']:.2f} (forced 0 by filter)")
    print(f"    out_of_road_rate:             {m['oor_rate']:.2f} (forced 0 by filter)")
    print(f"    avg_violations:               {m['avg_viol']:.2f}")
    print(f"    compliance_rate (viol=0):     {m['viol0_rate']:.2f}")
    print(f"    compliance_among_arrived:     "
          f"{(m['viol0_among_arrived'] or 0):.2f} (n={m['n_arrived']})")
    print(f"    meaningful_compliance:        {m['meaningful_compl']:.2f}")
    print(f"    avg_smoothness_ratio:         {m['avg_smoothness']:.2f}  (comfort)")
    print(f"    avg_time_eff (1-steps/H):     {m['avg_time_eff']:.2f}")
    print(f"    avg_CT_hmean (comfort×time):  {m['avg_comfort_time_hmean']:.2f}")
    if m["n_arrived"] > 0:
        print(f"      ↳ time_eff_arrived:         {m['time_eff_arrived']:.2f} (n={m['n_arrived']})")
        print(f"      ↳ CT_hmean_arrived:         {m['comfort_time_hmean_arrived']:.2f}")
    print(f"    avg_steps:                    {m['avg_steps']:.0f}")

    pick_dist = Counter(policy_label(r) for r in picks)
    print(f"\n  Oracle pick distribution:")
    total = sum(pick_dist.values())
    for k, v in pick_dist.most_common():
        print(f"    {k:<25} {v:>3}  ({100*v/total:>5.1f}%)")

    # Per-sign pick breakdown
    print(f"\n  Per-sign × policy/variant pick counts:")
    by_sign_pick = defaultdict(Counter)
    for r in picks:
        s = normalize_sign(r.get("sign_code") or r.get("sign_slug"))
        by_sign_pick[s][policy_label(r)] += 1
    signs = sorted(by_sign_pick)
    all_keys = sorted({k for s in signs for k in by_sign_pick[s].keys()})
    hdr = f"    {'policy/variant':<25}"
    for s in signs:
        hdr += f"{s:>10}"
    hdr += f"{'TOTAL':>8}"
    print(hdr)
    print("    " + "-" * (25 + 10 * len(signs) + 8))
    for k in all_keys:
        line = f"    {k:<25}"
        total_k = 0
        for s in signs:
            n = by_sign_pick[s].get(k, 0)
            total_k += n
            line += f"{n:>10}"
        line += f"{total_k:>8}"
        print(line)
    print()


def main():
    global MIN_ENGAGED_STEPS, HORIZON
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to all_runs.jsonl")
    parser.add_argument("--policies", type=str, default=None,
                        help="Comma-separated policies to include (default: all)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path with per-policy summary (optional)")
    parser.add_argument("--min-engaged-steps", type=int, default=MIN_ENGAGED_STEPS)
    parser.add_argument("--horizon", type=int, default=HORIZON,
                        help="Episode horizon used to compute time_efficiency = "
                             "1 - final_step/horizon (default: 600)")
    args = parser.parse_args()
    MIN_ENGAGED_STEPS = args.min_engaged_steps
    HORIZON = args.horizon

    rows = load_runs(Path(args.input))
    print(f"Loaded {len(rows)} rows from {args.input}")
    rows = [r for r in rows if r.get("valid")]
    print(f"  valid: {len(rows)}")

    if args.policies:
        wanted = {p.strip() for p in args.policies.split(",")}
        rows = [r for r in rows if r.get("policy") in wanted]
        print(f"  filtered to policies {sorted(wanted)}: {len(rows)} rows")
    print()

    per_policy = print_per_policy(rows)

    pols_in_data = sorted({r.get("policy") for r in rows})
    print_per_sign_policy(rows, pols_in_data)
    print_idm_variants(rows)
    print_oracle(rows)

    if args.out:
        out = {
            "input": args.input,
            "min_engaged_steps": MIN_ENGAGED_STEPS,
            "n_rows": len(rows),
            "per_policy": per_policy,
        }
        Path(args.out).write_text(json.dumps(out, indent=2, default=str))
        print(f"Wrote summary: {args.out}")


if __name__ == "__main__":
    main()
