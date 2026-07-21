#!/usr/bin/env python3
"""A/B comparison of carl legacy vs tracking over episodes_*.jsonl of two pilot runs.

Usage:
  python3 pilot_carl_speed_check.py --legacy <out_dir_legacy> --tracking <out_dir_tracking>

Each out_dir is the --benchmark-output of the corresponding run_benchmark.py run
(the script finds episodes_*.jsonl recursively). Prints a metric table and a verdict
on the plan's stop criteria: dest_rate no lower than −5 pp, OOR ≤ 1.5×, no steer_delta growth.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st


def load_episodes(root: str) -> list[dict]:
    files = glob.glob(os.path.join(root, "**", "episodes_*.jsonl"), recursive=True)
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        raise SystemExit(f"no episodes_*.jsonl under {root}")
    return rows


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("steps", 0) > 10]
    speeds = [r["distance_travelled_m"] / (r["steps"] * 0.1) * 3.6 for r in ok]
    return {
        "episodes": len(rows),
        "median_speed_kmh": st.median(speeds) if speeds else 0.0,
        "mean_speed_kmh": st.mean(speeds) if speeds else 0.0,
        "dest_rate": st.mean([bool(r.get("reached_dest")) for r in rows]),
        "crash_rate": st.mean([bool(r.get("crashed")) for r in rows]),
        "oor_rate": st.mean([bool(r.get("out_of_road")) for r in rows]),
        "hard_brake/ep": st.mean([r.get("hard_brake_count", 0) for r in rows]),
        "hard_accel/ep": st.mean([r.get("hard_accel_count", 0) for r in rows]),
        "steer_delta": st.mean([r.get("mean_abs_steer_delta", 0.0) for r in rows]),
        "sign_compliance": st.mean([r.get("sign_violations", 0) == 0 for r in rows]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy", required=True)
    ap.add_argument("--tracking", required=True)
    args = ap.parse_args()

    a = summarize(load_episodes(args.legacy))
    b = summarize(load_episodes(args.tracking))

    w = max(len(k) for k in a)
    print(f"{'metric':<{w}}  {'legacy':>10}  {'tracking':>10}  {'Δ':>8}")
    for k in a:
        va, vb = a[k], b[k]
        d = vb - va
        print(f"{k:<{w}}  {va:>10.3f}  {vb:>10.3f}  {d:>+8.3f}")

    print("\n--- verdict ---")
    faster = b["median_speed_kmh"] > a["median_speed_kmh"] + 3
    print(f"got faster:           {'YES' if faster else 'NO'} "
          f"({a['median_speed_kmh']:.1f} → {b['median_speed_kmh']:.1f} km/h)")
    checks = [
        ("dest_rate did not drop (>5 pp — stop)", b["dest_rate"] >= a["dest_rate"] - 0.05),
        ("OOR ≤ 1.5×", b["oor_rate"] <= max(a["oor_rate"], 0.02) * 1.5),
        ("lateral unaffected (steer_delta)", b["steer_delta"] <= a["steer_delta"] * 1.3 + 0.005),
        ("stopped riding the brake", b["hard_brake/ep"] < a["hard_brake/ep"]),
    ]
    ok_all = True
    for name, ok in checks:
        ok_all &= ok
        print(f"{'✓' if ok else '✗'} {name}")
    print("\nVERDICT:", "fix accepted — go run the full pilot/eval"
          if (faster and ok_all) else "red flags present — see table")


if __name__ == "__main__":
    main()
