#!/usr/bin/env python3
"""Compute per-baseline metrics directly from consolidated <baseline>_replays.jsonl.

Use this when episodes_*.jsonl are scattered/incomplete but you've already
consolidated sidecars via consolidate_replays.py. Outputs the same shape as
aggregate_per_var_baselines.py → same downstream report tooling works.

Output:
   <runs_var>/cumulative_from_replays.json     — per-baseline metrics
   prints summary table to stdout

Usage:
   python3 aggregate_from_consolidated_replays.py \\
       --runs-var benchmark_2node_eval/runs/var_0 \\
       --out benchmark_2node_eval/cumulative_from_replays.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _safe_mean(values: list) -> float | None:
    vals = [v for v in values
            if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _aggregate_baseline(rows: list[dict]) -> dict:
    """Compute per-baseline metrics from list of replay.json rows.

    Each row has top-level fields (scene_id, scene_uid, ...) plus a `metrics`
    sub-dict with the per-episode KPIs. We aggregate from `metrics`.
    """
    n_total = len(rows)
    if n_total == 0:
        return {"n_episodes_total": 0, "n_episodes_ok": 0}

    metrics_list = [r.get("metrics") or {} for r in rows]

    success = sum(1 for m in metrics_list if m.get("success"))
    arrived = sum(1 for m in metrics_list if m.get("arrived_dest"))
    crashed = sum(1 for m in metrics_list if m.get("crashed"))
    out_of_road = sum(1 for m in metrics_list if m.get("out_of_road"))
    crashed_ego = sum(1 for m in metrics_list if m.get("crashed_ego_fault"))
    crashed_npc = sum(1 for m in metrics_list if m.get("crashed_npc_fault"))

    total_step_viol = sum(int(m.get("total_violations") or 0) for m in metrics_list)
    total_event_viol = sum(int(m.get("violations_event_count") or 0) for m in metrics_list)
    total_in_zone = sum(int(m.get("in_zone_total_steps") or 0) for m in metrics_list)

    by_class_step: dict[str, int] = defaultdict(int)
    by_class_event: dict[str, int] = defaultdict(int)
    in_zone_by_class: dict[str, int] = defaultdict(int)
    for m in metrics_list:
        for cls, n in (m.get("violations_by_class_step") or {}).items():
            by_class_step[cls] += int(n)
        for cls, n in (m.get("violations_by_class_event") or {}).items():
            by_class_event[cls] += int(n)
        for cls, n in (m.get("in_zone_by_class_step") or {}).items():
            in_zone_by_class[cls] += int(n)

    return {
        "n_episodes_total": n_total,
        "n_episodes_ok": n_total,    # consolidated replays are only ok-scenes
        "n_failed": 0,
        "success_rate": success / n_total,
        "arrival_rate": arrived / n_total,
        "crash_rate": crashed / n_total,
        "crash_ego_rate": crashed_ego / n_total,
        "crash_npc_rate": crashed_npc / n_total,
        "out_of_road_rate": out_of_road / n_total,
        "avg_total_reward": _safe_mean([m.get("total_reward") for m in metrics_list]),
        "avg_steps": _safe_mean([m.get("final_step") for m in metrics_list]),
        "avg_route_completion": _safe_mean([m.get("route_completion") for m in metrics_list]),
        "avg_route_completion_pct": (
            _safe_mean([100 * (m.get("route_completion") or 0) for m in metrics_list])
        ),
        "avg_driving_score": _safe_mean([m.get("driving_score") for m in metrics_list]),
        "avg_driving_efficiency": _safe_mean([m.get("driving_efficiency") for m in metrics_list]),
        "avg_smoothness": _safe_mean([m.get("smoothness_ratio") for m in metrics_list]),
        "avg_min_ttc_sec": _safe_mean([m.get("min_ttc_sec") for m in metrics_list]),
        "avg_distance_travelled_m": _safe_mean([m.get("distance_travelled_m") for m in metrics_list]),
        "avg_hard_brake_count": _safe_mean([m.get("hard_brake_count") for m in metrics_list]),
        "avg_hard_accel_count": _safe_mean([m.get("hard_accel_count") for m in metrics_list]),
        "avg_mean_abs_lane_offset": _safe_mean([m.get("mean_abs_lane_offset") for m in metrics_list]),
        "avg_mean_abs_steer_delta": _safe_mean([m.get("mean_abs_steer_delta") for m in metrics_list]),
        # Violation breakdowns:
        "total_violation_steps": total_step_viol,
        "total_violation_events": total_event_viol,
        "violations_by_class_step_total": dict(by_class_step),
        "violations_by_class_event_total": dict(by_class_event),
        # In-zone exposure:
        "total_in_zone_steps": total_in_zone,
        "in_zone_by_class_step_total": dict(in_zone_by_class),
        "in_zone_violation_rate_by_class": {
            cls: round(by_class_step.get(cls, 0) / cnt, 4)
            for cls, cnt in in_zone_by_class.items() if cnt > 0
        },
    }


def _print_table(per_baseline: dict[str, dict]) -> None:
    cols = [("n", "n_episodes_ok", "5d"),
            ("succ%", "success_rate", "6.2%"),
            ("arr%", "arrival_rate", "6.2%"),
            ("crash%", "crash_rate", "6.2%"),
            ("oor%", "out_of_road_rate", "5.2%"),
            ("rew", "avg_total_reward", "8.2f"),
            ("DS", "avg_driving_score", "6.2f"),
            ("smth", "avg_smoothness", "5.3f"),
            ("ttc", "avg_min_ttc_sec", "5.2f"),
            ("vstep", "total_violation_steps", "7d"),
            ("vevt", "total_violation_events", "5d"),
            ("zone", "total_in_zone_steps", "7d")]
    header = f"  {'baseline':<42}  " + "  ".join(f"{c[0]:>{int(''.join(filter(str.isdigit, c[2])))}}" for c in cols)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for b in sorted(per_baseline.keys()):
        m = per_baseline[b]
        row = f"  {b:<42}  "
        cells = []
        for label, key, fmt in cols:
            v = m.get(key)
            if v is None:
                width = int("".join(filter(str.isdigit, fmt)))
                cells.append("—".rjust(width))
            elif "%" in fmt:
                width = int(fmt.split(".")[0])
                prec = int(fmt.split(".")[1].rstrip("%"))
                cells.append(f"{100*v:>{width}.{prec}f}")
            else:
                cells.append(format(v, fmt))
        print(row + "  ".join(cells))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-var", required=True,
                   help="Dir with <baseline>_replays.jsonl files (output of consolidate_replays.py).")
    p.add_argument("--out", default=None,
                   help="Output JSON path. Default: <runs-var>/cumulative_from_replays.json")
    args = p.parse_args()

    runs_var = Path(args.runs_var).resolve()
    if not runs_var.is_dir():
        print(f"ERROR: not a dir: {runs_var}", file=sys.stderr)
        sys.exit(2)

    jsonl_files = sorted(runs_var.glob("*_replays.jsonl"))
    if not jsonl_files:
        print(f"ERROR: no *_replays.jsonl found under {runs_var}", file=sys.stderr)
        sys.exit(2)

    print(f"[scan] found {len(jsonl_files)} consolidated baseline files")

    per_baseline: dict[str, dict] = {}
    for fp in jsonl_files:
        baseline = fp.stem.removesuffix("_replays")
        rows: list[dict] = []
        with fp.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        per_baseline[baseline] = _aggregate_baseline(rows)
        print(f"  [agg] {baseline}: {len(rows)} scenes")

    out_path = Path(args.out).resolve() if args.out else (runs_var / "cumulative_from_replays.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "vars_processed": [runs_var.name],
        "n_vars_processed": 1,
        "cumulative_through_latest": per_baseline,
        "per_var": {runs_var.name: per_baseline},
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[write] {out_path}")
    print()
    _print_table(per_baseline)
    print()
    print(f"Cumulative JSON: {out_path}")
    print(f"Markdown report: python3 merge_and_report_2node.py --node1 {out_path} --node2 {out_path} --out-dir {out_path.parent}")


if __name__ == "__main__":
    main()
