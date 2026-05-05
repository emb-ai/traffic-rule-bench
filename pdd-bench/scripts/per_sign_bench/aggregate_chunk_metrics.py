#!/usr/bin/env python3
"""Aggregate per-var baseline metrics → append to a single cumulative file.

After each var_<i>.json finishes processing on a node, this collects the
episode-level metrics from each baseline's episodes_<policy>.jsonl and
writes (a) a per-var summary and (b) a cumulative-through-var-i summary
into one rolling cumulative.json.

The cumulative file is rewritten atomically each call.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_episodes(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _safe_mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and (isinstance(v, (int, float))) and math.isfinite(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _aggregate_baseline(rows: list[dict]) -> dict:
    """Compute per-baseline metrics from a list of episode rows."""
    n_total = len(rows)
    ok_rows = [r for r in rows if r.get("ok")]
    n_ok = len(ok_rows)

    success = sum(1 for r in ok_rows if r.get("success"))
    arrived = sum(1 for r in ok_rows if r.get("reached_dest"))
    crashed = sum(1 for r in ok_rows if r.get("crashed"))
    out_of_road = sum(1 for r in ok_rows if r.get("out_of_road"))

    # Per-step + per-class accumulators
    total_violation_steps = sum(int(r.get("violations") or 0) for r in ok_rows)
    total_violation_events = sum(int(r.get("violations_event_count") or 0) for r in ok_rows)
    total_in_zone_steps = sum(int(r.get("in_zone_total_steps") or 0) for r in ok_rows)

    by_class_step: dict[str, int] = defaultdict(int)
    by_class_event: dict[str, int] = defaultdict(int)
    in_zone_by_class: dict[str, int] = defaultdict(int)

    for r in ok_rows:
        for cls, cnt in (r.get("violations_by_class_step") or {}).items():
            by_class_step[cls] += int(cnt)
        for cls, cnt in (r.get("violations_by_class_event") or {}).items():
            by_class_event[cls] += int(cnt)
        for cls, cnt in (r.get("in_zone_by_class_step") or {}).items():
            in_zone_by_class[cls] += int(cnt)

    return {
        "n_episodes_total": n_total,
        "n_episodes_ok": n_ok,
        "n_failed": n_total - n_ok,
        "success_rate": _safe_rate(success, n_ok),
        "arrival_rate": _safe_rate(arrived, n_ok),
        "crash_rate": _safe_rate(crashed, n_ok),
        "out_of_road_rate": _safe_rate(out_of_road, n_ok),
        "avg_total_reward": _safe_mean([r.get("total_reward") for r in ok_rows]),
        "avg_steps": _safe_mean([r.get("steps") for r in ok_rows]),
        "avg_route_completion_pct": _safe_mean([r.get("route_completion_pct") for r in ok_rows]),
        "avg_driving_score": _safe_mean([r.get("driving_score") for r in ok_rows]),
        "avg_driving_efficiency": _safe_mean([r.get("driving_efficiency") for r in ok_rows]),
        "avg_smoothness": _safe_mean([r.get("smoothness") for r in ok_rows]),
        "avg_min_ttc_sec": _safe_mean([r.get("min_ttc_sec") for r in ok_rows]),
        "avg_distance_travelled_m": _safe_mean([r.get("distance_travelled_m") for r in ok_rows]),
        "avg_hard_brake_count": _safe_mean([r.get("hard_brake_count") for r in ok_rows]),
        "avg_hard_accel_count": _safe_mean([r.get("hard_accel_count") for r in ok_rows]),
        # Violation breakdowns:
        "total_violation_steps": int(total_violation_steps),
        "total_violation_events": int(total_violation_events),
        "violations_by_class_step_total": dict(by_class_step),
        "violations_by_class_event_total": dict(by_class_event),
        # In-zone exposure:
        "total_in_zone_steps": int(total_in_zone_steps),
        "in_zone_by_class_step_total": dict(in_zone_by_class),
        # Derived: violation rate per step ego was in-zone (per-class).
        "in_zone_violation_rate_by_class": {
            cls: round(by_class_step.get(cls, 0) / cnt, 4)
            for cls, cnt in in_zone_by_class.items() if cnt > 0
        },
    }


def _merge_cumulative(prev: dict, this_var: dict) -> dict:
    """Add this var's per-baseline metrics into the cumulative roll-up."""
    out: dict[str, dict] = dict(prev) if prev else {}
    for baseline, m in this_var.items():
        if baseline not in out:
            out[baseline] = {
                "n_episodes_total": 0,
                "n_episodes_ok": 0,
                "n_failed": 0,
                "n_success": 0,
                "n_arrived": 0,
                "n_crashed": 0,
                "n_out_of_road": 0,
                "sum_total_reward": 0.0,
                "sum_steps": 0,
                "sum_route_completion_pct": 0.0,
                "sum_driving_score": 0.0,
                "sum_driving_efficiency": 0.0,
                "sum_smoothness": 0.0,
                "sum_distance_travelled_m": 0.0,
                "sum_hard_brake_count": 0.0,
                "sum_hard_accel_count": 0.0,
                "min_ttc_samples": [],
                "total_violation_steps": 0,
                "total_violation_events": 0,
                "violations_by_class_step_total": {},
                "violations_by_class_event_total": {},
                "total_in_zone_steps": 0,
                "in_zone_by_class_step_total": {},
                "vars_seen": [],
            }
        b = out[baseline]
        b["n_episodes_total"] += m["n_episodes_total"]
        b["n_episodes_ok"] += m["n_episodes_ok"]
        b["n_failed"] += m["n_failed"]
        b["n_success"] += int(m["n_episodes_ok"] * m["success_rate"])
        b["n_arrived"] += int(m["n_episodes_ok"] * m["arrival_rate"])
        b["n_crashed"] += int(m["n_episodes_ok"] * m["crash_rate"])
        b["n_out_of_road"] += int(m["n_episodes_ok"] * m["out_of_road_rate"])
        n_ok = m["n_episodes_ok"]
        for k_sum, k_avg in (
            ("sum_total_reward", "avg_total_reward"),
            ("sum_steps", "avg_steps"),
            ("sum_route_completion_pct", "avg_route_completion_pct"),
            ("sum_driving_score", "avg_driving_score"),
            ("sum_driving_efficiency", "avg_driving_efficiency"),
            ("sum_smoothness", "avg_smoothness"),
            ("sum_distance_travelled_m", "avg_distance_travelled_m"),
            ("sum_hard_brake_count", "avg_hard_brake_count"),
            ("sum_hard_accel_count", "avg_hard_accel_count"),
        ):
            v = m.get(k_avg)
            if v is not None and n_ok > 0:
                b[k_sum] += float(v) * n_ok
        if m.get("avg_min_ttc_sec") is not None and n_ok > 0:
            b["min_ttc_samples"].append((float(m["avg_min_ttc_sec"]), n_ok))
        b["total_violation_steps"] += m["total_violation_steps"]
        b["total_violation_events"] += m["total_violation_events"]
        for cls, cnt in m.get("violations_by_class_step_total", {}).items():
            b["violations_by_class_step_total"][cls] = (
                b["violations_by_class_step_total"].get(cls, 0) + int(cnt))
        for cls, cnt in m.get("violations_by_class_event_total", {}).items():
            b["violations_by_class_event_total"][cls] = (
                b["violations_by_class_event_total"].get(cls, 0) + int(cnt))
        b["total_in_zone_steps"] += m["total_in_zone_steps"]
        for cls, cnt in m.get("in_zone_by_class_step_total", {}).items():
            b["in_zone_by_class_step_total"][cls] = (
                b["in_zone_by_class_step_total"].get(cls, 0) + int(cnt))
    return out


def _cumulative_view(cum: dict) -> dict:
    """Compute current ratios from cumulative sums (called at write time)."""
    view: dict[str, Any] = {}
    for baseline, b in cum.items():
        n_ok = max(b["n_episodes_ok"], 1)
        ttc_total_w = sum(w for _, w in b["min_ttc_samples"])
        avg_ttc = (sum(v * w for v, w in b["min_ttc_samples"]) / ttc_total_w
                   if ttc_total_w > 0 else None)
        view[baseline] = {
            "n_episodes_total": b["n_episodes_total"],
            "n_episodes_ok": b["n_episodes_ok"],
            "n_failed": b["n_failed"],
            "success_rate": _safe_rate(b["n_success"], n_ok),
            "arrival_rate": _safe_rate(b["n_arrived"], n_ok),
            "crash_rate": _safe_rate(b["n_crashed"], n_ok),
            "out_of_road_rate": _safe_rate(b["n_out_of_road"], n_ok),
            "avg_total_reward": b["sum_total_reward"] / n_ok if n_ok else None,
            "avg_steps": b["sum_steps"] / n_ok if n_ok else None,
            "avg_route_completion_pct": b["sum_route_completion_pct"] / n_ok if n_ok else None,
            "avg_driving_score": b["sum_driving_score"] / n_ok if n_ok else None,
            "avg_driving_efficiency": b["sum_driving_efficiency"] / n_ok if n_ok else None,
            "avg_smoothness": b["sum_smoothness"] / n_ok if n_ok else None,
            "avg_distance_travelled_m": b["sum_distance_travelled_m"] / n_ok if n_ok else None,
            "avg_hard_brake_count": b["sum_hard_brake_count"] / n_ok if n_ok else None,
            "avg_hard_accel_count": b["sum_hard_accel_count"] / n_ok if n_ok else None,
            "avg_min_ttc_sec": avg_ttc,
            "total_violation_steps": b["total_violation_steps"],
            "total_violation_events": b["total_violation_events"],
            "violations_by_class_step_total": b["violations_by_class_step_total"],
            "violations_by_class_event_total": b["violations_by_class_event_total"],
            "total_in_zone_steps": b["total_in_zone_steps"],
            "in_zone_by_class_step_total": b["in_zone_by_class_step_total"],
            "in_zone_violation_rate_by_class": {
                cls: round(b["violations_by_class_step_total"].get(cls, 0) / cnt, 4)
                for cls, cnt in b["in_zone_by_class_step_total"].items() if cnt > 0
            },
        }
    return view


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True,
                   help="Directory with <baseline>/episodes_<policy>.jsonl per baseline.")
    p.add_argument("--var-name", required=True,
                   help="Identifier for this var (e.g. var_0).")
    p.add_argument("--cumulative-path", required=True,
                   help="Path to cumulative.json — atomic-rewritten each call.")
    p.add_argument("--node", default="?", help="Node label for the cumulative file.")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.is_dir():
        print(f"ERROR: runs-dir not found: {runs_dir}", file=sys.stderr)
        sys.exit(2)

    cum_path = Path(args.cumulative_path).resolve()
    cum_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-var aggregation. Two layouts are supported:
    #   1) Flat:    runs_dir/<baseline>/episodes_*.jsonl
    #   2) Chunked: runs_dir/chunk_<k>/<baseline>/episodes_*.jsonl
    # If chunk_*/ subdirs exist, episodes from EVERY chunk are merged in-memory.
    per_var: dict[str, dict] = {}

    # Collect (baseline_name → list of episodes_*.jsonl files) across both layouts.
    by_baseline: dict[str, list[Path]] = {}
    chunk_dirs = sorted(d for d in runs_dir.iterdir()
                        if d.is_dir() and d.name.startswith("chunk_"))

    if chunk_dirs:
        # Chunked layout: walk chunk_*/<baseline>/episodes_*.jsonl
        for cd in chunk_dirs:
            for bd in cd.iterdir():
                if not bd.is_dir():
                    continue
                ep_files = list(bd.glob("episodes_*.jsonl"))
                if ep_files:
                    by_baseline.setdefault(bd.name, []).extend(ep_files)

    # Flat layout (also used as fallback / additive after chunked).
    for baseline_dir in sorted(runs_dir.iterdir()):
        if not baseline_dir.is_dir() or baseline_dir.name.startswith("chunk_"):
            continue
        ep_files = list(baseline_dir.glob("episodes_*.jsonl"))
        if ep_files:
            by_baseline.setdefault(baseline_dir.name, []).extend(ep_files)

    for baseline, ep_files in sorted(by_baseline.items()):
        rows: list[dict] = []
        for f in ep_files:
            rows.extend(_load_episodes(f))
        per_var[baseline] = _aggregate_baseline(rows)
        layout = "chunked" if chunk_dirs and any("chunk_" in str(p) for p in ep_files) else "flat"
        print(f"[agg] {baseline} ({layout}, {len(ep_files)} files): "
              f"ok={per_var[baseline]['n_episodes_ok']}/{per_var[baseline]['n_episodes_total']}")

    # Read prior per-var blocks (per_var is the source-of-truth).
    prior_per_var: dict = {}
    if cum_path.exists():
        try:
            existing = json.loads(cum_path.read_text(encoding="utf-8"))
            prior_per_var = existing.get("per_var", {}) or {}
        except Exception:
            prior_per_var = {}
    # Overwrite this var's entry — idempotent: re-aggregating the same var
    # (e.g. when node 2 finishes its baselines after node 1 already aggregated)
    # replaces the previous partial snapshot without double-counting.
    prior_per_var[args.var_name] = per_var

    # Recompute cumulative from scratch by replaying every recorded var.
    # This is the idempotent path — no running state.json is needed because
    # cumulative is a pure function of `per_var`.
    state: dict = {}
    for v_name in sorted(prior_per_var.keys()):
        state = _merge_cumulative(state, prior_per_var[v_name])
    cumulative_view = _cumulative_view(state)
    state_path = cum_path.with_suffix(".state.json")

    output = {
        "node": args.node,
        "vars_processed": sorted(prior_per_var.keys()),
        "n_vars_processed": len(prior_per_var),
        "cumulative_through_latest": cumulative_view,
        "per_var": prior_per_var,
    }

    # Atomic write of public cumulative + private state.
    tmp = cum_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(cum_path)
    state_blob = {"cumulative_raw": state}
    tmp_state = state_path.with_suffix(".json.tmp")
    tmp_state.write_text(json.dumps(state_blob, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_state.replace(state_path)

    print(f"[ok] cumulative → {cum_path}  (node={args.node}, "
          f"{len(prior_per_var)} vars, {len(cumulative_view)} baselines)")


if __name__ == "__main__":
    main()
