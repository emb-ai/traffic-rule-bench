#!/usr/bin/env python3
"""Aggregate metrics_per_episode.csv → per-baseline / per-sign / per-var CSVs
and legacy cumulative JSON for downstream MD reports.

Slices produced:
  1. agg_per_baseline_var.csv       — one row per (baseline, var)
  2. agg_per_sign_baseline_var.csv  — one row per (sign, baseline, var)
  3. agg_per_baseline.csv           — one row per baseline (cumulative across vars)
  4. agg_per_sign_baseline.csv      — one row per (sign, baseline) cumulative

Plus, for backward compatibility with the existing MD-report scripts:
  5. cumulative.json                — {chunks, per_baseline, per_sign} schema for
                                       generate_cumulative_markdown_report.py and
                                       generate_category_aggregation_report.py
  6. cumulative_2node.json          — {vars_processed, cumulative_through_latest,
                                       per_var} schema for merge_and_report_2node.py

Usage:
  python3 aggregate_episode_metrics.py \
      --csv     /path/to/metrics_per_episode.csv \
      --out-dir /path/to/benchmark_2node_eval
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "expert_selection_exps"))

from select_experts import (  # noqa: E402
    BETA_DEFAULT,
    HORIZON_DEFAULT,
    SIGN_CLASS_MAP,
    time_eff,
    f1_score,
)


# End-of-zone signs (class name starts with "End"). Only these are eligible
# for paired grouping into "<major>.<minor>.x" buckets. Start-of-zone signs
# stay individual — pairing them across left/right or bus/bike doesn't make
# sense semantically (the action differs).
END_OF_ZONE_SIGNS: set[str] = {
    pdd for pdd, cls in SIGN_CLASS_MAP.items() if cls.startswith("End")
}


# Canonical baseline ordering for tables: all base versions first, then their
# rule-augmented counterparts. Within each block, families ordered as
# idm → ppo → carl → plant2.
BASELINE_ORDER: list[str] = [
    # === Base versions ===
    "idm_default", "idm_s1", "idm_s2", "idm_s3", "idm_s4",
    "ppo_expert",
    "carl",
    "plant2", "plant2_artem",
    # === Rule-augmented versions ===
    "comprehensive_rule_expert_default",
    "comprehensive_rule_expert_s1",
    "comprehensive_rule_expert_s2",
    "comprehensive_rule_expert_s3",
    "comprehensive_rule_expert_s4",
    "rule_compliant",
    "carl_rule",
    "plant2_rule",
]


def baseline_sort_key(baseline: str) -> tuple[int, str]:
    """Sort key putting known baselines in BASELINE_ORDER, unknown ones at the
    end alphabetically."""
    try:
        return (BASELINE_ORDER.index(baseline), baseline)
    except ValueError:
        return (len(BASELINE_ORDER), baseline)


def _to_bool(s: str) -> bool:
    return s == "True"


def sign_group(row_or_code) -> str:
    """Map a row (or bare pdd_code) to its 'paired' group label.

    Three classification paths:

    1. Paired-zone scene (manifest source contains "paired", e.g. pgmap_paired):
       group = "<pdd_code_start>+<pdd_code_end>" — e.g. "2.1+2.2", "5.31+5.32".
       This matches the manifest's actual zone semantics: ego enters at the
       start sign and the zone ends at the end sign.

    2. Standalone END-of-zone sign with variants (3+ dot-separated parts):
       group = "<major>.<minor>.x" — e.g. 5.12.1+5.12.2 -> "5.12.x",
       5.14.3+5.14.4 -> "5.14.x".

    3. Otherwise (START signs, END singletons, unknown codes): individual.

    Accepts either a row dict (preferred — uses manifest fields) or a bare
    string pdd_code (legacy — paired path 1 disabled).
    """
    if isinstance(row_or_code, dict):
        row = row_or_code
        pdd_code = row.get("pdd_code") or ""
        # Path 1 — paired manifest scene
        if row.get("is_paired_scene"):
            s = (row.get("pdd_code_start") or "").strip()
            e = (row.get("pdd_code_end") or "").strip()
            if s and e:
                return f"{s}+{e}"
    else:
        pdd_code = row_or_code or ""

    if not pdd_code:
        return "_unknown"
    # Path 2 — standalone END-of-zone variant
    parts = pdd_code.split(".")
    if len(parts) >= 3 and pdd_code in END_OF_ZONE_SIGNS:
        return f"{parts[0]}.{parts[1]}.x"
    # Path 3 — individual
    return pdd_code


def _to_int(s: str, default: int = 0) -> int:
    try:
        return int(s) if s != "" else default
    except (ValueError, TypeError):
        return default


def _to_float(s: str, default: float | None = None) -> float | None:
    try:
        return float(s) if s != "" else default
    except (ValueError, TypeError):
        return default


def _mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None and isinstance(v, (int, float))
            and math.isfinite(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _rate(num: int, den: int) -> float | None:
    return float(num) / float(den) if den > 0 else None


def _round_or_none(x, n=6):
    return None if x is None else round(float(x), n)


# ---------------------------------------------------------------------------
# Load CSV → list of dict-rows with typed values
# ---------------------------------------------------------------------------
def load_episode_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row = {
                "var_name": r["var_name"],
                "var_idx": _to_int(r["var_idx"]),
                "baseline": r["baseline"],
                "policy": r["policy"],
                "variant": r["variant"],
                "display_policy": r["display_policy"],
                "backend": r["backend"],
                "pdd_code": r["pdd_code"],
                "sign_slug": r["sign_slug"],
                "target_sign_class": r["target_sign_class"] or None,
                "is_no_entry_sign": _to_bool(r["is_no_entry_sign"]),
                "scene_id": r["scene_id"],
                "scene_uid": r["scene_uid"],
                "manifest_source": r.get("manifest_source", ""),
                "is_paired_scene": _to_bool(r.get("is_paired_scene", "")),
                "pdd_code_start": r.get("pdd_code_start", ""),
                "pdd_code_end": r.get("pdd_code_end", ""),
                "pdd_code_target": r.get("pdd_code_target", ""),
                "sign_type_start": r.get("sign_type_start", ""),
                "sign_type_end": r.get("sign_type_end", ""),
                "zone_length_m": _to_float(r.get("zone_length_m", "")),
                "valid": _to_bool(r["valid"]),
                "arrived_dest": _to_bool(r["arrived_dest"]),
                "crashed": _to_bool(r["crashed"]),
                "crashed_ego_fault": _to_bool(r["crashed_ego_fault"]),
                "crashed_npc_fault": _to_bool(r["crashed_npc_fault"]),
                "out_of_road": _to_bool(r["out_of_road"]),
                "success": _to_bool(r["success"]),
                "final_step": _to_int(r["final_step"]),
                "total_reward": _to_float(r["total_reward"]),
                "route_completion": _to_float(r["route_completion"]),
                "route_length_m": _to_float(r["route_length_m"]),
                "distance_travelled_m": _to_float(r["distance_travelled_m"]),
                "driving_score": _to_float(r["driving_score"]),
                "driving_efficiency": _to_float(r["driving_efficiency"]),
                "infraction_penalty": _to_float(r["infraction_penalty"]),
                "smoothness_ratio": _to_float(r["smoothness_ratio"]),
                "frame_smooth_ratio": _to_float(r["frame_smooth_ratio"]),
                "smooth_segments": _to_int(r["smooth_segments"]),
                "total_segments": _to_int(r["total_segments"]),
                "min_ttc_sec": _to_float(r["min_ttc_sec"]),
                "mean_abs_lane_offset": _to_float(r["mean_abs_lane_offset"]),
                "mean_abs_steer_delta": _to_float(r["mean_abs_steer_delta"]),
                "hard_brake_count": _to_int(r["hard_brake_count"]),
                "hard_accel_count": _to_int(r["hard_accel_count"]),
                "total_violations": _to_int(r["total_violations"]),
                "violations_event_count": _to_int(r["violations_event_count"]),
                "in_zone_total_steps": _to_int(r["in_zone_total_steps"]),
                "viol_high_sign": _to_int(r["viol_high_sign"]),
                "viol_high_traffic_light": _to_int(r["viol_high_traffic_light"]),
                "viol_high_crosswalk": _to_int(r["viol_high_crosswalk"]),
                "violations_by_class_step": json.loads(r["violations_by_class_step_json"] or "{}"),
                "violations_by_class_event": json.loads(r["violations_by_class_event_json"] or "{}"),
                "in_zone_by_class_step": json.loads(r["in_zone_by_class_step_json"] or "{}"),
                "target_violations_step": _to_int(r["target_violations_step"]) if r["target_violations_step"] != "" else None,
                "target_violations_event": _to_int(r["target_violations_event"]) if r["target_violations_event"] != "" else None,
                "target_in_zone_steps": _to_int(r["target_in_zone_steps"]) if r["target_in_zone_steps"] != "" else None,
                "target_in_zone": _to_bool(r["target_in_zone"]),
                "target_compliant_event": _to_bool(r["target_compliant_event"]) if r["target_compliant_event"] != "" else None,
                "target_compliant_step": _to_bool(r["target_compliant_step"]) if r["target_compliant_step"] != "" else None,
                "sign_compliant_high": _to_bool(r["sign_compliant_high"]),
                "tl_compliant": _to_bool(r["tl_compliant"]),
                "cw_compliant": _to_bool(r["cw_compliant"]),
                "dest_recomputed": _to_bool(r["dest_recomputed"]),
                "passes_filter": _to_bool(r["passes_filter"]),
                "comfort": _to_float(r["comfort"], 0.0) or 0.0,
            }
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Aggregation: compute one metric dict for a list of rows
# ---------------------------------------------------------------------------
def aggregate(rows: list[dict], beta: float = BETA_DEFAULT,
              horizon: int = HORIZON_DEFAULT) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    n_in_zone = sum(1 for r in rows if r["target_in_zone"])
    n_passing = sum(1 for r in rows if r["passes_filter"])
    n_arrived = sum(1 for r in rows if r["arrived_dest"])
    n_dest_recomp = sum(1 for r in rows if r["dest_recomputed"])
    n_crashed = sum(1 for r in rows if r["crashed"])
    n_oor = sum(1 for r in rows if r["out_of_road"])
    n_success = sum(1 for r in rows if r["success"])

    # --- High-level (3-key) compliance over ALL episodes
    n_sign_clean = sum(1 for r in rows if r["sign_compliant_high"])
    n_tl_clean = sum(1 for r in rows if r["tl_compliant"])
    n_cw_clean = sum(1 for r in rows if r["cw_compliant"])

    # --- High-level signs compliance restricted to in-zone subset
    in_zone_rows = [r for r in rows if r["target_in_zone"]]
    n_sign_clean_in_zone = sum(1 for r in in_zone_rows if r["sign_compliant_high"])

    # --- Per-episode (arrived AND sign_compliant) co-occurrence — averaged over
    # two denominators: all episodes (sr) and in-zone subset (x).
    n_arr_and_compl = sum(1 for r in rows
                          if r["arrived_dest"] and r["sign_compliant_high"])
    n_arr_and_compl_in_zone = sum(1 for r in in_zone_rows
                                  if r["arrived_dest"] and r["sign_compliant_high"])

    # --- Target-class (per-sign-class) compliance — only for rows where
    #     target_sign_class is known. Skip rows with no mapping (NaN-aware).
    rows_with_class = [r for r in rows if r["target_sign_class"]]
    n_with_class = len(rows_with_class)
    n_target_compliant_event = sum(1 for r in rows_with_class
                                    if r["target_compliant_event"])
    n_target_compliant_step = sum(1 for r in rows_with_class
                                   if r["target_compliant_step"])

    # --- Total compliance (no violations of any kind)
    n_clean_total = sum(1 for r in rows if r["total_violations"] == 0)

    # --- Cumulative violation totals + per-class
    sum_viol_steps = sum(r["total_violations"] for r in rows)
    sum_viol_events = sum(r["violations_event_count"] for r in rows)
    sum_in_zone_steps = sum(r["in_zone_total_steps"] for r in rows)
    by_class_step: Counter = Counter()
    by_class_event: Counter = Counter()
    in_zone_by_class: Counter = Counter()
    for r in rows:
        for cls, cnt in r["violations_by_class_step"].items():
            by_class_step[cls] += int(cnt or 0)
        for cls, cnt in r["violations_by_class_event"].items():
            by_class_event[cls] += int(cnt or 0)
        for cls, cnt in r["in_zone_by_class_step"].items():
            in_zone_by_class[cls] += int(cnt or 0)

    # --- Selection-style F1 (requires per-scene min/max final_step over passing rows)
    scene_minmax: dict[str, list[int]] = {}
    for r in rows:
        if not r["passes_filter"]:
            continue
        sid = r["scene_id"]
        fs = max(1, int(r["final_step"]))
        mm = scene_minmax.setdefault(sid, [10**9, 0])
        mm[0] = min(mm[0], fs)
        mm[1] = max(mm[1], fs)

    sum_te = sum_f1 = 0.0
    n_te = 0
    for r in rows:
        if not r["passes_filter"]:
            continue
        sid = r["scene_id"]
        mm = scene_minmax.get(sid, [1, 1])
        t = time_eff(r, mm[0], mm[1], formula="min_over_final")
        c = float(r["comfort"])
        sum_te += t
        sum_f1 += f1_score(t, c, beta)
        n_te += 1

    # --- Build output (None for empty denominators — caller renders as NaN/—)
    out: dict = {
        "n": n,
        "n_in_zone": n_in_zone,
        "n_passing": n_passing,
        "n_with_class": n_with_class,
        # Outcome rates
        "success_rate": _rate(n_success, n),
        "dest_rate": _rate(n_arrived, n),
        "dest_rate_recomputed": _rate(n_dest_recomp, n),
        "crash_rate": _rate(n_crashed, n),
        "out_of_road_rate": _rate(n_oor, n),
        "pass_rate": _rate(n_passing, n),
        # Compliance
        "sign_compliance_sr": _rate(n_sign_clean, n),
        "sign_compliance_x": _rate(n_sign_clean_in_zone, n_in_zone),
        "traffic_light_sr": _rate(n_tl_clean, n),
        "crosswalk_sr": _rate(n_cw_clean, n),
        "target_compliance_rate_event": _rate(n_target_compliant_event, n_with_class),
        "target_compliance_rate_step": _rate(n_target_compliant_step, n_with_class),
        "compliance_rate_total": _rate(n_clean_total, n),
        # Driving quality (avg)
        "avg_total_reward": _mean([r["total_reward"] for r in rows]),
        "avg_steps": _mean([r["final_step"] for r in rows]),
        "avg_route_completion": _mean([r["route_completion"] for r in rows]),
        "avg_route_completion_pct": _mean([
            (r["route_completion"] * 100) if r["route_completion"] is not None else None
            for r in rows]),
        "avg_route_length_m": _mean([r["route_length_m"] for r in rows]),
        "avg_distance_travelled_m": _mean([r["distance_travelled_m"] for r in rows]),
        "avg_driving_score": _mean([r["driving_score"] for r in rows]),
        "avg_driving_efficiency": _mean([r["driving_efficiency"] for r in rows]),
        "avg_smoothness": _mean([r["smoothness_ratio"] for r in rows]),
        "avg_frame_smoothness": _mean([r["frame_smooth_ratio"] for r in rows]),
        "avg_min_ttc_sec": _mean([r["min_ttc_sec"] for r in rows]),
        "avg_hard_brake_count": _mean([r["hard_brake_count"] for r in rows]),
        "avg_hard_accel_count": _mean([r["hard_accel_count"] for r in rows]),
        "avg_mean_abs_lane_offset": _mean([r["mean_abs_lane_offset"] for r in rows]),
        "avg_mean_abs_steer_delta": _mean([r["mean_abs_steer_delta"] for r in rows]),
        # Violation totals
        "total_violation_steps": sum_viol_steps,
        "total_violation_events": sum_viol_events,
        "total_in_zone_steps": sum_in_zone_steps,
        "avg_total_violations": _mean([r["total_violations"] for r in rows]),
        "target_avg_violations_event": _mean([
            r["target_violations_event"] for r in rows_with_class]),
        "target_avg_violations_step": _mean([
            r["target_violations_step"] for r in rows_with_class]),
        # Violation breakdowns by class (dicts)
        "violations_by_class_step_total": dict(by_class_step),
        "violations_by_class_event_total": dict(by_class_event),
        "in_zone_by_class_step_total": dict(in_zone_by_class),
        "in_zone_violation_rate_by_class": {
            cls: round(by_class_step.get(cls, 0) / cnt, 4)
            for cls, cnt in in_zone_by_class.items() if cnt > 0
        },
        # Selection-style
        "avg_comfort": _mean([r["comfort"] for r in rows]),
        "avg_time_eff_passing": (sum_te / n_te) if n_te else None,
        "avg_f1_passing": (sum_f1 / n_te) if n_te else None,
        # Per-episode (arrived ∧ sign_compliant) averaged over two subsets:
        #   _sr — over ALL episodes (denominator = n)
        #   _x  — over IN-ZONE episodes only (denominator = n_in_zone)
        # Pairs the convention used by sign_compliance_{sr,x}.
        "dest_x_comp_sr": _rate(n_arr_and_compl, n),
        "dest_x_comp_x":  _rate(n_arr_and_compl_in_zone, n_in_zone),
    }
    return out


# ---------------------------------------------------------------------------
# CSV writers (flat tables)
# ---------------------------------------------------------------------------
FLAT_METRIC_COLUMNS = [
    "n", "n_in_zone", "n_passing", "n_with_class",
    "success_rate", "dest_rate", "dest_rate_recomputed",
    "crash_rate", "out_of_road_rate", "pass_rate",
    "sign_compliance_sr", "sign_compliance_x",
    "traffic_light_sr", "crosswalk_sr",
    "target_compliance_rate_event", "target_compliance_rate_step",
    "compliance_rate_total",
    "avg_total_reward", "avg_steps", "avg_route_completion",
    "avg_route_completion_pct", "avg_route_length_m",
    "avg_distance_travelled_m",
    "avg_driving_score", "avg_driving_efficiency",
    "avg_smoothness", "avg_frame_smoothness",
    "avg_min_ttc_sec",
    "avg_hard_brake_count", "avg_hard_accel_count",
    "avg_mean_abs_lane_offset", "avg_mean_abs_steer_delta",
    "total_violation_steps", "total_violation_events", "total_in_zone_steps",
    "avg_total_violations",
    "target_avg_violations_event", "target_avg_violations_step",
    "avg_comfort", "avg_time_eff_passing", "avg_f1_passing",
    "dest_x_comp_sr", "dest_x_comp_x",
]


def _flat_row(metrics: dict) -> dict:
    """Strip nested dicts from a metrics dict to keep CSV flat. JSON-encode them."""
    out = {}
    for k in FLAT_METRIC_COLUMNS:
        v = metrics.get(k)
        out[k] = "" if v is None else (round(v, 6) if isinstance(v, float) else v)
    out["violations_by_class_step_total_json"] = json.dumps(
        metrics.get("violations_by_class_step_total") or {}, sort_keys=True)
    out["violations_by_class_event_total_json"] = json.dumps(
        metrics.get("violations_by_class_event_total") or {}, sort_keys=True)
    out["in_zone_by_class_step_total_json"] = json.dumps(
        metrics.get("in_zone_by_class_step_total") or {}, sort_keys=True)
    out["in_zone_violation_rate_by_class_json"] = json.dumps(
        metrics.get("in_zone_violation_rate_by_class") or {}, sort_keys=True)
    return out


def _is_paired_group(g: str) -> bool:
    """Group label is 'paired' if it's a paired-zone (contains '+') or
    END-variant (ends with '.x'). Used to decide if a GROUP row is worth
    emitting on top of its SIGN rows even when only 1 member appears."""
    return ("+" in g) or g.endswith(".x")


def write_paired_view(path: Path,
                       agg_pgb: dict[tuple[str, str], dict],
                       agg_pgpb: dict[tuple[str, str, str], dict],
                       members_pgb: dict[tuple[str, str], set[str]]) -> None:
    """Interleave per-group totals with per-member-sign breakdowns.

    For each (sign_group, baseline) the rows are:
      kind=GROUP, sign_group=2.1+2.2, pdd_code=,        <paired-zone total>
      kind=SIGN,  sign_group=2.1+2.2, pdd_code=2.1,     <within-group target=2.1>
      kind=SIGN,  sign_group=2.1+2.2, pdd_code=2.2,     <within-group target=2.2>

    GROUP row is emitted when the group is paired (label contains '+' or '.x')
    OR has 2+ pdd_code members. Pure singletons (sign_group == pdd_code, 1
    member) emit only their SIGN row to avoid duplication.

    SIGN rows use the WITHIN-GROUP aggregation (agg_pgpb) so paired-zone
    target metrics don't mix with non-paired same-pdd_code rows.
    """
    fieldnames = (["kind", "sign_group", "pdd_code", "baseline"] +
                  FLAT_METRIC_COLUMNS + [
                      "violations_by_class_step_total_json",
                      "violations_by_class_event_total_json",
                      "in_zone_by_class_step_total_json",
                      "in_zone_violation_rate_by_class_json",
                  ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for (g, b) in sorted(agg_pgb.keys(),
                              key=lambda kk: (kk[0], baseline_sort_key(kk[1]))):
            sign_members = sorted(members_pgb.get((g, b), set()))
            emit_group = _is_paired_group(g) or len(sign_members) > 1
            if emit_group:
                grp_metrics = agg_pgb[(g, b)]
                row = {"kind": "GROUP", "sign_group": g, "pdd_code": "", "baseline": b}
                row.update(_flat_row(grp_metrics))
                w.writerow(row)
            for s in sign_members:
                m = agg_pgpb.get((g, s, b))
                if m is None:
                    continue
                row = {"kind": "SIGN", "sign_group": g, "pdd_code": s, "baseline": b}
                row.update(_flat_row(m))
                w.writerow(row)


def write_paired_view_per_var(path: Path,
                                 agg_pgbv: dict[tuple[str, str, str], dict],
                                 agg_pgpbv: dict[tuple[str, str, str, str], dict],
                                 members_pgbv: dict[tuple[str, str, str], set[str]]) -> None:
    """Same as write_paired_view but with var_name as additional grouping key.
    GROUP rows for paired groups (label '+' or '.x') or 2+ members."""
    fieldnames = (["kind", "sign_group", "pdd_code", "baseline", "var_name"] +
                  FLAT_METRIC_COLUMNS + [
                      "violations_by_class_step_total_json",
                      "violations_by_class_event_total_json",
                      "in_zone_by_class_step_total_json",
                      "in_zone_violation_rate_by_class_json",
                  ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for (g, b, v) in sorted(agg_pgbv.keys(),
                                  key=lambda kk: (kk[0], baseline_sort_key(kk[1]), kk[2])):
            sign_members = sorted(members_pgbv.get((g, b, v), set()))
            emit_group = _is_paired_group(g) or len(sign_members) > 1
            if emit_group:
                grp_metrics = agg_pgbv[(g, b, v)]
                row = {"kind": "GROUP", "sign_group": g, "pdd_code": "",
                       "baseline": b, "var_name": v}
                row.update(_flat_row(grp_metrics))
                w.writerow(row)
            for s in sign_members:
                m = agg_pgpbv.get((g, s, b, v))
                if m is None:
                    continue
                row = {"kind": "SIGN", "sign_group": g, "pdd_code": s,
                       "baseline": b, "var_name": v}
                row.update(_flat_row(m))
                w.writerow(row)


def _grouped_sort_key(key: tuple, group_keys: list[str]) -> tuple:
    """Sort tuple where any 'baseline' position uses BASELINE_ORDER."""
    out = []
    for col, val in zip(group_keys, key):
        if col == "baseline":
            out.append(baseline_sort_key(val))
        else:
            out.append(val)
    return tuple(out)


def write_grouped_csv(path: Path, group_keys: list[str],
                       grouped: dict[tuple, dict]) -> None:
    fieldnames = group_keys + FLAT_METRIC_COLUMNS + [
        "violations_by_class_step_total_json",
        "violations_by_class_event_total_json",
        "in_zone_by_class_step_total_json",
        "in_zone_violation_rate_by_class_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for key in sorted(grouped.keys(),
                          key=lambda k: _grouped_sort_key(k, group_keys)):
            metrics = grouped[key]
            row = dict(zip(group_keys, key))
            row.update(_flat_row(metrics))
            w.writerow(row)


# ---------------------------------------------------------------------------
# Legacy JSON emitters
# ---------------------------------------------------------------------------
LEGACY_CUMULATIVE_FIELDS = [
    "n", "n_in_zone", "success_rate", "dest_rate",
    "sign_compliance_sr", "sign_compliance_x",
    "traffic_light_sr", "crosswalk_sr",
    "avg_driving_score", "avg_smoothness",
    "avg_steps", "avg_route_length_m", "avg_distance_travelled_m",
    "avg_route_completion", "dest_x_comp_sr", "dest_x_comp_x",
]


def _emit_legacy_per_baseline_block(metrics: dict) -> dict:
    """Block in the schema expected by generate_cumulative_markdown_report.py /
    generate_category_aggregation_report.py.

    `avg_efficiency` is renamed from our internal `avg_driving_efficiency`.
    Counts (n, n_in_zone) preserved as ints; rates and averages coerced to 0.0
    when None — legacy reports format with `:.3f` and don't handle None.
    """
    int_fields = {"n", "n_in_zone"}
    out = {}
    for f in LEGACY_CUMULATIVE_FIELDS:
        v = metrics.get(f)
        if f in int_fields:
            out[f] = int(v or 0)
        else:
            out[f] = 0.0 if v is None else float(v)
    eff = metrics.get("avg_driving_efficiency")
    out["avg_efficiency"] = 0.0 if eff is None else float(eff)
    return out


LEGACY_2NODE_FIELDS = [
    "success_rate", "crash_rate", "out_of_road_rate",
    "avg_total_reward", "avg_steps", "avg_route_completion_pct",
    "avg_driving_score", "avg_driving_efficiency", "avg_smoothness",
    "avg_min_ttc_sec", "avg_distance_travelled_m",
    "avg_hard_brake_count", "avg_hard_accel_count",
    "total_violation_steps", "total_violation_events",
    "violations_by_class_step_total", "violations_by_class_event_total",
    "total_in_zone_steps", "in_zone_by_class_step_total",
    "in_zone_violation_rate_by_class",
]


def _emit_2node_block(metrics: dict) -> dict:
    """Block in the schema expected by merge_and_report_2node.py
    (cumulative_through_latest / per_var entries). Coerce None to numeric
    defaults so the markdown formatter (`:.1f`/`:.3f`) doesn't crash."""
    n = int(metrics.get("n") or 0)
    arr = metrics.get("dest_rate")
    out = {
        "n_episodes_total": n,
        "n_episodes_ok": n,
        "n_failed": 0,
        "arrival_rate": 0.0 if arr is None else float(arr),
    }
    for f in LEGACY_2NODE_FIELDS:
        v = metrics.get(f)
        if isinstance(v, dict):
            out[f] = v
        elif v is None:
            out[f] = 0
        else:
            out[f] = v
    return out


def write_legacy_cumulative_json(out_path: Path,
                                   per_baseline: dict[str, dict],
                                   per_sign_baseline: dict[tuple[str, str], dict],
                                   chunks: list[str],
                                   per_signgroup_baseline: dict[tuple[str, str], dict]
                                       | None = None,
                                   members_pgb: dict[tuple[str, str], set[str]]
                                       | None = None) -> None:
    """Schema for generate_cumulative_markdown_report.py + category report.

    Per-sign tables get individual pdd_codes for ALL signs plus group keys
    (e.g. "2.1+2.2", "5.12.x") for paired groups — paired-zone scenes from
    the manifest, or END-of-zone variants with 2+ members. Pure singletons
    (one pdd_code, sign_group == pdd_code) are skipped to avoid duplicating
    the individual sign row.
    """
    per_sign: dict[str, dict[str, dict]] = defaultdict(dict)
    for (sign, baseline), m in per_sign_baseline.items():
        per_sign[baseline][sign] = _emit_legacy_per_baseline_block(m)
    if per_signgroup_baseline and members_pgb is not None:
        for (group, baseline), m in per_signgroup_baseline.items():
            members = members_pgb.get((group, baseline), set())
            # Emit GROUP block when paired (label '+' or '.x') or 2+ members
            if not (_is_paired_group(group) or len(members) > 1):
                continue
            per_sign[baseline][group] = _emit_legacy_per_baseline_block(m)

    out = {
        "chunks": chunks,
        "per_baseline": {b: _emit_legacy_per_baseline_block(m)
                          for b, m in sorted(per_baseline.items())},
        "per_sign": {b: dict(sorted(s.items()))
                      for b, s in sorted(per_sign.items())},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                         encoding="utf-8")


def write_2node_cumulative_json(out_path: Path,
                                  per_baseline: dict[str, dict],
                                  per_baseline_var: dict[tuple[str, str], dict],
                                  vars_processed: list[str]) -> None:
    """Schema for merge_and_report_2node.py."""
    per_var_block: dict[str, dict] = {v: {} for v in vars_processed}
    for (baseline, var_name), m in per_baseline_var.items():
        per_var_block.setdefault(var_name, {})[baseline] = _emit_2node_block(m)
    out = {
        "node": "merged_from_csv",
        "vars_processed": vars_processed,
        "n_vars_processed": len(vars_processed),
        "cumulative_through_latest": {b: _emit_2node_block(m)
                                       for b, m in sorted(per_baseline.items())},
        "per_var": {v: dict(sorted(b.items())) for v, b in sorted(per_var_block.items())},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                         encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="Path to metrics_per_episode.csv (output of build_episode_metrics_csv.py)")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory (will create aggregations/ and reports/ subdirs)")
    ap.add_argument("--beta", type=float, default=BETA_DEFAULT)
    ap.add_argument("--horizon", type=int, default=HORIZON_DEFAULT)
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not csv_path.exists():
        print(f"ERROR: csv not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    rows = load_episode_csv(csv_path)
    print(f"[load] {len(rows)} episode rows from {csv_path}")
    if not rows:
        print("ERROR: no rows", file=sys.stderr)
        sys.exit(2)

    # Group rows. sign_group is row-aware: paired-zone scenes get "<start>+<end>"
    # group key, standalone END-variants get "<major>.<minor>.x", everything
    # else is individual.
    by_baseline_var: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_sign_baseline_var: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    by_baseline: dict[str, list[dict]] = defaultdict(list)
    by_sign_baseline: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_signgroup_baseline: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_signgroup_baseline_var: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    # (sign_group, pdd_code, baseline) — needed for paired_view SIGN rows: the
    # per-sign breakdown WITHIN a paired group (otherwise mixes with non-paired
    # rows of the same pdd_code).
    by_sg_sign_baseline: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    by_sg_sign_baseline_var: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    # Members: which pdd_codes are inside each (sign_group, baseline) — used
    # to decide if a group has 2+ members and to drive paired_view rows.
    members_pgb: dict[tuple[str, str], set[str]] = defaultdict(set)
    members_pgbv: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for r in rows:
        b = r["baseline"]; v = r["var_name"]; s = r["pdd_code"] or "_unknown"
        g = sign_group(r)
        by_baseline_var[(b, v)].append(r)
        by_sign_baseline_var[(s, b, v)].append(r)
        by_baseline[b].append(r)
        by_sign_baseline[(s, b)].append(r)
        by_signgroup_baseline[(g, b)].append(r)
        by_signgroup_baseline_var[(g, b, v)].append(r)
        by_sg_sign_baseline[(g, s, b)].append(r)
        by_sg_sign_baseline_var[(g, s, b, v)].append(r)
        members_pgb[(g, b)].add(s)
        members_pgbv[(g, b, v)].add(s)

    # Aggregate
    print("[aggregate] computing 6 slices (4 base + 2 paired-group)...")
    agg_pbv = {k: aggregate(rs, args.beta, args.horizon) for k, rs in by_baseline_var.items()}
    agg_psbv = {k: aggregate(rs, args.beta, args.horizon) for k, rs in by_sign_baseline_var.items()}
    agg_pb = {k: aggregate(rs, args.beta, args.horizon) for k, rs in by_baseline.items()}
    agg_psb = {k: aggregate(rs, args.beta, args.horizon) for k, rs in by_sign_baseline.items()}
    agg_pgb = {k: aggregate(rs, args.beta, args.horizon) for k, rs in by_signgroup_baseline.items()}
    agg_pgbv = {k: aggregate(rs, args.beta, args.horizon) for k, rs in by_signgroup_baseline_var.items()}
    agg_pgpb = {k: aggregate(rs, args.beta, args.horizon) for k, rs in by_sg_sign_baseline.items()}
    agg_pgpbv = {k: aggregate(rs, args.beta, args.horizon) for k, rs in by_sg_sign_baseline_var.items()}

    # Write flat CSVs
    aggregations_dir = out_dir / "aggregations"
    write_grouped_csv(aggregations_dir / "agg_per_baseline_var.csv",
                       ["baseline", "var_name"], agg_pbv)
    write_grouped_csv(aggregations_dir / "agg_per_sign_baseline_var.csv",
                       ["pdd_code", "baseline", "var_name"], agg_psbv)
    write_grouped_csv(aggregations_dir / "agg_per_baseline.csv",
                       ["baseline"], {(b,): m for b, m in agg_pb.items()})
    write_grouped_csv(aggregations_dir / "agg_per_sign_baseline.csv",
                       ["pdd_code", "baseline"], agg_psb)
    write_grouped_csv(aggregations_dir / "agg_per_signgroup_baseline.csv",
                       ["sign_group", "baseline"], agg_pgb)
    write_grouped_csv(aggregations_dir / "agg_per_signgroup_baseline_var.csv",
                       ["sign_group", "baseline", "var_name"], agg_pgbv)
    print(f"[write] {aggregations_dir}/agg_per_baseline_var.csv  ({len(agg_pbv)} rows)")
    print(f"[write] {aggregations_dir}/agg_per_sign_baseline_var.csv  ({len(agg_psbv)} rows)")
    print(f"[write] {aggregations_dir}/agg_per_baseline.csv  ({len(agg_pb)} rows)")
    print(f"[write] {aggregations_dir}/agg_per_sign_baseline.csv  ({len(agg_psb)} rows)")
    print(f"[write] {aggregations_dir}/agg_per_signgroup_baseline.csv  ({len(agg_pgb)} rows)")
    print(f"[write] {aggregations_dir}/agg_per_signgroup_baseline_var.csv  ({len(agg_pgbv)} rows)")

    # Paired view: interleave group totals + member-sign breakdowns. Member
    # SIGN rows use within-group aggregations (agg_pgpb) so paired-zone metrics
    # don't mix with non-paired same-pdd_code rows.
    write_paired_view(aggregations_dir / "paired_view_baseline.csv",
                       agg_pgb, agg_pgpb, members_pgb)
    write_paired_view_per_var(aggregations_dir / "paired_view_baseline_var.csv",
                                agg_pgbv, agg_pgpbv, members_pgbv)
    print(f"[write] {aggregations_dir}/paired_view_baseline.csv")
    print(f"[write] {aggregations_dir}/paired_view_baseline_var.csv")

    # Write legacy JSONs
    reports_dir = out_dir / "reports"
    vars_processed = sorted({r["var_name"] for r in rows})
    cumulative_json = reports_dir / "cumulative.json"
    write_legacy_cumulative_json(cumulative_json, agg_pb, agg_psb,
                                   chunks=vars_processed,
                                   per_signgroup_baseline=agg_pgb,
                                   members_pgb=members_pgb)
    print(f"[write] {cumulative_json}  (per_baseline={len(agg_pb)}, "
          f"per_sign baselines={len({b for _, b in agg_psb})})")

    cumulative_2node = reports_dir / "cumulative_2node.json"
    write_2node_cumulative_json(cumulative_2node, agg_pb, agg_pbv, vars_processed)
    print(f"[write] {cumulative_2node}")

    print()
    print("Next steps:")
    print(f"  python3 {SCRIPT_DIR}/generate_cumulative_markdown_report.py \\")
    print(f"      --run-root {out_dir} --cumulative {cumulative_json}")
    print(f"  python3 {SCRIPT_DIR}/generate_category_aggregation_report.py \\")
    print(f"      --run-root {out_dir} --cumulative {cumulative_json} --preset task_oriented")
    print(f"  python3 {SCRIPT_DIR}/merge_and_report_2node.py \\")
    print(f"      --node1 {cumulative_2node} --node2 {cumulative_2node} --out-dir {out_dir}")


if __name__ == "__main__":
    main()
