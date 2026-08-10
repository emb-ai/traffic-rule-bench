#!/usr/bin/env python3
"""Compute expert-selection-style metrics from consolidated <baseline>_replays.jsonl.

Mirrors select_experts.py logic:
  - recompute_dest (rules 1.1/1.2 — no_entry compliance + horizon timeout)
  - passes_filter (Strategy-F-new): not crashed AND not out_of_road AND
                   compliance(target_sign_class) == True AND
                   dest_rate_new == 1 AND
                   final_step >= MIN_FINAL_STEP

Output matches the "Filter rates per policy_variant" section of
select_experts.print_summary:
  total | pass | rate | compl | arr | arr_n | crash | oor

Usage:
  python3 aggregate_expert_filter_metrics.py \\
      --runs-var benchmark_2node_eval/runs/var_0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Mirror constants from select_experts.py
NO_ENTRY_SIGNS = {"3.1", "3.2", "3.18.1", "3.18.2", "3.19"}
HORIZON_DEFAULT = 600
MIN_FINAL_STEP = 30

# PDD code → class name used for compliance / in-zone (must match violation
# emitter). Informational plates 2.1 / 2.3.x never violate — see select_experts.
SIGN_CLASS_MAP = {
    "2.1": "RightHandYieldSign", "2.2": "EndMainRoadSmartSign",
    "2.3": "YieldSign",
    "2.3.1": "YieldSign", "2.3.2": "YieldSign", "2.3.3": "YieldSign",
    "2.4": "YieldSign", "2.5": "StopSign",
    "3.1": "NoEntrySign", "3.2": "NoTrafficSign",
    "3.18.1": "NoRightTurnSign", "3.18.2": "NoLeftTurnSign", "3.19": "NoUTurnSign",
    "3.20": "NoOvertakingSign", "3.21": "EndOfNoOvertakingSign",
    "3.24": "SpeedLimitSign", "3.25": "EndOfSpeedLimitSign",
    "3.27": "NoStoppingAllowedSign", "3.31": "EndOfAllRestrictionsSign",
    "4.1.1": "LaneAllowedDirectionSign4_1_1", "4.1.2": "LaneAllowedDirectionSign4_1_2",
    "4.1.3": "LaneAllowedDirectionSign4_1_3", "4.1.4": "LaneAllowedDirectionSign4_1_4",
    "4.1.5": "LaneAllowedDirectionSign4_1_5", "4.1.6": "LaneAllowedDirectionSign4_1_6",
    "4.2.1": "DetourRightSign", "4.2.2": "DetourLeftSign", "4.2.3": "DetourEitherSign",
    "4.6": "MinimumSpeedLimitSign",
    "5.3": "OnlyAutoSign", "5.4": "EndOfOnlyAutoSign",
    "5.5": "OneWayEntrySignS", "5.7.1": "OneWayEntrySignR", "5.7.2": "OneWayEntrySignL",
    "5.11.1": "BusLaneRoadSign", "5.11.2": "BikeLaneRoadSign",
    "5.12.1": "EndBusLaneRoadSign", "5.12.2": "EndBikeLaneRoadSign",
    "5.13.1": "ExitToBusLaneSign", "5.13.2": "ExitToBusLaneSignLeft",
    "5.13.3": "ExitToBikeLaneSign", "5.13.4": "ExitToBikeLaneSignLeft",
    "5.14.1": "BusLaneSign", "5.14.2": "BikeLaneSign",
    "5.14.3": "EndBusLaneSign", "5.14.4": "EndBikeLaneSign",
    "5.15.2": "DirectionSign", "5.16": "BusStationSign",
    "5.31": "ZoneSpeedLimitSign", "5.32": "EndOfZoneSpeedLimitSign",
}


def _normalize_pdd(s: str) -> str:
    """slug → dotted: '5_11_1' → '5.11.1'. Pass through if already dotted."""
    if not s:
        return ""
    s = str(s)
    if "." in s:
        return s
    return s.replace("_", ".")


def _pdd_for_row(r: dict) -> str:
    """Extract PDD code from a sidecar replay row."""
    sr = r.get("source_row") or {}
    s = (r.get("pdd_code") or sr.get("pdd_code")
         or sr.get("_sign_code") or sr.get("sign_code"))
    if s:
        return _normalize_pdd(s)
    # Try sign_slug ('5_11_1')
    slug = r.get("sign_slug") or sr.get("_sign_slug")
    if slug:
        return _normalize_pdd(slug)
    # Fallback to internal sign_type (would need INTERNAL_TO_PDD map)
    return ""


def _is_compliant(metrics: dict, target_class: str) -> bool:
    """compliance per target sign class. metrics.violations_by_class_step is
    {ClassName: count}; treat 0 as compliant."""
    vbc = metrics.get("violations_by_class_step") or {}
    return int(vbc.get(target_class, 0) or 0) == 0


def _recompute_dest(r: dict, target_sign: str, target_class: str,
                    horizon: int = HORIZON_DEFAULT) -> bool:
    """Mirror select_experts.recompute_dest."""
    m = r.get("metrics") or {}
    arr_prev = bool(m.get("arrived_dest"))
    if target_sign in NO_ENTRY_SIGNS and _is_compliant(m, target_class):
        return True
    fs = int(m.get("final_step") or 0)
    if fs == horizon and not arr_prev:
        return True
    return arr_prev


def _passes_filter(r: dict, target_sign: str, target_class: str,
                   horizon: int = HORIZON_DEFAULT,
                   min_final_step: int = MIN_FINAL_STEP) -> bool:
    """Strategy-F-new (mirrors select_experts.passes_filter).
    Sidecar `metrics` is the source of truth for outcome flags."""
    m = r.get("metrics") or {}
    if m.get("crashed") or m.get("out_of_road"):
        return False
    if not _is_compliant(m, target_class):
        return False
    if min_final_step > 0:
        fs = int(m.get("final_step") or 0)
        if fs < min_final_step:
            return False
    return _recompute_dest(r, target_sign, target_class, horizon=horizon)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-var", required=True,
                    help="Dir with <baseline>_replays.jsonl files.")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: <runs-var>/expert_filter_metrics.json)")
    ap.add_argument("--horizon", type=int, default=HORIZON_DEFAULT)
    ap.add_argument("--min-final-step", type=int, default=MIN_FINAL_STEP)
    args = ap.parse_args()

    runs_var = Path(args.runs_var).resolve()
    if not runs_var.is_dir():
        print(f"ERROR: not a dir: {runs_var}", file=sys.stderr)
        sys.exit(2)

    jsonl_files = sorted(runs_var.glob("*_replays.jsonl"))
    if not jsonl_files:
        print(f"ERROR: no *_replays.jsonl found", file=sys.stderr)
        sys.exit(2)

    # pol → counter dict
    pol_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "pass": 0, "compl": 0,
                 "arr_prev": 0, "arr_new": 0, "crash": 0, "oor": 0,
                 "no_target_class": 0, "unknown_pdd": 0})
    # per-sign coverage
    sign_total: dict[str, set] = defaultdict(set)
    sign_passing: dict[str, set] = defaultdict(set)

    for fp in jsonl_files:
        baseline = fp.stem.removesuffix("_replays")
        with fp.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                pdd = _pdd_for_row(r)
                target_class = SIGN_CLASS_MAP.get(pdd)
                m = r.get("metrics") or {}
                scene_uid = r.get("scene_uid") or r.get("scene_id")
                s = pol_stats[baseline]
                s["total"] += 1

                if not pdd:
                    s["unknown_pdd"] += 1
                if not target_class:
                    s["no_target_class"] += 1

                # Coverage tracking
                if pdd:
                    sign_total[pdd].add(scene_uid)

                # Filter pass
                ok = (target_class is not None
                      and _passes_filter(r, pdd, target_class,
                                          horizon=args.horizon,
                                          min_final_step=args.min_final_step))
                if ok:
                    s["pass"] += 1
                    if pdd:
                        sign_passing[pdd].add(scene_uid)

                # Sub-counters
                if target_class and _is_compliant(m, target_class):
                    s["compl"] += 1
                if m.get("arrived_dest"):
                    s["arr_prev"] += 1
                if target_class and _recompute_dest(r, pdd, target_class, horizon=args.horizon):
                    s["arr_new"] += 1
                if m.get("crashed"):
                    s["crash"] += 1
                if m.get("out_of_road"):
                    s["oor"] += 1

    # ---- Print: Filter rates per policy_variant ----
    print("\n" + "=" * 100)
    print("Filter rates per policy_variant (compliance per target_sign_class)")
    print("=" * 100)
    print(f"{'policy_variant':<42} {'total':>6} {'pass':>5} {'rate':>6} "
          f"{'compl':>6} {'arr':>5} {'arr_n':>6} {'crash':>6} {'oor':>5} {'noPDD':>6}")
    print("-" * 100)
    for pol in sorted(pol_stats):
        s = pol_stats[pol]
        rate = s["pass"] / max(1, s["total"])
        print(f"{pol:<42} {s['total']:>6} {s['pass']:>5} {rate:>6.1%} "
              f"{s['compl']:>6} {s['arr_prev']:>5} {s['arr_new']:>6} "
              f"{s['crash']:>6} {s['oor']:>5} {s['unknown_pdd']:>6}")
    print("\nlegend: arr=arrived_dest, arr_n=dest_rate_new (post 1.1/1.2),"
          " noPDD=rows with empty pdd_code (excluded from compliance check)")

    # ---- Print: Coverage per sign ----
    print("\n" + "=" * 60)
    print("Coverage per sign (passing scenes / total scenes)")
    print("=" * 60)
    print(f"{'sign':<10} {'scenes':>7} {'passing':>8} {'cov%':>7}")
    print("-" * 50)
    for sign in sorted(sign_total):
        st = len(sign_total[sign])
        sp = len(sign_passing[sign])
        cov = sp / st if st else 0
        print(f"{sign:<10} {st:>7} {sp:>8} {cov:>6.0%}")

    # ---- Save JSON ----
    out_path = Path(args.out).resolve() if args.out else (runs_var / "expert_filter_metrics.json")
    out = {
        "horizon": args.horizon,
        "min_final_step": args.min_final_step,
        "per_baseline": {k: dict(v) for k, v in sorted(pol_stats.items())},
        "per_sign_coverage": {
            s: {"scenes": len(sign_total[s]),
                "passing": len(sign_passing[s]),
                "cov": (len(sign_passing[s]) / len(sign_total[s])) if sign_total[s] else 0.0}
            for s in sorted(sign_total)
        },
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[write] {out_path}")

    # ---- Save markdown ----
    md_path = out_path.with_suffix(".md")
    md = _make_markdown(pol_stats, sign_total, sign_passing,
                        horizon=args.horizon, min_final_step=args.min_final_step,
                        runs_var=runs_var)
    md_path.write_text(md, encoding="utf-8")
    print(f"[write] {md_path}")


def _make_markdown(pol_stats: dict, sign_total: dict, sign_passing: dict,
                   horizon: int, min_final_step: int, runs_var: Path) -> str:
    lines: list[str] = []
    lines.append("# Expert Filter Metrics Report")
    lines.append("")
    lines.append(f"**Source**: `{runs_var}/<baseline>_replays.jsonl`")
    lines.append(f"**Strategy**: F-new (mirror of `select_experts.passes_filter`)")
    lines.append(f"**Horizon**: {horizon}  |  **min_final_step**: {min_final_step}")
    lines.append("")

    # ---- Filter rates per policy_variant ----
    lines.append("## Filter rates per baseline")
    lines.append("")
    lines.append("Pass = `!crashed AND !out_of_road AND compliance(target_class) AND "
                 "final_step ≥ min_final_step AND dest_rate_new`")
    lines.append("")
    lines.append("`dest_rate_new` = `arrived_dest` OR (no_entry sign AND compliance) OR "
                 "(`final_step == horizon` AND not arrived).")
    lines.append("")
    lines.append("| Baseline | Total | Pass | **Rate** | Compliance | Arrived | Arrived_new | Crash | OOR | NoPDD |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for pol in sorted(pol_stats):
        s = pol_stats[pol]
        tot = s["total"]
        rate = s["pass"] / max(1, tot)
        lines.append(
            f"| `{pol}` | {tot} | {s['pass']} | **{100*rate:.1f}%** | "
            f"{s['compl']} ({100*s['compl']/max(1,tot):.1f}%) | "
            f"{s['arr_prev']} ({100*s['arr_prev']/max(1,tot):.1f}%) | "
            f"{s['arr_new']} ({100*s['arr_new']/max(1,tot):.1f}%) | "
            f"{s['crash']} ({100*s['crash']/max(1,tot):.1f}%) | "
            f"{s['oor']} ({100*s['oor']/max(1,tot):.1f}%) | "
            f"{s['unknown_pdd']} |"
        )
    lines.append("")

    # ---- Coverage per sign ----
    lines.append("## Coverage per sign")
    lines.append("")
    lines.append("How many scenes pass the filter (by at least one baseline) out of the total.")
    lines.append("")
    lines.append("| Sign | Scenes | Passing | Coverage |")
    lines.append("|---|---:|---:|---:|")
    for sign in sorted(sign_total):
        st = len(sign_total[sign])
        sp = len(sign_passing[sign])
        cov = sp / st if st else 0
        lines.append(f"| `{sign}` | {st} | {sp} | **{100*cov:.0f}%** |")
    lines.append("")

    # ---- Per-baseline ranking by pass rate ----
    lines.append("## Ranking by Pass Rate")
    lines.append("")
    ranked = sorted(pol_stats.items(),
                    key=lambda kv: -(kv[1]["pass"] / max(1, kv[1]["total"])))
    lines.append("| # | Baseline | Pass Rate |")
    lines.append("|---:|---|---:|")
    for i, (pol, s) in enumerate(ranked, 1):
        rate = s["pass"] / max(1, s["total"])
        lines.append(f"| {i} | `{pol}` | **{100*rate:.1f}%** |")
    lines.append("")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
