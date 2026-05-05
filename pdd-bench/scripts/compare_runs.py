#!/usr/bin/env python3
"""Compare two sets of replay_mini_new.py runs (baseline vs updated).

Reads summary.json files from --baseline and --updated result dirs
(produced by run_ab_comparison.sh) and prints a side-by-side table with
deltas for: dest_rate, avg_violations, crash_rate, out_of_road_rate,
oracle_dest_rate.

Usage:
    python3 pdd-bench/scripts/compare_runs.py \
        --baseline pdd-bench/ab_results/<ts>/baseline \
        --updated  pdd-bench/ab_results/<ts>/updated \
        --out      pdd-bench/ab_results/<ts>/comparison.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


_METRICS = [
    ("dest_rate",       "dest%",   100, ".1f"),
    ("avg_violations",  "viol",      1, ".2f"),
    ("crash_rate",      "crash%",  100, ".1f"),
    ("out_of_road_rate","oor%",    100, ".1f"),
]


def _load_results(root: Path) -> dict[tuple[str, str], dict]:
    """Return {(sign, policy): summary_dict} from a results dir."""
    out: dict = {}
    for d in sorted(root.glob("mini_new_*")):
        summary_p = d / "summary.json"
        if not summary_p.exists():
            continue
        try:
            s = json.loads(summary_p.read_text())
        except Exception:
            continue
        sign = s.get("sign_code", "?")
        policy = s.get("policy", "?")
        out[(sign, policy)] = s
    return out


def _get(s: dict, key: str, scale: float) -> float | None:
    # per_scene_oracle has the cleanest single-number; fall back to top-level
    oracle = s.get("per_scene_oracle", {})
    val = oracle.get(key, s.get(key))
    if val is None:
        return None
    return float(val) * scale


def compare(baseline: dict, updated: dict) -> list[dict]:
    all_keys = sorted(set(baseline) | set(updated))
    rows = []
    for key in all_keys:
        sign, policy = key
        b = baseline.get(key)
        u = updated.get(key)
        row = {"sign": sign, "policy": policy}
        for attr, label, scale, fmt in _METRICS:
            bv = _get(b, attr, scale) if b else None
            uv = _get(u, attr, scale) if u else None
            row[f"baseline_{attr}"] = bv
            row[f"updated_{attr}"] = uv
            row[f"delta_{attr}"] = (uv - bv) if (uv is not None and bv is not None) else None
        rows.append(row)
    return rows


def _fmt(v, fmt, suffix="") -> str:
    if v is None:
        return "  n/a"
    return f"{v:{fmt}}{suffix}"


def _delta_str(d, fmt) -> str:
    if d is None:
        return "    -"
    sign = "+" if d > 0 else ""
    return f"{sign}{d:{fmt}}"


def print_table(rows: list[dict]) -> None:
    # Header
    w_sign, w_pol = 8, 14
    col_w = 8  # per metric: base / upd / delta

    header1 = f"{'sign':<{w_sign}} {'policy':<{w_pol}}"
    header2 = f"{'':>{w_sign}} {'':>{w_pol}}"
    for _, label, scale, fmt in _METRICS:
        header1 += f"  {'── ' + label + ' ──':>24}"
        header2 += f"  {'base':>{col_w}} {'upd':>{col_w}} {'Δ':>{col_w}}"

    sep = "-" * (w_sign + 1 + w_pol + 2 + len(_METRICS) * 26)
    print()
    print(header1)
    print(header2)
    print(sep)

    prev_sign = None
    for r in rows:
        if r["sign"] != prev_sign and prev_sign is not None:
            print()
        prev_sign = r["sign"]
        line = f"{r['sign']:<{w_sign}} {r['policy']:<{w_pol}}"
        for attr, label, scale, fmt in _METRICS:
            b = _fmt(r.get(f"baseline_{attr}"), fmt)
            u = _fmt(r.get(f"updated_{attr}"), fmt)
            d = _delta_str(r.get(f"delta_{attr}"), fmt)
            line += f"  {b:>{col_w}} {u:>{col_w}} {d:>{col_w}}"
        print(line)

    print(sep)

    # Summary: average delta per metric across all sign×policy combinations
    print("\nMean Δ across all sign×policy combinations:")
    for attr, label, scale, fmt in _METRICS:
        deltas = [r[f"delta_{attr}"] for r in rows if r[f"delta_{attr}"] is not None]
        if deltas:
            mean_d = sum(deltas) / len(deltas)
            sign_str = "+" if mean_d > 0 else ""
            print(f"  {label:<12} {sign_str}{mean_d:{fmt}}")

    # Per-sign summary
    print("\nPer-sign oracle dest_rate (best policy per scene is in per_scene_oracle):")
    signs = sorted({r["sign"] for r in rows})
    for sign in signs:
        sign_rows = [r for r in rows if r["sign"] == sign]
        # aggregate: mean across policies for per_scene_oracle dest_rate
        b_vals = [r["baseline_dest_rate"] for r in sign_rows if r["baseline_dest_rate"] is not None]
        u_vals = [r["updated_dest_rate"]  for r in sign_rows if r["updated_dest_rate"]  is not None]
        b_mean = sum(b_vals) / len(b_vals) if b_vals else None
        u_mean = sum(u_vals) / len(u_vals) if u_vals else None
        delta  = (u_mean - b_mean) if (b_mean is not None and u_mean is not None) else None
        b_s = _fmt(b_mean, ".1f", "%")
        u_s = _fmt(u_mean, ".1f", "%")
        d_s = _delta_str(delta, ".1f") + "%" if delta is not None else "   -"
        print(f"  {sign:<8}  base={b_s}  upd={u_s}  Δ={d_s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Path to baseline results dir")
    ap.add_argument("--updated",  required=True, help="Path to updated results dir")
    ap.add_argument("--out", default=None, help="Write JSON comparison to this file")
    args = ap.parse_args()

    baseline = _load_results(Path(args.baseline))
    updated  = _load_results(Path(args.updated))

    if not baseline:
        print(f"[warn] no baseline summaries found in {args.baseline}")
    if not updated:
        print(f"[warn] no updated summaries found in {args.updated}")

    rows = compare(baseline, updated)
    print_table(rows)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nComparison JSON: {args.out}")


if __name__ == "__main__":
    main()
