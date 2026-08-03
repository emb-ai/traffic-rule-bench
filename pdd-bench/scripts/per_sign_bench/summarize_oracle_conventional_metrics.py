#!/usr/bin/env python3
"""Conventional metrics (Eff / Comfort / RC / Collision) for oracle_rule.

Uses the same episode-level definitions as ``summarize_ready_sign_test_metrics.py``:
  Efficiency       — driving_efficiency
  Comfort          — comfort (fallback: frame_smooth_ratio)
  Route Completion — 1 if arrived_dest else distance/route_length
  Collision        — crashed rate

Sources:
  * colleague detour  — smirnova/.../detour_v1/eval_test20/metrics_per_episode_oracle.csv
  * colleague speed   — smirnova/.../eval_fast/metrics_test20_oracle.csv
  * local ready signs — per_sign_bench/.../metrics_per_episode*_oracle.csv

Unlike the SCR summarizer, speed signs use ALL valid oracle episodes
(no A6 filter) — conventional metrics are not conditioned on compliance.

Examples:
  python summarize_oracle_conventional_metrics.py
  python summarize_oracle_conventional_metrics.py --list
  python summarize_oracle_conventional_metrics.py --only 4.3,5.19
  python summarize_oracle_conventional_metrics.py --out benchmark_output/oracle_conventional
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

PER_SIGN = Path(__file__).resolve().parent
SMIRNOVA = Path("/home/jovyan/shares/SR006.nfs2/smirnova/traffic-rule-bench/pdd-bench")

# Reuse RC / bool / float helpers from the ready-sign summarizer.
from summarize_ready_sign_test_metrics import (  # noqa: E402
    _mean,
    _to_bool,
    _to_float,
    route_completion,
)


@dataclass(frozen=True)
class OracleJob:
    label: str
    csv_path: Path
    codes: tuple[str, ...]


def _t20_oracle(bench: str, slug: str) -> Path:
    return (
        PER_SIGN / bench / "benchmark_output" / "test_metrics" / "test20_batch"
        / slug / "eval_out" / "metrics_per_episode_oracle.csv"
    )


def _eval_test_oracle(bench: str, *parts: str) -> Path:
    return (
        PER_SIGN / bench / "benchmark_output" / Path(*parts)
        / "metrics_per_episode_oracle.csv"
    )


ORACLE_JOBS: list[OracleJob] = [
    OracleJob("2.1", _t20_oracle("main_sign", "2_1"), ("2.1",)),
    OracleJob(
        "2.3.1-2.3.3",
        _t20_oracle("secondary_sign", "2_3_1_2_3_3"),
        ("2.3.1", "2.3.2", "2.3.3"),
    ),
    OracleJob("2.4", _t20_oracle("yield_sign", "2_4"), ("2.4",)),
    OracleJob("2.5", _t20_oracle("stop_sign", "2_5"), ("2.5",)),
    OracleJob(
        "3.1-3.2",
        _eval_test_oracle("no_entry_signs", "combined", "eval_out_test"),
        ("3.1", "3.2"),
    ),
    OracleJob(
        "3.24",
        SMIRNOVA
        / "benchmark_output_speed/balanced/run_v61_a6/eval_fast"
        / "metrics_test20_oracle.csv",
        ("3.24",),
    ),
    OracleJob(
        "4.2.1-4.2.3",
        SMIRNOVA
        / "benchmark_output/detour_v1/eval_test20/metrics_per_episode_oracle.csv",
        ("4.2.1", "4.2.2", "4.2.3"),
    ),
    OracleJob("4.3", _t20_oracle("roundabout_sign", "4_3"), ("4.3",)),
    OracleJob(
        "4.6",
        SMIRNOVA
        / "benchmark_output_speed/balanced/run_v61_a6/eval_fast"
        / "metrics_test20_oracle.csv",
        ("4.6",),
    ),
    OracleJob(
        "5.7.1-5.7.2",
        _eval_test_oracle("one_way_signs", "combined", "eval_out_test"),
        ("5.7.1", "5.7.2"),
    ),
    OracleJob(
        "5.15.1-5.15.2",
        _t20_oracle("lane_direction_signs", "5_15_1_5_15_2"),
        ("5.15.1", "5.15.2"),
    ),
    OracleJob("5.19", _t20_oracle("crosswalk_sign", "5_19"), ("5.19",)),
    OracleJob(
        "5.21",
        SMIRNOVA
        / "benchmark_output_speed/balanced/run_v61_a6/eval_fast"
        / "metrics_test20_oracle.csv",
        ("5.21",),
    ),
    OracleJob(
        "5.31",
        SMIRNOVA
        / "benchmark_output_speed/balanced/run_v61_a6/eval_fast"
        / "metrics_test20_oracle.csv",
        ("5.31",),
    ),
]

METRIC_KEYS = ("efficiency", "comfort", "route_completion", "collision")

_CSV_CACHE: dict[Path, list[dict]] = {}


def _norm_code(code: str) -> str:
    """Normalize CSV pdd_code quirks (e.g. floaty '4.60' → '4.6')."""
    code = (code or "").strip()
    if code in {"4.60", "4.6.0"}:
        return "4.6"
    return code


def _iter_csv(path: Path) -> list[dict]:
    cached = _CSV_CACHE.get(path)
    if cached is not None:
        return cached
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    _CSV_CACHE[path] = rows
    return rows


def load_oracle_episodes(job: OracleJob) -> list[dict]:
    if not job.csv_path.exists():
        raise FileNotFoundError(f"{job.label}: missing {job.csv_path}")
    want = set(job.codes)
    out: list[dict] = []
    for r in _iter_csv(job.csv_path):
        if (r.get("baseline") or "").strip() != "oracle_rule":
            continue
        if r.get("valid") not in ("", "True"):
            continue
        code = _norm_code(r.get("pdd_code") or "")
        if want and code and code not in want:
            continue
        comfort = _to_float(r.get("comfort", ""), None)
        if comfort is None:
            comfort = _to_float(r.get("frame_smooth_ratio", ""), None)
        out.append({
            "sign": job.label,
            "pdd_code": code or job.label,
            "efficiency": _to_float(r.get("driving_efficiency", ""), None),
            "comfort": comfort,
            "route_completion": route_completion(r),
            "collision": 1.0 if _to_bool(r.get("crashed", "")) else 0.0,
        })
    return out


def aggregate(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "efficiency": _mean(r["efficiency"] for r in rows),
        "comfort": _mean(r["comfort"] for r in rows),
        "route_completion": _mean(r["route_completion"] for r in rows),
        "collision": _mean(r["collision"] for r in rows),
    }


def macro_average(per_sign: dict[str, dict]) -> dict:
    items = [m for m in per_sign.values() if m.get("n", 0) > 0]
    if not items:
        return {"n": 0}
    out = {"n": sum(m["n"] for m in items)}
    for k in METRIC_KEYS:
        out[k] = _mean(m.get(k) for m in items)
    return out


def fmt(v: Optional[float], kind: str) -> str:
    if v is None:
        return ""
    if kind == "eff":
        return f"{v:.2f}"
    if kind == "comf":
        return f"{v:.3f}"
    return f"{100.0 * v:.1f}"


def write_csv(path: Path, rows: list[dict], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = keys + ["n", *METRIC_KEYS]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in keys}
            out["n"] = row.get("n", 0)
            for k in METRIC_KEYS:
                v = row.get(k)
                out[k] = "" if v is None else round(float(v), 6)
            w.writerow(out)


def md_table(title: str, rows: list[dict], key_cols: list[tuple[str, str]]) -> str:
    headers = [h for _, h in key_cols] + ["n", "Eff.", "Comfort", "RC (%)", "Coll. (%)"]
    lines = [f"## {title}", "", "| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    kinds = {"efficiency": "eff", "comfort": "comf",
             "route_completion": "pct", "collision": "pct"}
    for row in rows:
        cells = [str(row.get(k, "")) for k, _ in key_cols]
        cells.append(str(row.get("n", 0)))
        for mk in METRIC_KEYS:
            cells.append(fmt(row.get(mk), kinds[mk]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--only", default=None,
                    help="Comma-separated sign labels (default: all)")
    ap.add_argument(
        "--out",
        type=Path,
        default=PER_SIGN / "benchmark_output" / "oracle_conventional",
        help="Output directory",
    )
    ap.add_argument("--list", action="store_true",
                    help="List jobs / CSV readiness and exit")
    ap.add_argument("--print-md", action="store_true")
    args = ap.parse_args()

    jobs = ORACLE_JOBS
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        jobs = [j for j in jobs if j.label in want]
        missing = want - {j.label for j in jobs}
        if missing:
            print(f"ERROR: unknown labels: {sorted(missing)}", file=sys.stderr)
            sys.exit(2)

    if args.list:
        for j in ORACLE_JOBS:
            status = "ok" if j.csv_path.exists() else "MISSING"
            try:
                rel = j.csv_path.relative_to(PER_SIGN)
            except ValueError:
                rel = j.csv_path
            print(f"{j.label:16s}  [{status}]  {rel}")
        return

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    by_sign: dict[str, list[dict]] = {}
    all_rows: list[dict] = []
    skipped: list[str] = []

    for job in jobs:
        try:
            eps = load_oracle_episodes(job)
        except FileNotFoundError as e:
            print(f"[skip] {e}", file=sys.stderr)
            skipped.append(job.label)
            continue
        if not eps:
            print(f"[skip] {job.label}: 0 oracle_rule rows", file=sys.stderr)
            skipped.append(job.label)
            continue
        by_sign[job.label] = eps
        all_rows.extend(eps)
        print(f"[load] {job.label:16s}  n={len(eps):4d}  {job.csv_path.name}")

    if not all_rows:
        print("ERROR: no oracle episodes loaded", file=sys.stderr)
        sys.exit(1)

    per_sign_rows = []
    per_sign_agg: dict[str, dict] = {}
    for label in [j.label for j in jobs if j.label in by_sign]:
        m = aggregate(by_sign[label])
        per_sign_agg[label] = m
        per_sign_rows.append({"sign": label, **m})

    micro = {"avg": "micro", **aggregate(all_rows)}
    macro = {"avg": "macro", **macro_average(per_sign_agg)}

    write_csv(out_dir / "per_sign_oracle.csv", per_sign_rows, ["sign"])
    write_csv(out_dir / "overall_oracle.csv", [micro, macro], ["avg"])

    md = [
        "# Oracle conventional metrics (test set)",
        "",
        f"Baseline: `oracle_rule`. Signs loaded ({len(per_sign_rows)}): "
        + ", ".join(r["sign"] for r in per_sign_rows) + ".",
    ]
    if skipped:
        md.append(f"Skipped: {', '.join(skipped)}.")
    md += [
        "",
        "RC = 1 if arrived else distance_travelled / route_length.",
        "Collision = crash rate. Efficiency / Comfort = episode means.",
        "Detour + speed from smirnova oracle CSVs; other signs from local `*_oracle.csv`.",
        "Speed signs: all valid oracle episodes (no A6 filter).",
        "",
        md_table("Overall", [micro, macro], [("avg", "Avg")]),
        md_table("Per sign", per_sign_rows, [("sign", "Sign")]),
    ]
    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print()
    print(f"Wrote {out_dir}/")
    print(f"  per_sign_oracle.csv   ({len(per_sign_rows)} signs)")
    print(f"  overall_oracle.csv")
    print(f"  summary.md")
    print()
    print(md_table("Overall", [micro, macro], [("avg", "Avg")]))
    if args.print_md:
        print(md_table("Per sign", per_sign_rows, [("sign", "Sign")]))


if __name__ == "__main__":
    main()
