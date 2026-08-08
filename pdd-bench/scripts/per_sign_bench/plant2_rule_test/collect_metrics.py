#!/usr/bin/env python3
"""Collect every cumulative.json under output/ into one table.

Reads both report families of a checkpoint run:
  output/<run>/<label>/eval_out/reports/cumulative.json   per-sign eval_pipeline
  output/<run>/direct/reports/cumulative.json             speed signs + detour

Each file carries per_sign[baseline][pdd_code], so signs bundled in one label
(e.g. 5.7.1-5.7.2) and the four speed signs inside `direct` come out as
separate rows instead of one averaged number.

  python3 collect_metrics.py                       # every run
  python3 collect_metrics.py --runs 'fvexp30_*'    # subset
  python3 collect_metrics.py --metric dest_rate    # pivot another column
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

COLUMNS = [
    ("n", "N"),
    ("n_in_zone", "In-zone"),
    ("dest_rate", "Dest"),
    ("success_rate", "Success"),
    ("sign_compliance_sr", "Sign SR"),
    ("sign_compliance_x", "Sign SR (in-zone)"),
    ("avg_driving_score", "DS"),
    ("avg_efficiency", "Eff"),
    ("avg_smoothness", "Smooth"),
]


def _sign_key(sign: str) -> tuple:
    """Order sign codes numerically: 2.3.1 < 2.4 < 3.24 < 4.2.1 < 5.7.1."""
    parts = []
    for chunk in str(sign).replace("+", ".").split("."):
        parts.append((0, int(chunk)) if chunk.isdigit() else (1, 0, chunk))
    return tuple(parts)


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and abs(v) > 1.5):
        return str(int(v))
    return f"{v:.3f}"


def _pick_baseline(per: dict, wanted: str) -> str | None:
    if wanted in per:
        return wanted
    hits = [k for k in per if k.startswith(wanted) or wanted in k]
    if len(hits) == 1:
        return hits[0]
    return next(iter(per)) if len(per) == 1 else None


def collect(run_dir: Path, baseline: str) -> list[dict]:
    rows: list[dict] = []
    for cum_path in sorted(run_dir.rglob("reports/cumulative.json")):
        rel = cum_path.relative_to(run_dir).parts
        source = "direct" if rel[0] == "direct" else rel[0]
        try:
            data = json.loads(cum_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] unreadable {cum_path}: {exc}")
            continue
        per_sign = data.get("per_sign") or {}
        key = _pick_baseline(per_sign, baseline)
        if key is None:
            per_base = data.get("per_baseline") or {}
            key = _pick_baseline(per_base, baseline)
            if key is None:
                print(f"[warn] no baseline {baseline!r} in {cum_path}")
                continue
            signs = {"(all)": per_base[key]}
        else:
            signs = per_sign[key]
        for sign, m in sorted(signs.items()):
            row = {"run": run_dir.name, "source": source, "sign": sign,
                   "baseline": key}
            row.update({f: m.get(f) for f, _ in COLUMNS})
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="*", help="glob over output/ (default: all)")
    ap.add_argument("--baseline", default="plant2_default")
    ap.add_argument("--metric", default="sign_compliance_sr",
                    help="column for the checkpoint-vs-sign pivot")
    ap.add_argument("--out-dir", type=Path, default=HERE / "output" / "_all_metrics")
    args = ap.parse_args()

    rows: list[dict] = []
    for run_dir in sorted((HERE / "output").glob(args.runs)):
        if run_dir.is_dir():
            rows.extend(collect(run_dir, args.baseline))
    if not rows:
        raise SystemExit("no cumulative.json found — run refinalize.sh first")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "all_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    runs = sorted({r["run"] for r in rows})
    signs = sorted({r["sign"] for r in rows}, key=_sign_key)
    by_run: dict[str, dict[str, dict]] = {run: {} for run in runs}
    for r in rows:
        by_run[r["run"]][r["sign"]] = r

    md = ["# All metrics", "", f"Baseline: `{args.baseline}`", ""]
    for run in runs:
        md += [f"## {run}", "",
               "| Sign | Source | " + " | ".join(t for _, t in COLUMNS) + " |",
               "|---|---|" + "---:|" * len(COLUMNS)]
        for sign in signs:
            r = by_run[run].get(sign)
            if r is None:
                md.append(f"| {sign} | — |" + " — |" * len(COLUMNS))
            else:
                md.append(f"| {sign} | {r['source']} | "
                          + " | ".join(_fmt(r[f]) for f, _ in COLUMNS) + " |")
        md.append("")

    cell = {(r["run"], r["sign"]): r.get(args.metric) for r in rows}
    md += ["", f"## {args.metric}: sign x checkpoint", "",
           "| Sign | " + " | ".join(runs) + " |",
           "|---|" + "---:|" * len(runs)]
    for sign in signs:
        md.append(f"| {sign} | "
                  + " | ".join(_fmt(cell.get((run, sign))) for run in runs) + " |")

    md_path = args.out_dir / "all_metrics.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {md_path}\nWrote {csv_path}")


if __name__ == "__main__":
    main()
