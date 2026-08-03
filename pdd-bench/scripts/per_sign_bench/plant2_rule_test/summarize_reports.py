#!/usr/bin/env python3
"""Collect per-sign cumulative.json reports into one markdown/CSV summary.

Reads ``plant2_rule_test/output/<run-name>/<label>/eval_out/reports/cumulative.json``
produced by ``eval_checkpoint_on_test.py``.

Examples:
  python summarize_reports.py
  python summarize_reports.py --run-name plant2_rule_test --baseline plant2_rule_default
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Same order as eval_checkpoint_on_test.JOBS
LABELS = [
    "2.1",
    "2.3.1-2.3.3",
    "2.4",
    "2.5",
    "3.1-3.2",
    "4.3",
    "5.7.1-5.7.2",
    "5.15.1-5.15.2",
    "5.19",
]


def _slug(label: str) -> str:
    return label.replace(".", "_").replace("-", "_")


def _f(x) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _pick_float(m: dict, *keys: str) -> float | None:
    """First present numeric field; keep real 0.0 (do not use ``a or b``)."""
    for k in keys:
        if k not in m:
            continue
        v = m[k]
        if v is None or v == "":
            continue
        return _f(v)
    return None


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="plant2_rule_test")
    ap.add_argument(
        "--baseline",
        default="plant2_rule_default",
        help="baseline key inside cumulative.json per_baseline (default: plant2_rule_default)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where to write summary (default: output/<run-name>/_summary)",
    )
    args = ap.parse_args()

    root = HERE / "output" / args.run_name
    out_dir = args.out_dir or (root / "_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    missing: list[str] = []
    for label in LABELS:
        cum = root / _slug(label) / "eval_out" / "reports" / "cumulative.json"
        if not cum.is_file():
            missing.append(label)
            continue
        data = json.loads(cum.read_text(encoding="utf-8"))
        pb = data.get("per_baseline") or {}
        m = pb.get(args.baseline)
        if m is None:
            # try without _default / fuzzy
            keys = [k for k in pb if args.baseline in k or k.startswith(args.baseline)]
            if len(keys) == 1:
                m = pb[keys[0]]
                baseline = keys[0]
            else:
                print(f"[warn] {label}: baseline {args.baseline!r} not in {sorted(pb)}", file=sys.stderr)
                missing.append(label)
                continue
        else:
            baseline = args.baseline

        n = int(m.get("n", 0) or 0)
        in_zone = int(m.get("n_in_zone", m.get("in_zone_runs", 0)) or 0)
        rows.append(
            {
                "label": label,
                "baseline": baseline,
                "n": n,
                "n_in_zone": in_zone,
                "success_rate": _f(m.get("success_rate")),
                "dest_rate": _f(m.get("dest_rate")),
                "sign_compliance_sr": _pick_float(
                    m, "sign_compliance_sr", "sign_compliant_rate"
                ),
                "sign_compliance_in_zone": _pick_float(
                    m,
                    "sign_compliance_in_zone",
                    "sign_compliant_in_zone_rate",
                    "sign_compliance_x",
                ),
                "avg_efficiency": _pick_float(m, "avg_efficiency", "driving_efficiency"),
                "avg_smoothness": _pick_float(m, "avg_smoothness", "comfort"),
                "source": str(cum),
            }
        )

    if not rows:
        sys.exit(f"no cumulative.json found under {root}")

    # Markdown
    md_lines = [
        f"# plant2_rule test summary — `{args.run_name}`",
        "",
        f"Baseline: `{args.baseline}`",
        "",
        "| Sign | N | In-zone | Dest rate | Sign SR | Sign SR (in-zone) | Eff | Smooth |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['label']} | {r['n']} | {r['n_in_zone']} | "
            f"{_fmt(r['dest_rate'])} | {_fmt(r['sign_compliance_sr'])} | "
            f"{_fmt(r['sign_compliance_in_zone'])} | "
            f"{_fmt(r['avg_efficiency'])} | {_fmt(r['avg_smoothness'])} |"
        )
    if missing:
        md_lines += ["", f"Missing / incomplete: {', '.join(missing)}"]

    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "label",
        "baseline",
        "n",
        "n_in_zone",
        "success_rate",
        "dest_rate",
        "sign_compliance_sr",
        "sign_compliance_in_zone",
        "avg_efficiency",
        "avg_smoothness",
        "source",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(md_path.read_text(encoding="utf-8"))
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    if missing:
        print(f"[warn] missing: {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
