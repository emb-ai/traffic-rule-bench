#!/usr/bin/env python3
"""One policy-vs-oracle table across every sign, from the per-family ones.

The report is produced per family because each family recorded at its own
horizon, and the horizon decides whether an episode that ran out of steps
counts as an arrival. Re-running the report once over everything would need a
single horizon and would misjudge the families that do not share it, so the
per-family tables are combined here instead: rates weighted by the scenes they
were measured over, counts summed.

    python tools/oracle_report_all.py <collection_root> [--report report]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

COUNT_ROWS = ("n (episodes)", "n_scenes (common denominator)")
PICKS = "oracle picks"
_PAREN = re.compile(r"([-\d.]+)\s*\((\d+)\)")


def parse(cell: str):
    """Return (value, weight_override) for a table cell."""
    cell = (cell or "").strip()
    if not cell or cell in {"-", "n/a"}:
        return None, None
    m = _PAREN.match(cell)
    if m:
        return float(m.group(1)), int(m.group(2))
    try:
        return float(cell), None
    except ValueError:
        return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--report", default="report", help="per-family report dir")
    args = ap.parse_args()

    root = Path(args.root)
    experts: list[str] = []
    rows: dict[str, dict[str, list]] = {}
    order: list[str] = []
    n_fam = 0

    for fam in sorted(os.listdir(root)):
        f = root / fam / args.report / "oracle_metrics_summary.tsv"
        if not f.is_file():
            continue
        n_fam += 1
        with f.open() as fh:
            table = list(csv.reader(fh, delimiter="\t"))
        header, body = table[0], table[1:]
        cols = header[1:]
        for c in cols:
            if c not in experts:
                experts.append(c)
        scenes = None
        for r in body:
            if r[0].startswith("n_scenes"):
                scenes = parse(r[1])[0] or 0
        for r in body:
            name = r[0]
            key = PICKS if name.startswith(PICKS) else name
            if key not in rows:
                rows[key] = {}
                order.append(key)
            for c, cell in zip(cols, r[1:]):
                v, w = parse(cell)
                if v is None:
                    continue
                rows[key].setdefault(c, []).append(
                    (v, w if w is not None else (scenes or 0)))

    if not n_fam:
        print(f"no per-family reports under {root}/*/{args.report}/")
        return 2

    print(f"Unified over {n_fam} sign families\n")
    w = max(len(e) for e in experts) + 2
    print("%-38s" % "metric" + "".join("%*s" % (w, e) for e in experts))
    print("-" * (38 + w * len(experts)))
    for key in order:
        cells = []
        for e in experts:
            pairs = rows[key].get(e)
            if not pairs:
                cells.append("%*s" % (w, "-")); continue
            if key in COUNT_ROWS or key == PICKS:
                cells.append("%*d" % (w, sum(v for v, _ in pairs)))
            else:
                tot = sum(x for _, x in pairs)
                val = sum(v * x for v, x in pairs) / tot if tot else 0.0
                cells.append("%*.3f" % (w, val))
        print("%-38s" % key[:38] + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
