"""CLI: scan the harvest and write figures + markdown next to this package."""

from __future__ import annotations

import argparse
from pathlib import Path

from traffic_bench.scene_collection.analysis.figures import write_all
from traffic_bench.scene_collection.analysis.inventory import load_snapshot, summary_dict
from traffic_bench.scene_collection.analysis.report import write_report
from traffic_bench.scene_collection.paths import ANALYSIS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m traffic_bench.scene_collection analysis",
        description="Count harvest maps and write PNG figures + markdown.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ANALYSIS,
        help=f"output directory (default: {ANALYSIS})",
    )
    ap.add_argument(
        "--pdf",
        action="store_true",
        help="also write PDF copies of each figure",
    )
    args = ap.parse_args(argv)

    print("[analysis] Reading indexes and crop folders…")
    snap = load_snapshot()
    stats = summary_dict(snap)
    for name, row in stats["families"].items():
        print(f"  {name:10s}  P={row['index']:5d}  H={row['on_disk']:5d}  ({100 * row['coverage']:.1f}%)")

    out = args.out.expanduser().resolve()
    print(f"[analysis] Writing figures → {out}")
    written = write_all(snap, out, pdf=args.pdf)
    report = write_report(snap, out)
    print(f"[analysis] {len(written)} figure files, report {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
