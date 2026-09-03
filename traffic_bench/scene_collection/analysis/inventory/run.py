"""CLI for harvest inventory analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from traffic_bench.scene_collection.analysis.inventory.figures import write_all
from traffic_bench.scene_collection.analysis.inventory.harvest import (
    load_snapshot,
    summary_dict,
)
from traffic_bench.scene_collection.analysis.inventory.report import write_report
from traffic_bench.scene_collection.paths import SCENE_COLLECTION

DEFAULT_OUT = SCENE_COLLECTION / "analysis" / "inventory"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m traffic_bench.scene_collection analysis inventory",
        description="Count harvest maps and write PNG figures + README.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"package dir for README + summary + figures/ (default: {DEFAULT_OUT})",
    )
    ap.add_argument("--pdf", action="store_true", help="also write PDF copies")
    args = ap.parse_args(argv)

    print("[inventory] Reading crop folders on disk…")
    snap = load_snapshot()
    stats = summary_dict(snap)
    for name, row in stats["families"].items():
        print(f"  {name:10s}  on disk={row['on_disk']:5d}")

    out = args.out.expanduser().resolve()
    fig_dir = out / "figures"
    print(f"[inventory] Writing figures → {fig_dir}")
    written = write_all(snap, fig_dir, pdf=args.pdf)
    report = write_report(snap, out, figures_rel="figures")
    print(f"[inventory] {len(written)} figure files, report {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
