#!/usr/bin/env python3
"""Run Moscow scene harvest: download → netconvert → enumerate → crop T/X/O, dual_path, segment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def _run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *extra]
    print(f"\n=== {' '.join(cmd)} ===\n")
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-netconvert", action="store_true")
    ap.add_argument("--skip-enumerate", action="store_true")
    ap.add_argument("--skip-crop", action="store_true")
    ap.add_argument("--skip-dual-path", action="store_true")
    ap.add_argument("--skip-segment", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--shapes", default="T,X,O")
    ap.add_argument("--max-per-shape", type=int, default=None)
    ap.add_argument("--radius-m", type=float, default=80.0)
    ap.add_argument("--min-lane-m", type=float, default=10.0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dual-path-max-junctions", type=int, default=None)
    ap.add_argument("--dual-path-n-per-slot", type=int, default=1)
    ap.add_argument(
        "--dual-path-max-per-slot",
        type=int,
        default=0,
        help="Max dual_path scenes per (shape, slot); 0 = no cap",
    )
    args = ap.parse_args()

    build_args: list[str] = []
    if args.skip_download:
        build_args.append("--skip-download")
    if args.force_download:
        build_args.append("--force-download")
    if args.skip_netconvert:
        build_args.append("--skip-netconvert")
    _run("build_net.py", build_args)

    if not args.skip_enumerate:
        _run(
            "enumerate_junctions.py",
            [
                "--shapes",
                args.shapes,
                "--min-lane-m",
                str(args.min_lane_m),
            ],
        )

    if not args.skip_crop:
        crop_args = [
            "--shapes",
            args.shapes,
            "--radius-m",
            str(args.radius_m),
            "--workers",
            str(args.workers),
        ]
        if args.max_per_shape is not None:
            crop_args += ["--max-per-shape", str(args.max_per_shape)]
        if args.skip_existing:
            crop_args.append("--skip-existing")
        _run("crop_scenes.py", crop_args)

    if not args.skip_dual_path:
        dual_args = [
            "--n-per-junction-slot",
            str(args.dual_path_n_per_slot),
            "--max-per-slot",
            str(args.dual_path_max_per_slot),
            "--min-lane-m",
            str(args.min_lane_m),
        ]
        if args.dual_path_max_junctions is not None:
            dual_args += ["--max-junctions", str(args.dual_path_max_junctions)]
        if args.skip_existing:
            dual_args.append("--skip-existing")
        _run("crop_dual_path_scenes.py", dual_args)

    if not args.skip_segment:
        if not args.skip_enumerate:
            _run("enumerate_segments.py", [])
        seg_args = ["--max-scenes", "0"]
        if args.skip_existing:
            seg_args.append("--skip-existing")
        _run("crop_segment_scenes.py", seg_args)

    print("\n[run_pipeline] Finished. See README.md for layout and provenance.")


if __name__ == "__main__":
    main()
