#!/usr/bin/env python3
"""Run Moscow junction harvest: download → netconvert → enumerate → crop."""

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
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--shapes", default="T,X,O")
    ap.add_argument("--max-per-shape", type=int, default=None)
    ap.add_argument("--radius-m", type=float, default=80.0)
    ap.add_argument("--min-lane-m", type=float, default=10.0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
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

    print("\n[run_pipeline] Finished. See README.md for layout and provenance.")


if __name__ == "__main__":
    main()
