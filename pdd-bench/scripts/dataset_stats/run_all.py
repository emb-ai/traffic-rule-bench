#!/usr/bin/env python3
"""Run collect → plot/report pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    cmds = [
        [sys.executable, str(HERE / "collect_stats.py")],
        [sys.executable, str(HERE / "plot_and_report.py")],
    ]
    for cmd in cmds:
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(HERE))
        if proc.returncode != 0:
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
