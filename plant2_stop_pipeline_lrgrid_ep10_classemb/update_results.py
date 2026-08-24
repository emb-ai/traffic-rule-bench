#!/usr/bin/env python3
"""Rebuild RESULTS.md — delegates to ep20/30 merge script so longer-train
rows are preserved when this file is re-run from the ep10 workdir.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LONG = HERE.parent / "plant2_stop_pipeline_lrgrid_ep20_30_classemb"
MERGER = LONG / "update_results.py"


def main() -> None:
    os.environ.setdefault("WORK", str(LONG if LONG.is_dir() else HERE))
    os.environ.setdefault("RESULTS_MD", str(HERE / "RESULTS.md"))
    if MERGER.is_file():
        runpy.run_path(str(MERGER), run_name="__main__")
        return
    # Fallback: minimal ep10-only rewrite if sibling missing.
    print(f"WARN: merge script missing at {MERGER}; ep10-only fallback not implemented", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
