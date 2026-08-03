#!/usr/bin/env python3
"""DEPRECATED thin wrapper.

Full-cache iterkeys scanning is too slow (~5.9M keys). Use the fast extract path:

  extract_patch_2p5_cache.py
    - enumerate 2.5-subset sample keys (n_boxes-16 starts)
    - copy+patch from /tmp/plant2_ds_cache_spatial_aug into
      /tmp/plant2_ds_cache_2p5_tsfix
    - optional PlanTDataset materialize for misses

This wrapper forwards to extract_patch_2p5_cache.py.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "extract_patch_2p5_cache.py"


def main() -> int:
    # Default: reset dst and materialize misses so a naive launch still works.
    argv = list(sys.argv[1:])
    if "--help" in argv or "-h" in argv:
        sys.argv = [str(SCRIPT), "--help"]
        runpy.run_path(str(SCRIPT), run_name="__main__")
        return 0
    extras = []
    if "--reset-dst" not in argv:
        extras.append("--reset-dst")
    if "--materialize-missing" not in argv and "--no-materialize" not in argv:
        extras.append("--materialize-missing")
    argv = [a for a in argv if a != "--no-materialize"]
    sys.argv = [str(SCRIPT)] + extras + argv
    print(
        f"[patch_diskcache_2p5_target_speed] forwarding to {SCRIPT.name} "
        f"argv={sys.argv[1:]}",
        flush=True,
    )
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
