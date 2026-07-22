#!/usr/bin/env python3
"""Cropping is not required for no-entry signs 3.1 / 3.2.

Catalog scenes under ``pdd-bench/scenes/3.1`` and ``3.2`` are already local
OSM extracts around each sign. Import them with
``import_catalog_scenes.py`` instead of cropping core maps.
"""
from __future__ import annotations

import sys


def main() -> int:
    print(
        "crop_junction_scene.py is a no-op for no-entry signs (3.1 / 3.2).\n"
        "Catalog scenes are already local extracts — use:\n"
        "  python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.1\n"
        "  python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
