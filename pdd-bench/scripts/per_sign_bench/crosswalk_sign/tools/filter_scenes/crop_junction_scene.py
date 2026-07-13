#!/usr/bin/env python3
"""Backward-compatible alias for crop_crosswalk_scene.py."""
from __future__ import annotations

import sys
from pathlib import Path

FILTER_SCENES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FILTER_SCENES_DIR.parent.parent))

from tools.filter_scenes.crop_crosswalk_scene import main  # noqa: E402

if __name__ == "__main__":
    main()
