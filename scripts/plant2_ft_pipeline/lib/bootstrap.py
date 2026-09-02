"""Ensure pipeline root is on sys.path for `python subdir/script.py` invocations."""
from __future__ import annotations

import sys
from pathlib import Path

_PIPELINE_ROOT = Path(__file__).resolve().parent.parent


def ensure_pipeline_root() -> Path:
    root = str(_PIPELINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return _PIPELINE_ROOT
