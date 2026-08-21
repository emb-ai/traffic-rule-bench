"""Shared I/O helpers for collect scripts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def load_index(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_netconvert() -> str:
    for path in (
        shutil.which("netconvert"),
        str(Path.home() / ".local" / "bin" / "netconvert"),
        "/usr/local/bin/netconvert",
        "/usr/bin/netconvert",
    ):
        if path and Path(path).exists():
            return path
    raise FileNotFoundError(
        "netconvert not found. Install SUMO or add netconvert to PATH."
    )
