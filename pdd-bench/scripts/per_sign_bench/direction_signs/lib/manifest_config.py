"""Manifest configuration defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Shared with generate_manifest.py / run_benchmark.py CLI defaults.
DEFAULT_SPAWN_DISTANCE_BEFORE_END = 20.0

# Row fields that may be filled from manifest.json / real_manifest_summary.json.
EXPERIMENT_DEFAULT_KEYS = (
    "spawn_distance_before_end",
    "sign_distance_before_end",
    "spawn_velocity_ms",
    "traffic_density",
    "horizon",
)


def load_manifest_config(manifest_path: Path | str) -> dict[str, Any]:
    """Load experiment defaults next to a real_manifest.jsonl path."""
    path = Path(manifest_path)
    parent = path.parent if path.suffix else path
    for name in ("manifest.json", "real_manifest_summary.json"):
        cfg_path = parent / name
        if not cfg_path.is_file():
            continue
        try:
            with open(cfg_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            continue
    return {}


def enrich_manifest_row(row: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fill missing per-scene fields from experiment config and built-in defaults."""
    config = config or {}
    out = dict(row)

    if out.get("spawn_distance_before_end") is None:
        raw = config.get("spawn_distance_before_end", DEFAULT_SPAWN_DISTANCE_BEFORE_END)
        out["spawn_distance_before_end"] = float(raw)

    for key in EXPERIMENT_DEFAULT_KEYS:
        if key == "spawn_distance_before_end":
            continue
        if out.get(key) is None and key in config:
            out[key] = config[key]

    return out
