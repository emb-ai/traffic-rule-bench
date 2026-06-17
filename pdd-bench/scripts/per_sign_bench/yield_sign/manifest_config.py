"""Experiment-level defaults for yield-sign real manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Shared with generate_real_manifest.py / run_benchmark_real.py CLI defaults.
DEFAULT_SPAWN_DISTANCE_BEFORE_END = 20.0
DEFAULT_AUX_DISTANCE_FROM_INTERSECTION = 20.0
DEFAULT_AUX_LANES_OCCUPIED_MAX = 4

# Row fields that may be filled from manifest.json / real_manifest_summary.json.
EXPERIMENT_DEFAULT_KEYS = (
    "spawn_distance_before_end",
    "sign_distance_before_end",
    "spawn_velocity_ms",
    "traffic_density",
    "horizon",
    "auxiliary_agent",
    "aux_distance_from_intersection",
    "aux_convoy_size_max",
    "aux_convoy_gap_m",
    "aux_lanes_occupied_max",
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

    if out.get("aux_distance_from_intersection") is None:
        raw = config.get(
            "aux_distance_from_intersection", DEFAULT_AUX_DISTANCE_FROM_INTERSECTION
        )
        out["aux_distance_from_intersection"] = float(raw)

    for key in EXPERIMENT_DEFAULT_KEYS:
        if key in ("spawn_distance_before_end", "aux_distance_from_intersection"):
            continue
        if out.get(key) is None and key in config:
            out[key] = config[key]

    return out
