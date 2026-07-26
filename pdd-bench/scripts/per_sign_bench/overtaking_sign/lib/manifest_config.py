"""Manifest configuration defaults for overtaking_sign (3.20)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_SPAWN_DISTANCE_FROM_START = 3.0
DEFAULT_SIGN_DISTANCE_FROM_START = 2.0
DEFAULT_AUX_FRAC = 0.5
DEFAULT_WAIT_BEHIND_SUCCESS_S = 2.0
DEFAULT_WAIT_BEHIND_SPEED_MPS = 0.5
DEFAULT_WAIT_BEHIND_GAP_MAX_M = 18.0
DEFAULT_WAIT_BEHIND_GAP_MIN_M = 2.0
DEFAULT_MIN_EDGE_LENGTH_M = 60.0
DEFAULT_SPAWN_VELOCITY_MS = 2.5

EXPERIMENT_DEFAULT_KEYS = (
    "spawn_distance_from_start",
    "sign_distance_from_start",
    "aux_frac",
    "aux_long_m",
    "wait_behind_success_seconds",
    "wait_behind_speed_mps",
    "wait_behind_gap_max_m",
    "wait_behind_gap_min_m",
    "spawn_velocity_ms",
    "traffic_density",
    "horizon",
)


def load_manifest_config(manifest_path: Path | str) -> dict[str, Any]:
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


def enrich_manifest_row(
    row: dict[str, Any], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    config = config or {}
    out = dict(row)
    defaults = {
        "spawn_distance_from_start": DEFAULT_SPAWN_DISTANCE_FROM_START,
        "sign_distance_from_start": DEFAULT_SIGN_DISTANCE_FROM_START,
        "aux_frac": DEFAULT_AUX_FRAC,
        "wait_behind_success_seconds": DEFAULT_WAIT_BEHIND_SUCCESS_S,
        "wait_behind_speed_mps": DEFAULT_WAIT_BEHIND_SPEED_MPS,
        "wait_behind_gap_max_m": DEFAULT_WAIT_BEHIND_GAP_MAX_M,
        "wait_behind_gap_min_m": DEFAULT_WAIT_BEHIND_GAP_MIN_M,
        "spawn_velocity_ms": DEFAULT_SPAWN_VELOCITY_MS,
        "traffic_density": 0.0,
        "horizon": 600,
        "pdd_code": "3.20",
        "sign_code": "3.20",
        "force_opposite_as_peer": True,
        "opposite_peer_side": "left",
        "probe_overtake_disable_wait_success": True,
    }
    for key, default in defaults.items():
        if out.get(key) is None:
            out[key] = config.get(key, default)
    return out
