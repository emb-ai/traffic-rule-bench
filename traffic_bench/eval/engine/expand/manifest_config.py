"""Manifest configuration defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Shared with eval manifest / run defaults.
# Ego starts close enough to approach a ~5 m yield stop without a long idle.
DEFAULT_SPAWN_DISTANCE_BEFORE_END = 12.0
DEFAULT_AUX_DISTANCE_FROM_INTERSECTION = 20.0
DEFAULT_AUX_LANES_OCCUPIED_MAX = 4
# Cap ego travel / visual finish mark along the destination lane (4.3 / 3.2).
DEFAULT_DESTINATION_MAX_ALONG_M = 100.0
# Expert mandatory dwell at stop line after speed≈0 (sim steps; ×0.1 s ≈ seconds).
# Was 30 (~3.0 s); halved to 15 (~1.5 s).
DEFAULT_STOP_WAIT_STEPS = 15

# Blocked road (3.2) defaults — sign on forbidden lane from its start.
DEFAULT_SIGN_DISTANCE_FROM_START = 10.0
DEFAULT_BLOCKED_ROAD_SPAWN_DISTANCE_BEFORE_END = 25.0
DEFAULT_COMPLIANT_STOP_SUCCESS_SECONDS = 3.0
DEFAULT_COMPLIANT_STOP_MAX_DIST_M = 12.0
DEFAULT_COMPLIANT_STOP_SPEED_MPS = 0.5

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
    "stop_wait_steps",
    # Roundabout / blocked_road; only copied when present in experiment config.
    "destination_max_along_m",
    # Blocked road (3.2)
    "sign_distance_from_start",
    "compliant_stop_success_seconds",
    "compliant_stop_max_dist_m",
    "compliant_stop_speed_mps",
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

    # Roundabout-only: never invent a default for yield/main/stop rows.
    if out.get("destination_max_along_m") is None and "destination_max_along_m" in config:
        out["destination_max_along_m"] = float(config["destination_max_along_m"])

    if out.get("aux_distance_from_intersection") is None:
        raw = config.get(
            "aux_distance_from_intersection", DEFAULT_AUX_DISTANCE_FROM_INTERSECTION
        )
        out["aux_distance_from_intersection"] = float(raw)

    if out.get("stop_wait_steps") is None:
        raw = config.get("stop_wait_steps", DEFAULT_STOP_WAIT_STEPS)
        out["stop_wait_steps"] = int(raw)

    for key in EXPERIMENT_DEFAULT_KEYS:
        if key in (
            "spawn_distance_before_end",
            "destination_max_along_m",
            "aux_distance_from_intersection",
            "stop_wait_steps",
        ):
            continue
        if out.get(key) is None and key in config:
            out[key] = config[key]

    return out
