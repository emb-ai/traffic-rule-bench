"""Shared checks for whether a scene can enter generate_manifest (crosswalk 5.19)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from lib.crosswalk_layout import build_crosswalk_approaches, count_net_crossings, net_has_crossings
from lib.manifest_config import DEFAULT_SPAWN_DISTANCE_BEFORE_END
from lib.sumo_utils import load_scene_meta, resolve_net_file


@dataclass
class ManifestViabilityResult:
    viable: bool
    reason: str = ""
    detail: str = ""
    spawn_lane_count: int = 0
    scenario_count: int = 0


def check_manifest_viability(
    net_path: Path,
    *,
    meta: Optional[dict[str, Any]] = None,
    min_ego_lane_m: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    **_kwargs: Any,
) -> ManifestViabilityResult:
    """Return whether a scene would survive crosswalk manifest filters."""
    target_crosswalk_id = (meta or {}).get("crosswalk_id")

    if not net_path.is_file():
        return ManifestViabilityResult(
            viable=False,
            reason="missing_net",
            detail=str(net_path),
        )

    if not net_has_crossings(net_path):
        return ManifestViabilityResult(
            viable=False,
            reason="no_crossings",
            detail="SUMO net has no pedestrian crossing edges",
        )

    crossing_count = count_net_crossings(net_path)
    if crossing_count != 1:
        return ManifestViabilityResult(
            viable=False,
            reason="multiple_crossings",
            detail=f"expected exactly 1 crossing in cropped scene, found {crossing_count}",
        )

    approaches = build_crosswalk_approaches(net_path, min_approach_length=min_ego_lane_m)
    if target_crosswalk_id:
        approaches = [a for a in approaches if a.crosswalk_id == target_crosswalk_id]
    if approaches:
        return ManifestViabilityResult(
            viable=True,
            spawn_lane_count=len(approaches),
            scenario_count=len(approaches),
        )

    return ManifestViabilityResult(
        viable=False,
        reason="no_approach_lanes",
        detail=f"no vehicle approach lane >= {min_ego_lane_m:.0f}m toward a crossing",
    )


def check_scene_dir_viability(
    scene_dir: Path,
    *,
    min_ego_lane_m: float = DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    **_kwargs: Any,
) -> ManifestViabilityResult:
    """Check viability for a scene folder (meta.json + net.xml)."""
    meta_path = scene_dir / "meta.json"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = load_scene_meta(scene_dir)

    net_file = resolve_net_file(scene_dir, meta)
    net_path = scene_dir / net_file
    if not net_path.is_file():
        return ManifestViabilityResult(
            viable=False,
            reason="missing_net",
            detail=f"{scene_dir.name}: {net_file} not found",
        )

    return check_manifest_viability(net_path, meta=meta, min_ego_lane_m=min_ego_lane_m)
