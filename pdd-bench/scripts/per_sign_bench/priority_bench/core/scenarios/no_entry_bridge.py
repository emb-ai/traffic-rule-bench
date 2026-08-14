"""3.1 dual-path from crop ``meta.json`` (moscow dual_path harvest)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from .dual_path_scene import (
    DualPathScenario,
    dual_path_to_spawn_scenario,
    ego_spawn_lane_nums_for_dual,
    pick_meta_dual_path,
)
from .no_entry_sign_spec import (
    dual_path_role_dirs,
    get_no_entry_sign_spec,
    resolve_sign_class,
)


def discover_no_entry_dual_paths(
    net_path: Path,
    *,
    pdd_code: str,
    min_gain_m: float = 20.0,
    min_lane_length_m: float = 8.0,
    max_scenarios: int = 20,
    junction_ids: Optional[Sequence[str]] = None,
    scene_meta: Optional[dict] = None,
) -> List[DualPathScenario]:
    del net_path, min_gain_m, min_lane_length_m, max_scenarios
    spec_code = get_no_entry_sign_spec(pdd_code).pdd_code
    baseline_dirs, compliant_dirs = dual_path_role_dirs(spec_code)
    return pick_meta_dual_path(
        scene_meta,
        pdd_code=spec_code,
        baseline_dirs=baseline_dirs,
        compliant_dirs=compliant_dirs,
        junction_ids=junction_ids,
    )


__all__ = [
    "DualPathScenario",
    "discover_no_entry_dual_paths",
    "dual_path_to_spawn_scenario",
    "ego_spawn_lane_nums_for_dual",
    "get_no_entry_sign_spec",
    "resolve_sign_class",
]
