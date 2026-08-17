"""5.7.x dual-path from crop ``meta.json`` (moscow dual_path harvest).

Geometry lives on the scene: spawn, dest, both paths, ``wrong_dir_edges``.
No import from ``one_way_signs`` / ``_old``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from .dual_path_scene import (
    DualPathScenario,
    dual_path_to_spawn_scenario,
    ego_spawn_lane_nums_for_dual,
    pick_meta_dual_path,
)
from .one_way_sign_spec import (
    dual_path_role_dirs,
    get_one_way_sign_spec,
    resolve_sign_class,
)


def discover_one_way_dual_paths(
    net_path: Path,
    *,
    pdd_code: str,
    min_gain_m: float = 20.0,
    min_lane_length_m: float = 8.0,
    max_scenarios: int = 20,
    junction_ids: Optional[Sequence[str]] = None,
    arm_counts: Sequence[int] = (3, 4),
    scene_meta: Optional[dict] = None,
) -> List[DualPathScenario]:
    """Load crop-time dual-path from ``meta.json``. No net rediscovery."""
    del net_path, min_gain_m, min_lane_length_m, max_scenarios, arm_counts
    spec_code = get_one_way_sign_spec(pdd_code).pdd_code
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
    "discover_one_way_dual_paths",
    "dual_path_to_spawn_scenario",
    "ego_spawn_lane_nums_for_dual",
    "get_one_way_sign_spec",
    "resolve_sign_class",
]
