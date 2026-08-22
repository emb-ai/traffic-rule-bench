"""Shim — implementation is ``traffic_bench.eval.signs.dual_path``."""

from traffic_bench.eval.core.scenarios.one_way_sign_spec import (
    get_one_way_sign_spec,
    resolve_sign_class,
)
from traffic_bench.eval.signs.dual_path.scene import (
    DualPathScenario,
    dual_path_to_spawn_scenario,
    ego_spawn_lane_nums_for_dual,
)
from traffic_bench.eval.signs.dual_path.spec import discover_dual_paths as discover_one_way_dual_paths

__all__ = [
    "DualPathScenario",
    "discover_one_way_dual_paths",
    "dual_path_to_spawn_scenario",
    "ego_spawn_lane_nums_for_dual",
    "get_one_way_sign_spec",
    "resolve_sign_class",
]
