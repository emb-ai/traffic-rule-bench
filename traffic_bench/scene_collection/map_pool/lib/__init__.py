"""moscow_scenes.lib — dual-path and segment harvest helpers."""

from .dual_path import (
    DualPathScenario,
    build_edge_graph,
    crop_to_dual_path,
    dual_path_scenario_from_meta,
    fill_slots_for_junctions,
    find_dual_paths_for_slot,
)
from .roles import (
    SIGN_TO_SLOTS,
    SLOTS,
    scenario_matches_sign,
    sign_shape_policy,
    sign_to_slots,
)
from .segment import (
    CURVED_THRESHOLD,
    MIN_SEGMENT_LENGTH_M,
    STRAIGHT_THRESHOLD,
    SegmentCandidate,
    build_edge_metrics_cache,
    build_junction_positions_cache,
    calculate_straightness,
    enrich_lane_fields,
    osm_way_id_from_edge,
    pass_ok_from_indices,
)
from .stem import is_t_stem_approach
from .crosswalk_inject import (
    CrosswalkInjection,
    CrosswalkInjectionResult,
    calculate_crosswalk_positions,
    count_net_crossings,
    find_paired_edges,
    identify_main_edges,
    inject_crosswalk,
    validate_crosswalk_net,
)

__all__ = [
    # dual_path
    "DualPathScenario",
    "SLOTS",
    "SIGN_TO_SLOTS",
    "build_edge_graph",
    "crop_to_dual_path",
    "dual_path_scenario_from_meta",
    "fill_slots_for_junctions",
    "find_dual_paths_for_slot",
    # roles
    "scenario_matches_sign",
    "sign_shape_policy",
    "sign_to_slots",
    # segment
    "CURVED_THRESHOLD",
    "MIN_SEGMENT_LENGTH_M",
    "STRAIGHT_THRESHOLD",
    "SegmentCandidate",
    "build_edge_metrics_cache",
    "build_junction_positions_cache",
    "calculate_straightness",
    "enrich_lane_fields",
    "osm_way_id_from_edge",
    "pass_ok_from_indices",
    # stem
    "is_t_stem_approach",
    # crosswalk_inject
    "CrosswalkInjection",
    "CrosswalkInjectionResult",
    "calculate_crosswalk_positions",
    "count_net_crossings",
    "find_paired_edges",
    "identify_main_edges",
    "inject_crosswalk",
    "validate_crosswalk_net",
]
