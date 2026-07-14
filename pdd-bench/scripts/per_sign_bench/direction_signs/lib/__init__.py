"""Core library modules for direction signs 4.1.1–4.1.6."""

from .sumo_utils import (
    resolve_net_file,
    load_scene_meta,
    resolve_scene_dir,
    DEFAULT_NET_FILE,
)
from .manifest_config import (
    DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_AUX_LANES_OCCUPIED_MAX,
    load_manifest_config,
    enrich_manifest_row,
)
from .junction_priority_layout import (
    JunctionPriorityLayout,
    JunctionLayoutError,
    ApproachArm,
    build_junction_priority_layout,
    load_junction_priority_layout,
)
from .scene_augmentation import (
    SpawnScenario,
    augment_layout_for_scene,
    enumerate_spawn_scenarios,
)
from .auxiliary_agent import (
    AuxiliaryAgentsManager,
    add_auxiliary_agents,
    add_auxiliary_agent,
    resolve_aux_spawn_lanes,
    main_lane_keys_for_aux,
    select_occupied_main_lanes,
    DEFAULT_CONVOY_SIZE,
    DEFAULT_CONVOY_GAP_M,
)
from .direction_sign_spec import (
    DEFAULT_PDD_CODE,
    DIRECTION_SIGN_CODES,
    DIRECTION_SIGN_SPECS,
    SIGN_FAMILY,
    DirectionSignSpec,
    get_direction_sign_spec,
    resolve_sign_class,
)
