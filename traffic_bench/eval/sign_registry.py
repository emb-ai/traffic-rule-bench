"""Eval sign index: English id → family, spawn, data folder.

Official plate numbers (`sign_code`) are harvest / manifest keys and comments.
CLI and Hydra use the English id (`yield`, `direction_right`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from traffic_bench.eval.engine.spawn.scene_augmentation import SpawnStrategy

LayoutMode = Literal["main_main", "main_secondary", "roundabout"]
SignFamily = Literal[
    "junction",
    "roundabout",
    "blocked",
    "dual_path",
    "crosswalk",
    "detour",
    "speed",
]

EVAL = Path(__file__).resolve().parent
PACKAGE_ROOT = EVAL.parent
REPO_ROOT = PACKAGE_ROOT.parent
DATA = REPO_ROOT / "data"


@dataclass(frozen=True)
class SignProfile:
    """How one eval sign differs from others (spawn, plates, data folder)."""

    id: str
    family: SignFamily
    sign_code: str
    sign_type: str
    sign_name: str
    layout_mode: LayoutMode
    spawn_strategy: SpawnStrategy
    data_subdir: str = ""
    ego_road_class: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.data_subdir:
            object.__setattr__(self, "data_subdir", self.id)

    @property
    def pdd_code(self) -> str:
        """Harvest yaml / manifest row key (legacy attribute name)."""
        return self.sign_code


MAIN_ROAD = SignProfile(
    id="main",
    family="junction",
    sign_code="2.1",
    sign_type="main",
    sign_name="Main road (equal priority)",
    layout_mode="main_main",
    spawn_strategy="equal_priority",
    data_subdir="main_road",
)

SECONDARY_ROAD = SignProfile(
    id="secondary",
    family="junction",
    sign_code="2.3",
    sign_type="secondary",
    sign_name="Intersection with secondary road (2.3)",
    # Same geometry / ego / aux as yield (2.4). Main arms get 2.3.x plates
    # (X: 2.3.1; T: 2.3.2 + 2.3.3); secondary arms get YieldSign.
    layout_mode="main_secondary",
    spawn_strategy="yield",
    data_subdir="secondary_road",
    ego_road_class="secondary",
)

YIELD = SignProfile(
    id="yield",
    family="junction",
    sign_code="2.4",
    sign_type="yield",
    sign_name="Yield",
    layout_mode="main_secondary",
    spawn_strategy="yield",
    ego_road_class="secondary",
)

STOP = SignProfile(
    id="stop",
    family="junction",
    sign_code="2.5",
    sign_type="stop",
    sign_name="Stop",
    # Same junction geometry / spawn / aux as yield; only the plate +
    # stop-line violation differ (handled in run + StopSign class).
    layout_mode="main_secondary",
    spawn_strategy="yield",
    ego_road_class="secondary",
)

ROUNDABOUT = SignProfile(
    id="roundabout",
    family="roundabout",
    sign_code="4.3",
    sign_type="roundabout",
    sign_name="Roundabout circulation (4.3)",
    layout_mode="roundabout",
    spawn_strategy="roundabout",
    ego_road_class="secondary",
)

BLOCKED_ROAD = SignProfile(
    id="blocked_road",
    family="blocked",
    sign_code="3.2",
    sign_type="blocked_road",
    sign_name="Movement prohibited (3.2)",
    layout_mode="main_main",
    spawn_strategy="blocked_road",
)

ONE_WAY_RIGHT = SignProfile(
    id="one_way_right",
    family="dual_path",
    sign_code="5.7.1",
    sign_type="one_way",
    sign_name="Exit onto one-way road (right / 5.7.1)",
    layout_mode="main_main",
    spawn_strategy="one_way",
)

ONE_WAY_LEFT = SignProfile(
    id="one_way_left",
    family="dual_path",
    sign_code="5.7.2",
    sign_type="one_way",
    sign_name="Exit onto one-way road (left / 5.7.2)",
    layout_mode="main_main",
    spawn_strategy="one_way",
)

DIRECTION_STRAIGHT = SignProfile(
    id="direction_straight",
    family="dual_path",
    sign_code="4.1.1",
    sign_type="direction",
    sign_name="Proceed straight (4.1.1)",
    layout_mode="main_main",
    spawn_strategy="direction",
)

DIRECTION_RIGHT = SignProfile(
    id="direction_right",
    family="dual_path",
    sign_code="4.1.2",
    sign_type="direction",
    sign_name="Turn right (4.1.2)",
    layout_mode="main_main",
    spawn_strategy="direction",
)

DIRECTION_LEFT = SignProfile(
    id="direction_left",
    family="dual_path",
    sign_code="4.1.3",
    sign_type="direction",
    sign_name="Turn left (4.1.3)",
    layout_mode="main_main",
    spawn_strategy="direction",
)

DIRECTION_STRAIGHT_RIGHT = SignProfile(
    id="direction_straight_right",
    family="dual_path",
    sign_code="4.1.4",
    sign_type="direction",
    sign_name="Straight or right (4.1.4)",
    layout_mode="main_main",
    spawn_strategy="direction",
)

DIRECTION_STRAIGHT_LEFT = SignProfile(
    id="direction_straight_left",
    family="dual_path",
    sign_code="4.1.5",
    sign_type="direction",
    sign_name="Straight or left (4.1.5)",
    layout_mode="main_main",
    spawn_strategy="direction",
)

DIRECTION_LEFT_RIGHT = SignProfile(
    id="direction_left_right",
    family="dual_path",
    sign_code="4.1.6",
    sign_type="direction",
    sign_name="Right or left (4.1.6)",
    layout_mode="main_main",
    spawn_strategy="direction",
)

NO_TURN_RIGHT = SignProfile(
    id="no_turn_right",
    family="dual_path",
    sign_code="3.18.1",
    sign_type="no_turn",
    sign_name="No right turn (3.18.1)",
    layout_mode="main_main",
    spawn_strategy="no_turn",
)

NO_TURN_LEFT = SignProfile(
    id="no_turn_left",
    family="dual_path",
    sign_code="3.18.2",
    sign_type="no_turn",
    sign_name="No left turn (3.18.2)",
    layout_mode="main_main",
    spawn_strategy="no_turn",
)

NO_ENTRY = SignProfile(
    id="no_entry",
    family="dual_path",
    sign_code="3.1",
    sign_type="no_entry",
    sign_name="No entry (3.1)",
    layout_mode="main_main",
    spawn_strategy="no_entry",
)

DETOUR_RIGHT = SignProfile(
    id="detour_right",
    family="detour",
    sign_code="4.2.1",
    sign_type="detour",
    sign_name="Detour obstacle on the right (4.2.1)",
    layout_mode="main_main",
    spawn_strategy="detour",
)

DETOUR_LEFT = SignProfile(
    id="detour_left",
    family="detour",
    sign_code="4.2.2",
    sign_type="detour",
    sign_name="Detour obstacle on the left (4.2.2)",
    layout_mode="main_main",
    spawn_strategy="detour",
)

DETOUR_EITHER = SignProfile(
    id="detour_either",
    family="detour",
    sign_code="4.2.3",
    sign_type="detour",
    sign_name="Detour obstacle on either side (4.2.3)",
    layout_mode="main_main",
    spawn_strategy="detour",
)

SPEED_LIMIT = SignProfile(
    id="speed_limit",
    family="speed",
    sign_code="3.24",
    sign_type="speed",
    sign_name="Maximum speed limit (3.24)",
    layout_mode="main_main",
    spawn_strategy="speed_zone",
)

MIN_SPEED = SignProfile(
    id="min_speed",
    family="speed",
    sign_code="4.6",
    sign_type="speed",
    sign_name="Minimum speed (4.6)",
    layout_mode="main_main",
    spawn_strategy="speed_zone",
)

RESIDENTIAL_ZONE = SignProfile(
    id="residential_zone",
    family="speed",
    sign_code="5.21",
    sign_type="speed",
    sign_name="Residential zone (5.21)",
    layout_mode="main_main",
    spawn_strategy="speed_zone",
)

ZONE_SPEED_LIMIT = SignProfile(
    id="zone_speed_limit",
    family="speed",
    sign_code="5.31",
    sign_type="speed",
    sign_name="Zone speed limit (5.31)",
    layout_mode="main_main",
    spawn_strategy="speed_zone",
)

CROSSWALK = SignProfile(
    id="crosswalk",
    family="crosswalk",
    sign_code="5.19",
    sign_type="crosswalk",
    sign_name="Pedestrian crossing (5.19)",
    layout_mode="main_main",
    spawn_strategy="crosswalk",
)

_PROFILES = (
    MAIN_ROAD,
    SECONDARY_ROAD,
    YIELD,
    STOP,
    ROUNDABOUT,
    BLOCKED_ROAD,
    ONE_WAY_RIGHT,
    ONE_WAY_LEFT,
    DIRECTION_STRAIGHT,
    DIRECTION_RIGHT,
    DIRECTION_LEFT,
    DIRECTION_STRAIGHT_RIGHT,
    DIRECTION_STRAIGHT_LEFT,
    DIRECTION_LEFT_RIGHT,
    NO_TURN_RIGHT,
    NO_TURN_LEFT,
    NO_ENTRY,
    DETOUR_RIGHT,
    DETOUR_LEFT,
    DETOUR_EITHER,
    SPEED_LIMIT,
    MIN_SPEED,
    RESIDENTIAL_ZONE,
    ZONE_SPEED_LIMIT,
    CROSSWALK,
)

_BY_ID: dict[str, SignProfile] = {p.id: p for p in _PROFILES}
_BY_SIGN_CODE: dict[str, SignProfile] = {p.sign_code: p for p in _PROFILES}


def get_profile(sign_id: str) -> SignProfile:
    """Look up by English eval id. Harvest yaml keys fall back to ``sign_code``."""
    key = str(sign_id).strip()
    profile = _BY_ID.get(key) or _BY_SIGN_CODE.get(key)
    if profile is None:
        known = ", ".join(sorted(_BY_ID))
        raise KeyError(f"Unknown sign {sign_id!r}. Expected one of: {known}")
    return profile


def list_profiles() -> list[SignProfile]:
    return list(_PROFILES)


def profiles_in_family(family: SignFamily) -> list[SignProfile]:
    return [p for p in _PROFILES if p.family == family]


def package_root() -> Path:
    """``traffic_bench/`` package directory."""
    return PACKAGE_ROOT


def repo_root() -> Path:
    """Repository root (parent of the Python package)."""
    return REPO_ROOT


def artifact_root() -> Path:
    """Working artifacts: ``<repo>/data/{scenes,runs,trajectories}/``."""
    return DATA


def scenes_dir(profile: SignProfile) -> Path:
    return DATA / "scenes" / profile.data_subdir


def runs_dir(profile: SignProfile) -> Path:
    return DATA / "runs" / profile.data_subdir


def trajectories_dir(profile: SignProfile) -> Path:
    return DATA / "trajectories" / profile.data_subdir


def output_dir(profile: SignProfile) -> Path:
    """Hydra run parent (alias of ``runs_dir``)."""
    return runs_dir(profile)
