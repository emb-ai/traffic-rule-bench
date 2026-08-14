"""Sign profiles for priority-junction benches (2.1–2.5 / 4.1.x / 4.3 / 3.2 / 5.7.x)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from core.scenarios.scene_augmentation import SpawnStrategy

LayoutMode = Literal["main_main", "main_secondary", "roundabout"]


@dataclass(frozen=True)
class SignProfile:
    """Thin contract describing how one PDD priority sign differs from others."""

    id: str
    pdd_code: str
    sign_type: str
    sign_name: str
    layout_mode: LayoutMode
    spawn_strategy: SpawnStrategy
    data_subdir: str  # under priority_bench/data/
    output_code: str  # e.g. "2_1" for paths

    # Optional: ego must be on this road_class (None = any arm)
    ego_road_class: Optional[str] = None


MAIN_ROAD = SignProfile(
    id="main",
    pdd_code="2.1",
    sign_type="main",
    sign_name="Main road (equal priority)",
    layout_mode="main_main",
    spawn_strategy="equal_priority",
    data_subdir="main_road",
    output_code="2_1",
    ego_road_class=None,
)

SECONDARY_ROAD = SignProfile(
    id="secondary",
    pdd_code="2.3",
    sign_type="secondary",
    sign_name="Intersection with secondary road (2.3)",
    # Same geometry / ego / aux as yield (2.4). Main arms get 2.3.x plates
    # (X: 2.3.1; T: 2.3.2 + 2.3.3); secondary arms get YieldSign.
    layout_mode="main_secondary",
    spawn_strategy="yield",
    data_subdir="secondary_road",
    output_code="2_3",
    ego_road_class="secondary",
)

YIELD = SignProfile(
    id="yield",
    pdd_code="2.4",
    sign_type="yield",
    sign_name="Yield",
    layout_mode="main_secondary",
    spawn_strategy="yield",
    data_subdir="yield",
    output_code="2_4",
    ego_road_class="secondary",
)

STOP = SignProfile(
    id="stop",
    pdd_code="2.5",
    sign_type="stop",
    sign_name="Stop",
    # Same junction geometry / spawn / aux axes as yield; only the plate +
    # stop-line violation differ (handled in run_benchmark + StopSign class).
    layout_mode="main_secondary",
    spawn_strategy="yield",
    data_subdir="stop",
    output_code="2_5",
    ego_road_class="secondary",
)

ROUNDABOUT = SignProfile(
    id="roundabout",
    pdd_code="4.3",
    sign_type="roundabout",
    sign_name="Roundabout circulation (4.3)",
    layout_mode="roundabout",
    spawn_strategy="roundabout",
    data_subdir="roundabout",
    output_code="4_3",
    ego_road_class="secondary",
)

BLOCKED_ROAD = SignProfile(
    id="blocked_road",
    pdd_code="3.2",
    sign_type="blocked_road",
    sign_name="Movement prohibited (3.2)",
    layout_mode="main_main",
    spawn_strategy="blocked_road",
    data_subdir="blocked_road",
    output_code="3_2",
    ego_road_class=None,
)

# 5.7.1 / 5.7.2 share dual-path spawn; separate data pools + pdd_code.
ONE_WAY_RIGHT = SignProfile(
    id="one_way_right",
    pdd_code="5.7.1",
    sign_type="one_way",
    sign_name="Exit onto one-way road (right / 5.7.1)",
    layout_mode="main_main",
    spawn_strategy="one_way",
    data_subdir="one_way_right",
    output_code="5_7_1",
    ego_road_class=None,
)

ONE_WAY_LEFT = SignProfile(
    id="one_way_left",
    pdd_code="5.7.2",
    sign_type="one_way",
    sign_name="Exit onto one-way road (left / 5.7.2)",
    layout_mode="main_main",
    spawn_strategy="one_way",
    data_subdir="one_way_left",
    output_code="5_7_2",
    ego_road_class=None,
)

# 4.1.1–4.1.6 share dual-path spawn; separate data pools + pdd_code.
DIRECTION_STRAIGHT = SignProfile(
    id="direction_straight",
    pdd_code="4.1.1",
    sign_type="direction",
    sign_name="Proceed straight (4.1.1)",
    layout_mode="main_main",
    spawn_strategy="direction",
    data_subdir="direction_straight",
    output_code="4_1_1",
    ego_road_class=None,
)

DIRECTION_RIGHT = SignProfile(
    id="direction_right",
    pdd_code="4.1.2",
    sign_type="direction",
    sign_name="Turn right (4.1.2)",
    layout_mode="main_main",
    spawn_strategy="direction",
    data_subdir="direction_right",
    output_code="4_1_2",
    ego_road_class=None,
)

DIRECTION_LEFT = SignProfile(
    id="direction_left",
    pdd_code="4.1.3",
    sign_type="direction",
    sign_name="Turn left (4.1.3)",
    layout_mode="main_main",
    spawn_strategy="direction",
    data_subdir="direction_left",
    output_code="4_1_3",
    ego_road_class=None,
)

DIRECTION_STRAIGHT_RIGHT = SignProfile(
    id="direction_straight_right",
    pdd_code="4.1.4",
    sign_type="direction",
    sign_name="Straight or right (4.1.4)",
    layout_mode="main_main",
    spawn_strategy="direction",
    data_subdir="direction_straight_right",
    output_code="4_1_4",
    ego_road_class=None,
)

DIRECTION_STRAIGHT_LEFT = SignProfile(
    id="direction_straight_left",
    pdd_code="4.1.5",
    sign_type="direction",
    sign_name="Straight or left (4.1.5)",
    layout_mode="main_main",
    spawn_strategy="direction",
    data_subdir="direction_straight_left",
    output_code="4_1_5",
    ego_road_class=None,
)

DIRECTION_LEFT_RIGHT = SignProfile(
    id="direction_left_right",
    pdd_code="4.1.6",
    sign_type="direction",
    sign_name="Right or left (4.1.6)",
    layout_mode="main_main",
    spawn_strategy="direction",
    data_subdir="direction_left_right",
    output_code="4_1_6",
    ego_road_class=None,
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
)

_REGISTRY: dict[str, SignProfile] = {
    MAIN_ROAD.id: MAIN_ROAD,
    SECONDARY_ROAD.id: SECONDARY_ROAD,
    YIELD.id: YIELD,
    STOP.id: STOP,
    ROUNDABOUT.id: ROUNDABOUT,
    BLOCKED_ROAD.id: BLOCKED_ROAD,
    ONE_WAY_RIGHT.id: ONE_WAY_RIGHT,
    ONE_WAY_LEFT.id: ONE_WAY_LEFT,
    DIRECTION_STRAIGHT.id: DIRECTION_STRAIGHT,
    DIRECTION_RIGHT.id: DIRECTION_RIGHT,
    DIRECTION_LEFT.id: DIRECTION_LEFT,
    DIRECTION_STRAIGHT_RIGHT.id: DIRECTION_STRAIGHT_RIGHT,
    DIRECTION_STRAIGHT_LEFT.id: DIRECTION_STRAIGHT_LEFT,
    DIRECTION_LEFT_RIGHT.id: DIRECTION_LEFT_RIGHT,
    # aliases
    "2.1": MAIN_ROAD,
    "2_1": MAIN_ROAD,
    "main_road": MAIN_ROAD,
    "2.3": SECONDARY_ROAD,
    "2_3": SECONDARY_ROAD,
    "2.3.1": SECONDARY_ROAD,
    "2.3.2": SECONDARY_ROAD,
    "2.3.3": SECONDARY_ROAD,
    "secondary_road": SECONDARY_ROAD,
    "2.4": YIELD,
    "2_4": YIELD,
    "2.5": STOP,
    "2_5": STOP,
    "stop_sign": STOP,
    "4.3": ROUNDABOUT,
    "4_3": ROUNDABOUT,
    "3.2": BLOCKED_ROAD,
    "3_2": BLOCKED_ROAD,
    "blocked_road": BLOCKED_ROAD,
    "5.7.1": ONE_WAY_RIGHT,
    "5_7_1": ONE_WAY_RIGHT,
    "one_way_571": ONE_WAY_RIGHT,
    "5.7.2": ONE_WAY_LEFT,
    "5_7_2": ONE_WAY_LEFT,
    "one_way_572": ONE_WAY_LEFT,
    "4.1.1": DIRECTION_STRAIGHT,
    "4_1_1": DIRECTION_STRAIGHT,
    "4.1.2": DIRECTION_RIGHT,
    "4_1_2": DIRECTION_RIGHT,
    "4.1.3": DIRECTION_LEFT,
    "4_1_3": DIRECTION_LEFT,
    "4.1.4": DIRECTION_STRAIGHT_RIGHT,
    "4_1_4": DIRECTION_STRAIGHT_RIGHT,
    "4.1.5": DIRECTION_STRAIGHT_LEFT,
    "4_1_5": DIRECTION_STRAIGHT_LEFT,
    "4.1.6": DIRECTION_LEFT_RIGHT,
    "4_1_6": DIRECTION_LEFT_RIGHT,
}


def get_profile(sign_id: str) -> SignProfile:
    key = str(sign_id).strip()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted({p.id for p in _PROFILES}))
        raise KeyError(f"Unknown sign {sign_id!r}. Expected one of: {known}") from exc


def list_profiles() -> list[SignProfile]:
    return list(_PROFILES)


def package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_dir(profile: SignProfile) -> Path:
    return package_root() / "data" / profile.data_subdir


def scenes_dir(profile: SignProfile) -> Path:
    return data_dir(profile) / "scenes"


def output_dir(profile: SignProfile) -> Path:
    return data_dir(profile) / "output"
