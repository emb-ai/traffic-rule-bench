"""PDD 4.1.1–4.1.6 registry (owned by priority_bench)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


DIRECTION_SIGN_CODES: tuple[str, ...] = (
    "4.1.1",
    "4.1.2",
    "4.1.3",
    "4.1.4",
    "4.1.5",
    "4.1.6",
)
DEFAULT_PDD_CODE = "4.1.1"

_BASELINE_DIR_ORDER = ("t", "s", "r", "l")
_COMPLIANT_DIR_ORDER = ("s", "r", "l")
_CARDINAL_DIRS = frozenset({"s", "r", "l"})


@dataclass(frozen=True)
class DirectionSignSpec:
    pdd_code: str
    title: str
    allowed_dirs: FrozenSet[str]
    class_name: str


DIRECTION_SIGN_SPECS: dict[str, DirectionSignSpec] = {
    "4.1.1": DirectionSignSpec(
        pdd_code="4.1.1",
        title="Proceed straight",
        allowed_dirs=frozenset({"s"}),
        class_name="LaneAllowedDirectionSign4_1_1",
    ),
    "4.1.2": DirectionSignSpec(
        pdd_code="4.1.2",
        title="Turn right",
        allowed_dirs=frozenset({"r"}),
        class_name="LaneAllowedDirectionSign4_1_2",
    ),
    "4.1.3": DirectionSignSpec(
        pdd_code="4.1.3",
        title="Turn left",
        allowed_dirs=frozenset({"l"}),
        class_name="LaneAllowedDirectionSign4_1_3",
    ),
    "4.1.4": DirectionSignSpec(
        pdd_code="4.1.4",
        title="Straight or right",
        allowed_dirs=frozenset({"s", "r"}),
        class_name="LaneAllowedDirectionSign4_1_4",
    ),
    "4.1.5": DirectionSignSpec(
        pdd_code="4.1.5",
        title="Straight or left",
        allowed_dirs=frozenset({"s", "l"}),
        class_name="LaneAllowedDirectionSign4_1_5",
    ),
    "4.1.6": DirectionSignSpec(
        pdd_code="4.1.6",
        title="Right or left",
        allowed_dirs=frozenset({"l", "r"}),
        class_name="LaneAllowedDirectionSign4_1_6",
    ),
}


def get_direction_sign_spec(pdd_code: str | None = None) -> DirectionSignSpec:
    code = str(pdd_code or DEFAULT_PDD_CODE).strip()
    try:
        return DIRECTION_SIGN_SPECS[code]
    except KeyError as exc:
        known = ", ".join(DIRECTION_SIGN_CODES)
        raise ValueError(
            f"Unknown direction sign code {code!r}; expected one of: {known}"
        ) from exc


def dual_path_role_dirs(pdd_code: str) -> tuple[list[str], list[str]]:
    """``(baseline_dirs, compliant_dirs)``: forbidden short vs allowed long."""
    allowed = set(get_direction_sign_spec(pdd_code).allowed_dirs) & _CARDINAL_DIRS
    compliant = [d for d in _COMPLIANT_DIR_ORDER if d in allowed]
    baseline = [d for d in _BASELINE_DIR_ORDER if d in _CARDINAL_DIRS and d not in allowed]
    return baseline, compliant


def resolve_sign_class(pdd_code: str | None = None):
    from traffic_signs.lane_allowed_direction_sign import (
        LaneAllowedDirectionSign4_1_1,
        LaneAllowedDirectionSign4_1_2,
        LaneAllowedDirectionSign4_1_3,
        LaneAllowedDirectionSign4_1_4,
        LaneAllowedDirectionSign4_1_5,
        LaneAllowedDirectionSign4_1_6,
    )

    mapping = {
        "4.1.1": LaneAllowedDirectionSign4_1_1,
        "4.1.2": LaneAllowedDirectionSign4_1_2,
        "4.1.3": LaneAllowedDirectionSign4_1_3,
        "4.1.4": LaneAllowedDirectionSign4_1_4,
        "4.1.5": LaneAllowedDirectionSign4_1_5,
        "4.1.6": LaneAllowedDirectionSign4_1_6,
    }
    spec = get_direction_sign_spec(pdd_code)
    return mapping[spec.pdd_code]
