"""PDD 5.7.1 / 5.7.2 registry (owned by priority_bench).

5.7.1 — exit onto one-way to the right (left first-exit forbidden).
5.7.2 — exit onto one-way to the left (right first-exit forbidden).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


ONE_WAY_SIGN_CODES: tuple[str, ...] = ("5.7.1", "5.7.2")
DEFAULT_PDD_CODE = "5.7.1"

_BASELINE_DIR_ORDER = ("t", "s", "r", "l")
_COMPLIANT_DIR_ORDER = ("s", "r", "l")
_ROUTE_DIRS = frozenset({"s", "r", "l", "t"})


@dataclass(frozen=True)
class OneWaySignSpec:
    pdd_code: str
    title: str
    forbidden_dir: str  # "l" | "r"
    allowed_dirs: FrozenSet[str]
    class_name: str


ONE_WAY_SIGN_SPECS: dict[str, OneWaySignSpec] = {
    "5.7.1": OneWaySignSpec(
        pdd_code="5.7.1",
        title="Exit onto one-way road (right)",
        forbidden_dir="l",
        allowed_dirs=frozenset({"s", "r"}),
        class_name="OneWayEntrySignR",
    ),
    "5.7.2": OneWaySignSpec(
        pdd_code="5.7.2",
        title="Exit onto one-way road (left)",
        forbidden_dir="r",
        allowed_dirs=frozenset({"s", "l"}),
        class_name="OneWayEntrySignL",
    ),
}


def get_one_way_sign_spec(pdd_code: str | None = None) -> OneWaySignSpec:
    code = str(pdd_code or DEFAULT_PDD_CODE).strip()
    try:
        return ONE_WAY_SIGN_SPECS[code]
    except KeyError as exc:
        known = ", ".join(ONE_WAY_SIGN_CODES)
        raise ValueError(
            f"Unknown one-way sign code {code!r}; expected one of: {known}"
        ) from exc


def dirs_allowed_by_sign(pdd_code: str) -> FrozenSet[str]:
    return get_one_way_sign_spec(pdd_code).allowed_dirs


def dirs_forbidden_by_sign(pdd_code: str) -> FrozenSet[str]:
    return frozenset({get_one_way_sign_spec(pdd_code).forbidden_dir})


def dual_path_role_dirs(pdd_code: str) -> tuple[list[str], list[str]]:
    """``(baseline_dirs, compliant_dirs)`` for 5.7.x dual-path roles."""
    forbidden = set(dirs_forbidden_by_sign(pdd_code)) & _ROUTE_DIRS
    allowed = set(dirs_allowed_by_sign(pdd_code)) & _ROUTE_DIRS
    baseline = [d for d in _BASELINE_DIR_ORDER if d in forbidden]
    compliant = [d for d in _COMPLIANT_DIR_ORDER if d in allowed]
    return baseline, compliant


def resolve_sign_class(pdd_code: str | None = None):
    from traffic_signs.one_way_entry_sign import OneWayEntrySignL, OneWayEntrySignR

    mapping = {
        "5.7.1": OneWayEntrySignR,
        "5.7.2": OneWayEntrySignL,
    }
    spec = get_one_way_sign_spec(pdd_code)
    return mapping[spec.pdd_code]
