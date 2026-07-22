"""Shared registry for PDD 5.7.1 / 5.7.2 (one-way entry signs).

Same dual-path incentive as no-turn / direction signs:
  * baseline = shorter *forbidden* first exit
  * compliant = longer *allowed* first exit

5.7.1 marks exit onto a one-way road to the **right** (left entry blocked).
5.7.2 marks exit onto a one-way road to the **left** (right entry blocked).

Sign classes: ``OneWayEntrySignR`` (5.7.1), ``OneWayEntrySignL`` (5.7.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet


ONE_WAY_SIGN_CODES: tuple[str, ...] = (
    "5.7.1",
    "5.7.2",
)

DEFAULT_PDD_CODE = "5.7.1"
SIGN_FAMILY = "one_way"


@dataclass(frozen=True)
class OneWaySignSpec:
    """One one-way-entry sign (5.7.x)."""

    pdd_code: str
    title: str
    forbidden_dir: str  # "l" | "r"  — blocked first exit at the approach
    allowed_dirs: FrozenSet[str]
    class_name: str

    @property
    def output_slug(self) -> str:
        """Filesystem-safe folder slug, e.g. ``5_7_1``."""
        return self.pdd_code.replace(".", "_")

    @property
    def catalog_subdir(self) -> str:
        return self.pdd_code


ONE_WAY_SIGN_SPECS: dict[str, OneWaySignSpec] = {
    "5.7.1": OneWaySignSpec(
        pdd_code="5.7.1",
        title="Exit onto one-way road (right)",
        # OneWayEntrySignR: not_allowed_direction='l'
        forbidden_dir="l",
        allowed_dirs=frozenset({"s", "r"}),
        class_name="OneWayEntrySignR",
    ),
    "5.7.2": OneWaySignSpec(
        pdd_code="5.7.2",
        title="Exit onto one-way road (left)",
        # OneWayEntrySignL: not_allowed_direction='r'
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


def local_scenes_root(base: Path | str, pdd_code: str | None = None) -> Path:
    spec = get_one_way_sign_spec(pdd_code)
    return Path(base) / spec.output_slug


def local_core_scenes_root(base: Path | str, pdd_code: str | None = None) -> Path:
    return local_scenes_root(base, pdd_code) / "core"


def is_one_way_sign_code(pdd_code: str | None) -> bool:
    return str(pdd_code or "").strip() in ONE_WAY_SIGN_SPECS


def normalize_turn_direction(raw_dir: str | None) -> str:
    d = str(raw_dir or "").strip().lower()
    if d in ("r", "right"):
        return "r"
    if d in ("l", "left"):
        return "l"
    if d in ("s", "straight"):
        return "s"
    if d in ("t", "u", "uturn", "u-turn"):
        return "t"
    return d


def dirs_allowed_by_sign(pdd_code: str) -> FrozenSet[str]:
    return get_one_way_sign_spec(pdd_code).allowed_dirs


def dirs_forbidden_by_sign(pdd_code: str) -> FrozenSet[str]:
    return frozenset({get_one_way_sign_spec(pdd_code).forbidden_dir})


def resolve_sign_class(pdd_code: str | None = None):
    """Import and return the MetaDrive sign class for ``pdd_code``."""
    from traffic_signs.one_way_entry_sign import OneWayEntrySignL, OneWayEntrySignR

    mapping = {
        "5.7.1": OneWayEntrySignR,
        "5.7.2": OneWayEntrySignL,
    }
    spec = get_one_way_sign_spec(pdd_code)
    return mapping[spec.pdd_code]
