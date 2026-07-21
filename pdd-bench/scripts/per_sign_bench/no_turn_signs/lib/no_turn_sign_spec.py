"""Shared registry for PDD 3.18.1 / 3.18.2 / 3.19 (turn-prohibition signs).

Same dual-path incentive as direction signs 4.1.x:
  * baseline = shorter *forbidden* first exit
  * compliant = longer *allowed* first exit

Difference vs 4.1.x: these signs list the *prohibited* maneuver, so
``allowed_dirs`` is the complement of ``forbidden_dir`` among cardinal
directions (and U-turn for 3.19).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet


NO_TURN_SIGN_CODES: tuple[str, ...] = (
    "3.18.1",
    "3.18.2",
    "3.19",
)

DEFAULT_PDD_CODE = "3.18.1"
SIGN_FAMILY = "no_turn"


@dataclass(frozen=True)
class NoTurnSignSpec:
    """One turn-prohibition sign."""

    pdd_code: str
    title: str
    forbidden_dir: str  # "r" | "l" | "t"
    allowed_dirs: FrozenSet[str]
    class_name: str

    @property
    def output_slug(self) -> str:
        """Filesystem-safe folder slug, e.g. ``3_18_1``."""
        return self.pdd_code.replace(".", "_")

    @property
    def catalog_subdir(self) -> str:
        return self.pdd_code


NO_TURN_SIGN_SPECS: dict[str, NoTurnSignSpec] = {
    "3.18.1": NoTurnSignSpec(
        pdd_code="3.18.1",
        title="No right turn",
        forbidden_dir="r",
        allowed_dirs=frozenset({"s", "l"}),
        class_name="NoRightTurnSign",
    ),
    "3.18.2": NoTurnSignSpec(
        pdd_code="3.18.2",
        title="No left turn",
        forbidden_dir="l",
        allowed_dirs=frozenset({"s", "r"}),
        class_name="NoLeftTurnSign",
    ),
    "3.19": NoTurnSignSpec(
        pdd_code="3.19",
        title="No U-turn",
        forbidden_dir="t",
        allowed_dirs=frozenset({"s", "r", "l"}),
        class_name="NoUTurnSign",
    ),
}


def get_no_turn_sign_spec(pdd_code: str | None = None) -> NoTurnSignSpec:
    code = str(pdd_code or DEFAULT_PDD_CODE).strip()
    try:
        return NO_TURN_SIGN_SPECS[code]
    except KeyError as exc:
        known = ", ".join(NO_TURN_SIGN_CODES)
        raise ValueError(f"Unknown no-turn sign code {code!r}; expected one of: {known}") from exc


def local_scenes_root(base: Path | str, pdd_code: str | None = None) -> Path:
    spec = get_no_turn_sign_spec(pdd_code)
    return Path(base) / spec.output_slug


def local_core_scenes_root(base: Path | str, pdd_code: str | None = None) -> Path:
    return local_scenes_root(base, pdd_code) / "core"


def is_no_turn_sign_code(pdd_code: str | None) -> bool:
    return str(pdd_code or "").strip() in NO_TURN_SIGN_SPECS


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
    return get_no_turn_sign_spec(pdd_code).allowed_dirs


def dirs_forbidden_by_sign(pdd_code: str) -> FrozenSet[str]:
    return frozenset({get_no_turn_sign_spec(pdd_code).forbidden_dir})


def resolve_sign_class(pdd_code: str | None = None):
    """Import and return the MetaDrive sign class for ``pdd_code``."""
    from traffic_signs.no_turn_allowed import (
        NoLeftTurnSign,
        NoRightTurnSign,
        NoUTurnSign,
    )

    mapping = {
        "3.18.1": NoRightTurnSign,
        "3.18.2": NoLeftTurnSign,
        "3.19": NoUTurnSign,
    }
    spec = get_no_turn_sign_spec(pdd_code)
    return mapping[spec.pdd_code]
