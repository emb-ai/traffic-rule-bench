"""PDD 3.18.1 / 3.18.2 registry (owned by priority_bench).

Baseline = shorter *forbidden* first exit; compliant = longer *allowed* exit.
``allowed_dirs`` is the complement of ``forbidden_dir`` among {s,r,l}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


NO_TURN_SIGN_CODES: tuple[str, ...] = ("3.18.1", "3.18.2")
DEFAULT_PDD_CODE = "3.18.1"

_BASELINE_DIR_ORDER = ("r", "l", "s")
_COMPLIANT_DIR_ORDER = ("s", "r", "l")
_CARDINAL_DIRS = frozenset({"s", "r", "l"})


@dataclass(frozen=True)
class NoTurnSignSpec:
    pdd_code: str
    title: str
    forbidden_dir: str
    allowed_dirs: FrozenSet[str]
    class_name: str


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
}


def get_no_turn_sign_spec(pdd_code: str | None = None) -> NoTurnSignSpec:
    code = str(pdd_code or DEFAULT_PDD_CODE).strip()
    try:
        return NO_TURN_SIGN_SPECS[code]
    except KeyError as exc:
        known = ", ".join(NO_TURN_SIGN_CODES)
        raise ValueError(
            f"Unknown no-turn sign code {code!r}; expected one of: {known}"
        ) from exc


def dual_path_role_dirs(pdd_code: str) -> tuple[list[str], list[str]]:
    """``(baseline_dirs, compliant_dirs)`` for crop-meta filtering."""
    spec = get_no_turn_sign_spec(pdd_code)
    forbidden = {spec.forbidden_dir} & _CARDINAL_DIRS
    allowed = set(spec.allowed_dirs) & _CARDINAL_DIRS
    baseline = [d for d in _BASELINE_DIR_ORDER if d in forbidden]
    compliant = [d for d in _COMPLIANT_DIR_ORDER if d in allowed]
    return baseline, compliant


def resolve_sign_class(pdd_code: str | None = None):
    from pdd_bench.signs.no_turn_allowed import NoLeftTurnSign, NoRightTurnSign

    mapping = {
        "3.18.1": NoRightTurnSign,
        "3.18.2": NoLeftTurnSign,
    }
    spec = get_no_turn_sign_spec(pdd_code)
    return mapping[spec.pdd_code]
