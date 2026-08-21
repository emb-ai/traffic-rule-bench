"""PDD 3.1 registry (owned by priority_bench).

Dual-path: short baseline enters the forbidden road; long compliant detours.
All six moscow slots ``(baseline_dir, compliant_dir)`` are valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


NO_ENTRY_SIGN_CODES: tuple[str, ...] = ("3.1",)
DEFAULT_PDD_CODE = "3.1"

_CARDINAL_DIRS = frozenset({"s", "r", "l"})
_BASELINE_DIR_ORDER = ("l", "s", "r")
_COMPLIANT_DIR_ORDER = ("s", "r", "l")


@dataclass(frozen=True)
class NoEntrySignSpec:
    pdd_code: str
    title: str
    class_name: str
    # Any cardinal may be the short forbidden first exit / long compliant exit.
    allowed_dirs: FrozenSet[str] = frozenset({"s", "r", "l"})


NO_ENTRY_SIGN_SPECS: dict[str, NoEntrySignSpec] = {
    "3.1": NoEntrySignSpec(
        pdd_code="3.1",
        title="No entry",
        class_name="NoEntrySign",
    ),
}


def get_no_entry_sign_spec(pdd_code: str | None = None) -> NoEntrySignSpec:
    code = str(pdd_code or DEFAULT_PDD_CODE).strip()
    try:
        return NO_ENTRY_SIGN_SPECS[code]
    except KeyError as exc:
        known = ", ".join(NO_ENTRY_SIGN_CODES)
        raise ValueError(
            f"Unknown no-entry sign code {code!r}; expected one of: {known}"
        ) from exc


def dual_path_role_dirs(pdd_code: str) -> tuple[list[str], list[str]]:
    """Accept any crop dual-path slot (all six)."""
    del pdd_code
    baseline = [d for d in _BASELINE_DIR_ORDER if d in _CARDINAL_DIRS]
    compliant = [d for d in _COMPLIANT_DIR_ORDER if d in _CARDINAL_DIRS]
    return baseline, compliant


def resolve_sign_class(pdd_code: str | None = None):
    from traffic_bench.signs.no_entry_sign import NoEntrySign

    get_no_entry_sign_spec(pdd_code)  # validate
    return NoEntrySign
