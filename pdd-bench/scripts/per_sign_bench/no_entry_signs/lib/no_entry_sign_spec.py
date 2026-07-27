"""Shared registry for PDD 3.1 / 3.2 (no-entry / movement-prohibited).

Both members share the same incentive:
  * baseline (idm) drives onto the forbidden road past the sign
  * compliant experts stop strictly before the sign line

Members differ mainly by catalog scenes / core maps and the MetaDrive
sign class used for placement and violation detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


NO_ENTRY_SIGN_CODES: tuple[str, ...] = (
    "3.1",
    "3.2",
)

DEFAULT_PDD_CODE = "3.1"
SIGN_FAMILY = "no_entry"


@dataclass(frozen=True)
class NoEntrySignSpec:
    """One no-entry / movement-prohibited sign."""

    pdd_code: str
    title: str
    class_name: str

    @property
    def output_slug(self) -> str:
        """Filesystem-safe folder slug, e.g. ``3_1``."""
        return self.pdd_code.replace(".", "_")

    @property
    def catalog_subdir(self) -> str:
        return self.pdd_code


NO_ENTRY_SIGN_SPECS: dict[str, NoEntrySignSpec] = {
    "3.1": NoEntrySignSpec(
        pdd_code="3.1",
        title="No entry",
        class_name="NoEntrySign",
    ),
    "3.2": NoEntrySignSpec(
        pdd_code="3.2",
        title="Movement prohibited",
        class_name="NoTrafficSign",
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


def local_scenes_root(base: Path | str, pdd_code: str | None = None) -> Path:
    spec = get_no_entry_sign_spec(pdd_code)
    return Path(base) / spec.output_slug


def local_core_scenes_root(base: Path | str, pdd_code: str | None = None) -> Path:
    return local_scenes_root(base, pdd_code) / "core"


def is_no_entry_sign_code(pdd_code: str | None) -> bool:
    return str(pdd_code or "").strip() in NO_ENTRY_SIGN_SPECS


def resolve_sign_class(pdd_code: str | None = None):
    """Import and return the MetaDrive sign class for ``pdd_code``."""
    from traffic_signs.no_entry_sign import NoEntrySign
    from traffic_signs.no_traffic_sign import NoTrafficSign

    mapping = {
        "3.1": NoEntrySign,
        "3.2": NoTrafficSign,
    }
    spec = get_no_entry_sign_spec(pdd_code)
    return mapping[spec.pdd_code]
