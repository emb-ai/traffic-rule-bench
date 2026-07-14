"""Shared registry for PDD 4.1.1–4.1.6 (mandatory movement direction signs).

All six signs share the same junction scaffolding. They differ only by which
turn directions are allowed for ego (and later by scene/route generation).

Directions use the same codes as ``LaneAllowedDirectionSign`` / ``sumo_env``:
  s = straight, r = right, l = left
  (left-turn signs also permit a U-turn in the rule text; enforcement may map
  that via the sign class, not this folder's route filter.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable


# Canonical codes for the direction-sign family.
DIRECTION_SIGN_CODES: tuple[str, ...] = (
    "4.1.1",
    "4.1.2",
    "4.1.3",
    "4.1.4",
    "4.1.5",
    "4.1.6",
)

DEFAULT_PDD_CODE = "4.1.1"
SIGN_FAMILY = "direction"


@dataclass(frozen=True)
class DirectionSignSpec:
    """One member of the 4.1.x family."""

    pdd_code: str
    title_ru: str
    allowed_dirs: FrozenSet[str]
    # MetaDrive / traffic_signs class basename (without module path).
    class_name: str

    @property
    def output_slug(self) -> str:
        """Filesystem-safe folder slug, e.g. ``4_1_1``."""
        return self.pdd_code.replace(".", "_")

    @property
    def catalog_subdir(self) -> str:
        """Catalog folder under ``pdd-bench/scenes/``."""
        return self.pdd_code


DIRECTION_SIGN_SPECS: dict[str, DirectionSignSpec] = {
    "4.1.1": DirectionSignSpec(
        pdd_code="4.1.1",
        title_ru="Движение прямо",
        allowed_dirs=frozenset({"s"}),
        class_name="LaneAllowedDirectionSign4_1_1",
    ),
    "4.1.2": DirectionSignSpec(
        pdd_code="4.1.2",
        title_ru="Движение направо",
        allowed_dirs=frozenset({"r"}),
        class_name="LaneAllowedDirectionSign4_1_2",
    ),
    "4.1.3": DirectionSignSpec(
        pdd_code="4.1.3",
        title_ru="Движение налево",
        allowed_dirs=frozenset({"l"}),
        class_name="LaneAllowedDirectionSign4_1_3",
    ),
    "4.1.4": DirectionSignSpec(
        pdd_code="4.1.4",
        title_ru="Движение прямо или направо",
        allowed_dirs=frozenset({"s", "r"}),
        class_name="LaneAllowedDirectionSign4_1_4",
    ),
    "4.1.5": DirectionSignSpec(
        pdd_code="4.1.5",
        title_ru="Движение прямо или налево",
        allowed_dirs=frozenset({"s", "l"}),
        class_name="LaneAllowedDirectionSign4_1_5",
    ),
    "4.1.6": DirectionSignSpec(
        pdd_code="4.1.6",
        title_ru="Движение направо или налево",
        allowed_dirs=frozenset({"l", "r"}),
        class_name="LaneAllowedDirectionSign4_1_6",
    ),
}


def get_direction_sign_spec(pdd_code: str | None = None) -> DirectionSignSpec:
    """Resolve a PDD code to its family spec (default: 4.1.1)."""
    code = str(pdd_code or DEFAULT_PDD_CODE).strip()
    try:
        return DIRECTION_SIGN_SPECS[code]
    except KeyError as exc:
        known = ", ".join(DIRECTION_SIGN_CODES)
        raise ValueError(f"Unknown direction sign code {code!r}; expected one of: {known}") from exc


def is_direction_sign_code(pdd_code: str | None) -> bool:
    return str(pdd_code or "").strip() in DIRECTION_SIGN_SPECS


def normalize_turn_direction(raw_dir: str | None) -> str:
    """Normalize turn labels to ``s`` / ``r`` / ``l`` (and ``t`` for U-turn)."""
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


def dirs_allowed_by_sign(pdd_code: str, include_uturn_for_left: bool = True) -> FrozenSet[str]:
    """Allowed movement directions for a sign code.

    Per PDD text, signs that allow left also allow U-turn. When
    ``include_uturn_for_left`` is True, ``t`` is added whenever ``l`` is allowed.
    Scene-generation filters can ignore ``t`` until U-turn routes are supported.
    """
    spec = get_direction_sign_spec(pdd_code)
    dirs = set(spec.allowed_dirs)
    if include_uturn_for_left and "l" in dirs:
        dirs.add("t")
    return frozenset(dirs)


def resolve_sign_class(pdd_code: str | None = None):
    """Import and return the MetaDrive sign class for ``pdd_code``."""
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


def filter_routes_by_allowed_dirs(
    route_dirs: Iterable[str],
    pdd_code: str,
    *,
    include_uturn_for_left: bool = False,
) -> list[str]:
    """Keep route direction labels permitted by the sign.

    Placeholder helper for upcoming scene/manifest generation. Does not yet
    inspect SUMO connectivity — only filters an iterable of direction codes.
    """
    allowed = dirs_allowed_by_sign(pdd_code, include_uturn_for_left=include_uturn_for_left)
    out: list[str] = []
    for raw in route_dirs:
        d = normalize_turn_direction(raw)
        if d in allowed:
            out.append(d)
    return out
