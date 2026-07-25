"""Registry for PDD 5.15.1 (directions of movement by lanes).

One board over a multi-lane approach: each lane keeps its own allowed turn
set from SUMO topology (same codes as 5.15.2 / pavement arrows).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet


DIRECTION_SIGN_CODES: tuple[str, ...] = ("5.15.1",)
DEFAULT_PDD_CODE = "5.15.1"
SIGN_FAMILY = "lane_direction"


@dataclass(frozen=True)
class DirectionSignSpec:
    """5.15.1 board sign."""

    pdd_code: str
    title: str
    # Empty = per-lane dirs come from map topology, not a fixed class set.
    allowed_dirs: FrozenSet[str]
    class_name: str
    # Preferred catalog roots under pdd-bench/scenes/ (same OSM maps, other signs).
    # Import may still scan all catalogs via --catalogs all.
    catalog_subdirs: tuple[str, ...] = (
        "5.15.2",
        "4.1.1",
        "4.1.2",
        "4.1.3",
        "4.1.4",
        "4.1.5",
        "4.1.6",
    )

    @property
    def output_slug(self) -> str:
        return self.pdd_code.replace(".", "_")

    @property
    def catalog_subdir(self) -> str:
        """Primary catalog folder (first of ``catalog_subdirs``)."""
        return self.catalog_subdirs[0] if self.catalog_subdirs else "5.15.2"


DIRECTION_SIGN_SPECS: dict[str, DirectionSignSpec] = {
    "5.15.1": DirectionSignSpec(
        pdd_code="5.15.1",
        title="Directions of movement by lanes",
        allowed_dirs=frozenset(),
        class_name="LaneDirectionsSign",
    ),
}


def get_direction_sign_spec(pdd_code: str | None = None) -> DirectionSignSpec:
    code = str(pdd_code or DEFAULT_PDD_CODE).strip()
    try:
        return DIRECTION_SIGN_SPECS[code]
    except KeyError as exc:
        known = ", ".join(DIRECTION_SIGN_CODES)
        raise ValueError(f"Unknown lane-direction sign code {code!r}; expected one of: {known}") from exc


def local_scenes_root(base: Path | str, pdd_code: str | None = None) -> Path:
    spec = get_direction_sign_spec(pdd_code)
    return Path(base) / spec.output_slug


def local_core_scenes_root(base: Path | str, pdd_code: str | None = None) -> Path:
    return local_scenes_root(base, pdd_code) / "core"


def is_direction_sign_code(pdd_code: str | None) -> bool:
    return str(pdd_code or "").strip() in DIRECTION_SIGN_SPECS


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


def dirs_allowed_by_sign(pdd_code: str, include_uturn_for_left: bool = True) -> FrozenSet[str]:
    """5.15.1 has no fixed class-wide dirs; returns empty set."""
    del include_uturn_for_left
    return get_direction_sign_spec(pdd_code).allowed_dirs


def resolve_sign_class(pdd_code: str | None = None):
    from traffic_signs.lane_directions_sign import LaneDirectionsSign

    spec = get_direction_sign_spec(pdd_code)
    if spec.class_name != "LaneDirectionsSign":
        raise ValueError(f"Unexpected class {spec.class_name!r} for {spec.pdd_code}")
    return LaneDirectionsSign
