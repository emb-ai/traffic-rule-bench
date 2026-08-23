"""Plate table + crop-meta dual-path pick for 4.1 / 5.7 / 3.18 / 3.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, List, Optional, Sequence

from traffic_bench.eval.signs.dual_path.scene import DualPathScenario, pick_meta_dual_path

_CARDINAL = frozenset({"s", "r", "l"})
_ROUTE = frozenset({"s", "r", "l", "t"})


@dataclass(frozen=True)
class DualPathSignSpec:
    sign_code: str
    family: str
    title: str
    allowed_dirs: FrozenSet[str]
    class_name: str
    forbidden_dir: str = ""

    @property
    def pdd_code(self) -> str:
        return self.sign_code


SPECS: dict[str, DualPathSignSpec] = {
    "5.7.1": DualPathSignSpec(
        sign_code="5.7.1",
        family="one_way",
        title="Exit onto one-way road (right)",
        forbidden_dir="l",
        allowed_dirs=frozenset({"s", "r"}),
        class_name="OneWayEntrySignR",
    ),
    "5.7.2": DualPathSignSpec(
        sign_code="5.7.2",
        family="one_way",
        title="Exit onto one-way road (left)",
        forbidden_dir="r",
        allowed_dirs=frozenset({"s", "l"}),
        class_name="OneWayEntrySignL",
    ),
    "4.1.1": DualPathSignSpec(
        sign_code="4.1.1",
        family="direction",
        title="Proceed straight",
        allowed_dirs=frozenset({"s"}),
        class_name="LaneAllowedDirectionSign4_1_1",
    ),
    "4.1.2": DualPathSignSpec(
        sign_code="4.1.2",
        family="direction",
        title="Turn right",
        allowed_dirs=frozenset({"r"}),
        class_name="LaneAllowedDirectionSign4_1_2",
    ),
    "4.1.3": DualPathSignSpec(
        sign_code="4.1.3",
        family="direction",
        title="Turn left",
        allowed_dirs=frozenset({"l"}),
        class_name="LaneAllowedDirectionSign4_1_3",
    ),
    "4.1.4": DualPathSignSpec(
        sign_code="4.1.4",
        family="direction",
        title="Straight or right",
        allowed_dirs=frozenset({"s", "r"}),
        class_name="LaneAllowedDirectionSign4_1_4",
    ),
    "4.1.5": DualPathSignSpec(
        sign_code="4.1.5",
        family="direction",
        title="Straight or left",
        allowed_dirs=frozenset({"s", "l"}),
        class_name="LaneAllowedDirectionSign4_1_5",
    ),
    "4.1.6": DualPathSignSpec(
        sign_code="4.1.6",
        family="direction",
        title="Right or left",
        allowed_dirs=frozenset({"l", "r"}),
        class_name="LaneAllowedDirectionSign4_1_6",
    ),
    "3.18.1": DualPathSignSpec(
        sign_code="3.18.1",
        family="no_turn",
        title="No right turn",
        forbidden_dir="r",
        allowed_dirs=frozenset({"s", "l"}),
        class_name="NoRightTurnSign",
    ),
    "3.18.2": DualPathSignSpec(
        sign_code="3.18.2",
        family="no_turn",
        title="No left turn",
        forbidden_dir="l",
        allowed_dirs=frozenset({"s", "r"}),
        class_name="NoLeftTurnSign",
    ),
    "3.1": DualPathSignSpec(
        sign_code="3.1",
        family="no_entry",
        title="No entry",
        allowed_dirs=frozenset({"s", "r", "l"}),
        class_name="NoEntrySign",
    ),
}

_FAMILY_DEFAULT = {
    "one_way": "5.7.1",
    "direction": "4.1.1",
    "no_turn": "3.18.1",
    "no_entry": "3.1",
}


def _normalize_code(sign_code: str) -> str:
    code = str(sign_code).strip()
    if code in SPECS:
        return code
    dotted = code.replace("_", ".")
    if dotted in SPECS:
        return dotted
    return code


def get_spec(sign_code: str | None, *, family: str | None = None) -> DualPathSignSpec:
    if sign_code:
        code = _normalize_code(sign_code)
        spec = SPECS.get(code)
        if spec is not None:
            return spec
        if code in _FAMILY_DEFAULT:
            return SPECS[_FAMILY_DEFAULT[code]]
        raise ValueError(
            f"Unknown dual-path sign code {sign_code!r}; "
            f"expected one of: {', '.join(SPECS)}"
        )
    if family and family in _FAMILY_DEFAULT:
        return SPECS[_FAMILY_DEFAULT[family]]
    raise ValueError("get_spec needs a sign code or family=")


def dual_path_role_dirs(sign_code: str) -> tuple[list[str], list[str]]:
    """``(baseline_dirs, compliant_dirs)`` for crop-meta filtering."""
    spec = get_spec(sign_code)
    if spec.family == "one_way":
        forbidden = {spec.forbidden_dir} & _ROUTE
        allowed = set(spec.allowed_dirs) & _ROUTE
        baseline = [d for d in ("t", "s", "r", "l") if d in forbidden]
        compliant = [d for d in ("s", "r", "l") if d in allowed]
        return baseline, compliant
    if spec.family == "direction":
        allowed = set(spec.allowed_dirs) & _CARDINAL
        compliant = [d for d in ("s", "r", "l") if d in allowed]
        baseline = [d for d in ("t", "s", "r", "l") if d in _CARDINAL and d not in allowed]
        return baseline, compliant
    if spec.family == "no_turn":
        forbidden = {spec.forbidden_dir} & _CARDINAL
        allowed = set(spec.allowed_dirs) & _CARDINAL
        baseline = [d for d in ("r", "l", "s") if d in forbidden]
        compliant = [d for d in ("s", "r", "l") if d in allowed]
        return baseline, compliant
    # no_entry: any crop slot
    return [d for d in ("l", "s", "r") if d in _CARDINAL], [
        d for d in ("s", "r", "l") if d in _CARDINAL
    ]


def resolve_sign_class(sign_code: str | None = None, *, family: str | None = None):
    spec = get_spec(sign_code, family=family)
    if spec.family == "one_way":
        from traffic_bench.signs.dual_path.one_way import OneWayEntrySignL, OneWayEntrySignR

        return {"5.7.1": OneWayEntrySignR, "5.7.2": OneWayEntrySignL}[spec.sign_code]
    if spec.family == "direction":
        from traffic_bench.signs.dual_path.direction import (
            LaneAllowedDirectionSign4_1_1,
            LaneAllowedDirectionSign4_1_2,
            LaneAllowedDirectionSign4_1_3,
            LaneAllowedDirectionSign4_1_4,
            LaneAllowedDirectionSign4_1_5,
            LaneAllowedDirectionSign4_1_6,
        )

        return {
            "4.1.1": LaneAllowedDirectionSign4_1_1,
            "4.1.2": LaneAllowedDirectionSign4_1_2,
            "4.1.3": LaneAllowedDirectionSign4_1_3,
            "4.1.4": LaneAllowedDirectionSign4_1_4,
            "4.1.5": LaneAllowedDirectionSign4_1_5,
            "4.1.6": LaneAllowedDirectionSign4_1_6,
        }[spec.sign_code]
    if spec.family == "no_turn":
        from traffic_bench.signs.dual_path.no_turn import NoLeftTurnSign, NoRightTurnSign

        return {"3.18.1": NoRightTurnSign, "3.18.2": NoLeftTurnSign}[spec.sign_code]
    from traffic_bench.signs.dual_path.no_entry import NoEntrySign

    return NoEntrySign


def discover_dual_paths(
    net_path: Path | None = None,
    *,
    pdd_code: str,
    scene_meta: Optional[dict] = None,
    junction_ids: Optional[Sequence[str]] = None,
    min_gain_m: float = 20.0,
    min_lane_length_m: float = 8.0,
    max_scenarios: int = 20,
    arm_counts: Sequence[int] = (3, 4),
) -> List[DualPathScenario]:
    """Load crop-time dual-path from ``meta.json``. Extra kwargs are unused."""
    del net_path, min_gain_m, min_lane_length_m, max_scenarios, arm_counts
    spec = get_spec(pdd_code)
    baseline_dirs, compliant_dirs = dual_path_role_dirs(spec.sign_code)
    return pick_meta_dual_path(
        scene_meta,
        pdd_code=spec.sign_code,
        baseline_dirs=baseline_dirs,
        compliant_dirs=compliant_dirs,
        junction_ids=junction_ids,
    )
