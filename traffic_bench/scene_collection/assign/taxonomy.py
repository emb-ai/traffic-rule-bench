"""Semantic groups, behavioral families, and topology compatibility for assign."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Literal, Optional, Tuple

SemanticGroup = Literal["priority", "speed", "obstacle", "reroute"]
BehavioralFamily = Literal[
    "junction_priority",
    "direction_control",
    "turn_restriction",
    "obstacle_avoidance",
    "access_road_direction",
    "speed_control",
    "pedestrian_crossing",
    "roundabout",
    "lane_restriction",
]
TopologyKind = Literal["T", "X", "O", "dual_path", "segment"]

SEMANTIC_GROUP: Dict[str, SemanticGroup] = {
    "2.1": "priority",
    "2.3": "priority",
    "2.4": "priority",
    "2.5": "priority",
    "4.3": "priority",
    "5.19": "priority",
    "3.24": "speed",
    "4.6": "speed",
    "5.21": "speed",
    "5.31": "speed",
    "3.2": "obstacle",
    "4.2.1": "obstacle",
    "4.2.2": "obstacle",
    "4.2.3": "obstacle",
    "3.1": "reroute",
    "3.18.1": "reroute",
    "3.18.2": "reroute",
    "4.1.1": "reroute",
    "4.1.2": "reroute",
    "4.1.3": "reroute",
    "4.1.4": "reroute",
    "4.1.5": "reroute",
    "4.1.6": "reroute",
    "5.7.1": "reroute",
    "5.7.2": "reroute",
    "5.14.1": "obstacle",
    "5.14.2": "obstacle",
    "5.11.1": "obstacle",
    "5.11.2": "obstacle",
}

BEHAVIORAL_FAMILY: Dict[str, BehavioralFamily] = {
    "2.1": "junction_priority",
    "2.3": "junction_priority",
    "2.4": "junction_priority",
    "2.5": "junction_priority",
    "4.1.1": "direction_control",
    "4.1.2": "direction_control",
    "4.1.3": "direction_control",
    "4.1.4": "direction_control",
    "4.1.5": "direction_control",
    "4.1.6": "direction_control",
    "3.18.1": "turn_restriction",
    "3.18.2": "turn_restriction",
    "4.2.1": "obstacle_avoidance",
    "4.2.2": "obstacle_avoidance",
    "4.2.3": "obstacle_avoidance",
    "3.1": "access_road_direction",
    "3.2": "access_road_direction",
    "5.7.1": "access_road_direction",
    "5.7.2": "access_road_direction",
    "3.24": "speed_control",
    "4.6": "speed_control",
    "5.21": "speed_control",
    "5.31": "speed_control",
    "5.19": "pedestrian_crossing",
    "4.3": "roundabout",
    "5.14.1": "lane_restriction",
    "5.14.2": "lane_restriction",
    "5.11.1": "lane_restriction",
    "5.11.2": "lane_restriction",
}

# Process signs in this order so unique places go to early families first.
SIGN_ALLOC_ORDER: Tuple[str, ...] = (
    "4.3",
    "2.1",
    "2.3",
    "2.4",
    "2.5",
    "5.19",
    "3.24",
    "4.6",
    "5.21",
    "5.31",
    "3.2",
    "4.2.1",
    "4.2.2",
    "4.2.3",
    "3.1",
    "3.18.1",
    "3.18.2",
    "4.1.1",
    "4.1.2",
    "4.1.3",
    "4.1.4",
    "4.1.5",
    "4.1.6",
    "5.7.1",
    "5.7.2",
    "5.14.1",
    "5.14.2",
    "5.11.1",
    "5.11.2",
)

SEMANTIC_GROUP_ORDER: Tuple[SemanticGroup, ...] = (
    "priority",
    "speed",
    "obstacle",
    "reroute",
)


@dataclass(frozen=True)
class SignTaxonomy:
    pdd_code: str
    semantic_group: SemanticGroup
    behavioral_family: BehavioralFamily
    crop_kind: str
    topologies: FrozenSet[TopologyKind]


def semantic_group(pdd_code: str) -> SemanticGroup:
    try:
        return SEMANTIC_GROUP[str(pdd_code)]
    except KeyError as exc:
        raise KeyError(f"unknown PDD code for semantic group: {pdd_code!r}") from exc


def behavioral_family(pdd_code: str) -> BehavioralFamily:
    try:
        return BEHAVIORAL_FAMILY[str(pdd_code)]
    except KeyError as exc:
        raise KeyError(f"unknown PDD code for behavioral family: {pdd_code!r}") from exc


def compatible_topologies(*, crop_kind: str, shapes: List[str]) -> FrozenSet[TopologyKind]:
    kind = str(crop_kind or "junction").strip().lower()
    if kind == "segment":
        return frozenset({"segment"})
    if kind == "dual_path":
        return frozenset({"dual_path", *{s.upper() for s in shapes if s.upper() in {"T", "X"}}})
    return frozenset({s.upper() for s in shapes if s.upper() in {"T", "X", "O"}})


def sign_taxonomy(pdd_code: str, spec: dict) -> SignTaxonomy:
    spec = spec or {}
    shapes = [str(s).upper() for s in (spec.get("shapes") or ["T", "X"])]
    crop_kind = str(spec.get("crop_kind") or "junction").strip().lower()
    return SignTaxonomy(
        pdd_code=str(pdd_code),
        semantic_group=semantic_group(pdd_code),
        behavioral_family=behavioral_family(pdd_code),
        crop_kind=crop_kind,
        topologies=compatible_topologies(crop_kind=crop_kind, shapes=shapes),
    )


def sign_sort_key(pdd_code: str) -> Tuple[int, int, str]:
    order = {code: i for i, code in enumerate(SIGN_ALLOC_ORDER)}
    sem_order = {g: i for i, g in enumerate(SEMANTIC_GROUP_ORDER)}
    code = str(pdd_code)
    return (
        order.get(code, 999),
        sem_order.get(SEMANTIC_GROUP.get(code, "reroute"), 99),
        code,
    )
