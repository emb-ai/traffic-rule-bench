"""Tiered place reuse within a train/test split."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Set

from traffic_bench.scene_collection.assign.taxonomy import (
    behavioral_family,
    semantic_group,
)

ReuseTier = Literal[1, 2, 3]


@dataclass(frozen=True)
class SceneCandidate:
    scene_id: str
    place_id: str
    shape: str = ""
    slot: str = ""
    segment_type: str = ""


@dataclass
class SplitPlaceRegistry:
    """Tracks physical place reuse within one split (train or test)."""

    used: Dict[str, Set[str]] = field(default_factory=dict)
    tier_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def tier_for(self, place_id: str, pdd_code: str) -> Optional[ReuseTier]:
        owners = self.used.get(place_id)
        if not owners:
            return 1
        my_beh = behavioral_family(pdd_code)
        my_sem = semantic_group(pdd_code)
        owner_beh = {behavioral_family(s) for s in owners}
        owner_sem = {semantic_group(s) for s in owners}
        if owner_beh == {my_beh}:
            return 2
        if owner_sem == {my_sem} and my_beh not in owner_beh:
            return 3
        if owner_sem == {my_sem}:
            return 2
        return None

    def register(self, place_id: str, pdd_code: str, *, tier: ReuseTier) -> None:
        self.used.setdefault(place_id, set()).add(str(pdd_code))
        self.tier_counts[int(tier)] += 1

    def snapshot(self) -> Dict[str, List[str]]:
        return {pid: sorted(signs) for pid, signs in sorted(self.used.items())}


def pick_tiered(
    candidates: Iterable[SceneCandidate],
    *,
    registry: SplitPlaceRegistry,
    pdd_code: str,
    exclude_scene_ids: Set[str],
    rng: random.Random,
) -> Optional[tuple[SceneCandidate, ReuseTier]]:
    by_tier: Dict[int, List[SceneCandidate]] = {1: [], 2: [], 3: []}
    for cand in candidates:
        if cand.scene_id in exclude_scene_ids:
            continue
        tier = registry.tier_for(cand.place_id, pdd_code)
        if tier is None:
            continue
        by_tier[int(tier)].append(cand)
    for tier in (1, 2, 3):
        pool = by_tier[tier]
        if not pool:
            continue
        chosen = rng.choice(pool)
        return chosen, tier  # type: ignore[return-value]
    return None


def pick_many_tiered(
    candidates: Iterable[SceneCandidate],
    *,
    need: int,
    registry: SplitPlaceRegistry,
    pdd_code: str,
    rng: random.Random,
) -> tuple[List[SceneCandidate], Dict[int, int]]:
    picked: List[SceneCandidate] = []
    used_scenes: Set[str] = set()
    tier_hist: Dict[int, int] = defaultdict(int)
    cand_list = list(candidates)
    for _ in range(need):
        result = pick_tiered(
            cand_list,
            registry=registry,
            pdd_code=pdd_code,
            exclude_scene_ids=used_scenes,
            rng=rng,
        )
        if result is None:
            break
        chosen, tier = result
        picked.append(chosen)
        used_scenes.add(chosen.scene_id)
        registry.register(chosen.place_id, pdd_code, tier=tier)
        tier_hist[int(tier)] += 1
    return picked, dict(tier_hist)
