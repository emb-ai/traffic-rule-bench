"""Read indexes and crop folders. No plotting."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from traffic_bench.scene_collection.collect.dual_path.roles import SLOTS
from traffic_bench.scene_collection.collect.lib.io import load_index
from traffic_bench.scene_collection.paths import (
    DUAL_PATH_CANDIDATES,
    DUAL_PATH_CROPS,
    JUNCTION_CROPS,
    JUNCTIONS_INDEX,
    SEGMENT_CROPS,
    SEGMENTS_INDEX,
    TEST_IDS,
    TRAIN_IDS,
)

JUNCTION_SHAPES = ("T", "X", "O")
DUAL_PATH_SHAPES = ("T", "X")


def _has_net(scene_dir: Path) -> bool:
    return scene_dir.is_dir() and (scene_dir / "map.net.xml").is_file()


def _count_scene_dirs(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for child in root.iterdir() if _has_net(child))


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _latlon(row: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    try:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
    except (KeyError, TypeError, ValueError):
        return None
    return lat, lon


@dataclass(frozen=True)
class FamilyTally:
    index: int
    on_disk: int

    @property
    def coverage(self) -> float:
        return 0.0 if self.index == 0 else self.on_disk / self.index


@dataclass
class HarvestSnapshot:
    """One consistent read of the Moscow harvest."""

    junction_rows: List[Dict[str, Any]]
    segment_rows: List[Dict[str, Any]]
    dual_path_rows: List[Dict[str, Any]]
    junction_on_disk: Dict[str, int]
    dual_path_on_disk: Dict[Tuple[str, str], int]
    segment_on_disk: int
    train_ids: Dict[str, List[str]] = field(default_factory=dict)
    test_ids: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def junction(self) -> FamilyTally:
        return FamilyTally(len(self.junction_rows), sum(self.junction_on_disk.values()))

    @property
    def dual_path(self) -> FamilyTally:
        return FamilyTally(len(self.dual_path_rows), sum(self.dual_path_on_disk.values()))

    @property
    def segment(self) -> FamilyTally:
        return FamilyTally(len(self.segment_rows), self.segment_on_disk)

    def families(self) -> Dict[str, FamilyTally]:
        return {
            "junction": self.junction,
            "dual_path": self.dual_path,
            "segment": self.segment,
        }

    def junction_index_by_shape(self) -> Counter:
        return Counter(str(r.get("shape") or "") for r in self.junction_rows)

    def dual_path_index_by_cell(self) -> Dict[Tuple[str, str], int]:
        counts: Dict[Tuple[str, str], int] = {}
        for row in self.dual_path_rows:
            shape = str(row.get("shape") or "")
            slot = str(row.get("slot") or "")
            if shape and slot:
                counts[(shape, slot)] = counts.get((shape, slot), 0) + 1
        return counts

    def junction_geo(self) -> List[Tuple[float, float, str]]:
        out: List[Tuple[float, float, str]] = []
        for row in self.junction_rows:
            xy = _latlon(row)
            if xy is None:
                continue
            out.append((xy[0], xy[1], str(row.get("shape") or "?")))
        return out

    def segment_geo(self) -> List[Tuple[float, float, str]]:
        out: List[Tuple[float, float, str]] = []
        for row in self.segment_rows:
            xy = _latlon(row)
            if xy is None:
                continue
            out.append((xy[0], xy[1], str(row.get("segment_type") or "?")))
        return out

    def dual_path_geo(self) -> List[Tuple[float, float, str]]:
        by_jid = {
            str(r.get("junction_id") or ""): r
            for r in self.junction_rows
            if r.get("junction_id")
        }
        out: List[Tuple[float, float, str]] = []
        seen: set[str] = set()
        for row in self.dual_path_rows:
            jid = str(row.get("junction_id") or "")
            if not jid or jid in seen:
                continue
            parent = by_jid.get(jid)
            if parent is None:
                continue
            xy = _latlon(parent)
            if xy is None:
                continue
            seen.add(jid)
            out.append((xy[0], xy[1], str(row.get("shape") or "?")))
        return out


def count_junction_crops(root: Path = JUNCTION_CROPS) -> Dict[str, int]:
    return {shape: _count_scene_dirs(root / shape) for shape in JUNCTION_SHAPES}


def count_dual_path_crops(root: Path = DUAL_PATH_CROPS) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = {}
    for shape in DUAL_PATH_SHAPES:
        for slot in SLOTS:
            counts[(shape, slot)] = _count_scene_dirs(root / shape / slot)
    return counts


def count_segment_crops(root: Path = SEGMENT_CROPS) -> int:
    return _count_scene_dirs(root)


def _ids_by_shape(payload: Any) -> Dict[str, List[str]]:
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("by_shape", payload)
    if not isinstance(raw, dict):
        return {}
    return {str(k): [str(x) for x in v] for k, v in raw.items() if isinstance(v, list)}


def load_snapshot() -> HarvestSnapshot:
    return HarvestSnapshot(
        junction_rows=load_index(JUNCTIONS_INDEX) if JUNCTIONS_INDEX.is_file() else [],
        segment_rows=load_index(SEGMENTS_INDEX) if SEGMENTS_INDEX.is_file() else [],
        dual_path_rows=load_index(DUAL_PATH_CANDIDATES) if DUAL_PATH_CANDIDATES.is_file() else [],
        junction_on_disk=count_junction_crops(),
        dual_path_on_disk=count_dual_path_crops(),
        segment_on_disk=count_segment_crops(),
        train_ids=_ids_by_shape(_load_json(TRAIN_IDS)),
        test_ids=_ids_by_shape(_load_json(TEST_IDS)),
    )


def scene_example_dirs(
    snap: HarvestSnapshot,
    *,
    n_per_group: int = 2,
    seed: int = 0,
) -> Dict[str, List[Path]]:
    """Deterministic sample of cropped scenes that have a net (for the gallery)."""
    import random

    rng = random.Random(seed)
    picked: Dict[str, List[Path]] = {}

    def _sample(label: str, dirs: Sequence[Path]) -> None:
        ready = [p for p in dirs if _has_net(p)]
        ready.sort(key=lambda p: p.name)
        if len(ready) > n_per_group:
            ready = rng.sample(ready, n_per_group)
        if ready:
            picked[label] = ready

    for shape in JUNCTION_SHAPES:
        root = JUNCTION_CROPS / shape
        _sample(
            f"junction/{shape}",
            [root / str(r["scene_id"]) for r in snap.junction_rows if r.get("shape") == shape and r.get("scene_id")],
        )
    for shape in DUAL_PATH_SHAPES:
        dirs = [
            DUAL_PATH_CROPS / shape / str(r["slot"]) / str(r["scene_id"])
            for r in snap.dual_path_rows
            if r.get("shape") == shape and r.get("slot") and r.get("scene_id")
        ]
        _sample(f"dual_path/{shape}", dirs)
    for kind in ("straight", "curved"):
        dirs = [
            SEGMENT_CROPS / str(r["scene_id"])
            for r in snap.segment_rows
            if r.get("segment_type") == kind and r.get("scene_id")
        ]
        _sample(f"segment/{kind}", dirs)
    return picked


def summary_dict(snap: HarvestSnapshot) -> Dict[str, Any]:
    j_shape = snap.junction_index_by_shape()
    seg_type = Counter(str(r.get("segment_type") or "") for r in snap.segment_rows)
    lanes = Counter(int(r.get("lane_count") or 0) for r in snap.segment_rows)
    return {
        "families": {
            name: {"index": t.index, "on_disk": t.on_disk, "coverage": round(t.coverage, 4)}
            for name, t in snap.families().items()
        },
        "junction_by_shape": {
            "index": dict(j_shape),
            "on_disk": dict(snap.junction_on_disk),
        },
        "dual_path_by_slot": {
            "index": {f"{a}/{b}": n for (a, b), n in sorted(snap.dual_path_index_by_cell().items())},
            "on_disk": {f"{a}/{b}": n for (a, b), n in sorted(snap.dual_path_on_disk.items())},
        },
        "segment": {
            "by_type": dict(seg_type),
            "by_lane_count": {str(k): v for k, v in sorted(lanes.items())},
            "pass_right_ok": sum(1 for r in snap.segment_rows if r.get("pass_right_ok")),
            "pass_left_ok": sum(1 for r in snap.segment_rows if r.get("pass_left_ok")),
            "n_osm_ways": len({str(r.get("osm_way_id") or "") for r in snap.segment_rows if r.get("osm_way_id")}),
        },
        "split": {
            "train": {k: len(v) for k, v in snap.train_ids.items()},
            "test": {k: len(v) for k, v in snap.test_ids.items()},
        },
    }
