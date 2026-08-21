"""Moscow sign-pool bookkeeping (``moscow_pool.json``) and train/test filtering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

POOL_FILE = "moscow_pool.json"
VALID_SPLITS = frozenset({"all", "train", "test"})


def pool_path(scenes_dir: Path) -> Path:
    return Path(scenes_dir) / POOL_FILE


def load_moscow_pool(scenes_dir: Path) -> Optional[dict]:
    path = pool_path(scenes_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_moscow_pool(scenes_dir: Path, pool: dict) -> None:
    path = pool_path(scenes_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool, indent=2) + "\n", encoding="utf-8")


def scene_split_map(pool: Optional[dict]) -> Dict[str, str]:
    """Map scene_id → train|test from pool records."""
    out: Dict[str, str] = {}
    if not pool:
        return out
    for rec in pool.get("scenes") or []:
        sid = rec.get("scene_id")
        half = rec.get("split")
        if sid and half in ("train", "test"):
            out[str(sid)] = str(half)
    return out


def normalize_split(raw: Any) -> str:
    value = str(raw or "all").strip().lower()
    if value not in VALID_SPLITS:
        raise ValueError(
            f"paths.split must be one of {sorted(VALID_SPLITS)}, got {raw!r}"
        )
    return value


def filter_scene_dirs_by_split(
    scenes: Sequence[Path],
    *,
    split: str,
    scenes_dir: Path,
) -> tuple[List[Path], Dict[str, str], List[str]]:
    """Filter discovered scene dirs by ``paths.split``.

    Returns (filtered_dirs, split_by_scene_id, skipped_unknown_ids).

    When ``split == \"all\"``, returns all scenes; missing pool is OK.
    When ``split`` is train/test, ``moscow_pool.json`` is required.
    """
    split = normalize_split(split)
    pool = load_moscow_pool(scenes_dir)
    split_by_id = scene_split_map(pool)

    if split == "all":
        return list(scenes), split_by_id, []

    if pool is None:
        raise FileNotFoundError(
            f"paths.split={split!r} requires {pool_path(scenes_dir)} "
            "(run python -m traffic_bench.scene_collection materialize first)"
        )

    kept: List[Path] = []
    skipped: List[str] = []
    for scene_dir in scenes:
        half = split_by_id.get(scene_dir.name)
        if half is None:
            skipped.append(scene_dir.name)
            continue
        if half == split:
            kept.append(scene_dir)
    return kept, split_by_id, skipped


def count_splits(scene_ids: Iterable[str], split_by_id: Dict[str, str]) -> Dict[str, int]:
    counts = {"train": 0, "test": 0, "unknown": 0}
    for sid in scene_ids:
        half = split_by_id.get(sid)
        if half in ("train", "test"):
            counts[half] += 1
        else:
            counts["unknown"] += 1
    return counts
