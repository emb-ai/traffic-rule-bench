"""Discover scene dirs, apply split / caps, write ``real_manifest.jsonl`` + ``repro/``."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from traffic_bench.scene_collection.paths import SIGN_ALLOCATIONS, SIGNS_YAML
from traffic_bench.scene_collection.sign_scenes.filter.selection import (
    is_reserved_scene_dir,
    load_scene_selection,
    unapplied_rejected_scenes,
)
from traffic_bench.scene_collection.sign_scenes.materialize.pool_index import (
    count_splits,
    filter_scene_dirs_by_split,
    load_moscow_pool,
    normalize_split,
    pool_path,
)

DEFAULT_ALLOCATIONS = SIGN_ALLOCATIONS
DEFAULT_SIGNS_YAML = SIGNS_YAML


def discover_scenes(scenes_dir: Path) -> List[Path]:
    """Find all valid scene directories containing ``meta.json`` and a net file."""
    scenes = []
    for entry in sorted(scenes_dir.iterdir()):
        if not entry.is_dir() or is_reserved_scene_dir(entry.name):
            continue
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        net_file = meta.get("net_file", "map.net.xml")
        net_path = entry / net_file
        if net_path.exists():
            scenes.append(entry)
    return scenes


def assert_rejected_scenes_applied(scenes_dir: Path) -> None:
    """Fail if review rejects were not moved aside with ``--apply``."""
    pending = unapplied_rejected_scenes(scenes_dir)
    if not pending:
        return
    preview = ", ".join(pending[:8])
    more = f" (+{len(pending) - 8} more)" if len(pending) > 8 else ""
    raise SystemExit(
        f"[error] {len(pending)} scene(s) are marked reject in scene_selection.json "
        f"but still live under {scenes_dir.resolve()}.\n"
        f"  Run: python -m traffic_bench.scene_collection review --apply\n"
        f"  Pending: {preview}{more}"
    )


def load_scene_metadata(scene_dir: Path) -> Dict:
    """Load scene metadata from ``meta.json`` (lat/lon live there, not center.json)."""
    meta_path = scene_dir / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_split_filter(
    all_scenes: Sequence[Path],
    *,
    scenes_dir: Path,
    split: str,
) -> Tuple[List[Path], Dict[str, str]]:
    """Keep scenes in ``split``; log unknown ids. Raises ``SystemExit`` if the pool is missing."""
    split = normalize_split(split)
    try:
        scenes, split_by_id, skipped_unknown = filter_scene_dirs_by_split(
            list(all_scenes), split=split, scenes_dir=scenes_dir
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"[error] {exc}") from exc

    if skipped_unknown:
        preview = ", ".join(skipped_unknown[:8])
        more = f" (+{len(skipped_unknown) - 8} more)" if len(skipped_unknown) > 8 else ""
        print(
            f"  [split] Skipping {len(skipped_unknown)} scene(s) not in "
            f"{pool_path(scenes_dir).name}: {preview}{more}"
        )
    print(f"Split filter: {split} → {len(scenes)} scene(s)")
    return scenes, split_by_id


def append_scene_entries(
    entries: List[Dict],
    used_scene_ids: List[str],
    scene_entries: List[Dict],
    *,
    scene_dir: Path,
    meta: Dict,
    split_by_id: Dict[str, str],
) -> None:
    """Stamp ``split`` on rows and record the scene id."""
    scene_name = meta.get("scene_name", scene_dir.name)
    scene_split = split_by_id.get(scene_name) or split_by_id.get(scene_dir.name)
    for entry in scene_entries:
        entry["split"] = scene_split
    entries.extend(scene_entries)
    used_scene_ids.append(scene_dir.name)


def apply_max_total(
    entries: List[Dict],
    used_scene_ids: List[str],
    *,
    max_total: Optional[int],
    split: str,
    pdd_code: str,
    scene_id_key: str = "scene_id",
    log_under_cap: bool = False,
) -> Tuple[List[Dict], List[str], int]:
    """Shuffle-cap the global row list. Returns ``(entries, used_scene_ids, pre_total)``."""
    pre_total = len(entries)
    if max_total is not None and max_total >= 0 and pre_total > max_total:
        rng = random.Random(
            hash(("max_total_shuffle", int(max_total), split, pdd_code)) & 0xFFFFFFFF
        )
        rng.shuffle(entries)
        entries = entries[: int(max_total)]
        used_scene_ids = sorted(
            {str(e.get(scene_id_key)) for e in entries if e.get(scene_id_key)}
        )
        print(
            f"[max_total] Retained {len(entries)} of {pre_total} manifest entries "
            f"(shuffled, cap={max_total})"
        )
    elif log_under_cap and max_total is not None:
        print(
            f"[max_total] Manifest entries: {pre_total} "
            f"(under/at cap={max_total}, no trim)"
        )
    return entries, used_scene_ids, pre_total


def _file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_signs_quota(signs_yaml: Path = DEFAULT_SIGNS_YAML) -> dict:
    if not signs_yaml.is_file():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    with signs_yaml.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_repro_artifacts(
    *,
    output_dir: Path,
    scenes_dir: Path,
    split_filter: str,
    used_scene_ids: List[str],
    split_by_id: Dict[str, str],
    pdd_code: str,
) -> Path:
    """Write ``repro/`` snapshot for experiment reproduction."""
    repro_dir = output_dir / "repro"
    repro_dir.mkdir(parents=True, exist_ok=True)

    pool = load_moscow_pool(scenes_dir)
    selection = load_scene_selection(scenes_dir)
    pool_snapshot = {
        "scenes_dir": str(scenes_dir.resolve()),
        "moscow_pool": pool,
        "scene_selection": selection,
    }
    (repro_dir / "pool_snapshot.json").write_text(
        json.dumps(pool_snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    split_filter_doc = {
        "split": split_filter,
        "n_scenes": len(used_scene_ids),
        "scene_ids": list(used_scene_ids),
        "counts": count_splits(used_scene_ids, split_by_id),
    }
    (repro_dir / "split_filter.json").write_text(
        json.dumps(split_filter_doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    signs_cfg = _load_signs_quota()
    n_train = int(signs_cfg.get("n_train", 115)) if signs_cfg else None
    test_frac = float(signs_cfg.get("test_frac", 0.2)) if signs_cfg else None
    n_test = None
    if signs_cfg:
        n_test = signs_cfg.get("n_test")
        if n_test is None and n_train is not None and test_frac is not None:
            n_test = max(1, round(n_train * test_frac / (1.0 - test_frac)))
        else:
            n_test = int(n_test) if n_test is not None else None
        sign_spec = (signs_cfg.get("signs") or {}).get(str(pdd_code)) or {}
        if "n_train" in sign_spec:
            n_train = int(sign_spec["n_train"])
        if "n_test" in sign_spec:
            n_test = int(sign_spec["n_test"])

    alloc_path = DEFAULT_ALLOCATIONS
    allocations_ref = {
        "allocations_path": str(alloc_path.resolve()) if alloc_path.exists() else str(alloc_path),
        "allocations_sha256": _file_sha256(alloc_path),
        "signs_yaml": (
            str(DEFAULT_SIGNS_YAML.resolve())
            if DEFAULT_SIGNS_YAML.exists()
            else str(DEFAULT_SIGNS_YAML)
        ),
        "seed": signs_cfg.get("seed") if signs_cfg else None,
        "n_train": n_train,
        "n_test": n_test,
        "test_frac": test_frac,
        "pdd_code": pdd_code,
    }
    (repro_dir / "allocations_ref.json").write_text(
        json.dumps(allocations_ref, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return repro_dir


def write_real_manifest(
    *,
    output_dir: Path,
    scenes_dir: Path,
    entries: List[Dict],
    used_scene_ids: List[str],
    split_by_id: Dict[str, str],
    split: str,
    pdd_code: str,
    summary: Dict,
    announce: bool = True,
) -> Path:
    """Write ``real_manifest.jsonl``, summary, ``manifest.json``, and ``repro/``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "real_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, default=str) + "\n")

    if "split_counts" not in summary:
        summary["split_counts"] = count_splits(used_scene_ids, split_by_id)
    summary.setdefault("split_filter", split)
    summary.setdefault("generated_at", datetime.now().isoformat())
    summary.setdefault("scenes", list(used_scene_ids))

    summary_path = output_dir / "real_manifest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    manifest_meta_path = output_dir / "manifest.json"
    with open(manifest_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"entries_file": "real_manifest.jsonl", **summary},
            f,
            indent=2,
            ensure_ascii=False,
        )

    repro_dir = write_repro_artifacts(
        output_dir=output_dir,
        scenes_dir=scenes_dir,
        split_filter=split,
        used_scene_ids=used_scene_ids,
        split_by_id=split_by_id,
        pdd_code=pdd_code,
    )
    print(f"Wrote repro artifacts → {repro_dir}")
    if announce:
        print(
            f"\nGenerated {len(entries)} manifest entries from {len(used_scene_ids)} scenes"
        )
        print(f"  Manifest: {manifest_path}")
    return manifest_path
