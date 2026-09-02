#!/usr/bin/env python3
"""Global train/test split of Moscow map places (stratified by topology).

Units:
  * junction / roundabout (T/X/O): ``scene_id`` in ``junctions.jsonl``
  * segment (S): ``osm_way_id`` in ``segments.jsonl`` (one way never in both halves)
  * dual_path inherits junction split via ``junction_id`` at assign time

Does not assign scenes to signs — see ``python -m traffic_bench.scene_collection assign``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from traffic_bench.scene_collection.paths import JUNCTIONS_INDEX, SEGMENTS_INDEX, SPLITS

DEFAULT_INDEX = JUNCTIONS_INDEX
DEFAULT_SEGMENTS_INDEX = SEGMENTS_INDEX
DEFAULT_OUT_DIR = SPLITS
DEFAULT_SEED = 42
DEFAULT_TEST_FRAC = 0.2


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_by_shape(
    rows: List[Dict[str, Any]],
    *,
    test_frac: float,
    seed: int,
) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    by_shape: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        shape = str(row.get("shape") or "")
        scene_id = str(row.get("scene_id") or "")
        if shape and scene_id:
            by_shape[shape].append(scene_id)

    rng = random.Random(seed)
    train: Dict[str, List[str]] = {}
    test: Dict[str, List[str]] = {}
    for shape, ids in sorted(by_shape.items()):
        ids = sorted(set(ids))
        rng.shuffle(ids)
        n_test = int(round(len(ids) * test_frac))
        n_test = min(max(n_test, 0), len(ids))
        test[shape] = sorted(ids[:n_test])
        train[shape] = sorted(ids[n_test:])
    return train, test


def split_segments_by_type(
    rows: List[Dict[str, Any]],
    *,
    test_frac: float,
    seed: int,
) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Stratified split on ``osm_way_id`` within each ``segment_type``."""
    by_type: Dict[str, List[str]] = defaultdict(list)
    seen: Dict[str, str] = {}
    for row in rows:
        way = str(row.get("osm_way_id") or "").strip()
        seg_type = str(row.get("segment_type") or "unknown")
        if not way or way in seen:
            continue
        seen[way] = seg_type
        by_type[seg_type].append(way)

    rng = random.Random(seed)
    train: Dict[str, List[str]] = {}
    test: Dict[str, List[str]] = {}
    for seg_type, ways in sorted(by_type.items()):
        ways = sorted(set(ways))
        rng.shuffle(ways)
        n_test = int(round(len(ways) * test_frac))
        n_test = min(max(n_test, 0), len(ways))
        test[seg_type] = sorted(ways[:n_test])
        train[seg_type] = sorted(ways[n_test:])
    return train, test


def _flat(d: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    for key in sorted(d):
        out.extend(d[key])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--segments-index", type=Path, default=DEFAULT_SEGMENTS_INDEX)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    args = ap.parse_args()

    if not args.index.is_file():
        sys.exit(f"ERROR: index not found: {args.index}")

    rows = _load_jsonl(args.index)
    train, test = split_by_shape(rows, test_frac=args.test_frac, seed=args.seed)

    train_flat = set(_flat(train))
    test_flat = set(_flat(test))
    overlap = train_flat & test_flat
    if overlap:
        sys.exit(f"ERROR: junction train/test scene overlap ({len(overlap)} ids)")

    seg_train: Dict[str, List[str]] = {}
    seg_test: Dict[str, List[str]] = {}
    if args.segments_index.is_file():
        seg_rows = _load_jsonl(args.segments_index)
        seg_train, seg_test = split_segments_by_type(
            seg_rows, test_frac=args.test_frac, seed=args.seed
        )
        seg_overlap = set(_flat(seg_train)) & set(_flat(seg_test))
        if seg_overlap:
            sys.exit(f"ERROR: segment train/test osm_way overlap ({len(seg_overlap)} ways)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "seed": args.seed,
        "test_frac": args.test_frac,
        "junction_index": str(args.index.name),
        "segment_index": str(args.segments_index.name) if args.segments_index.is_file() else None,
        "units": {
            "junction": "scene_id",
            "segment": "osm_way_id",
            "dual_path": "junction_id (via junction split at assign)",
        },
        "stratify": {
            "junction": ["shape"],
            "segment": ["segment_type"],
        },
        "counts": {
            "junction_train": {k: len(v) for k, v in sorted(train.items())},
            "junction_test": {k: len(v) for k, v in sorted(test.items())},
            "junction_train_total": len(train_flat),
            "junction_test_total": len(test_flat),
            "segment_train": {k: len(v) for k, v in sorted(seg_train.items())},
            "segment_test": {k: len(v) for k, v in sorted(seg_test.items())},
            "segment_train_total": len(set(_flat(seg_train))),
            "segment_test_total": len(set(_flat(seg_test))),
        },
    }

    (args.out_dir / "train_ids.json").write_text(
        json.dumps({"by_shape": train, "all": sorted(train_flat), "meta": meta}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "test_ids.json").write_text(
        json.dumps({"by_shape": test, "all": sorted(test_flat), "meta": meta}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    if seg_train or seg_test:
        (args.out_dir / "segment_train_ids.json").write_text(
            json.dumps(
                {
                    "by_segment_type": seg_train,
                    "all": sorted(set(_flat(seg_train))),
                    "meta": meta,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (args.out_dir / "segment_test_ids.json").write_text(
            json.dumps(
                {
                    "by_segment_type": seg_test,
                    "all": sorted(set(_flat(seg_test))),
                    "meta": meta,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    (args.out_dir / "split_summary.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta["counts"], indent=2))
    print(f"[split] Wrote {args.out_dir / 'train_ids.json'}")
    print(f"[split] Wrote {args.out_dir / 'test_ids.json'}")
    if seg_train or seg_test:
        print(f"[split] Wrote {args.out_dir / 'segment_train_ids.json'}")
        print(f"[split] Wrote {args.out_dir / 'segment_test_ids.json'}")


if __name__ == "__main__":
    main()
