#!/usr/bin/env python3
"""Global 80/20 train/test split of moscow junctions by scene_id.

Stratified by shape (T / X / O). A scene_id never appears in both halves.
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

from traffic_bench.scene_collection.paths import JUNCTIONS_INDEX, SPLITS

DEFAULT_INDEX = JUNCTIONS_INDEX
DEFAULT_OUT_DIR = SPLITS
DEFAULT_SEED = 42
DEFAULT_TEST_FRAC = 0.2


def _load_index(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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


def _flat(d: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    for shape in sorted(d):
        out.extend(d[shape])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    args = ap.parse_args()

    if not args.index.is_file():
        sys.exit(f"ERROR: index not found: {args.index}")

    rows = _load_index(args.index)
    train, test = split_by_shape(rows, test_frac=args.test_frac, seed=args.seed)

    # Sanity: no overlap
    train_flat = set(_flat(train))
    test_flat = set(_flat(test))
    overlap = train_flat & test_flat
    if overlap:
        sys.exit(f"ERROR: train/test overlap ({len(overlap)} ids)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "seed": args.seed,
        "test_frac": args.test_frac,
        "index_file": str(args.index.name),
        "unit": "scene_id",
        "stratify": ["shape"],
        "counts": {
            "train": {k: len(v) for k, v in sorted(train.items())},
            "test": {k: len(v) for k, v in sorted(test.items())},
            "train_total": len(train_flat),
            "test_total": len(test_flat),
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
    (args.out_dir / "split_summary.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta["counts"], indent=2))
    print(f"[split] Wrote {args.out_dir / 'train_ids.json'}")
    print(f"[split] Wrote {args.out_dir / 'test_ids.json'}")


if __name__ == "__main__":
    main()
