#!/usr/bin/env python3
"""Train/test split for corridor candidates (all ways, no harvest quota).

Diversity balancing happens later at ``assign`` time per sign. This stage only
guarantees place-disjoint ``osm_way_id`` halves, stratified by subtype.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from traffic_bench.scene_collection.paths import (
    SEGMENT_TEST_IDS,
    SEGMENT_TRAIN_IDS,
    SEGMENTS_INDEX,
    SEGMENTS_SELECTION_SUMMARY,
    SPLITS,
)

DEFAULT_INDEX = SEGMENTS_INDEX
DEFAULT_SUMMARY = SEGMENTS_SELECTION_SUMMARY
DEFAULT_SEED = 42
DEFAULT_TEST_FRAC = 0.2

SUBTYPES = (
    "straight|1",
    "straight|2",
    "straight|3plus",
    "curved|1",
    "curved|2",
    "curved|3plus",
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_subtype(row: Dict[str, Any]) -> str:
    if row.get("subtype"):
        return str(row["subtype"])
    seg = str(row.get("segment_type") or "unknown")
    bucket = str(row.get("lane_bucket") or "")
    if not bucket:
        n = int(row.get("lane_count") or 0)
        bucket = "1" if n <= 1 else ("2" if n == 2 else "3plus")
    return f"{seg}|{bucket}"


def split_segments_by_type(
    rows: Sequence[Dict[str, Any]],
    *,
    test_frac: float,
    seed: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Stratified osm_way split within each segment_type (assign fallback)."""
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


def split_ways_by_subtype(
    rows: Sequence[Dict[str, Any]],
    *,
    test_frac: float,
    seed: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """Stratified osm_way split within each subtype. Returns train/test + way→row."""
    by_subtype: Dict[str, List[str]] = defaultdict(list)
    way_row: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        way = str(row.get("osm_way_id") or "").strip()
        if not way or way in way_row:
            continue
        subtype = _row_subtype(row)
        way_row[way] = row
        by_subtype[subtype].append(way)

    rng = random.Random(seed)
    train: Dict[str, List[str]] = {}
    test: Dict[str, List[str]] = {}
    for subtype, ways in sorted(by_subtype.items()):
        ways = sorted(set(ways))
        rng.shuffle(ways)
        n_test = int(round(len(ways) * test_frac))
        n_test = min(max(n_test, 0), len(ways))
        test[subtype] = sorted(ways[:n_test])
        train[subtype] = sorted(ways[n_test:])
    return train, test, way_row


def _by_segment_type_from_subtype(
    ways_by_subtype: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for subtype, ways in ways_by_subtype.items():
        seg_type = subtype.split("|", 1)[0]
        out[seg_type].extend(ways)
    return {k: sorted(set(v)) for k, v in sorted(out.items())}


def _flat(d: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    for key in sorted(d):
        out.extend(d[key])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    args = ap.parse_args()

    if not args.index.is_file():
        sys.exit(f"ERROR: segments index not found: {args.index}")

    rows = _load_jsonl(args.index)
    train_ways, test_ways, way_row = split_ways_by_subtype(
        rows, test_frac=args.test_frac, seed=args.seed
    )
    train_flat = set(_flat(train_ways))
    test_flat = set(_flat(test_ways))
    overlap = train_flat & test_flat
    if overlap:
        sys.exit(f"ERROR: train/test osm_way overlap ({len(overlap)} ways)")

    meta = {
        "seed": args.seed,
        "test_frac": args.test_frac,
        "unit": "osm_way_id",
        "harvest": "diverse_segment_v2",
        "mode": "full_pool_split",
        "note": "No harvest quota; crop all candidates. Balance at assign.",
        "n_index": len(rows),
        "n_unique_ways": len(way_row),
        "n_train_ways": len(train_flat),
        "n_test_ways": len(test_flat),
        "by_subtype_train": {k: len(v) for k, v in sorted(train_ways.items())},
        "by_subtype_test": {k: len(v) for k, v in sorted(test_ways.items())},
        "train_test_way_overlap": [],
    }

    SPLITS.mkdir(parents=True, exist_ok=True)
    train_by_type = _by_segment_type_from_subtype(train_ways)
    test_by_type = _by_segment_type_from_subtype(test_ways)
    SEGMENT_TRAIN_IDS.write_text(
        json.dumps(
            {
                "by_segment_type": train_by_type,
                "by_subtype": {k: sorted(v) for k, v in sorted(train_ways.items())},
                "all": sorted(train_flat),
                "meta": meta,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    SEGMENT_TEST_IDS.write_text(
        json.dumps(
            {
                "by_segment_type": test_by_type,
                "by_subtype": {k: sorted(v) for k, v in sorted(test_ways.items())},
                "all": sorted(test_flat),
                "meta": meta,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[select_segments] full pool split train={len(train_flat)} test={len(test_flat)}")
    print(f"[select_segments] train by subtype: {meta['by_subtype_train']}")
    print(f"[select_segments] test by subtype: {meta['by_subtype_test']}")
    print(f"[select_segments] Wrote {SEGMENT_TRAIN_IDS}")
    print(f"[select_segments] Wrote {SEGMENT_TEST_IDS}")
    print(f"[select_segments] Wrote {args.summary_out}")


if __name__ == "__main__":
    main()
