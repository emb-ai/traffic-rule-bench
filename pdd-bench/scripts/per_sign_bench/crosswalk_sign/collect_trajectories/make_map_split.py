#!/usr/bin/env python3
"""Split a crosswalk catalog/manifest 80/20 by unique MAPS (net_path).

Colleague equivalent: make_fv_map_split.py (A6 filter + stratified map split).
For crosswalk we do not yet gate on A6 (needs filtered metrics CSV); we split all
maps in the input catalog. A map never appears in both halves.

Stratification is by sign_code only (crosswalk has no v_target_kmh).

Outputs (next to the catalog, or at --out-prefix):
    <prefix>train80.jsonl   <prefix>test20.jsonl   <prefix>maps_split.json

Usage:
    python3 make_map_split.py \\
        --catalog ../benchmark_output/5_19/final_metrics_v1/real_manifest.jsonl

Then collect trajectories with:
    MANIFEST=.../catalog_train80.jsonl bash collect_trajectories.sh
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--catalog",
        required=True,
        help="catalog.jsonl or real_manifest.jsonl (must have net_path)",
    )
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-prefix",
        default=None,
        help="prefix for outputs (default: <catalog_dir>/catalog_)",
    )
    args = ap.parse_args()

    cat_p = Path(args.catalog)
    if not cat_p.is_file():
        sys.exit(f"ERROR: catalog not found: {cat_p}")
    prefix = (
        Path(args.out_prefix) if args.out_prefix else cat_p.with_name("catalog_")
    )

    rows_by_map: dict[str, list] = collections.defaultdict(list)
    for line in open(cat_p, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("valid") is False:
            continue
        net = r.get("net_path")
        if not net:
            continue
        rows_by_map[str(net)].append((line, r))

    maps_all = sorted(rows_by_map)
    if not maps_all:
        sys.exit("ERROR: no rows with net_path in catalog")

    # Stratify by sign_code (fallback "5.19")
    groups: dict[tuple, list] = collections.defaultdict(list)
    for m in maps_all:
        signs = tuple(
            sorted({str(r.get("sign_code") or r.get("pdd_code") or "5.19")
                    for _, r in rows_by_map[m]})
        )
        groups[signs].append(m)

    rng = random.Random(args.seed)
    for ms in groups.values():
        ms.sort()
        rng.shuffle(ms)

    # Largest-remainder quotas so small strata still contribute to test.
    quota = {b: len(ms) * args.test_frac for b, ms in groups.items()}
    take = {b: int(q) for b, q in quota.items()}
    target = round(len(maps_all) * args.test_frac)
    for b in sorted(groups, key=lambda b: quota[b] - take[b], reverse=True):
        if sum(take.values()) >= target:
            break
        if take[b] < len(groups[b]):
            take[b] += 1

    test_maps: set[str] = set()
    for b, ms in groups.items():
        test_maps.update(ms[: take[b]])

    train_p = prefix.with_name(prefix.name + "train80.jsonl")
    test_p = prefix.with_name(prefix.name + "test20.jsonl")
    split_p = prefix.with_name(prefix.name + "maps_split.json")

    stat: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    with open(train_p, "w", encoding="utf-8") as ftr, open(
        test_p, "w", encoding="utf-8"
    ) as fte:
        for m in maps_all:
            dst, tag = (fte, "te") if m in test_maps else (ftr, "tr")
            for line, r in rows_by_map[m]:
                dst.write(line + "\n")
                sign = str(r.get("sign_code") or r.get("pdd_code") or "5.19")
                stat[sign][tag] += 1

    meta = {
        "seed": args.seed,
        "test_frac": args.test_frac,
        "filter": "none (all maps; A6 gate deferred until crosswalk metrics CSV)",
        "rule": "split unique net_path; maps kept whole; never in both halves",
        "catalog": str(cat_p.resolve()),
        "n_maps": len(maps_all),
        "n_maps_test": len(test_maps),
        "test_maps": sorted(test_maps),
    }
    split_p.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")

    print(f"{'sign':<8}{'train':>8}{'test':>7}")
    for sign in sorted(stat):
        print(f"{sign:<8}{stat[sign]['tr']:>8}{stat[sign]['te']:>7}")
    tr = sum(c["tr"] for c in stat.values())
    te = sum(c["te"] for c in stat.values())
    print(
        f"\nrows: train={tr}  test={te}  "
        f"maps test={len(test_maps)}/{len(maps_all)}"
    )
    print(f"train: {train_p}\ntest:  {test_p}\nsplit: {split_p}")


if __name__ == "__main__":
    main()
