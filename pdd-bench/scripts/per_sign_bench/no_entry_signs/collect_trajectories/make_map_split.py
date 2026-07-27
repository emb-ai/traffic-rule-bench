#!/usr/bin/env python3
"""Split a no_entry combined catalog 80/20 by unique MAPS (net_path).

Stratification is by sign_code (3.1 / 3.2) so each sign keeps ~80% of its
equal-share maps after build_combined_catalog.py.

A map never appears in both halves.

Outputs (next to the catalog, or at --out-prefix):
    <prefix>train80.jsonl   <prefix>test20.jsonl   <prefix>maps_split.json

Usage:
    python3 make_map_split.py \\
        --catalog ../benchmark_output/combined/real_manifest_balanced.jsonl

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
        help="catalog.jsonl or real_manifest_balanced.jsonl (must have net_path)",
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

    # Stratify by sign_code (fallback "3.1")
    groups: dict[tuple, list] = collections.defaultdict(list)
    for m in maps_all:
        signs = tuple(
            sorted({str(r.get("sign_code") or r.get("pdd_code") or "3.1")
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

    train_by_sign: dict[str, list[str]] = collections.defaultdict(list)
    test_by_sign: dict[str, list[str]] = collections.defaultdict(list)
    stat: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    map_stat: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for m in maps_all:
        tag = "te" if m in test_maps else "tr"
        bucket = test_by_sign if tag == "te" else train_by_sign
        signs_on_map = {
            str(r.get("sign_code") or r.get("pdd_code") or "3.1")
            for _, r in rows_by_map[m]
        }
        for sign in signs_on_map:
            map_stat[sign][tag] += 1
        for line, r in rows_by_map[m]:
            sign = str(r.get("sign_code") or r.get("pdd_code") or "3.1")
            bucket[sign].append(line)
            stat[sign][tag] += 1

    def _interleave_write(path: Path, by_sign: dict[str, list[str]]) -> None:
        """Round-robin by sign so COUNT smokes see both 3.1 and 3.2 early."""
        keys = sorted(by_sign)
        idxs = {k: 0 for k in keys}
        with open(path, "w", encoding="utf-8") as fh:
            while True:
                progressed = False
                for k in keys:
                    i = idxs[k]
                    lst = by_sign[k]
                    if i < len(lst):
                        fh.write(lst[i] + "\n")
                        idxs[k] = i + 1
                        progressed = True
                if not progressed:
                    break

    _interleave_write(train_p, train_by_sign)
    _interleave_write(test_p, test_by_sign)

    meta = {
        "seed": args.seed,
        "test_frac": args.test_frac,
        "filter": "none (balanced combined catalog; equal maps per 3.1/3.2)",
        "rule": "split unique net_path; maps kept whole; never in both halves; "
                "stratified by sign_code",
        "catalog": str(cat_p.resolve()),
        "n_maps": len(maps_all),
        "n_maps_test": len(test_maps),
        "test_maps": sorted(test_maps),
    }
    split_p.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")

    print(f"{'sign':<8}{'train_rows':>11}{'test_rows':>10}"
          f"{'train_maps':>11}{'test_maps':>10}")
    for sign in sorted(stat):
        print(
            f"{sign:<8}{stat[sign]['tr']:>11}{stat[sign]['te']:>10}"
            f"{map_stat[sign]['tr']:>11}{map_stat[sign]['te']:>10}"
        )
    tr = sum(c["tr"] for c in stat.values())
    te = sum(c["te"] for c in stat.values())
    print(
        f"\nrows: train={tr}  test={te}  "
        f"maps test={len(test_maps)}/{len(maps_all)}"
    )
    print(f"train: {train_p}\ntest:  {test_p}\nsplit: {split_p}")


if __name__ == "__main__":
    main()
