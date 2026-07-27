#!/usr/bin/env python3
"""Build a balanced combined catalog for no_turn (3.18.1–3.18.2).

Equal map contribution:
  - Load per-code real_manifest.jsonl files
  - Group rows by (sign_code, net_path)
  - Take n = min(n_maps over codes) maps from each (seed=42 shuffle)
  - Keep ALL rows for the selected maps
  - Interleave signs so small COUNT smokes see all codes early

Writes under no_turn_signs/benchmark_output/combined/:
  real_manifest_balanced.jsonl
  real_manifest_unbalanced.jsonl
  balance_report.json

Usage:
  python build_combined_catalog.py
  python build_combined_catalog.py --seed 42 --out-dir ../benchmark_output/combined
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIGN_DIR = HERE.parent
DEFAULT_MANIFESTS = {
    "3.18.1": SIGN_DIR / "benchmark_output" / "3_18_1" / "final_metrics_v1" / "real_manifest.jsonl",
    "3.18.2": SIGN_DIR / "benchmark_output" / "3_18_2" / "final_metrics_v1" / "real_manifest.jsonl",
}
DEFAULT_OUT = SIGN_DIR / "benchmark_output" / "combined"
CODES = ["3.18.1", "3.18.2"]


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("valid") is False:
                continue
            rows.append(r)
    return rows


def _sign_of(row: dict, fallback: str) -> str:
    return str(row.get("sign_code") or row.get("pdd_code") or fallback).strip()


def _group_by_map(rows: list[dict], fallback_sign: str) -> dict[str, list[dict]]:
    by_map: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        net = r.get("net_path")
        if not net:
            continue
        code = _sign_of(r, fallback_sign)
        r = dict(r)
        r["sign_code"] = code
        r.setdefault("pdd_code", code)
        by_map[str(net)].append(r)
    return by_map


def _interleave(lists: list[list[dict]]) -> list[dict]:
    """Round-robin merge so COUNT=N smokes hit one row per sign when possible."""
    out: list[dict] = []
    idxs = [0] * len(lists)
    while True:
        progressed = False
        for i, lst in enumerate(lists):
            if idxs[i] < len(lst):
                out.append(lst[idxs[i]])
                idxs[i] += 1
                progressed = True
        if not progressed:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest-3_18_1", type=Path, default=DEFAULT_MANIFESTS["3.18.1"])
    ap.add_argument("--manifest-3_18_2", type=Path, default=DEFAULT_MANIFESTS["3.18.2"])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = {
        "3.18.1": args.manifest_3_18_1,
        "3.18.2": args.manifest_3_18_2,
    }
    for code, p in paths.items():
        if not p.is_file():
            sys.exit(f"ERROR: missing manifest for {code}: {p}")

    by_sign_maps: dict[str, dict[str, list[dict]]] = {}
    raw_counts: dict[str, dict] = {}
    unbalanced: list[dict] = []
    for code, p in paths.items():
        rows = _load_rows(p)
        by_map = _group_by_map(rows, code)
        by_sign_maps[code] = by_map
        raw_counts[code] = {
            "rows": len(rows),
            "maps": len(by_map),
            "path": str(p.resolve()),
        }
        for net in sorted(by_map):
            unbalanced.extend(by_map[net])
        print(f"[{code}] loaded {len(rows)} rows / {len(by_map)} maps from {p}")

    n = min(len(by_sign_maps["3.18.1"]), len(by_sign_maps["3.18.2"]))
    if n <= 0:
        sys.exit("ERROR: no maps available for balance")

    rng = random.Random(args.seed)
    selected: dict[str, list[str]] = {}
    balanced_by_sign: dict[str, list[dict]] = {}
    for code, by_map in by_sign_maps.items():
        maps = sorted(by_map)
        rng.shuffle(maps)
        keep = maps[:n]
        selected[code] = keep
        rows: list[dict] = []
        for m in keep:
            rows.extend(by_map[m])
        balanced_by_sign[code] = rows
        print(f"[{code}] selected {len(keep)}/{len(maps)} maps → {len(rows)} rows")

    for code in balanced_by_sign:
        balanced_by_sign[code].sort(
            key=lambda r: (
                str(r.get("scene_id") or ""),
                int(r.get("seed") or r.get("deterministic_seed") or 0),
                int(r.get("var_idx", 0) or 0),
            )
        )
    balanced = _interleave([balanced_by_sign["3.18.1"], balanced_by_sign["3.18.2"]])

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bal_path = out_dir / "real_manifest_balanced.jsonl"
    unbal_path = out_dir / "real_manifest_unbalanced.jsonl"
    report_path = out_dir / "balance_report.json"

    with open(bal_path, "w", encoding="utf-8") as f:
        for r in balanced:
            f.write(json.dumps(r, default=str) + "\n")
    with open(unbal_path, "w", encoding="utf-8") as f:
        for r in unbalanced:
            f.write(json.dumps(r, default=str) + "\n")

    bal_counts = {
        code: {
            "maps": len(selected[code]),
            "rows": len(balanced_by_sign[code]),
        }
        for code in CODES
    }
    overlaps = {}
    for i, a in enumerate(CODES):
        for b in CODES[i + 1:]:
            overlaps[f"{a}∩{b}"] = len(set(by_sign_maps[a]) & set(by_sign_maps[b]))
    report = {
        "seed": args.seed,
        "rule": "equal map contribution: n=min(n_maps for 3.18.1, 3.18.2); keep all rows for selected maps; interleave signs",
        "n_maps_per_sign": n,
        "raw": raw_counts,
        "balanced": bal_counts,
        "balanced_total_rows": len(balanced),
        "unbalanced_total_rows": len(unbalanced),
        "map_overlap": overlaps,
        "outputs": {
            "balanced": str(bal_path.resolve()),
            "unbalanced": str(unbal_path.resolve()),
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"{'sign':<8}{'maps':>8}{'rows':>8}")
    for code in CODES:
        print(f"{code:<8}{bal_counts[code]['maps']:>8}{bal_counts[code]['rows']:>8}")
    print(f"\nbalanced:   {bal_path}  ({len(balanced)} rows)")
    print(f"unbalanced: {unbal_path}  ({len(unbalanced)} rows)")
    print(f"report:     {report_path}")


if __name__ == "__main__":
    main()
