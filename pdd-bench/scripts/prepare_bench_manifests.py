#!/usr/bin/env python3
"""Filter sumo_manifest.jsonl files to exactly N unique maps per sign.

Groups manifest rows by net_path (= unique SUMO map), picks one row per map
(lowest v_idx, then var_idx), and writes filtered manifests under a new root
that replay_mini_new.py can read directly.

Usage:
    python3 prepare_bench_manifests.py \
        --src-root  /path/to/benchmark_output/mini \
        --out-root  /path/to/prepared_bench \
        --signs     "2.5 5.11.1 3.1 3.20 3.27" \
        --n-maps    10
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _pick_best_row(rows: list[dict]) -> dict:
    """From rows sharing the same net_path pick one deterministically.

    Prefer lowest v_idx, then lowest var_idx, then first by row order.
    """
    def _sort_key(r):
        return (
            int(r.get("v_idx") or 0),
            int(r.get("var_idx") or 0),
        )
    return sorted(rows, key=_sort_key)[0]


def prepare_manifests(
    src_root: Path,
    out_root: Path,
    signs: list[str],
    n_maps: int,
) -> dict[str, int]:
    """Write filtered manifests; return {sign_code: n_maps_written}."""
    results = {}

    for code in signs:
        code_dir = code.replace(".", "_")
        src_manifest = src_root / code_dir / "sumo" / "sumo_manifest.jsonl"

        if not src_manifest.exists():
            print(f"  [skip] {code}: manifest not found at {src_manifest}")
            results[code] = 0
            continue

        rows = [json.loads(l) for l in src_manifest.read_text().splitlines() if l.strip()]
        valid_rows = [r for r in rows if r.get("valid", True)]
        if not valid_rows:
            print(f"  [skip] {code}: no valid rows in manifest")
            results[code] = 0
            continue

        # Group by net_path
        by_map: dict[str, list[dict]] = defaultdict(list)
        for r in valid_rows:
            key = r.get("net_path") or r.get("scene_id", "unknown")
            by_map[key].append(r)

        # Pick one row per map, keep first n_maps maps (stable order)
        selected: list[dict] = []
        for net_path in list(by_map.keys())[:n_maps]:
            selected.append(_pick_best_row(by_map[net_path]))

        if len(selected) < n_maps:
            print(f"  [warn] {code}: only {len(selected)} unique maps "
                  f"(requested {n_maps})")

        # Write
        dst_manifest = out_root / code_dir / "sumo" / "sumo_manifest.jsonl"
        dst_manifest.parent.mkdir(parents=True, exist_ok=True)
        dst_manifest.write_text(
            "\n".join(json.dumps(r, default=str) for r in selected) + "\n"
        )
        results[code] = len(selected)
        print(f"  {code}: {len(selected)} unique maps → {dst_manifest}")

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True,
                    help="benchmark_output/mini (or mini_new) root dir")
    ap.add_argument("--out-root", required=True,
                    help="Output dir for filtered manifests")
    ap.add_argument("--signs", default="2.5 5.11.1 3.1 3.20 3.27",
                    help="Space-separated PDD sign codes")
    ap.add_argument("--n-maps", type=int, default=10,
                    help="Number of unique maps per sign (default: 10)")
    args = ap.parse_args()

    src = Path(args.src_root)
    out = Path(args.out_root)
    signs = args.signs.strip().split()

    print(f"Source:  {src}")
    print(f"Output:  {out}")
    print(f"Signs:   {signs}")
    print(f"n_maps:  {args.n_maps}")
    print()

    results = prepare_manifests(src, out, signs, args.n_maps)

    print()
    ok = sum(1 for v in results.values() if v == args.n_maps)
    partial = sum(1 for v in results.values() if 0 < v < args.n_maps)
    skipped = sum(1 for v in results.values() if v == 0)
    print(f"Done: {ok} full, {partial} partial, {skipped} skipped")


if __name__ == "__main__":
    main()
