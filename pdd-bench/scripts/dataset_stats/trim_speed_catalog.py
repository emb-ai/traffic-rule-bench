#!/usr/bin/env python3
"""Trim speed-sign catalog.jsonl by whole maps (not arbitrary rows).

Keeps every variant belonging to a selected ``net_path``, so map geometry stays
intact. Sampling is stratified by the existing train/test map split and is
deterministic (fixed seed).

Default target: ~1200 scenarios / sign (120 maps × 10 variants) ∈ [1000, 1500].
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_SRC = Path(
    "/home/jovyan/shares/SR006.nfs2/smirnova/traffic-rule-bench/pdd-bench"
    "/benchmark_output_speed/balanced/run_v61_a6/catalog.jsonl"
)
DEFAULT_SPLIT = Path(
    "/home/jovyan/shares/SR006.nfs2/smirnova/traffic-rule-bench/pdd-bench"
    "/benchmark_output_speed/balanced/run_v61_a6/maps_split.json"
)
DEFAULT_SIGNS = ("3.24", "4.6", "5.21", "5.31")


def load_rows(path: Path, signs: set[str]) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if str(r.get("sign_code") or r.get("pdd_code") or "") in signs:
                rows.append(r)
    return rows


def trim_by_maps(
    rows: list[dict],
    test_maps: set[str],
    target_scenarios: int,
    seed: int,
) -> tuple[list[dict], dict]:
    """Select whole maps until scenario count is near ``target_scenarios``."""
    by_sign: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        code = str(r.get("sign_code") or r.get("pdd_code"))
        by_sign[code][r["net_path"]].append(r)

    rng = random.Random(seed)
    kept: list[dict] = []
    report: dict[str, dict] = {}

    for code in sorted(by_sign):
        maps = by_sign[code]
        # rows per map (expect 10)
        per_map = {m: len(rs) for m, rs in maps.items()}
        typical = sorted(per_map.values())[len(per_map) // 2]
        n_maps_target = max(1, round(target_scenarios / max(typical, 1)))

        train = [m for m in maps if m not in test_maps]
        test = [m for m in maps if m in test_maps]
        # Preserve ~test_frac among selected maps (fallback to overall fraction).
        test_frac = (len(test) / len(maps)) if maps else 0.2
        n_test = min(len(test), round(n_maps_target * test_frac))
        n_train = min(len(train), n_maps_target - n_test)
        # If one side is short, top up from the other.
        if n_train + n_test < n_maps_target:
            deficit = n_maps_target - n_train - n_test
            extra_train = min(deficit, len(train) - n_train)
            n_train += extra_train
            deficit -= extra_train
            n_test += min(deficit, len(test) - n_test)

        rng.shuffle(train)
        rng.shuffle(test)
        selected = set(train[:n_train] + test[:n_test])

        sign_rows = []
        for m in sorted(selected):  # stable output order by map path
            # keep original within-map order
            sign_rows.extend(maps[m])
        kept.extend(sign_rows)

        report[code] = {
            "maps_total": len(maps),
            "maps_kept": len(selected),
            "maps_kept_train": n_train,
            "maps_kept_test": n_test,
            "scenarios_total": sum(per_map.values()),
            "scenarios_kept": len(sign_rows),
            "rows_per_map_typical": typical,
            "test_frac_original": round(test_frac, 4),
        }

    # Stable global order: by sign_code then net_path then var_idx
    kept.sort(
        key=lambda r: (
            str(r.get("sign_code") or ""),
            str(r.get("net_path") or ""),
            int(r.get("var_idx") or 0),
            int(r.get("seed") or 0),
        )
    )
    return kept, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--maps-split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output jsonl (default: <src_dir>/catalog_balanced_1k2.jsonl)",
    )
    ap.add_argument(
        "--also-copy-to",
        type=Path,
        default=None,
        help="Optional second output path (e.g. dataset_stats/data/...)",
    )
    ap.add_argument("--signs", nargs="+", default=list(DEFAULT_SIGNS))
    ap.add_argument(
        "--target-scenarios",
        type=int,
        default=1200,
        help="Target scenarios per sign (map-rounded; default 1200 ∈ [1k,1.5k])",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = args.out or (args.src.parent / "catalog_balanced_1k2.jsonl")
    signs = set(args.signs)

    split = json.loads(args.maps_split.read_text())
    test_maps = set(split.get("test_maps") or [])

    rows = load_rows(args.src, signs)
    kept, report = trim_by_maps(
        rows,
        test_maps=test_maps,
        target_scenarios=args.target_scenarios,
        seed=args.seed,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "source": str(args.src),
        "maps_split": str(args.maps_split),
        "seed": args.seed,
        "target_scenarios_per_sign": args.target_scenarios,
        "signs": sorted(signs),
        "n_rows_in": len(rows),
        "n_rows_out": len(kept),
        "per_sign": report,
        "policy": (
            "Sample whole net_path maps (all variants kept). "
            "Stratify by maps_split.test_maps. Deterministic shuffle(seed)."
        ),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    if args.also_copy_to is not None:
        args.also_copy_to.parent.mkdir(parents=True, exist_ok=True)
        args.also_copy_to.write_text(out.read_text())
        args.also_copy_to.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )

    print(f"Wrote {out} ({len(kept)} rows)")
    print(f"Meta  {meta_path}")
    for code, info in report.items():
        print(
            f"  {code}: maps {info['maps_kept']}/{info['maps_total']} "
            f"(train={info['maps_kept_train']}, test={info['maps_kept_test']}) "
            f"→ scenarios {info['scenarios_kept']}/{info['scenarios_total']}"
        )


if __name__ == "__main__":
    main()
