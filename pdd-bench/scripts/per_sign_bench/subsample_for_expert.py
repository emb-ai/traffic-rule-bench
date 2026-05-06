"""Stratified subsample of materialized JSONL manifests for expert replay.

Reads from multiple benchmark_output dirs, applies per-sign caps (HARD/MEDIUM/SIMPLE),
samples evenly across velocity indices, and writes to a new consolidated dir.
Original files are NOT modified.

Usage:
    python subsample_for_expert.py [--output-dir benchmark_output/sampled_for_expert]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

# Repo root (traffic-rule-bench/), overridable via REPO_ROOT env var.
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
_BENCH_OUT = REPO_ROOT / "pdd-bench/scripts/per_sign_bench/benchmark_output"

# ---------------------------------------------------------------------------
# Per-sign caps
# ---------------------------------------------------------------------------

CAPS: dict[str, int] = {
    # Unified target: 250 scenes per sign (when available).
    "2.1": 250, "2.2": 250,
    "2.3.1": 250, "2.3.2": 250, "2.3.3": 250,
    "2.4": 250,
    "3.1": 250, "3.2": 250,
    "3.18.2": 250, "3.19": 250, "3.20": 250,
    "3.24": 250, "3.25": 250, "3.27": 250, "3.31": 250,
    "4.2.1": 250, "4.2.2": 250, "4.2.3": 250, "4.6": 250,
    "5.11.1": 250, "5.11.2": 250,
    "5.12.1": 250, "5.12.2": 250,
    "5.13.1": 250, "5.13.2": 250, "5.13.3": 250, "5.13.4": 250,
    "5.14.1": 250, "5.14.2": 250, "5.14.3": 250,
    "5.31": 250, "5.32": 250,
}

SOURCE_DIRS = [
    _BENCH_OUT / "full_arbelyaev_250_metadrive",
    _BENCH_OUT / "additional_signs_for_full",
]

DEFAULT_OUTPUT = _BENCH_OUT / "sampled_for_expert"


def _vel_key(row: dict) -> int:
    """Unified velocity-index key across pgmap/citymap/paired schemas."""
    for field in ("v_idx", "spawn_velocity_idx", "spawn_velocity_ms"):
        v = row.get(field)
        if v is not None:
            # spawn_velocity_ms is a float — bucket into 5 bins for grouping
            if field == "spawn_velocity_ms":
                return int(round(float(v) * 2))   # arbitrary but stable
            return int(v)
    return 0


def stratified_sample(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample n rows from rows, distributed evenly across velocity groups."""
    if n >= len(rows):
        return list(rows)

    rng = random.Random(seed)

    # Group by velocity index
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        groups[_vel_key(r)].append(r)

    n_groups = len(groups)
    per_group = n // n_groups
    remainder = n % n_groups

    sampled: list[dict] = []
    group_keys = sorted(groups.keys())

    for i, k in enumerate(group_keys):
        take = per_group + (1 if i < remainder else 0)
        grp = groups[k]
        rng.shuffle(grp)
        sampled.extend(grp[:take])

    # Final safety trim (rounding edge cases)
    rng.shuffle(sampled)
    return sampled[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect all JSONL files per code
    # code → list of (jsonl_path, source_dir)
    jsonl_by_code: dict[str, list[Path]] = defaultdict(list)
    source_json_by_code: dict[str, Path] = {}

    for src in SOURCE_DIRS:
        for sign_dir in sorted(src.iterdir()):
            if not sign_dir.is_dir():
                continue
            code = sign_dir.name.replace("_", ".")
            if not code[0].isdigit():
                continue
            for jsonl in sign_dir.rglob("*materialized.jsonl"):
                jsonl_by_code[code].append(jsonl)
            sj = sign_dir / "source.json"
            if sj.exists():
                source_json_by_code[code] = sj

    summary = {}
    grand_total = 0

    for code in sorted(jsonl_by_code, key=lambda c: [int(x) if x.isdigit() else x for x in c.split(".")]):
        cap = CAPS.get(code)
        if cap is None:
            print(f"  [{code}] not in CAPS — skipping")
            continue

        # Merge all valid rows from all JSONL sources for this code
        all_rows: list[dict] = []
        for jsonl_path in jsonl_by_code[code]:
            for line in open(jsonl_path):
                try:
                    r = json.loads(line)
                    if r.get("valid", True):
                        r["_source_file"] = jsonl_path.name
                        all_rows.append(r)
                except Exception:
                    continue

        sampled = stratified_sample(all_rows, cap, seed=args.seed)

        sign_slug = code.replace(".", "_")
        out_dir = out_root / sign_slug
        out_dir.mkdir(parents=True, exist_ok=True)

        # Determine output filename (prefer citymap_ if any citymap rows, else pgmap)
        has_citymap = any("citymap" in r.get("_source_file", "") for r in sampled)
        has_paired  = any("paired"  in r.get("_source_file", "") for r in sampled)
        has_pgmap   = any("pgmap"   in r.get("_source_file", "") for r in sampled)

        # Write one merged JSONL (strip internal _source_file key)
        out_jsonl = out_dir / "materialized.jsonl"
        with open(out_jsonl, "w") as f:
            for r in sampled:
                r.pop("_source_file", None)
                f.write(json.dumps(r, default=str) + "\n")

        # Copy source.json for metadata
        if code in source_json_by_code:
            shutil.copy(source_json_by_code[code], out_dir / "source.json")

        n_available = len(all_rows)
        n_written   = len(sampled)
        grand_total += n_written

        vel_dist = defaultdict(int)
        for r in sampled:
            vel_dist[_vel_key(r)] += 1
        vel_str = " ".join(f"v{k}:{v}" for k, v in sorted(vel_dist.items()))

        backends = []
        if has_pgmap:   backends.append("pgmap")
        if has_paired:  backends.append("paired")
        if has_citymap: backends.append("citymap")

        print(f"  [{code}] {n_available:>5} → {n_written:>4}  ({'+'.join(backends)})  {vel_str}")
        summary[code] = {"available": n_available, "sampled": n_written, "cap": cap, "backends": backends}

    # Write summary
    with open(out_root / "sample_summary.json", "w") as f:
        json.dump({"grand_total": grand_total, "per_sign": summary}, f, indent=2)

    print(f"\nTotal scenes written: {grand_total}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
