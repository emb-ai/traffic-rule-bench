"""Full SUMO benchmark pipeline.

Reads a catalog (or builds one), materializes ALL rows in parallel using
subprocess workers, optionally drives an agent through each scene, and writes
sumo_manifest.jsonl + sumo_summary.json.

Usage (from a catalog):
    python -m sumo_space.sumo_pipeline \
        --catalog benchmark_sumo/sumo_scene_catalog_supported.json \
        --n-workers 16 \
        --output-dir benchmark_sumo

Usage (build catalog inline):
    python -m sumo_space.sumo_pipeline \
        --scenes-root pdd-bench/scenes \
        --n-per-category 250 --n-variations 10 \
        --n-workers 16 --output-dir benchmark_sumo
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Path bootstrap
SCRIPT_PATH = Path(__file__).resolve()
SUMO_DIR = SCRIPT_PATH.parent
BENCHMARK_DIR = SUMO_DIR.parent
SCRIPTS_DIR = BENCHMARK_DIR.parent
PDD_BENCH_DIR = SCRIPTS_DIR.parent
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"
for _p in (PDD_BENCH_DIR, METADRIVE_DIR, BENCHMARK_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from sumo_space.sumo_catalog import (
    build_catalog, save_catalog, load_catalog, DEFAULT_SPAWN_VELOCITIES,
)


# GOST sign codes that exist in /scenes/ and have Python class implementations
# in SIGN_TYPE_TO_CLASS. Unsupported new signs (3.21, 5.4) are not listed yet.
DEFAULT_SUPPORTED_SIGNS = [
    "2.1", "2.2", "2.3.1", "2.3.2", "2.3.3", "2.4", "2.5",
    "3.1", "3.2", "3.18.1", "3.18.2", "3.19", "3.20",
    "3.24", "3.25", "3.27", "3.31",
    "4.1.1", "4.1.2", "4.1.3", "4.1.4", "4.1.5", "4.1.6",
    "4.2.1", "4.2.2", "4.2.3", "4.6",
    "5.11.1", "5.11.2", "5.12.1", "5.12.2",
    "5.13.1", "5.13.2", "5.13.3", "5.13.4",
    "5.14.1", "5.14.2", "5.14.3",
    "5.3", "5.15.2", "5.16", "5.31", "5.32", "5.5",
    "5.7.1", "5.7.2",
]


def _worker(payload: dict) -> dict:
    """Pool worker — fresh subprocess per scene because MetaDrive engine is global."""
    import sys
    from pathlib import Path
    SCRIPT_PATH = Path(__file__).resolve()
    SUMO_DIR = SCRIPT_PATH.parent
    BENCHMARK_DIR = SUMO_DIR.parent
    SCRIPTS_DIR = BENCHMARK_DIR.parent
    PDD_BENCH_DIR = SCRIPTS_DIR.parent
    SDC_ROOT = PDD_BENCH_DIR.parent
    METADRIVE_DIR = SDC_ROOT / "metadrive"
    for _p in (PDD_BENCH_DIR, METADRIVE_DIR, BENCHMARK_DIR):
        _ps = str(_p)
        if _ps not in sys.path:
            sys.path.insert(0, _ps)
    from sumo_space.sumo_runner import materialize_sumo_scene

    row = payload["row"]
    try:
        return materialize_sumo_scene(
            row,
            density_cap=payload["density_cap"],
            drive_agent=payload["drive_agent"],
            agent_policy=payload["policy"],
            save_gif=None,
            max_steps=payload["max_steps"],
        )
    except Exception as exc:
        out = dict(row)
        out["valid"] = False
        out["failure_reason"] = f"worker_exception: {type(exc).__name__}: {exc}"
        return out


def run_pipeline(
    catalog_path: Path | None,
    scenes_root: Path | None,
    n_per_category: int,
    n_variations: int,
    sign_categories: list[str] | None,
    spawn_velocities: list[float],
    density_cap: float,
    drive_agent: bool,
    policy: str,
    max_steps: int,
    output_dir: Path,
    n_workers: int,
    seed: int,
    progress_every: int = 100,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Catalog
    if catalog_path is not None and Path(catalog_path).exists():
        print(f"Loading catalog from {catalog_path}...")
        catalog = load_catalog(catalog_path)
    else:
        print(f"Building catalog from {scenes_root}...")
        catalog = build_catalog(
            scenes_root=scenes_root,
            n_per_category=n_per_category,
            n_variations=n_variations,
            spawn_velocities=spawn_velocities,
            sign_categories=sign_categories,
            seed=seed,
        )
        catalog_path = output_dir / "sumo_scene_catalog.json"
        save_catalog(catalog, catalog_path)
        print(f"  saved to {catalog_path}")

    print(f"  catalog rows: {len(catalog):,}")
    print(f"  unique sign categories: {len(set(r['sign_code'] for r in catalog))}")

    # 2) Build payloads
    payloads = [
        {
            "row": row,
            "density_cap": density_cap,
            "drive_agent": drive_agent,
            "policy": policy,
            "max_steps": max_steps,
        }
        for row in catalog
    ]

    # 3) Parallel materialization
    print(f"\nMaterializing {len(payloads):,} scenes with {n_workers} workers...")
    print(f"  drive_agent={drive_agent}, policy={policy}, max_steps={max_steps}")
    print(f"  density_cap={density_cap}")

    from multiprocessing import get_context
    ctx = get_context("spawn")

    manifest_path = output_dir / "sumo_manifest.jsonl"
    failures = Counter()
    n_valid = 0
    n_done = 0
    t0 = time.time()

    # Stream results to disk so we can resume / inspect mid-run
    with open(manifest_path, "w") as f_out:
        with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(_worker, payloads):
                f_out.write(json.dumps(r, default=str) + "\n")
                f_out.flush()
                n_done += 1
                if r.get("valid"):
                    n_valid += 1
                else:
                    reason = (r.get("failure_reason") or "unknown").split(":")[0]
                    failures[reason] += 1

                if n_done % progress_every == 0:
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 1e-3)
                    eta = (len(payloads) - n_done) / max(rate, 1e-3)
                    print(f"  [{n_done}/{len(payloads)}] valid={n_valid} "
                          f"({100*n_valid/n_done:.0f}%) "
                          f"rate={rate:.1f}/s "
                          f"eta={eta/60:.0f}min", flush=True)

    elapsed = time.time() - t0

    # 4) Summary
    summary = {
        "catalog_rows": len(catalog),
        "n_materialized": n_done,
        "n_valid": n_valid,
        "valid_rate": round(n_valid / max(n_done, 1), 4),
        "wall_time_s": round(elapsed, 1),
        "throughput_per_s": round(n_done / max(elapsed, 1e-3), 2),
        "failures_by_reason": dict(failures.most_common()),
        "n_workers": n_workers,
        "drive_agent": drive_agent,
        "policy": policy,
        "max_steps": max_steps,
        "density_cap": density_cap,
    }

    # Per-sign breakdown
    by_sign_valid = defaultdict(int)
    by_sign_total = defaultdict(int)
    with open(manifest_path) as f_in:
        for line in f_in:
            r = json.loads(line)
            sign = r.get("sign_code", "?")
            by_sign_total[sign] += 1
            if r.get("valid"):
                by_sign_valid[sign] += 1
    summary["per_sign_valid_rate"] = {
        s: round(by_sign_valid[s] / max(by_sign_total[s], 1), 3)
        for s in sorted(by_sign_total)
    }

    summary_path = output_dir / "sumo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMO PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Materialized: {n_done:,}")
    print(f"Valid:        {n_valid:,} ({100*n_valid/max(n_done,1):.1f}%)")
    print(f"Wall time:    {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Throughput:   {n_done/max(elapsed,1e-3):.1f} scenes/s")
    print(f"\nFailure reasons:")
    for reason, n in failures.most_common(15):
        print(f"  {n:>6}× {reason}")
    print(f"\nPer-sign valid rate:")
    for sign, rate in sorted(summary["per_sign_valid_rate"].items()):
        bar = "█" * int(20 * rate)
        print(f"  {sign:8s} {100*rate:>5.1f}% {bar}")

    print(f"\nManifest: {manifest_path}")
    print(f"Summary:  {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Full SUMO benchmark pipeline")
    parser.add_argument("--catalog", type=str, default=None,
                        help="Path to existing catalog JSON. If omitted, build inline.")
    parser.add_argument("--scenes-root", type=str,
                        default=str(PDD_BENCH_DIR / "scenes"))
    parser.add_argument("--n-per-category", type=int, default=250)
    parser.add_argument("--n-variations", type=int, default=10)
    parser.add_argument("--sign-categories", type=str, default=None,
                        help="Comma-separated list. Default: all 24 supported codes.")
    parser.add_argument("--density-cap", type=float, default=1.0)
    parser.add_argument("--drive-agent", action="store_true",
                        help="Drive a real policy through each scene (slower, gives behavioral metrics)")
    parser.add_argument("--policy", type=str, default="idm",
                        choices=["comprehensive_rule_expert", "rule_compliant_expert", "idm"])
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="benchmark_sumo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    # Resolve net_path against THIS scenes root (not the hardcoded default), so
    # materialize_sumo_scene finds the .net.xml for non-default sets (scenes_uniq).
    # Set before spawning workers so forked/spawned workers inherit it.
    import os as _os
    _os.environ["PER_SIGN_SCENES_ROOT"] = str(Path(args.scenes_root).resolve())

    if args.sign_categories:
        sign_codes = [c.strip() for c in args.sign_categories.split(",") if c.strip()]
    else:
        sign_codes = DEFAULT_SUPPORTED_SIGNS

    run_pipeline(
        catalog_path=Path(args.catalog) if args.catalog else None,
        scenes_root=Path(args.scenes_root),
        n_per_category=args.n_per_category,
        n_variations=args.n_variations,
        sign_categories=sign_codes,
        spawn_velocities=DEFAULT_SPAWN_VELOCITIES,
        density_cap=args.density_cap,
        drive_agent=args.drive_agent,
        policy=args.policy,
        max_steps=args.max_steps,
        output_dir=Path(args.output_dir),
        n_workers=args.n_workers,
        seed=args.seed,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
