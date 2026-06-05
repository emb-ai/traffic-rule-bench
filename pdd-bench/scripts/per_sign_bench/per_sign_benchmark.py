
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_DIR = SCRIPT_PATH.parent
SCRIPTS_DIR = BENCHMARK_DIR.parent
PDD_BENCH_DIR = SCRIPTS_DIR.parent
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"
for _p in (PDD_BENCH_DIR, METADRIVE_DIR, BENCHMARK_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from factorized_space.space_definition import (
    PDD_TO_SYNTHETIC, PDD_HUMAN_NAME, PAIRED_SIGN_BUNDLES,
    PIPELINE_PGMAP, PIPELINE_CITYMAP, PIPELINE_NONE,
)
from factorized_space.index_codec import FactorizedSpaceCodec
from factorized_space.paired_space import PairedSpaceCodec
from factorized_space.materialized_sampler import sample_for_sign
from sumo_space.sumo_catalog import build_catalog as build_sumo_catalog


DEFAULT_SCENES_ROOT = str(PDD_BENCH_DIR / "scenes")

# Per-sign plan: which source(s) to use for benchmark scene construction.
# Values: "C" (synthetic only), "P" (real SUMO only), "CP" (both).
# This follows the current sign-coverage table. New signs without implementation
# yet (3.21, 5.4) are intentionally not included in the active plan.
DEFAULT_SOURCE_PLAN: Dict[str, str] = {
    "2.1": "CP",  "2.2": "C",   "2.3.1": "C",
    "2.3.2": "CP","2.3.3": "CP",
    "2.4": "CP",  "2.5": "P",
    "3.1": "P",   "3.18.1": "CP", "3.18.2": "CP", "3.19": "P",
    "3.2": "P",   "3.20": "CP",
    "3.24": "CP", "3.25": "C",   "3.27": "P",
    "3.31": "C",
    "4.1.1": "P", "4.1.2": "P", "4.1.3": "P",
    "4.1.4": "P", "4.1.5": "P", "4.1.6": "P",
    "4.2.1": "C", "4.2.2": "C", "4.2.3": "C",
    "4.6": "CP",
    "5.3": "P",   "5.5": "P",   "5.7.1": "P", "5.7.2": "P",
    "5.11.1": "CP","5.11.2": "C",
    "5.12.1": "C","5.12.2": "C",
    "5.13.1": "CP","5.13.2": "CP","5.13.3": "C","5.13.4": "C",
    "5.14.1": "C", "5.14.2": "CP","5.14.3": "C",
    "5.15.2": "P","5.16": "P",
    "5.31": "CP","5.32": "C",
}


PDD_CODE_ALIASES: Dict[str, str] = {
    # Spreadsheet/export aliases that encode the last PDD component as 2001,
    # 2002, ... Keep manifests on canonical code names used by the env.
    "2.3.2001": "2.3.1",
    "2.3.2002": "2.3.2",
    "2.3.2003": "2.3.3",
    "4.1.2001": "4.1.1",
    "4.1.2002": "4.1.2",
    "4.1.2003": "4.1.3",
    "4.1.2004": "4.1.4",
    "4.1.2005": "4.1.5",
    "4.1.2006": "4.1.6",
    "4.2.2001": "4.2.1",
    "4.2.2002": "4.2.2",
    "4.2.2003": "4.2.3",
    "5.7.2001": "5.7.1",
    "5.7.2002": "5.7.2",
    "5.11.2001": "5.11.1",
    "5.11.2002": "5.11.2",
    "5.12.2001": "5.12.1",
    "5.12.2002": "5.12.2",
}

UNSUPPORTED_TABLE_CODES: Dict[str, str] = {
    "3.21": "End of no-overtaking zone: no implementation/sign class yet",
    "5.4": "End of motor-vehicles-only road: no implementation/sign class yet",
}


def _normalize_pdd_code(code: str) -> str:
    code = code.strip()
    return PDD_CODE_ALIASES.get(code, code)


# ----------------------------------------------------------------------------
# Synthetic collectors (dry-run and actual)
# ----------------------------------------------------------------------------

def _collect_pgmap_specs(codec: FactorizedSpaceCodec, pdd_code: str,
                         n: int, seed: int) -> List[dict]:
    """Return up to `n` decoded base-scene dicts for PDD code (PGMap pipeline)."""
    pipeline, keys = PDD_TO_SYNTHETIC.get(pdd_code, (PIPELINE_NONE, []))
    if pipeline != PIPELINE_PGMAP or not keys:
        return []
    specs: List[dict] = []
    # Split budget across keys (4 for speed*/zone_speed*/minspeed, 1 otherwise).
    per_key = max(1, n // len(keys))
    for key in keys:
        picks = sample_for_sign(codec, key, n=per_key, seed=seed + hash(key) % 10000)
        for flat in picks:
            s = codec.decode(flat)
            s["pdd_code"] = pdd_code
            s["source"] = "pgmap"
            specs.append(s)
    return specs[:n]


def _collect_paired_specs(pcodec: PairedSpaceCodec, pdd_code: str,
                          n: int, seed: int) -> List[dict]:
    """Return paired-scene specs that include this PDD code on either end."""
    idxs = pcodec.indices_for_pdd(pdd_code)
    if not idxs:
        return []
    import random as _random
    rng = _random.Random(seed + 500)
    if n < len(idxs):
        idxs = rng.sample(idxs, n)
    out = []
    for i in idxs:
        s = pcodec.decode(i)
        s["pdd_code_target"] = pdd_code  # which side this row contributes to
        s["source"] = "pgmap_paired"
        out.append(s)
    return out


def _collect_citymap_plan(pdd_code: str, target_citymap: int, seed: int) -> dict:
    """Describe what CityMap args to use; actual materialization is separate.

    CityMap scenes are enumerated + materialized by citymap_pipeline in one
    pass (no cheap catalog). So we record the params here for the later
    materialize step to execute.
    """
    pipeline, keys = PDD_TO_SYNTHETIC.get(pdd_code, (PIPELINE_NONE, []))
    if pipeline != PIPELINE_CITYMAP or not keys:
        return {}
    return {
        "pdd_code": pdd_code,
        "sign_keys": keys,
        "target": target_citymap,
        "citymap_config": {
            "map_seeds": list(range(40)),       # ~40 maps × 8/sign → room for 250+
            "n_blocks": 8,                      # smaller grids empirically give 0 prohibition scenes
            "max_scenes_per_sign_per_map": 8,
            "sign_categories": ["prohibition"],
            "seed": seed,
        },
    }


# ----------------------------------------------------------------------------
# Real (SUMO) collectors
# ----------------------------------------------------------------------------

def _collect_real_rows(scenes_root: str, pdd_code: str, target: int,
                       n_variations: int, n_velocity_samples: int,
                       seed: int, brake_cfg: Optional[dict] = None) -> List[dict]:
    """Build the SUMO catalog rows for a single PDD code.

    n_per_category = min(target, available) — enforced by stratified_sample
    inside build_sumo_catalog. `brake_cfg` carries the braking-spawn parameters
    (used only for codes in BRAKING_SPAWN_CODES, e.g. 3.24).
    """
    brake_cfg = brake_cfg or {}
    rows = build_sumo_catalog(
        scenes_root=scenes_root,
        n_per_category=target,
        n_variations=n_variations,
        sign_categories=[pdd_code],
        seed=seed,
        n_v0_samples=int(brake_cfg.get("n_v0_samples", n_velocity_samples)),
        brake_decel_mps2=float(brake_cfg.get("brake_decel", 2.5)),
        brake_delay_s=float(brake_cfg.get("brake_delay", 1.0)),
        brake_margin_m=float(brake_cfg.get("brake_margin", 5.0)),
        v0_min_excess_mps=float(brake_cfg.get("v0_min_excess", 2.0)),
        v0_max_kmh=float(brake_cfg.get("v0_max_kmh", 80.0)),
    )
    for r in rows:
        r["pdd_code"] = pdd_code
        r["source"] = "sumo"
    return rows


def _take_first_n_scene_ids(rows: List[dict], n_scenes: int) -> List[dict]:
    """Keep all catalog rows for the first `n_scenes` unique scene IDs."""
    if n_scenes <= 0:
        return []
    selected_ids = []
    seen = set()
    for row in rows:
        scene_id = row.get("scene_id")
        if scene_id in seen:
            continue
        seen.add(scene_id)
        selected_ids.append(scene_id)
        if len(selected_ids) >= n_scenes:
            break
    allowed = set(selected_ids)
    return [row for row in rows if row.get("scene_id") in allowed]


# ----------------------------------------------------------------------------
# Plan / assemble per sign
# ----------------------------------------------------------------------------

def _plan_for_sign(
    pdd_code: str,
    source: str,
    target: int,
    min_synthetic_for_cp: int,
    codec: FactorizedSpaceCodec,
    pcodec: PairedSpaceCodec,
    scenes_root: str,
    n_variations: int,
    n_velocity_samples: int,
    seed: int,
    brake_cfg: Optional[dict] = None,
) -> dict:
    """Assemble the per-sign plan (without materializing CityMap scenes).

    Returns a dict with `synthetic_manifest` (list of specs) and
    `real_catalog_rows` and `citymap_plan` (optional).
    """
    pipeline, _keys = PDD_TO_SYNTHETIC.get(pdd_code, (PIPELINE_NONE, []))

    # Real count (via catalog) — short call, just enumerates scenes/<code>/
    real_rows = []
    if source in ("P", "CP"):
        real_rows = _collect_real_rows(
            scenes_root, pdd_code, target=target,
            n_variations=n_variations,
            n_velocity_samples=n_velocity_samples,
            seed=seed,
            brake_cfg=brake_cfg,
        )

    has_synthetic_source = bool(
        pipeline != PIPELINE_NONE or pcodec.indices_for_pdd(pdd_code)
    )

    # Budget split. For CP, reserve a bounded synthetic slice for diversity
    # without exceeding the per-sign target when both sources have enough data.
    n_real_scenes = len({r["scene_id"] for r in real_rows}) if real_rows else 0
    if source == "P":
        p_taken = min(target, n_real_scenes)
        c_taken_target = 0
    elif source == "C":
        p_taken = 0
        c_taken_target = target
    elif has_synthetic_source:
        synthetic_floor = min(min_synthetic_for_cp, target)
        real_budget = max(0, target - synthetic_floor)
        p_taken = min(real_budget, n_real_scenes)
        c_taken_target = max(0, target - p_taken)
    else:
        p_taken = min(target, n_real_scenes)
        c_taken_target = 0

    if real_rows and p_taken < n_real_scenes:
        real_rows = _take_first_n_scene_ids(real_rows, p_taken)

    # Paired first — shared between two PDD codes, prioritise for end-signs.
    paired_specs = _collect_paired_specs(pcodec, pdd_code, n=c_taken_target, seed=seed)

    # Single-sign synthetic — PGMap or CityMap
    remaining = c_taken_target - len(paired_specs)
    synthetic_specs: List[dict] = list(paired_specs)
    citymap_plan = {}

    if remaining > 0 and source in ("C", "CP"):
        if pipeline == PIPELINE_PGMAP:
            synthetic_specs.extend(
                _collect_pgmap_specs(codec, pdd_code, n=remaining, seed=seed)
            )
        elif pipeline == PIPELINE_CITYMAP:
            citymap_plan = _collect_citymap_plan(pdd_code, remaining, seed=seed)
        # pipeline == NONE → nothing to do (fall through; quality over quantity).

    citymap_target = int(citymap_plan.get("target", 0)) if citymap_plan else 0
    c_taken_total = len(synthetic_specs) + citymap_target

    return {
        "pdd_code": pdd_code,
        "human_name": PDD_HUMAN_NAME.get(pdd_code, "?"),
        "source_label": source,
        "pipeline": pipeline,
        "counts": {
            "real_scenes": n_real_scenes,
            "real_catalog_rows": len(real_rows),
            "synthetic_specs": len(synthetic_specs),
            "citymap_target": citymap_target,
            "p_taken": p_taken,
            "c_taken": c_taken_total,
            "total_base_scenes": p_taken + c_taken_total,
            "reached_target": (p_taken + c_taken_total) >= target,
        },
        "synthetic_specs": synthetic_specs,
        "real_rows": real_rows,
        "citymap_plan": citymap_plan,
    }


# ----------------------------------------------------------------------------
# Write per-sign output
# ----------------------------------------------------------------------------

def _write_sign(output_dir: Path, plan: dict) -> None:
    code = plan["pdd_code"]
    sign_dir = output_dir / code.replace(".", "_")
    sign_dir.mkdir(parents=True, exist_ok=True)

    # source.json — summary
    source_info = {
        "pdd_code": code,
        "human_name": plan["human_name"],
        "source_label": plan["source_label"],
        "pipeline": plan["pipeline"],
        "counts": plan["counts"],
    }
    if plan["citymap_plan"]:
        source_info["citymap_plan"] = plan["citymap_plan"]
    with open(sign_dir / "source.json", "w") as f:
        json.dump(source_info, f, indent=2, ensure_ascii=False)

    # synthetic_manifest.jsonl
    if plan["synthetic_specs"]:
        with open(sign_dir / "synthetic_manifest.jsonl", "w") as f:
            for spec in plan["synthetic_specs"]:
                f.write(json.dumps(spec, default=str) + "\n")

    # real_manifest.jsonl
    if plan["real_rows"]:
        with open(sign_dir / "real_manifest.jsonl", "w") as f:
            for row in plan["real_rows"]:
                f.write(json.dumps(row, default=str) + "\n")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-sign benchmark orchestrator")
    parser.add_argument("--output-dir", type=str, default="benchmark_output/per_sign")
    parser.add_argument("--scenes-root", type=str, default=DEFAULT_SCENES_ROOT)
    parser.add_argument("--target-per-sign", type=int, default=250)
    parser.add_argument("--min-synthetic-for-cp", type=int, default=50,
                        help="For C+P signs: reserve this many synthetic base scenes "
                             "inside the per-sign target when possible.")
    parser.add_argument("--n-variations", type=int, default=10)
    parser.add_argument("--n-velocity-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    # Braking-spawn (3.24): ego starts above the limit, placed d_required before
    # the sign so braking is actually tested. See sumo_catalog / sumo_env.
    parser.add_argument("--n-v0-samples", type=int, default=10,
                        help="v0 draws per scene for braking codes (default: 10).")
    parser.add_argument("--brake-decel", type=float, default=2.5,
                        help="Assumed comfortable deceleration (m/s²) for required distance.")
    parser.add_argument("--brake-delay", type=float, default=1.0,
                        help="Reaction/latency time (s) added to required distance.")
    parser.add_argument("--brake-margin", type=float, default=5.0,
                        help="Safety margin (m) added to required distance.")
    parser.add_argument("--v0-min-excess", type=float, default=2.0,
                        help="Minimum amount (m/s) ego v0 must exceed the sign limit.")
    parser.add_argument("--v0-max-kmh", type=float, default=80.0,
                        help="Upper cap (km/h) on sampled ego v0.")
    parser.add_argument("--only-codes", type=str, default=None,
                        help="Comma-separated PDD codes to process (default: all in plan).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan and print summary only; no files written.")
    parser.add_argument("--materialize", action="store_true",
                        help="After writing manifests, run materialization via MetaDrive/SUMO. "
                             "Requires xvfb-run on headless hosts.")
    parser.add_argument("--sumo-workers", type=int, default=8,
                        help="Parallel workers for SUMO materialization.")
    parser.add_argument("--citymap-maps", type=int, default=40,
                        help="Number of CityMap seeds for prohibition-with-detour scenes.")
    parser.add_argument("--horizon", type=int, default=1200,
                        help="Episode horizon (max steps) used during materialization.")
    parser.add_argument("--skip-materialize", type=str, default="",
                        help="Comma-separated list of backends to skip in materialize step: "
                             "pgmap,paired,citymap,sumo")
    args = parser.parse_args()

    codes = list(DEFAULT_SOURCE_PLAN.keys())
    if args.only_codes:
        raw_selected = {c.strip() for c in args.only_codes.split(",") if c.strip()}
        unsupported = raw_selected & set(UNSUPPORTED_TABLE_CODES)
        for code in sorted(unsupported):
            print(f"[warn] Unsupported code, skipping {code}: "
                  f"{UNSUPPORTED_TABLE_CODES[code]}", file=sys.stderr)
        selected = {
            _normalize_pdd_code(c)
            for c in raw_selected
            if c not in UNSUPPORTED_TABLE_CODES
        }
        codes = [c for c in codes if c in selected]
        missing = selected - set(codes)
        if missing:
            print(f"[warn] Unknown codes, skipping: {sorted(missing)}", file=sys.stderr)

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    codec = FactorizedSpaceCodec()
    pcodec = PairedSpaceCodec()

    brake_cfg = {
        "n_v0_samples": args.n_v0_samples if args.n_v0_samples is not None else args.n_velocity_samples,
        "brake_decel": args.brake_decel,
        "brake_delay": args.brake_delay,
        "brake_margin": args.brake_margin,
        "v0_min_excess": args.v0_min_excess,
        "v0_max_kmh": args.v0_max_kmh,
    }

    plans: List[dict] = []
    for code in codes:
        source = DEFAULT_SOURCE_PLAN[code]
        plan = _plan_for_sign(
            code, source,
            target=args.target_per_sign,
            min_synthetic_for_cp=args.min_synthetic_for_cp,
            codec=codec, pcodec=pcodec,
            scenes_root=args.scenes_root,
            n_variations=args.n_variations,
            n_velocity_samples=args.n_velocity_samples,
            seed=args.seed,
            brake_cfg=brake_cfg,
        )
        plans.append(plan)
        if not args.dry_run:
            _write_sign(output_dir, plan)

    # Print summary table
    print("\nPer-sign plan summary:")
    print(f"  {'PDD':8s} {'src':4s} {'pipe':8s} {'real':>6s} {'syn':>6s} {'total':>6s} "
          f"{'ok?':>4s}  name")
    for p in plans:
        c = p["counts"]
        print(f"  {p['pdd_code']:8s} {p['source_label']:4s} {p['pipeline']:8s} "
              f"{c['p_taken']:>6d} {c['c_taken']:>6d} {c['total_base_scenes']:>6d} "
              f"{'yes' if c['reached_target'] else 'NO':>4s}  {p['human_name']}")

    if not args.dry_run:
        # Summary file
        summary = {
            "target_per_sign": args.target_per_sign,
            "seed": args.seed,
            "per_sign": [
                {"pdd_code": p["pdd_code"], **p["counts"], "pipeline": p["pipeline"]}
                for p in plans
            ],
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote per-sign manifests to {output_dir}")

    if args.materialize and not args.dry_run:
        skip = {s.strip() for s in args.skip_materialize.split(",") if s.strip()}
        _materialize(plans, output_dir, args, skip=skip)


def _materialize(plans: List[dict], output_dir: Path, args, skip: set) -> None:
    """Dispatch materialization to the right backends per sign."""
    from collections import Counter as _Counter
    summary = {"pgmap": {}, "paired": {}, "citymap": {}, "sumo": {}}

    # Group specs by pipeline/source
    pgmap_jobs: Dict[str, List[dict]] = {}
    paired_jobs: Dict[str, List[dict]] = {}
    citymap_jobs: Dict[str, dict] = {}
    sumo_manifests: Dict[str, Path] = {}

    for p in plans:
        code_dir = output_dir / p["pdd_code"].replace(".", "_")
        pgmap = [s for s in p["synthetic_specs"] if s.get("source") == "pgmap"]
        paired = [s for s in p["synthetic_specs"] if s.get("source") == "pgmap_paired"]
        if pgmap:
            pgmap_jobs[p["pdd_code"]] = pgmap
        if paired:
            paired_jobs[p["pdd_code"]] = paired
        if p["citymap_plan"]:
            citymap_jobs[p["pdd_code"]] = p["citymap_plan"]
        if p["real_rows"]:
            sumo_manifests[p["pdd_code"]] = code_dir / "real_manifest.jsonl"

    # PGMap (single-sign)
    if "pgmap" not in skip and pgmap_jobs:
        from factorized_space.benchmark_runner import validate_and_materialize
        for code, specs in pgmap_jobs.items():
            print(f"\n[pgmap] {code}: {len(specs)} base specs × {args.n_velocity_samples} velocities")
            results = validate_and_materialize(
                specs, n_velocity_samples=args.n_velocity_samples, verbose=False,
            )
            out_path = output_dir / code.replace(".", "_") / "pgmap_materialized.jsonl"
            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, default=str) + "\n")
            valid = sum(1 for r in results if r.get("valid"))
            summary["pgmap"][code] = {"total": len(results), "valid": valid}
            print(f"  [pgmap] {code}: {valid}/{len(results)} valid → {out_path.name}")

    # Paired
    if "paired" not in skip and paired_jobs:
        from factorized_space.benchmark_runner import validate_and_materialize_paired
        for code, specs in paired_jobs.items():
            # Only materialize paired specs once per bundle (shared between two PDDs).
            # Dedup by paired_flat_index within this code's selection.
            seen = set(); unique = []
            for s in specs:
                idx = s.get("paired_flat_index")
                if idx in seen: continue
                seen.add(idx); unique.append(s)
            print(f"\n[paired] {code}: {len(unique)} unique paired specs × {args.n_velocity_samples}")
            results = validate_and_materialize_paired(
                unique, n_velocity_samples=args.n_velocity_samples, verbose=False,
            )
            out_path = output_dir / code.replace(".", "_") / "paired_materialized.jsonl"
            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, default=str) + "\n")
            valid = sum(1 for r in results if r.get("valid"))
            summary["paired"][code] = {"total": len(results), "valid": valid}
            print(f"  [paired] {code}: {valid}/{len(results)} valid → {out_path.name}")

    # CityMap
    if "citymap" not in skip and citymap_jobs:
        from citymap_space.citymap_pipeline import run_pipeline as run_citymap
        for code, plan in citymap_jobs.items():
            cfg = plan["citymap_config"]
            out_sub = output_dir / code.replace(".", "_") / "citymap"
            out_sub.mkdir(parents=True, exist_ok=True)
            print(f"\n[citymap] {code}: {len(cfg['map_seeds'])} maps × "
                  f"{cfg['max_scenes_per_sign_per_map']}/sign")
            run_citymap(
                map_seeds=cfg["map_seeds"][:args.citymap_maps],
                n_blocks=cfg["n_blocks"],
                max_scenes_per_sign_per_map=cfg["max_scenes_per_sign_per_map"],
                sign_categories=cfg["sign_categories"],
                spawn_velocities=None,                       # nuPlan sampling
                n_velocity_samples=args.n_velocity_samples,
                output_dir=str(out_sub),
            )
            # Filter by target sign_key
            manifest = out_sub / "citymap_manifest.json"
            if manifest.exists():
                all_rows = json.load(open(manifest))
                target_keys = set(plan["sign_keys"])
                filtered = [r for r in all_rows if r.get("sign_type") in target_keys]
                target_n = int(plan.get("target", len(filtered)))
                filtered = (
                    [r for r in filtered if r.get("valid")] +
                    [r for r in filtered if not r.get("valid")]
                )[:target_n]
                filt_path = out_sub.parent / "citymap_materialized.jsonl"
                with open(filt_path, "w") as f:
                    for r in filtered:
                        f.write(json.dumps(r, default=str) + "\n")
                valid = sum(1 for r in filtered if r.get("valid"))
                summary["citymap"][code] = {"total": len(filtered), "valid": valid}
                print(f"  [citymap] {code}: {valid}/{len(filtered)} valid → {filt_path.name}")

    # SUMO — uses its own pipeline with multiprocessing
    if "sumo" not in skip and sumo_manifests:
        from sumo_space.sumo_pipeline import run_pipeline as run_sumo
        for code, real_path in sumo_manifests.items():
            out_sub = output_dir / code.replace(".", "_") / "sumo"
            out_sub.mkdir(parents=True, exist_ok=True)
            print(f"\n[sumo] {code}: catalog={real_path.name}")
            try:
                run_sumo(
                    catalog_path=real_path,
                    scenes_root=Path(args.scenes_root),
                    n_per_category=args.target_per_sign,
                    n_variations=args.n_variations,
                    sign_categories=[code],
                    spawn_velocities=[0.0],            # legacy arg; runner re-samples
                    density_cap=1.0,
                    drive_agent=False,
                    policy="idm",
                    max_steps=args.horizon,
                    output_dir=out_sub,
                    n_workers=args.sumo_workers,
                    seed=args.seed,
                )
                sumo_manifest = out_sub / "sumo_manifest.jsonl"
                if sumo_manifest.exists():
                    n = sum(1 for _ in open(sumo_manifest))
                    valid = sum(1 for l in open(sumo_manifest) if json.loads(l).get("valid"))
                    summary["sumo"][code] = {"total": n, "valid": valid}
            except Exception as exc:
                print(f"  [sumo] {code}: FAILED ({type(exc).__name__}: {exc})")
                summary["sumo"][code] = {"error": str(exc)}

    # Write aggregate materialization summary
    mat_summary_path = output_dir / "materialization_summary.json"
    with open(mat_summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMaterialization summary → {mat_summary_path}")


if __name__ == "__main__":
    main()
