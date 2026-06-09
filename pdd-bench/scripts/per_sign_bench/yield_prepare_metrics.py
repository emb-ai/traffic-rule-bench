#!/usr/bin/env python3
"""
Launch benchmark runs for all 17 baselines from a JSONL manifest.

Iterates through all baseline configurations (idm variants, ppo, carl, plant2, 
and their rule-augmented versions) and runs yield_run_benchmark_mini.py with 
appropriate arguments.

Usage:
    python yield_prepare_metrics.py --manifest /path/to/pgmap_materialized.jsonl --out-dir /path/to/output
    python yield_prepare_metrics.py --manifest manifest.jsonl --out-dir output --dry-run
    python yield_prepare_metrics.py --manifest manifest.jsonl --out-dir output --baselines idm_default,carl
    python yield_prepare_metrics.py \
    --manifest benchmark_output/fixed/2_4/pgmap_materialized.jsonl \
    --out-dir benchmark_output/fixed/2_4 \
    --rerun-failed \
    --emit-replay-sidecar

All 17 baselines:
  Base versions:
    - idm_default, idm_s1, idm_s2, idm_s3, idm_s4
    - ppo_lidar
    - carl
    - plant2, plant2_artem
  Rule-augmented versions:
    - comprehensive_rule_expert_default, comprehensive_rule_expert_s1, 
      comprehensive_rule_expert_s2, comprehensive_rule_expert_s3, comprehensive_rule_expert_s4
    - rule_compliant (ppo_rule)
    - carl_rule
    - plant2_rule
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_DIR = SCRIPT_PATH.parent
SCRIPTS_DIR = BENCHMARK_DIR.parent
PDD_BENCH_DIR = SCRIPTS_DIR.parent
SDC_ROOT = PDD_BENCH_DIR.parent

RUN_BENCH_SCRIPT = BENCHMARK_DIR / "yield_run_benchmark.py"

import os

# Checkpoint paths — relative to repo root by default, overridable via env vars.
# PATH_PPO_EXPERT = ""
PATH_CARL = os.environ.get(
    "CARL_CKPT",
    str(SDC_ROOT / "CaRL/nuPlan/checkpoints/nuplan_51479_1B/model_best.pth"),
)
PATH_PLANT2 = os.environ.get(
    "PLANT2_CKPT",
    str(PDD_BENCH_DIR / "checkpoints/epoch%3D029_final_3.ckpt"),
)
PATH_ARTEM = os.environ.get(
    "PLANT2_CKPT_FINETUNE",
    str(SDC_ROOT / "expert_selection/plant2_training_out/plant2_supervised_2nd_final.pt"),
)

# =============================================================================
# Baseline Configuration
# =============================================================================

@dataclass
class BaselineConfig:
    """Configuration for a single baseline."""
    name: str                           # Baseline identifier (e.g., "idm_default")
    policy: str                         # Policy argument for --policy
    ego_variant: Optional[str] = None   # Variant for IDM-based policies (default, s1-s4)
    model_path: Optional[str] = None    # Model path for ppo/carl/plant2


# All 17 baselines in canonical order
BASELINE_ORDER = [
    # === Base versions ===
    "idm_default", "idm_s1", "idm_s2", "idm_s3", "idm_s4",
    "ppo_lidar",
    "carl",
    "plant2", "plant2_artem",
    # === Rule-augmented versions ===
    "comprehensive_rule_expert_default",
    "comprehensive_rule_expert_s1",
    "comprehensive_rule_expert_s2",
    "comprehensive_rule_expert_s3",
    "comprehensive_rule_expert_s4",
    "rule_compliant",  # ppo_rule
    "carl_rule",
    "plant2_rule",
]


def _get_baseline_config(baseline: str, model_paths: dict[str, str]) -> BaselineConfig:
    """Map baseline name to policy/variant/model configuration."""
    
    # IDM variants (base)
    if baseline == "idm_default":
        return BaselineConfig(name=baseline, policy="idm", ego_variant="default")
    if baseline == "idm_s1":
        return BaselineConfig(name=baseline, policy="idm", ego_variant="s1")
    if baseline == "idm_s2":
        return BaselineConfig(name=baseline, policy="idm", ego_variant="s2")
    if baseline == "idm_s3":
        return BaselineConfig(name=baseline, policy="idm", ego_variant="s3")
    if baseline == "idm_s4":
        return BaselineConfig(name=baseline, policy="idm", ego_variant="s4")
    
    # Comprehensive rule expert (IDM + rules)
    if baseline == "comprehensive_rule_expert_default":
        return BaselineConfig(name=baseline, policy="comprehensive_rule_expert", ego_variant="default")
    if baseline == "comprehensive_rule_expert_s1":
        return BaselineConfig(name=baseline, policy="comprehensive_rule_expert", ego_variant="s1")
    if baseline == "comprehensive_rule_expert_s2":
        return BaselineConfig(name=baseline, policy="comprehensive_rule_expert", ego_variant="s2")
    if baseline == "comprehensive_rule_expert_s3":
        return BaselineConfig(name=baseline, policy="comprehensive_rule_expert", ego_variant="s3")
    if baseline == "comprehensive_rule_expert_s4":
        return BaselineConfig(name=baseline, policy="comprehensive_rule_expert", ego_variant="s4")
    
    # PPO variants
    if baseline == "ppo_lidar":
        return BaselineConfig(
            name=baseline, 
            policy="ppo_lidar"
        )
    if baseline == "rule_compliant":
        return BaselineConfig(
            name=baseline, 
            policy="rule_compliant"
        )
    
    # CARL variants
    if baseline == "carl":
        return BaselineConfig(
            name=baseline, 
            policy="carl", 
            model_path=model_paths.get("carl")
        )
    if baseline == "carl_rule":
        return BaselineConfig(
            name=baseline, 
            policy="carl_rule", 
            model_path=model_paths.get("carl")
        )
    
    # PLANT2 variants
    if baseline == "plant2":
        return BaselineConfig(
            name=baseline, 
            policy="plant2", 
            model_path=model_paths.get("plant2")
        )
    if baseline == "plant2_artem":
        return BaselineConfig(
            name=baseline, 
            policy="plant2", 
            model_path=model_paths.get("plant2_artem")
        )
    if baseline == "plant2_rule":
        return BaselineConfig(
            name=baseline, 
            policy="plant2_rule", 
            model_path=model_paths.get("plant2")
        )
    
    raise ValueError(f"Unknown baseline: {baseline}")


def _build_command(
    config: BaselineConfig,
    manifest_path: Path,
    out_dir: Path,
    scenes_root: Optional[Path],
    backends: str,
    max_steps: int,
    rerun_failed: bool,
    emit_replay_sidecar: bool,
    extra_args: list[str],
) -> list[str]:
    """Build the command line for yield_run_benchmark_mini.py."""
    
    replay_root = out_dir / "replays"
    
    cmd = [
        sys.executable,
        str(RUN_BENCH_SCRIPT),
        "--policy", config.policy,
        "--run-name", config.name,
        "--manifest", str(manifest_path),
        "--backends", backends,
        "--max-steps", str(max_steps),
    ]
    
    # Add ego-variant for IDM-based policies
    if config.ego_variant is not None:
        cmd.extend(["--ego-variant", config.ego_variant])
    
    # Add model path if required
    if config.model_path is not None:
        cmd.extend(["--model-path", config.model_path])
    
    # Add scenes-root if specified
    if scenes_root is not None:
        cmd.extend(["--scenes-root", str(scenes_root)])
    
    # Add optional flags
    if rerun_failed:
        cmd.append("--rerun-failed")
    
    if emit_replay_sidecar:
        cmd.append("--emit-replay-sidecar")
        cmd.extend(["--replay-root", str(replay_root)])
    
    # Add any extra arguments
    cmd.extend(extra_args)
    
    return cmd


def _iter_jsonl_rows(path: Path):
    """Iterate over JSONL file rows."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def run_all_baselines(
    manifest_path: Path,
    out_dir: Path,
    baselines: list[str],
    model_paths: dict[str, str],
    scenes_root: Optional[Path],
    backends: str,
    max_steps: int,
    rerun_failed: bool,
    emit_replay_sidecar: bool,
    extra_args: list[str],
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, int]:
    """Run benchmarks for all specified baselines.
    
    Returns:
        dict mapping baseline name to exit code (0 = success)
    """
    results: dict[str, int] = {}
    
    print(f"\n{'='*60}")
    print(f"Running {len(baselines)} baselines")
    print(f"{'='*60}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Output dir: {out_dir}")
    print(f"  Backends: {backends}")
    print(f"  Max steps: {max_steps}")
    print(f"  Rerun failed: {rerun_failed}")
    print(f"  Emit replay sidecar: {emit_replay_sidecar}")
    if scenes_root:
        print(f"  Scenes root: {scenes_root}")
    print(f"\nBaselines: {', '.join(baselines)}")
    print()
    
    for i, baseline in enumerate(baselines, start=1):
        print(f"\n[{i}/{len(baselines)}] Running baseline: {baseline}")
        print("-" * 50)
        
        try:
            config = _get_baseline_config(baseline, model_paths)
        except ValueError as e:
            print(f"  [ERROR] {e}")
            results[baseline] = 1
            continue
        
        # Check if model path is required but not provided
        if config.policy in ("carl", "carl_rule", "plant2", "plant2_rule"):
            if config.model_path is None:
                print(f"  [SKIP] Model path required but not provided for {baseline}")
                print(f"         Use --model-paths '{baseline}:/path/to/model' to specify")
                results[baseline] = -1  # Skipped
                continue
        
        cmd = _build_command(
            config=config,
            manifest_path=manifest_path,
            out_dir=out_dir / baseline,
            scenes_root=scenes_root,
            backends=backends,
            max_steps=max_steps,
            rerun_failed=rerun_failed,
            emit_replay_sidecar=emit_replay_sidecar,
            extra_args=extra_args,
        )
        
        if verbose:
            print(f"  Command: {' '.join(cmd[:6])}...")
            print(f"           {' '.join(cmd[6:])}")
        
        if dry_run:
            print("  [DRY RUN] Would execute command")
            results[baseline] = 0
            continue
        
        # Create output directory
        baseline_out_dir = out_dir / baseline
        baseline_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Run the command
        print(f"  [RUNNING]...")
        res = subprocess.run(cmd, cwd=str(BENCHMARK_DIR))
        results[baseline] = res.returncode
        
        if res.returncode == 0:
            print(f"  [OK] Baseline {baseline} completed successfully")
        else:
            print(f"  [FAIL] Baseline {baseline} failed with code {res.returncode}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark for all 17 baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all baselines (IDM variants only, no model paths needed)
  python yield_prepare_metrics.py \\
      --manifest pgmap_materialized.jsonl \\
      --out-dir benchmark_runs

  # Run specific baselines
  python yield_prepare_metrics.py \\
      --manifest pgmap_materialized.jsonl \\
      --out-dir benchmark_runs \\
      --baselines idm_default,idm_s1,carl

  # Run with model paths for trained policies
  python yield_prepare_metrics.py \\
      --manifest pgmap_materialized.jsonl \\
      --out-dir benchmark_runs \\
      --model-paths "ppo_expert:/path/to/ppo.zip,carl:/path/to/carl.pt"

  # Dry run to see commands
  python yield_prepare_metrics.py \\
      --manifest pgmap_materialized.jsonl \\
      --out-dir benchmark_runs \\
      --dry-run

Model paths format:
  --model-paths "baseline1:/path1,baseline2:/path2"
  
  Required for: ppo_expert, rule_compliant, carl, carl_rule, plant2, plant2_artem, plant2_rule
"""
    )
    
    # Required arguments
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to the JSONL manifest file"
    )
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Output directory for results"
    )
    
    # Baseline selection
    parser.add_argument(
        "--baselines", type=str, default=None,
        help="Comma-separated list of baselines to run (default: all 17)"
    )
    parser.add_argument(
        "--skip-baselines", type=str, default=None,
        help="Comma-separated list of baselines to skip"
    )
    
    # Model paths for trained policies
    parser.add_argument(
        "--model-paths", type=str, default="",
        help="Model paths as 'baseline:path,baseline:path' (required for ppo/carl/plant2)"
    )
    
    # Benchmark arguments
    parser.add_argument(
        "--scenes-root", type=str, default=None,
        help="Root directory for scenes"
    )
    parser.add_argument(
        "--backends", type=str, default="pgmap",
        help="Comma-separated backends (default: sumo,pgmap,citymap)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=600,
        help="Maximum simulation steps (default: 600)"
    )
    parser.add_argument(
        "--rerun-failed", action="store_true",
        help="Rerun previously failed scenes"
    )
    parser.add_argument(
        "--emit-replay-sidecar", action="store_true",
        help="Emit per-episode replay.json sidecars"
    )
    
    # Execution options
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print detailed information"
    )
    
    # Pass-through arguments
    parser.add_argument(
        "extra_args", nargs="*",
        help="Additional arguments to pass to yield_run_benchmark_mini.py"
    )
    
    args = parser.parse_args()
    
    # Validate manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"[ERROR] Manifest file not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    
    # Parse model paths
    model_paths: dict[str, str] = {}

    model_paths['carl'] = PATH_CARL
    model_paths['plant2'] = PATH_PLANT2

    model_paths['plant2_artem'] = PATH_ARTEM
    
    model_paths['carl_rule'] = PATH_CARL
    model_paths['plant2_rule'] = PATH_PLANT2
    
    # Determine which baselines to run
    if args.baselines:
        baselines = [b.strip() for b in args.baselines.split(",")]
        # Validate baseline names
        for b in baselines:
            if b not in BASELINE_ORDER:
                print(f"[ERROR] Unknown baseline: {b}", file=sys.stderr)
                print(f"        Valid baselines: {', '.join(BASELINE_ORDER)}", file=sys.stderr)
                sys.exit(1)
    else:
        baselines = list(BASELINE_ORDER)
    
    # Apply skip filter
    if args.skip_baselines:
        skip_set = set(b.strip() for b in args.skip_baselines.split(","))
        baselines = [b for b in baselines if b not in skip_set]
    
    if not baselines:
        print("[ERROR] No baselines to run after applying filters", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    results = run_all_baselines(
        manifest_path=manifest_path,
        out_dir=out_dir,
        baselines=baselines,
        model_paths=model_paths,
        scenes_root=Path(args.scenes_root) if args.scenes_root else None,
        backends=args.backends,
        max_steps=args.max_steps,
        rerun_failed=args.rerun_failed,
        emit_replay_sidecar=args.emit_replay_sidecar,
        extra_args=args.extra_args,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    succeeded = [b for b, code in results.items() if code == 0]
    failed = [b for b, code in results.items() if code > 0]
    skipped = [b for b, code in results.items() if code < 0]
    
    print(f"\n  Succeeded: {len(succeeded)}")
    if succeeded:
        for b in succeeded:
            print(f"    - {b}")
    
    print(f"\n  Failed: {len(failed)}")
    if failed:
        for b in failed:
            print(f"    - {b} (exit code: {results[b]})")
    
    print(f"\n  Skipped: {len(skipped)}")
    if skipped:
        for b in skipped:
            print(f"    - {b}")
    
    # Write results summary
    summary_path = out_dir / "baseline_results.json"
    summary = {
        "manifest": str(manifest_path),
        "baselines_requested": baselines,
        "results": results,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to: {summary_path}")
    
    # Exit with error if any baseline failed
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
