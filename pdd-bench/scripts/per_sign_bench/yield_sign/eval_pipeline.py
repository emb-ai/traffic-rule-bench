#!/usr/bin/env python3
"""Run any subset of run_benchmark.py policies on a scene/manifest + unified report.

Per-policy rules:
  IDM family ({idm, modified_idm, comprehensive_rule_expert}):
      5 ego-variants are run (default, s1, s2, s3, s4) → 5 baselines per policy.
  NN policies ({rule_compliant, ppo_lidar, carl, carl_rule, plant2, plant2_rule}):
      1 baseline per policy (ego-variant does not apply).
  Checkpoint required for: carl, carl_rule, plant2, plant2_rule.

Supported input modes:
  1. Benchmark run folder:   --manifest <run_dir>  (reads <run_dir>/real_manifest.jsonl,
                             writes results into <run_dir>/eval_out/)
  2. ONE scene:              --manifest <m.jsonl> --scene-line N
  3. WHOLE manifest:         --manifest <m.jsonl>
  4. GROUP of manifests:     --manifest <m1.jsonl> <m2.jsonl> ...
  5. With a row cap:         --manifest ... --max-scenes K

Output layout:
    <out>/
      input_manifest.jsonl                             assembled input
      benchmark/full/policy_eval/<run_name>/           episodes_*.jsonl + summary
      runs/var_0/<run_name>/replays/<sign>/.../*.json  replay.json sidecars
      metrics_per_episode.csv
      aggregations/*.csv
      reports/cumulative.json
      reports/report_cumulative.md                     !!!! final table

Usage examples:
    # Benchmark run folder (reads real_manifest.jsonl, writes to <run_dir>/eval_out/)
    python3 eval_pipeline.py \\
        --policies    idm,modified_idm \\
        --manifest    benchmark_output/2_4/2026-06-11_13-33-40 \\
        --scenes-root scenes

    # IDM only (5 baselines)
    python3 eval_pipeline.py \\
        --policies    idm \\
        --manifest    <m.jsonl> \\
        --scenes-root <scenes>

    # Multiple IDM policies (5+5 = 10 baselines)
    python3 eval_pipeline.py \\
        --policies    idm,modified_idm \\
        --manifest    <m.jsonl> \\
        --scenes-root <scenes>

    # NN with checkpoint
    python3 eval_pipeline.py \\
        --policies    plant2 \\
        --model-paths plant2:/path/to/plant2.ckpt \\
        --manifest    <m.jsonl> \\
        --scenes-root <scenes>

    # Mix: 5 idm + 1 ppo_lidar + 1 carl + 1 plant2 = 8 baselines in one report
    python3 eval_pipeline.py \\
        --policies    idm,ppo_lidar,carl,plant2 \\
        --model-paths carl:/path/carl.ckpt,plant2:/path/plant2.ckpt \\
        --manifest    <m.jsonl> \\
        --scenes-root <scenes>

    # Single scene from manifest
    python3 eval_pipeline.py \\
        --policies   idm --scene-line 1 \\
        --manifest   <m.jsonl> --scenes-root <scenes>

Default: policies=idm, backends=sumo.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Policy categories 
IDM_FAMILY = {"idm", "modified_idm", "comprehensive_rule_expert"}
NN_NEED_CHECKPOINT = {"carl", "carl_rule", "plant2", "plant2_rule"}
NN_NO_CHECKPOINT = {"rule_compliant", "ppo_lidar"}
ALL_POLICIES = IDM_FAMILY | NN_NEED_CHECKPOINT | NN_NO_CHECKPOINT

EGO_VARIANTS = ["default", "s1", "s2", "s3", "s4"]
RUN_MANIFEST_NAME = "real_manifest.jsonl"
RUN_EVAL_OUT_DIRNAME = "eval_out"
BENCH_DIR = Path(__file__).resolve().parent
PDD_BENCH_DIR = BENCH_DIR.parent.parent.parent
CHECKPOINTS_DIR = PDD_BENCH_DIR / "checkpoints"

from lib.manifest_config import (
    DEFAULT_SPAWN_DISTANCE_BEFORE_END,
    enrich_manifest_row,
    load_manifest_config,
)

DEFAULT_MODEL_PATHS: dict[str, Path] = {
    "carl": CHECKPOINTS_DIR / "carl" / "nuplan_51479_1B" / "model_best.pth",
    "carl_rule": CHECKPOINTS_DIR / "carl" / "nuplan_51479_1B" / "model_best.pth",
    "plant2": CHECKPOINTS_DIR / "plant2_finetuned" / "plant2_supervised_2nd_final.pt",
    "plant2_rule": CHECKPOINTS_DIR / "plant2_finetuned" / "plant2_supervised_2nd_final.pt",
}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def parse_model_paths(spec: str | None) -> dict[str, str]:
    """Parse 'policy:path,policy:path' into a dict."""
    out: dict[str, str] = {}
    if not spec:
        return out
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            sys.exit(f"--model-paths: bad item {item!r}; expected 'policy:path'")
        k, v = item.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def resolve_model_paths(spec: str | None, policies: list[str]) -> dict[str, str]:
    """Parse --model-paths and fill in repo defaults for missing NN checkpoints."""
    paths = parse_model_paths(spec)
    for policy in policies:
        if policy not in NN_NEED_CHECKPOINT or policy in paths:
            continue
        default = DEFAULT_MODEL_PATHS.get(policy)
        if default is None:
            continue
        if not default.is_file():
            sys.exit(
                f"No --model-paths entry for {policy!r} and default checkpoint "
                f"not found: {default}"
            )
        paths[policy] = str(default)
        print(f"Using default checkpoint for {policy}: {default}")
    return paths


def resolve_manifest_path(path: Path) -> tuple[Path, Path | None]:
    """Resolve a manifest file or benchmark run directory.

    Returns (manifest_file, run_dir). run_dir is set when path is a directory.
    """
    path = path.expanduser()
    if path.is_file():
        return path.resolve(), None
    if path.is_dir():
        manifest = path / RUN_MANIFEST_NAME
        if not manifest.is_file():
            sys.exit(
                f"No {RUN_MANIFEST_NAME} in run directory: {path.resolve()}"
            )
        return manifest.resolve(), path.resolve()
    sys.exit(f"Manifest path not found: {path}")


def plan_baselines(policies: list[str]) -> list[tuple[str, str]]:
    """Return ordered list of (policy, ego_variant) tuples to run."""
    out: list[tuple[str, str]] = []
    for p in policies:
        if p in IDM_FAMILY:
            for v in EGO_VARIANTS:
                out.append((p, v))
        else:
            out.append((p, "default"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--policies", default="idm",
                   help=("comma-separated list of policies. "
                         f"Supported: {sorted(ALL_POLICIES)} (default: idm). "
                         "IDM family runs 5 ego-variants each; NN policies run once."))
    p.add_argument("--model-paths", default=None,
                   help=("checkpoints for NN policies that need them, "
                         "'policy:path,policy:path'. Defaults to files under "
                         f"{CHECKPOINTS_DIR} when omitted. Required for: "
                         f"{sorted(NN_NEED_CHECKPOINT)}"))
    p.add_argument("--manifest", nargs="+", required=True,
                   help=("one or more manifest .jsonl files, or benchmark run "
                         f"directories containing {RUN_MANIFEST_NAME} "
                         "(space-separated)"))
    p.add_argument("--scenes-root", default="pdd-bench/scenes",
                   help="path to pdd-bench/scenes")
    p.add_argument("--scene-line", type=int, default=None,
                   help="(optional) 1-based row number — pick a single scene; "
                        "only works when exactly ONE --manifest is given")
    p.add_argument("--max-scenes", type=int, default=None,
                   help="(optional) cap on total number of scenes after merging")
    p.add_argument("--backends", default="sumo",
                   help="comma-separated backends (default: sumo)")
    p.add_argument("--plant2-action-mode", default="pid",
                   choices=["pid", "wps_pure_pursuit"],
                   help="PlanT2 action mode (default: pid)")
    p.add_argument("--out-dir", default=None,
                   help=("output directory (default: ./eval_out, or "
                         f"<run_dir>/{RUN_EVAL_OUT_DIRNAME} when --manifest is a folder)"))
    p.add_argument("--no-auxiliary-agent", action="store_true",
                   help="Disable auxiliary agents on main road")
    args = p.parse_args()

    manifest_paths: list[Path] = []
    run_dirs: list[Path] = []
    for raw in args.manifest:
        manifest_path, run_dir = resolve_manifest_path(Path(raw))
        manifest_paths.append(manifest_path)
        if run_dir is not None:
            run_dirs.append(run_dir)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir).resolve()
    elif len(run_dirs) == 1 and len(args.manifest) == 1:
        out_dir = run_dirs[0] / RUN_EVAL_OUT_DIRNAME
        print(f"Using run eval output directory: {out_dir}")
    else:
        out_dir = Path("./eval_out").resolve()

    # --- 0. Validate policies + model-paths ---------------------------------
    policies = [s.strip() for s in args.policies.split(",") if s.strip()]
    bad = [p for p in policies if p not in ALL_POLICIES]
    if bad:
        sys.exit(f"Unknown policies: {bad}. Supported: {sorted(ALL_POLICIES)}")

    model_paths = resolve_model_paths(args.model_paths, policies)
    missing = [p for p in policies if p in NN_NEED_CHECKPOINT and p not in model_paths]
    if missing:
        sys.exit(f"--model-paths missing entries for: {missing}")

    baselines = plan_baselines(policies)
    print(f"Will run {len(baselines)} baseline(s):")
    for pol, v in baselines:
        print(f"  - {pol}_{v}")

    OUT = out_dir
    OUT.mkdir(parents=True, exist_ok=True)

    # Assemble input manifest (apply experiment defaults from manifest.json)
    all_lines: list[str] = []
    for m_path in manifest_paths:
        config = load_manifest_config(m_path)
        if config:
            cfg_path = m_path.parent / "manifest.json"
            if not cfg_path.is_file():
                cfg_path = m_path.parent / "real_manifest_summary.json"
            print(f"  + {cfg_path}: loaded experiment config")
            if config.get("spawn_distance_before_end") is not None:
                print(f"    spawn_distance_before_end={config['spawn_distance_before_end']}")
        else:
            print(
                f"  + {m_path}: no manifest.json/summary; "
                f"spawn_distance_before_end defaults to {DEFAULT_SPAWN_DISTANCE_BEFORE_END}"
            )
        lines = [ln for ln in m_path.read_text().splitlines() if ln.strip()]
        for ln in lines:
            row = enrich_manifest_row(json.loads(ln), config)
            all_lines.append(json.dumps(row, default=str))
        print(f"  + {m_path}: {len(lines)} rows")
    print(f"Total rows: {len(all_lines)}")

    if args.scene_line is not None:
        if len(manifest_paths) != 1:
            sys.exit("--scene-line only works with a single --manifest")
        if not (1 <= args.scene_line <= len(all_lines)):
            sys.exit(f"--scene-line {args.scene_line} out of range [1, {len(all_lines)}]")
        all_lines = [all_lines[args.scene_line - 1]]
        print(f"Selected row {args.scene_line}")

    if args.max_scenes is not None and args.max_scenes < len(all_lines):
        all_lines = all_lines[: args.max_scenes]
        print(f"Capped to first {args.max_scenes} rows")

    if not all_lines:
        sys.exit("No scenes left after filtering")

    input_manifest = OUT / "input_manifest.jsonl"
    input_manifest.write_text("\n".join(all_lines) + "\n")

    # Quick per-sign summary of the input
    sign_counts: dict[str, int] = {}
    for ln in all_lines:
        r = json.loads(ln)
        s = r.get("sign_code") or r.get("pdd_code") or "?"
        sign_counts[s] = sign_counts.get(s, 0) + 1
    print(f"Signs: {sign_counts}")

    #  Run all baselines 
    # --replay-root writes replays straight into the layout build_csv expects:
    # <OUT>/runs/var_0/<run_name>/replays/<sign>/by_sign/.../replay.json
    bench_root = OUT / "benchmark"
    for policy, variant in baselines:
        run_name = f"{policy}_{variant}"
        replay_root = OUT / "runs" / "var_0" / run_name / "replays"
        cmd = [
            sys.executable, str(BENCH_DIR / "run_benchmark.py"),
            "--policy",           policy,
            "--run-name",         run_name,
            "--manifest",         str(input_manifest),
            "--scenes-root",      args.scenes_root,
            # "--backends",         args.backends,
            "--ego-variant",      variant,
            "--benchmark-output", str(bench_root),
            "--emit-replay-sidecar",
            "--replay-root",      str(replay_root),
            "--save-gifs",
        ]
        if not args.no_auxiliary_agent:
            cmd.append("--auxiliary-agent")
        if policy in NN_NEED_CHECKPOINT:
            cmd += ["--model-path", model_paths[policy]]
        if policy in ("plant2", "plant2_rule"):
            cmd += ["--plant2-action-mode", args.plant2_action_mode]
        run(cmd)

    # metrics pipeline (build --  aggregate -- MD report) 
    no_manifests = OUT / "_no_manifests"  # placeholder for build_csv
    no_manifests.mkdir(exist_ok=True)

    run([
        sys.executable, str(BENCH_DIR.parent / "build_episode_metrics_csv.py"),
        "--runs-root",      str(OUT / "runs"),
        "--out",            str(OUT / "metrics_per_episode.csv"),
        "--vars",           "0",
        "--manifests-root", str(no_manifests),
    ])

    run([
        sys.executable, str(BENCH_DIR.parent / "aggregate_episode_metrics.py"),
        "--csv",     str(OUT / "metrics_per_episode.csv"),
        "--out-dir", str(OUT),
    ])

    run([
        sys.executable, str(BENCH_DIR.parent / "generate_cumulative_markdown_report.py"),
        "--run-root",   str(OUT),
        "--cumulative", str(OUT / "reports" / "cumulative.json"),
    ])

    # --- Done ---------------------------------------------------------------
    report = OUT / "reports" / "report_cumulative.md"
    print("\n" + "=" * 60)
    print("DONE.")
    print(f"  Baselines: {len(baselines)}")
    print(f"  CSV:    {OUT}/metrics_per_episode.csv")
    print(f"  Report: {report}")
    print("=" * 60)


if __name__ == "__main__":
    main()
