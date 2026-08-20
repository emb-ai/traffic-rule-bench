#!/usr/bin/env python3
"""Run any subset of run_benchmark.py policies on a scene/manifest + unified report.

Per-policy rules:
  IDM family ({idm, comprehensive_rule_expert}):
      5 ego-variants are run (default, s1, s2, s3, s4) → 5 baselines per policy.
  NN policies ({rule_compliant, ppo_lidar, carl, carl_rule, plant2, plant2_rule}):
      1 baseline per policy (ego-variant does not apply).
  Checkpoint required for: carl, carl_rule, plant2, plant2_rule.

Supported input modes:
  1. ONE scene:              --manifest <m.jsonl> --scene-line N
  2. WHOLE manifest:         --manifest <m.jsonl>
  3. GROUP of manifests:     --manifest <m1.jsonl> <m2.jsonl> ...
  4. With a row cap:         --manifest ... --max-scenes K

Output layout:
    <out>/
      input_manifest.jsonl                             assembled input
      benchmark/policy_eval/<run_name>/                episodes_*.jsonl
      runs/var_0/<run_name>/replays/<sign>/.../*.json  replay.json sidecars (opt-in)
      metrics_per_episode.csv
      aggregations/*.csv
      reports/cumulative.json
      reports/report_cumulative.md                     !!!! final table

Usage examples:
    # IDM only (5 baselines)
    python3 eval_pipeline.py \\
        --policies    idm \\
        --manifest    <m.jsonl> \\
        --scenes-root <scenes>

    # Multiple IDM-family policies (5+5 = 10 baselines)
    python3 eval_pipeline.py \\
        --policies    idm,comprehensive_rule_expert \\
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
IDM_FAMILY = {"idm", "comprehensive_rule_expert"}
NN_NEED_CHECKPOINT = {"carl", "carl_rule", "plant2", "plant2_rule"}
NN_NO_CHECKPOINT = {"rule_compliant", "ppo_lidar"}
ALL_POLICIES = IDM_FAMILY | NN_NEED_CHECKPOINT | NN_NO_CHECKPOINT

EGO_VARIANTS = ["default", "s1", "s2", "s3", "s4"]
BENCH_DIR = Path(__file__).resolve().parent

# A single baseline can die on a flaky native crash (e.g. SIGSEGV in
# MetaDrive/panda3d/SUMO). run_benchmark resumes from episodes_*.jsonl, so a
# retry accumulates progress past flaky crashes; after this many attempts we
# SKIP the baseline and continue rather than aborting the whole (multi-policy)
# job and losing every other baseline + the metrics build.
MAX_BASELINE_ATTEMPTS = 2


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


def plan_baselines(policies: list[str],
                   ego_variants: list[str]) -> list[tuple[str, str]]:
    """Return ordered list of (policy, ego_variant) tuples to run.

    IDM-family policies run once per requested ego-variant (`ego_variants`);
    NN policies always run once ("default" — the variant doesn't apply to them).
    """
    out: list[tuple[str, str]] = []
    for p in policies:
        if p in IDM_FAMILY:
            for v in ego_variants:
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
                         "IDM family runs once per --ego-variants; NN policies run once."))
    p.add_argument("--ego-variants", default=",".join(EGO_VARIANTS),
                   help=("comma-separated ego IDM variants to run for IDM-family "
                         f"policies (default: all = {','.join(EGO_VARIANTS)}). "
                         "Use e.g. 'default' to run a single idm baseline. "
                         "Ignored for NN policies."))
    p.add_argument("--model-paths", default=None,
                   help=("checkpoints for NN policies that need them, "
                         "'policy:path,policy:path'. Required for: "
                         f"{sorted(NN_NEED_CHECKPOINT)}"))
    p.add_argument("--manifest", nargs="+", required=True,
                   help="one or more manifest .jsonl files (space-separated)")
    p.add_argument("--scenes-root", required=True,
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
    p.add_argument("--out-dir", default="./eval_out",
                   help="output directory")
    p.add_argument("--emit-replay-sidecar", action="store_true",
                   help="also write replay.json sidecars (expert_actions / per-step "
                        "vars for deep analysis). NOT needed for metrics — the CSV is "
                        "built straight from episodes_*.jsonl.")
    p.add_argument("--save-gifs", action="store_true",
                   help="record a top-down GIF per scene (forwarded to run_benchmark)")
    p.add_argument("--gif-dir", default=None,
                   help="base dir for GIFs (default: <out-dir>/gifs); a per-run "
                        "subfolder is created so policies/variants don't collide")
    args = p.parse_args()

    # --- 0. Validate policies + model-paths ---------------------------------
    policies = [s.strip() for s in args.policies.split(",") if s.strip()]
    bad = [p for p in policies if p not in ALL_POLICIES]
    if bad:
        sys.exit(f"Unknown policies: {bad}. Supported: {sorted(ALL_POLICIES)}")

    model_paths = parse_model_paths(args.model_paths)
    missing = [p for p in policies if p in NN_NEED_CHECKPOINT and p not in model_paths]
    if missing:
        sys.exit(f"--model-paths missing entries for: {missing}")

    ego_variants = [v.strip() for v in args.ego_variants.split(",") if v.strip()]
    bad_v = [v for v in ego_variants if v not in EGO_VARIANTS]
    if bad_v:
        sys.exit(f"Unknown --ego-variants: {bad_v}. Supported: {EGO_VARIANTS}")
    if not ego_variants:
        sys.exit("--ego-variants is empty")

    baselines = plan_baselines(policies, ego_variants)
    print(f"Will run {len(baselines)} baseline(s):")
    for pol, v in baselines:
        print(f"  - {pol}_{v}")

    OUT = Path(args.out_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)

    # Assemble input manifest 
    all_lines: list[str] = []
    for m_path in args.manifest:
        lines = [ln for ln in Path(m_path).read_text().splitlines() if ln.strip()]
        all_lines.extend(lines)
        print(f"  + {m_path}: {len(lines)} rows")
    print(f"Total rows: {len(all_lines)}")

    if args.scene_line is not None:
        if len(args.manifest) != 1:
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
    # Metrics are built from episodes_*.jsonl (always written by run_benchmark) at
    # <bench_root>/policy_eval/<run_name>/episodes_<policy>.jsonl. Sidecars are
    # only written when --emit-replay-sidecar is passed (for deep per-step analysis).
    bench_root = OUT / "benchmark"
    failed: list[str] = []
    for policy, variant in baselines:
        run_name = f"{policy}_{variant}"
        cmd = [
            sys.executable, str(BENCH_DIR / "run_benchmark.py"),
            "--policy",           policy,
            "--run-name",         run_name,
            "--manifest",         str(input_manifest),
            "--scenes-root",      args.scenes_root,
            "--backends",         args.backends,
            "--ego-variant",      variant,
            "--benchmark-output", str(bench_root),
        ]
        if args.emit_replay_sidecar:
            replay_root = OUT / "runs" / "var_0" / run_name / "replays"
            cmd += ["--emit-replay-sidecar", "--replay-root", str(replay_root)]
        if policy in NN_NEED_CHECKPOINT:
            cmd += ["--model-path", model_paths[policy]]
        if policy in ("plant2", "plant2_rule"):
            cmd += ["--plant2-action-mode", args.plant2_action_mode]
        if args.save_gifs:
            gif_base = Path(args.gif_dir) if args.gif_dir else (OUT / "gifs")
            cmd += ["--save-gifs", "--gif-dir", str(gif_base / run_name)]
        # Fault-tolerant: a crash here (flaky native SIGSEGV) must NOT abort the
        # whole job. Retry (run_benchmark resumes from episodes_*.jsonl), then
        # skip the baseline and continue so the remaining policies + the metrics
        # build still run.
        ok = False
        for attempt in range(1, MAX_BASELINE_ATTEMPTS + 1):
            try:
                run(cmd)
                ok = True
                break
            except subprocess.CalledProcessError as exc:
                print(f"!! {run_name} crashed (attempt {attempt}/{MAX_BASELINE_ATTEMPTS}, "
                      f"rc={exc.returncode})", file=sys.stderr)
        if not ok:
            failed.append(run_name)
            print(f"!! SKIPPING {run_name} after {MAX_BASELINE_ATTEMPTS} attempts; "
                  f"continuing with the remaining baselines", file=sys.stderr)

    # metrics pipeline (build --  aggregate -- MD report)
    no_manifests = OUT / "_no_manifests"  # placeholder for build_csv
    no_manifests.mkdir(exist_ok=True)

    run([
        sys.executable, str(BENCH_DIR / "build_episode_metrics_csv.py"),
        "--episodes-root",  str(bench_root / "policy_eval"),
        "--out",            str(OUT / "metrics_per_episode.csv"),
        "--manifests-root", str(no_manifests),
    ])

    run([
        sys.executable, str(BENCH_DIR / "aggregate_episode_metrics.py"),
        "--csv",     str(OUT / "metrics_per_episode.csv"),
        "--out-dir", str(OUT),
    ])

    run([
        sys.executable, str(BENCH_DIR / "generate_cumulative_markdown_report.py"),
        "--run-root",   str(OUT),
        "--cumulative", str(OUT / "reports" / "cumulative.json"),
    ])

    # --- Done ---------------------------------------------------------------
    report = OUT / "reports" / "report_cumulative.md"
    print("\n" + "=" * 60)
    print("DONE.")
    print(f"  Baselines: {len(baselines)} ({len(baselines) - len(failed)} ok, {len(failed)} failed)")
    if failed:
        print(f"  FAILED baselines (skipped): {', '.join(failed)}")
    print(f"  CSV:    {OUT}/metrics_per_episode.csv")
    print(f"  Report: {report}")
    print("=" * 60)


if __name__ == "__main__":
    main()
