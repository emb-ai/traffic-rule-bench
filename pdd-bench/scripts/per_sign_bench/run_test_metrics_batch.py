#!/usr/bin/env python3
"""Batch-run eval_pipeline.py on test catalogs for many PDD signs.

Uses each bench's ``catalog_test20.jsonl`` (under ``benchmark_output/combined/``
or ``…/final_metrics_v1/``), optionally filters by ``pdd_code``, and launches
that bench's ``eval_pipeline.py``.

Examples:
  # dry-run: print commands only
  python run_test_metrics_batch.py --dry-run

  # only a few labels, smoke 2 scenes each
  python run_test_metrics_batch.py --only 2.5,3.1,4.1.1-4.1.6 --max-scenes 2

  # full batch (default policies: idm,carl,plant2,ppo_lidar)
  python run_test_metrics_batch.py --jobs 8

  # custom policies / continue after failures
  python run_test_metrics_batch.py \
    --policies idm,comprehensive_rule_expert,carl,carl_rule,plant2,plant2_rule,ppo_lidar \
    --keep-going
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PER_SIGN = Path(__file__).resolve().parent


@dataclass
class Job:
    """One eval_pipeline invocation."""

    label: str                          # user-facing id, e.g. "3.1-3.2"
    bench: str                          # folder under per_sign_bench/
    catalog: Path                       # source catalog_test20.jsonl
    codes: list[str]                    # keep rows with these pdd/sign codes
    # If set, all rows live under scenes/<scenes_subdir>/ (no net_path rewrite).
    scenes_subdir: Optional[str] = None
    # If True, rewrite net_path → <pdd_slug>/<net_path> and use scenes/ as root
    # (needed when combined catalog mixes several scenes/<slug>/ trees).
    prefix_net_with_code_slug: bool = False


def _slug(code: str) -> str:
    return str(code).replace(".", "_")


def _catalog(bench: str, *parts: str) -> Path:
    return PER_SIGN / bench / "benchmark_output" / Path(*parts)


# Jobs matching the user's requested sign list.
JOBS: list[Job] = [
    Job("2.1", "main_sign",
        _catalog("main_sign", "2_1", "final_metrics_v1", "catalog_test20.jsonl"),
        ["2.1"], scenes_subdir="2_1"),
    Job("2.3.1-2.3.3", "secondary_sign",
        _catalog("secondary_sign", "2_3", "final_metrics_v1", "catalog_test20.jsonl"),
        ["2.3.1", "2.3.2", "2.3.3"], scenes_subdir="2_3"),
    Job("2.4", "yield_sign",
        _catalog("yield_sign", "2_4", "final_metrics_v1", "catalog_test20.jsonl"),
        ["2.4"], scenes_subdir="2_4"),
    Job("2.5", "stop_sign",
        _catalog("stop_sign", "2_5", "final_metrics_v1", "catalog_test20.jsonl"),
        ["2.5"], scenes_subdir="2_5"),
    Job("4.3", "roundabout_sign",
        _catalog("roundabout_sign", "4_3", "final_metrics_v1", "catalog_test20.jsonl"),
        ["4.3"], scenes_subdir="4_3"),
    Job("5.19", "crosswalk_sign",
        _catalog("crosswalk_sign", "5_19", "final_metrics_v1", "catalog_test20.jsonl"),
        ["5.19"], scenes_subdir="5_19"),
    # Only 5.15.1 exists in catalog today; 5.15.2 rows will be empty until added.
    Job("5.15.1-5.15.2", "lane_direction_signs",
        _catalog("lane_direction_signs", "combined", "catalog_test20.jsonl"),
        ["5.15.1", "5.15.2"], prefix_net_with_code_slug=True),
    Job("3.1", "no_entry_signs",
        _catalog("no_entry_signs", "combined", "catalog_test20.jsonl"),
        ["3.1"], scenes_subdir="3_1"),
    Job("3.2", "no_entry_signs",
        _catalog("no_entry_signs", "combined", "catalog_test20.jsonl"),
        ["3.2"], scenes_subdir="3_2"),
    Job("3.1-3.2", "no_entry_signs",
        _catalog("no_entry_signs", "combined", "catalog_test20.jsonl"),
        ["3.1", "3.2"], prefix_net_with_code_slug=True),
    Job("3.18.1", "no_turn_signs",
        _catalog("no_turn_signs", "combined", "catalog_test20.jsonl"),
        ["3.18.1"], scenes_subdir="3_18_1"),
    Job("3.18.2", "no_turn_signs",
        _catalog("no_turn_signs", "combined", "catalog_test20.jsonl"),
        ["3.18.2"], scenes_subdir="3_18_2"),
    Job("3.18.1-3.18.2", "no_turn_signs",
        _catalog("no_turn_signs", "combined", "catalog_test20.jsonl"),
        ["3.18.1", "3.18.2"], prefix_net_with_code_slug=True),
    Job("4.1.1", "direction_signs",
        _catalog("direction_signs", "combined", "catalog_test20.jsonl"),
        ["4.1.1"], scenes_subdir="4_1_1"),
    Job("4.1.2", "direction_signs",
        _catalog("direction_signs", "combined", "catalog_test20.jsonl"),
        ["4.1.2"], scenes_subdir="4_1_2"),
    Job("4.1.3", "direction_signs",
        _catalog("direction_signs", "combined", "catalog_test20.jsonl"),
        ["4.1.3"], scenes_subdir="4_1_3"),
    Job("4.1.4", "direction_signs",
        _catalog("direction_signs", "combined", "catalog_test20.jsonl"),
        ["4.1.4"], scenes_subdir="4_1_4"),
    Job("4.1.5", "direction_signs",
        _catalog("direction_signs", "combined", "catalog_test20.jsonl"),
        ["4.1.5"], scenes_subdir="4_1_5"),
    Job("4.1.6", "direction_signs",
        _catalog("direction_signs", "combined", "catalog_test20.jsonl"),
        ["4.1.6"], scenes_subdir="4_1_6"),
    Job("4.1.1-4.1.6", "direction_signs",
        _catalog("direction_signs", "combined", "catalog_test20.jsonl"),
        ["4.1.1", "4.1.2", "4.1.3", "4.1.4", "4.1.5", "4.1.6"],
        prefix_net_with_code_slug=True),
    Job("5.7.1", "one_way_signs",
        _catalog("one_way_signs", "combined", "catalog_test20.jsonl"),
        ["5.7.1"], scenes_subdir="5_7_1"),
    Job("5.7.2", "one_way_signs",
        _catalog("one_way_signs", "combined", "catalog_test20.jsonl"),
        ["5.7.2"], scenes_subdir="5_7_2"),
    Job("5.7.1-5.7.2", "one_way_signs",
        _catalog("one_way_signs", "combined", "catalog_test20.jsonl"),
        ["5.7.1", "5.7.2"], prefix_net_with_code_slug=True),
]


def _row_code(row: dict) -> str:
    return str(row.get("pdd_code") or row.get("sign_code") or "").strip()


def _load_filtered(catalog: Path, codes: list[str], max_scenes: Optional[int]) -> list[dict]:
    want = set(codes)
    rows: list[dict] = []
    with open(catalog, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("valid") is False:
                continue
            if _row_code(row) not in want:
                continue
            rows.append(row)
            if max_scenes is not None and len(rows) >= max_scenes:
                break
    return rows


def _rewrite_net_paths(rows: list[dict], *, prefix_slug: bool) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        r = dict(row)
        net = str(r.get("net_path") or "")
        if prefix_slug and net and not net.startswith(_slug(_row_code(r)) + "/"):
            r["net_path"] = f"{_slug(_row_code(r))}/{net}"
        out.append(r)
    return out


def _prepare_manifest(job: Job, work_dir: Path, max_scenes: Optional[int]) -> tuple[Path, int]:
    if not job.catalog.is_file():
        raise FileNotFoundError(f"catalog missing: {job.catalog}")
    rows = _load_filtered(job.catalog, job.codes, max_scenes)
    if not rows:
        raise RuntimeError(f"no rows for codes={job.codes} in {job.catalog}")
    rows = _rewrite_net_paths(rows, prefix_slug=job.prefix_net_with_code_slug)
    out = work_dir / f"catalog_test_{job.label.replace('.', '_').replace('-', '_')}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return out, len(rows)


def _scenes_root(job: Job) -> Path:
    bench = PER_SIGN / job.bench
    if job.prefix_net_with_code_slug:
        return bench / "scenes"
    assert job.scenes_subdir
    return bench / "scenes" / job.scenes_subdir


def _out_dir(job: Job, run_name: str) -> Path:
    slug = job.label.replace(".", "_").replace("-", "_")
    return PER_SIGN / job.bench / "benchmark_output" / "test_metrics" / run_name / slug


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", default="idm,carl,plant2,ppo_lidar",
                    help="comma-separated policies for eval_pipeline (default: idm,carl,plant2,ppo_lidar)")
    ap.add_argument("--jobs", type=int, default=8, help="eval_pipeline --jobs (per-scene workers)")
    ap.add_argument("--max-scenes", type=int, default=None,
                    help="cap rows per job (smoke); default = full test catalog")
    ap.add_argument("--only", default=None,
                    help="comma-separated job labels to run (default: all known)")
    ap.add_argument("--run-name", default="test20_batch",
                    help="subdir under each bench's benchmark_output/test_metrics/")
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="where to write filtered manifests (default: <PER>/_batch_test_manifests/<run-name>)")
    ap.add_argument("--dry-run", action="store_true", help="print commands, do not run")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed job")
    ap.add_argument("--save-gifs", action="store_true", help="pass --save-gifs to eval_pipeline")
    ap.add_argument("--list", action="store_true", help="list jobs and exit")
    args = ap.parse_args()

    only = None
    if args.only:
        only = {x.strip() for x in args.only.split(",") if x.strip()}

    jobs = [j for j in JOBS if only is None or j.label in only]
    if args.list or only is not None:
        known = {j.label for j in JOBS}
        if only:
            unknown = only - known
            if unknown:
                print(f"[warn] unknown labels: {sorted(unknown)}", file=sys.stderr)

    if args.list:
        for j in JOBS:
            ok = j.catalog.is_file()
            print(f"{j.label:18s}  bench={j.bench:22s}  catalog={'OK' if ok else 'MISSING'}  codes={j.codes}")
        return

    if not jobs:
        sys.exit("no jobs selected")

    work_dir = args.work_dir or (PER_SIGN / "_batch_test_manifests" / args.run_name)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected {len(jobs)} job(s); work_dir={work_dir}")
    print(f"Policies: {args.policies}  jobs={args.jobs}  max_scenes={args.max_scenes}")

    failed: list[str] = []
    skipped: list[str] = []
    for i, job in enumerate(jobs, 1):
        print(f"\n======== [{i}/{len(jobs)}] {job.label} ({job.bench}) ========")
        eval_py = PER_SIGN / job.bench / "eval_pipeline.py"
        if not eval_py.is_file():
            print(f"[skip] no eval_pipeline.py in {job.bench}")
            skipped.append(job.label)
            continue
        if not job.catalog.is_file():
            print(f"[skip] catalog missing: {job.catalog}")
            skipped.append(job.label)
            continue

        try:
            manifest, n = _prepare_manifest(job, work_dir, args.max_scenes)
        except Exception as exc:
            print(f"[skip] prepare failed: {exc}")
            skipped.append(job.label)
            continue

        scenes = _scenes_root(job)
        if not scenes.is_dir():
            print(f"[skip] scenes-root missing: {scenes}")
            skipped.append(job.label)
            continue

        out = _out_dir(job, args.run_name)
        out.mkdir(parents=True, exist_ok=True)
        # Drop a pointer to source catalog for provenance.
        (out / "source_catalog.txt").write_text(
            f"label={job.label}\ncatalog={job.catalog}\ncodes={job.codes}\nn={n}\n",
            encoding="utf-8",
        )
        shutil.copy2(manifest, out / "input_catalog_test.jsonl")

        cmd = [
            sys.executable, str(eval_py),
            "--policies", args.policies,
            "--manifest", str(manifest),
            "--scenes-root", str(scenes),
            "--out-dir", str(out / "eval_out"),
            "--jobs", str(args.jobs),
        ]
        if args.save_gifs:
            cmd.append("--save-gifs")

        print(f"rows={n}  scenes={scenes}")
        print("$ " + " ".join(cmd))
        if args.dry_run:
            continue

        res = subprocess.run(cmd, cwd=str(PER_SIGN / job.bench))
        if res.returncode != 0:
            print(f"[FAIL] {job.label} exit={res.returncode}")
            failed.append(job.label)
            if not args.keep_going:
                break
        else:
            report = out / "eval_out" / "reports" / "report_cumulative.md"
            print(f"[ok] {job.label}  report={report if report.is_file() else '(pending)'}")

    print("\n======== SUMMARY ========")
    print(f"failed:  {failed or 'none'}")
    print(f"skipped: {skipped or 'none'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
