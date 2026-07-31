#!/usr/bin/env python3
"""Evaluate a plant2_rule checkpoint on ready test catalogs for selected PDD signs.

Uses existing test manifests (``catalog_test20.jsonl`` / ``input_manifest.jsonl``).
Does not regenerate catalogs.

Default policy: ``plant2_rule``. Pass your checkpoint via ``--model-paths``.

Examples:
  # list jobs + catalog availability
  python eval_checkpoint_on_test.py --list

  # dry-run (print commands only)
  python eval_checkpoint_on_test.py \\
      --model-paths plant2_rule:/path/to/your.ckpt \\
      --dry-run

  # smoke: only first N rows from each manifest
  python eval_checkpoint_on_test.py \\
      --model-paths plant2_rule:/path/to/your.ckpt \\
      --n-scenes 2 --only 2.1,2.4

  # full test for all signs in this package
  python eval_checkpoint_on_test.py \\
      --model-paths plant2_rule:/path/to/your.ckpt \\
      --jobs 8 --keep-going
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
PER_SIGN = HERE.parent
PDD_BENCH = PER_SIGN.parent.parent  # .../pdd-bench
MANIFESTS = HERE / "manifests"


@dataclass
class Job:
    """One eval_pipeline invocation on a filtered test catalog."""

    label: str
    catalog: Path
    codes: list[str]
    # Bench folder that owns eval_pipeline.py + scenes/.
    bench: str
    # If set: scenes root = <bench>/scenes/<scenes_subdir>
    scenes_subdir: Optional[str] = None
    # If True: rewrite net_path → <pdd_slug>/<net_path>
    prefix_net_with_code_slug: bool = False
    # If True: scenes root = <bench>/scenes (multi-code tree).
    # Set when net_path is already prefixed (or will be) and scenes live under scenes/<slug>/.
    mixed_scenes_root: bool = False


def _slug(code: str) -> str:
    return str(code).replace(".", "_")


JOBS: list[Job] = [
    Job(
        "2.1",
        MANIFESTS / "2_1_catalog_test20.jsonl",
        ["2.1"],
        bench="main_sign",
        scenes_subdir="2_1",
    ),
    Job(
        "2.3.1-2.3.3",
        MANIFESTS / "2_3_catalog_test20.jsonl",
        ["2.3.1", "2.3.2", "2.3.3"],
        bench="secondary_sign",
        scenes_subdir="2_3",
    ),
    Job(
        "2.4",
        MANIFESTS / "2_4_catalog_test20.jsonl",
        ["2.4"],
        bench="yield_sign",
        scenes_subdir="2_4",
    ),
    Job(
        "2.5",
        MANIFESTS / "2_5_catalog_test20.jsonl",
        ["2.5"],
        bench="stop_sign",
        scenes_subdir="2_5",
    ),
    # Combined test split (303 rows). net_path needs 3_1/ / 3_2/ prefix.
    Job(
        "3.1-3.2",
        MANIFESTS / "3_1_3_2_catalog_test20.jsonl",
        ["3.1", "3.2"],
        bench="no_entry_signs",
        prefix_net_with_code_slug=True,
        mixed_scenes_root=True,
    ),
    # Ready eval input from previous test20 batch (already filtered).
    Job(
        "4.3",
        MANIFESTS / "4_3_input_manifest.jsonl",
        ["4.3"],
        bench="roundabout_sign",
        scenes_subdir="4_3",
    ),
    # Prefixed net_path already (5_7_1/…, 5_7_2/…).
    Job(
        "5.7.1-5.7.2",
        MANIFESTS / "5_7_input_manifest.jsonl",
        ["5.7.1", "5.7.2"],
        bench="one_way_signs",
        mixed_scenes_root=True,
    ),
    # Prefixed net_path already (5_15_1/…).
    Job(
        "5.15.1-5.15.2",
        MANIFESTS / "5_15_input_manifest.jsonl",
        ["5.15.1", "5.15.2"],
        bench="lane_direction_signs",
        mixed_scenes_root=True,
    ),
    Job(
        "5.19",
        MANIFESTS / "5_19_input_manifest.jsonl",
        ["5.19"],
        bench="crosswalk_sign",
        scenes_subdir="5_19",
    ),
]


def _row_code(row: dict) -> str:
    return str(row.get("pdd_code") or row.get("sign_code") or "").strip()


def _resolve_catalog(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    return path


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
            if want and _row_code(row) not in want:
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
    catalog = _resolve_catalog(job.catalog)
    if not catalog.is_file():
        raise FileNotFoundError(f"catalog missing: {job.catalog} (resolved={catalog})")
    rows = _load_filtered(catalog, job.codes, max_scenes)
    if not rows:
        raise RuntimeError(f"no rows for codes={job.codes} in {catalog}")
    rows = _rewrite_net_paths(rows, prefix_slug=job.prefix_net_with_code_slug)
    out = work_dir / f"catalog_test_{job.label.replace('.', '_').replace('-', '_')}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return out, len(rows)


def _eval_pipeline(job: Job) -> Path:
    return PER_SIGN / job.bench / "eval_pipeline.py"


def _scenes_root(job: Job) -> Path:
    bench = PER_SIGN / job.bench
    if job.prefix_net_with_code_slug or job.mixed_scenes_root:
        return bench / "scenes"
    assert job.scenes_subdir, f"{job.label}: need scenes_subdir or mixed_scenes_root"
    return bench / "scenes" / job.scenes_subdir


def _out_dir(job: Job, run_name: str) -> Path:
    slug = job.label.replace(".", "_").replace("-", "_")
    return HERE / "output" / run_name / slug


def _cwd(job: Job) -> Path:
    return PER_SIGN / job.bench


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--policies",
        default="plant2_rule",
        help="comma-separated policies (default: plant2_rule)",
    )
    ap.add_argument(
        "--model-paths",
        default=None,
        help=(
            "checkpoints for NN policies, 'policy:path,...'. "
            "Required for plant2_rule unless the bench has a working default."
        ),
    )
    ap.add_argument("--jobs", type=int, default=8, help="eval_pipeline --jobs (per-scene workers)")
    ap.add_argument(
        "--scenes-per-job",
        type=int,
        default=1,
        help=(
            "scenes handled by one run_benchmark.py process "
            "(forwarded to eval_pipeline; default: 1)"
        ),
    )
    ap.add_argument(
        "--n-scenes",
        "--max-scenes",
        dest="n_scenes",
        type=int,
        default=None,
        metavar="N",
        help=(
            "use only the first N rows from each sign's test manifest "
            "(smoke / partial run). Default: full test catalog."
        ),
    )
    ap.add_argument(
        "--only",
        default=None,
        help="comma-separated job labels (default: all packaged signs)",
    )
    ap.add_argument(
        "--run-name",
        default="plant2_rule_test",
        help="subdir under plant2_rule_test/output/",
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="filtered manifests dir (default: plant2_rule_test/work/<run-name>)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print commands, do not run")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed job")
    ap.add_argument("--save-gifs", action="store_true", help="pass --save-gifs to eval_pipeline")
    ap.add_argument(
        "--plant2-action-mode",
        default="pid",
        choices=["pid", "wps_pure_pursuit"],
        help="PlanT2 action mode (default: pid)",
    )
    ap.add_argument("--list", action="store_true", help="list jobs and exit")
    args = ap.parse_args()

    only = None
    if args.only:
        only = {x.strip() for x in args.only.split(",") if x.strip()}

    jobs = [j for j in JOBS if only is None or j.label in only]
    if only is not None:
        known = {j.label for j in JOBS}
        unknown = only - known
        if unknown:
            print(f"[warn] unknown labels: {sorted(unknown)}", file=sys.stderr)

    if args.list:
        print(f"{'label':18s}  {'rows':>5s}  catalog  scenes  eval_pipeline")
        for j in JOBS:
            cat = _resolve_catalog(j.catalog)
            ok_cat = cat.is_file()
            n = 0
            if ok_cat:
                n = len(_load_filtered(cat, j.codes, None))
            scenes = _scenes_root(j)
            ev = _eval_pipeline(j)
            print(
                f"{j.label:18s}  {n:5d}  "
                f"catalog={'OK' if ok_cat else 'MISSING'}  "
                f"scenes={'OK' if scenes.is_dir() else 'MISSING'}  "
                f"eval={'OK' if ev.is_file() else 'MISSING'}  "
                f"codes={j.codes}"
            )
            if ok_cat:
                print(f"{'':18s}         → {cat}")
        return

    if not jobs:
        sys.exit("no jobs selected")

    if "plant2_rule" in {p.strip() for p in args.policies.split(",") if p.strip()}:
        if not args.model_paths or "plant2_rule:" not in args.model_paths:
            print(
                "[warn] plant2_rule selected but --model-paths has no plant2_rule:… entry; "
                "eval_pipeline will try the repo default checkpoint.",
                file=sys.stderr,
            )

    work_dir = args.work_dir or (HERE / "work" / args.run_name)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.scenes_per_job < 1:
        sys.exit("--scenes-per-job must be >= 1")

    print(f"Selected {len(jobs)} job(s); work_dir={work_dir}")
    print(
        f"Policies: {args.policies}  jobs={args.jobs}  "
        f"scenes_per_job={args.scenes_per_job}  n_scenes={args.n_scenes}"
    )
    print(f"model-paths: {args.model_paths}")
    print(f"Output root: {HERE / 'output' / args.run_name}")

    failed: list[str] = []
    skipped: list[str] = []
    for i, job in enumerate(jobs, 1):
        print(f"\n======== [{i}/{len(jobs)}] {job.label} ========")
        eval_py = _eval_pipeline(job)
        if not eval_py.is_file():
            print(f"[skip] no eval_pipeline.py: {eval_py}")
            skipped.append(job.label)
            continue

        try:
            manifest, n = _prepare_manifest(job, work_dir, args.n_scenes)
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
        (out / "source_catalog.txt").write_text(
            f"label={job.label}\n"
            f"catalog={_resolve_catalog(job.catalog)}\n"
            f"codes={job.codes}\n"
            f"n={n}\n"
            f"n_scenes_cap={args.n_scenes}\n"
            f"scenes_root={scenes}\n"
            f"eval_pipeline={eval_py}\n",
            encoding="utf-8",
        )
        shutil.copy2(manifest, out / "input_catalog_test.jsonl")

        cmd = [
            sys.executable,
            str(eval_py),
            "--policies",
            args.policies,
            "--manifest",
            str(manifest),
            "--scenes-root",
            str(scenes),
            "--out-dir",
            str(out / "eval_out"),
            "--jobs",
            str(args.jobs),
            "--scenes-per-job",
            str(args.scenes_per_job),
            "--plant2-action-mode",
            args.plant2_action_mode,
        ]
        if args.model_paths:
            cmd += ["--model-paths", args.model_paths]
        if args.save_gifs:
            cmd.append("--save-gifs")

        print(f"rows={n}  scenes={scenes}")
        print("$ " + " ".join(cmd))
        if args.dry_run:
            continue

        res = subprocess.run(cmd, cwd=str(_cwd(job)))
        if res.returncode != 0:
            print(f"[FAIL] {job.label} exit={res.returncode}")
            failed.append(job.label)
            if not args.keep_going:
                break
        else:
            report = out / "eval_out" / "reports" / "report_cumulative.md"
            print(f"[ok] {job.label}  report={report if report.is_file() else '(pending)'}")

    out_root = HERE / "output" / args.run_name
    summary_dir = out_root / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Per-sign metrics table (markdown + csv).
    sum_cmd = [
        sys.executable,
        str(HERE / "summarize_reports.py"),
        "--run-name",
        args.run_name,
        "--out-dir",
        str(summary_dir),
    ]
    sum_res = subprocess.run(sum_cmd, cwd=str(HERE))
    metrics_md = summary_dir / "summary.md"

    # Run status as markdown (failed / skipped / paths).
    ok_labels = [
        j.label
        for j in jobs
        if j.label not in failed and j.label not in skipped
    ]
    status_lines = [
        f"# plant2_rule eval run — `{args.run_name}`",
        "",
        f"- Policies: `{args.policies}`",
        f"- Jobs (parallel): `{args.jobs}`",
        f"- n_scenes: `{args.n_scenes}`",
        f"- plant2-action-mode: `{args.plant2_action_mode}`",
        f"- model-paths: `{args.model_paths}`",
        f"- Output: `{out_root}`",
        "",
        "## Status",
        "",
        f"| Status | Signs |",
        f"|---|---|",
        f"| ok | {', '.join(ok_labels) if ok_labels else '—'} |",
        f"| failed | {', '.join(failed) if failed else '—'} |",
        f"| skipped | {', '.join(skipped) if skipped else '—'} |",
        "",
        "## Metrics",
        "",
        f"See [`summary.md`]({metrics_md.name}) "
        f"(regenerate: `python summarize_reports.py --run-name {args.run_name}`).",
        "",
    ]
    if metrics_md.is_file():
        # Embed the metrics table into the run status doc.
        body = metrics_md.read_text(encoding="utf-8").strip()
        # Drop the duplicate top heading from summarize_reports.
        body_lines = body.splitlines()
        if body_lines and body_lines[0].startswith("#"):
            body_lines = body_lines[1:]
            while body_lines and not body_lines[0].strip():
                body_lines = body_lines[1:]
        status_lines.extend(body_lines)
        status_lines.append("")

    status_path = summary_dir / "run_summary.md"
    status_path.write_text("\n".join(status_lines), encoding="utf-8")

    print("\n======== SUMMARY ========")
    print(status_path.read_text(encoding="utf-8"))
    print(f"Wrote {status_path}")
    if sum_res.returncode != 0:
        print(f"[warn] summarize_reports exit={sum_res.returncode}", file=sys.stderr)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
