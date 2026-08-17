#!/usr/bin/env python3
"""Filter final_metrics / combined episode CSVs to the test split and rebuild reports.

Takes existing ``metrics_per_episode.csv`` under ``final_metrics_v1/eval_out/``,
keeps rows whose ``scene_id`` is in ``catalog_test20.jsonl``, then re-runs
``aggregate_episode_metrics.py`` + ``generate_cumulative_markdown_report.py``.

Output layout (mirrors eval_out reports):
  <sign>/final_metrics_v1/eval_out_test/
      metrics_per_episode.csv
      aggregations/
      reports/{cumulative.json, cumulative_2node.json, report_cumulative.md}
      source_catalog.txt
  <bench>/benchmark_output/combined/eval_out_test/   # for multi-code jobs

Examples:
  python filter_final_metrics_to_test.py --only 3.1,3.2,3.1-3.2,5.7.1,5.7.2,5.7.1-5.7.2
  python filter_final_metrics_to_test.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PER_SIGN = Path(__file__).resolve().parent
AGG = PER_SIGN / "aggregate_episode_metrics.py"
MD = PER_SIGN / "generate_cumulative_markdown_report.py"


@dataclass(frozen=True)
class Job:
    label: str
    # (metrics_csv, pdd_codes_to_keep_from_that_csv)
    sources: tuple[tuple[Path, tuple[str, ...]], ...]
    catalog: Path
    out_dir: Path


def _fm(bench: str, slug: str) -> Path:
    return (
        PER_SIGN / bench / "benchmark_output" / slug
        / "final_metrics_v1" / "eval_out" / "metrics_per_episode.csv"
    )


def _cat(bench: str) -> Path:
    return PER_SIGN / bench / "benchmark_output" / "combined" / "catalog_test20.jsonl"


def _out_sign(bench: str, slug: str) -> Path:
    return (
        PER_SIGN / bench / "benchmark_output" / slug
        / "final_metrics_v1" / "eval_out_test"
    )


def _out_combined(bench: str) -> Path:
    return PER_SIGN / bench / "benchmark_output" / "combined" / "eval_out_test"


JOBS: list[Job] = [
    Job(
        "3.1",
        ((_fm("no_entry_signs", "3_1"), ("3.1",)),),
        _cat("no_entry_signs"),
        _out_sign("no_entry_signs", "3_1"),
    ),
    Job(
        "3.2",
        ((_fm("no_entry_signs", "3_2"), ("3.2",)),),
        _cat("no_entry_signs"),
        _out_sign("no_entry_signs", "3_2"),
    ),
    Job(
        "3.1-3.2",
        (
            (_fm("no_entry_signs", "3_1"), ("3.1",)),
            (_fm("no_entry_signs", "3_2"), ("3.2",)),
        ),
        _cat("no_entry_signs"),
        _out_combined("no_entry_signs"),
    ),
    Job(
        "5.7.1",
        ((_fm("one_way_signs", "5_7_1"), ("5.7.1",)),),
        _cat("one_way_signs"),
        _out_sign("one_way_signs", "5_7_1"),
    ),
    Job(
        "5.7.2",
        ((_fm("one_way_signs", "5_7_2"), ("5.7.2",)),),
        _cat("one_way_signs"),
        _out_sign("one_way_signs", "5_7_2"),
    ),
    Job(
        "5.7.1-5.7.2",
        (
            (_fm("one_way_signs", "5_7_1"), ("5.7.1",)),
            (_fm("one_way_signs", "5_7_2"), ("5.7.2",)),
        ),
        _cat("one_way_signs"),
        _out_combined("one_way_signs"),
    ),
]


def _load_test_scene_ids(catalog: Path, codes: set[str]) -> set[str]:
    ids: set[str] = set()
    with catalog.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            code = str(row.get("pdd_code") or row.get("sign_code") or "").strip()
            if code not in codes:
                continue
            sid = row.get("scene_id")
            if sid:
                ids.add(str(sid))
    return ids


def _filter_csv(src: Path, codes: set[str], scene_ids: set[str]) -> list[dict]:
    rows: list[dict] = []
    with src.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise RuntimeError(f"empty CSV: {src}")
        for row in reader:
            code = (row.get("pdd_code") or "").strip()
            if codes and code and code not in codes:
                continue
            sid = (row.get("scene_id") or "").strip()
            if sid not in scene_ids:
                continue
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def run_job(job: Job, dry_run: bool) -> None:
    all_codes: set[str] = set()
    for _, codes in job.sources:
        all_codes.update(codes)

    if not job.catalog.is_file():
        raise FileNotFoundError(f"catalog missing: {job.catalog}")
    scene_ids = _load_test_scene_ids(job.catalog, all_codes)
    if not scene_ids:
        raise RuntimeError(f"no test scenes for codes={sorted(all_codes)} in {job.catalog}")

    merged: list[dict] = []
    fieldnames: list[str] | None = None
    for src, codes in job.sources:
        if not src.is_file():
            raise FileNotFoundError(f"metrics CSV missing: {src}")
        with src.open(encoding="utf-8") as fh:
            fieldnames = list(csv.DictReader(fh).fieldnames or [])
        part = _filter_csv(src, set(codes), scene_ids)
        print(f"  [{job.label}] {src.parent.parent.name}: kept {len(part)} rows "
              f"(codes={list(codes)}, test_scenes={len(scene_ids)})")
        merged.extend(part)

    if not merged:
        raise RuntimeError(f"{job.label}: no rows after filter")
    assert fieldnames is not None

    out = job.out_dir
    print(f"  [{job.label}] → {out}  (n={len(merged)})")
    if dry_run:
        return

    out.mkdir(parents=True, exist_ok=True)
    csv_out = out / "metrics_per_episode.csv"
    _write_csv(csv_out, merged, fieldnames)
    (out / "source_catalog.txt").write_text(
        f"label={job.label}\n"
        f"catalog={job.catalog}\n"
        f"codes={sorted(all_codes)}\n"
        f"n_test_scenes={len(scene_ids)}\n"
        f"n_episodes={len(merged)}\n"
        f"sources={[str(s) for s, _ in job.sources]}\n",
        encoding="utf-8",
    )

    # Copy filtered catalog rows for provenance (optional).
    cat_out = out / "input_catalog_test.jsonl"
    with job.catalog.open(encoding="utf-8") as fin, cat_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            code = str(row.get("pdd_code") or row.get("sign_code") or "").strip()
            if code in all_codes:
                fout.write(json.dumps(row, default=str) + "\n")

    agg = subprocess.run(
        [sys.executable, str(AGG), "--csv", str(csv_out), "--out-dir", str(out)],
        cwd=str(PER_SIGN),
    )
    if agg.returncode != 0:
        raise RuntimeError(f"aggregate failed for {job.label} (exit={agg.returncode})")

    md = subprocess.run(
        [sys.executable, str(MD), "--run-root", str(out)],
        cwd=str(PER_SIGN),
    )
    if md.returncode != 0:
        raise RuntimeError(f"markdown report failed for {job.label} (exit={md.returncode})")

    report = out / "reports" / "report_cumulative.md"
    print(f"  [ok] {job.label}  report={report}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None,
                    help="comma-separated labels (default: all jobs below)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list:
        for j in JOBS:
            ok = all(s.is_file() for s, _ in j.sources) and j.catalog.is_file()
            print(f"{j.label:14s}  ready={ok}  out={j.out_dir}")
        return

    only = None
    if args.only:
        only = {x.strip() for x in args.only.split(",") if x.strip()}
        unknown = only - {j.label for j in JOBS}
        if unknown:
            print(f"ERROR: unknown labels: {sorted(unknown)}", file=sys.stderr)
            sys.exit(2)

    jobs = [j for j in JOBS if only is None or j.label in only]
    failed: list[str] = []
    for i, job in enumerate(jobs, 1):
        print(f"\n======== [{i}/{len(jobs)}] {job.label} ========")
        try:
            run_job(job, dry_run=args.dry_run)
        except Exception as exc:
            print(f"[FAIL] {job.label}: {exc}", file=sys.stderr)
            failed.append(job.label)

    print("\n======== SUMMARY ========")
    print(f"failed: {failed or 'none'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
