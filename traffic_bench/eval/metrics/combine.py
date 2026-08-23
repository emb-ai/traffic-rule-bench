"""Merge per-sign ``metrics_per_episode.csv`` files into one overall report."""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

from traffic_bench.eval.metrics.csv import CSV_COLUMNS
from traffic_bench.eval.run_layout import (
    default_run_manifest_dir,
    point_debug_latest,
    resolve_manifest_in_dir,
)
from traffic_bench.eval.sign_registry import (
    hydra_sign_override,
    profiles_from_sign_value,
    resolve_repo_path,
    resolve_sign_token,
)

ALL_RUNS_REL = "data/runs/_all"


def eval_out_from_manifest_dir(man_dir: Path) -> Path:
    return resolve_manifest_in_dir(man_dir).parent / "eval_out"


def collect_eval_outs(sign_value: str) -> list[tuple[str, Path]]:
    profiles = profiles_from_sign_value(sign_value)
    if profiles is None:
        profiles = [resolve_sign_token(sign_value)]
    found: list[tuple[str, Path]] = []
    missing: list[str] = []
    for profile in profiles:
        name = hydra_sign_override(profile)
        man_dir = default_run_manifest_dir(profile)
        if man_dir is None:
            missing.append(name)
            continue
        try:
            out = eval_out_from_manifest_dir(man_dir)
        except FileNotFoundError:
            missing.append(name)
            continue
        if not (out / "metrics_per_episode.csv").is_file():
            missing.append(name)
            continue
        found.append((name, out))
    if not found:
        raise FileNotFoundError(
            "no per-sign metrics_per_episode.csv "
            f"(missing: {', '.join(missing) or sign_value})"
        )
    if missing:
        print(
            f"[combine] skip {len(missing)} sign(s) without CSV: {', '.join(missing)}",
            file=sys.stderr,
        )
    return found


def infer_folder(eval_outs: list[Path]) -> str:
    names = {path.parent.name for path in eval_outs}
    if names == {"test"}:
        return "test"
    if names == {"train"}:
        return "train"
    return "debug"


def combined_dest(folder: str) -> Path:
    root = resolve_repo_path(ALL_RUNS_REL)
    if folder == "debug":
        dest = root / datetime.now().strftime("debug/%Y-%m-%d_%H-%M-%S")
        dest.mkdir(parents=True, exist_ok=True)
        point_debug_latest(dest)
        return dest
    dest = root / folder
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def concat_csvs(csv_paths: list[Path], out_path: Path) -> int:
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for path in csv_paths:
            with path.open(encoding="utf-8", newline="") as src:
                for row in csv.DictReader(src):
                    writer.writerow(row)
                    n += 1
    return n


def write_combined_report(eval_outs: list[Path], dest: Path) -> Path:
    from traffic_bench.eval.cli import _run_module_main
    from traffic_bench.eval.metrics import aggregate as aggregate_mod
    from traffic_bench.eval.metrics import report as report_mod

    csvs = [path / "metrics_per_episode.csv" for path in eval_outs]
    csvs = [path for path in csvs if path.is_file()]
    if not csvs:
        raise FileNotFoundError("no metrics_per_episode.csv to combine")
    merged = dest / "metrics_per_episode.csv"
    n_rows = concat_csvs(csvs, merged)
    print(f"[combine] {len(csvs)} CSV(s) → {n_rows} rows → {merged}")
    agg = _run_module_main(
        aggregate_mod,
        ["--csv", str(merged), "--out-dir", str(dest)],
    )
    if agg != 0:
        raise RuntimeError(f"metrics aggregate failed (exit {agg})")
    report_path = dest / "reports" / "report_cumulative.md"
    rep = _run_module_main(
        report_mod,
        [
            "--run-root",
            str(dest),
            "--cumulative",
            str(dest / "reports" / "cumulative.json"),
        ],
    )
    if rep != 0:
        raise RuntimeError(f"metrics report failed (exit {rep})")
    return report_path


def combine_from_eval_outs(eval_outs: list[Path], folder: str | None = None) -> Path:
    dest = combined_dest(folder or infer_folder(eval_outs))
    return write_combined_report(eval_outs, dest)


def main() -> None:
    raw_sign: str | None = None
    folder: str | None = None
    for arg in sys.argv[1:]:
        if arg in ("-h", "--help"):
            print(
                "usage: python -m traffic_bench.eval metrics combine sign=all\n\n"
                "Merge per-sign metrics_per_episode.csv into one overall report "
                "at data/runs/_all/."
            )
            return
        if arg.startswith("sign="):
            raw_sign = arg.split("=", 1)[1]
        elif arg.startswith("paths.split="):
            folder = arg.split("=", 1)[1].strip() or None
    if not raw_sign:
        print(
            "ERROR: metrics combine needs sign=all (or a comma-separated list)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    pairs = collect_eval_outs(raw_sign)
    outs = [path for _, path in pairs]
    report = combine_from_eval_outs(outs, folder)
    print(f"\n======== combined report ({len(pairs)} signs) ========")
    print(report)


if __name__ == "__main__":
    main()
