"""Shared eval helpers: Sign SR, FV-fast, tag/ckpt resolution."""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from _env import bench_dir, metrics_root, plan_t, resolve_python, setup_eval_thread_env, signs_dir, trb_root

NFS2 = Path("/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova")
DEFAULT_MANIFEST = NFS2 / "traffic-rule-bench/pdd-bench/benchmark_output_speed/balanced/run_v61_a6/catalog_fv_test20.jsonl"
DEFAULT_SCENES = NFS2 / "traffic-rule-bench/pdd-bench/scenes_balanced"
DEFAULT_MANIFEST_DETOUR = NFS2 / "traffic-rule-bench/pdd-bench/benchmark_output/detour_v1/catalog_fv_test20.jsonl"
DEFAULT_SCENES_DETOUR = NFS2 / "sdc/pdd-bench/scenes"


@dataclass
class SignsEvalConfig:
    ckpt: Path
    tag: str
    gpu: str = "0"
    only_signs: str | None = None
    jobs: int = 8
    scenes_per_job: int = 20
    metrics_root: Path | None = None
    max_retries: int = 3
    python: Path | None = None


def signs_output_dir(tag: str) -> Path:
    return signs_dir() / "output" / tag


def signs_done(tag: str) -> bool:
    return (signs_output_dir(tag) / "_summary/summary.md").is_file()


def fv_done(out_dir: Path) -> bool:
    return (out_dir / "reports/report_cumulative.md").is_file()


def setup_metrics_tag(metrics: Path, tag: str, ckpt: Path) -> Path:
    tag_dir = metrics / tag
    (tag_dir / "logs").mkdir(parents=True, exist_ok=True)
    (tag_dir / "ckpt.txt").write_text(f"{ckpt}\n")
    out = signs_output_dir(tag)
    out.mkdir(parents=True, exist_ok=True)
    link = tag_dir / "signs"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(out)
    return tag_dir


def tag_from_ckpt(ckpt: Path) -> str | None:
    bn = ckpt.name
    patterns = [
        (r"best_(\d+)_fvexp30_sign_(lr[0-9e]+)_", r"fvexp30_sign_\2_best\1"),
        (r"last_ft_fvexp30_sign_(lr[0-9e]+)_", r"fvexp30_sign_\1_lastft"),
        (r"best_(\d+)_fvexp30_spatial_(lr[0-9e]+)_", r"fvexp30_spatial_\2_best\1"),
        (r"epoch=(\d+)_fvexp30_spatial_(lr[0-9e]+)_", r"fvexp30_spatial_\2_ep\1"),
        (r"best_(\d+)_fvexp30_(lr[0-9e]+)_", r"fvexp30_\2_best\1"),
        (r"epoch=(\d+)_fvexp30_(lr[0-9e]+)_", r"fvexp30_\2_ep\1"),
    ]
    for pat, fmt in patterns:
        m = re.search(pat, bn)
        if m:
            return m.expand(fmt)
    return None


def resolve_ckpt_spatial(lr: str, slot: str, ckpt_root: Path | None = None) -> Path | None:
    root = ckpt_root or plan_t() / "checkpoints_ft"
    d = root / f"fvexp30_spatial_lr{lr}"
    if slot == "best":
        hits = sorted(d.glob(f"best_*_fvexp30_spatial_lr{lr}_1.ckpt"))
        return hits[0] if hits else None
    if slot.startswith("ep"):
        ep = slot[2:]
        hits = list(d.glob(f"epoch={ep}_fvexp30_spatial_lr{lr}_1.ckpt"))
        return hits[0] if hits else None
    return None


def make_spatial_tag(lr: str, slot: str, ckpt: Path, suffix: str = "") -> str:
    bn = ckpt.name
    if slot == "best":
        m = re.search(r"best_(\d+)_", bn)
        base = f"fvexp30_spatial_lr{lr}_best{m.group(1)}" if m else f"fvexp30_spatial_lr{lr}_best"
    else:
        m = re.search(r"epoch=(\d+)_", bn)
        base = f"fvexp30_spatial_lr{lr}_ep{m.group(1)}" if m else f"fvexp30_spatial_lr{lr}_{slot}"
    return base + suffix


def run_signs_eval(cfg: SignsEvalConfig) -> int:
    setup_eval_thread_env()
    py = resolve_python(str(cfg.python) if cfg.python else None)
    sd = signs_dir()
    metrics = cfg.metrics_root or metrics_root()
    tag_dir = setup_metrics_tag(metrics, cfg.tag, cfg.ckpt)
    logf = tag_dir / "logs/eval_checkpoint.log"

    if cfg.only_signs:
        (tag_dir / "eval_filter.txt").write_text(f"ONLY_SIGNS={cfg.only_signs}\n")

    if signs_done(cfg.tag):
        print(f"SIGNS SKIP {cfg.tag}")
        return 0

    for attempt in range(1, cfg.max_retries + 1):
        if signs_done(cfg.tag):
            return 0
        print(f"SIGNS START tag={cfg.tag} gpu={cfg.gpu} attempt={attempt}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cfg.gpu
        cmd = [
            str(py), "-u", str(sd / "eval_checkpoint_on_test.py"),
            "--policies", "plant2",
            "--model-paths", f"plant2:{cfg.ckpt}",
            "--jobs", str(cfg.jobs),
            "--scenes-per-job", str(cfg.scenes_per_job),
            "--keep-going",
            "--run-name", cfg.tag,
        ]
        if cfg.only_signs:
            cmd.extend(["--only", cfg.only_signs])
        with logf.open("a") as log:
            rc = subprocess.run(cmd, cwd=str(sd), env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            subprocess.run(
                [str(py), str(sd / "summarize_reports.py"),
                 "--run-name", cfg.tag, "--baseline", "plant2_default",
                 "--out-dir", str(signs_output_dir(cfg.tag) / "_summary")],
                cwd=str(sd), stdout=log, stderr=subprocess.STDOUT,
            )
        if signs_done(cfg.tag):
            print(f"SIGNS DONE {cfg.tag} rc={rc}")
            return 0
        print(f"SIGNS FAIL {cfg.tag} rc={rc}")
        time.sleep(10)
    return 1


def _filter_manifest(manifest: Path, out: Path, exclude_codes: list[str]) -> Path:
    if not exclude_codes:
        return manifest
    excl = "|".join(re.escape(c) for c in exclude_codes)
    pat = re.compile(rf'"sign_code"\s*:\s*"({excl})"')
    lines = [ln for ln in manifest.read_text().splitlines() if ln.strip() and not pat.search(ln)]
    out.write_text("\n".join(lines) + ("\n" if lines else ""))
    return out


def run_fv_fast(
    *,
    ckpt: Path,
    out: Path,
    manifest: Path = DEFAULT_MANIFEST,
    scenes: Path = DEFAULT_SCENES,
    gpus: list[str] | None = None,
    nshards: int = 28,
    concurrency: int = 28,
    exclude_codes: list[str] | None = None,
    max_steps: int = 1500,
    python: Path | None = None,
) -> int:
    setup_eval_thread_env()
    py = resolve_python(str(python) if python else None)
    repo = trb_root() / "pdd-bench"
    gpus = gpus or ["0", "1", "2", "3", "4", "5", "6"]

    if "catalog_fv_test20.jsonl" not in str(manifest):
        raise SystemExit(f"MANIFEST must be catalog_fv_test20.jsonl, got {manifest}")
    rows = sum(1 for _ in manifest.open())
    if rows > 5000:
        raise SystemExit(f"MANIFEST has {rows} rows (>5000)")

    out.mkdir(parents=True, exist_ok=True)
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "parts").mkdir(exist_ok=True)

    src_manifest = _filter_manifest(
        manifest, out / "manifest_filtered.jsonl", exclude_codes or ["3.25", "5.22", "5.32"]
    )
    lines = [ln for ln in src_manifest.read_text().splitlines() if ln.strip()]
    for old in shard_dir.glob("shard_*.jsonl"):
        old.unlink()
    shards: list[Path] = []
    for i, line in enumerate(lines):
        sf = shard_dir / f"shard_{i % nshards:02d}.jsonl"
        with sf.open("a") as f:
            f.write(line + "\n")
        if sf not in shards:
            shards.append(sf)

    done_dir = out / "_done"
    done_dir.mkdir(exist_ok=True)
    bench = repo / "scripts/per_sign_bench"

    def run_one(gpu: str, shard_file: Path, sidx: int) -> int:
        tag = f"plant2_default_s{sidx:02d}"
        if (done_dir / f"{tag}.ok").is_file():
            return 0
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        args = [
            str(py), str(bench / "run_benchmark.py"),
            "--policy", "plant2",
            "--run-name", "plant2_default",
            "--ego-variant", "default",
            "--manifest", str(shard_file),
            "--scenes-root", str(scenes),
            "--backends", "sumo",
            "--max-steps", str(max_steps),
            "--benchmark-output", str(out / "parts" / tag / "benchmark"),
            "--model-path", str(ckpt),
            "--plant2-action-mode", "pid",
        ]
        logf = out / "logs" / f"{tag}.log"
        with logf.open("a") as log:
            rc = subprocess.run(args, cwd=str(repo), env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        (done_dir / f"{tag}.{'ok' if rc == 0 else 'fail'}").write_text("")
        return rc

    jobs: list[tuple[str, Path, int]] = []
    for sidx, sf in enumerate(sorted(shard_dir.glob("shard_*.jsonl"))):
        gpu = gpus[sidx % len(gpus)]
        jobs.append((gpu, sf, sidx))

    fail = 0
    with ProcessPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(run_one, g, sf, i) for g, sf, i in jobs]
        for fut in as_completed(futs):
            if fut.result() != 0:
                fail += 1

    empty = out / "_no_manifests"
    empty.mkdir(exist_ok=True)
    comb = out / "metrics_per_episode.csv"
    comb.write_text("")
    for pe in sorted(out.glob("parts/*/policy_eval")):
        tag = pe.parent.parent.name
        csv = out / "parts" / f"_csv_{tag}.csv"
        subprocess.run(
            [str(py), str(bench / "build_episode_metrics_csv.py"),
             "--episodes-root", str(pe), "--out", str(csv), "--manifests-root", str(empty)],
            capture_output=True,
        )
        if csv.is_file() and csv.stat().st_size:
            text = csv.read_text()
            if comb.stat().st_size:
                comb.write_text(comb.read_text() + "\n".join(text.splitlines()[1:]) + "\n")
            else:
                comb.write_text(text)

    subprocess.run([str(py), str(bench / "aggregate_episode_metrics.py"), "--csv", str(comb), "--out-dir", str(out)])
    subprocess.run([str(py), str(bench / "generate_cumulative_markdown_report.py"), "--run-root", str(out)])
    print(f"REPORT: {out / 'reports/report_cumulative.md'}")
    return fail
