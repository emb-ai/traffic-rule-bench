"""Shared eval helpers: Sign SR, FV-fast, tag/ckpt resolution."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from lib.env import bench_dir, metrics_root, plan_t, resolve_python, setup_eval_thread_env, shepelev, signs_dir, trb_root

NFS2 = Path("/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova")
DEFAULT_MANIFEST = NFS2 / "traffic-rule-bench/pdd-bench/benchmark_output_speed/balanced/run_v61_a6/catalog_fv_test20.jsonl"
DEFAULT_SCENES = NFS2 / "traffic-rule-bench/pdd-bench/scenes_balanced"
DEFAULT_MANIFEST_DETOUR = NFS2 / "traffic-rule-bench/pdd-bench/benchmark_output/detour_v1/catalog_fv_test20.jsonl"
DEFAULT_SCENES_DETOUR = NFS2 / "sdc/pdd-bench/scenes"


TRAJECTORY_EXPERTS_2P5 = (
    lambda: shepelev()
    / "collected_trajectories/traj-priority-signs/traj_stop_2_5_train80/experts/all_runs_dedup.jsonl"
)
STOP_SCENES_2P5 = (
    lambda: trb_root() / "pdd-bench/scripts/per_sign_bench/stop_sign/scenes/2_5"
)


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
    trajectory: str | None = None
    save_gifs: bool = False
    save_predictions: bool = False
    force_rerun: bool = False


def signs_output_dir(tag: str) -> Path:
    return signs_dir() / "output" / tag


def signs_done(tag: str) -> bool:
    return (signs_output_dir(tag) / "_summary/summary.md").is_file()


def trajectory_done(tag: str, *, predictions_path: Path | None = None) -> bool:
    report = signs_output_dir(tag) / "2_5/eval_out/reports/report_cumulative.md"
    if not report.is_file():
        return False
    if predictions_path is not None and not predictions_path.is_file():
        return False
    return True


def normalize_trajectory_uid(trajectory: str) -> tuple[str, str]:
    """Return (scene_uid, plant2_route_name)."""
    t = trajectory.strip()
    if t.endswith("_v0_default"):
        return t[: -len("_default")], t
    if t.endswith("_v0"):
        return t, f"{t}_default"
    if t.endswith("_default"):
        return t.replace("_default", ""), t
    return t, f"{t}_default"


def build_trajectory_manifest(trajectory: str, work_dir: Path) -> Path:
    """One-row SUMO manifest for a train dump route (2.5 experts catalog)."""
    uid, route_name = normalize_trajectory_uid(trajectory)
    src = TRAJECTORY_EXPERTS_2P5()
    if not src.is_file():
        raise FileNotFoundError(f"experts catalog missing: {src}")
    row = None
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cand = json.loads(line)
        if cand.get("scene_uid") != uid:
            continue
        if cand.get("variant") != "default":
            continue
        if cand.get("policy") != "comprehensive_rule_expert":
            continue
        row = cand
        break
    if row is None:
        raise RuntimeError(f"no comprehensive_rule_expert/default row for scene_uid={uid!r}")

    # Merge eval-critical spawn/route fields from replay sidecar when present.
    sidecar_path = row.get("sidecar_path")
    if sidecar_path:
        sc = Path(str(sidecar_path))
        if sc.is_file():
            sidecar = json.loads(sc.read_text(encoding="utf-8"))
            source = sidecar.get("source_row") or {}
            for key in (
                "spawn_distance_before_end",
                "sign_distance_before_end",
                "destination_lane_id",
                "destination_edge_id",
                "road_id",
                "spawn_lane_num",
                "auxiliary_agent",
                "aux_distance_from_intersection",
                "aux_convoy_size",
                "aux_convoy_gap_m",
                "aux_lanes_occupied",
                "aux_road_id",
                "aux_spawn_lane_num",
                "aux_spawn_lane_index",
                "aux_destination_lane_id",
                "aux_destination_edge_id",
                "sign_spawn_distance",
                "junction_layout",
                "spawn_velocity_ms",
            ):
                if key in source and source[key] is not None:
                    row[key] = source[key]

    row = dict(row)
    row["pdd_code"] = row.get("sign_code") or "2.5"
    row["plant2_route"] = route_name
    row.setdefault("spawn_distance_before_end", 20.0)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"trajectory_{route_name}.jsonl"
    out.write_text(json.dumps(row, default=str) + "\n", encoding="utf-8")
    return out


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


def run_trajectory_eval(cfg: SignsEvalConfig) -> int:
    """Sign SR eval on a single train trajectory via stop_sign/eval_pipeline.py."""
    setup_eval_thread_env()
    py = resolve_python(str(cfg.python) if cfg.python else None)
    metrics = cfg.metrics_root or metrics_root()
    tag_dir = setup_metrics_tag(metrics, cfg.tag, cfg.ckpt)
    logf = tag_dir / "logs/eval_trajectory.log"
    (tag_dir / "trajectory.txt").write_text(f"{cfg.trajectory}\n")
    predictions_path = None
    if cfg.save_predictions and cfg.trajectory:
        predictions_path = tag_dir / f"{cfg.trajectory}_predictions.jsonl"
        if predictions_path.exists():
            predictions_path.unlink()

    if not cfg.force_rerun and trajectory_done(
        cfg.tag, predictions_path=predictions_path if cfg.save_predictions else None
    ):
        print(f"TRAJECTORY SKIP {cfg.tag}")
        return 0

    work_dir = tag_dir / "work"
    manifest = build_trajectory_manifest(cfg.trajectory, work_dir)
    out_root = signs_output_dir(cfg.tag) / "2_5"
    out_root.mkdir(parents=True, exist_ok=True)
    eval_out = out_root / "eval_out"
    if cfg.save_predictions:
        ep_cache = eval_out / "benchmark/full/policy_eval/plant2_default/episodes_plant2.jsonl"
        if ep_cache.is_file():
            ep_cache.unlink()
    scenes = STOP_SCENES_2P5()
    eval_py = trb_root() / "pdd-bench/scripts/per_sign_bench/stop_sign/eval_pipeline.py"
    if not eval_py.is_file():
        raise FileNotFoundError(eval_py)
    if not scenes.is_dir():
        raise FileNotFoundError(scenes)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    if cfg.save_predictions and predictions_path is not None:
        env["PLANT2_DEBUG_LOG_PATH"] = str(predictions_path)
    cmd = [
        str(py),
        "-u",
        str(eval_py),
        "--policies",
        "plant2",
        "--model-paths",
        f"plant2:{cfg.ckpt}",
        "--manifest",
        str(manifest),
        "--scenes-root",
        str(scenes),
        "--out-dir",
        str(eval_out),
        "--jobs",
        "1",
        "--scenes-per-job",
        "1",
    ]
    if cfg.save_gifs:
        cmd.append("--save-gifs")

    print(f"TRAJECTORY START tag={cfg.tag} route={cfg.trajectory} gpu={cfg.gpu}")
    print("$ " + " ".join(cmd))
    with logf.open("a") as log:
        rc = subprocess.run(
            cmd,
            cwd=str(eval_py.parent),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    gif_dirs = [
        eval_out / "benchmark/full/policy_eval/plant2_default/gifs",
        eval_out / "gifs",
    ]
    gifs: list[Path] = []
    for gif_dir in gif_dirs:
        if gif_dir.is_dir():
            gifs.extend(sorted(gif_dir.glob("*.gif")))
    report = eval_out / "reports/report_cumulative.md"
    print(f"TRAJECTORY rc={rc} report={report} gifs={len(gifs)}")
    for g in gifs:
        print(f"  GIF: {g}")
    if gifs and cfg.save_gifs:
        tag_dir = (cfg.metrics_root or metrics_root()) / cfg.tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        dst = tag_dir / f"{cfg.trajectory}.gif"
        if not dst.exists() or dst.stat().st_mtime < gifs[-1].stat().st_mtime:
            shutil.copy2(gifs[-1], dst)
            print(f"  copied -> {dst}")
    if predictions_path is not None and predictions_path.is_file():
        n_lines = sum(1 for _ in predictions_path.open())
        print(f"  predictions: {predictions_path} ({n_lines} steps)")
    return 0 if report.is_file() else rc or 1


def run_signs_eval(cfg: SignsEvalConfig) -> int:
    if cfg.trajectory:
        return run_trajectory_eval(cfg)

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
