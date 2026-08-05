#!/usr/bin/env python3
"""Unified PlanT2 L1 data dump (experts / fv / lane / rebuild-signs).

Subcommands:
  rebuild-signs   Parallel rebuild with PDD signs (default superset)
  experts         Priority/detour experts only (sequential)
  fv              FV nodeA speed-limit experts only
  lane            Lane 5.15.1 only (sharded)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from _env import bench_dir, pipeline_dir, resolve_python, shepelev
from _utils import WorkerPool, default_dump_max_workers, prepare_fv_experts

SM_MNT = Path("/mnt/virtual_ai0001053-01202_SR006-nfs2/smirnova")
ZINK_BENCH_DEFAULT = Path(
    "/mnt/virtual_ai0001053-01202_SR006-nfs2/zinkovich/zinkovich/"
    "traffic-rule-bench/pdd-bench/scripts/per_sign_bench"
)
DETOUR_SCENES_DEFAULT = SM_MNT / "sdc/pdd-bench/scenes"

EXPERT_SIGNS: dict[str, tuple[str, str]] = {
    "yield": ("traj-priority-signs/traj_yield_2_4_train80/experts", "yield_sign/scenes/2_4"),
    "stop": ("traj-priority-signs/traj_stop_2_5_train80/experts", "stop_sign/scenes/2_5"),
    "secondary": ("traj-priority-signs/traj_secondary_2_3_train80/experts", "secondary_sign/scenes/2_3"),
    "main": ("traj-priority-signs/traj_main_2_1_train80/experts", "main_sign/scenes/2_1"),
    "roundabout": ("traj-priority-signs/traj_roundabout_4_3_train80/experts", "roundabout_sign/scenes/4_3"),
    "detour": ("traffic-rule-bench-traj/experts_detour_train80", "__DETOUR__"),
}
FV_SIGNS = ["3.24", "4.6", "5.21", "5.31"]


@dataclass
class DumpJob:
    name: str
    experts: Path
    scenes: Path
    out_dir: Path
    n: int


def _replay_cmd(
    py: Path,
    script: Path,
    experts: Path,
    scenes: Path,
    out_dir: Path,
    *,
    backends: str,
    start: int,
    count: int | None,
    save_gifs: bool,
    gif_subdir: str,
) -> list[str]:
    cmd = [
        str(py),
        str(script),
        "--experts",
        str(experts),
        "--scenes-root",
        str(scenes),
        "--save-plant2-dir",
        str(out_dir),
        "--backends",
        backends,
        "--ego-mode",
        "recorded",
        "--npc-mode",
        "recorded",
        "--start",
        str(start),
    ]
    if count is not None:
        cmd.extend(["--count", str(count)])
    if save_gifs:
        cmd.extend(["--save-gifs", "--gif-dir", str(out_dir / "gifs" / gif_subdir)])
    return cmd


def _run_experts(args: argparse.Namespace) -> int:
    py = resolve_python(args.python_exe)
    script = bench_dir() / "expert_replay_inenv.py"
    ct = pipeline_dir()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    experts_file = f"experts_scene_uid_{args.experts_rank}.jsonl"
    ok = fail = 0

    for sign in args.signs:
        rel_exp, rel_sc = EXPERT_SIGNS[sign]
        experts = ct / rel_exp / experts_file
        scenes = args.detour_scenes if rel_sc == "__DETOUR__" else args.zink_bench / rel_sc
        if not experts.is_file() or not scenes.is_dir():
            print(f"[FAIL] {sign}: missing inputs")
            fail += 1
            continue
        cmd = _replay_cmd(
            py, script, experts, scenes, args.out_dir,
            backends=args.backends, start=args.start, count=args.count,
            save_gifs=args.save_gifs, gif_subdir=sign,
        )
        print(f"[run] {sign}: {' '.join(cmd)}")
        if args.dry_run:
            ok += 1
            continue
        logf = log_dir / f"{sign}_{ts}.log"
        with logf.open("w") as f:
            rc = subprocess.run(cmd, cwd=str(bench_dir()), stdout=f, stderr=subprocess.STDOUT).returncode
        fail += rc != 0
        ok += rc == 0
        print(f"[{'ok' if rc == 0 else 'FAIL'}] {sign} rc={rc}")
    print(f"=== experts done ok={ok} fail={fail} ===")
    return fail


def _run_fv(args: argparse.Namespace) -> int:
    py = resolve_python(args.python_exe)
    script = bench_dir() / "expert_replay_inenv.py"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    experts_dir = args.out_dir / "experts"
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    prepare_fv_experts(
        src=args.fv_experts_src,
        out_dir=experts_dir,
        signs=args.fv_signs,
        node_filter=args.node_filter,
    )
    max_jobs = args.max_jobs if args.max_jobs > 0 else len(args.fv_signs)
    pool = WorkerPool(max_jobs)
    rc_map: dict[str, int] = {}

    for sign in args.fv_signs:
        slug = sign.replace(".", "_")
        experts = experts_dir / f"experts_{slug}_top1.jsonl"
        if not experts.is_file() or experts.stat().st_size == 0:
            print(f"[FAIL] {sign}: empty/missing experts")
            rc_map[sign] = 1
            continue
        cmd = _replay_cmd(
            py, script, experts, args.scenes_root, args.out_dir,
            backends=args.backends, start=args.start, count=args.count,
            save_gifs=args.save_gifs, gif_subdir=slug,
        )
        logf = log_dir / f"{slug}_{ts}.log"
        if args.dry_run:
            print(f"[dry-run] {sign}")
            rc_map[sign] = 0
            continue
        pool.spawn(sign, cmd, cwd=bench_dir(), log_path=logf)
    if not args.dry_run:
        rc_map.update(pool.wait_all())
    fail = sum(1 for r in rc_map.values() if r != 0)
    print(f"=== fv done fail={fail} ===")
    return fail


def _run_lane(args: argparse.Namespace) -> int:
    py = resolve_python(args.python_exe)
    script = bench_dir() / "expert_replay_inenv.py"
    experts = (
        pipeline_dir()
        / "traj-priority-signs/traj_lane_5_15_train80/experts"
        / f"experts_scene_uid_{args.experts_rank}.jsonl"
    )
    scenes = args.zink_bench / "lane_direction_signs/scenes/5_15_1"
    for p, label in ((script, "script"), (experts, "experts"), (scenes, "scenes")):
        if not p.exists():
            raise SystemExit(f"ERROR: missing {label}: {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_experts = sum(1 for _ in experts.open())
    count = min(args.count, max(0, n_experts - args.start))
    if count <= 0:
        raise SystemExit("ERROR: nothing to dump")
    n_shards = min(args.n_shards, count)
    pool = WorkerPool(n_shards)
    rc: dict[int, int] = {}

    for shard in range(n_shards):
        st = args.start + shard * count // n_shards
        en = args.start + (shard + 1) * count // n_shards
        cnt = en - st
        if cnt <= 0:
            continue
        cmd = _replay_cmd(
            py, script, experts, scenes, args.out_dir,
            backends=args.backends, start=st, count=cnt,
            save_gifs=args.save_gifs, gif_subdir=f"lane_shard{shard}",
        )
        logf = log_dir / f"lane_shard{shard}_{ts}.log"
        if args.dry_run:
            print(f"[dry-run] shard={shard} count={cnt}")
            rc[shard] = 0
            continue
        pool.spawn(str(shard), cmd, cwd=bench_dir(), log_path=logf)
    if not args.dry_run:
        for k, v in pool.wait_all().items():
            rc[int(k)] = v
    fail = sum(1 for v in rc.values() if v != 0)
    print(f"=== lane done fail={fail} ===")
    return fail


def _build_rebuild_jobs(args: argparse.Namespace) -> list[DumpJob]:
    ct = pipeline_dir()
    experts_file = f"experts_scene_uid_{args.experts_rank}.jsonl"
    jobs: list[DumpJob] = []

    def add(name: str, experts: Path, scenes: Path, out: Path, count_cap: int | None = None) -> None:
        if not experts.is_file() or not scenes.is_dir():
            print(f"[skip] {name}: missing inputs")
            return
        n = sum(1 for _ in experts.open())
        if count_cap is not None:
            n = min(n, count_cap)
        if n <= 0:
            print(f"[skip] {name}: n=0")
            return
        jobs.append(DumpJob(name, experts, scenes, out, n))

    for sign, (rel_exp, rel_sc) in EXPERT_SIGNS.items():
        scenes = args.detour_scenes if rel_sc == "__DETOUR__" else args.zink_bench / rel_sc
        add(f"exp:{sign}", ct / rel_exp / experts_file, scenes, args.out_exp)

    prepare_fv_experts(
        src=args.fv_experts_src,
        out_dir=args.out_fv / "experts",
        signs=FV_SIGNS,
        node_filter="nodeA",
    )
    for sign in FV_SIGNS:
        slug = sign.replace(".", "_")
        add(
            f"fv:{sign}",
            args.out_fv / "experts" / f"experts_{slug}_top1.jsonl",
            args.fv_scenes,
            args.out_fv,
        )

    add(
        "lane:5.15.1",
        ct / "traj-priority-signs/traj_lane_5_15_train80/experts" / experts_file,
        args.zink_bench / "lane_direction_signs/scenes/5_15_1",
        args.out_lane,
    )

    if args.jobs:
        want = set(args.jobs)
        jobs = [j for j in jobs if j.name in want]
    return jobs


def _run_rebuild_signs(args: argparse.Namespace) -> int:
    py = resolve_python(args.python_exe)
    script = bench_dir() / "expert_replay_inenv.py"
    nproc = os.cpu_count() or 8
    max_workers = args.max_workers or default_dump_max_workers(nproc)

    for d in (args.out_exp, args.out_fv, args.out_lane):
        (d / "logs").mkdir(parents=True, exist_ok=True)
    log_root = shepelev() / "collected_trajectories/logs_dump_signs"
    log_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = log_root / f"rebuild_signs_{ts}.log"

    jobs = _build_rebuild_jobs(args)
    tasks: list[tuple[str, int, int, Path, Path, Path, str]] = []
    for job in jobs:
        n_shards = max(1, min((job.n + args.target_per_shard - 1) // args.target_per_shard, job.n))
        n_shards = min(n_shards, max_workers * 2)
        slug_base = job.name.replace(":", "_").replace(".", "_")
        print(f"  job {job.name} n={job.n} -> shards={n_shards}")
        for s in range(n_shards):
            st = s * job.n // n_shards
            en = (s + 1) * job.n // n_shards
            cnt = en - st
            if cnt > 0:
                tasks.append((f"{slug_base}_s{s}", st, cnt, job.experts, job.scenes, job.out_dir, job.name))

    print(f"PLAN jobs={len(jobs)} shards={len(tasks)} max_workers={max_workers}")
    summary.write_text(f"PLAN shards={len(tasks)} max_workers={max_workers}\n")

    if args.dry_run:
        for t in tasks[:15]:
            print(t[:3], t[6])
        return 0

    pool = WorkerPool(max_workers)
    rc_files: dict[str, Path] = {}

    for slug, start, count, experts, scenes, out, name in tasks:
        out.mkdir(parents=True, exist_ok=True)
        logf = out / "logs" / f"{slug}_{ts}.log"
        rc_file = out / "logs" / f"{slug}_{ts}.rc"
        cmd = _replay_cmd(
            py, script, experts, scenes, out,
            backends=args.backends, start=start, count=count,
            save_gifs=args.save_gifs, gif_subdir=slug,
        )
        print(f"[spawn] {name} {slug} start={start} count={count}")
        pool.spawn(slug, cmd, cwd=bench_dir(), log_path=logf)
        rc_files[slug] = rc_file

    results = pool.wait_all()
    ok = fail = 0
    for slug, code in results.items():
        rc_files[slug].write_text(f"{code}\n")
        if code == 0:
            ok += 1
        else:
            fail += 1
    print(f"=== rebuild-signs done ok={ok} fail={fail} ===")
    return fail


def _common_parser(sub: argparse._SubParsersAction) -> None:
    for name, help_ in (
        ("rebuild-signs", "Parallel full rebuild → *_signs trees"),
        ("experts", "Priority/detour experts (sequential)"),
        ("fv", "FV nodeA experts"),
        ("lane", "Lane 5.15.1 (sharded)"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--python", dest="python_exe", default=None)
        p.add_argument("--backends", default="sumo")
        p.add_argument("--save-gifs", action="store_true")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--experts-rank", default="top1", choices=("top1", "top2"))
        p.add_argument(
            "--zink-bench", type=Path, default=ZINK_BENCH_DEFAULT,
        )
        p.add_argument("--detour-scenes", type=Path, default=DETOUR_SCENES_DEFAULT)


def main(argv: list[str] | None = None) -> int:
    sh = shepelev()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    _common_parser(sub)

    rs = sub.choices["rebuild-signs"]
    rs.add_argument("--out-exp", type=Path, default=sh / "plant2_l1_from_experts_signs")
    rs.add_argument("--out-fv", type=Path, default=sh / "plant2_l1_traj_fv_nodeA_signs")
    rs.add_argument("--out-lane", type=Path, default=sh / "plant2_l1_lane_signs")
    rs.add_argument("--max-workers", type=int, default=None)
    rs.add_argument("--target-per-shard", type=int, default=120)
    rs.add_argument("--jobs", nargs="+", default=None, help="e.g. exp:stop fv:3.24")
    rs.add_argument(
        "--fv-experts-src",
        type=Path,
        default=SM_MNT / "experts_fv_train80/experts_scene_uid_top1.jsonl",
    )
    rs.add_argument(
        "--fv-scenes",
        type=Path,
        default=SM_MNT / "traffic-rule-bench/pdd-bench/scenes_balanced",
    )

    ex = sub.choices["experts"]
    ex.add_argument("--out-dir", type=Path, default=sh / "plant2_l1_from_experts")
    ex.add_argument("--signs", nargs="+", default=list(EXPERT_SIGNS.keys()), choices=list(EXPERT_SIGNS.keys()))
    ex.add_argument("--count", type=int, default=None)
    ex.add_argument("--start", type=int, default=0)

    fv = sub.choices["fv"]
    fv.add_argument("--out-dir", type=Path, default=sh / "plant2_l1_traj_fv_nodeA")
    fv.add_argument("--fv-experts-src", type=Path, default=SM_MNT / "experts_fv_train80/experts_scene_uid_top1.jsonl")
    fv.add_argument("--scenes-root", type=Path, default=SM_MNT / "traffic-rule-bench/pdd-bench/scenes_balanced")
    fv.add_argument("--node-filter", default="nodeA")
    fv.add_argument("--fv-signs", nargs="+", default=FV_SIGNS)
    fv.add_argument("--count", type=int, default=None)
    fv.add_argument("--start", type=int, default=0)
    fv.add_argument("--max-jobs", type=int, default=0)

    ln = sub.choices["lane"]
    ln.add_argument("--out-dir", type=Path, default=sh / "plant2_l1_lane300")
    ln.add_argument("--count", type=int, default=300)
    ln.add_argument("--start", type=int, default=0)
    ln.add_argument("--n-shards", type=int, default=8)

    args = parser.parse_args(argv)
    handlers = {
        "rebuild-signs": _run_rebuild_signs,
        "experts": _run_experts,
        "fv": _run_fv,
        "lane": _run_lane,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
