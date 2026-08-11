#!/usr/bin/env python3
"""Unified PlanT2 L1 data dump (experts / fv / lane / rebuild-signs).

Subcommands:
  rebuild-signs   Parallel rebuild with PDD signs (default superset)
  experts         Priority/detour experts only (sequential)
  fv              FV nodeA speed-limit experts only
  lane            Lane 5.15.1 only (sharded)
  one             Dump exactly one random expert trajectory
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import contextlib
import importlib.util
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from lib.env import shepelev, trb_root
from lib.utils import default_dump_max_workers, prepare_fv_experts

EXPERT_REPLAY_FOR_PLANT2 = (
    trb_root()
    / "pdd-bench"
    / "scripts"
    / "per_sign_bench"
    / "expert_replay_for_plant2.py"
)
_expert_replay_mod: object | None = None

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


def _traj_root() -> Path:
    return shepelev() / "collected_trajectories"


def _bench_local() -> Path:
    return trb_root() / "pdd-bench/scripts/per_sign_bench"


def _experts_path(rel_exp: str, experts_rank: str) -> Path:
    return _traj_root() / rel_exp / f"experts_scene_uid_{experts_rank}.jsonl"


def _scenes_path(rel_sc: str, args: argparse.Namespace) -> Path:
    if rel_sc == "__DETOUR__":
        return args.detour_scenes
    local = _bench_local() / rel_sc
    if local.is_dir():
        return local
    return args.zink_bench / rel_sc


def _load_expert_replay():
    global _expert_replay_mod
    if _expert_replay_mod is not None:
        return _expert_replay_mod
    if not EXPERT_REPLAY_FOR_PLANT2.is_file():
        raise SystemExit(f"ERROR: missing replay module: {EXPERT_REPLAY_FOR_PLANT2}")
    spec = importlib.util.spec_from_file_location(
        "expert_replay_for_plant2",
        EXPERT_REPLAY_FOR_PLANT2,
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"ERROR: cannot import replay module: {EXPERT_REPLAY_FOR_PLANT2}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _expert_replay_mod = mod
    return mod


def _load_run_batch() -> Callable[..., dict]:
    return _load_expert_replay().run_batch


def _sign_experts_scenes(sign: str, args: argparse.Namespace) -> tuple[Path, Path]:
    if sign not in EXPERT_SIGNS:
        raise SystemExit(f"ERROR: unknown --sign {sign!r}; choose from {list(EXPERT_SIGNS)}")
    rel_exp, rel_sc = EXPERT_SIGNS[sign]
    return _experts_path(rel_exp, args.experts_rank), _scenes_path(rel_sc, args)


def _backend_for_row(row: dict) -> str:
    if row.get("backend"):
        return str(row["backend"])
    sidecar_path = row.get("sidecar_path")
    if not sidecar_path:
        return "sumo"
    sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    return str(sidecar.get("backend") or "sumo")


def _expert_row_candidates(experts_path: Path, backends: str) -> list[tuple[int, dict]]:
    allowed = {b.strip() for b in backends.split(",") if b.strip()}
    out: list[tuple[int, dict]] = []
    with experts_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if _backend_for_row(row) in allowed:
                out.append((i, row))
    return out


def _pick_random_expert(
    experts_path: Path,
    *,
    backends: str,
    seed: int | None,
) -> tuple[int, dict]:
    candidates = _expert_row_candidates(experts_path, backends)
    if not candidates:
        raise SystemExit(f"ERROR: no expert rows matching backends={backends!r} in {experts_path}")
    rng = random.Random(seed)
    return rng.choice(candidates)


def _replay_desc(
    experts: Path,
    scenes: Path,
    out_dir: Path,
    *,
    backends: str,
    start: int,
    count: int | None,
) -> str:
    parts = [
        f"experts={experts}",
        f"scenes={scenes}",
        f"out={out_dir}",
        f"backends={backends}",
        f"start={start}",
    ]
    if count is not None:
        parts.append(f"count={count}")
    return " ".join(parts)


def _run_replay_batch(
    experts: Path,
    scenes: Path,
    out_dir: Path,
    *,
    backends: str,
    start: int,
    count: int | None,
    save_gifs: bool = False,
    log_path: Path | None = None,
) -> int:
    if save_gifs:
        print(f"[warn] --save-gifs ignored ({EXPERT_REPLAY_FOR_PLANT2.name} has no gif export)")
    run_batch = _load_run_batch()
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as logf:
                with contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
                    summary = run_batch(
                        experts.resolve(),
                        scenes.resolve(),
                        out_dir.resolve(),
                        count=count,
                        start=start,
                        backends=backends,
                    )
                    print(json.dumps(summary, indent=2))
        else:
            summary = run_batch(
                experts.resolve(),
                scenes.resolve(),
                out_dir.resolve(),
                count=count,
                start=start,
                backends=backends,
            )
            print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        msg = f"ERROR: {exc}\n"
        print(msg, end="" if log_path is None else "", file=sys.stderr)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(msg, encoding="utf-8")
        return 1


def _replay_batch_job(
    experts: str,
    scenes_root: str,
    save_plant2_dir: str,
    backends: str,
    start: int,
    count: int | None,
    log_path: str | None,
    save_gifs: bool,
) -> int:
    return _run_replay_batch(
        Path(experts),
        Path(scenes_root),
        Path(save_plant2_dir),
        backends=backends,
        start=start,
        count=count,
        save_gifs=save_gifs,
        log_path=Path(log_path) if log_path else None,
    )


@dataclass
class DumpJob:
    name: str
    experts: Path
    scenes: Path
    out_dir: Path
    n: int


def _run_one(args: argparse.Namespace) -> int:
    mod = _load_expert_replay()
    from bench.plant2_frames import ensure_slurm_dummy, plant2_route_dir

    if args.sign:
        experts, scenes = _sign_experts_scenes(args.sign, args)
    elif args.experts is not None and args.scenes_root is not None:
        experts, scenes = args.experts.resolve(), args.scenes_root.resolve()
    else:
        raise SystemExit("ERROR: provide --sign stop (etc.) or both --experts and --scenes-root")

    for label, path in (("experts", experts), ("scenes", scenes)):
        if not path.exists():
            raise SystemExit(f"ERROR: missing {label}: {path}")

    line_idx, row = _pick_random_expert(experts, backends=args.backends, seed=args.seed)
    pkl_path, sidecar_path, scene_uid, variant, backend = mod.resolve_expert_paths(row)
    route_dir = plant2_route_dir(args.out_dir, scene_uid, variant)

    plan = {
        "experts": str(experts),
        "experts_line": line_idx,
        "scenes_root": str(scenes),
        "backend": backend,
        "scene_uid": scene_uid,
        "variant": variant,
        "pkl_path": str(pkl_path),
        "sidecar_path": str(sidecar_path),
        "out_dir": str(args.out_dir),
        "route_dir": str(route_dir),
    }
    print(json.dumps({"plan": plan}, indent=2))

    if args.dry_run:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ensure_slurm_dummy(args.out_dir)

    if route_dir.exists() and (route_dir / "results.json.gz").is_file() and not args.force:
        print(json.dumps({
            "status": "skipped",
            "reason": "route already exists (use --force to re-dump)",
            **plan,
        }, indent=2))
        return 0

    try:
        result = mod.dump_plant2(
            pkl_path,
            sidecar_path,
            scenes_root=scenes,
            save_plant2_dir=args.out_dir,
            max_steps=args.max_steps,
        )
    except Exception as exc:
        print(json.dumps({"status": "fail", "error": str(exc), **plan}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps({"status": "ok", **plan, "result": result}, indent=2))
    return 0


def _run_experts(args: argparse.Namespace) -> int:
    _load_run_batch()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    experts_file = f"experts_scene_uid_{args.experts_rank}.jsonl"
    ok = fail = 0

    for sign in args.signs:
        rel_exp, rel_sc = EXPERT_SIGNS[sign]
        experts = _experts_path(rel_exp, args.experts_rank)
        scenes = _scenes_path(rel_sc, args)
        if not experts.is_file() or not scenes.is_dir():
            print(f"[FAIL] {sign}: missing inputs")
            fail += 1
            continue
        start = args.start
        count = args.count
        if args.random_one:
            candidates = _expert_row_candidates(experts, args.backends)
            if not candidates:
                print(f"[FAIL] {sign}: no expert rows for backends={args.backends!r}")
                fail += 1
                continue
            rng = random.Random(args.seed)
            line_idx, _ = rng.choice(candidates)
            start = line_idx
            count = 1
            print(f"[random-one] {sign}: experts line {line_idx}")
        desc = _replay_desc(
            experts, scenes, args.out_dir,
            backends=args.backends, start=start, count=count,
        )
        print(f"[run] {sign}: {desc}")
        if args.dry_run:
            ok += 1
            continue
        logf = log_dir / f"{sign}_{ts}.log"
        rc = _run_replay_batch(
            experts, scenes, args.out_dir,
            backends=args.backends, start=start, count=count,
            save_gifs=args.save_gifs, log_path=logf,
        )
        fail += rc != 0
        ok += rc == 0
        print(f"[{'ok' if rc == 0 else 'FAIL'}] {sign} rc={rc}")
    print(f"=== experts done ok={ok} fail={fail} ===")
    return fail


def _run_fv(args: argparse.Namespace) -> int:
    _load_run_batch()
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
    rc_map: dict[str, int] = {}
    pending: list[tuple[str, dict[str, object]]] = []

    for sign in args.fv_signs:
        slug = sign.replace(".", "_")
        experts = experts_dir / f"experts_{slug}_top1.jsonl"
        if not experts.is_file() or experts.stat().st_size == 0:
            print(f"[FAIL] {sign}: empty/missing experts")
            rc_map[sign] = 1
            continue
        logf = log_dir / f"{slug}_{ts}.log"
        if args.dry_run:
            print(f"[dry-run] {sign}")
            rc_map[sign] = 0
            continue
        pending.append((
            sign,
            {
                "experts": str(experts),
                "scenes_root": str(args.scenes_root),
                "save_plant2_dir": str(args.out_dir),
                "backends": args.backends,
                "start": args.start,
                "count": args.count,
                "log_path": str(logf),
                "save_gifs": args.save_gifs,
            },
        ))

    if pending:
        with ProcessPoolExecutor(max_workers=max_jobs) as pool:
            futures = {pool.submit(_replay_batch_job, **job): key for key, job in pending}
            for fut in as_completed(futures):
                rc_map[futures[fut]] = fut.result()
    fail = sum(1 for r in rc_map.values() if r != 0)
    print(f"=== fv done fail={fail} ===")
    return fail


def _run_lane(args: argparse.Namespace) -> int:
    _load_run_batch()
    experts = _experts_path("traj-priority-signs/traj_lane_5_15_train80/experts", args.experts_rank)
    scenes = _scenes_path("lane_direction_signs/scenes/5_15_1", args)
    for p, label in ((EXPERT_REPLAY_FOR_PLANT2, "replay module"), (experts, "experts"), (scenes, "scenes")):
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
    rc: dict[int, int] = {}
    pending: list[tuple[int, dict[str, object]]] = []

    for shard in range(n_shards):
        st = args.start + shard * count // n_shards
        en = args.start + (shard + 1) * count // n_shards
        cnt = en - st
        if cnt <= 0:
            continue
        logf = log_dir / f"lane_shard{shard}_{ts}.log"
        if args.dry_run:
            print(f"[dry-run] shard={shard} count={cnt}")
            rc[shard] = 0
            continue
        pending.append((
            shard,
            {
                "experts": str(experts),
                "scenes_root": str(scenes),
                "save_plant2_dir": str(args.out_dir),
                "backends": args.backends,
                "start": st,
                "count": cnt,
                "log_path": str(logf),
                "save_gifs": args.save_gifs,
            },
        ))

    if pending:
        with ProcessPoolExecutor(max_workers=n_shards) as pool:
            futures = {pool.submit(_replay_batch_job, **job): shard for shard, job in pending}
            for fut in as_completed(futures):
                rc[futures[fut]] = fut.result()
    fail = sum(1 for v in rc.values() if v != 0)
    print(f"=== lane done fail={fail} ===")
    return fail


def _build_rebuild_jobs(args: argparse.Namespace) -> list[DumpJob]:
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
        add(
            f"exp:{sign}",
            _experts_path(rel_exp, args.experts_rank),
            _scenes_path(rel_sc, args),
            args.out_exp,
        )

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
        _experts_path("traj-priority-signs/traj_lane_5_15_train80/experts", args.experts_rank),
        _scenes_path("lane_direction_signs/scenes/5_15_1", args),
        args.out_lane,
    )

    if args.jobs:
        want = set(args.jobs)
        jobs = [j for j in jobs if j.name in want]
    return jobs


def _run_rebuild_signs(args: argparse.Namespace) -> int:
    _load_run_batch()
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

    rc_files: dict[str, Path] = {}
    pending: list[tuple[str, dict[str, object], Path]] = []

    for slug, start, count, experts, scenes, out, name in tasks:
        out.mkdir(parents=True, exist_ok=True)
        logf = out / "logs" / f"{slug}_{ts}.log"
        rc_file = out / "logs" / f"{slug}_{ts}.rc"
        print(f"[spawn] {name} {slug} start={start} count={count}")
        pending.append((
            slug,
            {
                "experts": str(experts),
                "scenes_root": str(scenes),
                "save_plant2_dir": str(out),
                "backends": args.backends,
                "start": start,
                "count": count,
                "log_path": str(logf),
                "save_gifs": args.save_gifs,
            },
            rc_file,
        ))
        rc_files[slug] = rc_file

    results: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_replay_batch_job, **job): slug for slug, job, _ in pending}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

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
        ("one", "Dump one random expert trajectory"),
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
    ex.add_argument(
        "--random-one",
        action="store_true",
        help="pick one random expert row per --sign (overrides --start/--count)",
    )
    ex.add_argument("--seed", type=int, default=None, help="RNG seed for --random-one")

    one = sub.choices["one"]
    one.add_argument("--out-dir", type=Path, default=sh / "plant2_l1_one_random")
    one.add_argument(
        "--sign",
        default="stop",
        choices=list(EXPERT_SIGNS.keys()),
        help="expert pool to sample from (default: stop / 2.5)",
    )
    one.add_argument("--experts", type=Path, default=None, help="override experts jsonl")
    one.add_argument("--scenes-root", type=Path, default=None, help="override scenes dir")
    one.add_argument("--seed", type=int, default=None, help="RNG seed")
    one.add_argument("--max-steps", type=int, default=1500)
    one.add_argument(
        "--force",
        action="store_true",
        help="re-dump even if route_dir/results.json.gz already exists",
    )

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
        "one": _run_one,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
