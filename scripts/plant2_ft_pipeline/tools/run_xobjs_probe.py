#!/usr/bin/env python
"""Instrumented plant2 rollouts that record whether the 2.5 stop-sign token
reaches ``x_objs`` at eval time.

Runs the belyaev ``priority_bench/run_benchmark.py`` once per selected episode
with ``PLANT2_XOBJS_LOG_PATH`` pointed at a per-episode JSONL trace. The trace
is produced by the env-gated ``_maybe_log_xobjs`` hook in
``pdd-bench/agents/plant2_in_metadrive/plant2_adapter.py`` and holds, per frame:
the object token types in ``x_objs``, the 2.5 token's ego-frame pose, the
ground-truth sign poses read straight from ``traffic_sign_manager``, ego speed,
the model's desired speed and the commanded action.

Everything is written under --work (a scratch dir); the shared eval outputs and
the pipeline driver script are never touched.

Usage:
    python run_xobjs_probe.py --work <scratch> --group violators
    python run_xobjs_probe.py --work <scratch> --group compliant
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TRB_ROOT = Path("/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench")
PY = Path("/home/jovyan/.mlspace/envs/zinkovich-plant2/bin/python")
EVAL_PRIORITY = TRB_ROOT / "pdd-bench" / "scripts" / "per_sign_bench" / "priority_bench"
CKPT = (TRB_ROOT / "plant2" / "PlanT" / "checkpoints_ft" / "stop_signfix_lr3e4_ep20"
        / "last_ft_stop_signfix_lr3e4_ep20_1.ckpt")
FULL_EVAL = TRB_ROOT / "plant2_stop_pipeline_signfix" / "eval_test" / "full"
MANIFEST_ALL = FULL_EVAL / "input_manifest.jsonl"
METRICS_CSV = FULL_EVAL / "metrics_per_episode.csv"
# Maps already vendored for the 8 violating junctions.
SCENES_VENDORED = (TRB_ROOT / "plant2_stop_pipeline_signfix" / "eval_test"
                   / "gifs_violations" / "scenes")
# Read-only source for the junctions that were not part of the violation set.
SCENES_UPSTREAM = Path("/home/jovyan/shares/SR006.nfs2/zinkovich/zinkovich"
                       "/traffic-rule-bench/pdd-bench/moscow_scenes/scenes")


def scene_uid(row: dict) -> str:
    return (f"{row.get('scene_id')}_lane{int(row.get('spawn_lane_num', 0) or 0)}"
            f"_seed{int(row.get('seed'))}_v{int(row.get('var_idx', 0) or 0)}")


def require(path: Path, what: str) -> Path:
    """Re-verify a pinned path right before use (the tree is being restructured)."""
    if not path.exists():
        raise FileNotFoundError(f"{what} vanished: {path}")
    return path


def load_groups() -> tuple[dict[str, dict], list[str], list[str]]:
    require(MANIFEST_ALL, "input manifest")
    require(METRICS_CSV, "metrics csv")
    rows = {}
    for line in MANIFEST_ALL.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[scene_uid(r)] = r
    violators, compliant = [], []
    with METRICS_CSV.open() as f:
        for m in csv.DictReader(f):
            uid = m["scene_uid"]
            (violators if m["sign_compliant_high"] == "False" else compliant).append(uid)
    return rows, violators, compliant


def pick_compliant(compliant: list[str], violators: list[str], limit: int) -> list[str]:
    """Compliant episodes from mixed junctions first (same map as a violator, so
    the comparison isolates behaviour from geometry), then one per
    always-compliant junction."""
    def junction(uid: str) -> str:
        return uid.split("_lane")[0]

    viol_junctions = {junction(u) for u in violators}
    mixed = [u for u in compliant if junction(u) in viol_junctions]
    others, seen = [], set()
    for u in compliant:
        j = junction(u)
        if j in viol_junctions or j in seen:
            continue
        seen.add(j)
        others.append(u)
    return (mixed + others)[:limit]


def vendor_scene(scene_id: str, dest_root: Path) -> None:
    dest = dest_root / scene_id
    if (dest / "map.net.xml").is_file():
        return
    for src in (SCENES_VENDORED / scene_id,
                SCENES_UPSTREAM / "T" / scene_id,
                SCENES_UPSTREAM / "X" / scene_id):
        if (src / "map.net.xml").is_file():
            dest.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, dest / f.name)
            return
    raise FileNotFoundError(f"no map.net.xml for {scene_id}")


def run_episode(uid: str, row: dict, work: Path, scenes: Path, timeout_s: int) -> dict:
    man = work / "manifests" / f"{uid}.jsonl"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps(row) + "\n")

    log = work / "logs" / f"{uid}.xobjs.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        log.unlink()
    out_dir = work / "runs" / uid
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = work / "logs" / f"{uid}.stdout.log"

    env = dict(os.environ)
    env.update({
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "SDL_AUDIODRIVER": "dummy",
        "PER_SIGN_COMPLIANT_NPC": "1",
        "WANDB_MODE": "offline",
        "PLANT2_DUMP_SIGN_CLASSES": "2.5",
        "PYTHONPATH": ":".join([
            str(TRB_ROOT / "metadrive"),
            str(TRB_ROOT / "pdd-bench"),
            str(TRB_ROOT / "pdd-bench" / "scripts" / "per_sign_bench"),
        ]),
        "PLANT2_XOBJS_LOG_PATH": str(log),
    })

    cmd = [
        str(require(PY, "python")), "-u", "run_benchmark.py",
        "--policy", "plant2",
        "--model-path", str(require(CKPT, "checkpoint")),
        "--run-name", "xobjs_probe",
        "--manifest", str(man),
        "--scenes-root", str(scenes),
        "--output-dir", str(out_dir),
        "--plant2-action-mode", "pid",
        "--emit-replay-sidecar",
        "--replay-root", str(out_dir / "replays"),
        "--force-rerun",
    ]
    t0 = time.time()
    with stdout_path.open("w") as f:
        try:
            rc = subprocess.call(cmd, cwd=str(EVAL_PRIORITY), env=env,
                                 stdout=f, stderr=subprocess.STDOUT,
                                 timeout=timeout_s)
        except subprocess.TimeoutExpired:
            rc = -9
    frames = sum(1 for _ in log.open()) if log.exists() else 0
    return {"scene_uid": uid, "rc": rc, "frames": frames,
            "elapsed_s": round(time.time() - t0, 1),
            "log": str(log), "out_dir": str(out_dir)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--group", choices=["violators", "compliant", "both"],
                    default="violators")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--compliant-limit", type=int, default=9)
    ap.add_argument("--timeout-s", type=int, default=1800)
    args = ap.parse_args()

    work = Path(args.work).resolve()
    (work / "logs").mkdir(parents=True, exist_ok=True)
    scenes = work / "scenes"
    scenes.mkdir(parents=True, exist_ok=True)

    rows, violators, compliant = load_groups()
    selected: list[str] = []
    if args.group in ("violators", "both"):
        selected += violators
    if args.group in ("compliant", "both"):
        selected += pick_compliant(compliant, violators, args.compliant_limit)
    if args.limit:
        selected = selected[: args.limit]

    print(f"[probe] {len(selected)} episodes: {args.group}", flush=True)
    for uid in selected:
        vendor_scene(rows[uid]["scene_id"], scenes)

    results = []
    summary_path = work / f"probe_results_{args.group}.jsonl"
    with summary_path.open("w") as sf:
        for i, uid in enumerate(selected, 1):
            print(f"[probe {i}/{len(selected)}] {uid}", flush=True)
            r = run_episode(uid, rows[uid], work, scenes, args.timeout_s)
            r["group"] = "violator" if uid in violators else "compliant"
            results.append(r)
            sf.write(json.dumps(r) + "\n")
            sf.flush()
            print(f"[probe {i}/{len(selected)}] rc={r['rc']} frames={r['frames']} "
                  f"{r['elapsed_s']}s", flush=True)

    bad = [r for r in results if r["rc"] != 0 or r["frames"] == 0]
    print(f"[probe] done: {len(results)} episodes, {len(bad)} problematic")
    print(f"[probe] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
