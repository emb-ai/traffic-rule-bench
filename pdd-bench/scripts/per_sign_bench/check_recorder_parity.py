#!/usr/bin/env python3
"""Prove recorder<->eval parity for one policy on one manifest row.

Runs the SAME (scene, seed, policy) through run_benchmark.py (the corrected
eval, with --emit-replay-sidecar) and through expert_replay.py (the trajectory
recorder), then diffs the two replay.json sidecars:

  * every shared `metrics` field must be identical,
  * the per-step ego actions (expert_actions) must be identical,
  * identity fields (scene_uid, sign_slug, policy, variant) must match.

Exit code 0 = parity holds; 1 = mismatch (printed field by field).

Usage (CPU policy — no checkpoint):
  python3 check_recorder_parity.py --manifest <m.jsonl> --scenes-root <scenes> \
      --policy comprehensive_rule_expert

NN policies (pin the GPU so both runs share one device):
  CUDA_VISIBLE_DEVICES=0 python3 check_recorder_parity.py --manifest <m.jsonl> \
      --scenes-root <scenes> --policy carl_rule --model-path $CARL_CKPT
  CUDA_VISIBLE_DEVICES=0 python3 check_recorder_parity.py --manifest <m.jsonl> \
      --scenes-root <scenes> --policy plant2_rule --model-path $PLANT2_CKPT \
      --plant2-action-mode pid
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _pick_row(manifest: Path, scene_id: str | None, row_index: int,
              scenes_root: Path) -> dict:
    """First valid row (or by scene_id / index) whose net file resolves."""
    idx = -1
    for line in open(manifest, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("valid") is False:
            continue
        net = row.get("net_path")
        if net:
            p = Path(net)
            if not (p if p.is_absolute() else scenes_root / p).exists():
                continue
        if scene_id is not None:
            if row.get("scene_id") == scene_id:
                return row
            continue
        idx += 1
        if idx == row_index:
            return row
    raise SystemExit(f"no matching row (scene_id={scene_id!r}, "
                     f"row_index={row_index}) in {manifest}")


def _run(cmd: list[str], log_path: Path) -> None:
    print(f"[run] {' '.join(cmd)}\n      log: {log_path}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT,
                             cwd=str(SCRIPT_DIR))
    if rc != 0:
        print(f"FAIL: exit={rc}; tail of {log_path}:", file=sys.stderr)
        print("\n".join(log_path.read_text().splitlines()[-25:]), file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--backend", default="sumo",
                    choices=["sumo", "pgmap", "paired", "citymap"])
    ap.add_argument("--scenes-root", default=str(SCRIPT_DIR.parents[1] / "scenes"))
    ap.add_argument("--policy", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--plant2-action-mode", default="pid",
                    choices=["pid", "wps_pure_pursuit"])
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--scene-id", default=None,
                    help="Pick the row with this scene_id (default: --row-index)")
    ap.add_argument("--row-index", type=int, default=0,
                    help="Pick the N-th valid row (default 0)")
    ap.add_argument("--workdir", default="/tmp/recorder_parity",
                    help="Scratch dir for the two runs (wiped per policy)")
    args = ap.parse_args()

    manifest = Path(args.manifest).resolve()
    scenes_root = Path(args.scenes_root).resolve()
    work = Path(args.workdir).resolve() / args.policy
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    row = _pick_row(manifest, args.scene_id, args.row_index, scenes_root)
    one_row = work / "one_row.jsonl"
    one_row.write_text(json.dumps(row) + "\n", encoding="utf-8")
    print(f"[row] scene={row.get('scene_id')} sign={row.get('sign_code')} "
          f"seed={row.get('seed') or row.get('deterministic_seed')} "
          f"braking={bool(row.get('braking_spawn'))}")

    py = sys.executable
    model_args = (["--model-path", args.model_path] if args.model_path else [])

    # 1) corrected eval, sidecar on
    _run([py, "run_benchmark.py",
          "--policy", args.policy, "--run-name", f"parity_{args.policy}",
          "--manifest", str(one_row), "--backends", args.backend,
          "--scenes-root", str(scenes_root), "--max-steps", str(args.max_steps),
          "--benchmark-output", str(work / "eval"), "--emit-replay-sidecar",
          "--plant2-action-mode", args.plant2_action_mode, *model_args],
         work / "eval.log")

    # 2) recorder (no spawn-velocity sampling — must match the eval exactly)
    _run([py, "expert_replay.py",
          "--manifest", str(one_row), "--backend", args.backend,
          "--policy", args.policy, "--count", "1",
          "--max-steps", str(args.max_steps),
          "--scenes-root", str(scenes_root),
          "--output-dir", str(work / "rec"),
          "--plant2-action-mode", args.plant2_action_mode, *model_args],
         work / "rec.log")

    ev_path = next((work / "eval").rglob("replay.json"), None)
    rc_path = next((work / "rec").rglob("replay.json"), None)
    if ev_path is None or rc_path is None:
        raise SystemExit(f"FAIL: sidecar missing (eval={ev_path}, rec={rc_path})")
    ev, rc = json.load(open(ev_path)), json.load(open(rc_path))

    me, mr = ev["metrics"], rc["metrics"]
    shared = sorted(set(me) & set(mr))
    bad = [k for k in shared if me[k] != mr[k]]
    acts_ok = ev.get("expert_actions") == rc.get("expert_actions")
    ident_bad = [k for k in ("scene_uid", "sign_slug", "policy", "variant")
                 if ev.get(k) != rc.get(k)]
    pkl = next((work / "rec").rglob("replay.pkl"), None)

    print(f"\n=== parity: {args.policy} on {row.get('scene_id')} ===")
    print(f"shared metric fields: {len(shared)}  mismatches: {bad or 'NONE'}")
    for k in bad:
        print(f"  {k}: eval={me[k]!r}  rec={mr[k]!r}")
    print(f"expert_actions identical: {acts_ok} "
          f"({len(ev.get('expert_actions') or [])} vs {len(rc.get('expert_actions') or [])} steps)")
    if ident_bad:
        for k in ident_bad:
            print(f"identity mismatch {k}: eval={ev.get(k)!r} rec={rc.get(k)!r}")
    print(f"recorded pkl: {pkl} "
          f"({pkl.stat().st_size} bytes)" if pkl else "recorded pkl: MISSING")
    print(f"key metrics: arrived={me.get('arrived_dest')} crash={me.get('crashed')} "
          f"steps={me.get('final_step')} viol_evt={me.get('violations_event_count')} "
          f"in_zone={me.get('in_zone_total_steps')}")

    ok = not bad and acts_ok and not ident_bad and pkl is not None
    print("\nPARITY:", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
