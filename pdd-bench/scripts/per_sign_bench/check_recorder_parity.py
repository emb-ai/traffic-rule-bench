#!/usr/bin/env python3
"""Prove recorder<->eval parity for one policy on one manifest row.

Runs the SAME (scene, seed, policy) through run_benchmark.py (the corrected
eval, with --emit-replay-sidecar) and through expert_replay.py (the trajectory
recorder), then diffs the two replay.json sidecars:

  * every shared `metrics` field must be identical,
  * the per-step ego actions (expert_actions) must be identical,
  * identity fields (scene_uid, sign_slug, policy, variant) must match.

Exit code 0 = parity holds; 1 = mismatch (printed field by field).

With --save-gifs it additionally:
  * records a top-down GIF in BOTH runs (eval + recorder),
  * replays the recorded pkl in-env (expert_replay_inenv.py) with its own GIF,
  * compares the GIFs frame by frame (eval vs recorder must be near-identical;
    pkl-replay is auto-aligned by frame offset and compared without the text
    overlay), and writes side_by_side.gif for eyeballing.

Usage (CPU policy — no checkpoint):
  python3 check_recorder_parity.py --manifest <m.jsonl> --scenes-root <scenes> \
      --policy comprehensive_rule_expert

NN policies (pin the GPU so both runs share one device):
  CUDA_VISIBLE_DEVICES=0 python3 check_recorder_parity.py --manifest <m.jsonl> \
      --scenes-root <scenes> --policy carl_rule --model-path $CARL_CKPT
  CUDA_VISIBLE_DEVICES=0 python3 check_recorder_parity.py --manifest <m.jsonl> \
      --scenes-root <scenes> --policy plant2_rule --model-path $PLANT2_CKPT \
      --plant2-action-mode pid --save-gifs
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


def _gif_frames(path: Path, crop_top: int = 0):
    """Load GIF frames as int16 RGB arrays (optionally cropping the top strip
    where the eval/recorder text overlay lives)."""
    import numpy as np
    from PIL import Image, ImageSequence
    return [np.asarray(f.convert("RGB"), dtype=np.int16)[crop_top:]
            for f in ImageSequence.Iterator(Image.open(path))]


def _gif_diff(a_frames, b_frames, offsets=(0,)):
    """Best (offset, mean_abs_diff, pct_pixels_gt30) over the given offsets."""
    import numpy as np
    best = None
    for off in offsets:
        means, pcts, n = [], [], 0
        for i in range(len(a_frames)):
            j = i + off
            if 0 <= j < len(b_frames) and a_frames[i].shape == b_frames[j].shape:
                d = np.abs(a_frames[i] - b_frames[j])
                means.append(float(d.mean()))
                pcts.append(float((d > 30).mean() * 100))
                n += 1
        if not n:
            continue
        cand = (off, sum(means) / n, sum(pcts) / n, n)
        if best is None or cand[1] < best[1]:
            best = cand
    return best  # (offset, mean, pct>30, n_frames) or None


def _side_by_side(a_path: Path, b_path: Path, out: Path, offset: int,
                  label_a: str, label_b: str) -> None:
    from PIL import Image, ImageDraw, ImageSequence
    fa = [f.convert("RGB").copy() for f in ImageSequence.Iterator(Image.open(a_path))]
    fb = [f.convert("RGB").copy() for f in ImageSequence.Iterator(Image.open(b_path))]
    combo = []
    for i in range(len(fa)):
        j = i + offset
        if not (0 <= j < len(fb)):
            continue
        aa = fa[i].resize((420, 420)); bb = fb[j].resize((420, 420))
        c = Image.new("RGB", (852, 452), "white")
        c.paste(aa, (2, 30)); c.paste(bb, (430, 30))
        dr = ImageDraw.Draw(c)
        dr.text((6, 8), f"{label_a}  frame {i}", fill="black")
        dr.text((434, 8), f"{label_b}  frame {j}", fill="black")
        combo.append(c)
    if combo:
        combo[0].save(out, save_all=True, append_images=combo[1:],
                      duration=40, loop=0)


def _gif_checks(work: Path, rec_sidecar: Path, py: str, max_steps: int) -> bool:
    """GIF part of the round-trip: replay pkl in-env with a GIF, then compare
    eval-vs-recorder GIFs and recorder-vs-pkl-replay GIFs. Returns ok flag."""
    try:
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        print(f"[gif] SKIP comparison (pip install pillow numpy): {exc}")
        return True

    eval_gif = next((work / "eval").rglob("*.gif"), None)
    rec_gif = next((work / "rec").rglob("replay.gif"), None)
    pkl = next((work / "rec").rglob("replay.pkl"), None)
    if not (eval_gif and rec_gif and pkl):
        print(f"[gif] FAIL: missing artifacts (eval_gif={eval_gif}, "
              f"rec_gif={rec_gif}, pkl={pkl})")
        return False

    # Replay the recorded pkl in-env with its own GIF.
    replay_gif = work / "replay_from_pkl.gif"
    _run([py, "expert_replay_inenv.py", "--pkl", str(pkl),
          "--sidecar", str(rec_sidecar), "--ego-mode", "recorded",
          "--npc-mode", "recorded", "--save-gif", str(replay_gif),
          "--max-steps", str(max_steps)], work / "inenv.log")
    inenv_log = (work / "inenv.log").read_text()
    violations_match = '"violations_match": true' in inenv_log
    print(f"[gif] in-env replay violations_match: {violations_match}")

    ok = violations_match

    # 1) eval GIF vs recorder GIF: same rollout code, same overlay — must be
    #    near-identical at offset 0 (residual = GIF palette noise).
    d = _gif_diff(_gif_frames(eval_gif), _gif_frames(rec_gif), offsets=(0,))
    if d:
        off, mean, pct, n = d
        verdict = "OK" if (mean < 1.0 and pct < 1.5) else "MISMATCH"
        print(f"[gif] eval vs recorder: {n} frames, mean_diff={mean:.2f}/255, "
              f">30px={pct:.2f}%  -> {verdict}")
        ok = ok and verdict == "OK"
    else:
        print("[gif] eval vs recorder: no comparable frames -> MISMATCH")
        ok = False

    # 2) recorder GIF vs pkl-replay GIF: replay renders a couple of extra lead
    #    frames and has no text overlay — crop the top strip, auto-align offset.
    d = _gif_diff(_gif_frames(rec_gif, crop_top=140),
                  _gif_frames(replay_gif, crop_top=140),
                  offsets=range(-3, 4))
    if d:
        off, mean, pct, n = d
        verdict = "OK" if (mean < 1.0 and pct < 1.5) else "MISMATCH"
        print(f"[gif] recorder vs pkl-replay: offset={off:+d}, {n} frames, "
              f"mean_diff={mean:.2f}/255, >30px={pct:.2f}%  -> {verdict}")
        ok = ok and verdict == "OK"
        _side_by_side(rec_gif, replay_gif, work / "side_by_side.gif", off,
                      "RECORD (live)", "REPLAY (from pkl)")
        print(f"[gif] side-by-side: {work / 'side_by_side.gif'}")
    else:
        print("[gif] recorder vs pkl-replay: no comparable frames -> MISMATCH")
        ok = False

    print(f"[gif] files: eval={eval_gif}\n            rec={rec_gif}\n"
          f"            replay={replay_gif}")
    return ok


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
    ap.add_argument("--save-gifs", action="store_true",
                    help="Record GIFs in both runs, replay the pkl in-env with "
                         "a GIF, and compare the frames (needs pillow+numpy)")
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
    gif_args = (["--save-gifs"] if args.save_gifs else [])

    # 1) corrected eval, sidecar on
    _run([py, "run_benchmark.py",
          "--policy", args.policy, "--run-name", f"parity_{args.policy}",
          "--manifest", str(one_row), "--backends", args.backend,
          "--scenes-root", str(scenes_root), "--max-steps", str(args.max_steps),
          "--benchmark-output", str(work / "eval"), "--emit-replay-sidecar",
          "--plant2-action-mode", args.plant2_action_mode,
          *gif_args, *model_args],
         work / "eval.log")

    # 2) recorder (no spawn-velocity sampling — must match the eval exactly)
    _run([py, "expert_replay.py",
          "--manifest", str(one_row), "--backend", args.backend,
          "--policy", args.policy, "--count", "1",
          "--max-steps", str(args.max_steps),
          "--scenes-root", str(scenes_root),
          "--output-dir", str(work / "rec"),
          "--plant2-action-mode", args.plant2_action_mode,
          *gif_args, *model_args],
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

    if args.save_gifs:
        print()
        ok = _gif_checks(work, rc_path, py, args.max_steps) and ok

    print("\nPARITY:", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
