"""Full scene reconstruction (NPCs from pkl) + live comprehensive rule expert.

For each recorded scene (pkl + .meta.json sidecar) found under
`<mini-new-root>/<code>/expert/replays/`, we rebuild the env via
`expert_replay_inenv.replay_in_our_env` with:

    ego_mode = "live"       → ComprehensiveRuleExpertPolicy drives the ego
    npc_mode = "recorded"   → NPCs/pedestrians frozen to recorded tracks (pkl)

Road width is taken from SUMO_DEFAULT_CONFIG (`min_lane_width=4.5`) — no extra
flags needed.

GIF recording style mirrors run_benchmark.py (semantic top-down,
target_agent_heading_up, screen_record + generate_gif).

Outputs:
    sdc/pdd-bench/replay_results/inenv_<code>_<ts>/
        episode_results.jsonl
        summary.json
    sdc/pdd-bench/gifs/inenv_<code>_<ts>/
        <scene_id>.gif
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PDD_BENCH_DIR = SCRIPT_PATH.parent.parent
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"
PER_SIGN_DIR = PDD_BENCH_DIR / "scripts" / "per_sign_bench"
for _p in (PDD_BENCH_DIR, METADRIVE_DIR, PER_SIGN_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)


def _find_pairs(replays_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return list of (scene_id, pkl_path, sidecar_path) found in `replays_dir`."""
    pairs = []
    for sidecar in sorted(replays_dir.glob("*.meta.json")):
        scene_id = sidecar.stem.replace(".meta", "")
        pkl = sidecar.with_name(f"sd_{scene_id}.pkl")
        if pkl.exists():
            pairs.append((scene_id, pkl, sidecar))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign-code", default="2.5")
    ap.add_argument("--mini-new-root", default="/Users/victoria_s/sdc_new_signs/mini_new")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--limit", type=int, default=None,
                    help="Max scenes to replay (default: all pairs)")
    ap.add_argument("--no-gifs", action="store_true")
    ap.add_argument("--output-dir", default=None,
                    help="Metrics dir (default: sdc/pdd-bench/replay_results/inenv_<code>_<ts>)")
    ap.add_argument("--gifs-dir", default=None,
                    help="Gifs dir (default: sdc/pdd-bench/gifs/inenv_<code>_<ts>)")
    args = ap.parse_args()

    code_dir = args.sign_code.replace(".", "_")
    replays_dir = Path(args.mini_new_root) / code_dir / "expert" / "replays"
    if not replays_dir.exists():
        print(f"ERROR: no replays dir: {replays_dir}", file=sys.stderr)
        sys.exit(1)

    pairs = _find_pairs(replays_dir)
    if not pairs:
        print(f"ERROR: no pkl/sidecar pairs in {replays_dir}", file=sys.stderr)
        sys.exit(1)
    if args.limit is not None:
        pairs = pairs[:args.limit]

    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PDD_BENCH_DIR / "replay_results" / f"inenv_{code_dir}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gifs_dir:
        gifs_dir = Path(args.gifs_dir)
    else:
        gifs_dir = PDD_BENCH_DIR / "gifs" / f"inenv_{code_dir}_{ts}"
    if not args.no_gifs:
        gifs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Replays dir: {replays_dir}")
    print(f"Found pairs: {len(pairs)}")
    print(f"Output dir:  {output_dir}")
    print(f"GIFs dir:    {gifs_dir if not args.no_gifs else '(off)'}")
    print(f"Mode:        ego=live (ComprehensiveRuleExpertPolicy), npc=recorded")
    print()

    logging.getLogger().setLevel(logging.ERROR)

    from expert_replay_inenv import replay_in_our_env

    t0 = time.time()
    results: list[dict] = []
    for i, (scene_id, pkl, sidecar) in enumerate(pairs):
        gif_path = (gifs_dir / f"{scene_id}.gif") if not args.no_gifs else None
        print(f"[{i+1}/{len(pairs)}] {scene_id}  pkl={pkl.name}")
        try:
            res = replay_in_our_env(
                pkl, sidecar,
                ego_mode="live",       # comprehensive expert drives (via agent_policy default)
                npc_mode="recorded",   # NPCs/pedestrians frozen to recorded pkl tracks
                render_2d=False,
                render_3d=False,
                save_gif=gif_path,
                max_steps=args.max_steps,
            )
            res["gif_path"] = str(gif_path) if gif_path else None
        except Exception as exc:
            print(f"   [ERROR] {type(exc).__name__}: {exc}")
            res = {
                "scene_id": scene_id,
                "error": f"{type(exc).__name__}: {exc}",
                "arrived_dest": False,
                "crashed": False,
                "steps_run": 0,
                "violations_match": False,
                "gif_path": None,
            }

        results.append(res)
        with open(output_dir / "episode_results.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")

        outcome = ("arrived" if res.get("arrived_dest") else
                   "crashed" if res.get("crashed") else "other")
        print(f"   -> {outcome}  steps={res.get('steps_run')}  "
              f"viol_match={res.get('violations_match')}")

    n = len(results) or 1
    arrived = sum(1 for r in results if r.get("arrived_dest"))
    crashed = sum(1 for r in results if r.get("crashed"))
    errors = sum(1 for r in results if "error" in r)

    summary = {
        "sign_code": args.sign_code,
        "mode": "ego=live(comprehensive) npc=recorded",
        "n_episodes": n,
        "n_errors": errors,
        "dest_rate": round(arrived / n, 3),
        "crash_rate": round(crashed / n, 3),
        "violations_match_rate": round(
            sum(1 for r in results if r.get("violations_match")) / n, 3),
        "wall_time_s": round(time.time() - t0, 1),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 80)
    print(f"per-scene outcomes (n={n})")
    print("=" * 80)
    print(f"{'scene_id':<30} {'arrived':>8} {'crashed':>8} {'steps':>6}  gif")
    print("-" * 80)
    for r in results:
        print(f"{r.get('scene_id',''):<30} "
              f"{str(r.get('arrived_dest','')):>8} "
              f"{str(r.get('crashed','')):>8} "
              f"{str(r.get('steps_run','')):>6}  "
              f"{Path(r['gif_path']).name if r.get('gif_path') else '-'}")
    print()
    print("=" * 80)
    print("summary")
    print("=" * 80)
    for k, v in summary.items():
        print(f"  {k:<25} = {v}")


if __name__ == "__main__":
    main()
