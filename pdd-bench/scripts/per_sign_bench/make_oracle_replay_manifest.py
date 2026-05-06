#!/usr/bin/env python3
"""Build oracle replay manifest from select_experts.py output.

Takes experts_selected.jsonl (one winner per scene); for each pick it reads
the sidecar (replay.json), pulls source_row + env_config_summary +
expert metadata, and writes oracle_replay_manifest.jsonl — a format ready
to feed back into expert_replay.py to reproduce the scenes.

Each row contains:
  - all source_row fields (net_path, seed, sign_type, road_id, lane_num,
    spawn_velocity_ms, sign_spawn_distance, pdd_code, ...)
  - winner_policy / winner_variant + metrics (f1, time_eff, comfort)
  - pkl_path / gif_path / sidecar_path — paths to the recorded trajectory
  - env_config_summary (horizon, traffic_density, seed)

Usage:
  python3 make_oracle_replay_manifest.py \\
      --picks   experts_selected.jsonl \\
      --output  oracle_replay_manifest.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        print(f"[error] not found: {p}", file=sys.stderr)
        sys.exit(1)
    rows = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
    return rows


def build_replay_row(pick: dict) -> dict | None:
    """Merge source_row from sidecar with winner metadata."""
    sc_path = pick.get("sidecar_path")
    if not sc_path or not Path(sc_path).exists():
        print(f"[warn] sidecar missing for {pick['sign']}/{pick['scene_id']}: "
              f"{sc_path}", file=sys.stderr)
        return None
    try:
        sidecar = json.loads(Path(sc_path).read_text())
    except Exception as e:
        print(f"[warn] sidecar parse error: {sc_path}: {e}", file=sys.stderr)
        return None

    src = sidecar.get("source_row") or {}
    env_cfg = sidecar.get("env_config_summary") or {}
    metrics = sidecar.get("metrics") or {}

    out = dict(src)

    out.setdefault("scene_id", pick["scene_id"])
    out.setdefault("sign_code", pick["sign"])
    out.setdefault("seed", env_cfg.get("seed"))
    if "horizon" not in out:
        out["horizon"] = env_cfg.get("horizon", 600)

    for k in ("road_id", "spawn_lane_num", "block_sequence",
              "lane_num", "lane_width", "traffic_density"):
        if k not in out and k in env_cfg:
            out[k] = env_cfg[k]

    out["initial_speed_mps"] = metrics.get("initial_speed_mps") or out.get("spawn_velocity_ms") or 0.0
    out["recorded_final_step"] = metrics.get("final_step")
    out["recorded_total_violations"] = metrics.get("total_violations")
    out["recorded_violations_by_class"] = metrics.get("violations_by_class") or {}
    out["recorded_arrived_dest"] = metrics.get("arrived_dest")
    out["recorded_route_completion"] = metrics.get("route_completion")
    out["recorded_total_reward"] = metrics.get("total_reward")
    out["recorded_frame_smooth_ratio"] = metrics.get("frame_smooth_ratio")

    out["oracle_winner_policy"] = pick["winner_policy"]
    out["oracle_winner_variant"] = pick["winner_variant"]
    out["oracle_best_idm_variant"] = pick.get("best_idm_variant")
    out["oracle_f1_score"] = pick["f1_score"]
    out["oracle_time_eff"] = pick["time_eff"]
    out["oracle_comfort"] = pick["comfort"]
    out["oracle_scene_min_step"] = pick.get("scene_min_step")
    out["oracle_beta"] = pick.get("beta", 0.25)
    out["oracle_passing_candidates_n"] = pick.get("passing_candidates_n")

    out["pkl_path"] = pick.get("pkl_path")
    out["gif_path"] = pick.get("gif_path")
    out["sidecar_path"] = pick.get("sidecar_path")

    return out


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--picks", required=True,
                   help="experts_selected.jsonl from select_experts.py")
    p.add_argument("--output", default="oracle_replay_manifest.jsonl",
                   help="output jsonl manifest for replay")
    p.add_argument("--require-pkl", action="store_true",
                   help="skip picks whose pkl file is missing")
    args = p.parse_args()

    picks = load_jsonl(args.picks)
    print(f"Loaded {len(picks)} picks from {args.picks}")

    out_rows = []
    skipped_no_sidecar = 0
    skipped_no_pkl = 0
    for pick in picks:
        row = build_replay_row(pick)
        if row is None:
            skipped_no_sidecar += 1
            continue
        if args.require_pkl:
            pkl = row.get("pkl_path")
            if not pkl or not Path(pkl).exists():
                skipped_no_pkl += 1
                continue
        out_rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"\nWrote {len(out_rows)} replay rows → {out_path}")
    if skipped_no_sidecar:
        print(f"  skipped (no sidecar):  {skipped_no_sidecar}")
    if skipped_no_pkl:
        print(f"  skipped (no pkl):      {skipped_no_pkl}")

    # Per-sign summary
    by_sign = {}
    for r in out_rows:
        sign = r.get("sign_code") or r.get("pdd_code") or "?"
        by_sign.setdefault(sign, []).append(r)
    print("\nReplay manifest per sign:")
    for sign in sorted(by_sign):
        items = by_sign[sign]
        winners = {}
        for r in items:
            w = f"{r.get('oracle_winner_policy')}_{r.get('oracle_winner_variant')}"
            winners[w] = winners.get(w, 0) + 1
        wstr = ", ".join(f"{k}×{v}" for k, v in sorted(winners.items(),
                                                         key=lambda kv: -kv[1]))
        print(f"  {sign:<10}: {len(items):>3} scenes  ({wstr})")


if __name__ == "__main__":
    main()
