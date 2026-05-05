h#!/usr/bin/env python3
"""Aggregate the output of collect_sumo_trajectories.sh into a per-(sign, scene)
index of all agents that recorded each scene.

Input: OUT_BASE produced by collect_sumo_trajectories.sh. The script reads
    OUT_BASE/_merged/all_runs.jsonl  (preferred — produced by the merge step), or
    OUT_BASE/<policy>/<sign>/all_runs.jsonl  (per-policy, fallback)

Each row in all_runs.jsonl describes one (scene × variant × policy) recording.
For each (sign_slug, scene_uid) we collect every agent that produced a replay,
where "agent" is `policy` for non-IDM policies and `policy_variant` for the
multi-variant comprehensive policy (matches the on-disk variant_full naming).

Output (jsonl, one line per (sign, scene)):
    {
      "sign_slug": "2_1",
      "scene_uid": "scene123_lane2_seed0_v0",
      "scene_id":  "scene123",
      "n_agents":  6,
      "agents": {
        "comprehensive_default": {
          "valid": true, "outcome": "reached",
          "pkl":     ".../replay.pkl",
          "sidecar": ".../replay.json",
          "arrived_dest": true, "crashed": false,
          "out_of_road": false, "total_violations": 0,
          "initial_speed_mps": 8.3, "total_reward": 12.4
        },
        "rule_compliant": { ... },
        "carl":   { ... },
        "plant2": { ... }
      }
    }

Usage:
    ./collect_recorded_scenes.py /path/to/OUT_BASE
    ./collect_recorded_scenes.py /path/to/OUT_BASE --out custom.jsonl
    ./collect_recorded_scenes.py /path/to/OUT_BASE --include-failed
    ./collect_recorded_scenes.py /path/to/OUT_BASE --signs 2_1,3_1
    ./collect_recorded_scenes.py /path/to/OUT_BASE --require carl,plant2,comprehensive_default,rule_compliant
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


OUTCOME_FIELDS = (
    "arrived_dest", "crashed", "crash_attribution", "out_of_road",
    "total_violations", "total_reward", "initial_speed_mps",
    "n_steps", "failure_reason",
)


def _agent_id(row: dict) -> str:
    """Match the variant_full naming used by expert_replay.run_batch_multi_variant:
    comprehensive uses `<policy>_<variant>`; everything else is just `<policy>`.
    """
    policy = row.get("policy") or "?"
    variant = row.get("variant") or "default"
    if policy == "comprehensive":
        return f"{policy}_{variant}"
    return policy


def _outcome(row: dict) -> str:
    if not row.get("valid"):
        return "invalid"
    if row.get("arrived_dest"):
        return "reached"
    if row.get("crashed"):
        return "crash_ego" if row.get("crash_attribution") == "ego" else "crash_npc"
    if row.get("out_of_road"):
        return "out_of_road"
    return "timeout"


def _iter_runs(out_base: Path):
    """Yield rows from all_runs.jsonl. Prefer the merged file; otherwise walk
    every <policy>/<sign>/all_runs.jsonl under OUT_BASE."""
    merged = out_base / "_merged" / "all_runs.jsonl"
    if merged.exists() and merged.stat().st_size > 0:
        sources = [merged]
    else:
        sources = sorted(out_base.glob("*/*/all_runs.jsonl"))
    if not sources:
        sys.exit(f"ERROR: no all_runs.jsonl under {out_base} "
                 f"(checked {merged} and <policy>/<sign>/all_runs.jsonl)")
    print(f"[info] reading {len(sources)} all_runs.jsonl source(s)", file=sys.stderr)
    for src in sources:
        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line), src
                except json.JSONDecodeError:
                    continue


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_base", type=Path,
                   help="OUT_BASE directory produced by collect_sumo_trajectories.sh")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSONL (default: OUT_BASE/_merged/scenes_by_sign.jsonl)")
    p.add_argument("--include-failed", action="store_true",
                   help="Include rows where valid=false (default: skip)")
    p.add_argument("--signs", type=str, default="",
                   help="Comma-separated whitelist of sign_slugs to keep")
    p.add_argument("--require", type=str, default="",
                   help="Comma-separated agent ids; only emit scenes covered by ALL of them. "
                        "Use the on-disk agent naming, e.g. "
                        "'comprehensive_default,rule_compliant,carl,plant2'.")
    args = p.parse_args()

    if not args.out_base.is_dir():
        sys.exit(f"ERROR: {args.out_base} is not a directory")

    out_path = args.out or (args.out_base / "_merged" / "scenes_by_sign.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sign_filter = {s for s in args.signs.split(",") if s} if args.signs else None
    required = {s for s in args.require.split(",") if s} if args.require else None

    # (sign_slug, scene_uid) -> { agent_id: row_summary }
    bucket: dict[tuple[str, str], dict] = defaultdict(dict)
    scene_id_lookup: dict[tuple[str, str], str] = {}

    n_rows = 0
    n_skipped_invalid = 0
    n_skipped_filter = 0
    for row, _src in _iter_runs(args.out_base):
        n_rows += 1
        if not args.include_failed and not row.get("valid"):
            n_skipped_invalid += 1
            continue
        sign = row.get("sign_slug") or row.get("sign_code")
        scene_uid = row.get("scene_uid")
        if not sign or not scene_uid:
            continue
        if sign_filter is not None and sign not in sign_filter:
            n_skipped_filter += 1
            continue

        agent = _agent_id(row)
        key = (sign, scene_uid)
        # Last writer wins on duplicates (resume can append twice).
        bucket[key][agent] = {
            "valid": bool(row.get("valid")),
            "outcome": _outcome(row),
            "pkl": row.get("pkl_path"),
            "sidecar": row.get("sidecar_path"),
            **{k: row.get(k) for k in OUTCOME_FIELDS if k in row},
        }
        if row.get("scene_id") is not None:
            scene_id_lookup[key] = row["scene_id"]

    # Per-agent and per-sign totals
    agent_totals: dict[str, int] = defaultdict(int)
    sign_totals: dict[str, int] = defaultdict(int)
    sign_agent_totals: dict[tuple[str, str], int] = defaultdict(int)
    n_emitted = 0
    n_dropped_require = 0

    with open(out_path, "w") as fo:
        for (sign, scene_uid), agents in sorted(bucket.items()):
            if required is not None and not required.issubset(agents.keys()):
                n_dropped_require += 1
                continue
            entry = {
                "sign_slug": sign,
                "scene_uid": scene_uid,
                "scene_id": scene_id_lookup.get((sign, scene_uid)),
                "n_agents": len(agents),
                "agents": agents,
            }
            fo.write(json.dumps(entry, default=str) + "\n")
            n_emitted += 1
            sign_totals[sign] += 1
            for a in agents:
                agent_totals[a] += 1
                sign_agent_totals[(sign, a)] += 1

    # ---- summary --------------------------------------------------------
    print(f"\nRead {n_rows} rows from all_runs.jsonl source(s).")
    if n_skipped_invalid:
        print(f"  skipped invalid: {n_skipped_invalid} (use --include-failed to keep)")
    if n_skipped_filter:
        print(f"  skipped by --signs filter: {n_skipped_filter}")
    if n_dropped_require:
        print(f"  dropped by --require {sorted(required)}: {n_dropped_require}")
    print(f"Wrote {n_emitted} (sign, scene) entries → {out_path}")

    if not n_emitted:
        return 0

    print(f"\n{'sign':<14}{'scenes':>8}  per-agent counts")
    print("-" * 70)
    all_agents = sorted(agent_totals)
    for sign in sorted(sign_totals):
        per = "  ".join(
            f"{a}={sign_agent_totals.get((sign, a), 0)}" for a in all_agents
        )
        print(f"{sign:<14}{sign_totals[sign]:>8}  {per}")
    print("-" * 70)
    totals_line = "  ".join(f"{a}={agent_totals[a]}" for a in all_agents)
    print(f"{'TOTAL':<14}{n_emitted:>8}  {totals_line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
