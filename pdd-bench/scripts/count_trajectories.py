#!/usr/bin/env python3
"""Count collected trajectories under OUT_BASE.

Layout we expect (created by collect_sumo_trajectories.sh):
    OUT_BASE/<policy>/<sign>/by_sign/<sign>/by_scene/<scene_uid>/<variant>/replay.pkl

Usage:
    ./count_trajectories.py /path/to/traj_all_experts_ffff
    ./count_trajectories.py /path/to/traj_all_experts_ffff --target 2000
    ./count_trajectories.py /path/to/traj_all_experts_ffff --policy plant2 --details
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def collect_counts(out_base: Path) -> dict:
    """Walk OUT_BASE and count valid replays per (policy, sign, variant).

    A replay is "valid" if both replay.pkl and replay.json exist and are
    non-empty (matches PDD_BENCH_RESUME=1 logic in expert_replay.py).
    """
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    invalid: dict[tuple[str, str, str], int] = defaultdict(int)
    scene_uids: dict[tuple[str, str], set] = defaultdict(set)

    for policy_dir in sorted(out_base.iterdir()):
        if not policy_dir.is_dir() or policy_dir.name.startswith("_"):
            continue
        policy = policy_dir.name
        for sign_dir in sorted(policy_dir.iterdir()):
            if not sign_dir.is_dir():
                continue
            sign = sign_dir.name
            by_scene = sign_dir / "by_sign" / sign / "by_scene"
            if not by_scene.is_dir():
                continue
            for scene_dir in by_scene.iterdir():
                if not scene_dir.is_dir():
                    continue
                scene_uid = scene_dir.name
                for variant_dir in scene_dir.iterdir():
                    if not variant_dir.is_dir():
                        continue
                    variant = variant_dir.name
                    pkl = variant_dir / "replay.pkl"
                    sidecar = variant_dir / "replay.json"
                    key = (policy, sign, variant)
                    if (pkl.exists() and pkl.stat().st_size > 0
                            and sidecar.exists() and sidecar.stat().st_size > 0):
                        counts[key] += 1
                        scene_uids[(policy, sign)].add(scene_uid)
                    else:
                        invalid[key] += 1

    return {"counts": counts, "invalid": invalid, "unique_scenes": scene_uids}


def manifest_total(full_dir: Path, sign: str) -> int | None:
    """Return number of rows in real_manifest.jsonl for given sign, or None."""
    f = full_dir / sign / "real_manifest.jsonl"
    if not f.exists():
        return None
    try:
        with open(f) as fh:
            return sum(1 for _ in fh)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_base", type=Path,
                    help="Path to OUT_BASE (where comprehensive/, plant2/ etc live)")
    p.add_argument("--full-dir", type=Path, default=None,
                    help="benchmark_output/full/ — to compare with manifest size "
                         "(default: try to auto-resolve)")
    p.add_argument("--target", type=int, default=None,
                    help="Target episode count per (policy, sign). "
                         "For comprehensive add ×5 internally for 4 extra variants.")
    p.add_argument("--policy", type=str, default=None,
                    help="Only show this policy (carl/plant2/comprehensive/rule_compliant)")
    p.add_argument("--details", action="store_true",
                    help="Also show breakdown by variant for comprehensive")
    p.add_argument("--missing", action="store_true",
                    help="Only show (policy, sign) pairs that are NOT complete")
    p.add_argument("--json", action="store_true",
                    help="Output JSON instead of pretty table")
    args = p.parse_args()

    if not args.out_base.is_dir():
        print(f"ERROR: {args.out_base} does not exist", file=sys.stderr)
        sys.exit(1)

    # Auto-resolve full_dir relative to out_base if not given
    if args.full_dir is None:
        sdc_root = args.out_base.parent
        guess = sdc_root / "pdd-bench" / "scripts" / "per_sign_bench" / "benchmark_output" / "full"
        if guess.is_dir():
            args.full_dir = guess

    data = collect_counts(args.out_base)
    counts = data["counts"]
    invalid = data["invalid"]
    unique_scenes = data["unique_scenes"]

    # Per (policy, sign) totals (sum across variants)
    per_ps: dict[tuple[str, str], int] = defaultdict(int)
    for (pol, sign, _var), n in counts.items():
        per_ps[(pol, sign)] += n

    policies = sorted({p for (p, _, _) in counts})
    if args.policy:
        policies = [args.policy] if args.policy in policies else []

    if args.json:
        all_unique = defaultdict(set)
        for (pol, sign), uids in unique_scenes.items():
            all_unique[sign] |= uids
        out = {
            "out_base": str(args.out_base),
            "grand_total_trajectories": sum(counts.values()),
            "grand_total_unique_scenarios": sum(len(u) for u in all_unique.values()),
            "signs_with_data": len(all_unique),
            "by_policy": {pol: sum(n for (p, _, _), n in counts.items() if p == pol)
                           for pol in policies},
            "by_policy_sign": {f"{p}/{s}": n for (p, s), n in per_ps.items()},
            "by_policy_sign_variant": {f"{p}/{s}/{v}": n
                                          for (p, s, v), n in counts.items()},
            "unique_scenes_per_sign": {f"{p}/{s}": len(uids)
                                          for (p, s), uids in unique_scenes.items()},
            "unique_scenes_total_per_sign": {s: len(u) for s, u in all_unique.items()},
        }
        print(json.dumps(out, indent=2, default=str))
        return

    # Pretty table — header
    print(f"\n{'='*78}")
    print(f"OUT_BASE: {args.out_base}")
    if args.full_dir:
        print(f"FULL_DIR: {args.full_dir}")
    print(f"{'='*78}")

    # Grand totals
    grand_total = sum(counts.values())
    grand_invalid = sum(invalid.values())
    # Unique scenarios covered across ALL policies (scene_uid per sign)
    all_unique_scenes: dict[str, set] = defaultdict(set)
    for (pol, sign), uids in unique_scenes.items():
        all_unique_scenes[sign] |= uids
    total_unique_scenes = sum(len(u) for u in all_unique_scenes.values())
    n_signs_with_data = len(all_unique_scenes)

    # Manifest sums (if FULL_DIR provided)
    manifest_total_rows = None
    manifest_per_sign: dict[str, int] = {}
    if args.full_dir is not None:
        for sign in all_unique_scenes:
            m = manifest_total(args.full_dir, sign)
            if m is not None:
                manifest_per_sign[sign] = m
        if manifest_per_sign:
            manifest_total_rows = sum(manifest_per_sign.values())

    print(f"\n--- GRAND TOTAL ---")
    print(f"  trajectories (valid replay.pkl): {grand_total:>8}")
    print(f"  invalid / partial dirs:          {grand_invalid:>8}")
    print(f"  unique scenarios covered:        {total_unique_scenes:>8}"
          + (f"  / {manifest_total_rows} в манифесте"
              if manifest_total_rows else ""))
    print(f"  signs with data:                 {n_signs_with_data:>8}")
    if manifest_per_sign:
        coverage_pct = 100.0 * total_unique_scenes / max(manifest_total_rows, 1)
        print(f"  scenario coverage:               {coverage_pct:>7.1f}%")

    # Per-policy totals
    print(f"\n--- Total by policy ---")
    for pol in policies:
        total = sum(n for (p, _, _), n in counts.items() if p == pol)
        n_signs = sum(1 for (p, _) in per_ps if p == pol)
        bad = sum(n for (p, _, _), n in invalid.items() if p == pol)
        # unique scenarios for this policy
        n_uniq = sum(len(u) for (p, _), u in unique_scenes.items() if p == pol)
        print(f"  {pol:<18} total={total:>6}   signs={n_signs:>3}"
              f"   unique_scenes={n_uniq:>5}"
              f"   invalid_dirs={bad:>4}")

    # Per (policy, sign)
    print(f"\n--- By policy × sign ---")
    print(f"{'policy':<18}{'sign':<10}{'count':>8}  {'unique':>8}", end="")
    if args.full_dir is not None:
        print(f"  {'manifest':>9}  {'comp_target':>12}  {'progress':>10}")
    else:
        print()

    all_signs = sorted({s for (p, s) in per_ps if not args.policy or p == args.policy})

    rows = []
    for pol in policies:
        for sign in all_signs:
            n = per_ps.get((pol, sign), 0)
            uniq = len(unique_scenes.get((pol, sign), set()))
            manifest_n = manifest_total(args.full_dir, sign) if args.full_dir else None

            # Expected count for completeness:
            if args.target is not None:
                target = args.target
            elif manifest_n is not None:
                target = manifest_n
            else:
                target = None
            # comprehensive expects target × 5 (default + 4 IDM variants)
            if pol == "comprehensive" and target is not None:
                target *= 5

            is_complete = target is not None and n >= target
            if args.missing and is_complete:
                continue
            if args.missing and target is None:
                continue
            rows.append((pol, sign, n, uniq, manifest_n, target, is_complete))

    for pol, sign, n, uniq, manifest_n, target, is_complete in rows:
        line = f"{pol:<18}{sign:<10}{n:>8}  {uniq:>8}"
        if args.full_dir is not None:
            line += f"  {(manifest_n if manifest_n is not None else '-'):>9}"
            if target is not None:
                pct = 100.0 * n / target if target > 0 else 0.0
                tag = "OK" if is_complete else f"{pct:5.1f}%"
                line += f"  {target:>12}  {tag:>10}"
            else:
                line += f"  {'-':>12}  {'-':>10}"
        print(line)

    # Variant breakdown for --details (only for comprehensive)
    if args.details:
        print(f"\n--- Variant breakdown (comprehensive only) ---")
        print(f"{'sign':<10}{'default':>9}{'s1':>6}{'s2':>6}{'s3':>6}{'s4':>6}{'total':>8}")
        for sign in all_signs:
            counts_per_var = {v: counts.get(("comprehensive", sign, f"comprehensive_{v}"), 0)
                              for v in ("default", "s1", "s2", "s3", "s4")}
            total = sum(counts_per_var.values())
            if total == 0:
                continue
            line = f"{sign:<10}"
            for v in ("default", "s1", "s2", "s3", "s4"):
                line += f"{counts_per_var[v]:>6}"
                if v == "default":
                    line = line[:9] + " " + line[9:]  # extra space after default
            line += f"{total:>8}"
            print(line)

    print()


if __name__ == "__main__":
    main()
