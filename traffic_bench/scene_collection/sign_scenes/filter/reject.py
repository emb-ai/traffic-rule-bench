#!/usr/bin/env python3
"""Reject scenes that cannot produce manifest scenarios, then optionally refill.

Runs after materialize: maps that fail the same viability checks used for
harvest are marked reject, moved to ``scenes/_rejected/``, and the pool can
be topped up to ``signs.yaml`` quotas.

``--sign`` is the eval profile id (same as ``python -m traffic_bench.eval manifest sign=...``).

Examples:
    python -m traffic_bench.scene_collection reject --sign roundabout --dry-run
    python -m traffic_bench.scene_collection reject --sign roundabout --apply --refill
    python -m traffic_bench.scene_collection reject --sign roundabout --apply --refill --loop
    python -m traffic_bench.scene_collection reject --all --apply --refill --loop
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

from traffic_bench.eval.engine.expand.manifest_config import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
)
from traffic_bench.eval.engine.expand.manifest_viability import check_scene_dir_viability
from traffic_bench.scene_collection.paths import REPO_ROOT
from traffic_bench.scene_collection.sign_scenes.filter.selection import (
    REJECTED_SUBDIR,
    apply_rejected_scenes,
    is_reserved_scene_dir,
    set_scene_reject,
)
from traffic_bench.eval.sign_registry import (
    SignProfile,
    get_profile,
    list_profiles,
    scenes_dir as profile_scenes_dir,
)

# Match generate_manifest ``parse_sumo_net_for_spawn_lanes`` default min length.
DEFAULT_MIN_EGO_LANE_M = 20.0


def _live_scene_dirs(scenes_root: Path) -> List[Path]:
    out: List[Path] = []
    if not scenes_root.is_dir():
        return out
    for entry in sorted(scenes_root.iterdir()):
        if is_reserved_scene_dir(entry.name):
            continue
        if not entry.is_dir() and not entry.is_symlink():
            continue
        if (entry / "meta.json").is_file() or (entry / "map.net.xml").is_file():
            out.append(entry)
    return out


def reject_unusable(
    *,
    scenes_root: Path,
    strategy: str,
    min_ego_lane_m: float,
    aux_distance_from_intersection: float,
    auxiliary_enabled: bool,
    dry_run: bool,
    pdd_code: str | None = None,
) -> list[dict]:
    """Mark non-viable live scenes as reject. Returns audit rows."""
    rejected_rows: list[dict] = []
    for scene_dir in _live_scene_dirs(scenes_root):
        result = check_scene_dir_viability(
            scene_dir,
            strategy=strategy,  # type: ignore[arg-type]
            min_ego_lane_m=min_ego_lane_m,
            aux_distance_from_intersection=aux_distance_from_intersection,
            auxiliary_enabled=auxiliary_enabled,
            pdd_code=pdd_code,
        )
        if result.viable:
            continue
        row = {
            "scene_id": scene_dir.name,
            "reason": result.reason or "not_viable",
            "detail": result.detail or "",
            "scenario_count": int(result.scenario_count),
            "spawn_lane_count": int(result.spawn_lane_count),
        }
        rejected_rows.append(row)
        tag = f"{row['reason']}" + (f" ({row['detail']})" if row["detail"] else "")
        if dry_run:
            print(f"  [dry-run] would reject {scene_dir.name}: {tag}")
            continue
        set_scene_reject(
            scenes_root,
            scene_dir.name,
            reason=row["reason"],
            detail=row["detail"],
        )
        print(f"  [reject] {scene_dir.name}: {tag}")
    return rejected_rows


def _run_refill(sign: str, scenes_dir: Optional[Path]) -> int:
    cmd = [
        sys.executable,
        "-m",
        "traffic_bench.scene_collection",
        "materialize",
        "--sign",
        sign,
        "--refill",
    ]
    if scenes_dir is not None:
        cmd.extend(["--scenes-dir", str(scenes_dir)])
    print("[reject-unusable] Running:", " ".join(cmd))
    return int(subprocess.call(cmd, cwd=str(REPO_ROOT)))


def _run_one_sign(
    *,
    profile: SignProfile,
    scenes_root: Path,
    dry_run: bool,
    apply: bool,
    refill: bool,
    loop: bool,
    max_loops: int,
    min_ego_lane_m: float,
    aux_distance_m: float,
    no_auxiliary: bool,
    scenes_dir_override: Optional[Path],
) -> list[dict]:
    """Reject→apply→refill for one sign. Returns audit rows."""
    strategy = profile.spawn_strategy
    auxiliary_enabled = (not no_auxiliary) and strategy in (
        "yield",
        "roundabout",
    )
    all_audit: list[dict] = []
    n_loops = max(1, int(max_loops)) if loop else 1

    for loop_i in range(1, n_loops + 1):
        print(
            f"[reject-unusable] sign={profile.id} ({profile.pdd_code}) strategy={strategy} "
            f"scenes={scenes_root} (loop {loop_i}/{n_loops})"
        )
        live_before = len(_live_scene_dirs(scenes_root))
        rows = reject_unusable(
            scenes_root=scenes_root,
            strategy=strategy,
            min_ego_lane_m=float(min_ego_lane_m),
            aux_distance_from_intersection=float(aux_distance_m),
            auxiliary_enabled=auxiliary_enabled,
            dry_run=bool(dry_run),
            pdd_code=profile.pdd_code,
        )
        all_audit.extend(rows)
        print(
            f"[reject-unusable] live={live_before} newly_rejected={len(rows)} "
            f"reasons={dict(Counter(r['reason'] for r in rows))}"
        )

        if dry_run:
            break

        if apply and rows:
            only = [r["scene_id"] for r in rows]
            moved, total = apply_rejected_scenes(
                scenes_root, dry_run=False, only=only
            )
            print(
                f"[reject-unusable] applied {moved}/{total} → "
                f"{scenes_root / REJECTED_SUBDIR}"
            )
        elif apply and not rows:
            print("[reject-unusable] nothing to apply")

        if not refill:
            break

        rc = _run_refill(profile.id, scenes_dir_override)
        if rc != 0:
            sys.exit(rc)

        if not loop or not rows:
            if not rows:
                print("[reject-unusable] stable: all live scenes are viable")
            break
    else:
        if loop:
            print(
                f"[reject-unusable] stopped after {n_loops} loops "
                f"for {profile.id} (still seeing rejects; pool may be exhausted)"
            )
    return all_audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--sign",
        metavar="ID",
        help=(
            "Eval sign id, same as `python -m traffic_bench.eval manifest sign=...` "
            f"(e.g. yield, roundabout, crosswalk). "
            f"Known: {', '.join(sorted(p.id for p in list_profiles()))}"
        ),
    )
    g.add_argument(
        "--all",
        action="store_true",
        help="Run every eval sign profile that has a data/scenes/<id>/ folder",
    )
    ap.add_argument(
        "--scenes-dir",
        type=Path,
        default=None,
        help="Scenes root for --sign (default: data/scenes/<profile>); invalid with --all",
    )
    ap.add_argument(
        "--min-ego-lane-m",
        type=float,
        default=DEFAULT_MIN_EGO_LANE_M,
        help=f"Min approach lane length (default {DEFAULT_MIN_EGO_LANE_M})",
    )
    ap.add_argument(
        "--aux-distance-m",
        type=float,
        default=DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
        help=f"Aux distance from intersection (default {DEFAULT_AUX_DISTANCE_FROM_INTERSECTION})",
    )
    ap.add_argument(
        "--no-auxiliary",
        action="store_true",
        help="Do not require a viable aux arm (layout-only viability)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print which scenes would be rejected",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help=f"Move rejected scene dirs to {REJECTED_SUBDIR}/",
    )
    ap.add_argument(
        "--refill",
        action="store_true",
        help="After apply, top up kept counts via materialize_scenes --refill",
    )
    ap.add_argument(
        "--loop",
        action="store_true",
        help="Repeat reject→apply→refill until no new rejects (requires --apply --refill)",
    )
    ap.add_argument(
        "--max-loops",
        type=int,
        default=10,
        help="Safety cap for --loop (default 10)",
    )
    ap.add_argument(
        "--audit",
        type=Path,
        default=None,
        help="Optional path to write rejected rows as JSONL",
    )
    args = ap.parse_args()

    if not args.all and not args.sign:
        args.sign = "roundabout"
    if args.all and args.scenes_dir is not None:
        sys.exit("ERROR: --scenes-dir cannot be used with --all")
    if args.loop and not (args.apply and args.refill):
        sys.exit("ERROR: --loop requires both --apply and --refill")
    if args.refill and not args.apply and not args.dry_run:
        print(
            "[warn] --refill without --apply: rejects stay live and refill "
            "will not see a shortfall from them"
        )

    if args.all:
        profiles = []
        for profile in list_profiles():
            root = profile_scenes_dir(profile)
            if root.is_dir():
                profiles.append(profile)
            else:
                print(f"[reject-unusable] skip {profile.id}: no scenes dir {root}")
        if not profiles:
            sys.exit("ERROR: --all found no data/scenes/<sign>/ directories")
        print(
            f"[reject-unusable] --all: {len(profiles)} sign(s): "
            f"{', '.join(p.id for p in profiles)}"
        )
    else:
        profiles = [get_profile(args.sign)]

    all_audit: list[dict] = []
    for i, profile in enumerate(profiles, 1):
        if args.all:
            print(f"\n======== [{i}/{len(profiles)}] {profile.id} ========")
        scenes_root = (
            args.scenes_dir.expanduser().resolve()
            if args.scenes_dir is not None
            else profile_scenes_dir(profile)
        )
        rows = _run_one_sign(
            profile=profile,
            scenes_root=scenes_root,
            dry_run=bool(args.dry_run),
            apply=bool(args.apply),
            refill=bool(args.refill),
            loop=bool(args.loop),
            max_loops=int(args.max_loops),
            min_ego_lane_m=float(args.min_ego_lane_m),
            aux_distance_m=float(args.aux_distance_m),
            no_auxiliary=bool(args.no_auxiliary),
            scenes_dir_override=args.scenes_dir,
        )
        all_audit.extend(
            {**row, "sign": profile.id} for row in rows
        )

    if args.audit is not None:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        with args.audit.open("w", encoding="utf-8") as f:
            for row in all_audit:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[reject-unusable] wrote audit → {args.audit}")

    if all_audit and not args.apply and not args.dry_run:
        target = "--all" if args.all else f"--sign {profiles[0].id}"
        print(
            "[reject-unusable] Next:\n"
            f"  python -m traffic_bench.scene_collection reject "
            f"{target} --apply --refill"
        )


if __name__ == "__main__":
    main()
