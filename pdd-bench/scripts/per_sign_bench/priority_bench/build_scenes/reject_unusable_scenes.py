#!/usr/bin/env python3
"""Reject scenes that cannot produce manifest scenarios, then optionally refill.

Runs at the **build_scenes** stage (not generate_manifest): maps that fail the
same viability checks used for harvest are marked reject, moved to
``scenes/_rejected/``, and the pool can be topped up to ``signs.yaml`` quotas.

Examples:
    # Dry-run: print which live scenes would be rejected for 4.3
    python build_scenes/reject_unusable_scenes.py --sign 4.3 --dry-run

    # Reject + move to _rejected/ + refill to quota
    python build_scenes/reject_unusable_scenes.py --sign 4.3 --apply --refill

    # Loop reject→refill until the live pool is fully viable (or pool empty)
    python build_scenes/reject_unusable_scenes.py --sign 4.3 --apply --refill --loop
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

BUILD_SCENES_DIR = Path(__file__).resolve().parent
PRIORITY_BENCH = BUILD_SCENES_DIR.parent
sys.path.insert(0, str(PRIORITY_BENCH))

from core.manifest.manifest_config import (  # noqa: E402
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
)
from core.manifest.manifest_viability import check_scene_dir_viability  # noqa: E402
from core.pool.scene_selection import (  # noqa: E402
    REJECTED_SUBDIR,
    apply_rejected_scenes,
    is_reserved_scene_dir,
    set_scene_reject,
)
from signs import get_profile, scenes_dir as profile_scenes_dir  # noqa: E402

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
        str(BUILD_SCENES_DIR / "materialize_scenes.py"),
        "--sign",
        sign,
        "--refill",
    ]
    if scenes_dir is not None:
        cmd.extend(["--scenes-dir", str(scenes_dir)])
    print("[reject-unusable] Running:", " ".join(cmd))
    return int(subprocess.call(cmd, cwd=str(PRIORITY_BENCH)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sign", default="4.3", help="Sign code / profile id (e.g. 4.3, roundabout)")
    ap.add_argument(
        "--scenes-dir",
        type=Path,
        default=None,
        help="Scenes root (default: data/<profile>/scenes)",
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

    profile = get_profile(args.sign)
    scenes_root = (
        args.scenes_dir.expanduser().resolve()
        if args.scenes_dir is not None
        else profile_scenes_dir(profile)
    )
    strategy = profile.spawn_strategy
    auxiliary_enabled = (not args.no_auxiliary) and strategy in (
        "yield",
        "roundabout",
    )

    if args.loop and not (args.apply and args.refill):
        sys.exit("ERROR: --loop requires both --apply and --refill")
    if args.refill and not args.apply and not args.dry_run:
        print(
            "[warn] --refill without --apply: rejects stay live and refill "
            "will not see a shortfall from them"
        )

    max_loops = max(1, int(args.max_loops)) if args.loop else 1
    all_audit: list[dict] = []

    for loop_i in range(1, max_loops + 1):
        print(
            f"[reject-unusable] sign={profile.pdd_code} strategy={strategy} "
            f"scenes={scenes_root} (loop {loop_i}/{max_loops})"
        )
        live_before = len(_live_scene_dirs(scenes_root))
        rows = reject_unusable(
            scenes_root=scenes_root,
            strategy=strategy,
            min_ego_lane_m=float(args.min_ego_lane_m),
            aux_distance_from_intersection=float(args.aux_distance_m),
            auxiliary_enabled=auxiliary_enabled,
            dry_run=bool(args.dry_run),
        )
        all_audit.extend(rows)
        print(
            f"[reject-unusable] live={live_before} newly_rejected={len(rows)} "
            f"reasons={dict(Counter(r['reason'] for r in rows))}"
        )

        if args.dry_run:
            break

        if args.apply and rows:
            only = [r["scene_id"] for r in rows]
            moved, total = apply_rejected_scenes(
                scenes_root, dry_run=False, only=only
            )
            print(
                f"[reject-unusable] applied {moved}/{total} → "
                f"{scenes_root / REJECTED_SUBDIR}"
            )
        elif args.apply and not rows:
            print("[reject-unusable] nothing to apply")

        if not args.refill:
            break

        rc = _run_refill(str(args.sign), args.scenes_dir)
        if rc != 0:
            sys.exit(rc)

        if not args.loop or not rows:
            # Stable when this pass found nothing to reject.
            if not rows:
                print("[reject-unusable] stable: all live scenes are viable")
            break
    else:
        if args.loop:
            print(
                f"[reject-unusable] stopped after {max_loops} loops "
                "(still seeing rejects; pool may be exhausted of viable maps)"
            )

    if args.audit is not None:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        with args.audit.open("w", encoding="utf-8") as f:
            for row in all_audit:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[reject-unusable] wrote audit → {args.audit}")

    if all_audit and not args.apply and not args.dry_run:
        print(
            "[reject-unusable] Next:\n"
            f"  python build_scenes/reject_unusable_scenes.py --sign {args.sign} "
            "--apply --refill"
        )


if __name__ == "__main__":
    main()
