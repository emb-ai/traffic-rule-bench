#!/usr/bin/env python3
"""Discover dual-path atoms on the Moscow city net and crop path-union scenes.

Fills a **fixed shared pool** per ``(shape, slot)`` under::

    scenes/dual_path/{T,X}/{slot}/<scene_id>/

Default ``--max-per-slot 500``. Same idea as junction T/X/O: many signs
(5.7 / 3.18 / 4.1 / 3.1) sample the **same** train/test maps (shared pool).
Pool size is therefore **not** ``n_train+n_test`` for one sign. We still cap
because each dual_path scene is a path-union netconvert (costlier than
junction-only); 500 ≈ order of the X junction inventory per bucket, enough
diversity under 80/20 split and slot/stem filters. At most one atom per
junction per slot; junctions shuffled with ``--seed``; early-stop when full.

Sign-free: no ``pdd_code`` in meta. Allocate later via ``lib.roles.sign_to_slots``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
PRIORITY_BENCH = ROOT.parent / "priority_bench"
for p in (str(ROOT), str(SCRIPTS), str(PRIORITY_BENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.layout.junction_priority_layout import JunctionLayoutError  # noqa: E402
from lib.dual_path import crop_to_dual_path, fill_slots_for_junctions  # noqa: E402
from lib.roles import SLOTS, slots_from_iterable  # noqa: E402

DEFAULT_NET = ROOT / "nets" / "moscow.net.xml"
DEFAULT_INDEX = ROOT / "index" / "junctions.jsonl"
DEFAULT_OUT = ROOT / "scenes" / "dual_path"
DEFAULT_CANDIDATES = ROOT / "index" / "dual_path_candidates.jsonl"
# Shared-pool inventory per (shape, slot). Not one-sign quota (80+20).
# ~X-scale; dual_path is rarer/heavier than junction-only so we cap compute.
DEFAULT_MAX_PER_SLOT = 500


def _load_index(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _count_existing(out_root: Path) -> Dict[Tuple[str, str], int]:
    """Count on-disk dual_path scenes per (shape, slot)."""
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    if not out_root.is_dir():
        return counts
    for shape_dir in out_root.iterdir():
        if not shape_dir.is_dir() or shape_dir.name not in ("T", "X"):
            continue
        for slot_dir in shape_dir.iterdir():
            if not slot_dir.is_dir():
                continue
            n = sum(1 for p in slot_dir.iterdir() if (p / "map.net.xml").is_file())
            if n:
                counts[(shape_dir.name, slot_dir.name)] = n
    return dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", type=Path, default=DEFAULT_NET)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--candidates-out", type=Path, default=DEFAULT_CANDIDATES)
    ap.add_argument("--shapes", default="T,X", help="Comma-separated T and/or X")
    ap.add_argument("--slots", default=",".join(SLOTS), help="Comma-separated slots")
    ap.add_argument(
        "--n-per-junction-slot",
        type=int,
        default=1,
        help="Max atoms from one junction for a given slot (default 1)",
    )
    ap.add_argument(
        "--max-per-slot",
        type=int,
        default=DEFAULT_MAX_PER_SLOT,
        help=(
            f"Max scenes per (shape, slot) shared pool (default {DEFAULT_MAX_PER_SLOT}; "
            "not one-sign 80+20 quota — many signs reuse the same maps)"
        ),
    )
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed for junction order")
    ap.add_argument("--min-gain-m", type=float, default=20.0)
    ap.add_argument("--min-lane-m", type=float, default=8.0)
    ap.add_argument("--margin-m", type=float, default=40.0)
    ap.add_argument("--max-junctions", type=int, default=None, help="Debug: cap junctions scanned")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--discover-only",
        action="store_true",
        help="Write candidates JSONL only; do not crop",
    )
    args = ap.parse_args()

    shapes = {s.strip().upper() for s in args.shapes.split(",") if s.strip()}
    slots = slots_from_iterable(s.strip() for s in args.slots.split(",") if s.strip())

    if not args.net.is_file():
        raise SystemExit(f"City net not found: {args.net}")
    if not args.index.is_file():
        raise SystemExit(f"Junction index not found: {args.index}")

    rows = [r for r in _load_index(args.index) if str(r.get("shape") or "").upper() in shapes]
    if args.max_junctions is not None:
        rows = rows[: max(0, int(args.max_junctions))]

    already = _count_existing(args.out) if args.skip_existing else {}
    if already:
        print(f"[dual_path] existing on disk: {dict(sorted(already.items()))}")

    print(
        f"[dual_path] net={args.net} junctions={len(rows)} "
        f"slots={list(slots)} max_per_slot={args.max_per_slot} "
        f"n_per_junction={args.n_per_junction_slot} seed={args.seed}"
    )
    t0 = time.time()
    filled = fill_slots_for_junctions(
        args.net,
        junction_rows=rows,
        slots=slots,
        n_per_junction_slot=int(args.n_per_junction_slot),
        max_per_shape_slot=int(args.max_per_slot),
        min_gain_m=float(args.min_gain_m),
        min_lane_length_m=float(args.min_lane_m),
        seed=int(args.seed),
        already_filled=already,
    )
    print(f"[dual_path] discovered {len(filled)} new atoms in {time.time() - t0:.1f}s")

    args.candidates_out.parent.mkdir(parents=True, exist_ok=True)
    with args.candidates_out.open("w", encoding="utf-8") as f:
        for row, sc in filled:
            shape = str(row.get("shape") or "").upper()
            rec = {
                "junction_id": sc.junction_id,
                "shape": shape,
                "slot": sc.slot,
                "ego_edge_id": sc.ego_edge_id,
                "dest_edge_id": sc.dest_edge_id,
                "baseline_dir": sc.baseline_dir,
                "compliant_dir": sc.compliant_dir,
                "gain_m": sc.gain_m,
                "ego_is_t_stem": sc.ego_is_t_stem,
                "carriageway_pair": sc.carriageway_pair,
                "scene_id": sc.scene_id(shape),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[dual_path] wrote {args.candidates_out}")

    if args.discover_only:
        return

    ok = 0
    fail = 0
    skipped = 0
    t1 = time.time()
    for row, sc in filled:
        shape = str(row.get("shape") or "").upper()
        scene_name = sc.scene_id(shape)
        scene_dir = args.out / shape / sc.slot / scene_name
        if args.skip_existing and (scene_dir / "map.net.xml").is_file():
            skipped += 1
            continue
        try:
            crop_to_dual_path(
                source_net=args.net,
                scenario=sc,
                output_dir=args.out / shape / sc.slot,
                shape=shape,
                base_row=row,
                margin_m=float(args.margin_m),
            )
            ok += 1
            if ok % 25 == 0:
                print(f"  cropped {ok}/{len(filled)} …")
        except (JunctionLayoutError, OSError, ValueError) as exc:
            fail += 1
            print(f"  FAIL {scene_name}: {exc}")
    print(
        f"[dual_path] crop done in {time.time() - t1:.1f}s: "
        f"ok={ok} fail={fail} skipped={skipped}"
    )


if __name__ == "__main__":
    main()
