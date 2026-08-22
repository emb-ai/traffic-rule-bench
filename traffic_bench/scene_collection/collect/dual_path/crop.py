#!/usr/bin/env python3
"""Discover dual-path atoms on the Moscow city net and crop path-union scenes.

Fills a shared pool per ``(shape, slot)`` under::

    crops/dual_path/{T,X}/{slot}/<scene_id>/

Default ``--max-per-slot 0`` (no cap: crop until the city runs out).
Signs sample the same train/test maps. At most one atom per junction per slot;
junctions shuffled with ``--seed``. Interrupt-safe with ``--skip-existing``.
Sign-free: no ``pdd_code`` in meta. Allocate later via ``collect.dual_path.roles.sign_to_slots``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from traffic_bench.eval.core.layout.junction_priority_layout import JunctionLayoutError
from traffic_bench.scene_collection.collect.dual_path.graph import (
    crop_to_dual_path,
    fill_slots_for_junctions,
)
from traffic_bench.scene_collection.collect.dual_path.roles import SLOTS, slots_from_iterable
from traffic_bench.scene_collection.paths import DUAL_PATH_CANDIDATES, DUAL_PATH_CROPS, JUNCTIONS_INDEX, MOSCOW_NET

DEFAULT_NET = MOSCOW_NET
DEFAULT_INDEX = JUNCTIONS_INDEX
DEFAULT_OUT = DUAL_PATH_CROPS
DEFAULT_CANDIDATES = DUAL_PATH_CANDIDATES
# Shared-pool inventory per (shape, slot). Not one-sign quota (80+20).
# ~X-scale; dual_path is rarer/heavier than junction-only so we cap compute.
DEFAULT_MAX_PER_SLOT = 0  # 0 = no cap; harvest all of P


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


def _existing_scene_ids(out_root: Path) -> Set[str]:
    ids: Set[str] = set()
    if not out_root.is_dir():
        return ids
    for shape_dir in out_root.iterdir():
        if not shape_dir.is_dir() or shape_dir.name not in ("T", "X"):
            continue
        for slot_dir in shape_dir.iterdir():
            if not slot_dir.is_dir():
                continue
            for scene_dir in slot_dir.iterdir():
                if scene_dir.is_dir() and (scene_dir / "map.net.xml").is_file():
                    ids.add(scene_dir.name)
    return ids


def _seed_candidates_from_disk(out_root: Path, cand_f) -> int:
    """Write candidate rows for already-cropped scenes (resume completeness)."""
    n = 0
    if not out_root.is_dir():
        return 0
    for shape_dir in sorted(out_root.iterdir()):
        if not shape_dir.is_dir() or shape_dir.name not in ("T", "X"):
            continue
        for slot_dir in sorted(shape_dir.iterdir()):
            if not slot_dir.is_dir():
                continue
            for scene_dir in sorted(slot_dir.iterdir()):
                meta_path = scene_dir / "meta.json"
                if not (scene_dir / "map.net.xml").is_file() or not meta_path.is_file():
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                dp = meta.get("dual_path") or {}
                rec = {
                    "junction_id": meta.get("junction_id"),
                    "shape": shape_dir.name,
                    "slot": meta.get("slot") or slot_dir.name,
                    "ego_edge_id": meta.get("road_id"),
                    "dest_edge_id": meta.get("destination_edge_id"),
                    "baseline_dir": meta.get("baseline_dir") or dp.get("baseline_dir"),
                    "compliant_dir": meta.get("compliant_dir") or dp.get("compliant_dir"),
                    "gain_m": dp.get("gain_m") or meta.get("dual_path_gain_m"),
                    "ego_is_t_stem": meta.get("ego_is_t_stem"),
                    "carriageway_pair": meta.get("carriageway_pair"),
                    "scene_id": meta.get("scene_id") or scene_dir.name,
                }
                cand_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    if n:
        cand_f.flush()
    return n


def _candidate_record(row: dict, sc, shape: str) -> dict:
    return {
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
            "Max scenes per (shape, slot); 0 = no cap (harvest all of P). "
            "Not one-sign 80+20 quota — many signs reuse the same maps."
        ),
    )
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed for junction order")
    ap.add_argument("--min-gain-m", type=float, default=20.0)
    ap.add_argument("--min-lane-m", type=float, default=8.0)
    ap.add_argument("--margin-m", type=float, default=40.0)
    ap.add_argument("--max-junctions", type=int, default=None, help="Debug: cap junctions scanned")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Resume-safe: count on-disk scenes toward caps and skip re-crops",
    )
    ap.add_argument(
        "--png-only",
        action="store_true",
        help="Write custom_cropped.png for existing crops (no netconvert)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for --png-only (default 8)",
    )
    ap.add_argument(
        "--discover-only",
        action="store_true",
        help="Write candidates JSONL only; do not crop",
    )
    ap.add_argument(
        "--batch-discover",
        action="store_true",
        help="Old mode: discover all atoms first, then crop (not interrupt-safe)",
    )
    args = ap.parse_args()

    if args.png_only:
        from traffic_bench.scene_collection.preview import backfill_previews

        backfill_previews(
            args.out,
            skip_existing=args.skip_existing,
            workers=args.workers,
        )
        return

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
    existing_ids = _existing_scene_ids(args.out) if args.skip_existing else set()
    if already:
        print(f"[dual_path] existing on disk: {dict(sorted(already.items()))}")
        print(f"[dual_path] existing scene_ids: {len(existing_ids)}", flush=True)

    max_per_slot = int(args.max_per_slot) if int(args.max_per_slot) > 0 else 10**9
    cap_note = "unlimited" if int(args.max_per_slot) <= 0 else str(args.max_per_slot)
    print(
        f"[dual_path] net={args.net} junctions={len(rows)} "
        f"slots={list(slots)} max_per_slot={cap_note} "
        f"n_per_junction={args.n_per_junction_slot} seed={args.seed} "
        f"incremental={not args.batch_discover and not args.discover_only}",
        flush=True,
    )

    args.candidates_out.parent.mkdir(parents=True, exist_ok=True)
    # Rewrite candidates; seed from on-disk scenes so resume keeps a full index.
    cand_f = args.candidates_out.open("w", encoding="utf-8")
    seeded = _seed_candidates_from_disk(args.out, cand_f) if args.skip_existing else 0
    if seeded:
        print(f"[dual_path] seeded {seeded} candidates from existing scenes", flush=True)

    stats = {"ok": 0, "fail": 0, "skipped": 0}

    def _write_candidate(row: dict, sc, shape: str) -> None:
        cand_f.write(json.dumps(_candidate_record(row, sc, shape), ensure_ascii=False) + "\n")
        cand_f.flush()

    def on_atom(row: dict, sc, shape: str) -> bool:
        """Crop immediately; count toward pool only if scene ends up on disk."""
        scene_name = sc.scene_id(shape)
        scene_dir = args.out / shape / sc.slot / scene_name
        if args.skip_existing and (scene_dir / "map.net.xml").is_file():
            stats["skipped"] += 1
            _write_candidate(row, sc, shape)
            return True
        if args.discover_only:
            _write_candidate(row, sc, shape)
            return True
        try:
            crop_to_dual_path(
                source_net=args.net,
                scenario=sc,
                output_dir=args.out / shape / sc.slot,
                shape=shape,
                base_row=row,
                margin_m=float(args.margin_m),
            )
            stats["ok"] += 1
            _write_candidate(row, sc, shape)
            if stats["ok"] == 1 or stats["ok"] % 10 == 0:
                print(
                    f"  cropped {stats['ok']} "
                    f"(fail={stats['fail']} skip={stats['skipped']}) "
                    f"last={scene_name}",
                    flush=True,
                )
            return True
        except (JunctionLayoutError, OSError, ValueError) as exc:
            stats["fail"] += 1
            print(f"  FAIL {scene_name}: {exc}", flush=True)
            return False

    t0 = time.time()
    if args.batch_discover or args.discover_only:
        filled = fill_slots_for_junctions(
            args.net,
            junction_rows=rows,
            slots=slots,
            n_per_junction_slot=int(args.n_per_junction_slot),
            max_per_shape_slot=max_per_slot,
            min_gain_m=float(args.min_gain_m),
            min_lane_length_m=float(args.min_lane_m),
            seed=int(args.seed),
            already_filled=already,
            existing_scene_ids=existing_ids,
            on_atom=on_atom if args.discover_only else None,
        )
        print(f"[dual_path] discovered {len(filled)} atoms in {time.time() - t0:.1f}s")
        if args.discover_only:
            cand_f.close()
            print(f"[dual_path] wrote {args.candidates_out}")
            return
        for row, sc in filled:
            shape = str(row.get("shape") or "").upper()
            on_atom(row, sc, shape)
    else:
        filled = fill_slots_for_junctions(
            args.net,
            junction_rows=rows,
            slots=slots,
            n_per_junction_slot=int(args.n_per_junction_slot),
            max_per_shape_slot=max_per_slot,
            min_gain_m=float(args.min_gain_m),
            min_lane_length_m=float(args.min_lane_m),
            seed=int(args.seed),
            already_filled=already,
            existing_scene_ids=existing_ids,
            on_atom=on_atom,
        )
        print(
            f"[dual_path] done in {time.time() - t0:.1f}s: "
            f"kept={len(filled)} ok={stats['ok']} fail={stats['fail']} "
            f"skipped={stats['skipped']}",
            flush=True,
        )

    cand_f.close()
    print(f"[dual_path] wrote {args.candidates_out}", flush=True)
    if args.batch_discover:
        print(
            f"[dual_path] crop stats: ok={stats['ok']} fail={stats['fail']} "
            f"skipped={stats['skipped']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
