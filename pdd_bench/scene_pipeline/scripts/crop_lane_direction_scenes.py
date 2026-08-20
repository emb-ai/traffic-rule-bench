#!/usr/bin/env python3
"""Discover + crop lane-change dual-path scenes for PDD 5.15.1.

Shared pool under::

    scenes/lane_direction/{T,X}/<scene_id>/

Each atom: multi-lane approach where a peer lane has an exclusive L/R exit the
spawn lane lacks. Meta ``dual_path.kind=lane_change``; preview draws
wrong (orange) vs correct (blue + LC) paths like the old lane_direction harvest.

Sign-free harvest; allocate later with ``crop_kind: lane_direction`` for 5.15.1.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]

from pdd_bench.bench.core.layout.junction_priority_layout import JunctionLayoutError
from pdd_bench.scene_pipeline.lib.lane_direction import (
    DualPathScenario,
    build_edge_graph,
    crop_scene_to_dual_path_scenario,
    dual_path_scenario_from_meta,
    find_dual_path_scenarios,
)

DEFAULT_NET = ROOT / "nets" / "moscow.net.xml"
DEFAULT_INDEX = ROOT / "index" / "junctions.jsonl"
DEFAULT_OUT = ROOT / "scenes" / "lane_direction"
DEFAULT_CANDIDATES = ROOT / "index" / "lane_direction_candidates.jsonl"
DEFAULT_MAX_PER_SHAPE = 500


def _load_index(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _count_existing(out_root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    if not out_root.is_dir():
        return counts
    for shape_dir in out_root.iterdir():
        if not shape_dir.is_dir() or shape_dir.name not in ("T", "X"):
            continue
        n = sum(1 for p in shape_dir.iterdir() if (p / "map.net.xml").is_file())
        if n:
            counts[shape_dir.name] = n
    return dict(counts)


def _existing_scene_ids(out_root: Path) -> Set[str]:
    ids: Set[str] = set()
    if not out_root.is_dir():
        return ids
    for shape_dir in out_root.iterdir():
        if not shape_dir.is_dir() or shape_dir.name not in ("T", "X"):
            continue
        for scene_dir in shape_dir.iterdir():
            if scene_dir.is_dir() and (scene_dir / "map.net.xml").is_file():
                ids.add(scene_dir.name)
    return ids


def _render_preview(scene_dir: Path, scenario: DualPathScenario) -> None:
    """Wrong (red/baseline) vs correct (green/compliant) path overlay."""
    try:
        from tools.render_map import parse_sumo_net, render_network
    except ImportError as exc:
        print(f"  [preview] skip (render_map import failed: {exc})")
        return

    net_path = scene_dir / "map.net.xml"
    if not net_path.is_file():
        return
    edges, junctions = parse_sumo_net(net_path)
    # Wrong: stay on spawn lane's natural exit spur.
    baseline = [scenario.ego_edge_id, *list(scenario.turn_path)]
    # Correct: after LC onto target lane, exclusive turn → dest.
    compliant = [scenario.ego_edge_id, *list(scenario.straight_path)]
    out_png = scene_dir / "custom_cropped.png"
    render_network(
        edges,
        junctions,
        out_png,
        figsize=(6, 6),
        dpi=120,
        marker_xy=scenario.junction_center_xy,
        baseline_edge_ids=baseline,
        compliant_edge_ids=compliant,
        legend=True,
    )
    # Annotate spawn/target lanes in the legend title via a sidecar note in meta
    # is enough for review; edge overlay matches dual_path reroute style.


def _candidate_record(row: dict, sc: DualPathScenario, shape: str) -> dict:
    return {
        "junction_id": sc.junction_id,
        "shape": shape,
        "ego_edge_id": sc.ego_edge_id,
        "dest_edge_id": sc.dest_edge_id,
        "spawn_lane_num": sc.ego_lane_num,
        "target_lane_num": sc.target_lane_num,
        "baseline_dir": sc.turn_dir,
        "compliant_dir": sc.compliant_dir,
        "gain_m": sc.gain_m,
        "scene_id": sc.scene_id(shape),
        "crop_kind": "lane_direction",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", type=Path, default=DEFAULT_NET)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--candidates-out", type=Path, default=DEFAULT_CANDIDATES)
    ap.add_argument("--shapes", default="T,X")
    ap.add_argument(
        "--max-per-shape",
        type=int,
        default=DEFAULT_MAX_PER_SHAPE,
        help=f"Max scenes per shape (default {DEFAULT_MAX_PER_SHAPE})",
    )
    ap.add_argument("--n-per-junction", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-lane-m", type=float, default=21.0)
    ap.add_argument("--min-arms", type=int, default=2)
    ap.add_argument("--margin-m", type=float, default=40.0)
    ap.add_argument("--max-junctions", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--discover-only", action="store_true")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    shapes = {s.strip().upper() for s in args.shapes.split(",") if s.strip()}
    if not args.net.is_file():
        raise SystemExit(f"City net not found: {args.net}")
    if not args.index.is_file():
        raise SystemExit(f"Junction index not found: {args.index}")

    rows = [r for r in _load_index(args.index) if str(r.get("shape") or "").upper() in shapes]
    rng = random.Random(int(args.seed))
    rng.shuffle(rows)
    if args.max_junctions is not None:
        rows = rows[: max(0, int(args.max_junctions))]

    already = _count_existing(args.out) if args.skip_existing else {}
    existing_ids = _existing_scene_ids(args.out) if args.skip_existing else set()
    filled: Dict[str, int] = {s: int(already.get(s, 0)) for s in shapes}

    print(
        f"[lane_direction] net={args.net} junctions={len(rows)} "
        f"shapes={sorted(shapes)} max_per_shape={args.max_per_shape} "
        f"n_per_junc={args.n_per_junction} seed={args.seed} "
        f"existing={dict(sorted(filled.items()))}",
        flush=True,
    )

    print("[lane_direction] building city edge graph…", flush=True)
    t_graph = time.time()
    graph = build_edge_graph(args.net)
    print(f"[lane_direction] graph ready in {time.time() - t_graph:.1f}s", flush=True)

    args.candidates_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    cand_f = args.candidates_out.open("w", encoding="utf-8")

    stats = {"ok": 0, "fail": 0, "skipped": 0, "no_scenario": 0}
    t0 = time.time()

    for row in rows:
        if all(filled.get(s, 0) >= int(args.max_per_shape) for s in shapes):
            break
        shape = str(row.get("shape") or "").upper()
        if shape not in shapes or filled.get(shape, 0) >= int(args.max_per_shape):
            continue
        jid = str(row.get("junction_id") or "").strip()
        if not jid:
            continue

        try:
            scenarios = find_dual_path_scenarios(
                args.net,
                junction_ids=[jid],
                min_lane_length_m=float(args.min_lane_m),
                min_arms=int(args.min_arms),
                max_scenarios=max(1, int(args.n_per_junction)),
                dests_per_arm=1,
                graph=graph,
            )
        except Exception as exc:
            stats["fail"] += 1
            print(f"  FAIL discover {jid}: {exc}", flush=True)
            continue

        if not scenarios:
            stats["no_scenario"] += 1
            continue

        for sc in scenarios[: max(1, int(args.n_per_junction))]:
            if filled.get(shape, 0) >= int(args.max_per_shape):
                break
            scene_name = sc.scene_id(shape)
            scene_dir = args.out / shape / scene_name
            if args.skip_existing and (scene_dir / "map.net.xml").is_file():
                stats["skipped"] += 1
                filled[shape] = filled.get(shape, 0) + 1
                cand_f.write(
                    json.dumps(_candidate_record(row, sc, shape), ensure_ascii=False) + "\n"
                )
                cand_f.flush()
                png = scene_dir / "custom_cropped.png"
                if not args.no_preview and not png.is_file():
                    try:
                        cropped = dual_path_scenario_from_meta(
                            json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
                        )
                        if cropped is not None:
                            _render_preview(scene_dir, cropped)
                    except Exception as exc:
                        print(f"  [preview] {scene_name}: {exc}", flush=True)
                continue
            if scene_name in existing_ids:
                stats["skipped"] += 1
                continue

            if args.discover_only:
                cand_f.write(
                    json.dumps(_candidate_record(row, sc, shape), ensure_ascii=False) + "\n"
                )
                cand_f.flush()
                filled[shape] = filled.get(shape, 0) + 1
                stats["ok"] += 1
                continue

            try:
                cropped = crop_scene_to_dual_path_scenario(
                    sc,
                    source_net=args.net,
                    output_dir=args.out / shape,
                    shape=shape,
                    margin_m=float(args.margin_m),
                    base_row=row,
                )
                if not args.no_preview:
                    _render_preview(args.out / shape / cropped.scene_id(shape), cropped)
                stats["ok"] += 1
                filled[shape] = filled.get(shape, 0) + 1
                existing_ids.add(cropped.scene_id(shape))
                cand_f.write(
                    json.dumps(
                        _candidate_record(row, cropped, shape), ensure_ascii=False
                    )
                    + "\n"
                )
                cand_f.flush()
                if stats["ok"] == 1 or stats["ok"] % 10 == 0:
                    print(
                        f"  cropped {stats['ok']} "
                        f"(fail={stats['fail']} skip={stats['skipped']} "
                        f"empty={stats['no_scenario']}) "
                        f"last={cropped.scene_id(shape)}",
                        flush=True,
                    )
            except (JunctionLayoutError, OSError, ValueError) as exc:
                stats["fail"] += 1
                print(f"  FAIL {scene_name}: {exc}", flush=True)

    cand_f.close()
    print(
        f"[lane_direction] done in {time.time() - t0:.1f}s: "
        f"ok={stats['ok']} fail={stats['fail']} skipped={stats['skipped']} "
        f"no_scenario={stats['no_scenario']} filled={dict(sorted(filled.items()))}",
        flush=True,
    )
    print(f"[lane_direction] wrote {args.candidates_out}", flush=True)


if __name__ == "__main__":
    main()
