#!/usr/bin/env python3
"""filter_scenes_ego_on_sign_lane.py — for the given signs, walk over the
manifests, dry-init each scene, and keep only those where ego spawns on the
lane that carries the sign.

Output: <out>/<preset>/<sign_slug>/<manifest>.jsonl — layout compatible with
        run_benchmark_mini.py --benchmark-output <out> --preset <preset>.

Usage:
  python3 filter_scenes_ego_on_sign_lane.py \
    --src /path/to/full_test_250_x10 \
    --signs 5.11.1 5.11.2 5.13.1 5.13.2 5.13.3 5.13.4 5.14.1 5.14.2 4.2.1 4.2.2 4.2.3 \
    --scenes-root /path/to/pdd-bench/scenes \
    --out /path/to/pdd-bench/scripts/per_sign_bench/benchmark_filtered \
    --preset mini \
    --backends sumo,pgmap,paired,citymap \
    --report /tmp/filter_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_benchmark_mini import (    # noqa: E402
    _build_pgmap_env, _build_sumo_env,
    _place_pgmap_sign, _unwrap_base_env,
    _choose_manifest, _load_jsonl_rows, _slug_to_code,
)


MANIFEST_NAMES = {
    "pgmap":   "pgmap_materialized.jsonl",
    "paired":  "paired_materialized.jsonl",
    "citymap": "citymap_materialized.jsonl",
}


def lane_index_to_tuple(lane):
    if lane is None: return None
    idx = getattr(lane, "index", None)
    if idx is None: return None
    if isinstance(idx, (list, tuple)):
        return tuple(idx)
    return idx


def get_sign_lanes(env):
    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return set()
    lanes = set()
    for s in getattr(sign_mgr, "spawned_objects", {}).values():
        idx = lane_index_to_tuple(getattr(s, "lane", None))
        if idx is not None:
            lanes.add(idx)
    return lanes


def is_ego_on_sign_lane(row: dict, backend: str, scenes_root: Path) -> tuple[bool, str | None]:
    """Returns (ego_on_sign_lane, error_str_or_None)."""
    seed = int(row.get("seed") or row.get("deterministic_seed") or 0)
    try:
        if backend in ("pgmap", "paired", "citymap"):
            env = _build_pgmap_env(row, max_steps=1)
        elif backend == "sumo":
            env = _build_sumo_env(row, scenes_root=scenes_root, max_steps=1)
        else:
            return False, f"unsupported backend: {backend}"
    except Exception as exc:
        return False, f"build_env: {type(exc).__name__}: {exc}"

    try:
        if backend == "sumo":
            env_seed = (int(row.get("sign_id", 0)) + int(row.get("var_idx", 0))) % 100000
        elif backend == "citymap":
            env_seed = int(row.get("map_seed") or row.get("seed") or 0)
        else:
            env_seed = seed
        env.reset(seed=env_seed)
        base_env = _unwrap_base_env(env)

        if backend in ("pgmap", "paired", "citymap"):
            ok = _place_pgmap_sign(base_env, row, seed)
            if not ok:
                return False, "place_sign_failed"

        ego = base_env.agent
        ego_lane = lane_index_to_tuple(ego.lane) if ego and ego.lane else None
        sign_lanes = get_sign_lanes(base_env)
        if ego_lane is None or not sign_lanes:
            return False, "no_lane_info"
        def _lane_num(idx):
            if isinstance(idx, (list, tuple)) and idx:
                return idx[-1]
            return idx
        ego_num = _lane_num(ego_lane)
        sign_nums = {_lane_num(s) for s in sign_lanes}
        return (ego_num in sign_nums), None
    except Exception as exc:
        return False, f"reset/place: {type(exc).__name__}: {exc}"
    finally:
        try:
            env.close()
        except Exception:
            pass


def find_sumo_manifest(sign_dir: Path) -> Path | None:
    p1 = sign_dir / "sumo" / "sumo_manifest.jsonl"
    p2 = sign_dir / "real_manifest.jsonl"
    if p1.exists() and p1.stat().st_size > 0: return p1
    if p2.exists() and p2.stat().st_size > 0: return p2
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True,
                   help="full_test_250_x10 with sign subdirs (2_1/, 4_2_1/, …)")
    p.add_argument("--signs", nargs="+", required=True,
                   help="Sign codes (dotted form): 5.11.1 4.2.1 ...")
    p.add_argument("--scenes-root", required=True,
                   help="Directory with net_path files for sumo (pdd-bench/scenes)")
    p.add_argument("--out", required=True,
                   help="Output dir — <out>/<preset>/<sign>/... will be created")
    p.add_argument("--preset", default="mini")
    p.add_argument("--backends", default="sumo,pgmap,paired,citymap")
    p.add_argument("--report", default=None,
                   help="Where to write the summary JSON report (optional)")
    p.add_argument("--all-kept-json", default=None,
                   help="Where to write JSON listing ALL kept scenes (full "
                        "manifest rows + sign_code + backend). Not written by "
                        "default. Example: /tmp/all_kept_scenes.json")
    p.add_argument("--limit-per-sign", type=int, default=None,
                   help="Cap the number of processed rows per (sign, backend)")
    p.add_argument("--early-stop-per-scene-id", action="store_true", default=True,
                   help="Once a variant with ego_on_sign_lane=True is found for a "
                        "scene_id, skip the remaining variants of that scene "
                        "(default: True). Disable with --no-early-stop")
    p.add_argument("--no-early-stop", dest="early_stop_per_scene_id",
                   action="store_false")
    args = p.parse_args()

    logging.getLogger().setLevel(logging.CRITICAL)

    src = Path(args.src).resolve()
    if not src.is_dir():
        raise SystemExit(f"src not found: {src}")
    scenes_root = Path(args.scenes_root).resolve()
    out_root = Path(args.out) / args.preset
    out_root.mkdir(parents=True, exist_ok=True)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    target_signs = set(args.signs)

    stats = defaultdict(lambda: defaultdict(lambda: {
        "in": 0, "kept": 0, "dropped": 0, "errors": 0,
        "skipped_by_early_stop": 0,
        "error_types": defaultdict(int),
    }))
    all_kept_global: list = []

    sign_dirs = sorted([d for d in src.iterdir()
                        if d.is_dir() and d.name[:1].isdigit()])
    sign_dirs = [d for d in sign_dirs if _slug_to_code(d.name) in target_signs]
    print(f"Found {len(sign_dirs)} sign dirs (out of {len(target_signs)} requested)",
          file=sys.stderr)

    for sign_dir in sign_dirs:
        sign_slug = sign_dir.name
        sign_code = _slug_to_code(sign_slug)
        out_sign_dir = out_root / sign_slug
        out_sign_dir.mkdir(parents=True, exist_ok=True)

        for backend in backends:
            if backend == "sumo":
                manifest = find_sumo_manifest(sign_dir)
                out_name = manifest.name if manifest else None
                if manifest and "sumo" in manifest.parts:
                    out_path = out_sign_dir / "sumo" / manifest.name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    out_path = out_sign_dir / (out_name or "real_manifest.jsonl")
            else:
                fname = MANIFEST_NAMES[backend]
                manifest = sign_dir / fname
                out_path = out_sign_dir / fname

            if manifest is None or not manifest.exists() or manifest.stat().st_size == 0:
                continue

            rows = _load_jsonl_rows(manifest)
            if args.limit_per_sign:
                rows = rows[: args.limit_per_sign]

            s = stats[sign_code][backend]
            s["in"] = len(rows)

            kept_rows = []
            covered_scene_ids: set = set()
            n_skipped_by_early_stop = 0
            for i, row in enumerate(rows):
                if "valid" in row and not row["valid"]:
                    continue
                if backend in ("pgmap", "paired", "citymap") and not row.get("sign_type"):
                    row["sign_type"] = sign_code

                sid = row.get("scene_id")
                if (args.early_stop_per_scene_id and sid is not None
                        and sid in covered_scene_ids):
                    n_skipped_by_early_stop += 1
                    continue

                on_lane, err = is_ego_on_sign_lane(row, backend, scenes_root)
                if err:
                    s["errors"] += 1
                    s["error_types"][err.split(":")[0]] += 1
                    continue
                if on_lane:
                    s["kept"] += 1
                    kept_rows.append(row)
                    if sid is not None:
                        covered_scene_ids.add(sid)
                    all_kept_global.append({
                        **row,
                        "_sign_code": sign_code,
                        "_sign_slug": sign_slug,
                        "_backend":   backend,
                    })
                else:
                    s["dropped"] += 1

                if (i + 1) % 25 == 0:
                    print(f"  [{sign_code}/{backend}] {i+1}/{len(rows)}  "
                          f"kept={s['kept']} dropped={s['dropped']} err={s['errors']} "
                          f"skipped(early)={n_skipped_by_early_stop}",
                          file=sys.stderr)
            s["skipped_by_early_stop"] = n_skipped_by_early_stop

            if kept_rows:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    for r in kept_rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"[ok] {sign_code:<10} {backend:<8} "
                      f"kept={s['kept']:>4}/{s['in']:<4} → {out_path}",
                      file=sys.stderr)
            else:
                print(f"[empty] {sign_code:<10} {backend:<8} "
                      f"0/{s['in']} kept (none on sign lane)",
                      file=sys.stderr)

    print(f"\n{'sign':<10} {'backend':<8} {'in':>5} {'kept':>5} {'drop':>5} {'err':>4} {'skip':>5} {'%kept':>6}",
          file=sys.stderr)
    total_in = total_kept = total_drop = total_err = total_skip = 0
    report = {}
    for sign in sorted(stats):
        report[sign] = {}
        for backend, s in sorted(stats[sign].items()):
            n_in, n_keep, n_drop, n_err = s["in"], s["kept"], s["dropped"], s["errors"]
            n_skip = s.get("skipped_by_early_stop", 0)
            pct = (100 * n_keep / n_in) if n_in else 0
            print(f"{sign:<10} {backend:<8} {n_in:>5} {n_keep:>5} {n_drop:>5} {n_err:>4} {n_skip:>5} {pct:>5.1f}%",
                  file=sys.stderr)
            total_in += n_in; total_kept += n_keep
            total_drop += n_drop; total_err += n_err
            total_skip += n_skip
            report[sign][backend] = {
                "in": n_in, "kept": n_keep, "dropped": n_drop, "errors": n_err,
                "skipped_by_early_stop": n_skip,
                "error_types": dict(s["error_types"]),
            }
    pct = (100 * total_kept / total_in) if total_in else 0
    print(f"{'TOTAL':<10} {'':<8} {total_in:>5} {total_kept:>5} {total_drop:>5} {total_err:>4} {total_skip:>5} {pct:>5.1f}%",
          file=sys.stderr)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nReport: {args.report}", file=sys.stderr)

    if args.all_kept_json:
        Path(args.all_kept_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.all_kept_json).write_text(
            json.dumps(all_kept_global, indent=2, ensure_ascii=False, default=str))
        print(f"All kept scenes ({len(all_kept_global)}): {args.all_kept_json}",
              file=sys.stderr)

    print(f"\nFiltered manifests in: {out_root}", file=sys.stderr)


if __name__ == "__main__":
    main()
