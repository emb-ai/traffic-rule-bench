#!/usr/bin/env python3
"""Build cumulative.json in the LEGACY format (compatible with
generate_cumulative_markdown_report.py / generate_chunk_markdown_report.py).

Reads episodes_*.jsonl from runs/var_<i>/chunk_*/baseline/ (and flat layout)
OR consolidated <baseline>_replays.jsonl (sidecar format) — auto-detects.

Output:
  $RUN_ROOT/reports/chunk_<chunk_name>.json    — per-chunk metrics (legacy)
  $RUN_ROOT/reports/cumulative.json            — weighted-merge cumulative

Fields produced (matches aggregate_chunk_metrics.py output):
  per_baseline.<baseline>:
    n, success_rate, dest_rate,
    sign_compliance_sr, traffic_light_sr, crosswalk_sr,
    avg_driving_score, avg_efficiency, avg_smoothness, avg_steps,
    avg_route_length_m, avg_distance_travelled_m
  per_sign.<baseline>.<sign_type>: same fields

Usage:
  python3 aggregate_legacy_format.py \\
      --run-root benchmark_2node_eval \\
      --chunk-name var_0 \\
      --horizon 600
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

NO_ENTRY_SIGNS = {"3.1", "3.2", "3.18.1", "3.18.2", "3.19"}

# Reverse map: internal sign-key (PGMap manifest convention) → PDD code.
# Built from factorized_space.space_definition.PDD_TO_SYNTHETIC + paired bundles.
# Used to normalize per-sign breakdown when row.sign_type carries the internal
# name (PGMap rows) instead of the PDD code (SUMO rows).
INTERNAL_TO_PDD: dict[str, str] = {
    # Priority (2.x)
    "main_road": "2.1",
    "end_main_road": "2.2",
    "secondary_road": "2.3.1",
    "secondary_road_right": "2.3.2",
    "secondary_road_left": "2.3.3",
    "yield": "2.4",
    # Prohibitions / mandatory (3.x, 4.x)
    "no_entry": "3.1",
    "no_traffic": "3.2",
    "no_right_turn": "3.18.1",
    "no_left_turn": "3.18.2",
    "no_uturn": "3.19",
    "no_overtaking": "3.20",
    "no_stop": "3.27",
    "speed20": "3.24", "speed30": "3.24", "speed40": "3.24", "speed60": "3.24",
    "end_speed20": "3.25", "end_speed30": "3.25", "end_speed40": "3.25", "end_speed60": "3.25",
    "end_all": "3.31",
    "detour_right": "4.2.1",
    "detour_left": "4.2.2",
    "detour_either": "4.2.3",
    "minspeed30": "4.6", "minspeed40": "4.6", "minspeed50": "4.6", "minspeed60": "4.6",
    # Informational (5.x)
    "bus_lane_road": "5.11.1",
    "bike_lane_road": "5.11.2",
    "end_bus_lane_road": "5.12.1",
    "end_bike_lane_road": "5.12.2",
    "exit_to_bus_lane": "5.13.1",
    "exit_to_bus_lane_left": "5.13.2",
    "exit_to_bike_lane": "5.13.3",
    "exit_to_bike_lane_left": "5.13.4",
    "bus_lane": "5.14.1",
    "bike_lane": "5.14.2",
    "end_bus_lane": "5.14.3",
    "end_bike_lane": "5.14.4",
    "zone_speed20": "5.31", "zone_speed30": "5.31", "zone_speed40": "5.31", "zone_speed60": "5.31",
    "end_zone_speed20": "5.32", "end_zone_speed30": "5.32", "end_zone_speed40": "5.32", "end_zone_speed60": "5.32",
    # Other
    "stop": "2.5",
    "bus_station": "5.16",
    "only_auto": "4.1.1",
}


def _extract_violations(m: dict) -> dict:
    """Pull per-class step-counts from sidecar metrics.

    Sources, in priority:
      1. metrics.violations_by_class.<bucket>     — 3-bucket per-step counter
         {sign, traffic_light, crosswalk}. Always populated by run_benchmark.
      2. metrics.<bucket>_violations              — older flat field name.
      3. metrics.total_violations                 — last-resort for `sign` only,
         so sign_compliance_sr reflects *some* violation signal.

    NOTE: metrics.violations_by_class_step is intentionally NOT used. Despite
    the similar name, its keys are sign class-names (StopSign, TrafficLightSign,
    MainRoadSign, ...) — incompatible with this 3-bucket lookup. See
    run_benchmark.py:_violation_bucket for the bucket mapping.

    Per-episode binarization (>0 step-count → violated) is done in _agg.
    """
    vbc = m.get("violations_by_class") or {}
    total = int(m.get("total_violations") or 0)

    def _get(cls: str, *aliases: str) -> int:
        if cls in vbc:
            return int(vbc[cls] or 0)
        for a in (cls, *aliases):
            v = m.get(f"{a}_violations")
            if v is not None:
                return int(v or 0)
        return -1  # sentinel: nothing found

    sign_v = _get("sign")
    tl_v = _get("traffic_light", "tl")
    cw_v = _get("crosswalk", "cw")

    if sign_v < 0:
        sign_v = total

    return {
        "sign_violations": max(sign_v, 0),
        "traffic_light_violations": max(tl_v, 0),
        "crosswalk_violations": max(cw_v, 0),
    }


def _normalize_sign_type(s: str) -> str:
    """Convert internal sign key (e.g. 'minspeed40') → PDD code (e.g. '4.6').
    PDD codes pass through unchanged. Unknown keys returned as-is."""
    if not s:
        return s
    s = str(s)
    # PDD codes start with a digit and contain a dot — e.g. "5.11.1".
    if s and s[0].isdigit() and "." in s:
        return s
    return INTERNAL_TO_PDD.get(s, s)


def _sign_from_replay_path(path: str | None) -> str | None:
    """Extract sign slug from sidecar path layout:
       .../replays/<sign>/by_sign/<sign>/by_scene/<scene_uid>/<expert>/replay.json
    Returns the slug verbatim ('2_1', 'minspeed40', ...). Caller normalizes it
    via _normalize_sign_type (which converts '2_1' → '2.1' via dot fix below)."""
    if not path:
        return None
    parts = str(path).split("/")
    try:
        i = parts.index("replays")
    except ValueError:
        return None
    if i + 1 >= len(parts):
        return None
    slug = parts[i + 1]
    # PDD codes are slugified with '_' instead of '.' on disk: '2_1' → '2.1'.
    if slug and slug[0].isdigit() and "_" in slug and not any(c.isalpha() for c in slug):
        return slug.replace("_", ".")
    return slug

# Fields aggregated with weighted mean across chunks.
WEIGHTED_FIELDS = (
    "success_rate", "dest_rate",
    "sign_compliance_sr", "traffic_light_sr", "crosswalk_sr",
    "avg_driving_score", "avg_efficiency", "avg_smoothness", "avg_steps",
    "avg_route_length_m", "avg_distance_travelled_m",
)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _normalize_row(r: dict) -> dict:
    """Normalize a row from EITHER episodes_*.jsonl or sidecar replay.json
    into the flat shape that _agg() expects."""
    # If sidecar shape: top-level + nested `metrics`.
    if "metrics" in r and isinstance(r["metrics"], dict):
        m = r["metrics"]
        # PRIMARY: sign-slug derived from sidecar path (the manifest folder under
        # which writer placed this scene: replays/<sign_slug>/by_sign/...).
        # FALLBACK: row fields, in case _replay_path is missing or malformed.
        sr = r.get("source_row") or {}
        sign_type = (
            _sign_from_replay_path(r.get("_replay_path"))
            or r.get("pdd_code")
            or r.get("sign_key")
            or sr.get("sign_code")
            or sr.get("pdd_code")
            or sr.get("_sign_code")
            or sr.get("sign_type")
            or sr.get("sign_type_start")
            or sr.get("sign_type_end")
            or ""
        )
        # Map sidecar fields → episode-row field names expected by _agg.
        return {
            "ok": True,
            "sign_type": _normalize_sign_type(str(sign_type or "")),
            "reached_dest": bool(m.get("arrived_dest", False)),
            "crashed": bool(m.get("crashed", False)),
            "out_of_road": bool(m.get("out_of_road", False)),
            "steps": int(m.get("final_step") or 0),
            # Per-class violation counters with multi-key fallback (see helper).
            **_extract_violations(m),
            "violations": int(m.get("total_violations") or 0),
            "driving_score": float(m.get("driving_score") or 0.0),
            "driving_efficiency": float(m.get("driving_efficiency") or 0.0),
            "smoothness": float(m.get("smoothness_ratio") or 0.0),
            "route_length_m": float(m.get("route_length_m") or 0.0),
            "distance_travelled_m": float(m.get("distance_travelled_m") or 0.0),
            "in_zone": int(m.get("in_zone_total_steps") or 0) >= 1,
        }
    # Already an episodes-row. Normalize sign_type to PDD code in-place.
    # Paired scenes have sign_type=None but carry sign_type_end / sign_type_start.
    r = dict(r)
    sign_type = (
        r.get("sign_type")
        or r.get("sign_code")
        or r.get("pdd_code")
        or r.get("sign_type_start")    # paired scene → start sign (defines zone rule)
        or r.get("sign_type_end")      # paired scene → end-of-zone sign as fallback
        or ""
    )
    r["sign_type"] = _normalize_sign_type(str(sign_type or ""))
    return r


def _compute_success(r: dict, horizon: int) -> bool:
    """Mirror aggregate_chunk_metrics._compute_success."""
    reached_dest = bool(r.get("reached_dest", False))
    crashed = bool(r.get("crashed", False))
    out_of_road = bool(r.get("out_of_road", False))
    sign = str(r.get("sign_type") or "")
    steps = int(r.get("steps") or 0)
    sign_viol = int(r.get("sign_violations", r.get("violations", 0)) or 0)
    compliance = sign_viol == 0
    if reached_dest and not crashed and not out_of_road:
        return True
    if steps >= horizon and not crashed and not out_of_road:
        return True
    if sign in NO_ENTRY_SIGNS and (not reached_dest) and compliance and not crashed and not out_of_road:
        return True
    return False


def _agg(rows: list[dict], horizon: int) -> dict:
    if not rows:
        return {"n": 0, "n_in_zone": 0, "sign_compliance_x": 0.0}
    n = len(rows)
    success = sum(1 for r in rows if _compute_success(r, horizon))
    dest = sum(1 for r in rows if bool(r.get("reached_dest", False)))
    sign_clean = sum(1 for r in rows
                     if int(r.get("sign_violations", r.get("violations", 0)) or 0) == 0)
    tl_clean = sum(1 for r in rows
                   if int(r.get("traffic_light_violations", 0) or 0) == 0)
    cw_clean = sum(1 for r in rows
                   if int(r.get("crosswalk_violations", 0) or 0) == 0)
    # In-zone subset: episodes where ego actually entered the sign's zone
    # (in_zone bool is True iff in_zone_total_steps >= 1 per sidecar).
    n_in_zone = sum(1 for r in rows if r.get("in_zone"))
    sign_clean_x = sum(1 for r in rows
                       if r.get("in_zone")
                       and int(r.get("sign_violations", r.get("violations", 0)) or 0) == 0)
    sign_compliance_x = (sign_clean_x / n_in_zone) if n_in_zone > 0 else 0.0
    return {
        "n": n,
        "success_rate": success / n,
        "dest_rate": dest / n,
        "sign_compliance_sr": sign_clean / n,
        "traffic_light_sr": tl_clean / n,
        "crosswalk_sr": cw_clean / n,
        "n_in_zone": n_in_zone,
        "sign_compliance_x": sign_compliance_x,
        "avg_driving_score": sum(float(r.get("driving_score", 0.0) or 0.0) for r in rows) / n,
        "avg_efficiency": sum(float(r.get("driving_efficiency", 0.0) or 0.0) for r in rows) / n,
        "avg_smoothness": sum(float(r.get("smoothness", 0.0) or 0.0) for r in rows) / n,
        "avg_steps": sum(int(r.get("steps", 0) or 0) for r in rows) / n,
        "avg_route_length_m": sum(float(r.get("route_length_m", 0.0) or 0.0) for r in rows) / n,
        "avg_distance_travelled_m": sum(float(r.get("distance_travelled_m", 0.0) or 0.0) for r in rows) / n,
    }


def _gather_rows_for_baseline(runs_dir: Path, baseline: str) -> list[dict]:
    """Collect all rows for ONE baseline. Priority order (sign_type derives from
    the manifest folder under which the sidecar was placed, so sidecars win):
       1. sidecars:          runs_dir/(chunk_*/)?baseline/replays/**/replay.json
       2. consolidated:      runs_dir/<baseline>_replays.jsonl      (sidecar shape, has _replay_path)
       3. flat episodes:     runs_dir/(chunk_*/)?baseline/episodes_*.jsonl     (fallback only)
    """
    rows: list[dict] = []

    # 1: walk replays/**/replay.json sidecars directly (chunked + flat layouts)
    sidecar_paths: list[Path] = []
    for cd in sorted(runs_dir.glob("chunk_*")):
        bd = cd / baseline
        if bd.is_dir():
            sidecar_paths.extend(bd.rglob("replay.json"))
    bd_flat = runs_dir / baseline
    if bd_flat.is_dir():
        sidecar_paths.extend(bd_flat.rglob("replay.json"))
    sidecar_paths = sorted(set(p.resolve() for p in sidecar_paths))

    # Dedup by scene_uid (paired scenes have sidecars in BOTH start- and end-sign
    # folders; we want one row per unique scene). First encountered wins
    # (chunked layout comes first thanks to the sort above).
    seen_uids: set[str] = set()
    for sp in sidecar_paths:
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        uid = str(data.get("scene_uid") or data.get("scene_id") or sp)
        if uid in seen_uids:
            continue
        seen_uids.add(uid)
        data["_replay_path"] = str(sp)
        rows.append(_normalize_row(data))

    if rows:
        return rows

    # 2: fallback to pre-consolidated sidecar jsonl (also has _replay_path)
    consolidated = runs_dir / f"{baseline}_replays.jsonl"
    if consolidated.exists():
        for r in _read_jsonl(consolidated):
            rows.append(_normalize_row(r))
        if rows:
            return rows

    # 3: last-resort fallback to episodes_*.jsonl (no path → may bucket as "")
    ep_files: list[Path] = []
    for cd in sorted(runs_dir.glob("chunk_*")):
        bd = cd / baseline
        if bd.is_dir():
            ep_files.extend(bd.glob("episodes_*.jsonl"))
    if bd_flat.is_dir():
        ep_files.extend(bd_flat.glob("episodes_*.jsonl"))
    ep_files = sorted(set(p.resolve() for p in ep_files))

    for fp in ep_files:
        for r in _read_jsonl(fp):
            if r.get("ok"):
                rows.append(_normalize_row(r))

    return rows


def _discover_baselines(runs_dir: Path) -> list[str]:
    names: set[str] = set()
    for cd in runs_dir.glob("chunk_*"):
        if cd.is_dir():
            for bd in cd.iterdir():
                if bd.is_dir():
                    names.add(bd.name)
    for bd in runs_dir.iterdir():
        if (bd.is_dir() and not bd.name.startswith("chunk_")
                and not bd.name.startswith("_")):
            names.add(bd.name)
    for fp in runs_dir.glob("*_replays.jsonl"):
        names.add(fp.stem.removesuffix("_replays"))
    return sorted(names)


def _merge_weighted(old: dict, new: dict) -> dict:
    n_old = int(old.get("n", 0) or 0)
    n_new = int(new.get("n", 0) or 0)
    n_tot = n_old + n_new
    if n_tot <= 0:
        return old or new
    merged = {"n": n_tot}
    for key in WEIGHTED_FIELDS:
        merged[key] = (
            float(old.get(key, 0.0)) * n_old + float(new.get(key, 0.0)) * n_new
        ) / n_tot
    # n_in_zone: simple sum across chunks.
    nz_old = int(old.get("n_in_zone", 0) or 0)
    nz_new = int(new.get("n_in_zone", 0) or 0)
    nz_tot = nz_old + nz_new
    merged["n_in_zone"] = nz_tot
    # sign_compliance_x: weighted-mean by n_in_zone (rate over in-zone subset).
    if nz_tot > 0:
        merged["sign_compliance_x"] = (
            float(old.get("sign_compliance_x", 0.0)) * nz_old
            + float(new.get("sign_compliance_x", 0.0)) * nz_new
        ) / nz_tot
    else:
        merged["sign_compliance_x"] = 0.0
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True,
                    help="Benchmark root containing runs/<chunk_name>/")
    ap.add_argument("--chunk-name", required=True,
                    help="Chunk name (e.g. var_0). Reads from $RUN_ROOT/runs/<chunk_name>.")
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--cumulative-out", default=None,
                    help="Custom path for cumulative.json. Default: <run-root>/reports/cumulative.json. "
                         "Use to build separate per-var cumulatives.")
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    runs_dir = run_root / "runs" / args.chunk_name
    if not runs_dir.is_dir():
        print(f"ERROR: not a dir: {runs_dir}", file=sys.stderr)
        sys.exit(2)

    baselines = _discover_baselines(runs_dir)
    if not baselines:
        print(f"ERROR: no baselines found under {runs_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"[scan] {runs_dir}: {len(baselines)} baseline(s)")

    rows_by_baseline: dict[str, list[dict]] = {}
    for b in baselines:
        rows = _gather_rows_for_baseline(runs_dir, b)
        rows_by_baseline[b] = rows
        print(f"  [agg] {b}: {len(rows)} ok rows")

    per_baseline = {b: _agg(rs, horizon=args.horizon) for b, rs in sorted(rows_by_baseline.items())}

    # Per-sign breakdown — skip rows with empty sign_type (provenance unknown).
    # Logged separately as "<unknown>" only if non-trivial count exists.
    per_sign: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    n_unknown_per_baseline: dict[str, int] = defaultdict(int)
    for b, rs in rows_by_baseline.items():
        for r in rs:
            sign = str(r.get("sign_type") or "").strip()
            if not sign:
                n_unknown_per_baseline[b] += 1
                continue
            per_sign[b][sign].append(r)
    per_sign_metrics = {
        b: {s: _agg(rs, horizon=args.horizon) for s, rs in sorted(by_s.items())}
        for b, by_s in sorted(per_sign.items())
    }
    # Warn on baselines with high unknown-sign-type counts.
    for b, n in n_unknown_per_baseline.items():
        if n > 0:
            print(f"  [warn] {b}: {n} rows have empty sign_type — excluded from per-sign breakdown")

    out_dir = run_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-chunk legacy report
    chunk_report = out_dir / f"chunk_{args.chunk_name}.json"
    chunk_report.write_text(
        json.dumps({
            "chunk": args.chunk_name,
            "horizon": args.horizon,
            "per_baseline": per_baseline,
            "per_sign": per_sign_metrics,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 2. Cumulative (weighted merge — append-style)
    cumulative_path = (Path(args.cumulative_out).resolve() if args.cumulative_out
                       else out_dir / "cumulative.json")
    cumulative_path.parent.mkdir(parents=True, exist_ok=True)
    cumulative = (json.loads(cumulative_path.read_text(encoding="utf-8"))
                  if cumulative_path.exists()
                  else {"chunks": [], "per_baseline": {}, "per_sign": {}})

    if args.chunk_name not in cumulative["chunks"]:
        cumulative["chunks"].append(args.chunk_name)

    for b, m in per_baseline.items():
        old = cumulative["per_baseline"].get(b)
        cumulative["per_baseline"][b] = _merge_weighted(old or {}, m) if old else m

    for b, by_sign in per_sign_metrics.items():
        if b not in cumulative["per_sign"] or not isinstance(cumulative["per_sign"][b], dict):
            cumulative["per_sign"][b] = {}
        for s, m in by_sign.items():
            old = cumulative["per_sign"][b].get(s)
            cumulative["per_sign"][b][s] = _merge_weighted(old or {}, m) if old else m

    cumulative_path.write_text(json.dumps(cumulative, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"Chunk report: {chunk_report}")
    print(f"Cumulative:   {cumulative_path}")
    print()
    print("Generate markdown reports:")
    print(f"  python3 generate_chunk_markdown_report.py --run-root {run_root} --chunk-name {args.chunk_name}")
    print(f"  python3 generate_cumulative_markdown_report.py --run-root {run_root}")
    print(f"  python3 generate_category_aggregation_report.py --run-root {run_root}")


if __name__ == "__main__":
    main()
