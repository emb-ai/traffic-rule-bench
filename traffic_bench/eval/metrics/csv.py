#!/usr/bin/env python3
"""Build a single episode-level CSV — the source of truth for downstream aggregations.

Two input modes (mutually exclusive):

  --episodes-root  (PRIMARY)  <policy_eval>/<run_name>/episodes_*.jsonl
      Built straight from the per-episode JSONL that eval run ALWAYS writes.
      No replay.json sidecar needed.

  --runs-root      (legacy)   <runs-root>/var_<i>/<baseline>_replays.jsonl
      Reads consolidated replays / replay.json sidecars.

Either way, one CSV row per episode carries all `metrics` fields plus precomputed
flags (target compliance, recomputed dest, passes_filter) needed by
  generate_cumulative_markdown_report.py
  expert_selection_exps/select_from_replays.py

Usage:
  python -m traffic_bench.eval metrics csv \
      --episodes-root <out>/benchmark/policy_eval \
      --out           <out>/metrics_per_episode.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ORACLE_DIR = SCRIPT_DIR.parent.parent / "oracle"

from traffic_bench.oracle.select_experts import (
    SIGN_CLASS_MAP,
    NO_ENTRY_SIGNS,
    HORIZON_DEFAULT,
    MIN_FINAL_STEP,
    MIN_ROUTE_COMPLETION,
    normalize_sign,
    recompute_dest,
    passes_filter,
)


POLICY_DISPLAY_NAME: dict[str, str] = {
    "comprehensive_rule_expert_default": "idm_rule_default",
    "comprehensive_rule_expert_s1": "idm_rule_s1",
    "comprehensive_rule_expert_s2": "idm_rule_s2",
    "comprehensive_rule_expert_s3": "idm_rule_s3",
    "comprehensive_rule_expert_s4": "idm_rule_s4",
    "ppo_expert": "ppo",
    "rule_compliant": "ppo_rule",
}


# Speed-sign subclasses registered in run_benchmark_mini_speed.py.
# SIGN_CLASS_MAP maps PDD → BASE class, but episode metrics record violations
# and in_zone steps under the SUBCLASS name (e.g. "MinimumSpeedLimit20"
# instead of "MinimumSpeedLimitSign"). Without this mapping all speed runs
# count target_in_zone=0 and target_compliant=True.
#
# Priority plates 2.1 / 2.3.x are informational and never violate; SIGN_CLASS_MAP
# already points those PDDs at RightHandYieldSign / YieldSign (the classes that
# own the approach zone and emit violations).
TARGET_CLASS_SUBCLASSES: dict[str, list[str]] = {
    "SpeedLimitSign":         ["SpeedLimitSign15"],
    "EndOfSpeedLimitSign":    ["EndOfSpeedLimitSign15"],
    "ZoneSpeedLimitSign":     ["ZoneSpeedLimitSign15"],
    "EndOfZoneSpeedLimitSign":["EndOfZoneSpeedLimitSign15"],
    "MinimumSpeedLimitSign":  ["MinimumSpeedLimit20"],
}


def _resolve_target_classes(base_class: str | None) -> list[str]:
    """Return [base_class] plus any registered subclasses (for speed signs)."""
    if not base_class:
        return []
    out = [base_class]
    out.extend(TARGET_CLASS_SUBCLASSES.get(base_class, []))
    return out


def _sum_class_keys(d: dict, classes: list[str]) -> int:
    """Sum integer values from d for any of the given class keys (skip missing)."""
    total = 0
    for c in classes:
        v = d.get(c)
        if v is None:
            continue
        try:
            total += int(v)
        except (TypeError, ValueError):
            continue
    return total


VAR_DIR_RE = re.compile(r"^var_(\d+)$")


def _to_int(v, default=0) -> int:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _to_float(v, default=None):
    try:
        if v is None:
            return default
        f = float(v)
        return f
    except (TypeError, ValueError):
        return default


def _bool(v) -> bool:
    return bool(v)


def _ensure_dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _episode_to_replay(ep: dict) -> dict:
    """Normalize a run_benchmark `episodes_*.jsonl` row into the replay-shaped dict
    that `_build_row` consumes.

    This is what lets the metrics CSV be built from `episodes_*.jsonl` directly,
    with NO replay.json sidecar needed. The episode row is flat (run_benchmark
    schema); the sidecar nests metrics under `metrics`. The renames below mirror
    run_benchmark's `sidecar_metrics` block one-to-one.
    """
    rc_pct = ep.get("route_completion_pct")
    metrics = {
        "arrived_dest": bool(ep.get("reached_dest")),
        # sidecar `metrics.crashed` is info["crash"] alone (without OOR);
        # run_benchmark exposes that as `crashed_raw`.
        "crashed": bool(ep.get("crashed_raw", ep.get("crashed"))),
        "crashed_ego_fault": bool(ep.get("crashed_ego_fault")),
        "crashed_npc_fault": bool(ep.get("crashed_npc_fault")),
        "out_of_road": bool(ep.get("out_of_road")),
        "success": bool(ep.get("success")),
        "final_step": ep.get("steps"),
        "total_reward": ep.get("total_reward"),
        "route_completion": (float(rc_pct) / 100.0) if rc_pct is not None else None,
        "route_length_m": ep.get("route_length_m"),
        "distance_travelled_m": ep.get("distance_travelled_m"),
        "driving_score": ep.get("driving_score"),
        "driving_efficiency": ep.get("driving_efficiency"),
        "infraction_penalty": ep.get("infraction_penalty"),
        "smoothness_ratio": ep.get("smoothness"),
        "frame_smooth_ratio": ep.get("smoothness_frame_ratio"),
        "smooth_segments": ep.get("smooth_segments"),
        "total_segments": ep.get("smooth_total_segments"),
        "min_ttc_sec": ep.get("min_ttc_sec"),
        "mean_abs_lane_offset": ep.get("mean_abs_lane_offset"),
        "mean_abs_steer_delta": ep.get("mean_abs_steer_delta"),
        "hard_brake_count": ep.get("hard_brake_count"),
        "hard_accel_count": ep.get("hard_accel_count"),
        "total_violations": ep.get("violations"),
        "violations_by_class": {
            "sign": ep.get("sign_violations", 0),
            "traffic_light": ep.get("traffic_light_violations", 0),
            "crosswalk": ep.get("crosswalk_violations", 0),
        },
        "violations_by_class_step": ep.get("violations_by_class_step") or {},
        "violations_by_class_event": ep.get("violations_by_class_event") or {},
        "in_zone_total_steps": ep.get("in_zone_total_steps", 0),
        "in_zone_by_class_step": ep.get("in_zone_by_class_step") or {},
        "violations_event_count": ep.get("violations_event_count", 0),
    }
    pdd = ep.get("sign_type") or ep.get("pdd_code") or ""
    return {
        "scene_id": ep.get("scene_id"),
        "scene_uid": ep.get("scene_uid"),
        "backend": ep.get("backend") or "",
        "pdd_code": pdd,
        "sign_slug": ep.get("sign_slug") or (str(pdd).replace(".", "_") if pdd else ""),
        "policy": ep.get("policy") or "",
        "variant": ep.get("variant") or "",
        "valid": bool(ep.get("ok", True)),
        "source_row": {},
        "metrics": metrics,
    }


def _build_row(replay: dict, var_name: str, var_idx: int, baseline: str,
                manifest_lookup: dict[tuple[int, str], dict] | None = None) -> dict | None:
    """Convert one consolidated replay JSON object into a flat CSV row dict.

    If `manifest_lookup` is provided, enriches the row with manifest-derived
    fields for paired-zone scenes: `manifest_source`, `is_paired_scene`,
    `pdd_code_start`, `pdd_code_end`, `pdd_code_target`, `sign_type_start`,
    `sign_type_end`, `zone_length_m`. These are used by the aggregator to
    correctly group paired scenes by their (start, end) pair.

    Returns None if essential fields are missing (cannot identify the episode).
    """
    metrics = _ensure_dict(replay.get("metrics"))
    pdd_code_raw = replay.get("pdd_code") or replay.get("sign_slug") or ""
    pdd_code = normalize_sign(str(pdd_code_raw)) if pdd_code_raw else ""
    sign_slug = (replay.get("sign_slug")
                 or (pdd_code.replace(".", "_") if pdd_code else ""))
    if not pdd_code:
        # Fallback to source_row
        sr = _ensure_dict(replay.get("source_row"))
        pdd_code = normalize_sign(str(sr.get("sign_code") or sr.get("pdd_code") or "")) or ""
        if pdd_code and not sign_slug:
            sign_slug = pdd_code.replace(".", "_")

    scene_id = replay.get("scene_id")
    scene_uid = replay.get("scene_uid")
    if not scene_id or not scene_uid:
        return None

    # Manifest lookup — paired-scene fields come from chunks/var_<i>.jsonl which
    # carries pdd_code_start / pdd_code_end / source. Match by (var_idx, scene_id).
    manifest_source = ""
    is_paired_scene = False
    pdd_code_start = ""
    pdd_code_end = ""
    pdd_code_target = pdd_code    # default — replay's own code
    sign_type_start = ""
    sign_type_end = ""
    zone_length_m = ""
    if manifest_lookup is not None:
        mr = manifest_lookup.get((var_idx, scene_id))
        if mr:
            manifest_source = str(mr.get("source") or "")
            is_paired_scene = ("paired" in manifest_source.lower())
            if is_paired_scene:
                pdd_code_start = str(mr.get("pdd_code_start") or "")
                pdd_code_end = str(mr.get("pdd_code_end") or "")
                # rewrite_speed_manifests.py drops `pdd_code` from paired rows
                # and rarely sets `pdd_code_target` — fall back to `pdd_code_start`
                # so target_class lookup works downstream.
                pdd_code_target = str(mr.get("pdd_code_target") or pdd_code_start or pdd_code)
                sign_type_start = str(mr.get("sign_type_start") or "")
                sign_type_end = str(mr.get("sign_type_end") or "")
                zl = mr.get("zone_length_m")
                zone_length_m = "" if zl is None else float(zl)

    # For paired scenes pdd_code is empty; pdd_code_target carries the target
    # (start) sign. For non-paired scenes pdd_code_target == pdd_code.
    target_pdd = pdd_code_target or pdd_code
    target_class = SIGN_CLASS_MAP.get(target_pdd) if target_pdd else None
    is_no_entry = bool(target_pdd in NO_ENTRY_SIGNS)

    # High-level violations dict in replay.json: keys are {"sign","traffic_light","crosswalk"}
    vbc_high = _ensure_dict(metrics.get("violations_by_class"))
    viol_high_sign = _to_int(vbc_high.get("sign"), 0)
    viol_high_tl = _to_int(vbc_high.get("traffic_light"), 0)
    viol_high_cw = _to_int(vbc_high.get("crosswalk"), 0)

    # Per-class breakdowns: keys are sign-class names (e.g., "MainRoadSign")
    vbc_step = _ensure_dict(metrics.get("violations_by_class_step"))
    vbc_event = _ensure_dict(metrics.get("violations_by_class_event"))
    in_zone_by_class = _ensure_dict(metrics.get("in_zone_by_class_step"))

    # Lookup expands base class to its registered subclasses (speed signs only;
    # see TARGET_CLASS_SUBCLASSES). For non-speed PDDs this collapses to
    # [target_class] and behavior is unchanged.
    target_classes = _resolve_target_classes(target_class)
    target_violations_step = _sum_class_keys(vbc_step, target_classes) if target_class else None
    target_violations_event = _sum_class_keys(vbc_event, target_classes) if target_class else None
    target_in_zone_steps = _sum_class_keys(in_zone_by_class, target_classes) if target_class else None
    target_in_zone = (target_in_zone_steps is not None and target_in_zone_steps > 0)
    target_compliant_event = (target_violations_event == 0) if target_class else None
    target_compliant_step = (target_violations_step == 0) if target_class else None

    # Build a candidate row in select_experts.passes_filter shape so we can
    # reuse recompute_dest / passes_filter directly. Critically, set
    # `violations_by_class` to per-class counts (event-based, integer per
    # sign-class name) — this is what passes_filter.is_compliant() expects.
    pf_row = {
        "valid": bool(replay.get("valid", True)),
        "crashed": _bool(metrics.get("crashed")),
        "out_of_road": _bool(metrics.get("out_of_road")),
        "arrived_dest": _bool(metrics.get("arrived_dest")),
        "final_step": _to_int(metrics.get("final_step"), 0),
        "route_completion": _to_float(metrics.get("route_completion"), 0.0) or 0.0,
        "violations_by_class": vbc_event,    # per-class events (used for is_compliant)
    }
    if target_class:
        dest_recomputed = bool(recompute_dest(pf_row, target_pdd, target_class, HORIZON_DEFAULT))
        passes = bool(passes_filter(pf_row, target_pdd, target_class,
                                     HORIZON_DEFAULT, MIN_ROUTE_COMPLETION, MIN_FINAL_STEP))
    else:
        dest_recomputed = pf_row["arrived_dest"]
        passes = False

    # Variant / display_policy mapping: derive from baseline name and replay.policy/variant.
    policy = replay.get("policy") or ""
    variant = replay.get("variant")
    display_policy = POLICY_DISPLAY_NAME.get(baseline, baseline)

    row = {
        "var_name": var_name,
        "var_idx": var_idx,
        "baseline": baseline,
        "policy": policy,
        "variant": variant or "",
        "display_policy": display_policy,
        "backend": replay.get("backend") or "",
        "pdd_code": pdd_code,
        "sign_slug": sign_slug,
        "target_sign_class": target_class or "",
        "is_no_entry_sign": is_no_entry,
        "scene_id": scene_id,
        "scene_uid": scene_uid,
        # Manifest-derived paired-scene fields (empty for non-paired scenes)
        "manifest_source": manifest_source,
        "is_paired_scene": is_paired_scene,
        "pdd_code_start": pdd_code_start,
        "pdd_code_end": pdd_code_end,
        "pdd_code_target": pdd_code_target,
        "sign_type_start": sign_type_start,
        "sign_type_end": sign_type_end,
        "zone_length_m": zone_length_m,
        # top-level
        "valid": bool(replay.get("valid", True)),
        # metrics — outcomes
        "arrived_dest": _bool(metrics.get("arrived_dest")),
        "crashed": _bool(metrics.get("crashed")),
        "crashed_ego_fault": _bool(metrics.get("crashed_ego_fault")),
        "crashed_npc_fault": _bool(metrics.get("crashed_npc_fault")),
        "out_of_road": _bool(metrics.get("out_of_road")),
        "success": _bool(metrics.get("success")),
        # metrics — performance
        "final_step": _to_int(metrics.get("final_step"), 0),
        "total_reward": _to_float(metrics.get("total_reward")),
        "route_completion": _to_float(metrics.get("route_completion")),
        "route_length_m": _to_float(metrics.get("route_length_m")),
        "distance_travelled_m": _to_float(metrics.get("distance_travelled_m")),
        "driving_score": _to_float(metrics.get("driving_score")),
        "driving_efficiency": _to_float(metrics.get("driving_efficiency")),
        "infraction_penalty": _to_float(metrics.get("infraction_penalty")),
        # metrics — comfort
        "smoothness_ratio": _to_float(metrics.get("smoothness_ratio")),
        "frame_smooth_ratio": _to_float(metrics.get("frame_smooth_ratio")),
        "smooth_segments": _to_int(metrics.get("smooth_segments"), 0),
        "total_segments": _to_int(metrics.get("total_segments"), 0),
        # metrics — safety
        "min_ttc_sec": _to_float(metrics.get("min_ttc_sec")),
        "mean_abs_lane_offset": _to_float(metrics.get("mean_abs_lane_offset")),
        "mean_abs_steer_delta": _to_float(metrics.get("mean_abs_steer_delta")),
        "hard_brake_count": _to_int(metrics.get("hard_brake_count"), 0),
        "hard_accel_count": _to_int(metrics.get("hard_accel_count"), 0),
        # metrics — violations (high-level + counts)
        "total_violations": _to_int(metrics.get("total_violations"), 0),
        "violations_event_count": _to_int(metrics.get("violations_event_count"), 0),
        "in_zone_total_steps": _to_int(metrics.get("in_zone_total_steps"), 0),
        "viol_high_sign": viol_high_sign,
        "viol_high_traffic_light": viol_high_tl,
        "viol_high_crosswalk": viol_high_cw,
        # JSON-encoded per-class breakdowns
        "violations_by_class_step_json": json.dumps(vbc_step, ensure_ascii=False, sort_keys=True),
        "violations_by_class_event_json": json.dumps(vbc_event, ensure_ascii=False, sort_keys=True),
        "in_zone_by_class_step_json": json.dumps(in_zone_by_class, ensure_ascii=False, sort_keys=True),
        # Target-class derived
        "target_violations_step": target_violations_step if target_violations_step is not None else "",
        "target_violations_event": target_violations_event if target_violations_event is not None else "",
        "target_in_zone_steps": target_in_zone_steps if target_in_zone_steps is not None else "",
        "target_in_zone": target_in_zone,
        "target_compliant_event": target_compliant_event if target_compliant_event is not None else "",
        "target_compliant_step": target_compliant_step if target_compliant_step is not None else "",
        # High-level compliance
        "sign_compliant_high": (
            bool(target_compliant_event)
            if target_class == "PedestrianYieldRule" and target_compliant_event is not None
            else (viol_high_sign == 0)
        ),
        "tl_compliant": (viol_high_tl == 0),
        "cw_compliant": (viol_high_cw == 0),
        # Recomputed dest + filter (require target_class)
        "dest_recomputed": dest_recomputed,
        "passes_filter": passes,
        # Comfort copy (==frame_smooth_ratio) for explicit selection-style scoring
        "comfort": _to_float(metrics.get("frame_smooth_ratio"), 0.0) or 0.0,
    }
    return row


CSV_COLUMNS = [
    "var_name", "var_idx", "baseline", "policy", "variant", "display_policy",
    "backend", "pdd_code", "sign_slug", "target_sign_class", "is_no_entry_sign",
    "scene_id", "scene_uid",
    "manifest_source", "is_paired_scene",
    "pdd_code_start", "pdd_code_end", "pdd_code_target",
    "sign_type_start", "sign_type_end", "zone_length_m",
    "valid",
    "arrived_dest", "crashed", "crashed_ego_fault", "crashed_npc_fault",
    "out_of_road", "success",
    "final_step", "total_reward", "route_completion", "route_length_m",
    "distance_travelled_m", "driving_score", "driving_efficiency",
    "infraction_penalty",
    "smoothness_ratio", "frame_smooth_ratio", "smooth_segments", "total_segments",
    "min_ttc_sec", "mean_abs_lane_offset", "mean_abs_steer_delta",
    "hard_brake_count", "hard_accel_count",
    "total_violations", "violations_event_count", "in_zone_total_steps",
    "viol_high_sign", "viol_high_traffic_light", "viol_high_crosswalk",
    "violations_by_class_step_json", "violations_by_class_event_json",
    "in_zone_by_class_step_json",
    "target_violations_step", "target_violations_event",
    "target_in_zone_steps", "target_in_zone",
    "target_compliant_event", "target_compliant_step",
    "sign_compliant_high", "tl_compliant", "cw_compliant",
    "dest_recomputed", "passes_filter",
    "comfort",
]


def _iter_jsonl(fp: Path):
    """Yield (lineno, dict) for each parseable JSON object in a JSONL file.

    Truncated/malformed lines are skipped with a warning to stderr.
    """
    bad = 0
    with fp.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError as e:
                bad += 1
                print(f"  [warn] {fp.name} line {i}: bad json (len={len(line)}, err={e})",
                      file=sys.stderr)
    if bad:
        print(f"  [warn] {fp.name}: skipped {bad} malformed lines", file=sys.stderr)


def _load_manifest_lookup(manifests_root: Path,
                            wanted_var_idxs: set[int]) -> dict[tuple[int, str], dict]:
    """Load chunks/var_<i>/var_<i>.jsonl into a (var_idx, scene_id) -> manifest_row map.

    The chunks manifest carries paired-scene fields (pdd_code_start/_end/_target,
    source=pgmap_paired, etc.) which the consolidated replay JSONL doesn't.
    """
    lookup: dict[tuple[int, str], dict] = {}
    if not manifests_root.is_dir():
        print(f"  [warn] manifests-root not found: {manifests_root}", file=sys.stderr)
        return lookup
    n_loaded = 0
    n_paired = 0
    for var_idx in sorted(wanted_var_idxs):
        mp = manifests_root / f"var_{var_idx}" / f"var_{var_idx}.jsonl"
        if not mp.exists():
            continue
        for _, r in _iter_jsonl(mp):
            sid = r.get("scene_id")
            if not sid:
                continue
            lookup[(var_idx, str(sid))] = r
            n_loaded += 1
            if "paired" in str(r.get("source") or "").lower():
                n_paired += 1
    print(f"  [manifest] loaded {n_loaded} rows ({n_paired} paired) from "
          f"{manifests_root}/var_*/var_*.jsonl")
    return lookup


def _write_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {out_path}  ({len(rows)} rows × {len(CSV_COLUMNS)} cols)")


def _build_from_episodes(episodes_root: Path, out_path: Path,
                          manifests_root_arg: str | None) -> None:
    """Build the metrics CSV from run_benchmark `episodes_*.jsonl` (unified path).

    Layout: <episodes_root>/<run_name>/episodes_*.jsonl, where <run_name> is
    "<policy>_<variant>" (the baseline). Only ok=true episodes are emitted. All
    episodes are treated as var_0 (a single-policy run uses var_0).
    """
    if not episodes_root.is_dir():
        print(f"ERROR: not a directory: {episodes_root}", file=sys.stderr)
        sys.exit(2)

    manifests_root = (Path(manifests_root_arg).resolve() if manifests_root_arg
                      else (episodes_root.parent / "chunks").resolve())
    manifest_lookup = _load_manifest_lookup(manifests_root, wanted_var_idxs={0})

    seen: dict[tuple[int, str, str], dict] = {}
    n_total = n_kept = n_dup = n_no_id = n_skip = 0
    for run_dir in sorted(d for d in episodes_root.iterdir() if d.is_dir()):
        baseline = run_dir.name
        eps = sorted(run_dir.glob("episodes_*.jsonl"))
        n_for_baseline = 0
        for fp in eps:
            for _, ep in _iter_jsonl(fp):
                if not ep.get("ok"):
                    n_skip += 1
                    continue
                n_total += 1
                row = _build_row(_episode_to_replay(ep), "var_0", 0, baseline,
                                 manifest_lookup)
                if row is None:
                    n_no_id += 1
                    continue
                key = (0, baseline, row["scene_uid"])
                if key in seen:
                    n_dup += 1
                seen[key] = row
                n_kept += 1
                n_for_baseline += 1
        if eps:
            print(f"  [scan] {baseline}: {len(eps)} episodes file(s), "
                  f"{n_for_baseline} episodes")

    rows = list(seen.values())
    print(f"[stats] read={n_total} kept={len(rows)} dups_overwritten={n_dup} "
          f"no_scene_uid={n_no_id} skipped_not_ok={n_skip}")
    _write_csv(rows, out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--episodes-root",
                     help="policy_eval dir containing <run_name>/episodes_*.jsonl. "
                          "PRIMARY path — builds the CSV straight from the per-episode "
                          "JSONL that run_benchmark always writes (no sidecar needed).")
    src.add_argument("--runs-root",
                     help="Directory containing var_<i>/<baseline>_replays.jsonl. "
                          "Legacy path — reads consolidated replays / replay.json sidecars.")
    ap.add_argument("--out", required=True,
                    help="Output CSV path (e.g. metrics_per_episode.csv)")
    ap.add_argument("--vars", default="all",
                    help="Comma-separated var indices (e.g. '0,1,2') or 'all' (default). "
                         "Only used with --runs-root.")
    ap.add_argument("--manifests-root", default=None,
                    help="Directory with var_<i>/var_<i>.jsonl manifests (for paired-scene "
                         "info: pdd_code_start/end/target). Default: <runs-root>/../chunks")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --episodes-root: build directly from episodes_*.jsonl (unified path).
    if args.episodes_root:
        _build_from_episodes(Path(args.episodes_root).resolve(), out_path,
                             args.manifests_root)
        return

    runs_root = Path(args.runs_root).resolve()
    if not runs_root.is_dir():
        print(f"ERROR: not a directory: {runs_root}", file=sys.stderr)
        sys.exit(2)

    # Collect var_<i> directories
    var_dirs: list[tuple[int, Path]] = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        m = VAR_DIR_RE.match(d.name)
        if m:
            var_dirs.append((int(m.group(1)), d))
    if args.vars != "all":
        wanted = {int(s.strip()) for s in args.vars.split(",") if s.strip()}
        var_dirs = [(i, d) for i, d in var_dirs if i in wanted]
    if not var_dirs:
        print(f"ERROR: no var_<i> dirs found in {runs_root}", file=sys.stderr)
        sys.exit(2)

    # Load manifest lookup for paired-scene enrichment
    manifests_root = (Path(args.manifests_root).resolve() if args.manifests_root
                      else (runs_root.parent / "chunks").resolve())
    manifest_lookup = _load_manifest_lookup(manifests_root,
                                              wanted_var_idxs={i for i, _ in var_dirs})

    # Dedup by (var_idx, baseline, scene_uid). Last write wins (matches reruns).
    seen: dict[tuple[int, str, str], dict] = {}
    n_total = 0
    n_kept = 0
    n_dup = 0
    n_no_id = 0

    n_sidecar_total = 0

    for var_idx, vdir in var_dirs:
        var_name = vdir.name
        jsonls = sorted(vdir.glob("*_replays.jsonl"))
        # Step 1: read consolidated <baseline>_replays.jsonl files
        consolidated_baselines: set[str] = set()
        for fp in jsonls:
            baseline = fp.stem[:-len("_replays")] if fp.stem.endswith("_replays") else fp.stem
            consolidated_baselines.add(baseline)
            for _, replay in _iter_jsonl(fp):
                n_total += 1
                row = _build_row(replay, var_name, var_idx, baseline, manifest_lookup)
                if row is None:
                    n_no_id += 1
                    continue
                key = (var_idx, baseline, row["scene_uid"])
                if key in seen:
                    n_dup += 1
                seen[key] = row
                n_kept += 1

        # Step 2: fall back to sidecar replay.json for baselines without
        # consolidated jsonl. Sidecars live at:
        #   <var_dir>/<baseline>/replays/<sign>/by_sign/<sign>/by_scene/<scene_uid>/<expert>/replay.json
        sidecar_baselines: dict[str, int] = {}
        for baseline_dir in sorted(d for d in vdir.iterdir() if d.is_dir()):
            baseline = baseline_dir.name
            if baseline.startswith("chunk_"):
                continue    # legacy chunk subdir, contains only run.log
            if baseline in consolidated_baselines:
                continue    # already covered by consolidated jsonl
            replays_dir = baseline_dir / "replays"
            if not replays_dir.is_dir():
                continue
            n_for_baseline = 0
            for sidecar in replays_dir.glob("*/by_sign/*/by_scene/*/*/replay.json"):
                try:
                    replay = json.loads(sidecar.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as e:
                    print(f"  [warn] {sidecar}: {e}", file=sys.stderr)
                    continue
                n_total += 1
                n_sidecar_total += 1
                n_for_baseline += 1
                row = _build_row(replay, var_name, var_idx, baseline, manifest_lookup)
                if row is None:
                    n_no_id += 1
                    continue
                key = (var_idx, baseline, row["scene_uid"])
                if key in seen:
                    n_dup += 1
                seen[key] = row
                n_kept += 1
            if n_for_baseline:
                sidecar_baselines[baseline] = n_for_baseline

        if jsonls or sidecar_baselines:
            extra = (f", sidecar baselines: {len(sidecar_baselines)} "
                     f"({sum(sidecar_baselines.values())} replays)"
                     if sidecar_baselines else "")
            print(f"  [scan] {var_name}: {len(jsonls)} consolidated baseline files{extra}")
        else:
            print(f"  [scan] {var_name}: no data (no jsonl, no sidecars)")

    rows = list(seen.values())
    print(f"[stats] read={n_total} (sidecars={n_sidecar_total}) "
          f"kept={len(rows)} dups_overwritten={n_dup} no_scene_uid={n_no_id}")
    _write_csv(rows, out_path)


if __name__ == "__main__":
    main()
