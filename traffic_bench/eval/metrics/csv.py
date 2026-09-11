#!/usr/bin/env python3
"""Build a single episode-level CSV — the source of truth for downstream aggregations.

Two input modes (mutually exclusive):

  --episodes-root  (PRIMARY)  <policy_eval>/<run_name>/episodes_*.jsonl
      Built straight from the per-episode JSONL that eval run ALWAYS writes.
      No replay.json sidecar needed.

  --runs-root      (legacy)   <runs-root>/var_<i>/<baseline>_replays.jsonl
      Reads consolidated replays / replay.json sidecars.

Both need ``--manifest``: the manifest the episodes were run from. Every
episode is matched to its manifest row by scene_id, and its map (``map_id``)
is the directory of that row's ``net_path`` (metrics/map_id.py). Nothing is
guessed and nothing is skipped: a missing manifest, an episode the manifest
does not list, a failed episode, a record without ids or a malformed value
raises with the file and line (metrics/README.md, "Errors").

Either way, one CSV row per episode carries all `metrics` fields plus precomputed
flags (target compliance, recomputed dest, passes_filter) needed by
  generate_cumulative_markdown_report.py
  expert_selection_exps/select_from_replays.py

Usage:
  python -m traffic_bench.eval metrics csv \
      --episodes-root <eval_out>/benchmark/full/policy_eval \
      --manifest      <split>/real_manifest.jsonl \
      --out           <eval_out>/metrics_per_episode.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from contextlib import contextmanager
from pathlib import Path

from traffic_bench.oracle.select.filter import (
    SIGN_CLASS_MAP,
    NO_ENTRY_SIGNS,
    HORIZON_DEFAULT,
    MIN_FINAL_STEP,
    MIN_ROUTE_COMPLETION,
    normalize_sign,
    recompute_dest,
    passes_filter,
)
from traffic_bench.agents.policy_names import canonical_policy_name
from traffic_bench.eval.metrics.map_id import ManifestError, map_id_from_manifest_row


# Display names for baselines recorded under legacy spellings. Baselines are
# canonicalised on load (canonical_policy_name), so this only matters for
# ``ppo_expert`` and any name that slipped past normalisation.
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
    "SpeedLimitSign":         ["SpeedLimitSign20", "SpeedLimitSign30", "SpeedLimitSign40", "SpeedLimitSign60"],
    "EndOfSpeedLimitSign":    ["EndOfSpeedLimitSign20", "EndOfSpeedLimitSign30", "EndOfSpeedLimitSign40", "EndOfSpeedLimitSign60"],
    "ZoneSpeedLimitSign":     ["ZoneSpeedLimitSign20", "ZoneSpeedLimitSign30", "ZoneSpeedLimitSign40", "ZoneSpeedLimitSign60"],
    "EndOfZoneSpeedLimitSign":["EndOfZoneSpeedLimitSign20", "EndOfZoneSpeedLimitSign30", "EndOfZoneSpeedLimitSign40", "EndOfZoneSpeedLimitSign60"],
    "MinimumSpeedLimitSign":  ["MinimumSpeedLimit30", "MinimumSpeedLimit40", "MinimumSpeedLimit50", "MinimumSpeedLimit60"],
}


def _resolve_target_classes(base_class: str | None) -> list[str]:
    """Return [base_class] plus any registered subclasses (for speed signs)."""
    if not base_class:
        return []
    out = [base_class]
    out.extend(TARGET_CLASS_SUBCLASSES.get(base_class, []))
    return out


def _sum_class_keys(d: dict, classes: list[str]) -> int:
    """Sum the integer counts of d for the given class keys (absent keys count 0)."""
    total = 0
    for c in classes:
        v = d.get(c)
        if v is None:
            continue
        total += _to_int(v)
    return total


VAR_DIR_RE = re.compile(r"^var_(\d+)$")


# ---------------------------------------------------------------------------
# Strict value parsing: an absent value (None) takes the documented default,
# anything present must parse. Nothing is coerced to a default on error.
# ---------------------------------------------------------------------------
def _to_int(v, default=0) -> int:
    """``default`` when absent (None); otherwise an int, a bool, an integral
    float or a decimal string. Anything else raises ValueError."""
    if v is None:
        return default
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not v.is_integer():
            raise ValueError(f"expected an integer, got {v!r}")
        return int(v)
    if isinstance(v, str):
        return int(v)
    raise ValueError(f"expected an integer, got {type(v).__name__} {v!r}")


def _to_float(v, default=None):
    """``default`` when absent (None); otherwise a finite number or its
    string. Anything else, NaN and infinities included, raises ValueError."""
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError(f"expected a number, got {type(v).__name__} {v!r}")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"non-finite value {v!r}")
    return f


def _bool(v) -> bool:
    """An episode flag: absent (None) is False; otherwise a bool or 0 / 1.
    Anything else raises ValueError."""
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    raise ValueError(f"expected a bool flag, got {type(v).__name__} {v!r}")


def _dict_or_empty(v, what: str) -> dict:
    """A per-class breakdown: absent (None) is empty; otherwise a dict."""
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    raise ValueError(f"{what}: expected a dict, got {type(v).__name__} {v!r}")


def _required_id(rec: dict, key: str) -> str:
    """An identifier every record must carry (scene_id, scene_uid)."""
    v = rec.get(key)
    if not isinstance(v, str) or not v:
        raise ValueError(f"record has no {key} (got {v!r}); it cannot be matched "
                         "to the manifest or deduplicated")
    return v


@contextmanager
def _located(where: str):
    """Prefix a ManifestError / ValueError raised inside with ``where``
    (file and line), keeping its type."""
    try:
        yield
    except ManifestError as e:
        raise ManifestError(f"{where}: {e}") from e
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from e


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
        "arrived_dest": _bool(ep.get("reached_dest")),
        # sidecar `metrics.crashed` is info["crash"] alone (without OOR);
        # run_benchmark exposes that as `crashed_raw`.
        "crashed": _bool(ep.get("crashed_raw", ep.get("crashed"))),
        "crashed_ego_fault": _bool(ep.get("crashed_ego_fault")),
        "crashed_npc_fault": _bool(ep.get("crashed_npc_fault")),
        "out_of_road": _bool(ep.get("out_of_road")),
        "success": _bool(ep.get("success")),
        "final_step": ep.get("steps"),
        "total_reward": ep.get("total_reward"),
        "route_completion": (_to_float(rc_pct) / 100.0) if rc_pct is not None else None,
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
        "violations_by_class_step": _dict_or_empty(
            ep.get("violations_by_class_step"), "violations_by_class_step"),
        "violations_by_class_event": _dict_or_empty(
            ep.get("violations_by_class_event"), "violations_by_class_event"),
        "in_zone_total_steps": ep.get("in_zone_total_steps", 0),
        "in_zone_by_class_step": _dict_or_empty(
            ep.get("in_zone_by_class_step"), "in_zone_by_class_step"),
        "violations_event_count": ep.get("violations_event_count", 0),
    }
    pdd = ep.get("sign_type") or ep.get("pdd_code") or ""
    return {
        "scene_id": _required_id(ep, "scene_id"),
        "scene_uid": _required_id(ep, "scene_uid"),
        "backend": ep.get("backend") or "",
        "pdd_code": pdd,
        "sign_slug": ep.get("sign_slug") or (str(pdd).replace(".", "_") if pdd else ""),
        "policy": ep.get("policy") or "",
        "variant": ep.get("variant") or "",
        "valid": ep.get("ok") is True,
        "source_row": {},
        "metrics": metrics,
    }


def _build_row(replay: dict, var_name: str, var_idx: int, baseline: str,
               manifest: "ManifestIndex") -> dict:
    """Convert one consolidated replay JSON object into a flat CSV row dict.

    The replay's scene is looked up in ``manifest`` (``ManifestIndex.row``
    raises when the manifest does not list it). That row gives the episode's
    physical map (``map_id`` = directory of ``net_path``) and, for paired-zone
    scenes, the fields the aggregator groups on: ``manifest_source``,
    ``is_paired_scene``, ``pdd_code_start``, ``pdd_code_end``,
    ``pdd_code_target``, ``sign_type_start``, ``sign_type_end``,
    ``zone_length_m``.

    Raises ValueError when the replay lacks its ids or a value is malformed.
    """
    metrics = replay.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"replay has no metrics dict (got {type(metrics).__name__})")
    pdd_code_raw = replay.get("pdd_code") or replay.get("sign_slug") or ""
    pdd_code = normalize_sign(str(pdd_code_raw)) if pdd_code_raw else ""
    sign_slug = (replay.get("sign_slug")
                 or (pdd_code.replace(".", "_") if pdd_code else ""))
    if not pdd_code:
        # Legacy replays without pdd_code: the embedded manifest row has it.
        sr = _dict_or_empty(replay.get("source_row"), "source_row")
        pdd_code = normalize_sign(str(sr.get("sign_code") or sr.get("pdd_code") or "")) or ""
        if pdd_code and not sign_slug:
            sign_slug = pdd_code.replace(".", "_")

    scene_id = _required_id(replay, "scene_id")
    scene_uid = _required_id(replay, "scene_uid")

    # The manifest row the episode was run from, matched by (var_idx,
    # scene_id): the physical map and the paired-zone fields.
    mr = manifest.row(var_idx, scene_id)
    map_id = map_id_from_manifest_row(mr)
    net_path = str(mr["net_path"]).strip()
    manifest_source = str(mr.get("source") or "")
    is_paired_scene = ("paired" in manifest_source.lower())
    pdd_code_start = ""
    pdd_code_end = ""
    pdd_code_target = pdd_code    # default — replay's own code
    sign_type_start = ""
    sign_type_end = ""
    zone_length_m = ""
    if is_paired_scene:
        pdd_code_start = str(mr.get("pdd_code_start") or "")
        pdd_code_end = str(mr.get("pdd_code_end") or "")
        # rewrite_speed_manifests.py drops `pdd_code` from paired rows
        # and rarely sets `pdd_code_target` — use `pdd_code_start` then
        # so target_class lookup works downstream.
        pdd_code_target = str(mr.get("pdd_code_target") or pdd_code_start or pdd_code)
        sign_type_start = str(mr.get("sign_type_start") or "")
        sign_type_end = str(mr.get("sign_type_end") or "")
        zl = mr.get("zone_length_m")
        zone_length_m = "" if zl is None else _to_float(zl)

    # For paired scenes pdd_code is empty; pdd_code_target carries the target
    # (start) sign. For non-paired scenes pdd_code_target == pdd_code.
    target_pdd = pdd_code_target or pdd_code
    target_class = SIGN_CLASS_MAP.get(target_pdd) if target_pdd else None
    is_no_entry = bool(target_pdd in NO_ENTRY_SIGNS)

    # High-level violations dict in replay.json: keys are {"sign","traffic_light","crosswalk"}
    vbc_high = _dict_or_empty(metrics.get("violations_by_class"), "violations_by_class")
    viol_high_sign = _to_int(vbc_high.get("sign"), 0)
    viol_high_tl = _to_int(vbc_high.get("traffic_light"), 0)
    viol_high_cw = _to_int(vbc_high.get("crosswalk"), 0)

    # Per-class breakdowns: keys are sign-class names (e.g., "MainRoadSign")
    vbc_step = _dict_or_empty(metrics.get("violations_by_class_step"), "violations_by_class_step")
    vbc_event = _dict_or_empty(metrics.get("violations_by_class_event"), "violations_by_class_event")
    in_zone_by_class = _dict_or_empty(metrics.get("in_zone_by_class_step"), "in_zone_by_class_step")

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

    # Build a candidate row in select.filter.passes_filter shape so we can
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
    # Legacy spellings (comprehensive_rule_expert / rule_compliant) → canonical ids.
    baseline = canonical_policy_name(baseline)
    policy = canonical_policy_name(replay.get("policy") or "")
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
        # Physical map from the manifest row (metrics/map_id.py)
        "map_id": map_id,
        "net_path": net_path,
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
        # SR&Dest: obeyed the target sign AND reached the destination. A run
        # that crashes before the sign has no violation but no arrival either,
        # so it scores 0 instead of passing as "compliant".
        "sr_and_dest": (
            (bool(target_compliant_event) and bool(pf_row["arrived_dest"]))
            if target_compliant_event is not None else ""
        ),
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
    "scene_id", "scene_uid", "map_id", "net_path",
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
    "sr_and_dest",
    "sign_compliant_high", "tl_compliant", "cw_compliant",
    "dest_recomputed", "passes_filter",
    "comfort",
]


def _iter_jsonl(fp: Path):
    """Yield (lineno, dict) for each JSON object of a JSONL file. A line that
    is not a JSON object (a torn write, a stray value) raises ValueError with
    the file and line."""
    with fp.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{fp}:{i}: malformed JSON ({e.msg}, line of {len(line)} chars)") from e
            if not isinstance(rec, dict):
                raise ValueError(f"{fp}:{i}: expected a JSON object, got {type(rec).__name__}")
            yield i, rec


# ---------------------------------------------------------------------------
# Manifest: the only source of an episode's map
# ---------------------------------------------------------------------------
# Fields of a manifest row that _build_row reads. Two rows with the same
# scene_id must agree on them, or the episodes of that scene_id have no
# single map.
_MANIFEST_MATCH_FIELDS: tuple[str, ...] = (
    "net_path", "source", "pdd_code_start", "pdd_code_end", "pdd_code_target",
    "sign_type_start", "sign_type_end", "zone_length_m",
)


class ManifestIndex:
    """Manifest rows by (var_idx, scene_id). A lookup that misses raises."""

    def __init__(self, rows: dict[tuple[int, str], dict], sources: list[Path]):
        self.rows = rows
        self.sources = list(sources)

    def row(self, var_idx: int, scene_id: str) -> dict:
        try:
            return self.rows[(int(var_idx), scene_id)]
        except KeyError:
            raise ManifestError(
                f"scene_id {scene_id!r} (var {var_idx}) is not in the manifest "
                f"{', '.join(str(p) for p in self.sources)}: the episode was run from "
                "another manifest, so its map cannot be determined") from None


def load_manifest(manifest: Path, var_idxs: set[int]) -> ManifestIndex:
    """The manifest the episodes were run from.

    ``manifest`` is a manifest JSONL file (``real_manifest.jsonl``, or the
    ``input_manifest.jsonl`` that ``run policies=…`` writes), registered under
    every var index scored, or a ``chunks/`` directory that must hold
    ``var_<i>/var_<i>.jsonl`` for every var index scored. Every row needs a
    scene_id and a net_path naming ``<scene_dir>/<net file>``; a scene_id
    listed twice must name the same net and paired-zone fields. Anything else
    raises: without the manifest the map of an episode is unknown, and
    per-map metrics cannot be computed.
    """
    if manifest.is_file():
        sources: list[tuple[int | None, Path]] = [(None, manifest)]
    elif manifest.is_dir():
        sources = [(i, manifest / f"var_{i}" / f"var_{i}.jsonl") for i in sorted(var_idxs)]
        missing = [str(p) for _, p in sources if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{manifest} is read as a chunks/ manifest and lacks "
                f"{', '.join(missing)}; pass the manifest file the episodes were run from")
    else:
        raise FileNotFoundError(
            f"manifest not found: {manifest}. Per-map metrics cannot be computed "
            "without the manifest the episodes were run from")
    rows: dict[tuple[int, str], dict] = {}
    n_rows = n_paired = 0
    for var_idx, path in sources:
        idxs = [var_idx] if var_idx is not None else sorted(var_idxs)
        n_file = 0
        for lineno, r in _iter_jsonl(path):
            with _located(f"{path}:{lineno}"):
                sid = r.get("scene_id")
                if not isinstance(sid, str) or not sid:
                    raise ManifestError("manifest row has no scene_id")
                map_id_from_manifest_row(r)
                ident = tuple(r.get(f) for f in _MANIFEST_MATCH_FIELDS)
                for i in idxs:
                    prev = rows.get((i, sid))
                    if prev is not None and tuple(
                            prev.get(f) for f in _MANIFEST_MATCH_FIELDS) != ident:
                        raise ManifestError(
                            f"scene_id {sid!r} is listed twice with different net_path "
                            "or paired-zone fields, so its episodes cannot be matched "
                            "to one map")
                    rows.setdefault((i, sid), r)
            n_file += 1
            if "paired" in str(r.get("source") or "").lower():
                n_paired += 1
        if n_file == 0:
            raise ManifestError(f"manifest {path} has no rows")
        n_rows += n_file
    print(f"  [manifest] {n_rows} rows ({n_paired} paired) from "
          + ", ".join(str(p) for _, p in sources))
    return ManifestIndex(rows, [p for _, p in sources])


def _write_csv(rows: list[dict], out_path: Path) -> None:
    """Write all rows or nothing: rows are checked against the schema first,
    and the file is written to a temporary name and moved into place."""
    expected = set(CSV_COLUMNS)
    for r in rows:
        if set(r) != expected:
            raise ValueError(f"row {r.get('scene_uid')!r} does not match the CSV schema: "
                             f"{sorted(set(r) ^ expected)}")
    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    tmp.replace(out_path)
    print(f"[write] {out_path}  ({len(rows)} rows × {len(CSV_COLUMNS)} cols)")


def _build_from_episodes(episodes_root: Path, out_path: Path, manifest: Path) -> None:
    """Build the metrics CSV from run_benchmark `episodes_*.jsonl` (unified path).

    Layout: <episodes_root>/<run_name>/episodes_*.jsonl, where <run_name> is
    "<policy>_<variant>" (the baseline). All episodes are treated as var_0 (a
    single-policy run uses var_0). A rerun appends to the same file, so the
    last record of a (baseline, scene_uid) wins. If that record is a failed
    episode (``ok`` is not true) the build stops: dropping it would change the
    map's set of augmentations without a trace.
    """
    if not episodes_root.is_dir():
        raise FileNotFoundError(f"episodes root is not a directory: {episodes_root}")
    index = load_manifest(manifest, {0})

    last: dict[tuple[int, str, str], tuple[Path, int, dict]] = {}
    n_read = n_dup = 0
    for run_dir in sorted(d for d in episodes_root.iterdir() if d.is_dir()):
        baseline = run_dir.name
        eps = sorted(run_dir.glob("episodes_*.jsonl"))
        if not eps:
            print(f"  [scan] {baseline}: no episodes_*.jsonl, not a run directory",
                  file=sys.stderr)
            continue
        n_for_baseline = 0
        for fp in eps:
            for lineno, ep in _iter_jsonl(fp):
                with _located(f"{fp}:{lineno}"):
                    uid = _required_id(ep, "scene_uid")
                key = (0, baseline, uid)
                if key in last:
                    n_dup += 1
                last[key] = (fp, lineno, ep)
                n_read += 1
                n_for_baseline += 1
        print(f"  [scan] {baseline}: {len(eps)} episodes file(s), {n_for_baseline} records")
    if not last:
        raise ValueError(f"no episode records under {episodes_root}")

    failed = [(fp, ln, ep) for fp, ln, ep in last.values() if ep.get("ok") is not True]
    if failed:
        shown = "; ".join(f"{fp}:{ln} {ep.get('scene_uid')}" for fp, ln, ep in failed[:5])
        more = " …" if len(failed) > 5 else ""
        raise ValueError(
            f"{len(failed)} episode(s) failed to run (ok is not true) and have no later "
            f"successful record: {shown}{more}. Re-run them (rerun_failed=true) before "
            "scoring; dropping them would bias the per-map means.")

    rows = []
    for (_, baseline, _uid), (fp, lineno, ep) in last.items():
        with _located(f"{fp}:{lineno}"):
            rows.append(_build_row(_episode_to_replay(ep), "var_0", 0, baseline, index))
    print(f"[stats] records={n_read} episodes={len(rows)} dups_overwritten={n_dup}")
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
    ap.add_argument("--manifest", required=True,
                    help="The manifest the episodes were run from: a manifest JSONL "
                         "(real_manifest.jsonl, or the input_manifest.jsonl that "
                         "`run policies=…` writes), or a chunks/ dir holding "
                         "var_<i>/var_<i>.jsonl for every var scored (--runs-root). "
                         "An episode's map is the directory of its row's net_path; an "
                         "episode the manifest does not list is an error.")
    ap.add_argument("--out", required=True,
                    help="Output CSV path (e.g. metrics_per_episode.csv)")
    ap.add_argument("--vars", default="all",
                    help="Comma-separated var indices (e.g. '0,1,2') or 'all' (default). "
                         "Only used with --runs-root.")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest).resolve()

    # --episodes-root: build directly from episodes_*.jsonl (unified path).
    if args.episodes_root:
        _build_from_episodes(Path(args.episodes_root).resolve(), out_path, manifest)
        return

    runs_root = Path(args.runs_root).resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs root is not a directory: {runs_root}")

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
        raise FileNotFoundError(f"no var_<i> dirs in {runs_root} (vars={args.vars})")

    index = load_manifest(manifest, {i for i, _ in var_dirs})

    # Dedup by (var_idx, baseline, scene_uid). Last write wins (matches reruns).
    seen: dict[tuple[int, str, str], dict] = {}
    n_total = 0
    n_dup = 0
    n_sidecar_total = 0

    for var_idx, vdir in var_dirs:
        var_name = vdir.name
        jsonls = sorted(vdir.glob("*_replays.jsonl"))
        # Step 1: read consolidated <baseline>_replays.jsonl files
        consolidated_baselines: set[str] = set()
        for fp in jsonls:
            baseline = fp.stem[:-len("_replays")] if fp.stem.endswith("_replays") else fp.stem
            consolidated_baselines.add(baseline)
            for lineno, replay in _iter_jsonl(fp):
                n_total += 1
                with _located(f"{fp}:{lineno}"):
                    row = _build_row(replay, var_name, var_idx, baseline, index)
                key = (var_idx, baseline, row["scene_uid"])
                if key in seen:
                    n_dup += 1
                seen[key] = row

        # Step 2: baselines without a consolidated jsonl are read from their
        # replay.json sidecars:
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
                with _located(str(sidecar)):
                    replay = json.loads(sidecar.read_text(encoding="utf-8"))
                    if not isinstance(replay, dict):
                        raise ValueError(f"expected a JSON object, got {type(replay).__name__}")
                    row = _build_row(replay, var_name, var_idx, baseline, index)
                n_total += 1
                n_sidecar_total += 1
                n_for_baseline += 1
                key = (var_idx, baseline, row["scene_uid"])
                if key in seen:
                    n_dup += 1
                seen[key] = row
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
    if not rows:
        raise ValueError(f"no replays under {runs_root}")
    print(f"[stats] read={n_total} (sidecars={n_sidecar_total}) "
          f"kept={len(rows)} dups_overwritten={n_dup}")
    _write_csv(rows, out_path)


if __name__ == "__main__":
    main()
