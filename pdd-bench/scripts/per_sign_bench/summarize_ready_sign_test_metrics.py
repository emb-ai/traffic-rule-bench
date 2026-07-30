#!/usr/bin/env python3
"""Summarize Eff / Comfort / RC / Collision / SCR on the ready-sign test set.

Sources (test split):
  * ``test_metrics/test20_batch/<label>/eval_out/metrics_per_episode.csv``
    — signs evaluated via ``run_test_metrics_batch.py``
  * ``*/final_metrics_v1/eval_out_test/`` or ``combined/eval_out_test/``
    — filtered by ``filter_final_metrics_to_test.py`` (3.1–3.2, 5.7.1–5.7.2)
  * on-the-fly filter of ``final_metrics_v1/eval_out`` by ``catalog_test20.jsonl``
    — 3.18.1–3.18.2, 4.1.1–4.1.6
  * ``speed_signs/.../metrics_per_episode_test20.csv`` — per-code 3.24 / 4.6 / 5.21 / 5.31
  * ``detour_sign/.../eval_test20/metrics_per_episode.csv`` — pooled mean over 4.2.1–4.2.3

Metrics (per episode, then averaged):
  Efficiency       — driving_efficiency
  Comfort          — comfort (= frame_smooth_ratio)
  Route Completion — 1.0 if arrived_dest else min(1, distance/route_length)
                     (route_completion field in CSV is currently always 0)
  Collision        — crashed (rate)
  Sign Compliance  — priority signs: sign_compliant_high over ALL episodes
                     (= report "Sign compliance SR");
                     speed signs (3.24/4.6/5.21/5.31): A6 filter (F3∩F2) +
                     target_compliant_event among in-zone
                     (= fv_split_report «Compliance (in-zone)»);
                     3.24/5.31 additionally uniform-average over v_target strata;
                     all other signs: sign_compliant_high among target_in_zone
                     (= report "Sign compliance (in-zone)" / sign_compliance_x)

Averages:
  micro — weighted by #episodes (default; fills paper-style planner rows)
  macro — unweighted mean of per-sign means
  SCR (overall) — mean of category means
                 (priority / prohibitory / mandatory / special),
                 where each category mean is the unweighted mean of
                 per-sign SCR within that category

Examples:
  python summarize_ready_sign_test_metrics.py
  python summarize_ready_sign_test_metrics.py --baselines idm_default,carl_default,plant2_default
  python summarize_ready_sign_test_metrics.py --out /tmp/ready_test_summary
  python summarize_ready_sign_test_metrics.py --list
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

PER_SIGN = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SignJob:
    """One row-group in the paper table (one sign or a pooled group)."""

    label: str
    sources: tuple[Path, ...]
    codes: tuple[str, ...]
    # If set, keep only episodes whose scene_id is in this test catalog
    # (for codes in ``codes``). Needed when CSV is full final_metrics.
    catalog: Optional[Path] = None


def _t20(bench: str, slug: str) -> Path:
    return (
        PER_SIGN / bench / "benchmark_output" / "test_metrics" / "test20_batch"
        / slug / "eval_out" / "metrics_per_episode.csv"
    )


def _fm(bench: str, slug: str) -> Path:
    return (
        PER_SIGN / bench / "benchmark_output" / slug
        / "final_metrics_v1" / "eval_out" / "metrics_per_episode.csv"
    )


def _eval_test(bench: str, *parts: str) -> Path:
    return PER_SIGN / bench / "benchmark_output" / Path(*parts) / "metrics_per_episode.csv"


def _cat(bench: str) -> Path:
    return PER_SIGN / bench / "benchmark_output" / "combined" / "catalog_test20.jsonl"


# Full paper-ready sign list (order = table row order).
READY_JOBS: list[SignJob] = [
    SignJob("2.1", (_t20("main_sign", "2_1"),), ("2.1",)),
    SignJob(
        "2.3.1-2.3.3",
        (_t20("secondary_sign", "2_3_1_2_3_3"),),
        ("2.3.1", "2.3.2", "2.3.3"),
    ),
    SignJob("2.4", (_t20("yield_sign", "2_4"),), ("2.4",)),
    SignJob("2.5", (_t20("stop_sign", "2_5"),), ("2.5",)),
    SignJob(
        "3.1-3.2",
        (_eval_test("no_entry_signs", "combined", "eval_out_test"),),
        ("3.1", "3.2"),
    ),
    # SignJob(
    #     "3.18.1-3.18.2",
    #     (_fm("no_turn_signs", "3_18_1"), _fm("no_turn_signs", "3_18_2")),
    #     ("3.18.1", "3.18.2"),
    #     catalog=_cat("no_turn_signs"),
    # ),
    SignJob(
        "3.24",
        (
            PER_SIGN / "speed_signs" / "benchmark_output" / "run_v61_a6"
            / "eval_fast" / "metrics_per_episode_test20.csv",
        ),
        ("3.24",),
    ),
    # SignJob(
    #     "4.1.1-4.1.6",
    #     tuple(_fm("direction_signs", f"4_1_{i}") for i in range(1, 7)),
    #     tuple(f"4.1.{i}" for i in range(1, 7)),
    #     catalog=_cat("direction_signs"),
    # ),
    SignJob(
        "4.2.1-4.2.3",
        (_eval_test("detour_sign", "4_2", "eval_test20"),),
        ("4.2.1", "4.2.2", "4.2.3"),
    ),
    SignJob("4.3", (_t20("roundabout_sign", "4_3"),), ("4.3",)),
    SignJob(
        "4.6",
        (
            PER_SIGN / "speed_signs" / "benchmark_output" / "run_v61_a6"
            / "eval_fast" / "metrics_per_episode_test20.csv",
        ),
        ("4.6",),
    ),
    SignJob(
        "5.7.1-5.7.2",
        (_eval_test("one_way_signs", "combined", "eval_out_test"),),
        ("5.7.1", "5.7.2"),
    ),
    SignJob(
        "5.15.1-5.15.2",
        (_t20("lane_direction_signs", "5_15_1_5_15_2"),),
        ("5.15.1", "5.15.2"),
    ),
    SignJob("5.19", (_t20("crosswalk_sign", "5_19"),), ("5.19",)),
    SignJob(
        "5.21",
        (
            PER_SIGN / "speed_signs" / "benchmark_output" / "run_v61_a6"
            / "eval_fast" / "metrics_per_episode_test20.csv",
        ),
        ("5.21",),
    ),
    SignJob(
        "5.31",
        (
            PER_SIGN / "speed_signs" / "benchmark_output" / "run_v61_a6"
            / "eval_fast" / "metrics_per_episode_test20.csv",
        ),
        ("5.31",),
    ),
]


def _fm_test(bench: str, slug: str) -> Path:
    return (
        PER_SIGN / bench / "benchmark_output" / slug
        / "final_metrics_v1" / "eval_out_test" / "metrics_per_episode.csv"
    )


# Extra per-code rows for the «Запрещенная полоса» block (not in overall SCR).
DETAIL_JOBS: list[SignJob] = [
    SignJob("3.1", (_fm_test("no_entry_signs", "3_1"),), ("3.1",)),
    SignJob("3.2", (_fm_test("no_entry_signs", "3_2"),), ("3.2",)),
    SignJob(
        "3.18.1",
        (_fm("no_turn_signs", "3_18_1"),),
        ("3.18.1",),
        catalog=_cat("no_turn_signs"),
    ),
    SignJob(
        "3.18.2",
        (_fm("no_turn_signs", "3_18_2"),),
        ("3.18.2",),
        catalog=_cat("no_turn_signs"),
    ),
]


# Canonical planner order for paper-style tables.
BASELINE_ORDER: list[str] = [
    "idm_default", "idm_s1", "idm_s2", "idm_s3", "idm_s4",
    "ppo_lidar_default",
    "carl_default",
    "plant2_default",
    "comprehensive_rule_expert_default",
    "comprehensive_rule_expert_s1",
    "comprehensive_rule_expert_s2",
    "comprehensive_rule_expert_s3",
    "comprehensive_rule_expert_s4",
    "rule_compliant_default",
    "carl_rule_default",
    "plant2_rule_default",
]

DISPLAY_NAME: dict[str, str] = {
    "idm_default": "IDM (default)",
    "idm_s1": "IDM-s1",
    "idm_s2": "IDM-s2",
    "idm_s3": "IDM-s3",
    "idm_s4": "IDM-s4",
    "ppo_lidar_default": "PPO",
    "carl_default": "CaRL",
    "plant2_default": "PlanT-2",
    "comprehensive_rule_expert_default": "IDM^e (default)",
    "comprehensive_rule_expert_s1": "IDM^e-s1",
    "comprehensive_rule_expert_s2": "IDM^e-s2",
    "comprehensive_rule_expert_s3": "IDM^e-s3",
    "comprehensive_rule_expert_s4": "IDM^e-s4",
    "rule_compliant_default": "PPO^e",
    "carl_rule_default": "CaRL^e",
    "plant2_rule_default": "PlanT-2^e",
}

# Unify naming across benches (no_entry / one_way / no_turn / direction use modified_idm).
BASELINE_ALIASES: dict[str, str] = {
    "modified_idm_default": "comprehensive_rule_expert_default",
    "modified_idm_s1": "comprehensive_rule_expert_s1",
    "modified_idm_s2": "comprehensive_rule_expert_s2",
    "modified_idm_s3": "comprehensive_rule_expert_s3",
    "modified_idm_s4": "comprehensive_rule_expert_s4",
    "idm_rule_default": "comprehensive_rule_expert_default",
    "idm_rule_s1": "comprehensive_rule_expert_s1",
    "idm_rule_s2": "comprehensive_rule_expert_s2",
    "idm_rule_s3": "comprehensive_rule_expert_s3",
    "idm_rule_s4": "comprehensive_rule_expert_s4",
}

# Drop known duplicate / obsolete policy tags.
SKIP_BASELINES: frozenset[str] = frozenset({
    "plant2_rule_default_old",
})

# PDD category grouping for overall SCR
# (mean over signs in group → mean over the 4 groups).
SIGN_GROUPS: dict[str, tuple[str, ...]] = {
    "priority": ("2.1", "2.3.1-2.3.3", "2.4", "2.5"),
    "prohibitory": ("3.1-3.2", "3.18.1-3.18.2", "3.24"),
    "mandatory": ("4.1.1-4.1.6", "4.2.1-4.2.3", "4.3", "4.6"),
    "special": ("5.7.1-5.7.2", "5.15.1-5.15.2", "5.19", "5.21", "5.31"),
}
GROUP_ORDER: list[str] = ["priority", "prohibitory", "mandatory", "special"]
GROUP_DISPLAY: dict[str, str] = {
    "priority": "Priority",
    "prohibitory": "Prohibitory",
    "mandatory": "Mandatory",
    "special": "Special",
}

# Speed-sign SCR matches fv_split_report A6 (F3∩F2); 3.24/5.31 also
# uniform-average over v_target_kmh (same as UNIFORM_SIGNS in recount_filtered_metrics).
SPEED_CSV = (
    PER_SIGN / "speed_signs" / "benchmark_output" / "run_v61_a6"
    / "eval_fast" / "metrics_per_episode_test20.csv"
)
SPEED_CATALOG = (
    PER_SIGN / "speed_signs" / "benchmark_output" / "run_v61_a6" / "catalog.jsonl"
)
SPEED_SIGNS: frozenset[str] = frozenset({"3.24", "4.6", "5.21", "5.31"})
UNIFORM_SPEED_SIGNS: frozenset[str] = frozenset({"3.24", "5.31"})

# Paper SCR table layout (matches the spreadsheet template).
PAPER_SECTIONS: list[tuple[str, tuple[str, ...]]] = [
    ("Priority", SIGN_GROUPS["priority"]),
    ("Prohibitory", SIGN_GROUPS["prohibitory"]),
    ("Mandatory", SIGN_GROUPS["mandatory"]),
    ("Special", SIGN_GROUPS["special"]),
    (
        "Запрещенная полоса (направление)",
        ("3.1", "3.2", "3.18.1", "3.18.2"),
    ),
]

# Columns 1–8 base, 9–16 expert (same display names as in the paper sheet).
PAPER_BASE_BASELINES: list[str] = [
    "idm_default", "idm_s1", "idm_s2", "idm_s3", "idm_s4",
    "ppo_lidar_default", "carl_default", "plant2_default",
]
PAPER_EXPERT_BASELINES: list[str] = [
    "comprehensive_rule_expert_default",
    "comprehensive_rule_expert_s1",
    "comprehensive_rule_expert_s2",
    "comprehensive_rule_expert_s3",
    "comprehensive_rule_expert_s4",
    "rule_compliant_default",
    "carl_rule_default",
    "plant2_rule_default",
]
PAPER_COL_HEADERS: list[str] = [
    "IDM (default)", "IDM s1", "IDM s2", "IDM s3", "IDM s4",
    "PPO", "CaRL", "PlanT-2",
]


def _to_bool(s: str) -> bool:
    return s == "True"


def _to_float(s: str, default: Optional[float] = None) -> Optional[float]:
    if s is None or s == "":
        return default
    try:
        v = float(s)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _normalize_baseline(name: str) -> Optional[str]:
    if not name or name in SKIP_BASELINES:
        return None
    return BASELINE_ALIASES.get(name, name)


def route_completion(row: dict) -> Optional[float]:
    """CARLA-style RC. CSV route_completion is broken (always 0), so recompute."""
    if _to_bool(row.get("arrived_dest", "")):
        return 1.0
    rl = _to_float(row.get("route_length_m", ""), None)
    dist = _to_float(row.get("distance_travelled_m", ""), None)
    if rl is None or rl <= 0 or dist is None:
        return None
    return float(min(1.0, max(0.0, dist / rl)))


def _load_test_scene_ids(catalog: Path, codes: set[str]) -> set[str]:
    ids: set[str] = set()
    with catalog.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            code = str(row.get("pdd_code") or row.get("sign_code") or "").strip()
            if codes and code not in codes:
                continue
            sid = row.get("scene_id")
            if sid:
                ids.add(str(sid))
    return ids


# Shared CSV row cache (speed_signs CSV is reused for 3.24 / 4.6 / 5.21 / 5.31).
_CSV_ROW_CACHE: dict[Path, list[dict]] = {}
_SPEED_A6_CACHE: Optional[list[dict]] = None


def _iter_csv_rows(path: Path) -> list[dict]:
    cached = _CSV_ROW_CACHE.get(path)
    if cached is not None:
        return cached
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    _CSV_ROW_CACHE[path] = rows
    return rows


def _extract_seed(scene_uid: str, row: dict) -> int:
    m = re.search(r"_seed(\d+)_", scene_uid or "")
    if m:
        return int(m.group(1))
    return int(_to_float(row.get("seed", ""), 0) or 0)


def _is_rule_baseline(baseline: str) -> bool:
    # Matches recount_filtered_metrics: baseline name contains "rule"
    # (comprehensive_rule_expert_*, carl_rule_*, plant2_rule_*, rule_compliant_*).
    return "rule" in baseline


def _load_speed_a6_episodes() -> list[dict]:
    """Load speed test20 CSV, apply A6 = F3∩F2, attach v_target_kmh.

    SCR field = target_compliant_event (same as fv_split_report Compliance in-zone).
    """
    global _SPEED_A6_CACHE
    if _SPEED_A6_CACHE is not None:
        return _SPEED_A6_CACHE
    if not SPEED_CSV.exists():
        raise FileNotFoundError(f"speed metrics missing: {SPEED_CSV}")
    if not SPEED_CATALOG.exists():
        raise FileNotFoundError(f"speed catalog missing: {SPEED_CATALOG}")

    vtarget: dict[tuple[str, int], Optional[float]] = {}
    with SPEED_CATALOG.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (str(r.get("scene_id") or ""), int(r.get("seed") or 0))
            vt = r.get("v_target_kmh")
            vtarget[key] = None if vt is None or vt == "" else float(vt)

    raw: list[dict] = []
    for r in _iter_csv_rows(SPEED_CSV):
        if r.get("valid") not in ("", "True"):
            continue
        code = (r.get("pdd_code") or "").strip()
        if code not in SPEED_SIGNS:
            continue
        baseline = _normalize_baseline(r.get("baseline") or "")
        if baseline is None:
            continue
        uid = (r.get("scene_uid") or "").strip()
        sid = (r.get("scene_id") or "").strip()
        seed = _extract_seed(uid, r)
        in_zone = (
            _to_bool(r.get("target_in_zone", ""))
            or (_to_float(r.get("target_in_zone_steps", ""), 0.0) or 0.0) > 0
        )
        tce_raw = r.get("target_compliant_event", "")
        if tce_raw is None or tce_raw == "":
            tce: Optional[float] = None
        else:
            tce = 1.0 if _to_bool(str(tce_raw)) else 0.0
        comfort = _to_float(r.get("comfort", ""), None)
        if comfort is None:
            comfort = _to_float(r.get("frame_smooth_ratio", ""), None)
        raw.append({
            "pdd_code": code,
            "baseline": baseline,
            "scene_uid": uid,
            "scene_id": sid,
            "v_target_kmh": vtarget.get((sid, seed)),
            "efficiency": _to_float(r.get("driving_efficiency", ""), None),
            "comfort": comfort,
            "route_completion": route_completion(r),
            "collision": 1.0 if _to_bool(r.get("crashed", "")) else 0.0,
            # Speed SCR uses target compliance, not high-level sign_compliant_high.
            "sign_compliance": tce,
            "in_zone": in_zone,
            "arrived": 1.0 if _to_bool(r.get("arrived_dest", "")) else 0.0,
            "is_rule": _is_rule_baseline(baseline),
        })

    f2 = {ep["scene_uid"] for ep in raw if ep["in_zone"]}
    f3 = {
        ep["scene_uid"] for ep in raw
        if ep["is_rule"] and ep["arrived"] == 1.0 and ep["sign_compliance"] == 1.0
    }
    a6 = f2 & f3
    filtered = [ep for ep in raw if ep["scene_uid"] in a6]
    print(
        f"[speed-A6] test20 episodes={len(raw)}  F2={len(f2)}  F3={len(f3)}  "
        f"A6={len(a6)}  kept={len(filtered)}",
        flush=True,
    )
    _SPEED_A6_CACHE = filtered
    return filtered


def load_episodes(job: SignJob, baselines: Optional[set[str]]) -> list[dict]:
    # Speed signs: dedicated A6 path (shared across 3.24/4.6/5.21/5.31 jobs).
    if set(job.codes) <= SPEED_SIGNS and all(
        p.resolve() == SPEED_CSV.resolve() for p in job.sources
    ):
        want_codes = set(job.codes)
        out: list[dict] = []
        for ep in _load_speed_a6_episodes():
            if ep["pdd_code"] not in want_codes:
                continue
            if baselines is not None and ep["baseline"] not in baselines:
                continue
            out.append({
                "sign_label": job.label,
                "pdd_code": ep["pdd_code"],
                "baseline": ep["baseline"],
                "efficiency": ep["efficiency"],
                "comfort": ep["comfort"],
                "route_completion": ep["route_completion"],
                "collision": ep["collision"],
                "sign_compliance": ep["sign_compliance"],
                "in_zone": ep["in_zone"],
                "arrived": ep["arrived"],
                "v_target_kmh": ep["v_target_kmh"],
                "scene_uid": ep["scene_uid"],
                "scr_kind": "speed_a6",
            })
        return out

    missing = [p for p in job.sources if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing metrics CSV for {job.label}: "
            + ", ".join(str(p) for p in missing)
        )

    want_codes = set(job.codes)
    scene_ids: Optional[set[str]] = None
    if job.catalog is not None:
        if not job.catalog.exists():
            raise FileNotFoundError(
                f"missing test catalog for {job.label}: {job.catalog}"
            )
        scene_ids = _load_test_scene_ids(job.catalog, want_codes)
        if not scene_ids:
            raise RuntimeError(
                f"{job.label}: no test scenes in {job.catalog} for {sorted(want_codes)}"
            )

    out = []
    for metrics_csv in job.sources:
        for r in _iter_csv_rows(metrics_csv):
            if r.get("valid") not in ("", "True"):
                continue
            code = (r.get("pdd_code") or "").strip()
            if want_codes and code and code not in want_codes:
                continue
            if scene_ids is not None:
                sid = (r.get("scene_id") or "").strip()
                if sid not in scene_ids:
                    continue
            baseline = _normalize_baseline(r.get("baseline") or "")
            if baseline is None:
                continue
            if baselines is not None and baseline not in baselines:
                continue
            comfort = _to_float(r.get("comfort", ""), None)
            if comfort is None:
                comfort = _to_float(r.get("frame_smooth_ratio", ""), None)
            eff = _to_float(r.get("driving_efficiency", ""), None)
            rc = route_completion(r)
            out.append({
                "sign_label": job.label,
                "pdd_code": code or job.label,
                "baseline": baseline,
                "efficiency": eff,
                "comfort": comfort,
                "route_completion": rc,
                "collision": 1.0 if _to_bool(r.get("crashed", "")) else 0.0,
                "sign_compliance": (
                    1.0 if _to_bool(r.get("sign_compliant_high", "")) else 0.0
                ),
                "in_zone": (
                    _to_bool(r.get("target_in_zone", ""))
                    or (_to_float(r.get("target_in_zone_steps", ""), 0.0) or 0.0) > 0
                ),
                "arrived": 1.0 if _to_bool(r.get("arrived_dest", "")) else 0.0,
                "v_target_kmh": None,
                "scene_id": (r.get("scene_id") or "").strip(),
                "scene_uid": (r.get("scene_uid") or "").strip(),
                "scr_kind": "default",
            })
    return out


def _mean(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None and math.isfinite(v)]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _is_priority_sign(label: str) -> bool:
    return label in SIGN_GROUPS["priority"]


def _scr_for_rows(rows: list[dict], label: str) -> tuple[Optional[float], str]:
    """Return (SCR, mode) with the paper-correct denominator per sign family."""
    if _is_priority_sign(label):
        return _mean(r["sign_compliance"] for r in rows), "all"
    if label in UNIFORM_SPEED_SIGNS:
        by_vt: dict[Optional[float], list[dict]] = defaultdict(list)
        for r in rows:
            by_vt[r.get("v_target_kmh")].append(r)
        strata: list[Optional[float]] = []
        for vt, rs in by_vt.items():
            if vt is None:
                continue
            inz = [r for r in rs if r.get("in_zone")]
            strata.append(_mean(r["sign_compliance"] for r in inz))
        return _mean(strata), "speed_a6_uniform"
    if label in SPEED_SIGNS or (rows and rows[0].get("scr_kind") == "speed_a6"):
        inz = [r for r in rows if r.get("in_zone")]
        return _mean(r["sign_compliance"] for r in inz), "speed_a6_in_zone"
    inz = [r for r in rows if r.get("in_zone")]
    return _mean(r["sign_compliance"] for r in inz), "in_zone"


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    in_zone_rows = [r for r in rows if r.get("in_zone")]
    label = rows[0].get("sign_label", "")
    scr, scr_mode = _scr_for_rows(rows, label)
    return {
        "n": n,
        "n_in_zone": len(in_zone_rows),
        "efficiency": _mean(r["efficiency"] for r in rows),
        "comfort": _mean(r["comfort"] for r in rows),
        "route_completion": _mean(r["route_completion"] for r in rows),
        "collision": _mean(r["collision"] for r in rows),
        "sign_compliance": scr,
        "sign_compliance_mode": scr_mode,
        "dest_rate": _mean(r["arrived"] for r in rows),
    }


def baseline_sort_key(b: str) -> tuple:
    try:
        return (0, BASELINE_ORDER.index(b), b)
    except ValueError:
        return (1, 0, b)


def sign_sort_key(label: str) -> tuple:
    order = [j.label for j in READY_JOBS]
    try:
        return (0, order.index(label), label)
    except ValueError:
        return (1, 0, label)


def fmt(v: Optional[float], kind: str) -> str:
    if v is None:
        return ""
    if kind == "eff":
        return f"{v:.2f}"
    if kind == "comf":
        return f"{v:.3f}"
    # rates as percent
    return f"{100.0 * v:.1f}"


METRIC_COLS = [
    ("efficiency", "Eff.", "eff"),
    ("comfort", "Comfort", "comf"),
    ("route_completion", "RC (%)", "pct"),
    ("collision", "Coll. (%)", "pct"),
    ("sign_compliance", "SCR (%)", "pct"),
]


def write_csv(path: Path, rows: list[dict], extra_keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = extra_keys + ["n"] + [k for k, _, _ in METRIC_COLS] + ["dest_rate"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in extra_keys}
            out["n"] = row.get("n", 0)
            for k, _, _ in METRIC_COLS:
                v = row.get(k)
                out[k] = "" if v is None else round(float(v), 6)
            dr = row.get("dest_rate")
            out["dest_rate"] = "" if dr is None else round(float(dr), 6)
            w.writerow(out)


def markdown_table(title: str, rows: list[dict], key_cols: list[tuple[str, str]]) -> str:
    headers = [h for _, h in key_cols] + ["n"] + [h for _, h, _ in METRIC_COLS]
    lines = [f"## {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        cells = []
        for k, _ in key_cols:
            if k == "baseline":
                cells.append(DISPLAY_NAME.get(row[k], row[k]))
            else:
                cells.append(str(row.get(k, "")))
        cells.append(str(row.get("n", 0)))
        for k, _, kind in METRIC_COLS:
            cells.append(fmt(row.get(k), kind))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def macro_average(per_sign: dict[str, dict]) -> dict:
    """Unweighted mean over sign labels that have n>0."""
    items = [m for m in per_sign.values() if m.get("n", 0) > 0]
    if not items:
        return {"n": 0}
    out = {"n": sum(m["n"] for m in items)}
    for k, _, _ in METRIC_COLS:
        out[k] = _mean(m.get(k) for m in items)
    out["dest_rate"] = _mean(m.get("dest_rate") for m in items)
    return out


def per_group_scr(per_sign: dict[str, dict]) -> dict[str, Optional[float]]:
    """Unweighted mean SCR within each PDD category (over available signs)."""
    out: dict[str, Optional[float]] = {}
    for group in GROUP_ORDER:
        labels = SIGN_GROUPS[group]
        scrs = []
        for label in labels:
            m = per_sign.get(label)
            if m is None or m.get("n", 0) <= 0:
                continue
            v = m.get("sign_compliance")
            if v is not None and math.isfinite(v):
                scrs.append(float(v))
        out[group] = _mean(scrs)
    return out


def category_scr(per_sign: dict[str, dict]) -> Optional[float]:
    """Overall SCR = mean of priority/prohibitory/mandatory/special means."""
    return _mean(per_group_scr(per_sign).values())


def markdown_group_scr_table(rows: list[dict]) -> str:
    """Compact Planner × Group SCR table (percent)."""
    lines = [
        "## Per-group SCR (inputs to overall SCR)",
        "",
        "| Planner | Group | n_signs | SCR (%) |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {planner} | {group} | {n} | {scr} |".format(
                planner=DISPLAY_NAME.get(row["baseline"], row["baseline"]),
                group=row["group"],
                n=row["n_signs"],
                scr=fmt(row.get("sign_compliance"), "pct"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _scr_lookup(
    by_sign_base: dict[tuple[str, str], dict],
    sign: str,
    baseline: str,
) -> Optional[float]:
    m = by_sign_base.get((sign, baseline))
    if m is None or m.get("n", 0) <= 0:
        return None
    v = m.get("sign_compliance")
    if v is None or not math.isfinite(v):
        return None
    return float(v)


def _fmt_scr_cell(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{v:.2f}"


def write_paper_scr_tables(
    out_dir: Path,
    by_sign_base_agg: dict[tuple[str, str], dict],
    loaded_main_labels: list[str],
) -> tuple[Path, Path]:
    """Write paper-layout SCR table (CSV + MD).

    Columns: 8 base planners + 8 expert planners (same display names).
    Rows: Priority / Prohibitory / Mandatory / Special signs, then the
    «Запрещенная полоса» detail block, then total (# main signs loaded).
    """
    all_baselines = PAPER_BASE_BASELINES + PAPER_EXPERT_BASELINES
    # CSV uses distinct headers for the expert block.
    csv_headers = (
        ["subgroup", "sign"]
        + [f"base:{h}" for h in PAPER_COL_HEADERS]
        + [f"expert:{h}" for h in PAPER_COL_HEADERS]
    )
    csv_rows: list[dict] = []
    md_lines = [
        "# Sign Compliance (SCR) — paper layout",
        "",
        "Values = SCR on the test split. "
        "Priority: overall Sign compliance SR. "
        "Speed (3.24/4.6/5.21/5.31): A6 (F3∩F2) + target_compliant_event in-zone "
        "(3.24/5.31 uniform over v_target). "
        "Other signs: Sign compliance (in-zone) via sign_compliant_high.",
        "Columns 1–8: base planners. Columns 9–16: rule/expert counterparts "
        "(IDM→`comprehensive_rule_expert`, PPO→`rule_compliant`, "
        "CaRL→`carl_rule`, PlanT-2→`plant2_rule`).",
        "Empty cell = sign not loaded / missing metrics / no episodes in denominator.",
        "",
        "| subgroup | sign | "
        + " | ".join(PAPER_COL_HEADERS)
        + " | "
        + " | ".join(PAPER_COL_HEADERS)
        + " |",
        "| --- | --- | "
        + " | ".join(["---"] * 16)
        + " |",
        "| | | "
        + " | ".join(["*base*"] * 8)
        + " | "
        + " | ".join(["*expert*"] * 8)
        + " |",
    ]

    main_sign_set = {lab for _, labs in PAPER_SECTIONS[:4] for lab in labs}

    for section, labels in PAPER_SECTIONS:
        for i, sign in enumerate(labels):
            cells = [_scr_lookup(by_sign_base_agg, sign, b) for b in all_baselines]
            csv_row = {
                "subgroup": section if i == 0 else "",
                "sign": sign,
            }
            for h, v, prefix in (
                *[(h, cells[j], "base") for j, h in enumerate(PAPER_COL_HEADERS)],
                *[(h, cells[8 + j], "expert") for j, h in enumerate(PAPER_COL_HEADERS)],
            ):
                csv_row[f"{prefix}:{h}"] = "" if v is None else round(v, 6)
            csv_rows.append(csv_row)

            md_lines.append(
                "| {sub} | {sign} | {vals} |".format(
                    sub=section if i == 0 else "",
                    sign=sign,
                    vals=" | ".join(_fmt_scr_cell(v) for v in cells),
                )
            )

        # Category mean row (only for the four main groups).
        if section in GROUP_DISPLAY.values():
            # Map display name back to group key.
            gkey = next(k for k, v in GROUP_DISPLAY.items() if v == section)
            mean_cells: list[Optional[float]] = []
            for b in all_baselines:
                per_sign = {
                    lab: by_sign_base_agg[(lab, b)]
                    for lab in SIGN_GROUPS[gkey]
                    if (lab, b) in by_sign_base_agg
                }
                mean_cells.append(
                    _mean(
                        m.get("sign_compliance")
                        for m in per_sign.values()
                        if m.get("n", 0) > 0
                    )
                )
            csv_row = {"subgroup": "", "sign": f"{section} mean"}
            for j, h in enumerate(PAPER_COL_HEADERS):
                v = mean_cells[j]
                csv_row[f"base:{h}"] = "" if v is None else round(v, 6)
            for j, h in enumerate(PAPER_COL_HEADERS):
                v = mean_cells[8 + j]
                csv_row[f"expert:{h}"] = "" if v is None else round(v, 6)
            csv_rows.append(csv_row)
            md_lines.append(
                "| | **{lab} mean** | {vals} |".format(
                    lab=section,
                    vals=" | ".join(
                        f"**{_fmt_scr_cell(v)}**" if v is not None else ""
                        for v in mean_cells
                    ),
                )
            )

    # Overall SCR = mean of the 4 category means (same as category_scr).
    overall_cells: list[Optional[float]] = []
    for b in all_baselines:
        per_sign = {
            lab: by_sign_base_agg[(lab, b)]
            for lab in main_sign_set
            if (lab, b) in by_sign_base_agg
        }
        overall_cells.append(category_scr(per_sign))
    n_loaded = sum(1 for lab in main_sign_set if any(
        (lab, b) in by_sign_base_agg for b in all_baselines
    ))
    csv_row = {"subgroup": "total", "sign": str(n_loaded)}
    for j, h in enumerate(PAPER_COL_HEADERS):
        v = overall_cells[j]
        csv_row[f"base:{h}"] = "" if v is None else round(v, 6)
    for j, h in enumerate(PAPER_COL_HEADERS):
        v = overall_cells[8 + j]
        csv_row[f"expert:{h}"] = "" if v is None else round(v, 6)
    csv_rows.append(csv_row)
    md_lines.append(
        "| **total** | **{n}** | {vals} |".format(
            n=n_loaded,
            vals=" | ".join(
                f"**{_fmt_scr_cell(v)}**" if v is not None else ""
                for v in overall_cells
            ),
        )
    )
    md_lines.append("")
    md_lines.append(
        f"Main signs present in this run: "
        f"{', '.join(lab for lab in loaded_main_labels if lab in main_sign_set) or '(none)'}."
    )
    md_lines.append("")

    csv_path = out_dir / "scr_paper_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_headers)
        w.writeheader()
        for row in csv_rows:
            w.writerow(row)

    md_path = out_dir / "scr_paper_table.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return csv_path, md_path


def job_status(job: SignJob) -> str:
    missing_src = [p for p in job.sources if not p.exists()]
    if missing_src:
        return f"MISSING csv ({len(missing_src)}/{len(job.sources)})"
    if job.catalog is not None and not job.catalog.exists():
        return "MISSING catalog"
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--only",
        default=None,
        help="Comma-separated sign labels to include (default: all ready signs)",
    )
    ap.add_argument(
        "--baselines",
        default=None,
        help="Comma-separated baselines to keep (default: all)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for CSV/MD "
             "(default: per_sign_bench/benchmark_output/ready_test_summary)",
    )
    ap.add_argument(
        "--print-md",
        action="store_true",
        help="Also print markdown tables to stdout",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List configured sign jobs and source readiness, then exit",
    )
    args = ap.parse_args()

    if args.list:
        for j in READY_JOBS:
            src = "; ".join(
                str(p.relative_to(PER_SIGN)) if p.is_relative_to(PER_SIGN) else str(p)
                for p in j.sources
            )
            cat = ""
            if j.catalog is not None:
                cat = f"  catalog={j.catalog.relative_to(PER_SIGN)}"
            print(f"{j.label:16s}  {job_status(j):24s}  codes={list(j.codes)}")
            print(f"  {'':16s}  {src}{cat}")
        return

    jobs = READY_JOBS
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        jobs = [j for j in jobs if j.label in want]
        missing = want - {j.label for j in jobs}
        if missing:
            print(f"ERROR: unknown labels: {sorted(missing)}", file=sys.stderr)
            sys.exit(2)

    baselines: Optional[set[str]] = None
    if args.baselines:
        baselines = {x.strip() for x in args.baselines.split(",") if x.strip()}
        # Also accept aliases on the CLI.
        baselines = {
            _normalize_baseline(b) or b for b in baselines
        }

    out_dir = (
        args.out or (PER_SIGN / "benchmark_output" / "ready_test_summary")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    loaded_labels: list[str] = []
    for job in jobs:
        try:
            rows = load_episodes(job, baselines)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[warn] {e}", file=sys.stderr)
            continue
        print(f"[load] {job.label}: {len(rows)} episodes from {len(job.sources)} csv(s)")
        if not rows:
            print(f"[warn] {job.label}: 0 episodes after filters", file=sys.stderr)
            continue
        all_rows.extend(rows)
        loaded_labels.append(job.label)

    # Detail rows (3.1 / 3.2 / 3.18.x) for the paper «Запрещенная полоса» block only.
    detail_rows: list[dict] = []
    for job in DETAIL_JOBS:
        try:
            rows = load_episodes(job, baselines)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[warn][detail] {e}", file=sys.stderr)
            continue
        print(f"[load:detail] {job.label}: {len(rows)} episodes")
        detail_rows.extend(rows)

    if not all_rows:
        print("ERROR: no episodes loaded", file=sys.stderr)
        sys.exit(2)

    # --- per (sign, baseline)
    by_sign_base: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_base: dict[str, list[dict]] = defaultdict(list)
    by_sign: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_sign_base[(r["sign_label"], r["baseline"])].append(r)
        by_base[r["baseline"]].append(r)
        by_sign[r["sign_label"]].append(r)

    # Aggregates used by the paper SCR table (main + detail labels).
    by_sign_base_agg: dict[tuple[str, str], dict] = {
        (sign, base): aggregate(rs) for (sign, base), rs in by_sign_base.items()
    }
    detail_by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in detail_rows:
        detail_by[(r["sign_label"], r["baseline"])].append(r)
    for key, rs in detail_by.items():
        by_sign_base_agg[key] = aggregate(rs)

    per_sign_base_rows = []
    for (sign, base), rs in sorted(
        by_sign_base.items(),
        key=lambda kv: (sign_sort_key(kv[0][0]), baseline_sort_key(kv[0][1])),
    ):
        m = aggregate(rs)
        per_sign_base_rows.append({"sign": sign, "baseline": base, **m})

    # micro overall per baseline (SCR overridden by category average below)
    micro_rows = []
    for base in sorted(by_base.keys(), key=baseline_sort_key):
        micro_rows.append({"baseline": base, "avg": "micro", **aggregate(by_base[base])})

    # macro overall per baseline = mean of per-sign means
    macro_rows = []
    per_group_scr_rows = []
    for base in sorted(by_base.keys(), key=baseline_sort_key):
        per_sign = {
            sign: aggregate(rs)
            for (sign, b), rs in by_sign_base.items()
            if b == base
        }
        m = macro_average(per_sign)
        # Overall SCR: mean(group means), group = mean of per-sign SCRs.
        group_scrs = per_group_scr(per_sign)
        m["sign_compliance"] = category_scr(per_sign)
        macro_rows.append({"baseline": base, "avg": "macro", **m})
        for group in GROUP_ORDER:
            per_group_scr_rows.append({
                "baseline": base,
                "group": group,
                "n_signs": sum(
                    1 for lab in SIGN_GROUPS[group]
                    if per_sign.get(lab, {}).get("n", 0) > 0
                ),
                "sign_compliance": group_scrs.get(group),
            })

    # Micro overall also uses category SCR (not episode-weighted SCR).
    for row in micro_rows:
        per_sign = {
            sign: aggregate(rs)
            for (sign, b), rs in by_sign_base.items()
            if b == row["baseline"]
        }
        row["sign_compliance"] = category_scr(per_sign)

    # per-sign (all baselines pooled) — useful sanity check
    per_sign_rows = []
    for sign in sorted(by_sign.keys(), key=sign_sort_key):
        per_sign_rows.append({"sign": sign, **aggregate(by_sign[sign])})

    write_csv(out_dir / "per_sign_baseline.csv", per_sign_base_rows, ["sign", "baseline"])
    write_csv(out_dir / "overall_micro_by_baseline.csv", micro_rows, ["baseline", "avg"])
    write_csv(out_dir / "overall_macro_by_baseline.csv", macro_rows, ["baseline", "avg"])
    write_csv(out_dir / "per_sign_all_baselines.csv", per_sign_rows, ["sign"])

    # Per-group SCR CSV (custom columns).
    gpath = out_dir / "per_group_scr_by_baseline.csv"
    with gpath.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["baseline", "group", "n_signs", "sign_compliance"]
        )
        w.writeheader()
        for row in per_group_scr_rows:
            out = {
                "baseline": row["baseline"],
                "group": row["group"],
                "n_signs": row["n_signs"],
                "sign_compliance": (
                    "" if row["sign_compliance"] is None
                    else round(float(row["sign_compliance"]), 6)
                ),
            }
            w.writerow(out)

    paper_csv, paper_md = write_paper_scr_tables(
        out_dir, by_sign_base_agg, loaded_labels
    )

    skipped = [j.label for j in jobs if j.label not in loaded_labels]
    md_parts = [
        "# Ready-sign test-set metrics",
        "",
        f"Signs loaded ({len(loaded_labels)}): {', '.join(loaded_labels)}",
    ]
    if skipped:
        md_parts.append(f"Skipped (missing data): {', '.join(skipped)}")
    md_parts += [
        "",
        "RC = 1 if arrived else distance_travelled / route_length "
        "(CSV `route_completion` is always 0).",
        "Per-sign SCR: priority → overall SR; "
        "speed → A6 + target_compliant_event in-zone "
        "(3.24/5.31 uniform v_target); "
        "others → sign_compliant_high in-zone.",
        "Overall SCR = mean of category means "
        "(priority / prohibitory / mandatory / special), "
        "each category = unweighted mean of its per-sign SCRs.",
        "Collision = crash rate. Efficiency / Comfort = episode means.",
        "Speed signs (3.24 / 4.6 / 5.21 / 5.31) from "
        "`speed_signs/.../metrics_per_episode_test20.csv` (separate rows).",
        "Detour (4.2.1–4.2.3) pooled mean from `detour_sign/.../eval_test20`.",
        "`modified_idm_*` / `idm_rule_*` aliased to `comprehensive_rule_expert_*`.",
        "",
        markdown_table(
            "Overall (micro-average across ready signs; SCR = category mean)",
            micro_rows,
            [("baseline", "Planner")],
        ),
        markdown_table(
            "Overall (macro-average across ready signs; SCR = category mean)",
            macro_rows,
            [("baseline", "Planner")],
        ),
        markdown_group_scr_table(per_group_scr_rows),
        markdown_table(
            "Per sign × baseline",
            per_sign_base_rows,
            [("sign", "Sign"), ("baseline", "Planner")],
        ),
        "",
        f"Paper SCR layout: `{paper_md.name}` / `{paper_csv.name}`.",
        "",
    ]
    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md_parts), encoding="utf-8")

    print()
    print(f"Wrote {out_dir}/")
    print(f"  overall_micro_by_baseline.csv  ({len(micro_rows)} planners)")
    print(f"  overall_macro_by_baseline.csv  ({len(macro_rows)} planners)")
    print(f"  per_group_scr_by_baseline.csv  ({len(per_group_scr_rows)} rows)")
    print(f"  per_sign_baseline.csv          ({len(per_sign_base_rows)} rows)")
    print(f"  per_sign_all_baselines.csv     ({len(per_sign_rows)} signs)")
    print(f"  {paper_csv.name}")
    print(f"  {paper_md.name}")
    print(f"  summary.md")
    if skipped:
        print(f"  skipped: {skipped}")

    if args.print_md:
        print()
        print(paper_md.read_text(encoding="utf-8"))
        print(markdown_table(
            "Overall (macro-average; SCR = category mean)",
            macro_rows,
            [("baseline", "Planner")],
        ))


if __name__ == "__main__":
    main()
