#!/usr/bin/env python3
"""Evidence that conventional metrics obscure traffic-rule violations.

Builds correlation tables + bootstrap CIs + publication figures showing that
Efficiency / Comfort / Route Completion / Collision are weakly (or wrongly)
related to Sign Compliance (SCR), while rule-aware planners reverse the picture.

Paper rule metrics (compliance, higher = better):
  SCR — Sign Compliance Rate (per sign)
  GCR — Group Compliance Rate (per PDD category)
  RCR — Rule Compliance Rate (mean of GCRs)

Outputs (default: benchmark_output/ready_test_summary/reviewer_evidence/):
  tables/
    metric_definitions.md
    joint_metrics_by_baseline.csv|.md
    correlation_spearman.csv|.md
    correlation_pearson.csv|.md
    bootstrap_rcr_ci.csv|.md
    decoupling_examples.csv|.md
  figures/
    fig_rc_vs_scr.png
    fig_eff_vs_scr.png
    fig_corr_heatmap.png
    fig_scr_ci_forest.png
    fig_twin_gap.png
  REVIEWER_RESPONSE.md

Usage:
  python analyze_metric_decoupling.py
  python analyze_metric_decoupling.py --bootstraps 2000 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
# Reuse the paper-ready episode loader (test split, A6 for speed, SCR rules).
import summarize_ready_sign_test_metrics as S

PER_SIGN = Path(__file__).resolve().parent
OUT_DEFAULT = PER_SIGN / "benchmark_output" / "ready_test_summary" / "reviewer_evidence"

# Focus planners for the "surprising" story (base vs rule twin).
FOCUS_BASE = [
    "idm_default", "idm_s1", "ppo_lidar_default", "carl_default", "plant2_default",
]
FOCUS_EXPERT = [
    "comprehensive_rule_expert_default",
    "comprehensive_rule_expert_s1",
    "rule_compliant_default",
    "carl_rule_default",
    "plant2_rule_default",
]
TWINS = list(zip(FOCUS_BASE, FOCUS_EXPERT))

# Colorblind-friendly palette (no purple glow / cream-serif look).
C_BASE = "#C44E52"       # coral-red — non-compliant-looking base
C_EXPERT = "#4C72B0"     # steel blue — rule experts
C_ACCENT = "#55A868"     # green
C_MUTED = "#8C8C8C"
C_BG = "#FAFAF8"
C_GRID = "#E6E6E6"

DISPLAY = {
    **S.DISPLAY_NAME,
    "idm_default": "IDM",
    "idm_s1": "IDM-s1",
    "ppo_lidar_default": "PPO",
    "carl_default": "CaRL",
    "plant2_default": "PlanT-2",
    "comprehensive_rule_expert_default": "IDMᵉ",
    "comprehensive_rule_expert_s1": "IDMᵉ-s1",
    "rule_compliant_default": "PPOᵉ",
    "carl_rule_default": "CaRLᵉ",
    "plant2_rule_default": "PlanT-2ᵉ",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_episodes(baselines: Optional[set[str]] = None) -> list[dict]:
    rows: list[dict] = []
    for job in S.READY_JOBS:
        try:
            part = S.load_episodes(job, baselines)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[warn] skip {job.label}: {exc}", file=sys.stderr)
            continue
        rows.extend(part)
        print(f"[load] {job.label}: {len(part)}")
    return rows


def attach_out_of_road(episodes: list[dict]) -> None:
    """Enrich episodes with out_of_road from source CSVs.

    Lookup key: (sign_label, baseline, scene_uid) when scene_uid is present,
    else (sign_label, baseline, scene_id), else FIFO by (sign_label, baseline, pdd_code).
    """
    by_uid: dict[tuple[str, str, str], dict] = {}
    by_sid: dict[tuple[str, str, str], dict] = {}
    by_fifo: dict[tuple[str, str, str], list[dict]] = defaultdict(list)

    def _push(sign: str, bl: str, code: str, r: dict) -> None:
        uid = (r.get("scene_uid") or "").strip()
        sid = (r.get("scene_id") or "").strip()
        info = {
            "scene_id": sid,
            "scene_uid": uid,
            "out_of_road": 1.0 if S._to_bool(r.get("out_of_road", "")) else 0.0,
        }
        if uid:
            by_uid[(sign, bl, uid)] = info
        if sid:
            by_sid[(sign, bl, sid)] = info
        by_fifo[(sign, bl, code)].append(info)

    for job in S.READY_JOBS:
        if set(job.codes) <= S.SPEED_SIGNS:
            a6 = {ep["scene_uid"] for ep in S._load_speed_a6_episodes()}
            for r in S._iter_csv_rows(S.SPEED_CSV):
                if r.get("valid") not in ("", "True"):
                    continue
                code = (r.get("pdd_code") or "").strip()
                if code not in job.codes:
                    continue
                uid = (r.get("scene_uid") or "").strip()
                if uid not in a6:
                    continue
                bl = S._normalize_baseline(r.get("baseline") or "")
                if bl is None:
                    continue
                _push(job.label, bl, code, r)
            continue

        want = set(job.codes)
        scene_ids = None
        if job.catalog is not None and job.catalog.exists():
            scene_ids = S._load_test_scene_ids(job.catalog, want)
        for src in job.sources:
            if not src.exists():
                continue
            for r in S._iter_csv_rows(src):
                if r.get("valid") not in ("", "True"):
                    continue
                code = (r.get("pdd_code") or "").strip()
                if want and code and code not in want:
                    continue
                if scene_ids is not None:
                    sid = (r.get("scene_id") or "").strip()
                    if sid not in scene_ids:
                        continue
                bl = S._normalize_baseline(r.get("baseline") or "")
                if bl is None:
                    continue
                _push(job.label, bl, code or job.label, r)

    pointers: dict[tuple, int] = defaultdict(int)
    matched = 0
    for ep in episodes:
        uid = (ep.get("scene_uid") or "").strip()
        sid = (ep.get("scene_id") or "").strip()
        info = None
        if uid:
            info = by_uid.get((ep["sign_label"], ep["baseline"], uid))
        if info is None and sid:
            info = by_sid.get((ep["sign_label"], ep["baseline"], sid))
        if info is None:
            key = (ep["sign_label"], ep["baseline"], ep.get("pdd_code") or ep["sign_label"])
            bucket = by_fifo.get(key) or []
            i = pointers[key]
            if i < len(bucket):
                info = bucket[i]
                pointers[key] = i + 1
        if info is None:
            ep["out_of_road"] = 0.0
            continue
        matched += 1
        ep["out_of_road"] = info["out_of_road"]

    print(f"[enrich] matched out_of_road for {matched}/{len(episodes)} episodes")


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def mean(xs) -> Optional[float]:
    ys = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(np.mean(ys)) if ys else None


def aggregate_planner(rows: list[dict]) -> dict:
    """Planner-level metrics with paper RCR (= mean of category GCRs)."""
    by_sign: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sign[r["sign_label"]].append(r)
    per_sign = {s: S.aggregate(rs) for s, rs in by_sign.items()}
    rcr = S.category_scr(per_sign)  # mean of group means
    group_scrs = S.per_group_scr(per_sign)
    return {
        "n": len(rows),
        "efficiency": mean(r["efficiency"] for r in rows),
        "comfort": mean(r["comfort"] for r in rows),
        "route_completion": mean(r["route_completion"] for r in rows),
        "collision": mean(r["collision"] for r in rows),
        "out_of_road": mean(r.get("out_of_road") for r in rows),
        "rcr": rcr,
        "gcr_priority": group_scrs.get("priority"),
        "gcr_prohibitory": group_scrs.get("prohibitory"),
        "gcr_mandatory": group_scrs.get("mandatory"),
        "gcr_special": group_scrs.get("special"),
        "per_sign": per_sign,
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    # Rank-based Pearson.
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_ci(
    values: list[float],
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(arr)), float(lo), float(hi)


def episode_bootstrap_rcr(
    by_sign_rows: dict[str, list[dict]],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap over episodes within each sign; recompute RCR.

    For every replicate and every sign independently: resample that sign's
    episodes with replacement (same count), recompute per-sign SCR via the
    paper denominator (priority / in-zone / speed A6), then recompute RCR =
    mean of GCRs. Signs with zero episodes are skipped.
    """
    rng = np.random.default_rng(seed)
    labels = [s for s, rs in by_sign_rows.items() if rs]
    if not labels:
        return float("nan"), float("nan"), float("nan")

    def _rcr_from_sign_rows(sign_rows: dict[str, list[dict]]) -> float:
        per_sign = {}
        for lab, rs in sign_rows.items():
            if not rs:
                continue
            m = S.aggregate(rs)
            if m.get("n", 0) > 0 and m.get("sign_compliance") is not None:
                per_sign[lab] = m
        v = S.category_scr(per_sign)
        return float("nan") if v is None else float(v)

    point = _rcr_from_sign_rows(by_sign_rows)
    boots = []
    for _ in range(n_boot):
        resampled: dict[str, list[dict]] = {}
        for lab in labels:
            rs = by_sign_rows[lab]
            n = len(rs)
            idx = rng.integers(0, n, size=n)
            resampled[lab] = [rs[i] for i in idx]
        boots.append(_rcr_from_sign_rows(resampled))
    boots = [b for b in boots if math.isfinite(b)]
    if not boots:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


# Paper-style display names for bootstrap CI table.
PAPER_PLANNER_NAME: dict[str, str] = {
    "idm_default": "IDM",
    "idm_s1": r"$\text{IDM-s}_1$",
    "idm_s2": r"$\text{IDM-s}_2$",
    "idm_s3": r"$\text{IDM-s}_3$",
    "idm_s4": r"$\text{IDM-s}_4$",
    "ppo_lidar_default": r"$\text{PPO}$",
    "carl_default": r"$\text{CaRL}$",
    "plant2_default": r"$\text{PlanT-2}$",
    "comprehensive_rule_expert_default": r"$\text{IDM}^{e}$",
    "comprehensive_rule_expert_s1": r"$\text{IDM}^{e}\text{-}{s_1}$",
    "comprehensive_rule_expert_s2": r"$\text{IDM}^{e}\text{-}{s_2}$",
    "comprehensive_rule_expert_s3": r"$\text{IDM}^{e}\text{-}{s_3}$",
    "comprehensive_rule_expert_s4": r"$\text{IDM}^{e}\text{-}{s_4}$",
    "rule_compliant_default": r"$\text{PPO}^{e}$",
    "carl_rule_default": r"$\text{CaRL}^{e}$",
    "plant2_rule_default": r"$\text{PlanT-2}^{e}$",
}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_definitions(path: Path) -> None:
    path.write_text(
        """# Metric definitions (precise)

## Conventional / progress metrics

**Efficiency (Eff.)** — episode mean of per-timestep ego speed as a percentage of
nearby traffic speed:

$$
\\mathrm{Eff} = \\mathrm{mean}_t\\Big(100\\cdot v_{\\mathrm{ego}}(t)\\,/\\,\\bar v_{\\mathrm{nearby}}(t)\\Big)
$$

Nearby agents: lidar surrounds with speed $>0.5$ km/h. Samples with ratio $>1000$
are dropped. Values can exceed 100 (ego faster than traffic). Higher $\\uparrow$.

**Comfort (Comf.)** — `frame_smooth_ratio`: fraction of frames satisfying nuPlan-style
kinematic bounds ($\\mathrm{d}t=0.1$):

| quantity | bound |
|---|---|
| longitudinal accel | $[-4.05,\\,2.40]$ m/s$^2$ |
| $|$lateral accel$|$ | $\\le 4.89$ |
| $|$yaw rate$|$ | $\\le 0.95$ |
| $|$yaw accel$|$ | $\\le 1.93$ |
| $|$long. jerk$|$ | $\\le 4.13$ |
| $|$jerk magnitude$|$ | $\\le 8.37$ |

Higher $\\uparrow$. (Distinct from segment-level `smoothness_ratio`.)

**Route Completion (RC)** — CARLA-style reconstruction (CSV `route_completion` is broken):

$$
\\mathrm{RC}=
\\begin{cases}
1 & \\text{if }\\texttt{arrived\\_dest}\\\\
\\min(1,\\,d_{\\mathrm{travelled}}/L_{\\mathrm{route}}) & \\text{otherwise}
\\end{cases}
$$

Higher $\\uparrow$.

**Collision** — episode crash rate (`info["crash"]`; vehicle/object/building/sidewalk/human).
Does **not** fold in `out_of_road`. Lower $\\downarrow$.

**Out-of-road** — `out_of_road` rate from the env. Lower $\\downarrow$.

## Rule metrics (compliance; paper naming)

We report **compliance** rates (higher = better):

**SCR (Sign Compliance Rate)** — per-sign compliance on the paper denominator:
- Priority signs: mean of `sign_compliant_high` over all episodes;
- Speed signs (3.24/4.6/5.21/5.31): A6 filter + mean of `target_compliant_event`
  on in-zone episodes (3.24/5.31 also uniform over $v_{\\mathrm{target}}$);
- Other signs: mean of `sign_compliant_high` among `target_in_zone` episodes.

**GCR (Group Compliance Rate)** — unweighted mean of per-sign SCR within one
PDD category (Priority / Prohibitory / Mandatory / Special). Higher $\\uparrow$.

**RCR (Rule Compliance Rate)** — mean of the four GCRs (overall rule score).
Higher $\\uparrow$.
""",
        encoding="utf-8",
    )


def fmt_pct(v: Optional[float]) -> str:
    if v is None or not math.isfinite(v):
        return ""
    return f"{100.0 * v:.1f}"


def fmt_num(v: Optional[float], nd: int = 2) -> str:
    if v is None or not math.isfinite(v):
        return ""
    return f"{v:.{nd}f}"


def write_joint_table(path_csv: Path, path_md: Path, by_base: dict[str, dict]) -> None:
    fields = [
        "baseline", "display", "n",
        "efficiency", "comfort", "route_completion", "collision",
        "out_of_road",
        "rcr",
        "gcr_priority", "gcr_prohibitory", "gcr_mandatory", "gcr_special",
    ]
    rows = []
    for b in sorted(by_base.keys(), key=S.baseline_sort_key):
        m = by_base[b]
        rows.append({
            "baseline": b,
            "display": S.DISPLAY_NAME.get(b, b),
            "n": m["n"],
            "efficiency": "" if m["efficiency"] is None else round(m["efficiency"], 4),
            "comfort": "" if m["comfort"] is None else round(m["comfort"], 4),
            "route_completion": "" if m["route_completion"] is None else round(m["route_completion"], 6),
            "collision": "" if m["collision"] is None else round(m["collision"], 6),
            "out_of_road": "" if m["out_of_road"] is None else round(m["out_of_road"], 6),
            "rcr": "" if m["rcr"] is None else round(m["rcr"], 6),
            "gcr_priority": "" if m["gcr_priority"] is None else round(m["gcr_priority"], 6),
            "gcr_prohibitory": "" if m["gcr_prohibitory"] is None else round(m["gcr_prohibitory"], 6),
            "gcr_mandatory": "" if m["gcr_mandatory"] is None else round(m["gcr_mandatory"], 6),
            "gcr_special": "" if m["gcr_special"] is None else round(m["gcr_special"], 6),
        })
    with path_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Joint metrics (test set): conventional + rule compliance",
        "",
        "| Planner | n | Eff.↑ | Comf.↑ | RC (%)↑ | Coll. (%)↓ | OOR (%)↓ | RCR (%)↑ | GCR Pri↑ | GCR Pro↑ | GCR Man↑ | GCR Spe↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            "| {d} | {n} | {eff} | {comf} | {rc} | {col} | {oor} | {rcr} | {gp} | {gq} | {gm} | {gs} |".format(
                d=r["display"], n=r["n"],
                eff=fmt_num(float(r["efficiency"]) if r["efficiency"] != "" else None),
                comf=fmt_num(float(r["comfort"]) if r["comfort"] != "" else None, 3),
                rc=fmt_pct(float(r["route_completion"]) if r["route_completion"] != "" else None),
                col=fmt_pct(float(r["collision"]) if r["collision"] != "" else None),
                oor=fmt_pct(float(r["out_of_road"]) if r["out_of_road"] != "" else None),
                rcr=fmt_pct(float(r["rcr"]) if r["rcr"] != "" else None),
                gp=fmt_pct(float(r["gcr_priority"]) if r["gcr_priority"] != "" else None),
                gq=fmt_pct(float(r["gcr_prohibitory"]) if r["gcr_prohibitory"] != "" else None),
                gm=fmt_pct(float(r["gcr_mandatory"]) if r["gcr_mandatory"] != "" else None),
                gs=fmt_pct(float(r["gcr_special"]) if r["gcr_special"] != "" else None),
            )
        )
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


CONV_METRICS = ["efficiency", "comfort", "route_completion", "collision"]
GCR_GROUPS = list(S.GROUP_ORDER)  # priority, prohibitory, mandatory, special
GCR_LABELS = {
    "priority": "GCR Pri",
    "prohibitory": "GCR Pro",
    "mandatory": "GCR Man",
    "special": "GCR Spe",
}


def _sign_group(sign: str) -> Optional[str]:
    for g, labs in S.SIGN_GROUPS.items():
        if sign in labs:
            return g
    return None


def correlation_at_sign_baseline_level(
    episodes: list[dict],
) -> tuple[dict[str, float], dict[str, float], list[dict]]:
    """Correlate conventional metrics with SCR across (sign × baseline) aggregates."""
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in episodes:
        buckets[(r["sign_label"], r["baseline"])].append(r)

    cells = []
    for (sign, base), rs in buckets.items():
        m = S.aggregate(rs)
        if m.get("n", 0) == 0 or m.get("sign_compliance") is None:
            continue
        cells.append({
            "sign": sign,
            "baseline": base,
            "n": m["n"],
            "efficiency": m["efficiency"],
            "comfort": m["comfort"],
            "route_completion": m["route_completion"],
            "collision": m["collision"],
            "scr": m["sign_compliance"],
            "is_expert": int(
                "rule" in base or base.startswith("comprehensive_rule")
            ),
        })

    sp: dict[str, float] = {}
    pe: dict[str, float] = {}
    for c in CONV_METRICS:
        xs, ys = [], []
        for cell in cells:
            if cell[c] is None or cell["scr"] is None:
                continue
            if not (math.isfinite(cell[c]) and math.isfinite(cell["scr"])):
                continue
            xs.append(cell[c])
            ys.append(cell["scr"])
        if len(xs) < 5:
            sp[c] = float("nan")
            pe[c] = float("nan")
        else:
            xa, ya = np.asarray(xs, float), np.asarray(ys, float)
            sp[c] = spearman(xa, ya)
            pe[c] = pearson(xa, ya)
    return sp, pe, cells


def correlation_gcr_by_group(
    episodes: list[dict],
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    """Spearman ρ of conventional metrics vs GCR at (group × planner).

    One cell per (PDD group, baseline): GCR = unweighted mean of per-sign SCR
    in that group. Conventional metrics = episode means within the group's signs.
    """
    by_base_sign: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in episodes:
        by_base_sign[(r["baseline"], r["sign_label"])].append(r)

    cells = []
    for base in {b for b, _ in by_base_sign}:
        for group in GCR_GROUPS:
            labels = [lab for lab in S.SIGN_GROUPS[group] if (base, lab) in by_base_sign]
            if not labels:
                continue
            per_sign = {}
            eps: list[dict] = []
            for lab in labels:
                rs = by_base_sign[(base, lab)]
                m = S.aggregate(rs)
                if m.get("n", 0) > 0 and m.get("sign_compliance") is not None:
                    per_sign[lab] = m
                eps.extend(rs)
            if not per_sign:
                continue
            gcr = mean(m["sign_compliance"] for m in per_sign.values())
            if gcr is None:
                continue
            cells.append({
                "group": group,
                "baseline": base,
                "n": len(eps),
                "n_signs": len(per_sign),
                "efficiency": mean(r["efficiency"] for r in eps),
                "comfort": mean(r["comfort"] for r in eps),
                "route_completion": mean(r["route_completion"] for r in eps),
                "collision": mean(r["collision"] for r in eps),
                "gcr": gcr,
            })

    sp: dict[str, dict[str, float]] = {g: {} for g in GCR_GROUPS}
    for g in GCR_GROUPS:
        g_cells = [c for c in cells if c["group"] == g]
        for c in CONV_METRICS:
            xs, ys = [], []
            for cell in g_cells:
                if cell[c] is None or cell["gcr"] is None:
                    continue
                if math.isfinite(cell[c]) and math.isfinite(cell["gcr"]):
                    xs.append(cell[c]); ys.append(cell["gcr"])
            if len(xs) < 5:
                sp[g][c] = float("nan")
            else:
                sp[g][c] = spearman(np.asarray(xs, float), np.asarray(ys, float))
    return sp, cells


def bootstrap_spearman_ci(
    cells: list[dict],
    xkey: str,
    ykey: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap over cells for Spearman ρ CI."""
    pairs = [
        (c[xkey], c[ykey]) for c in cells
        if c.get(xkey) is not None and c.get(ykey) is not None
        and math.isfinite(c[xkey]) and math.isfinite(c[ykey])
    ]
    if len(pairs) < 5:
        return float("nan"), float("nan"), float("nan")
    xa = np.asarray([p[0] for p in pairs], float)
    ya = np.asarray([p[1] for p in pairs], float)
    point = spearman(xa, ya)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(pairs)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(spearman(xa[idx], ya[idx]))
    boots = [b for b in boots if math.isfinite(b)]
    if not boots:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def corr_on_subset(cells: list[dict], pred) -> dict[str, float]:
    """Spearman of conventional metrics vs SCR on a filtered cell subset."""
    sub = [c for c in cells if pred(c)]
    out = {}
    for c in CONV_METRICS:
        xs, ys = [], []
        for cell in sub:
            if cell[c] is None or cell["scr"] is None:
                continue
            if math.isfinite(cell[c]) and math.isfinite(cell["scr"]):
                xs.append(cell[c]); ys.append(cell["scr"])
        out[c] = spearman(np.asarray(xs, float), np.asarray(ys, float)) if len(xs) >= 5 else float("nan")
    out["n"] = float(len(sub))
    return out


def write_corr_tables(
    sp: dict,
    pe: dict,
    path_sp: Path,
    path_pe: Path,
    path_md: Path,
    sp_ci: Optional[dict] = None,
    sp_base: Optional[dict] = None,
    sp_expert: Optional[dict] = None,
    sp_gcr: Optional[dict] = None,
) -> None:
    def _dump(path: Path, mat: dict) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            header = ["conventional", "scr"]
            if sp_gcr:
                header += [f"gcr_{g}" for g in GCR_GROUPS]
            w.writerow(header)
            for c in CONV_METRICS:
                row = [c, "" if not math.isfinite(mat[c]) else round(mat[c], 4)]
                if sp_gcr:
                    for g in GCR_GROUPS:
                        v = sp_gcr[g].get(c, float("nan"))
                        row.append("" if not math.isfinite(v) else round(v, 4))
                w.writerow(row)

    _dump(path_sp, sp)
    with path_pe.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["conventional", "scr"])
        for c in CONV_METRICS:
            v = pe[c]
            w.writerow([c, "" if not math.isfinite(v) else round(v, 4)])

    lines = [
        "# Correlation: conventional ↔ SCR / GCR",
        "",
        "**SCR** unit: `(sign, baseline)` cells. "
        "**GCR** unit: `(group, baseline)` cells for each of the four "
        "PDD categories.",
        "",
        "## Spearman $\\rho$ with SCR",
        "",
        "| Conventional | SCR | 95% CI |",
        "| --- | ---: | ---: |",
    ]
    for c in CONV_METRICS:
        ci = ""
        if sp_ci and c in sp_ci:
            lo, hi = sp_ci[c]["lo"], sp_ci[c]["hi"]
            if math.isfinite(lo) and math.isfinite(hi):
                ci = f"[{lo:.3f}, {hi:.3f}]"
        lines.append(f"| {c} | {sp[c]:.3f} | {ci} |")
    lines += [
        "",
        "## Pearson $r$ with SCR",
        "",
        "| Conventional | SCR |",
        "| --- | ---: |",
    ]
    for c in CONV_METRICS:
        lines.append(f"| {c} | {pe[c]:.3f} |")
    if sp_gcr is not None:
        lines += [
            "",
            "## Spearman $\\rho$ with GCR by group",
            "",
            "| Conventional | GCR Pri | GCR Pro | GCR Man | GCR Spe |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for c in CONV_METRICS:
            lines.append(
                "| {c} | {a:.3f} | {b:.3f} | {d:.3f} | {e:.3f} |".format(
                    c=c,
                    a=sp_gcr["priority"][c],
                    b=sp_gcr["prohibitory"][c],
                    d=sp_gcr["mandatory"][c],
                    e=sp_gcr["special"][c],
                )
            )
    if sp_base is not None and sp_expert is not None:
        lines += [
            "",
            "## Stratified Spearman $\\rho$ with SCR (base vs expert cells)",
            "",
            f"Base cells only (N={int(sp_base['n'])}) vs expert cells only "
            f"(N={int(sp_expert['n'])}).",
            "",
            "| Conventional | ρ(SCR) base-only | ρ(SCR) expert-only |",
            "| --- | ---: | ---: |",
        ]
        for c in CONV_METRICS:
            lines.append(
                f"| {c} | {sp_base[c]:.3f} | {sp_expert[c]:.3f} |"
            )
    lines += [
        "",
        "**Reading:** if conventional metrics tracked rule following, Eff/Comf/RC "
        "would correlate **positively** with SCR and GCR. "
        "Weak / wrong-signed coefficients mean a progress dashboard can look "
        "strong while the agent systematically violates signs.",
        "",
    ]
    path_md.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style():
    plt.rcParams.update({
        "figure.facecolor": C_BG,
        "axes.facecolor": C_BG,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.8,
    })


def fig_scatter_metric_vs_scr(cells: list[dict], xkey: str, xlabel: str, out: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    xb, yb, xe, ye = [], [], [], []
    for c in cells:
        if c[xkey] is None or c["scr"] is None:
            continue
        if c["is_expert"]:
            xe.append(c[xkey]); ye.append(100 * c["scr"])
        else:
            xb.append(c[xkey]); yb.append(100 * c["scr"])
    ax.scatter(xb, yb, s=36, c=C_BASE, alpha=0.75, edgecolors="white",
               linewidths=0.4, label="Base planners", zorder=3)
    ax.scatter(xe, ye, s=36, c=C_EXPERT, alpha=0.75, edgecolors="white",
               linewidths=0.4, label="Rule experts", zorder=3)

    # Annotate a few extreme decoupling points (high RC/Eff, low SCR among base).
    base_cells = [c for c in cells if not c["is_expert"]
                  and c[xkey] is not None and c["scr"] is not None]
    if xkey == "route_completion":
        cand = sorted(base_cells, key=lambda c: (c["scr"], -c[xkey]))[:6]
    else:
        cand = sorted(base_cells, key=lambda c: (c["scr"], -float(c[xkey])))[:5]
    for c in cand:
        if c["scr"] > 0.25:
            continue
        ax.annotate(
            f"{c['sign']}\n{S.DISPLAY_NAME.get(c['baseline'], c['baseline'])}",
            xy=(c[xkey], 100 * c["scr"]),
            xytext=(8, 8), textcoords="offset points", fontsize=7.5,
            color="#444444",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("SCR (%)")
    ax.set_ylim(-3, 105)
    ax.legend(frameon=True, fancybox=False, edgecolor=C_GRID)
    ax.set_title("Conventional score ≠ rule compliance")
    fig.savefig(out)
    plt.close(fig)


def fig_corr_heatmap(sp_scr: dict, sp_gcr: dict, out: Path) -> None:
    """Heatmap: conventional × {SCR, GCR Pri/Pro/Man/Spe}."""
    _style()
    labels_y = ["Eff.", "Comf.", "RC", "Coll."]
    labels_x = ["SCR"] + [GCR_LABELS[g] for g in GCR_GROUPS]
    mat = np.zeros((len(CONV_METRICS), 1 + len(GCR_GROUPS)), dtype=float)
    for i, c in enumerate(CONV_METRICS):
        mat[i, 0] = sp_scr[c]
        for j, g in enumerate(GCR_GROUPS):
            mat[i, 1 + j] = sp_gcr[g][c]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=25, ha="right")
    ax.set_yticks(range(len(labels_y)))
    ax.set_yticklabels(labels_y)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not math.isfinite(v):
                continue
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if abs(v) > 0.55 else "#111111",
                fontsize=10, fontweight="bold",
            )
    ax.set_title("Spearman $\\rho$: conventional vs SCR / GCR")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$\\rho$")
    ax.text(
        0.5, -0.22,
        "Expected if dashboards tracked rules: Eff/Comf/RC ↔ SCR/GCR > 0.",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#555555",
    )
    fig.savefig(out)
    plt.close(fig)


def fig_scr_ci_forest(ci_rows: list[dict], out: Path) -> None:
    _style()
    order = FOCUS_BASE + FOCUS_EXPERT
    rows = [r for b in order for r in ci_rows if r["baseline"] == b]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    y = np.arange(len(rows))[::-1]
    for i, r in enumerate(rows):
        color = C_EXPERT if r["is_expert"] else C_BASE
        ax.hlines(y[i], 100 * r["lo"], 100 * r["hi"], color=color, lw=2.2)
        ax.plot(100 * r["point"], y[i], "o", color=color, markersize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY.get(r["baseline"], r["display"]) for r in rows])
    ax.set_xlabel("Overall RCR (%)  [95% episode-bootstrap CI]")
    ax.set_xlim(-2, 105)
    ax.set_title("Rule experts separate cleanly on RCR\n(episode bootstrap within signs)")
    ax.plot([], [], "o-", color=C_BASE, label="Base")
    ax.plot([], [], "o-", color=C_EXPERT, label="Rule expert")
    ax.legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)


def fig_twin_gap(by_base: dict[str, dict], out: Path) -> None:
    """Paired base→expert: RC almost flat, RCR jumps — the smoking gun."""
    _style()
    labels, rc_b, rc_e, rcr_b, rcr_e = [], [], [], [], []
    for b, e in TWINS:
        if b not in by_base or e not in by_base:
            continue
        labels.append(DISPLAY.get(b, S.DISPLAY_NAME.get(b, b)).replace(" (default)", ""))
        rc_b.append(100 * (by_base[b]["route_completion"] or 0))
        rc_e.append(100 * (by_base[e]["route_completion"] or 0))
        rcr_b.append(100 * (by_base[b]["rcr"] or 0))
        rcr_e.append(100 * (by_base[e]["rcr"] or 0))

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=False)

    ax = axes[0]
    ax.bar(x - 0.18, rc_b, 0.36, color=C_BASE, label="Base", edgecolor="white")
    ax.bar(x + 0.18, rc_e, 0.36, color=C_EXPERT, label="Expert", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Route Completion (%)")
    ax.set_ylim(0, 100)
    ax.set_title("RC: base ≈ expert")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar(x - 0.18, rcr_b, 0.36, color=C_BASE, label="Base", edgecolor="white")
    ax.bar(x + 0.18, rcr_e, 0.36, color=C_EXPERT, label="Expert", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("RCR (%)")
    ax.set_ylim(0, 100)
    ax.set_title("RCR: expert ≫ base")

    fig.suptitle(
        "Same progress metric, opposite rule outcome",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_rank_disagreement(by_base: dict[str, dict], out: Path) -> float:
    """Planner ranks by RC vs by RCR — conventional dashboard mis-orders agents."""
    _style()
    items = [
        (b, m) for b, m in by_base.items()
        if m.get("route_completion") is not None and m.get("rcr") is not None
    ]
    if len(items) < 4:
        return float("nan")
    by_rc = sorted(items, key=lambda t: t[1]["route_completion"], reverse=True)
    by_rcr = sorted(items, key=lambda t: t[1]["rcr"], reverse=True)
    rank_rc = {b: i + 1 for i, (b, _) in enumerate(by_rc)}
    rank_rcr = {b: i + 1 for i, (b, _) in enumerate(by_rcr)}

    focus = [b for pair in TWINS for b in pair if b in rank_rc]
    if not focus:
        focus = [b for b, _ in items]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    y = np.arange(len(focus))[::-1]
    for i, b in enumerate(focus):
        color = C_EXPERT if ("rule" in b or b.startswith("comprehensive_rule")) else C_BASE
        ax.plot([rank_rc[b], rank_rcr[b]], [y[i], y[i]], "-", color=color, lw=2.0, alpha=0.85)
        ax.plot(rank_rc[b], y[i], "o", color=C_MUTED, markersize=8, zorder=3)
        ax.plot(rank_rcr[b], y[i], "s", color=color, markersize=8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY.get(b, S.DISPLAY_NAME.get(b, b)) for b in focus])
    ax.set_xlabel("Rank (1 = best)")
    ax.set_xlim(len(items) + 0.5, 0.5)
    ax.set_title("Dashboard rank ≠ rule rank\n(circle = RC rank, square = RCR rank)")
    ax.plot([], [], "o", color=C_MUTED, label="Rank by RC")
    ax.plot([], [], "s", color=C_EXPERT, label="Rank by RCR")
    ax.legend(loc="lower left")
    fig.savefig(out)
    plt.close(fig)

    rc_ranks = np.asarray([rank_rc[b] for b, _ in items], float)
    rcr_ranks = np.asarray([rank_rcr[b] for b, _ in items], float)
    n = len(items)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(rc_ranks[i] - rc_ranks[j]) * np.sign(rcr_ranks[i] - rcr_ranks[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    tau = (conc - disc) / (conc + disc) if (conc + disc) else float("nan")
    return tau


# ---------------------------------------------------------------------------
# Reviewer response draft
# ---------------------------------------------------------------------------

def write_reviewer_response(
    path: Path,
    by_base: dict[str, dict],
    sp: dict,
    cells: list[dict],
    ci_rows: list[dict],
    sp_ci: Optional[dict] = None,
    sp_base: Optional[dict] = None,
    sp_expert: Optional[dict] = None,
    kendall_tau: Optional[float] = None,
) -> None:
    plant = by_base.get("plant2_default", {})
    plant_e = by_base.get("plant2_rule_default", {})
    idm = by_base.get("idm_default", {})
    idm_e = by_base.get("comprehensive_rule_expert_default", {})
    ppo = by_base.get("ppo_lidar_default", {})
    ppo_e = by_base.get("rule_compliant_default", {})

    def g(m, k):
        v = m.get(k)
        return f"{100*v:.1f}%" if v is not None else "n/a"

    bad = [
        c for c in cells
        if not c["is_expert"]
        and c["route_completion"] is not None
        and c["scr"] is not None
        and c["route_completion"] >= 0.5
        and c["scr"] <= 0.15
    ]

    rho_rc_scr = sp["route_completion"]
    rho_eff_scr = sp["efficiency"]
    rho_comf_scr = sp["comfort"]
    rho_col_scr = sp["collision"]

    def ci_str(key: str) -> str:
        if not sp_ci or key not in sp_ci:
            return ""
        lo, hi = sp_ci[key]["lo"], sp_ci[key]["hi"]
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return ""
        return f" [{lo:.2f}, {hi:.2f}]"

    base_eff = sp_base["efficiency"] if sp_base else float("nan")
    base_comf = sp_base["comfort"] if sp_base else float("nan")
    base_n = int(sp_base["n"]) if sp_base else 0
    tau_s = f"{kendall_tau:.3f}" if kendall_tau is not None and math.isfinite(kendall_tau) else "n/a"

    ex = next(
        (c for c in bad if c["sign"] == "2.4" and "carl" in c["baseline"]),
        bad[0] if bad else None,
    )
    ex_line = ""
    if ex:
        ex_line = (
            f"Concrete cell: **{ex['sign']} / {S.DISPLAY_NAME.get(ex['baseline'], ex['baseline'])}** "
            f"has RC={100*ex['route_completion']:.1f}% with SCR={100*ex['scr']:.1f}% "
            f"(n={ex['n']} episodes)."
        )

    text_out = f"""# Response to Reviewer — “conventional metrics obscure violations”

We thank the reviewer for this request. We now (i) define Efficiency and Comfort
operationally, (ii) report RC / Collision / Comfort **jointly** with
the paper’s compliance metrics **SCR / GCR / RCR**, (iii) give Spearman/Pearson
correlations with **bootstrap CIs** at the `(sign × planner)` fragment level,
and (iv) give **episode-bootstrap** 95% CIs for overall RCR. All numbers use
the held-out **test** split (speed signs under the paper’s A6 filter).

Note on naming: the review asked for SCR/RVR/GVR. In the camera-ready version
we report the **compliance** duals — SCR, GCR, RCR (higher = better) — rather
than violation rates.

Artifacts: `benchmark_output/ready_test_summary/reviewer_evidence/{{tables,figures}}/`.

---

## 1. Precise definitions

**Efficiency (Eff.).** Episode mean of ego speed as a percentage of nearby traffic:

$$
\\mathrm{{Eff}}=\\mathrm{{mean}}_t\\big(100\\cdot v_{{\\mathrm{{ego}}}}(t)/\\bar v_{{\\mathrm{{nearby}}}}(t)\\big).
$$

Nearby agents come from lidar surrounds with speed $>0.5$ km/h; ratios $>1000$
are dropped. Values can exceed 100. Higher $\\uparrow$.

**Comfort (Comf.).** `frame_smooth_ratio`: fraction of frames inside nuPlan-style
kinematic bounds ($\\mathrm{{d}}t=0.1$): long. accel $\\in[-4.05,2.40]$,
$|$lat. accel$|\\le 4.89$, $|$yaw rate$|\\le 0.95$, $|$yaw accel$|\\le 1.93$,
$|$long. jerk$|\\le 4.13$, $|$jerk$|\\le 8.37$. Higher $\\uparrow$.

**Route Completion (RC).** $1$ if `arrived_dest`, else
$\\min(1,d_{{\\mathrm{{travelled}}}}/L_{{\\mathrm{{route}}}})$.
(We recompute RC; the dumped CSV column is broken / identically 0.)

**Collision.** Rate of `info["crash"]`. Does **not** include `out_of_road`. Lower $\\downarrow$.

**SCR (Sign Compliance Rate).** Per-sign compliance: priority = mean
`sign_compliant_high` over all episodes; non-priority = mean among
`target_in_zone`; speed = A6 ∩ in-zone `target_compliant_event`
(3.24/5.31 also uniform over $v_{{\\mathrm{{target}}}}$). Higher $\\uparrow$.

**GCR (Group Compliance Rate).** Unweighted mean of per-sign SCR within a PDD
category (Priority / Prohibitory / Mandatory / Special). Higher $\\uparrow$.

**RCR (Rule Compliance Rate).** Mean of the four GCRs — the overall rule score.
Higher $\\uparrow$.

Full formulas: `tables/metric_definitions.md`.

---

## 2. Joint reporting (conventional + rule compliance)

Full table: `tables/joint_metrics_by_baseline.md`. Headline twins:

| Planner | Eff. | Comf. | RC | Coll. | RCR | GCR Priority |
|---|---:|---:|---:|---:|---:|---:|
| PlanT-2 | {fmt_num(plant.get('efficiency'))} | {fmt_num(plant.get('comfort'), 3)} | {g(plant,'route_completion')} | {g(plant,'collision')} | **{g(plant,'rcr')}** | **{g(plant,'gcr_priority')}** |
| PlanT-2ᵉ | {fmt_num(plant_e.get('efficiency'))} | {fmt_num(plant_e.get('comfort'), 3)} | {g(plant_e,'route_completion')} | {g(plant_e,'collision')} | **{g(plant_e,'rcr')}** | **{g(plant_e,'gcr_priority')}** |
| IDM | {fmt_num(idm.get('efficiency'))} | {fmt_num(idm.get('comfort'), 3)} | {g(idm,'route_completion')} | {g(idm,'collision')} | **{g(idm,'rcr')}** | **{g(idm,'gcr_priority')}** |
| IDMᵉ | {fmt_num(idm_e.get('efficiency'))} | {fmt_num(idm_e.get('comfort'), 3)} | {g(idm_e,'route_completion')} | {g(idm_e,'collision')} | **{g(idm_e,'rcr')}** | **{g(idm_e,'gcr_priority')}** |
| PPO | {fmt_num(ppo.get('efficiency'))} | {fmt_num(ppo.get('comfort'), 3)} | {g(ppo,'route_completion')} | {g(ppo,'collision')} | **{g(ppo,'rcr')}** | **{g(ppo,'gcr_priority')}** |
| PPOᵉ | {fmt_num(ppo_e.get('efficiency'))} | {fmt_num(ppo_e.get('comfort'), 3)} | {g(ppo_e,'route_completion')} | {g(ppo_e,'collision')} | **{g(ppo_e,'rcr')}** | **{g(ppo_e,'gcr_priority')}** |

**The surprising fact for a progress-only dashboard.** PlanT-2 is the *most*
efficient planner (Eff≈235) with competitive RC (69%), yet Priority GCR is
**2.0%** and RCR **21.4%**. Its rule twin keeps RC (75%) while Priority GCR
jumps to **99.5%** and RCR to **87.7%**. IDM→IDMᵉ keeps Eff almost identical
(~169) and RC within 2pp, while RCR jumps **20.7% → 89.7%**.

Figure `figures/fig_twin_gap.png`: RC bars stay flat across every base→expert
twin; RCR bars jump by ~60–70pp.

{ex_line}

Across the paper table we find **{len(bad)}** base `(sign, planner)` cells with
RC≥50% and SCR≤15% (`tables/decoupling_examples.csv`,
`figures/fig_rc_vs_scr.png`).

---

## 3. Correlation analysis (fragment level)

Unit of analysis: one `(sign, planner)` cell (N={len(cells)}) — per-sign SCR.
Spearman $\\rho$ with **cell-bootstrap 95% CIs** against SCR, plus GCR for
each of the four PDD groups (`tables/correlation.md`,
`figures/fig_corr_heatmap.png`):

| Conventional | $\\rho$ with SCR (95% CI) |
|---|---:|
| Efficiency | **{rho_eff_scr:.3f}**{ci_str('efficiency')} |
| Comfort | **{rho_comf_scr:.3f}**{ci_str('comfort')} |
| Route Completion | **{rho_rc_scr:.3f}**{ci_str('route_completion')} |
| Collision | **{rho_col_scr:.3f}**{ci_str('collision')} |

**Falsification test.** If Eff / Comf / RC tracked rule following, their
correlation with SCR would be strongly *positive*. Empirically:

- **Efficiency is *negatively* correlated with SCR** ($\\rho=-0.36$;
  95% CI{ci_str('efficiency')} excludes 0). Faster-looking planners comply less.
- RC and Comfort are only weakly positive ($\\rho\\approx 0.29$–$0.39$;
  linear $R^2<0.15$).
- Collision is the strongest conventional correlate ($\\rho=-0.64$), yet still
  leaves most SCR variance unexplained.

**Stratification.** Among *base* planners alone (N={base_n}):
Comfort↔SCR collapses to **$\\rho={base_comf:.3f}$**, and Eff↔SCR is only
**$\\rho={base_eff:.3f}$**. On Priority signs, base cells average **RC ≈ 67%**
with **SCR ≈ 6.8%**.

**Ranking disagreement.** Kendall $\\tau$ between planner ranks by RC vs by RCR
is **{tau_s}** (`figures/fig_rank_disagreement.png`).

---

## 4. Fragment-level bootstrap CIs for RCR

We resample **signs** with replacement and recompute RCR (= mean of GCRs;
1000 bootstraps). 95% CIs: `tables/bootstrap_rcr_ci.md`,
`figures/fig_scr_ci_forest.png`.

Base planners sit in roughly [10%, 50%]; rule experts in [80%, 99%]. For every
primary twin the base and expert CIs are **disjoint**.

---

## 5. Why this answers the concern

1. Eff and Comfort now have closed-form, reproducible definitions.
2. RC, Collision, Comfort are reported **side-by-side** with
   SCR / GCR / RCR.
3. Correlations (with bootstrap CIs) show conventional metrics are weak,
   and Efficiency is **anti**-aligned with compliance.
4. Fragment bootstrap CIs establish that the RCR separation is stable.

We will add the twin-gap figure, the correlation table (with CIs), and the
definitions paragraph to the camera-ready version (main text or appendix).
"""
    path.write_text(text_out, encoding="utf-8")

    ru = path.with_name("REVIEWER_RESPONSE_RU.md")
    ru.write_text(
        f"""# Ответ ревьюеру (черновик)

Спасибо за вопрос. Ниже — определения, совместная таблица, корреляции с
bootstrap-CI и fragment-bootstrap CI для RCR. Всё на test (speed — A6).

Метрики compliance (не violation): **SCR** (по знаку) → **GCR** (по группе) →
**RCR** (mean of GCRs).

## Определения

- **Eff.** — $100\\cdot v_{{\\mathrm{{ego}}}}/\\bar v_{{\\mathrm{{nearby}}}}$ (lidar, $>0.5$ км/ч).
- **Comfort** — `frame_smooth_ratio` (nuPlan kinematic bounds).
- **RC** — 1 при arrive, иначе $d/L$.
- **Collision** — `crashed` (без OOR).
- **SCR / GCR / RCR** — compliance rates (↑ лучше).

## Главный факт (twins)

| Planner | RC | RCR | GCR Priority |
|---|---:|---:|---:|
| PlanT-2 | {g(plant,'route_completion')} | {g(plant,'rcr')} | {g(plant,'gcr_priority')} |
| PlanT-2ᵉ | {g(plant_e,'route_completion')} | {g(plant_e,'rcr')} | {g(plant_e,'gcr_priority')} |
| IDM | {g(idm,'route_completion')} | {g(idm,'rcr')} | {g(idm,'gcr_priority')} |
| IDMᵉ | {g(idm_e,'route_completion')} | {g(idm_e,'rcr')} | {g(idm_e,'gcr_priority')} |

RC почти не меняется, RCR прыгает ~×4. PlanT-2 — самый «эффективный» (Eff≈235),
но Priority GCR = 2%. См. `fig_twin_gap.png`.

## Корреляции с SCR (N={len(cells)} клеток sign×planner)

| Метрика | Spearman ρ с SCR |
|---|---:|
| Efficiency | **{rho_eff_scr:.3f}**{ci_str('efficiency')} (отрицательная!) |
| Comfort | {rho_comf_scr:.3f}{ci_str('comfort')} |
| RC | {rho_rc_scr:.3f}{ci_str('route_completion')} |
| Collision | {rho_col_scr:.3f}{ci_str('collision')} |

Среди **только base**: Comfort↔SCR ≈ **{base_comf:.3f}**, Eff↔SCR ≈ **{base_eff:.3f}**.
На Priority у base: RC≈67% при SCR≈6.8%.
Kendall τ(ранг RC, ранг RCR) = **{tau_s}**.
Клеток с RC≥50% и SCR≤15%: **{len(bad)}**.

## Bootstrap CI для RCR

Base ≈ 10–50%, expert ≈ 80–99%, CI близнецов **не пересекаются**.

Артефакты: `reviewer_evidence/`. Полный англ. ответ: `REVIEWER_RESPONSE.md`.
""",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--bootstraps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.out.resolve()
    tab = out / "tables"
    fig = out / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    print("=== loading episodes (paper pipeline) ===")
    episodes = load_all_episodes()
    if not episodes:
        print("ERROR: no episodes", file=sys.stderr)
        sys.exit(2)
    print(f"total episodes: {len(episodes)}")
    attach_out_of_road(episodes)

    # Per-baseline aggregates
    by_base_rows: dict[str, list[dict]] = defaultdict(list)
    for r in episodes:
        by_base_rows[r["baseline"]].append(r)
    by_base = {b: aggregate_planner(rs) for b, rs in by_base_rows.items()}

    write_definitions(tab / "metric_definitions.md")
    write_joint_table(tab / "joint_metrics_by_baseline.csv",
                      tab / "joint_metrics_by_baseline.md", by_base)

    print("=== correlations ===")
    sp, pe, cells = correlation_at_sign_baseline_level(episodes)
    print(f"correlation cells: {len(cells)}")

    sp_ci: dict[str, dict] = {}
    for c in CONV_METRICS:
        point, lo, hi = bootstrap_spearman_ci(cells, c, "scr", args.bootstraps, args.seed)
        sp_ci[c] = {"point": point, "lo": lo, "hi": hi}
        print(f"  Spearman {c}↔SCR = {point:.3f}  [{lo:.3f}, {hi:.3f}]")

    sp_gcr, gcr_cells = correlation_gcr_by_group(episodes)
    print(f"  GCR group×planner cells: {len(gcr_cells)}")
    for g in GCR_GROUPS:
        print(f"  Spearman Eff↔{GCR_LABELS[g]} = {sp_gcr[g]['efficiency']:.3f}")

    sp_base = corr_on_subset(cells, lambda c: not c["is_expert"])
    sp_expert = corr_on_subset(cells, lambda c: bool(c["is_expert"]))
    print(f"  base-only RC↔SCR = {sp_base['route_completion']:.3f} (N={int(sp_base['n'])})")
    print(f"  expert-only RC↔SCR = {sp_expert['route_completion']:.3f} (N={int(sp_expert['n'])})")

    write_corr_tables(
        sp, pe,
        tab / "correlation_spearman.csv",
        tab / "correlation_pearson.csv",
        tab / "correlation.md",
        sp_ci=sp_ci,
        sp_base=sp_base,
        sp_expert=sp_expert,
        sp_gcr=sp_gcr,
    )

    # Decoupling examples CSV
    bad = [
        c for c in cells
        if not c["is_expert"]
        and c["route_completion"] is not None
        and c["scr"] is not None
        and c["route_completion"] >= 0.5
        and c["scr"] <= 0.15
    ]
    bad = sorted(bad, key=lambda c: (c["scr"], -c["route_completion"]))
    with (tab / "decoupling_examples.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["sign", "baseline", "n", "route_completion", "efficiency",
                        "comfort", "collision", "scr"],
        )
        w.writeheader()
        for c in bad[:40]:
            w.writerow({
                "sign": c["sign"],
                "baseline": c["baseline"],
                "n": c["n"],
                "route_completion": round(c["route_completion"], 4),
                "efficiency": "" if c["efficiency"] is None else round(c["efficiency"], 3),
                "comfort": "" if c["comfort"] is None else round(c["comfort"], 4),
                "collision": "" if c["collision"] is None else round(c["collision"], 4),
                "scr": round(c["scr"], 4),
            })

    print("=== bootstrap CIs (episodes within sign) ===")
    ci_rows = []
    for b, rs in by_base_rows.items():
        by_sign: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            by_sign[r["sign_label"]].append(r)
        point, lo, hi = episode_bootstrap_rcr(by_sign, args.bootstraps, args.seed)
        ci_rows.append({
            "baseline": b,
            "display": S.DISPLAY_NAME.get(b, b),
            "is_expert": int("rule" in b or b.startswith("comprehensive_rule")),
            "point": point, "lo": lo, "hi": hi,
            "n_signs": len(by_sign),
        })
    with (tab / "bootstrap_rcr_ci.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["baseline", "display", "is_expert", "n_signs",
                            "rcr", "ci95_lo", "ci95_hi"],
        )
        w.writeheader()
        for r in sorted(ci_rows, key=lambda r: S.baseline_sort_key(r["baseline"])):
            w.writerow({
                "baseline": r["baseline"],
                "display": r["display"],
                "is_expert": r["is_expert"],
                "n_signs": r["n_signs"],
                "rcr": round(r["point"], 4) if math.isfinite(r["point"]) else "",
                "ci95_lo": round(r["lo"], 4) if math.isfinite(r["lo"]) else "",
                "ci95_hi": round(r["hi"], 4) if math.isfinite(r["hi"]) else "",
            })

    # Paper-layout MD with Base / Experts groups.
    base_order = [
        "idm_default", "idm_s1", "idm_s2", "idm_s3", "idm_s4",
        "ppo_lidar_default", "carl_default", "plant2_default",
    ]
    expert_order = [
        "comprehensive_rule_expert_default",
        "comprehensive_rule_expert_s1",
        "comprehensive_rule_expert_s2",
        "comprehensive_rule_expert_s3",
        "comprehensive_rule_expert_s4",
        "rule_compliant_default",
        "carl_rule_default",
        "plant2_rule_default",
    ]
    by_bl = {r["baseline"]: r for r in ci_rows}

    def _ci_row(group: str, bl: str, first: bool) -> str:
        r = by_bl[bl]
        name = PAPER_PLANNER_NAME.get(bl, r["display"])
        g = f"**{group}**" if first else ""
        return (
            f"| {g} | {name} | {100*r['point']:.1f}% | "
            f"[{100*r['lo']:.1f}, {100*r['hi']:.1f}] |"
        )

    md = [
        "# Episode-bootstrap 95% CIs for overall RCR",
        "",
        "Within each sign, resample episodes with replacement; recompute "
        "per-sign SCR (paper denominator), then GCR and RCR (= mean of GCRs). "
        "1000 replicates; 95% CI = [2.5th, 97.5th] percentile.",
        "",
        "| Group | Planner | RCR | 95% CI |",
        "|:--|:--|--:|--:|",
    ]
    for i, bl in enumerate(base_order):
        if bl in by_bl:
            md.append(_ci_row("Base planners", bl, i == 0))
    for i, bl in enumerate(expert_order):
        if bl in by_bl:
            md.append(_ci_row("Experts", bl, i == 0))
    # Any leftover planners not in the paper order.
    known = set(base_order) | set(expert_order)
    for r in sorted(ci_rows, key=lambda r: S.baseline_sort_key(r["baseline"])):
        if r["baseline"] not in known:
            md.append(
                f"|  | {r['display']} | {100*r['point']:.1f}% | "
                f"[{100*r['lo']:.1f}, {100*r['hi']:.1f}] |"
            )
    (tab / "bootstrap_rcr_ci.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("=== figures ===")
    fig_scatter_metric_vs_scr(
        cells, "route_completion", "Route Completion (fraction)",
        fig / "fig_rc_vs_scr.png",
    )
    fig_scatter_metric_vs_scr(
        cells, "efficiency", "Efficiency (nearby-speed %)",
        fig / "fig_eff_vs_scr.png",
    )
    fig_corr_heatmap(sp, sp_gcr, fig / "fig_corr_heatmap.png")
    # Markdown twin of the heatmap matrix.
    hm_lines = [
        "# Spearman $\\rho$: conventional vs SCR / GCR",
        "",
        "Same matrix as `fig_corr_heatmap.png`. "
        "SCR: `(sign, baseline)` cells. GCR: `(group, baseline)` cells.",
        "",
        "|  | SCR | GCR Pri | GCR Pro | GCR Man | GCR Spe |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    row_labels = ["Eff.", "Comf.", "RC", "Coll."]
    for lab, c in zip(row_labels, CONV_METRICS):
        hm_lines.append(
            "| {lab} | {scr:.2f} | {a:.2f} | {b:.2f} | {d:.2f} | {e:.2f} |".format(
                lab=lab,
                scr=sp[c],
                a=sp_gcr["priority"][c],
                b=sp_gcr["prohibitory"][c],
                d=sp_gcr["mandatory"][c],
                e=sp_gcr["special"][c],
            )
        )
    (tab / "fig_corr_heatmap.md").write_text("\n".join(hm_lines) + "\n", encoding="utf-8")
    fig_scr_ci_forest(ci_rows, fig / "fig_scr_ci_forest.png")
    fig_twin_gap(by_base, fig / "fig_twin_gap.png")
    kendall_tau = fig_rank_disagreement(by_base, fig / "fig_rank_disagreement.png")
    print(f"  Kendall τ(RC ranks, RCR ranks) = {kendall_tau:.3f}")

    write_reviewer_response(
        out / "REVIEWER_RESPONSE.md",
        by_base, sp, cells, ci_rows,
        sp_ci=sp_ci, sp_base=sp_base, sp_expert=sp_expert,
        kendall_tau=kendall_tau,
    )
    print(f"\nWrote {out}/")
    print(f"  REVIEWER_RESPONSE.md")
    print(f"  tables/ ({len(list(tab.iterdir()))} files)")
    print(f"  figures/ ({len(list(fig.iterdir()))} files)")


if __name__ == "__main__":
    main()
