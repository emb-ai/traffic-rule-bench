#!/usr/bin/env python3
"""Recount per-baseline metrics under scene-filtration scenarios (post-hoc, no re-eval).

Inputs: metrics_per_episode.csv + the eval manifest (for v_target_kmh per scene row).
Scene granularity: scene_uid (= scene_id + spawn lane + seed + variation).

Filters (composable):
  F1  --require-rule-arrived : keep scene_uids where >=1 RULE-baseline episode arrived_dest.
  F2  --require-in-zone      : keep scene_uids where >=1 episode (any baseline) has
                               target_in_zone steps.
  U   uniform v_target       : post-stratify metrics to equal weights over the sign's
                               v_target buckets (applied to UNIFORM_SIGNS only).

Scenarios reported: S0 base, S1 F1, S2 F2, S3 F1+F2, S4 U, S5 F1+U, S6 F2+U, S7 F1+F2+U.

Metrics per (sign, baseline): compliance (target_compliant_event), dest_rate,
dest_x_compliance = mean(arrived AND compliant), crash_rate, out_of_road_rate, n episodes.

Output: one markdown report (--out-md) + per-scenario CSVs (--out-csv-dir).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Signs whose v_target distribution is post-stratified to uniform in U-scenarios.
UNIFORM_SIGNS = (3.24, 5.31)
DEFAULT_NONRULE = ["idm_default", "carl_default", "plant2_default", "ppo_lidar_default"]

USECOLS = [
    "scene_uid", "scene_id", "baseline", "pdd_code", "arrived_dest",
    "target_in_zone", "target_compliant_event", "crashed", "out_of_road",
    # для cumulative-style таблиц (формат report_cumulative.md)
    "success", "tl_compliant", "cw_compliant", "sign_compliant_high",
    "driving_score", "driving_efficiency", "smoothness_ratio",
    "route_length_m", "distance_travelled_m",
]

# Отображаемые имена политик — как в generate_cumulative_markdown_report.py
POLICY_DISPLAY_NAME = {
    "comprehensive_rule_expert_default": "idm_rule_default",
    "comprehensive_rule_expert_s1": "idm_rule_s1",
    "comprehensive_rule_expert_s2": "idm_rule_s2",
    "comprehensive_rule_expert_s3": "idm_rule_s3",
    "comprehensive_rule_expert_s4": "idm_rule_s4",
}

SCENARIOS = {
    "S0 base": (False, False, False),          # (F1, F2, uniform)
    "S1 F1": (True, False, False),
    "S2 F2": (False, True, False),
    "S3 F1+F2": (True, True, False),
    "S4 unif": (False, False, True),
    "S5 F1+unif": (True, False, True),
    "S6 F2+unif": (False, True, True),
    "S7 F1+F2+unif": (True, True, True),
}


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def load(csv_path: Path, manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=USECOLS, low_memory=False)
    for col in ("arrived_dest", "target_in_zone", "crashed", "out_of_road",
                "success", "tl_compliant", "cw_compliant", "sign_compliant_high"):
        df[col] = _to_bool(df[col])
    # target_compliant_event may hold NaN for rows without a class mapping — keep NaN
    if df.target_compliant_event.dtype != bool:
        df["target_compliant_event"] = df.target_compliant_event.map(
            {True: True, False: False, "True": True, "False": False}).astype("boolean")
    df["seed"] = df.scene_uid.str.extract(r"_seed(\d+)_")[0].astype("int64")

    man = pd.DataFrame(
        json.loads(line) for line in open(manifest_path, encoding="utf-8"))
    man = man[["scene_id", "seed", "v_target_kmh"]]
    df = df.merge(man, on=["scene_id", "seed"], how="left")

    df["is_rule"] = df.baseline.str.contains("rule")
    return df


def scenario_mask(df: pd.DataFrame, f1: bool, f2: bool) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if f1:
        ok = set(df.loc[df.is_rule & df.arrived_dest, "scene_uid"])
        mask &= df.scene_uid.isin(ok)
    if f2:
        ok = set(df.loc[df.target_in_zone, "scene_uid"])
        mask &= df.scene_uid.isin(ok)
    return mask


def _agg(g: pd.DataFrame) -> pd.Series:
    comp = g.target_compliant_event
    return pd.Series({
        "compliance": comp.mean(),
        "dest_rate": g.arrived_dest.mean(),
        "dest_x_compliance": (g.arrived_dest & comp.fillna(False)).mean(),
        "crash_rate": g.crashed.mean(),
        "out_of_road_rate": g.out_of_road.mean(),
        "n": len(g),
    })


def per_baseline(sub: pd.DataFrame, uniform: bool) -> pd.DataFrame:
    rows = {}
    for bl, g in sub.groupby("baseline"):
        code = float(g.pdd_code.iloc[0])
        if uniform and code in UNIFORM_SIGNS:
            strata = g.groupby("v_target_kmh").apply(_agg, include_groups=False)
            row = strata.drop(columns="n").mean()
            row["n"] = strata.n.sum()
        else:
            row = _agg(g)
        rows[bl] = row
    out = pd.DataFrame(rows).T
    out["n"] = out.n.astype(int)
    return out.astype({"n": int})


PAIRS = [
    ("idm_default", "comprehensive_rule_expert_default"),
    ("carl_default", "carl_rule_default"),
    ("plant2_default", "plant2_rule_default"),
    ("ppo_lidar_default", "rule_compliant_default"),
]


def strategy_comparison(df: pd.DataFrame) -> dict[float, pd.DataFrame]:
    """Качество бенчмарка = парный зазор compliance (rule − base) по PAIRS.

    Стратегии: A0 без фильтров; A1 F1+F2; для UNIFORM_SIGNS — A2 unif,
    A3 жёсткий сабсет v∈{20,30}, A4 v=20; A5 = F3 (сцена решаема: >=1 rule-эксперт
    доехал И соблюдал) + F2 (+ v∈{20,30} для UNIFORM_SIGNS).
    """
    f1 = set(df.loc[df.is_rule & df.arrived_dest, "scene_uid"])
    f2 = set(df.loc[df.target_in_zone, "scene_uid"])
    f3 = set(df.loc[df.is_rule & df.arrived_dest
                    & df.target_compliant_event.fillna(False), "scene_uid"])

    def evaluate(sub: pd.DataFrame, uniform: bool) -> dict:
        def comp(bl):
            g = sub[sub.baseline == bl]
            if not len(g):
                return float("nan")
            if uniform:
                return g.groupby("v_target_kmh").target_compliant_event.mean().mean()
            return g.target_compliant_event.mean()
        gaps = {}
        for b, r in PAIRS:
            cb, cr = comp(b), comp(r)
            if pd.notna(cb) and pd.notna(cr):
                gaps[b.split("_")[0]] = cr - cb
        bases = [comp(b) for b, _ in PAIRS if pd.notna(comp(b))]
        rules = [comp(r) for _, r in PAIRS if pd.notna(comp(r))]
        return {
            "base": np.mean(bases), "rule": np.mean(rules),
            "mean_gap": np.mean(list(gaps.values())),
            "min_gap": min(gaps.values()),
            "слабейшая пара": min(gaps, key=gaps.get),
            "uid": sub.scene_uid.nunique(),
        }

    out: dict[float, pd.DataFrame] = {}
    for code in sorted(df.pdd_code.unique()):
        d = df[df.pdd_code == code]
        d12 = d[d.scene_uid.isin(f1 & f2)]
        strats = {"A0 без фильтров": (d, False), "A1 F1+F2": (d12, False)}
        if float(code) in UNIFORM_SIGNS:
            strats["A2 F1+F2+unif"] = (d12, True)
            strats["A3 F1+F2, v∈{20,30}"] = (d12[d12.v_target_kmh.isin([20, 30])], False)
            strats["A4 F1+F2, v=20"] = (d12[d12.v_target_kmh == 20], False)
            strats["A5 F3+F2, v∈{20,30}"] = (
                d[d.scene_uid.isin(f3 & f2) & d.v_target_kmh.isin([20, 30])], False)
            strats["A6 F3+F2+unif"] = (d[d.scene_uid.isin(f3 & f2)], True)
        else:
            strats["A5 F3+F2"] = (d[d.scene_uid.isin(f3 & f2)], False)
        out[float(code)] = pd.DataFrame(
            {name: evaluate(sub, u) for name, (sub, u) in strats.items()}).T
    return out


CUM_COLS = ["Runs", "In-zone runs", "Success rate", "Dest rate", "TL SR", "CW SR",
            "Sign compliance SR", "Sign compliance (in-zone)", "Avg driving score",
            "Avg efficiency", "Avg smoothness", "Avg route len (m)", "Avg distance (m)"]


def _cum_rates(g: pd.DataFrame) -> pd.Series:
    inz = g[g.target_in_zone]
    return pd.Series({
        "Runs": len(g),
        "In-zone runs": int(g.target_in_zone.sum()),
        "Success rate": g.success.mean(),
        "Dest rate": g.arrived_dest.mean(),
        "TL SR": g.tl_compliant.mean(),
        "CW SR": g.cw_compliant.mean(),
        "Sign compliance SR": g.sign_compliant_high.mean(),
        "Sign compliance (in-zone)": inz.sign_compliant_high.mean() if len(inz) else float("nan"),
        "Avg driving score": g.driving_score.mean(),
        "Avg efficiency": g.driving_efficiency.mean(),
        "Avg smoothness": g.smoothness_ratio.mean(),
        "Avg route len (m)": g.route_length_m.mean(),
        "Avg distance (m)": g.distance_travelled_m.mean(),
    })


def cum_table(sub: pd.DataFrame, uniform: bool) -> pd.DataFrame:
    """Таблица в формате report_cumulative для одного знака (или Overall)."""
    rows = {}
    for bl, g in sub.groupby("baseline"):
        code = float(g.pdd_code.iloc[0])
        if uniform and code in UNIFORM_SIGNS:
            strata = g.groupby("v_target_kmh").apply(_cum_rates, include_groups=False)
            row = strata.drop(columns=["Runs", "In-zone runs"]).mean()
            row["Runs"] = int(strata["Runs"].sum())
            row["In-zone runs"] = int(strata["In-zone runs"].sum())
        else:
            row = _cum_rates(g)
        rows[POLICY_DISPLAY_NAME.get(bl, bl)] = row
    return pd.DataFrame(rows).T[CUM_COLS].sort_index()


def overall_from_signs(per_sign: dict[float, pd.DataFrame]) -> pd.DataFrame:
    """Overall = средневзвешенное per-sign таблиц (веса — Runs; in-zone колонка — In-zone runs)."""
    policies = sorted({p for t in per_sign.values() for p in t.index})
    rows = {}
    for p in policies:
        runs = inz = 0
        acc = {c: 0.0 for c in CUM_COLS if c not in ("Runs", "In-zone runs",
                                                     "Sign compliance (in-zone)")}
        acc_x = 0.0
        for t in per_sign.values():
            if p not in t.index:
                continue
            r = t.loc[p]
            w, wz = r["Runs"], r["In-zone runs"]
            runs += int(w)
            inz += int(wz)
            for c in acc:
                acc[c] += r[c] * w
            if pd.notna(r["Sign compliance (in-zone)"]):
                acc_x += r["Sign compliance (in-zone)"] * wz
        row = {c: acc[c] / runs for c in acc}
        row["Sign compliance (in-zone)"] = acc_x / inz if inz else float("nan")
        row["Runs"], row["In-zone runs"] = runs, inz
        rows[p] = row
    return pd.DataFrame(rows).T[CUM_COLS]


def cum_md(t: pd.DataFrame) -> str:
    t = t.copy()
    t["Runs"] = t.Runs.astype(int)
    t["In-zone runs"] = t["In-zone runs"].astype(int)
    for c in t.columns:
        if t[c].dtype.kind == "f":
            t[c] = t[c].map(lambda x: "—" if pd.isna(x) else f"{x:.3f}")
    header = ["Policy"] + list(t.columns)
    out = ["| " + " | ".join(header) + " |",
           "|---|" + "|".join("---:" for _ in t.columns) + "|"]
    for idx, r in t.iterrows():
        out.append("| `" + str(idx) + "` | " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def cumulative_by_strategy(df: pd.DataFrame) -> str:
    """report_cumulative-style таблицы для каждой стратегии фильтрации."""
    f1 = set(df.loc[df.is_rule & df.arrived_dest, "scene_uid"])
    f2 = set(df.loc[df.target_in_zone, "scene_uid"])
    f3 = set(df.loc[df.is_rule & df.arrived_dest
                    & df.target_compliant_event.fillna(False), "scene_uid"])
    v_uni = sorted(UNIFORM_SIGNS)

    # (подпись, uid-фильтр, uniform, v-сабсет для UNIFORM_SIGNS)
    strategies = [
        ("A0 — без фильтров", None, False, None),
        ("A1 — F1+F2 (≥1 rule-эксперт доехал; есть in-zone шаги)", f1 & f2, False, None),
        (f"A2 — F1+F2 + равные веса v_target 20/30/40 (знаки {v_uni})", f1 & f2, True, None),
        (f"A3 — F1+F2 + только v∈{{20,30}} (знаки {v_uni})", f1 & f2, False, [20, 30]),
        (f"A4 — F1+F2 + только v=20 (знаки {v_uni})", f1 & f2, False, [20]),
        (f"A5 — F3+F2 (сцена решаема rule-экспертом) + v∈{{20,30}} (знаки {v_uni})",
         f3 & f2, False, [20, 30]),
        (f"A6 — F3+F2 + равные веса v_target 20/30/40 (знаки {v_uni}) — РЕКОМЕНДУЕМАЯ",
         f3 & f2, True, None),
    ]

    lines = ["# Cumulative Benchmark Results — по стратегиям фильтрации\n",
             "Формат таблиц идентичен report_cumulative.md. Фильтры: F1 — ≥1 rule-эксперт "
             "доехал; F2 — есть in-zone шаги хоть у одного бейзлайна; F3 — ≥1 rule-эксперт "
             "доехал И был compliant («сцена решаема»). v-сабсеты/веса применяются только к "
             f"знакам {v_uni}; остальные знаки получают те же uid-фильтры.\n"]
    for label, uids, uniform, vsub in strategies:
        d = df if uids is None else df[df.scene_uid.isin(uids)]
        per_sign: dict[float, pd.DataFrame] = {}
        for code in sorted(d.pdd_code.unique()):
            ds = d[d.pdd_code == code]
            if vsub is not None and float(code) in UNIFORM_SIGNS:
                ds = ds[ds.v_target_kmh.isin(vsub)]
            per_sign[float(code)] = cum_table(ds, uniform)
        lines.append(f"\n---\n\n## Стратегия {label}\n")
        lines.append("### Overall (weighted by runs)\n")
        lines.append(cum_md(overall_from_signs(per_sign)) + "\n")
        for code, t in per_sign.items():
            lines.append(f"### Sign `{code}`\n")
            lines.append(cum_md(t) + "\n")
    return "\n".join(lines)


def drop_stats(df: pd.DataFrame) -> pd.DataFrame:
    total = df.groupby("pdd_code").scene_uid.nunique()
    rows = {}
    for name, (f1, f2) in {
        "F1 (>=1 rule-эксперт доехал)": (True, False),
        "F2 (есть in-zone шаги)": (False, True),
        "F1+F2": (True, True),
    }.items():
        kept = df[scenario_mask(df, f1, f2)].groupby("pdd_code").scene_uid.nunique()
        kept = kept.reindex(total.index, fill_value=0)
        rows[name] = pd.Series(
            {c: f"{total[c] - kept[c]} из {total[c]} ({(total[c]-kept[c])/total[c]*100:.1f}%)"
             for c in total.index})
    return pd.DataFrame(rows)


def md_table(t: pd.DataFrame) -> str:
    """GitHub-markdown таблица без зависимости от tabulate."""
    t = t.copy()
    for c in t.columns:
        if t[c].dtype.kind == "f":
            t[c] = t[c].map(lambda x: "—" if pd.isna(x) else f"{x:.3f}")
        else:
            t[c] = t[c].astype(str).replace({"nan": "—", "None": "—"})
    header = [str(t.index.name or "")] + [str(c) for c in t.columns]
    rows = [[str(i)] + list(r) for i, r in zip(t.index, t.values)]
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def fmt_table(t: pd.DataFrame, float_cols=None) -> str:
    return md_table(t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="metrics_per_episode.csv")
    ap.add_argument("--manifest", required=True, help="eval manifest .jsonl (v_target_kmh)")
    ap.add_argument("--out-md", required=True, help="markdown report path")
    ap.add_argument("--out-csv-dir", default=None, help="dir for per-scenario CSVs")
    ap.add_argument("--cumulative-md", default=None,
                    help="доп. отчёт: report_cumulative-style таблицы для каждой стратегии")
    args = ap.parse_args()

    df = load(Path(args.csv), Path(args.manifest))
    signs = sorted(float(c) for c in df.pdd_code.unique())
    lines: list[str] = []
    lines.append("# Recount: сценарии фильтрации сцен (пост-хок по сыгранным данным)\n")
    lines.append(f"Источник: `{args.csv}`; эпизодов {len(df)}, "
                 f"scene_uid {df.scene_uid.nunique()}, знаки {signs}.\n")
    lines.append("Фильтры: **F1** — scene_uid с >=1 доехавшим rule-экспертом; "
                 "**F2** — scene_uid, где хоть у одного бейзлайна есть in-zone шаги; "
                 f"**unif** — равные веса v_target-страт (только знаки {list(UNIFORM_SIGNS)}).\n")

    lines.append("## Отбрасывание сцен (scene_uid)\n")
    lines.append(md_table(drop_stats(df)) + "\n")

    csv_dir = Path(args.out_csv_dir) if args.out_csv_dir else None
    if csv_dir:
        csv_dir.mkdir(parents=True, exist_ok=True)

    group_summary: dict[str, dict[str, float]] = {}
    for code in signs:
        d_sign = df[df.pdd_code == code]
        lines.append(f"\n## Знак {code}\n")

        # v_target распределение
        vt = (d_sign.drop_duplicates("scene_uid").v_target_kmh
              .value_counts().sort_index())
        if float(code) in UNIFORM_SIGNS:
            lines.append("Распределение v_target по сценам: "
                         + ", ".join(f"{int(k)} км/ч — {v}" for k, v in vt.items()) + "\n")

        comp_matrix = {}
        for sname, (f1, f2, uni) in SCENARIOS.items():
            if uni and float(code) not in UNIFORM_SIGNS:
                continue
            sub = d_sign[scenario_mask(d_sign, f1, f2)]
            t = per_baseline(sub, uni)
            comp_matrix[sname] = t.compliance
            if csv_dir:
                safe = sname.replace(" ", "_").replace("+", "")
                t.round(4).to_csv(csv_dir / f"sign{code}_{safe}.csv")
            key = f"{code}|{sname}"
            avail = [b for b in DEFAULT_NONRULE if b in t.index]
            group_summary[key] = {
                "default non-rule": t.loc[avail, "compliance"].mean(),
                "rule": t.loc[t.index.str.contains("rule"), "compliance"].mean(),
            }

        m = pd.DataFrame(comp_matrix)
        m["rule"] = m.index.str.contains("rule")
        m = m.sort_values(["rule", m.columns[0]])
        lines.append("### Target compliance по сценариям\n")
        lines.append(fmt_table(m.drop(columns="rule")) + "\n")

        # Полные метрики: базовый и максимально фильтрованный сценарии
        for sname in ("S0 base",
                      "S7 F1+F2+unif" if float(code) in UNIFORM_SIGNS else "S3 F1+F2"):
            f1, f2, uni = SCENARIOS[sname]
            t = per_baseline(d_sign[scenario_mask(d_sign, f1, f2)], uni)
            t = t.sort_values("compliance")
            lines.append(f"### Полные метрики — {sname}\n")
            lines.append(fmt_table(t) + "\n")

    lines.append("\n## Выбор стратегии: качество бенчмарка = парный зазор (rule − base)\n")
    lines.append("Пары: idm/idm_rule, carl/carl_rule, plant2/plant2_rule, ppo/ppo_rule. "
                 "F3 — «сцена решаема»: ≥1 rule-эксперт доехал И соблюдал.\n")
    for code, tbl in strategy_comparison(df).items():
        lines.append(f"\n### Знак {code}\n")
        t = tbl.copy()
        for c in ("base", "rule", "mean_gap", "min_gap"):
            t[c] = pd.to_numeric(t[c]).round(3)
        t["uid"] = t.uid.astype(int)
        lines.append(md_table(t) + "\n")

    lines.append("\n## Сводка: групповое среднее compliance (default non-rule / rule)\n")
    rows = {}
    for key, vals in group_summary.items():
        code, sname = key.split("|")
        rows.setdefault(code, {})[sname] = (
            f"{vals['default non-rule']:.3f} / {vals['rule']:.3f}")
    lines.append(md_table(pd.DataFrame(rows).T) + "\n")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"written: {out_md}")
    if csv_dir:
        print(f"csv dir: {csv_dir}")

    if args.cumulative_md:
        cum_path = Path(args.cumulative_md)
        cum_path.parent.mkdir(parents=True, exist_ok=True)
        cum_path.write_text(cumulative_by_strategy(df), encoding="utf-8")
        print(f"written: {cum_path}")


if __name__ == "__main__":
    main()
