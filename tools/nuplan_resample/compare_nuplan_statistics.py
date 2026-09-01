#!/usr/bin/env python3
"""Compare an old and a new nuplan_statistics directory, and what it does to
the quantities the benchmark actually consumes (through NuPlanSampler and
traffic_density_levels).

Run:
  python compare_nuplan_statistics.py --old <old dir> --new <new dir> \
      [--out report.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_dir(d: Path) -> dict:
    out = {}
    out["speeds"] = pd.read_csv(d / "speeds.csv")["speed"].dropna().to_numpy()
    out["acc_pos"] = pd.read_csv(d / "acc_pos.csv")["acceleration"].dropna().to_numpy()
    out["acc_neg"] = pd.read_csv(d / "acc_neg.csv")["deceleration"].dropna().to_numpy()
    out["following"] = pd.read_csv(d / "following.csv")["following_distance"].dropna().to_numpy()
    out["routes"] = pd.read_csv(d / "routes.csv")
    out["densities"] = pd.read_csv(d / "densities.csv")
    out["lane_changes"] = pd.read_csv(d / "lane_changes.csv")
    cfg = d / "metadrive_config.json"
    out["config"] = json.loads(cfg.read_text()) if cfg.exists() else {}
    return out


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def size_dist_like_sampler(routes: pd.DataFrame, config: dict) -> dict:
    """Exactly the logic of NuPlanSampler._prepare_distributions."""
    if "length" in routes.columns and "width" in routes.columns:
        def classify(row):
            length, width = row.get("length", 4.5), row.get("width", 1.8)
            if length < 4.0:
                return "s"
            elif length < 4.8:
                return "s" if width < 1.8 else "m"
            elif length < 5.2:
                return "m"
            elif length < 6.0:
                return "l"
            return "xl"
        return routes.apply(classify, axis=1).value_counts(normalize=True).to_dict()
    return config.get("size_prob", {}) or {}


def lane_rate(data: dict) -> float:
    km = data["routes"]["distance"].sum() / 1000.0
    return len(data["lane_changes"]) / km if km > 0 else float("nan")


def idm_lane_change_steps(rate_per_km: float) -> int:
    # exactly the formula in agent_profile_bank.apply_profile_to_idm_class
    return int(max(50, 1250.0 / max(rate_per_km, 1.0)))


def density_levels(counts: np.ndarray, cap: float = 0.5) -> list[float]:
    # exactly the formula in traffic_density_levels: p25/50/75 -> /80, cap
    return [round(float(np.clip(np.percentile(counts, p) / 80.0, 0.0, cap)), 4)
            for p in (25, 50, 75)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    old = load_dir(Path(args.old))
    new = load_dir(Path(args.new))

    lines = []
    w = lines.append
    w("# nuplan_statistics: previous vs recomputed\n")
    w(f"- previous: `{args.old}`")
    w(f"- new:      `{args.new}`\n")

    w("## Raw distributions\n")
    w("| quantity | n prev | n new | median prev | median new | p95 prev | p95 new |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for key, col in (("speeds", "speed, m/s"), ("acc_pos", "acceleration, m/s²"),
                     ("acc_neg", "deceleration, m/s²"), ("following", "distance, m")):
        o, n = old[key], new[key]
        w(f"| {col} | {len(o)} | {len(n)} | {q(o,50):.2f} | {q(n,50):.2f} "
          f"| {q(o,95):.2f} | {q(n,95):.2f} |")
    w("")

    w("## Derived quantities the benchmark consumes\n")
    w("| parameter | previous | new | where it is used |")
    w("|---|---:|---:|---|")
    w(f"| MAX_SPEED = p95(speeds), m/s | {q(old['speeds'],95):.2f} "
      f"| {q(new['speeds'],95):.2f} | agent_profile_bank |")
    w(f"| CREEP_SPEED = p5(speeds), m/s | {q(old['speeds'],5):.2f} "
      f"| {q(new['speeds'],5):.2f} | agent_profile_bank |")
    ro, rn = old["routes"], new["routes"]
    w(f"| spawn_velocity: median initial_speed | {ro['initial_speed'].median():.2f} "
      f"| {rn['initial_speed'].median():.2f} | sample_spawn_velocity |")
    w(f"| spawn_velocity: p5–p95 | {ro['initial_speed'].quantile(.05):.2f}–"
      f"{ro['initial_speed'].quantile(.95):.2f} | {rn['initial_speed'].quantile(.05):.2f}–"
      f"{rn['initial_speed'].quantile(.95):.2f} | ego spawn, braking scenes |")

    lr_o, lr_n = lane_rate(old), lane_rate(new)
    w(f"| lane_change rate, events/km | {lr_o:.2f} | {lr_n:.3f} | LANE_CHANGE_FREQ |")
    w(f"| IDMPolicy.LANE_CHANGE_FREQ, steps | {idm_lane_change_steps(lr_o)} "
      f"| {idm_lane_change_steps(lr_n)} | apply_profile_to_idm_class |")

    do, dn = old["densities"], new["densities"]
    w(f"| density levels (count, p25/50/75 → /80) | {density_levels(do['count'].to_numpy())} "
      f"| {density_levels(dn['count'].to_numpy())} | traffic_density_levels |")
    for alt in ("count_moving", "count_r50", "count_moving_r50"):
        if alt in dn.columns:
            w(f"| density levels, variant `{alt}` | - "
              f"| {density_levels(dn[alt].to_numpy())} | (candidate to replace count) |")

    so = size_dist_like_sampler(ro, old["config"])
    sn = size_dist_like_sampler(rn, new["config"])
    fmt = lambda d: ", ".join(f"{k}:{v:.3f}" for k, v in sorted(d.items()))
    w(f"| size_dist (NPC types) | {fmt(so)} | {fmt(sn)} | install_npc_vehicle_type_hook |")
    w("")

    prov = new["config"].get("_provenance", {})
    if prov:
        w("## Provenance of the new version\n")
        w("```json")
        w(json.dumps({k: v for k, v in prov.items() if k != "definitions"},
                     indent=2, ensure_ascii=False))
        w("```")
        w("### Definitions\n")
        for k, v in prov.get("definitions", {}).items():
            w(f"- **{k}** — {v}")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
