#!/usr/bin/env python3
"""A/B-сравнение carl legacy vs tracking по episodes_*.jsonl двух пилотных прогонов.

Использование:
  python3 pilot_carl_speed_check.py --legacy <out_dir_legacy> --tracking <out_dir_tracking>

Каждый out_dir — --benchmark-output соответствующего запуска run_benchmark.py
(скрипт сам найдёт episodes_*.jsonl рекурсивно). Печатает таблицу метрик и вердикт
по стоп-критериям из плана: dest_rate не ниже −5 пп, OOR ≤ 1.5×, steer_delta без роста.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st


def load_episodes(root: str) -> list[dict]:
    files = glob.glob(os.path.join(root, "**", "episodes_*.jsonl"), recursive=True)
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        raise SystemExit(f"нет episodes_*.jsonl под {root}")
    return rows


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("steps", 0) > 10]
    speeds = [r["distance_travelled_m"] / (r["steps"] * 0.1) * 3.6 for r in ok]
    return {
        "episodes": len(rows),
        "median_speed_kmh": st.median(speeds) if speeds else 0.0,
        "mean_speed_kmh": st.mean(speeds) if speeds else 0.0,
        "dest_rate": st.mean([bool(r.get("reached_dest")) for r in rows]),
        "crash_rate": st.mean([bool(r.get("crashed")) for r in rows]),
        "oor_rate": st.mean([bool(r.get("out_of_road")) for r in rows]),
        "hard_brake/ep": st.mean([r.get("hard_brake_count", 0) for r in rows]),
        "hard_accel/ep": st.mean([r.get("hard_accel_count", 0) for r in rows]),
        "steer_delta": st.mean([r.get("mean_abs_steer_delta", 0.0) for r in rows]),
        "sign_compliance": st.mean([r.get("sign_violations", 0) == 0 for r in rows]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy", required=True)
    ap.add_argument("--tracking", required=True)
    args = ap.parse_args()

    a = summarize(load_episodes(args.legacy))
    b = summarize(load_episodes(args.tracking))

    w = max(len(k) for k in a)
    print(f"{'метрика':<{w}}  {'legacy':>10}  {'tracking':>10}  {'Δ':>8}")
    for k in a:
        va, vb = a[k], b[k]
        d = vb - va
        print(f"{k:<{w}}  {va:>10.3f}  {vb:>10.3f}  {d:>+8.3f}")

    print("\n--- вердикт ---")
    faster = b["median_speed_kmh"] > a["median_speed_kmh"] + 3
    print(f"ускорился:            {'ДА' if faster else 'НЕТ'} "
          f"({a['median_speed_kmh']:.1f} → {b['median_speed_kmh']:.1f} км/ч)")
    checks = [
        ("dest_rate не упал (>5пп — стоп)", b["dest_rate"] >= a["dest_rate"] - 0.05),
        ("OOR ≤ 1.5×", b["oor_rate"] <= max(a["oor_rate"], 0.02) * 1.5),
        ("латераль не задета (steer_delta)", b["steer_delta"] <= a["steer_delta"] * 1.3 + 0.005),
        ("перестал ехать на тормозе", b["hard_brake/ep"] < a["hard_brake/ep"]),
    ]
    ok_all = True
    for name, ok in checks:
        ok_all &= ok
        print(f"{'✓' if ok else '✗'} {name}")
    print("\nИТОГ:", "фикс принят — можно гнать полный пилот/eval"
          if (faster and ok_all) else "есть красные флаги — см. таблицу")


if __name__ == "__main__":
    main()
