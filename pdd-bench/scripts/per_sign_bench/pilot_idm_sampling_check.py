#!/usr/bin/env python3
"""Check ego-IDM sample diversity (s1–s4) on one scene set.

Usage:
  python3 pilot_idm_sampling_check.py --runs <dir1> <dir2> ...

Each dir is the --benchmark-output of one run_benchmark.py run with --ego-variant
(default/s1/s2/s3/s4); episodes_*.jsonl are found recursively. The script shows:
  1) per-variant summary: sampled NORMAL_SPEED, actual speed,
     dest/compliance — how much the variants differ on average;
  2) "scene × variant" matrix (NORMAL_SPEED → actual speed):
     on ONE scene, different variants = different drivers;
  3) corr(sample, behavior) — whether the params translate into driving style.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st


def load(root: str) -> list[dict]:
    rows = []
    for f in glob.glob(os.path.join(root, "**", "episodes_*.jsonl"), recursive=True):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    eps = []
    for d in args.runs:
        for r in load(d):
            if r.get("steps", 0) <= 10:
                continue
            p = r.get("ego_params") or {}
            eps.append({
                "variant": r.get("variant", "?"),
                "scene": r.get("scene_id", "?"),
                "ns": p.get("NORMAL_SPEED"),          # m/s (None for default)
                "acc": p.get("ACC_FACTOR"),
                "dw": p.get("DISTANCE_WANTED"),
                "v": r["distance_travelled_m"] / (r["steps"] * 0.1) * 3.6,  # km/h
                "dest": bool(r.get("reached_dest")),
                "comp": r.get("sign_violations", 0) == 0,
            })
    if not eps:
        raise SystemExit("no episodes found")

    variants = sorted({e["variant"] for e in eps})
    scenes = sorted({e["scene"] for e in eps})

    print("=== 1. Per-variant summary ===")
    print(f"{'variant':<9} {'NS m/s (min–max)':>20} {'v act km/h':>12} "
          f"{'dest':>6} {'compl':>6}")
    for v in variants:
        g = [e for e in eps if e["variant"] == v]
        ns = [e["ns"] for e in g if e["ns"] is not None]
        ns_str = (f"{st.mean(ns):5.1f} ({min(ns):.1f}–{max(ns):.1f})"
                  if ns else "  default (10.0)")
        print(f"{v:<9} {ns_str:>20} {st.mean([e['v'] for e in g]):>12.1f} "
              f"{st.mean([e['dest'] for e in g]):>6.2f} "
              f"{st.mean([e['comp'] for e in g]):>6.2f}")

    print("\n=== 2. Scene × variant: sampled NORMAL_SPEED m/s → actual km/h ===")
    hdr = f"{'scene':<22}" + "".join(f"{v:>16}" for v in variants)
    print(hdr)
    for s in scenes:
        cells = []
        for v in variants:
            g = [e for e in eps if e["scene"] == s and e["variant"] == v]
            if not g:
                cells.append(f"{'—':>16}")
            else:
                e = g[0]
                ns = f"{e['ns']:.1f}" if e["ns"] is not None else "def"
                cells.append(f"{ns + '→' + format(e['v'], '.0f'):>16}")
        print(f"{s:<22}" + "".join(cells))

    withp = [e for e in eps if e["ns"] is not None]
    if len(withp) > 3:
        ns = [e["ns"] for e in withp]
        vv = [e["v"] for e in withp]
        mn, mv = st.mean(ns), st.mean(vv)
        cov = sum((a - mn) * (b - mv) for a, b in zip(ns, vv))
        corr = cov / ((sum((a - mn) ** 2 for a in ns) ** 0.5)
                      * (sum((b - mv) ** 2 for b in vv) ** 0.5) or 1.0)
        spread = st.mean([st.pstdev([e["ns"] for e in withp if e["scene"] == s])
                          for s in scenes
                          if len([e for e in withp if e["scene"] == s]) > 1])
        print(f"\n=== 3. corr(sampled NORMAL_SPEED, actual speed) = {corr:.2f} "
              f"(params drive behavior, expect >0.6)")
        print(f"mean NS spread across variants on one scene: {spread:.1f} m/s "
              f"(driver diversity within a scene)")


if __name__ == "__main__":
    main()
