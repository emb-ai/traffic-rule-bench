#!/usr/bin/env python3
"""Ego-vs-sign geometry check for one recorded episode.

Usage:  python3 check_sign_distance.py <episode_dir_with_replay.pkl_and_json>

Prints the ego->sign distance profile: where the minimum happened tells
whether the ego actually drove THROUGH the sign point (correct direction)
or started at the sign and drove away (flipped edge orientation).
"""
import json
import math
import pickle
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PDD_BENCH_DIR = SCRIPT_PATH.parent.parent.parent
METADRIVE_DIR = PDD_BENCH_DIR.parent / "metadrive"
for _p in (PDD_BENCH_DIR, METADRIVE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> None:
    ep = Path(sys.argv[1])
    sc = json.load(open(ep / "replay.json"))
    d = pickle.load(open(ep / "replay.pkl", "rb"))

    sdc = d["metadata"]["sdc_id"]
    pos = d["tracks"][sdc]["state"]["position"]

    for s in sc.get("signs", []):
        sp = s.get("position_world")
        if not sp:
            print(f"{s['sign_class']}: no world position in sidecar")
            continue
        ds = [math.hypot(float(p[0]) - sp[0], float(p[1]) - sp[1]) for p in pos]
        i = min(range(len(ds)), key=ds.__getitem__)
        print(f"sign {s['sign_class']} @ ({sp[0]:.1f}, {sp[1]:.1f})")
        print(f"  ego->sign: start {ds[0]:.1f} m | min {ds[i]:.1f} m at step {i} "
              f"of {len(ds)} | end {ds[-1]:.1f} m")
        if ds[i] < 4.0 and 0 < i < len(ds) - 1:
            verdict = "PASSED THROUGH the sign (direction OK)"
        elif i == 0:
            verdict = "started AT the sign and moved AWAY (flipped orientation?)"
        else:
            verdict = "never approached the sign"
        print(f"  verdict: {verdict}")
        print("  first 60 steps, every 4th:",
              [round(x, 1) for x in ds[:60:4]])


if __name__ == "__main__":
    main()
