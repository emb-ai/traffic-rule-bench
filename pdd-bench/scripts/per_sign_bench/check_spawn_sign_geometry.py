#!/usr/bin/env python3
"""Batch spawn-vs-sign geometry audit over selected expert trajectories.

For every pick: reconstruct the ego track from the recorded pkl (the ego is
identified by matching the track's path length against the sidecar's
distance_travelled_m — unambiguous, unlike speed matching), then compare:

  start_dist  = ego spawn -> sign world point
  min_dist    = closest approach to the sign point (and at which step)
  expectation = manifest sign_s (ego must spawn ~sign_s metres before the sign)

A row is SUSPICIOUS when start_dist exceeds sign_s by more than --slack:
the ego did not spawn on the sign approach the manifest describes.

Usage:
  python3 check_spawn_sign_geometry.py \\
      --picks $OUT/experts_scene_uid_top1.jsonl \\
      [--path-map OLD=NEW] [--slack 30] [--limit 300] [--out-csv audit.csv]
"""
import argparse
import csv
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


def ego_track(d: dict, sc: dict):
    """-> (positions, how) with the ego chosen by path-length match."""
    if "metadata" in d and "tracks" in d:
        sdc = d["metadata"]["sdc_id"]
        pos = [(float(p[0]), float(p[1]))
               for p in d["tracks"][sdc]["state"]["position"]]
        return pos, "sd:sdc_id"

    frames = [fg[0].step_info if fg else {} for fg in d["frame"]]
    n = len(frames)
    target = float(sc.get("metrics", {}).get("distance_travelled_m") or 0.0)
    best_id, best_gap = None, 1e18
    for oid in frames[0]:
        if sum(1 for f in frames if oid in f) / max(1, n) < 0.95:
            continue
        pts = [f[oid]["position"] for f in frames if oid in f]
        plen = sum(math.hypot(float(b[0]) - float(a[0]),
                              float(b[1]) - float(a[1]))
                   for a, b in zip(pts, pts[1:]))
        gap = abs(plen - target)
        if gap < best_gap:
            best_gap, best_id = gap, oid
    if best_id is None:
        return None, "no-candidate"
    pos = [(float(f[best_id]["position"][0]), float(f[best_id]["position"][1]))
           for f in frames if best_id in f]
    return pos, f"raw:pathlen_gap={best_gap:.1f}m"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--picks", required=True,
                    help="experts_scene_uid_top1.jsonl (or any picks jsonl)")
    ap.add_argument("--path-map", default=None, help="OLD=NEW path prefix swap")
    ap.add_argument("--slack", type=float, default=30.0,
                    help="allowed start_dist excess over sign_s, m (default 30)")
    ap.add_argument("--limit", type=int, default=0,
                    help="audit only the first N picks (0 = all)")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    old = new = None
    if args.path_map:
        old, new = args.path_map.split("=", 1)

    rows_out = []
    n_ok = n_susp = n_err = 0
    for i, line in enumerate(open(args.picks, encoding="utf-8")):
        if args.limit and i >= args.limit:
            break
        p = json.loads(line)
        sp_path = p.get("sidecar_path") or ""
        pk_path = p.get("pkl_path") or ""
        if old:
            sp_path = sp_path.replace(old, new)
            pk_path = pk_path.replace(old, new)
        rec = {"sign": p.get("sign"), "scene_uid": p.get("scene_uid"),
               "policy": f"{p.get('winner_policy')}_{p.get('winner_variant')}"}
        try:
            sc = json.load(open(sp_path))
            d = pickle.load(open(pk_path, "rb"))
            signs = [s for s in sc.get("signs", []) if s.get("position_world")]
            if not signs:
                raise ValueError("no sign position in sidecar")
            spos = signs[0]["position_world"]
            pos, how = ego_track(d, sc)
            if pos is None:
                raise ValueError(how)
            ds = [math.hypot(a - spos[0], b - spos[1]) for a, b in pos]
            k = min(range(len(ds)), key=ds.__getitem__)
            sign_s = float(sc.get("source_row", {}).get("sign_s")
                           or sc.get("source_row", {}).get("sign_spawn_distance")
                           or 0.0)
            susp = ds[0] > sign_s + args.slack
            rec.update({"start_dist_m": round(ds[0], 1),
                        "min_dist_m": round(ds[k], 1), "min_step": k,
                        "steps": len(ds), "sign_s": round(sign_s, 1),
                        "ego_how": how,
                        "verdict": "SUSPICIOUS" if susp else "ok"})
            n_susp += susp
            n_ok += (not susp)
        except Exception as exc:
            rec.update({"verdict": f"ERROR:{type(exc).__name__}",
                        "error": str(exc)[:120]})
            n_err += 1
        rows_out.append(rec)
        if rec["verdict"] != "ok":
            print(rec)

    print(f"\naudited: {len(rows_out)}  ok: {n_ok}  suspicious: {n_susp}  "
          f"errors: {n_err}")
    by_sign = {}
    for r in rows_out:
        if r["verdict"] == "SUSPICIOUS":
            by_sign[r["sign"]] = by_sign.get(r["sign"], 0) + 1
    if by_sign:
        print("suspicious per sign:", dict(sorted(by_sign.items())))

    if args.out_csv and rows_out:
        keys = sorted({k for r in rows_out for k in r})
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows_out)
        print(f"csv: {args.out_csv}")


if __name__ == "__main__":
    main()
