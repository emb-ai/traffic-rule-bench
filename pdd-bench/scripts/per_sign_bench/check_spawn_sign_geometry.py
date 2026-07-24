#!/usr/bin/env python3
"""Batch spawn-vs-sign geometry audit over recorded trajectories.

For every episode: reconstruct the ego track from the recorded pkl (the ego is
identified by matching the track's path length against the sidecar's
distance_travelled_m — unambiguous, unlike speed matching), then measure the
closest approach to the sidecar's sign world point and read the zone-tracker
steps from the metrics.

Verdict per row (the zone tracker is the authority, not raw geometry —
non-braking routes legitimately spawn 100-200 m up the approach edge):
  ok                    in_zone_total_steps > 0 and closest approach <= 8 m
  ZONE_OK_SIGN_POS_BAD  zone was hit but the ego never came near the sign
                        point (spawned inside the zone / wrong route leg)
  NO_ZONE_CONTACT       the trajectory never touched the sign zone (vacuous
                        compliance — excluded from the dataset)

Inputs (one of):
  --picks     experts_*.jsonl from selection (audits the winners)
  --all-runs  all_runs_dedup.jsonl (audits EVERY candidate episode — feeds
              select_experts_coverage.py --geometry-audit for re-selection)

Usage:
  python3 check_spawn_sign_geometry.py \\
      --all-runs $OUT/all_runs_dedup.jsonl --workers 16 \\
      [--path-map OLD=NEW] [--limit 300] --out-csv audit_all.csv
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

OK_MAX_DIST_M = 8.0


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


def audit_episode(task: tuple) -> dict:
    """(sign, scene_uid, policy, variant, sidecar_path, pkl_path) -> record."""
    sign, uid, policy, variant, sp_path, pk_path = task
    rec = {"sign": sign, "scene_uid": uid, "policy": policy, "variant": variant}
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
        in_zone = int((sc.get("metrics") or {}).get("in_zone_total_steps")
                      or 0)
        if in_zone > 0 and ds[k] <= OK_MAX_DIST_M:
            verdict = "ok"
        elif in_zone > 0:
            verdict = "ZONE_OK_SIGN_POS_BAD"
        else:
            verdict = "NO_ZONE_CONTACT"
        rec.update({"start_dist_m": round(ds[0], 1),
                    "min_dist_m": round(ds[k], 1), "min_step": k,
                    "steps": len(ds), "in_zone_steps": in_zone,
                    "ego_how": how, "verdict": verdict})
    except Exception as exc:
        rec.update({"verdict": f"ERROR:{type(exc).__name__}",
                    "error": str(exc)[:120]})
    return rec


def _map_path(p: str, old, new) -> str:
    return p.replace(old, new) if old and p else (p or "")


def load_tasks(args) -> list:
    old = new = None
    if args.path_map:
        old, new = args.path_map.split("=", 1)
    tasks = []
    src = args.all_runs or args.picks
    from_picks = args.picks is not None
    for i, line in enumerate(open(src, encoding="utf-8")):
        if args.limit and i >= args.limit:
            break
        r = json.loads(line)
        if from_picks:
            sign, policy, variant = (r.get("sign"), r.get("winner_policy"),
                                     r.get("winner_variant"))
        else:
            if not r.get("valid"):
                continue
            sign = r.get("sign_code") or r.get("sign_slug")
            policy, variant = r.get("policy"), r.get("variant")
        tasks.append((sign, r.get("scene_uid"), policy, variant,
                      _map_path(r.get("sidecar_path"), old, new),
                      _map_path(r.get("pkl_path"), old, new)))
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--picks", default=None,
                    help="experts_*.jsonl — audit the selection winners")
    ap.add_argument("--all-runs", default=None,
                    help="all_runs_dedup.jsonl — audit every valid candidate")
    ap.add_argument("--path-map", default=None, help="OLD=NEW path prefix swap")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="audit only the first N rows (0 = all)")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()
    if bool(args.picks) == bool(args.all_runs):
        sys.exit("ERROR: provide exactly one of --picks / --all-runs")

    tasks = load_tasks(args)
    print(f"episodes to audit: {len(tasks)}  workers: {args.workers}")

    rows_out = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for j, rec in enumerate(pool.imap_unordered(
                    audit_episode, tasks, chunksize=16)):
                rows_out.append(rec)
                if (j + 1) % 1000 == 0:
                    print(f"  ...{j + 1}/{len(tasks)}", flush=True)
    else:
        for j, task in enumerate(tasks):
            rows_out.append(audit_episode(task))
            if (j + 1) % 1000 == 0:
                print(f"  ...{j + 1}/{len(tasks)}", flush=True)

    from collections import Counter, defaultdict
    by_v = Counter(r["verdict"].split(":")[0] for r in rows_out)
    print("\nverdicts:", dict(by_v.most_common()))
    tab = defaultdict(Counter)
    for r in rows_out:
        tab[r["sign"]][r["verdict"].split(":")[0]] += 1
    for s in sorted(tab):
        t = tab[s]
        print(f"  {s}: total {sum(t.values())} "
              f"ok {t.get('ok', 0)} zone_far {t.get('ZONE_OK_SIGN_POS_BAD', 0)} "
              f"no_zone {t.get('NO_ZONE_CONTACT', 0)} err {t.get('ERROR', 0)}")

    if args.out_csv and rows_out:
        keys = sorted({k for r in rows_out for k in r})
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows_out)
        print(f"csv: {args.out_csv}")


if __name__ == "__main__":
    main()
