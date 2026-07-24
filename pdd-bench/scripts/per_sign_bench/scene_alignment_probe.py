#!/usr/bin/env python3
"""Spawn-vs-sign alignment probe: build each scene exactly like the recorder
(env reset only, no rollout) and report where the ego and the sign actually
are — same lane? sign ahead of the spawn? how far?

Per row classification:
  SAME_LANE_AHEAD   ego on the sign's lane, sign ahead of the spawn — correct
  APPROACH_ON_ROUTE ego spawns upstream, but the sign edge IS on the route —
                    correct (approach room before the sign)
  SAME_LANE_BEHIND  ego on the sign's lane but PAST the sign (spawned inside
                    the zone) — broken variation
  SAME_EDGE_OTHER_LANE  same edge, different lane index
  OTHER_EDGE        ego spawns on a different edge than the sign
  NO_SIGN / ERROR   sign object missing / env build failed

Usage (server, plant2 env, from per_sign_bench):
  python3 scene_alignment_probe.py \\
      --catalog $SM/.../catalog_fv_train80.jsonl \\
      --scenes-root $SM/scenes_balanced \\
      [--per-scene-id 1] [--limit 0] [--relocate 1] --out-csv alignment.csv
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_DIR = SCRIPT_PATH.parent
PDD_BENCH_DIR = BENCHMARK_DIR.parent.parent
METADRIVE_DIR = PDD_BENCH_DIR.parent / "metadrive"
for _p in (PDD_BENCH_DIR, METADRIVE_DIR, BENCHMARK_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _lane_id(lane) -> str:
    idx = getattr(lane, "index", None)
    return idx if isinstance(idx, str) else str(tuple(idx) if idx else None)


def _edge_of(lane_id: str) -> str:
    # sumo lane ids look like "lane_35571371#0_0" -> edge "lane_35571371#0"
    return lane_id.rsplit("_", 1)[0] if "_" in lane_id else lane_id


def probe_row(row: dict, scenes_root, relocate: bool) -> dict:
    from expert_replay import _build_env, _scene_uid
    from bench import env_builders as _eb

    _eb.RELOCATE_EGO_TO_SIGN_LANE = relocate
    rec = {"scene_uid": _scene_uid(row), "sign": row.get("sign_code"),
           "manifest_sign_s": row.get("sign_s"),
           "spawn_lane_num": row.get("spawn_lane_num")}
    env = None
    try:
        env = _build_env(row, "sumo", max_steps=10, record_episode=False,
                         ego_policy_cls=None, render=False,
                         scenes_root=scenes_root)
        env_seed = (int(row.get("sign_id", 0) or 0)
                    + int(row.get("var_idx", 0) or 0)) % 100000
        env.reset(seed=env_seed)

        veh = env.vehicle
        ego_lane = veh.lane
        ego_lid = _lane_id(ego_lane)
        ego_s = float(ego_lane.local_coordinates(veh.position)[0])

        mgr = getattr(env.engine, "traffic_sign_manager", None)
        signs = list(getattr(mgr, "signs", []) or [])
        if not signs:
            rec["verdict"] = "NO_SIGN"
            return rec
        s = signs[0]
        sign_lid = _lane_id(s.lane)
        sign_s = float(getattr(s, "placement_long", 0.0))
        try:
            ego_on_sign_lane_s = float(
                s.lane.local_coordinates(veh.position)[0])
        except Exception:
            ego_on_sign_lane_s = None

        nav = getattr(veh, "navigation", None)
        ckpts = [str(c) for c in (getattr(nav, "checkpoints", None) or [])]
        sign_on_route = sign_lid in ckpts

        rec.update({"ego_lane": ego_lid, "ego_s": round(ego_s, 1),
                    "sign_lane": sign_lid, "sign_s_runtime": round(sign_s, 1),
                    "sign_class": type(s).__name__,
                    "sign_on_route": sign_on_route,
                    "gap_m": (round(sign_s - ego_on_sign_lane_s, 1)
                              if ego_on_sign_lane_s is not None else None)})
        if ego_lid == sign_lid:
            rec["verdict"] = ("SAME_LANE_AHEAD"
                              if sign_s - ego_s > 0 else "SAME_LANE_BEHIND")
        elif sign_on_route:
            rec["verdict"] = "APPROACH_ON_ROUTE"   # spawn upstream, sign ahead on route
        elif _edge_of(ego_lid) == _edge_of(sign_lid):
            rec["verdict"] = "SAME_EDGE_OTHER_LANE"
        else:
            rec["verdict"] = "OTHER_EDGE"
        return rec
    except Exception as exc:
        rec["verdict"] = f"ERROR:{type(exc).__name__}"
        rec["error"] = str(exc)[:120]
        return rec
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--scenes-root", required=True)
    ap.add_argument("--per-scene-id", type=int, default=1,
                    help="probe at most N variations per scene_id (default 1)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--relocate", type=int, default=1,
                    help="RELOCATE_EGO_TO_SIGN_LANE (1 = IDM-family recording)")
    ap.add_argument("--signs", nargs="*", default=None)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    rows = []
    seen = Counter()
    for line in open(args.catalog, encoding="utf-8"):
        r = json.loads(line)
        if r.get("valid") is False:
            continue
        if args.signs and str(r.get("sign_code")) not in args.signs:
            continue
        sid = r.get("scene_id")
        if seen[sid] >= args.per_scene_id:
            continue
        seen[sid] += 1
        rows.append(r)
        if args.limit and len(rows) >= args.limit:
            break
    print(f"probing {len(rows)} rows "
          f"({len(seen)} scene_ids, relocate={bool(args.relocate)})", flush=True)

    out = []
    for i, r in enumerate(rows):
        rec = probe_row(r, Path(args.scenes_root), bool(args.relocate))
        out.append(rec)
        if rec["verdict"] not in ("SAME_LANE_AHEAD", "APPROACH_ON_ROUTE"):
            print(rec, flush=True)
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(rows)}", flush=True)

    by_v = Counter(r["verdict"].split(":")[0] for r in out)
    print("\nverdicts:", dict(by_v.most_common()))
    by_sign = {}
    for r in out:
        by_sign.setdefault(r["sign"], Counter())[r["verdict"].split(":")[0]] += 1
    for s in sorted(by_sign):
        print(f"  {s}: {dict(by_sign[s].most_common())}")

    if args.out_csv and out:
        keys = sorted({k for r in out for k in r})
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(out)
        print(f"csv: {args.out_csv}")


if __name__ == "__main__":
    main()
