#!/usr/bin/env python
"""Throwaway diagnostic: why does PDD sign 2.5 vanish from dump boxes for C3 routes?

Read-only. Scans the L1 dump + the expert sidecars and reports, per route:
  - sign world position implied by dump boxes frame 0000 (via ego_matrix)
  - sign world position recorded in the sidecar (position_world)
  - ego jump between dump frames 0000 and 0001
  - min ego->sign distance over seq>=5 for both sign anchors
  - whether class "2.5" appears in boxes at seq>=5 (C1) or not (C3)
"""
import gzip
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

DUMP = Path("/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench/"
            "plant2_stop_pipeline_debug400/plant2_l1_stop_train/data")
EXPERTS = Path("/home/jovyan/shares/SR006.nfs2/zinkovich/zinkovich/traffic-rule-bench/"
               "pdd-bench/scripts/per_sign_bench/priority_bench/data/stop/trajectories/"
               "debug_train_400/experts/experts_scene_uid_top1.jsonl")
SEQ_MIN = 5


def rj(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def sign_world_from_box(box, ego_matrix):
    """boxes position is CARLA-style ego frame: x=forward, y=right.

    _ego_xy() returns (local_x, -local_y) where local = convert_to_local_coordinates.
    ego_matrix rows are [[cos,-sin,0,x],[sin,cos,0,y],...] i.e. world = R @ local + t
    with local y=left. So local_left = -box_y.
    """
    x = float(box["position"][0])
    y_left = -float(box["position"][1])
    m = np.asarray(ego_matrix, dtype=float)
    R = m[:2, :2]
    t = m[:2, 3]
    return R @ np.array([x, y_left]) + t


def scan_route(rdir: Path):
    mdir = rdir / "measurements"
    bdir = rdir / "boxes"
    frames = sorted(p.name for p in mdir.glob("*.json.gz"))
    n = len(frames)
    if n == 0:
        return None
    out = {"route": rdir.name, "n_frames": n}

    m0 = rj(mdir / frames[0])
    ego0 = np.asarray(m0["pos_global"], dtype=float)
    b0 = rj(bdir / frames[0])
    signs0 = [b for b in b0 if str(b.get("class", "")) == "2.5"]
    out["sign_in_box0"] = len(signs0)
    if signs0:
        sw = sign_world_from_box(signs0[0], m0["ego_matrix"])
        out["sign_world_box0"] = [float(sw[0]), float(sw[1])]
        out["sign_dist_frame0"] = float(math.hypot(*signs0[0]["position"][:2]))
    else:
        out["sign_world_box0"] = None
        out["sign_dist_frame0"] = None

    # ego trajectory
    ego = np.zeros((n, 2))
    ego[0] = ego0

    def _one(i):
        return i, np.asarray(rj(mdir / frames[i])["pos_global"], dtype=float)

    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, p in ex.map(_one, range(1, n)):
            ego[i] = p
    out["ego_jump_0_1"] = float(np.linalg.norm(ego[1] - ego[0])) if n > 1 else None
    out["ego_traj_len"] = float(np.linalg.norm(np.diff(ego, axis=0), axis=1).sum())

    # sign presence in boxes at seq>=5
    def _b(i):
        bb = rj(bdir / frames[i])
        codes = {str(x.get("pdd_code")) for x in bb if x.get("pdd_code")}
        return i, ("2.5" in codes), codes

    present = {}
    all_codes: set = set()
    codes_ge5: set = set()
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, ok, codes in ex.map(_b, range(n)):
            present[i] = ok
            all_codes |= codes
            if i >= SEQ_MIN:
                codes_ge5 |= codes
    out["pdd_codes_any_frame"] = sorted(all_codes)
    out["pdd_codes_ge5"] = sorted(codes_ge5)
    idx_ge5 = [i for i in range(n) if i >= SEQ_MIN]
    out["n_frames_with_sign_all"] = int(sum(present.values()))
    out["n_frames_with_sign_ge5"] = int(sum(present[i] for i in idx_ge5))
    out["first_frame_with_sign"] = next((i for i in range(n) if present[i]), None)
    out["last_frame_with_sign"] = next((i for i in range(n - 1, -1, -1) if present[i]), None)
    out["_ego"] = ego
    return out


def main():
    rows = [json.loads(l) for l in EXPERTS.read_text().splitlines() if l.strip()]
    routes = {}
    for r in rows:
        uid = r["scene_uid"]
        var = r.get("winner_variant") or "default"
        routes[f"{uid}_{var}"] = r

    dirs = sorted(p for p in DUMP.iterdir() if p.is_dir())
    print(f"dump dirs={len(dirs)} manifest rows={len(rows)} "
          f"matched={sum(1 for d in dirs if d.name in routes)}", flush=True)

    only = sys.argv[1:] or None
    results = []
    for k, d in enumerate(dirs):
        if only and d.name not in only:
            continue
        row = routes.get(d.name)
        rec = scan_route(d)
        if rec is None:
            continue
        sc_path = Path(row["sidecar_path"]) if row else None
        rec["backend"] = None
        rec["policy"] = None
        rec["sidecar_signs"] = []
        if sc_path and sc_path.exists():
            sc = json.loads(sc_path.read_text())
            rec["backend"] = sc.get("backend")
            rec["policy"] = sc.get("policy")
            for si in sc.get("signs", []) or []:
                li = si.get("lane_index")
                if isinstance(li, (list, tuple)) and all(
                        isinstance(c, str) and len(c) == 1 for c in li):
                    li = "".join(li)
                rec["sidecar_signs"].append({
                    "sign_class": si.get("sign_class"),
                    "position_world": si.get("position_world"),
                    "lane_index": li,
                    "longitudinal_offset": si.get("longitudinal_offset"),
                    "lateral_offset": si.get("lateral_offset"),
                })
            ecs = sc.get("env_config_summary") or {}
            rec["ecs_keys"] = sorted(ecs.keys())
            for kk in ("spawn_distance_before_end", "sign_distance_before_end",
                       "lane_length", "seed"):
                if kk in ecs:
                    rec["ecs_" + kk] = ecs[kk]
            srow = sc.get("source_row") or {}
            for kk in ("spawn_distance_before_end", "sign_distance_before_end",
                       "lane_length", "n_lanes", "sign_id", "var_idx", "lane_index"):
                if kk in srow:
                    rec["row_" + kk] = srow[kk]
        ego = rec.pop("_ego")
        # min distances over seq>=5
        seg = ego[SEQ_MIN:] if len(ego) > SEQ_MIN else ego
        if rec["sign_world_box0"] is not None:
            sw = np.asarray(rec["sign_world_box0"])
            rec["min_d_boxsign_ge5"] = float(np.linalg.norm(seg - sw, axis=1).min())
            rec["d_boxsign_frame1"] = float(np.linalg.norm(ego[1] - sw)) if len(ego) > 1 else None
        else:
            rec["min_d_boxsign_ge5"] = None
            rec["d_boxsign_frame1"] = None
        pw = None
        for si in rec["sidecar_signs"]:
            if si["sign_class"] == "StopSign" and si["position_world"]:
                pw = np.asarray(si["position_world"][:2], dtype=float)
                break
        rec["n_stopsigns_sidecar"] = sum(
            1 for si in rec["sidecar_signs"] if si["sign_class"] == "StopSign")
        rec["sidecar_sign_classes"] = [si["sign_class"] for si in rec["sidecar_signs"]]
        rec["sidecar_world"] = pw.tolist() if pw is not None else None
        if pw is not None:
            rec["min_d_sidecar_ge5"] = float(np.linalg.norm(seg - pw, axis=1).min())
            rec["d_ego0_sidecar"] = float(np.linalg.norm(ego[0] - pw))
            rec["n_frames_within30_sidecar_ge5"] = int(
                (np.linalg.norm(seg - pw, axis=1) <= 30.0).sum())
        else:
            rec["min_d_sidecar_ge5"] = None
            rec["d_ego0_sidecar"] = None
            rec["n_frames_within30_sidecar_ge5"] = None
        if pw is not None and rec["sign_world_box0"] is not None:
            rec["d_box0sign_to_sidecar"] = float(
                np.linalg.norm(np.asarray(rec["sign_world_box0"]) - pw))
        else:
            rec["d_box0sign_to_sidecar"] = None
        rec["cat"] = "C1" if rec["n_frames_with_sign_ge5"] > 0 else "C3"
        results.append(rec)
        if (k + 1) % 20 == 0:
            print(f"  ... {k+1}/{len(dirs)}", flush=True)

    outp = Path(__file__).parent / "outputs_debug" / "probe_sign25_c1c3.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=1))
    print("wrote", outp, "n=", len(results))


if __name__ == "__main__":
    main()
