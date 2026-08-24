#!/usr/bin/env python
"""Summarise the per-frame x_objs traces written by run_xobjs_probe.py.

For every episode: how many frames carried the 2.5 stop-sign token in x_objs,
how many frames the sign existed in traffic_sign_manager, the detection distance
range, and whether any frame lost the token while the sign was in range (which
is what a collector/cutoff bug would look like). Also reports the model's
desired speed and commanded action over the approach so the presence of the
token can be weighed against the behaviour it produced.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def gt25(rec: dict):
    for m in rec.get("mgr") or []:
        if m.get("pdd") == "2.5":
            return m
    return None


def approach_window(recs: list[dict], near_m: float = 12.0) -> list[dict]:
    """The first approach to the stop line: from the first frame where the sign
    is ahead (ego-frame x > 0) and within ``near_m``, up to the frame where the
    ego passes it (x <= 0).

    Anchoring on "sign still ahead" rather than on a step offset keeps violators
    and compliant episodes comparable: a compliant episode stops, waits, then
    accelerates away, so a fixed window before the pass frame would measure the
    departure, not the approach."""
    start = None
    for i, r in enumerate(recs):
        s = r.get("s25") or gt25(r)
        if s and s["x"] > 0.0 and s["d"] <= near_m:
            start = i
            break
    if start is None:
        return []
    out = []
    for r in recs[start:]:
        s = r.get("s25") or gt25(r)
        if s is None or s["x"] <= 0.0:
            break
        out.append(r)
    return out


def summarise(uid: str, path: Path, group: str) -> dict:
    recs = load(path)
    if not recs:
        return {"scene_uid": uid, "group": group, "frames": 0}
    in_x = [r for r in recs if r.get("n25")]
    in_mgr = [r for r in recs if gt25(r)]
    # A frame where the sign is within the 30 m gate but absent from x_objs
    # is the signature of a collector / filtering bug.
    dropped = [r for r in recs
               if gt25(r) and gt25(r)["d"] <= 30.0 and not r.get("n25")]
    dists = [r["s25"]["d"] for r in in_x]
    win = approach_window(recs)
    accel = [r["act"][1] for r in win]
    speeds = [r["v"] for r in win]
    dspeed = [r.get("desired_speed") for r in win if r.get("desired_speed") is not None]
    win_have_token = sum(1 for r in win if r.get("n25"))
    return {
        "scene_uid": uid,
        "group": group,
        "frames": len(recs),
        "frames_25_in_xobjs": len(in_x),
        "frames_25_in_manifest_mgr": len(in_mgr),
        "frames_25_dropped_in_range": len(dropped),
        "first_detect_d": round(dists[0], 2) if dists else None,
        "max_detect_d": round(max(dists), 2) if dists else None,
        "min_detect_d": round(min(dists), 2) if dists else None,
        "approach_frames": len(win),
        "approach_frames_with_token": win_have_token,
        "approach_min_ego_speed": round(min(speeds), 2) if speeds else None,
        "approach_mean_accel": round(st.mean(accel), 3) if accel else None,
        "approach_braking_frac": (round(sum(1 for a in accel if a < -0.05) / len(accel), 2)
                                  if accel else None),
        "approach_min_desired_speed": round(min(dspeed), 2) if dspeed else None,
        "approach_frames_desired_stop": sum(1 for d in dspeed if d < 0.05),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    work = Path(args.work).resolve()

    groups: dict[str, str] = {}
    for res in sorted(work.glob("probe_results_*.jsonl")):
        for line in res.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                groups[r["scene_uid"]] = r.get("group", "?")

    rows = []
    for path in sorted((work / "logs").glob("*.xobjs.jsonl")):
        uid = path.name[: -len(".xobjs.jsonl")]
        rows.append(summarise(uid, path, groups.get(uid, "?")))

    rows.sort(key=lambda r: (r["group"], r["scene_uid"]))
    out_path = Path(args.out) if args.out else (work / "xobjs_report.json")
    out_path.write_text(json.dumps(rows, indent=2))

    hdr = (f"{'group':10} {'frames':>6} {'in_xobjs':>8} {'in_mgr':>7} {'drop':>5} "
           f"{'first_d':>7} {'min_d':>6} {'appr':>5} {'appr_tok':>8} {'v_min':>6} "
           f"{'acc':>6} {'brake':>6} {'ds_min':>6} {'ds_stop':>7}  scene_uid")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['group']:10} {r['frames']:6d} {r['frames_25_in_xobjs']:8d} "
              f"{r['frames_25_in_manifest_mgr']:7d} {r['frames_25_dropped_in_range']:5d} "
              f"{str(r['first_detect_d']):>7} {str(r['min_detect_d']):>6} "
              f"{r['approach_frames']:5d} {r['approach_frames_with_token']:8d} "
              f"{str(r['approach_min_ego_speed']):>6} "
              f"{str(r['approach_mean_accel']):>6} {str(r['approach_braking_frac']):>6} "
              f"{str(r['approach_min_desired_speed']):>6} "
              f"{r['approach_frames_desired_stop']:7d}  {r['scene_uid']}")

    for g in ("violator", "compliant"):
        sub = [r for r in rows if r["group"] == g and r["frames"]]
        if not sub:
            continue
        full = sum(1 for r in sub
                   if r["approach_frames_with_token"] == r["approach_frames"] > 0)
        print(f"\n[{g}] n={len(sub)}  "
              f"2.5 token on EVERY approach frame: {full}/{len(sub)}  "
              f"dropped-in-range frames: {sum(r['frames_25_dropped_in_range'] for r in sub)}")
        acc = [r["approach_mean_accel"] for r in sub if r["approach_mean_accel"] is not None]
        brk = [r["approach_braking_frac"] for r in sub if r["approach_braking_frac"] is not None]
        vmin = [r["approach_min_ego_speed"] for r in sub
                if r["approach_min_ego_speed"] is not None]
        dsm = [r["approach_min_desired_speed"] for r in sub
               if r["approach_min_desired_speed"] is not None]
        stopped = sum(1 for v in vmin if v < 0.5)
        if acc:
            print(f"[{g}] approach: mean accel {st.mean(acc):+.3f}  "
                  f"braking frac {st.mean(brk):.2f}  "
                  f"min ego speed {st.mean(vmin):.2f} m/s  "
                  f"min desired speed {st.mean(dsm):.2f} m/s  "
                  f"came to a halt: {stopped}/{len(vmin)}")
    print(f"\nreport: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
