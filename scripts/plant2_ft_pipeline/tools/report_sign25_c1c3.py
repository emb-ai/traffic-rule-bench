#!/usr/bin/env python
"""Aggregate the probe_sign25_c1c3.py output into C1 vs C3 distributions."""
import json
from pathlib import Path

import numpy as np

P = Path(__file__).parent / "outputs_debug" / "probe_sign25_c1c3.json"
recs = json.loads(P.read_text())
print(f"routes={len(recs)}")

c1 = [r for r in recs if r["cat"] == "C1"]
c3 = [r for r in recs if r["cat"] == "C3"]
print(f"C1={len(c1)} ({100*len(c1)/len(recs):.1f}%)  C3={len(c3)} ({100*len(c3)/len(recs):.1f}%)")

print("\nbackends:", {b: sum(1 for r in recs if r['backend'] == b) for b in
                      {r['backend'] for r in recs}})
for cat, g in (("C1", c1), ("C3", c3)):
    pol = {}
    for r in g:
        pol[r["policy"]] = pol.get(r["policy"], 0) + 1
    print(f"  {cat} policies:", dict(sorted(pol.items(), key=lambda kv: -kv[1])))

print("\nsidecar sign classes present:")
for cat, g in (("C1", c1), ("C3", c3)):
    n_stop = sum(1 for r in g if r["n_stopsigns_sidecar"] > 0)
    n_pw = sum(1 for r in g if r["sidecar_world"] is not None)
    print(f"  {cat}: routes with a StopSign in sidecar = {n_stop}/{len(g)}; "
          f"with position_world = {n_pw}/{len(g)}")

print("\npdd codes ever written to boxes (any frame):")
codes = {}
for r in recs:
    for c in r["pdd_codes_any_frame"]:
        codes[c] = codes.get(c, 0) + 1
print(" ", codes)

print("\nsign present in boxes frame 0000:",
      f"C1 {sum(1 for r in c1 if r['sign_in_box0'])}/{len(c1)}",
      f"C3 {sum(1 for r in c3 if r['sign_in_box0'])}/{len(c3)}")


def q(g, key):
    v = np.array([r[key] for r in g if r.get(key) is not None], dtype=float)
    if v.size == 0:
        return "n/a"
    return (f"n={v.size:3d} min={v.min():8.2f} p10={np.percentile(v,10):8.2f} "
            f"med={np.median(v):8.2f} p90={np.percentile(v,90):8.2f} max={v.max():8.2f}")


KEYS = [
    ("ego_jump_0_1", "ego jump frame0->frame1 (m)"),
    ("sign_dist_frame0", "ego->env-sign distance at frame 0 (m)"),
    ("d_box0sign_to_sidecar", "env-sign(world) -> sidecar StopSign(world) (m)"),
    ("min_d_boxsign_ge5", "min ego->env-sign over seq>=5 (m)"),
    ("min_d_sidecar_ge5", "min ego->sidecar StopSign over seq>=5 (m)"),
    ("n_frames_within30_sidecar_ge5", "#frames seq>=5 within 30m of sidecar sign"),
    ("n_frames_with_sign_ge5", "#frames seq>=5 with 2.5 in boxes"),
    ("n_frames", "#frames"),
    ("row_spawn_distance_before_end", "row.spawn_distance_before_end"),
    ("row_sign_distance_before_end", "row.sign_distance_before_end"),
]
print()
for k, label in KEYS:
    print(f"{label}")
    print(f"   C1: {q(c1, k)}")
    print(f"   C3: {q(c3, k)}")

# discriminator check
print("\n--- discriminator: min_d_boxsign_ge5 vs 30m ---")
for cat, g in (("C1", c1), ("C3", c3)):
    v = [r["min_d_boxsign_ge5"] for r in g if r["min_d_boxsign_ge5"] is not None]
    print(f"  {cat}: <=30m {sum(1 for x in v if x <= 30)}/{len(v)}  "
          f">30m {sum(1 for x in v if x > 30)}/{len(v)}")

print("\n--- would the sidecar-anchored sign be inside 30m? (post-fix estimate) ---")
for cat, g in (("C1", c1), ("C3", c3)):
    ok = [r for r in g if r["min_d_sidecar_ge5"] is not None and r["min_d_sidecar_ge5"] <= 30]
    print(f"  {cat}: {len(ok)}/{len(g)} routes would have >=1 frame with 2.5")
    if ok:
        fr = np.array([r["n_frames_within30_sidecar_ge5"] for r in ok], dtype=float)
        tot = np.array([r["n_frames"] for r in ok], dtype=float)
        print(f"       frames-in-range: med={np.median(fr):.0f} "
              f"(median coverage {np.median(fr/tot)*100:.0f}% of the route)")
allok = [r for r in recs if r["min_d_sidecar_ge5"] is not None and r["min_d_sidecar_ge5"] <= 30]
print(f"  TOTAL: {len(allok)}/{len(recs)} routes")

print("\n--- ego jump split at 30m ---")
for cat, g in (("C1", c1), ("C3", c3)):
    v = [r["ego_jump_0_1"] for r in g if r["ego_jump_0_1"] is not None]
    print(f"  {cat}: jump<=30m {sum(1 for x in v if x <= 30)}/{len(v)}  "
          f"jump>30m {sum(1 for x in v if x > 30)}/{len(v)}")

print("\n--- outliers: C1 routes with a LARGE env-sign/sidecar mismatch ---")
for r in sorted(c1, key=lambda r: -(r["d_box0sign_to_sidecar"] or 0))[:5]:
    print(f"  {r['route'][:60]:60s} d_mismatch={r['d_box0sign_to_sidecar']:7.1f} "
          f"jump={r['ego_jump_0_1']:6.1f} minbox={r['min_d_boxsign_ge5']:6.1f} "
          f"nsign={r['n_frames_with_sign_ge5']}")
print("--- C3 routes with the SMALLEST mismatch ---")
for r in sorted(c3, key=lambda r: (r["d_box0sign_to_sidecar"] or 1e9))[:5]:
    print(f"  {r['route'][:60]:60s} d_mismatch={r['d_box0sign_to_sidecar']:7.1f} "
          f"jump={r['ego_jump_0_1']:6.1f} minbox={r['min_d_boxsign_ge5']:6.1f} "
          f"minside={r['min_d_sidecar_ge5']:6.1f}")

print("\n--- per-route examples ---")
ex = ["junc_1025291468_lane0_seed777595480_v1_default",
      "junc_1106337009_lane0_seed2343079505_v0_default",
      "junc_12114233576_lane0_seed2211617675_v0_s4",
      "junc_1259970778_lane0_seed2766674357_v0_default",
      "junc_1269994286_lane0_seed895955374_v9_default"]
byname = {r["route"]: r for r in recs}
for name in ex:
    r = byname.get(name)
    if r is None:
        cand = [k for k in byname if k.startswith(name.rsplit("_", 1)[0])]
        print(f"  {name}: NOT FOUND (candidates: {cand})")
        continue
    print(f"  {r['route']} cat={r['cat']} frames={r['n_frames']}")
    print(f"     env-sign world  {r['sign_world_box0']}  (d@frame0={r['sign_dist_frame0']:.2f}m)")
    print(f"     sidecar StopSign {r['sidecar_world']}  mismatch={r['d_box0sign_to_sidecar']:.1f}m")
    print(f"     ego jump 0->1 = {r['ego_jump_0_1']:.1f}m  "
          f"min d(ego,env-sign)|seq>=5 = {r['min_d_boxsign_ge5']:.1f}m  "
          f"min d(ego,sidecar)|seq>=5 = {r['min_d_sidecar_ge5']:.1f}m  "
          f"frames<30m(sidecar) = {r['n_frames_within30_sidecar_ge5']}")

# C1 examples for contrast
print("\n--- 3 C1 examples ---")
for r in sorted(c1, key=lambda r: -r["n_frames_with_sign_ge5"])[:3]:
    print(f"  {r['route']} cat={r['cat']} frames={r['n_frames']} "
          f"sign_frames_ge5={r['n_frames_with_sign_ge5']}")
    print(f"     env-sign world  {r['sign_world_box0']}  sidecar {r['sidecar_world']}  "
          f"mismatch={r['d_box0sign_to_sidecar']:.1f}m")
    print(f"     ego jump 0->1 = {r['ego_jump_0_1']:.1f}m  "
          f"min d(ego,env-sign)|seq>=5 = {r['min_d_boxsign_ge5']:.1f}m  "
          f"min d(ego,sidecar)|seq>=5 = {r['min_d_sidecar_ge5']:.1f}m")
