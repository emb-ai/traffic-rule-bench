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


def _ego_track(d: dict, sc: dict) -> list:
    """Ego positions per step, from either pkl format.

    ScenarioDescription: tracks[metadata.sdc_id].state.position.
    Raw dump_episode:    frame -> [FrameInfo, ...] with .step_info[obj_id];
    the ego is identified as the frame-0 object whose speed is closest to
    the sidecar's initial_speed_mps (present through the whole episode).
    """
    if "metadata" in d and "tracks" in d:
        sdc = d["metadata"]["sdc_id"]
        st = d["tracks"][sdc]["state"]
        vels = [math.hypot(float(v[0]), float(v[1]))
                for v in st.get("velocity", [])]
        return ([(float(p[0]), float(p[1])) for p in st["position"]], vels)

    frames = [fg[0].step_info if fg else {} for fg in d["frame"]]
    v0 = float(sc.get("metrics", {}).get("initial_speed_mps")
               or sc.get("initial_speed_mps") or 0.0)
    best_id, best_gap = None, 1e9
    n_frames = len(frames)
    for oid, st in frames[0].items():
        presence = sum(1 for f in frames if oid in f) / max(1, n_frames)
        if presence < 0.95:
            continue
        vel = st.get("velocity", (0.0, 0.0))
        speed = math.hypot(float(vel[0]), float(vel[1]))
        gap = abs(speed - v0)
        if gap < best_gap:
            best_gap, best_id = gap, oid
    if best_id is None:
        raise SystemExit("ERROR: could not identify the ego in raw frames")
    print(f"[info] raw-frame pkl; ego={best_id} "
          f"(speed gap to sidecar v0: {best_gap:.2f} m/s)")
    pos, vels = [], []
    for f in frames:
        if best_id not in f:
            continue
        st = f[best_id]
        pos.append((float(st["position"][0]), float(st["position"][1])))
        vel = st.get("velocity", (0.0, 0.0))
        vels.append(math.hypot(float(vel[0]), float(vel[1])))
    return pos, vels


def main() -> None:
    ep = Path(sys.argv[1])
    sc = json.load(open(ep / "replay.json"))
    d = pickle.load(open(ep / "replay.pkl", "rb"))
    pos, vels = _ego_track(d, sc)

    print(f"ego start {pos[0]} -> end {pos[-1]}")
    if vels:
        # Braking-spawn discriminator: WHERE the ego first held <= 20 km/h
        # is where the speed-limit sign actually acted.
        lim = 5.7  # 20 km/h + epsilon, m/s
        first_slow = next((k for k in range(len(vels))
                           if all(v <= lim for v in vels[k:k + 20])), None)
        print(f"speed m/s: v0 {vels[0]:.1f} | step30 {vels[min(30, len(vels)-1)]:.1f} "
              f"| min {min(vels):.1f} | end {vels[-1]:.1f}")
        print(f"first sustained <=20 km/h at step: {first_slow}")

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
