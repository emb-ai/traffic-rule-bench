"""Collect statistics + histograms for the 3.24 braking-spawn SUMO scenes.

For every braking-spawn scene it builds the env, reads the resolved spawn
(via _braking_spawn_info) and the planned route, and records:
  - v_target_kmh   — the speed limit being checked (from the net edge)
  - v0_kmh         — ego initial speed (sampled above the limit)
  - route_len_m    — length of the planned route spawn→…→sign→destination
  - d_required_m / d_achieved_m, status (OK / insufficient / invalid)

Outputs (to --out-dir):
  - braking_scene_stats.jsonl   (per-scene rows)
  - hist_route_length.png, hist_speed_limit.png, hist_initial_speed.png
  - prints a summary table.

Run:
  ~/miniconda3/envs/metadrive_sdc/bin/python scripts/vis_env/braking_3_24_stats.py \
      --scenes-root <.../scenes> --out-dir <.../stats>
"""
import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

SCRIPT = Path(__file__).resolve()
PDD = SCRIPT.parent.parent.parent
for p in (str(PDD), str(PDD.parent / "metadrive"), str(PDD / "scripts" / "per_sign_bench")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sumo_space.sumo_catalog import build_catalog  # noqa: E402
from envs.sumo_env import TrafficSignSumoEnv  # noqa: E402


def _route_length_m(env) -> float:
    """Sum of lane lengths over the ego's planned route checkpoints (dedup)."""
    veh = env.agent if hasattr(env, "agent") else env.vehicle
    nav = getattr(veh, "navigation", None)
    rn = env.current_map.road_network
    ckpts = list(getattr(nav, "checkpoints", None) or [])
    total, prev = 0.0, None
    for c in ckpts:
        if c == prev:
            continue
        prev = c
        try:
            total += float(rn.get_lane(c).length)
        except Exception:
            pass
    return total


def collect(scenes_root: Path, target: int, n_v0: int, only_v_idx=None):
    rows = build_catalog(scenes_root=scenes_root, n_per_category=target, n_variations=1,
                         sign_categories=["3.24"], seed=42, n_v0_samples=n_v0)
    if only_v_idx is not None:
        rows = [r for r in rows if int(r.get("v_idx", 0)) == int(only_v_idx)]
        print(f"[filter] only v_idx={only_v_idx}: {len(rows)} rows (1 sample per scene)")
    out = []
    for i, r in enumerate(rows):
        cfg = dict(use_render=False, manual_control=False, use_mesh_terrain=False,
                   log_level=logging.CRITICAL, map_name=str(scenes_root / r["net_path"]),
                   sign_type="3.24", sign_spawn_distance=float(r["sign_s"]),
                   num_scenarios=100000, spawn_lane_num=int(r.get("spawn_lane_num", 0)),
                   ego_braking_spawn=True, ego_spawn_v0_ms=float(r["spawn_velocity_ms"]),
                   ego_brake_d_required=float(r["d_required_m"]),
                   ego_v_target_kmh=float(r["v_target_kmh"]),
                   vehicle_config={"show_lidar": False, "spawn_lane_index": r["road_id"],
                                   "destination": r.get("destination_lane_id")})
        try:
            env = TrafficSignSumoEnv(cfg)
            env.reset(seed=int(r["sign_id"]) % 100000)
        except Exception:
            continue
        info = getattr(env, "_braking_spawn_info", None) or {}
        status = ("invalid" if info.get("braking_invalid")
                  else "insufficient" if info.get("insufficient_runway") else "ok")
        out.append({
            "scene_id": r["scene_id"],
            "v_target_kmh": float(r["v_target_kmh"]),
            "v0_kmh": round(float(r["spawn_velocity_ms"]) * 3.6, 2),
            "route_len_m": round(_route_length_m(env), 2),
            "d_required_m": float(r["d_required_m"]),
            "d_achieved_m": float(info.get("ego_d_achieved_m", 0.0)),
            "routed_through_sign": bool(info.get("routed_through_sign")),
            "status": status,
        })
        env.close()
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(rows)} scenes")
    return out


def _hist(values, title, xlabel, path, bins=20, color="#3b76af"):
    plt.figure(figsize=(7, 4.2))
    plt.hist(values, bins=bins, color=color, edgecolor="white")
    plt.title(f"{title}  (n={len(values)})")
    plt.xlabel(xlabel)
    plt.ylabel("scenes")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target", type=int, default=250)
    ap.add_argument("--n-v0", type=int, default=1)
    ap.add_argument("--only-v-idx", type=int, default=None,
                    help="keep only this v0 sample index per scene (1 variant). "
                         "Omit to use ALL samples (all rows).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Tag outputs so the 'all samples' and 'single variant' runs don't overwrite.
    tag = f"_v{args.only_v_idx}" if args.only_v_idx is not None else "_all"
    stats = collect(Path(args.scenes_root), args.target, args.n_v0, only_v_idx=args.only_v_idx)

    with open(out_dir / f"braking_scene_stats{tag}.jsonl", "w") as f:
        for s in stats:
            f.write(json.dumps(s) + "\n")

    usable = [s for s in stats if s["status"] != "invalid"]
    rl = [s["route_len_m"] for s in usable]
    lim = [s["v_target_kmh"] for s in usable]
    v0 = [s["v0_kmh"] for s in usable]
    d2sign = [s["d_achieved_m"] for s in usable]   # distance route-start(spawn) → sign

    scope = "all samples" if args.only_v_idx is None else f"1 sample (v_idx={args.only_v_idx})"
    _hist(d2sign, f"Distance spawn → sign [{scope}]", "distance to sign, m",
          out_dir / f"hist_spawn_to_sign{tag}.png", color="#8064a2")
    _hist(rl, f"Route length spawn→sign→dest [{scope}]", "route length, m",
          out_dir / f"hist_route_length{tag}.png", color="#3b76af")
    _hist(lim, f"Speed limit checked [{scope}]", "limit, km/h",
          out_dir / f"hist_speed_limit{tag}.png", bins=[0, 15, 25, 35, 45, 55, 65, 75], color="#c0504d")
    _hist(v0, f"Ego initial speed v0 [{scope}]", "v0, km/h",
          out_dir / f"hist_initial_speed{tag}.png", color="#4f9d69")

    def stat(x):
        a = np.array(x) if x else np.array([0.0])
        return f"min={a.min():.1f} med={np.median(a):.1f} mean={a.mean():.1f} max={a.max():.1f}"

    print(f"\n=== 3.24 braking-scene stats [{scope}]: {len(stats)} rows "
          f"({len(usable)} usable, {len(stats)-len(usable)} invalid) ===")
    print(f"dist spawn→sign(m):{stat(d2sign)}")
    print(f"route length (m):  {stat(rl)}")
    print(f"speed limit (kmh): {stat(lim)}")
    print(f"initial v0 (kmh):  {stat(v0)}")
    print(f"PNGs + jsonl → {out_dir}")


if __name__ == "__main__":
    main()
