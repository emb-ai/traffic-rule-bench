"""Render top-down GIFs of combined (paired) SUMO zone scenes.

Drives the ego through a combined start+end scene (produced by
sumo_space/sumo_zone_pairing.py) and records a top-down GIF showing both sign
icons and the route through the zone.

Usage:
    python scripts/vis_env/view_paired_sumo.py --out-root <scenes_paired dir> \
        [--code 3.24] [--only <scene_id substring>] [--steps 220]
GIFs are written to <out-root>/gifs/.
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
PDD = SCRIPT.parent.parent.parent          # .../pdd-bench
for p in (str(PDD), str(PDD.parent / "metadrive")):
    if p not in sys.path:
        sys.path.insert(0, p)

from envs.sumo_env import TrafficSignSumoEnv  # noqa: E402


def render_row(row, out_root: Path, gifs: Path, steps: int) -> str:
    net = str(out_root / row["net_path"])   # keep the symlink so meta.json beside it is read
    cfg = dict(
        use_render=False, manual_control=False, use_mesh_terrain=False,
        log_level=logging.CRITICAL, map_name=net, sign_type=row["sign_code"],
        sign_spawn_distance=30.0, num_scenarios=100000,
        vehicle_config={"show_lidar": False, "spawn_lane_index": row["road_id"],
                        "destination": row.get("destination_lane_id")},
    )
    env = TrafficSignSumoEnv(cfg)
    env.reset(seed=int(row["sign_id"]) % 100000)
    frames = 0
    for _ in range(steps):
        _, _, term, trunc, _ = env.step([0.0, 1.0])
        try:
            env.render(mode="top_down", film_size=(1600, 1600), scaling=8.0,
                       screen_size=(700, 700), semantic_map=True,
                       target_agent_heading_up=True, screen_record=True, window=False)
            frames += 1
        except Exception as exc:
            env.close()
            return f"render err: {exc}"
        if term or trunc:
            break
    gif = gifs / f"{row['scene_id']}.gif"
    msg = "no renderer"
    if getattr(env, "top_down_renderer", None) is not None:
        env.top_down_renderer.generate_gif(str(gif), duration=40)
        msg = f"{gif}  ({frames} frames, zone {row.get('zone_length_m')}m / {row.get('n_zone_edges')} edges)"
    env.close()
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True, help="scenes_paired dir (has paired_manifest.jsonl)")
    ap.add_argument("--code", default="3.24", help="pdd_code_start filter (3.24 or 5.31; empty=all)")
    ap.add_argument("--only", default=None, help="substring filter on scene_id")
    ap.add_argument("--steps", type=int, default=220)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    gifs = out_root / "gifs"
    gifs.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in open(out_root / "paired_manifest.jsonl")]
    if args.code:
        rows = [r for r in rows if r["pdd_code_start"] == args.code]
    if args.only:
        rows = [r for r in rows if args.only in r["scene_id"]]

    print(f"Rendering {len(rows)} scene(s) -> {gifs}")
    for r in rows:
        try:
            print(f"{r['scene_id']}: {render_row(r, out_root, gifs, args.steps)}")
        except Exception as exc:
            print(f"{r['scene_id']}: FAIL {type(exc).__name__}: {exc}")
    print("DONE")


if __name__ == "__main__":
    main()
