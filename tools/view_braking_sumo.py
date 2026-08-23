"""Render top-down GIFs of the 3.24 braking-spawn SUMO scenes.

Builds the env with the braking-spawn spec from a real_manifest.jsonl row (ego
starts above the limit, placed d_required before the sign), drives a rule-
following policy, and records a top-down GIF so you can watch the approach +
braking to the limit at the sign.

Usage:
    python tools/vis/view_braking_sumo.py \
        --manifest <.../3_24/real_manifest.jsonl> \
        --scenes-root <.../scenes> [--only <scene_id substr>] [--max N] [--skip-invalid]
GIFs → <manifest dir>/gifs/.
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
PDD = SCRIPT.parent.parent
METADRIVE_DIR = PDD.parent / "third_party" / "metadrive"
for p in (str(METADRIVE_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

from traffic_bench.envs.sumo import TrafficSignSumoEnv


def _make_policy(veh, seed):
    try:
        from traffic_bench.agents.idm_rule import ComprehensiveRuleExpertPolicy
        return ComprehensiveRuleExpertPolicy(veh, seed)
    except Exception:
        try:
            from metadrive.policy.idm_policy import IDMPolicy
            return IDMPolicy(veh, seed)
        except Exception:
            return None


def render_row(row, scenes_root: Path, gifs: Path, steps: int) -> str:
    cfg = dict(
        use_render=False, manual_control=False, use_mesh_terrain=False,
        log_level=logging.CRITICAL, map_name=str(scenes_root / row["net_path"]),
        sign_type=row["sign_code"], sign_spawn_distance=float(row.get("sign_s", 0.0)),
        num_scenarios=100000, spawn_lane_num=int(row.get("spawn_lane_num", 0)),
        ego_braking_spawn=bool(row.get("braking_spawn")),
        ego_spawn_v0_ms=float(row.get("spawn_velocity_ms", 0.0)),
        ego_brake_d_required=float(row.get("d_required_m", 0.0)),
        ego_v_target_kmh=float(row.get("v_target_kmh", 0.0)),
        ego_brake_decel=float(row.get("brake_decel_mps2", 2.5)),
        ego_brake_delay=float(row.get("brake_delay_s", 1.0)),
        ego_brake_margin=float(row.get("brake_margin_m", 5.0)),
        vehicle_config={"show_lidar": False, "spawn_lane_index": row["road_id"]},
    )
    if row.get("destination_lane_id"):
        cfg["vehicle_config"]["destination"] = row["destination_lane_id"]

    env = TrafficSignSumoEnv(cfg)
    env.reset(seed=int(row["sign_id"]) % 100000)
    info = getattr(env, "_braking_spawn_info", {}) or {}
    veh = env.agent if hasattr(env, "agent") else env.vehicle
    policy = _make_policy(veh, int(row["sign_id"]) % 100000)

    v_at_sign = None
    frames = 0
    for _ in range(steps):
        try:
            action = policy.act(veh.name) if policy is not None else [0.0, 0.0]
        except Exception:
            action = [0.0, 0.0]
        _, _, term, trunc, _ = env.step(action)
        # record speed when ego first enters the sign's drivable zone
        try:
            for s in env.engine.traffic_sign_manager.signs:
                if hasattr(s, "is_vehicle_in_zone") and s.is_vehicle_in_zone(veh) and v_at_sign is None:
                    v_at_sign = round(float(veh.speed_km_h), 1)
        except Exception:
            pass
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
    gif = gifs / f"{row['scene_id']}_v{row.get('v_idx', 0)}.gif"
    msg = "no renderer"
    if getattr(env, "top_down_renderer", None) is not None:
        env.top_down_renderer.generate_gif(str(gif), duration=40)
        msg = (f"{gif.name}  v0={info.get('ego_spawn_v0_ms')}m/s limit={row.get('v_target_kmh')}km/h "
               f"d_req={info.get('ego_d_required_m')} d_ach={info.get('ego_d_achieved_m')} "
               f"v@sign={v_at_sign}km/h "
               f"{'INVALID' if info.get('braking_invalid') else ('insuff' if info.get('insufficient_runway') else 'OK')}")
    env.close()
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help=".../3_24/real_manifest.jsonl")
    ap.add_argument("--scenes-root", required=True, help="root the net_path is relative to")
    ap.add_argument("--only", default=None)
    ap.add_argument("--max", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--skip-invalid", action="store_true",
                    help="skip scenes the env would mark braking_invalid (no runway)")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    gifs = manifest.parent / "gifs"
    gifs.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in open(manifest)]
    # one row per scene (v_idx 0) for a clean overview
    seen, uniq = set(), []
    for r in rows:
        if args.only and args.only not in r["scene_id"]:
            continue
        if r["scene_id"] in seen:
            continue
        seen.add(r["scene_id"]); uniq.append(r)
    uniq = uniq[: args.max]

    print(f"Rendering up to {len(uniq)} scene(s) -> {gifs}")
    for r in uniq:
        try:
            print(f"{r['scene_id']}: {render_row(r, Path(args.scenes_root), gifs, args.steps)}")
        except Exception as exc:
            print(f"{r['scene_id']}: FAIL {type(exc).__name__}: {exc}")
    print("DONE")


if __name__ == "__main__":
    main()
