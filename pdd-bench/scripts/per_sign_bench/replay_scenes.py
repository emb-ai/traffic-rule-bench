"""Replay a few scenes from an already-materialized manifest with rendering.

The main per_sign_benchmark materialization runs scenes with use_render=False
and no agent driving (just 50-step sanity validation). This script takes a
handful of scenes for one PDD code and re-runs them with render + IDM policy,
optionally saving a top-down GIF per scene.

Usage:
    python replay_scenes.py --code 2.1 --count 2 --save-gifs
    python replay_scenes.py --manifest benchmark_output/mini/2_1/pgmap_materialized.jsonl --count 3
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_DIR = SCRIPT_PATH.parent
SCRIPTS_DIR = BENCHMARK_DIR.parent
PDD_BENCH_DIR = SCRIPTS_DIR.parent
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"
for _p in (PDD_BENCH_DIR, METADRIVE_DIR, BENCHMARK_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)


# ---------------------------------------------------------------------------
# Run-benchmark compatible NPC setup with nuPlan-sampled params
# ---------------------------------------------------------------------------
# scripts/run_benchmark.py / scripts/sim.py use:
#   * SumoTrafficManager (real SUMO trajectories) for SUMO scenes
#   * ModifiedIDMPolicy for ego
#   * traffic_density = 0.1, tl_speed_factor = 20.0
#   * IDM defaults (NORMAL_SPEED=30 km/h, DISTANCE_WANTED=10, TIME_WANTED=1.5)
#
# For replay we keep that exact setup but apply a nuPlan profile to the NPC
# policies with SAFE BOUNDS: urban-range speeds only (≤ 40 km/h), minimum
# following distances/times that prevent rear-end crashes, lane-change freq
# from nuPlan. This gives per-scene NPC flavour from nuPlan while behaving
# like run_benchmark.py traffic.

# Applies to both IDMPolicy (PGMap NPCs) and SumoTrajectoryIDMPolicy (SUMO NPCs).
_NPC_SAFE_BOUNDS = {
    "NORMAL_SPEED_KMH_MIN": 22.0,
    "NORMAL_SPEED_KMH_MAX": 38.0,
    "MAX_SPEED_KMH": 50.0,
    "DEACC_ABS_MAX": 6.0,
    # DISTANCE_WANTED / TIME_WANTED are never overridden — stay at MetaDrive
    # defaults (10 m / 1.5 s). Exempting them keeps NPCs from tail-gating.
}

REPLAY_TRAFFIC_DENSITY = 0.1   # matches run_benchmark.py base_config


def _clip(v, lo, hi): return max(lo, min(hi, v))


def _apply_nuplan_npc_profile(profile: dict):
    """Apply a nuPlan-sampled profile to NPC IDM classes, clipped to safe urban bounds.

    Targets both PGMap (`IDMPolicy`) and SUMO (`SumoTrajectoryIDMPolicy`) NPCs.
    Aggressive nuPlan values (highway-speed NORMAL_SPEED, 3 m
    following distance) are clipped so surrounding traffic doesn't rear-end a
    slow replay ego, while still giving per-scene variability.
    """
    b = _NPC_SAFE_BOUNDS
    normal_kmh = _clip(float(profile.get("NORMAL_SPEED", 8)) * 3.6,
                        b["NORMAL_SPEED_KMH_MIN"], b["NORMAL_SPEED_KMH_MAX"])
    max_kmh = _clip(float(profile.get("MAX_SPEED", 12)) * 3.6,
                     normal_kmh, b["MAX_SPEED_KMH"])
    acc = _clip(float(profile.get("ACC_FACTOR", 1.0)), 0.5, 2.5)
    deacc = -_clip(abs(float(profile.get("DEACC_FACTOR", 3.0))),
                    2.0, b["DEACC_ABS_MAX"])
    lcf_per_km = float(profile.get("LANE_CHANGE_FREQ", 1.0))
    lcf_steps = int(max(50, 1250.0 / max(lcf_per_km, 1.0)))

    # DISTANCE_WANTED / TIME_WANTED intentionally NOT applied — keep MetaDrive
    # defaults (10 m / 1.5 s) so NPC IDM doesn't tail-gate ego.
    try:
        from metadrive.policy.idm_policy import IDMPolicy
        IDMPolicy.NORMAL_SPEED = normal_kmh
        IDMPolicy.MAX_SPEED = max_kmh
        IDMPolicy.ACC_FACTOR = acc
        IDMPolicy.DEACC_FACTOR = deacc
        IDMPolicy.LANE_CHANGE_FREQ = lcf_steps
    except ImportError:
        pass
    # SUMO NPCs inherit from TrajectoryIDMPolicy but override NORMAL_SPEED; keep
    # them consistent.
    try:
        from envs.sumo_idm_policy import SumoTrajectoryIDMPolicy
        SumoTrajectoryIDMPolicy.NORMAL_SPEED = normal_kmh
    except ImportError:
        pass


def _render_top_down(env, window: bool, screen_record: bool):
    """Call env.render in top-down mode with semantic colouring.

    `semantic_map=True` + `semantic_broken_line=True` draw lanes with OSM-like
    colours (driveable, sidewalk, bike lane, crosswalk, …) and broken lane
    dividers — much easier to read than the default greyscale.

    window=True opens a pygame window; screen_record=True captures frames for
    later GIF (mutually independent, both can be on at once).
    """
    env.render(
        mode="top_down",
        film_size=(2400, 2400), scaling=12.0, screen_size=(800, 800),
        semantic_map=True,
        semantic_broken_line=True,
        draw_target_vehicle_trajectory=True,
        target_agent_heading_up=True,
        screen_record=screen_record, window=window,
    )


def _replay_pgmap(spec: dict, render_3d: bool, render_2d: bool,
                   save_gif: str | None, max_steps: int,
                   spawn_zero: bool = False, manual_control: bool = False) -> dict:
    """Replay a PGMap scene (from pgmap_materialized.jsonl) with IDM ego."""
    from envs.traffic_sign_env import TrafficSignEnv
    from metadrive.component.pgblock.first_block import FirstPGBlock
    from metadrive.policy.idm_policy import IDMPolicy
    from factorized_space.benchmark_runner import (
        SIGN_CLASS_MAP, install_npc_vehicle_type_hook,
        DETOUR_KEYS, RESTRICTED_BEGIN_KEYS, LANE_CHANGE_KEYS, BEGIN_TO_END,
        _get_route_lanes, _pick_route_lane, _pick_detour_lane,
        _pick_rightmost_lane, _pick_lane_for_lane_change,
        _override_route_intent, _spawn_cyclists_on_lane,
    )
    from factorized_space.ego_defaults import apply_ego_defaults
    from factorized_space.space_definition import BIKE_RELATED_SIGNS
    from traffic_signs.detour_obstacle import spawn_detour_obstacle

    seed = int(spec.get("seed") or spec.get("deterministic_seed") or (100_000 + int(spec.get("flat_index", 0))))
    np.random.seed(seed)
    random.seed(seed)

    # NPC profile — sampled from nuPlan (matches mini/full benchmark flavour)
    # but clipped to safe urban bounds to match run_benchmark.py behaviour.
    from factorized_space.agent_profile_bank import sample_one_profile
    profile = spec if any(k.startswith("profile_") for k in spec) else {}
    if profile:
        profile = {k.replace("profile_", ""): v for k, v in profile.items()
                    if k.startswith("profile_")}
    else:
        profile = sample_one_profile(seed)
    _apply_nuplan_npc_profile(profile)
    install_npc_vehicle_type_hook()

    vehicle_config = {"show_lidar": False}
    spawn_vel = 0.0 if spawn_zero else float(spec.get("spawn_velocity_ms", 0) or 0)
    if spawn_vel > 0:
        vehicle_config["spawn_velocity"] = [spawn_vel, 0]
        vehicle_config["spawn_velocity_car_frame"] = True

    spawn_lane = spec["spawn_lane_index"]
    is_ramp_merge = spec.get("block_id") == "r" and spec.get("route_intent") == "merge"
    is_ramp_exit = spec.get("block_id") == "R" and spec.get("route_intent") == "exit"
    if is_ramp_merge:
        spawn_lane_tuple = ("2r1_0_", "2r1_1_", 0)
    elif is_ramp_exit:
        spawn_lane_tuple = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, spec["lane_num"] - 1)
    else:
        spawn_lane_tuple = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, spawn_lane)

    config = dict(
        start_seed=seed,
        use_render=render_3d,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.WARNING,
        traffic_density=REPLAY_TRAFFIC_DENSITY,
        horizon=max_steps,
        vehicle_config=vehicle_config,
        map_config={
            "type": "block_sequence",
            "config": spec["block_sequence"],
            "lane_num": spec["lane_num"],
            "lane_width": spec["lane_width"],
        },
        random_spawn_lane_index=False,
        agent_configs={
            "default_agent": dict(
                use_special_color=True,
                spawn_lane_index=spawn_lane_tuple,
            ),
        },
    )

    env = TrafficSignEnv(config)
    try:
        env.reset(seed=seed)

        # Drive ego with IDM (expert-ish). Override IDM class to ego defaults.
        ego_policy = env.engine.get_policy(env.vehicle.id)
        if ego_policy is not None:
            apply_ego_defaults(ego_policy)

        if not _override_route_intent(env, spec):
            return {"ok": False, "reason": "route_intent_override_failed"}

        route_lanes = _get_route_lanes(env)
        if not route_lanes:
            return {"ok": False, "reason": "no_route_lanes"}

        sign_key = spec["sign_type"]
        sign_class = SIGN_CLASS_MAP[sign_key]
        sign_mgr = env.engine.traffic_sign_manager
        rn = env.current_map.road_network
        veh = env.vehicle

        if sign_key in DETOUR_KEYS:
            lane = _pick_detour_lane(route_lanes, sign_class, min_length=30.0)
        elif sign_key in LANE_CHANGE_KEYS:
            vidx = getattr(veh.lane, "index", None)
            vnum = vidx[2] if (vidx and len(vidx) >= 3) else 0
            lane = _pick_lane_for_lane_change(route_lanes, vnum, road_network=rn)
        elif sign_key in RESTRICTED_BEGIN_KEYS:
            lane = _pick_rightmost_lane(route_lanes, min_length=30.0)
        else:
            lane = _pick_route_lane(route_lanes, min_length=15.0, road_network=rn)

        if lane is None:
            return {"ok": False, "reason": "no_suitable_lane"}

        if sign_key in RESTRICTED_BEGIN_KEYS:
            sign = sign_mgr.add_sign(sign_class, lane=lane, use_random_lane=False)
            if sign_key in BEGIN_TO_END:
                sign_mgr.add_sign(BEGIN_TO_END[sign_key], lane=sign.lane, use_random_lane=False)
        else:
            sign = sign_mgr.add_sign(sign_class, lane=lane, use_random_lane=False)

        if sign_key in DETOUR_KEYS and sign is not None:
            spawn_detour_obstacle(env.engine, sign.lane, sign)

        if sign_key in BIKE_RELATED_SIGNS and sign is not None:
            _spawn_cyclists_on_lane(env, sign.lane, seed, n=3)

        # Drive with IDM policy. Let the engine use the default NPC IDM for ego.
        policy = IDMPolicy(veh, seed)
        step = 0; arrived = False; crashed = False
        while step < max_steps:
            try:
                action = policy.act(veh.name)
            except Exception:
                action = [0.0, 0.0]
            _, _, terminated, truncated, info = env.step(action)
            if save_gif:
                _render_top_down(env, window=False, screen_record=True)
            elif render_2d:
                _render_top_down(env, window=True, screen_record=False)
            elif render_3d:
                env.render()
            step += 1
            if terminated or truncated:
                arrived = bool(info.get("arrive_dest", False))
                crashed = bool(info.get("crash", False))
                break

        if save_gif:
            try:
                if hasattr(env, "top_down_renderer") and env.top_down_renderer is not None:
                    env.top_down_renderer.generate_gif(save_gif, duration=40)
            except Exception:
                pass

        return {"ok": True, "steps": step, "arrived": arrived, "crashed": crashed}
    finally:
        try: env.close()
        except Exception: pass


def _replay_sumo(row: dict, render_3d: bool, render_2d: bool,
                  save_gif: str | None, max_steps: int,
                  spawn_zero: bool = False) -> dict:
    """Replay a SUMO catalog row with an IDM agent driving."""
    # For any visualisation (live window OR GIF) route through _replay_sumo_live
    # which controls traffic_density and NPC-profile bounds so surrounding
    # SUMO NPCs don't rear-end the slow replay ego.
    if render_2d or render_3d or save_gif:
        return _replay_sumo_live(row, render_3d=render_3d, render_2d=render_2d,
                                  save_gif=save_gif, max_steps=max_steps,
                                  spawn_zero=spawn_zero)
    # Pure metric replay (no visuals): keep materialize_sumo_scene for parity
    # with the full pipeline.
    from sumo_space.sumo_runner import materialize_sumo_scene
    result = materialize_sumo_scene(
        row,
        density_cap=1.0,
        drive_agent=True,
        agent_policy="idm",
        max_steps=max_steps,
        save_gif=save_gif,
    )
    return {"ok": bool(result.get("valid")), **{k: result.get(k)
        for k in ("arrived_dest", "crashed", "n_violations", "passed_sign",
                  "episode_length_steps", "failure_reason")}}


def _replay_sumo_live(row: dict, render_3d: bool, render_2d: bool,
                      save_gif: str | None, max_steps: int,
                      spawn_zero: bool = False) -> dict:
    """SUMO replay with an explicit render loop (needed for live windows).

    Mirrors scripts/run_benchmark.py: real SUMO trajectories via
    SumoTrafficManager, conservative IDM defaults (no aggressive nuPlan
    profile) so surrounding traffic doesn't rear-end the slow ego.
    """
    from envs.sumo_env import TrafficSignSumoEnv
    from envs.sumo_traffic_manager import SumoTrafficManager
    from metadrive.policy.idm_policy import IDMPolicy, ModifiedIDMPolicy
    from factorized_space.agent_profile_bank import sample_one_profile
    from factorized_space.benchmark_runner import install_npc_vehicle_type_hook
    from factorized_space.ego_defaults import apply_ego_defaults

    # Sample nuPlan profile and apply to SUMO NPC policies with safe bounds.
    _seed_for_profile = int(row.get("seed", row.get("deterministic_seed", 0)) or 1)
    _apply_nuplan_npc_profile(sample_one_profile(_seed_for_profile))

    # sumo_manifest.jsonl stores the seed as `deterministic_seed`; raw catalog
    # rows (real_manifest.jsonl) use plain `seed`. Support both.
    seed_val = row.get("seed", row.get("deterministic_seed"))
    if seed_val is None:
        # Last resort: derive from scene_id + v_idx
        import hashlib
        h = hashlib.sha256(f"{row.get('scene_id','?')}|{row.get('v_idx',0)}".encode()).digest()
        seed_val = int.from_bytes(h[:4], "big")
    seed = int(seed_val)
    # SUMO ego always spawns at 0 m/s (IDM accelerates from rest). We ignore
    # any `spawn_velocity_ms` in the row and any `--spawn-zero` flag becomes
    # redundant here (kept for API parity with PGMap).
    spawn_v = 0.0
    install_npc_vehicle_type_hook()

    SumoTrafficManager.EGO_SAFE_RADIUS = 15
    config = dict(
        use_render=render_3d,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.WARNING,
        map_name=row["net_path"],
        sign_type=row["sign_code"],
        # Match run_benchmark.py: keep surrounding SUMO traffic light and IDM-
        # compatible so it doesn't slam into the slow ego.
        traffic_density=REPLAY_TRAFFIC_DENSITY,
        tl_speed_factor=20.0,
        horizon=max_steps,
        num_scenarios=100000,
        vehicle_config={"show_lidar": False},
    )
    if row.get("road_id"):
        config["vehicle_config"]["spawn_lane_index"] = row["road_id"]
    if "spawn_lane_num" in row:
        config["spawn_lane_num"] = int(row["spawn_lane_num"])

    class _EnvWithTraffic(TrafficSignSumoEnv):
        @classmethod
        def default_config(cls):
            cfg = super().default_config()
            cfg["traffic_density"] = 0.0
            return cfg
        def setup_engine(self):
            super().setup_engine()
            self.engine.update_manager("traffic_manager", SumoTrafficManager())

    env = _EnvWithTraffic(config)
    try:
        sign_id = int(row.get("sign_id", 0))
        env.reset(seed=(sign_id + int(row.get("var_idx", 0))) % 100000)
        ego_policy = env.engine.get_policy(env.vehicle.id)
        if ego_policy is not None:
            apply_ego_defaults(ego_policy)
        if spawn_v > 0:
            env.vehicle.set_velocity([spawn_v, 0.0])

        policy = IDMPolicy(env.vehicle, seed)
        step = 0; arrived = False; crashed = False
        while step < max_steps:
            try: action = policy.act(env.vehicle.name)
            except Exception: action = [0.0, 0.0]
            _, _, terminated, truncated, info = env.step(action)
            if save_gif:
                _render_top_down(env, window=False, screen_record=True)
            elif render_2d:
                _render_top_down(env, window=True, screen_record=False)
            elif render_3d:
                env.render()
            step += 1
            if terminated or truncated:
                arrived = bool(info.get("arrive_dest", False))
                crashed = bool(info.get("crash", False))
                break
        if save_gif:
            try:
                if hasattr(env, "top_down_renderer") and env.top_down_renderer is not None:
                    env.top_down_renderer.generate_gif(save_gif, duration=40)
            except Exception:
                pass
        return {"ok": True, "steps": step, "arrived": arrived, "crashed": crashed}
    finally:
        try: env.close()
        except Exception: pass


def main():
    parser = argparse.ArgumentParser(description="Replay benchmark scenes with rendering")
    parser.add_argument("--manifest", type=str,
                        help="Full path to a *_materialized.jsonl or real_manifest.jsonl")
    parser.add_argument("--code", type=str,
                        help="PDD code (e.g. '2.1' or '2_1'); uses benchmark_output/<preset>/<code>/")
    parser.add_argument("--preset", type=str, default="mini",
                        help="Which preset output to use when --code is set")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "pgmap", "paired", "sumo"],
                        help="Which manifest to replay (auto: first available in priority)")
    parser.add_argument("--count", type=int, default=2, help="How many scenes to replay")
    parser.add_argument("--start", type=int, default=0, help="Start index in the manifest")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--render-3d", action="store_true",
                        help="Live 3D MetaDrive window (Panda3D). macOS/Linux with display only.")
    parser.add_argument("--render-2d", action="store_true",
                        help="Live top-down 2D window (pygame). Works anywhere with display.")
    parser.add_argument("--save-gifs", action="store_true",
                        help="Save top-down GIF per scene to ./replays/ (no live window needed).")
    parser.add_argument("--spawn-zero", action="store_true",
                        help="Spawn ego with 0 m/s (override any spawn_velocity_ms in manifest).")
    parser.add_argument("--manual-control", action="store_true",
                        help="Drive ego with keyboard (W/A/S/D). Requires --render-3d.")
    args = parser.parse_args()

    # Locate manifest
    manifest_path = None
    backend = args.backend
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        # infer backend from path
        name = manifest_path.name
        if backend == "auto":
            if "sumo" in str(manifest_path): backend = "sumo"
            elif "paired" in name: backend = "paired"
            else: backend = "pgmap"
    elif args.code:
        # Look for benchmark_output either under per_sign_bench/ or under the
        # legacy ../benchmark/ (whichever exists).
        code_slug = args.code.replace(".", "_")
        candidates_root = [
            BENCHMARK_DIR / "benchmark_output" / args.preset / code_slug,
            SCRIPTS_DIR / "benchmark" / "benchmark_output" / args.preset / code_slug,
        ]
        code_dir = next((c for c in candidates_root if c.exists()), None)
        if code_dir is None:
            print(f"ERROR: no benchmark_output/{args.preset}/{code_slug} found in:\n  "
                  + "\n  ".join(str(c) for c in candidates_root), file=sys.stderr)
            sys.exit(1)
        print(f"Using output: {code_dir}")
        candidates = [
            ("pgmap",  code_dir / "pgmap_materialized.jsonl"),
            ("paired", code_dir / "paired_materialized.jsonl"),
            ("sumo",   code_dir / "sumo" / "sumo_manifest.jsonl"),
        ]
        if backend == "auto":
            for b, p in candidates:
                if p.exists() and p.stat().st_size > 0:
                    backend, manifest_path = b, p; break
        else:
            manifest_path = dict(candidates).get(backend)
        if manifest_path is None or not manifest_path.exists():
            print(f"ERROR: no {backend} manifest under {code_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Need --manifest or --code", file=sys.stderr); sys.exit(1)

    print(f"Replaying: {manifest_path}")
    print(f"  backend={backend}  count={args.count}  start={args.start}  max_steps={args.max_steps}")

    rows = []
    with open(manifest_path) as f:
        for i, line in enumerate(f):
            if i < args.start: continue
            if len(rows) >= args.count: break
            try: rows.append(json.loads(line))
            except Exception: continue
    if not rows:
        print("No rows to replay.", file=sys.stderr); sys.exit(1)

    replays_dir = Path("replays"); replays_dir.mkdir(exist_ok=True)
    results = []
    for i, row in enumerate(rows):
        sid = row.get("scene_id") or row.get("flat_index") or f"row{i}"
        gif_path = str(replays_dir / f"{sid}.gif") if args.save_gifs else None
        print(f"\n[{i+1}/{len(rows)}] scene_id={sid}  sign={row.get('sign_type', row.get('sign_code'))}")
        if backend in ("pgmap", "paired"):
            r = _replay_pgmap(row, render_3d=args.render_3d, render_2d=args.render_2d,
                               save_gif=gif_path, max_steps=args.max_steps,
                               spawn_zero=args.spawn_zero)
        elif backend == "sumo":
            r = _replay_sumo(row, render_3d=args.render_3d, render_2d=args.render_2d,
                              save_gif=gif_path, max_steps=args.max_steps,
                              spawn_zero=args.spawn_zero)
        else:
            r = {"ok": False, "reason": f"unknown backend {backend}"}
        results.append({"scene_id": sid, **r})
        print(f"  → {r}")
        if gif_path and Path(gif_path).exists():
            print(f"  GIF: {gif_path}")

    print(f"\nDone: {sum(1 for r in results if r.get('ok'))}/{len(results)} ok")


if __name__ == "__main__":
    main()
