"""Replay an expert-recorded scene inside our own env.

Reads:
  * pkl — ScenarioDescription with ego/NPC/pedestrian/cyclist tracks (MetaDrive)
  * sidecar .meta.json — sign info, original source row, expert actions

Reconstructs the exact TrafficSignEnv / TrafficSignSumoEnv from source_row +
places the sign at the recorded offsets, then runs one of:
  ego_mode = "recorded"   ← replay expert actions exactly
             "new_policy" ← drive a fresh policy (e.g. IDM) for comparison
             "live"       ← whatever env.config["agent_policy"] dictates

  npc_mode = "recorded"   ← NPCs frozen to pkl tracks via ScenarioReplayPolicy
             "live"       ← NPCs keep their own managers (PG/SumoTraffic)

Usage:
    python expert_replay_inenv.py \\
        --sidecar benchmark_output/mini/2_5/expert/replays/sumo_2.5_80410.meta.json \\
        --pkl     benchmark_output/mini/2_5/expert/replays/sd_sumo_2.5_80410.pkl \\
        --ego-mode recorded --npc-mode recorded --render-2d
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Optional

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


def _render_top_down(env, window: bool, screen_record: bool):
    env.render(
        mode="top_down",
        film_size=(2400, 2400), scaling=12.0, screen_size=(800, 800),
        semantic_map=True, semantic_broken_line=True,
        draw_target_vehicle_trajectory=True,
        target_agent_heading_up=True,
        screen_record=screen_record, window=window,
    )


def replay_in_our_env(
    pkl_path: Path, sidecar_path: Path,
    ego_mode: str = "recorded",
    npc_mode: str = "recorded",
    render_2d: bool = False,
    render_3d: bool = False,
    save_gif: Optional[Path] = None,
    max_steps: int = 500,
) -> dict:
    sidecar = json.load(open(sidecar_path))
    scenario = pickle.load(open(pkl_path, "rb"))

    from expert_replay import _build_env    # re-use env-builder
    from factorized_space.benchmark_runner import SIGN_CLASS_MAP
    from factorized_space.ego_defaults import apply_ego_defaults

    row = sidecar["source_row"]
    backend = sidecar["backend"]

    # Build env WITHOUT auto-policy, WITHOUT recording.
    env = _build_env(row, backend, max_steps=max_steps,
                      record_episode=False,
                      ego_policy_cls=None,
                      render=bool(render_3d))

    # Extract per-step NPC states from pkl FrameInfo for manual replay.
    # pkl format: {frame: [[FrameInfo, ...], ...], ...}
    # Each FrameInfo.step_info[obj_id] = {position, heading_theta, velocity, ...}
    npc_frames = []
    if npc_mode == "recorded" and "frame" in scenario:
        for frame_group in scenario["frame"]:
            if frame_group:
                npc_frames.append(frame_group[0].step_info)
            else:
                npc_frames.append({})
        print(f"[info] loaded {len(npc_frames)} replay frames, "
              f"objects in first: {len(npc_frames[0]) if npc_frames else 0}")

    seed = int(sidecar["env_config_summary"].get("seed") or 0)
    if backend == "sumo":
        env_seed = (int(row.get("sign_id", 0)) + int(row.get("var_idx", 0))) % 100000
    else:
        env_seed = seed
    np.random.seed(seed)
    random.seed(seed)

    try:
        env.reset(seed=env_seed)

        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        rn = env.current_map.road_network
        re_add_signs = (backend != "sumo")
        for sign_info in (sidecar.get("signs", []) if re_add_signs else []):
            cls_name = sign_info["sign_class"]
            sign_cls = None
            for k, v in SIGN_CLASS_MAP.items():
                if v.__name__ == cls_name:
                    sign_cls = v; break
            if sign_cls is None or sign_mgr is None:
                continue
            lane_idx = sign_info.get("lane_index")
            lane = None
            if lane_idx:
                try:
                    lane = rn.get_lane(tuple(lane_idx))
                except Exception:
                    lane = None
            if lane is None and env.vehicle is not None:
                lane = env.vehicle.lane
            if lane is None:
                continue
            try:
                sign_mgr.add_sign(sign_cls, lane=lane,
                                   longitudinal_offset=sign_info.get("longitudinal_offset", 0.0),
                                   lateral_offset=sign_info.get("lateral_offset", 0.0),
                                   use_random_lane=False)
            except Exception as exc:
                print(f"[warn] failed to re-add sign {cls_name}: {exc}")

        # Ego policy override if needed
        try:
            ego_policy = env.engine.get_policy(env.vehicle.id)
            if ego_policy is not None:
                apply_ego_defaults(ego_policy)
        except Exception:
            pass


        all_live_objs = dict(env.engine.get_objects())
        ego_id = env.vehicle.id
        ego_rec_id = None
        obj_map = {}  # recorded_id → live_obj

        if npc_mode == "recorded" and npc_frames:
            rec_frame0 = npc_frames[0]
            ego_pos = env.vehicle.position

            # 1. Find ego in recording
            best_ego_dist = 10.0
            for rid, rstate in rec_frame0.items():
                rpos = rstate.get("position", [0, 0, 0])
                d = ((rpos[0] - ego_pos[0]) ** 2 + (rpos[1] - ego_pos[1]) ** 2) ** 0.5
                if d < best_ego_dist:
                    best_ego_dist = d
                    ego_rec_id = rid
            if ego_rec_id:
                obj_map[ego_rec_id] = env.vehicle

            # 2. Match ALL other recorded objects to live objects by proximity
            used_live = {ego_id}
            for rid, rstate in rec_frame0.items():
                if rid == ego_rec_id:
                    continue
                rpos = rstate.get("position", [0, 0, 0])
                best_live_id = None
                best_dist = 8.0
                for lid, lobj in all_live_objs.items():
                    if lid in used_live:
                        continue
                    try:
                        lpos = lobj.position
                        d = ((rpos[0] - lpos[0]) ** 2 + (rpos[1] - lpos[1]) ** 2) ** 0.5
                    except Exception:
                        continue
                    if d < best_dist:
                        best_dist = d
                        best_live_id = lid
                if best_live_id is not None:
                    obj_map[rid] = all_live_objs[best_live_id]
                    used_live.add(best_live_id)

            print(f"[info] 1:1 replay: matched {len(obj_map)} objects "
                  f"(ego + {len(obj_map)-1} NPC/ped) of {len(rec_frame0)} recorded, "
                  f"{len(all_live_objs)} live")

        expert_actions = sidecar.get("expert_actions", [])
        violations_replay = []
        n_replay_frames = len(npc_frames) if npc_frames else max_steps

        for step in range(min(max_steps, n_replay_frames)):
            # 1. Set ALL object states from pkl BEFORE env.step
            if npc_mode == "recorded" and step < len(npc_frames):
                frame_data = npc_frames[step]
                for rid, live_obj in obj_map.items():
                    state = frame_data.get(rid)
                    if state is None:
                        continue
                    try:
                        live_obj.set_position(state["position"])
                        live_obj.set_heading_theta(float(state["heading_theta"]))
                        if "velocity" in state:
                            live_obj.set_velocity(state["velocity"])
                    except Exception:
                        pass

            # 2. Step env (for rendering + sign checks). Action from recording.
            if ego_mode == "recorded" and step < len(expert_actions):
                action = expert_actions[step]
            else:
                action = [0.0, 0.0]
            _, _, term, trunc, info = env.step(action)

            # 3. Override positions AGAIN after step (physics may have moved them)
            if npc_mode == "recorded" and step < len(npc_frames):
                frame_data = npc_frames[step]
                for rid, live_obj in obj_map.items():
                    state = frame_data.get(rid)
                    if state is None:
                        continue
                    try:
                        live_obj.set_position(state["position"])
                        live_obj.set_heading_theta(float(state["heading_theta"]))
                        if "velocity" in state:
                            live_obj.set_velocity(state["velocity"])
                    except Exception:
                        pass

            if sign_mgr is not None:
                for s_obj, v in sign_mgr.check_all_violations(env.vehicle):
                    if v:
                        violations_replay.append({
                            "step": step, "sign_class": type(s_obj).__name__,
                        })

            if save_gif:
                _render_top_down(env, window=False, screen_record=True)
            elif render_2d:
                _render_top_down(env, window=True, screen_record=False)
            elif render_3d:
                env.render()

            if term or trunc:
                break

        if save_gif:
            try:
                save_gif.parent.mkdir(parents=True, exist_ok=True)
                if hasattr(env, "top_down_renderer") and env.top_down_renderer is not None:
                    env.top_down_renderer.generate_gif(str(save_gif), duration=40)
            except Exception:
                pass

        original_violations = sidecar.get("violations_timeline", [])
        match = (
            len(violations_replay) == len(original_violations)
            and all(a.get("sign_class") == b.get("sign_class")
                    and a.get("step") == b.get("step")
                    for a, b in zip(violations_replay, original_violations))
        )
        result = {
            "scene_id": sidecar["scene_id"],
            "ego_mode": ego_mode,
            "npc_mode": npc_mode,
            "steps_run": step + 1,
            "arrived_dest": bool(info.get("arrive_dest", False)),
            "crashed": bool(info.get("crash", False)),
            "violations_replay": violations_replay,
            "violations_original": original_violations,
            "violations_match": match,
        }
        return result
    finally:
        try:
            env.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Replay expert scene in our env")
    parser.add_argument("--pkl", type=str)
    parser.add_argument("--sidecar", type=str)
    parser.add_argument("--scene-id", type=str,
                        help="If given with --preset and --code, resolve paths auto")
    parser.add_argument("--code", type=str)
    parser.add_argument("--preset", type=str, default="mini")
    parser.add_argument("--ego-mode", choices=["recorded", "new_policy", "live"],
                        default="recorded")
    parser.add_argument("--npc-mode", choices=["recorded", "live"], default="recorded")
    parser.add_argument("--render-2d", action="store_true")
    parser.add_argument("--render-3d", action="store_true")
    parser.add_argument("--save-gif", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    if args.scene_id and args.code:
        code_slug = args.code.replace(".", "_")
        base = BENCHMARK_DIR / "benchmark_output" / args.preset / code_slug / "expert" / "replays"
        if not base.exists():
            base = SCRIPTS_DIR / "benchmark" / "benchmark_output" / args.preset / code_slug / "expert" / "replays"
        pkl = base / f"sd_{args.scene_id}.pkl"
        sidecar = base / f"{args.scene_id}.meta.json"
    else:
        if not args.pkl or not args.sidecar:
            print("Need --pkl + --sidecar OR --scene-id + --code", file=sys.stderr)
            sys.exit(1)
        pkl = Path(args.pkl)
        sidecar = Path(args.sidecar)

    if not pkl.exists() or not sidecar.exists():
        print(f"Missing files: {pkl} exists={pkl.exists()}, {sidecar} exists={sidecar.exists()}",
              file=sys.stderr)
        sys.exit(1)

    result = replay_in_our_env(
        pkl, sidecar,
        ego_mode=args.ego_mode, npc_mode=args.npc_mode,
        render_2d=args.render_2d, render_3d=args.render_3d,
        save_gif=Path(args.save_gif) if args.save_gif else None,
        max_steps=args.max_steps,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "violations_replay"
                       and k != "violations_original"}, indent=2, ensure_ascii=False))
    print(f"violations_match: {result['violations_match']}")


if __name__ == "__main__":
    main()
