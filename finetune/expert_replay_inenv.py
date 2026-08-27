"""Replay an expert-recorded scene inside our own env.

Reads:
  * pkl — ScenarioDescription / raw FrameInfo dump with ego+NPC tracks
  * sidecar .meta.json — sign info, original source row, expert actions

Reconstructs the exact TrafficSignEnv / TrafficSignSumoEnv from source_row +
places the sign at the recorded offsets, then runs one of:
  ego_mode = "recorded"   ← replay expert actions exactly
             "new_policy" ← drive a fresh policy (e.g. IDM) for comparison
             "live"       ← whatever env.config["agent_policy"] dictates

  npc_mode = "recorded"   ← NPCs teleported each step from pkl tracks
             "live"       ← NPCs keep their own managers (PG/SumoTraffic)

With ``--save-plant2-dir``, dumps PlanT2 boxes/measurements during recorded
replay (after set_position, before env.step).

Usage (single scene):
    python expert_replay_inenv.py \\
        --sidecar .../replay.json --pkl .../replay.pkl \\
        --ego-mode recorded --npc-mode recorded \\
        --scenes-root /path/to/scenes_balanced \\
        --save-plant2-dir /path/to/plant2_out --save-gif /tmp/one.gif

Usage (batch from experts_*.jsonl — PlanT2 collection):
    python expert_replay_inenv.py \\
        --experts /path/to/experts_scene_uid_top1.jsonl \\
        --scenes-root /path/to/scenes_balanced \\
        --save-plant2-dir /path/to/plant2_out \\
        --count 5 --save-gifs
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
PLANT2_DIR = SCRIPT_PATH.parent
PDD_BENCH_DIR = PLANT2_DIR.parent
SIGN_BENCH_DIR = PDD_BENCH_DIR / "bench"
# Historical aliases used by --scene-id replay path lookup.
BENCHMARK_DIR = SIGN_BENCH_DIR
SCRIPTS_DIR = PDD_BENCH_DIR / "scripts"
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "third_party" / "metadrive"
for _p in (METADRIVE_DIR,):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)


def _resolve_sign_class(cls_name: str):
    """Resolve a traffic-sign class by ``__name__`` (sidecar stores class names)."""
    import importlib
    import pkgutil
    import traffic_bench.signs
    for info in pkgutil.walk_packages(traffic_bench.signs.__path__, traffic_bench.signs.__name__ + "."):
        try:
            mod = importlib.import_module(info.name)
        except Exception:
            continue
        obj = getattr(mod, cls_name, None)
        if isinstance(obj, type):
            return obj
    return None


def _render_top_down(env, window: bool, screen_record: bool):
    env.render(
        mode="top_down",
        film_size=(2400, 2400), scaling=12.0, screen_size=(800, 800),
        semantic_map=True, semantic_broken_line=True,
        draw_target_vehicle_trajectory=True,
        target_agent_heading_up=True,
        screen_record=screen_record, window=window,
    )


def _xy(pos) -> np.ndarray:
    return np.asarray(pos[:2], dtype=np.float64)


def _resolve_ego_rec_id(scenario, npc_frames) -> Optional[str]:
    """Prefer FrameInfo agent map over proximity — spawn can be metres off."""
    try:
        frame0 = scenario["frame"][0][0]
        ato = getattr(frame0, "_agent_to_object", None) or {}
        for agent_name in (getattr(frame0, "agents", None) or ["default_agent"]):
            rid = ato.get(agent_name)
            if rid and npc_frames and rid in npc_frames[0]:
                return rid
        # Fallback: any agent mapping present in frame 0.
        for rid in ato.values():
            if npc_frames and rid in npc_frames[0]:
                return rid
    except Exception:
        pass
    return None


def _apply_state(live_obj, state) -> bool:
    try:
        live_obj.set_position(state["position"])
        live_obj.set_heading_theta(float(state["heading_theta"]))
        if "velocity" in state:
            live_obj.set_velocity(state["velocity"])
        return True
    except Exception:
        return False


def _match_recorded_to_live(
    frame_data: dict,
    live_objs: dict,
    ego_live,
    ego_rec_id: Optional[str],
    max_dist: float = 1e9,
) -> tuple[dict, int, int]:
    """Greedy proximity match of recorded IDs → live objects for one frame.

    Returns (obj_map, n_mapped, n_unmatched_recorded).
    Ego is always bound to ``ego_live`` when ``ego_rec_id`` is known.

    ``max_dist`` defaults to effectively unlimited so vehicles parked far away
    (see ``_park_unmatched_live``) can be reassigned when new recorded agents
    appear. Candidates are sorted so near matches win before far ones.
    """
    obj_map: dict = {}
    used_live = set()
    if ego_rec_id and ego_rec_id in frame_data and ego_live is not None:
        obj_map[ego_rec_id] = ego_live
        used_live.add(getattr(ego_live, "id", None) or id(ego_live))

    # Build live candidate list (skip ego).
    candidates = []
    ego_id = getattr(ego_live, "id", None)
    for lid, lobj in live_objs.items():
        if ego_id is not None and lid == ego_id:
            continue
        try:
            candidates.append((lid, lobj, _xy(lobj.position)))
        except Exception:
            continue

    # Greedy: nearest unused live object per recorded state (non-ego).
    # Sort recorded by distance to nearest candidate so tight matches win first.
    rec_items = []
    for rid, state in frame_data.items():
        if rid == ego_rec_id:
            continue
        try:
            rpos = _xy(state["position"])
        except Exception:
            continue
        rec_items.append((rid, state, rpos))

    def _nearest_dist(rpos):
        if not candidates:
            return 1e9
        return min(float(np.linalg.norm(rpos - c[2])) for c in candidates
                   if c[0] not in used_live) if any(c[0] not in used_live for c in candidates) else 1e9

    rec_items.sort(key=lambda it: _nearest_dist(it[2]))

    unmatched = 0
    for rid, state, rpos in rec_items:
        best_lid = None
        best_obj = None
        best_dist = max_dist
        for lid, lobj, lpos in candidates:
            if lid in used_live:
                continue
            d = float(np.linalg.norm(rpos - lpos))
            if d < best_dist:
                best_dist = d
                best_lid = lid
                best_obj = lobj
        if best_lid is not None:
            obj_map[rid] = best_obj
            used_live.add(best_lid)
        else:
            unmatched += 1
    return obj_map, len(obj_map), unmatched


def _park_unmatched_live(live_objs: dict, obj_map: dict, ego_live) -> None:
    """Move live objects not used for replay far away so they don't pollute sensors."""
    used = {id(o) for o in obj_map.values()}
    if ego_live is not None:
        used.add(id(ego_live))
    park = np.array([-10000.0, -10000.0, 1.0])
    for lid, lobj in live_objs.items():
        if id(lobj) in used:
            continue
        try:
            lobj.set_position(park)
        except Exception:
            pass


_TRAFFIC_MGR_KEYS = (
    "sumo_traffic_manager",
    "traffic_manager",
    "pg_traffic_manager",
    "pedestrian_manager",
    "crosswalk_pedestrian_manager",
    "crosswalk_yield_enforcer",
    "crosswalk_yield_enforcer_manager",
)


def _pause_live_traffic_managers(env) -> list:
    """No-op live traffic before/after_step so they don't fight recorded teleports."""
    patched = []
    engine = getattr(env, "engine", None)
    managers = getattr(engine, "managers", None) if engine is not None else None
    if not managers:
        return patched

    def _noop_before(*args, **kwargs):
        return {}

    def _noop_after(*args, **kwargs):
        return {}

    for key, mgr in list(managers.items()):
        key_l = str(key).lower()
        cls_l = type(mgr).__name__.lower()
        if not (key_l in _TRAFFIC_MGR_KEYS
                or any(t.replace("_", "") in cls_l.replace("_", "")
                       for t in _TRAFFIC_MGR_KEYS)):
            # Also match by class name fragments.
            if not any(x in cls_l for x in (
                    "sumotraffic", "trafficmanager", "pgtraffic",
                    "crosswalkpedestrian", "crosswalkyield")):
                continue
        for meth, noop in (("before_step", _noop_before),
                           ("after_step", _noop_after)):
            if not hasattr(mgr, meth):
                continue
            patched.append((mgr, meth, getattr(mgr, meth)))
            setattr(mgr, meth, noop)
    return patched


def _restore_patched(patched: list) -> None:
    for obj, meth, orig in patched:
        try:
            setattr(obj, meth, orig)
        except Exception:
            pass


def replay_in_our_env(
    pkl_path: Path, sidecar_path: Path,
    ego_mode: str = "recorded",
    npc_mode: str = "recorded",
    render_2d: bool = False,
    render_3d: bool = False,
    save_gif: Optional[Path] = None,
    max_steps: int = 500,
    scenes_root: Optional[Path] = None,
    save_plant2_dir: Optional[Path] = None,
) -> dict:
    sidecar = json.load(open(sidecar_path))
    scenario = pickle.load(open(pkl_path, "rb"))

    from expert_replay import _build_env    # re-use env-builder
    import env_flags as _env_flags
    from traffic_bench.eval.engine.traffic.ego_defaults import apply_ego_defaults

    row = sidecar["source_row"]
    backend = sidecar["backend"]

    # Mirror the RECORDING's relocate mode. NN policies are recorded with ego
    # left on the manifest road (RELOCATE_EGO_TO_SIGN_LANE=False); IDM-family
    # with True. The module default (True) would rebuild a DIFFERENT scene for
    # NN recordings — ego spawn/route mismatch, instant termination in replay.
    _idm_family = {"idm", "idm_rule", "ppo_rule"}
    _rec_policy = str(sidecar.get("policy") or "")
    _env_flags.RELOCATE_EGO_TO_SIGN_LANE = (
        (_rec_policy in _idm_family) if _rec_policy else True)
    print(f"[replay] relocate_ego_to_sign_lane="
          f"{_env_flags.RELOCATE_EGO_TO_SIGN_LANE} (policy={_rec_policy or '?'})")

    # Build env WITHOUT auto-policy, WITHOUT recording.
    env = _build_env(row, backend, max_steps=max_steps,
                      record_episode=False,
                      ego_policy_cls=None,
                      render=bool(render_3d),
                      scenes_root=scenes_root)

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

    traffic_patched: list = []
    try:
        env.reset(seed=env_seed)

        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        rn = env.current_map.road_network
        re_add_signs = (backend != "sumo")
        for sign_info in (sidecar.get("signs", []) if re_add_signs else []):
            cls_name = sign_info["sign_class"]
            sign_cls = _resolve_sign_class(cls_name)
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


        ego_rec_id = None
        replay_diag = {"steps": 0, "updated_sum": 0, "unmatched_sum": 0,
                       "ego_err_sum": 0.0, "cov_lt1m_sum": 0, "n_rec_sum": 0}

        if npc_mode == "recorded" and npc_frames:
            # Prefer FrameInfo agent map — proximity matching can bind ego to a
            # nearby NPC when live spawn is metres off the recorded ego pose.
            ego_rec_id = _resolve_ego_rec_id(scenario, npc_frames)
            if ego_rec_id is None:
                # Last resort: proximity (legacy behaviour).
                ego_pos = env.vehicle.position
                best_ego_dist = 50.0
                for rid, rstate in npc_frames[0].items():
                    rpos = rstate.get("position", [0, 0, 0])
                    d = float(np.linalg.norm(_xy(rpos) - _xy(ego_pos)))
                    if d < best_ego_dist:
                        best_ego_dist = d
                        ego_rec_id = rid
                print(f"[warn] ego_rec_id via proximity only "
                      f"(dist={best_ego_dist:.2f}): {ego_rec_id}")
            else:
                rpos = npc_frames[0][ego_rec_id]["position"]
                d0 = float(np.linalg.norm(_xy(rpos) - _xy(env.vehicle.position)))
                print(f"[info] ego_rec_id from agent map: {ego_rec_id} "
                      f"(live spawn offset {d0:.2f} m)")

            traffic_patched = _pause_live_traffic_managers(env)
            print(f"[info] paused {len(traffic_patched)} traffic manager hooks "
                  f"for recorded NPC replay")

            # Initial match for logging (re-matched every step below).
            obj_map0, n_map0, n_un0 = _match_recorded_to_live(
                npc_frames[0], dict(env.engine.get_objects()),
                env.vehicle, ego_rec_id)
            print(f"[info] per-frame rematch ready: frame0 matched {n_map0}/"
                  f"{len(npc_frames[0])} recorded "
                  f"(ego + {max(0, n_map0 - 1)} NPC/ped), "
                  f"unmatched_rec={n_un0}, live={len(env.engine.get_objects())}")

        expert_actions = sidecar.get("expert_actions", [])
        violations_replay = []
        prev_violating: set = set()
        arrived_any = False
        n_replay_frames = len(npc_frames) if npc_frames else max_steps
        # Prefer expert_actions length when recorded ego — matches original final_step.
        if ego_mode == "recorded" and expert_actions:
            n_replay_frames = min(n_replay_frames, len(expert_actions)) if npc_frames else len(expert_actions)

        plant2_collector = None
        plant2_path: Optional[str] = None
        if save_plant2_dir is not None:
            from plant2_frames import (
                Plant2FrameCollector, ensure_slurm_dummy, plant2_route_dir,
            )
            ensure_slurm_dummy(Path(save_plant2_dir))
            plant2_collector = Plant2FrameCollector(row)

        def _teleport_frame(frame_data: dict) -> tuple[int, int, Optional[float]]:
            """Rematch + set_position for one recorded frame. Returns
            (n_updated, n_unmatched_rec, ego_pos_err)."""
            live_objs = dict(env.engine.get_objects())
            obj_map, _, n_unmatched = _match_recorded_to_live(
                frame_data, live_objs, env.vehicle, ego_rec_id)
            n_updated = 0
            ego_err = None
            for rid, live_obj in obj_map.items():
                state = frame_data.get(rid)
                if state is None:
                    continue
                if _apply_state(live_obj, state):
                    n_updated += 1
                    if rid == ego_rec_id:
                        try:
                            ego_err = float(np.linalg.norm(
                                _xy(live_obj.position) - _xy(state["position"])))
                        except Exception:
                            pass
            _park_unmatched_live(live_objs, obj_map, env.vehicle)
            return n_updated, n_unmatched, ego_err

        for step in range(min(max_steps, n_replay_frames)):
            # 1. Rematch + teleport ALL objects from pkl BEFORE env.step
            #    (IDs churn across frames in SUMO dumps — fixed obj_map breaks).
            if npc_mode == "recorded" and step < len(npc_frames):
                n_upd, n_un, ego_err = _teleport_frame(npc_frames[step])
                replay_diag["steps"] += 1
                replay_diag["updated_sum"] += n_upd
                replay_diag["unmatched_sum"] += n_un
                replay_diag["n_rec_sum"] += len(npc_frames[step])
                if ego_err is not None:
                    replay_diag["ego_err_sum"] += ego_err
                # Coverage: how many recorded poses have a live obj within 1 m.
                try:
                    live_arr = []
                    for o in env.engine.get_objects().values():
                        try:
                            live_arr.append(_xy(o.position))
                        except Exception:
                            pass
                    if live_arr:
                        live_arr_np = np.stack(live_arr)
                        close = 0
                        for st in npc_frames[step].values():
                            rp = _xy(st["position"])
                            if float(np.linalg.norm(live_arr_np - rp, axis=1).min()) < 1.0:
                                close += 1
                        replay_diag["cov_lt1m_sum"] += close
                except Exception:
                    pass
                if step in (0, 1, 50, 100, 200) or (step % 200 == 0):
                    print(f"[replay] step={step} updated={n_upd}/"
                          f"{len(npc_frames[step])} unmatched_rec={n_un} "
                          f"ego_err={ego_err}")

            # PlanT2: capture AFTER teleporting ego+NPC to this frame's recorded
            # positions and BEFORE env.step. Matches PlanTDataset pre-step
            # convention with objects at the recorded frame (training fidelity).
            if plant2_collector is not None:
                plant2_collector.on_step(env, row)

            # 2. Step env (for rendering + sign checks). Action from recording.
            if ego_mode == "recorded" and step < len(expert_actions):
                action = expert_actions[step]
            else:
                action = [0.0, 0.0]
            _, _, term, trunc, info = env.step(action)
            # arrive_dest is a momentary flag; when the loop continues past the
            # arrival step (--save-plant2-dir path), the last info no longer
            # carries it — accumulate over the whole playback.
            arrived_any = arrived_any or bool(info.get("arrive_dest", False))

            # 3. Rematch + teleport AGAIN after step (physics may have moved them)
            if npc_mode == "recorded" and step < len(npc_frames):
                _teleport_frame(npc_frames[step])

            if sign_mgr is not None:
                # Edge-counted, mirroring the recorder's violations_timeline:
                # one entry per not-violating -> violating transition per class.
                current_violating = set()
                for s_obj, v in sign_mgr.check_all_violations(env.vehicle):
                    if v:
                        cls = type(s_obj).__name__
                        current_violating.add(cls)
                        if cls not in prev_violating:
                            violations_replay.append({
                                "step": step, "sign_class": cls,
                            })
                prev_violating = current_violating

            if save_gif:
                _render_top_down(env, window=False, screen_record=True)
            elif render_2d:
                _render_top_down(env, window=True, screen_record=False)
            elif render_3d:
                env.render()

            # Default inenv stops on env term/trunc. When dumping PlanT2 from
            # recorded tracks, continue through the pkl — live crash/arrive
            # flags between teleports are not ground truth for training length.
            if (term or trunc) and save_plant2_dir is None:
                break

        if save_gif:
            try:
                save_gif.parent.mkdir(parents=True, exist_ok=True)
                if hasattr(env, "top_down_renderer") and env.top_down_renderer is not None:
                    env.top_down_renderer.generate_gif(str(save_gif), duration=40)
            except Exception:
                pass

        arrived = bool(arrived_any or info.get("arrive_dest", False))
        crashed = bool(info.get("crash", False)) or bool(info.get("crash_vehicle", False))
        out_of_road = bool(info.get("out_of_road", False))

        if plant2_collector is not None and save_plant2_dir is not None:
            from plant2_frames import plant2_route_dir
            scene_uid = (
                sidecar.get("scene_uid")
                or row.get("scene_uid")
                or sidecar.get("scene_id")
                or "unknown"
            )
            variant = sidecar.get("variant") or "default"
            route_dir = plant2_route_dir(Path(save_plant2_dir), str(scene_uid), str(variant))
            # Prefer original sidecar success when present (recorded replay
            # fidelity); else derive from terminal info.
            metrics = sidecar.get("metrics") or {}
            if "success" in metrics:
                success = bool(metrics["success"])
            else:
                success = bool(arrived and not crashed and not out_of_road)
            # Carry the flags, not just their conjunction: "left the road at the
            # end" and "crashed into the obstacle" are both Failed here, and only
            # the second is unusable for imitation.
            n_frames = plant2_collector.flush(
                route_dir, success=success,
                reason={"arrived_dest": arrived,
                        "crashed": crashed,
                        "out_of_road": out_of_road})
            plant2_path = str(route_dir)
            print(f"[plant2] flushed {n_frames} frames → {route_dir}")

        original_violations = sidecar.get("violations_timeline", [])
        # Same events in the same order; the step may differ by one frame — the
        # replay teleports recorded states BEFORE env.step, so violation
        # detection can shift by a single step relative to the live run.
        match = (
            len(violations_replay) == len(original_violations)
            and all(a.get("sign_class") == b.get("sign_class")
                    and abs(int(a.get("step", -99)) - int(b.get("step", 99))) <= 1
                    for a, b in zip(violations_replay, original_violations))
        )
        result = {
            "scene_id": sidecar["scene_id"],
            "ego_mode": ego_mode,
            "npc_mode": npc_mode,
            "steps_run": step + 1,
            "arrived_dest": arrived,
            "crashed": crashed,
            "violations_replay": violations_replay,
            "violations_original": original_violations,
            "violations_match": match,
            "ego_rec_id": ego_rec_id,
        }
        if replay_diag["steps"] > 0:
            ns = replay_diag["steps"]
            result["replay_npc_diag"] = {
                "mean_updated": replay_diag["updated_sum"] / ns,
                "mean_unmatched_rec": replay_diag["unmatched_sum"] / ns,
                "mean_n_rec": replay_diag["n_rec_sum"] / ns,
                "mean_cov_lt1m": replay_diag["cov_lt1m_sum"] / ns,
                "mean_ego_err": replay_diag["ego_err_sum"] / ns,
            }
            d = result["replay_npc_diag"]
            print(f"[replay] NPC diag: updated={d['mean_updated']:.1f}/"
                  f"{d['mean_n_rec']:.1f} unmatched_rec={d['mean_unmatched_rec']:.1f} "
                  f"cov_lt1m={d['mean_cov_lt1m']:.1f} ego_err={d['mean_ego_err']:.4f}m")
        if plant2_path is not None:
            result["plant2_path"] = plant2_path
            result["plant2_frames"] = len(plant2_collector.step_records) if plant2_collector else 0
        return result
    finally:
        try:
            _restore_patched(traffic_patched)
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass


def _load_expert_rows(path: Path, count: int | None, start: int) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if count is not None and len(rows) >= count:
                break
    return rows


def _resolve_expert_paths(expert_row: dict) -> tuple[Path, Path, str, str, str]:
    """Return (pkl, sidecar, scene_uid, variant, backend) from an experts_*.jsonl row."""
    sidecar_path = expert_row.get("sidecar_path") or expert_row.get("winning_sidecar")
    pkl_path = expert_row.get("pkl_path")
    if not sidecar_path:
        raise ValueError("expert row missing sidecar_path")
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"sidecar not found: {sidecar_path}")

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    if not pkl_path:
        pkl_path = sidecar.get("pkl_path")
    if not pkl_path:
        raise ValueError(f"expert row / sidecar missing pkl_path: {sidecar_path}")
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"pkl not found: {pkl_path}")

    scene_uid = (
        expert_row.get("scene_uid")
        or sidecar.get("scene_uid")
        or sidecar.get("scene_id")
        or "?"
    )
    variant = (
        sidecar.get("variant")
        or expert_row.get("winner_variant")
        or expert_row.get("variant")
        or "default"
    )
    backend = (
        sidecar.get("backend")
        or expert_row.get("backend")
        or (sidecar.get("source_row") or {}).get("_backend")
        or "sumo"
    )
    return pkl_path, sidecar_path, str(scene_uid), str(variant), str(backend)


def run_experts_batch(
    experts_path: Path,
    scenes_root: Path,
    save_plant2_dir: Path,
    *,
    count: int | None = None,
    start: int = 0,
    max_steps: int = 1500,
    backends: str = "sumo",
    save_gifs: bool = False,
    gif_dir: Path | None = None,
    ego_mode: str = "recorded",
    npc_mode: str = "recorded",
) -> dict:
    """Batch-replay experts_*.jsonl rows and dump PlanT2 frames.

    Returns summary ``{ok, fail, skip, wall_s}``.
    """
    from plant2_frames import ensure_slurm_dummy, plant2_route_dir

    save_plant2_dir = Path(save_plant2_dir).resolve()
    save_plant2_dir.mkdir(parents=True, exist_ok=True)
    ensure_slurm_dummy(save_plant2_dir)

    gif_root: Path | None = None
    if save_gifs:
        gif_root = (Path(gif_dir).resolve() if gif_dir
                    else save_plant2_dir / "gifs")
        gif_root.mkdir(parents=True, exist_ok=True)

    allowed = {b.strip() for b in backends.split(",") if b.strip()}
    expert_rows = _load_expert_rows(Path(experts_path), count, start)
    if not expert_rows:
        raise SystemExit(f"No expert rows loaded from {experts_path}")

    logging.info(
        "experts=%s n=%d plant2_dir=%s scenes_root=%s ego=%s npc=%s",
        experts_path, len(expert_rows), save_plant2_dir, scenes_root,
        ego_mode, npc_mode,
    )

    ok_count = fail_count = skip_count = 0
    t0 = time.time()

    for i, expert_row in enumerate(expert_rows, start=1):
        scene_uid = expert_row.get("scene_uid") or "?"
        try:
            pkl_path, sidecar_path, scene_uid, variant, backend = _resolve_expert_paths(
                expert_row)
        except Exception as exc:
            logging.error("[%d/%d] %s resolve failed: %s",
                          i, len(expert_rows), scene_uid, exc)
            fail_count += 1
            continue

        if backend not in allowed:
            logging.info("[%d/%d] %s skip backend=%s",
                         i, len(expert_rows), scene_uid, backend)
            skip_count += 1
            continue

        route_dir = plant2_route_dir(save_plant2_dir, scene_uid, variant)
        if (route_dir / "results.json.gz").exists():
            logging.info("[%d/%d] %s already exists → %s",
                         i, len(expert_rows), scene_uid, route_dir)
            skip_count += 1
            continue

        gif_path = (
            gif_root / f"{scene_uid}_{variant}.gif"
            if gif_root is not None else None
        )

        expected_steps = expert_row.get("final_step")
        if expected_steps is None:
            expected_steps = (expert_row.get("metrics") or {}).get("final_step")
        logging.info("[%d/%d] %s variant=%s backend=%s expected_steps=%s pkl=%s",
                     i, len(expert_rows), scene_uid, variant, backend,
                     expected_steps, pkl_path)

        ep_t0 = time.time()
        try:
            result = replay_in_our_env(
                pkl_path, sidecar_path,
                ego_mode=ego_mode,
                npc_mode=npc_mode,
                save_gif=gif_path,
                max_steps=max_steps,
                scenes_root=Path(scenes_root),
                save_plant2_dir=save_plant2_dir,
            )
        except Exception as exc:
            import traceback
            logging.error("[%d/%d] %s ERROR %s: %s",
                          i, len(expert_rows), scene_uid, type(exc).__name__, exc)
            traceback.print_exc()
            fail_count += 1
            continue

        n_frames = None
        if route_dir.exists():
            n_frames = len(list((route_dir / "boxes").glob("*.json.gz")))
        logging.info(
            "[%d/%d] %s done steps_run=%s plant2_frames=%s arrived=%s "
            "crashed=%s violations_match=%s elapsed_s=%.1f plant2=%s gif=%s",
            i, len(expert_rows), scene_uid, result.get("steps_run"),
            n_frames, result.get("arrived_dest"), result.get("crashed"),
            result.get("violations_match"),
            time.time() - ep_t0, result.get("plant2_path"), gif_path,
        )
        if result.get("plant2_path") and (route_dir / "results.json.gz").exists():
            ok_count += 1
        else:
            fail_count += 1

    summary = {
        "ok": ok_count,
        "fail": fail_count,
        "skip": skip_count,
        "wall_s": round(time.time() - t0, 1),
        "plant2_dir": str(save_plant2_dir),
    }
    logging.info("=" * 60)
    logging.info("Done: ok=%d fail=%d skip=%d wall_s=%.1f → %s/data/",
                 ok_count, fail_count, skip_count, summary["wall_s"],
                 save_plant2_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Replay expert scene(s) in-env; optional PlanT2 dump",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Single-scene inputs
    parser.add_argument("--pkl", type=str)
    parser.add_argument("--sidecar", type=str)
    parser.add_argument("--scene-id", type=str,
                        help="If given with --preset and --code, resolve paths auto")
    parser.add_argument("--code", type=str)
    parser.add_argument("--preset", type=str, default="mini")
    # Batch from experts_*.jsonl
    parser.add_argument("--experts", type=str, default=None,
                        help="experts_scene_uid_top1.jsonl (or top2). "
                             "Batch mode: replay each row's pkl+sidecar.")
    parser.add_argument("--count", type=int, default=None,
                        help="Limit number of expert rows (batch mode)")
    parser.add_argument("--start", type=int, default=0,
                        help="Skip first N expert rows (batch mode)")
    parser.add_argument("--backends", type=str, default="sumo",
                        help="Comma-separated backends to keep (batch mode)")
    # Common
    parser.add_argument("--ego-mode", choices=["recorded", "new_policy", "live"],
                        default="recorded")
    parser.add_argument("--npc-mode", choices=["recorded", "live"], default="recorded")
    parser.add_argument("--render-2d", action="store_true")
    parser.add_argument("--render-3d", action="store_true")
    parser.add_argument("--save-gif", type=str, default=None,
                        help="Single-scene GIF path")
    parser.add_argument("--save-gifs", action="store_true",
                        help="Batch mode: write a GIF per episode")
    parser.add_argument("--gif-dir", type=str, default=None,
                        help="Batch GIF dir (default: <save-plant2-dir>/gifs)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Default 500 (single) / 1500 (batch/--save-plant2-dir)")
    parser.add_argument("--scenes-root", type=str, default=None,
                        help="Root dir sidecar net_path entries resolve against "
                             "(default: traffic-bench/scenes)")
    parser.add_argument("--save-plant2-dir", type=str, default=None,
                        help="Dump PlanT2 boxes/measurements under "
                             "<dir>/data/<scene_uid>_<variant>/")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # ── Batch mode: --experts ────────────────────────────────────────────
    if args.experts:
        if not args.scenes_root:
            print("ERROR: --experts requires --scenes-root", file=sys.stderr)
            sys.exit(1)
        if not args.save_plant2_dir:
            print("ERROR: --experts requires --save-plant2-dir "
                  "(PlanT2 collection is the batch purpose)", file=sys.stderr)
            sys.exit(1)
        max_steps = args.max_steps if args.max_steps is not None else 1500
        summary = run_experts_batch(
            Path(args.experts).resolve(),
            Path(args.scenes_root).resolve(),
            Path(args.save_plant2_dir).resolve(),
            count=args.count,
            start=args.start,
            max_steps=max_steps,
            backends=args.backends,
            save_gifs=args.save_gifs,
            gif_dir=Path(args.gif_dir) if args.gif_dir else None,
            ego_mode=args.ego_mode,
            npc_mode=args.npc_mode,
        )
        print(json.dumps(summary, indent=2))
        return

    # ── Single-scene mode ────────────────────────────────────────────────
    max_steps = args.max_steps if args.max_steps is not None else 500
    if args.save_plant2_dir and args.max_steps is None:
        max_steps = 1500

    if args.scene_id and args.code:
        code_slug = args.code.replace(".", "_")
        base = BENCHMARK_DIR / "benchmark_output" / args.preset / code_slug / "expert" / "replays"
        if not base.exists():
            base = SCRIPTS_DIR / "benchmark" / "benchmark_output" / args.preset / code_slug / "expert" / "replays"
        pkl = base / f"sd_{args.scene_id}.pkl"
        sidecar = base / f"{args.scene_id}.meta.json"
    else:
        if not args.pkl or not args.sidecar:
            print("Need --pkl + --sidecar, OR --scene-id + --code, "
                  "OR --experts (batch)", file=sys.stderr)
            sys.exit(1)
        pkl = Path(args.pkl)
        sidecar = Path(args.sidecar)

    if not pkl.exists() or not sidecar.exists():
        print(f"Missing files: {pkl} exists={pkl.exists()}, {sidecar} exists={sidecar.exists()}",
              file=sys.stderr)
        sys.exit(1)

    plant2_dir = None
    if args.save_plant2_dir:
        from plant2_frames import ensure_slurm_dummy
        plant2_dir = Path(args.save_plant2_dir).resolve()
        plant2_dir.mkdir(parents=True, exist_ok=True)
        ensure_slurm_dummy(plant2_dir)

    result = replay_in_our_env(
        pkl, sidecar,
        ego_mode=args.ego_mode, npc_mode=args.npc_mode,
        render_2d=args.render_2d, render_3d=args.render_3d,
        save_gif=Path(args.save_gif) if args.save_gif else None,
        max_steps=max_steps,
        scenes_root=Path(args.scenes_root) if args.scenes_root else None,
        save_plant2_dir=plant2_dir,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "violations_replay"
                       and k != "violations_original"}, indent=2, ensure_ascii=False))
    print(f"violations_match: {result['violations_match']}")


if __name__ == "__main__":
    main()
