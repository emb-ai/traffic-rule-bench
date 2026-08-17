"""PlanT2 dump from recorded expert trajectories (simplified).

Opens pkl + sidecar, replays NPC poses in env, collects frames via
``bench.plant2_frames.Plant2FrameCollector``.

Usage:
    python expert_replay_for_plant2_v2.py \\
        --experts .../experts_scene_uid_top1.jsonl \\
        --scenes-root .../stop_sign/scenes/2_5 \\
        --save-plant2-dir /tmp/plant2_out --count 10
"""
from __future__ import annotations

import argparse
import json
import pickle
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

_IDM_FAMILY = {"idm", "comprehensive_rule_expert", "rule_compliant"}
_TRAFFIC_MGR_KEYS = (
    "sumo_traffic_manager", "traffic_manager", "pg_traffic_manager",
    "pedestrian_manager", "crosswalk_pedestrian_manager",
    "crosswalk_yield_enforcer", "crosswalk_yield_enforcer_manager",
)


def _sign_lib_roots(scenes_root: Path | None) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    if scenes_root is not None:
        p = Path(scenes_root).resolve()
        for _ in range(8):
            if (p / "lib" / "auxiliary_agent.py").is_file():
                s = str(p.resolve())
                if s not in seen:
                    seen.add(s)
                    roots.append(Path(s))
                break
            if p.parent == p:
                break
            p = p.parent
    for lib_aux in BENCHMARK_DIR.glob("*/lib/auxiliary_agent.py"):
        s = str(lib_aux.parent.parent.resolve())
        if s not in seen:
            seen.add(s)
            roots.append(Path(s))
    return roots


def load_pkl(pkl_path: Path, scenes_root: Path | None):
    inserted: list[str] = []
    for root in _sign_lib_roots(scenes_root):
        s = str(root)
        if s not in sys.path:
            sys.path.insert(0, s)
            inserted.append(s)
    with open(pkl_path, "rb") as f:
        scenario = pickle.load(f)
    for s in inserted:
        sys.path.remove(s)
    for mod in list(sys.modules):
        if mod == "lib" or mod.startswith("lib."):
            del sys.modules[mod]
    return scenario


def xy(pos) -> np.ndarray:
    return np.asarray(pos[:2], dtype=np.float64)


def resolve_ego_rec_id(scenario, npc_frames) -> str | None:
    frame0 = scenario["frame"][0][0]
    ato = getattr(frame0, "_agent_to_object", None) or {}
    for agent_name in getattr(frame0, "agents", None) or ["default_agent"]:
        rid = ato.get(agent_name)
        if rid and npc_frames and rid in npc_frames[0]:
            return rid
    for rid in ato.values():
        if npc_frames and rid in npc_frames[0]:
            return rid
    return None


def apply_state(live_obj, state) -> None:
    live_obj.set_position(state["position"])
    live_obj.set_heading_theta(float(state["heading_theta"]))
    if "velocity" in state and getattr(live_obj, "_body", None) is not None:
        live_obj.set_velocity(state["velocity"])


def match_recorded_to_live(frame_data, live_objs, ego_live, ego_rec_id, engine=None):
    from metadrive.component.vehicle.base_vehicle import BaseVehicle

    obj_map: dict = {}
    used_live: set = set()
    if ego_rec_id and ego_rec_id in frame_data and ego_live is not None:
        obj_map[ego_rec_id] = ego_live
        used_live.add(getattr(ego_live, "id", None) or id(ego_live))

    candidates: list = []
    seen_vehicle_ids: set[int] = set()
    ego_id = getattr(ego_live, "id", None)

    def _add_vehicle(lid, lobj) -> None:
        if lobj is ego_live:
            return
        if ego_id is not None and lid == ego_id:
            return
        if not isinstance(lobj, BaseVehicle):
            return
        vid = id(lobj)
        if vid in seen_vehicle_ids:
            return
        seen_vehicle_ids.add(vid)
        candidates.append((lid, lobj, xy(lobj.position)))

    for lid, lobj in live_objs.items():
        _add_vehicle(lid, lobj)
    if engine is not None:
        aux_mgr = getattr(engine, "auxiliary_agent_manager", None)
        if aux_mgr is not None:
            for lobj in getattr(aux_mgr, "auxiliary_vehicles", []) or []:
                _add_vehicle(getattr(lobj, "id", id(lobj)), lobj)

    rec_items = []
    for rid, state in frame_data.items():
        if rid == ego_rec_id:
            continue
        if "Vehicle" not in str(state.get("type", "")):
            continue
        rec_items.append((rid, state, xy(state["position"])))

    def nearest_dist(rpos):
        best = 1e9
        for lid, _, lpos in candidates:
            if lid not in used_live:
                best = min(best, float(np.linalg.norm(rpos - lpos)))
        return best

    rec_items.sort(key=lambda it: nearest_dist(it[2]))

    for rid, state, rpos in rec_items:
        best_lid = None
        best_obj = None
        best_dist = 1e9
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
    return obj_map


def park_unmatched_live(live_objs, obj_map, ego_live, engine=None) -> None:
    from metadrive.component.vehicle.base_vehicle import BaseVehicle

    used = {id(o) for o in obj_map.values()}
    if ego_live is not None:
        used.add(id(ego_live))
    aux_ids: set[int] = set()
    if engine is not None:
        aux_mgr = getattr(engine, "auxiliary_agent_manager", None)
        if aux_mgr is not None:
            for v in getattr(aux_mgr, "auxiliary_vehicles", []) or []:
                aux_ids.add(id(v))
    park = np.array([-10000.0, -10000.0, 1.0])
    for lobj in live_objs.values():
        if id(lobj) in used or id(lobj) in aux_ids:
            continue
        if not isinstance(lobj, BaseVehicle):
            continue
        lobj.set_position(park)


def pause_traffic_managers(env) -> list:
    patched = []
    engine = env.engine
    managers = engine.managers
    if not managers:
        return patched

    def noop(*args, **kwargs):
        return {}

    for key, mgr in list(managers.items()):
        key_l = str(key).lower()
        cls_l = type(mgr).__name__.lower()
        if not (key_l in _TRAFFIC_MGR_KEYS or any(
            t.replace("_", "") in cls_l.replace("_", "") for t in _TRAFFIC_MGR_KEYS
        )):
            if not any(x in cls_l for x in (
                "sumotraffic", "trafficmanager", "pgtraffic",
                "crosswalkpedestrian", "crosswalkyield",
            )):
                continue
        for meth in ("before_step", "after_step"):
            if hasattr(mgr, meth):
                patched.append((mgr, meth, getattr(mgr, meth)))
                setattr(mgr, meth, noop)
    return patched


def restore_patched(patched: list) -> None:
    for obj, meth, orig in patched:
        setattr(obj, meth, orig)


def npc_frames_from_scenario(scenario) -> list[dict]:
    frames = []
    for frame_group in scenario["frame"]:
        frames.append(frame_group[0].step_info if frame_group else {})
    return frames


def spawn_auxiliary_from_row(env, row: dict, scenes_root: Path) -> None:
    """Spawn convoy NPCs via ``AuxiliaryAgentsManager`` (same as run_benchmark)."""
    if not row.get("auxiliary_agent"):
        return

    inserted: list[str] = []
    for root in _sign_lib_roots(scenes_root):
        s = str(root)
        if s not in sys.path:
            sys.path.insert(0, s)
            inserted.append(s)
    try:
        from lib.auxiliary_agent import (
            DEFAULT_CONVOY_GAP_M,
            DEFAULT_CONVOY_SIZE,
            DEFAULT_DISTANCE_FROM_INTERSECTION,
            DEFAULT_SPAWN_VELOCITY_MS,
            add_auxiliary_agents,
            resolve_aux_spawn_plan,
        )
        from lib.lane_keys import make_lane_key

        aux_distance_from_intersection = float(
            row.get("aux_distance_from_intersection", DEFAULT_DISTANCE_FROM_INTERSECTION)
        )
        aux_spawn_velocity_ms = float(
            row.get("aux_spawn_velocity_ms", DEFAULT_SPAWN_VELOCITY_MS)
        )
        aux_convoy_size = int(row.get("aux_convoy_size", DEFAULT_CONVOY_SIZE))
        aux_convoy_gap_m = float(row.get("aux_convoy_gap_m", DEFAULT_CONVOY_GAP_M))
        aux_lanes_occupied = int(row.get("aux_lanes_occupied", 1))
        aux_policy = str(row.get("aux_policy", "idm"))
        aux_release_when_ego_within_m = float(row.get("aux_release_when_ego_within_m", 20.0))

        ego_lane_index = (
            make_lane_key(str(row.get("road_id")), int(row.get("spawn_lane_num", 0) or 0))
            if row.get("road_id")
            else str(getattr(env.vehicle.lane, "index", ""))
        )

        aux_spawn_lanes, aux_destination_lanes, alternate_spawn_dest_map = (
            resolve_aux_spawn_plan(
                row,
                ego_lane_index=str(ego_lane_index),
                incoming_lanes=None,
                aux_lanes_occupied=aux_lanes_occupied,
                aux_distance_from_intersection=aux_distance_from_intersection,
            )
        )
        aux_destination_lanes = [dest or None for dest in aux_destination_lanes]

        if not aux_spawn_lanes:
            print("[plant2_dump] aux: no spawn lanes resolved", flush=True)
            return

        mgr = add_auxiliary_agents(
            env,
            spawn_lane_indices=aux_spawn_lanes,
            outgoing_lanes=None,
            distance_from_intersection=aux_distance_from_intersection,
            policy=aux_policy,
            spawn_velocity_ms=aux_spawn_velocity_ms,
            destination_lanes=aux_destination_lanes,
            ego_vehicle=env.vehicle,
            ego_spawn_lane_index=ego_lane_index,
            ego_release_distance_before_end=aux_release_when_ego_within_m,
            convoy_size=aux_convoy_size,
            convoy_gap_m=aux_convoy_gap_m,
            alternate_spawn_dest_map=alternate_spawn_dest_map,
        )
        if mgr is not None:
            print(
                f"[plant2_dump] aux spawned: lanes={len(aux_spawn_lanes)} "
                f"vehicles={len(mgr.auxiliary_vehicles)}",
                flush=True,
            )
    finally:
        for s in inserted:
            sys.path.remove(s)
        for mod in list(sys.modules):
            if mod == "lib" or mod.startswith("lib."):
                del sys.modules[mod]


def readd_signs(env, sidecar, backend) -> None:
    if backend == "sumo":
        return
    from factorized_space.benchmark_runner import SIGN_CLASS_MAP

    sign_mgr = env.engine.traffic_sign_manager
    rn = env.current_map.road_network
    for sign_info in sidecar.get("signs", []):
        cls_name = sign_info["sign_class"]
        sign_cls = next(
            (v for v in SIGN_CLASS_MAP.values() if v.__name__ == cls_name), None)
        if sign_cls is None:
            continue
        lane_idx = sign_info.get("lane_index")
        lane = rn.get_lane(tuple(lane_idx)) if lane_idx else env.vehicle.lane
        sign_mgr.add_sign(
            sign_cls, lane=lane,
            longitudinal_offset=sign_info.get("longitudinal_offset", 0.0),
            lateral_offset=sign_info.get("lateral_offset", 0.0),
            use_random_lane=False,
        )


def dump_plant2(
    pkl_path: Path,
    sidecar_path: Path,
    *,
    scenes_root: Path,
    save_plant2_dir: Path,
    max_steps: int = 1500,
) -> dict:
    from bench.plant2_frames import Plant2FrameCollector, ensure_slurm_dummy, plant2_route_dir
    from bench import env_builders
    from expert_replay import _build_env

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    scenario = load_pkl(pkl_path, scenes_root)
    row = sidecar["source_row"]
    backend = sidecar["backend"]

    rec_policy = str(sidecar.get("policy") or "")
    env_builders.RELOCATE_EGO_TO_SIGN_LANE = (
        (rec_policy in _IDM_FAMILY) if rec_policy else True)

    env = _build_env(
        row, backend, max_steps=max_steps,
        record_episode=False, ego_policy_cls=None, render=False,
        scenes_root=scenes_root,
    )

    npc_frames = npc_frames_from_scenario(scenario)
    seed = int(sidecar["env_config_summary"].get("seed") or 0)
    env_seed = (
        (int(row.get("sign_id", 0)) + int(row.get("var_idx", 0))) % 100000
        if backend == "sumo" else seed)
    np.random.seed(seed)
    random.seed(seed)

    ensure_slurm_dummy(save_plant2_dir)
    collector = Plant2FrameCollector(row)
    expert_actions = sidecar.get("expert_actions", [])
    n_frames = len(npc_frames)
    if expert_actions:
        n_frames = min(n_frames, len(expert_actions)) if npc_frames else len(expert_actions)
    n_frames = min(n_frames, max_steps)

    patched = []
    env.reset(seed=env_seed)
    readd_signs(env, sidecar, backend)
    spawn_auxiliary_from_row(env, row, scenes_root)

    ego_rec_id = resolve_ego_rec_id(scenario, npc_frames)
    if ego_rec_id is None:
        ego_pos = env.vehicle.position
        best_dist = 50.0
        for rid, rstate in npc_frames[0].items():
            d = float(np.linalg.norm(xy(rstate["position"]) - xy(ego_pos)))
            if d < best_dist:
                best_dist = d
                ego_rec_id = rid

    patched = pause_traffic_managers(env)

    def teleport(frame_data: dict) -> None:
        live_objs = dict(env.engine.get_objects())
        obj_map = match_recorded_to_live(
            frame_data, live_objs, env.vehicle, ego_rec_id, engine=env.engine)
        for rid, live_obj in obj_map.items():
            state = frame_data.get(rid)
            if state is not None:
                apply_state(live_obj, state)
        park_unmatched_live(live_objs, obj_map, env.vehicle, engine=env.engine)

    for step in range(n_frames):
        if step < len(npc_frames):
            teleport(npc_frames[step])
        collector.on_step(env, row)
        action = expert_actions[step] if step < len(expert_actions) else [0.0, 0.0]
        env.step(action)
        if step < len(npc_frames):
            teleport(npc_frames[step])

    restore_patched(patched)
    env.close()

    scene_uid = sidecar.get("scene_uid") or row.get("scene_uid") or sidecar.get("scene_id")
    variant = sidecar.get("variant") or "default"
    route_dir = plant2_route_dir(save_plant2_dir, str(scene_uid), str(variant))
    metrics = sidecar.get("metrics") or {}
    success = bool(metrics["success"]) if "success" in metrics else True
    n_written = collector.flush(route_dir, success=success)

    return {
        "scene_uid": scene_uid,
        "variant": variant,
        "steps": n_frames,
        "plant2_frames": n_written,
        "plant2_path": str(route_dir),
    }


def load_expert_rows(path: Path, count: int | None, start: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if count is not None and len(rows) >= count:
                break
    return rows


def resolve_expert_paths(expert_row: dict) -> tuple[Path, Path, str, str, str]:
    # Expert rows carry absolute paths written on whichever node collected them;
    # the same volume answers to more than one mount point, so resolve against
    # the equivalent roots before declaring a replay missing.
    from bench.mount_paths import resolve_shared_path

    sidecar_path = resolve_shared_path(expert_row["sidecar_path"])
    pkl_path = resolve_shared_path(expert_row.get("pkl_path") or json.loads(
        sidecar_path.read_text(encoding="utf-8"))["pkl_path"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    scene_uid = expert_row.get("scene_uid") or sidecar.get("scene_uid") or "?"
    variant = sidecar.get("variant") or expert_row.get("winner_variant") or "default"
    backend = sidecar.get("backend") or expert_row.get("backend") or "sumo"
    return pkl_path, sidecar_path, str(scene_uid), str(variant), str(backend)


def run_batch(
    experts_path: Path,
    scenes_root: Path,
    save_plant2_dir: Path,
    *,
    count: int | None = None,
    start: int = 0,
    max_steps: int = 1500,
    backends: str = "sumo",
) -> dict:
    from bench.plant2_frames import ensure_slurm_dummy, plant2_route_dir

    save_plant2_dir.mkdir(parents=True, exist_ok=True)
    ensure_slurm_dummy(save_plant2_dir)
    allowed = {b.strip() for b in backends.split(",") if b.strip()}
    rows = load_expert_rows(experts_path, count, start)
    ok = fail = skip = 0

    for expert_row in rows:
        pkl_path, sidecar_path, scene_uid, variant, backend = resolve_expert_paths(expert_row)
        if backend not in allowed:
            skip += 1
            continue
        route_dir = plant2_route_dir(save_plant2_dir, scene_uid, variant)
        if (route_dir / "results.json.gz").exists():
            skip += 1
            continue
        dump_plant2(
            pkl_path, sidecar_path,
            scenes_root=scenes_root,
            save_plant2_dir=save_plant2_dir,
            max_steps=max_steps,
        )
        ok += 1

    return {"ok": ok, "fail": fail, "skip": skip, "plant2_dir": str(save_plant2_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="PlanT2 dump from expert trajectories (v2)")
    parser.add_argument("--experts", type=Path, required=True)
    parser.add_argument("--scenes-root", type=Path, required=True)
    parser.add_argument("--save-plant2-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--backends", type=str, default="sumo")
    parser.add_argument("--pkl", type=Path, default=None)
    parser.add_argument("--sidecar", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.save_plant2_dir.resolve()
    scenes_root = args.scenes_root.resolve()

    if args.pkl and args.sidecar:
        result = dump_plant2(
            args.pkl, args.sidecar,
            scenes_root=scenes_root,
            save_plant2_dir=out_dir,
            max_steps=args.max_steps,
        )
        print(json.dumps(result, indent=2))
        return

    summary = run_batch(
        args.experts.resolve(), scenes_root, out_dir,
        count=args.count, start=args.start,
        max_steps=args.max_steps, backends=args.backends,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
