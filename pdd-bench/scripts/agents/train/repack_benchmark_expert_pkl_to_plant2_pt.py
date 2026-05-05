#!/usr/bin/env python3
"""
Repack per-sign benchmark expert replays (sd_*.pkl + *.meta.json) into .pt files
compatible with train_plant2_from_carl_trajectories.py.

Replays each scene like expert_replay_inenv.py: one MetaDrive step per recorded
ego action (``len(expert_actions)``), until the episode terminates or truncates.
Logs metadrive_obs_to_plant2_batch per step, then adds ego_pos_world_future_4 /
ego_pos_world_future_4_s3 using the same logic as collect_metadrive_carl_plant2_trajectories.py.

Requires --sdc-root to a checkout that contains pdd-bench/scripts/per_sign_bench/
(e.g. smirnova/sdc). For SUMO scenes recorded on another machine, use
--remap-net-path OLD:NEW so source_row[\"net_path\"] resolves (e.g. map
/Users/.../sdc -> .../arbelyaev/sdc).

Example::

  export SDL_VIDEODRIVER=dummy
  python repack_benchmark_expert_pkl_to_plant2_pt.py \\
    --benchmark-root .../benchmark_output/mini \\
    --output-dir ./plant2_pt \\
    --sdc-root .../smirnova/sdc \\
    --remap-net-path /Users/victoria_s/sdc_new_signs/sdc:/home/you/arbelyaev/sdc \\
    --num-workers 4
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import os
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

if os.environ.get("SDL_VIDEODRIVER") is None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


def _find_train_script_sdc_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for parent in (current, *current.parents):
        if (parent / "pdd-bench").is_dir() and (parent / "metadrive").is_dir():
            return parent
    raise RuntimeError("Could not locate SDC root (expected pdd-bench and metadrive next to each other)")


def _inject_import_paths(sdc_root: Path) -> None:
    pdd = sdc_root / "pdd-bench"
    md = sdc_root / "metadrive"
    bench = pdd / "scripts" / "per_sign_bench"
    if not (bench / "expert_replay.py").is_file():
        raise FileNotFoundError(
            f"Missing {bench / 'expert_replay.py'} — pass --sdc-root to a tree with per_sign_bench."
        )
    for p in (pdd, md, bench):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)


def _override_npc_policies_to_replay(env, scenario: dict) -> int:
    try:
        from metadrive.policy.replay_policy import ScenarioNetLikeReplayPolicy as ReplayPolicy
    except ImportError:
        try:
            from metadrive.policy.replay_policy import ReplayPolicy  # type: ignore
        except ImportError:
            print("[warn] ReplayPolicy not available — NPCs will not be frozen")
            return 0

    n_overridden = 0
    tracks = scenario.get("tracks", {})
    for obj_id, obj in list(env.engine.get_objects().items()):
        if obj_id == env.vehicle.id:
            continue
        track = tracks.get(obj_id)
        if track is None:
            continue
        try:
            env.engine.add_policy(obj_id, ReplayPolicy, obj, track)
            n_overridden += 1
        except Exception:
            continue
    return n_overridden


def _apply_net_path_remap(row: dict, remap: Optional[Tuple[str, str]]) -> dict:
    if remap is None or "net_path" not in row:
        return row
    old_prefix, new_prefix = remap
    out = copy.deepcopy(row)
    npth = out.get("net_path")
    if isinstance(npth, str) and npth.startswith(old_prefix):
        out["net_path"] = new_prefix + npth[len(old_prefix) :]
    return out


def _repack_one(
    pkl_path: Path,
    sidecar_path: Path,
    out_path: Path,
    max_steps: int,
    remap: Optional[Tuple[str, str]],
    input_bev: bool,
    input_ego_speed: bool,
) -> Dict[str, Any]:
    from expert_replay import _build_env  # type: ignore
    from factorized_space.benchmark_runner import SIGN_CLASS_MAP  # type: ignore
    from factorized_space.ego_defaults import apply_ego_defaults  # type: ignore
    from metadrive.policy.metadrive_obs_to_plant2 import metadrive_obs_to_plant2_batch  # type: ignore

    sidecar = json.load(open(sidecar_path, encoding="utf-8"))
    scenario = pickle.load(open(pkl_path, "rb"))

    row = _apply_net_path_remap(sidecar["source_row"], remap)
    backend = sidecar["backend"]

    env = _build_env(
        row,
        backend,
        max_steps=max_steps,
        record_episode=False,
        ego_policy_cls=None,
        render=False,
    )
    seed = int(sidecar["env_config_summary"].get("seed") or 0)
    if backend == "sumo":
        env_seed = (int(row.get("sign_id", 0)) + int(row.get("var_idx", 0))) % 100000
    else:
        env_seed = seed
    np.random.seed(seed)
    random.seed(seed)

    base_env = getattr(env, "unwrapped", env)
    steps_out: List[Dict[str, Any]] = []

    try:
        env.reset(seed=env_seed)

        sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
        rn = env.current_map.road_network
        for sign_info in sidecar.get("signs", []):
            cls_name = sign_info["sign_class"]
            sign_cls = None
            for _k, v in SIGN_CLASS_MAP.items():
                if v.__name__ == cls_name:
                    sign_cls = v
                    break
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
                sign_mgr.add_sign(
                    sign_cls,
                    lane=lane,
                    longitudinal_offset=sign_info.get("longitudinal_offset", 0.0),
                    lateral_offset=sign_info.get("lateral_offset", 0.0),
                    use_random_lane=False,
                )
            except Exception as exc:
                print(f"[warn] failed to re-add sign {cls_name}: {exc}")

        try:
            ego_policy = env.engine.get_policy(env.vehicle.id)
            if ego_policy is not None:
                apply_ego_defaults(ego_policy)
        except Exception:
            pass

        n_over = _override_npc_policies_to_replay(env, scenario)
        print(f"  [info] NPC replay policies overridden: {n_over}")

        expert_actions = list(sidecar.get("expert_actions") or [])
        n_actions = len(expert_actions)
        if n_actions == 0:
            print(f"[warn] no expert_actions in sidecar {sidecar_path}, 0 rollout steps", file=sys.stderr)
        info: Dict[str, Any] = {}
        for step in range(n_actions):
            ego_pos_before = np.asarray(getattr(base_env.agent, "position")[:2], dtype=np.float32)
            ego_heading_before = float(getattr(base_env.agent, "heading_theta", 0.0))

            plant2_batch = metadrive_obs_to_plant2_batch(
                base_env.engine,
                base_env.agent,
                route_ego_20x2=None,
                speed_limit_kmh=None,
                max_objects=30,
                max_distance=75.0,
                range_factor_front=16.0,
                input_bev=input_bev,
                input_ego_speed=input_ego_speed,
                bev_resolution=128,
                bev_size_meters=64.0,
                device="cpu",
            )
            plant2_batch_save: Dict[str, Any] = {}
            for k, v in plant2_batch.items():
                if v is None:
                    plant2_batch_save[k] = None
                elif torch.is_tensor(v):
                    plant2_batch_save[k] = v.cpu().numpy()
                else:
                    plant2_batch_save[k] = v
            plant2_batch_save["target_speed"] = np.array(
                [[float(getattr(base_env.agent, "speed", 0.0))]], dtype=np.float32
            )

            action = expert_actions[step]

            _, _r, term, trunc, info = env.step(action)

            ego_pos_after = np.asarray(
                getattr(base_env.agent, "position", [0.0, 0.0, 0.0])[:2], dtype=np.float32
            )

            steps_out.append(
                {
                    "plant2_batch": plant2_batch_save,
                    "ego_pos_world_before": ego_pos_before,
                    "ego_heading_before": ego_heading_before,
                    "ego_pos_world_after": ego_pos_after,
                    "action_env": np.asarray(action, dtype=np.float32),
                    "step_idx": np.array([float(step)], dtype=np.float32),
                    "terminated": bool(term),
                    "truncated": bool(trunc),
                }
            )

            if term or trunc:
                break

        n = len(steps_out)
        if n > 0:
            pos_after = [np.asarray(s["ego_pos_world_after"], dtype=np.float32) for s in steps_out]
            for i in range(n):
                fut_consec = []
                for k in range(4):
                    j = i + k
                    fut_consec.append(pos_after[j] if j < n else pos_after[-1])
                steps_out[i]["ego_pos_world_future_4"] = np.stack(fut_consec, axis=0).astype(np.float32)

                fut_s3 = []
                for k in range(1, 5):
                    j = i + k * 3
                    fut_s3.append(pos_after[j] if j < n else pos_after[-1])
                steps_out[i]["ego_pos_world_future_4_s3"] = np.stack(fut_s3, axis=0).astype(np.float32)

        ep_data: Dict[str, Any] = {
            "scene_id": sidecar.get("scene_id"),
            "backend": backend,
            "pdd_code": sidecar.get("pdd_code"),
            "source_sidecar": str(sidecar_path),
            "source_pkl": str(pkl_path),
            "reset_seed": env_seed,
            "seed": seed,
            "steps": steps_out,
            "num_steps": len(steps_out),
            "metrics_sidecar": sidecar.get("metrics"),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ep_data, out_path)
        return {"ok": True, "steps": len(steps_out), "out": str(out_path)}
    finally:
        try:
            env.close()
        except Exception:
            pass


def _repack_worker(task: Tuple[Any, ...]) -> Dict[str, Any]:
    """Picklable entry point for ProcessPoolExecutor (spawn). One env per call."""
    (
        sdc_root_s,
        pkl_s,
        sidecar_s,
        out_s,
        max_steps,
        remap,
        input_bev,
        input_ego_speed,
        task_i,
        n_total,
        rel_display,
    ) = task
    try:
        _inject_import_paths(Path(sdc_root_s))
        r = _repack_one(
            Path(pkl_s),
            Path(sidecar_s),
            Path(out_s),
            max_steps,
            remap,
            input_bev,
            input_ego_speed,
        )
        return {
            "ok": bool(r.get("ok")),
            "steps": r.get("steps"),
            "out": r.get("out"),
            "error": None,
            "task_i": task_i,
            "n_total": n_total,
            "rel": rel_display,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "task_i": task_i,
            "n_total": n_total,
            "rel": rel_display,
            "pkl": pkl_s,
        }


def _parse_remap(s: Optional[str]) -> Optional[Tuple[str, str]]:
    if not s:
        return None
    if ":" not in s:
        raise ValueError("--remap-net-path must be OLD:NEW")
    old, new = s.split(":", 1)
    if not old:
        raise ValueError("empty OLD prefix in --remap-net-path")
    return (old, new)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repack benchmark expert sd_*.pkl + meta.json into Plant2 .pt trajectories."
    )
    parser.add_argument("--benchmark-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sdc-root", type=str, default=None)
    parser.add_argument("--remap-net-path", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N pkls (0=all)")
    parser.add_argument(
        "--no-bev",
        action="store_true",
        help="input_bev=False (CaRL collector uses True by default)",
    )
    parser.add_argument(
        "--no-ego-speed-input",
        action="store_true",
        help="input_ego_speed=False (CaRL collector uses True by default)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel worker processes (each runs one episode at a time). "
        "Uses spawn; keep N modest (2–4) to avoid OOM. Default 1 = sequential.",
    )
    args = parser.parse_args()

    sdc_root = Path(args.sdc_root).resolve() if args.sdc_root else _find_train_script_sdc_root(Path(__file__))
    if args.num_workers < 1:
        print("--num-workers must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.num_workers == 1:
        _inject_import_paths(sdc_root)

    benchmark_root = Path(args.benchmark_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    remap = _parse_remap(args.remap_net_path)
    input_bev = not args.no_bev
    input_ego_speed = not args.no_ego_speed_input

    pkls = sorted(benchmark_root.glob("**/expert/replays/sd_*.pkl"))
    if not pkls:
        print(f"No sd_*.pkl under {benchmark_root}/**/expert/replays/", file=sys.stderr)
        sys.exit(1)

    if args.limit > 0:
        pkls = pkls[: args.limit]

    tasks: List[Tuple[Any, ...]] = []
    skipped = 0
    for i, pkl_path in enumerate(pkls, 1):
        name = pkl_path.name
        if not name.startswith("sd_") or not name.endswith(".pkl"):
            continue
        scene_part = name[len("sd_") : -4]
        sidecar = pkl_path.parent / f"{scene_part}.meta.json"
        if not sidecar.is_file():
            print(f"[skip] no sidecar for {pkl_path.name}", file=sys.stderr)
            skipped += 1
            continue

        safe_id = scene_part.replace("/", "_")
        out_pt = out_dir / f"{safe_id}_plant2.pt"
        rel = str(pkl_path.relative_to(benchmark_root))
        tasks.append(
            (
                str(sdc_root),
                str(pkl_path.resolve()),
                str(sidecar.resolve()),
                str(out_pt.resolve()),
                args.max_steps,
                remap,
                input_bev,
                input_ego_speed,
                i,
                len(pkls),
                rel,
            )
        )

    ok, failed = 0, skipped

    if not tasks:
        print("No tasks to run (missing sidecars or empty glob).", file=sys.stderr)
        sys.exit(1)

    if args.num_workers == 1:
        for t in tasks:
            (
                _sdc,
                pkl_s,
                sidecar_s,
                out_s,
                max_steps,
                remap_t,
                input_bev_t,
                input_ego_speed_t,
                task_i,
                n_total,
                rel,
            ) = t
            print(f"[{task_i}/{n_total}] {rel} -> {Path(out_s).name}")
            try:
                r = _repack_one(
                    Path(pkl_s),
                    Path(sidecar_s),
                    Path(out_s),
                    max_steps,
                    remap_t,
                    input_bev_t,
                    input_ego_speed_t,
                )
                if r.get("ok"):
                    ok += 1
                else:
                    failed += 1
            except Exception as exc:
                print(f"[fail] {pkl_s}: {type(exc).__name__}: {exc}", file=sys.stderr)
                failed += 1
    else:
        ctx = multiprocessing.get_context("spawn")
        n_workers = min(args.num_workers, len(tasks))
        print(
            f"Parallel repack: {len(tasks)} tasks, {n_workers} workers (spawn)",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = {pool.submit(_repack_worker, t): t for t in tasks}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    print(f"[fail] worker crashed: {exc}", file=sys.stderr)
                    failed += 1
                    continue
                label = f"[{res['task_i']}/{res['n_total']}] {res['rel']}"
                if res.get("ok"):
                    ok += 1
                    print(f"{label} -> ok steps={res.get('steps')}", flush=True)
                else:
                    failed += 1
                    err = res.get("error", "unknown")
                    print(f"{label} -> FAIL {err}", file=sys.stderr, flush=True)

    print(f"Done. ok={ok} failed={failed} output_dir={out_dir}")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
