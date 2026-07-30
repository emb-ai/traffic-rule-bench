"""Run a single policy over a per-sign benchmark manifest and write per-episode
metrics (+ optional replay sidecars / GIFs).

Thin CLI orchestrator. Heavy lifting lives in `bench/`:
  bench.env_builders   — build pgmap/sumo env from a manifest row + place signs
  bench.policy_factory — load NN checkpoints, resolve the ego BasePolicy class
  bench.episode_metrics— pure per-episode metric helpers (efficiency/TTC/smoothness)
  bench.sign_eval      — violation / zone-of-effect / crash-attribution helpers
  bench.manifest_io    — manifest reading, scene collection, resume keys
  bench.util           — seeding + tiny row/env helpers

Usage examples:
  # IDM on a SUMO manifest (no checkpoint needed):
  python run_benchmark.py --policy idm --run-name idm_default \\
      --manifest <m.jsonl> --scenes-root scenes --backends sumo \\
      --benchmark-output <out>

  # Sampled IDM ego variant (s1..s4):
  python run_benchmark.py --policy idm --ego-variant s1 --run-name idm_s1 \\
      --manifest <m.jsonl> --scenes-root scenes --backends sumo --benchmark-output <out>

  # NN policy (needs --model-path):
  python run_benchmark.py --policy plant2 --model-path <ckpt> --run-name plant2 \\
      --manifest <m.jsonl> --scenes-root scenes --backends sumo --benchmark-output <out>

  # One scene + GIF:
  python run_benchmark.py --policy idm --run-name idm --manifest <m.jsonl> \\
      --scenes-root scenes --backends sumo --scene-uid <uid> --save-gifs

Typically invoked per-policy by eval_pipeline.py (which also builds the CSV +
aggregations). Keep the CLI flags stable — eval_pipeline calls this by name.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# Path bootstrap: add this script's dir to sys.path so `bench` is importable,
# then bench._paths performs the full setup (pdd-bench root, metadrive, CaRL) as
# an import side effect — single source of truth for paths (no duplicate here).
_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
from bench._paths import PDD_BENCH_DIR  # noqa: E402  (import sets sys.path)

# --- extracted helpers (see bench/) ---
# run_benchmark stays backend-agnostic: env construction + per-backend seed +
# post-reset sign placement are encapsulated in build_env_for_row, and ego-policy
# resolution/variant sampling in make_ego_policy. To add a scene/backend
# peculiarity, edit those modules — NOT this file.
from bench.util import (_row_seed, _row_sign_code, _seed_everything,
                        _unwrap_base_env, _error_result)
from bench.manifest_io import (load_manifest_rows, select_rows_to_run,
                               _episode_key_from_result, _load_existing_results)
from bench.sign_eval import (_format_violation, _violation_bucket,
                             _ego_in_sign_zone, _extract_sign_info,
                             _ego_at_fault_for_crash)
from bench.episode_metrics import (_safe_float, _route_completion_percent,
                                   _route_length_m, _infraction_penalty,
                                   _nearby_speed_percentage, _min_ttc_seconds,
                                   _compute_smoothness)
from bench.env_builders import build_env_for_row
from bench import env_builders as _env_builders
from bench.policy_factory import _load_policy_models, make_ego_policy

_PHYSICS_DT = 0.1   # MetaDrive physics step (s); used for accel/jerk derivation.


@dataclass
class Rollout:
    """All mutable state for one episode — the loop writes straight into this.

    Single source: there are NO separate pre-loop local accumulators. Fields above
    the divider are the episode RECORD (read by run_one_episode to build the
    episodes row + sidecar); fields below are transient loop scratch (intermediate
    buffers / previous-step trackers) that the caller ignores.
    """
    # --- episode record (read by run_one_episode) ---
    steps: int = 0
    total_reward: float = 0.0
    violations: int = 0
    sign_violations: int = 0
    traffic_light_violations: int = 0
    crosswalk_violations: int = 0
    # Per-step per-class counter: {StopSign: 50, TrafficLightSign: 5} — +1 every
    # step a sign of that class is violating.
    violations_by_class_step: dict = field(default_factory=dict)
    # Edge-counted: +1 only on a "not violating → started violating" transition.
    violations_event_count: int = 0
    violations_by_class_event: dict = field(default_factory=dict)
    violations_timeline: list = field(default_factory=list)
    # Steps ego was inside (or approaching) each sign's zone of effect.
    in_zone_total_steps: int = 0
    in_zone_by_class_step: dict = field(default_factory=dict)
    hard_brake_count: int = 0
    hard_accel_count: int = 0
    distance_travelled_m: float = 0.0
    min_ttc: float | None = None
    mean_abs_lane_offset: float | None = None
    mean_abs_steer_delta: float | None = None
    smoothness: dict = field(default_factory=dict)
    smoothness_step_vars: list = field(default_factory=list)
    # Per-step expert actions (mirror expert_replay.py:543) — written to sidecar.
    expert_actions: list = field(default_factory=list)
    sign_info_snapshot: list = field(default_factory=list)
    crashed: bool = False
    out_of_road: bool = False
    reached_dest: bool = False
    crashed_flag_raw: bool = False
    crash_attribution: str | None = None
    crashed_ego_fault: bool = False
    crashed_npc_fault: bool = False
    route_completion_pct: float = 0.0
    infraction_penalty: float = 1.0
    driving_score: float = 0.0
    driving_efficiency: float = 0.0
    route_length_m: float | None = None
    route_length_source: str = "none"
    # --- transient loop scratch (intermediate; not part of the record) ---
    speed_pct_samples: list = field(default_factory=list)
    abs_lane_offsets: list = field(default_factory=list)
    steer_delta_abs: list = field(default_factory=list)
    visited_lane_lengths: dict = field(default_factory=dict)
    prev_speed_mps: float | None = None
    prev_heading: float | None = None
    prev_long_acc: float | None = None
    prev_lat_acc: float | None = None
    prev_yaw_rate: float | None = None
    prev_action_steer: float | None = None
    prev_violated_class_names: set = field(default_factory=set)
    last_violation_texts: list = field(default_factory=list)
    violation_text_ttl: int = 0
    last_info: dict = field(default_factory=dict)


def _run_rollout(env, base_env, policy_obj, *, max_steps: int,
                 save_gif: Path | None = None,
                 step_hook: Callable[[], None] | None = None) -> Rollout:
    """Step the ego policy through one reset env, writing all state into a Rollout.

    Pure rollout — no manifest/identity/output concerns. All episode state lives
    on the returned Rollout (no separate local accumulators); only per-iteration
    temporaries are local. run_one_episode wraps this with env setup and record
    assembly (episodes row + optional sidecar).

    ``step_hook`` (optional) runs once per step *before* ``policy.act`` / ``env.step``,
    while the env is still in the pre-action state (PlanT2 frame capture, freezes).
    Default ``None`` leaves eval/recorder behaviour unchanged.
    """
    r = Rollout()
    # One-shot snapshot of placed signs after sign-placement (post-reset).
    r.sign_info_snapshot = _extract_sign_info(base_env)

    for step in range(max_steps):
        if step_hook is not None:
            step_hook()
        action = policy_obj.act(base_env.vehicle.name)
        r.expert_actions.append([float(action[0]), float(action[1])])

        obs, reward, terminated, truncated, info = env.step(action)
        r.last_info = info
        r.total_reward += float(reward)
        r.steps += 1

        vehicle = base_env.agent
        sign_mgr = getattr(base_env.engine, "traffic_sign_manager", None)
        current_violation_texts = []
        current_violated_class_names: set = set()
        if sign_mgr is not None and vehicle is not None:
            # In-zone tracking (per-step, per-class). Done once per step
            # over all signs — independent from violation check below.
            step_in_any_zone = False
            for _s in sign_mgr.signs:
                if _ego_in_sign_zone(_s, vehicle):
                    step_in_any_zone = True
                    cls = type(_s).__name__
                    r.in_zone_by_class_step[cls] = r.in_zone_by_class_step.get(cls, 0) + 1
            if step_in_any_zone:
                r.in_zone_total_steps += 1

            current_violations = sign_mgr.check_all_violations(vehicle)
            for _sign, violated in current_violations:
                if violated:
                    r.violations += 1
                    bucket = _violation_bucket(_sign)
                    if bucket == "traffic_light":
                        r.traffic_light_violations += 1
                    elif bucket == "crosswalk":
                        r.crosswalk_violations += 1
                    else:
                        r.sign_violations += 1
                    current_violation_texts.append(_format_violation(_sign, vehicle))
                    cls_name = type(_sign).__name__
                    r.violations_by_class_step[cls_name] = (
                        r.violations_by_class_step.get(cls_name, 0) + 1)
                    current_violated_class_names.add(cls_name)
                    if cls_name not in r.prev_violated_class_names:
                        r.violations_event_count += 1
                        r.violations_by_class_event[cls_name] = (
                            r.violations_by_class_event.get(cls_name, 0) + 1)
                        try:
                            rule = _sign.get_rule_description() or ""
                        except Exception:
                            rule = ""
                        r.violations_timeline.append({
                            "step": int(step),
                            "sign_class": cls_name,
                            "rule": rule,
                        })
            r.prev_violated_class_names = current_violated_class_names
            if current_violation_texts:
                r.last_violation_texts = current_violation_texts[:3]
                r.violation_text_ttl = 40
            elif r.violation_text_ttl > 0:
                r.violation_text_ttl -= 1
            else:
                r.last_violation_texts = []

        # Per-step ego metrics (efficiency proxy + quality/safety kinematics).
        if vehicle is not None:
            # Bench2Drive-like efficiency proxy: ego speed vs surrounding traffic.
            sp = _nearby_speed_percentage(vehicle)
            if sp is not None:
                r.speed_pct_samples.append(float(sp))

            speed_mps = _safe_float(getattr(vehicle, "speed", 0.0), 0.0)
            r.distance_travelled_m += max(0.0, speed_mps) * _PHYSICS_DT
            heading = _safe_float(getattr(vehicle, "heading_theta", 0.0), 0.0)

            lane = getattr(vehicle, "lane", None)
            if lane is not None:
                try:
                    _long, lat = lane.local_coordinates(vehicle.position)
                    r.abs_lane_offsets.append(abs(float(lat)))
                except Exception:
                    pass
                try:
                    lane_idx = getattr(lane, "index", None)
                    lane_key = repr(lane_idx) if lane_idx is not None else f"lane_obj_{id(lane)}"
                    lane_len = float(getattr(lane, "length", 0.0) or 0.0)
                    if lane_len > 0.0 and lane_key not in r.visited_lane_lengths:
                        r.visited_lane_lengths[lane_key] = lane_len
                except Exception:
                    pass

            step_ttc = _min_ttc_seconds(vehicle)
            if step_ttc is not None and step_ttc > 0.0:
                r.min_ttc = step_ttc if r.min_ttc is None else min(r.min_ttc, step_ttc)

            cur_steer = _safe_float(action[0], 0.0)
            if r.prev_action_steer is not None:
                r.steer_delta_abs.append(abs(cur_steer - r.prev_action_steer))
            r.prev_action_steer = cur_steer

            if r.prev_speed_mps is not None and r.prev_heading is not None:
                long_acc = (speed_mps - r.prev_speed_mps) / _PHYSICS_DT
                yaw_delta = math.atan2(math.sin(heading - r.prev_heading), math.cos(heading - r.prev_heading))
                yaw_rate = yaw_delta / _PHYSICS_DT
                lat_acc = speed_mps * yaw_rate

                if long_acc < -3.0:
                    r.hard_brake_count += 1
                if long_acc > 2.5:
                    r.hard_accel_count += 1

                if r.prev_long_acc is not None and r.prev_lat_acc is not None and r.prev_yaw_rate is not None:
                    long_jerk = (long_acc - r.prev_long_acc) / _PHYSICS_DT
                    lat_jerk = (lat_acc - r.prev_lat_acc) / _PHYSICS_DT
                    yaw_acc = (yaw_rate - r.prev_yaw_rate) / _PHYSICS_DT
                    jerk_mag = float(math.sqrt(long_jerk * long_jerk + lat_jerk * lat_jerk))
                    r.smoothness_step_vars.append(
                        {
                            "long_acc": float(long_acc),
                            "lat_acc": float(lat_acc),
                            "yaw_rate": float(yaw_rate),
                            "yaw_acc": float(yaw_acc),
                            "long_jerk": float(long_jerk),
                            "jerk_mag": jerk_mag,
                        }
                    )

                r.prev_long_acc = long_acc
                r.prev_lat_acc = lat_acc
                r.prev_yaw_rate = yaw_rate

            r.prev_speed_mps = speed_mps
            r.prev_heading = heading

        if terminated or truncated:
            r.reached_dest = bool(info.get("arrive_dest", False))
            r.out_of_road = bool(info.get("out_of_road", False))
            r.crashed = bool(info.get("crash", False) or r.out_of_road)
            break

        # text_dict is only assembled when GIF recording is on (it's only
        # used as the render() text overlay). Skip the work otherwise.
        text_dict: dict = {}
        if save_gif:
            text_dict = {
                "Step": step,
                "Speed": f"{vehicle.speed_km_h:.2f} km/h",
                "Violations": r.sign_violations + r.crosswalk_violations,
            }

        if save_gif:
            try:
                base_env.render(
                    mode="top_down",
                    film_size=(4800, 4800), scaling=24.0,
                    screen_size=(800, 800),
                    semantic_map=True,
                    semantic_broken_line=True,
                    draw_target_vehicle_trajectory=True,
                    target_agent_heading_up=True,
                    screen_record=True, window=False,
                    text=text_dict,
                )
            except Exception:
                pass

    r.route_completion_pct = _route_completion_percent(r.last_info, r.reached_dest)
    r.infraction_penalty = _infraction_penalty(crashed=r.crashed, out_of_road=r.out_of_road, violations=r.violations)
    r.driving_score = r.route_completion_pct * r.infraction_penalty
    r.driving_efficiency = float(np.mean(r.speed_pct_samples)) if r.speed_pct_samples else 0.0
    r.smoothness = _compute_smoothness(r.smoothness_step_vars, segment_len=20)
    r.route_length_m = _route_length_m(r.last_info)
    r.route_length_source = "info"
    if r.route_length_m is None:
        approx = float(sum(r.visited_lane_lengths.values()))
        if approx > 0.0:
            r.route_length_m = approx
            r.route_length_source = "visited_lanes"
        else:
            r.route_length_source = "none"

    # Sidecar uses raw info["crash"] (without OOR) per expert_replay schema;
    # episodes JSONL uses (crash OR OOR) per legacy run_benchmark convention.
    r.crashed_flag_raw = bool(r.last_info.get("crash", False)) if r.last_info else False
    if r.crashed_flag_raw or bool(getattr(base_env.agent, "crash_vehicle", False)):
        try:
            r.crash_attribution = "ego" if _ego_at_fault_for_crash(base_env.agent, base_env.engine) else "npc"
        except Exception:
            r.crash_attribution = None
    r.crashed_ego_fault = bool(r.crashed_flag_raw and r.crash_attribution == "ego")
    r.crashed_npc_fault = bool(r.crashed_flag_raw and r.crash_attribution == "npc")

    r.mean_abs_lane_offset = float(np.mean(r.abs_lane_offsets)) if r.abs_lane_offsets else None
    r.mean_abs_steer_delta = float(np.mean(r.steer_delta_abs)) if r.steer_delta_abs else None
    return r


def build_sidecar_metrics(r: Rollout) -> dict:
    """Sidecar `metrics` block from a finished Rollout — the single source of
    truth for the replay.json metrics schema (also imported by expert_replay.py,
    so the recorder and the eval can never drift apart on metric semantics)."""
    return {
        "arrived_dest": bool(r.reached_dest),
        "crashed": r.crashed_flag_raw,
        "crash_attribution": r.crash_attribution,
        "crashed_ego_fault": r.crashed_ego_fault,
        "crashed_npc_fault": r.crashed_npc_fault,
        "out_of_road": bool(r.out_of_road),
        "final_step": int(r.steps),
        # Per-step counts (frames where any sign was violating)
        "total_violations": int(r.violations),
        # 3-bucket per-step (run_benchmark-style):
        "violations_by_class": {
            "sign": int(r.sign_violations),
            "traffic_light": int(r.traffic_light_violations),
            "crosswalk": int(r.crosswalk_violations),
        },
        # Per-class per-step (NEW — combines both styles):
        # {StopSign: 50, TrafficLightSign: 5, ...}
        "violations_by_class_step": dict(r.violations_by_class_step),
        # Steps ego was inside (or approaching) any sign's zone.
        # Pair with violations_by_class_step → per-zone violation rate.
        "in_zone_total_steps": int(r.in_zone_total_steps),
        "in_zone_by_class_step": dict(r.in_zone_by_class_step),
        # Edge-counts (expert_replay style — one event per class
        # transition not-violating → started-violating):
        "violations_event_count": int(r.violations_event_count),
        "violations_by_class_event": dict(r.violations_by_class_event),
        "violations_timeline": list(r.violations_timeline),
        "route_completion": (float(r.route_completion_pct) / 100.0
                              if r.route_completion_pct else 0.0),
        "total_reward": round(float(r.total_reward), 4),
        "smoothness_ratio": r.smoothness["smoothness_ratio"],
        "frame_smooth_ratio": r.smoothness["frame_smooth_ratio"],
        "smooth_segments": r.smoothness["smooth_segments"],
        "total_segments": r.smoothness["total_segments"],
        "driving_score": float(r.driving_score),
        "driving_efficiency": float(r.driving_efficiency),
        "infraction_penalty": float(r.infraction_penalty),
        "min_ttc_sec": float(r.min_ttc) if r.min_ttc is not None else None,
        "mean_abs_lane_offset": r.mean_abs_lane_offset,
        "mean_abs_steer_delta": r.mean_abs_steer_delta,
        "hard_brake_count": int(r.hard_brake_count),
        "hard_accel_count": int(r.hard_accel_count),
        "route_length_m": (float(r.route_length_m)
                            if r.route_length_m is not None else None),
        "distance_travelled_m": float(r.distance_travelled_m),
        "success": bool(r.reached_dest and not r.crashed_flag_raw and not r.out_of_road),
    }


def run_one_episode(
    row: dict,
    backend: str,
    policy_type: str,
    models: dict,
    scenes_root: Path,
    max_steps: int,
    ego_variant: str,
    ego_sample_seed_base: int,
    replay_root: Path | None = None,
    save_gif: Path | None = None,
) -> dict:
    """Build the env for a manifest row, run one rollout, assemble the record.

    Thin orchestrator: env construction lives in build_env_for_row, the policy in
    make_ego_policy, the step loop + metrics in _run_rollout. This function only
    wires them together and shapes the episodes row (+ optional replay sidecar).
    """
    seed = _row_seed(row)
    _seed_everything(seed)

    env, env_seed, post_reset = build_env_for_row(
        row, backend, scenes_root=scenes_root, max_steps=max_steps)

    try:
        obs, info = env.reset(seed=env_seed)
        base_env = _unwrap_base_env(env)
        if hasattr(base_env, "engine") and hasattr(base_env.engine, "np_random"):
            base_env.engine.np_random = np.random.RandomState(seed)

        # Backend-specific post-reset setup (e.g. pgmap sign placement). Returns an
        # error string if the episode can't be set up, else None.
        setup_error = post_reset(base_env)
        if setup_error:
            return _error_result(row, setup_error, backend=backend)

        # Resolve + instantiate the ego BasePolicy and apply the IDM ego variant.
        # Braking-spawn: default ego-IDM "holds v0" (desired speed >= spawn speed)
        # so a sign-unaware agent enters the zone above the limit instead of decaying
        # to 36 km/h right at the sign (vacuous v40 compliance). Rule-expert: sign-capped.
        ego_hold_speed_ms = None
        if row.get("braking_spawn"):
            try:
                ego_hold_speed_ms = float(row.get("spawn_velocity_ms") or 0.0) or None
            except (TypeError, ValueError):
                ego_hold_speed_ms = None
        policy_obj, sampled_ego_params = make_ego_policy(
            policy_type, models, base_env, seed,
            ego_variant=ego_variant, ego_sample_seed_base=ego_sample_seed_base,
            ego_hold_speed_ms=ego_hold_speed_ms)

        r = _run_rollout(env, base_env, policy_obj,
                         max_steps=max_steps, save_gif=save_gif)

        # Stable per-episode identity. Computed once and embedded in BOTH the
        # episodes record (below) and the optional sidecar — so the metrics CSV
        # can be built from episodes_*.jsonl alone, no sidecar required.
        sign_slug = str(_row_sign_code(row) or "").replace(".", "_")
        scene_id_for_uid = row.get("scene_id") or f"scene_{seed}"
        lane_for_uid = int(row.get("spawn_lane_num", 0) or 0)
        var_for_uid = int(row.get("var_idx", 0) or 0)
        scene_uid = f"{scene_id_for_uid}_lane{lane_for_uid}_seed{seed}_v{var_for_uid}"

        if replay_root is not None:
            try:
                expert_subdir = f"{policy_type}_{ego_variant}" if ego_variant else policy_type

                out_replay = (Path(replay_root) / sign_slug / "by_sign" / sign_slug
                              / "by_scene" / scene_uid / expert_subdir)
                out_replay.mkdir(parents=True, exist_ok=True)
                sidecar_path = out_replay / "replay.json"

                sidecar_metrics = build_sidecar_metrics(r)
                sidecar = {
                    "scene_id": scene_id_for_uid,
                    "scene_uid": scene_uid,
                    "backend": backend,
                    "pdd_code": (row.get("pdd_code") or row.get("sign_code")
                                 or row.get("sign_type")),
                    "sign_key": row.get("sign_type"),
                    "sign_slug": sign_slug,
                    "policy": policy_type,
                    "variant": ego_variant,
                    "source_row": row,
                    "env_config_summary": {
                        "map_name": row.get("net_path"),
                        "road_id": row.get("road_id"),
                        "spawn_lane_num": row.get("spawn_lane_num"),
                        "lane_num": row.get("lane_num"),
                        "lane_width": row.get("lane_width"),
                        "horizon": max_steps,
                        "seed": seed,
                    },
                    "signs": r.sign_info_snapshot,
                    "expert_actions": r.expert_actions,
                    "smoothness_step_vars": r.smoothness_step_vars,
                    "metrics": sidecar_metrics,
                    "ego_idm_params": (sampled_ego_params if sampled_ego_params is not None
                                        else "DEFAULT_EGO_PARAMS"),
                    "pkl_path": None,
                    "sidecar_path": str(sidecar_path),
                    "valid": True,
                }
                with open(sidecar_path, "w", encoding="utf-8") as _sf:
                    json.dump(sidecar, _sf, default=str)
            except Exception:
                # Fail soft — episodes JSONL still gets written below.
                pass

        return {
            "ok": True,
            "backend": backend,
            "scene_id": row.get("scene_id"),
            "sign_type": _row_sign_code(row),
            "seed": seed,
            # Identity (lets build_episode_metrics_csv consume episodes_*.jsonl
            # directly — no replay.json sidecar needed):
            "policy": policy_type,
            "scene_uid": scene_uid,
            "sign_slug": sign_slug,
            "total_reward": r.total_reward,
            "steps": r.steps,
            "violations": r.violations,
            "sign_violations": int(r.sign_violations),
            "traffic_light_violations": int(r.traffic_light_violations),
            "crosswalk_violations": int(r.crosswalk_violations),
            "crashed": r.crashed,
            # `crashed` is (crash OR out_of_road) per legacy convention; `crashed_raw`
            # is info["crash"] alone (matches the sidecar `metrics.crashed`).
            "crashed_raw": r.crashed_flag_raw,
            "crash_attribution": r.crash_attribution,
            "crashed_ego_fault": r.crashed_ego_fault,
            "crashed_npc_fault": r.crashed_npc_fault,
            "out_of_road": r.out_of_road,
            "reached_dest": r.reached_dest,
            "success": r.reached_dest and not r.crashed,
            "route_completion_pct": r.route_completion_pct,
            "infraction_penalty": r.infraction_penalty,
            "driving_score": r.driving_score,
            "driving_efficiency": r.driving_efficiency,
            "smoothness": r.smoothness["smoothness_ratio"],
            "smoothness_frame_ratio": r.smoothness["frame_smooth_ratio"],
            "smooth_segments": r.smoothness["smooth_segments"],
            "smooth_total_segments": r.smoothness["total_segments"],
            "hard_brake_count": int(r.hard_brake_count),
            "hard_accel_count": int(r.hard_accel_count),
            "mean_abs_lane_offset": r.mean_abs_lane_offset,
            "mean_abs_steer_delta": r.mean_abs_steer_delta,
            "min_ttc_sec": float(r.min_ttc) if r.min_ttc is not None else None,
            "route_length_m": float(r.route_length_m) if r.route_length_m is not None else None,
            "route_length_source": r.route_length_source,
            "distance_travelled_m": float(r.distance_travelled_m),
            "variant": ego_variant,
            "ego_params": sampled_ego_params,
            # Per-class per-step counts (granular run_benchmark-style)
            "violations_by_class_step": dict(r.violations_by_class_step),
            # Edge-counted violations (expert_replay style)
            "violations_event_count": int(r.violations_event_count),
            "violations_by_class_event": dict(r.violations_by_class_event),
            "violations_timeline": list(r.violations_timeline),
            # Per-class in-zone exposure (for per-zone violation rate)
            "in_zone_total_steps": int(r.in_zone_total_steps),
            "in_zone_by_class_step": dict(r.in_zone_by_class_step),
        }
    finally:
        if save_gif is not None:
            try:
                save_gif.parent.mkdir(parents=True, exist_ok=True)
                renderer = getattr(_unwrap_base_env(env), "top_down_renderer", None)
                if renderer is not None:
                    renderer.generate_gif(str(save_gif), duration=40)
            except Exception:
                pass
        try:
            env.close()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run policies on per_sign_bench full manifests")
    parser.add_argument("--policy", required=True,
                        choices=["idm", "comprehensive_rule_expert",
                                 "rule_compliant", "ppo_lidar",
                                 "carl", "carl_rule",
                                 "plant2", "plant2_rule"])
    parser.add_argument("--model-path", type=str, default=None,
                        help="Required for carl/plant2")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--benchmark-output", type=str, default="benchmark_output",
                        help="Base output dir (relative to cwd, or absolute); episodes "
                             "go to <it>/policy_eval/<run_name>/")
    parser.add_argument("--scenes-root", type=str, default=str(PDD_BENCH_DIR / "scenes"))
    parser.add_argument("--backends", type=str, default="sumo,pgmap,paired,citymap",
                        help="Comma-separated: sumo,pgmap,paired,citymap")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--ego-variant", type=str, default="default",
                        help="Ego IDM variant label: default or s1/s2/s3/s4")
    parser.add_argument("--ego-sample-seed-base", type=int, default=42,
                        help="Base seed for sampled IDM ego variants")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="Recompute scenes with existing failed records (ok=false)")
    parser.add_argument("--skip-error-episodes", action="store_true",
                        help="When used with --rerun-failed, keep previously errored episodes skipped")
    parser.add_argument("--emit-replay-sidecar", action="store_true",
                        help="Also emit per-(scene_uid, variant) replay.json sidecar in expert_replay layout (no pkl).")
    parser.add_argument("--replay-root", type=str, default=None,
                        help="Output dir for sidecar files (used with --emit-replay-sidecar). "
                             "Default: <out_dir>/replays")
    parser.add_argument("--scene-id", type=str, default=None,
                        help="Run only the scene with this scene_id.")
    parser.add_argument("--scene-uid", type=str, default=None,
                        help="Run only the scene matching this exact UID "
                             "<backend>:<scene_id>:<sign_type>:<seed>. "
                             "Mutually exclusive with --scene-id.")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to the *.jsonl manifest to evaluate (the only "
                             "source of scenes).")
    parser.add_argument("--save-gifs", action="store_true",
                        help="Record top-down GIF per episode (slow, ~3-5x overhead).")
    parser.add_argument("--gif-dir", type=str, default=None,
                        help="Directory for GIFs (default: <out_dir>/gifs).")
    parser.add_argument("--plant2-action-mode", type=str, default="pid",
                        choices=["pid", "wps_pure_pursuit"],
                        help="How PlanT2 converts pred_plan -> action. "
                             "'pid' (default) = plant2_predictions_to_action with "
                             "PCHIP+LateralPID on pred_path. 'wps_pure_pursuit' = "
                             "pure-pursuit on pred_wps[1] + softmax(pred_speed) "
                             "throttle (matches eval_plant2_wps_steer.py). "
                             "Applies to --policy plant2 and plant2_rule.")
    parser.add_argument("--relocate-ego-to-sign-lane", type=str, default="auto",
                        choices=["auto", "true", "false"],
                        help="After sign placement, teleport ego onto the "
                             "sign-topology lane. 'auto' (default) = True for "
                             "idm/comprehensive_rule_expert/rule_compliant, False "
                             "for NN policies (plant2/carl/ppo_lidar) — matches "
                             "1300c1e (NN policies fail when relocated off the "
                             "manifest road_id). 'true'/'false' force the value.")
    return parser


def main():
    args = build_parser().parse_args()

    if args.scene_id and args.scene_uid:
        raise ValueError("--scene-id and --scene-uid are mutually exclusive")

    assert args.ego_variant in ("default", "s1", "s2", "s3", "s4"), \
        f"--ego-variant must be one of default/s1/s2/s3/s4, got {args.ego_variant!r}"

    # relocate_ego_to_sign_lane: NN policies (plant2/carl/ppo_lidar) need ego left
    # on the manifest road_id, not teleported onto the sign lane (cf 1300c1e).
    _idm_family = {"idm", "comprehensive_rule_expert", "rule_compliant"}
    if args.relocate_ego_to_sign_lane == "auto":
        _env_builders.RELOCATE_EGO_TO_SIGN_LANE = args.policy in _idm_family
    else:
        _env_builders.RELOCATE_EGO_TO_SIGN_LANE = (args.relocate_ego_to_sign_lane == "true")
    print(f"relocate_ego_to_sign_lane: {_env_builders.RELOCATE_EGO_TO_SIGN_LANE}")

    logging.getLogger().setLevel(getattr(logging, "CRITICAL"))

    # Standard CLI resolution: relative paths resolve against the current working
    # directory, absolute paths are used as-is (eval_pipeline passes absolute).
    benchmark_output_dir = Path(args.benchmark_output).resolve()

    scenes_root = Path(args.scenes_root).resolve()
    if not scenes_root.exists():
        raise ValueError(f"Scenes root not found: {scenes_root}")

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    allowed = {"sumo", "pgmap", "paired", "citymap"}
    bad = [b for b in backends if b not in allowed]
    if bad:
        raise ValueError(f"Unsupported backends: {bad}; allowed={sorted(allowed)}")

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"--manifest not found: {manifest_path}")

    print(f"Policy: {args.policy}")
    print(f"Backends: {backends}")
    print(f"Manifest: {manifest_path}")

    rows = load_manifest_rows(manifest_path, backends,
                              scene_id=args.scene_id, scene_uid=args.scene_uid)
    if not rows:
        raise RuntimeError(
            "No scenes selected. Check --manifest/--scene-id/--scene-uid/--backends")

    print(f"Selected scenes: {len(rows)}")
    models = _load_policy_models(
        args.policy, args.model_path, plant2_action_mode=args.plant2_action_mode,
    )

    out_dir = benchmark_output_dir / "policy_eval" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / f"episodes_{args.policy}.jsonl"

    # Sidecar root (expert_replay layout) — only if requested.
    replay_root: Path | None = None
    if args.emit_replay_sidecar:
        replay_root = Path(args.replay_root) if args.replay_root else (out_dir / "replays")
        replay_root.mkdir(parents=True, exist_ok=True)
        print(f"Sidecars: {replay_root}")

    # GIF output dir — only when --save-gifs is set.
    gifs_dir: Path | None = None
    if args.save_gifs:
        gifs_dir = Path(args.gif_dir) if args.gif_dir else (out_dir / "gifs")
        gifs_dir.mkdir(parents=True, exist_ok=True)
        print(f"GIFs: {gifs_dir}")

    existing_results = _load_existing_results(episodes_path)
    existing_by_key: dict[tuple[str, str, str, int], dict] = {}
    for r in existing_results:
        existing_by_key[_episode_key_from_result(r)] = r

    ego_params_manifest_path = out_dir / "ego_params_manifest.json"
    ego_params_manifest = {
        "policy": args.policy,
        "ego_variant": args.ego_variant,
        "ego_sample_seed_base": args.ego_sample_seed_base,
        "params_per_scene_uid": {},
    }

    rows_to_run, skipped = select_rows_to_run(
        rows, existing_by_key,
        rerun_failed=args.rerun_failed,
        skip_error_episodes=args.skip_error_episodes)

    print(f"Resume: loaded {len(existing_results)} existing episodes, skip {skipped}, run {len(rows_to_run)}")

    results_by_key: dict[tuple[str, str, str, int], dict] = dict(existing_by_key)
    write_mode = "a" if episodes_path.exists() else "w"
    with open(episodes_path, write_mode, encoding="utf-8") as f:
        for idx, row in enumerate(rows_to_run, start=1):
            backend = str(row["_backend"])
            scene_id = row.get("scene_id")
            sign_code = row.get("_sign_code")
            print(f"[{idx}/{len(rows_to_run)}] backend={backend} sign={sign_code} scene={scene_id}")
            try:
                gif_path = None
                if gifs_dir is not None:
                    seed_val = _row_seed(row)
                    var_idx = int(row.get("var_idx", 0) or 0)
                    uid = f"{scene_id or 'scene'}_v{var_idx}_s{seed_val}"
                    gif_path = gifs_dir / f"{uid}_{args.policy}_{args.ego_variant}.gif"
                episode_t0 = time.time()
                r = run_one_episode(
                    row=row,
                    backend=backend,
                    policy_type=args.policy,
                    models=models,
                    scenes_root=scenes_root,
                    max_steps=args.max_steps,
                    ego_variant=args.ego_variant,
                    ego_sample_seed_base=args.ego_sample_seed_base,
                    replay_root=replay_root,
                    save_gif=gif_path,
                )
                episode_dt = time.time() - episode_t0
                print(f"{args.policy}  elapsed_s={episode_dt:.3f}")
            except Exception as exc:
                r = _error_result(row, exc, backend=backend)
            results_by_key[_episode_key_from_result(r)] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()

            # Persist sampled ego IDM params per scene_uid (atomic rewrite).
            if args.ego_variant != "default" and r.get("ok") and r.get("ego_params"):
                key = ":".join(str(x) for x in _episode_key_from_result(r))
                k_idx = int(args.ego_variant[1:]) if args.ego_variant.startswith("s") else 0
                sample_seed = int(args.ego_sample_seed_base) + int(r.get("seed") or 0) + k_idx * 1000003
                ego_params_manifest["params_per_scene_uid"][key] = {
                    "sample_seed": sample_seed,
                    "params": r["ego_params"],
                }
                tmp = ego_params_manifest_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(ego_params_manifest, indent=2, default=str),
                               encoding="utf-8")
                tmp.replace(ego_params_manifest_path)

    results: list[dict] = list(results_by_key.values())
    ok_runs = sum(1 for r in results if r.get("ok"))
    print("\n=== Done ===")
    print(f"Episodes OK: {ok_runs}/{len(results)}")
    print(f"Episodes: {episodes_path}")
    print("Aggregate with: build_episode_metrics_csv.py --episodes-root "
          f"{out_dir.parent}")


if __name__ == "__main__":
    main()
