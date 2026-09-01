"""Closed-loop episode: wrap env, apply row, step, optional GIF."""

from __future__ import annotations

import json
import logging
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from traffic_bench.eval.engine.sim.metadrive_sumo_patch import apply_metadrive_sumo_via_patch
from traffic_bench.eval.engine.sim.checkpoints import (
    DEFAULT_MODEL_PATHS,
    NN_NEED_CHECKPOINT,
    PLAIN_PLANT2_POLICIES,
    PLANT2_POLICIES,
    resolve_nn_checkpoint,
)
from traffic_bench.eval.engine.sim.top_down_text_patch import apply_top_down_violations_text_patch
from traffic_bench.eval.engine.sim.top_down_path_conflict_patch import (
    apply_top_down_path_conflict_overlay_patch,
    is_path_conflict_overlay_enabled,
    set_path_conflict_overlay_enabled,
)
from traffic_bench.eval.engine.sim.top_down_local_film_patch import apply_top_down_local_film_patch
from traffic_bench.eval.engine.sim.sign_eval import (
    _ego_at_fault_for_crash,
    _ego_in_sign_zone,
    _extract_sign_info,
    _format_violation,
    _violation_bucket,
)

apply_metadrive_sumo_via_patch()
apply_top_down_violations_text_patch()
apply_top_down_path_conflict_overlay_patch()
apply_top_down_local_film_patch()

from traffic_bench.envs.sumo import TrafficSignSumoEnv
from traffic_bench.envs.traffic import SumoTrafficManager
from traffic_bench.agents.idm_rule import ComprehensiveRuleExpertPolicy
from traffic_bench.agents.ppo_rule import RuleCompliantExpertPolicy
from metadrive.policy.idm_policy import ModifiedIDMPolicy
from metadrive.policy.expert_policy import ExpertPolicy
from traffic_bench.eval.engine.traffic.ego_defaults import (
    apply_ego_defaults,
    apply_ego_sampled,
    numpy_legacy_seed,
    sample_ego_params,
)
from traffic_bench.signs.junction import (
    MainRoadSign,
    YieldSign,
)
from traffic_bench.eval.engine.map.lane_keys import clamp_lane_key_to_graph, lane_edge_id, make_lane_key
from traffic_bench.eval.signs.dual_path.nav import (
    OneWaySumoTrafficManager,
    install_one_way_compliant_nav_route,
    resolve_row_background_excluded_edges,
)
from traffic_bench.eval.run.place import place_signs_for_row
from traffic_bench.eval.signs.blocked.place import (
    ego_compliant_stop_before_blocked_road,
    row_is_blocked_road as _row_is_blocked_road,
)
from traffic_bench.eval.signs.dual_path.place import (
    COMPLIANT_NAV_POLICIES,
    resolve_row_for_policy,
    row_is_one_way as _row_is_one_way,
    row_uses_dual_path_nav as _row_uses_dual_path_nav,
)
from traffic_bench.eval.signs.junction.place import (
    place_right_hand_yield_tracker,
    row_is_secondary_road as _row_is_secondary_road,
    row_is_stop as _row_is_stop,
    row_is_yield as _row_is_yield,
)
from traffic_bench.eval.signs.roundabout.place import (
    layout_from_row as roundabout_layout_from_row,
    row_is_roundabout as _row_is_roundabout,
)
from traffic_bench.eval.signs.crosswalk.place import (
    install_segment_crosswalk_geometry,
    row_is_crosswalk as _row_is_crosswalk,
)
from traffic_bench.eval.signs.detour.place import row_is_detour as _row_is_detour
from traffic_bench.eval.signs.speed.place import row_is_speed as _row_is_speed
from traffic_bench.eval.engine.spawn.auxiliary_agent import (
    DEFAULT_CONVOY_GAP_M,
    DEFAULT_CONVOY_SIZE,
    DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END,
    DEFAULT_SPAWN_VELOCITY_MS,
    add_auxiliary_agents,
    resolve_aux_spawn_plan,
)
from traffic_bench.eval.engine.expand.manifest_config import (
    DEFAULT_AUX_LANES_OCCUPIED_MAX,
    DEFAULT_COMPLIANT_STOP_MAX_DIST_M,
    DEFAULT_COMPLIANT_STOP_SPEED_MPS,
    DEFAULT_COMPLIANT_STOP_SUCCESS_SECONDS,
    DEFAULT_DESTINATION_MAX_ALONG_M,
)
from traffic_bench.eval.engine.map.junction_priority_layout import (
    JunctionLayoutError,
    build_junction_priority_layout,
)
from traffic_bench.eval.engine.expand.manifest_config import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_STOP_WAIT_STEPS,
    enrich_manifest_row,
    load_manifest_config,
)

from traffic_bench.eval.run.env import (
    _apply_destination_along_cap,
    _apply_manifest_ego_destination,
    _apply_manifest_ego_spawn_lane,
    _apply_manifest_ego_spawn_velocity,
    _apply_manifest_profile_to_npcs,
    _analyze_junction_lanes,
    _build_sumo_env,
    _ego_reached_capped_destination,
    _reposition_ego_before_lane_end,
    _resolve_sign_spawn_distance,
    _wrap_for_policy,
    _manifest_horizon,
)
from traffic_bench.eval.run.gif import _topdown_gif_film_and_scaling
from traffic_bench.eval.run.policy import _load_policy_models, resolve_model_path
from traffic_bench.eval.run.score import (
    _compute_smoothness,
    _infraction_penalty,
    _is_aux_in_main_zone,
    _is_ego_in_yield_zone,
    _min_ttc_seconds,
    _nearby_speed_percentage,
    _route_completion_percent,
    _route_length_m,
    _safe_float,
    _unwrap_base_env,
    aggregate_results,
)

EVAL_DIR = Path(__file__).resolve().parent.parent
PDD_BENCH_DIR = EVAL_DIR.parent
SDC_ROOT = PDD_BENCH_DIR.parent


def _row_is_main_secondary(row: dict) -> bool:
    """Yield / stop / secondary / roundabout already carry a yield-style tracker."""
    return (
        _row_is_yield(row)
        or _row_is_stop(row)
        or _row_is_secondary_road(row)
        or _row_is_roundabout(row)
    )


def _slug_to_code(slug: str) -> str:
    return slug.replace("_", ".")



def _load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _load_enriched_manifest_rows(path: Path) -> list[dict]:
    config = load_manifest_config(path)
    return [enrich_manifest_row(row, config) for row in _load_jsonl_rows(path)]


def _choose_manifest(code_dir: Path) -> Path | None:
    """Find real_manifest.jsonl for SUMO scenes."""
    p1 = code_dir / "sumo" / "sumo_manifest.jsonl"
    p2 = code_dir / "real_manifest.jsonl"
    if p1.exists() and p1.stat().st_size > 0:
        return p1
    if p2.exists() and p2.stat().st_size > 0:
        return p2
    return None


def collect_rows(
    benchmark_output_dir: Path,
    only_codes: set[str],
    max_scenes_per_sign: int | None,
    unique_scene_id: bool = False,
) -> list[dict]:
    """Iterate manifests and collect rows for evaluation (SUMO/real maps only)."""
    rows: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    seen_scene_ids: set[str] = set()

    sign_dirs = sorted([d for d in benchmark_output_dir.iterdir() if d.is_dir() and d.name[:1].isdigit()])
    for sign_dir in sign_dirs:
        sign_code = _slug_to_code(sign_dir.name)
        if only_codes and sign_code not in only_codes:
            continue

        manifest = _choose_manifest(sign_dir)
        if manifest is None:
            continue

        config = load_manifest_config(manifest)
        for row in _load_jsonl_rows(manifest):
            row = enrich_manifest_row(row, config)
            if "valid" in row and not row["valid"]:
                continue
            if unique_scene_id:
                sid_key = str(row.get("scene_id") or "")
                if sid_key in seen_scene_ids:
                    continue
                seen_scene_ids.add(sid_key)
            if max_scenes_per_sign is not None and counts[sign_code] >= max_scenes_per_sign:
                continue
            row["_backend"] = "sumo"
            row["_sign_code"] = sign_code
            rows.append(row)
            counts[sign_code] += 1

    return rows







def run_one_episode(
    row: dict,
    policy_type: str,
    models: dict,
    scenes_root: Path,
    max_steps: int,
    ego_variant: str,
    ego_sample_seed_base: int,
    replay_root: Path | None = None,
    save_gif: Path | None = None,
    gif_window_m: float = 80.0,
    hide_signs: bool = False,
    draw_path_conflict: bool = False,
    auxiliary_agent: bool = False,
    aux_distance_from_intersection: float = DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    aux_policy: str = "idm",
    aux_spawn_velocity_ms: float = DEFAULT_SPAWN_VELOCITY_MS,
    aux_release_when_ego_within_m: float = DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END,
    aux_convoy_size: int = DEFAULT_CONVOY_SIZE,
    aux_convoy_gap_m: float = DEFAULT_CONVOY_GAP_M,
    aux_lanes_occupied: int = DEFAULT_AUX_LANES_OCCUPIED_MAX,
    record_episode: bool = False,
    replay_layout: str = "legacy",
) -> dict:
    seed = int(row.get("seed") or row.get("deterministic_seed") or 0)
    set_path_conflict_overlay_enabled(bool(draw_path_conflict) and save_gif is not None)
    np.random.seed(numpy_legacy_seed(seed))
    random.seed(seed)
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    env = _build_sumo_env(row, scenes_root=scenes_root, max_steps=max_steps)

    raw_env = env
    env = _wrap_for_policy(env, policy_type)

    if record_episode:
        # Shared patch: tolerate post-reset sign/aux spawn before first step.
        from traffic_bench.eval.engine.sim.record_manager_patch import patch_record_manager_once
        patch_record_manager_once()
        # RecordManager reads this off global_config on reset.
        try:
            raw_env.config["record_episode"] = True
        except Exception:
            pass
        try:
            env.config["record_episode"] = True
        except Exception:
            pass

    policy_cls = None
    if policy_type == "idm":
        policy_cls = ModifiedIDMPolicy  # Good driving, no sign compliance
    elif policy_type == "comprehensive_rule_expert":
        policy_cls = ComprehensiveRuleExpertPolicy
    elif policy_type == "rule_compliant":
        policy_cls = RuleCompliantExpertPolicy
    elif policy_type == "ppo_lidar":
        policy_cls = ExpertPolicy
    elif policy_type in ("carl", "carl_rule") or policy_type in PLANT2_POLICIES:
        policy_cls = models.get("policy_cls")
        if policy_cls is None:
            raise RuntimeError(f"policy_cls for --policy {policy_type} not loaded; "
                               "check _load_policy_models")

    try:
        if record_episode:
            from traffic_bench.eval.engine.sim.record_manager_patch import patch_record_manager_once
            patch_record_manager_once()

        def _row_int(key: str, default: int = 0) -> int:
            val = row.get(key, default)
            try:
                return int(val)
            except (TypeError, ValueError):
                return default

        env_seed = (_row_int("sign_id") + _row_int("var_idx")) % 100000
        obs, info = env.reset(seed=env_seed)
        base_env = _unwrap_base_env(env)
        try:
            if hasattr(base_env, "engine") and hasattr(base_env.engine, "np_random"):
                base_env.engine.np_random = np.random.RandomState(seed)
        except Exception:
            pass

        # Manifest spawn lane + distance before intersection
        if _row_uses_dual_path_nav(row):
            row = resolve_row_for_policy(row, policy_type)
        _apply_manifest_ego_spawn_lane(base_env, row)
        spawn_distance = float(row.get("spawn_distance_before_end", 0) or 0)
        if spawn_distance > 0:
            _reposition_ego_before_lane_end(base_env, spawn_distance)
        if _row_is_speed(row):
            _apply_manifest_ego_spawn_velocity(base_env, row)
        _apply_manifest_ego_destination(base_env, row)
        if _row_is_detour(row) or _row_is_speed(row):
            _apply_destination_along_cap(base_env, row)
        install_segment_crosswalk_geometry(base_env, row)

        # Dual-path dest is already policy-resolved (truncated finish).
        # Plain baselines: MetaDrive unrestricted set_route — short violating
        # path is a property of the map, we do not pin or block anything.
        # Rule-compliant: rebuild with the forbidden branch blocked.
        if _row_uses_dual_path_nav(row):
            if policy_type in COMPLIANT_NAV_POLICIES:
                install_one_way_compliant_nav_route(base_env, row)
            else:
                nav = getattr(base_env.vehicle, "navigation", None)
                n_ck = len(getattr(nav, "checkpoints", None) or [])
                print(
                    f"[DualPathNav] kept MetaDrive default route "
                    f"({n_ck} checkpoints) for {policy_type}"
                )
            _apply_destination_along_cap(base_env, row)

        # Validate route: check that destination is different from spawn.
        # Detour finishes on the same obstacle edge (along-cap), so skip this.
        nav = getattr(base_env.vehicle, "navigation", None)
        if nav is not None and not _row_is_detour(row) and not _row_is_speed(row):
            checkpoints = getattr(nav, "checkpoints", [])
            spawn_lane_idx = getattr(base_env.vehicle.lane, "index", None)
            if checkpoints and spawn_lane_idx:
                if len(checkpoints) <= 1 or checkpoints[-1] == spawn_lane_idx or checkpoints[0] == checkpoints[-1]:
                    scene_id = row.get("scene_id", "unknown")
                    dest = row.get("destination_lane_id", "unknown")
                    print(f"[RouteValidation] INVALID: {scene_id} - route loops back to spawn. "
                          f"spawn={spawn_lane_idx}, dest={dest}, checkpoints={checkpoints[:3]}...")
                    return {
                        "ok": False,
                        "error": f"Invalid route: spawn and destination are the same or unreachable",
                        "scene_id": scene_id,
                    }

        # Place MainRoadSign (+ RH yield tracker) under RecordManager guard.
        sign_distance = float(row.get("sign_distance_before_end", 20.0))
        _rm = getattr(base_env.engine, "record_manager", None) if record_episode else None
        _rm_original_add_spawn = None
        _signs_pre = set(base_env.engine._spawned_objects.keys())
        if _rm is not None:
            _rm_original_add_spawn = _rm.add_spawn_info
            _rm.add_spawn_info = lambda *a, **kw: None
        try:
            place_signs_for_row(
                base_env,
                row,
                scenes_root=scenes_root,
                distance_before_end=sign_distance,
                show_model=not hide_signs,
            )
            if (
                not _row_is_main_secondary(row)
                and not _row_is_blocked_road(row)
                and not _row_uses_dual_path_nav(row)
                and not _row_is_crosswalk(row)
                and not _row_is_detour(row)
                and not _row_is_speed(row)
            ):
                place_right_hand_yield_tracker(
                    base_env,
                    row,
                    scenes_root=scenes_root,
                    distance_before_end=sign_distance,
                )
        finally:
            if record_episode:
                _signs_post = set(base_env.engine._spawned_objects.keys())
                for _sid in _signs_post - _signs_pre:
                    obj = base_env.engine._spawned_objects.get(_sid)
                    body = getattr(obj, "_body", None) if obj is not None else None
                    if obj is not None and body is None:
                        base_env.engine._spawned_objects.pop(_sid, None)
                if _rm is not None and _rm_original_add_spawn is not None:
                    _rm.add_spawn_info = _rm_original_add_spawn

        # Analyze and print junction lanes (for debugging/info only)
        incoming_lanes, outgoing_lanes = _analyze_junction_lanes(base_env)

        policy_obj = None
        sampled_ego_params = None
        if policy_cls is not None:
            policy_obj = policy_cls(base_env.vehicle, seed)
            if policy_type in ("idm", "comprehensive_rule_expert"):
                if ego_variant == "default":
                    apply_ego_defaults(policy_obj)
                elif ego_variant.startswith("s") and ego_variant[1:].isdigit():
                    k = int(ego_variant[1:])
                    sample_seed = int(ego_sample_seed_base) + int(seed) + k * 1000003
                    sampled_ego_params = sample_ego_params(sample_seed)
                    apply_ego_sampled(policy_obj, sampled_ego_params)
                else:
                    apply_ego_defaults(policy_obj)
            if hasattr(policy_obj, "STOP_WAIT_STEPS"):
                policy_obj.STOP_WAIT_STEPS = int(
                    row.get("stop_wait_steps", DEFAULT_STOP_WAIT_STEPS)
                )

        # Add auxiliary agents on every incoming lane (except ego's road)
        # aux_agent_mgr = None
        # if auxiliary_agent:
        #     ego_lane_index = getattr(base_env.vehicle.lane, "index", "")
        #     aux_spawn_lanes = [
        #         lane["lane_name"]
        #         for lane in incoming_lanes
        #         # if lane["edge_id"] not in ego_lane_index
        #     ]
        #     if aux_spawn_lanes:
        #         aux_agent_mgr = add_auxiliary_agents(
        #             base_env,
        #             spawn_lane_indices=aux_spawn_lanes,
        #             distance_from_intersection=aux_distance_from_intersection,
        #         )
        #         print(f"[AuxAgent] Spawned on {len(aux_spawn_lanes)} incoming lane(s)")
        #     else:
        #         print("[AuxAgent] No incoming lanes available for auxiliary agents")

        aux_agent_mgr = None
        aux_spawn_lanes: list[str] = []
        want_aux = bool(row.get("auxiliary_agent", auxiliary_agent))
        if want_aux and not _row_is_blocked_road(row) and not _row_uses_dual_path_nav(row):
            aux_distance_from_intersection = float(
                row.get("aux_distance_from_intersection", aux_distance_from_intersection)
            )
            aux_spawn_velocity_ms = float(
                row.get("aux_spawn_velocity_ms", aux_spawn_velocity_ms)
            )
            aux_convoy_size = int(row.get("aux_convoy_size", aux_convoy_size))
            aux_convoy_gap_m = float(row.get("aux_convoy_gap_m", aux_convoy_gap_m))
            aux_lanes_occupied = int(row.get("aux_lanes_occupied", aux_lanes_occupied))
            ego_lane_index = (
                make_lane_key(str(row.get("road_id")), int(row.get("spawn_lane_num", 0) or 0))
                if row.get("road_id")
                else str(getattr(base_env.vehicle.lane, "index", ""))
            )

            aux_spawn_lanes, aux_destination_lanes, alternate_spawn_dest_map, aux_spawn_longs, ring_circulate = (
                resolve_aux_spawn_plan(
                    row,
                    ego_lane_index=str(ego_lane_index),
                    incoming_lanes=incoming_lanes,
                    aux_lanes_occupied=aux_lanes_occupied,
                    aux_distance_from_intersection=aux_distance_from_intersection,
                    scenes_root=scenes_root,
                )
            )
            aux_destination_lanes = [
                dest or None for dest in aux_destination_lanes
            ]

            # Keep release distance >= ego spawn offset so aux is not held while a
            # yielding ego freezes outside the release radius.
            ego_spawn_before_end = float(row.get("spawn_distance_before_end", 0) or 0)
            release_before_end = float(aux_release_when_ego_within_m)
            if release_before_end > 0 and ego_spawn_before_end > 0:
                release_before_end = max(release_before_end, ego_spawn_before_end)

            if aux_spawn_lanes:
                aux_agent_mgr = add_auxiliary_agents(
                    base_env,
                    spawn_lane_indices=aux_spawn_lanes,
                    outgoing_lanes=outgoing_lanes,
                    distance_from_intersection=aux_distance_from_intersection,
                    policy=aux_policy,
                    spawn_velocity_ms=aux_spawn_velocity_ms,
                    destination_lanes=aux_destination_lanes,
                    ego_vehicle=base_env.vehicle,
                    ego_spawn_lane_index=ego_lane_index,
                    ego_release_distance_before_end=release_before_end,
                    convoy_size=aux_convoy_size,
                    convoy_gap_m=aux_convoy_gap_m,
                    alternate_spawn_dest_map=alternate_spawn_dest_map,
                    spawn_longitudinal_by_lane=aux_spawn_longs,
                    ring_circulate_by_lane=ring_circulate,
                    junction_layout=row.get("junction_layout"),
                )
                if aux_agent_mgr is not None:
                    print(
                        f"[AuxAgent] lanes={len(aux_spawn_lanes)}, "
                        f"convoy_size={aux_convoy_size}, gap={aux_convoy_gap_m}m, "
                        f"spawned={aux_agent_mgr.get_status().get('count', 0)}"
                    )

        total_reward = 0.0
        violations = 0
        sign_violations = 0
        traffic_light_violations = 0
        crosswalk_violations = 0
        violations_by_class_step: dict[str, int] = {}
        in_zone_total_steps = 0
        in_zone_by_class_step: dict[str, int] = {}
        violations_event_count = 0
        violations_by_class_event: dict[str, int] = {}
        violations_timeline: list[dict] = []
        prev_violated_class_names: set = set()
        steps = 0
        crashed = False
        reached_dest = False
        out_of_road = False
        last_violation_texts = []
        violation_text_ttl = 0
        speed_pct_samples: list[float] = []
        min_ttc: float | None = None
        abs_lane_offsets: list[float] = []
        steer_delta_abs: list[float] = []
        hard_brake_count = 0
        hard_accel_count = 0
        distance_travelled_m = 0.0
        visited_lane_lengths: dict[str, float] = {}

        dt = 0.1
        prev_speed_mps: float | None = None
        prev_heading: float | None = None
        prev_long_acc: float | None = None
        prev_lat_acc: float | None = None
        prev_yaw_rate: float | None = None
        prev_action_steer: float | None = None
        smoothness_step_vars: list[dict] = []
        expert_actions: list[list[float]] = []
        sign_info_snapshot = _extract_sign_info(base_env)
        last_info: dict = {}

        is_blocked_road_row = _row_is_blocked_road(row)
        compliant_stop_steps = 0
        compliant_stop_success = False
        stop_success_s = float(
            row.get(
                "compliant_stop_success_seconds",
                DEFAULT_COMPLIANT_STOP_SUCCESS_SECONDS,
            )
            or DEFAULT_COMPLIANT_STOP_SUCCESS_SECONDS
        )
        stop_success_steps = max(1, int(round(stop_success_s / dt)))
        stop_max_dist_m = float(
            row.get("compliant_stop_max_dist_m", DEFAULT_COMPLIANT_STOP_MAX_DIST_M)
            or DEFAULT_COMPLIANT_STOP_MAX_DIST_M
        )
        stop_speed_max = float(
            row.get("compliant_stop_speed_mps", DEFAULT_COMPLIANT_STOP_SPEED_MPS)
            or DEFAULT_COMPLIANT_STOP_SPEED_MPS
        )

        # Re-apply after spawn/signs/policy — nav may have been rebuilt.
        _apply_destination_along_cap(base_env, row)

        episode_horizon = _manifest_horizon(row, max_steps)
        for step in range(episode_horizon):
            if policy_obj is not None:
                action = policy_obj.act(base_env.vehicle.name)
            else:
                action = [0.0, 0.0]
            try:
                expert_actions.append([float(action[0]), float(action[1])])
            except Exception:
                expert_actions.append([0.0, 0.0])

            obs, reward, terminated, truncated, info = env.step(action)

            last_info = info
            total_reward += float(reward)
            steps += 1

            vehicle = base_env.agent
            sign_mgr = getattr(base_env.engine, "traffic_sign_manager", None)
            current_violation_texts = []
            current_violated_class_names: set = set()
            current_violations = []
            if sign_mgr is not None and vehicle is not None:
                step_in_any_zone = False
                for _s in sign_mgr.signs:
                    if _ego_in_sign_zone(_s, vehicle):
                        step_in_any_zone = True
                        cls = type(_s).__name__
                        in_zone_by_class_step[cls] = in_zone_by_class_step.get(cls, 0) + 1
                for rule in getattr(sign_mgr, "rules", []):
                    if type(rule).__name__ != "PedestrianYieldRule":
                        continue
                    try:
                        ped_status = rule.get_status(vehicle)
                    except Exception:
                        continue
                    if (
                        ped_status.get("in_yield_zone")
                        or ped_status.get("in_crosswalk")
                        or ped_status.get("in_no_stop_zone")
                    ):
                        step_in_any_zone = True
                        in_zone_by_class_step["PedestrianYieldRule"] = (
                            in_zone_by_class_step.get("PedestrianYieldRule", 0) + 1
                        )
                if step_in_any_zone:
                    in_zone_total_steps += 1

                for sign in sign_mgr.signs:
                    if sign._is_violating(vehicle):
                        sign_violations += 1
                        violations += 1

                current_violations = sign_mgr.check_all_violations(vehicle)
                for _sign, violated in current_violations:
                    if violated:
                        violations += 1
                        bucket = _violation_bucket(_sign)
                        if bucket == "traffic_light":
                            traffic_light_violations += 1
                        elif bucket == "crosswalk":
                            crosswalk_violations += 1
                        else:
                            sign_violations += 1
                        current_violation_texts.append(_format_violation(_sign, vehicle))
                        cls_name = type(_sign).__name__
                        violations_by_class_step[cls_name] = (
                            violations_by_class_step.get(cls_name, 0) + 1)
                        current_violated_class_names.add(cls_name)
                        if cls_name not in prev_violated_class_names:
                            violations_event_count += 1
                            violations_by_class_event[cls_name] = (
                                violations_by_class_event.get(cls_name, 0) + 1)
                            try:
                                rule = _sign.get_rule_description() or ""
                            except Exception:
                                rule = ""
                            violations_timeline.append({
                                "step": int(step),
                                "sign_class": cls_name,
                                "rule": rule,
                            })
                prev_violated_class_names = current_violated_class_names
                if current_violation_texts:
                    last_violation_texts = current_violation_texts[:3]
                    violation_text_ttl = 40
                elif violation_text_ttl > 0:
                    violation_text_ttl -= 1
                else:
                    last_violation_texts = []

            if vehicle is not None:
                sp = _nearby_speed_percentage(vehicle)
                if sp is not None:
                    speed_pct_samples.append(float(sp))

            if vehicle is not None:
                speed_mps = _safe_float(getattr(vehicle, "speed", 0.0), 0.0)
                distance_travelled_m += max(0.0, speed_mps) * dt
                heading = _safe_float(getattr(vehicle, "heading_theta", 0.0), 0.0)

                lane = getattr(vehicle, "lane", None)
                if lane is not None:
                    try:
                        _long, lat = lane.local_coordinates(vehicle.position)
                        abs_lane_offsets.append(abs(float(lat)))
                    except Exception:
                        pass
                    try:
                        lane_idx = getattr(lane, "index", None)
                        lane_key = repr(lane_idx) if lane_idx is not None else f"lane_obj_{id(lane)}"
                        lane_len = float(getattr(lane, "length", 0.0) or 0.0)
                        if lane_len > 0.0 and lane_key not in visited_lane_lengths:
                            visited_lane_lengths[lane_key] = lane_len
                    except Exception:
                        pass

                step_ttc = _min_ttc_seconds(vehicle)
                if step_ttc is not None and step_ttc > 0.0:
                    min_ttc = step_ttc if min_ttc is None else min(min_ttc, step_ttc)

                cur_steer = _safe_float(action[0], 0.0)
                if prev_action_steer is not None:
                    steer_delta_abs.append(abs(cur_steer - prev_action_steer))
                prev_action_steer = cur_steer

                if prev_speed_mps is not None and prev_heading is not None:
                    long_acc = (speed_mps - prev_speed_mps) / dt
                    yaw_delta = math.atan2(math.sin(heading - prev_heading), math.cos(heading - prev_heading))
                    yaw_rate = yaw_delta / dt
                    lat_acc = speed_mps * yaw_rate

                    if long_acc < -3.0:
                        hard_brake_count += 1
                    if long_acc > 2.5:
                        hard_accel_count += 1

                    if prev_long_acc is not None and prev_lat_acc is not None and prev_yaw_rate is not None:
                        long_jerk = (long_acc - prev_long_acc) / dt
                        lat_jerk = (lat_acc - prev_lat_acc) / dt
                        yaw_acc = (yaw_rate - prev_yaw_rate) / dt
                        jerk_mag = float(math.sqrt(long_jerk * long_jerk + lat_jerk * lat_jerk))
                        smoothness_step_vars.append(
                            {
                                "long_acc": float(long_acc),
                                "lat_acc": float(lat_acc),
                                "yaw_rate": float(yaw_rate),
                                "yaw_acc": float(yaw_acc),
                                "long_jerk": float(long_jerk),
                                "jerk_mag": jerk_mag,
                            }
                        )

                    prev_long_acc = long_acc
                    prev_lat_acc = lat_acc
                    prev_yaw_rate = yaw_rate

                prev_speed_mps = speed_mps
                prev_heading = heading

            # Finish: compliant stop (3.2 success) and/or capped dest (3.2 / 4.3).
            # Violation on 3.2 is recorded by NoTrafficSign when ego passes the sign.
            dest_cap_m = 0.0
            capped_arrive = False
            if is_blocked_road_row:
                if sign_violations == 0 and ego_compliant_stop_before_blocked_road(
                    base_env,
                    vehicle,
                    max_dist_before_sign_m=stop_max_dist_m,
                    speed_max_mps=stop_speed_max,
                ):
                    compliant_stop_steps += 1
                    if compliant_stop_steps >= stop_success_steps:
                        capped_arrive = True
                        compliant_stop_success = True
                        print(
                            f"[NoTrafficSign] Compliant stop for {stop_success_s:.1f}s "
                            f"before sign → arrive_dest (step={steps})"
                        )
                else:
                    compliant_stop_steps = 0

            if (
                row.get("destination_max_along_m") is not None
                or is_blocked_road_row
                or _row_is_roundabout(row)
                or _row_uses_dual_path_nav(row)
                or _row_is_crosswalk(row)
                or _row_is_detour(row)
                or _row_is_speed(row)
            ):
                raw_cap = row.get("destination_max_along_m")
                if raw_cap is None:
                    dest_cap_m = float(DEFAULT_DESTINATION_MAX_ALONG_M)
                else:
                    dest_cap_m = float(raw_cap or 0.0)
                stored = getattr(vehicle, "_priority_bench_dest_along_m", None)
                if stored is not None:
                    try:
                        dest_cap_m = float(stored)
                    except (TypeError, ValueError):
                        pass
            capped_arrive = capped_arrive or _ego_reached_capped_destination(
                vehicle,
                max_along_m=dest_cap_m,
                allow_same_lane=_row_is_detour(row) or _row_is_speed(row),
            )
            natural_done = bool(terminated or truncated)

            text_dict: dict = {}
            if save_gif:
                aux_vehicles = []
                if aux_agent_mgr is not None:
                    try:
                        aux_vehicles = list(aux_agent_mgr.auxiliary_vehicles)
                    except Exception:
                        aux_vehicles = []
                text_dict = {
                    "Step": step,
                    "Speed": f"{vehicle.speed_km_h:.2f} km/h" if vehicle else "n/a",
                    "Violations": sign_violations + crosswalk_violations,
                    "is_aux_in_main_zone": _is_aux_in_main_zone(
                        sign_mgr, aux_vehicles, ego_vehicle=vehicle
                    ),
                    "is_ego_in_yield_zone": _is_ego_in_yield_zone(sign_mgr, vehicle),
                }
                if draw_path_conflict or is_path_conflict_overlay_enabled():
                    text_dict["paths"] = "cyan=ego magenta=auxX yellow=X amber=zone"

            # Render before breaking so arrive/terminate frames are in the GIF.
            if save_gif:
                try:
                    screen_size = (800, 800)
                    film_size, scaling = _topdown_gif_film_and_scaling(
                        base_env,
                        screen_size=screen_size,
                        window_m=gif_window_m,
                    )
                    base_env.render(
                        mode="top_down",
                        film_size=film_size,
                        scaling=float(scaling),
                        screen_size=screen_size,
                        semantic_map=True,
                        semantic_broken_line=True,
                        draw_target_vehicle_trajectory=True,
                        target_agent_heading_up=True,
                        screen_record=True, window=False,
                        text=text_dict,
                    )
                except Exception:
                    pass

            if capped_arrive:
                reached_dest = True
                info["arrive_dest"] = True
                last_info = info
                break

            if natural_done:
                reached_dest = bool(info.get("arrive_dest", False))
                out_of_road = bool(info.get("out_of_road", False))
                crashed = bool(info.get("crash", False) or out_of_road)
                break

        route_completion_pct = _route_completion_percent(last_info, reached_dest)
        infraction_penalty = _infraction_penalty(crashed=crashed, out_of_road=out_of_road, violations=violations)
        driving_score = route_completion_pct * infraction_penalty
        driving_efficiency = float(np.mean(speed_pct_samples)) if speed_pct_samples else 0.0
        smoothness = _compute_smoothness(smoothness_step_vars, segment_len=20)
        route_length_m = _route_length_m(last_info)
        route_length_source = "info"
        if route_length_m is None:
            approx = float(sum(visited_lane_lengths.values()))
            if approx > 0.0:
                route_length_m = approx
                route_length_source = "visited_lanes"
            else:
                route_length_source = "none"

        crashed_flag_raw = bool(last_info.get("crash", False)) if last_info else False
        crash_attribution = None
        if crashed_flag_raw or bool(getattr(base_env.agent, "crash_vehicle", False)):
            try:
                crash_attribution = "ego" if _ego_at_fault_for_crash(base_env.agent, base_env.engine) else "npc"
            except Exception:
                crash_attribution = None

        pkl_path_str: str | None = None
        dump_error: str | None = None

        scene_uid = scene_uid_from_row(row, seed)
        if replay_root is not None:
            try:
                _sign_for_path = (row.get("_sign_code") or row.get("sign_code")
                                  or row.get("pdd_code") or row.get("sign_type") or "")
                sign_slug = str(_sign_for_path).replace(".", "_")
                expert_subdir = f"{policy_type}_{ego_variant}" if ego_variant else policy_type

                if replay_layout == "flat":
                    out_replay = (
                        Path(replay_root) / "by_scene" / scene_uid / expert_subdir
                    )
                else:
                    out_replay = (
                        Path(replay_root) / sign_slug / "by_sign" / sign_slug
                        / "by_scene" / scene_uid / expert_subdir
                    )
                out_replay.mkdir(parents=True, exist_ok=True)
                sidecar_path = out_replay / "replay.json"
                output_pkl = out_replay / "replay.pkl"

                if record_episode:
                    scenario_desc = None
                    try:
                        from metadrive.scenario.utils import (
                            convert_recorded_scenario_exported,
                        )
                        raw_frames = base_env.engine.record_manager.episode_info
                        scenario_desc = convert_recorded_scenario_exported(
                            raw_frames, to_dict=True
                        )
                    except Exception:
                        scenario_desc = None
                    if scenario_desc is not None:
                        with open(output_pkl, "wb") as f:
                            pickle.dump(scenario_desc, f)
                        pkl_path_str = str(output_pkl)
                    else:
                        try:
                            base_env.engine.dump_episode(str(output_pkl))
                            if output_pkl.is_file() and output_pkl.stat().st_size > 0:
                                pkl_path_str = str(output_pkl)
                            else:
                                dump_error = "dump_episode wrote empty file"
                        except Exception as exc:
                            dump_error = f"dump_episode: {type(exc).__name__}: {exc}"
                            pkl_path_str = None

                sidecar_metrics = {
                    "arrived_dest": bool(reached_dest),
                    "crashed": crashed_flag_raw,
                    "crash_attribution": crash_attribution,
                    "crashed_ego_fault": bool(crashed_flag_raw and crash_attribution == "ego"),
                    "crashed_npc_fault": bool(crashed_flag_raw and crash_attribution == "npc"),
                    "out_of_road": bool(out_of_road),
                    "final_step": int(steps),
                    "total_violations": int(violations),
                    "violations_by_class": {
                        "sign": int(sign_violations),
                        "traffic_light": int(traffic_light_violations),
                        "crosswalk": int(crosswalk_violations),
                    },
                    "violations_by_class_step": dict(violations_by_class_step),
                    "in_zone_total_steps": int(in_zone_total_steps),
                    "in_zone_by_class_step": dict(in_zone_by_class_step),
                    "violations_event_count": int(violations_event_count),
                    "violations_by_class_event": dict(violations_by_class_event),
                    "violations_timeline": list(violations_timeline),
                    "route_completion": (float(route_completion_pct) / 100.0
                                          if route_completion_pct else 0.0),
                    "total_reward": round(float(total_reward), 4),
                    "smoothness_ratio": smoothness["smoothness_ratio"],
                    "frame_smooth_ratio": smoothness["frame_smooth_ratio"],
                    "smooth_segments": smoothness["smooth_segments"],
                    "total_segments": smoothness["total_segments"],
                    "driving_score": float(driving_score),
                    "driving_efficiency": float(driving_efficiency),
                    "infraction_penalty": float(infraction_penalty),
                    "min_ttc_sec": float(min_ttc) if min_ttc is not None else None,
                    "mean_abs_lane_offset": (float(np.mean(abs_lane_offsets))
                                              if abs_lane_offsets else None),
                    "mean_abs_steer_delta": (float(np.mean(steer_delta_abs))
                                              if steer_delta_abs else None),
                    "hard_brake_count": int(hard_brake_count),
                    "hard_accel_count": int(hard_accel_count),
                    "route_length_m": (float(route_length_m)
                                        if route_length_m is not None else None),
                    "distance_travelled_m": float(distance_travelled_m),
                    "success": (
                        bool(compliant_stop_success and not crashed_flag_raw and not out_of_road)
                        if is_blocked_road_row
                        else bool(reached_dest and not crashed_flag_raw and not out_of_road)
                    ),
                }
                sidecar = {
                    "scene_id": row.get("scene_id") or f"scene_{seed}",
                    "scene_uid": scene_uid,
                    "backend": "sumo",
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
                        "horizon": episode_horizon,
                        "seed": seed,
                    },
                    "signs": sign_info_snapshot,
                    "expert_actions": expert_actions,
                    "smoothness_step_vars": smoothness_step_vars,
                    "metrics": sidecar_metrics,
                    "ego_idm_params": (sampled_ego_params if sampled_ego_params is not None
                                        else "DEFAULT_EGO_PARAMS"),
                    "pkl_path": pkl_path_str,
                    "sidecar_path": str(sidecar_path),
                    "valid": True,
                }
                if dump_error:
                    sidecar["dump_error"] = dump_error
                with open(sidecar_path, "w", encoding="utf-8") as _sf:
                    json.dump(sidecar, _sf, default=str)
            except Exception:
                pass

        return {
            "ok": True,
            "backend": "sumo",
            "scene_id": row.get("scene_id"),
            "scene_uid": scene_uid,
            "policy": policy_type,
            "sign_type": row.get("_sign_code") or row.get("sign_code") or row.get("pdd_code") or row.get("sign_type"),
            "seed": seed,
            "total_reward": total_reward,
            "steps": steps,
            "violations": violations,
            "sign_violations": int(sign_violations),
            "traffic_light_violations": int(traffic_light_violations),
            "crosswalk_violations": int(crosswalk_violations),
            "crashed": crashed,
            "out_of_road": out_of_road,
            "reached_dest": reached_dest,
            "success": (
                bool(compliant_stop_success)
                if is_blocked_road_row
                else bool(reached_dest and not crashed)
            ),
            "route_completion_pct": route_completion_pct,
            "infraction_penalty": infraction_penalty,
            "driving_score": driving_score,
            "driving_efficiency": driving_efficiency,
            "smoothness": smoothness["smoothness_ratio"],
            "smoothness_frame_ratio": smoothness["frame_smooth_ratio"],
            "smooth_segments": smoothness["smooth_segments"],
            "smooth_total_segments": smoothness["total_segments"],
            "hard_brake_count": int(hard_brake_count),
            "hard_accel_count": int(hard_accel_count),
            "mean_abs_lane_offset": float(np.mean(abs_lane_offsets)) if abs_lane_offsets else None,
            "mean_abs_steer_delta": float(np.mean(steer_delta_abs)) if steer_delta_abs else None,
            "min_ttc_sec": float(min_ttc) if min_ttc is not None else None,
            "route_length_m": float(route_length_m) if route_length_m is not None else None,
            "route_length_source": route_length_source,
            "distance_travelled_m": float(distance_travelled_m),
            "variant": ego_variant,
            "ego_params": sampled_ego_params,
            "violations_by_class_step": dict(violations_by_class_step),
            "violations_event_count": int(violations_event_count),
            "violations_by_class_event": dict(violations_by_class_event),
            "violations_timeline": list(violations_timeline),
            "in_zone_total_steps": int(in_zone_total_steps),
            "in_zone_by_class_step": dict(in_zone_by_class_step),
            "pkl_path": pkl_path_str,
            "dump_error": dump_error,
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
        if raw_env is not env:
            try:
                raw_env.close()
            except Exception:
                pass



def scene_uid_from_row(row: dict, seed: int | None = None) -> str:
    resolved_seed = int(seed if seed is not None else row.get("seed") or row.get("deterministic_seed") or 0)
    scene_id = row.get("scene_id") or f"scene_{resolved_seed}"
    lane = int(row.get("spawn_lane_num", 0) or 0)
    var = int(row.get("var_idx", 0) or 0)
    return f"{scene_id}_lane{lane}_seed{resolved_seed}_v{var}"


def _episode_key_from_row(row: dict) -> tuple[str, str, int]:
    return (
        str(row.get("scene_id", "")),
        str(row.get("_sign_code") or row.get("sign_code") or row.get("pdd_code") or row.get("sign_type") or ""),
        int(row.get("seed") or row.get("deterministic_seed") or -1),
    )


def _episode_key_from_result(r: dict) -> tuple[str, str, int]:
    return (
        str(r.get("scene_id", "")),
        str(r.get("sign_type", "")),
        int(r.get("seed") or -1),
    )


def _load_existing_results(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows
