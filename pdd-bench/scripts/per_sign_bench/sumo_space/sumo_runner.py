"""Per-scene SUMO materialization.

Creates a fresh TrafficSignSumoEnv (with SumoTrafficManager injected) for one
scene, applies an nuPlan-derived profile, resets, optionally drives an agent
through the full episode, and returns a result dict matching the unified
manifest schema.

No 30-step "validation" rollout — that was a smoke test for PGMap/CityMap
that's not needed when we already drive the actual agent through the scene.
"""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

# Path bootstrap
SCRIPT_PATH = Path(__file__).resolve()
SUMO_DIR = SCRIPT_PATH.parent
BENCHMARK_DIR = SUMO_DIR.parent
SCRIPTS_DIR = BENCHMARK_DIR.parent
PDD_BENCH_DIR = SCRIPTS_DIR.parent
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"
SCENES_ROOT = PDD_BENCH_DIR / "scenes"
for _p in (PDD_BENCH_DIR, METADRIVE_DIR, BENCHMARK_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)


def _build_env(catalog_row: dict, profile: dict):
    """Construct an _EnvWithTraffic with the SumoTrafficManager injected."""
    from envs.sumo_env import TrafficSignSumoEnv
    from envs.sumo_traffic_manager import SumoTrafficManager

    # Reduce ego safe radius so NPCs spawn close enough to actually interact
    # with the agent. 5m = ~1 vehicle length — absolute minimum to avoid
    # immediate collision; NPCs spawn directly within the agent's lane
    # segment so traffic interactions start from step 0.
    SumoTrafficManager.EGO_SAFE_RADIUS = 15
    # Sign stays at its REAL distance_from_start — no minimum floor. For braking
    # scenes the approach runway comes from spawning ego upstream along the graph.
    sign_spawn_distance = float(catalog_row.get("sign_spawn_distance",
                                                catalog_row.get("distance_from_start", 0.0)) or 0.0)

    config = dict(
        use_render=False,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.CRITICAL,
        # net_path is relative to the scenes root the catalog was built from.
        # Honor PER_SIGN_SCENES_ROOT (set by sumo_pipeline from --scenes-root) so a
        # non-default scene set (e.g. scenes_uniq) resolves correctly; else default.
        map_name=str(Path(os.environ.get("PER_SIGN_SCENES_ROOT") or SCENES_ROOT)
                     / catalog_row["net_path"]),
        sign_type=catalog_row["sign_code"],
        sign_spawn_distance=sign_spawn_distance,
        traffic_density=float(profile["traffic_density"]),
        horizon=int(profile.get("horizon_steps", 600)),
        tl_speed_factor=float(catalog_row.get("tl_speed_factor", 20.0)),
        min_route_hops_after_spawn=int(catalog_row.get("min_route_hops_after_spawn", 2)),
        max_route_hops_after_spawn=int(catalog_row.get("max_route_hops_after_spawn", 4)),
        num_scenarios=100000,  # allow seeds in [0, 100000) for variation
        vehicle_config={"show_lidar": False},
    )
    if catalog_row.get("road_id"):
        config["vehicle_config"]["spawn_lane_index"] = catalog_row["road_id"]
    # destination_lane_id: opt-in (see expert_replay.py for why — BFS hangs
    # on some scenes). Enable via env var for experiments.
    import os as _os
    if _os.environ.get("PER_SIGN_USE_DESTINATION") == "1" and catalog_row.get("destination_lane_id"):
        config["vehicle_config"]["destination"] = catalog_row["destination_lane_id"]
    # Combined zone pair: force the destination to the end-sign edge so the route
    # passes through BOTH signs (start zone -> ... -> end sign).
    if catalog_row.get("is_paired") and catalog_row.get("destination_lane_id"):
        config["vehicle_config"]["destination"] = catalog_row["destination_lane_id"]

    # Spawn on a specific lane of the chosen road. Catalog entries for SUMO
    # enumerate all lanes per scene (§11.5); pass `spawn_lane_num` through
    # config so sumo_env's reset() teleports ego onto it.
    if "spawn_lane_num" in catalog_row:
        config["spawn_lane_num"] = int(catalog_row["spawn_lane_num"])

    # Detour (4.2.x): catalog splits scenes into cones+sign vs sign-only.
    if "detour_cones" in catalog_row:
        config["spawn_detour_cones"] = bool(catalog_row["detour_cones"])

    # Braking-spawn (3.24): ego starts above the limit, placed d_required before
    # the sign (resolved up the road graph in sumo_env). Pass the spec through.
    if catalog_row.get("braking_spawn"):
        config["ego_braking_spawn"] = True
        config["ego_spawn_mode"] = str(catalog_row.get("spawn_mode", "brake"))
        config["ego_spawn_v0_ms"] = float(catalog_row.get("spawn_velocity_ms", 0.0) or 0.0)
        config["ego_brake_d_required"] = float(catalog_row.get("d_required_m", 0.0) or 0.0)
        config["ego_v_target_kmh"] = float(catalog_row.get("v_target_kmh", 0.0) or 0.0)
        config["ego_brake_decel"] = float(catalog_row.get("brake_decel_mps2", 2.5) or 2.5)
        config["ego_brake_delay"] = float(catalog_row.get("brake_delay_s", 1.0) or 1.0)
        config["ego_brake_margin"] = float(catalog_row.get("brake_margin_m", 5.0) or 5.0)
        # Route forward through the sign toward the recorded destination.
        if catalog_row.get("destination_lane_id"):
            config["vehicle_config"]["destination"] = catalog_row["destination_lane_id"]

    class _EnvWithTraffic(TrafficSignSumoEnv):
        @classmethod
        def default_config(cls):
            cfg = super().default_config()
            cfg["traffic_density"] = 0.0
            return cfg

        def setup_engine(self):
            super().setup_engine()
            self.engine.update_manager("traffic_manager", SumoTrafficManager())

    return _EnvWithTraffic(config)


def _capture_reset_state(env) -> dict:
    """Build a 'fingerprint' from the post-reset state, used by dedup logic.

    Cheap and unique enough: start lane, navigation route, NPC count + position hash.
    Also computes the **physical route length** (sum of lane lengths from current
    lane to destination) so downstream stats can report scene duration / coverage.
    """
    fp: dict = {}
    try:
        veh = env.agent if hasattr(env, "agent") else env.vehicle
        fp["start_lane"] = str(getattr(veh, "lane_index", "?"))
        fp["start_pos"] = (round(float(veh.position[0]), 1), round(float(veh.position[1]), 1))
    except Exception:
        fp["start_lane"] = "?"
        fp["start_pos"] = (0.0, 0.0)

    try:
        nav = veh.navigation
        ckpts = list(nav.checkpoints or [])
        fp["route_checkpoints"] = [str(c) for c in ckpts]
        fp["dest_node"] = str(ckpts[-1]) if ckpts else None
        # Compute physical route length. SUMO/EdgeRoadNetwork uses
        # graph[lane_name] -> lane_info(namedtuple) where each entry has
        # `.lane.length`. PG networks use graph[start][end] -> [lanes].
        rn = env.current_map.road_network
        graph = getattr(rn, "graph", None)
        total_length = 0.0
        n_lanes = 0
        if graph is not None:
            for ck in ckpts:
                try:
                    info = graph.get(ck) if hasattr(graph, "get") else None
                    if info is not None and hasattr(info, "lane"):
                        # SUMO: lane_info namedtuple
                        ln = info.lane
                        total_length += float(getattr(ln, "length", 0.0))
                        n_lanes += 1
                except Exception:
                    continue
        fp["n_route_lanes"] = n_lanes
        fp["route_length_m"] = round(total_length, 1)
    except Exception:
        fp["route_checkpoints"] = []
        fp["dest_node"] = None
        fp["n_route_lanes"] = 0
        fp["route_length_m"] = 0.0

    try:
        tm = env.engine.traffic_manager
        traffic_vehicles = getattr(tm, "_traffic_vehicles", []) or []
        fp["n_npcs"] = len(traffic_vehicles)
        positions = []
        for v in traffic_vehicles:
            try:
                positions.append((round(float(v.position[0]), 1), round(float(v.position[1]), 1)))
            except Exception:
                pass
        fp["npc_positions"] = positions
    except Exception:
        fp["n_npcs"] = 0
        fp["npc_positions"] = []

    return fp


def materialize_sumo_scene(
    catalog_row: dict,
    density_cap: float = 1.0,
    drive_agent: bool = False,
    agent_policy: str = "comprehensive_rule_expert",
    save_gif: str | None = None,
    max_steps: int = 600,
) -> dict:
    """Materialize a single SUMO scene from a catalog row.

    Args:
        catalog_row: dict from sumo_scene_catalog.json (scene_id, net_path,
            sign_code, road_id, spawn_velocity_ms, seed, ...)
        density_cap: passed to sample_one_profile (1.0 = no clipping for SUMO)
        drive_agent: if True, drive a real agent through the scene for max_steps
            and record violations / dest_rate / crash_rate
        agent_policy: which policy to use for drive_agent
            ('comprehensive_rule_expert' | 'rule_compliant_expert' | 'idm')
        save_gif: optional path to save a top-down GIF of the episode (only used
            when drive_agent=True)
        max_steps: episode horizon when drive_agent=True

    Returns: dict with materialization result + (optionally) drive metrics.
    """
    from factorized_space.agent_profile_bank import (
        sample_one_profile, apply_profile_to_idm_class,
    )
    from factorized_space.benchmark_runner import install_npc_vehicle_type_hook
    from factorized_space.ego_defaults import apply_ego_defaults

    seed = int(catalog_row["seed"])

    # Ego spawn velocity is FIXED at 0 m/s for SUMO — IDM accelerates from rest,
    # matching the original SUMO-replay behaviour. Any `spawn_velocity_ms` in the
    # catalog row is ignored (kept for schema backward compat).
    spawn_v = 0.0

    # 1. Sample NPC profile (deterministic from seed)
    profile = sample_one_profile(
        seed, density_cap=density_cap, horizon_steps=max_steps
    )

    # 2. Apply IDM params to NPC policy class attrs (NPC-only)
    apply_profile_to_idm_class(profile)

    # 2b. Rule-compliant NPCs: cap surrounding-traffic speed to the sign limit so
    # they don't violate it (opt-in via PER_SIGN_COMPLIANT_NPC=1). SUMO NPCs use
    # SumoTrajectoryIDMPolicy, which has its OWN NORMAL_SPEED — so we must cap it
    # there (and on IDMPolicy) rather than rely on the sampled profile.
    npc_speed_cap_kmh = None
    import os as _os2
    if _os2.environ.get("PER_SIGN_COMPLIANT_NPC") == "1":
        v_lim = float(catalog_row.get("v_target_kmh", 0.0) or 0.0)
        if v_lim > 0:
            npc_speed_cap_kmh = v_lim
            try:
                from metadrive.policy.idm_policy import IDMPolicy
                from envs.sumo_idm_policy import SumoTrajectoryIDMPolicy
                for cls in (IDMPolicy, SumoTrajectoryIDMPolicy):
                    cls.NORMAL_SPEED = min(float(getattr(cls, "NORMAL_SPEED", v_lim)), v_lim)
                    cls.MAX_SPEED = v_lim
            except Exception:
                pass

    # 3. NPC vehicle type: use nuPlan-derived weights for PGTrafficManager.
    install_npc_vehicle_type_hook()

    # 3. Build env (one fresh env per scene — SumoMapManager binds to a single map)
    np.random.seed(seed)
    random.seed(seed)

    result: dict = {
        "scene_id": catalog_row["scene_id"],
        "sign_id": catalog_row["sign_id"],
        "sign_type": catalog_row["sign_code"],
        "sign_code": catalog_row["sign_code"],
        "road_id": catalog_row.get("road_id", ""),
        "net_path": catalog_row["net_path"],
        "sign_spawn_distance": float(catalog_row.get("sign_spawn_distance",
                              catalog_row.get("distance_from_start", 0.0)) or 0.0),
        "distance_from_start": catalog_row.get("distance_from_start"),
        "destination_lane_id": catalog_row.get("destination_lane_id"),
        "traffic_density": float(profile["traffic_density"]),
        "horizon_steps": int(profile.get("horizon_steps", 600)),
        "tl_speed_factor": float(catalog_row.get("tl_speed_factor", 20.0)),
        "min_route_hops_after_spawn": int(catalog_row.get("min_route_hops_after_spawn", 10)),
        "max_route_hops_after_spawn": int(catalog_row.get("max_route_hops_after_spawn", 10)),
        "latitude": catalog_row.get("latitude", 0.0),
        "longitude": catalog_row.get("longitude", 0.0),
        "spawn_velocity_ms": spawn_v,
        "v_idx": catalog_row.get("v_idx", 0),
        "var_idx": catalog_row.get("var_idx", 0),
        "spawn_lane_num": catalog_row.get("spawn_lane_num", 0),
        "deterministic_seed": seed,
        "source": "sumo",
        "npc_compliant": npc_speed_cap_kmh is not None,
        "npc_speed_cap_kmh": npc_speed_cap_kmh,
        "valid": False,
        "failure_reason": None,
    }
    for k, v in profile.items():
        result[f"profile_{k}"] = v

    # Braking-spawn (3.24): carry the full spec into the manifest so the EVAL
    # path (run_benchmark._build_sumo_env) can reconstruct the SAME upstream
    # spawn + NPC corridor restriction. The env recomputes placement
    # deterministically from (spawn_velocity_ms, d_required_m, v_target_kmh,
    # brake params, sign_s), so eval matches materialization exactly.
    if catalog_row.get("braking_spawn"):
        result["braking_spawn"] = True
        result["spawn_velocity_ms"] = float(catalog_row.get("spawn_velocity_ms", 0.0) or 0.0)
        result["d_required_m"] = float(catalog_row.get("d_required_m", 0.0) or 0.0)
        result["v_target_kmh"] = float(catalog_row.get("v_target_kmh", 0.0) or 0.0)
        result["v_target_raw_kmh"] = float(catalog_row.get("v_target_raw_kmh", 0.0) or 0.0)
        result["brake_decel_mps2"] = float(catalog_row.get("brake_decel_mps2", 2.5) or 2.5)
        result["brake_delay_s"] = float(catalog_row.get("brake_delay_s", 1.0) or 1.0)
        result["brake_margin_m"] = float(catalog_row.get("brake_margin_m", 5.0) or 5.0)
        result["sign_s"] = float(catalog_row.get("sign_s",
                                                 catalog_row.get("distance_from_start", 0.0)) or 0.0)

    env = None
    # Original run_benchmark.py uses seed=int(sign_id) for deterministic routing.
    # For variations we offset by var_idx so each variation gets a different
    # navigation route choice while still being reproducible.
    sign_id_int = int(catalog_row.get("sign_id", 0))
    var_idx = int(catalog_row.get("var_idx", 0))
    env_seed = (sign_id_int + var_idx) % 100000
    try:
        env = _build_env(catalog_row, profile)
        try:
            obs, info = env.reset(seed=env_seed)
        except TypeError as exc:
            # Most common failure: road_id from meta.json doesn't map to a parsed
            # MetaDrive lane → sign_lane is None → NoneType TypeErrors deep in
            # sumo_env.reset() during sign placement.
            msg = str(exc)
            if "NoneType" in msg:
                raise RuntimeError(f"sign_lane_not_found: road_id={catalog_row.get('road_id')}")
            raise

        # Ego isolation: apply DEFAULT_EGO_PARAMS on ego-policy instance
        try:
            ego_policy = env.engine.get_policy(env.vehicle.id) if hasattr(env, "engine") else None
            if ego_policy is not None:
                apply_ego_defaults(ego_policy)
        except Exception:
            pass

        # Set spawn velocity post-reset
        if spawn_v > 0:
            try:
                veh = env.agent if hasattr(env, "agent") else env.vehicle
                veh.set_velocity([spawn_v, 0.0])
            except Exception:
                pass

        # Capture fingerprint from reset state
        result["fingerprint"] = _capture_reset_state(env)
        # Record the actual braking spawn (3.24): where ego was placed, achieved
        # distance, final v0, and whether the runway was insufficient.
        brake_info = getattr(env, "_braking_spawn_info", None)
        if brake_info:
            result.update(brake_info)
        result["valid"] = True
    except Exception as exc:
        result["valid"] = False
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        return result

    # 4. Optionally drive an agent through the full episode
    if drive_agent:
        try:
            drive_metrics = _drive_episode(env, agent_policy, max_steps, save_gif)
            result.update(drive_metrics)
            # Drivability filter: a scene is only valid if the rule-aware REFERENCE
            # driver completes it cleanly — reaches the destination, no crash, no
            # off-road. Scenes where even the expert fails have bad geometry (sharp
            # junctions / curves the ego can't hold) and are dropped from the bench.
            drivable = (drive_metrics.get("arrived_dest")
                        and not drive_metrics.get("crashed")
                        and not drive_metrics.get("out_of_road"))
            if not drivable:
                result["valid"] = False
                reasons = []
                if not drive_metrics.get("arrived_dest"): reasons.append("no_dest")
                if drive_metrics.get("crashed"): reasons.append("crash")
                if drive_metrics.get("out_of_road"): reasons.append("off_road")
                result["failure_reason"] = "undrivable:" + ",".join(reasons)
        except Exception as exc:
            result["drive_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                env.close()
            except Exception:
                pass
    else:
        try:
            env.close()
        except Exception:
            pass

    return result


def _drive_episode(env, agent_policy: str, max_steps: int, save_gif: str | None) -> dict:
    """Drive a real policy through the env until termination or max_steps.

    Records dest_rate, crash, out_of_road, n_violations, route_completion.
    Optionally saves a top-down GIF.
    """
    from envs.sumo_env import TrafficSignSumoEnv  # noqa
    from agents.policies.comprehensive_rule_expert import ComprehensiveRuleExpertPolicy
    from agents.policies.rule_compliant_expert import RuleCompliantExpertPolicy
    from metadrive.policy.idm_policy import IDMPolicy

    policy_map = {
        "comprehensive_rule_expert": ComprehensiveRuleExpertPolicy,
        "rule_compliant_expert": RuleCompliantExpertPolicy,
        "idm": IDMPolicy,
    }
    PolicyCls = policy_map.get(agent_policy, ComprehensiveRuleExpertPolicy)

    veh = env.agent if hasattr(env, "agent") else env.vehicle
    seed_for_policy = int(env.engine.global_random_seed) if env.engine else 0
    policy = PolicyCls(veh, seed_for_policy)

    arrived = False
    crashed = False
    out_of_road = False
    n_violations = 0
    violations_log: list = []
    step = 0
    last_info: dict = {}

    sign_mgr = env.engine.traffic_sign_manager

    # Capture sign positions/lanes BEFORE driving so we can check whether the
    # ego actually reached them. A scene where the agent terminates before
    # reaching the sign is meaningless for sign compliance evaluation.
    sign_lane_ids: set = set()
    sign_positions: list = []
    try:
        for sign in sign_mgr.signs:
            if sign.lane is not None:
                sign_lane_ids.add(str(sign.lane.index))
            if hasattr(sign, "position"):
                sign_positions.append((float(sign.position[0]), float(sign.position[1])))
    except Exception:
        pass

    visited_lanes: set = set()
    min_dist_to_sign = float("inf")
    passed_sign = False

    while step < max_steps:
        try:
            action = policy.act(veh.name)
        except Exception:
            action = [0.0, 0.0]

        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as exc:
            return {
                "drove": False,
                "drive_error": f"step_failed: {exc}",
                "episode_length_steps": step,
            }
        last_info = info

        # Track whether ego visits the sign's lane (definition of "passed sign")
        # and how close it gets to the sign position. Threshold accounts for
        # multi-lane roads where the sign is on an adjacent lane (~3-4m apart).
        try:
            cur_lane = str(getattr(veh, "lane_index", "?"))
            visited_lanes.add(cur_lane)
            if cur_lane in sign_lane_ids:
                passed_sign = True
            for sx, sy in sign_positions:
                d = ((veh.position[0] - sx) ** 2 + (veh.position[1] - sy) ** 2) ** 0.5
                if d < min_dist_to_sign:
                    min_dist_to_sign = d
                if d < 10.0:  # ~2 lane widths — covers same-road adjacent lanes
                    passed_sign = True
            # If any sign reports a violation, the agent is definitionally
            # within the sign's evaluation region → mark as passed.
            if n_violations > 0:
                passed_sign = True
        except Exception:
            pass

        # Check sign violations every step
        try:
            for sign, violated in sign_mgr.check_all_violations(veh):
                if violated:
                    n_violations += 1
                    violations_log.append({
                        "step": step,
                        "sign": type(sign).__name__,
                        "rule": sign.get_rule_description(),
                    })
        except Exception:
            pass

        if save_gif:
            try:
                text_dict = {
                    "Step": step,
                    "Speed": f"{veh.speed_km_h:.1f} km/h",
                    "Violations": n_violations,
                }
                env.render(
                    mode="top_down",
                    film_size=(4800, 4800),
                    scaling=24.0,
                    screen_size=(600, 600),
                    semantic_map=True,
                    semantic_broken_line=True,
                    draw_target_vehicle_trajectory=True,
                    target_agent_heading_up=True,
                    screen_record=True,
                    window=False,
                    text=text_dict,
                )
            except Exception:
                pass

        step += 1
        if terminated or truncated:
            arrived = bool(info.get("arrive_dest", False))
            crashed = bool(info.get("crash", False))
            out_of_road = bool(info.get("out_of_road", False))
            break

    # Save GIF
    if save_gif:
        try:
            if hasattr(env, "top_down_renderer") and env.top_down_renderer is not None:
                env.top_down_renderer.generate_gif(save_gif, duration=40)
        except Exception:
            pass

    route_completion = 0.0
    try:
        route_completion = float(getattr(veh.navigation, "route_completion", 0.0))
    except Exception:
        pass

    return {
        "drove": True,
        "arrived_dest": arrived,
        "crashed": crashed,
        "out_of_road": out_of_road,
        "n_violations": n_violations,
        "violations_log": violations_log,
        "route_completion": round(route_completion, 4),
        "episode_length_steps": step,
        "max_steps": max_steps,
        # Sign-coverage metrics — critical for benchmark relevance
        "passed_sign": passed_sign,
        "min_dist_to_sign_m": round(min_dist_to_sign, 2) if min_dist_to_sign != float("inf") else None,
        "n_sign_lanes": len(sign_lane_ids),
    }
