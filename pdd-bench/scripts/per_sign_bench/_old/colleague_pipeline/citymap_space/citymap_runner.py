"""Materialize CityMap scenes against a long-lived env session.

CityMap generation is non-deterministic across env instances even with the
same seed. To work around this we open ONE env per (map_seed, n_blocks),
enumerate scenes from its actual graph, then materialize each scene by
mutating spawn config + reset, overriding navigation + placing the sign.
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Path bootstrap
SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_DIR = SCRIPT_PATH.parent.parent
SCRIPTS_DIR = BENCHMARK_DIR.parent
PDD_BENCH_DIR = SCRIPTS_DIR.parent
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"
for _p in (PDD_BENCH_DIR, METADRIVE_DIR, BENCHMARK_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.road_network import Road

from .citymap_env import CityMapTrafficSignEnv
from .citymap_scene_enumerator import CityMapScene
from factorized_space.benchmark_runner import (
    SIGN_CLASS_MAP,
    DETOUR_KEYS,
    BEGIN_TO_END,
    spawn_detour_obstacle,
    install_npc_vehicle_type_hook,
)
from factorized_space.ego_defaults import apply_ego_defaults


def _set_spawn_lane(env: CityMapTrafficSignEnv, lane_index: tuple) -> None:
    env.config["agent_configs"]["default_agent"]["spawn_lane_index"] = lane_index


def _apply_profile(env: CityMapTrafficSignEnv, profile: dict) -> None:
    """Mutate env.config and IDMPolicy class attrs so the profile parameters
    take effect on next reset.

    Three categories:
      1. env.config keys read by managers per reset:
           traffic_density, horizon
      2. IDMPolicy class attributes read by every IDM-controlled vehicle
         (NPC + ego if expert): NORMAL_SPEED, ACC_FACTOR, DEACC_FACTOR,
         DISTANCE_WANTED, TIME_WANTED, LANE_CHANGE_FREQ
      3. Vehicle spawn velocity is injected separately via spawn_velocity_ms
    """
    # ---- env.config (read per reset by PGTrafficManager) ----
    td = float(profile.get("traffic_density", 0.0))
    env.config["traffic_density"] = td
    horizon = int(profile.get("horizon_steps", 500))
    env.config["horizon"] = horizon

    # ---- IDM class attributes (read by every IDM vehicle in this episode) ----
    # nuPlan profile values are in m/s; IDMPolicy.NORMAL_SPEED is in km/h.
    from metadrive.policy.idm_policy import IDMPolicy
    if "NORMAL_SPEED" in profile:
        IDMPolicy.NORMAL_SPEED = float(profile["NORMAL_SPEED"]) * 3.6  # m/s → km/h
    if "MAX_SPEED" in profile:
        IDMPolicy.MAX_SPEED = float(profile["MAX_SPEED"]) * 3.6
    if "ACC_FACTOR" in profile:
        IDMPolicy.ACC_FACTOR = float(profile["ACC_FACTOR"])
    if "DEACC_FACTOR" in profile:
        # IDMPolicy uses NEGATIVE deacc by convention (see policy source)
        IDMPolicy.DEACC_FACTOR = -abs(float(profile["DEACC_FACTOR"]))
    # DISTANCE_WANTED / TIME_WANTED intentionally NOT applied — keep MetaDrive
    # defaults (10 m / 1.5 s) so NPC IDM doesn't tail-gate ego.
    if "LANE_CHANGE_FREQ" in profile:
        # LANE_CHANGE_FREQ is per-km in profile; IDMPolicy uses cooldown (steps).
        # Convert by typical 8 m/s for 1km = 125 s = 1250 steps; divide.
        rate_per_km = float(profile["LANE_CHANGE_FREQ"])
        IDMPolicy.LANE_CHANGE_FREQ = int(max(50, 1250.0 / max(rate_per_km, 1.0)))


def _verify_route_includes_sign(env, sign_edge):
    checkpoints = list(getattr(env.vehicle.navigation, "checkpoints", []))
    return any(
        checkpoints[i] == sign_edge[0] and checkpoints[i + 1] == sign_edge[1]
        for i in range(len(checkpoints) - 1)
    ), checkpoints


def _checkpoints_pass_through(checkpoints, edge):
    return any(
        checkpoints[i] == edge[0] and checkpoints[i + 1] == edge[1]
        for i in range(len(checkpoints) - 1)
    )


def _install_explicit_route(nav, road_network, checkpoints):
    """Override navigation with a caller-supplied list of checkpoint nodes.

    Mimics what ``NodeNetworkNavigation.set_route`` does internally but with
    an explicit path instead of recomputing shortest_path. Used for
    prohibition scenes where the agent must follow the detour (alternative)
    route, not MetaDrive's natural shortest path through the forbidden edge.

    The first edge of ``checkpoints`` MUST equal the agent's spawn lane edge,
    otherwise navigation localisation will be inconsistent.
    """
    assert len(checkpoints) >= 2, f"need at least 2 checkpoints, got {checkpoints}"
    nav.checkpoints = list(checkpoints)
    if len(checkpoints) <= 2:
        nav._target_checkpoints_index = [0, 0]
    else:
        nav._target_checkpoints_index = [0, 1]

    nav.final_road = Road(checkpoints[-2], checkpoints[-1])
    final_lanes = nav.final_road.get_lanes(road_network)
    nav.final_lane = final_lanes[-1]
    nav._navi_info.fill(0.0)

    target_road_1_start = checkpoints[0]
    target_road_1_end = checkpoints[1]
    nav.current_ref_lanes = road_network.graph[target_road_1_start][target_road_1_end]
    nav.next_ref_lanes = (
        road_network.graph[checkpoints[1]][checkpoints[2]]
        if len(checkpoints) > 2 else None
    )
    nav.current_road = Road(target_road_1_start, target_road_1_end)
    nav.next_road = Road(checkpoints[1], checkpoints[2]) if len(checkpoints) > 2 else None

    nav.total_length = 0.0
    nav.travelled_length = 0.0
    nav._last_long_in_ref_lane = 0.0
    for c1, c2 in zip(checkpoints[:-1], checkpoints[1:]):
        nav.total_length += road_network.graph[c1][c2][0].length


def materialize_scene_in_env(
    env: CityMapTrafficSignEnv,
    scene: CityMapScene,
    profile: dict,
    spawn_velocity_ms: float = 8.0,
    deterministic_seed: int = None,
    max_validation_steps: int = 30,
) -> dict:
    """Materialize a scene in the given long-lived env session.

    Tries to spawn the agent at scene.spawn_node_from. If env reset fails (the
    safe-spawn heuristic was wrong) falls back to FirstPGBlock and re-routes.
    """
    seed = deterministic_seed if deterministic_seed is not None else (scene.map_seed * 1_000_000 + 1)
    np.random.seed(seed)
    random.seed(seed)

    # Install NPC-vehicle-type hook (idempotent). Applies nuPlan-derived
    # vehicle-type distribution to every NPC spawned by PGTrafficManager.
    install_npc_vehicle_type_hook()

    result = {
        "scene_id": scene.scene_id,
        "map_seed": scene.map_seed,
        "n_blocks": scene.n_blocks,
        "lane_num": scene.lane_num,
        "sign_type": scene.sign_placement.sign_key,
        "route_length": scene.route_length,
        "has_alternative": scene.has_alternative,
        "alternative_length": scene.alternative_length,
        "primary_length": scene.primary_length,
        "detour_extra_m": (
            (scene.route_length - scene.primary_length)
            if scene.has_alternative and scene.primary_length is not None else None
        ),
        "spawn_velocity_ms": spawn_velocity_ms,
        "profile_id": profile.get("profile_id", -1),
        "spawn_fallback": False,
        "deterministic_seed": seed,
    }
    # Embed full profile parameters with profile_ prefix so the evaluator can
    # compute diversity and realism metrics on the same fields as PGMap records.
    for k, v in profile.items():
        if k != "profile_id":
            result[f"profile_{k}"] = v

    sign_edge = (scene.sign_placement.edge_from, scene.sign_placement.edge_to)

    # For prohibition scenes the agent must follow the detour, not the natural
    # shortest path. The detour is `scene.route_path`; it starts with the spawn
    # edge and avoids the forbidden sign edge.
    use_detour = bool(scene.has_alternative) and scene.route_path is not None and len(scene.route_path) >= 2

    # Apply profile parameters BEFORE spawn attempts so traffic + horizon are set
    _apply_profile(env, profile)

    def _setup_navigation():
        """Set the navigation route after a successful env.reset.

        Returns (ok, checkpoints, failure_reason). For prohibition scenes the
        natural shortest path goes through the forbidden edge, so we install
        the explicit alternative path stored on the scene; the verification
        flips polarity (route must NOT contain the sign edge).
        """
        try:
            env.vehicle.navigation.set_route(env.vehicle.lane.index, scene.dest_node)
        except Exception as exc:
            return False, [], f"set_route: {exc}"

        if use_detour:
            try:
                _install_explicit_route(
                    env.vehicle.navigation,
                    env.current_map.road_network,
                    scene.route_path,
                )
            except Exception as exc:
                return False, list(env.vehicle.navigation.checkpoints), f"install_alt_route: {exc}"

        checkpoints = list(env.vehicle.navigation.checkpoints)
        if use_detour:
            if _checkpoints_pass_through(checkpoints, sign_edge):
                return False, checkpoints, "alt_route_still_through_sign"
            if checkpoints[-1] != scene.dest_node:
                return False, checkpoints, "alt_route_wrong_dest"
        else:
            if not _checkpoints_pass_through(checkpoints, sign_edge):
                return False, checkpoints, "sign_not_on_route"
        return True, checkpoints, None

    # ---- Attempt 1: spawn at the scene-specified node ----
    target_spawn = (scene.spawn_node_from, scene.spawn_node_to, scene.spawn_lane_idx)
    spawn_ok = False
    nav_failure_reason = None
    try:
        _set_spawn_lane(env, target_spawn)
        env.reset(seed=scene.map_seed)
        spawn_ok = True
    except Exception:
        spawn_ok = False

    if spawn_ok:
        ok, checkpoints, reason = _setup_navigation()
        if not ok:
            spawn_ok = False
            nav_failure_reason = reason
            result["scene_route_failed_checkpoints"] = checkpoints

    # ---- Attempt 2: fallback to FirstPGBlock ----
    # NB: only meaningful for non-detour scenes — the alternative path is
    # tied to the scene's spawn edge, so falling back to FirstPGBlock would
    # break the navigation contract.
    if not spawn_ok:
        if use_detour:
            result["valid"] = False
            result["failure_reason"] = nav_failure_reason or "detour_spawn_failed"
            return result
        result["spawn_fallback"] = True
        try:
            _set_spawn_lane(env, (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0))
            env.reset(seed=scene.map_seed)
            env.vehicle.navigation.set_route(env.vehicle.lane.index, scene.dest_node)
        except Exception as exc:
            result["valid"] = False
            result["failure_reason"] = f"fallback_reset: {exc}"
            return result
        on_route, checkpoints = _verify_route_includes_sign(env, sign_edge)
        if not on_route:
            result["valid"] = False
            result["failure_reason"] = "sign_not_on_route"
            result["route_checkpoints"] = checkpoints
            return result

    # ---- Ego isolation: NPC profile is in IDMPolicy class attrs; override
    # ego-policy instance attrs so ego uses DEFAULT_EGO_PARAMS. Safe no-op
    # for non-IDM policies.
    try:
        ego_policy = env.engine.get_policy(env.vehicle.id) if hasattr(env, "engine") else None
        if ego_policy is not None:
            apply_ego_defaults(ego_policy)
    except Exception:
        pass

    # ---- Set spawn velocity post-reset ----
    if spawn_velocity_ms > 0:
        try:
            env.vehicle.set_velocity([float(spawn_velocity_ms), 0.0])
        except Exception:
            pass

    # ---- Place the sign ----
    rn = env.current_map.road_network
    try:
        target_lanes = rn.graph[scene.sign_placement.edge_from][scene.sign_placement.edge_to]
    except KeyError:
        result["valid"] = False
        result["failure_reason"] = "sign_edge_missing"
        return result
    lane_idx = min(scene.sign_placement.lane_idx, len(target_lanes) - 1)
    target_lane = target_lanes[lane_idx]

    sign_class = SIGN_CLASS_MAP.get(scene.sign_placement.sign_key)
    if sign_class is None:
        result["valid"] = False
        result["failure_reason"] = f"unknown_sign:{scene.sign_placement.sign_key}"
        return result

    sign_mgr = env.engine.traffic_sign_manager
    try:
        sign = sign_mgr.add_sign(sign_class, lane=target_lane, use_random_lane=False)
        if scene.sign_placement.sign_key in BEGIN_TO_END:
            sign_mgr.add_sign(BEGIN_TO_END[scene.sign_placement.sign_key],
                              lane=sign.lane, use_random_lane=False)
        if scene.sign_placement.sign_key in DETOUR_KEYS and sign is not None:
            spawn_detour_obstacle(env.engine, sign.lane, sign)
            if not getattr(sign, "detour_feasible", True):
                result["valid"] = False
                result["failure_reason"] = "detour_infeasible"
                return result
    except Exception as exc:
        result["valid"] = False
        result["failure_reason"] = f"sign_placement: {exc}"
        return result

    # ---- Quick validation steps ----
    try:
        for _ in range(max_validation_steps):
            obs, reward, term, trunc, info = env.step([0, 0])
            if term or trunc:
                break
    except Exception as exc:
        result["valid"] = False
        result["failure_reason"] = f"validation_step: {exc}"
        return result

    result["valid"] = True
    result["failure_reason"] = None
    return result
