"""Build pgmap/sumo envs from a manifest row + sign placement."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from bench._paths import PDD_BENCH_DIR  # noqa: F401  (ensures sys.path)
from bench.util import _row_seed
from envs.sumo_env import TrafficSignSumoEnv
from envs.sumo_traffic_manager import SumoTrafficManager
from envs.traffic_sign_env import TrafficSignEnv


_SUMO_SIGN_DISTANCE_CACHE: dict[Path, float] = {}


_PROFILE_KEYS = (
    "NORMAL_SPEED",
    "MAX_SPEED",
    "CREEP_SPEED",
    "ACC_FACTOR",
    "DEACC_FACTOR",
    "DISTANCE_WANTED",
    "TIME_WANTED",
    "LANE_CHANGE_FREQ",
    "traffic_density",
    "horizon_steps",
)


def _manifest_profile(row: dict) -> dict:
    profile: dict = {}
    for key in _PROFILE_KEYS:
        if f"profile_{key}" in row:
            profile[key] = row[f"profile_{key}"]
        elif key in row:
            profile[key] = row[key]
    return profile


def _manifest_traffic_density(row: dict, default: float) -> float:
    profile = _manifest_profile(row)
    val = profile.get("traffic_density", default)
    return float(val)


def _manifest_horizon(row: dict, fallback: int) -> int:
    profile = _manifest_profile(row)
    val = profile.get("horizon_steps", fallback)
    return int(val)


def _apply_manifest_profile_to_npcs(row: dict) -> None:
    profile = _manifest_profile(row)
    if not profile:
        return
    from factorized_space.agent_profile_bank import apply_profile_to_idm_class

    apply_profile_to_idm_class(profile)


def _build_pgmap_env(row: dict, max_steps: int) -> TrafficSignEnv:
    from metadrive.component.pgblock.first_block import FirstPGBlock

    seed = _row_seed(row)
    _apply_manifest_profile_to_npcs(row)
    traffic_density = _manifest_traffic_density(row, default=0.1)
    horizon = _manifest_horizon(row, fallback=max_steps)
    spawn_lane = int(row["spawn_lane_index"])
    is_ramp_merge = row.get("block_id") == "r" and row.get("route_intent") == "merge"
    is_ramp_exit = row.get("block_id") == "R" and row.get("route_intent") == "exit"
    if is_ramp_merge:
        spawn_lane_tuple = ("2r1_0_", "2r1_1_", 0)
    elif is_ramp_exit:
        spawn_lane_tuple = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, int(row["lane_num"]) - 1)
    else:
        spawn_lane_tuple = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, spawn_lane)

    vehicle_config = {"show_lidar": False}
    spawn_vel = float(row.get("spawn_velocity_ms", 0.0) or 0.0)
    if spawn_vel > 0:
        vehicle_config["spawn_velocity"] = [spawn_vel, 0.0]
        vehicle_config["spawn_velocity_car_frame"] = True

    config = dict(
        start_seed=seed,
        use_render=False,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.CRITICAL,
        traffic_density=traffic_density,
        horizon=horizon,
        vehicle_config=vehicle_config,
        map_config={
            "type": "block_sequence",
            "config": row["block_sequence"],
            "lane_num": row["lane_num"],
            "lane_width": row["lane_width"],
        },
        random_spawn_lane_index=False,
        agent_configs={
            "default_agent": dict(
                use_special_color=True,
                spawn_lane_index=spawn_lane_tuple,
            )
        },
    )
    return TrafficSignEnv(config)


def _place_pgmap_sign(env: TrafficSignEnv, row: dict, seed: int) -> bool:
    from factorized_space.benchmark_runner import (
        BEGIN_TO_END,
        DETOUR_KEYS,
        LANE_CHANGE_KEYS,
        RESTRICTED_BEGIN_KEYS,
        SIGN_CLASS_MAP,
        _get_route_lanes,
        _override_route_intent,
        _pick_detour_lane,
        _pick_lane_for_lane_change,
        _pick_rightmost_lane,
        _pick_route_lane,
        _spawn_cyclists_on_lane,
    )
    from factorized_space.space_definition import BIKE_RELATED_SIGNS
    from factorized_space.sign_placement import zone_pair_offsets
    from traffic_signs.detour_obstacle import spawn_detour_obstacle

    if row.get("sign_type") is None and row.get("sign_type_start") and row.get("sign_type_end"):
        # Paired scene: place start + end signs on one lane segment.
        if not _override_route_intent(env, row):
            return False
        route_lanes = _get_route_lanes(env)
        if not route_lanes:
            return False

        zone_length = float(row.get("zone_length_m", 20.0))
        start_long, end_long, min_lane = zone_pair_offsets(zone_length)
        lane = _pick_route_lane(route_lanes, min_length=min_lane, road_network=env.current_map.road_network)
        if lane is None:
            return False

        sign_start_cls = SIGN_CLASS_MAP.get(row["sign_type_start"])
        sign_end_cls = SIGN_CLASS_MAP.get(row["sign_type_end"])
        if sign_start_cls is None or sign_end_cls is None:
            return False

        sign_mgr = env.engine.traffic_sign_manager
        half_w = lane.width_at(0) / 2 + 0.8
        # Offsets are meters from the lane START (longitudinal_from_start).
        try:
            sign_mgr.add_sign(
                sign_start_cls,
                lane=lane,
                longitudinal_offset=start_long,
                lateral_offset=half_w,
                use_random_lane=False,
            )
            sign_mgr.add_sign(
                sign_end_cls,
                lane=lane,
                longitudinal_offset=end_long,
                lateral_offset=half_w,
                use_random_lane=False,
            )
            # Truncate the start zone at the end sign (otherwise no effect).
            sign_mgr.build_zones()
            return True
        except Exception:
            return False

    sign_key = row["sign_type"]
    if sign_key not in SIGN_CLASS_MAP:
        return False
    sign_cls = SIGN_CLASS_MAP[sign_key]

    if not _override_route_intent(env, row):
        return False

    route_lanes = _get_route_lanes(env)
    if not route_lanes:
        return False

    veh = env.vehicle
    rn = env.current_map.road_network
    sign_mgr = env.engine.traffic_sign_manager

    if sign_key in DETOUR_KEYS:
        lane = _pick_detour_lane(route_lanes, sign_cls, min_length=30.0)
    elif sign_key in LANE_CHANGE_KEYS:
        vidx = getattr(veh.lane, "index", None)
        vnum = vidx[2] if (vidx and len(vidx) >= 3) else 0
        lane = _pick_lane_for_lane_change(route_lanes, vnum, road_network=rn)
    elif sign_key in RESTRICTED_BEGIN_KEYS:
        lane = _pick_rightmost_lane(route_lanes, min_length=30.0)
    else:
        lane = _pick_route_lane(route_lanes, min_length=15.0, road_network=rn)

    if lane is None:
        return False

    if sign_key in RESTRICTED_BEGIN_KEYS:
        sign = sign_mgr.add_sign(sign_cls, lane=lane, use_random_lane=False)
        if sign_key in BEGIN_TO_END:
            sign_mgr.add_sign(BEGIN_TO_END[sign_key], lane=sign.lane, use_random_lane=False)
    else:
        sign = sign_mgr.add_sign(sign_cls, lane=lane, use_random_lane=False)

    if sign_key in DETOUR_KEYS and sign is not None:
        spawn_detour_obstacle(env.engine, sign.lane, sign)
    if sign_key in BIKE_RELATED_SIGNS and sign is not None:
        _spawn_cyclists_on_lane(env, sign.lane, seed, n=3)

    return True


def _apply_manifest_npc_speed_cap(row: dict) -> None:
    """Cap NPC speed to the sign limit when the manifest marks them compliant.

    The manifest is the single source of truth: materialization records
    `npc_compliant` / `npc_speed_cap_kmh`, and eval re-applies the SAME cap to the
    NPC IDM policy classes (IDMPolicy / SumoTrajectoryIDMPolicy) — independent of
    the PER_SIGN_COMPLIANT_NPC env var. No-op when the row isn't compliant.
    """
    if not row.get("npc_compliant"):
        return
    cap = float(row.get("npc_speed_cap_kmh", 0.0) or 0.0)
    if cap <= 0:
        return
    from metadrive.policy.idm_policy import IDMPolicy
    from envs.sumo_idm_policy import SumoTrajectoryIDMPolicy
    for cls in (IDMPolicy, SumoTrajectoryIDMPolicy):
        cls.NORMAL_SPEED = min(float(getattr(cls, "NORMAL_SPEED", cap)), cap)
        cls.MAX_SPEED = cap


# Set by run_benchmark.main() before building envs. True (default) teleports ego
# onto the sign-topology lane after sign placement; run_benchmark flips this to
# False for NN policies (plant2/carl/ppo_lidar), matching upstream 1300c1e —
# those policies fail when relocated off the manifest road_id.
RELOCATE_EGO_TO_SIGN_LANE = True


def _build_sumo_env(row: dict, scenes_root: Path, max_steps: int) -> TrafficSignSumoEnv:
    SumoTrafficManager.EGO_SAFE_RADIUS = 15
    _apply_manifest_profile_to_npcs(row)
    _apply_manifest_npc_speed_cap(row)
    traffic_density = _manifest_traffic_density(row, default=0.1)
    horizon = _manifest_horizon(row, fallback=max_steps)
    net_path = str(scenes_root / row["net_path"]) if not str(row["net_path"]).startswith("/") else str(row["net_path"])
    is_braking = bool(row.get("braking_spawn"))
    # Braking scenes (3.24): keep the sign at its REAL offset (runway comes from
    # the upstream spawn, not the legacy 30 m floor which would displace it).
    if is_braking:
        sign_spawn_distance = float(row.get("sign_s",
                                            row.get("sign_spawn_distance",
                                                    row.get("distance_from_start", 0.0))) or 0.0)
    else:
        sign_spawn_distance = _resolve_sign_spawn_distance(row, scenes_root)

    vehicle_config: dict = {"show_lidar": False}
    spawn_vel = float(row.get("spawn_velocity_ms", 0.0) or 0.0)
    # For braking scenes the velocity is set by the env's upstream spawn
    # (_spawn_ego_before_sign); don't also pre-set it via vehicle_config.
    if spawn_vel > 0 and not is_braking:
        vehicle_config["spawn_velocity"] = [spawn_vel, 0.0]
        vehicle_config["spawn_velocity_car_frame"] = True

    config = dict(
        use_render=False,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.CRITICAL,
        map_name=net_path,
        sign_type=row.get("sign_code") or row.get("sign_type"),
        traffic_density=traffic_density,
        tl_speed_factor=float(row.get("tl_speed_factor", 20.0)),
        sign_spawn_distance=sign_spawn_distance,
        min_route_hops_after_spawn=int(row.get("min_route_hops_after_spawn", 10)),
        max_route_hops_after_spawn=int(row.get("max_route_hops_after_spawn", 10)),
        horizon=horizon,
        num_scenarios=100000,
        vehicle_config=vehicle_config,
        debug_one_way_sign_selection=bool(row.get("debug_one_way_sign_selection", False)),
        relocate_ego_to_sign_lane=RELOCATE_EGO_TO_SIGN_LANE,
    )
    if row.get("road_id"):
        config["vehicle_config"]["spawn_lane_index"] = row["road_id"]
    if "spawn_lane_num" in row:
        config["spawn_lane_num"] = int(row["spawn_lane_num"])
    # Pin the destination lane if the manifest specifies one — keeps ego from
    # taking BFS-picked routes that wander away from the sign zone. Mirrors
    # expert_replay.py:389-391 (their env-var gate, here unconditional).
    if row.get("destination_lane_id"):
        config["vehicle_config"]["destination"] = row["destination_lane_id"]

    # Braking-spawn (3.24): reconstruct the upstream spawn + NPC corridor
    # restriction/relocation exactly as at materialization. Mirrors
    # sumo_runner._build_env so the EVAL actor spawn is correct.
    if is_braking:
        config["ego_braking_spawn"] = True
        config["ego_spawn_v0_ms"] = float(row.get("spawn_velocity_ms", 0.0) or 0.0)
        config["ego_brake_d_required"] = float(row.get("d_required_m", 0.0) or 0.0)
        config["ego_v_target_kmh"] = float(row.get("v_target_kmh", 0.0) or 0.0)
        config["ego_brake_decel"] = float(row.get("brake_decel_mps2", 2.5) or 2.5)
        config["ego_brake_delay"] = float(row.get("brake_delay_s", 1.0) or 1.0)
        config["ego_brake_margin"] = float(row.get("brake_margin_m", 5.0) or 5.0)

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


def _resolve_sign_spawn_distance(row: dict, scenes_root: Path) -> float:
    """Sign offset along the edge = real distance_from_start (no minimum floor)."""
    direct = row.get("sign_spawn_distance")
    if direct is not None:
        return float(direct)

    direct = row.get("distance_from_start")
    if direct is not None:
        return float(direct)

    net_path = row.get("net_path")
    if not net_path:
        return 0.0

    net_file = Path(str(net_path))
    scene_dir = (scenes_root / net_file).parent if not net_file.is_absolute() else net_file.parent
    meta_path = scene_dir / "meta.json"

    if meta_path in _SUMO_SIGN_DISTANCE_CACHE:
        return _SUMO_SIGN_DISTANCE_CACHE[meta_path]

    distance = 0.0
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            distance = float(meta.get("distance_from_start", 0.0) or 0.0)
        except Exception:
            distance = 0.0

    _SUMO_SIGN_DISTANCE_CACHE[meta_path] = distance
    return distance


def build_env_for_row(row: dict, backend: str, scenes_root: Path, max_steps: int):
    """Build the env for a manifest row and return (env, env_seed, post_reset).

    This is the SINGLE place that knows about backends. run_benchmark stays
    backend-agnostic: it just resets `env` with `env_seed`, then calls
    `post_reset(base_env)` once after reset.

      * env        — an unreset MetaDrive env for this row.
      * env_seed   — the per-backend reset seed.
      * post_reset — callback(base_env) -> error_str | None. Runs backend-specific
                     setup that must happen AFTER reset (pgmap sign placement).
                     Returns an error string if the episode can't be set up
                     (recorded as ok=False), else None.

    To add a backend or a scene peculiarity, edit HERE — not run_benchmark.
    """
    seed = _row_seed(row)

    if backend in ("pgmap", "paired", "citymap"):
        env = _build_pgmap_env(row, max_steps=max_steps)
        env_seed = (int(row.get("map_seed") or row.get("seed") or 0)
                    if backend == "citymap" else seed)

        def post_reset(base_env):
            ok = _place_pgmap_sign(base_env, row, seed)
            return None if ok else "failed_to_place_pgmap_sign"

        return env, env_seed, post_reset

    if backend == "sumo":
        env = _build_sumo_env(row, scenes_root=scenes_root, max_steps=max_steps)
        env_seed = (int(row.get("sign_id", 0)) + int(row.get("var_idx", 0))) % 100000
        return env, env_seed, lambda base_env: None

    raise ValueError(f"Unsupported backend: {backend}")
