"""Build the SUMO env and apply spawn / destination from a manifest row."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from traffic_bench.envs.sumo import TrafficSignSumoEnv
from traffic_bench.envs.traffic import SumoTrafficManager
from traffic_bench.eval.engine.expand.manifest_config import (
    DEFAULT_DESTINATION_MAX_ALONG_M,
    DEFAULT_STOP_WAIT_STEPS,
)
from traffic_bench.eval.engine.map.junction_priority_layout import (
    JunctionLayoutError,
    build_junction_priority_layout,
)
from traffic_bench.eval.engine.map.lane_keys import clamp_lane_key_to_graph, lane_edge_id, make_lane_key
from traffic_bench.eval.signs.blocked.place import row_is_blocked_road as _row_is_blocked_road
from traffic_bench.eval.signs.crosswalk.place import row_is_crosswalk as _row_is_crosswalk
from traffic_bench.eval.signs.detour.place import row_is_detour as _row_is_detour
from traffic_bench.eval.signs.restricted_lane.place import (
    row_is_restricted_lane as _row_is_restricted_lane,
)

# Reserved lane: no background car on the ego edge within this many metres past
# the ego teleport point (front headway at the 5 m/s spawn speed plus margin).
RESTRICTED_SPAWN_CLEAR_M = 30.0
from traffic_bench.eval.signs.dual_path.nav import (
    OneWaySumoTrafficManager,
    resolve_row_background_excluded_edges,
)
from traffic_bench.eval.signs.dual_path.place import (
    row_is_one_way as _row_is_one_way,
    row_uses_dual_path_nav as _row_uses_dual_path_nav,
)
from traffic_bench.eval.signs.junction.nav import (
    JunctionOutgoingTrafficManager,
    resolve_row_background_spawn_edges,
)
from traffic_bench.eval.signs.junction.place import row_is_junction as _row_is_junction
from traffic_bench.eval.signs.roundabout.place import row_is_roundabout as _row_is_roundabout
from traffic_bench.eval.signs.speed.place import row_is_speed as _row_is_speed

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
    # Spawn density only. Never use profile_traffic_density here: that field is
    # the raw nuPlan sample (stats); traffic_density is background after aux.
    if row.get("traffic_density") is not None:
        return float(row["traffic_density"])
    return float(default)


def _manifest_horizon(row: dict, fallback: int) -> int:
    profile = _manifest_profile(row)
    val = profile.get("horizon_steps")
    if val is None:
        val = row.get("horizon", fallback)
    return int(val)


def _apply_manifest_profile_to_npcs(row: dict) -> None:
    profile = _manifest_profile(row)
    if not profile:
        return
    from traffic_bench.eval.engine.traffic.agent_profile_bank import apply_profile_to_idm_class

    apply_profile_to_idm_class(profile)


def _build_sumo_env(row: dict, scenes_root: Path, max_steps: int) -> TrafficSignSumoEnv:
    SumoTrafficManager.EGO_SAFE_RADIUS = 15
    _apply_manifest_profile_to_npcs(row)
    traffic_density = _manifest_traffic_density(row, default=0.0)
    horizon = _manifest_horizon(row, fallback=max_steps)
    net_path = str(scenes_root / row["net_path"]) if not str(row["net_path"]).startswith("/") else str(row["net_path"])
    sign_spawn_distance = _resolve_sign_spawn_distance(row, scenes_root)
    # Speed plates: no background traffic between the ego and the plate. The
    # approach is what the plate is judged on, and an NPC placed on it either
    # boxes the ego in or gets rear-ended before the sign is even reached. The
    # cut uses the row's sign_s -- the same longitude place_speed_signs puts the
    # plate at (signs/speed/place.py) -- so the two cannot drift apart;
    # sign_spawn_distance is the junction-style fallback and is not the plate.
    if _row_is_speed(row) and row.get("road_id"):
        _sign_s = row.get("sign_s")
        traffic_after_lng = float(_sign_s if _sign_s is not None else sign_spawn_distance)
        traffic_after_edge = str(row["road_id"])
        traffic_after_kmh = float(row.get("v_target_kmh") or 0.0)
        # Share of NPCs honouring the plate; rows made before the field
        # existed get 1.0, the old all-compliant behaviour.
        traffic_compliance = float(row.get("npc_compliance_rate", 1.0) or 0.0)
    elif _row_is_restricted_lane(row) and row.get("road_id"):
        # Reserved lane: the ego is teleported to spawn_offset_from_start
        # AFTER traffic has spawned, so the keep-clear guard saw it at the edge
        # start. On the ego's edge let background cars start only past the
        # teleport point; the zone and the neighbouring lane keep their traffic.
        # A ladder starting 5 m past the ego put the first car within its
        # 3-second braking distance: ppo/carl baselines and experts alike crashed
        # at step ~30 in a fifth of the rows. Keep the front headway clear.
        traffic_after_lng = (
            float(row.get("spawn_offset_from_start") or 0.0) + RESTRICTED_SPAWN_CLEAR_M
        )
        traffic_after_edge = str(row["road_id"])
        traffic_after_kmh = 0.0
        traffic_compliance = 1.0
    else:
        traffic_after_lng = -1.0
        traffic_after_edge = ""
        traffic_after_kmh = 0.0
        traffic_compliance = 1.0
    background_excluded_edges = (
        resolve_row_background_excluded_edges(row, net_path)
        if _row_is_one_way(row)
        else []
    )
    use_junction_outgoing_traffic = (
        (_row_is_junction(row) or _row_is_roundabout(row))
        and traffic_density > 0.0
    )
    background_spawn_edges = (
        resolve_row_background_spawn_edges(row, net_path)
        if use_junction_outgoing_traffic
        else []
    )

    vehicle_config: dict = {"show_lidar": False}
    spawn_vel = float(row.get("spawn_velocity_ms", 0.0) or 0.0)
    if spawn_vel > 0:
        vehicle_config["spawn_velocity"] = [spawn_vel, 0.0]
        vehicle_config["spawn_velocity_car_frame"] = True

    use_ped = bool(row.get("use_pedestrian_manager", False))
    use_yield = bool(row.get("use_pedestrian_yield_rule", False))
    ped_cfg = dict(row.get("pedestrian_manager") or {})

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
        traffic_spawn_after_lng=traffic_after_lng,
        traffic_spawn_after_edge=traffic_after_edge,
        traffic_spawn_after_kmh=traffic_after_kmh,
        traffic_npc_compliance_rate=traffic_compliance,
        min_route_hops_after_spawn=int(row.get("min_route_hops_after_spawn", 10)),
        max_route_hops_after_spawn=int(row.get("max_route_hops_after_spawn", 10)),
        horizon=horizon,
        num_scenarios=100000,
        vehicle_config=vehicle_config,
        debug_one_way_sign_selection=bool(row.get("debug_one_way_sign_selection", False)),
        show_lane_arrows=row.get("show_lane_arrows", False),
        show_traffic_lights=row.get("show_traffic_lights", False),
        show_npc_vehicles=row.get("show_npc_vehicles", False),
        background_excluded_edges=list(background_excluded_edges),
        background_spawn_edges=list(background_spawn_edges),
        skip_auto_signs=True,
        use_pedestrian_manager=use_ped,
        use_pedestrian_yield_rule=use_yield,
        enforce_pedestrian_yield_for_traffic=False,
    )
    if ped_cfg:
        config["pedestrian_manager"] = ped_cfg
    if row.get("road_id"):
        config["vehicle_config"]["spawn_lane_index"] = row["road_id"]
    if "spawn_lane_num" in row:
        config["spawn_lane_num"] = int(row["spawn_lane_num"])
    if row.get("destination_lane_id"):
        config["vehicle_config"]["destination"] = row["destination_lane_id"]

    is_one_way = _row_is_one_way(row)

    class _RealMapEnv(TrafficSignSumoEnv):
        @classmethod
        def default_config(cls):
            cfg = super().default_config()
            cfg["traffic_density"] = 0.0
            cfg["show_lane_arrows"] = True
            cfg["show_traffic_lights"] = True
            cfg["show_npc_vehicles"] = True
            cfg["skip_auto_signs"] = False
            cfg["background_excluded_edges"] = []
            cfg["background_spawn_edges"] = []
            return cfg

        def setup_engine(self):
            super().setup_engine()
            # Only add SumoTrafficManager if traffic_density > 0
            # Otherwise keep the default SimpleTrafficManager (no NPC spawning)
            if self.config.get("traffic_density", 0.0) > 0:
                if is_one_way:
                    mgr = OneWaySumoTrafficManager()
                elif (
                    (_row_is_junction(row) or _row_is_roundabout(row))
                    and self.config.get("background_spawn_edges")
                ):
                    mgr = JunctionOutgoingTrafficManager()
                else:
                    mgr = SumoTrafficManager()
                self.engine.update_manager("traffic_manager", mgr)

        def reset(self, *, seed=None):
            # Skip TrafficSignSumoEnv.reset() sign creation by calling grandparent directly
            if self.config.get("skip_auto_signs", False):
                # Call BaseEnv.reset() directly, skipping TrafficSignSumoEnv.reset()
                from metadrive.envs import BaseEnv
                obs, info = BaseEnv.reset(self, seed=seed)
                return obs, info
            else:
                return super().reset(seed=seed)

    return _RealMapEnv(config)


def _resolve_sign_spawn_distance(row: dict, scenes_root: Path) -> float:
    direct = row.get("sign_spawn_distance")
    if direct is not None:
        return max(float(direct), 30.0)

    direct = row.get("distance_from_start")
    if direct is not None:
        return max(float(direct), 30.0)

    net_path = row.get("net_path")
    if not net_path:
        return 0.0

    net_file = Path(str(net_path))
    scene_dir = (scenes_root / net_file).parent if not net_file.is_absolute() else net_file.parent
    meta_path = scene_dir / "meta.json"

    if meta_path in _SUMO_SIGN_DISTANCE_CACHE:
        return max(_SUMO_SIGN_DISTANCE_CACHE[meta_path], 30.0)

    distance = 0.0
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            distance = float(meta.get("distance_from_start", 0.0) or 0.0)
        except Exception:
            distance = 0.0

    _SUMO_SIGN_DISTANCE_CACHE[meta_path] = distance
    return max(distance, 30.0)


def _wrap_for_policy(env, policy_type: str):
    return env


def _format_lane_pos(pos) -> str:
    """Format lane position for logging (handles numpy arrays)."""
    if pos is None:
        return "N/A"
    return f"({float(pos[0]):.1f}, {float(pos[1]):.1f})"


def _analyze_junction_lanes(env) -> dict:
    """Analyze and print incoming/outgoing lanes of the junction.
    
    Incoming lanes: lanes that feed INTO the junction (have exit_lanes)
    Outgoing lanes: lanes that exit FROM the junction (have entry_lanes but no exit_lanes)
    
    Args:
        env: The environment instance.
        
    Returns:
        Dict with 'incoming' and 'outgoing' lane lists.
    """
    result = {"incoming": [], "outgoing": [], "junction_id": None}
    
    road_network = env.engine.current_map.road_network
    graph = road_network.graph
    
    incoming_lanes = []
    outgoing_lanes = []
    junction_id = None
    
    for lane_name, lane_info in graph.items():
        # Find the junction polygon
        if lane_name.startswith("junction"):
            junction_id = lane_name
            continue
        
        # Skip non-lane entries
        if not lane_name.startswith("lane_"):
            continue
        
        exit_lanes = getattr(lane_info, "exit_lanes", None) or []
        entry_lanes = getattr(lane_info, "entry_lanes", None) or []
        
        # Extract edge ID from lane name (e.g., "lane_46710990#1_0" -> "46710990#1")
        raw_name = lane_name[5:] if lane_name.startswith("lane_") else lane_name
        edge_id = raw_name.rsplit("_", 1)[0] if "_" in raw_name else raw_name
        
        lane_obj = None
        lane_length = 0.0
        start_pos = None
        end_pos = None
        
        try:
            lane_obj = road_network.get_lane(lane_name)
            lane_length = lane_obj.length
            # Get start and end positions of the lane
            start_pos = lane_obj.position(0.0, 0.0)  # Beginning of lane
            end_pos = lane_obj.position(lane_length, 0.0)  # End of lane
        except Exception:
            pass
        
        lane_data = {
            "lane_name": lane_name,
            "edge_id": edge_id,
            "length": lane_length,
            "exit_lanes": exit_lanes,
            "entry_lanes": entry_lanes,
            "start_pos": start_pos,
            "end_pos": end_pos,
        }
        
        # Incoming: has exit_lanes (feeds INTO junction)
        # Outgoing: has entry_lanes but no exit_lanes (exits FROM junction)
        if exit_lanes:
            incoming_lanes.append(lane_data)
        elif entry_lanes and not exit_lanes:
            outgoing_lanes.append(lane_data)
    
    result["incoming"] = incoming_lanes
    result["outgoing"] = outgoing_lanes
    result["junction_id"] = junction_id

    return result.get("incoming", []), result.get("outgoing", [])


def _apply_manifest_ego_spawn_lane(env, row: dict) -> bool:
    """Teleport ego onto the manifest parallel lane (needed when skip_auto_signs=True)."""
    road_id = row.get("road_id")
    if not road_id:
        return False
    lane_num = int(row.get("spawn_lane_num", 0) or 0)
    target_key = make_lane_key(str(road_id), lane_num)
    try:
        vehicle = env.agent
        if vehicle is None:
            return False
        road_network = env.engine.current_map.road_network
        clamped_spawn = clamp_lane_key_to_graph(target_key, road_network.graph)
        if clamped_spawn and clamped_spawn != target_key:
            print(f"[EgoSpawn] Clamped spawn {target_key} -> {clamped_spawn}")
            target_key = clamped_spawn
        target_lane = road_network.get_lane(target_key)
        start_long = min(1.0, target_lane.length - 0.1)
        pos = target_lane.position(start_long, 0.0)
        heading = target_lane.heading_theta_at(start_long)
        vehicle.set_position(pos)
        vehicle.set_heading_theta(heading)
        try:
            vehicle.spawn_place = pos.copy()
        except Exception:
            pass
        if hasattr(env, "_refresh_navigation_after_spawn"):
            env._refresh_navigation_after_spawn(target_lane)
        else:
            vehicle.reset_navigation(target_lane)
        return True
    except Exception as exc:
        print(f"[EgoSpawn] Could not teleport to {target_key}: {exc}")
        return False


def _apply_manifest_ego_spawn_velocity(env, row: dict) -> None:
    """Re-apply spawn speed after skip_auto_signs teleport."""
    v = float(row.get("spawn_velocity_ms") or 0.0)
    if v <= 0:
        return
    vehicle = getattr(env, "agent", None) or getattr(env, "vehicle", None)
    if vehicle is None:
        return
    try:
        vehicle.set_velocity([v, 0.0], in_local_frame=True)
    except TypeError:
        try:
            vehicle.set_velocity([v, 0.0])
        except Exception:
            pass
    except Exception as exc:
        print(f"[EgoSpawn] Could not set spawn velocity {v:.2f} m/s: {exc}")


def _apply_manifest_ego_destination(env, row: dict) -> Optional[str]:
    """Clamp ego destination to a real graph lane and re-bind navigation."""
    dest = row.get("destination_lane_id")
    if not dest:
        return None
    try:
        vehicle = env.agent
        if vehicle is None:
            return None
        road_network = env.engine.current_map.road_network
        clamped = clamp_lane_key_to_graph(str(dest), road_network.graph)
        if not clamped:
            return None
        if clamped != str(dest):
            print(f"[EgoDest] Clamped destination {dest} -> {clamped}")
        spawn_key = getattr(vehicle.lane, "index", None)
        if spawn_key and vehicle.navigation is not None:
            vehicle.navigation.set_route(spawn_key, clamped)
        _apply_destination_along_cap(env, row)
        return clamped
    except Exception as exc:
        print(f"[EgoDest] Could not apply destination {dest}: {exc}")
        return None


def _apply_destination_along_cap(env, row: dict) -> None:
    """Cap ego finish to ``min(cap, final_lane.length-5)`` on the final lane.

    Used by roundabout (4.3) and blocked_road (3.2): sets
    ``_priority_bench_dest_along_m`` for GIF/top-down and moves MetaDrive
    ``_dest_node_path``. Arrive for both signs uses the same cap (compliant-stop
    remains an alternate success path for 3.2). Violation on 3.2 is driving
    past the no-entry sign (sign manager), not a separate past-sign distance.
    """
    raw = row.get("destination_max_along_m")
    if raw is None and not (
        _row_is_roundabout(row)
        or _row_is_blocked_road(row)
        or _row_uses_dual_path_nav(row)
        or _row_is_detour(row)
        or _row_is_speed(row)
    ):
        return
    if raw is None and _row_uses_dual_path_nav(row):
        return
    try:
        cap = float(DEFAULT_DESTINATION_MAX_ALONG_M if raw is None else raw)
    except (TypeError, ValueError):
        return
    if cap <= 0.0:
        return

    vehicle = getattr(env, "agent", None) or getattr(env, "vehicle", None)
    if vehicle is None:
        return
    nav = getattr(vehicle, "navigation", None)
    final = getattr(nav, "final_lane", None) if nav is not None else None
    if final is None:
        return
    try:
        target = min(cap, max(0.5, float(final.length) - 5.0))
    except Exception:
        return

    try:
        vehicle._priority_bench_dest_along_m = float(target)
    except Exception:
        pass
    try:
        if nav is not None:
            nav._priority_bench_dest_along_m = float(target)
    except Exception:
        pass

    try:
        from metadrive.utils.coordinates_shift import panda_vector

        dest_path = getattr(nav, "_dest_node_path", None)
        if dest_path is not None:
            check_point = final.position(target, 0.0)
            height = float(getattr(nav, "MARK_HEIGHT", 1.0) or 1.0)
            dest_path.setPos(
                panda_vector(float(check_point[0]), float(check_point[1]), height)
            )
    except Exception:
        pass

    try:
        if _row_is_blocked_road(row):
            label = "Blocked-road"
        elif _row_uses_dual_path_nav(row):
            label = "Dual-path"
        elif _row_is_crosswalk(row):
            label = "Crosswalk"
        elif _row_is_detour(row):
            label = "Detour"
        elif _row_is_speed(row):
            label = "Speed"
        elif _row_is_junction(row):
            label = "Junction"
        else:
            label = "Roundabout"
        print(
            f"[EgoDest] {label} destination cap at {target:.1f}m "
            f"on final lane (len={float(final.length):.1f}m)"
        )
    except Exception:
        pass


# Back-compat alias (older call sites / notebooks).
_apply_roundabout_destination_cap = _apply_destination_along_cap


def _lane_index_road_key(lane_index) -> tuple | str | None:
    """Comparable road identity for a MetaDrive / SUMO lane index."""
    if lane_index is None:
        return None
    if isinstance(lane_index, str):
        try:
            return lane_edge_id(lane_index) or lane_index
        except Exception:
            return lane_index
    try:
        if len(lane_index) >= 2:
            return (lane_index[0], lane_index[1])
    except Exception:
        pass
    return lane_index


def _same_road_lane_index(a, b) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    return _lane_index_road_key(a) == _lane_index_road_key(b)


def _ego_reached_capped_destination(
    vehicle,
    *,
    max_along_m: float,
    arrive_tol_m: float = 2.0,
    allow_same_lane: bool = False,
) -> bool:
    """True when ego is on the route's final exit lane at/after the cap.

    Destination point = ``min(max_along_m, final_lane.length - 5)`` along the
    navigation final lane (MetaDrive end criterion, capped when the exit is
    longer). Works with ``EdgeNetworkNavigation`` (SUMO: checkpoints are lane
    indices) and node-network checkpoints.

    ``max_along_m <= 0`` disables the cap.
    """
    if vehicle is None:
        return False
    try:
        cap = float(max_along_m)
    except (TypeError, ValueError):
        return False
    if cap <= 0.0:
        return False

    nav = getattr(vehicle, "navigation", None)
    final = getattr(nav, "final_lane", None) if nav else None
    lane = getattr(vehicle, "lane", None)
    if nav is None or final is None or lane is None:
        return False

    lane_idx = getattr(lane, "index", None)
    final_idx = getattr(final, "index", None)
    if lane_idx is None or final_idx is None:
        return False

    checkpoints = list(getattr(nav, "checkpoints", None) or [])
    if checkpoints:
        first_cp = checkpoints[0]
        last_cp = checkpoints[-1]
        # Degenerate route (spawn lane == dest lane) — never early-arrive,
        # except detour: finish is a along-cap on the same obstacle edge.
        if (
            not allow_same_lane
            and _same_road_lane_index(first_cp, last_cp)
            and len(checkpoints) <= 2
        ):
            return False
        # Must be on the final checkpoint / final_lane road — not the approach.
        on_final = _same_road_lane_index(lane_idx, last_cp) or _same_road_lane_index(
            lane_idx, final_idx
        )
        if not on_final:
            return False
        if not _same_road_lane_index(first_cp, last_cp) and _same_road_lane_index(
            lane_idx, first_cp
        ):
            return False
    elif not _same_road_lane_index(lane_idx, final_idx):
        return False

    try:
        lane_len = float(final.length)
        # Use the vehicle's current lane coords (same road as final).
        long, _lat = lane.local_coordinates(vehicle.position)
    except Exception:
        return False

    target = min(cap, max(0.5, lane_len - 5.0))
    return float(long) >= (target - float(arrive_tol_m))


def _reposition_ego_before_lane_end(env, distance_before_end: float) -> bool:
    """Reposition the ego vehicle to a specific distance before the lane end.
    
    Args:
        env: The environment instance.
        distance_before_end: Distance in meters before lane end to place the vehicle.
        
    Returns True if repositioning succeeded, False otherwise.
    """
    try:
        vehicle = env.agent
        if vehicle is None:
            return False
        
        lane = vehicle.lane
        if lane is None:
            return False
        
        lane_length = lane.length
        # spawn_longitude is from lane START, so: lane_length - distance_before_end
        spawn_long = max(1.0, min(lane_length - distance_before_end, lane_length - 0.1))
        
        pos = lane.position(spawn_long, 0.0)
        heading = lane.heading_theta_at(spawn_long)
        
        vehicle.set_position(pos)
        vehicle.set_heading_theta(heading)
        
        # Update spawn_place so navigation uses the new position
        try:
            vehicle.spawn_place = pos.copy()
        except Exception:
            pass
        
        # Rebuild navigation from new position
        if hasattr(env, "_refresh_navigation_after_spawn"):
            env._refresh_navigation_after_spawn(lane)
        else:
            try:
                vehicle.reset_navigation(lane)
            except Exception:
                pass
        
        return True
    except Exception as e:
        print(f"[EgoReposition] Failed to reposition ego: {e}")
        return False



