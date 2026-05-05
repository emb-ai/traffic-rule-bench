"""Record ComprehensiveRuleExpertPolicy replays on benchmark scenes.

For each scene in a benchmark manifest, runs the expert through the scene
with MetaDrive's native RecordManager on, then:
  * dumps a ScenarioDescription pkl (ego+NPC+map+pedestrians+cyclists+TL)
  * writes a sidecar JSON with the sign, per-step violations, expert actions,
    and the original source row (so the scene can be rebuilt in-env).

Output is laid out as a ScenarioNet-compatible dataset via
`save_dataset()` so it can be replayed in `ScenarioEnv` out of the box,
or in our `TrafficSignEnv`/`TrafficSignSumoEnv` via expert_replay_inenv.py.

Usage:
    # one PDD code, 5 scenes:
    python expert_replay.py --code 2.5 --count 5

    # explicit manifest:
    python expert_replay.py \\
        --manifest benchmark_output/mini/2_5/real_manifest.jsonl \\
        --backend sumo --count 5 \\
        --output-dir benchmark_output/mini/2_5/expert

    # aggregate already-written per-sign outputs into global summary:
    python expert_replay.py --aggregate --preset mini
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_DIR = SCRIPT_PATH.parent
SCRIPTS_DIR = BENCHMARK_DIR.parent
PDD_BENCH_DIR = SCRIPTS_DIR.parent
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"
SCENES_ROOT = PDD_BENCH_DIR / "scenes"
for _p in (PDD_BENCH_DIR, METADRIVE_DIR, BENCHMARK_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# Legacy-layout fallback: per_sign_bench/benchmark_output/ preferred, but we also
# read from sibling benchmark/benchmark_output/ when per_sign_bench has none.
LEGACY_OUTPUT_DIR = SCRIPTS_DIR / "benchmark"


def _resolve_policy_cls(policy_name: str):
    """Lazy-import policy classes so we don't pull CARL/torch unless needed."""
    if policy_name == "comprehensive":
        from agents.policies.comprehensive_rule_expert import ComprehensiveRuleExpertPolicy
        return ComprehensiveRuleExpertPolicy
    if policy_name == "carl":
        from agents.policies.carl_sign_compliant import CarlSignCompliantPolicy
        return CarlSignCompliantPolicy
    if policy_name == "rule_compliant":
        from agents.policies.rule_compliant_expert import RuleCompliantExpertPolicy
        return RuleCompliantExpertPolicy
    if policy_name == "plant2":
        from agents.policies.plant2_sign_compliant import PlanT2SignCompliantPolicy
        return PlanT2SignCompliantPolicy
    raise ValueError(f"Unknown policy {policy_name!r}; "
                      "expected one of: comprehensive, carl, rule_compliant, plant2")


POLICY_NAMES = ("comprehensive", "carl", "rule_compliant", "plant2")


# ---------------------------------------------------------------------------
# RecordManager tolerance patch
# ---------------------------------------------------------------------------
# Between env.reset() finishing (RecordManager.after_reset clears reset_frame
# and current_frames to None) and the first env.step() (before_step creates a
# new current_frames), RecordManager has no active frame — but our
# TrafficSignManager.add_sign and sumo_env.reset both spawn sign objects in
# that window, which crashes on `current_frame.spawn_info` assertion.
#
# Sign objects are static — we don't need them in the recording (they're in
# the sidecar). So we monkey-patch RecordManager.add_spawn_info at module
# load to be a no-op when no frame is active. This is done once per process
# and only affects expert_replay.py / expert_replay_inenv.py processes.

_RM_PATCHED = False

def _patch_record_manager_once():
    global _RM_PATCHED
    if _RM_PATCHED:
        return
    try:
        from metadrive.manager.record_manager import RecordManager
        from metadrive.utils.utils import is_map_related_class
        from metadrive.constants import ObjectState
    except ImportError as exc:
        # MetaDrive not importable yet — try again after env is built.
        print(f"[warn] RecordManager patch deferred: {exc}")
        return

    def _tolerant_add_spawn_info(self, obj, object_class, kwargs):
        if is_map_related_class(object_class) or not self.engine.record_episode:
            return
        # If no frame is currently active (between reset and first step), skip
        # silently — this is the case for sign spawns that happen post-reset.
        if self.reset_frame is None and self.current_frames is None:
            return
        try:
            frame = self.current_frame
        except (TypeError, AttributeError):
            return
        name = obj.name
        if name in frame.spawn_info:
            return  # idempotent
        self._episode_obj_names.add(name)
        frame.spawn_info[name] = {
            ObjectState.CLASS: object_class,
            ObjectState.INIT_KWARGS: kwargs,
            ObjectState.NAME: name,
        }

    RecordManager.add_spawn_info = _tolerant_add_spawn_info

    # collect_objects_states iterates ALL engine.get_objects() on every step.
    # Traffic-sign objects are not map-related (so they're not filtered out),
    # but they have no physics body → `obj.get_state()` → `obj.velocity` →
    # `self._body.hasPythonTag(...)` crashes with 'NoneType has no attribute'.
    # Skip such bodyless objects safely.
    import copy as _copy

    def _tolerant_collect_objects_states(self):
        from metadrive.utils.utils import is_map_related_instance
        policy_mapping = self.engine.get_policies()
        frame = self.current_frame
        for name, obj in self.engine.get_objects().items():
            if is_map_related_instance(obj):
                continue
            if getattr(obj, "_body", None) is None:
                continue  # static / bodyless objects (e.g. traffic signs)
            try:
                frame.step_info[name] = obj.get_state()
            except Exception:
                continue
            if name in policy_mapping:
                try:
                    frame.policy_info[name] = policy_mapping[name].get_state()
                except Exception:
                    pass
        frame.agents = list(self.engine.agents.keys())
        frame._agent_to_object = _copy.deepcopy(self.engine.agent_manager._agent_to_object)
        frame._object_to_agent = _copy.deepcopy(self.engine.agent_manager._object_to_agent)

    RecordManager.collect_objects_states = _tolerant_collect_objects_states

    # add_policy_info: pedestrian_manager (and other managers) call
    # engine.add_policy() during reset (e.g. _spawn_due_tracks → _spawn_pedestrian
    # → add_policy(ScenarioNetLikeReplayPolicy)). At that point reset_frame may
    # be None and current_frames is None — accessing self.current_frame crashes
    # with `'NoneType' object is not subscriptable`. Skip silently when no
    # frame is active (the policy is still registered in engine.get_policies()
    # via add_policy itself; we only skip the recording side-effect).
    try:
        from metadrive.constants import PolicyState
        from metadrive.base_class.base_object import BaseObject
    except ImportError:
        PolicyState = None
        BaseObject = None

    def _tolerant_add_policy_info(self, name, policy_class, *args, **kwargs):
        if not self.engine.record_episode:
            return
        if self.reset_frame is None and self.current_frames is None:
            return
        try:
            frame = self.current_frame
        except (TypeError, AttributeError):
            return
        if name in frame.policy_spawn_info:
            return  # idempotent
        # Filter BaseObject args/kwargs (match original record_manager logic).
        filtered_args = []
        for arg in args:
            if BaseObject is not None and isinstance(arg, BaseObject):
                filtered_args.append(BaseObject)
            else:
                filtered_args.append(arg)
        filtered_kwargs = {}
        for k, v in kwargs.items():
            if BaseObject is not None and isinstance(v, BaseObject):
                filtered_kwargs[k] = BaseObject
            else:
                filtered_kwargs[k] = v
        if PolicyState is not None:
            frame.policy_spawn_info[name] = {
                PolicyState.POLICY_CLASS: policy_class,
                PolicyState.ARGS: filtered_args,
                PolicyState.KWARGS: filtered_kwargs,
                PolicyState.OBJ_NAME: name,
            }

    RecordManager.add_policy_info = _tolerant_add_policy_info
    _RM_PATCHED = True

# Apply at import so everything downstream (including env.reset calls inside
# expert_replay_inenv.py) is covered.
_patch_record_manager_once()


# ---------------------------------------------------------------------------
# Smoothness — port of run_benchmark_mini._compute_smoothness, applied per
# episode to ego step_vars (long_acc, lat_acc, yaw_rate, yaw_acc, long_jerk,
# jerk_mag). Frame thresholds are nuPlan p95 envelopes for safe driving.
# ---------------------------------------------------------------------------
def _compute_smoothness(step_vars: list, segment_len: int = 20) -> dict:
    if not step_vars:
        return {
            "smoothness_ratio": 0.0,
            "smooth_segments": 0,
            "total_segments": 0,
            "frame_smooth_ratio": 0.0,
        }

    def _frame_ok(v: dict) -> bool:
        return (
            -4.05 <= v["long_acc"] <= 2.40
            and abs(v["lat_acc"]) <= 4.89
            and abs(v["yaw_rate"]) <= 0.95
            and abs(v["yaw_acc"]) <= 1.93
            and abs(v["long_jerk"]) <= 4.13
            and abs(v["jerk_mag"]) <= 8.37
        )

    frame_flags = [_frame_ok(v) for v in step_vars]
    frame_smooth_ratio = float(np.mean(frame_flags)) if frame_flags else 0.0

    total_segments = max(0, len(step_vars) // segment_len)
    smooth_segments = 0
    for i in range(total_segments):
        seg = frame_flags[i * segment_len:(i + 1) * segment_len]
        if all(seg):
            smooth_segments += 1
    smoothness_ratio = (smooth_segments / total_segments) if total_segments > 0 else 0.0

    return {
        "smoothness_ratio": float(smoothness_ratio),
        "smooth_segments": int(smooth_segments),
        "total_segments": int(total_segments),
        "frame_smooth_ratio": float(frame_smooth_ratio),
    }


# ---------------------------------------------------------------------------
# Env builder — wraps replay_scenes env construction, flips record_episode on
# ---------------------------------------------------------------------------

def _build_env(row: dict, backend: str, max_steps: int,
                record_episode: bool = True, ego_policy_cls=None,
                render: bool = False):
    """Build an env for one scene.

    Defaults match the recording pipeline (record_episode=True, expert ego).
    For replay pass record_episode=False and ego_policy_cls=None so the env
    accepts external actions via env.step(action).
    """
    from factorized_space.agent_profile_bank import sample_one_profile, apply_profile_to_idm_class
    from factorized_space.benchmark_runner import install_npc_vehicle_type_hook

    install_npc_vehicle_type_hook()
    seed_for_profile = int(row.get("seed") or row.get("deterministic_seed") or 1)
    apply_profile_to_idm_class(sample_one_profile(seed_for_profile))

    if ego_policy_cls is None:
        # Default to expert policy only if caller didn't override.
        from agents.policies.comprehensive_rule_expert import ComprehensiveRuleExpertPolicy
        ego_policy_cls = ComprehensiveRuleExpertPolicy

    if backend in ("pgmap", "paired"):
        return _build_pgmap_env(row, max_steps, ego_policy_cls,
                                 record_episode=record_episode, render=render)
    elif backend == "sumo":
        return _build_sumo_env(row, max_steps, ego_policy_cls,
                                record_episode=record_episode, render=render)
    elif backend == "citymap":
        return _build_citymap_env(row, max_steps, ego_policy_cls,
                                    record_episode=record_episode, render=render)
    raise ValueError(f"Unknown backend {backend!r}")


def _build_pgmap_env(row: dict, max_steps: int, ego_policy_cls,
                      record_episode: bool = True, render: bool = False):
    from envs.traffic_sign_env import TrafficSignEnv
    from metadrive.component.pgblock.first_block import FirstPGBlock

    spawn_lane = int(row["spawn_lane_index"])
    is_ramp_merge = row.get("block_id") == "r" and row.get("route_intent") == "merge"
    is_ramp_exit = row.get("block_id") == "R" and row.get("route_intent") == "exit"
    if is_ramp_merge:
        spawn_lane_tuple = ("2r1_0_", "2r1_1_", 0)
    elif is_ramp_exit:
        spawn_lane_tuple = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, row["lane_num"] - 1)
    else:
        spawn_lane_tuple = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, spawn_lane)

    config = dict(
        start_seed=int(row.get("seed") or row.get("deterministic_seed") or 0),
        use_render=render,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.WARNING,
        traffic_density=0.1,                       # run_benchmark.py default
        horizon=max_steps,
        record_episode=record_episode,
        vehicle_config={"show_lidar": False},
        map_config={
            "type": "block_sequence",
            "config": row["block_sequence"],
            "lane_num": row["lane_num"],
            "lane_width": row["lane_width"],
        },
        random_spawn_lane_index=False,
        agent_configs={"default_agent": dict(
            use_special_color=True,
            spawn_lane_index=spawn_lane_tuple,
        )},
        # Drives sign placement inside TrafficSignEnv.reset() — symmetric with
        # SUMO. Manager has signs at step 0 (verifier + reward apply from t=0)
        # and TopDownRenderer caches their icons on first render() (GIFs show
        # signs).
        scene_row=row,
    )
    if ego_policy_cls is not None:
        config["agent_policy"] = ego_policy_cls
    return TrafficSignEnv(config)


def _build_sumo_env(row: dict, max_steps: int, ego_policy_cls,
                     record_episode: bool = True, render: bool = False):
    from envs.sumo_env import TrafficSignSumoEnv
    from envs.sumo_traffic_manager import SumoTrafficManager
    SumoTrafficManager.EGO_SAFE_RADIUS = 15

    # Resolve env params from manifest row, falling back to legacy defaults.
    # This matches replay_mini_new.py behavior so scenes are simulated
    # consistently across both entry points (esp. profile_traffic_density,
    # which can be 0.08-0.86 — hardcoding 0.1 would skew dest/crash rates).
    def _row_get(*keys):
        for k in keys:
            v = row.get(k)
            if v is not None:
                return v
        return None

    resolved_density = _row_get("traffic_density", "profile_traffic_density")
    if resolved_density is None:
        resolved_density = 0.1
    resolved_tl = _row_get("tl_speed_factor")
    if resolved_tl is None:
        resolved_tl = 20.0
    resolved_horizon = _row_get("horizon", "horizon_steps", "profile_horizon_steps")
    if resolved_horizon is None:
        resolved_horizon = max_steps

    # Sign placement offset along the lane. Mirrors replay_mini_new.py:
    # use row.sign_spawn_distance / row.distance_from_start, fall back to
    # meta.json next to the .net.xml. No floor — uses the actual recorded
    # distance from start (can be very small if the sign is at the start of
    # a short edge, which is fine — it matches the real OSM topology).
    def _resolve_sign_spawn_distance() -> float:
        d = row.get("sign_spawn_distance")
        if d is None:
            d = row.get("distance_from_start")
        if d is None:
            try:
                map_path = SCENES_ROOT / row["net_path"]
                meta_path = map_path.parent / "meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    d = float(meta.get("distance_from_start", 0.0) or 5.0)
            except Exception:
                d = 0.0
        return float(d or 0.0)

    resolved_sign_spawn = max(_resolve_sign_spawn_distance(), 30.0)

    # Spawn velocity (m/s). Priority:
    #   1. EGO_SPAWN_VELOCITY_MS env var (CLI override via --spawn-velocity-ms)
    #   2. row.spawn_velocity_ms / row.spawn_velocity from manifest
    #   3. 0.0
    import os as _os_sv
    resolved_spawn_velocity = _os_sv.environ.get("EGO_SPAWN_VELOCITY_MS")
    if resolved_spawn_velocity is not None:
        resolved_spawn_velocity = float(resolved_spawn_velocity)
    else:
        resolved_spawn_velocity = _row_get("spawn_velocity_ms", "spawn_velocity")
    if resolved_spawn_velocity is None:
        resolved_spawn_velocity = 0.0

    vehicle_config = {
        "show_lidar": False,
        "spawn_velocity": [resolved_spawn_velocity, 0],
        "spawn_velocity_car_frame": True,
    }

    config = dict(
        use_render=render,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.WARNING,
        map_name=str(SCENES_ROOT / row["net_path"]),
        sign_type=row["sign_code"],
        sign_spawn_distance=float(resolved_sign_spawn),
        # Tight nav range: ego only needs to traverse the sign zone
        # (intersection + 1-2 lanes past it). Default was 10/10 which leads
        # to BFS picking distant destinations across SUMO-graph cycles → ego
        # drives in loops, episodes hit horizon at 600 steps.
        min_route_hops_after_spawn=2,
        max_route_hops_after_spawn=3,
        traffic_density=float(resolved_density),
        tl_speed_factor=float(resolved_tl),
        horizon=int(resolved_horizon),
        record_episode=record_episode,
        num_scenarios=100000,
        vehicle_config=vehicle_config,
    )
    if ego_policy_cls is not None:
        config["agent_policy"] = ego_policy_cls
    if row.get("road_id"):
        config["vehicle_config"]["spawn_lane_index"] = row["road_id"]

    import os as _os
    if _os.environ.get("PER_SIGN_USE_DESTINATION") == "1" and row.get("destination_lane_id"):
        config["vehicle_config"]["destination"] = row["destination_lane_id"]
    if "spawn_lane_num" in row:
        config["spawn_lane_num"] = int(row["spawn_lane_num"])

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


def _build_citymap_env(row: dict, max_steps: int, ego_policy_cls,
                        record_episode: bool = True, render: bool = False):

    from citymap_space.citymap_env import CityMapTrafficSignEnv
    map_seed = int(row.get("map_seed") or row.get("seed") or 0)
    config = dict(
        start_seed=map_seed,
        num_scenarios=1,
        use_render=render,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.WARNING,
        map=int(row.get("n_blocks", 8)),
        traffic_density=0.1,
        horizon=max_steps,
        record_episode=record_episode,
        vehicle_config={"show_lidar": False},
    )
    if ego_policy_cls is not None:
        config["agent_policy"] = ego_policy_cls
    return CityMapTrafficSignEnv(config)


# ---------------------------------------------------------------------------
# Record one scene
# ---------------------------------------------------------------------------

def _ego_at_fault_for_crash(ego, engine, contact_dist: float = 4.0) -> bool:
    """Heuristic ego-vs-NPC crash attribution (port of replay_mini_new helper).

    Returns True iff a colliding NPC is in ego's forward half AND ego speed > 0.5
    km/h — meaning ego drove into it. Otherwise NPC fault.
    """
    import math as _m
    try:
        ego_pos = ego.position
        ego_heading = ego.heading_theta
        ego_speed_kmh = float(getattr(ego, "speed_km_h", 0.0))
    except Exception:
        return True
    cos_h, sin_h = _m.cos(ego_heading), _m.sin(ego_heading)
    try:
        objs = engine.get_objects(lambda o: o is not ego).values()
    except Exception:
        return True
    from metadrive.component.vehicle.base_vehicle import BaseVehicle
    for obj in objs:
        if not isinstance(obj, BaseVehicle):
            continue
        try:
            dx = obj.position[0] - ego_pos[0]
            dy = obj.position[1] - ego_pos[1]
            dist = _m.hypot(dx, dy)
        except Exception:
            continue
        if dist > contact_dist:
            continue
        rel_x = cos_h * dx + sin_h * dy
        if rel_x > 0.0 and ego_speed_kmh > 0.5:
            return True
    return False


def _extract_sign_info(env) -> list[dict]:
    """Snapshot signs currently placed in the env (after reset + sign placement)."""
    signs = []
    sign_mgr = getattr(env.engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return signs
    for s in sign_mgr.signs:
        lane = getattr(s, "lane", None)
        lane_index = list(getattr(lane, "index", ())) if lane is not None else None
        pos = None
        try:
            pos = [float(s.position[0]), float(s.position[1])]
        except Exception:
            pass
        signs.append({
            "sign_class": type(s).__name__,
            "lane_index": lane_index,
            "longitudinal_offset": float(getattr(s, "longitudinal_offset", 0.0)),
            "lateral_offset": float(getattr(s, "lateral_offset", 0.0)),
            "position_world": pos,
        })
    return signs


def record_expert_replay(
    row: dict, backend: str,
    output_pkl: Path, output_sidecar: Path,
    max_steps: int = 500,
    save_gif: Optional[Path] = None,
    verbose: bool = False,
    policy_cls: Optional[type] = None,
    ego_params: Optional[dict] = None,
    extra_sidecar: Optional[dict] = None,
    sample_ego_velocity: bool = False,
) -> dict:
    """Run expert on one scene; write pkl (ScenarioDescription) + sidecar JSON.
    Returns a metrics dict suitable for expert_replays.jsonl.

    Args:
        policy_cls: ego policy class. None → ComprehensiveRuleExpertPolicy
            (preserves prior behaviour).
        ego_params: if None, calls apply_ego_defaults() on the IDM ego policy
            (legacy default). If a dict, calls apply_ego_sampled() with it
            instead — used for nuPlan-sampled IDM variants. Ignored for
            non-IDM policies (carl/rule_compliant), where the dict, if any,
            is still recorded in the sidecar for traceability but has no
            effect on policy behaviour.
        extra_sidecar: optional dict merged into the sidecar JSON (e.g.,
            policy/variant identifiers).
    """
    from metadrive.scenario.utils import convert_recorded_scenario_exported
    from factorized_space.ego_defaults import apply_ego_defaults, apply_ego_sampled
    from traffic_signs.detour_obstacle import spawn_detour_obstacle  # noqa: side-effects
    from factorized_space.benchmark_runner import (
        SIGN_CLASS_MAP, BIKE_RELATED_SIGNS, DETOUR_KEYS, RESTRICTED_BEGIN_KEYS,
        LANE_CHANGE_KEYS, BEGIN_TO_END,
        _get_route_lanes, _pick_route_lane, _pick_rightmost_lane,
        _pick_detour_lane, _pick_lane_for_lane_change,
        _override_route_intent, _spawn_cyclists_on_lane,
    )

    seed = int(row.get("seed") or row.get("deterministic_seed") or 0)
    np.random.seed(seed)
    random.seed(seed)

    import os as _os_sv2
    if sample_ego_velocity and _os_sv2.environ.get("EGO_SPAWN_VELOCITY_MS") is None:
        from factorized_space.agent_profile_bank import sample_spawn_velocity as _svs
        row = dict(row, spawn_velocity_ms=_svs(seed))

    env = _build_env(row, backend, max_steps, ego_policy_cls=policy_cls)
    veh = None
    violations_timeline: list[dict] = []
    expert_actions: list[list[float]] = []
    violations_by_class: Counter = Counter()
    info: dict = {}
    step = 0
    total_reward = 0.0
    prev_violated_keys: set = set()
    reason = None

    # Smoothness step_vars accumulator — filled inside the rollout loop.
    smoothness_step_vars: list[dict] = []
    _prev_speed_mps: Optional[float] = None
    _prev_yaw: Optional[float] = None
    _prev_long_acc: Optional[float] = None
    _prev_lat_acc: Optional[float] = None
    _prev_yaw_rate: Optional[float] = None

    # Make sure the RecordManager patch is in place — module-load attempt can
    # fail if MetaDrive wasn't yet importable. By now env construction has
    # imported everything, so the patch should stick.
    _patch_record_manager_once()

    try:
        # SUMO env has `num_scenarios=100000`, so env seed must be < 100000.
        # Our `seed` may be a hash larger than that — clamp it for reset while
        # keeping the full value in sidecar for reproducibility.
        if backend == "sumo":
            sign_id_int = int(row.get("sign_id", 0))
            var_idx = int(row.get("var_idx", 0))
            env_seed = (sign_id_int + var_idx) % 100000
        elif backend == "citymap":
            env_seed = int(row.get("map_seed") or row.get("seed") or 0)
        else:
            env_seed = seed
        _, reset_info = env.reset(seed=env_seed)
        veh = env.vehicle

        # Capture ego initial speed right after reset for sidecar verification.
        try:
            initial_speed_mps = float(getattr(veh, "speed", 0.0))
            initial_speed_kmh = float(getattr(veh, "speed_km_h", initial_speed_mps * 3.6))
            print(f"[ego] initial speed: {initial_speed_mps:.2f} m/s "
                  f"({initial_speed_kmh:.1f} km/h)  scene={row.get('scene_id')}")
        except Exception:
            initial_speed_mps = 0.0
            initial_speed_kmh = 0.0

        # PGMap signs are now placed inside TrafficSignEnv.reset() (driven by
        # config["scene_row"]). If placement failed there, propagate the
        # failure reason exactly like the legacy post-reset block did.
        if backend == "pgmap" and isinstance(reset_info, dict):
            if reset_info.get("sign_placement_ok") is False:
                reason = reset_info.get("sign_placement_reason") or "sign_placement_failed"

        # Ego isolation already implicit via agent_policy; still apply defaults
        # (or sampled overrides) to the ego policy instance so IDM base attrs
        # (NORMAL_SPEED etc.) on ego match run_benchmark.py conventions.
        try:
            ego_policy = env.engine.get_policy(veh.id)
            if ego_policy is not None:
                if ego_params is not None:
                    apply_ego_sampled(ego_policy, ego_params)
                else:
                    apply_ego_defaults(ego_policy)
        except Exception:
            pass

        sign_key = row.get("sign_type")

        # RecordManager is in a "no active frame" state between env.reset() and
        # the first env.step() — attempting to spawn objects here crashes on
        # RecordManager.add_spawn_info (current_frames is None). Temporarily
        # disable add_spawn_info for the duration of our sign placement.
        # Also: traffic-sign objects have no physics body, so their `get_state`
        # would crash during `record_manager.collect_objects_states` on every
        # step — we detach them from engine._spawned_objects after spawn so
        # RecordManager never iterates over them. They remain live in
        # traffic_sign_manager.signs (rendered/placed correctly) and in the
        # sidecar JSON.
        _rm = getattr(env.engine, "record_manager", None)
        _rm_original_add_spawn = None
        _signs_pre = set(env.engine._spawned_objects.keys())
        if _rm is not None:
            _rm_original_add_spawn = _rm.add_spawn_info
            _rm.add_spawn_info = lambda *a, **kw: None

        # PGMap sign placement now happens inside TrafficSignEnv.reset() (via
        # config["scene_row"]). The block below remains for paired/citymap
        # backends if they need explicit post-reset placement; for pgmap it is
        # intentionally a no-op (signs are already in sign_mgr).

        # Detach signs (and any other static objects just spawned) from
        # engine._spawned_objects so record_manager.collect_objects_states
        # doesn't iterate over them (they have no physics body → crash).
        # We keep references via sign_mgr so they stay alive and rendered.
        _signs_post = set(env.engine._spawned_objects.keys())
        _sign_ids_recorded = _signs_post - _signs_pre
        for _sid in _sign_ids_recorded:
            obj = env.engine._spawned_objects.get(_sid)
            body = getattr(obj, "_body", None) if obj is not None else None
            if obj is not None and body is None:
                env.engine._spawned_objects.pop(_sid, None)

        # Restore RecordManager.add_spawn_info so NPC spawns during stepping
        # get captured into the recording.
        if _rm is not None and _rm_original_add_spawn is not None:
            _rm.add_spawn_info = _rm_original_add_spawn

        if reason is not None:
            return _fail_result(row, backend, output_pkl, output_sidecar, reason)

        sign_info_snapshot = _extract_sign_info(env)
        sign_mgr = env.engine.traffic_sign_manager

        # Get the expert policy instance — we must call policy.act() explicitly
        # and pass the result to env.step(). agent_policy in config does NOT
        # auto-drive ego; env.step(action) uses whatever action we pass.
        ego_policy_instance = env.engine.get_policy(veh.id)

        for step in range(max_steps):
            # Expert computes action:
            try:
                action = ego_policy_instance.act(veh.name)
                action = [float(action[0]), float(action[1])]
            except Exception:
                action = [0.0, 0.0]

            expert_actions.append(action)
            _, reward, term, trunc, info = env.step(action)
            try:
                total_reward += float(reward)
            except Exception:
                pass

            # Smoothness step_vars: long_acc/lat_acc/yaw_rate/yaw_acc/long_jerk/jerk_mag
            # Mirrors run_benchmark_mini.py:757-883. dt comes from physics step size.
            try:
                import math as _math
                dt = float(env.engine.global_config.get("physics_world_step_size", 0.02))
                speed_mps = float(getattr(veh, "speed", 0.0))
                yaw = float(getattr(veh, "heading_theta", 0.0))
                if _prev_speed_mps is not None and _prev_yaw is not None and dt > 0:
                    long_acc = (speed_mps - _prev_speed_mps) / dt
                    yaw_delta = (yaw - _prev_yaw + _math.pi) % (2 * _math.pi) - _math.pi
                    yaw_rate = yaw_delta / dt
                    lat_acc = speed_mps * yaw_rate
                    if (_prev_long_acc is not None and _prev_lat_acc is not None
                            and _prev_yaw_rate is not None):
                        long_jerk = (long_acc - _prev_long_acc) / dt
                        lat_jerk = (lat_acc - _prev_lat_acc) / dt
                        yaw_acc = (yaw_rate - _prev_yaw_rate) / dt
                        jerk_mag = float(_math.sqrt(long_jerk * long_jerk
                                                     + lat_jerk * lat_jerk))
                        smoothness_step_vars.append({
                            "long_acc": float(long_acc),
                            "lat_acc": float(lat_acc),
                            "yaw_rate": float(yaw_rate),
                            "yaw_acc": float(yaw_acc),
                            "long_jerk": float(long_jerk),
                            "jerk_mag": float(jerk_mag),
                        })
                    _prev_long_acc = long_acc
                    _prev_lat_acc = lat_acc
                    _prev_yaw_rate = yaw_rate
                _prev_speed_mps = speed_mps
                _prev_yaw = yaw
            except Exception:
                pass

            # Violations delta
            current_keys: set = set()
            for s_obj, violated in sign_mgr.check_all_violations(veh):
                if not violated:
                    continue
                key = (type(s_obj).__name__, step)
                if type(s_obj).__name__ not in prev_violated_keys:
                    violations_timeline.append({
                        "step": step,
                        "sign_class": type(s_obj).__name__,
                        "rule": getattr(s_obj, "get_rule_description", lambda: "")() or "",
                    })
                    violations_by_class[type(s_obj).__name__] += 1
                current_keys.add(type(s_obj).__name__)
            prev_violated_keys = current_keys

            if save_gif:
                try:
                    env.render(mode="top_down", film_size=(2400, 2400), scaling=12.0,
                                screen_size=(800, 800), semantic_map=True,
                                semantic_broken_line=True,
                                draw_target_vehicle_trajectory=True,
                                target_agent_heading_up=True,
                                screen_record=True, window=False)
                except Exception:
                    pass

            if term or trunc:
                break

        # Dump episode data.
        # Try ScenarioDescription conversion first (works for PGMap); fallback
        # to raw FrameInfo format (works for any env; replayable via
        # env.config["replay_episode"] = episode_info).
        output_pkl.parent.mkdir(parents=True, exist_ok=True)
        scenario_desc = None
        try:
            raw_frames = env.engine.record_manager.episode_info
            scenario_desc = convert_recorded_scenario_exported(raw_frames, to_dict=True)
        except Exception:
            scenario_desc = None

        if scenario_desc is not None:
            with open(output_pkl, "wb") as f:
                pickle.dump(scenario_desc, f)
        else:
            # Fallback: save raw episode_info (FrameInfo list) — replay via
            # env.config["replay_episode"]=pickle.load(f); env.reset().
            try:
                episode_info = env.engine.dump_episode(str(output_pkl))
                scenario_desc = episode_info  # non-None → marks valid
            except Exception as exc:
                reason = f"dump_episode: {type(exc).__name__}: {exc}"
                scenario_desc = None

        # GIF
        if save_gif:
            try:
                save_gif.parent.mkdir(parents=True, exist_ok=True)
                if hasattr(env, "top_down_renderer") and env.top_down_renderer is not None:
                    env.top_down_renderer.generate_gif(str(save_gif), duration=40)
            except Exception:
                pass

        route_completion = 0.0
        try:
            route_completion = float(getattr(veh.navigation, "route_completion", 0.0))
        except Exception:
            pass

        crashed_flag = bool(info.get("crash", False))
        crash_attribution = None
        if crashed_flag or bool(getattr(veh, "crash_vehicle", False)):
            crash_attribution = "ego" if _ego_at_fault_for_crash(veh, env.engine) else "npc"

        smoothness = _compute_smoothness(smoothness_step_vars)

        metrics = {
            "arrived_dest": bool(info.get("arrive_dest", False)),
            "crashed": crashed_flag,
            "crash_attribution": crash_attribution,
            "crashed_ego_fault": crashed_flag and crash_attribution == "ego",
            "crashed_npc_fault": crashed_flag and crash_attribution == "npc",
            "out_of_road": bool(info.get("out_of_road", False)),
            "final_step": step + 1,
            "total_violations": int(sum(violations_by_class.values())),
            "violations_by_class": dict(violations_by_class),
            "route_completion": round(route_completion, 4),
            "total_reward": round(total_reward, 4),
            "smoothness_ratio": smoothness["smoothness_ratio"],
            "frame_smooth_ratio": smoothness["frame_smooth_ratio"],
            "initial_speed_mps": float(initial_speed_mps),
            "initial_speed_kmh": float(initial_speed_kmh),
            "smooth_segments": smoothness["smooth_segments"],
            "total_segments": smoothness["total_segments"],
        }

        # Sidecar
        sidecar = {
            "scene_id": row.get("scene_id") or row.get("flat_index") or f"scene_{seed}",
            "backend": backend,
            "pdd_code": row.get("pdd_code") or row.get("sign_code") or row.get("sign_type"),
            "sign_key": row.get("sign_type"),
            "source_row": row,
            "env_config_summary": {
                "map_name": row.get("net_path"),
                "road_id": row.get("road_id"),
                "spawn_lane_num": row.get("spawn_lane_num"),
                "block_sequence": row.get("block_sequence"),
                "lane_num": row.get("lane_num"),
                "lane_width": row.get("lane_width"),
                "traffic_density": 0.1,
                "horizon": max_steps,
                "seed": seed,
            },
            "signs": sign_info_snapshot,
            "expert_actions": expert_actions,
            "violations_timeline": violations_timeline,
            "smoothness_step_vars": smoothness_step_vars,
            "metrics": metrics,
            "dump_error": reason,
            "ego_idm_params": (ego_params if ego_params is not None
                               else "DEFAULT_EGO_PARAMS"),
        }
        if extra_sidecar:
            sidecar.update(extra_sidecar)
        output_sidecar.parent.mkdir(parents=True, exist_ok=True)
        with open(output_sidecar, "w") as f:
            json.dump(sidecar, f, default=str)

        return {
            "scene_id": sidecar["scene_id"],
            "backend": backend,
            "sign_type": sidecar["sign_key"],
            "sign_code": sidecar["pdd_code"],
            "seed": seed,
            "pkl_path": str(output_pkl.relative_to(output_pkl.parent.parent))
                          if scenario_desc is not None else None,
            "sidecar_path": str(output_sidecar.relative_to(output_sidecar.parent.parent)),
            "gif_path": str(save_gif) if save_gif else None,
            "valid": bool(scenario_desc is not None),
            **metrics,
            "failure_reason": reason,
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def _fail_result(row, backend, output_pkl, output_sidecar, reason):
    return {
        "scene_id": row.get("scene_id") or row.get("flat_index"),
        "backend": backend,
        "valid": False,
        "failure_reason": reason,
    }


# ---------------------------------------------------------------------------
# Batch + aggregation
# ---------------------------------------------------------------------------

def _iter_manifest_rows(path: Path, count: int, start: int):
    with open(path) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if count is not None and len(out := getattr(_iter_manifest_rows, "_buf", [])) >= count:
                return
            try:
                yield json.loads(line)
            except Exception:
                continue


def run_batch(manifest_path: Path, backend: str, output_dir: Path,
              count: int, start: int, max_steps: int,
              save_gifs: bool = False,
              sample_ego_velocity: bool = False) -> dict:
    output_dir = Path(output_dir)
    replays_dir = output_dir / "replays"
    gifs_dir = output_dir / "gifs" if save_gifs else None
    replays_dir.mkdir(parents=True, exist_ok=True)
    if gifs_dir:
        gifs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(manifest_path) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if count is not None and len(rows) >= count:
                break
    if not rows:
        return {"error": f"no rows in {manifest_path}", "n": 0}

    index_lines = []
    metrics_accum = {"total": 0, "valid": 0, "arrived": 0, "crashed": 0, "out_of_road": 0,
                      "violations": 0, "violations_by_class": Counter()}

    t0 = time.time()
    for i, row in enumerate(rows):
        scene_id = row.get("scene_id") or row.get("flat_index") or f"scene_{i}"
        var_idx = row.get("var_idx")
        seed_val = row.get("seed") or row.get("deterministic_seed")
        uid = scene_id
        if var_idx is not None:
            uid = f"{uid}_v{var_idx}"
        if seed_val is not None:
            uid = f"{uid}_s{seed_val}"
        pkl = replays_dir / f"sd_{uid}.pkl"
        sidecar = replays_dir / f"{uid}.meta.json"
        gif = (gifs_dir / f"{uid}.gif") if gifs_dir else None
        print(f"[{i+1}/{len(rows)}] {scene_id} backend={backend}", flush=True)
        result = record_expert_replay(row, backend, pkl, sidecar,
                                        max_steps=max_steps, save_gif=gif,
                                        sample_ego_velocity=sample_ego_velocity)
        index_lines.append(result)
        metrics_accum["total"] += 1
        if result.get("valid"):
            metrics_accum["valid"] += 1
            metrics_accum["arrived"] += int(result.get("arrived_dest", False))
            metrics_accum["crashed"] += int(result.get("crashed", False))
            metrics_accum["out_of_road"] += int(result.get("out_of_road", False))
            metrics_accum["violations"] += int(result.get("total_violations", 0))
            for k, v in (result.get("violations_by_class") or {}).items():
                metrics_accum["violations_by_class"][k] += v

    # Per-sign summary
    n = max(metrics_accum["valid"], 1)
    summary = {
        "pdd_code": (rows[0].get("pdd_code") or rows[0].get("sign_code")
                      or rows[0].get("sign_type")),
        "backend": backend,
        "n_episodes": metrics_accum["total"],
        "n_valid": metrics_accum["valid"],
        "dest_rate": round(metrics_accum["arrived"] / n, 3),
        "crash_rate": round(metrics_accum["crashed"] / n, 3),
        "out_of_road_rate": round(metrics_accum["out_of_road"] / n, 3),
        "avg_violations": round(metrics_accum["violations"] / n, 3),
        "violations_by_class": dict(metrics_accum["violations_by_class"]),
        "wall_time_s": round(time.time() - t0, 1),
    }

    with open(output_dir / "expert_replays.jsonl", "w") as f:
        for r in index_lines:
            f.write(json.dumps(r, default=str) + "\n")
    with open(output_dir / "expert_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Write dataset_summary.pkl for ScenarioEnv compatibility
    try:
        _write_dataset_summary(replays_dir, index_lines)
    except Exception as exc:
        print(f"  [warn] dataset_summary write failed: {exc}")

    print(f"\n{summary}")
    return summary


def _write_dataset_summary(replays_dir: Path, index_lines: list[dict]):
    """Write ScenarioNet dataset_summary.pkl so ScenarioEnv can read this dir."""
    summary = {}
    for row in index_lines:
        if not row.get("valid"):
            continue
        pkl_name = f"sd_{row['scene_id']}.pkl"
        if not (replays_dir / pkl_name).exists():
            continue
        summary[row["scene_id"]] = pkl_name
    with open(replays_dir / "dataset_summary.pkl", "wb") as f:
        pickle.dump(summary, f)


def aggregate_global(preset: str = "mini",
                      output_root: Optional[Path] = None) -> dict:
    """Scan benchmark_output/<preset>/*/expert/expert_summary.json and combine."""
    if output_root is None:
        for candidate in (BENCHMARK_DIR / "benchmark_output" / preset,
                           LEGACY_OUTPUT_DIR / "benchmark_output" / preset):
            if candidate.exists():
                output_root = candidate
                break
    if output_root is None or not output_root.exists():
        return {"error": f"no benchmark_output for preset={preset}"}

    per_sign = {}
    totals = Counter()
    violations_global = Counter()
    for code_dir in sorted(output_root.iterdir()):
        if not code_dir.is_dir():
            continue
        summary_path = code_dir / "expert" / "expert_summary.json"
        if not summary_path.exists():
            continue
        s = json.load(open(summary_path))
        per_sign[code_dir.name.replace("_", ".")] = s
        totals["n_episodes"] += s.get("n_episodes", 0)
        totals["n_valid"] += s.get("n_valid", 0)
        for k, v in s.get("violations_by_class", {}).items():
            violations_global[k] += v

    out = {
        "preset": preset,
        "total_signs": len(per_sign),
        "total_episodes": totals["n_episodes"],
        "total_valid": totals["n_valid"],
        "violations_top10": violations_global.most_common(10),
        "per_sign": per_sign,
    }
    with open(output_root / "expert_global_summary.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


# ---------------------------------------------------------------------------
# Multi-variant trajectory recording: runs one policy with default + N
# nuPlan-sampled IDM variants, writes the new by_sign/by_scene layout, and
# appends one row per (scene × variant) to all_runs.jsonl. After all policies
# are recorded for a run, --build-oracle-manifest aggregates the jsonl rows
# into oracle_manifest.jsonl (best variant per scene_uid).
# ---------------------------------------------------------------------------


def _scene_uid(row: dict) -> str:
    sid = row.get("scene_id") or row.get("flat_index") or "scene"
    seed = row.get("seed") or row.get("deterministic_seed") or 0
    lane = row.get("spawn_lane_num") or row.get("spawn_lane_index") or 0
    # Include var_idx so PGMAP rows replicated with different var_idx
    # (different NPC profile) don't collide on the same output dir.
    var_idx = row.get("var_idx")
    if var_idx is not None:
        return f"{sid}_lane{lane}_seed{seed}_v{int(var_idx)}"
    return f"{sid}_lane{lane}_seed{seed}"


def _ego_score_key(record: dict) -> tuple:
    """Lower-is-better key for oracle selection.

    Mirrors the priority used in /tmp/analyze_winners.py:
        reached_dest desc → ego-fault crash asc → out_of_road asc →
        any crash asc → violations asc → total_reward desc.
    """
    reached = bool(record.get("arrived_dest", False))
    ego_crash = 1 if (record.get("crashed", False)
                       and record.get("crash_attribution") == "ego") else 0
    any_crash = 1 if record.get("crashed", False) else 0
    oor = 1 if record.get("out_of_road", False) else 0
    viol = int(record.get("total_violations", 0))
    reward = float(record.get("total_reward", 0.0))
    return (-int(reached), ego_crash, oor, any_crash, viol, -reward)


def run_batch_multi_variant(
    manifest_path: Path, backend: str, run_root: Path,
    sign_slug: str, policy_name: str,
    count: int, start: int, max_steps: int,
    extra_samples: int = 0, sample_seed_base: int = 0,
    save_gifs: bool = False,
    sample_ego_velocity: bool = False,
    skip_variants: Optional[set[str]] = None,
) -> dict:
    """Record one policy across all scenes in the manifest, with `1 + extra_samples`
    variants per scene for IDM policies. Appends rows to <run_root>/all_runs.jsonl.

    Output layout under run_root:
        by_sign/<sign_slug>/by_scene/<scene_uid>/<variant_id>/replay.pkl
        by_sign/<sign_slug>/by_scene/<scene_uid>/<variant_id>/replay.json
        all_runs.jsonl   (appended; one row per (scene × variant))
    """
    from factorized_space.ego_defaults import sample_ego_params

    def _episode_already_recorded(pkl_path: Path, sidecar_path: Path,
                                   gif_path: Optional[Path],
                                   require_gif: bool) -> bool:
        if not (pkl_path.exists() and pkl_path.stat().st_size > 0):
            return False
        if not (sidecar_path.exists() and sidecar_path.stat().st_size > 0):
            return False
        if require_gif and gif_path is not None:
            if not (gif_path.exists() and gif_path.stat().st_size > 0):
                return False
        return True


    policy_cls = _resolve_policy_cls(policy_name)
    is_idm_based = (policy_name == "comprehensive")
    n_extra = extra_samples if is_idm_based else 0

    rows = []
    with open(manifest_path) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if count is not None and len(rows) >= count:
                break
    if not rows:
        return {"error": f"no rows in {manifest_path}", "n": 0}

    sign_root = run_root / "by_sign" / sign_slug / "by_scene"
    sign_root.mkdir(parents=True, exist_ok=True)
    all_runs_path = run_root / "all_runs.jsonl"

    variants: list[tuple[str, Optional[dict]]] = [("default", None)]
    for k in range(1, n_extra + 1):
        seed_k = sample_seed_base + k * 1000003
        variants.append((f"s{k}", sample_ego_params(seed_k)))

    # Optionally drop variants the caller knows are already valid elsewhere
    # (used by the optim/scene_dispatcher.py to avoid recomputing IDM variants
    # that already have a valid replay for this scene_id at some other var_idx).
    # Accepts either bare ids ("default", "s2") or fully-qualified
    # ("comprehensive_default", "comprehensive_s2"). Hierarchy is preserved
    # because seed_base × position determines params — skipping one variant
    # does not shift the seeds of the others.
    if skip_variants:
        _skip_ids = set()
        for v in skip_variants:
            _skip_ids.add(v)
            if v.startswith(f"{policy_name}_"):
                _skip_ids.add(v[len(policy_name) + 1:])
        variants = [(vid, p) for (vid, p) in variants if vid not in _skip_ids]
        if not variants:
            return {"info": "all variants skipped via --skip-variants",
                    "n": 0, "skipped_existing": 0}

    n_total = len(rows) * len(variants)
    t0 = time.time()
    counter = 0
    valid_count = 0
    skipped_existing = 0

    # Resume mode: skip episodes whose pkl + sidecar are already on disk and
    # non-empty. Activated via env var PDD_BENCH_RESUME=1 (set by callers like
    # run_all_experts_parallel.sh after a tmux/nohup kill).
    resume_mode = os.environ.get("PDD_BENCH_RESUME", "0") == "1"
    if resume_mode:
        print(f"[resume] PDD_BENCH_RESUME=1 — пропуск уже записанных эпизодов",
              flush=True)

    with open(all_runs_path, "a") as ao:
        for i, row in enumerate(rows):
            scene_uid = _scene_uid(row)
            for variant_id, params in variants:
                counter += 1
                variant_full = f"{policy_name}_{variant_id}" if is_idm_based else policy_name
                out_dir = sign_root / scene_uid / variant_full
                pkl_path = out_dir / "replay.pkl"
                sidecar_path = out_dir / "replay.json"
                gif_path = (out_dir / "replay.gif") if save_gifs else None

                tag = (f"[{counter}/{n_total}] sign={sign_slug} scene={scene_uid} "
                       f"variant={variant_full}")

                if resume_mode and _episode_already_recorded(pkl_path, sidecar_path,
                                                             gif_path, save_gifs):
                    print(f"{tag}  [skip: already recorded]", flush=True)
                    skipped_existing += 1
                    continue

                print(tag, flush=True)

                try:
                    result = record_expert_replay(
                        row, backend, pkl_path, sidecar_path,
                        max_steps=max_steps, save_gif=gif_path,
                        policy_cls=policy_cls,
                        ego_params=params,
                        sample_ego_velocity=sample_ego_velocity,
                        extra_sidecar={
                            "policy": policy_name,
                            "variant": variant_id,
                            "sign_slug": sign_slug,
                            "scene_uid": scene_uid,
                        },
                    )
                except Exception as exc:
                    import traceback
                    print(f"  [ERROR] {type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    result = {
                        "scene_id": row.get("scene_id"),
                        "valid": False,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }

                row_out = dict(result)
                row_out.update({
                    "policy": policy_name,
                    "variant": variant_id,
                    "sign_slug": sign_slug,
                    "scene_uid": scene_uid,
                    "pkl_path": str(pkl_path) if result.get("valid") else None,
                    "sidecar_path": str(sidecar_path),
                    "ego_idm_params": (params if params is not None
                                       else "DEFAULT_EGO_PARAMS"),
                })
                ao.write(json.dumps(row_out, default=str) + "\n")
                ao.flush()
                if result.get("valid"):
                    valid_count += 1

    return {
        "sign_slug": sign_slug,
        "policy": policy_name,
        "n_total": n_total,
        "n_valid": valid_count,
        "wall_time_s": round(time.time() - t0, 1),
    }


def build_oracle_manifest(run_root: Path) -> dict:
    """Read all_runs.jsonl, pick best variant per scene_uid, write oracle_manifest.jsonl."""
    runs_path = run_root / "all_runs.jsonl"
    if not runs_path.exists():
        return {"error": f"no all_runs.jsonl under {run_root}"}

    by_scene: dict[str, list[dict]] = {}
    with open(runs_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("valid"):
                continue
            uid = r.get("scene_uid") or _scene_uid(r)
            by_scene.setdefault(uid, []).append(r)

    out_lines = []
    for uid in sorted(by_scene):
        candidates = by_scene[uid]
        keyed = sorted(
            candidates,
            key=lambda c: (_ego_score_key(c), c.get("policy", ""), c.get("variant", "")),
        )
        winner = keyed[0]
        out_lines.append({
            "scene_uid": uid,
            "sign_slug": winner.get("sign_slug"),
            "scene_id": winner.get("scene_id"),
            "policy": winner.get("policy"),
            "variant": winner.get("variant"),
            "winner_id": (f"{winner.get('policy')}_{winner.get('variant')}"
                          if winner.get("policy") == "comprehensive"
                          else winner.get("policy")),
            "winning_pkl": winner.get("pkl_path"),
            "winning_sidecar": winner.get("sidecar_path"),
            "score_key": list(_ego_score_key(winner)),
            "outcome": (
                "reached" if winner.get("arrived_dest") else
                "crash_ego" if (winner.get("crashed") and winner.get("crash_attribution") == "ego") else
                "crash_npc" if winner.get("crashed") else
                "out_of_road" if winner.get("out_of_road") else
                "timeout"
            ),
            "n_candidates": len(candidates),
        })

    manifest_path = run_root / "oracle_manifest.jsonl"
    with open(manifest_path, "w") as f:
        for r in out_lines:
            f.write(json.dumps(r, default=str) + "\n")
    return {
        "n_scenes": len(out_lines),
        "manifest_path": str(manifest_path),
    }


# ---------------------------------------------------------------------------
# Output-dir resolution
# ---------------------------------------------------------------------------

def _resolve_code_dir(code: str, preset: str) -> Optional[Path]:
    code_slug = code.replace(".", "_")
    for candidate in (BENCHMARK_DIR / "benchmark_output" / preset / code_slug,
                       LEGACY_OUTPUT_DIR / "benchmark_output" / preset / code_slug):
        if candidate.exists():
            return candidate
    return None


def _pick_manifest_and_backend(code_dir: Path, backend: str) -> tuple[Path, str]:
    """Prefer PGMap synthetic → paired → CityMap → SUMO catalog, unless user chose one."""
    candidates = [
        ("pgmap",  code_dir / "pgmap_materialized.jsonl"),
        ("pgmap",  code_dir / "synthetic_manifest.jsonl"),
        ("paired", code_dir / "paired_materialized.jsonl"),
        ("citymap",code_dir / "citymap_materialized.jsonl"),
        ("sumo",   code_dir / "sumo" / "sumo_manifest.jsonl"),
        ("sumo",   code_dir / "real_manifest.jsonl"),
    ]
    if backend != "auto":
        for b, p in candidates:
            if b == backend and p.exists() and p.stat().st_size > 0:
                return p, b
        raise FileNotFoundError(f"No {backend} manifest under {code_dir}")
    for b, p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p, b
    raise FileNotFoundError(f"No manifests at all under {code_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Record expert replays for one PDD code")
    parser.add_argument("--manifest", type=str, help="Path to *.jsonl manifest")
    parser.add_argument("--code", type=str, help="PDD code; auto-resolves manifest")
    parser.add_argument("--preset", type=str, default="mini")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "pgmap", "paired", "sumo", "citymap"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--save-gifs", action="store_true")
    parser.add_argument("--spawn-velocity-ms", type=float, default=None,
                        help="Override ego initial velocity (m/s). If unset, "
                             "uses row.spawn_velocity_ms from manifest (default 0).")
    parser.add_argument("--aggregate", action="store_true",
                        help="Only rebuild expert_global_summary.json from existing per-sign summaries")

    # Multi-variant trajectory recording (new layout)
    parser.add_argument("--policy", type=str, default=None, choices=POLICY_NAMES,
                        help="Ego policy. If unset, runs the legacy single-policy "
                             "(comprehensive) batch and writes the old expert/ layout.")
    parser.add_argument("--carl-model-path", type=str, default=None,
                        help="Required when --policy carl: path to CaRL checkpoint")
    parser.add_argument("--carl-device", type=str, default="cpu",
                        help="Torch device for CaRL (default: cpu)")
    parser.add_argument("--plant2-model-path", type=str, default=None,
                        help="Required when --policy plant2: path to PlanT2 checkpoint")
    parser.add_argument("--plant2-repo-dir", type=str, default=None,
                        help="Required when --policy plant2: plant2 repo root")
    parser.add_argument("--plant2-device", type=str, default="cpu",
                        help="Torch device for PlanT2 (default: cpu)")
    parser.add_argument("--ego-extra-samples", type=int, default=0,
                        help="N additional rollouts per scene with nuPlan-sampled ego "
                             "IDM params (only meaningful for --policy comprehensive). "
                             "Default 0: only the apply_ego_defaults baseline is recorded.")
    parser.add_argument("--ego-sample-seed-base", type=int, default=0,
                        help="Base offset for sampled-variant seeds (default 0)")
    parser.add_argument("--skip-variants", type=str, default=None,
                        help="Comma-separated list of comprehensive variants to NOT record. "
                             "Accepts bare ids ('default', 's2') or fully qualified "
                             "('comprehensive_s2'). Used by optim/scene_dispatcher.py to "
                             "avoid recomputing IDM variants that already have a valid "
                             "replay elsewhere. Hierarchy is preserved (seeds depend on "
                             "position, not on which variants are skipped).")
    parser.add_argument("--sample-ego-spawn-velocity", action="store_true",
                        help="Sample ego initial spawn velocity from nuPlan distribution "
                             "(sample_spawn_velocity(row.seed)) instead of using 0.0 from manifest.")
    parser.add_argument("--build-oracle-manifest", action="store_true",
                        help="Read <run-dir>/all_runs.jsonl and write oracle_manifest.jsonl")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Run root for --build-oracle-manifest")

    args = parser.parse_args()

    # Propagate spawn_velocity override to env builder via env var (read in
    # _build_sumo_env). None = use manifest value.
    if args.spawn_velocity_ms is not None:
        import os as _os_main
        _os_main.environ["EGO_SPAWN_VELOCITY_MS"] = str(args.spawn_velocity_ms)
        print(f"[ego] spawn_velocity override = {args.spawn_velocity_ms} m/s "
              f"({args.spawn_velocity_ms*3.6:.1f} km/h)")

    if args.aggregate:
        out = aggregate_global(preset=args.preset)
        print(json.dumps({"preset": out.get("preset"),
                           "total_signs": out.get("total_signs"),
                           "total_episodes": out.get("total_episodes"),
                           "total_valid": out.get("total_valid")}, indent=2))
        return

    if args.build_oracle_manifest:
        if not args.run_dir:
            print("ERROR: --build-oracle-manifest requires --run-dir", file=sys.stderr)
            sys.exit(1)
        out = build_oracle_manifest(Path(args.run_dir))
        print(json.dumps(out, indent=2))
        return

    backend = args.backend
    if args.manifest:
        manifest_path = Path(args.manifest)
    elif args.code:
        code_dir = _resolve_code_dir(args.code, args.preset)
        if code_dir is None:
            print(f"ERROR: no benchmark output for code={args.code} preset={args.preset}",
                  file=sys.stderr)
            sys.exit(1)
        manifest_path, backend = _pick_manifest_and_backend(code_dir, backend)
    else:
        print("Need --manifest or --code", file=sys.stderr)
        sys.exit(1)

    if args.policy is None:
        # Legacy single-policy (comprehensive) batch — preserves the old layout
        # (output_dir/replays/, output_dir/expert_replays.jsonl, etc.) so existing
        # callers like run_expert_5xx.sh keep working bit-for-bit.
        if args.output_dir:
            output_dir = Path(args.output_dir)
        elif args.code:
            code_dir = _resolve_code_dir(args.code, args.preset)
            output_dir = code_dir / "expert"
        else:
            output_dir = manifest_path.parent / "expert"

        print(f"Manifest: {manifest_path}")
        print(f"Backend:  {backend}")
        print(f"Output:   {output_dir}")
        run_batch(manifest_path, backend, output_dir,
                  count=args.count, start=args.start,
                  max_steps=args.max_steps, save_gifs=args.save_gifs,
                  sample_ego_velocity=args.sample_ego_spawn_velocity)
        return

    # New multi-variant mode: --policy was specified.
    if args.policy == "carl":
        if not args.carl_model_path:
            print("ERROR: --policy carl requires --carl-model-path <checkpoint>",
                  file=sys.stderr)
            sys.exit(1)
        from agents.policies.carl_sign_compliant import CarlSignCompliantPolicy
        CarlSignCompliantPolicy.set_checkpoint(args.carl_model_path,
                                                device=args.carl_device)
    if args.policy == "plant2":
        if not args.plant2_model_path or not args.plant2_repo_dir:
            print("ERROR: --policy plant2 requires --plant2-model-path "
                  "<checkpoint> and --plant2-repo-dir <repo>", file=sys.stderr)
            sys.exit(1)
        from agents.policies.plant2_sign_compliant import PlanT2SignCompliantPolicy
        PlanT2SignCompliantPolicy.set_checkpoint(args.plant2_model_path,
                                                  args.plant2_repo_dir,
                                                  device=args.plant2_device)
    if args.output_dir is None:
        print("ERROR: --policy mode requires --output-dir (run root)", file=sys.stderr)
        sys.exit(1)
    run_root = Path(args.output_dir)
    run_root.mkdir(parents=True, exist_ok=True)

    # sign_slug used in by_sign/<slug>/
    if args.code:
        sign_slug = args.code.replace(".", "_")
    else:
        sign_slug = manifest_path.parent.name

    print(f"Manifest:    {manifest_path}")
    print(f"Backend:     {backend}")
    print(f"Run root:    {run_root}")
    print(f"Policy:      {args.policy}  extra_samples={args.ego_extra_samples}")
    skip_variants = (
        {v.strip() for v in args.skip_variants.split(",") if v.strip()}
        if args.skip_variants else None
    )
    out = run_batch_multi_variant(
        manifest_path, backend, run_root,
        sign_slug=sign_slug,
        policy_name=args.policy,
        count=args.count, start=args.start,
        max_steps=args.max_steps,
        extra_samples=args.ego_extra_samples,
        sample_seed_base=args.ego_sample_seed_base,
        save_gifs=args.save_gifs,
        sample_ego_velocity=args.sample_ego_spawn_velocity,
        skip_variants=skip_variants,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
