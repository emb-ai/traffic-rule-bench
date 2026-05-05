"""Replay mini_new SUMO scenes under ComprehensiveRuleExpertPolicy.

Mirrors the run_benchmark.py GIF-recording style (per-step top-down render
with semantic layers, per-step overlay text, GIF via top_down_renderer.generate_gif)
but driven off a mini_new manifest and using the comprehensive rule expert.

The widened lane width is already baked into SUMO_DEFAULT_CONFIG
(`min_lane_width=4.5`), so it applies automatically here.

Usage:
    python pdd-bench/scripts/replay_mini_new.py --sign-code 2.5

Outputs:
    <mini-new-root>/<code>/expert_rerun_<timestamp>/
        gifs/<scene_id>.gif
        episode_results.jsonl   (per-episode metrics)
        summary.json            (aggregate dest_rate/crash_rate/oor_rate)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
PDD_BENCH_DIR = SCRIPT_PATH.parent.parent
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"
SCENES_ROOT = PDD_BENCH_DIR / "scenes"  # overridden by --scenes-root
PER_SIGN_BENCH_DIR = PDD_BENCH_DIR / "scripts" / "per_sign_bench"
for _p in (PDD_BENCH_DIR, METADRIVE_DIR, PER_SIGN_BENCH_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from envs.sumo_env import TrafficSignSumoEnv
from envs.sumo_traffic_manager import SumoTrafficManager
from agents.policies.comprehensive_rule_expert import ComprehensiveRuleExpertPolicy
from agents.policies.rule_compliant_expert import RuleCompliantExpertPolicy
from agents.policies.carl_sign_compliant import CarlSignCompliantPolicy
from agents.policies.plant2_sign_compliant import PlanT2SignCompliantPolicy
from agents.policies.base_policies import CarlBasePolicy, PlanT2BasePolicy
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.expert_policy import ExpertPolicy

# nuPlan-sampled IDM ego variants (only meaningful for IDM-based policies).
# apply_ego_sampled is a setattr-only no-op on non-IDM policies, so it's safe
# to call uniformly.
from factorized_space.ego_defaults import (
    apply_ego_defaults, apply_ego_sampled, sample_ego_params,
)

POLICY_MAP = {
    # With sign compliance
    "idm_signs":    ComprehensiveRuleExpertPolicy,  # IDM + signs
    "ppo_signs":    RuleCompliantExpertPolicy,       # PPO + signs
    "carl_signs":   CarlSignCompliantPolicy,         # CaRL + signs
    "plant2_signs": PlanT2SignCompliantPolicy,        # PlanT2 + signs
    # Without sign compliance (baseline)
    "idm":          IDMPolicy,                       # pure IDM
    "ppo":          ExpertPolicy,                    # pure PPO
    "carl":         CarlBasePolicy,                  # CaRL, no signs
    "plant2":       PlanT2BasePolicy,                # PlanT2, no signs
}

_SUMO_SIGN_DISTANCE_CACHE: dict[Path, float] = {}


class _EnvWithTraffic(TrafficSignSumoEnv):
    """Same wrapper as run_benchmark.py: swaps the default traffic manager
    for the SUMO-traffic variant and zeroes traffic_density in defaults
    (actual density is set via config passed at construction)."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg["traffic_density"] = 0.0
        return cfg

    def setup_engine(self):
        super().setup_engine()
        self.engine.update_manager("traffic_manager", SumoTrafficManager())


def _vehicle_crash_attribution(ego, engine, contact_dist: float = 4.0):
    """Heuristic fault attribution for a vehicle–vehicle collision.

    Returns ``(ego_fault, contact_npcs)``.  Rule:
        * Any colliding NPC in ego's forward half (rel_x > 0 in ego frame)
          AND ego speed > 0.5 km/h  → ego drove into it → ego fault.
        * Otherwise (rear/side contact, or ego stationary) → NPC fault.
    """
    import math as _m

    ego_pos = ego.position
    ego_heading = ego.heading_theta
    ego_speed_kmh = float(getattr(ego, "speed_km_h", 0.0))
    cos_h, sin_h = _m.cos(ego_heading), _m.sin(ego_heading)

    traffic_manager = getattr(engine, "traffic_manager", None)
    traffic_ids = set()
    if traffic_manager is not None:
        try:
            traffic_ids = {getattr(v, "id", None) for v in traffic_manager.traffic_vehicles}
        except Exception:
            traffic_ids = set()

    try:
        objs = engine.get_objects(lambda o: o is not ego).values()
    except Exception:
        return True, []  # no way to attribute, assume ego fault

    from metadrive.component.vehicle.base_vehicle import BaseVehicle
    contact_npcs = []
    ego_fault = False
    for obj in objs:
        if not isinstance(obj, BaseVehicle):
            continue
        obj_id = getattr(obj, "id", None)
        if traffic_ids and obj_id not in traffic_ids:
            continue
        try:
            dx = obj.position[0] - ego_pos[0]
            dy = obj.position[1] - ego_pos[1]
            dist = _m.hypot(dx, dy)
        except Exception:
            continue
        if dist > contact_dist:
            continue
        contact_npcs.append(obj)
        # World -> ego-local: rotate by -ego_heading
        rel_x = cos_h * dx + sin_h * dy   # +ve = in front of ego
        # Front-half contact AND ego moving -> ego fault.
        if rel_x > 0.0 and ego_speed_kmh > 0.5:
            ego_fault = True

    if not contact_npcs:
        return True, []  # vehicle crash without a removable NPC is unsafe to ignore
    return ego_fault, contact_npcs


def _remove_npc_fault_vehicles(engine, npcs) -> int:
    """Remove NPCs that hit ego and clear SumoTrafficManager bookkeeping."""
    npc_ids = []
    for npc in npcs:
        npc_id = getattr(npc, "id", None)
        if npc_id is not None and npc_id not in npc_ids:
            npc_ids.append(npc_id)
    if not npc_ids:
        return 0

    traffic_manager = getattr(engine, "traffic_manager", None)
    cleared = False
    if traffic_manager is not None:
        try:
            traffic_manager.clear_objects(npc_ids)
            cleared = True
        except Exception:
            pass
    if not cleared:
        try:
            engine.clear_objects(npc_ids)
            cleared = True
        except Exception:
            pass

    npc_id_set = set(npc_ids)
    if traffic_manager is not None:
        traffic_vehicles = getattr(traffic_manager, "_traffic_vehicles", None)
        if isinstance(traffic_vehicles, list):
            traffic_manager._traffic_vehicles = [
                v for v in traffic_vehicles
                if getattr(v, "id", None) not in npc_id_set
            ]
        for attr in ("_spawn_step", "_stuck_counter", "_wrong_way_counter"):
            store = getattr(traffic_manager, attr, None)
            if isinstance(store, dict):
                for npc_id in npc_ids:
                    store.pop(npc_id, None)

    return len(npc_ids) if cleared else 0


def _format_violation(sign, vehicle) -> str:
    sign_name = type(sign).__name__
    lane = getattr(sign, "lane", None)
    lane_idx = getattr(lane, "index", None)
    parts = [sign_name]
    if lane_idx is not None:
        parts.append(f"L:{lane_idx}")
    try:
        if lane is not None:
            veh_long = float(lane.local_coordinates(vehicle.position)[0])
            parts.append(f"d={float(lane.length - veh_long):.1f}m")
    except Exception:
        pass
    return " | ".join(parts)


def _resolve_sign_spawn_distance(row: dict, map_path: Path) -> float:
    """Match run_benchmark.py: use meta.distance_from_start, fallback to 30m."""
    direct = row.get("sign_spawn_distance")
    if direct is None:
        direct = row.get("distance_from_start")
    if direct is None:
        meta_path = map_path.parent / "meta.json"
        if meta_path not in _SUMO_SIGN_DISTANCE_CACHE:
            distance = 0.0
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                distance = float(meta.get("distance_from_start", 0.0) or 0.0)
            except Exception:
                distance = 0.0
            _SUMO_SIGN_DISTANCE_CACHE[meta_path] = distance
        direct = _SUMO_SIGN_DISTANCE_CACHE[meta_path]
    return max(float(direct or 0.0), 30.0)


def _first_present(row: dict, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _build_config(row: dict, max_steps: int | None, policy_cls,
                  min_lane_width: float = 6.0,
                  traffic_density: float | None = None,
                  use_manifest_destination: bool = False) -> dict:
    """Build the env config for one manifest row, mirroring run_benchmark.py
    base_config but plugging in the requested ego policy class."""
    net_path = Path(str(row["net_path"]))
    map_path = net_path if net_path.is_absolute() else (SCENES_ROOT / net_path)
    fingerprint = row.get("fingerprint", {}) or {}
    road_id = row.get("road_id") or fingerprint.get("start_lane")
    spawn_lane_num = int(row.get("spawn_lane_num", 0))
    sign_type = row.get("sign_code") or row.get("sign_type") or row.get("pdd_code")
    sign_spawn_distance = _resolve_sign_spawn_distance(row, map_path)
    resolved_traffic_density = _first_present(
        {"traffic_density": traffic_density} if traffic_density is not None else {},
        "traffic_density",
    )
    if resolved_traffic_density is None:
        resolved_traffic_density = _first_present(row, "traffic_density", "profile_traffic_density")
    if resolved_traffic_density is None:
        resolved_traffic_density = 0.0
    resolved_horizon = max_steps
    if resolved_horizon is None:
        resolved_horizon = _first_present(row, "horizon", "horizon_steps", "profile_horizon_steps")
    if resolved_horizon is None:
        resolved_horizon = 600

    # run_benchmark.py does not force a destination for SUMO scenes; it lets
    # TrafficSignSumoEnv rebuild navigation from road_id/spawn_lane_num. Keep
    # manifest destinations opt-in for experiments/debugging only.
    destination = (row.get("destination_lane_id")
                   or fingerprint.get("dest_node"))
    start_lane = fingerprint.get("start_lane")

    # Guard against pathological destinations that make
    # EdgeRoadNetwork.shortest_path hang in BFS:
    #   1. destination == start_lane -> self-loop, infinite walk
    #   2. junction-internal lanes (`lane_:<junction>_...`) aren't routable
    #      endpoints in the graph.
    def _is_safe_destination(dest: str | None, start: str | None) -> bool:
        if not dest:
            return False
        if start and dest == start:
            return False
        if dest.startswith("lane_:"):
            return False
        return True

    vehicle_config = {"spawn_lane_index": road_id, "show_lidar": False}
    if use_manifest_destination and _is_safe_destination(destination, start_lane):
        vehicle_config["destination"] = destination

    return dict(
        use_render=False,
        vehicle_config=vehicle_config,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.ERROR,
        map_name=str(map_path),
        sign_type=sign_type,
        sign_spawn_distance=sign_spawn_distance,
        spawn_lane_num=spawn_lane_num,
        traffic_density=float(resolved_traffic_density),
        tl_speed_factor=float(_first_present(row, "tl_speed_factor") or 20.0),
        horizon=int(resolved_horizon),
        min_route_hops_after_spawn=int(_first_present(row, "min_route_hops_after_spawn") or 10),
        max_route_hops_after_spawn=int(_first_present(row, "max_route_hops_after_spawn") or 10),
        # Vehicle crashes are attributed in this replay loop.  Ego-fault
        # crashes terminate; NPC-fault contacts remove the NPC and continue.
        crash_vehicle_done=False,
        # Large num_scenarios so the manifest's big deterministic_seed values
        # pass the `seed < num_scenarios` assert in base_env._reset_global_seed.
        num_scenarios=100000,
        agent_policy=policy_cls,
        min_lane_width=min_lane_width,
    )


def _run_one(row: dict, output_gifs_dir: Path, max_steps: int | None,
             save_gif: bool, policy_cls, show_window: bool = False,
             min_lane_width: float = 6.0,
             traffic_density: float | None = None,
             use_manifest_destination: bool = False,
             ego_params: dict | None = None,
             variant: str = "default") -> dict:
    scene_id = row.get("scene_id", "unknown")
    raw_seed = int(row.get("deterministic_seed") or row.get("seed") or 0)
    spawn_lane_num = int(row.get("spawn_lane_num", 0))

    # MetaDrive asserts `seed < num_scenarios`. run_benchmark.py uses sign_id
    # as the env seed; SUMO materialization offsets by var_idx. Keep raw_seed
    # for numpy/random so NPC-profile randomness still differs across rows.
    np.random.seed(raw_seed)
    random.seed(raw_seed)
    if row.get("sign_id") is not None:
        seed = (int(row.get("sign_id", 0)) + int(row.get("var_idx", 0))) % 100000
    else:
        seed = raw_seed % 100000

    config = _build_config(
        row,
        max_steps,
        policy_cls,
        min_lane_width=min_lane_width,
        traffic_density=traffic_density,
        use_manifest_destination=use_manifest_destination,
    )
    episode_horizon = int(config.get("horizon") or 600)
    env = None
    try:
        EnvClass = _EnvWithTraffic if float(config.get("traffic_density", 0.0)) > 0.0 else TrafficSignSumoEnv
        env = EnvClass(config)
        obs, info = env.reset(seed=seed)
        base_env = env.unwrapped

        # Apply ego IDM-parameter overrides (default or sampled). For non-IDM
        # policies (carl/plant2/rule_compliant) these setattr calls are no-ops
        # behaviour-wise but still recorded in `variant` for traceability.
        try:
            ego_policy = base_env.engine.get_policy(base_env.agent.id)
            if ego_policy is not None:
                if ego_params is not None:
                    apply_ego_sampled(ego_policy, ego_params)
                else:
                    apply_ego_defaults(ego_policy)
        except Exception as _exc:
            print(f"  [warn] failed to apply ego params: {_exc}")
        total_reward = 0.0
        violations = 0
        steps = 0
        crashed = False
        reached_dest = False
        out_of_road = False
        crash_attrib = None
        violated_sign_ids = set()
        last_violation_texts: list[str] = []
        violation_text_ttl = 0
        done_reason = "timeout"

        sm = base_env.engine.traffic_sign_manager
        registered = [type(s).__name__ for s in sm.signs if type(s).__name__ != "TrafficLightSign"]
        print(f"[{scene_id}] seed={seed} lane={spawn_lane_num} signs={registered}")

        for step in range(episode_horizon):
            action = [0.0, 0.0]  # ignored: agent_policy drives ego
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

            vehicle = base_env.agent

            current_violations = sm.check_all_violations(vehicle)
            current_violation_names, current_violation_texts = [], []
            compliance_lines = []
            _mgr_rules = {id(r) for r in sm.rules}
            for sign, violated_for_report in current_violations:
                sign_type_name = type(sign).__name__
                if sign_type_name == "TrafficLightSign":
                    continue
                is_rule = id(sign) in _mgr_rules
                name = str(getattr(sign, "id", None) or sign_type_name) if is_rule else sign_type_name

                if not is_rule and hasattr(sign, "_is_violating"):
                    try:
                        currently_violating = bool(sign._is_violating(vehicle))
                    except Exception:
                        currently_violating = bool(violated_for_report)
                else:
                    currently_violating = bool(violated_for_report) or (name in violated_sign_ids)

                if violated_for_report:
                    violations += 1
                    current_violation_names.append(name)
                    violated_sign_ids.add(name)
                    current_violation_texts.append(_format_violation(sign, vehicle))

                compliance_lines.append(f"{name}: {'VIOLATED' if currently_violating else 'OK'}")

            status = ", ".join(compliance_lines) if compliance_lines else "no signs"
            marker = "VIOLATION" if current_violation_names else "ok      "
            try:
                lane_idx = vehicle.lane.index
            except Exception:
                lane_idx = "?"
            print(f"  step={step:04d} {marker} speed={vehicle.speed_km_h:5.1f} km/h "
                  f"lane={str(lane_idx):>28} | {status}")

            if current_violation_texts:
                last_violation_texts = current_violation_texts[:3]
                violation_text_ttl = 40
            elif violation_text_ttl > 0:
                violation_text_ttl -= 1
            else:
                last_violation_texts = []

            current_violations_str = ", ".join(current_violation_names) if current_violation_names else "-"
            text_dict = {
                "Scene": scene_id,
                "Step": step,
                "Speed": f"{vehicle.speed_km_h:.2f} km/h",
                "Lane": str(lane_idx),
                "Lane width": f"{getattr(vehicle.lane, 'width', 0.0):.2f}",
                "Violations (total)": violations,
                "Violation now": current_violations_str,
            }
            for i, line in enumerate(compliance_lines[:6]):
                text_dict[f"Compliance[{i}]"] = line
            if current_violation_texts:
                text_dict["Violation"] = current_violation_texts[0]
                if len(current_violation_texts) > 1:
                    text_dict["Violation +"] = f"+{len(current_violation_texts) - 1} more"
            elif last_violation_texts:
                text_dict["Last violation"] = last_violation_texts[0]
                if len(last_violation_texts) > 1:
                    text_dict["Last violation +"] = f"+{len(last_violation_texts) - 1} more"

            vehicle_crash = bool(info.get("crash_vehicle", False)) or bool(
                getattr(vehicle, "crash_vehicle", False)
            )
            npc_fault_vehicle_crash = False
            if vehicle_crash:
                ego_fault, contact_npcs = _vehicle_crash_attribution(vehicle, base_env.engine)
                if ego_fault:
                    crashed = True
                    crash_attrib = "ego"
                    done_reason = "crashed"
                    print("  crash: ego-fault vehicle collision")
                    break

                npc_fault_vehicle_crash = True
                if crash_attrib is None:
                    crash_attrib = "npc"
                removed = _remove_npc_fault_vehicles(base_env.engine, contact_npcs)
                vehicle.crash_vehicle = False
                print(f"  crash: npc-fault vehicle collision, removed_npc={removed}")

            env.unwrapped.render(mode="topdown",
                                 text=text_dict,
                                 screen_record=save_gif,
                                 window=show_window,
                                 film_size=(10000, 10000),
                                 scaling=10,
                                 screen_size=(600, 600),
                                 semantic_map=True,
                                 semantic_broken_line=True,
                                 draw_target_vehicle_trajectory=True,
                                 target_agent_heading_up=True)

            done = terminated or truncated
            if done:
                reached_dest = bool(info.get("arrive_dest", False))
                out_of_road = bool(info.get("out_of_road", False)) or bool(getattr(vehicle, "crash_sidewalk", False))
                object_or_human_crash = (
                    bool(info.get("crash_object", False))
                    or bool(info.get("crash_building", False))
                    or bool(info.get("crash_human", False))
                    or bool(getattr(vehicle, "crash_object", False))
                    or bool(getattr(vehicle, "crash_building", False))
                    or bool(getattr(vehicle, "crash_human", False))
                )
                if (npc_fault_vehicle_crash and not truncated and not reached_dest
                        and not out_of_road and not object_or_human_crash):
                    continue
                if reached_dest:
                    done_reason = "arrived"
                elif out_of_road:
                    done_reason = "out_of_road"
                elif object_or_human_crash:
                    crashed = True
                    done_reason = "crashed"
                else:
                    done_reason = "done_other"
                break

        # Save GIF
        gif_path = None
        if save_gif:
            try:
                renderer = getattr(base_env, "top_down_renderer", None)
                if renderer is not None and getattr(renderer, "_screen_frames", None):
                    output_gifs_dir.mkdir(parents=True, exist_ok=True)
                    suffix = f"_lane{spawn_lane_num}"
                    if variant and variant != "default":
                        suffix += f"_{variant}"
                    gif_path = str(output_gifs_dir / f"{scene_id}{suffix}.gif")
                    renderer.generate_gif(gif_name=gif_path, duration=40)
                    print(f"  gif: {gif_path}")
            except Exception as exc:
                print(f"  [warn] gif save failed: {exc}")

        return {
            "scene_id": scene_id,
            "seed": seed,
            "spawn_lane_num": spawn_lane_num,
            "variant": variant,
            "ego_params": ego_params,
            "sign_spawn_distance": config.get("sign_spawn_distance"),
            "traffic_density": config.get("traffic_density"),
            "horizon": config.get("horizon"),
            "tl_speed_factor": config.get("tl_speed_factor"),
            "min_route_hops_after_spawn": config.get("min_route_hops_after_spawn"),
            "max_route_hops_after_spawn": config.get("max_route_hops_after_spawn"),
            "use_manifest_destination": use_manifest_destination,
            "destination_lane_id": row.get("destination_lane_id"),
            "steps": steps,
            "total_reward": round(total_reward, 2),
            "violations": violations,
            "reached_dest": reached_dest,
            "crashed": crashed,
            "crash_attribution": crash_attrib,  # "ego" | "npc" | None
            "out_of_road": out_of_road,
            "done_reason": done_reason,
            "gif_path": gif_path,
        }
    finally:
        # Always shut the MetaDrive singleton down, even if env construction
        # or reset() failed — otherwise the engine stays alive and every
        # subsequent scene hits "Can not call this API after engine init".
        # close_engine() from engine_utils calls singleton.close() *before*
        # nulling the field; on a partially-initialised engine close() can
        # throw, leaving the singleton set. We force-null it here.
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            from metadrive.engine.base_engine import BaseEngine
            if BaseEngine.singleton is not None:
                try:
                    BaseEngine.singleton.close()
                except Exception:
                    pass
                BaseEngine.singleton = None
        except Exception:
            pass


def _print_summary(episode_results: list[dict], summary: dict) -> None:
    print()
    print("=" * 100)
    print(f"per-scene trip outcomes (n={len(episode_results)})")
    print("=" * 100)
    print(f"{'scene_id':<30} {'lane':>4} {'reason':>12} {'steps':>5} {'viol':>4}  gif")
    print("-" * 100)
    for r in episode_results:
        gif_short = Path(r["gif_path"]).name if r.get("gif_path") else "-"
        print(f"{r['scene_id']:<30} {r['spawn_lane_num']:>4} {r['done_reason']:>12} "
              f"{r['steps']:>5} {r['violations']:>4}  {gif_short}")

    print()
    print("=" * 100)
    print("aggregate summary")
    print("=" * 100)
    for k, v in summary.items():
        print(f"  {k:<20} = {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign-code", default="2.5")
    ap.add_argument("--mini-new-root", default="/Users/victoria_s/sdc_new_signs/mini_new")
    ap.add_argument("--scenes-root", default=None,
                    help="Root folder containing SUMO .net.xml files referenced by net_path "
                         "in the manifest (default: pdd-bench/scenes next to this script)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Override episode horizon. Default: use row horizon/profile_horizon_steps, fallback 600.")
    ap.add_argument("--limit", type=int, default=None, help="Max scenes (default: all)")
    ap.add_argument("--policy", default="idm_signs",
                    choices=list(POLICY_MAP.keys()),
                    help="Ego policy. With signs: idm_signs, ppo_signs, carl_signs, plant2_signs. "
                         "Without signs: idm, ppo, carl, plant2.")
    ap.add_argument("--carl-model-path", default=None,
                    help="Path to CaRL PPO checkpoint (required for --policy carl)")
    ap.add_argument("--carl-device", default="cpu",
                    help="Torch device for CaRL model (cpu or cuda). Default: cpu")
    ap.add_argument("--plant2-model-path", default=None,
                    help="Path to PlanT2 checkpoint (required for --policy plant2)")
    ap.add_argument("--plant2-repo-dir", default=None,
                    help="Path to plant2 repo root (default: SDC_ROOT/plant2)")
    ap.add_argument("--plant2-device", default=None,
                    help="Torch device for PlanT2 model (default: auto cuda/cpu)")
    ap.add_argument("--no-gifs", action="store_true", help="Skip GIF rendering")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write metrics JSON (default: sdc/pdd-bench/replay_results/mini_new_<code>_<ts>)")
    ap.add_argument("--gifs-dir", default=None,
                    help="Where to write GIFs (default: sdc/pdd-bench/gifs/mini_new_<code>_<ts>)")
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle scenes before applying --limit (random sample)")
    ap.add_argument("--shuffle-seed", type=int, default=42,
                    help="Seed for shuffle (default: 42) — same seed = same scenes across policies")
    ap.add_argument("--min-lane-width", type=float, default=6.0,
                    help="Minimum lane width in metres (default: 6.0 — widened; use 0.0 for raw SUMO widths)")
    ap.add_argument("--traffic-density", type=float, default=None,
                    help="Override SUMO traffic density. Default: use row traffic_density/profile_traffic_density, fallback 0.0")
    ap.add_argument("--use-manifest-destination", action="store_true",
                    help="Experimental: force vehicle_config.destination from "
                         "destination_lane_id/fingerprint.dest_node. Default off "
                         "to match run_benchmark.py SUMO launches.")
    ap.add_argument("--window", action="store_true",
                    help="Show live pygame top-down window during rendering (default: off — only GIFs)")
    ap.add_argument("--ego-extra-samples", type=int, default=0,
                    help="N additional rollouts per scene with nuPlan-sampled IDM ego "
                         "params (default 0 = run only the default-params variant). "
                         "Sampling is reproducible via --ego-sample-seed-base.")
    ap.add_argument("--ego-sample-seed-base", type=int, default=0,
                    help="Base seed for sampled IDM variants. Each sample k uses "
                         "seed = base + k * 1000003 (default 0).")
    args = ap.parse_args()

    global SCENES_ROOT
    if args.scenes_root is not None:
        SCENES_ROOT = Path(args.scenes_root)

    code_dir = args.sign_code.replace(".", "_")
    manifest = Path(args.mini_new_root) / code_dir / "sumo" / "sumo_manifest.jsonl"
    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        sys.exit(1)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_tag = f"mini_new_{code_dir}_{args.policy}_{ts}"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PDD_BENCH_DIR / "replay_results" / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gifs_dir:
        gifs_dir = Path(args.gifs_dir)
    else:
        gifs_dir = PDD_BENCH_DIR / "gifs" / run_tag
    if not args.no_gifs:
        gifs_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    if args.shuffle:
        random.Random(args.shuffle_seed).shuffle(rows)
    if args.limit is not None:
        rows = rows[:args.limit]
    policy_cls = POLICY_MAP[args.policy]
    if args.policy in ("carl_signs", "carl"):
        if not args.carl_model_path:
            print(f"ERROR: --policy {args.policy} requires --carl-model-path <checkpoint>",
                  file=sys.stderr)
            sys.exit(1)
        policy_cls.set_checkpoint(args.carl_model_path, device=args.carl_device)
    elif args.policy in ("plant2_signs", "plant2"):
        if not args.plant2_model_path:
            print(f"ERROR: --policy {args.policy} requires --plant2-model-path <checkpoint>",
                  file=sys.stderr)
            sys.exit(1)
        plant2_repo = Path(args.plant2_repo_dir) if args.plant2_repo_dir else (SDC_ROOT / "plant2")
        if args.plant2_device:
            plant2_device = args.plant2_device
        else:
            try:
                import torch
                plant2_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                plant2_device = "cpu"
        policy_cls.set_checkpoint(args.plant2_model_path, plant2_repo, device=plant2_device)
    print(f"Manifest:   {manifest}  ({len(rows)} scenes)")
    print(f"Policy:     {args.policy} ({policy_cls.__name__})")
    print(f"Output dir: {output_dir}")
    print(f"GIFs dir:   {gifs_dir if not args.no_gifs else '(off)'}")
    print()

    logging.getLogger().setLevel(logging.ERROR)

    # Build the list of (variant_name, ego_params) per scene:
    # - default: ego_params=None → apply_ego_defaults
    # - sN (N>=1): ego_params from sample_ego_params(seed_base + N*1000003)
    variants: list[tuple[str, dict | None]] = [("default", None)]
    for k in range(1, max(0, int(args.ego_extra_samples)) + 1):
        seed_k = int(args.ego_sample_seed_base) + k * 1000003
        try:
            params = sample_ego_params(seed_k)
        except Exception as exc:
            print(f"[warn] sample_ego_params(seed={seed_k}) failed: {exc} — skipping s{k}")
            continue
        variants.append((f"s{k}", params))
    if len(variants) > 1:
        print(f"Variants per scene: {len(variants)} "
              f"(1 default + {len(variants) - 1} sampled)")

    t0 = time.time()
    episode_results: list[dict] = []
    total_runs = len(rows) * len(variants)
    run_idx = 0
    for row in rows:
        for variant_name, ego_params in variants:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] scene={row.get('scene_id','?')} variant={variant_name}")
            try:
                result = _run_one(row, gifs_dir, args.max_steps,
                                  save_gif=not args.no_gifs,
                                  policy_cls=policy_cls,
                                  min_lane_width=args.min_lane_width,
                                  traffic_density=args.traffic_density,
                                  use_manifest_destination=args.use_manifest_destination,
                                  show_window=args.window,
                                  ego_params=ego_params,
                                  variant=variant_name)
            except Exception as exc:
                import traceback
                print(f"  [ERROR] {type(exc).__name__}: {exc}")
                traceback.print_exc()
                result = {
                    "scene_id": row.get("scene_id", "?"),
                    "seed": row.get("deterministic_seed", 0),
                    "spawn_lane_num": row.get("spawn_lane_num", 0),
                    "variant": variant_name,
                    "ego_params": ego_params,
                    "error": f"{type(exc).__name__}: {exc}",
                    "reached_dest": False,
                    "crashed": False,
                    "out_of_road": False,
                    "done_reason": "error",
                    "steps": 0,
                    "violations": 0,
                    "gif_path": None,
                }
            episode_results.append(result)
            with open(output_dir / "episode_results.jsonl", "w") as f:
                for r in episode_results:
                    f.write(json.dumps(r, default=str) + "\n")

    def _agg(rows):
        """Aggregate metrics over a list of episode-result dicts."""
        n = len(rows) or 1
        dest = sum(1 for r in rows if r.get("reached_dest"))
        ego_crash = sum(1 for r in rows if r.get("crash_attribution") == "ego")
        npc_crash = sum(1 for r in rows if r.get("crash_attribution") == "npc")
        oor = sum(1 for r in rows if r.get("out_of_road"))
        errors = sum(1 for r in rows if r.get("done_reason") == "error")
        return {
            "n_episodes": n,
            "n_errors": errors,
            "dest_rate": round(dest / n, 3),
            "crash_rate": round(ego_crash / n, 3),
            "crashes_ego_fault": ego_crash,
            "crashes_npc_fault": npc_crash,
            "out_of_road_rate": round(oor / n, 3),
            "avg_violations": round(
                sum(r.get("violations", 0) for r in rows) / n, 2
            ),
        }

    # Per-scene oracle: for each scene_id, pick the variant with the best
    # outcome (dest=+1000, viol=-10, crashed=-500, oor=-200). For non-IDM
    # policies all variants are identical, so the pick is trivial; for
    # comprehensive (IDM) this gives the within-policy oracle.
    def _score(r):
        return (
            int(bool(r.get("reached_dest", False))) * 1000
            - int(r.get("violations", 0)) * 10
            - int(bool(r.get("crashed", False))) * 500
            - int(bool(r.get("out_of_road", False))) * 200
        )

    by_scene: dict = {}
    for r in episode_results:
        sid = r.get("scene_id", "?")
        cur = by_scene.get(sid)
        if cur is None or _score(r) > _score(cur):
            by_scene[sid] = r
    oracle_episodes = list(by_scene.values())

    default_episodes = [r for r in episode_results if r.get("variant", "default") == "default"]

    summary = {
        "sign_code": args.sign_code,
        "policy": args.policy,
        "n_scenes": len(by_scene),
        "n_variants_per_scene": len(variants),
        "wall_time_s": round(time.time() - t0, 1),
        # default-params only (legacy comparison)
        "default": _agg(default_episodes),
        # All runs (default + N sampled)
        "all_variants": _agg(episode_results),
        # Within-policy oracle: best variant per scene
        "per_scene_oracle": _agg(oracle_episodes),
    }
    # Top-level shortcuts for compatibility with previous summary readers
    summary.update({
        "n_episodes": summary["all_variants"]["n_episodes"],
        "n_errors": summary["all_variants"]["n_errors"],
        "dest_rate": summary["all_variants"]["dest_rate"],
        "crash_rate": summary["all_variants"]["crash_rate"],
        "crashes_ego_fault": summary["all_variants"]["crashes_ego_fault"],
        "crashes_npc_fault": summary["all_variants"]["crashes_npc_fault"],
        "out_of_road_rate": summary["all_variants"]["out_of_road_rate"],
        "avg_violations": summary["all_variants"]["avg_violations"],
    })
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(episode_results, summary)


if __name__ == "__main__":
    main()
