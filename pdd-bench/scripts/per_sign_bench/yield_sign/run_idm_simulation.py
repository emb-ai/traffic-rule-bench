#!/usr/bin/env python3
"""Run IDM vehicle simulation on a SUMO scene and save GIF.

Usage:
    python run_idm_simulation.py savvinskaya_3
    python run_idm_simulation.py savvinskaya_3 --max-steps 300 --traffic-density 0.2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PDD_BENCH_DIR = SCRIPT_DIR.parent.parent.parent
SCENES_DIR_DEFAULT = SCRIPT_DIR / "scenes"

# Add paths for imports
sys.path.insert(0, str(PDD_BENCH_DIR))


def resolve_scene_dir(scenes_dir: Path, scene_name: str) -> Path:
    scene_dir = scenes_dir / scene_name
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene folder not found: {scene_dir}")
    return scene_dir


def load_scene_meta(scene_dir: Path) -> dict:
    """Load meta.json from a scene directory."""
    meta_path = scene_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {scene_dir}")
    with open(meta_path) as f:
        return json.load(f)


def build_catalog_row(scene_dir: Path, meta: dict, var_idx: int = 0) -> dict:
    """Build a catalog row from scene meta.json."""
    scene_name = meta.get("scene_name", scene_dir.name)
    
    # Find .net.xml file
    net_file = meta.get("net_file")
    if not net_file:
        net_files = list(scene_dir.glob("*.net.xml"))
        if net_files:
            net_file = net_files[0].name
        else:
            raise FileNotFoundError(f"No .net.xml file found in {scene_dir}")
    
    # Build relative path from scenes root
    net_path = f"{scene_dir.name}/{net_file}"
    
    # Deterministic seed
    seed = (hash(scene_name) + var_idx) % (2**32)
    
    return {
        "scene_id": f"sumo_{scene_name}",
        "sign_code": "2.4",  # yield sign
        "sign_id": 0,
        "road_id": str(meta.get("road_id", "")),
        "net_path": net_path,
        "sign_spawn_distance": 30.0,
        "distance_from_start": 0.0,
        "destination_lane_id": None,
        "n_lanes": 1,
        "spawn_lane_num": 0,
        "var_idx": var_idx,
        "seed": seed,
        "spawn_velocity_ms": 0.0,
        "source": "sumo",
    }


def build_env(catalog_row: dict, scenes_root: Path, traffic_density: float, max_steps: int):
    """Build the SUMO environment."""
    from envs.sumo_env import TrafficSignSumoEnv
    from envs.sumo_traffic_manager import SumoTrafficManager
    
    SumoTrafficManager.EGO_SAFE_RADIUS = 15
    
    sign_spawn_distance = max(float(catalog_row.get("sign_spawn_distance", 30.0)), 30.0)
    map_path = str(scenes_root / catalog_row["net_path"])
    
    config = dict(
        use_render=False,
        manual_control=False,
        use_mesh_terrain=False,
        log_level=logging.CRITICAL,
        map_name=map_path,
        sign_type="2.4",  # yield sign
        sign_spawn_distance=sign_spawn_distance,
        traffic_density=traffic_density,
        horizon=max_steps,
        tl_speed_factor=20.0,
        min_route_hops_after_spawn=10,
        max_route_hops_after_spawn=10,
        num_scenarios=100000,
        vehicle_config={"show_lidar": False},
    )
    
    if catalog_row.get("road_id"):
        config["vehicle_config"]["spawn_lane_index"] = catalog_row["road_id"]
    if catalog_row.get("spawn_lane_num") is not None:
        config["spawn_lane_num"] = int(catalog_row["spawn_lane_num"])

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


def make_policy(policy_type: str, vehicle, seed: int):
    """Create a policy instance."""
    from metadrive.policy.idm_policy import IDMPolicy
    
    if policy_type == "idm":
        return IDMPolicy(vehicle, seed)
    
    # Try to import advanced policies
    try:
        from policies.modified_idm import ModifiedIDMPolicy
        if policy_type == "modified_idm":
            return ModifiedIDMPolicy(vehicle, seed)
    except ImportError:
        pass
    
    try:
        from policies.comprehensive_rule_expert import ComprehensiveRuleExpertPolicy
        if policy_type == "comprehensive_rule_expert":
            return ComprehensiveRuleExpertPolicy(vehicle, seed)
    except ImportError:
        pass
    
    try:
        from policies.rule_compliant_expert import RuleCompliantExpertPolicy
        if policy_type == "rule_compliant":
            return RuleCompliantExpertPolicy(vehicle, seed)
    except ImportError:
        pass
    
    # Fallback to IDM
    print(f"Policy '{policy_type}' not found, falling back to IDM")
    return IDMPolicy(vehicle, seed)


def main():
    parser = argparse.ArgumentParser(
        description="Run IDM simulation on SUMO scene and save GIF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("scene", help="Scene folder name under scenes/ (e.g. savvinskaya_3)")
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=SCENES_DIR_DEFAULT,
        help=f"Scenes root directory (default: {SCENES_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output GIF path (default: scenes/<scene>/simulation.gif)",
    )
    parser.add_argument("--policy", default="idm",
                        choices=["idm", "modified_idm", "comprehensive_rule_expert", "rule_compliant"],
                        help="Ego policy (default: idm)")
    parser.add_argument("--max-steps", type=int, default=600, help="Max simulation steps (default: 600)")
    parser.add_argument("--traffic-density", type=float, default=0.0,
                        help="Traffic density 0.0-1.0 (default: 0.0, no other vehicles)")
    parser.add_argument("--var-idx", type=int, default=0, help="Variation index for seed")
    parser.add_argument("--gif-duration-ms", type=int, default=40, help="GIF frame duration in ms (default: 40)")
    parser.add_argument("--scaling", type=float, default=12.0, help="Top-down view scaling (default: 12.0)")
    parser.add_argument("--screen-size", type=int, default=800, help="Screen size for GIF (default: 800)")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.CRITICAL)

    scenes_dir = Path(args.scenes_dir)
    scene_dir = resolve_scene_dir(scenes_dir, args.scene)

    out_gif = args.out if args.out is not None else scene_dir / "simulation.gif"
    out_gif = out_gif.resolve()
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    # Load scene
    print(f"Loading scene: {scene_dir}")
    meta = load_scene_meta(scene_dir)
    catalog_row = build_catalog_row(scene_dir, meta, var_idx=args.var_idx)
    
    print(f"  scene_name: {meta.get('scene_name', scene_dir.name)}")
    print(f"  seed: {catalog_row['seed']}")

    # Set random seed
    seed = catalog_row["seed"]
    np.random.seed(seed)

    # Build environment
    print(f"\nBuilding environment (traffic_density={args.traffic_density})...")
    env = build_env(catalog_row, scenes_dir, args.traffic_density, args.max_steps)

    # Reset
    env_seed = (catalog_row["sign_id"] + args.var_idx) % 100000
    obs, info = env.reset(seed=env_seed)
    print(f"  Environment reset OK")

    # Create policy
    policy = make_policy(args.policy, env.vehicle, seed)
    print(f"  Policy: {args.policy}")

    # Run simulation
    print(f"\nRunning {args.max_steps} steps...")
    total_reward = 0.0
    for step in range(args.max_steps):
        action = policy.act(env.vehicle.name)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Record frame
        try:
            env.render(
                mode="top_down",
                film_size=(2400, 2400),
                scaling=args.scaling,
                screen_size=(args.screen_size, args.screen_size),
                semantic_map=True,
                semantic_broken_line=True,
                draw_target_vehicle_trajectory=True,
                target_agent_heading_up=True,
                screen_record=True,
                window=False,
            )
        except Exception as e:
            if step == 0:
                print(f"  Warning: render failed: {e}")

        if terminated or truncated:
            break

        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}...")

    # Results
    arrived = info.get("arrive_dest", False)
    crashed = info.get("crash", False) or info.get("out_of_road", False)
    print(f"\nDone: steps={step + 1}, arrived={arrived}, crashed={crashed}, reward={total_reward:.2f}")

    # Save GIF
    try:
        if hasattr(env, "top_down_renderer") and env.top_down_renderer is not None:
            env.top_down_renderer.generate_gif(str(out_gif), duration=args.gif_duration_ms)
            print(f"\nGIF saved: {out_gif} ({out_gif.stat().st_size / 1024:.0f} KB)")
        else:
            print("Warning: no top_down_renderer - GIF not saved")
    except Exception as e:
        print(f"Error saving GIF: {e}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
