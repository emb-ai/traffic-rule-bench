#!/usr/bin/env python3
"""Debug: Run simulation on a scene and save GIF."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Path setup - tools/ is inside stop_sign/
TOOLS_DIR = Path(__file__).resolve().parent
STOP_SIGN_DIR = TOOLS_DIR.parent
PDD_BENCH_DIR = STOP_SIGN_DIR.parent.parent.parent
SDC_ROOT = PDD_BENCH_DIR.parent
SCENES_ROOT = STOP_SIGN_DIR / "scenes"
CHECKPOINTS_DIR = PDD_BENCH_DIR / "checkpoints"

# Add paths for imports
sys.path.insert(0, str(PDD_BENCH_DIR))
sys.path.insert(0, str(STOP_SIGN_DIR))

from lib.sumo_utils import resolve_net_file, load_scene_meta, find_first_edge_id, DEFAULT_NET_FILE

# Policy categories
IDM_FAMILY = {"idm", "modified_idm", "comprehensive_rule_expert"}
NN_NEED_CHECKPOINT = {"carl", "carl_rule", "plant2", "plant2_rule"}
NN_NO_CHECKPOINT = {"rule_compliant", "ppo_lidar"}
ALL_POLICIES = IDM_FAMILY | NN_NEED_CHECKPOINT | NN_NO_CHECKPOINT

# Default checkpoint paths
DEFAULT_MODEL_PATHS = {
    "carl": CHECKPOINTS_DIR / "carl" / "nuplan_51479_1B" / "model_best.pth",
    "carl_rule": CHECKPOINTS_DIR / "carl" / "nuplan_51479_1B" / "model_best.pth",
    "plant2": CHECKPOINTS_DIR / "plant2_finetuned" / "plant2_supervised_2nd_final.pt",
    "plant2_rule": CHECKPOINTS_DIR / "plant2_finetuned" / "plant2_supervised_2nd_final.pt",
}


def build_catalog_row(scene_dir: Path, meta: dict, var_idx: int = 0) -> dict:
    """Build a catalog row from scene meta.json."""
    scene_name = meta.get("scene_name", scene_dir.name)
    
    net_file = resolve_net_file(scene_dir, meta)
    
    # Build relative path from scenes root
    net_path = f"{scene_dir.name}/{net_file}"
    
    # Get road_id from meta or find first edge in network
    road_id = meta.get("road_id", "")
    if not road_id:
        net_full_path = scene_dir / net_file
        road_id = find_first_edge_id(net_full_path) or ""
        if road_id:
            print(f"  auto-selected road_id: {road_id}")
    
    # Deterministic seed
    seed = (hash(scene_name) + var_idx) % (2**32)
    
    return {
        "scene_id": f"sumo_{scene_name}",
        "sign_code": "2.5",  # stop sign
        "sign_id": 0,
        "road_id": road_id,
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
        sign_type="2.5",  # stop sign
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


def load_policy_models(policy: str, model_path: str | None, plant2_action_mode: str = "pid"):
    """Load model checkpoints and resolve the policy class.
    
    Returns dict with "policy_cls" key.
    """
    policy_cls = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if policy == "carl":
        if not model_path:
            raise ValueError("--model-path is required for --policy carl")
        from agents.policies.plain_carl_policy import PlainCarlPolicy
        PlainCarlPolicy.set_checkpoint(model_path, device=device)
        policy_cls = PlainCarlPolicy
        
    elif policy == "plant2":
        if not model_path:
            raise ValueError("--model-path is required for --policy plant2")
        PLANT2_PATH = SDC_ROOT / "plant2"
        from agents.policies.plain_plant2_policy import PlainPlanT2Policy
        PlainPlanT2Policy.set_checkpoint(
            model_path, PLANT2_PATH, device=device, action_mode=plant2_action_mode,
        )
        policy_cls = PlainPlanT2Policy
        
    elif policy == "carl_rule":
        if not model_path:
            raise ValueError("--model-path is required for --policy carl_rule")
        from agents.policies.carl_sign_compliant import CarlSignCompliantPolicy
        CarlSignCompliantPolicy.set_checkpoint(model_path, device=device)
        policy_cls = CarlSignCompliantPolicy
        
    elif policy == "plant2_rule":
        if not model_path:
            raise ValueError("--model-path is required for --policy plant2_rule")
        PLANT2_PATH = SDC_ROOT / "plant2"
        from agents.policies.plant2_sign_compliant import PlanT2SignCompliantPolicy
        PlanT2SignCompliantPolicy.set_checkpoint(
            model_path, PLANT2_PATH, device=device, action_mode=plant2_action_mode,
        )
        policy_cls = PlanT2SignCompliantPolicy
    
    return {"policy_cls": policy_cls}


def make_policy(policy_type: str, vehicle, seed: int, models: dict | None = None):
    """Create a policy instance."""
    from metadrive.policy.idm_policy import IDMPolicy
    
    # IDM family
    if policy_type == "idm":
        return IDMPolicy(vehicle, seed)
    
    if policy_type == "modified_idm":
        from agents.policies.modified_idm_sign_compliant import ModifiedIDMSignCompliantPolicy
        return ModifiedIDMSignCompliantPolicy(vehicle, seed)
    
    if policy_type == "comprehensive_rule_expert":
        from agents.policies.comprehensive_rule_expert import ComprehensiveRuleExpertPolicy
        return ComprehensiveRuleExpertPolicy(vehicle, seed)
    
    if policy_type == "rule_compliant":
        from agents.policies.rule_compliant_expert import RuleCompliantExpertPolicy
        return RuleCompliantExpertPolicy(vehicle, seed)
    
    if policy_type == "ppo_lidar":
        from metadrive.policy.expert_policy import ExpertPolicy
        return ExpertPolicy(vehicle, seed)
    
    # NN policies (CARL, PLANT)
    if policy_type in NN_NEED_CHECKPOINT:
        if models is None or models.get("policy_cls") is None:
            raise RuntimeError(f"policy_cls for --policy {policy_type} not loaded")
        policy_cls = models["policy_cls"]
        return policy_cls(vehicle, seed)
    
    # Fallback to IDM
    print(f"Policy '{policy_type}' not found, falling back to IDM")
    return IDMPolicy(vehicle, seed)


def resolve_model_path(policy: str, model_path: str | None) -> str | None:
    """Resolve model path, using default if not provided."""
    if model_path:
        return model_path
    
    if policy in NN_NEED_CHECKPOINT:
        default = DEFAULT_MODEL_PATHS.get(policy)
        if default and default.is_file():
            print(f"Using default checkpoint for {policy}: {default}")
            return str(default)
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run simulation (IDM/CARL/PLANT) on SUMO scene and save GIF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("scene", help="Scene name (e.g. savvinskaya_3, check)")
    parser.add_argument("--out", default=None, help="Output GIF path (default: scenes/<scene>/simulation.gif)")
    parser.add_argument("--policy", default="idm",
                        choices=sorted(ALL_POLICIES),
                        help="Policy to run (default: idm)")
    parser.add_argument("--model-path", default=None,
                        help="Checkpoint path for CARL/PLANT policies. "
                             "If not provided, uses default from checkpoints/")
    parser.add_argument("--plant2-action-mode", default="pid",
                        choices=["pid", "wps_pure_pursuit"],
                        help="PLANT2 action mode (default: pid)")
    parser.add_argument("--max-steps", type=int, default=600,
                        help="Max simulation steps (default: 600)")
    parser.add_argument("--traffic-density", type=float, default=0.0,
                        help="Traffic density 0.0-1.0 (default: 0.0)")
    parser.add_argument("--var-idx", type=int, default=0,
                        help="Variation index for seed")
    parser.add_argument("--gif-duration-ms", type=int, default=40,
                        help="GIF frame duration in ms (default: 40)")
    parser.add_argument("--scaling", type=float, default=24.0,
                        help="Top-down view scaling (default: 24.0)")
    parser.add_argument("--screen-size", type=int, default=800,
                        help="Screen size for GIF (default: 800)")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.CRITICAL)

    # Resolve scene directory
    scene_dir = SCENES_ROOT / args.scene
    if not scene_dir.exists():
        # Try as full path
        scene_dir = Path(args.scene)
        if not scene_dir.exists():
            sys.exit(f"Scene not found: {args.scene}\n"
                     f"  Tried: {SCENES_ROOT / args.scene}\n"
                     f"  Tried: {Path(args.scene)}")
    scene_dir = scene_dir.resolve()

    # Default output path
    if args.out:
        out_gif = Path(args.out).resolve()
    else:
        out_gif = scene_dir / f"simulation-{args.policy}.gif"
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    # Load scene
    print(f"Loading scene: {scene_dir}")
    meta = load_scene_meta(scene_dir)
    catalog_row = build_catalog_row(scene_dir, meta, var_idx=args.var_idx)
    
    print(f"  scene_name: {meta.get('scene_name', scene_dir.name)}")
    print(f"  policy: {args.policy}")

    # Resolve model path for NN policies
    model_path = resolve_model_path(args.policy, args.model_path)
    if args.policy in NN_NEED_CHECKPOINT and not model_path:
        default = DEFAULT_MODEL_PATHS.get(args.policy)
        sys.exit(f"--model-path required for {args.policy}. "
                 f"Default not found at: {default}")

    # Load policy models if needed
    models = None
    if args.policy in NN_NEED_CHECKPOINT:
        print(f"  checkpoint: {model_path}")
        models = load_policy_models(args.policy, model_path, args.plant2_action_mode)

    # Set random seed
    seed = catalog_row["seed"]
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    # Build environment
    print(f"\nBuilding environment (traffic_density={args.traffic_density})...")
    env = build_env(catalog_row, SCENES_ROOT, args.traffic_density, args.max_steps)

    # Reset
    env_seed = (catalog_row["sign_id"] + args.var_idx) % 100000
    obs, info = env.reset(seed=env_seed)
    print(f"  Environment reset OK")

    # Create policy
    policy = make_policy(args.policy, env.vehicle, seed, models)
    print(f"  Policy created: {type(policy).__name__}")

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
                film_size=(4800, 4800),
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
