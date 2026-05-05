import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from metadrive.component.map.base_map import BaseMap
from metadrive.utils.doc_utils import generate_gif


def _find_pdd_bench_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for parent in (current, *current.parents):
        if (parent / "envs").is_dir() and (parent / "traffic_signs").is_dir():
            return parent
    raise RuntimeError("Could not locate pdd-bench root")


SCRIPT_PATH = Path(__file__).resolve()
PDD_BENCH_DIR = _find_pdd_bench_root(SCRIPT_PATH)
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"

for path in (PDD_BENCH_DIR, METADRIVE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from traffic_signs.traffic_sign_obs import TrafficSignObservation
from envs.sumo_env import TrafficSignSumoEnv

def create_env(need_monitor=False, scene_dir=""):
    meta_path = scene_dir / "meta.json"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    net_file = meta["net_file"]
    map_path = scene_dir / net_file
    
    road_id = meta["road_id"]

    env_config = dict(
        use_render=False,
        vehicle_config={"spawn_lane_index": road_id},
        manual_control=False,
        use_mesh_terrain=False,
        map_name=str(map_path.absolute()),
        agent_observation = TrafficSignObservation,
        discrete_action=True,
        
    )
    env = TrafficSignSumoEnv(env_config)
    if need_monitor:
        env = Monitor(env)
    return env

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument("--sign-type", type=str, required=True, help='e.g. "2.5"')
    parser.add_argument("--test_seeds", type=str, default="", help='Comma-separated seeds, e.g. "42,100,200". If empty, use sign_id as seed.')
    parser.add_argument(
        "--scenes-dir",
        type=str,
        default=str(PDD_BENCH_DIR / "scenes"),
        help="Path to scenes root",
    )
    parser.add_argument("--max_train_scenes", type=int, default=5, required=True, help="Limit number of scenes to run")
    args = parser.parse_args()

    scenes_root = Path(args.scenes_dir)
    sign_type_dir = scenes_root / args.sign_type
    
    if not sign_type_dir.exists():
        raise ValueError(f"No scenes found for sign type '{args.sign_type}' in {sign_type_dir}")

    scene_dirs = sorted([d for d in sign_type_dir.iterdir() if d.is_dir()])
    if args.max_train_scenes:
        scene_dirs = scene_dirs[:args.max_train_scenes]

    if not scene_dirs:
        raise ValueError(f"No scene directories in {sign_type_dir}")

        
    if args.test_seeds.strip():
        test_seeds = [int(s) for s in args.test_seeds.split(",")]
    else:
        test_seeds = None


    env=create_env(False, scene_dirs[0])
    env.reset()
    # ret = env.render(mode="topdown",
    #                 screen_record=True,
    #                 window=False,                
    #                 screen_size=(600, 600))
    env.close()
    # plt.axis("off")
    # plt.imshow(ret)

    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
    from stable_baselines3.common.utils import set_random_seed
    from functools import partial
    from IPython.display import clear_output
    import os
        

    set_random_seed(0)
    # 4 subprocess to rollout
    train_env=SubprocVecEnv([partial(create_env, True, scene_dir) for scene_dir in scene_dirs])
    model = PPO("MlpPolicy",
                train_env,
                n_steps=4096,
                verbose=1)
    model.learn(total_timesteps=1000 if os.getenv('TEST_DOC') else 300_000,
                log_interval=4)

    clear_output()
    print("Training is finished! Generate gif ...")

    # evaluation
    for test_seed in test_seeds:
        total_reward = 0
        env=create_env(False, scene_dirs[0])
        obs, _ = env.reset()
        try:
            for i in range(1000):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                ret = env.render(mode="topdown",
                                screen_record=True,
                                screen_size=(600, 600),
                                semantic_map=True)
                if done:
                    print("episode_reward", total_reward)
                    break

            env.top_down_renderer.generate_gif()
        finally:
            env.close()
    print("gif generation is finished ...")
