import argparse
import logging
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
PDD_BENCH_DIR = REPO_ROOT / "traffic_bench"
SDC_ROOT = REPO_ROOT
METADRIVE_DIR = SDC_ROOT / "third_party" / "metadrive"

for path in (METADRIVE_DIR,):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from metadrive.component.lane.point_lane import PointLane
from metadrive.component.vehicle.vehicle_type import SVehicle
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs import BaseEnv
from metadrive.manager.base_manager import BaseManager
from metadrive.manager.sumo_map_manager import SumoMapManager
from metadrive.obs.observation_base import DummyObservation
from metadrive.policy.idm_policy import TrajectoryIDMPolicy
from metadrive.utils.pg.utils import ray_localization

from metadrive.envs.top_down_env import TopDownMetaDrive

from metadrive.policy.idm_policy import ManualControlPolicy

from traffic_bench.signs.stop_sign import StopSign
from traffic_bench.signs.no_stopping_allowed_sign import NoStoppingAllowedSign
from traffic_bench.signs.speed_limit_sign import SpeedLimitSign

SIGN_TYPE_TO_CLASS = {
    "2.5": StopSign,
    "3.27": NoStoppingAllowedSign,
    "3.24": SpeedLimitSign
    
}

from traffic_bench.envs.sumo_env import TrafficSignSumoEnv
        

import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--road-id", type=str, required=True)
    parser.add_argument("--sign-type", type=str, default="2.5")
    parser.add_argument("--map-name", type=str, default="map.net.xml")
    args = parser.parse_args()
    
    env = TrafficSignSumoEnv(
        dict(
            use_render=False,
            vehicle_config={"spawn_lane_index": args.road_id},
            manual_control=True,
            # traffic_density=0.1,
            use_mesh_terrain=False,
            log_level=logging.CRITICAL,
            window_size =  (1200, 800),
            show_coordinates = True,
            map_name=args.map_name,
            sign_type=args.sign_type
        )
    )
    env.reset()

    c_r = 0
    violations_sum = 0 
    for i in range(1000):
        obs, reward, termination, truncate, info, = env.step(env.action_space.sample())
        c_r += reward
        if env.agent.crash_vehicle:
            print("💥 I collided with a car!")


        vehicle = env.agent
        violations = env.engine.traffic_sign_manager.check_all_violations(vehicle)
        for sign, violated in violations:
            if violated:
                violations_sum += 1
                print(f"❌ Violation: {sign.get_rule_description()}")
            
        text_dict = {
            "Step": i,
            "Speed": f"{vehicle.speed_km_h:.2f} km/h",
            "Violations":
                violations_sum,
            "Reward": c_r
        }
        


        env.render(
            mode="top_down",
            text=text_dict,
            film_size=(2000, 2000),
            semantic_map=True,
            draw_target_vehicle_trajectory=True
        )

        if termination:
            if info["arrive_dest"]:
                print("🎯 Success! I reached the goal.")
                break
            elif info["out_of_road"]:
                print("⚠️ I went off the road!")
            elif info.get("crash", False):
                print("💥 There was a collision")
            elif info["max_step"]:
                print("⏳ The time of the episode is over.")
            else:
                print("⏹️ The episode ended for a difwwwwwwwwferent reason.")
            # break
    env.close()
