from pdd_bench.envs.traffic_sign_env import TrafficSignEnv
from pdd_bench.signs.speed_limit_sign import SpeedLimitSign
from pdd_bench.signs.end_of_zone_signs import EndOfSpeedLimitSign, EndOfAllRestrictionsSign, EndOfZoneSpeedLimitSign
from pdd_bench.signs.no_stopping_allowed_sign import NoStoppingAllowedSign
from pdd_bench.signs.zone_signs import ZoneSpeedLimitSign
from metadrive.component.map.base_map import BaseMap
from pdd_bench.signs.no_overtaking_sign import NoOvertakingSign
from pdd_bench.signs.end_of_zone_signs import EndOfNoOvertakingSign, EndOfOnlyAutoSign
from pdd_bench.signs.only_auto_sign import OnlyAutoSign

import logging


config = {
    "map": "SSXS",
    # "X" = Just an intersection
    # "T" = T-intersection
    # "O" = Roundabout
    # "C" = Curve
    # "SS" = Two straights (no intersection)
    "out_of_road_done": False,
    "manual_control": True,
    "use_render": False,
    "window_size": (1200, 800),
    "show_coordinates": True,
    "log_level": logging.CRITICAL,
    "horizon": 10000,
    "vehicle_config": {
        # "vehicle_model": "xl",
        "show_lidar": False,
        "enable_reverse": True,
    },
    "traffic_density": 0.0,
}


def main():
    env = TrafficSignEnv(config)

    try:
        obs, info = env.reset()

        sign_mgr = env.engine.traffic_sign_manager
        lane = env.vehicle.lane

        sign = sign_mgr.add_sign(
            OnlyAutoSign,
            lane=lane,
            longitudinal_offset=30
        )
        end_sign = sign_mgr.add_sign(
            EndOfOnlyAutoSign,
            lane=lane,
            longitudinal_offset=60
        )
        sign_mgr.build_zones()
        
        c_r = 0.0
        violations_sum = 0
        step = 0 

        for step in range(config["horizon"]):
            action = [0.0, 0.0]

            o, r, terminated, truncated, info = env.step(action)
            c_r += r
            done = terminated or truncated
            vehicle = env.vehicle

            for sign in sign_mgr.signs:
                if sign._is_violating(vehicle):
                    print(f"❌ Violation: {sign.get_rule_description()}")
                    violations_sum += 1
                else:
                    print(f"✅ No violations")

            text_dict = {
                "Step": step,
                "Speed": f"{vehicle.speed_km_h:.2f} km/h",
                "Violations": violations_sum,
                "Lane": vehicle.lane.index,
                "Lane closest": env.engine.current_map.road_network.get_closest_lane_index(vehicle.position)[0]
            }

            env.render(
                mode="top_down",
                text=text_dict,
                film_size=(2000, 2000),
                semantic_map=False,
            )

            if done:
                if info["arrive_dest"]:
                    print("🎯 Success! I reached the goal.")
                elif info["out_of_road"]:
                    print("⚠️ I went off the road!")
                elif info.get("crash", False):
                    print("💥 There was a collision")
                elif info["max_step"]:
                    print("⏳ The time of the episode is over.")
                else:
                    print("⏹️ The episode ended for a different reason.")
                break

    except KeyboardInterrupt:
        print("\n🛑 The simulation was interrupted by the user.")
    except Exception as e:
        print(f"⚠️ Error: {e}")
        raise
    finally:
        env.close()
        print("✅ Env is closed.")


if __name__ == "__main__":
    main()

