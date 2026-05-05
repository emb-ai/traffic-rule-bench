from envs.traffic_sign_env import TrafficSignEnv
from traffic_signs.speed_limit_sign import SpeedLimitSign
from traffic_signs.end_of_zone_signs import EndOfSpeedLimitSign, EndOfAllRestrictionsSign, EndOfZoneSpeedLimitSign
from traffic_signs.no_stopping_allowed_sign import NoStoppingAllowedSign
from traffic_signs.zone_signs import ZoneSpeedLimitSign
from metadrive.component.map.base_map import BaseMap 

import logging


config = {
    "map": "SSXS",
    # "X" = Just an intersection
    # "T" = T-intersection
    # "O" = Roundabout
    # "C" = Curve
    # "SS" = Two straights (no intersection)
    "manual_control": True,
    "use_render": False,
    "window_size": (1200, 800),
    "show_coordinates": True,
    "log_level": logging.CRITICAL,
    "horizon": 10000,
    "vehicle_config": {
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

        speed = sign_mgr.add_sign(
            SpeedLimitSign,
            lane=lane,
            speed_limit=20,
            longitudinal_offset=30
        )
        zone = sign_mgr.add_sign(
            ZoneSpeedLimitSign,
            lane=lane,
            speed_limit=40,
            longitudinal_offset=80
        )
        end = sign_mgr.add_sign(
            EndOfZoneSpeedLimitSign,
            lane=lane,
            speed_limit=40,
            longitudinal_offset=120
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
                "Reward": c_r,
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

