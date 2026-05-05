from envs.traffic_sign_env import TrafficSignEnv
from traffic_signs.no_turn_allowed import NoLeftTurnSign, NoRightTurnSign, NoUTurnSign
import logging


SIGN_ROAD_FROM = ">>>"
SIGN_ROAD_TO = "1S0_0_"
SIGN_LANE_INDEX = 0

config = {
    "map": "ST",
    "out_of_road_done": False,
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


def get_sign_lane(road_network, lane_index=0):
    lanes = road_network.graph.get(SIGN_ROAD_FROM, {}).get(SIGN_ROAD_TO, [])
    assert lanes, f"Sign road not found: {SIGN_ROAD_FROM} -> {SIGN_ROAD_TO}"
    idx = min(lane_index, len(lanes) - 1)
    return lanes[idx]


def main():
    env = TrafficSignEnv(config)

    try:
        env.reset()
        sign_mgr = env.engine.traffic_sign_manager
        road_network = env.engine.current_map.road_network

        sign_lane = get_sign_lane(road_network, lane_index=SIGN_LANE_INDEX)
        sign = sign_mgr.add_sign(
            NoRightTurnSign,
            lane=sign_lane,
            longitudinal_offset=0
        )
        sign_mgr.build_zones()

        violations_sum = 0
        for step in range(config["horizon"]):
            action = [0.0, 0.0]
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            vehicle = env.vehicle

            for traffic_sign in sign_mgr.signs:
                if traffic_sign._is_violating(vehicle):
                    violations_sum += 1

            text_dict = {
                "Step": step,
                "Speed": f"{vehicle.speed_km_h:.2f} km/h",
                "Violations": violations_sum,
                "Lane": vehicle.lane.index,
                "Lane closest": road_network.get_closest_lane_index(vehicle.position)[0],
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

