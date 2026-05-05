import logging
import sys
from pathlib import Path


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

from envs.traffic_sign_env import TrafficSignEnv
from traffic_signs.stop_sign import StopSign
from traffic_signs.speed_limit_sign import SpeedLimitSign30
from traffic_signs.no_stopping_allowed_sign import NoStoppingAllowedSign
from traffic_signs.no_overtaking_sign import NoOvertakingSign


config = {
    "map": "T",
    "manual_control": True,
    "use_render": False,
    "window_size": (1200, 800),
    "show_coordinates": True,
    "log_level": logging.CRITICAL,
    "vehicle_config": {
        "show_lidar": False,
        "enable_reverse": True,
    },
    "traffic_density": 0.5,
}

env = TrafficSignEnv(config)

try:
    obs, info = env.reset()



    sign_mgr = env.engine.traffic_sign_manager
    

    stop_sign = sign_mgr.add_sign(
        NoOvertakingSign,
        # longitudinal_offset=-3.0
    )

    # no_stop_sign = sign_mgr.add_sign(
    #     NoStoppingAllowedSign,
    #     # zone_length=30.0
    # )
    
    # stop_sign = sign_mgr.add_sign(
    #     SpeedLimitSign30,
    #     zone_length=150
    # )

    print("✅ Signs added. Controls: Arrows / WASD.")
    
    sign_info_lines = []
    for sign in sign_mgr.signs:
        sign_type = type(sign).__name__
        if hasattr(sign, 'top_down_color_name'):
            color_str = sign.top_down_color_name
            sign_info_lines.append(f"{sign_type} - {color_str}")

    c_r = 0
    violations_sum = 0 
    for step in range(50000):
        if config["manual_control"]:
            action = [0.0, 0.0]
        else:
            action = env.action_space.sample()

        o, r, terminated, truncated, info = env.step(action)
        c_r += r
        done = terminated or truncated
        vehicle = env.vehicle

        violations = sign_mgr.check_all_violations(vehicle)
        for sign, violated in violations:
            if violated:
                violations_sum += 1
                print(f"❌ Violation: {sign.get_rule_description()}")

        text_dict = {
            "Step": step,
            "Speed": f"{vehicle.speed_km_h:.2f} km/h",
            "Violations":
                violations_sum,
            "Reward": c_r,
            "Current lane: ": vehicle.lane.index
        }

        # for i, info_s in enumerate(sign_info_lines):
        #     text_dict[f"Sign {i+1}"] = info_s
    
        env.render(
            mode="top_down",
            text=text_dict,
            film_size=(2000, 2000),
            semantic_map=False
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
