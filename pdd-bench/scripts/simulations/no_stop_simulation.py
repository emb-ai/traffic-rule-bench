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

from metadrive.envs.top_down_env import TopDownMetaDrive
from traffic_signs.no_stopping_allowed_sign import NoStoppingAllowedSign


class NoStopSimulation:
    def __init__(self, start_only=True):
        self.config = {
            "num_scenarios": 1,
            "traffic_density": 0.0,
            "manual_control": True,
            "use_render": False,
            "debug": False,
            "map": "S",
            "window_size": (1200, 800),
            "show_coordinates": True,
            "vehicle_config": {
                "show_lidar": True,
                "enable_reverse": True,
                "show_dest_mark": True
            }
        }
        self.startOnly = start_only
        self.env = None
        self.no_stop_sign = None

    def setup_environment(self):
        self.env = TopDownMetaDrive(self.config)
        return self.env.reset()

    def add_no_stop_sign(self, lng, lat=0, gap=30, lat_offset=5):
        if self.env is None:
            return

        lane = self.env.current_map.road_network.graph[">>"][">>>"][0]

        position = [lng, lat - lat_offset]

        zone_length = None if self.startOnly else gap

        self.no_stop_sign = self.env.engine.spawn_object(
            NoStoppingAllowedSign,
            lane=lane,
            zone_length=zone_length,
            random_seed=5
        )

        if not self.startOnly:
            end_position = [lng + gap, lat - lat_offset]
            self.env.engine.spawn_object(
                NoStoppingAllowedSign,
                lane=lane,
                position=end_position,
                is_end_sign=True,
                random_seed=6
            )

    def run_simulation(self):
        try:
            self.setup_environment()
            self.add_no_stop_sign(30, 0, gap=30, lat_offset=5)

            for step in range(2000):
                action = [0, 0.1]
                o, r, d, t, info = self.env.step(action)

                vehicle = self.env.vehicle
                violation = False

                if self.no_stop_sign:
                    violation = self.no_stop_sign.check_violation(vehicle)
                    if violation:
                        print("❌ VIOLATION! Stop in a restricted area.")

                self.env.render(
                    mode="top_down",
                    text={
                        "Violation": violation,
                        "Speed": f"{vehicle.speed:.2f}"
                    },
                    film_size=(2000, 2000)
                )

                if d:
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
            pass
        finally:
            if self.env:
                self.env.close()
