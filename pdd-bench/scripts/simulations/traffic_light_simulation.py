from metadrive.envs.top_down_env import TopDownMetaDrive
from traffic_light_sign import TrafficLightSign


class TrafficLightSimulation:
    def __init__(self):
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
        self.env = None
        self.no_stop_sign = None

    def setup_environment(self):
        self.env = TopDownMetaDrive(self.config)
        return self.env.reset()

    def add_sign(self):
        if self.env is None:
            return

        lane = self.env.current_map.road_network.graph[">>"][">>>"][0]


        self.no_stop_sign = self.env.engine.spawn_object(
            TrafficLightSign,
            lane=lane,
            times_map = {"green" : 500, "red" : 300, "yellow" : 50},
            random_seed=5
        )


    def run_simulation(self):
        try:
            self.setup_environment()
            self.add_sign()
            time = 0
            current_mode = None
            if self.no_stop_sign:
                current_mode = self.no_stop_sign.get_mode()
            for step in range(2000):
                if self.no_stop_sign:
                    if time > self.no_stop_sign.get_time()[current_mode]:
                        self.no_stop_sign.change_mode()
                        current_mode = self.no_stop_sign.get_mode()
                        time = 0

                action = [0, 0.1]
                o, r, d, t, info = self.env.step(action)

                vehicle = self.env.vehicle
                violation = False

                if self.no_stop_sign:
                    violation = self.no_stop_sign.check_violation(vehicle)
                    if violation:
                        print("❌ VIOLATION! go on red light.")

                self.env.render(
                    mode="top_down",
                    text={
                        "Violation": violation,
                        "Speed": f"{vehicle.speed:.2f}",
                        "mode" : f"{current_mode}",
                        "step" : time,
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

                time += 1
        except KeyboardInterrupt:
            pass
        finally:
            if self.env:
                self.env.close()