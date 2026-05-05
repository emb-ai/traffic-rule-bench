from metadrive.envs.top_down_env import TopDownMetaDrive
from traffic_signs.traffic_sign_manager import TrafficSignManager


def build_topdown_env(observation_cls):
    class _TopDownMetaDriveWithStopSigns(TopDownMetaDrive):
        def setup_engine(self):
            super().setup_engine()
            self.engine.register_manager("traffic_sign_manager", TrafficSignManager())

        def get_single_observation(self, _=None):
            return observation_cls(
                self.config["vehicle_config"],
                onscreen=False,
                clip_rgb=self.config["norm_pixel"],
                frame_stack=self.config["frame_stack"],
                post_stack=self.config["post_stack"],
                frame_skip=self.config["frame_skip"],
                resolution=(self.config["resolution_size"], self.config["resolution_size"]),
                max_distance=self.config["distance"],
            )

    return _TopDownMetaDriveWithStopSigns


__all__ = ["build_topdown_env"]
