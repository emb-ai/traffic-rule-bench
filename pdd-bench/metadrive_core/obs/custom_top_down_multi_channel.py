"""
TopDownMultiChannel wrapper for multiple number of metadrive envs's bev obseravtions 
num_stacks = 3 (vs origin is 2 + frame_stack)
stack_traffic_flow is on 
"""

from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel


class CustomTopDownMultiChannel(TopDownMultiChannel):
    def __init__(self, vehicle_config, clip_rgb, onscreen, resolution, max_distance, frame_stack=1, post_stack=1, frame_skip=1):
        super().__init__(
            vehicle_config, clip_rgb, onscreen=onscreen, 
            resolution=resolution, max_distance=max_distance,
            frame_stack=frame_stack, post_stack=post_stack, frame_skip=frame_skip
        )

        self.num_stacks = 3
    def observe(self, *args, **kwargs):
        obs = super().observe(*args, **kwargs)
        return obs




