# This class extends TopDownMultiChannel to draw STOP signs in the own extra channel 


from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from metadrive.component.static_object.traffic_object import TrafficObject
from metadrive.utils import import_pygame
import numpy as np

import gymnasium as gym
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.obs.top_down_obs_impl import WorldSurface, COLOR_BLACK, ObjectGraphics, LaneGraphics, \
    ObservationWindowMultiChannel, ObservationWindow
import math
from collections import deque

from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.component.lane.point_lane import PointLane
from metadrive.constants import Decoration, DEFAULT_AGENT
from metadrive.utils import clip

from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.navigation_module.edge_network_navigation import EdgeNetworkNavigation
from metadrive.component.navigation_module.trajectory_navigation import TrajectoryNavigation

pygame = import_pygame()
COLOR_BLACK = pygame.Color("black")
DEFAULT_TRAJECTORY_LANE_WIDTH = 3


class TopDownMultiChannel6Channels(TopDownMultiChannel):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_stacks = 6  # road_network, past_pos, traffic_flow, target_vehicle, extra traffic_signs
    
    def init_canvas(self):
        super().init_canvas()
        # add new canvas
        self.canvas_signs = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        
    
    def init_obs_window(self):
        super().init_obs_window()
        #  ObservationWindow for canvas_signs
        self.obs_window_signs = ObservationWindow(
            max_range=(self.max_distance, self.max_distance),
            resolution=self.resolution
        )
    
    def draw_map(self) -> pygame.Surface:
        """
        canvas_signs setup like other canvas
        """
        super().draw_map()
        
        b_box = self.road_network.get_bounding_box()
        self.canvas_signs.fill(COLOR_BLACK)
        x_len = b_box[1] - b_box[0]
        y_len = b_box[3] - b_box[2]
        max_len = max(x_len, y_len) + 20  
        scaling = self.MAP_RESOLUTION[1] / max_len - 0.1
        assert scaling > 0

        # real-world distance * scaling = pixel in canvas
        self.canvas_signs.scaling = scaling

        centering_pos = ((b_box[0] + b_box[1]) / 2, (b_box[2] + b_box[3]) / 2)
        self.canvas_signs.move_display_window_to(centering_pos)
        
        # like obs_window.reset
        self.obs_window_signs.reset(self.canvas_signs)
        
    def draw_scene(self):
        ret = super().draw_scene()

        
        assert len(self.engine.agents) == 1, "Don't support multi-agent top-down observation yet!"
        vehicle = self.engine.agents[DEFAULT_AGENT]
        pos = self.canvas_runtime.pos2pix(*vehicle.position)
        clip_size = (int(self.obs_window.get_size()[0] * 1.1), int(self.obs_window.get_size()[0] * 1.1))
        self._refresh(self.canvas_signs, pos, clip_size)

        # draw STOP signs on canvas_signs like top_down_obs_with_signs.py
        signs_drawn = 0
        traffic_objects = None
        try:
            traffic_objects = self.engine.get_objects(
                lambda o: isinstance(o, TrafficObject) and hasattr(o, 'top_down_color')
            )

            if traffic_objects:
                for sign in traffic_objects.values():
                    if hasattr(sign, 'top_down_color') and hasattr(sign, 'top_down_length'):
                        if not self.canvas_signs.is_visible(sign.position):
                            continue
                        color = (
                            tuple(sign.top_down_color)
                            if isinstance(sign.top_down_color, list)
                            else sign.top_down_color
                        )
                        
                        w = self.canvas_signs.pix(sign.top_down_width)
                        h = self.canvas_signs.pix(sign.top_down_length)

                        position = [
                            *self.canvas_signs.pos2pix(sign.position[0], sign.position[1])
                        ]

                        heading = sign.heading_theta if hasattr(sign, 'heading_theta') else 0
                        angle = -np.rad2deg(heading)

                        box = [
                            pygame.math.Vector2(p) for p in
                            [(-h / 2, -w / 2), (-h / 2, w / 2),
                             (h / 2, w / 2), (h / 2, -w / 2)]
                        ]
                        box_rotate = [p.rotate(angle) + position for p in box]

                        pygame.draw.polygon(self.canvas_signs, color, box_rotate)

                        signs_drawn += 1
        except Exception as e:
            import traceback
            print(f"  Error rendering signs: {e}")
            traceback.print_exc()

        if not hasattr(self, '_sign_debug_count'):
            self._sign_debug_count = 0
        if self._sign_debug_count < 3:
            print(
                f"  !!! DEBUG: Signs found "
                f"{len(traffic_objects) if traffic_objects else 0}, Drawn: {signs_drawn}"
            )
            self._sign_debug_count += 1

        # 5) Render canvas_signs using obs_window_signs.render() for proper scaling and rotation like other canvases 
        if not hasattr(self, 'obs_window_signs') or self.obs_window_signs is None:
            from metadrive.obs.top_down_obs_impl import ObservationWindow
            self.obs_window_signs = ObservationWindow(
                max_range=(self.max_distance, self.max_distance),
                resolution=self.resolution
            )
            self.obs_window_signs.reset(self.canvas_signs)
        
        signs_rendered = self.obs_window_signs.render(
            self.canvas_signs,
            position=pos,
            heading=vehicle.heading_theta
        )
        ret["traffic_signs"] = signs_rendered
        self._last_signs_rendered = signs_rendered
        return ret
    
    def get_observation_window(self):
        ret = super().get_observation_window()
        ret["past_pos"] = self.canvas_past_pos
        # Use the rendered result from draw_scene, if available
        # Otherwise, get it from obs_window_signs
        if hasattr(self, '_last_signs_rendered'):
            ret["traffic_signs"] = self._last_signs_rendered
        else:
            # Fallback: obtained directly from obs_window_signs
            ret["traffic_signs"] = self.obs_window_signs.get_observation_window()
        return ret
    
    
    def observe(self, vehicle: BaseVehicle):
        self.render()
        surface_dict = self.get_observation_window()
        
        #  correct resolution 
        for key in ["road_network", "traffic_flow", "target_vehicle", "past_pos", "traffic_signs"]:
            if key in surface_dict and surface_dict[key] is not None:
                current_size = surface_dict[key].get_size()
                if current_size != self.resolution:
                    surface_dict[key] = pygame.transform.smoothscale(
                        surface_dict[key], self.resolution
                    )
        
        img_dict = {k: pygame.surfarray.array3d(surface) for k, surface in surface_dict.items()}
        
        # Gray scale
        img_dict = {k: self._transform(img) for k, img in img_dict.items()}
        
        if self._should_fill_stack:
            self.stack_past_pos.clear()
            self.stack_traffic_flow.clear()
            for _ in range(self.stack_traffic_flow.maxlen):
                self.stack_traffic_flow.append(img_dict["traffic_flow"])
            self._should_fill_stack = False
        self.stack_traffic_flow.append(img_dict["traffic_flow"])

        # traffic_flow
        current_traffic_flow = list(self.stack_traffic_flow)[-1] if len(self.stack_traffic_flow) > 0 else img_dict["traffic_flow"]
        additional_channel = list(self.stack_traffic_flow)[-2] if len(self.stack_traffic_flow) > 1 else img_dict["traffic_flow"]
        
        def check_channel(channel, name=""):
            if len(channel.shape) == 3:
                channel = channel[:, :, 0]
            elif len(channel.shape) != 2:
                raise ValueError(f"Channel {name} has an unexpected shape: {channel.shape}")
            return channel
        
        # Render channels
        road_network = check_channel(img_dict["road_network"], "road_network")
        past_pos = check_channel(img_dict["past_pos"], "past_pos")
        traffic_flow = check_channel(current_traffic_flow, "traffic_flow")
        target_vehicle = check_channel(img_dict["target_vehicle"], "target_vehicle")
        # traffic_signs from  img_dict
        traffic_signs = check_channel(img_dict.get("traffic_signs", np.zeros_like(road_network)), "traffic_signs")
        additional_channel = check_channel(additional_channel, "additional")
        
        # Combine all channels
        img = [
            road_network * 2, 
            past_pos,  
            traffic_flow,
            target_vehicle, 
            additional_channel,  
            traffic_signs,
        ]
        
        # Stack
        img = np.stack(img, axis=2)
        if self.norm_pixel:
            img = np.clip(img, 0, 1.0)
        else:
            img = np.clip(img, 0, 255)
        return np.transpose(img, (1, 0, 2))

    def draw_navigation_node(self, canvas, color=(128, 128, 128)):
        checkpoints = self.target_vehicle.navigation.checkpoints
        for i, c in enumerate(checkpoints[:-1]):
            lanes = self.road_network.graph[c][checkpoints[i + 1]]
            for lane in lanes:
                LaneGraphics.draw_drivable_area(lane, canvas, color=color)

    def draw_navigation_trajectory(self, canvas, color=(128, 128, 128)):
        lane = PointLane(self.target_vehicle.navigation.checkpoints, DEFAULT_TRAJECTORY_LANE_WIDTH)
        LaneGraphics.draw_drivable_area(lane, canvas, color=color)

    def _get_stack_indices(self, length, frame_skip=None):
        frame_skip = frame_skip or self.frame_skip
        num = int(math.ceil(length / frame_skip))
        indices = []
        for i in range(num):
            indices.append(length - 1 - i * frame_skip)
        return indices

    @property
    def observation_space(self):
        shape = self.obs_shape + (self.num_stacks, )
        if self.norm_pixel:
            return gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
        else:
            return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)