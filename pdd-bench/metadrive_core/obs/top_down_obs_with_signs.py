# This class extends TopDownMultiChannel to draw STOP signs in the road_network channel 

from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from metadrive.component.static_object.traffic_object import TrafficObject
import numpy as np
import pygame


class TopDownMultiChannelWithSigns(TopDownMultiChannel):
    """
    An extended version of TopDownMultiChannel that renders STOP signs.

    Signs are drawn in the road_network channel (canvas_road_network) in yellow.
    The observation shape remains unchanged: (H, W, 3).
    """
    
    def draw_scene(self):
        ret = super().draw_scene()
        
        #  STOP on canvas_road_network (road_network ch)
        signs_drawn = 0
        try:
            from metadrive.constants import DEFAULT_AGENT
            vehicle = self.engine.agents[DEFAULT_AGENT]
            
            traffic_objects = self.engine.get_objects(
                lambda o: isinstance(o, TrafficObject) and hasattr(o, 'top_down_color')
            )
            
            if traffic_objects:
                for sign in traffic_objects.values():
                    if hasattr(sign, 'top_down_color') and hasattr(sign, 'top_down_length'):
                        if not self.canvas_road_network.is_visible(sign.position):
                            continue
                        
                        color = tuple(sign.top_down_color) if isinstance(sign.top_down_color, list) else sign.top_down_color
                        
                        w = self.canvas_road_network.pix(sign.top_down_width)
                        h = self.canvas_road_network.pix(sign.top_down_length)
                        
                        position = [*self.canvas_road_network.pos2pix(sign.position[0], sign.position[1])]
                        
                        heading = sign.heading_theta if hasattr(sign, 'heading_theta') else 0
                        angle = -np.rad2deg(heading)
                        
                        box = [
                            pygame.math.Vector2(p) for p in 
                            [(-h / 2, -w / 2), (-h / 2, w / 2), (h / 2, w / 2), (h / 2, -w / 2)]
                        ]
                        box_rotate = [p.rotate(angle) + position for p in box]
                        
                        pygame.draw.polygon(self.canvas_road_network, color, box_rotate)
                        
                        signs_drawn += 1
        except Exception as e:
            import traceback
            print(f"Error rendering characters: {e}")
            traceback.print_exc()
        # debugging 
        if not hasattr(self, '_sign_debug_count'):
            self._sign_debug_count = 0
        if self._sign_debug_count < 3:
            print(f"!!! DEBUG: Signs found: {len(traffic_objects) if 'traffic_objects' in locals() and traffic_objects else 0}, Drawn: {signs_drawn}")
            self._sign_debug_count += 1
        
        # Redraw the result with the updated canvas_road_network (with signs)
        vehicle = self.engine.agents[DEFAULT_AGENT]
        pos = self.canvas_runtime.pos2pix(*vehicle.position)
        
        ret = self.obs_window.render(
            canvas_dict=dict(
                road_network=self.canvas_road_network,  #  STOP sign 
                traffic_flow=self.canvas_runtime,  
                target_vehicle=self.canvas_ego,
            ),
            position=pos,
            heading=vehicle.heading_theta
        )
        ret["past_pos"] = self.canvas_past_pos
        return ret