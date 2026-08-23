import numpy as np

from traffic_bench.signs.base import BaseTrafficSign


class MainRoadSign(BaseTrafficSign):
    def __init__(
        self, 
        lane, 
        intersection_name: str = None, 
        **kwargs
    ):
        super().__init__(
            lane, 
            icon_path="main_road.png", 
            **kwargs
        )
        self.intersection_name = intersection_name
        self.is_priority_sign = True
        self.priority_type = "main"
    
    def _is_violating(self, vehicle) -> bool:
        return False
    
    def get_rule_description(self) -> str:
        return "Main road - you have priority at the intersection"
    
    @property
    def top_down_color(self):
        return [255, 204, 0]
    
    @property  
    def top_down_color_name(self):
        return "yellow"


class EndMainRoadSign(BaseTrafficSign):
    def __init__(
        self, 
        lane, 
        intersection_name: str = None, 
        **kwargs
    ):
        super().__init__(
            lane, 
            icon_path="end_main_road.png", 
            **kwargs
        )
        self.intersection_name = intersection_name
        self.is_priority_sign = True
        self.priority_type = "end_main"
    
    def _is_violating(self, vehicle) -> bool:
        return False
    
    def get_rule_description(self) -> str:
        return "End of main road - priority road ends"
    
    @property
    def top_down_color(self):
        return [200, 200, 200]
    
    @property
    def top_down_color_name(self):
        return "grey"



from traffic_bench.signs.junction.yield_sign import YieldSign
from traffic_bench.signs.extra.traffic_light import TrafficLightSign


class EndMainRoadSmartSign(YieldSign):
    """2.2 logic:
    - if lane has traffic-light signals -> check as traffic-light rule
    - else -> behave like yield (2.4)
    """

    def __init__(self, lane, intersection_name: str = None, debug_priority: bool = True, **kwargs):
        tl_speed_factor = kwargs.get("tl_speed_factor", 1.0)
        icon_path = kwargs.pop("icon_path", "end_main_road.png")
        super().__init__(
            lane,
            intersection_name=intersection_name,
            main_road_lanes=None,
            debug_priority=debug_priority,
            icon_path=icon_path,
            **kwargs,
        )
        self.priority_type = "end_main_smart"
        self._tl_rule = None
        if getattr(lane, "tl_signals", None):
            self._tl_rule = TrafficLightSign(
                lane,
                sim_step_duration=self.engine.global_config.get("physics_world_step_size", 0.1),
                tl_speed_factor=tl_speed_factor,
                show_model=False,
            )
            
    def get_top_down_icon_poses(self):
        """Draw the icon only on the lane where the sign is placed."""
        road_network = getattr(getattr(self.engine, "current_map", None), "road_network", None)
        if road_network is None or self.lane is None:
            return []
        
        try:
            pos = self.lane.position(self.placement_long, self._lateral_offset)
            heading = self.lane.heading_theta_at(self.placement_long) + np.pi / 2
            return [(pos, heading)]
        except Exception:
            return []

    def update_state(self):
        if self._tl_rule is not None:
            self._tl_rule.update_state()

    def _is_violating(self, vehicle) -> bool:
        if self._tl_rule is not None:
            return self._tl_rule._is_violating(vehicle)
        return super()._is_violating(vehicle)
