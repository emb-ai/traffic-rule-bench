from traffic_bench.signs.base_traffic_sign import BaseTrafficSign, same_road_check

class NoStoppingAllowedSign(BaseTrafficSign):
    HEIGHT = 3.0
    LENGTH = 0.15
    WIDTH = 0.8
    def __init__(self, lane, zone_length=None, **kwargs):
        super().__init__(lane, icon_path="3.27.png", **kwargs)
        if not zone_length:
            self.zone_length = self.lane.length - self.placement_long
        else:
            self.zone_length = zone_length
        # Clamp zone_end to lane boundary
        self.zone_start = self.placement_long
        self.zone_end = min(self.lane.length, self.placement_long + self.zone_length)

    def _create_visual_model(self):
        pass

    def _is_violating(self, vehicle) -> bool:
        veh_lane = getattr(vehicle, "lane", None)
        if veh_lane is None:
            return False
        if not same_road_check(
            getattr(veh_lane, "index", None),
            getattr(self.lane, "index", None),
        ):
            return False
        veh_long = self.lane.local_coordinates(vehicle.position)[0]
        if self.zone_start <= veh_long <= self.zone_end and vehicle.speed < 0.1:
            return True
        return False

    def get_rule_description(self) -> str:
        return f"No stopping allowed for {self.zone_length}m ahead"

    @property
    def top_down_color(self):
        return [0, 255, 0]
    
    @property
    def top_down_color_name(self):
        return "green"