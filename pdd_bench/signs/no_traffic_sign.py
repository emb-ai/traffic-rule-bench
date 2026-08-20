from pdd_bench.signs.base_traffic_sign import BaseTrafficSign, same_road_check

class NoTrafficSign(BaseTrafficSign):


    def __init__(self, lane, icon_path="3.2.png", zone_length=None, **kwargs):
        super().__init__(lane, icon_path=icon_path, **kwargs)
        self.sign_line_position = float(self.placement_long)
        
        self.zone_start = max(0.0, self.sign_line_position - self.DEFAULT_ZONE_BEFORE)
        if zone_length is None:
            self.zone_end = self.sign_line_position + self.DEFAULT_ZONE_AFTER
            self.zone_length = self.zone_end - self.zone_start
        else:
            self.zone_length = max(0.0, float(zone_length))
            self.zone_end = self.zone_start + self.zone_length

    def _is_violating(self, vehicle) -> bool:
        veh_lane = getattr(vehicle, "lane", None)
        if veh_lane is None:
            return False
        if not same_road_check(
            getattr(veh_lane, "index", None),
            getattr(self.lane, "index", None),
        ):
            return False
        try:
            veh_long = self.lane.local_coordinates(vehicle.position)[0]
        except Exception:
            return False

        if not (self.zone_start <= veh_long <= self.zone_end):
            return False
        
        stop_long = self.sign_line_position

        if veh_long >= stop_long + 0.3:
            return True
        return False


    def get_rule_description(self) -> str:
        desc = "Closed to all vehicles (sign 3.2) – no entry for any vehicle"
        return desc

    @property
    def top_down_color(self):
        return [255, 0, 0]

    @property
    def top_down_color_name(self):
        return "red"

    @property
    def top_down_length(self):
        return 1.0

    @property
    def top_down_width(self):
        return 1.0


__all__ = ["NoTrafficSign"]