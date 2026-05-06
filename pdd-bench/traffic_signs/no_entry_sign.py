from traffic_signs.base_traffic_sign import BaseTrafficSign, same_road_check
from metadrive.engine.asset_loader import AssetLoader
import numpy as np
    
class NoEntrySign(BaseTrafficSign):
    
    def __init__(self, lane, zone_length=None, **kwargs):
        super().__init__(lane, icon_path="no_entry_sign.jpg", **kwargs)
        self._vehicle_states_local = {}
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

        vid = vehicle.id
        if vid not in self._vehicle_states_local:
            self._vehicle_states_local[vid] = {"driving_before_line": False}

        state = self._vehicle_states_local[vid]

        stop_long = self.sign_line_position

        dist_to_line = stop_long - veh_long
        if veh_long < stop_long and dist_to_line <= 2.0:
            state["driving_before_line"] = True

        if veh_long >= stop_long + 0.3:
            if state["driving_before_line"]:
                return True
        return False


    def get_rule_description(self) -> str:
        return "No entry (sign 3.1) — entry forbidden for all vehicles in this direction"

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


__all__ = ["NoEntrySign"]