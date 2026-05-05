from traffic_signs.base_traffic_sign import BaseTrafficSign

class MinimumSpeedLimitSign(BaseTrafficSign):
    def __init__(self, lane, min_speed=40,
                 icon_path="minimum_speed50.png", zone_length=None, **kwargs):
        
        if hasattr(lane, 'speed'):
            self.min_speed = round(lane.speed * 3.6) - 10
        else:
            self.min_speed = min_speed
        lane.min_speed_limit = self.min_speed
        
        icon_map = {
            20: "minimum_speed50.png",
            30: "minimum_speed50.png",
            40: "minimum_speed50.png",
            50: "minimum_speed50.png",
            60: "minimum_speed50.png",
        }
        icon_path = icon_map.get(int(self.min_speed), "minimum_speed50.png")
        
        super().__init__(lane, 
                        icon_path=icon_path, **kwargs)
        
        self.zone_start = self.placement_long + 10
        if zone_length is None:
            self.zone_length = self.lane.length - self.zone_start
        else:
            self.zone_length = max(0.0, float(zone_length))
        
        self.zone_end = min(self.lane.length, self.zone_start + self.zone_length)

    def _create_visual_model(self):
        pass

    def _is_violating(self, vehicle) -> bool:
        try:
            veh_long = self.lane.local_coordinates(vehicle.position)[0]
        except Exception:
            return False
        if self.zone_start <= veh_long <= self.zone_end:
            if vehicle.speed_km_h < self.min_speed - 2.0:
                return True
        return False

    def get_rule_description(self) -> str:
        return f"Speed below minimum limit ({self.min_speed} km/h)"

    @property
    def top_down_color(self):
        return [255, 255, 255]

    @property
    def top_down_color_name(self):
        return "cyan"

class MinimumSpeedLimit30(MinimumSpeedLimitSign):
    def __init__(self, lane, zone_length=None, **kwargs):
        super().__init__(lane, min_speed=30, icon_path="minimum_speed30.png",
                         zone_length=zone_length, **kwargs)

class MinimumSpeedLimit40(MinimumSpeedLimitSign):
    def __init__(self, lane, zone_length=None, **kwargs):
        super().__init__(lane, min_speed=40, icon_path="minimum_speed40.png",
                         zone_length=zone_length, **kwargs)

class MinimumSpeedLimit50(MinimumSpeedLimitSign):
    def __init__(self, lane, zone_length=None, **kwargs):
        super().__init__(lane, min_speed=50, icon_path="minimum_speed50.png",
                         zone_length=zone_length, **kwargs)

class MinimumSpeedLimit60(MinimumSpeedLimitSign):
    def __init__(self, lane, zone_length=None, **kwargs):
        super().__init__(lane, min_speed=60, icon_path="minimum_speed60.png",
                         zone_length=zone_length, **kwargs)