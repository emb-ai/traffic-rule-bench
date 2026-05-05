from typing import Optional
from traffic_signs.base_traffic_sign import BaseTrafficSign
from traffic_signs.speed_limit_sign import SpeedLimitSign
from traffic_signs.no_stopping_allowed_sign import NoStoppingAllowedSign
from traffic_signs.zone_signs import ZoneSpeedLimitSign
from traffic_signs.no_overtaking_sign import NoOvertakingSign
from traffic_signs.only_auto_sign import OnlyAutoSign

RESTRICTED_ZONE_SIGN_TYPES = (
    SpeedLimitSign,
    ZoneSpeedLimitSign,
    NoStoppingAllowedSign,
    NoOvertakingSign,
    OnlyAutoSign,
)


class BaseEndOfZoneSign(BaseTrafficSign):
    def __init__(
        self,
        lane,
        icon_path: Optional[str] = None,
        longitudinal_offset: float = 0.0,
        **kwargs
    ):
        super().__init__(
            lane,
            icon_path=icon_path,
            longitudinal_offset=longitudinal_offset,
            **kwargs
        )
    
    def update_zones(self):
        self._terminate_zones()
    
    def _terminate_zones(self):
        pass
    
    def _create_visual_model(self):
        pass
    
    def _is_violating(self, vehicle) -> bool:
        return False
    
    @property
    def top_down_color(self):
        return [200, 200, 200]
    
    @property
    def top_down_color_name(self):
        return "gray"

    def _truncate_zone(self, sign):
        sign.zone_end = min(sign.zone_end, self.placement_long)
        sign.zone_length = max(0.0, sign.zone_end - sign.zone_start)
        sign._is_active = False


class EndOfSpeedLimitSign(BaseEndOfZoneSign):
    def __init__(
        self,
        lane,
        speed_limit: float = 20,
        longitudinal_offset: float = 0.0,
        **kwargs
    ):
        self.speed_limit = speed_limit
        icon_path = f"3.25_{int(speed_limit)}.png"
        
        super().__init__(
            lane, 
            icon_path=icon_path, 
            longitudinal_offset=longitudinal_offset, 
            **kwargs
        )
    
    def _terminate_zones(self):
        for sign in self.engine.traffic_sign_manager.get_signs_before(self, SpeedLimitSign):
            if sign.speed_limit == self.speed_limit:
                self._truncate_zone(sign)
    
    def get_rule_description(self) -> str:
        return f"End of {self.speed_limit} km/h speed limit zone"


class EndOfSpeedLimitSign20(EndOfSpeedLimitSign):
    def __init__(self, lane, **kwargs):
        super().__init__(lane, speed_limit=20, **kwargs)


class EndOfSpeedLimitSign30(EndOfSpeedLimitSign):
    def __init__(self, lane, **kwargs):
        super().__init__(lane, speed_limit=30, **kwargs)


class EndOfSpeedLimitSign40(EndOfSpeedLimitSign):
    def __init__(self, lane, **kwargs):
        super().__init__(lane, speed_limit=40, **kwargs)


class EndOfSpeedLimitSign60(EndOfSpeedLimitSign):
    def __init__(self, lane, **kwargs):
        super().__init__(lane, speed_limit=60, **kwargs)



class EndOfZoneSpeedLimitSign(BaseEndOfZoneSign):
    def __init__(
        self,
        lane,
        speed_limit: float = 20,
        longitudinal_offset: float = 0.0,
        **kwargs
    ):
        self.speed_limit = speed_limit
        icon_path = f"5.32_{int(speed_limit)}.png"
        
        super().__init__(
            lane, 
            icon_path=icon_path, 
            longitudinal_offset=longitudinal_offset, 
            **kwargs
        )
    
    def _terminate_zones(self):
        for sign in self.engine.traffic_sign_manager.get_signs_before(self, ZoneSpeedLimitSign):
            if sign.speed_limit == self.speed_limit:
                self._truncate_zone(sign)
    
    def get_rule_description(self) -> str:
        return f"End of zone speed limit ({self.speed_limit} km/h)"


class EndOfZoneSpeedLimitSign20(EndOfZoneSpeedLimitSign):
    def __init__(self, lane, **kwargs):
        super().__init__(lane, speed_limit=20, **kwargs)


class EndOfZoneSpeedLimitSign30(EndOfZoneSpeedLimitSign):
    def __init__(self, lane, **kwargs):
        super().__init__(lane, speed_limit=30, **kwargs)


class EndOfZoneSpeedLimitSign40(EndOfZoneSpeedLimitSign):
    def __init__(self, lane, **kwargs):
        super().__init__(lane, speed_limit=40, **kwargs)


class EndOfZoneSpeedLimitSign60(EndOfZoneSpeedLimitSign):
    def __init__(self, lane, **kwargs):
        super().__init__(lane, speed_limit=60, **kwargs)


class EndOfAllRestrictionsSign(BaseEndOfZoneSign):
    def __init__(
        self, 
        lane,
        longitudinal_offset: float = 0.0,
        **kwargs
    ):
        super().__init__(
            lane, 
            icon_path="3.31.png", 
            longitudinal_offset=longitudinal_offset, 
            **kwargs
        )
    
    def _terminate_zones(self):
        for sign in self.engine.traffic_sign_manager.get_signs_before(self, RESTRICTED_ZONE_SIGN_TYPES):
            self._truncate_zone(sign)
    
    def get_rule_description(self) -> str:
        return "End of all restrictions zone"


class EndOfNoOvertakingSign(BaseEndOfZoneSign):
    def __init__(self, lane, longitudinal_offset: float = 0.0, **kwargs):
        super().__init__(
            lane,
            icon_path="3.21.png",
            longitudinal_offset=longitudinal_offset,
            **kwargs,
        )

    def _terminate_zones(self):
        for sign in self.engine.traffic_sign_manager.get_signs_before(self, NoOvertakingSign):
            self._truncate_zone(sign)

    def get_rule_description(self) -> str:
        return "End of overtaking prohibition zone"


class EndOfOnlyAutoSign(BaseEndOfZoneSign):
    def __init__(self, lane, longitudinal_offset: float = 0.0, **kwargs):
        super().__init__(
            lane,
            icon_path="5.4.png",
            longitudinal_offset=longitudinal_offset,
            **kwargs,
        )

    def _terminate_zones(self):
        for sign in self.engine.traffic_sign_manager.get_signs_before(self, OnlyAutoSign):
            self._truncate_zone(sign)

    def get_rule_description(self) -> str:
        return "End of passenger-only lane restriction"
