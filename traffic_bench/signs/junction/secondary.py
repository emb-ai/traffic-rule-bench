from traffic_bench.signs.base import BaseTrafficSign


class SecondaryRoadSign(BaseTrafficSign):
    """Sign 2.3.1 – Intersection with a secondary road (X crossroad variant)."""

    def __init__(self, lane, intersection_name: str = None, **kwargs):
        super().__init__(lane, icon_path="secondary_road.png", **kwargs)
        self.intersection_name = intersection_name
        self.is_priority_sign = True
        self.priority_type = "secondary_road_ahead"

    def _is_violating(self, vehicle) -> bool:
        return False

    def get_rule_description(self) -> str:
        return "Intersection with secondary road ahead (2.3.1) - you have priority"

    @property
    def top_down_color(self):
        return [255, 204, 0]

    @property
    def top_down_color_name(self):
        return "yellow"


class SecondaryRoadLeftSign(BaseTrafficSign):
    """Sign 2.3.3 – Secondary road on the left (T crossroad variant)."""

    def __init__(self, lane, intersection_name: str = None, **kwargs):
        super().__init__(lane, icon_path="secondary_road_left.png", **kwargs)
        self.intersection_name = intersection_name
        self.is_priority_sign = True
        self.priority_type = "secondary_road_left"

    def _is_violating(self, vehicle) -> bool:
        return False

    def get_rule_description(self) -> str:
        return "Secondary road on the left (2.3.3) - you have priority"

    @property
    def top_down_color(self):
        return [255, 204, 0]

    @property
    def top_down_color_name(self):
        return "yellow"



class SecondaryRoadRightSign(BaseTrafficSign):
    """Sign 2.3.2 – Secondary road on the right (T crossroad variant)."""

    def __init__(self, lane, intersection_name: str = None, **kwargs):
        super().__init__(lane, icon_path="secondary_road_right.png", **kwargs)
        self.intersection_name = intersection_name
        self.is_priority_sign = True
        self.priority_type = "secondary_road_right"

    def _is_violating(self, vehicle) -> bool:
        return False

    def get_rule_description(self) -> str:
        return "Secondary road on the right (2.3.2) - you have priority"

    @property
    def top_down_color(self):
        return [255, 204, 0]

    @property
    def top_down_color_name(self):
        return "yellow"

