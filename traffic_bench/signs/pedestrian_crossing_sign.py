"""Pedestrian crossing sign (PDD 5.19) — visual plate only.

Yield-to-pedestrian enforcement is handled by PedestrianYieldRule +
CrosswalkPedestrianManager (see envs/pedestrian_manager.py). This class
only places the 5.19 icon beside the approach lane.
"""

from traffic_bench.signs.base_traffic_sign import BaseTrafficSign


class PedestrianCrossingSign(BaseTrafficSign):
    """PDD 5.19 pedestrian crossing plate (icon)."""

    def __init__(self, lane, **kwargs):
        kwargs.setdefault("icon_path", "5.19.png")
        super().__init__(lane, **kwargs)

    def _create_visual_model(self):
        pass

    def _is_violating(self, vehicle) -> bool:
        # Violations come from PedestrianYieldRule, not this plate.
        return False

    def get_rule_description(self) -> str:
        return (
            "Pedestrian crossing (5.19): drivers must yield to pedestrians "
            "on the marked crosswalk (enforced via PedestrianYieldRule)"
        )
