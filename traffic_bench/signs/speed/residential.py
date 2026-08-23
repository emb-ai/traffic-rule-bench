"""Residential-zone signs 5.21 (residential zone) / 5.22 (end of residential zone).

Modeled as a fixed 20 km/h speed-limit ZONE (per RF traffic code §17 a residential zone
caps speed at 20 km/h). 5.21 opens the zone, 5.22 closes it — exactly the
start/end-of-zone pattern of 5.31/5.32, but the limit is ALWAYS 20 (we do NOT
read it from the road's `lane.speed`).
"""
import os

from traffic_bench.signs.base import BaseTrafficSign, ICONS_DIR
from traffic_bench.signs.speed.zone import ZoneSpeedLimitSign
from traffic_bench.signs.speed.end_of_zone import EndOfZoneSpeedLimitSign

RESIDENTIAL_ZONE_LIMIT_KMH = 20


def _icon_or_fallback(preferred: str, fallback: str) -> str:
    """Use `preferred` icon if it exists in icons/, else `fallback` (cosmetic;
    icon is only loaded for --save-gifs rendering)."""
    return preferred if os.path.exists(os.path.join(ICONS_DIR, preferred)) else fallback


class ResidentialZoneSign(ZoneSpeedLimitSign):
    """5.21 — residential zone entry: a 20 km/h zone in effect until 5.22.

    Unlike ZoneSpeedLimitSign (which derives the limit from `lane.speed`), the
    residential-zone limit is fixed at 20 km/h regardless of the road.
    """

    def __init__(self, lane, longitudinal_offset: float = 0.0, **kwargs):
        self.speed_limit = RESIDENTIAL_ZONE_LIMIT_KMH
        lane.speed_limit = RESIDENTIAL_ZONE_LIMIT_KMH
        # Call the grandparent directly (skip ZoneSpeedLimitSign.__init__, which
        # would overwrite the limit from lane.speed).
        BaseTrafficSign.__init__(
            self, lane,
            longitudinal_offset=longitudinal_offset,
            longitudinal_from_start=True,
            icon_path=_icon_or_fallback("residential_zone.png", "zone_speed_20.png"),
            **kwargs,
        )
        self.zone_start = self.placement_long
        self.zone_length = float('inf')
        self.zone_end = float('inf')
        self._is_active = True
        self.zone_edges = None
        self.zone_end_s = None

    def get_rule_description(self) -> str:
        return "Residential zone: exceeding 20 km/h"


class EndOfResidentialZoneSign(EndOfZoneSpeedLimitSign):
    """5.22 — end of residential zone: closes the preceding 5.21 zone."""

    def __init__(self, lane, longitudinal_offset: float = 0.0, **kwargs):
        super().__init__(
            lane,
            speed_limit=RESIDENTIAL_ZONE_LIMIT_KMH,
            longitudinal_offset=longitudinal_offset,
            **kwargs,
        )
        self.icon_path = os.path.join(
            ICONS_DIR, _icon_or_fallback("end_residential_zone.png", "end_zone_speed_20.png"))

    def _terminate_zones(self):
        for sign in self.engine.traffic_sign_manager.get_signs_before(self, ResidentialZoneSign):
            self._truncate_zone(sign)

    def get_rule_description(self) -> str:
        return "End of residential zone"
