import numpy as np

from traffic_bench.signs.base import BaseTrafficSign
from traffic_bench.signs.junction.yield_sign import YieldSign


class RoundaboutSign(BaseTrafficSign):
    """Sign 4.3 — roundabout ahead (informational plate on the approach)."""

    def __init__(self, lane, intersection_name: str = None, **kwargs):
        super().__init__(lane, icon_path="roundabout.png", **kwargs)
        self.intersection_name = intersection_name
        self.is_priority_sign = True
        self.priority_type = "roundabout_ahead"

    def _is_violating(self, vehicle) -> bool:
        return False

    def get_rule_description(self) -> str:
        return "Roundabout ahead (4.3) — yield to traffic on the circle"

    @property
    def top_down_color(self):
        return [255, 204, 0]

    @property
    def top_down_color_name(self):
        return "yellow"


class RoundaboutYieldSign(YieldSign):
    """Invisible yield tracker for 4.3 — ego on spoke yields to ring traffic."""

    ENTRY_CONFLICT_BEFORE_M = 20.0
    ENTRY_CONFLICT_AFTER_M = 5.0

    def __init__(
        self,
        lane,
        intersection_name: str = None,
        ring_road_lanes: list = None,
        entry_incoming_lanes: list = None,
        entry_junction_xy: tuple[float, float] | list[float] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("show_model", False)
        kwargs["icon_path"] = None
        incoming = list(entry_incoming_lanes or [])
        conflict_lanes = incoming
        if not conflict_lanes:
            conflict_lanes = list(ring_road_lanes or [])
        super().__init__(
            lane,
            intersection_name=intersection_name,
            main_road_lanes=conflict_lanes,
            auto_detect_main_roads=False,
            **kwargs,
        )
        self.priority_type = "roundabout_yield"
        self.icon_path = None
        if entry_junction_xy is not None:
            self._entry_junction_xy = np.array(
                [float(entry_junction_xy[0]), float(entry_junction_xy[1])],
                dtype=np.float64,
            )
        else:
            self._entry_junction_xy = None

    def _junction_at_lane_end(self, lane) -> bool:
        """True when the entry junction is at ``lane.length`` (SUMO to-node end)."""
        if self._entry_junction_xy is None:
            return True
        try:
            p0 = np.array(lane.position(0.0, 0.0), dtype=np.float64)
            p1 = np.array(lane.position(float(lane.length), 0.0), dtype=np.float64)
        except Exception:
            return True
        return float(np.linalg.norm(p1 - self._entry_junction_xy)) <= float(
            np.linalg.norm(p0 - self._entry_junction_xy)
        )

    def _conflict_longitudinal_range(self, lane) -> tuple[float, float]:
        """Ring tail upstream of ego entry, plus a short downstream tail past lane end."""
        before_m = self.ENTRY_CONFLICT_BEFORE_M
        after_m = self.ENTRY_CONFLICT_AFTER_M
        return (
            max(0.0, float(lane.length) - before_m),
            float(lane.length) + after_m,
        )

    def _is_vehicle_in_main_road_conflict_zone(self, vehicle) -> bool:
        """True when ring traffic is in the monitored conflict arc at ego's entry."""
        if not self.main_road_lanes:
            return False

        try:
            vehicle_pos = vehicle.position
            vehicle_heading = vehicle.heading_theta
            vehicle_lane = getattr(vehicle, "lane", None)
            vehicle_lane_idx = getattr(vehicle_lane, "index", None)
            if vehicle_lane_idx is None:
                return False
            v_segment = (vehicle_lane_idx[0], vehicle_lane_idx[1])
        except Exception:
            return False

        main_segments = set()
        for ln in self.main_road_lanes:
            ln_idx = getattr(ln, "index", None)
            if ln_idx and len(ln_idx) >= 2:
                main_segments.add((ln_idx[0], ln_idx[1]))

        if v_segment in main_segments and vehicle_lane is not None:
            try:
                long_pos, lat_pos = vehicle_lane.local_coordinates(vehicle_pos)
                zone_start, zone_end = self._conflict_longitudinal_range(vehicle_lane)
                if zone_start <= long_pos <= zone_end and abs(lat_pos) <= vehicle_lane.width * 1.5:
                    return True
            except Exception:
                pass

        for lane in self.main_road_lanes:
            try:
                lane_idx = getattr(lane, "index", None)
                if lane_idx is None or len(lane_idx) < 2:
                    continue
                lane_segment = (lane_idx[0], lane_idx[1])
                long_pos, lat_pos = lane.local_coordinates(vehicle_pos)
                zone_start, zone_end = self._conflict_longitudinal_range(lane)

                if zone_start <= long_pos <= zone_end:
                    if abs(lat_pos) <= lane.width * 1.5:
                        if v_segment == lane_segment:
                            return True
                        if v_segment not in main_segments:
                            lane_heading = lane.heading_theta_at(
                                min(max(long_pos, 0.0), lane.length)
                            )
                            heading_diff = abs(vehicle_heading - lane_heading)
                            heading_diff = min(heading_diff, 2 * np.pi - heading_diff)
                            if heading_diff < np.pi / 2:
                                return True
            except Exception:
                continue
        return False

    def get_top_down_aux_conflict_zones(self) -> list[dict]:
        """Lane segments where auxiliary ring traffic triggers a 4.3 violation."""
        zones: list[dict] = []
        for lane in self.main_road_lanes:
            try:
                long_start, long_end = self._conflict_longitudinal_range(lane)
                zones.append(
                    {
                        "lane": lane,
                        "long_start": float(long_start),
                        "long_end": float(long_end),
                        "kind": "incoming",
                    }
                )
            except Exception:
                continue
        return zones

    def get_rule_description(self) -> str:
        return (
            "Roundabout (4.3) — must not leave the approach zone "
            "while traffic is present on the ring within "
            f"{self.ENTRY_CONFLICT_BEFORE_M:.0f} m upstream of the ego entry junction "
            f"(plus {self.ENTRY_CONFLICT_AFTER_M:.0f} m past the lane end)"
        )

