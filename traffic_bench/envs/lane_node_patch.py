"""SUMO LaneNode tweaks for PDD benches without patching the metadrive package."""

from __future__ import annotations


_LIGHT_MOTOR_ALLOW = frozenset(
    {"motorcycle", "moped", "bicycle", "delivery", "passenger", "bus", "taxi"}
)


def apply_sumo_lane_node_patch() -> None:
    """Import motorcycle/moped/bicycle lanes as drivable when configured."""
    from metadrive.utils.sumo.map_utils import LaneNode

    if getattr(LaneNode, "_traffic_bench_lane_node_patch_applied", False):
        return

    LaneNode.TREAT_LIGHT_VEHICLE_AS_DRIVING = False

    @classmethod
    def allows_light_motor_vehicle(cls, sumolib_obj) -> bool:
        return any(sumolib_obj.allows(token) for token in _LIGHT_MOTOR_ALLOW)

    @classmethod
    def _coerce_light_motor_to_driving(cls, sumolib_obj) -> bool:
        return cls.TREAT_LIGHT_VEHICLE_AS_DRIVING and cls.allows_light_motor_vehicle(
            sumolib_obj
        )

    _orig_init = LaneNode.__init__

    def _patched_init(self, sumolib_obj):
        _orig_init(self, sumolib_obj)
        if self.type != "sidewalk":
            return
        if not LaneNode._coerce_light_motor_to_driving(sumolib_obj):
            return
        self.type = "driving"
        raw_width = sumolib_obj.getWidth()
        if LaneNode.MIN_LANE_WIDTH > 0:
            self.width = max(raw_width, LaneNode.MIN_LANE_WIDTH)
        else:
            self.width = raw_width

    LaneNode.allows_light_motor_vehicle = allows_light_motor_vehicle
    LaneNode._coerce_light_motor_to_driving = _coerce_light_motor_to_driving
    LaneNode.__init__ = _patched_init
    LaneNode._traffic_bench_lane_node_patch_applied = True
