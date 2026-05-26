from traffic_signs.base_traffic_sign import BaseTrafficSign, same_road_check
from metadrive.engine.asset_loader import AssetLoader

class StopSign(BaseTrafficSign):
    # How far *before* the stop line we monitor and accept a near-stop. Base class
    # uses 10 m; for stop signs 5 m is enough and avoids long false “in zone” runs.
    APPROACH_BEFORE_LINE_M = 7.5

    def __init__(self, lane, zone_length=None, **kwargs):
        super().__init__(lane, icon_path="2.5.png", **kwargs)
        self._vehicle_states_stop = {}
        
        # Zone of action: small zone around the sign (not to end of lane)
        # Sign position is the stop line by default
        self.stop_line_position = float(self.placement_long)
        
        self.zone_start = max(0.0, self.stop_line_position - self.APPROACH_BEFORE_LINE_M)
        if zone_length is None:
            # Default zone extends from zone_start to zone_after
            self.zone_end = self.stop_line_position + self.DEFAULT_ZONE_AFTER
            self.zone_length = self.zone_end - self.zone_start
        else:
            self.zone_length = max(0.0, float(zone_length))
            self.zone_end = self.zone_start + self.zone_length

    def _create_visual_model(self):
        from panda3d.core import NodePath
        model_path = AssetLoader.file_path("models", "traffic_sign", "stop_sign.gltf")
        model = self.loader.loadModel(model_path)
        model.setPos(0, 0, self.sign_height)
        model.setH(-90)
        self._visual_model = model.instanceTo(self.origin)

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

        vid = vehicle.id
        in_zone = self.zone_start <= veh_long <= self.zone_end

        # Clear approach memory when not in the sign zone (new run at the sign / re-approach).
        if not in_zone:
            self._vehicle_states_stop.pop(vid, None)
            return False

        if vid not in self._vehicle_states_stop:
            self._vehicle_states_stop[vid] = {"stopped_before_line": False}
        state = self._vehicle_states_stop[vid]

        stop_long = self.stop_line_position
        # Treat as "stopped" in sim (same order of magnitude as idle creep).
        speed_stop_threshold_mps = 0.5

        # A legal stop is anywhere before the line within the approach zone, not
        # only in the last 2 m. (Approach length is ``APPROACH_BEFORE_LINE_M``.)
        if veh_long < stop_long and float(getattr(vehicle, "speed", 0.0) or 0.0) < speed_stop_threshold_mps:
            state["stopped_before_line"] = True

        # After crossing slightly past the line, fail if we never had a stop before it.
        if veh_long >= stop_long + 0.3:
            if not state["stopped_before_line"]:
                return True
        return False

    def get_rule_description(self) -> str:
        return "Must stop before stop line"
    
    @property
    def top_down_color(self):
        return [255, 255, 255]
    
    @property
    def top_down_color_name(self):
        return "yellow"


__all__ = ["StopSign"]
