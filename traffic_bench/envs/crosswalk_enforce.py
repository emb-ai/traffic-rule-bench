from __future__ import annotations

from typing import Dict, List

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.manager.base_manager import BaseManager

from traffic_bench.signs.crosswalk.yield_rule import PedestrianYieldRule


class CrosswalkYieldEnforcerManager(BaseManager):
    """
    Force traffic vehicles to yield before occupied crosswalks.
    This manager is intended for NPC/IDM traffic only, not for ego control.
    """

    PRIORITY = 11

    def __init__(self):
        super().__init__()
        raw_cfg = self.engine.global_config.get("pedestrian_yield_enforcer", {})
        self._cfg = raw_cfg.get_dict() if hasattr(raw_cfg, "get_dict") else dict(raw_cfg)
        self.enabled = bool(self.engine.global_config.get("enforce_pedestrian_yield_for_traffic", False)) and bool(
            self._cfg.get("enabled", True)
        )
        self.steer_value = float(self._cfg.get("steer_value", 0.0))
        self.brake_value = float(self._cfg.get("brake_value", -1.0))
        self.max_forced_per_step = int(self._cfg.get("max_forced_per_step", -1))
        self._fallback_rule = PedestrianYieldRule()

    def before_step(self, *args, **kwargs):
        if not self.enabled:
            return {}

        rule = self._get_yield_rule()
        if rule is None:
            return {"crosswalk_yield_enforcer": {"forced_brake": 0}}

        forced = 0
        for vehicle in self._iter_traffic_vehicles():
            # Crosswalk yield only (pedestrians). TL compliance is handled
            # inside SumoTrajectoryIDMPolicy._should_stop_for_red().
            if not self._should_force_stop(vehicle, rule):
                continue
            self._apply_force_brake(vehicle)
            forced += 1
            if self.max_forced_per_step > 0 and forced >= self.max_forced_per_step:
                break

        return {"crosswalk_yield_enforcer": {"forced_brake": forced}}

    @staticmethod
    def _should_force_stop(vehicle, rule) -> bool:
        try:
            return bool(rule.should_vehicle_stop(vehicle))
        except Exception:
            return False

    def _apply_force_brake(self, vehicle):
        vehicle.before_step([self.steer_value, self.brake_value])

    def _should_stop_for_red_tl(self, vehicle) -> bool:
        """Check if vehicle should stop for a red traffic light.

        Same logic as SumoTrajectoryIDMPolicy._should_stop_for_red but applied
        externally via force-brake (like crosswalks). When light turns green
        this returns False → force brake stops → policy resumes cleanly.
        """
        try:
            sign_mgr = getattr(self.engine, "traffic_sign_manager", None)
            if sign_mgr is None:
                return False
            veh_lane = vehicle.lane
            if veh_lane is None:
                return False
            veh_idx = getattr(veh_lane, "index", None)

            for sign in sign_mgr.signs:
                if type(sign).__name__ != "TrafficLightSign":
                    continue
                sign_idx = getattr(sign.lane, "index", None)
                if veh_idx is None or sign_idx is None:
                    continue
                try:
                    if veh_idx[0] != sign_idx[0] or veh_idx[1] != sign_idx[1]:
                        continue
                except (IndexError, TypeError):
                    continue

                veh_long = sign.lane.local_coordinates(vehicle.position)[0]
                if not (sign.zone_start <= veh_long <= sign.zone_end):
                    continue
                if not sign.current_states:
                    continue

                # Check signals for directions reachable from vehicle's lane.
                # NPC vehicles on PointLane don't have `turns` → fall back to
                # checking ALL signal states (conservative: stop if any is red).
                veh_to_lanes = set()
                for turn in getattr(veh_lane, "turns", []) or []:
                    to = turn.get("to_lane") if isinstance(turn, dict) else None
                    if to:
                        veh_to_lanes.add(to)

                if veh_to_lanes:
                    relevant = [s for to_lane, s in sign.current_states.items()
                                if to_lane in veh_to_lanes]
                else:
                    # No turn info → use all states (conservative)
                    relevant = list(sign.current_states.values())

                if not relevant:
                    continue
                any_green = any(s in ("g", "G") for s in relevant)
                if not any_green:
                    dist_to_end = sign.lane.length - veh_long
                    if dist_to_end < 30:
                        return True
        except Exception:
            pass
        return False

    def _get_yield_rule(self):
        sign_manager = getattr(self.engine, "traffic_sign_manager", None)
        if sign_manager is None:
            return self._fallback_rule

        for rule in getattr(sign_manager, "rules", []):
            if isinstance(rule, PedestrianYieldRule):
                return rule
            if getattr(rule, "id", None) == "pedestrian_yield_rule" and hasattr(rule, "should_vehicle_stop"):
                return rule
        return self._fallback_rule

    def _iter_traffic_vehicles(self) -> List[BaseVehicle]:
        traffic_manager = getattr(self.engine, "traffic_manager", None)
        candidates: List[BaseVehicle] = []

        if traffic_manager is not None and hasattr(traffic_manager, "traffic_vehicles"):
            candidates = list(getattr(traffic_manager, "traffic_vehicles"))
        elif traffic_manager is not None and hasattr(traffic_manager, "vehicles"):
            candidates = list(getattr(traffic_manager, "vehicles"))
        else:
            candidates = list(self.engine.get_objects(lambda o: isinstance(o, BaseVehicle)).values())

        active_agent_ids = set()
        active_agent_names = set()
        agent_manager = getattr(self.engine, "agent_manager", None)
        if agent_manager is not None and hasattr(agent_manager, "active_agents"):
            active_agents = getattr(agent_manager, "active_agents")
            active_agent_ids = set(active_agents.keys())
            active_agent_names = {getattr(obj, "name", None) for obj in active_agents.values()}

        uniq: Dict[str, BaseVehicle] = {}
        for vehicle in candidates:
            if vehicle is None:
                continue
            vehicle_id = getattr(vehicle, "id", None)
            vehicle_name = getattr(vehicle, "name", None)
            if vehicle_id in active_agent_ids or vehicle_name in active_agent_names:
                continue
            uniq_key = str(vehicle_id if vehicle_id is not None else vehicle_name if vehicle_name is not None else id(vehicle))
            uniq[uniq_key] = vehicle
        return list(uniq.values())
