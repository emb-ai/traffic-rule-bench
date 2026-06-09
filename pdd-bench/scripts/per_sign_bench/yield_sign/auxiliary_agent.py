"""
Auxiliary agents for yield sign scenarios.

Creates stationary NPC vehicles on specified lanes near the intersection.

Usage:
    from scripts.per_sign_bench.yield_sign.auxiliary_agent import add_auxiliary_agents

    # After env.reset()
    aux_mgr = add_auxiliary_agents(
        env,
        spawn_lane_indices=["lane_46710989#1_0", "lane_46934779#0_0"],
        distance_from_intersection=5.0,
    )
"""

from __future__ import annotations

import logging
from typing import List, Optional

from metadrive.manager.base_manager import BaseManager
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.policy.base_policy import BasePolicy


DEFAULT_DISTANCE_FROM_INTERSECTION = 0  # meters before intersection (end of lane)
MIN_SPAWN_LANE_LENGTH = 30.0  # meters


class StationaryPolicy(BasePolicy):
    """Policy that keeps the vehicle completely stationary at 0 m/s."""

    def __init__(self, control_object, random_seed: int):
        super().__init__(control_object=control_object, random_seed=random_seed)

    def act(self, *args, **kwargs):
        return [0.0, -0.5]


class AuxiliaryAgentsManager(BaseManager):
    """Manager that spawns one stationary vehicle per incoming lane."""

    def __init__(
        self,
        spawn_lane_indices: List[str],
        distance_from_intersection: float = DEFAULT_DISTANCE_FROM_INTERSECTION,
    ):
        super().__init__()
        self._spawn_lane_indices = list(spawn_lane_indices)
        self._distance_from_intersection = distance_from_intersection
        self._aux_vehicles: List[BaseVehicle] = []

    def reset(self):
        self._aux_vehicles = []

    def after_reset(self):
        self._spawn_auxiliary_vehicles()

    def _spawn_auxiliary_vehicles(self):
        from metadrive.component.vehicle.vehicle_type import DefaultVehicle

        road_network = self.engine.current_map.road_network
        self._aux_vehicles = []

        for spawn_lane_index in self._spawn_lane_indices:
            lane = road_network.get_lane(spawn_lane_index)

            spawn_long = max(1.0, lane.length - self._distance_from_intersection)

            # SUMO maps use EdgeRoadNetwork: spawn_lane_index must be an edge id
            # (e.g. "46710990#1"), not a full lane key ("lane_46710990#1_0").
            # Passing a full lane key makes find_rightmost_lane_by_road_id() fail and
            # fall back to a random lane via os.urandom — different every subprocess.
            raw_name = spawn_lane_index[5:] if spawn_lane_index.startswith("lane_") else spawn_lane_index
            edge_id = raw_name.rsplit("_", 1)[0] if "_" in raw_name else raw_name

            vehicle_config = {
                "spawn_lane_index": edge_id,
                "spawn_longitude": lane.length,
                "spawn_lateral": 0.0,
                "enable_reverse": False,
                "navigation_module": None,
            }

            try:
                aux_vehicle = self.spawn_object(
                    DefaultVehicle,
                    vehicle_config=vehicle_config,
                )
                aux_vehicle.set_velocity([0.0, 0.0], in_local_frame=True)
                self.add_policy(
                    aux_vehicle.id,
                    StationaryPolicy,
                    aux_vehicle,
                    self.generate_seed(),
                )
                self._aux_vehicles.append(aux_vehicle)
                logging.info(
                    f"[AuxAgent] Spawned on {spawn_lane_index} at {spawn_long:.1f}m "
                    f"(lane_length={lane.length:.1f}m)"
                )
            except Exception as e:
                logging.warning(f"[AuxAgent] Failed to spawn on {spawn_lane_index}: {e}")

    def before_step(self):
        if not self._aux_vehicles:
            return {}
        for aux_vehicle in self._aux_vehicles:
            try:
                policy = self.engine.get_policy(aux_vehicle.name)
                if policy is not None:
                    action = policy.act()
                    aux_vehicle.before_step(action)
            except Exception as e:
                logging.debug(f"[AuxAgent] Policy execution error: {e}")
        return {}

    def after_step(self, *args, **kwargs):
        for aux_vehicle in self._aux_vehicles:
            try:
                aux_vehicle.after_step()
            except Exception:
                pass
        return {}

    @property
    def auxiliary_vehicles(self) -> List[BaseVehicle]:
        return list(self._aux_vehicles)

    @property
    def auxiliary_vehicle(self) -> Optional[BaseVehicle]:
        """First auxiliary vehicle (backward compatibility)."""
        return self._aux_vehicles[0] if self._aux_vehicles else None

    def get_status(self) -> dict:
        if not self._aux_vehicles:
            return {"exists": False, "count": 0, "agents": []}

        agents = []
        for aux_vehicle, lane_index in zip(self._aux_vehicles, self._spawn_lane_indices):
            try:
                agents.append({
                    "spawn_lane": lane_index,
                    "position": list(aux_vehicle.position),
                    "speed_mps": float(aux_vehicle.speed) if hasattr(aux_vehicle, "speed") else 0.0,
                })
            except Exception:
                agents.append({"spawn_lane": lane_index, "error": "status unavailable"})

        return {
            "exists": True,
            "count": len(self._aux_vehicles),
            "agents": agents,
        }


def add_auxiliary_agents(
    env,
    spawn_lane_indices: List[str],
    distance_from_intersection: float = DEFAULT_DISTANCE_FROM_INTERSECTION,
) -> Optional[AuxiliaryAgentsManager]:
    """
    Add stationary auxiliary agents on multiple lanes.
    !!! TODO: DOES NOT work when spawning several agents on different lanes
    """
    if not spawn_lane_indices:
        return None

    if not hasattr(env, "engine") or env.engine is None:
        logging.error("[AuxAgent] Environment has no engine")
        return None

    manager = AuxiliaryAgentsManager(
        spawn_lane_indices=spawn_lane_indices,
        distance_from_intersection=distance_from_intersection,
    )
    env.engine.register_manager("auxiliary_agent_manager", manager)
    manager.after_reset()
    return manager


def add_auxiliary_agent(
    env,
    spawn_lane_index: str,
    distance_from_intersection: float = DEFAULT_DISTANCE_FROM_INTERSECTION,
) -> Optional[AuxiliaryAgentsManager]:
    """Add a single stationary auxiliary agent (backward compatibility)."""
    return add_auxiliary_agents(
        env,
        spawn_lane_indices=[spawn_lane_index],
        distance_from_intersection=distance_from_intersection,
    )
