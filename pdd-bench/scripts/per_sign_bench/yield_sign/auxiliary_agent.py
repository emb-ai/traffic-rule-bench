"""
Auxiliary agents for yield sign scenarios.

Spawns NPC vehicles on incoming lanes near the intersection. By default they
use IDM to drive through the junction to a reachable outgoing lane.

Usage:
    from scripts.per_sign_bench.yield_sign.auxiliary_agent import add_auxiliary_agents

    # After env.reset()
    aux_mgr = add_auxiliary_agents(
        env,
        spawn_lane_indices=["lane_46710989#1_0"],
        outgoing_lanes=outgoing_lanes,
        distance_from_intersection=5.0,
        policy="idm",
    )
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

from metadrive.manager.base_manager import BaseManager
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.component.navigation_module.edge_network_navigation import EdgeNetworkNavigation


DEFAULT_DISTANCE_FROM_INTERSECTION = 5.0
DEFAULT_SPAWN_VELOCITY_MS = 5.0
DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END = 5.0
AuxPolicyType = Literal["idm", "stationary"]


class StationaryPolicy(BasePolicy):
    """Policy that keeps the vehicle completely stationary at 0 m/s."""

    def __init__(self, control_object, random_seed: int):
        super().__init__(control_object=control_object, random_seed=random_seed)

    def act(self, *args, **kwargs):
        return [0.0, -0.5]


class AuxiliaryIDMPolicy(IDMPolicy):
    """IDM policy tuned for a single fixed route through the junction."""

    def __init__(self, control_object, random_seed: int):
        super().__init__(control_object=control_object, random_seed=random_seed)
        self.enable_lane_change = False
        self.enable_idm_overtake = False


class GatedAuxiliaryIDMPolicy(AuxiliaryIDMPolicy):
    """IDM that stays stopped until ego is near the end of its spawn lane."""

    def __init__(
        self,
        control_object,
        random_seed: int,
        ego_vehicle,
        ego_spawn_lane_index: str,
        release_distance_before_end: float = DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END,
        release_speed_ms: float = DEFAULT_SPAWN_VELOCITY_MS,
    ):
        super().__init__(control_object=control_object, random_seed=random_seed)
        self._ego_vehicle = ego_vehicle
        self._ego_spawn_lane_index = ego_spawn_lane_index
        self._release_distance_before_end = float(release_distance_before_end)
        self._release_speed_ms = float(release_speed_ms)
        self.released = self._release_distance_before_end <= 0

    def ego_distance_to_spawn_lane_end(self) -> float:
        """Meters from ego to the end of its spawn lane (along lane centerline)."""
        try:
            road_network = self.engine.current_map.road_network
            lane = road_network.get_lane(self._ego_spawn_lane_index)
            longitudinal, _ = lane.local_coordinates(self._ego_vehicle.position)
            return float(lane.length - longitudinal)
        except Exception:
            return float("inf")

    def act(self, *args, **kwargs):
        if not self.released:
            if self.ego_distance_to_spawn_lane_end() <= self._release_distance_before_end:
                self.released = True
                self.control_object.set_velocity(
                    [self._release_speed_ms, 0.0], in_local_frame=True
                )
                logging.info(
                    "[AuxAgent] Released IDM: ego within %.1fm of spawn lane end",
                    self._release_distance_before_end,
                )
            else:
                return [0.0, -0.5]
        return super().act(*args, **kwargs)


def _lane_edge_id(lane_name: str) -> str:
    raw_name = lane_name[5:] if lane_name.startswith("lane_") else lane_name
    return raw_name.rsplit("_", 1)[0] if "_" in raw_name else raw_name


def pick_destination_outgoing_lane(
    spawn_lane_index: str,
    outgoing_lanes: List[dict],
    road_network,
) -> Optional[str]:
    """Pick a reachable outgoing lane as the navigation destination."""
    if not outgoing_lanes:
        return None

    outgoing_names = {lane["lane_name"] for lane in outgoing_lanes}
    if spawn_lane_index not in road_network.graph:
        return outgoing_lanes[0]["lane_name"]

    visited = set()
    queue = [spawn_lane_index]
    while queue:
        lane_name = queue.pop(0)
        if lane_name in visited:
            continue
        visited.add(lane_name)
        if lane_name in outgoing_names and lane_name != spawn_lane_index:
            return lane_name

        lane_info = road_network.graph.get(lane_name)
        if lane_info is None:
            continue
        for next_lane in getattr(lane_info, "exit_lanes", None) or []:
            if next_lane not in visited:
                queue.append(next_lane)

    spawn_info = road_network.graph.get(spawn_lane_index)
    if spawn_info is not None:
        for next_lane in getattr(spawn_info, "exit_lanes", None) or []:
            if next_lane in outgoing_names:
                return next_lane

    return outgoing_lanes[0]["lane_name"]


class AuxiliaryAgentsManager(BaseManager):
    """Manager that spawns NPC vehicles on incoming lanes near the junction."""

    def __init__(
        self,
        spawn_lane_indices: List[str],
        outgoing_lanes: Optional[List[dict]] = None,
        distance_from_intersection: float = DEFAULT_DISTANCE_FROM_INTERSECTION,
        policy: AuxPolicyType = "idm",
        spawn_velocity_ms: float = DEFAULT_SPAWN_VELOCITY_MS,
        destination_lanes: Optional[List[str]] = None,
        ego_vehicle=None,
        ego_spawn_lane_index: Optional[str] = None,
        ego_release_distance_before_end: float = DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END,
    ):
        super().__init__()
        self._spawn_lane_indices = list(spawn_lane_indices)
        self._outgoing_lanes = list(outgoing_lanes or [])
        self._distance_from_intersection = distance_from_intersection
        self._policy = policy
        self._spawn_velocity_ms = spawn_velocity_ms
        self._destination_lanes = list(destination_lanes or [])
        self._ego_vehicle = ego_vehicle
        self._ego_spawn_lane_index = ego_spawn_lane_index
        self._ego_release_distance_before_end = float(ego_release_distance_before_end)
        self._aux_vehicles: List[BaseVehicle] = []
        self._spawn_destinations: List[Optional[str]] = []
        self._aux_policies: List[BasePolicy] = []

    def reset(self):
        self._aux_vehicles = []
        self._spawn_destinations = []
        self._aux_policies = []

    def after_reset(self):
        self._spawn_auxiliary_vehicles()

    def _spawn_auxiliary_vehicles(self):
        from metadrive.component.vehicle.vehicle_type import DefaultVehicle

        road_network = self.engine.current_map.road_network
        self._aux_vehicles = []
        self._spawn_destinations = []

        for idx, spawn_lane_index in enumerate(self._spawn_lane_indices):
            lane = road_network.get_lane(spawn_lane_index)
            spawn_long = max(1.0, lane.length - self._distance_from_intersection)
            edge_id = _lane_edge_id(spawn_lane_index)

            if idx < len(self._destination_lanes) and self._destination_lanes[idx]:
                destination_lane = self._destination_lanes[idx]
            else:
                destination_lane = pick_destination_outgoing_lane(
                    spawn_lane_index, self._outgoing_lanes, road_network
                )

            vehicle_config = {
                "spawn_lane_index": edge_id,
                "spawn_longitude": spawn_long,
                "spawn_lateral": 0.0,
                "enable_reverse": False,
                "navigation_module": EdgeNetworkNavigation,
                "destination": destination_lane,
                "show_navi_mark": False,
                "show_dest_mark": False,
                "show_line_to_dest": False,
            }

            try:
                aux_vehicle = self.spawn_object(
                    DefaultVehicle,
                    vehicle_config=vehicle_config,
                )

                correct_pos = lane.position(spawn_long, 0.0)
                correct_heading = lane.heading_theta_at(spawn_long)
                aux_vehicle.set_position([float(correct_pos[0]), float(correct_pos[1])])
                aux_vehicle.set_heading_theta(correct_heading)

                if destination_lane and aux_vehicle.navigation is not None:
                    aux_vehicle.navigation.set_route(spawn_lane_index, destination_lane)

                if self._policy == "idm":
                    use_gated = (
                        self._ego_vehicle is not None
                        and self._ego_spawn_lane_index
                        and self._ego_release_distance_before_end > 0
                    )
                    if use_gated:
                        aux_vehicle.set_velocity([0.0, 0.0], in_local_frame=True)
                        self.add_policy(
                            aux_vehicle.id,
                            GatedAuxiliaryIDMPolicy,
                            aux_vehicle,
                            self.generate_seed(),
                            ego_vehicle=self._ego_vehicle,
                            ego_spawn_lane_index=self._ego_spawn_lane_index,
                            release_distance_before_end=self._ego_release_distance_before_end,
                            release_speed_ms=self._spawn_velocity_ms,
                        )
                    else:
                        aux_vehicle.set_velocity(
                            [self._spawn_velocity_ms, 0.0], in_local_frame=True
                        )
                        self.add_policy(
                            aux_vehicle.id,
                            AuxiliaryIDMPolicy,
                            aux_vehicle,
                            self.generate_seed(),
                        )
                    self._aux_policies.append(self.get_policy(aux_vehicle.id))
                else:
                    aux_vehicle.set_velocity([0.0, 0.0], in_local_frame=True)
                    self.add_policy(
                        aux_vehicle.id,
                        StationaryPolicy,
                        aux_vehicle,
                        self.generate_seed(),
                    )
                    self._aux_policies.append(self.get_policy(aux_vehicle.id))

                self._aux_vehicles.append(aux_vehicle)
                self._spawn_destinations.append(destination_lane)
                logging.info(
                    f"[AuxAgent] Spawned on {spawn_lane_index} at {spawn_long:.1f}m "
                    f"(lane_length={lane.length:.1f}m, policy={self._policy}, "
                    f"destination={destination_lane})"
                )
                print(
                    f"[AuxAgent] Spawned on {spawn_lane_index} -> {destination_lane} "
                    f"({self._policy})"
                )
            except Exception as e:
                logging.warning(f"[AuxAgent] Failed to spawn on {spawn_lane_index}: {e}")
                print(f"[AuxAgent] Failed to spawn on {spawn_lane_index}: {e}")

    def before_step(self):
        if not self._aux_vehicles:
            return {}
        for aux_vehicle in self._aux_vehicles:
            try:
                policy = self.engine.get_policy(aux_vehicle.name)
                if (
                    isinstance(policy, GatedAuxiliaryIDMPolicy)
                    and not policy.released
                ):
                    aux_vehicle.set_velocity([0.0, 0.0], in_local_frame=True)
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
        for aux_vehicle, lane_index, destination, policy in zip(
            self._aux_vehicles,
            self._spawn_lane_indices,
            self._spawn_destinations,
            self._aux_policies,
        ):
            try:
                agent_status = {
                    "spawn_lane": lane_index,
                    "destination_lane": destination,
                    "position": list(aux_vehicle.position),
                    "speed_mps": float(aux_vehicle.speed) if hasattr(aux_vehicle, "speed") else 0.0,
                    "policy": self._policy,
                }
                if isinstance(policy, GatedAuxiliaryIDMPolicy):
                    agent_status["released"] = policy.released
                    agent_status["ego_dist_to_spawn_lane_end_m"] = (
                        policy.ego_distance_to_spawn_lane_end()
                    )
                agents.append(agent_status)
            except Exception:
                agents.append({
                    "spawn_lane": lane_index,
                    "destination_lane": destination,
                    "error": "status unavailable",
                })

        return {
            "exists": True,
            "count": len(self._aux_vehicles),
            "policy": self._policy,
            "agents": agents,
        }


def add_auxiliary_agents(
    env,
    spawn_lane_indices: List[str],
    outgoing_lanes: Optional[List[dict]] = None,
    distance_from_intersection: float = DEFAULT_DISTANCE_FROM_INTERSECTION,
    policy: AuxPolicyType = "idm",
    spawn_velocity_ms: float = DEFAULT_SPAWN_VELOCITY_MS,
    destination_lanes: Optional[List[str]] = None,
    ego_vehicle=None,
    ego_spawn_lane_index: Optional[str] = None,
    ego_release_distance_before_end: float = DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END,
) -> Optional[AuxiliaryAgentsManager]:
    """Add auxiliary agents on incoming lanes."""
    if not spawn_lane_indices:
        return None

    if not hasattr(env, "engine") or env.engine is None:
        logging.error("[AuxAgent] Environment has no engine")
        return None

    manager = AuxiliaryAgentsManager(
        spawn_lane_indices=spawn_lane_indices,
        outgoing_lanes=outgoing_lanes,
        distance_from_intersection=distance_from_intersection,
        policy=policy,
        spawn_velocity_ms=spawn_velocity_ms,
        destination_lanes=destination_lanes,
        ego_vehicle=ego_vehicle,
        ego_spawn_lane_index=ego_spawn_lane_index,
        ego_release_distance_before_end=ego_release_distance_before_end,
    )
    env.engine.register_manager("auxiliary_agent_manager", manager)
    manager.after_reset()
    return manager


def add_auxiliary_agent(
    env,
    spawn_lane_index: str,
    outgoing_lanes: Optional[List[dict]] = None,
    distance_from_intersection: float = DEFAULT_DISTANCE_FROM_INTERSECTION,
    policy: AuxPolicyType = "idm",
    spawn_velocity_ms: float = DEFAULT_SPAWN_VELOCITY_MS,
    destination_lane: Optional[str] = None,
    ego_vehicle=None,
    ego_spawn_lane_index: Optional[str] = None,
    ego_release_distance_before_end: float = DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END,
) -> Optional[AuxiliaryAgentsManager]:
    """Add a single auxiliary agent (backward compatibility)."""
    destination_lanes = [destination_lane] if destination_lane else None
    return add_auxiliary_agents(
        env,
        spawn_lane_indices=[spawn_lane_index],
        outgoing_lanes=outgoing_lanes,
        distance_from_intersection=distance_from_intersection,
        policy=policy,
        spawn_velocity_ms=spawn_velocity_ms,
        destination_lanes=destination_lanes,
        ego_vehicle=ego_vehicle,
        ego_spawn_lane_index=ego_spawn_lane_index,
        ego_release_distance_before_end=ego_release_distance_before_end,
    )
