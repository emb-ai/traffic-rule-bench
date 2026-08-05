"""Auxiliary agents for priority-junction benches (equal-priority + yield)."""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

from metadrive.manager.base_manager import BaseManager
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.component.navigation_module.edge_network_navigation import EdgeNetworkNavigation
from metadrive.utils.math import wrap_to_pi


DEFAULT_DISTANCE_FROM_INTERSECTION = 20.0
DEFAULT_SPAWN_VELOCITY_MS = 5.0
# Must be >= typical ego spawn_distance_before_end so gated aux starts when ego
# is already near the junction (avoids yield-vs-gate deadlock).
DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END = 15.0
DEFAULT_CONVOY_SIZE = 3
DEFAULT_CONVOY_GAP_M = 10.0
MIN_SPAWN_LONGITUDE_M = 3.0
# Don't despawn for arrive_destination checks until aux has been driving a bit.
ARRIVE_GRACE_STEPS = 10
AuxPolicyType = Literal["idm", "stationary"]


def min_aux_spawn_lane_length(
    aux_distance_from_intersection: float,
    convoy_size: int = 1,
    convoy_gap_m: float = DEFAULT_CONVOY_GAP_M,
) -> float:
    """Minimum incoming lane length to place a full aux convoy.

    Lead spawns at ``length - aux_distance``; slot ``i`` at
    ``lead - i * gap``. The last slot must stay ``>= MIN_SPAWN_LONGITUDE_M``, so::

        length >= aux_distance + (convoy_size - 1) * gap + MIN_SPAWN_LONGITUDE_M
    """
    n = max(1, int(convoy_size))
    gap = max(0.0, float(convoy_gap_m))
    return (
        float(aux_distance_from_intersection)
        + float(n - 1) * gap
        + MIN_SPAWN_LONGITUDE_M
    )


def is_viable_aux_lane_length(
    lane_length: float,
    aux_distance_from_intersection: float,
    convoy_size: int = 1,
    convoy_gap_m: float = DEFAULT_CONVOY_GAP_M,
) -> bool:
    return float(lane_length) >= min_aux_spawn_lane_length(
        aux_distance_from_intersection,
        convoy_size=convoy_size,
        convoy_gap_m=convoy_gap_m,
    )


def max_convoy_size_for_lane_length(
    lane_length: float,
    aux_distance_from_intersection: float,
    convoy_gap_m: float = DEFAULT_CONVOY_GAP_M,
    convoy_size_cap: int = DEFAULT_CONVOY_SIZE,
) -> int:
    """Largest convoy that fully fits on a lane of the given length (0 if none)."""
    gap = max(0.0, float(convoy_gap_m))
    cap = max(1, int(convoy_size_cap))
    best = 0
    for n in range(1, cap + 1):
        if is_viable_aux_lane_length(
            lane_length, aux_distance_from_intersection, n, gap
        ):
            best = n
        else:
            break
    return best


def apply_aux_cruise_speed(aux_policy, speed_ms: float) -> None:
    """Set auxiliary IDM cruise target to a fixed speed (m/s)."""
    if aux_policy is None:
        return
    speed_kmh = float(speed_ms) * 3.6
    if hasattr(aux_policy, "NORMAL_SPEED"):
        aux_policy.NORMAL_SPEED = speed_kmh
    if hasattr(aux_policy, "MAX_SPEED"):
        aux_policy.MAX_SPEED = max(speed_kmh, getattr(aux_policy, "MAX_SPEED", speed_kmh))
    if hasattr(aux_policy, "target_speed"):
        aux_policy.target_speed = speed_kmh


class StationaryPolicy(BasePolicy):
    """Policy that keeps the vehicle completely stationary at 0 m/s."""

    def __init__(self, control_object, random_seed: int):
        super().__init__(control_object=control_object, random_seed=random_seed)

    def act(self, *args, **kwargs):
        return [0.0, -0.5]


class AuxiliaryIDMPolicy(IDMPolicy):
    """IDM that sticks tightly to the routed lane centerline (incl. turns)."""

    # Look-ahead along the lane for heading (meters); longer helps on sharp turns.
    HEADING_LOOKAHEAD_M = 4.0

    def __init__(self, control_object, random_seed: int):
        super().__init__(control_object=control_object, random_seed=random_seed)
        self.enable_lane_change = False
        self.enable_idm_overtake = False
        # Stronger than stock IDM so aux stays on the reference line through
        # junction connectors instead of cutting across / skipping the turn.
        self.heading_pid = PIDController(2.8, 0.01, 4.5)
        self.lateral_pid = PIDController(1.0, 0.002, 0.25)

    def steering_control(self, target_lane) -> float:
        if target_lane is None:
            return 0.0
        ego_vehicle = self.control_object
        long, lat = target_lane.local_coordinates(ego_vehicle.position)
        lookahead = min(
            self.HEADING_LOOKAHEAD_M,
            max(1.0, float(getattr(target_lane, "length", self.HEADING_LOOKAHEAD_M)) - long),
        )
        lane_heading = target_lane.heading_theta_at(long + lookahead)
        v_heading = ego_vehicle.heading_theta
        steering = self.heading_pid.get_result(-wrap_to_pi(lane_heading - v_heading))
        steering += self.lateral_pid.get_result(-lat)
        return float(steering)

    def move_to_next_road(self):
        """Advance along navigation checkpoints; do not snap off the route."""
        navigation = getattr(self.control_object, "navigation", None)
        current_lanes = getattr(navigation, "current_ref_lanes", None) if navigation else None
        if not current_lanes:
            return super().move_to_next_road()

        # Prefer the exact checkpoint lane when it is among the current ref set.
        checkpoint_lane = None
        ckpt_idx = getattr(navigation, "current_checkpoint_lane_index", None)
        if ckpt_idx is not None:
            try:
                checkpoint_lane = navigation.map.road_network.get_lane(ckpt_idx)
            except Exception:
                checkpoint_lane = None
        if checkpoint_lane is not None and checkpoint_lane in current_lanes:
            self.routing_target_lane = checkpoint_lane
            return True

        if self.routing_target_lane is None:
            self.routing_target_lane = current_lanes[0]
            return True

        if self.routing_target_lane in current_lanes:
            return True

        # Only step forward onto a successor that is still on the planned route.
        checkpoints = list(getattr(navigation, "checkpoints", None) or [])
        for lane in current_lanes:
            if self.routing_target_lane.is_previous_lane_of(lane):
                if not checkpoints or lane.index in checkpoints:
                    self.routing_target_lane = lane
                    return True
            if checkpoints and lane.index in checkpoints:
                self.routing_target_lane = lane
                return True
        return False


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


from .lane_keys import lane_edge_id, lane_num_from_key, make_lane_key, parse_lane_key


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
        convoy_size: int = DEFAULT_CONVOY_SIZE,
        convoy_gap_m: float = DEFAULT_CONVOY_GAP_M,
        alternate_spawn_dest_map: Optional[dict] = None,
    ):
        super().__init__()
        self._requested_spawn_lane_indices = list(spawn_lane_indices)
        self._outgoing_lanes = list(outgoing_lanes or [])
        self._distance_from_intersection = distance_from_intersection
        self._policy = policy
        self._spawn_velocity_ms = float(
            spawn_velocity_ms if spawn_velocity_ms is not None else DEFAULT_SPAWN_VELOCITY_MS
        )
        self._destination_lanes = list(destination_lanes or [])
        self._alternate_spawn_dest_map = dict(alternate_spawn_dest_map or {})
        self._ego_vehicle = ego_vehicle
        self._ego_spawn_lane_index = ego_spawn_lane_index
        self._ego_release_distance_before_end = float(ego_release_distance_before_end)
        self._convoy_size = max(1, int(convoy_size))
        self._convoy_gap_m = max(1.0, float(convoy_gap_m))
        self._aux_vehicles: List[BaseVehicle] = []
        self._spawn_lane_indices: List[str] = []
        self._spawn_destinations: List[Optional[str]] = []
        self._convoy_positions: List[int] = []
        self._aux_policies: List[BasePolicy] = []

    def reset(self):
        self._aux_vehicles = []
        self._spawn_lane_indices = []
        self._spawn_destinations = []
        self._convoy_positions = []
        self._aux_policies = []

    def after_reset(self):
        self._spawn_auxiliary_vehicles()

    def _spawn_vehicle_on_lane(
        self,
        spawn_lane_index: str,
        spawn_long: float,
        destination_lane: Optional[str],
        convoy_position: int,
    ) -> bool:
        from metadrive.component.vehicle.vehicle_type import DefaultVehicle

        road_network = self.engine.current_map.road_network
        lane = road_network.get_lane(spawn_lane_index)
        edge_id, lane_num = parse_lane_key(spawn_lane_index)

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
            if aux_vehicle.navigation is not None:
                aux_vehicle.reset_navigation(lane)

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
                aux_policy = self.get_policy(aux_vehicle.id)
                apply_aux_cruise_speed(aux_policy, self._spawn_velocity_ms)
                if isinstance(aux_policy, GatedAuxiliaryIDMPolicy):
                    aux_policy._release_speed_ms = self._spawn_velocity_ms
                self._aux_policies.append(aux_policy)
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
            self._spawn_lane_indices.append(spawn_lane_index)
            self._spawn_destinations.append(destination_lane)
            self._convoy_positions.append(convoy_position)
            logging.info(
                f"[AuxAgent] Spawned convoy slot {convoy_position + 1}/{self._convoy_size} "
                f"on {spawn_lane_index} at {spawn_long:.1f}m "
                f"(lane_length={lane.length:.1f}m, policy={self._policy}, "
                f"destination={destination_lane})"
            )
            return True
        except Exception as e:
            logging.warning(
                f"[AuxAgent] Failed to spawn convoy slot {convoy_position + 1} "
                f"on {spawn_lane_index}: {e}"
            )
            print(
                f"[AuxAgent] Failed to spawn convoy slot {convoy_position + 1} "
                f"on {spawn_lane_index}: {e}"
            )
            return False

    def _spawn_auxiliary_vehicles(self):
        road_network = self.engine.current_map.road_network
        self._aux_vehicles = []
        self._spawn_lane_indices = []
        self._spawn_destinations = []
        self._convoy_positions = []
        self._aux_policies = []

        for idx, spawn_lane_index in enumerate(self._requested_spawn_lane_indices):
            candidate_lanes = [spawn_lane_index]
            for alt_lane in self._alternate_spawn_dest_map:
                if alt_lane not in candidate_lanes:
                    candidate_lanes.append(alt_lane)

            spawned_on_lane = 0
            used_lane = None
            used_destination = None
            for candidate_lane in candidate_lanes:
                if candidate_lane not in road_network.graph:
                    if candidate_lane == spawn_lane_index:
                        logging.warning(
                            f"[AuxAgent] Lane {candidate_lane} not found in road network; skipping"
                        )
                    continue
                lane = road_network.get_lane(candidate_lane)
                lead_spawn_long = lane.length - self._distance_from_intersection
                if lead_spawn_long < MIN_SPAWN_LONGITUDE_M:
                    if candidate_lane == spawn_lane_index:
                        logging.warning(
                            f"[AuxAgent] Lane {candidate_lane} too short for convoy "
                            f"(lead at {lead_spawn_long:.1f}m, sim length={lane.length:.1f}m)"
                        )
                    continue

                # Manifest / layout destination wins for the requested spawn lane.
                # Alternate-map entries are straight-through fallbacks for other arms.
                if (
                    candidate_lane == spawn_lane_index
                    and idx < len(self._destination_lanes)
                    and self._destination_lanes[idx]
                ):
                    destination_lane = self._destination_lanes[idx]
                elif candidate_lane in self._alternate_spawn_dest_map:
                    destination_lane = self._alternate_spawn_dest_map[candidate_lane]
                else:
                    destination_lane = pick_destination_outgoing_lane(
                        candidate_lane, self._outgoing_lanes, road_network
                    )

                spawned_on_lane = 0
                for convoy_idx in range(self._convoy_size):
                    spawn_long = lead_spawn_long - convoy_idx * self._convoy_gap_m
                    if spawn_long < MIN_SPAWN_LONGITUDE_M:
                        break
                    if self._spawn_vehicle_on_lane(
                        candidate_lane,
                        spawn_long,
                        destination_lane,
                        convoy_idx,
                    ):
                        spawned_on_lane += 1

                if spawned_on_lane:
                    used_lane = candidate_lane
                    used_destination = destination_lane
                    break

            if spawned_on_lane and used_lane:
                print(
                    f"[AuxAgent] Convoy x{spawned_on_lane} on {used_lane} "
                    f"-> {used_destination} ({self._policy}, gap={self._convoy_gap_m:.1f}m)"
                )

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

    def _should_despawn(self, aux_vehicle) -> tuple[bool, str]:
        """Return (True, reason) when the aux agent has left the road or arrived."""
        if getattr(aux_vehicle, "on_lane", True) is False:
            return True, "off_lane"
        if bool(getattr(aux_vehicle, "out_of_route", False)):
            return True, "out_of_route"
        if bool(getattr(aux_vehicle, "crash_sidewalk", False)):
            return True, "crash_sidewalk"

        age = int(getattr(self.engine, "episode_step", 0) or 0)
        if age <= ARRIVE_GRACE_STEPS:
            return False, ""

        try:
            policy = self.engine.get_policy(aux_vehicle.name)
            if bool(getattr(policy, "arrive_destination", False)):
                return True, "arrived"
        except Exception:
            pass

        navigation = getattr(aux_vehicle, "navigation", None)
        final_lane = getattr(navigation, "final_lane", None) if navigation is not None else None
        if final_lane is not None:
            try:
                long, lat = final_lane.local_coordinates(aux_vehicle.position)
                lane_w = float(getattr(final_lane, "width", 3.5) or 3.5)
                near_end = (final_lane.length - 5.0) < long < (final_lane.length + 5.0)
                on_lane_lat = abs(lat) <= (lane_w / 2.0 + 1.0)
                if near_end and on_lane_lat:
                    return True, "arrived_final_lane"
            except Exception:
                pass

        return False, ""

    def _remove_aux_at(self, idx: int, reason: str) -> None:
        aux_vehicle = self._aux_vehicles[idx]
        lane = (
            self._spawn_lane_indices[idx]
            if idx < len(self._spawn_lane_indices)
            else "?"
        )
        try:
            self.clear_objects([aux_vehicle.id])
        except Exception as exc:
            logging.debug(f"[AuxAgent] clear_objects failed for {aux_vehicle.id}: {exc}")
        for seq in (
            self._aux_vehicles,
            self._spawn_lane_indices,
            self._spawn_destinations,
            self._convoy_positions,
            self._aux_policies,
        ):
            if idx < len(seq):
                seq.pop(idx)
        print(f"[AuxAgent] Despawned {lane} ({reason})")

    def after_step(self, *args, **kwargs):
        if not self._aux_vehicles:
            return {}

        to_remove: list[tuple[int, str]] = []
        for idx, aux_vehicle in enumerate(self._aux_vehicles):
            try:
                aux_vehicle.after_step()
            except Exception:
                to_remove.append((idx, "after_step_error"))
                continue
            should, reason = self._should_despawn(aux_vehicle)
            if should:
                to_remove.append((idx, reason))

        for idx, reason in reversed(to_remove):
            self._remove_aux_at(idx, reason)
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
        for aux_vehicle, lane_index, destination, policy, convoy_pos in zip(
            self._aux_vehicles,
            self._spawn_lane_indices,
            self._spawn_destinations,
            self._aux_policies,
            self._convoy_positions,
        ):
            try:
                agent_status = {
                    "spawn_lane": lane_index,
                    "destination_lane": destination,
                    "convoy_position": convoy_pos,
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
            "convoy_size": self._convoy_size,
            "convoy_gap_m": self._convoy_gap_m,
            "lanes_occupied": len(set(self._spawn_lane_indices)),
            "policy": self._policy,
            "agents": agents,
        }


def main_lane_keys_for_aux(
    junction_layout: Optional[dict],
    ego_edge_id: Optional[str] = None,
    main_lane_keys: Optional[List[str]] = None,
) -> List[str]:
    """Main-road lane keys for aux spawning, excluding ego's approach arm."""
    if main_lane_keys:
        if not ego_edge_id:
            return sorted(main_lane_keys)
        return sorted(
            k for k in main_lane_keys if lane_edge_id(k) != ego_edge_id
        )
    if not junction_layout:
        return []
    keys: List[str] = []
    for arm in junction_layout.get("arms", []):
        if arm.get("road_class") != "main":
            continue
        if ego_edge_id and arm.get("edge_id") == ego_edge_id:
            continue
        keys.extend(arm.get("lane_keys", []))
    return sorted(keys)


def select_occupied_main_lanes(
    all_main_lane_keys: List[str],
    n_lanes_occupied: int,
    prefer_lane_key: Optional[str] = None,
) -> List[str]:
    if not all_main_lane_keys:
        return []
    n = max(1, min(int(n_lanes_occupied), len(all_main_lane_keys)))
    ordered = sorted(all_main_lane_keys)
    if prefer_lane_key and prefer_lane_key in ordered:
        ordered.remove(prefer_lane_key)
        ordered.insert(0, prefer_lane_key)
    return ordered[:n]


from .junction_priority_layout import right_arm_edge_id


def right_lane_keys_for_aux(
    junction_layout: Optional[dict],
    ego_edge_id: Optional[str] = None,
) -> List[str]:
    """Lane keys on the incoming arm to ego's right (right-hand conflict)."""
    if not junction_layout or not ego_edge_id:
        return []
    right_edge = right_arm_edge_id(junction_layout, ego_edge_id)
    if not right_edge:
        return []
    for arm in junction_layout.get("arms", []):
        if arm.get("edge_id") == right_edge:
            return sorted(arm.get("lane_keys", []))
    return []


def viable_right_aux_lane_keys(
    junction_layout: Optional[dict],
    aux_distance_from_intersection: float,
    ego_edge_id: Optional[str] = None,
    convoy_size: int = 1,
    convoy_gap_m: float = DEFAULT_CONVOY_GAP_M,
) -> List[str]:
    """Right-arm lane keys with enough length for a full aux convoy."""
    if not junction_layout or not ego_edge_id:
        return []
    right_edge = right_arm_edge_id(junction_layout, ego_edge_id)
    if not right_edge:
        return []
    min_required = min_aux_spawn_lane_length(
        aux_distance_from_intersection,
        convoy_size=convoy_size,
        convoy_gap_m=convoy_gap_m,
    )
    for arm in junction_layout.get("arms", []):
        if arm.get("edge_id") != right_edge:
            continue
        min_len = float(arm.get("min_lane_length") or 0.0)
        if min_len < min_required:
            return []
        return sorted(arm.get("lane_keys", []))
    return []


def has_viable_right_aux_lanes(
    junction_layout: Optional[dict],
    aux_distance_from_intersection: float,
    ego_edge_id: Optional[str] = None,
    convoy_size: int = 1,
    convoy_gap_m: float = DEFAULT_CONVOY_GAP_M,
) -> bool:
    return bool(
        viable_right_aux_lane_keys(
            junction_layout,
            aux_distance_from_intersection,
            ego_edge_id,
            convoy_size=convoy_size,
            convoy_gap_m=convoy_gap_m,
        )
    )


def viable_aux_arms(
    junction_layout: Optional[dict],
    aux_distance_from_intersection: float,
    ego_edge_id: Optional[str] = None,
    convoy_size: int = 1,
    convoy_gap_m: float = DEFAULT_CONVOY_GAP_M,
) -> List[dict]:
    """Return main-road arms with lanes long enough for a full aux convoy.

    A lane is viable if
    ``min_lane_length >= aux_distance + (convoy_size-1)*gap + MIN_SPAWN_LONGITUDE_M``.
    """
    if not junction_layout:
        return []
    min_required = min_aux_spawn_lane_length(
        aux_distance_from_intersection,
        convoy_size=convoy_size,
        convoy_gap_m=convoy_gap_m,
    )
    viable: List[dict] = []
    for arm in junction_layout.get("arms", []):
        if arm.get("road_class") != "main":
            continue
        if ego_edge_id and arm.get("edge_id") == ego_edge_id:
            continue
        min_len = arm.get("min_lane_length", 0.0)
        if min_len >= min_required:
            viable.append(arm)
    return viable


def viable_aux_lane_keys(
    junction_layout: Optional[dict],
    aux_distance_from_intersection: float,
    ego_edge_id: Optional[str] = None,
    convoy_size: int = 1,
    convoy_gap_m: float = DEFAULT_CONVOY_GAP_M,
) -> List[str]:
    """Lane keys on main-road arms with enough length for a full aux convoy."""
    keys: List[str] = []
    for arm in viable_aux_arms(
        junction_layout,
        aux_distance_from_intersection,
        ego_edge_id,
        convoy_size=convoy_size,
        convoy_gap_m=convoy_gap_m,
    ):
        keys.extend(arm.get("lane_keys", []))
    return sorted(keys)


def has_viable_aux_lanes(
    junction_layout: Optional[dict],
    aux_distance_from_intersection: float,
) -> bool:
    """Check if a junction layout has any main-road arm with sufficient lane length.
    
    Returns True if at least one main-road arm (across all possible ego edges)
    has lanes long enough for aux spawning.
    """
    if not junction_layout:
        return False
    min_required = min_aux_spawn_lane_length(aux_distance_from_intersection)
    for arm in junction_layout.get("arms", []):
        if arm.get("road_class") != "main":
            continue
        min_len = arm.get("min_lane_length", 0.0)
        if min_len >= min_required:
            return True
    return False


def resolve_aux_spawn_lanes(
    row: dict,
    ego_lane_index: str,
    incoming_lanes: Optional[List[dict]] = None,
    aux_lanes_occupied: int = 1,
    aux_distance_from_intersection: float = DEFAULT_DISTANCE_FROM_INTERSECTION,
) -> List[str]:
    """Resolve which lane indices should carry auxiliary convoys for this episode."""
    lanes_n = int(row.get("aux_lanes_occupied", aux_lanes_occupied) or aux_lanes_occupied)
    aux_distance = float(
        row.get("aux_distance_from_intersection", aux_distance_from_intersection)
    )
    convoy_size = int(row.get("aux_convoy_size", 1) or 1)
    convoy_gap_m = float(row.get("aux_convoy_gap_m", DEFAULT_CONVOY_GAP_M) or DEFAULT_CONVOY_GAP_M)

    ego_edge = lane_edge_id(str(ego_lane_index)) if ego_lane_index else None
    if row.get("road_id"):
        ego_edge = str(row["road_id"])

    junction_layout = row.get("junction_layout")
    if junction_layout and junction_layout.get("mode") == "main_main":
        right_keys = viable_right_aux_lane_keys(
            junction_layout,
            aux_distance,
            ego_edge,
            convoy_size=convoy_size,
            convoy_gap_m=convoy_gap_m,
        )
        if not right_keys:
            right_keys = row.get("right_lane_keys") or right_lane_keys_for_aux(
                junction_layout, ego_edge
            )
        occupied = row.get("aux_occupied_lane_keys")
        if occupied:
            filtered = [key for key in occupied if key in right_keys] if right_keys else list(occupied)
            if filtered:
                return filtered[:lanes_n]
        if right_keys:
            return select_occupied_main_lanes(right_keys, lanes_n)
        return []

    viable_keys = viable_aux_lane_keys(
        junction_layout,
        aux_distance,
        ego_edge,
        convoy_size=convoy_size,
        convoy_gap_m=convoy_gap_m,
    )
    viable_set = set(viable_keys)

    def _filter_viable(keys: List[str]) -> List[str]:
        if not viable_set:
            return keys
        return [key for key in keys if key in viable_set]

    scenario_lane = row.get("aux_spawn_lane_index")
    if scenario_lane and lanes_n == 1:
        lane = str(scenario_lane)
        if not viable_set or lane in viable_set:
            return [lane]
        if viable_keys:
            return viable_keys[:1]
        return []

    occupied = row.get("aux_occupied_lane_keys")
    if occupied:
        filtered = _filter_viable(list(occupied))
        if filtered:
            return filtered[:lanes_n]

    main_keys = viable_keys or main_lane_keys_for_aux(
        junction_layout,
        ego_edge_id=ego_edge,
        main_lane_keys=row.get("main_lane_keys"),
    )
    lanes_n = int(row.get("aux_lanes_occupied", aux_lanes_occupied) or aux_lanes_occupied)
    if main_keys:
        return select_occupied_main_lanes(main_keys, lanes_n)

    # Legacy single-lane fallback
    spawn_lane = row.get("aux_spawn_lane_index")
    if not spawn_lane and row.get("aux_road_id") is not None:
        aux_lane_num = int(row.get("aux_spawn_lane_num", 0) or 0)
        spawn_lane = make_lane_key(str(row["aux_road_id"]), aux_lane_num)
    if spawn_lane is not None:
        spawn_lane = str(spawn_lane)
        if not viable_set or spawn_lane in viable_set:
            return [spawn_lane]
        if viable_keys:
            return viable_keys[:1]
        return []
    if incoming_lanes:
        for lane in incoming_lanes:
            if lane["edge_id"] not in str(ego_lane_index):
                candidate = lane["lane_name"]
                if not viable_set or candidate in viable_set:
                    return [candidate]
        if viable_keys:
            return viable_keys[:1]
    return []


def resolve_aux_destination_lane_key(
    junction_layout: Optional[dict],
    spawn_lane_key: str,
) -> Optional[str]:
    """Straight-through destination lane key (fallback when no turn is specified)."""
    if not junction_layout:
        return None

    edge_id = lane_edge_id(spawn_lane_key)
    lane_num = lane_num_from_key(spawn_lane_key)
    arm = None
    for candidate in junction_layout.get("arms", []):
        if candidate.get("edge_id") == edge_id:
            arm = candidate
            break
    if arm is None:
        return None

    straight_to = [
        edge
        for edge in arm.get("straight_to", [])
        if edge and not str(edge).startswith(":")
    ]
    if not straight_to:
        return None

    dest_edge = straight_to[0]
    for candidate in junction_layout.get("arms", []):
        if candidate.get("edge_id") != dest_edge:
            continue
        keys = candidate.get("lane_keys", [])
        for key in keys:
            if lane_num_from_key(key) == lane_num:
                return key
        if keys:
            return keys[min(lane_num, len(keys) - 1)]
    return make_lane_key(dest_edge, lane_num)


def resolve_aux_destination_lane_key_for_edge(
    junction_layout: Optional[dict],
    spawn_lane_key: str,
    dest_edge_id: str,
) -> Optional[str]:
    """Outgoing lane key on ``dest_edge_id`` matching the spawn lane index."""
    if not junction_layout or not dest_edge_id:
        return None
    lane_num = lane_num_from_key(spawn_lane_key)
    for candidate in junction_layout.get("arms", []):
        if candidate.get("edge_id") != dest_edge_id:
            continue
        keys = candidate.get("lane_keys", [])
        for key in keys:
            if lane_num_from_key(key) == lane_num:
                return key
        if keys:
            return keys[min(lane_num, len(keys) - 1)]
    return make_lane_key(dest_edge_id, lane_num)


def resolve_aux_spawn_plan(
    row: dict,
    ego_lane_index: str,
    incoming_lanes: Optional[List[dict]] = None,
    aux_lanes_occupied: int = 1,
    aux_distance_from_intersection: float = DEFAULT_DISTANCE_FROM_INTERSECTION,
) -> tuple[List[str], List[str], dict]:
    """Resolve aux spawn lanes, destinations, and alternate spawn->dest fallbacks."""
    spawn_lanes = resolve_aux_spawn_lanes(
        row,
        ego_lane_index=ego_lane_index,
        incoming_lanes=incoming_lanes,
        aux_lanes_occupied=aux_lanes_occupied,
        aux_distance_from_intersection=aux_distance_from_intersection,
    )
    junction_layout = row.get("junction_layout")
    ego_edge = lane_edge_id(str(ego_lane_index)) if ego_lane_index else None
    if row.get("road_id"):
        ego_edge = str(row["road_id"])

    viable_keys = viable_aux_lane_keys(
        junction_layout,
        float(row.get("aux_distance_from_intersection", aux_distance_from_intersection)),
        ego_edge,
        convoy_size=int(row.get("aux_convoy_size", 1) or 1),
        convoy_gap_m=float(
            row.get("aux_convoy_gap_m", DEFAULT_CONVOY_GAP_M) or DEFAULT_CONVOY_GAP_M
        ),
    )
    # Straight-through defaults for fallback spawn lanes only.
    alternate_spawn_dest_map: dict = {}
    for lane_key in viable_keys:
        dest = resolve_aux_destination_lane_key(junction_layout, lane_key)
        if dest:
            alternate_spawn_dest_map[lane_key] = dest

    manifest_dest = row.get("aux_destination_lane_id")
    manifest_dest_edge = row.get("aux_destination_edge_id")
    manifest_spawn = row.get("aux_spawn_lane_index")
    if manifest_dest and manifest_spawn:
        alternate_spawn_dest_map[str(manifest_spawn)] = str(manifest_dest)
    elif manifest_dest_edge and manifest_spawn:
        resolved = resolve_aux_destination_lane_key_for_edge(
            junction_layout, str(manifest_spawn), str(manifest_dest_edge)
        )
        if resolved:
            alternate_spawn_dest_map[str(manifest_spawn)] = resolved
            manifest_dest = resolved

    destination_lanes: List[str] = []
    for idx, spawn_lane in enumerate(spawn_lanes):
        dest = None
        if manifest_dest and (
            (manifest_spawn and spawn_lane == str(manifest_spawn))
            or (idx == 0 and not manifest_spawn)
        ):
            dest = str(manifest_dest)
        if not dest:
            dest = alternate_spawn_dest_map.get(spawn_lane)
        destination_lanes.append(dest or "")

    return spawn_lanes, destination_lanes, alternate_spawn_dest_map


def add_auxiliary_agents(
    env,
    spawn_lane_indices: List[str],
    outgoing_lanes: Optional[List[dict]] = None,
    distance_from_intersection: float = DEFAULT_DISTANCE_FROM_INTERSECTION,
    policy: AuxPolicyType = "idm",
    spawn_velocity_ms: Optional[float] = None,
    destination_lanes: Optional[List[str]] = None,
    ego_vehicle=None,
    ego_spawn_lane_index: Optional[str] = None,
    ego_release_distance_before_end: float = DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END,
    convoy_size: int = DEFAULT_CONVOY_SIZE,
    convoy_gap_m: float = DEFAULT_CONVOY_GAP_M,
    alternate_spawn_dest_map: Optional[dict] = None,
) -> Optional[AuxiliaryAgentsManager]:
    """Add auxiliary agents on incoming lanes (optionally as a convoy per lane)."""
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
        convoy_size=convoy_size,
        convoy_gap_m=convoy_gap_m,
        alternate_spawn_dest_map=alternate_spawn_dest_map,
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
    spawn_velocity_ms: Optional[float] = None,
    destination_lane: Optional[str] = None,
    ego_vehicle=None,
    ego_spawn_lane_index: Optional[str] = None,
    ego_release_distance_before_end: float = DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END,
    convoy_size: int = DEFAULT_CONVOY_SIZE,
    convoy_gap_m: float = DEFAULT_CONVOY_GAP_M,
) -> Optional[AuxiliaryAgentsManager]:
    """Add auxiliary agents on one lane (backward compatibility)."""
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
        convoy_size=convoy_size,
        convoy_gap_m=convoy_gap_m,
    )
