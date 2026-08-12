"""Auxiliary agents for priority-junction benches (equal-priority + yield)."""

from __future__ import annotations

import logging
from pathlib import Path
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


from .lane_keys import (
    clamp_lane_key_to_graph,
    lane_edge_id,
    lane_num_from_key,
    make_lane_key,
    parse_lane_key,
    pick_lane_key_on_edge,
)


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
        spawn_longitudinal_by_lane: Optional[dict] = None,
        ring_circulate_by_lane: Optional[dict] = None,
        junction_layout: Optional[dict] = None,
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
        self._spawn_longitudinal_by_lane = {
            str(k): float(v) for k, v in dict(spawn_longitudinal_by_lane or {}).items()
        }
        self._ring_circulate_by_lane = {
            str(k): bool(v) for k, v in dict(ring_circulate_by_lane or {}).items()
        }
        self._junction_layout = dict(junction_layout or {})
        self._ring_lane_keys: List[str] = []
        for arm in self._junction_layout.get("arms", []):
            if arm.get("road_class") != "main":
                continue
            for key in arm.get("lane_keys", []) or []:
                if key and str(key) not in self._ring_lane_keys:
                    self._ring_lane_keys.append(str(key))
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
        self._ring_circulate_flags: List[bool] = []

    def reset(self):
        self._aux_vehicles = []
        self._spawn_lane_indices = []
        self._spawn_destinations = []
        self._convoy_positions = []
        self._aux_policies = []
        self._ring_circulate_flags = []

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
                destination_lane = clamp_lane_key_to_graph(
                    destination_lane, road_network.graph
                )
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
            ring_circ = bool(self._ring_circulate_by_lane.get(str(spawn_lane_index), False))
            self._ring_circulate_flags.append(ring_circ)
            try:
                aux_vehicle._pdd_ring_circulate = ring_circ
                aux_vehicle._pdd_spawn_lane_key = str(spawn_lane_index)
            except Exception:
                pass
            if ring_circ:
                print(
                    f"[AuxAgent] Ring-circulate aux on {spawn_lane_index} "
                    f"(1-hop ring dest; will keep looping)"
                )
            logging.info(
                f"[AuxAgent] Spawned convoy slot {convoy_position + 1}/{self._convoy_size} "
                f"on {spawn_lane_index} at {spawn_long:.1f}m "
                f"(lane_length={lane.length:.1f}m, policy={self._policy}, "
                f"destination={destination_lane}, ring_circulate={ring_circ})"
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

    def _convoy_slots_for_lead_lane(
        self,
        lead_lane_key: str,
        lead_spawn_long: float,
        road_network,
    ) -> List[tuple[str, float]]:
        """Return (lane_key, longitude) per convoy slot; may spill upstream on ring."""
        n = self._convoy_size
        gap = self._convoy_gap_m
        # Default: same lane, decreasing longitude.
        same_lane: List[tuple[str, float]] = []
        ok_same = True
        for i in range(n):
            spawn_long = lead_spawn_long - i * gap
            if spawn_long < MIN_SPAWN_LONGITUDE_M:
                ok_same = False
                break
            same_lane.append((lead_lane_key, float(spawn_long)))
        if ok_same and len(same_lane) == n:
            return same_lane

        layout = self._junction_layout
        if not layout or layout.get("mode") != "roundabout":
            return same_lane  # partial same-lane fallback

        try:
            from .roundabout_aux import (
                merge_lane_lengths_from_layout,
                resolve_convoy_spawn_slots,
            )
            from .roundabout_yield_zone import entry_conflict_ring_edges
        except Exception:
            return same_lane

        edge_id = lane_edge_id(lead_lane_key)
        lane_num = lane_num_from_key(lead_lane_key)
        lengths = merge_lane_lengths_from_layout(layout, {})
        # Prefer actual MetaDrive lane length when available.
        try:
            md_lane = road_network.get_lane(lead_lane_key)
            if md_lane is not None:
                lengths[(edge_id, lane_num)] = float(md_lane.length)
        except Exception:
            pass

        ego_edge = None
        if self._ego_spawn_lane_index:
            ego_edge = lane_edge_id(str(self._ego_spawn_lane_index))
        left = (
            set(entry_conflict_ring_edges(layout, ego_edge))
            if ego_edge
            else {edge_id}
        )
        conflict_edge = edge_id if edge_id in left or not left else next(iter(left))
        slots = resolve_convoy_spawn_slots(
            layout,
            conflict_edge,
            lane_num,
            lengths,
            self._distance_from_intersection,
            n,
            gap,
            allowed_ring_edges=left if left else None,
        )
        if not slots:
            # Retry with lead edge as conflict (manifest may already be remapped).
            if conflict_edge != edge_id:
                slots = resolve_convoy_spawn_slots(
                    layout,
                    edge_id,
                    lane_num,
                    lengths,
                    self._distance_from_intersection,
                    n,
                    gap,
                    allowed_ring_edges=None,
                )
        if not slots:
            return same_lane

        # Override lead longitude with the resolved sim longitude when present.
        out: List[tuple[str, float]] = []
        for slot in slots:
            key = slot.spawn_lane_key
            long_val = float(slot.spawn_longitudinal)
            if slot.convoy_index == 0 and key == lead_lane_key:
                long_val = float(lead_spawn_long)
            # Prefer a graph lane that exists (sibling remap).
            if key not in getattr(road_network, "graph", {}):
                alt = None
                for candidate in self._ring_lane_keys:
                    if lane_edge_id(candidate) == slot.spawn_edge_id and (
                        candidate in road_network.graph
                    ):
                        alt = candidate
                        break
                if alt is None:
                    continue
                key = alt
            out.append((key, long_val))
        return out if len(out) == n else same_lane

    def _destination_for_spawn_lane(
        self,
        spawn_lane_index: str,
        *,
        lead_lane_index: str,
        lead_idx: int,
    ) -> Optional[str]:
        road_network = self.engine.current_map.road_network
        if (
            spawn_lane_index == lead_lane_index
            and lead_idx < len(self._destination_lanes)
            and self._destination_lanes[lead_idx]
        ):
            destination_lane = self._destination_lanes[lead_idx]
        elif spawn_lane_index in self._alternate_spawn_dest_map:
            destination_lane = self._alternate_spawn_dest_map[spawn_lane_index]
        else:
            destination_lane = pick_destination_outgoing_lane(
                spawn_lane_index, self._outgoing_lanes, road_network
            )
        return clamp_lane_key_to_graph(destination_lane, road_network.graph)

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
                if candidate_lane in self._spawn_longitudinal_by_lane:
                    lead_spawn_long = float(
                        self._spawn_longitudinal_by_lane[candidate_lane]
                    )
                else:
                    lead_spawn_long = lane.length - self._distance_from_intersection
                lead_spawn_long = min(lead_spawn_long, float(lane.length) - 0.1)
                if lead_spawn_long < MIN_SPAWN_LONGITUDE_M:
                    if candidate_lane == spawn_lane_index:
                        logging.warning(
                            f"[AuxAgent] Lane {candidate_lane} too short for convoy "
                            f"(lead at {lead_spawn_long:.1f}m, sim length={lane.length:.1f}m)"
                        )
                    continue

                slot_spawns = self._convoy_slots_for_lead_lane(
                    candidate_lane, lead_spawn_long, road_network
                )
                if not slot_spawns:
                    continue

                spawned_on_lane = 0
                last_dest = None
                for convoy_idx, (slot_lane, spawn_long) in enumerate(slot_spawns):
                    if slot_lane not in road_network.graph:
                        break
                    slot_lane_obj = road_network.get_lane(slot_lane)
                    spawn_long = min(float(spawn_long), float(slot_lane_obj.length) - 0.1)
                    if spawn_long < MIN_SPAWN_LONGITUDE_M:
                        break
                    destination_lane = self._destination_for_spawn_lane(
                        slot_lane,
                        lead_lane_index=candidate_lane,
                        lead_idx=idx,
                    )
                    # Spillover followers on the ring always circulate.
                    if (
                        slot_lane != candidate_lane
                        and (self._junction_layout or {}).get("mode") == "roundabout"
                    ):
                        self._ring_circulate_by_lane[slot_lane] = True
                    if self._spawn_vehicle_on_lane(
                        slot_lane,
                        spawn_long,
                        destination_lane,
                        convoy_idx,
                    ):
                        spawned_on_lane += 1
                        last_dest = destination_lane

                if spawned_on_lane:
                    used_lane = candidate_lane
                    used_destination = last_dest
                    spill = sum(
                        1 for lane_key, _ in slot_spawns if lane_key != candidate_lane
                    )
                    if spill:
                        print(
                            f"[AuxAgent] Convoy spillover: {spill} follower(s) "
                            f"on upstream ring from lead {candidate_lane}"
                        )
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
                # Only treat as arrived when physically on the final road —
                # projecting a ring position onto a nearby exit lane was a
                # false-positive that despawned mid-lane aux early.
                cur_lane = getattr(aux_vehicle, "lane", None)
                cur_idx = getattr(cur_lane, "index", None) if cur_lane is not None else None
                fin_idx = getattr(final_lane, "index", None)
                on_final = False
                if cur_idx is not None and fin_idx is not None:
                    try:
                        on_final = (cur_idx[0], cur_idx[1]) == (fin_idx[0], fin_idx[1])
                    except Exception:
                        on_final = cur_idx == fin_idx
                if on_final:
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
            self._ring_circulate_flags,
        ):
            if idx < len(seq):
                seq.pop(idx)
        print(f"[AuxAgent] Despawned {lane} ({reason})")

    def _resolve_string_lane_key(self, aux_vehicle) -> Optional[str]:
        """Map MetaDrive lane (often a tuple index) to a string lane_* key."""
        road_network = self.engine.current_map.road_network
        cur_lane = getattr(aux_vehicle, "lane", None)
        cur_idx = getattr(cur_lane, "index", None) if cur_lane is not None else None
        if isinstance(cur_idx, str) and cur_idx in getattr(road_network, "graph", {}):
            return cur_idx

        spawn_key = getattr(aux_vehicle, "_pdd_spawn_lane_key", None)
        candidates: List[str] = []
        if spawn_key:
            candidates.append(str(spawn_key))
        candidates.extend(self._ring_lane_keys)
        # Also try every main-arm lane key from the layout.
        for arm in (self._junction_layout or {}).get("arms", []):
            if arm.get("road_class") != "main":
                continue
            for key in arm.get("lane_keys", []) or []:
                candidates.append(str(key))

        pos = getattr(aux_vehicle, "position", None)
        if pos is None:
            return str(spawn_key) if spawn_key else None

        best = None
        best_score = 1e9
        seen: set[str] = set()
        for key in candidates:
            if not key or key in seen:
                continue
            seen.add(key)
            if key not in road_network.graph:
                clamped = clamp_lane_key_to_graph(key, road_network.graph)
                if not clamped or clamped not in road_network.graph:
                    continue
                key = clamped
            try:
                lane = road_network.get_lane(key)
                long, lat = lane.local_coordinates(pos)
                if long < -2.0 or long > float(lane.length) + 2.0:
                    continue
                score = abs(float(lat)) + max(0.0, -float(long)) + max(
                    0.0, float(long) - float(lane.length)
                )
                if score < best_score:
                    best_score = score
                    best = key
            except Exception:
                continue
        if best is not None:
            return best
        return str(spawn_key) if spawn_key else None

    def _pick_next_ring_dest(self, aux_vehicle) -> Optional[str]:
        """Immediate next ring-lane key (one hop ahead) for circulation."""
        if not self._ring_lane_keys and not self._junction_layout:
            return None
        road_network = self.engine.current_map.road_network
        cur_key = self._resolve_string_lane_key(aux_vehicle)
        if not cur_key:
            return None
        cur_edge = lane_edge_id(cur_key)
        cur_lane_num = lane_num_from_key(cur_key)

        main_edges = {
            str(arm.get("edge_id"))
            for arm in self._junction_layout.get("arms", [])
            if arm.get("road_class") == "main" and arm.get("edge_id")
        }
        lane_keys_by_edge = self._junction_layout.get("lane_keys_by_edge") or {
            str(arm.get("edge_id")): list(arm.get("lane_keys", []))
            for arm in self._junction_layout.get("arms", [])
            if arm.get("edge_id")
        }

        next_edges: List[str] = []
        for arm in self._junction_layout.get("arms", []):
            if str(arm.get("edge_id")) != str(cur_edge):
                continue
            for bucket in ("straight_to", "outgoing_to"):
                for edge in arm.get(bucket, []) or []:
                    eid = str(edge)
                    if eid in main_edges and eid not in next_edges:
                        next_edges.append(eid)

        candidates: List[str] = []
        for edge in next_edges:
            key = pick_lane_key_on_edge(edge, cur_lane_num, lane_keys_by_edge)
            if key and key != cur_key:
                candidates.append(key)
        if not candidates:
            for key in self._ring_lane_keys:
                if lane_edge_id(key) == cur_edge:
                    continue
                if lane_num_from_key(key) == cur_lane_num:
                    candidates.append(key)
        if not candidates:
            candidates = [
                key
                for key in self._ring_lane_keys
                if lane_edge_id(key) != cur_edge
            ]
        # Never re-route to the lane we are already on.
        candidates = [key for key in candidates if key and key != cur_key]
        if not candidates:
            return None

        pick = candidates[0]
        return clamp_lane_key_to_graph(pick, road_network.graph) or pick

    def _continue_ring_circulation(self, aux_vehicle) -> bool:
        """Re-route a ring-only aux so it keeps driving instead of despawning."""
        if not getattr(aux_vehicle, "_pdd_ring_circulate", False):
            return False
        nav = getattr(aux_vehicle, "navigation", None)
        if nav is None:
            return False
        route_from = self._resolve_string_lane_key(aux_vehicle)
        dest = self._pick_next_ring_dest(aux_vehicle)
        if not route_from or not dest or route_from == dest:
            return False
        try:
            nav.set_route(route_from, dest)
            try:
                # Keep the *current* lane as fallback; dest is only a waypoint.
                aux_vehicle._pdd_spawn_lane_key = route_from
            except Exception:
                pass
            try:
                aux_vehicle.out_of_route = False
            except Exception:
                pass
            try:
                policy = self.engine.get_policy(aux_vehicle.name)
                if policy is not None and hasattr(policy, "arrive_destination"):
                    policy.arrive_destination = False
            except Exception:
                pass
            print(
                f"[AuxAgent] Ring-circulate re-route "
                f"{getattr(aux_vehicle, 'id', '?')} {route_from} -> {dest}"
            )
            return True
        except Exception as exc:
            logging.debug("[AuxAgent] Ring re-route failed: %s", exc)
            return False

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
            if should and reason in (
                "arrived",
                "arrived_final_lane",
                "out_of_route",
            ):
                if self._continue_ring_circulation(aux_vehicle):
                    continue
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

    For roundabout layouts, only the left-hand conflict-arc ring edges relative
    to ego are considered. Arcs shorter than ``MIN_CONFLICT_ARC_LENGTH_M`` are
    rejected; longer arcs may clamp the lead offset. Convoy followers may
    spill onto upstream ring hops when the conflict arc is short.
    """
    if not junction_layout:
        return []

    if junction_layout.get("mode") == "roundabout":
        from .roundabout_aux import (
            convoy_fits_with_spillover,
            merge_lane_lengths_from_layout,
            resolve_aux_spawn_placement,
        )
        from .roundabout_yield_zone import (
            all_entry_conflict_ring_edges,
            entry_conflict_ring_edges,
        )

        lengths = merge_lane_lengths_from_layout(junction_layout, {})
        arms_by_edge = {
            str(arm.get("edge_id")): arm
            for arm in junction_layout.get("arms", [])
            if arm.get("edge_id")
        }

        if ego_edge_id:
            candidate_edges = set(
                entry_conflict_ring_edges(junction_layout, ego_edge_id)
            )
        else:
            # Scene-level gate (no ego yet): any entry conflict arc that can
            # place an aux counts as viable.
            candidate_edges = set(all_entry_conflict_ring_edges(junction_layout))

        # Lead stays on the conflict edge; convoy may spill upstream.
        spawn_edge_ids: set[str] = set()
        for conflict_edge in candidate_edges:
            conflict_arm = arms_by_edge.get(conflict_edge)
            if conflict_arm is None:
                continue
            lane_nums = sorted(
                {
                    lane_num_from_key(str(key))
                    for key in conflict_arm.get("lane_keys", [])
                }
            ) or [0]
            for lane_num in lane_nums:
                placement = resolve_aux_spawn_placement(
                    junction_layout,
                    conflict_edge,
                    lane_num,
                    lengths,
                    aux_distance_from_intersection,
                    allowed_ring_edges=candidate_edges,
                )
                if placement is None:
                    continue
                if not convoy_fits_with_spillover(
                    junction_layout,
                    conflict_edge,
                    lane_num,
                    lengths,
                    aux_distance_from_intersection,
                    convoy_size,
                    convoy_gap_m,
                    allowed_ring_edges=candidate_edges,
                ):
                    continue
                spawn_edge_ids.add(placement.spawn_edge_id)

        viable: List[dict] = []
        for edge_id in sorted(spawn_edge_ids):
            arm = arms_by_edge.get(edge_id)
            if arm is None:
                continue
            if ego_edge_id and edge_id == str(ego_edge_id):
                continue
            viable.append(arm)
        return viable

    min_required = min_aux_spawn_lane_length(
        aux_distance_from_intersection,
        convoy_size=convoy_size,
        convoy_gap_m=convoy_gap_m,
    )
    viable = []
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
    *,
    convoy_size: int = 1,
    convoy_gap_m: float = DEFAULT_CONVOY_GAP_M,
) -> bool:
    """True when at least one aux spawn lane is viable on the conflict arc."""
    return bool(
        viable_aux_lane_keys(
            junction_layout,
            aux_distance_from_intersection,
            convoy_size=convoy_size,
            convoy_gap_m=convoy_gap_m,
        )
    )


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


def _sibling_lane_keys(
    junction_layout: Optional[dict],
    spawn_lane_key: str,
) -> List[str]:
    """Other lane keys on the same edge as ``spawn_lane_key`` (same arm)."""
    if not junction_layout:
        return []
    edge_id = lane_edge_id(spawn_lane_key)
    for arm in junction_layout.get("arms", []) or []:
        if arm.get("edge_id") != edge_id:
            continue
        keys = [str(k) for k in (arm.get("lane_keys") or []) if k]
        # Prefer higher lane indices first — on rings the outer (0) lane often
        # peels to an exit while the inner lane continues circulating.
        siblings = [k for k in keys if k != str(spawn_lane_key)]
        siblings.sort(key=lambda k: lane_num_from_key(k), reverse=True)
        return siblings
    return []


def resolve_aux_destination_lane_key(
    junction_layout: Optional[dict],
    spawn_lane_key: str,
    *,
    route_index: Optional["VehicleRouteIndex"] = None,
    preferred_dest_edge: Optional[str] = None,
) -> Optional[str]:
    """Pick a destination lane key reachable from ``spawn_lane_key``.

    Prefers ``preferred_dest_edge``, then arm straight-through, then any
    outgoing arm exit. When ``route_index`` is provided, only edges that this
    *lane* can actually enter are used (outer lanes often only turn).
    """
    if not junction_layout:
        return None

    edge_id = lane_edge_id(spawn_lane_key)
    lane_num = lane_num_from_key(spawn_lane_key)
    arm = None
    for candidate in junction_layout.get("arms", []):
        if candidate.get("edge_id") == edge_id:
            arm = candidate
            break

    lane_keys_by_edge = junction_layout.get("lane_keys_by_edge") or {
        candidate.get("edge_id"): list(candidate.get("lane_keys", []))
        for candidate in junction_layout.get("arms", [])
        if candidate.get("edge_id")
    }

    candidates: List[str] = []
    if preferred_dest_edge and str(preferred_dest_edge) not in candidates:
        candidates.append(str(preferred_dest_edge))
    if arm is not None:
        for edge in arm.get("straight_to", []):
            if edge and str(edge) not in candidates and not str(edge).startswith(":"):
                candidates.append(str(edge))
        for edge in arm.get("outgoing_to", []):
            if edge and str(edge) not in candidates and not str(edge).startswith(":"):
                candidates.append(str(edge))

    if route_index is None:
        raise ValueError(
            "[AuxAgent] route_index is required to resolve aux destinations "
            "(refusing silent straight-through fallback)"
        )

    if not route_index.has_exit(edge_id, lane_num):
        return None

    is_roundabout = junction_layout.get("mode") == "roundabout"
    main_edges = {
        str(arm.get("edge_id"))
        for arm in junction_layout.get("arms", [])
        if arm.get("road_class") == "main" and arm.get("edge_id")
    }

    def _key_for_dest(dest_edge: str) -> Optional[str]:
        allowed = route_index.reachable_lanes_on_edge(
            edge_id, lane_num, dest_edge, max_hops=16
        )
        return pick_lane_key_on_edge(
            dest_edge,
            lane_num,
            lane_keys_by_edge,
            allowed_lane_nums=sorted(allowed),
        )

    preferred = str(preferred_dest_edge) if preferred_dest_edge else None
    spawn_on_main = bool(main_edges) and edge_id in main_edges

    # Roundabout ring traffic: always one-hop on the ring. Even when the ego
    # exit is SUMO-reachable (esp. outer lane, 2+ hops), MetaDrive often marks
    # mid-lane aux out_of_route mid-route and despawns them. Circulation is
    # continued by AuxiliaryAgentsManager re-route on arrive / out_of_route.
    if is_roundabout and spawn_on_main:
        ranked = route_index.reachable_real_edges_with_hops(
            edge_id, lane_num, max_hops=16
        )
        next_hops = [
            edge
            for edge, hops in ranked
            if hops == 1 and edge in main_edges and edge != edge_id
        ]
        for dest_edge in next_hops:
            key = _key_for_dest(dest_edge)
            if key:
                return key
        ring_ranked = [
            (edge, hops)
            for edge, hops in ranked
            if edge in main_edges and hops >= 2 and edge != edge_id
        ]
        ring_ranked.sort(key=lambda item: item[1], reverse=True)
        for dest_edge, _hops in ring_ranked:
            key = _key_for_dest(dest_edge)
            if key:
                return key
        return None

    if preferred and route_index.can_reach_edge(
        edge_id, lane_num, preferred, max_hops=16
    ):
        key = _key_for_dest(preferred)
        if key:
            return key

    # Non-roundabout (or spoke spawn): try arm exits in listed order.
    for dest_edge in candidates:
        if preferred and dest_edge == preferred:
            continue
        if not route_index.can_reach_edge(
            edge_id, lane_num, dest_edge, max_hops=16
        ):
            continue
        key = _key_for_dest(dest_edge)
        if key:
            return key

    # Generic farthest-reachable fallback.
    ranked = route_index.reachable_real_edges_with_hops(
        edge_id, lane_num, max_hops=16
    )
    if not ranked:
        return None

    ordered: List[str] = []
    all_ranked = sorted(ranked, key=lambda item: item[1], reverse=True)
    for edge, _hops in all_ranked:
        if edge == edge_id:
            continue
        if edge not in ordered:
            ordered.append(edge)

    for dest_edge in ordered:
        key = _key_for_dest(dest_edge)
        if key:
            return key
    return None


def resolve_aux_destination_lane_key_for_edge(
    junction_layout: Optional[dict],
    spawn_lane_key: str,
    dest_edge_id: str,
    *,
    route_index: Optional["VehicleRouteIndex"] = None,
) -> Optional[str]:
    """Outgoing lane key on ``dest_edge_id`` if reachable from the spawn lane."""
    if not junction_layout or not dest_edge_id:
        return None
    return resolve_aux_destination_lane_key(
        junction_layout,
        spawn_lane_key,
        route_index=route_index,
        preferred_dest_edge=str(dest_edge_id),
    )


def _load_route_index_from_row(row: dict, scenes_root: Optional[Path | str] = None):
    """Load SUMO route index; resolve relative ``net_path`` against scenes_root.

    Raises if ``net_path`` is missing or cannot be resolved — callers must not
    silently fall back to unclamped straight-through destinations.
    """
    net_path = row.get("net_path")
    if not net_path:
        raise ValueError(
            "[AuxAgent] row is missing net_path; cannot build route index for aux destinations"
        )

    path = Path(str(net_path))
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if scenes_root:
            candidates.append(Path(scenes_root) / path)
        for key in ("scenes_root", "scenes_dir"):
            root = row.get(key)
            if root:
                candidates.append(Path(str(root)) / path)
        candidates.append(path)

    # De-dupe while preserving order.
    seen: set[str] = set()
    unique_candidates: List[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    resolved = next((candidate for candidate in unique_candidates if candidate.is_file()), None)
    if resolved is None:
        raise FileNotFoundError(
            "[AuxAgent] Could not resolve net_path for route index. "
            f"net_path={net_path!r}, scenes_root={scenes_root!r}, "
            f"tried={[str(c) for c in unique_candidates]}"
        )

    from .sumo_utils import load_vehicle_route_index

    return load_vehicle_route_index(resolved)


def resolve_aux_spawn_plan(
    row: dict,
    ego_lane_index: str,
    incoming_lanes: Optional[List[dict]] = None,
    aux_lanes_occupied: int = 1,
    aux_distance_from_intersection: float = DEFAULT_DISTANCE_FROM_INTERSECTION,
    route_index=None,
    scenes_root: Optional[Path | str] = None,
) -> tuple[List[str], List[str], dict, dict, dict]:
    """Resolve aux spawn lanes, destinations, alternate spawn->dest, longitudes,
    and roundabout ring-circulate flags.
    """
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

    if route_index is None:
        route_index = _load_route_index_from_row(row, scenes_root=scenes_root)

    # Roundabout: aux always aims for the same exit as ego.
    ego_dest_preferred: Optional[str] = None
    if (junction_layout or {}).get("mode") == "roundabout":
        if row.get("destination_edge_id"):
            ego_dest_preferred = str(row["destination_edge_id"])
        elif row.get("destination_lane_id"):
            ego_dest_preferred = lane_edge_id(str(row["destination_lane_id"]))

    # Roundabout: always spawn on the left-hand conflict arc with the full
    # aux_distance. Older manifests may still point upstream; short arcs
    # (insufficient length) clear aux spawn so generation-time rejects stay
    # consistent at runtime.
    conflict_spawn_longs: dict = {}
    if (junction_layout or {}).get("mode") == "roundabout" and ego_edge:
        try:
            from .roundabout_aux import (
                MIN_CONFLICT_ARC_LENGTH_M,
                merge_lane_lengths_from_layout,
                resolve_aux_spawn_placement,
            )
            from .roundabout_yield_zone import entry_conflict_ring_edges

            left = entry_conflict_ring_edges(junction_layout, ego_edge)
            left_set = set(left)
            lengths = merge_lane_lengths_from_layout(junction_layout, {})
            aux_distance = float(
                row.get(
                    "aux_distance_from_intersection",
                    aux_distance_from_intersection,
                )
            )
            lanes_n = max(1, int(aux_lanes_occupied or 1))
            needs_rebase = (not spawn_lanes) or any(
                lane_edge_id(key) not in left_set for key in spawn_lanes
            )

            def _placements_on_conflict() -> List[tuple[str, float]]:
                found: List[tuple[str, float]] = []
                for conflict_edge in left:
                    arm = None
                    for candidate in (junction_layout or {}).get("arms", []) or []:
                        if candidate.get("edge_id") == conflict_edge:
                            arm = candidate
                            break
                    lane_nums = sorted(
                        {
                            lane_num_from_key(str(k))
                            for k in (arm.get("lane_keys") if arm else []) or []
                        }
                    ) or [0]
                    for lane_num in lane_nums:
                        placement = resolve_aux_spawn_placement(
                            junction_layout,
                            conflict_edge,
                            lane_num,
                            lengths,
                            aux_distance,
                            allowed_ring_edges=left_set,
                        )
                        if placement is None:
                            continue
                        key = placement.spawn_lane_key
                        found.append((key, float(placement.spawn_longitudinal)))
                return found

            placed = _placements_on_conflict() if left else []
            for key, long_val in placed:
                conflict_spawn_longs[key] = long_val

            if left and not placed:
                if spawn_lanes:
                    logging.info(
                        "[AuxAgent] Conflict arc too short "
                        f"(need >= {MIN_CONFLICT_ARC_LENGTH_M:.0f}m); "
                        f"clearing aux spawn (was {spawn_lanes})"
                    )
                spawn_lanes = []
            elif left and needs_rebase and placed:
                rebuilt = []
                for key, _long_val in placed:
                    if key in rebuilt:
                        continue
                    rebuilt.append(key)
                    if len(rebuilt) >= lanes_n:
                        break
                if rebuilt:
                    logging.info(
                        "[AuxAgent] Rebased roundabout aux spawn onto conflict "
                        "arc %s (was %s)",
                        rebuilt,
                        spawn_lanes,
                    )
                    spawn_lanes = rebuilt[:lanes_n]
            elif left and spawn_lanes and placed:
                # Already on the conflict arc (e.g. viable-lane remap): still
                # refresh longitudes so stale upstream offsets are not reused.
                for key in spawn_lanes:
                    if key not in conflict_spawn_longs:
                        edge = lane_edge_id(key)
                        for place_key, long_val in placed:
                            if lane_edge_id(place_key) == edge:
                                conflict_spawn_longs[key] = long_val
                                break
        except Exception as exc:
            logging.debug("[AuxAgent] Conflict-arc rebase skipped: %s", exc)

    # Drop dead-end approach lanes (no SUMO connections — common on outer lanes).
    # When a requested lane has no exit, prefer a sibling on the same edge that
    # does (roundabout outer ring lanes often only peel off to an exit spoke).
    if spawn_lanes:
        kept: List[str] = []
        seen_kept: set[str] = set()
        for lane_key in spawn_lanes:
            edge_id = lane_edge_id(lane_key)
            lane_num = lane_num_from_key(lane_key)
            if route_index.has_exit(edge_id, lane_num):
                if lane_key not in seen_kept:
                    kept.append(lane_key)
                    seen_kept.add(lane_key)
                continue
            remapped = False
            for alt in _sibling_lane_keys(junction_layout, lane_key):
                if alt in seen_kept:
                    continue
                if route_index.has_exit(lane_edge_id(alt), lane_num_from_key(alt)):
                    logging.info(
                        "[AuxAgent] Remapping aux spawn %s -> %s "
                        "(no SUMO exit on requested lane)",
                        lane_key,
                        alt,
                    )
                    kept.append(alt)
                    seen_kept.add(alt)
                    remapped = True
                    break
            if not remapped:
                logging.info(
                    "[AuxAgent] Skipping aux spawn %s: no SUMO exit connections",
                    lane_key,
                )
        spawn_lanes = kept

    viable_keys = viable_aux_lane_keys(
        junction_layout,
        float(row.get("aux_distance_from_intersection", aux_distance_from_intersection)),
        ego_edge,
        convoy_size=int(row.get("aux_convoy_size", 1) or 1),
        convoy_gap_m=float(
            row.get("aux_convoy_gap_m", DEFAULT_CONVOY_GAP_M) or DEFAULT_CONVOY_GAP_M
        ),
    )
    viable_keys = [
        key
        for key in viable_keys
        if route_index.has_exit(lane_edge_id(key), lane_num_from_key(key))
    ]

    manifest_dest = row.get("aux_destination_lane_id")
    manifest_dest_edge = row.get("aux_destination_edge_id") or (
        lane_edge_id(str(manifest_dest)) if manifest_dest else None
    )
    if ego_dest_preferred:
        # Roundabout rule: aux destination edge is always ego's exit.
        manifest_dest_edge = ego_dest_preferred
        if row.get("destination_lane_id"):
            manifest_dest = str(row["destination_lane_id"])
    manifest_spawn = row.get("aux_spawn_lane_index")
    lane_keys_by_edge = (junction_layout or {}).get("lane_keys_by_edge") or {}

    # Per-lane destinations: each approach lane may only allow a subset of turns.
    alternate_spawn_dest_map: dict = {}
    for lane_key in viable_keys:
        preferred = ego_dest_preferred or (
            str(manifest_dest_edge)
            if manifest_dest_edge and lane_key == str(manifest_spawn)
            else None
        )
        dest = resolve_aux_destination_lane_key(
            junction_layout,
            lane_key,
            route_index=route_index,
            preferred_dest_edge=preferred,
        )
        if dest:
            alternate_spawn_dest_map[lane_key] = dest

    if manifest_dest and manifest_spawn:
        preferred_edge = (
            ego_dest_preferred
            or (str(manifest_dest_edge) if manifest_dest_edge else None)
        )
        resolved = resolve_aux_destination_lane_key(
            junction_layout,
            str(manifest_spawn),
            route_index=route_index,
            preferred_dest_edge=preferred_edge,
        )
        if resolved:
            alternate_spawn_dest_map[str(manifest_spawn)] = resolved
            manifest_dest = resolved
        elif preferred_edge:
            # Manifest dest unreachable from this lane — fall back per-lane.
            manifest_dest = None

    destination_lanes: List[str] = []
    kept_spawns: List[str] = []
    for idx, spawn_lane in enumerate(spawn_lanes):
        preferred_edge = ego_dest_preferred
        if preferred_edge is None and manifest_dest and (
            (manifest_spawn and spawn_lane == str(manifest_spawn))
            or (idx == 0 and not manifest_spawn)
        ):
            preferred_edge = lane_edge_id(str(manifest_dest))

        candidates = [spawn_lane] + [
            alt
            for alt in _sibling_lane_keys(junction_layout, spawn_lane)
            if alt not in kept_spawns
        ]
        chosen_spawn: Optional[str] = None
        dest: Optional[str] = None
        for candidate in candidates:
            dest = resolve_aux_destination_lane_key(
                junction_layout,
                candidate,
                route_index=route_index,
                preferred_dest_edge=preferred_edge,
            )
            if dest:
                chosen_spawn = candidate
                if candidate != spawn_lane:
                    logging.info(
                        "[AuxAgent] Remapping aux spawn %s -> %s "
                        "(no reachable ring/exit dest on requested lane)",
                        spawn_lane,
                        candidate,
                    )
                break
        if not chosen_spawn or not dest:
            logging.info(
                "[AuxAgent] No reachable dest for aux spawn %s; skipping lane",
                spawn_lane,
            )
            continue

        if lane_keys_by_edge:
            # Keep the resolved dest lane index (may differ from spawn lane —
            # e.g. left turn from lane 0 onto dest lane 1).
            dest = pick_lane_key_on_edge(
                lane_edge_id(dest),
                lane_num_from_key(dest),
                lane_keys_by_edge,
            )
        kept_spawns.append(chosen_spawn)
        destination_lanes.append(dest or "")
        alternate_spawn_dest_map[chosen_spawn] = dest

    ring_circulate_by_lane: dict = {}
    if (junction_layout or {}).get("mode") == "roundabout":
        main_edges = {
            str(arm.get("edge_id"))
            for arm in (junction_layout or {}).get("arms", [])
            if arm.get("road_class") == "main" and arm.get("edge_id")
        }
        for spawn_lane in kept_spawns:
            # All ring-spawned aux hop one segment at a time.
            if lane_edge_id(spawn_lane) in main_edges:
                ring_circulate_by_lane[spawn_lane] = True

    spawn_longitudinal_by_lane: dict = {}
    if conflict_spawn_longs:
        for spawn_lane in kept_spawns:
            if spawn_lane in conflict_spawn_longs:
                spawn_longitudinal_by_lane[spawn_lane] = float(
                    conflict_spawn_longs[spawn_lane]
                )
            else:
                # Sibling on same conflict edge: reuse any known longitude.
                edge = lane_edge_id(spawn_lane)
                for key, val in conflict_spawn_longs.items():
                    if lane_edge_id(key) == edge:
                        spawn_longitudinal_by_lane[spawn_lane] = float(val)
                        break
    manifest_long = row.get("aux_spawn_longitudinal")
    manifest_spawn = row.get("aux_spawn_lane_index")
    if (
        not spawn_longitudinal_by_lane
        and manifest_long is not None
        and kept_spawns
    ):
        long_val = float(manifest_long)
        # Prefer applying the manifest longitude onto the (possibly remapped)
        # spawn lanes that share the manifest edge; fall back to all kept.
        edge = lane_edge_id(str(manifest_spawn)) if manifest_spawn else None
        applied = False
        for spawn_lane in kept_spawns:
            if edge is None or lane_edge_id(spawn_lane) == edge:
                spawn_longitudinal_by_lane[spawn_lane] = long_val
                applied = True
        if not applied:
            for spawn_lane in kept_spawns:
                spawn_longitudinal_by_lane[spawn_lane] = long_val
    if (
        (junction_layout or {}).get("mode") == "roundabout"
        and kept_spawns
        and ego_edge
        and len(spawn_longitudinal_by_lane) < len(kept_spawns)
    ):
        # Fill any lanes still missing a longitude (older manifests / other edges).
        try:
            from .roundabout_aux import (
                merge_lane_lengths_from_layout,
                resolve_aux_spawn_placement,
            )
            from .roundabout_yield_zone import entry_conflict_ring_edges

            lengths = merge_lane_lengths_from_layout(junction_layout, {})
            left = entry_conflict_ring_edges(junction_layout, ego_edge)
            aux_distance = float(
                row.get(
                    "aux_distance_from_intersection",
                    aux_distance_from_intersection,
                )
            )
            for spawn_lane in kept_spawns:
                if spawn_lane in spawn_longitudinal_by_lane:
                    continue
                spawn_edge = lane_edge_id(spawn_lane)
                spawn_ln = lane_num_from_key(spawn_lane)
                conflict_edge = spawn_edge
                if left and spawn_edge not in left:
                    conflict_edge = left[0]
                placement = resolve_aux_spawn_placement(
                    junction_layout,
                    conflict_edge,
                    spawn_ln,
                    lengths,
                    aux_distance,
                    allowed_ring_edges=set(left) if left else None,
                )
                if placement is not None and placement.spawn_lane_key == spawn_lane:
                    spawn_longitudinal_by_lane[spawn_lane] = float(
                        placement.spawn_longitudinal
                    )
                elif placement is not None and lane_edge_id(spawn_lane) == (
                    placement.spawn_edge_id
                ):
                    # Sibling lane on the placement edge: reuse longitude.
                    spawn_longitudinal_by_lane[spawn_lane] = float(
                        placement.spawn_longitudinal
                    )
        except Exception:
            pass
    return (
        kept_spawns,
        destination_lanes,
        alternate_spawn_dest_map,
        spawn_longitudinal_by_lane,
        ring_circulate_by_lane,
    )



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
    spawn_longitudinal_by_lane: Optional[dict] = None,
    ring_circulate_by_lane: Optional[dict] = None,
    junction_layout: Optional[dict] = None,
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
        spawn_longitudinal_by_lane=spawn_longitudinal_by_lane,
        ring_circulate_by_lane=ring_circulate_by_lane,
        junction_layout=junction_layout,
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
