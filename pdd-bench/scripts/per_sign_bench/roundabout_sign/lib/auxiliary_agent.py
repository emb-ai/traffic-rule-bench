"""Auxiliary agents (main road NPCs) for yield sign scenarios."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from metadrive.manager.base_manager import BaseManager
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.component.navigation_module.edge_network_navigation import EdgeNetworkNavigation

from .lane_keys import lane_edge_id, lane_num_from_key, make_lane_key, parse_lane_key


DEFAULT_DISTANCE_FROM_INTERSECTION = 20.0
DEFAULT_SPAWN_VELOCITY_MS = 5.0
DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END = 5.0
DEFAULT_CONVOY_SIZE = 3
DEFAULT_CONVOY_GAP_M = 10.0
MIN_SPAWN_LONGITUDE_M = 3.0
# Advance ring-loop destination when this close to the end of the current dest lane.
RING_LOOP_ADVANCE_M = 8.0
# Despawn aux that left the roadway (treated like a soft collision / vanish).
OFFROAD_DESPAWN_GRACE_STEPS = 20
OFFROAD_DESPAWN_STREAK = 3
OFFROAD_LATERAL_MARGIN_M = 1.5


def achievable_convoy_sizes(
    lead_spawn_long: float,
    *,
    convoy_gap_m: float = DEFAULT_CONVOY_GAP_M,
    max_convoy: int = DEFAULT_CONVOY_SIZE,
    min_spawn_long: float = MIN_SPAWN_LONGITUDE_M,
) -> List[int]:
    """Return convoy sizes supported on one lane (spatial or sequential at lead point)."""
    _ = (lead_spawn_long, convoy_gap_m, min_spawn_long)
    cap = max(1, int(max_convoy))
    return list(range(1, cap + 1))


def manifest_entry_spawn_fingerprint(entry: dict) -> tuple:
    """Hashable key for the simulation-relevant spawn layout (dedup identical variants)."""
    aux_keys = tuple(sorted(str(k) for k in (entry.get("aux_occupied_lane_keys") or [])))
    if not aux_keys and entry.get("aux_spawn_lane_index"):
        aux_keys = (str(entry["aux_spawn_lane_index"]),)
    spawn_long = entry.get("aux_spawn_longitudinal")
    return (
        str(entry.get("scene_id") or ""),
        str(entry.get("road_id") or ""),
        int(entry.get("spawn_lane_num") or 0),
        str(entry.get("destination_edge_id") or ""),
        str(entry.get("destination_lane_id") or ""),
        aux_keys,
        int(entry.get("aux_convoy_size") or 1),
        len(aux_keys) if aux_keys else int(entry.get("aux_lanes_occupied") or 1),
        round(float(spawn_long), 2) if spawn_long is not None else None,
    )


@dataclass
class _PendingConvoySpawn:
    """Convoy slot waiting for the predecessor to vacate the lead spawn point."""

    spawn_lane_index: str
    lead_spawn_long: float
    destination_lane: Optional[str]
    convoy_position: int
    wait_slot: int
    lane_convoy: List[Optional[BaseVehicle]]


AuxPolicyType = Literal["idm", "stationary"]


def min_aux_spawn_lane_length(aux_distance_from_intersection: float) -> float:
    """Minimum incoming lane length required to place an aux convoy on one segment."""
    return float(aux_distance_from_intersection) + MIN_SPAWN_LONGITUDE_M


def is_viable_aux_lane_length(
    lane_length: float,
    aux_distance_from_intersection: float,
) -> bool:
    return float(lane_length) >= min_aux_spawn_lane_length(aux_distance_from_intersection)


def _layout_arms(junction_layout: Optional[dict]) -> List[dict]:
    if not junction_layout:
        return []
    return list(junction_layout.get("arms", []))


def _arm_for_edge(junction_layout: Optional[dict], edge_id: str) -> Optional[dict]:
    for arm in _layout_arms(junction_layout):
        if arm.get("edge_id") == edge_id:
            return arm
    return None


def upstream_ring_arm(
    junction_layout: Optional[dict],
    edge_id: str,
) -> Optional[dict]:
    """Ring segment immediately upstream of ``edge_id`` (feeds into its ``from_node``)."""
    arm = _arm_for_edge(junction_layout, edge_id)
    if arm is None or arm.get("road_class") != "main":
        return None
    from_node = str(arm.get("from_node", ""))
    if not from_node:
        return None
    upstream: List[dict] = []
    for candidate in _layout_arms(junction_layout):
        if candidate.get("road_class") != "main":
            continue
        if str(candidate.get("to_node", "")) == from_node:
            upstream.append(candidate)
    if not upstream:
        return None
    return max(upstream, key=lambda item: float(item.get("min_lane_length", 0.0) or 0.0))


def lane_length_for_spawn(
    edge_id: str,
    lane_num: int,
    lane_lengths: Dict[Tuple[str, int], float],
    junction_layout: Optional[dict],
) -> float:
    length = float(lane_lengths.get((edge_id, lane_num), 0.0) or 0.0)
    if length > 0.0:
        return length
    arm = _arm_for_edge(junction_layout, edge_id)
    if arm is not None:
        return float(arm.get("min_lane_length", 0.0) or 0.0)
    return 0.0


@dataclass(frozen=True)
class AuxSpawnPlacement:
    """Resolved aux spawn lane and longitudinal offset along it."""

    spawn_edge_id: str
    spawn_lane_num: int
    spawn_longitudinal: float
    conflict_edge_id: str
    conflict_lane_num: int

    @property
    def spawn_lane_key(self) -> str:
        return make_lane_key(self.spawn_edge_id, self.spawn_lane_num)


def resolve_aux_spawn_placement(
    junction_layout: Optional[dict],
    edge_id: str,
    lane_num: int,
    lane_lengths: Dict[Tuple[str, int], float],
    aux_distance_from_intersection: float = DEFAULT_DISTANCE_FROM_INTERSECTION,
    *,
    allowed_ring_edges: Optional[set[str]] = None,
) -> Optional[AuxSpawnPlacement]:
    """Place aux ``aux_distance`` before the junction, extending onto upstream ring if needed."""
    aux_distance = float(aux_distance_from_intersection)
    lane_length = lane_length_for_spawn(edge_id, lane_num, lane_lengths, junction_layout)
    if lane_length <= 0.0:
        return None

    if allowed_ring_edges is not None and edge_id not in allowed_ring_edges:
        return None

    if lane_length >= aux_distance + MIN_SPAWN_LONGITUDE_M:
        spawn_long = lane_length - aux_distance
        if spawn_long < MIN_SPAWN_LONGITUDE_M:
            spawn_long = MIN_SPAWN_LONGITUDE_M
        if spawn_long > max(lane_length - 0.1, MIN_SPAWN_LONGITUDE_M):
            return None
        return AuxSpawnPlacement(
            spawn_edge_id=edge_id,
            spawn_lane_num=lane_num,
            spawn_longitudinal=float(spawn_long),
            conflict_edge_id=edge_id,
            conflict_lane_num=lane_num,
        )

    remainder = aux_distance - lane_length
    upstream = upstream_ring_arm(junction_layout, edge_id)
    if upstream is not None:
        up_edge = str(upstream.get("edge_id", ""))
        if up_edge and (
            allowed_ring_edges is None or up_edge in allowed_ring_edges
        ):
            up_length = lane_length_for_spawn(
                up_edge, lane_num, lane_lengths, junction_layout
            )
            if up_length >= remainder + MIN_SPAWN_LONGITUDE_M:
                spawn_long = up_length - remainder
                if spawn_long >= MIN_SPAWN_LONGITUDE_M:
                    return AuxSpawnPlacement(
                        spawn_edge_id=up_edge,
                        spawn_lane_num=lane_num,
                        spawn_longitudinal=float(spawn_long),
                        conflict_edge_id=edge_id,
                        conflict_lane_num=lane_num,
                    )

    if allowed_ring_edges is not None and edge_id not in allowed_ring_edges:
        return None
    spawn_long = max(MIN_SPAWN_LONGITUDE_M, lane_length - MIN_SPAWN_LONGITUDE_M)
    if spawn_long > max(lane_length - 0.1, MIN_SPAWN_LONGITUDE_M):
        return None
    return AuxSpawnPlacement(
        spawn_edge_id=edge_id,
        spawn_lane_num=lane_num,
        spawn_longitudinal=float(spawn_long),
        conflict_edge_id=edge_id,
        conflict_lane_num=lane_num,
    )


def is_aux_lane_viable_with_ring_extension(
    junction_layout: Optional[dict],
    edge_id: str,
    lane_num: int,
    lane_lengths: Dict[Tuple[str, int], float],
    aux_distance_from_intersection: float = DEFAULT_DISTANCE_FROM_INTERSECTION,
    *,
    allowed_ring_edges: Optional[set[str]] = None,
) -> bool:
    return (
        resolve_aux_spawn_placement(
            junction_layout,
            edge_id,
            lane_num,
            lane_lengths,
            aux_distance_from_intersection,
            allowed_ring_edges=allowed_ring_edges,
        )
        is not None
    )


def merge_lane_lengths_from_layout(
    junction_layout: Optional[dict],
    lane_lengths: Dict[Tuple[str, int], float],
) -> Dict[Tuple[str, int], float]:
    """Fill missing (edge, lane) lengths from junction arm minima."""
    merged = dict(lane_lengths)
    for arm in _layout_arms(junction_layout):
        edge_id = str(arm.get("edge_id", ""))
        min_len = float(arm.get("min_lane_length", 0.0) or 0.0)
        if not edge_id or min_len <= 0.0:
            continue
        for lane_key in arm.get("lane_keys", []):
            lane_num = lane_num_from_key(str(lane_key))
            merged.setdefault((edge_id, lane_num), min_len)
    return merged


def ordered_ring_lane_cycle(
    junction_layout: Optional[dict],
    *,
    lane_num: int = 0,
) -> List[str]:
    """Return main/ring lane keys in circulation order (a closed cycle when possible)."""
    main_arms = [
        arm for arm in _layout_arms(junction_layout) if arm.get("road_class") == "main"
    ]
    if not main_arms:
        return []

    by_from: Dict[str, dict] = {}
    for arm in main_arms:
        from_node = str(arm.get("from_node", ""))
        if from_node and from_node not in by_from:
            by_from[from_node] = arm

    start = main_arms[0]
    cycle_arms: List[dict] = []
    seen_edges: set[str] = set()
    cur: Optional[dict] = start
    while cur is not None:
        edge_id = str(cur.get("edge_id", ""))
        if not edge_id or edge_id in seen_edges:
            break
        seen_edges.add(edge_id)
        cycle_arms.append(cur)
        nxt = by_from.get(str(cur.get("to_node", "")))
        if nxt is None:
            break
        cur = nxt
        if len(cycle_arms) > len(main_arms) + 1:
            break

    keys: List[str] = []
    for arm in cycle_arms:
        arm_keys = [str(k) for k in (arm.get("lane_keys") or [])]
        chosen: Optional[str] = None
        for key in arm_keys:
            if lane_num_from_key(key) == int(lane_num):
                chosen = key
                break
        if chosen is None and arm_keys:
            chosen = arm_keys[min(int(lane_num), len(arm_keys) - 1)]
        if chosen is None:
            chosen = make_lane_key(str(arm.get("edge_id")), int(lane_num))
        keys.append(chosen)
    return keys


def next_ring_lane_in_cycle(cycle: List[str], lane_key: Optional[str]) -> Optional[str]:
    """Next lane in the ring cycle after ``lane_key`` (wraps around)."""
    if not cycle:
        return None
    if not lane_key:
        return cycle[0]
    edge = lane_edge_id(str(lane_key))
    for i, key in enumerate(cycle):
        if key == lane_key or lane_edge_id(key) == edge:
            return cycle[(i + 1) % len(cycle)]
    return cycle[0]


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
        ring_loop_lanes: Optional[List[str]] = None,
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
        self._spawn_longitudinal_by_lane = dict(spawn_longitudinal_by_lane or {})
        self._ring_loop_lanes = list(ring_loop_lanes or [])
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
        self._aux_spawn_steps: List[int] = []
        self._offroad_streaks: List[int] = []
        self._pending_convoy_spawns: List[_PendingConvoySpawn] = []
        self._despawned_offroad_count = 0

    def reset(self):
        self._aux_vehicles = []
        self._spawn_lane_indices = []
        self._spawn_destinations = []
        self._convoy_positions = []
        self._aux_policies = []
        self._aux_spawn_steps = []
        self._offroad_streaks = []
        self._pending_convoy_spawns = []
        self._despawned_offroad_count = 0

    def after_reset(self):
        if self._ring_loop_lanes:
            try:
                road_network = self.engine.current_map.road_network
                filtered = [
                    key for key in self._ring_loop_lanes if key in road_network.graph
                ]
            except Exception:
                filtered = list(self._ring_loop_lanes)
            if len(filtered) < 2:
                logging.warning(
                    "[AuxAgent] ring loop disabled: need >=2 ring lanes in map "
                    f"(got {len(filtered)} from {len(self._ring_loop_lanes)})"
                )
                self._ring_loop_lanes = []
            else:
                self._ring_loop_lanes = filtered
                print(
                    f"[AuxAgent] Ring loop enabled: {len(self._ring_loop_lanes)} lane(s) "
                    f"{' → '.join(self._ring_loop_lanes[:4])}"
                    + (" → …" if len(self._ring_loop_lanes) > 4 else "")
                    + " → (cycle)"
                )
        self._spawn_auxiliary_vehicles()

    def _loop_destination_for_spawn(self, spawn_lane_index: str) -> Optional[str]:
        if len(self._ring_loop_lanes) < 2:
            return None
        return next_ring_lane_in_cycle(self._ring_loop_lanes, spawn_lane_index)

    def _should_advance_ring_destination(
        self,
        vehicle: BaseVehicle,
        destination_lane: Optional[str],
    ) -> bool:
        if not destination_lane or len(self._ring_loop_lanes) < 2:
            return False
        try:
            road_network = self.engine.current_map.road_network
            lane = road_network.get_lane(destination_lane)
            longitudinal, lateral = lane.local_coordinates(vehicle.position)
            on_dest = abs(float(lateral)) < 4.0 and float(longitudinal) >= -2.0
            near_end = float(longitudinal) >= float(lane.length) - RING_LOOP_ADVANCE_M
            if on_dest and near_end:
                return True
            # Already left the destination edge onto the next ring segment.
            current = getattr(vehicle, "lane_index", None)
            if current is None:
                return False
            current_key = str(current)
            if current_key == destination_lane:
                return False
            expected_next = next_ring_lane_in_cycle(self._ring_loop_lanes, destination_lane)
            return (
                expected_next is not None
                and (
                    current_key == expected_next
                    or lane_edge_id(current_key) == lane_edge_id(expected_next)
                )
            )
        except Exception:
            return False

    def _advance_ring_destination(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._aux_vehicles):
            return
        vehicle = self._aux_vehicles[idx]
        old_dest = self._spawn_destinations[idx] if idx < len(self._spawn_destinations) else None
        new_dest = next_ring_lane_in_cycle(self._ring_loop_lanes, old_dest)
        if not new_dest or new_dest == old_dest:
            return
        start = None
        current = getattr(vehicle, "lane_index", None)
        if current is not None:
            start = str(current)
        if not start:
            start = old_dest
        try:
            if vehicle.navigation is not None and start:
                vehicle.navigation.set_route(start, new_dest)
            self._spawn_destinations[idx] = new_dest
            logging.debug(
                "[AuxAgent] Ring loop advance %s → %s (from %s)",
                old_dest,
                new_dest,
                start,
            )
        except Exception as exc:
            logging.debug("[AuxAgent] Ring loop advance failed: %s", exc)

    def _maybe_advance_ring_routes(self) -> None:
        if len(self._ring_loop_lanes) < 2:
            return
        for idx, (vehicle, destination) in enumerate(
            zip(self._aux_vehicles, self._spawn_destinations)
        ):
            if self._should_advance_ring_destination(vehicle, destination):
                self._advance_ring_destination(idx)

    def _vehicle_longitudinal_on_lane(self, vehicle: BaseVehicle, lane_index: str) -> Optional[float]:
        try:
            lane = self.engine.current_map.road_network.get_lane(lane_index)
            longitudinal, _ = lane.local_coordinates(vehicle.position)
            return float(longitudinal)
        except Exception:
            return None

    def _predecessor_cleared_spawn_point(
        self,
        predecessor: BaseVehicle,
        lane_index: str,
        lead_spawn_long: float,
    ) -> bool:
        """True when the predecessor has moved far enough to reuse the lead spawn point."""
        long_pos = self._vehicle_longitudinal_on_lane(predecessor, lane_index)
        if long_pos is None:
            return False
        return long_pos >= float(lead_spawn_long) + self._convoy_gap_m

    def _try_spawn_pending_convoys(self) -> None:
        if not self._pending_convoy_spawns:
            return

        still_pending: List[_PendingConvoySpawn] = []
        for pending in self._pending_convoy_spawns:
            if pending.wait_slot >= 0:
                predecessor = pending.lane_convoy[pending.wait_slot]
                if predecessor is None:
                    # Never spawned or already despawned — do not block forever.
                    pass
                elif not self._predecessor_cleared_spawn_point(
                    predecessor,
                    pending.spawn_lane_index,
                    pending.lead_spawn_long,
                ):
                    still_pending.append(pending)
                    continue

            if self._spawn_vehicle_on_lane(
                pending.spawn_lane_index,
                pending.lead_spawn_long,
                pending.destination_lane,
                pending.convoy_position,
            ):
                pending.lane_convoy[pending.convoy_position] = self._aux_vehicles[-1]
                print(
                    f"[AuxAgent] Sequential convoy slot {pending.convoy_position + 1}/"
                    f"{self._convoy_size} on {pending.spawn_lane_index} "
                    f"at {pending.lead_spawn_long:.1f}m "
                    f"(after slot {pending.wait_slot + 1} cleared)"
                )
            else:
                still_pending.append(pending)
        self._pending_convoy_spawns = still_pending

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
            self._aux_spawn_steps.append(int(getattr(self.engine, "episode_step", 0) or 0))
            self._offroad_streaks.append(0)
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
        self._aux_spawn_steps = []
        self._offroad_streaks = []
        self._pending_convoy_spawns = []

        for idx, spawn_lane_index in enumerate(self._requested_spawn_lane_indices):
            candidate_lanes = [spawn_lane_index]
            for alt_lane in self._alternate_spawn_dest_map:
                if alt_lane not in candidate_lanes:
                    candidate_lanes.append(alt_lane)

            spawned_on_lane = 0
            used_lane = None
            used_destination = None
            for candidate_lane in candidate_lanes:
                lane = road_network.get_lane(candidate_lane)
                if candidate_lane in self._spawn_longitudinal_by_lane:
                    lead_spawn_long = float(self._spawn_longitudinal_by_lane[candidate_lane])
                else:
                    lead_spawn_long = lane.length - self._distance_from_intersection
                if lead_spawn_long < MIN_SPAWN_LONGITUDE_M:
                    if candidate_lane == spawn_lane_index:
                        logging.warning(
                            f"[AuxAgent] Lane {candidate_lane} too short for convoy "
                            f"(lead at {lead_spawn_long:.1f}m, sim length={lane.length:.1f}m)"
                        )
                    continue

                if candidate_lane in self._alternate_spawn_dest_map:
                    destination_lane = self._alternate_spawn_dest_map[candidate_lane]
                elif idx < len(self._destination_lanes) and self._destination_lanes[idx]:
                    destination_lane = self._destination_lanes[idx]
                else:
                    destination_lane = pick_destination_outgoing_lane(
                        candidate_lane, self._outgoing_lanes, road_network
                    )
                loop_dest = self._loop_destination_for_spawn(candidate_lane)
                if loop_dest:
                    destination_lane = loop_dest

                spawned_on_lane = 0
                lane_convoy: List[Optional[BaseVehicle]] = [None] * self._convoy_size
                lane_pending: List[_PendingConvoySpawn] = []
                for convoy_idx in range(self._convoy_size):
                    spawn_long = lead_spawn_long - convoy_idx * self._convoy_gap_m
                    if spawn_long >= MIN_SPAWN_LONGITUDE_M:
                        if self._spawn_vehicle_on_lane(
                            candidate_lane,
                            spawn_long,
                            destination_lane,
                            convoy_idx,
                        ):
                            spawned_on_lane += 1
                            lane_convoy[convoy_idx] = self._aux_vehicles[-1]
                        elif convoy_idx == 0:
                            break
                        else:
                            lane_pending.append(
                                _PendingConvoySpawn(
                                    spawn_lane_index=candidate_lane,
                                    lead_spawn_long=lead_spawn_long,
                                    destination_lane=destination_lane,
                                    convoy_position=convoy_idx,
                                    wait_slot=convoy_idx - 1,
                                    lane_convoy=lane_convoy,
                                )
                            )
                    else:
                        if convoy_idx == 0:
                            break
                        lane_pending.append(
                            _PendingConvoySpawn(
                                spawn_lane_index=candidate_lane,
                                lead_spawn_long=lead_spawn_long,
                                destination_lane=destination_lane,
                                convoy_position=convoy_idx,
                                wait_slot=convoy_idx - 1,
                                lane_convoy=lane_convoy,
                            )
                        )

                if spawned_on_lane > 0 or lane_pending:
                    if lane_pending:
                        self._pending_convoy_spawns.extend(lane_pending)
                    used_lane = candidate_lane
                    used_destination = destination_lane
                    break

            if used_lane is not None:
                pending_n = sum(
                    1
                    for pending in self._pending_convoy_spawns
                    if pending.spawn_lane_index == used_lane
                )
                print(
                    f"[AuxAgent] Convoy x{spawned_on_lane} on {used_lane} "
                    f"-> {used_destination} ({self._policy}, gap={self._convoy_gap_m:.1f}m)"
                    + (f", +{pending_n} sequential pending" if pending_n else "")
                )

    def _aux_is_off_road(self, vehicle: BaseVehicle) -> bool:
        """True when the aux left the drivable surface (sidewalk / off-lane)."""
        if bool(getattr(vehicle, "crash_sidewalk", False)):
            return True
        on_lane = getattr(vehicle, "on_lane", None)
        if on_lane is False:
            return True
        if bool(getattr(vehicle, "out_of_route", False)):
            return True
        try:
            lane = getattr(vehicle, "lane", None)
            if lane is not None:
                _longitudinal, lateral = lane.local_coordinates(vehicle.position)
                half = float(getattr(lane, "width", 3.5) or 3.5) * 0.5 + OFFROAD_LATERAL_MARGIN_M
                if abs(float(lateral)) > half:
                    return True
        except Exception:
            pass
        return False

    def _despawn_aux_at(self, idx: int, reason: str) -> None:
        if idx < 0 or idx >= len(self._aux_vehicles):
            return
        vehicle = self._aux_vehicles[idx]
        for pending in self._pending_convoy_spawns:
            for slot, other in enumerate(pending.lane_convoy):
                if other is vehicle:
                    pending.lane_convoy[slot] = None
        vid = getattr(vehicle, "id", None)
        try:
            if vid is not None:
                self.clear_objects([vid])
            elif hasattr(vehicle, "destroy"):
                vehicle.destroy()
        except Exception as exc:
            logging.debug("[AuxAgent] clear_objects failed for off-road aux: %s", exc)
        print(
            f"[AuxAgent] Despawned off-road aux "
            f"(convoy_pos={self._convoy_positions[idx] if idx < len(self._convoy_positions) else '?'}, "
            f"reason={reason})"
        )
        self._despawned_offroad_count += 1
        for lst in (
            self._aux_vehicles,
            self._spawn_lane_indices,
            self._spawn_destinations,
            self._convoy_positions,
            self._aux_policies,
            self._aux_spawn_steps,
            self._offroad_streaks,
        ):
            if idx < len(lst):
                lst.pop(idx)

    def _despawn_offroad_auxiliaries(self) -> None:
        if not self._aux_vehicles:
            return
        episode_step = int(getattr(self.engine, "episode_step", 0) or 0)
        while len(self._offroad_streaks) < len(self._aux_vehicles):
            self._offroad_streaks.append(0)
        while len(self._aux_spawn_steps) < len(self._aux_vehicles):
            self._aux_spawn_steps.append(episode_step)

        to_remove: List[int] = []
        for idx, vehicle in enumerate(self._aux_vehicles):
            spawn_step = self._aux_spawn_steps[idx]
            if episode_step - spawn_step < OFFROAD_DESPAWN_GRACE_STEPS:
                self._offroad_streaks[idx] = 0
                continue
            if self._aux_is_off_road(vehicle):
                self._offroad_streaks[idx] += 1
            else:
                self._offroad_streaks[idx] = 0
            if self._offroad_streaks[idx] >= OFFROAD_DESPAWN_STREAK:
                to_remove.append(idx)
        for idx in reversed(to_remove):
            reason = "crash_sidewalk" if bool(
                getattr(self._aux_vehicles[idx], "crash_sidewalk", False)
            ) else "off_road"
            self._despawn_aux_at(idx, reason)

    def before_step(self):
        self._try_spawn_pending_convoys()
        self._maybe_advance_ring_routes()
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
        self._despawn_offroad_auxiliaries()
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
            return {
                "exists": False,
                "count": 0,
                "agents": [],
                "despawned_offroad": self._despawned_offroad_count,
            }

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
            "pending_convoy_spawns": len(self._pending_convoy_spawns),
            "lanes_occupied": len(set(self._spawn_lane_indices)),
            "policy": self._policy,
            "ring_loop": len(self._ring_loop_lanes) >= 2,
            "ring_loop_lanes": list(self._ring_loop_lanes),
            "despawned_offroad": self._despawned_offroad_count,
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


def filter_lane_keys_in_road_network(road_network, lane_keys: List[str]) -> List[str]:
    """Keep lane keys that exist in the MetaDrive road network."""
    return [key for key in lane_keys if key in road_network.graph]


def select_spawnable_lanes(
    road_network,
    lane_keys: List[str],
    n_lanes_occupied: int = 1,
    *,
    prefer_lane_key: Optional[str] = None,
    min_length: float = MIN_SPAWN_LONGITUDE_M,
) -> List[str]:
    """Pick longest sim lanes that are long enough to place at least one aux vehicle."""
    viable: List[tuple[float, str]] = []
    for lane_key in lane_keys:
        if lane_key not in road_network.graph:
            continue
        try:
            length = float(road_network.get_lane(lane_key).length)
        except Exception:
            continue
        if length >= float(min_length):
            viable.append((length, lane_key))
    viable.sort(key=lambda item: (-item[0], item[1]))
    ordered = [lane_key for _, lane_key in viable]
    if prefer_lane_key and prefer_lane_key in ordered:
        ordered.remove(prefer_lane_key)
        ordered.insert(0, prefer_lane_key)
    if not ordered:
        return []
    n = max(1, min(int(n_lanes_occupied), len(ordered)))
    return ordered[:n]


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


def viable_aux_arms(
    junction_layout: Optional[dict],
    aux_distance_from_intersection: float,
    ego_edge_id: Optional[str] = None,
    lane_lengths: Optional[Dict[Tuple[str, int], float]] = None,
) -> List[dict]:
    """Return main-road arms where at least one lane can host aux (with ring extension)."""
    if not junction_layout:
        return []
    from .roundabout_yield_zone import all_entry_conflict_ring_edges, conflict_aux_ring_edge_ids

    lengths = merge_lane_lengths_from_layout(junction_layout, lane_lengths or {})
    if ego_edge_id:
        allowed_edges = set(conflict_aux_ring_edge_ids(junction_layout, ego_edge_id))
    else:
        allowed_edges = set(all_entry_conflict_ring_edges(junction_layout))

    viable: List[dict] = []
    for arm in _layout_arms(junction_layout):
        if arm.get("road_class") != "main":
            continue
        if ego_edge_id and arm.get("edge_id") == ego_edge_id:
            continue
        edge_id = str(arm.get("edge_id", ""))
        if edge_id not in allowed_edges:
            continue
        lane_nums = sorted(
            {lane_num_from_key(str(key)) for key in arm.get("lane_keys", [])}
        ) or [0]
        if any(
            is_aux_lane_viable_with_ring_extension(
                junction_layout,
                edge_id,
                lane_num,
                lengths,
                aux_distance_from_intersection,
                allowed_ring_edges=allowed_edges,
            )
            for lane_num in lane_nums
        ):
            viable.append(arm)
    return viable


def viable_aux_lane_keys(
    junction_layout: Optional[dict],
    aux_distance_from_intersection: float,
    ego_edge_id: Optional[str] = None,
) -> List[str]:
    """Lane keys on main-road arms with enough length for aux spawning."""
    keys: List[str] = []
    for arm in viable_aux_arms(junction_layout, aux_distance_from_intersection, ego_edge_id):
        keys.extend(arm.get("lane_keys", []))
    return sorted(keys)


def has_viable_aux_lanes(
    junction_layout: Optional[dict],
    aux_distance_from_intersection: float,
    lane_lengths: Optional[Dict[Tuple[str, int], float]] = None,
) -> bool:
    """True if any main-road arm can host aux (possibly via upstream ring extension)."""
    return bool(
        viable_aux_arms(
            junction_layout,
            aux_distance_from_intersection,
            lane_lengths=lane_lengths,
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

    ego_edge = lane_edge_id(str(ego_lane_index)) if ego_lane_index else None
    if row.get("road_id"):
        ego_edge = str(row["road_id"])

    junction_layout = row.get("junction_layout")
    viable_keys = viable_aux_lane_keys(junction_layout, aux_distance, ego_edge)
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
    """Straight-through destination lane key for an aux spawn lane."""
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


def resolve_aux_spawn_plan(
    row: dict,
    ego_lane_index: str,
    incoming_lanes: Optional[List[dict]] = None,
    aux_lanes_occupied: int = 1,
    aux_distance_from_intersection: float = DEFAULT_DISTANCE_FROM_INTERSECTION,
) -> tuple[List[str], List[str], dict, dict]:
    """Resolve aux spawn lanes, destinations, alternate fallbacks, and spawn longitudes."""
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
    )
    alternate_spawn_dest_map: dict = {}
    for lane_key in viable_keys:
        dest = resolve_aux_destination_lane_key(junction_layout, lane_key)
        if dest:
            alternate_spawn_dest_map[lane_key] = dest

    destination_lanes: List[str] = []
    spawn_longitudinal_by_lane: dict = {}
    for idx, spawn_lane in enumerate(spawn_lanes):
        dest = alternate_spawn_dest_map.get(spawn_lane)
        if not dest and idx == 0 and row.get("aux_destination_lane_id"):
            manifest_spawn = row.get("aux_spawn_lane_index")
            if manifest_spawn and spawn_lane == str(manifest_spawn):
                dest = str(row["aux_destination_lane_id"])
        destination_lanes.append(dest or "")
        manifest_long = row.get("aux_spawn_longitudinal")
        if manifest_long is not None and spawn_lane == str(row.get("aux_spawn_lane_index") or ""):
            spawn_longitudinal_by_lane[str(spawn_lane)] = float(manifest_long)

    return spawn_lanes, destination_lanes, alternate_spawn_dest_map, spawn_longitudinal_by_lane


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
    ring_loop_lanes: Optional[List[str]] = None,
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
        ring_loop_lanes=ring_loop_lanes,
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
