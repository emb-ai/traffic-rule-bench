def same_road_check(v_idx, s_idx) -> bool:
    """Return True if two lane indices refer to the same road/edge (any
    parallel peer-lane counts as same road). Works for both SUMO string
    indices ("lane_<edge>_<N>") and PGMap tuple indices ((from, to, num)).
    Returns False if either index is None or cannot be compared.

    Defined at module top so it can be imported by traffic-sign submodules
    before the heavier metadrive imports below trigger circular-import
    issues via idm_policy → junction.yield_sign → signs.base.
    """
    if v_idx is None or s_idx is None:
        return False
    if isinstance(v_idx, str) and isinstance(s_idx, str):
        return v_idx.rsplit("_", 1)[0] == s_idx.rsplit("_", 1)[0]
    try:
        return v_idx[0] == s_idx[0] and v_idx[1] == s_idx[1]
    except (IndexError, TypeError):
        return v_idx == s_idx


from abc import ABC, abstractmethod
import numpy as np
from metadrive.component.static_object.traffic_object import TrafficObject
from metadrive.constants import Semantics
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from typing import Optional

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ICONS_DIR = os.path.join(SCRIPT_DIR, "icons")


class BaseTrafficSign(TrafficObject, ABC):

    COLLISION_MASK = TrafficObject.COLLISION_MASK
    SEMANTIC_LABEL = Semantics.TRAFFIC_SIGN.label
    
    DEFAULT_ZONE_BEFORE = 10.0  # meters before sign
    DEFAULT_ZONE_AFTER = 5.0    # meters after sign

    HEIGHT = 2.0
    WIDTH = 0.6
    DEPTH = 0.1

    def __init__(
        self,
        lane,
        longitudinal_offset=0.0,
        lateral_offset=None,
        name=None,
        random_seed=None,
        show_model=True,
        icon_path=None,
        longitudinal_from_start=False,
        **kwargs
    ):
        self.lane = lane
        self._show_model = show_model
        self._vehicle_states = {}

        # `longitudinal_offset` convention:
        #   longitudinal_from_start=False (default, legacy): offset measured from
        #     the lane END (target_long = lane.length + offset). Used by Stop,
        #     MinSpeed, restricted-lane and other approach-style signs.
        #   longitudinal_from_start=True: offset IS the distance from lane START
        #     (target_long = offset). Used by speed/zone/end-of-zone signs so a
        #     start sign and its end sign share one coordinate frame.
        if longitudinal_from_start:
            target_long = longitudinal_offset
        else:
            target_long = lane.length + longitudinal_offset
        # self.placement_long = np.clip(target_long, 0.1, lane.length - 0.1)
        self.placement_long = np.maximum(target_long, 0.1)

        if lateral_offset is None:
            lane_width = lane.width_at(self.placement_long)
            shoulder_width = 0.8
            lateral_offset = lane_width / 2 + shoulder_width

        self._lateral_offset = lateral_offset
        position = lane.position(self.placement_long, lateral_offset)
        heading_theta = lane.heading_theta_at(self.placement_long) + np.pi / 2
        self._heading_theta = heading_theta
        self._position = np.array(position)

        # kwargs are ignored
        super().__init__(
            position=position,
            heading_theta=heading_theta,
            lane=lane,
            random_seed=random_seed,
            name=name,
        )
        if icon_path is not None:
            self.icon_path = os.path.join(ICONS_DIR, icon_path)
        else:
            self.icon_path = None

        if self.render and self._show_model:
            self._create_visual_model()

    def _create_visual_model(self):
        pass
    
    @staticmethod
    def _lane_index_parts(lane_idx):
        if lane_idx is None:
            return None, None
        if isinstance(lane_idx, tuple) and len(lane_idx) >= 3:
            return (lane_idx[0], lane_idx[1]), int(lane_idx[2])
        if isinstance(lane_idx, str) and lane_idx.startswith("lane_"):
            core = lane_idx[5:]
            if "_" not in core:
                return None, None
            prefix, last = core.rsplit("_", 1)
            try:
                return prefix, int(last)
            except Exception:
                return prefix, None
        return None, None

    @classmethod
    def _is_pre_junction_lane_change(cls, src_lane_obj, current_lane_id) -> bool:
        if src_lane_obj is None or current_lane_id is None:
            return False
        incoming = set(getattr(src_lane_obj, "incoming_junction_lanes", None) or [])
        if not incoming:
            return False
        if current_lane_id not in incoming:
            return False

        src_key, src_lane_num = cls._lane_index_parts(getattr(src_lane_obj, "index", None))
        cur_key, cur_lane_num = cls._lane_index_parts(current_lane_id)
        if src_key is None or cur_key is None:
            return False
        if src_key != cur_key:
            return False
        if src_lane_num is None or cur_lane_num is None:
            return False
        return abs(src_lane_num - cur_lane_num) == 1
    

    def check_violation(self, vehicle, for_reward=False) -> bool:
        road_network = self.engine.current_map.road_network
        if isinstance(road_network, NodeRoadNetwork):
            if not self.is_in_drivable_area(vehicle):
                return False
        else:
            current_lane = vehicle.lane_index
            if "lane_:" in current_lane or "junction_" in current_lane:
                return False

        vid = vehicle.id
        key = "reported_for_reward" if for_reward else "reported_for_metrics"
        state = self._vehicle_states.setdefault(
            vid,
            {
                "reported_for_reward": False,
                "reported_for_metrics": False,
                "pending_violation": False,
            },
        )

        currently_violating = self._is_violating(vehicle)

        if not currently_violating and not state.get("pending_violation", False):
            # edge-triggered: re-arm once the violation condition clears
            state["reported_for_reward"] = False
            state["reported_for_metrics"] = False
            return False

        if state[key]:
            return False

        if state.get("pending_violation", False):
            state[key] = True
            if state["reported_for_reward"] and state["reported_for_metrics"]:
                state["pending_violation"] = False
            return True

        if not currently_violating:
            return False

        state["pending_violation"] = True
        state[key] = True
        if state["reported_for_reward"] and state["reported_for_metrics"]:
            state["pending_violation"] = False
        return True

    @abstractmethod
    def _is_violating(self, vehicle) -> bool:
        raise NotImplementedError


    @abstractmethod
    def get_rule_description(self) -> str:
        raise NotImplementedError


    @property
    def top_down_length(self):
        return 1

    @property
    def top_down_width(self):
        return 1
    
    @property
    def position(self):
        """Safe position that handles empty origin NodePath."""
        try:
            if hasattr(self, "origin") and not self.origin.isEmpty():
                return super().position
        except Exception:
            pass
        return self._fallback_position()
    
    @property
    def heading_theta(self):
        """Safe heading that handles empty origin NodePath."""
        try:
            if hasattr(self, "origin") and not self.origin.isEmpty():
                return super().heading_theta
        except Exception:
            pass
        return self._fallback_heading()

    def _fallback_position(self):
        if hasattr(self, "_position"):
            return self._position
        lane = getattr(self, "lane", None)
        if lane is None:
            return np.array([0.0, 0.0])
        try:
            lateral_offset = getattr(self, "_lateral_offset", lane.width_at(self.placement_long) / 2 + 0.8)
            return np.array(lane.position(self.placement_long, lateral_offset))
        except Exception:
            return np.array([0.0, 0.0])

    def _fallback_heading(self):
        if hasattr(self, "_heading_theta"):
            return self._heading_theta
        lane = getattr(self, "lane", None)
        if lane is None:
            return 0.0
        try:
            return lane.heading_theta_at(self.placement_long) + np.pi / 2
        except Exception:
            return 0.0

    @staticmethod
    def _wrap_pi(x: float) -> float:
        return float((x + np.pi) % (2 * np.pi) - np.pi)

    @staticmethod
    def _get_lane_index(obj):
        """
        Best-effort lane index fetch for lanes, agents, or other objects.
        """
        if obj is None:
            return None
        lane = getattr(obj, "lane", None)
        if lane is not None:
            idx = getattr(lane, "index", None)
            if idx is not None:
                return idx
        idx = getattr(obj, "lane_index", None)
        if idx is not None:
            return idx
        idx = getattr(obj, "index", None)
        if idx is not None:
            return idx
        return None

    @staticmethod
    def _same_road_direction(idx_a, idx_b):
        """
        Check if two lane indices are on the same road in the same direction.

        Supported index formats:
          * NodeRoadNetwork (PG maps): tuple ``(from_node, to_node, lane_num)``
            → same road iff ``from_node`` and ``to_node`` match.
          * EdgeRoadNetwork (SUMO / Scenario maps): string
            ``"<edge_id>_<lane_num>"`` where ``<edge_id>`` itself may contain
            underscores. We strip the LAST ``_<digits>`` component to recover
            the edge id, then compare edge ids.
        """
        if idx_a is None or idx_b is None:
            return False
        try:
            if len(idx_a) < 2 or len(idx_b) < 2:
                return False
        except Exception:
            return False
        if isinstance(idx_a, str) and isinstance(idx_b, str):
            def edge_id(s: str) -> str:
                # Drop trailing "_<lanenum>" so "gneE5_2" -> "gneE5" and
                # ":J1_0_0" -> ":J1_0" (internal SUMO edge ids).
                head, sep, tail = s.rpartition("_")
                if sep and tail.isdigit():
                    return head
                return s
            return edge_id(idx_a) == edge_id(idx_b)
        return idx_a[0] == idx_b[0] and idx_a[1] == idx_b[1]

    @staticmethod
    def _node_sign(node_name) -> str:
        node_name = str(node_name)
        return "-" if node_name.startswith("-") else "+"

    def _lane_direction_signature(self, lane_index):
        """
        Signature based on start/end node sign prefixes only.
        Works across split lane pieces like ('>', '>>', i), ('>>', '>>>', i), etc.
        """
        if not lane_index:
            return None
        try:
            if len(lane_index) < 2:
                return None
        except TypeError:
            return None
        start_node, end_node = lane_index[0], lane_index[1]
        return self._node_sign(start_node), self._node_sign(end_node)

    @staticmethod
    def _is_same_signature(sign_signature, current_signature) -> bool:
        return (
            sign_signature is not None
            and current_signature is not None
            and current_signature == sign_signature
        )

    @staticmethod
    def _is_opposite_signature(sign_signature, current_signature) -> bool:
        if sign_signature is None or current_signature is None:
            return False
        sign_start, sign_end = sign_signature
        opposite_signature = (
            "-" if sign_start == "+" else "+",
            "-" if sign_end == "+" else "+",
        )
        return current_signature == opposite_signature

    def _is_sumo_network(self) -> bool:
        return not isinstance(self.engine.current_map.road_network, NodeRoadNetwork)

    @staticmethod
    def _sumo_edge_id_from_lane_index(lane_index):
        if lane_index is None:
            return None
        lane_str = str(lane_index)
        if lane_str.startswith("lane_"):
            lane_str = lane_str[len("lane_"):]
        parts = lane_str.split("_")
        # Strip trailing lane-number suffixes, e.g.:
        # lane_794172375#0_0 -> 794172375#0
        # lane_:843318379_0_0 -> :843318379
        for _ in range(2):
            if len(parts) > 1 and parts[-1].isdigit():
                parts = parts[:-1]
            else:
                break
        if len(parts) == 0:
            return None
        return "_".join(parts)

    def _sumo_direction_key(self, lane_index):
        edge_id = self._sumo_edge_id_from_lane_index(lane_index)
        if edge_id is None:
            return None
        direction = "-" if edge_id.startswith("-") else "+"
        road_id = edge_id[1:] if edge_id.startswith("-") else edge_id
        return road_id, direction

    def _is_sumo_lane_on_sign_road(self, sign_lane_index, lane_index, allowed_directions="all") -> bool:
        sign_key = self._sumo_direction_key(sign_lane_index)
        lane_key = self._sumo_direction_key(lane_index)
        if sign_key is None or lane_key is None:
            return False
        sign_road, sign_dir = sign_key
        lane_road, lane_dir = lane_key
        if allowed_directions == "all":
            allowed_directions = ("+", "-")
        elif allowed_directions == "same":
            allowed_directions = (sign_dir)
        elif allowed_directions == "opposite":
            allowed_directions = ("-" if sign_dir == "+" else "+")
        else:
            raise ValueError(f"Invalid allowed directions: {allowed_directions}")
        return lane_road == sign_road and lane_dir in allowed_directions

    def _vehicle_edge_and_s(self, vehicle):
        """Return (directed SUMO edge id, longitudinal s) of the vehicle on its
        CURRENT lane. Edge id keeps direction/segment (e.g. '-787071935#2') so it
        can be matched against an ordered multi-edge zone path."""
        edge = self._sumo_edge_id_from_lane_index(getattr(vehicle, "lane_index", None))
        s = None
        lane = getattr(vehicle, "lane", None)
        if lane is not None:
            try:
                s = float(lane.local_coordinates(vehicle.position)[0])
            except Exception:
                s = None
        return edge, s

    def _in_multi_edge_zone(self, vehicle):
        """Membership for a zone spanning several connected edges.

        Uses `self.zone_edges` (ordered list of directed edge ids from the start
        sign's edge to the end sign's edge), `self.zone_start` (offset on the
        first edge) and `self.zone_end_s` (offset on the last edge). Returns
        True/False, or None when this sign has no multi-edge zone configured (so
        the caller falls back to the single-lane check)."""
        edges = getattr(self, "zone_edges", None)
        if not edges:
            return None
        edge, s = self._vehicle_edge_and_s(vehicle)
        if edge is None or edge not in edges or s is None:
            return False
        if edge == edges[0] and s < float(self.zone_start) - self.LONGITUDINAL_TOLERANCE:
            return False
        if edge == edges[-1] and s > float(self.zone_end_s) + self.LONGITUDINAL_TOLERANCE:
            return False
        return True

    def _heading_aligned(self, agent, tolerance_rad: float = 1.1):
        """
        Direction check using heading alignment.
        Returns True/False if headings are comparable, otherwise None.
        """
        try:
            lane_heading = float(self.lane.heading_theta_at(self.placement_long))
        except Exception:
            return None
        agent_heading = getattr(agent, "heading_theta", None)
        if agent_heading is None:
            return None
        diff = abs(self._wrap_pi(lane_heading - float(agent_heading)))
        return diff <= tolerance_rad

    LATERAL_TOLERANCE = 0.5
    LONGITUDINAL_TOLERANCE = 1.0

    def _on_sign_lane_geometrically(self, agent):
        """
        Geometry-based check: is the agent close enough to self.lane
        (lateral distance <= half lane width + tolerance) and within the
        lane's longitudinal extent. Returns (is_on_lane, veh_long).
        """
        if self.lane is None:
            return False, None
        try:
            veh_long, veh_lat = self.lane.local_coordinates(agent.position)
        except Exception:
            return False, None
        try:
            half_width = float(self.lane.width_at(veh_long)) / 2.0
        except Exception:
            half_width = float(getattr(self.lane, "width", 3.5)) / 2.0
        on_lat = abs(float(veh_lat)) <= half_width + self.LATERAL_TOLERANCE
        on_long = (
            -self.LONGITUDINAL_TOLERANCE
            <= float(veh_long)
            <= float(self.lane.length) + self.LONGITUDINAL_TOLERANCE
        )
        return bool(on_lat and on_long), float(veh_long)

    def is_in_drivable_area(self, agent) -> bool:
        """If sign applies to the current agent direction/lane.

        Works with both NodeRoadNetwork (MetaDrive PG) and SUMO/Scenario maps.
        Decision order:
          1. Lane indices resolve on both sides -> same-road/direction match wins.
          2. Otherwise fall back to geometry: agent must be on this sign's lane
             laterally+longitudinally AND heading-aligned (when computable).
        """
        if self.lane is None:
            return False

        sign_lane_index = self._get_lane_index(self.lane)
        if sign_lane_index is not None:
            try:
                if len(sign_lane_index) < 2:
                    sign_lane_index = None
            except TypeError:
                sign_lane_index = None

        agent_lane_index = self._get_lane_index(agent)

        aligned = self._heading_aligned(agent)

        same_dir_current = False
        if sign_lane_index is not None and agent_lane_index is not None:
            same_dir_current = self._same_road_direction(agent_lane_index, sign_lane_index)

        same_dir_ref = False
        has_ref_lanes = False
        navigation = getattr(agent, "navigation", None)
        if agent_lane_index is None and sign_lane_index is not None:
            if navigation is not None and hasattr(navigation, "current_ref_lanes"):
                ref_lanes = navigation.current_ref_lanes or []
                has_ref_lanes = len(ref_lanes) > 0
                for ref_lane in ref_lanes:
                    ref_idx = self._get_lane_index(ref_lane)
                    if self._same_road_direction(ref_idx, sign_lane_index):
                        same_dir_ref = True
                        break

        # Strong directional info from lane indices: trust it.
        has_dir_info = sign_lane_index is not None and (
            agent_lane_index is not None or has_ref_lanes
        )
        if has_dir_info:
            if same_dir_current or same_dir_ref:
                return True
            return False

        # Fallback (typical on SUMO maps): geometry + heading.
        on_sign_lane, _ = self._on_sign_lane_geometrically(agent)
        if not on_sign_lane:
            return False
        if aligned is False:
            return False
        return True
        if aligned is None:
            return True
        return aligned

    def _find_next_intersection_distance(self) -> Optional[float]:
        """
        Find the distance to the next intersection/crossroad along the lane.
        Handles both SUMO edge-based networks (flat graph: key -> edge_lane)
        and MetaDrive PG block-based networks (nested graph: from -> {to -> [lanes]}).
        """
        
        road_network = self.engine.current_map.road_network
        start_node, end_node, matched_entry = self._find_lane_in_graph(road_network)
        
        if not isinstance(road_network, NodeRoadNetwork):
            return self.lane.length
        
        start_node, end_node = None, None
        
        for from_node, to_dict in road_network.graph.items():
            for to_node, lanes in to_dict.items():
                if self.lane in lanes:
                    start_node = from_node
                    end_node = to_node
                    break
            if start_node is not None:
                break
        
        assert start_node is not None and end_node is not None, 'start_node or end_node is None'
        remaining = self._entry_lane_length(matched_entry) - self.placement_long

        # SUMO flat graph: each edge already spans between intersections
        if end_node is None:
            return remaining

        # MetaDrive PG nested graph: walk forward through connected segments
        total_distance = remaining
        visited = {start_node}
        current = end_node
        prev_lanes = road_network.graph.get(start_node, {}).get(end_node, [])

        while current is not None:
            if current in visited:
                break
            if current not in road_network.graph:
                break

            outgoing = road_network.graph[current]
            if not isinstance(outgoing, dict):
                break

            candidates = {k: v for k, v in outgoing.items() if k not in visited}

            if not candidates:
                break

            if len(candidates) == 1:
                next_node = next(iter(candidates))
            else:
                next_node = self._find_spatial_continuation(prev_lanes, candidates)
                if next_node is None:
                    return total_distance

            next_lanes = candidates[next_node]
            if next_lanes:
                total_distance += next_lanes[0].length
                prev_lanes = next_lanes

            visited.add(current)
            current = next_node

        return total_distance

    def _find_lane_in_graph(self, road_network):
        """Locate the sign's lane in the road network graph.
        Returns (key, to_node, matched_entry).
        SUMO flat graph:    returns (lane_key, None, edge_lane).
        MetaDrive nested graph: returns (from_node, to_node, lane_obj)."""
        for key, value in road_network.graph.items():
            # SUMO flat graph: value is an edge_lane with a .lane attribute
            if hasattr(value, 'lane') and not isinstance(value, dict):
                if value.lane is self.lane:
                    return key, None, value
                continue

            # MetaDrive PG nested graph: value is {to_node: [lane_list]}
            if isinstance(value, dict):
                for to_node, lanes in value.items():
                    for entry in lanes:
                        if entry is self.lane:
                            return key, to_node, entry
        return None, None, None

    @staticmethod
    def _entry_lane_length(entry) -> float:
        """Get lane length from either an edge_lane wrapper or a raw lane."""
        if hasattr(entry, 'lane'):
            return entry.lane.length
        return entry.length

    def _find_spatial_continuation(self, prev_lanes, candidates):
        """At a node with multiple outgoing edges, find the unique spatial continuation of the current road."""
        if not prev_lanes:
            return None

        ref = prev_lanes[0]
        end_pos = np.array(ref.position(ref.length, 0))

        threshold = 5.0
        matches = []
        for to_node, lanes in candidates.items():
            if not lanes:
                continue
            start_pos = np.array(lanes[0].position(0, 0))
            if np.linalg.norm(end_pos - start_pos) < threshold:
                matches.append(to_node)

        return matches[0] if len(matches) == 1 else None
