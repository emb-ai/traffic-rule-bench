from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from metadrive.component.map.base_map import BaseMap
from metadrive.component.pgblock.pg_block import PGBlock


class RoadPriority(Enum):
    """Priority level for a road approach at an intersection."""
    MAIN = "main"                   
    SECONDARY = "secondary"         
    EQUAL = "equal"                


@dataclass
class ApproachInfo:
    """Information about a single approach to an intersection."""
    road_start_node: str
    road_end_node: str
    lanes: list
    priority: RoadPriority
    heading: float = 0.0
    
    @property
    def road_id(self) -> str:
        return f"{self.road_start_node}->{self.road_end_node}"


@dataclass  
class IntersectionPriority:
    """Priority information for a single intersection."""
    block_name: str
    block_type: str  # "X", "T", or "O"
    approaches: list[ApproachInfo] = field(default_factory=list)
    center_position: tuple[float, float] | None = None
    sign_style: str = "main_road"  # "main_road" (2.1+2.4), "secondary_road" (2.3.x+2.4), "equal" (no signs)
    parallel_pairs: list[tuple[ApproachInfo, ApproachInfo]] = field(default_factory=list)
    
    def get_main_approaches(self) -> list[ApproachInfo]:
        return [a for a in self.approaches if a.priority == RoadPriority.MAIN]
    
    def get_secondary_approaches(self) -> list[ApproachInfo]:
        return [a for a in self.approaches if a.priority == RoadPriority.SECONDARY]
    
    def get_parallel_approach(self, approach: ApproachInfo) -> ApproachInfo | None:
        """Find the approach that is parallel (opposite direction) to the given approach."""
        for a1, a2 in self.parallel_pairs:
            if a1.road_id == approach.road_id:
                return a2
            if a2.road_id == approach.road_id:
                return a1
        return None


@dataclass
class MapPriorityNetwork:
    """Complete priority network for a MetaDrive map."""
    intersections: dict[str, IntersectionPriority] = field(default_factory=dict)
    
    def get_lane_priority(self, lane) -> RoadPriority | None:
        """Get the priority for a specific lane at any intersection."""
        lane_index = lane.index
            
        for intersection in self.intersections.values():
            for approach in intersection.approaches:
                for approach_lane in approach.lanes:
                    if approach_lane.index == lane_index:
                        return approach.priority
        return None
    
    def get_intersection_for_lane(self, lane) -> IntersectionPriority | None:
        """Find which intersection (if any) a lane approaches."""
        lane_index = lane.index
        for intersection in self.intersections.values():
            for approach in intersection.approaches:
                for approach_lane in approach.lanes:
                    if approach_lane.index == lane_index:
                        return intersection
        return None


class RoadPriorityAssigner:
    """Assigns road priorities to intersections in a MetaDrive PG map."""
    
    def __init__(self, seed: int | None = None):
        """Initialize the priority assigner."""
        self.rng = random.Random(seed)
        
    def assign_priorities(self, current_map: BaseMap) -> MapPriorityNetwork:
        """Assign priorities to all intersections in the map."""
        priority_network = MapPriorityNetwork()
        x_blocks = [b for b in current_map.blocks if b.ID in ("X", "U")]
        x_styles = self._build_balanced_random_styles(
            len(x_blocks),
            ["equal", "main_road", "secondary_road"],
        )
        x_style_by_name = {
            block.name: style for block, style in zip(x_blocks, x_styles)
        }
        
        for block in current_map.blocks:
            if block.ID == "X" or block.ID == "U":
                intersection = self._assign_x_intersection_priority(
                    block,
                    current_map,
                    style=x_style_by_name.get(block.name),
                )
                priority_network.intersections[block.name] = intersection
                
            elif block.ID == "T":
                intersection = self._assign_t_intersection_priority(block, current_map)
                priority_network.intersections[block.name] = intersection
                
            elif block.ID == "O":
                intersection = self._assign_roundabout_priority(block, current_map)
                priority_network.intersections[block.name] = intersection
                
        return priority_network

    def _build_balanced_random_styles(self, count: int, styles: list[str]) -> list[str]:
        """Build random style list with guaranteed coverage when possible.

        If ``count >= len(styles)``, each style appears at least once.
        Otherwise returns a random subset of unique styles.
        """
        if count <= 0:
            return []
        if count < len(styles):
            picked = list(styles)
            self.rng.shuffle(picked)
            return picked[:count]

        result = list(styles)
        while len(result) < count:
            result.append(self.rng.choice(styles))
        self.rng.shuffle(result)
        return result
    
    def _get_approach_heading(self, lanes: list) -> float:
        """Calculate average heading for an approach's lanes."""
        if not lanes:
            return 0.0
        headings = []
        for lane in lanes:
            try:
                heading = lane.heading_theta_at(lane.length)
                headings.append(heading)
            except Exception:
                pass
        return float(np.mean(headings)) if headings else 0.0
    
    def _get_socket_approaches(
        self, 
        block: PGBlock, 
        road_network
    ) -> list[ApproachInfo]:
        """Extract approach information from block sockets."""
        approaches = []
        
        sockets = block.get_socket_list()
        for socket in sockets:
            if socket.negative_road is None:
                continue
                
            neg_road = socket.negative_road
            start_node = neg_road.start_node
            end_node = neg_road.end_node
            
            try:
                lanes = road_network.graph.get(start_node, {}).get(end_node, [])
            except Exception:
                lanes = []
            
            if lanes:
                heading = self._get_approach_heading(lanes)
                approach = ApproachInfo(
                    road_start_node=start_node,
                    road_end_node=end_node,
                    lanes=list(lanes),
                    priority=RoadPriority.EQUAL,
                    heading=heading
                )
                approaches.append(approach)
                
        pre_socket = block.pre_block_socket
        if pre_socket and pre_socket.positive_road:
            pos_road = pre_socket.positive_road
            start_node = pos_road.start_node
            end_node = pos_road.end_node
            
            try:
                lanes = road_network.graph.get(start_node, {}).get(end_node, [])
            except Exception:
                lanes = []
                
            if lanes:
                heading = self._get_approach_heading(lanes)
                approach = ApproachInfo(
                    road_start_node=start_node,
                    road_end_node=end_node,
                    lanes=list(lanes),
                    priority=RoadPriority.EQUAL,
                    heading=heading
                )
                approaches.append(approach)
                
        return approaches
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)
    
    def _are_opposite_directions(
        self, 
        heading1: float, 
        heading2: float, 
        tolerance: float = 0.5
    ) -> bool:
        """Check if two headings are roughly opposite (180 degrees apart)."""
        diff = abs(self._normalize_angle(heading1 - heading2))
        return abs(diff - np.pi) < tolerance
    
    def _are_perpendicular(
        self, 
        heading1: float, 
        heading2: float, 
        tolerance: float = 0.5
    ) -> bool:
        """Check if two headings are roughly perpendicular (90 degrees apart)."""
        diff = abs(self._normalize_angle(heading1 - heading2))
        return abs(diff - np.pi/2) < tolerance or abs(diff - 3*np.pi/2) < tolerance

    def _assign_x_intersection_priority(
        self, 
        block: PGBlock, 
        current_map: BaseMap,
        style: str | None = None,
    ) -> IntersectionPriority:
        """Assign priorities for X (4-way) intersection.

        Randomly picks one of three configurations:
        - "equal"          – all approaches equivalent, no priority signs
        - "main_road"      – signs 2.1 + 2.4
        - "secondary_road" – signs 2.3.1 + 2.4
        """
        road_network = current_map.road_network
        approaches = self._get_socket_approaches(block, road_network)
        
        assert len(approaches) == 4, f"X intersection {block.name} should have 4 approaches"

        if style is None:
            style = self.rng.choice(["equal", "main_road", "secondary_road"])

        # Find parallel pairs (opposite directions - roads going straight through)
        parallel_pairs = self._find_opposite_pairs(approaches)

        intersection = IntersectionPriority(
            block_name=block.name,
            block_type="X",
            approaches=approaches,
            sign_style=style,
            parallel_pairs=parallel_pairs,
        )

        if style == "equal":
            for approach in approaches:
                approach.priority = RoadPriority.EQUAL
        else:
            main_pair_idx = self.rng.randint(0, 1)
            main_pair, secondary_pair = parallel_pairs[main_pair_idx], parallel_pairs[1 - main_pair_idx]
            for approach in main_pair:
                approach.priority = RoadPriority.MAIN
            for approach in secondary_pair:
                approach.priority = RoadPriority.SECONDARY

        return intersection
    
    def _find_opposite_pairs(
        self, 
        approaches: list[ApproachInfo]
    ) -> list[tuple[ApproachInfo, ApproachInfo]]:
        """Find pairs of approaches with opposite headings."""
        pairs = []
        used = set()
        
        for i, a1 in enumerate(approaches):
            if i in used:
                continue
            for j, a2 in enumerate(approaches):
                if j <= i or j in used:
                    continue
                if self._are_opposite_directions(a1.heading, a2.heading):
                    pairs.append((a1, a2))
                    used.add(i)
                    used.add(j)
                    break
                    
        return pairs

    def _assign_t_intersection_priority(
        self, 
        block: PGBlock, 
        current_map: BaseMap
    ) -> IntersectionPriority:
        """Assign priorities for T-intersection.

        Randomly picks one of three configurations:
        - "equal"          – all approaches equivalent, no priority signs
        - "main_road"      – signs 2.1 + 2.4
        - "secondary_road" – signs 2.3.2/2.3.3 + 2.4
        """
        road_network = current_map.road_network
        approaches = self._get_socket_approaches(block, road_network)
        
        assert len(approaches) == 3, f"T intersection {block.name} should have 3 approaches"

        through_pair = None
        dead_end = None
        
        for i, a1 in enumerate(approaches):
            for j, a2 in enumerate(approaches):
                if j <= i:
                    continue
                if self._are_opposite_directions(a1.heading, a2.heading):
                    through_pair = (a1, a2)
                    remaining_indices = [k for k in range(len(approaches)) if k != i and k != j]
                    if remaining_indices:
                        dead_end = approaches[remaining_indices[0]]
                    break
            if through_pair:
                break

        # Store parallel pairs (the through pair for T intersections)
        parallel_pairs = [through_pair] if through_pair else []

        if not through_pair:
            intersection = IntersectionPriority(
                block_name=block.name,
                block_type="T",
                approaches=approaches,
                sign_style="equal",
                parallel_pairs=[],
            )
            for approach in approaches:
                approach.priority = RoadPriority.EQUAL
            return intersection

        style = self.rng.choice(["equal", "main_road", "secondary_road"])

        intersection = IntersectionPriority(
            block_name=block.name,
            block_type="T",
            approaches=approaches,
            sign_style=style,
            parallel_pairs=parallel_pairs,
        )

        if style == "equal":
            for approach in approaches:
                approach.priority = RoadPriority.EQUAL
        else:
            for approach in through_pair:
                approach.priority = RoadPriority.MAIN
            if dead_end:
                dead_end.priority = RoadPriority.SECONDARY
                
        return intersection

    def _assign_roundabout_priority(
        self, 
        block: PGBlock, 
        current_map: BaseMap
    ) -> IntersectionPriority:
        """Assign priorities for roundabout."""
        road_network = current_map.road_network
        approaches = self._get_socket_approaches(block, road_network)
        
        for approach in approaches:
            approach.priority = RoadPriority.SECONDARY
        
        # Find canonical 4 circulating lanes of the roundabout.
        internal_lanes = []
        for start_node, connections in road_network.graph.items():
            for end_node, lanes in connections.items():
                if block.name not in start_node or block.name not in end_node:
                    continue
                if start_node.startswith("-") or end_node.startswith("-"):
                    continue
                try:
                    start_road_idx = int(start_node.split("_")[-2])
                    end_road_idx = int(end_node.split("_")[-2])
                except Exception:
                    continue

                if start_road_idx == 0 and end_road_idx == 1:
                    internal_lanes.extend(lanes)

        if not internal_lanes:
            for start_node, connections in road_network.graph.items():
                for end_node, lanes in connections.items():
                    if (
                        block.name in start_node
                        and block.name in end_node
                        and not start_node.startswith("-")
                        and not end_node.startswith("-")
                    ):
                        internal_lanes.extend(lanes)
        
        if internal_lanes:
            heading = self._get_approach_heading(internal_lanes)
            internal_approach = ApproachInfo(
                road_start_node=f"{block.name}_internal",
                road_end_node=f"{block.name}_internal",
                lanes=internal_lanes,
                priority=RoadPriority.MAIN,
                heading=heading
            )
            approaches.append(internal_approach)
        
        intersection = IntersectionPriority(
            block_name=block.name,
            block_type="O",
            approaches=approaches
        )
        return intersection


def build_priority_network(
    current_map: BaseMap, 
    seed: int | None = None
) -> MapPriorityNetwork:
    """Convenience function to build priority network for a map."""
    assigner = RoadPriorityAssigner(seed=seed)
    return assigner.assign_priorities(current_map)



class PrioritySignPlacer:
    """Places priority signs based on a MapPriorityNetwork.

    Supported sign sets per intersection ``sign_style``:
    - ``"equal"``          – no signs (equivalent roads)
    - ``"main_road"``      – 2.1 (MainRoadSign) + 2.4 (YieldSign)
    - ``"secondary_road"`` – 2.3.x (SecondaryRoad*Sign) + 2.4 (YieldSign)

    Additionally places 2.2 (EndMainRoadSign) when a road loses its main
    priority between two consecutive intersections.
    """
    
    SIGN_DISTANCE_BEFORE_INTERSECTION = 0
    ADVANCE_SIGN_DISTANCE = 30 
    END_MAIN_SIGN_AFTER_INTERSECTION = 30

    def __init__(self, sign_manager, priority_network: MapPriorityNetwork):
        self.sign_manager = sign_manager
        self.priority_network = priority_network
        self.placed_signs: list = []
    
    def place_all_signs(self) -> list:
        """Place all priority signs according to the network."""
        from traffic_signs.priority_signs import (
            MainRoadSign, YieldSign,
            SecondaryRoadSign, SecondaryRoadLeftSign, SecondaryRoadRightSign,
        )
        
        self.placed_signs = []
        road_network = self.sign_manager.engine.current_map.road_network
        
        for intersection_name, intersection in self.priority_network.intersections.items():
            if intersection.sign_style == "equal":
                continue

            for approach in intersection.approaches:
                if "_internal" in approach.road_start_node:
                    continue
                    
                lane = sorted(approach.lanes, key=lambda ln: ln.index[2])[-1]
                
                if approach.priority == RoadPriority.MAIN:
                    if intersection.sign_style == "secondary_road":
                        sign_cls = self._pick_secondary_road_sign_class(
                            intersection, approach,
                            SecondaryRoadSign, SecondaryRoadLeftSign, SecondaryRoadRightSign,
                        )
                        adv_lane, adv_offset = self._find_lane_for_advance_sign(
                            approach, self.ADVANCE_SIGN_DISTANCE, road_network,
                        )
                        sign = self._place_sign(
                            sign_cls,
                            lane=adv_lane,
                            longitudinal_offset=adv_offset,
                            intersection_name=intersection_name,
                        )
                    else:
                        sign_position = self._calculate_sign_position(lane)
                        sign = self._place_sign(
                            MainRoadSign,
                            lane=lane,
                            longitudinal_offset=sign_position,
                            intersection_name=intersection_name,
                        )

                elif approach.priority == RoadPriority.SECONDARY:
                    sign_position = self._calculate_sign_position(lane)
                    relevant_main_lanes = self._get_relevant_main_lanes_for_approach(
                        intersection, approach,
                    )
                    sign = self._place_sign(
                        YieldSign,
                        lane=lane,
                        longitudinal_offset=sign_position,
                        intersection_name=intersection_name,
                        main_road_lanes=relevant_main_lanes,
                    )
                else:
                    sign = None

                if sign:
                    self.placed_signs.append(sign)

        self._place_end_main_road_signs(road_network)
        return self.placed_signs

    @staticmethod
    def _pick_secondary_road_sign_class(
        intersection: IntersectionPriority,
        main_approach: ApproachInfo,
        cls_x, cls_left, cls_right,
    ):
        """Choose the correct SecondaryRoad*Sign class for *main_approach*.

        - X intersection  → 2.3.1 (cls_x)
        - T intersection  → 2.3.2 or 2.3.3 depending on which side the dead-end
          approach is relative to the main approach's travel direction.
        """
        if intersection.block_type != "T":
            return cls_x

        dead_end = next(
            (a for a in intersection.approaches if a.priority == RoadPriority.SECONDARY),
            None,
        )
        if dead_end is None:
            return cls_x

        angle_diff = main_approach.heading - dead_end.heading
        return cls_left if np.sin(angle_diff) > 0 else cls_right

    def _find_lane_for_advance_sign(
        self,
        approach: ApproachInfo,
        target_distance: float,
        road_network,
    ) -> tuple:
        """Return ``(lane, longitudinal_offset)`` for a sign placed
        *target_distance* metres before the intersection, walking backward
        through consecutive road segments if necessary.
        """
        lane = sorted(approach.lanes, key=lambda ln: ln.index[2])[-1]
        remaining = target_distance

        if lane.length >= remaining:
            return lane, -remaining

        remaining -= lane.length

        reverse_map: dict[str, list] = {}
        for start_node, connections in road_network.graph.items():
            if not isinstance(connections, dict):
                continue
            for end_node, lanes_list in connections.items():
                reverse_map.setdefault(end_node, []).append((start_node, lanes_list))

        current_node = approach.road_start_node
        visited = {approach.road_end_node}
        best_lane = lane

        while remaining > 0 and current_node not in visited:
            visited.add(current_node)
            predecessors = reverse_map.get(current_node, [])
            if len(predecessors) != 1:
                break
            prev_start, prev_lanes = predecessors[0]
            if not prev_lanes:
                break
            best_lane = sorted(prev_lanes, key=lambda ln: ln.index[2])[-1]
            if best_lane.length >= remaining:
                return best_lane, -remaining
            remaining -= best_lane.length
            current_node = prev_start

        return best_lane, -(best_lane.length - 0.1)

    def _place_end_main_road_signs(self, road_network) -> None:
        """Place sign 2.2 wherever a road *loses* its main-road status."""
        from traffic_signs.priority_signs import EndMainRoadSign

        approach_by_end_node: dict[str, list[tuple[str, ApproachInfo]]] = {}
        approach_by_start_node: dict[str, list[tuple[str, ApproachInfo]]] = {}
        
        for iname, inter in self.priority_network.intersections.items():
            for appr in inter.approaches:
                if "_internal" in appr.road_start_node:
                    continue
                approach_by_end_node.setdefault(appr.road_end_node, []).append((iname, appr))
                approach_by_start_node.setdefault('-' + appr.road_start_node, []).append((iname, appr))

        reverse_map: dict[str, list] = {}
        for start_node, connections in road_network.graph.items():
            if not isinstance(connections, dict):
                continue
            for end_node, lanes_list in connections.items():
                reverse_map.setdefault(end_node, []).append((start_node, lanes_list))

        # TODO: inconsistent road priority assignments 16X2_1 17S0_0 -> -16X0_1 -16X0_0
        for iname, intersection in self.priority_network.intersections.items():
            for approach in intersection.approaches:
                if "_internal" in approach.road_start_node:
                    continue

                road_is_main_here = (
                    intersection.sign_style != "equal"
                    and approach.priority == RoadPriority.MAIN
                )
                if road_is_main_here:
                    continue

                upstream_iname, found_approach, found_by_start = self._trace_to_upstream_intersection(
                    approach, iname, reverse_map, approach_by_end_node, approach_by_start_node,
                )
                if upstream_iname is None:
                    continue

                upstream_inter = self.priority_network.intersections.get(upstream_iname)
                if upstream_inter is None:
                    continue

                if found_by_start:
                    approach_to_check = found_approach
                else:
                    approach_to_check = upstream_inter.get_parallel_approach(found_approach)
                if approach_to_check is None:
                    continue

                road_was_main = (
                    upstream_inter.sign_style != "equal"
                    and approach_to_check.priority == RoadPriority.MAIN
                )
                if not road_was_main:
                    continue

                lane = sorted(approach.lanes, key=lambda ln: ln.index[2])[-1]
                end_sign = self._place_sign(
                    EndMainRoadSign,
                    lane=lane,
                    longitudinal_offset=-self.END_MAIN_SIGN_AFTER_INTERSECTION,
                    intersection_name=iname,
                )
                if end_sign:
                    self.placed_signs.append(end_sign)

    def _trace_to_upstream_intersection(
        self,
        approach: ApproachInfo,
        own_intersection: str,
        reverse_map: dict[str, list],
        approach_by_end_node: dict[str, list[tuple[str, ApproachInfo]]],
        approach_by_start_node: dict[str, list[tuple[str, ApproachInfo]]],
        n_max_steps: int = 50,
    ) -> tuple[str | None, ApproachInfo | None, bool]:
        """Walk backward through road nodes until hitting an intersection."""
        current_node = approach.road_start_node
        current_lanes = list(approach.lanes)
        visited: set[str] = set()

        for _ in range(n_max_steps):
            if current_node in visited:
                break
            visited.add(current_node)

            if current_node in approach_by_end_node:
                for iname, ending_approach in approach_by_end_node[current_node]:
                    if iname != own_intersection:
                        return iname, ending_approach, False

            if current_node in approach_by_start_node:
                for iname, found_approach in approach_by_start_node[current_node]:
                    if iname != own_intersection:
                        return iname, found_approach, True

            predecessors = reverse_map.get(current_node, [])
            if not predecessors:
                break
            chosen_predecessor = self._select_continuous_predecessor(
                predecessors,
                current_lanes,
            )
            if chosen_predecessor is None:
                break

            prev_start, prev_lanes = chosen_predecessor
            current_node = prev_start
            current_lanes = list(prev_lanes)

        return None, None, False

    def _select_continuous_predecessor(self, predecessors: list, current_lanes: list):
        """Pick predecessor segment that best connects to current segment."""
        if not current_lanes:
            return predecessors[0] if predecessors else None

        ref_lane = current_lanes[0]
        try:
            ref_start_pos = np.array(ref_lane.position(0.0, 0.0))
        except Exception:
            return predecessors[0] if predecessors else None

        best = None
        best_dist = float("inf")
        for candidate in predecessors:
            _, lanes = candidate
            if not lanes:
                continue
            lane0 = lanes[0]
            try:
                cand_end_pos = np.array(lane0.position(lane0.length, 0.0))
            except Exception:
                continue
            dist = float(np.linalg.norm(cand_end_pos - ref_start_pos))
            if dist < best_dist:
                best_dist = dist
                best = candidate

        if best is not None and best_dist < 8.0:
            return best
        return None

    def _collect_main_road_lanes(self, intersection: IntersectionPriority) -> list:
        """Collect all lanes from main road approaches for an intersection."""
        main_lanes = []
        for approach in intersection.approaches:
            if approach.priority == RoadPriority.MAIN:
                main_lanes.extend(approach.lanes)
        return main_lanes

    def _get_relevant_main_lanes_for_approach(
        self,
        intersection: IntersectionPriority,
        secondary_approach: ApproachInfo,
    ) -> list:
        """Return main-road lanes that are relevant to one secondary approach."""
        main_lanes = self._collect_main_road_lanes(intersection)
        if not main_lanes:
            return []

        if intersection.block_type != "O":
            return main_lanes

        road_network = self.sign_manager.engine.current_map.road_network
        approach_end = secondary_approach.road_end_node
        next_nodes = list(road_network.graph.get(approach_end, {}).keys())
        if not next_nodes:
            return main_lanes

        merge_node = next_nodes[0]

        lane_by_segment: dict[tuple, list] = {}
        for lane in main_lanes:
            seg = (lane.index[0], lane.index[1])
            lane_by_segment.setdefault(seg, []).append(lane)
            
        incoming_to_merge = None
        for seg_start, to_dict in road_network.graph.items():
            if merge_node not in to_dict or seg_start == approach_end:
                continue
            if seg_start.startswith("-") or merge_node.startswith("-"):
                continue
            if "O" in seg_start and "O" in merge_node:
                incoming_to_merge = (seg_start, merge_node)
                break

        if incoming_to_merge is None:
            return main_lanes

        upstream_node = incoming_to_merge[0]
        predecessor_seg = None
        for seg_start, to_dict in road_network.graph.items():
            if upstream_node not in to_dict:
                continue
            seg = (seg_start, upstream_node)
            if seg in lane_by_segment:
                predecessor_seg = seg
                break

        if predecessor_seg is None and incoming_to_merge in lane_by_segment:
            predecessor_seg = incoming_to_merge

        if predecessor_seg is None:
            return main_lanes

        return lane_by_segment[predecessor_seg]
    
    def _calculate_sign_position(self, lane) -> float:
        """Signs are placed before the intersection (negative offset from end of lane)."""
        return -min(self.SIGN_DISTANCE_BEFORE_INTERSECTION, lane.length)
    
    def _place_sign(self, sign_class, lane, longitudinal_offset, intersection_name, **kwargs):
        """Place a single sign using the sign manager."""
        sign = self.sign_manager.add_sign(
            sign_class,
            lane=lane,
            use_random_lane=False,
            check_zone_overlap=False,
            longitudinal_offset=longitudinal_offset,
            intersection_name=intersection_name,
            **kwargs
        )
        return sign


def place_priority_signs(sign_manager, priority_network: MapPriorityNetwork) -> list:
    placer = PrioritySignPlacer(sign_manager, priority_network)
    return placer.place_all_signs()

