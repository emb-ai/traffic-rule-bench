"""
Solid line crossing checks for MetaDrive.

Tracks the vehicle trajectory and tests whether it crosses solid lane markings.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque
from shapely.geometry import LineString, Point
from metadrive.constants import PGLineType, MetaDriveType
from metadrive.component.lane.abs_lane import AbstractLane


class SolidLineDetector:
    """
    Detect trajectory intersections with solid lane markings.

    Tracks vehicle positions and tests against solid segments on lane boundaries.
    """

    def __init__(self, trajectory_history_size: int = 1000):
        """
        Args:
            trajectory_history_size: Max stored trajectory samples
        """
        self.trajectory_history = deque(maxlen=trajectory_history_size)
        self.intersections = []
        self.current_lane = None
        self.previous_lane = None

    def add_position(self, position, lane: Optional[AbstractLane] = None):
        """
        Append a vehicle sample to the trajectory history.

        Args:
            position: (x, y) as ndarray, Vector, list, or tuple
            lane: Current lane, if known
        """
        # Normalize to numpy
        if isinstance(position, np.ndarray):
            pos_array = position.copy()
        elif hasattr(position, '__iter__'):
            # Vector, list, tuple, etc.
            pos_array = np.array([float(position[0]), float(position[1])])
            if len(position) > 2:
                pos_array = np.array([float(position[0]), float(position[1]), float(position[2])])
        else:
            raise ValueError(f"Unsupported position type: {type(position)}")

        self.trajectory_history.append((pos_array, lane))
        self.previous_lane = self.current_lane
        self.current_lane = lane

    def get_lane_boundaries(self, lane: AbstractLane, sample_interval: float = 1.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Left and right lane boundary polylines.

        Args:
            lane: Lane to sample
            sample_interval: Longitudinal sampling step (meters)

        Returns:
            (left_boundary, right_boundary) point lists
        """
        if lane is None:
            return [], []

        left_boundary = []
        right_boundary = []

        # Sample along lane length
        for s in np.arange(0, lane.length, sample_interval):
            width = lane.width_at(s)
            # Left edge in travel direction
            left_pos = lane.position(s, lateral=width / 2)
            left_boundary.append(left_pos[:2])
            # Right edge
            right_pos = lane.position(s, lateral=-width / 2)
            right_boundary.append(right_pos[:2])

        # Include end point
        if lane.length > 0:
            width = lane.width_at(lane.length)
            left_pos = lane.position(lane.length, lateral=width / 2)
            left_boundary.append(left_pos[:2])
            right_pos = lane.position(lane.length, lateral=-width / 2)
            right_boundary.append(right_pos[:2])

        return left_boundary, right_boundary

    def get_solid_line_segments(self, lane: AbstractLane, sample_interval: float = 1.0) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Solid-line segments on lane edges as polylines.

        Args:
            lane: Lane to inspect
            sample_interval: Longitudinal sampling step (meters)

        Returns:
            List of (start, end, side) with side 'left' or 'right'
        """
        if lane is None:
            return []

        solid_segments = []

        # Boundary line types when available
        if hasattr(lane, 'line_types'):
            line_types = lane.line_types
        else:
            # Default: treat both edges as potentially solid
            line_types = (PGLineType.CONTINUOUS, PGLineType.CONTINUOUS)

        left_boundary, right_boundary = self.get_lane_boundaries(lane, sample_interval)

        # Left edge
        if len(left_boundary) > 1 and self.is_solid_line(line_types[0]):
            for i in range(len(left_boundary) - 1):
                solid_segments.append((
                    np.array(left_boundary[i]),
                    np.array(left_boundary[i + 1]),
                    'left'
                ))

        # Right edge
        if len(right_boundary) > 1 and self.is_solid_line(line_types[1]):
            for i in range(len(right_boundary) - 1):
                solid_segments.append((
                    np.array(right_boundary[i]),
                    np.array(right_boundary[i + 1]),
                    'right'
                ))

        return solid_segments

    def is_solid_line(self, line_type) -> bool:
        """
        Whether a boundary type counts as solid for this detector.
        """
        if line_type is None:
            return False

        solid_types = [
            PGLineType.CONTINUOUS,
            MetaDriveType.LINE_SOLID_SINGLE_WHITE,
            MetaDriveType.LINE_SOLID_DOUBLE_WHITE,
            MetaDriveType.LINE_SOLID_SINGLE_YELLOW,
            MetaDriveType.LINE_SOLID_DOUBLE_YELLOW,
            PGLineType.SIDE,  # Road edge treated as solid
        ]

        return line_type in solid_types or MetaDriveType.is_solid_line(line_type)

    def check_intersection(self,
                          trajectory_segment: Tuple[np.ndarray, np.ndarray],
                          solid_line_segment: Tuple[np.ndarray, np.ndarray]) -> bool:
        """
        Whether a trajectory segment intersects a solid segment.
        """
        try:
            traj_start, traj_end = trajectory_segment
            line_start, line_end, _ = solid_line_segment

            traj_line = LineString([(traj_start[0], traj_start[1]),
                                    (traj_end[0], traj_end[1])])
            line_line = LineString([(line_start[0], line_start[1]),
                                   (line_end[0], line_end[1])])

            return traj_line.intersects(line_line)
        except Exception as e:
            return False

    def detect_crossings(self,
                        current_map=None,
                        all_lanes: Optional[List[AbstractLane]] = None) -> List[Dict]:
        """
        Scan trajectory history for new solid-line crossings.
        """
        if len(self.trajectory_history) < 2:
            return []

        intersections = []

        # Collect candidate lanes
        lanes_to_check = []
        if all_lanes is not None:
            lanes_to_check = all_lanes
        elif current_map is not None and hasattr(current_map, 'road_network'):
            try:
                graph = current_map.road_network.graph
                for start_node, end_node, lanes in graph.edges(data='lanes'):
                    if lanes:
                        lanes_to_check.extend(lanes)
            except:
                pass

        # Always include current and previous lane
        if self.current_lane is not None:
            if not any(lane is self.current_lane or
                      (hasattr(lane, 'index') and hasattr(self.current_lane, 'index') and
                       lane.index == self.current_lane.index)
                      for lane in lanes_to_check):
                lanes_to_check.append(self.current_lane)
        if self.previous_lane is not None:
            if not any(lane is self.previous_lane or
                      (hasattr(lane, 'index') and hasattr(self.previous_lane, 'index') and
                       lane.index == self.previous_lane.index)
                      for lane in lanes_to_check):
                lanes_to_check.append(self.previous_lane)

        for lane in lanes_to_check:
            if lane is None:
                continue

            solid_segments = self.get_solid_line_segments(lane)

            for i in range(len(self.trajectory_history) - 1):
                pos1, _ = self.trajectory_history[i]
                pos2, _ = self.trajectory_history[i + 1]

                trajectory_segment = (pos1[:2], pos2[:2])

                for solid_segment in solid_segments:
                    if self.check_intersection(trajectory_segment, solid_segment):
                        # Deduplicate; ensure pos2 is a numpy 2-vector
                        pos2_array = np.array(pos2[:2]) if not isinstance(pos2, np.ndarray) else pos2[:2].copy()
                        intersection_info = {
                            'position': pos2_array,
                            'lane_id': lane.index if hasattr(lane, 'index') else None,
                            'side': solid_segment[2],
                            'step': i + 1,
                            'timestamp': len(self.trajectory_history) - (len(self.trajectory_history) - i - 1)
                        }

                        is_duplicate = False
                        for existing in intersections:
                            if (np.linalg.norm(existing['position'] - intersection_info['position']) < 2.0 and
                                existing['lane_id'] == intersection_info['lane_id'] and
                                existing['side'] == intersection_info['side']):
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            intersections.append(intersection_info)

        self.intersections.extend(intersections)

        return intersections

    def get_all_intersections(self) -> List[Dict]:
        """All recorded intersection events (copy)."""
        return self.intersections.copy()

    def reset(self):
        """Clear trajectory and recorded crossings."""
        self.trajectory_history.clear()
        self.intersections = []
        self.current_lane = None
        self.previous_lane = None

    def get_trajectory(self) -> np.ndarray:
        """Stacked (N, 2) positions from history."""
        if len(self.trajectory_history) == 0:
            return np.array([]).reshape(0, 2)

        positions = [pos[:2] for pos, _ in self.trajectory_history]
        return np.array(positions)
