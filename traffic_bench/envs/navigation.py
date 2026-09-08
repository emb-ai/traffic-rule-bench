"""Edge-network navigation whose current lane is decided by geometry.

MetaDrive's ``EdgeNetworkNavigation._get_current_lane`` asks the physics
world which lane polygon lies under the vehicle (``ray_localization``). On
long SUMO lanes (XY crops keep whole edges of 1-3 km) the polygon meshes have
seams and the ray misses the lane the vehicle is centred on for a step or
two, returning the neighbour instead: the ego flickered between lane 1 and
lane 2 while its lateral offset on lane 2 was 0.0 m. Every consumer of
``vehicle.lane`` then misbehaves -- IDM steers towards the wrong lane centre
(a baseline left the reserved lane "by accident"), lane-scoped plates fire on
the wrong lane, lane-change logic re-plans.

Here the reference lanes are tested geometrically first: the lane whose
centreline is closest to the vehicle centre, provided the centre lies inside
that lane's width (plus a small margin), is the current lane. The ray result
is only the fallback when the vehicle is on none of the reference lanes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from metadrive.component.navigation_module.edge_network_navigation import EdgeNetworkNavigation
from metadrive.utils.pg.utils import ray_localization

# The vehicle centre may sit this far outside the lane width and still count
# as on that lane (a lane change counts until the centre crosses the line).
LATERAL_MARGIN_M = 0.5
LONGITUDINAL_MARGIN_M = 1.0


def geometric_lane_pick(lanes, position) -> Optional[Tuple[object, float]]:
    """(lane, |lateral|) of the reference lane the point is geometrically on, or None."""
    best = None
    for lane in lanes or ():
        try:
            lon, lat = lane.local_coordinates(position)
            length = float(lane.length)
            if lon < -LONGITUDINAL_MARGIN_M or lon > length + LONGITUDINAL_MARGIN_M:
                continue
            width = float(lane.width_at(min(max(lon, 0.0), length)))
        except Exception:
            continue
        if abs(float(lat)) > width / 2.0 + LATERAL_MARGIN_M:
            continue
        if best is None or abs(float(lat)) < best[1]:
            best = (lane, abs(float(lat)))
    return best


class GeometricEdgeNavigation(EdgeNetworkNavigation):
    """EdgeNetworkNavigation with a geometry-first current-lane test."""

    def _get_current_lane(self, ego_vehicle):
        position = ego_vehicle.position
        picked = geometric_lane_pick(self.current_ref_lanes, position)
        if picked is None:
            nx_ckpt = self._target_checkpoints_index[-1]
            on_last_road = nx_ckpt == self.checkpoints[-1] or self.next_ref_lanes is None
            if not on_last_road:
                picked = geometric_lane_pick(self.next_ref_lanes, position)
        if picked is not None:
            lane = picked[0]
            return lane, lane.index, True
        # Off every reference lane (junction, shoulder, far off-road): MetaDrive's ray test.
        possible_lanes, on_lane = ray_localization(
            ego_vehicle.heading, position, ego_vehicle.engine, use_heading_filter=False, return_on_lane=True
        )
        for lane, index, _dist in possible_lanes:
            if lane in (self.current_ref_lanes or ()):
                return lane, index, on_lane
        nx_ckpt = self._target_checkpoints_index[-1]
        if nx_ckpt == self.checkpoints[-1] or self.next_ref_lanes is None:
            return (*possible_lanes[0][:-1], on_lane) if len(possible_lanes) > 0 else (None, None, on_lane)
        for lane, index, _dist in possible_lanes:
            if lane in self.next_ref_lanes:
                return lane, index, on_lane
        return (*possible_lanes[0][:-1], on_lane) if len(possible_lanes) > 0 else (None, None, on_lane)
