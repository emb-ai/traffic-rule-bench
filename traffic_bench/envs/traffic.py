"""
Traffic manager for SUMO-based environments.
Spawns TrajectoryIDM-controlled vehicles that follow PointLane trajectories
built from the navigation route — the same approach used in ScenarioNet.
"""
import json
import logging
import math
import os

# Suppress verbose NPC warnings (DRIFT, TRAJ_SHORT, SPAWN REPORT, RESPAWN, REMOVE).
# All logging.warning() calls in this module use the root logger directly;
# redirect them through a module-level logger so we can control the level.
_log = logging.getLogger(__name__)
_log.setLevel(logging.ERROR)

import numpy as np

from metadrive.component.lane.point_lane import PointLane
from metadrive.manager.base_manager import BaseManager
from traffic_bench.envs.npc_idm import SumoTrajectoryIDMPolicy


class SumoTrafficManager(BaseManager):
    # Point guard only: a spawn position this close to the ego is skipped so a
    # car cannot materialise inside it. The 50 m radius this replaces was applied
    # per LANE and blanked the whole road the ego was on, so no NPC was ever
    # within reach: min_ttc_sec came back null in all 319 speed episodes and
    # driving_efficiency had no support at all.
    EGO_KEEP_CLEAR_M = 12.0
    # On the ego's OWN lane the point guard is not enough. Slots are laid every
    # 12 m from the lane start and each car gets a nuPlan spawn speed of up to
    # 80 km/h with no regard to what is ahead of it, so a car placed one slot
    # behind a freshly spawned ego closes the gap in under a second. In the
    # 5-variant pilot 299 of 300 crashes were NPC-attributed and 236 of them
    # happened within 20 m of the spawn, identically for all eight experts --
    # the scene killed the ego before any policy had acted. Behind the ego on
    # its lane nothing spawns within EGO_REAR_KEEP_CLEAR_M, and a car further
    # back within EGO_REAR_SPEED_MATCH_M starts no faster than the ego does;
    # ahead, EGO_FRONT_KEEP_CLEAR_M keeps the ego from being boxed in at once.
    EGO_REAR_KEEP_CLEAR_M = 40.0
    EGO_REAR_SPEED_MATCH_M = 80.0
    # Ahead of the ego on its lane the clear distance scales with the ego's
    # spawn speed: a car 28 m ahead crawling at 3 m/s in front of an ego doing
    # 11.7 m/s is a collision at step 16 whatever the policy does (traced on
    # min_speed seg_1278634838_0). Nothing spawns inside EGO_FRONT_HEADWAY_S
    # seconds of travel, and a leader inside twice that starts no slower than
    # the ego minus EGO_FRONT_SPEED_SLACK_MPS.
    EGO_FRONT_KEEP_CLEAR_M = 20.0
    EGO_FRONT_HEADWAY_S = 3.0
    EGO_FRONT_SPEED_SLACK_MPS = 2.0
    # Speed scenes only (config traffic_spawn_after_*): on the ego's edge no NPC
    # spawns before the plate plus this margin, so nothing stands on the plate.
    TRAFFIC_AFTER_SIGN_MARGIN_M = 5.0
    VEHICLE_MIN_GAP = 12   # metres between any two spawned vehicles
    VEHICLE_GAP_ON_LANE = 12  # metres between vehicles on the same lane
    MAX_PER_LANE = 6       # max vehicles per lane
    IDM_ACT_BATCH_SIZE = 5
    # NPC↔NPC pile-ups: if two traffic cars contact and either has
    # crash_vehicle, remove both so they don't block the skill scene.
    NPC_NPC_CRASH_DIST = 4.0  # metres — approximate contact envelope
    # Stuck-cascade prevention: previously 2.0 km/h + 40 steps (~4 s) was so
    # aggressive that any traffic-light queue or yield negotiation culled
    # the entire surrounding fleet in one tick. Now we only count *truly*
    # stationary vehicles, and give them 20 s of grace before removal.
    STUCK_SPEED_KMH = 0.5   # only truly stopped vehicles count as stuck
    STUCK_TIMEOUT = 60      # steps (~6 s at 10Hz) stuck before removal — faster clearance; initial-spawn still protected by ARRIVE_GRACE_STEPS
    # Stuck counter only increments if the vehicle is ALSO off-track (heading
    # error > threshold or lateral drift > threshold). Vehicles stopped in a
    # legitimate traffic jam (correct heading, on lane center) won't accumulate
    # stuck ticks and won't be removed.
    STUCK_HEADING_ERR_THRESH = 15.0  # degrees — below this, vehicle is "pointing right"
    STUCK_LATERAL_THRESH = 2.0       # metres — below this, vehicle is "on track"

    # Wrong-way / circling detector: NPC is moving but its heading is badly
    # misaligned with its routing PointLane for several consecutive seconds.
    # Covers cases that fall between `not on_lane` and stuck-detection:
    # vehicle is on-surface, moving, but spinning / driving backwards.
    MOVING_MIN_SPEED_KMH = 1.0
    MOVING_HEADING_ERR_THRESH = 60.0   # degrees — > this and moving = wrong-way
    MOVING_MISALIGNED_STEPS = 15       # ~1.5 s at 10 Hz sustained before removal

    def __init__(self):
        super(SumoTrafficManager, self).__init__()
        self._traffic_vehicles = []
        self.density = self.engine.global_config.get("traffic_density", 0.1)
        self._idm_count = 0
        self._stuck_counter = {}  # vehicle_id → steps stuck
        self._wrong_way_counter = {}  # vehicle_id → consecutive misaligned steps
        self._spawn_step = {}     # vehicle_id → episode_step at spawn

        # Cached nuPlan sampler for spawn velocities. Loaded lazily on first use
        # so non-SUMO contexts (and tests without nuplan_statistics) keep working.
        # Spawning NPCs with realistic non-zero initial velocity prevents the
        # "stuck cascade" — when batch-spawned cars at speed=0 in a tight queue
        # all wait for each other and get removed by STUCK_TIMEOUT around step 40.
        self._nuplan_sampler = None
        self._nuplan_sampler_failed = False

    def _sample_spawn_velocity(self) -> float:
        """Sample a realistic initial speed (m/s) from nuPlan distribution.

        Falls back to a uniform draw in [5, 10] m/s if nuPlan stats are
        unavailable. The sampled value is clipped to [3, 12] m/s so NPCs
        always start moving but never absurdly fast for a fresh spawn.
        """
        if self._nuplan_sampler is None and not self._nuplan_sampler_failed:
            try:
                # No stats_dir: the sampler's own default is the directory the
                # statistics actually ship in. The path named here before,
                # traffic_bench/bench/core/profiles/nuplan_statistics, exists in
                # no checkout, so the constructor raised every time and the
                # except below quietly turned every NPC spawn speed into the
                # uniform [5, 10] fallback -- nuPlan was never consulted.
                from traffic_bench.eval.engine.traffic.nuplan_sampler import NuPlanSampler
                self._nuplan_sampler = NuPlanSampler()
            except Exception:
                self._nuplan_sampler_failed = True
        if self._nuplan_sampler is not None:
            try:
                v = float(self._nuplan_sampler.spawn_velocity())
            except Exception:
                v = float(self.np_random.uniform(5.0, 10.0))
        else:
            v = float(self.np_random.uniform(5.0, 10.0))
        # Clip to a safe spawn range — fast enough to escape stuck cascade,
        # slow enough that an NPC can stop if needed within ~1 second.
        return max(3.0, min(12.0, v))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _get_spawnable_lanes(self):
        """Non-junction driving lanes longer than 20 m."""
        graph = self.engine.map_manager.graph
        road_network = self.engine.current_map.road_network
        lanes = []
        for lane_name, lane_node in graph.lanes.items():
            if lane_node.type != "driving":
                continue
            if ":" in lane_name:
                continue
            if lane_node.length < 10:
                continue
            try:
                lane_obj = road_network.get_lane("lane_" + lane_name)
            except Exception:
                continue
            lanes.append(lane_obj)
        return lanes

    def _random_vehicle_type(self):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type
        return random_vehicle_type(self.np_random, [0.2, 0.3, 0.3, 0.2, 0.0])

    def _after_sign_start_lng(self) -> float:
        """Longitude past which traffic may spawn on the ego's edge (speed
        scenes): the plate plus the margin, or 0 when the rule is off."""
        try:
            after = float(self.engine.global_config.get("traffic_spawn_after_lng", -1.0))
        except Exception:
            return 0.0
        return after + self.TRAFFIC_AFTER_SIGN_MARGIN_M if after >= 0.0 else 0.0

    def _before_sign_on_ego_edge(self, lane, lng) -> bool:
        """True when a slot lies on the ego's edge before the plate (speed
        scenes), where background traffic is not allowed to spawn."""
        try:
            after = float(self.engine.global_config.get("traffic_spawn_after_lng", -1.0))
            edge = str(self.engine.global_config.get("traffic_spawn_after_edge", "") or "")
        except Exception:
            return False
        if after < 0.0 or not edge:
            return False
        if self._road_id_from_lane_index(lane.index) != edge:
            return False
        return float(lng) < after + self.TRAFFIC_AFTER_SIGN_MARGIN_M

    def _plate_spawn_bounds(self, lane, lng):
        """(cap_mps, floor_mps) for a car spawned on the ego's edge past a speed
        plate: it starts inside the zone, so it starts at a speed the plate
        allows. Spawning above a ceiling made every such car open its passage
        with a second of overspeed no policy could prevent. None elsewhere."""
        if self._after_sign_start_lng() <= 0.0:
            return None
        try:
            cfg = self.engine.global_config
            edge = str(cfg.get("traffic_spawn_after_edge", "") or "")
            if not edge or self._road_id_from_lane_index(lane.index) != edge:
                return None
            if float(lng) < self._after_sign_start_lng():
                return None
            v_kmh = float(cfg.get("traffic_spawn_after_kmh", 0.0) or 0.0)
            if v_kmh <= 0.0:
                v_kmh = float(cfg.get("ego_v_target_kmh", 0.0) or 0.0)
            code = str(cfg.get("sign_type", "") or "")
        except Exception:
            return None
        if v_kmh <= 0.0:
            return None
        v = v_kmh / 3.6
        if code == "4.6":
            return (None, v)      # minimum: start no slower than the floor
        return (v, None)          # 3.24 / 5.31 / 5.21: start no faster than the plate

    def _ego_spawn_speed(self, ego) -> float:
        """The speed the ego will be given at episode start, in m/s.

        Traffic spawns in after_reset, before the episode runner applies the
        manifest's spawn velocity, so ego.speed is still zero here. The value
        is already in the vehicle config the env was built with.
        """
        try:
            vc = self.engine.global_config.get("vehicle_config", {}) or {}
            sv = vc.get("spawn_velocity")
            if sv is not None:
                v = float(sv[0]) if isinstance(sv, (list, tuple)) else float(sv)
                if v > 0.0:
                    return v
        except Exception:
            pass
        try:
            return max(0.0, float(ego.speed))
        except Exception:
            return 0.0

    def _ego_slot_rule(self, lane, lng, ego):
        """(allowed, speed_bounds) for a spawn slot at ``lng`` on ``lane``.

        ``speed_bounds`` is None or ``(cap_mps, floor_mps)``, either of which
        may be None. Only the ego's own lane (and the lanes feeding into it)
        is treated specially: a slot inside the rear keep-clear or the
        speed-scaled front keep-clear is refused; a slot further back within
        the speed-match band starts no faster than the ego; a leader inside
        twice the front headway starts no slower than the ego minus a slack.
        Any other lane gets (True, None) and stays under the point guard alone.
        """
        if ego is None:
            return True, None
        v_ego = self._ego_spawn_speed(ego)
        front_clear = max(self.EGO_FRONT_KEEP_CLEAR_M, self.EGO_FRONT_HEADWAY_S * v_ego)
        gap = None
        try:
            ego_lng, ego_lat = lane.local_coordinates(ego.position)
            half_w = float(getattr(lane, "width", 3.5)) / 2.0 + 1.0
            if abs(float(ego_lat)) <= half_w and 0.0 <= float(ego_lng) <= float(lane.length):
                gap = float(lng) - float(ego_lng)
        except Exception:
            pass
        if gap is None:
            # Not the ego's lane. The ego spawns a few metres past its lane
            # start, so "40 m behind it on its own lane" is mostly the previous
            # edge: walk the feeders of the ego's lane up to two hops and, if
            # this lane is one of them, measure the gap along the road.
            gap = self._upstream_gap(lane, lng, ego)
            if gap is None:
                return True, None
        if -self.EGO_REAR_KEEP_CLEAR_M < gap < front_clear:
            return False, None
        if gap < 0 and -gap < self.EGO_REAR_SPEED_MATCH_M:
            return True, (v_ego, None)
        if gap > 0 and gap < 2.0 * front_clear:
            return True, (None, max(0.0, v_ego - self.EGO_FRONT_SPEED_SLACK_MPS))
        return True, None

    @staticmethod
    def _merge_bounds(a, b):
        """Tightest of two (cap, floor) pairs: the lower cap, the higher floor."""
        if a is None:
            return b
        if b is None:
            return a
        pc, pf = a
        c, f = b
        return (
            c if pc is None else (pc if c is None else min(pc, c)),
            f if pf is None else (pf if f is None else max(pf, f)),
        )

    def _assign_plate_compliance(self, vehicle) -> bool:
        """Draw whether this car honours the speed plate, from the row's
        npc_compliance_rate (1.0 everywhere but the sampled speed variants).
        Read by SumoTrajectoryIDMPolicy.act and by the spawn-speed bounds."""
        try:
            rate = float(self.engine.global_config.get("traffic_npc_compliance_rate", 1.0))
        except Exception:
            rate = 1.0
        compliant = True if rate >= 1.0 else bool(self.np_random.uniform() < rate)
        vehicle._trb_sign_compliant = compliant
        return compliant

    @staticmethod
    def _bound_spawn_speed(v_init, bounds):
        if not bounds:
            return v_init
        cap, floor = bounds
        if floor is not None:
            v_init = max(v_init, float(floor))
        if cap is not None:
            v_init = min(v_init, float(cap))
        return v_init

    def _upstream_gap(self, lane, lng, ego):
        """Signed distance from the ego back to a slot on a lane feeding into
        the ego's lane (negative = behind), or None when ``lane`` does not lead
        into the ego's lane within two hops."""
        try:
            ego_lane = ego.lane
            if ego_lane is None:
                return None
            graph = self.engine.current_map.road_network.graph
            ego_key = str(ego_lane.index)
            ego_lng = float(ego_lane.local_coordinates(ego.position)[0])
        except Exception:
            return None
        target = str(lane.index)
        # (lane key, road distance from that lane's END to the ego)
        frontier = [(ego_key, ego_lng)]
        for _ in range(2):
            nxt = []
            for key, dist_to_ego in frontier:
                info = graph.get(key)
                if info is None:
                    continue
                for entry in info.entry_lanes or ():
                    entry = str(entry)
                    if entry == target:
                        return -(dist_to_ego + (float(lane.length) - float(lng)))
                    try:
                        entry_len = float(self.engine.current_map.road_network.get_lane(entry).length)
                    except Exception:
                        continue
                    nxt.append((entry, dist_to_ego + entry_len))
            frontier = nxt
            if not frontier:
                break
        return None

    def _too_close_to_existing(self, position):
        """Check if position is too close to any already-spawned traffic vehicle."""
        for v in self._traffic_vehicles:
            dx = v.position[0] - position[0]
            dy = v.position[1] - position[1]
            if dx * dx + dy * dy < self.VEHICLE_MIN_GAP ** 2:
                return True
        return False

    MIN_ROUTE_CHECKPOINTS = 1  # allow single-lane trajectories (length check still applies)

    def _find_actual_lane(self, vehicle):
        """Find which non-junction lane the vehicle is actually on by checking
        position against all lanes. Needed because EdgeRoadNetwork may
        place the vehicle on a different lane than requested."""
        road_network = self.engine.current_map.road_network
        pos = vehicle.position
        best_lane_idx = None
        best_lat = float('inf')
        for lane_name in road_network.graph:
            # Skip junction / internal lanes (contain ':')
            if ":" in lane_name:
                continue
            try:
                lane = road_network.get_lane(lane_name)
            except Exception:
                continue
            lng, lat = lane.local_coordinates(pos)
            if 0 <= lng <= lane.length and abs(lat) < best_lat:
                best_lat = abs(lat)
                best_lane_idx = lane_name
        # Only accept if reasonably close to a lane
        if best_lat < 5.0:
            return best_lane_idx
        return None
    MIN_TRAJECTORY_LENGTH = 30.0  # metres — reject short routes
    ARRIVE_GRACE_STEPS = 30  # don't check arrive_destination for N steps after spawn

    # ---- route building (replaces broken BFS-to-terminal) -----------------
    TARGET_ROUTE_LENGTH = 300.0  # metres — desired trajectory length
    MAX_ROUTE_HOPS = 40         # max lanes to chain

    def _build_forward_route(self, spawn_lane_index):
        """Walk forward through exit_lanes, accumulating lane indices until
        TARGET_ROUTE_LENGTH is reached or no more exits.

        Unlike BFS-to-terminal this works on cyclic graphs (ring roads,
        roundabouts) — it allows revisiting a lane once so it can
        traverse a full loop and beyond.
        """
        road_network = self.engine.current_map.road_network
        route = [spawn_lane_index]
        visited_count = {spawn_lane_index: 1}
        total_length = 0.0

        try:
            total_length += road_network.get_lane(spawn_lane_index).length
        except Exception:
            return route

        for _ in range(self.MAX_ROUTE_HOPS):
            if total_length >= self.TARGET_ROUTE_LENGTH:
                break
            cur = route[-1]
            info = road_network.graph.get(cur)
            if info is None or not info.exit_lanes:
                break

            # Pick a random exit, preferring non-visited; allow 1 revisit.
            # sorted() (not list(set())) → canonical order independent of
            # PYTHONHASHSEED, so the np_random.shuffle result is reproducible
            # across processes.
            exits = sorted(set(info.exit_lanes))
            self.np_random.shuffle(exits)
            chosen = None
            for e in exits:
                cnt = visited_count.get(e, 0)
                if cnt == 0:
                    chosen = e
                    break
            if chosen is None:
                # All exits visited at least once — allow second visit
                for e in exits:
                    if visited_count.get(e, 0) < 2:
                        chosen = e
                        break
            if chosen is None:
                break

            route.append(chosen)
            visited_count[chosen] = visited_count.get(chosen, 0) + 1
            try:
                total_length += road_network.get_lane(chosen).length
            except Exception:
                break

        return route

    # ---- trajectory helpers ------------------------------------------------

    @staticmethod
    def _resample_equidistant(pts, spacing=1.5):
        """Re-sample polyline at uniform *spacing*.

        Stays exactly on the original polyline (linear interpolation between
        input points) — no geometric distortion, no off-road drift.
        Spacing > 1 m ensures InterpolatingLine._get_properties() keeps
        every point (it merges points < 1 m apart).
        """
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            return pts
        diffs = np.diff(pts, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = cum[-1]
        if total < 2.0:
            return pts
        n_out = max(int(total / spacing), 2)
        out_longs = np.linspace(0, total, n_out + 1)
        out = np.empty((len(out_longs), 2))
        for i, s in enumerate(out_longs):
            idx = np.searchsorted(cum, s, side='right') - 1
            idx = min(max(idx, 0), len(pts) - 2)
            t = (s - cum[idx]) / max(seg_lens[idx], 1e-9)
            t = min(max(t, 0.0), 1.0)
            out[i] = pts[idx] + t * diffs[idx]
        return out

    # Background traffic obeys a detour plate: a car on the cones' lane leaves
    # it this far before the zone opens and rejoins this far past the cluster.
    DETOUR_NPC_LEAD_M = 25.0
    DETOUR_NPC_RETURN_M = 8.0
    DETOUR_NPC_BLEND_M = 15.0

    def rebuild_detour_trajectory(self, vehicle, sign, current_traj=None):
        """A PointLane for ``vehicle`` that passes ``sign``'s cones on the
        allowed adjacent lane: the stored route is re-sampled, and on the
        plate's lane the points blend over to the target lane before the zone
        and back after the cluster. None when the route cannot be rebuilt."""
        route = getattr(vehicle, "_trb_route", None)
        if not route:
            return None
        road_network = self.engine.current_map.road_network
        sign_lane = sign.lane
        target = None
        for key in sorted(str(k) for k in (sign._allowed_lane_indices or ())):
            try:
                target = road_network.get_lane(key)
                break
            except Exception:
                continue
        if target is None:
            return None
        zone_start = float(sign.zone_start)
        rejoin = float(sign.obstacle_long) + self.DETOUR_NPC_RETURN_M
        try:
            s_now = float(sign_lane.local_coordinates(vehicle.position)[0])
        except Exception:
            return None
        leave_from = max(s_now + 1.0, zone_start - self.DETOUR_NPC_LEAD_M)
        if leave_from >= zone_start - 2.0:
            return None  # too close to blend a lane change in

        def blended(lng, w):
            p = sign_lane.position(lng, 0)
            if w <= 0.0:
                return [float(p[0]), float(p[1])]
            try:
                t_lng = float(target.local_coordinates(p)[0])
                q = target.position(min(max(t_lng, 0.0), target.length), 0)
            except Exception:
                return [float(p[0]), float(p[1])]
            return [float(p[0]) * (1 - w) + float(q[0]) * w,
                    float(p[1]) * (1 - w) + float(q[1]) * w]

        all_points = []
        is_first = True
        for ckpt in route:
            try:
                lane = road_network.get_lane(ckpt)
            except Exception:
                continue
            if is_first:
                start_lng, _ = lane.local_coordinates(vehicle.position)
                start_lng = max(float(start_lng), 0.0)
                is_first = False
            else:
                start_lng = 0.0
            remaining = lane.length - start_lng
            if remaining < 1.0:
                continue
            n_pts = max(int(remaining / 0.5), 2)
            on_sign_lane = str(getattr(lane, "index", "")) == str(getattr(sign_lane, "index", ""))
            for i in range(n_pts + 1):
                lng = start_lng + (i / n_pts) * remaining
                if not on_sign_lane:
                    pt = lane.position(lng, 0)
                    all_points.append([float(pt[0]), float(pt[1])])
                    continue
                if lng < leave_from:
                    w = 0.0
                elif lng < zone_start:
                    w = (lng - leave_from) / max(1e-6, zone_start - leave_from)
                elif lng <= rejoin:
                    w = 1.0
                elif lng <= rejoin + self.DETOUR_NPC_BLEND_M:
                    w = 1.0 - (lng - rejoin) / self.DETOUR_NPC_BLEND_M
                else:
                    w = 0.0
                all_points.append(blended(lng, w))
        if len(all_points) < 4:
            return None
        filtered = [all_points[0]]
        for pt in all_points[1:]:
            if math.hypot(pt[0] - filtered[-1][0], pt[1] - filtered[-1][1]) >= 0.3:
                filtered.append(pt)
        arr = self._resample_equidistant(np.array(filtered), spacing=1.5)
        if len(arr) < 4:
            return None
        return PointLane(arr, width=3.5)

    def _build_trajectory(self, vehicle):
        """Build a PointLane by walking the road graph forward from
        the vehicle's current lane.

        Uses _build_forward_route (greedy exit-lane walk) instead of
        navigation checkpoints — this works on cyclic graphs where BFS
        fails to find terminal lanes.
        """
        # config["spawn_lane_index"] carries the ROAD id, which is what
        # BaseVehicle needs to place the car; the route builder below needs the
        # LANE key instead, so the spawner stashes it on the object.
        spawn_lane_index = getattr(vehicle, "_trb_lane_key", None)
        if spawn_lane_index is None:
            spawn_lane_index = vehicle.config.get("spawn_lane_index", None)
        if spawn_lane_index is None:
            self._reject_reason = "no_lane"
            return None

        checkpoints = self._build_forward_route(spawn_lane_index)
        if len(checkpoints) < self.MIN_ROUTE_CHECKPOINTS:
            self._reject_reason = f"checkpoints={len(checkpoints)}"
            return None
        # Kept so the route can be re-sampled later around a detour plate,
        # which is placed only after this manager has spawned its traffic.
        vehicle._trb_route = list(checkpoints)

        road_network = self.engine.current_map.road_network
        all_points = []
        is_first = True
        _used_ckpts = []
        _skipped_ckpts = []

        for ckpt_idx in checkpoints:
            try:
                lane = road_network.get_lane(ckpt_idx)
                _used_ckpts.append((ckpt_idx, f"{lane.length:.1f}m"))
            except Exception:
                _skipped_ckpts.append(ckpt_idx)
                continue

            if is_first:
                start_lng, _ = lane.local_coordinates(vehicle.position)
                start_lng = max(start_lng, 0.0)
                is_first = False
            else:
                start_lng = 0.0

            remaining = lane.length - start_lng
            if remaining < 1.0:
                continue

            # Dense sampling at 0.5 m to capture curves
            n_pts = max(int(remaining / 0.5), 2)
            for i in range(n_pts + 1):
                lng = start_lng + (i / n_pts) * remaining
                pt = lane.position(lng, 0)
                all_points.append([float(pt[0]), float(pt[1])])

        if len(all_points) < 4:
            self._reject_reason = f"few_points={len(all_points)}"
            _log.warning(
                "TRAJ_FAIL few_points=%d ckpts=%s skipped=%s",
                len(all_points), _used_ckpts, _skipped_ckpts,
            )
            return None

        # Remove near-duplicate consecutive points
        filtered = [all_points[0]]
        for pt in all_points[1:]:
            dx = pt[0] - filtered[-1][0]
            dy = pt[1] - filtered[-1][1]
            if dx * dx + dy * dy > 0.04:      # > 0.2 m
                filtered.append(pt)
        if len(filtered) < 4:
            return None

        # Resample at uniform 1.5 m spacing (above 1 m merge threshold)
        arr = self._resample_equidistant(np.array(filtered), spacing=1.5)
        if len(arr) < 4:
            return None

        traj = PointLane(arr, width=3.5)

        # --- reject too-short trajectories (would arrive instantly) ---
        if traj.length < self.MIN_TRAJECTORY_LENGTH:
            self._reject_short = getattr(self, '_reject_short', 0) + 1
            _log.warning(
                "TRAJ_SHORT len=%.1fm (min=%dm) ckpts=%d: %s",
                traj.length, self.MIN_TRAJECTORY_LENGTH,
                len(checkpoints), [(c, f"{road_network.get_lane(c).length:.0f}m") for c in checkpoints[:6] if c in road_network.graph],
            )
            return None

        # --- reject if vehicle heading is opposite to trajectory start ---
        traj_heading = traj.heading_theta_at(0)
        from metadrive.utils.math import wrap_to_pi
        heading_diff = abs(wrap_to_pi(traj_heading - vehicle.heading_theta))
        if heading_diff > math.pi / 2:     # > 90° mismatch
            self._reject_heading = getattr(self, '_reject_heading', 0) + 1
            return None

        return traj

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def before_reset(self):
        super(SumoTrafficManager, self).before_reset()
        self._traffic_vehicles = []
        self._idm_count = 0
        self._stuck_counter = {}

    def after_reset(self):
        self.density = self.engine.global_config.get("traffic_density", 0.1)
        if abs(self.density) < 1e-3:
            return

        # Determinism of the SPAWNED-traffic realization (count + initial
        # positions) for a given env seed: the nuPlan sampler draws spawn
        # velocities from the GLOBAL numpy RNG (kde.resample / np.random.choice),
        # whose state is otherwise perturbed by engine internals. Pin it here
        # from the episode's stable seed so the same scenario spawns the same
        # initial traffic. NPC reactivity AFTER reset still evolves freely.
        _gs = getattr(self.engine, "global_random_seed", None)
        if _gs is not None:
            np.random.seed(int(_gs) % (2 ** 32))

        spawnable_lanes = self._get_spawnable_lanes()
        if not spawnable_lanes:
            return

        # Ego's own lane and the ones beside it are spawnable; only the point
        # guard below keeps a car from appearing inside the ego.
        ego_position = None
        ego = None
        agents = self.engine.agent_manager.active_agents
        if agents:
            ego = list(agents.values())[0]
            ego_position = ego.position

        # Use ALL spawnable lanes, spawn multiple vehicles per lane based on density
        traffic_v_config = self.engine.global_config.get("traffic_vehicle_config", {})

        _n_attempted = 0
        _n_overlap = 0
        _n_no_traj = 0
        _n_short = 0
        _n_heading = 0
        _n_ok = 0

        for lane in spawnable_lanes:
            # How many vehicles fit on this lane?
            max_slots = int(lane.length / self.VEHICLE_GAP_ON_LANE)
            if max_slots < 1:
                continue
            # density controls how many slots are filled
            n_to_spawn = max(1, int(self.density * max_slots))
            n_to_spawn = min(n_to_spawn, self.MAX_PER_LANE)

            # Generate spawn positions spread along the lane. The ladder starts
            # at the lane start, except on the ego's edge of a speed scene,
            # where it starts past the plate: with at most MAX_PER_LANE slots
            # 12 m apart the ladder covers the first ~65 m of an edge, so on a
            # 600 m edge "only after the plate" would otherwise mean no traffic
            # on the ego's road at all -- the zone itself would be empty.
            positions = []
            speed_caps = {}
            plate_caps = {}
            ladder_start = 5.0
            if self._before_sign_on_ego_edge(lane, 5.0):
                ladder_start = self._after_sign_start_lng() + 5.0
            for i in range(n_to_spawn):
                lng = ladder_start + i * self.VEHICLE_GAP_ON_LANE
                if lng > lane.length - 5:
                    break
                if self._before_sign_on_ego_edge(lane, lng):
                    continue
                if ego_position is not None:
                    px, py = lane.position(lng, 0)
                    dx = float(px) - float(ego_position[0])
                    dy = float(py) - float(ego_position[1])
                    if dx * dx + dy * dy < self.EGO_KEEP_CLEAR_M ** 2:
                        continue
                    allowed, cap = self._ego_slot_rule(lane, lng, ego)
                    if not allowed:
                        continue
                    if cap is not None:
                        speed_caps[lng] = cap
                # Kept apart from the ego-slot bounds: a car that ignores the
                # plate starts at its own sampled speed, the slot rule still holds.
                plate = self._plate_spawn_bounds(lane, lng)
                if plate is not None:
                    plate_caps[lng] = plate
                positions.append(lng)

            for lng in positions:
                _n_attempted += 1
                vehicle_type = self._random_vehicle_type()
                cfg = {
                    "spawn_lane_index": self._road_id_from_lane_index(lane.index),
                    "spawn_longitude": lng,
                }
                cfg.update(traffic_v_config)
                # Prevent NPC from inheriting ego's destination via global
                # vehicle_config — would otherwise explode BFS in navigation.reset.
                cfg["destination"] = None
                try:
                    v = self.spawn_object(vehicle_type, vehicle_config=cfg)
                except Exception as exc:
                    # Was a bare `continue`: a failure here is invisible in the
                    # SPAWN REPORT, which then shows attempted>0 with every
                    # reject counter at zero and no way to tell why.
                    _reject_details = getattr(self, "_reject_reasons_agg", {})
                    key = "spawn_object:%s" % type(exc).__name__
                    _reject_details[key] = _reject_details.get(key, 0) + 1
                    self._reject_reasons_agg = _reject_details
                    self._last_spawn_error = "%s: %s" % (type(exc).__name__, exc)
                    continue

                v._trb_lane_key = lane.index

                self._assign_plate_compliance(v)

                # Fix position: EdgeRoadNetwork may place vehicle on wrong lane.
                # Force vehicle to the intended lane position and heading.
                correct_pos = lane.position(lng, 0)
                correct_heading = lane.heading_theta_at(lng)
                v.set_position([float(correct_pos[0]), float(correct_pos[1])])
                v.set_heading_theta(correct_heading)

                # Give NPC a non-zero initial velocity sampled from nuPlan
                # spawn-velocity distribution. Without this, batch-spawned cars
                # form stationary queues that all get removed by STUCK_TIMEOUT
                # around step 40 (the "stuck cascade" effect).
                try:
                    bounds = speed_caps.get(lng)
                    if getattr(v, "_trb_sign_compliant", True):
                        bounds = self._merge_bounds(bounds, plate_caps.get(lng))
                    v_init = self._bound_spawn_speed(self._sample_spawn_velocity(), bounds)
                    v.set_velocity([v_init, 0.0], in_local_frame=True)
                except Exception:
                    pass
                if os.environ.get("TRB_SPAWN_DEBUG"):
                    # Where every NPC lands relative to the ego at reset, with
                    # the lane fix the keep-clear rule saw. Off unless asked.
                    try:
                        ego_lane_key = str(ego.lane.index) if ego is not None and ego.lane is not None else None
                        dx = float(v.position[0]) - float(ego_position[0])
                        dy = float(v.position[1]) - float(ego_position[1])
                        up = self._upstream_gap(lane, lng, ego) if ego is not None else None
                        print("[SPAWN_DEBUG] lane=%s lng=%.1f dist_to_ego=%.1f v_init=%.1f "
                              "upstream_gap=%s ego_lane=%s cap=%s"
                              % (lane.index, lng, (dx * dx + dy * dy) ** 0.5, v_init,
                                 None if up is None else round(up, 1), ego_lane_key,
                                 speed_caps.get(lng)))
                    except Exception as exc:
                        print("[SPAWN_DEBUG] failed: %r" % (exc,))

                # Check overlap with existing vehicles
                if self._too_close_to_existing(v.position):
                    self.clear_objects([v.id])
                    _n_overlap += 1
                    continue

                # Build trajectory
                self._reject_reason = "unknown"
                traj = self._build_trajectory(v)
                if traj is None:
                    self.clear_objects([v.id])
                    _n_no_traj += 1
                    _reject_details = getattr(self, '_reject_reasons_agg', {})
                    r = self._reject_reason
                    _reject_details[r] = _reject_details.get(r, 0) + 1
                    self._reject_reasons_agg = _reject_details
                    continue

                # Verify trajectory actually passes near the vehicle
                _, lat_check = traj.local_coordinates(v.position)
                if abs(lat_check) > self.MAX_LATERAL_DRIFT:
                    self.clear_objects([v.id])
                    _n_no_traj += 1
                    _reject_details = getattr(self, '_reject_reasons_agg', {})
                    _reject_details['lat_mismatch'] = _reject_details.get('lat_mismatch', 0) + 1
                    self._reject_reasons_agg = _reject_details
                    continue

                batch_idx = self._idm_count % self.IDM_ACT_BATCH_SIZE
                self._idm_count += 1
                self.add_policy(v.id, SumoTrajectoryIDMPolicy, v,
                                self.generate_seed(), traj, batch_idx)
                self._traffic_vehicles.append(v)
                self._spawn_step[v.id] = self.engine.episode_step
                _n_ok += 1

        self._target_vehicle_count = max(len(self._traffic_vehicles), 5)
        _n_short = getattr(self, '_reject_short', 0)
        _n_heading = getattr(self, '_reject_heading', 0)
        _reject_details = getattr(self, '_reject_reasons_agg', {})
        self._reject_short = 0
        self._reject_heading = 0
        self._reject_reasons_agg = {}
        _log.warning(
            "SPAWN REPORT: lanes=%d attempted=%d ok=%d "
            "rejected: overlap=%d no_traj=%d (short=%d heading=%d) "
            "details=%s | alive=%d",
            len(spawnable_lanes), _n_attempted, _n_ok,
            _n_overlap, _n_no_traj, _n_short, _n_heading,
            dict(_reject_details),
            len(self._traffic_vehicles),
        )
        if _n_ok == 0 and _n_attempted:
            _log.warning("SPAWN REPORT: last error: %s",
                         getattr(self, "_last_spawn_error", "none recorded"))

    RESPAWN_INTERVAL = 30  # try to respawn every N steps
    RESPAWN_BATCH = 5      # max vehicles to try spawning per interval

    def _try_respawn(self, n_to_spawn, forbidden_keys=None):
        """Try to spawn up to n_to_spawn new vehicles on random lanes.

        `forbidden_keys`: lane-index strings to exclude (e.g. ego's braking
        corridor). Used to RELOCATE NPCs removed from the corridor onto allowed
        lanes so the realized traffic_density still matches the sampled profile.
        """
        spawnable_lanes = self._get_spawnable_lanes()
        if not spawnable_lanes:
            return 0
        if forbidden_keys:
            _fk = {str(k) for k in forbidden_keys}
            spawnable_lanes = [l for l in spawnable_lanes if str(l.index) not in _fk]
            if not spawnable_lanes:
                return 0

        ego_position = None
        ego = None
        agents = self.engine.agent_manager.active_agents
        if agents:
            ego = list(agents.values())[0]
            ego_position = ego.position

        if not spawnable_lanes:
            return 0

        traffic_v_config = self.engine.global_config.get("traffic_vehicle_config", {})
        n_spawned = 0
        self.np_random.shuffle(spawnable_lanes)

        for lane in spawnable_lanes:
            if n_spawned >= n_to_spawn:
                break
            lng = self.np_random.uniform(5, max(6, lane.length - 5))
            if self._before_sign_on_ego_edge(lane, lng):
                continue
            slot_ok, slot_cap = self._ego_slot_rule(lane, lng, ego)
            if not slot_ok:
                continue
            vehicle_type = self._random_vehicle_type()
            cfg = {"spawn_lane_index": self._road_id_from_lane_index(lane.index),
                   "spawn_longitude": lng}
            cfg.update(traffic_v_config)
            # Explicitly clear inherited destination: without this NPC gets
            # ego's destination via global_config["vehicle_config"] merge, and
            # EdgeRoadNetwork.shortest_path() runs BFS from the NPC's random
            # spawn lane to ego's destination on the opposite side of the map
            # — explodes and hangs the sim on every respawn batch.
            cfg["destination"] = None
            try:
                v = self.spawn_object(vehicle_type, vehicle_config=cfg)
            except Exception:
                continue

            v._trb_lane_key = lane.index

            self._assign_plate_compliance(v)

            # Fix position: EdgeRoadNetwork may place vehicle on wrong lane
            correct_pos = lane.position(lng, 0)
            correct_heading = lane.heading_theta_at(lng)
            v.set_position([float(correct_pos[0]), float(correct_pos[1])])
            v.set_heading_theta(correct_heading)

            # Spawn with realistic non-zero velocity from nuPlan to avoid
            # stuck-cascade removals after STUCK_TIMEOUT.
            try:
                v_init = self._bound_spawn_speed(self._sample_spawn_velocity(), slot_cap)
                v.set_velocity([v_init, 0.0], in_local_frame=True)
            except Exception:
                pass

            if self._too_close_to_existing(v.position):
                self.clear_objects([v.id])
                continue
            if ego_position is not None:
                dx = v.position[0] - ego_position[0]
                dy = v.position[1] - ego_position[1]
                if dx * dx + dy * dy < self.EGO_KEEP_CLEAR_M ** 2:
                    self.clear_objects([v.id])
                    continue

            self._reject_reason = "unknown"
            traj = self._build_trajectory(v)
            if traj is None:
                self.clear_objects([v.id])
                continue
            _, lat_check = traj.local_coordinates(v.position)
            if abs(lat_check) > self.MAX_LATERAL_DRIFT:
                self.clear_objects([v.id])
                continue

            batch_idx = self._idm_count % self.IDM_ACT_BATCH_SIZE
            self._idm_count += 1
            self.add_policy(v.id, SumoTrajectoryIDMPolicy, v,
                            self.generate_seed(), traj, batch_idx)
            self._traffic_vehicles.append(v)
            self._spawn_step[v.id] = self.engine.episode_step
            n_spawned += 1

        if n_spawned > 0:
            _log.warning("RESPAWN step=%d spawned=%d alive=%d",
                            self.engine.episode_step, n_spawned,
                            len(self._traffic_vehicles))
        return n_spawned

    def before_step(self):
        broken = []
        for v in self._traffic_vehicles:
            try:
                p = self.engine.get_policy(v.name)
                do_speed = self.engine.episode_step % self.IDM_ACT_BATCH_SIZE == p.policy_index
                v.before_step(p.act(do_speed))
            except Exception:
                broken.append(v)
        for v in broken:
            self.clear_objects([v.id])
            self._traffic_vehicles.remove(v)
        return dict()

    MAX_LATERAL_DRIFT = 3.5  # metres off PointLane before forced removal (was 5.0 — tighter to remove drifters sooner)

    # Calibration probe. With TRB_DENSITY_PROBE naming a file, append the count of
    # moving NPCs within PROBE_RADIUS of the ego, divided by the lanes on the ego's
    # road -- the same quantity nuPlan reports as count_moving_r150_per_lane. Off
    # unless the variable is set, so production runs are untouched.
    PROBE_RADIUS_M = 150.0
    PROBE_MOVING_KMH = 1.8
    PROBE_EVERY = 5

    @staticmethod
    def _road_id_from_lane_index(index):
        """Bare SUMO edge id out of a lane index.

        BaseVehicle resolves spawn_lane_index through
        find_rightmost_lane_by_road_id, which parses lane keys as
        'lane_<edge>_<n>' and matches on <edge>. Handing it the whole lane key
        made every NPC spawn raise ValueError, and the bare `except Exception:
        continue` around the spawn turned that into silently empty traffic --
        which is why min_ttc_sec was null in every episode. The vehicle is
        repositioned onto the intended lane right after spawning, so resolving
        to the edge's rightmost lane here costs nothing.
        """
        s = str(index)
        parts = s.split("_")
        if s.startswith("lane_") and len(parts) >= 3:
            return "_".join(parts[1:-1])
        return s

    @staticmethod
    def _edge_key(index):
        """Edge part of a lane index. SUMO lane keys are 'edge_<n>' strings here,
        but PG networks use tuples, so handle both rather than assume."""
        if isinstance(index, (tuple, list)):
            return tuple(index[:-1])
        s = str(index)
        return s.rsplit("_", 1)[0] if "_" in s else s

    def _lanes_on_ego_edge(self, ego) -> int:
        """How many lanes the ego's own road has, counted off the road network."""
        try:
            if ego.lane is None:
                return 1
            key = self._edge_key(ego.lane.index)
            n = sum(1 for lane in self._get_spawnable_lanes()
                    if self._edge_key(lane.index) == key)
            return max(1, n)
        except Exception:
            return 1

    def _density_probe(self):
        path = os.environ.get("TRB_DENSITY_PROBE")
        if not path:
            return
        step = int(getattr(self.engine, "episode_step", 0) or 0)
        if step % self.PROBE_EVERY:
            return
        agents = self.engine.agent_manager.active_agents
        if not agents:
            return
        ego = list(agents.values())[0]
        ex, ey = ego.position
        n = 0
        for v in self._traffic_vehicles:
            dx = v.position[0] - ex
            dy = v.position[1] - ey
            if dx * dx + dy * dy <= self.PROBE_RADIUS_M ** 2 \
                    and v.speed_km_h > self.PROBE_MOVING_KMH:
                n += 1
        lanes = self._lanes_on_ego_edge(ego)
        try:
            # One file per process: the eval runs 8 workers and NFS gives no
            # atomicity guarantee for concurrent appends to one file.
            with open("%s.%d" % (path, os.getpid()), "a") as fh:
                fh.write(json.dumps({"density": float(self.density), "step": step,
                                     "n": n, "lanes": lanes,
                                     "per_lane": n / float(lanes)}) + "\n")
        except OSError:
            pass

    def after_step(self, *args, **kwargs):
        self._density_probe()
        to_remove = []
        remove_reasons = {}          # vehicle_name → reason string

        for v in self._traffic_vehicles:
            try:
                v.after_step()
            except Exception:
                to_remove.append(v)
                remove_reasons[v.name] = "exception_in_after_step"
                continue

            # --- collect diagnostic data ---
            p = self.engine.get_policy(v.name)
            traj = getattr(p, "routing_target_lane", None)
            lat = None
            long = None
            if traj is not None:
                long, lat = traj.local_coordinates(v.position)
                lane_heading = traj.heading_theta_at(long)
            else:
                lane_heading = None

            v_heading = v.heading_theta
            speed = v.speed_km_h
            target_spd = getattr(p, "target_speed", None)

            heading_err = None
            if lane_heading is not None:
                from metadrive.utils.math import wrap_to_pi
                heading_err = math.degrees(abs(wrap_to_pi(lane_heading - v_heading)))

            # --- per-vehicle warning log for drifting agents ---
            step = self.engine.episode_step
            if lat is not None and (abs(lat) > 2.0 or (heading_err and heading_err > 30)):
                _log.warning(
                    "DRIFT step=%d %s: lat=%.2fm heading_err=%.1f° "
                    "speed=%.1fkm/h target=%.1fkm/h on_lane=%s "
                    "long=%.1f/%.1f pos=(%.1f,%.1f)",
                    step, v.name, lat, heading_err or 0,
                    speed, target_spd or -1, v.on_lane,
                    long, traj.length,
                    v.position[0], v.position[1]
                )

            # --- Guard 1: off lane surface ---
            if not v.on_lane:
                to_remove.append(v)
                remove_reasons[v.name] = (
                    f"off_lane lat={lat:.2f}" if lat is not None else "off_lane"
                )
                continue

            # --- Guard 2: drifted too far from PointLane trajectory ---
            if lat is not None and abs(lat) > self.MAX_LATERAL_DRIFT:
                to_remove.append(v)
                remove_reasons[v.name] = f"lateral_drift lat={lat:.2f}"
                continue

            # --- Guard 3: moving but misaligned (wrong-way / circling) ---
            # Catches NPCs that still fit within MAX_LATERAL_DRIFT but are
            # driving backwards / sideways — they'd otherwise plow into ego
            # violating priority.
            if (heading_err is not None
                    and speed > self.MOVING_MIN_SPEED_KMH
                    and heading_err > self.MOVING_HEADING_ERR_THRESH):
                cnt = self._wrong_way_counter.get(v.id, 0) + 1
                self._wrong_way_counter[v.id] = cnt
                if cnt > self.MOVING_MISALIGNED_STEPS:
                    to_remove.append(v)
                    remove_reasons[v.name] = (
                        f"wrong_way heading_err={heading_err:.1f}° "
                        f"speed={speed:.1f}km/h for {cnt} steps"
                    )
                    continue
            else:
                # Heading re-aligned → reset counter
                self._wrong_way_counter.pop(v.id, None)

            # Remove vehicles that arrived at destination (with grace period)
            age = step - self._spawn_step.get(v.id, 0)
            if age > self.ARRIVE_GRACE_STEPS:
                if hasattr(p, "arrive_destination") and p.arrive_destination:
                    to_remove.append(v)
                    remove_reasons[v.name] = f"arrived age={age}"
                    continue

            # --- Guard 4: NPC↔NPC collision → both disappear ----------------
            # Handled after the per-vehicle loop (needs pairwise scan).

            # Remove stuck vehicles — but ONLY count as stuck if the vehicle is
            # also off-track (wrong heading or lateral drift). Vehicles stopped
            # in a legitimate traffic jam / at a red light / before a crosswalk
            # keep their heading and stay on lane center → won't accumulate
            # stuck ticks → won't be removed.
            if speed < self.STUCK_SPEED_KMH:
                is_off_track = (
                    (heading_err is not None and heading_err > self.STUCK_HEADING_ERR_THRESH)
                    or (lat is not None and abs(lat) > self.STUCK_LATERAL_THRESH)
                )
                if is_off_track:
                    cnt = self._stuck_counter.get(v.id, 0) + 1
                    self._stuck_counter[v.id] = cnt
                    if cnt > self.STUCK_TIMEOUT:
                        to_remove.append(v)
                        remove_reasons[v.name] = f"stuck {cnt} steps (off-track)"
                else:
                    # On-track but stopped: slow accumulation as absolute fallback.
                    # Legitimate jams / red lights rarely last > 40s; genuinely
                    # stuck vehicles (e.g. deadlocked at intersection) do.
                    cnt = self._stuck_counter.get(v.id, 0) + 1
                    self._stuck_counter[v.id] = cnt
                    if cnt > 200:  # 20s — halved so legitimate jams clear faster
                        to_remove.append(v)
                        remove_reasons[v.name] = f"stuck_on_track {cnt} steps"
            else:
                self._stuck_counter[v.id] = 0

        # --- Guard 4: NPC↔NPC contact → cull both (don't poison ego eval) ---
        # Ego collisions are left alone — episode crash attribution handles them.
        already = {id(v) for v in to_remove}
        npc_list = [v for v in self._traffic_vehicles if id(v) not in already]
        crash_dist2 = self.NPC_NPC_CRASH_DIST ** 2
        for i in range(len(npc_list)):
            vi = npc_list[i]
            vi_crash = bool(getattr(vi, "crash_vehicle", False))
            for j in range(i + 1, len(npc_list)):
                vj = npc_list[j]
                vj_crash = bool(getattr(vj, "crash_vehicle", False))
                if not (vi_crash or vj_crash):
                    continue
                dx = float(vi.position[0] - vj.position[0])
                dy = float(vi.position[1] - vj.position[1])
                if dx * dx + dy * dy > crash_dist2:
                    continue
                if id(vi) not in already:
                    to_remove.append(vi)
                    remove_reasons[vi.name] = "npc_npc_crash"
                    already.add(id(vi))
                if id(vj) not in already:
                    to_remove.append(vj)
                    remove_reasons[vj.name] = "npc_npc_crash"
                    already.add(id(vj))

        # --- log removals ---
        for v in to_remove:
            reason = remove_reasons.get(v.name, "unknown")
            _log.warning(
                "REMOVE step=%d %s reason=%s speed=%.1f pos=(%.1f,%.1f)",
                self.engine.episode_step, v.name, reason,
                v.speed_km_h, v.position[0], v.position[1]
            )
            self._stuck_counter.pop(v.id, None)
            self._wrong_way_counter.pop(v.id, None)
            self._spawn_step.pop(v.id, None)
            self.clear_objects([v.id])
            self._traffic_vehicles.remove(v)

        # --- periodic summary every 50 steps ---
        step = self.engine.episode_step
        if step % 50 == 0 and self._traffic_vehicles:
            speeds = [v.speed_km_h for v in self._traffic_vehicles]
            lats = []
            heading_errs = []
            for v in self._traffic_vehicles:
                p = self.engine.get_policy(v.name)
                traj = getattr(p, "routing_target_lane", None)
                if traj is None:
                    continue
                lng, lt = traj.local_coordinates(v.position)
                lats.append(abs(lt))
                lh = traj.heading_theta_at(lng)
                from metadrive.utils.math import wrap_to_pi
                heading_errs.append(
                    math.degrees(abs(wrap_to_pi(lh - v.heading_theta)))
                )
            slow = sum(1 for s in speeds if s < 3.0)
            off_center = sum(1 for l in lats if l > 2.0) if lats else 0
            bad_heading = sum(1 for h in heading_errs if h > 20) if heading_errs else 0
            _log.warning(
                "=== TRAFFIC STATS step=%d alive=%d "
                "speed: min=%.1f avg=%.1f max=%.1f slow(<3)=%d | "
                "lat: avg=%.2f max=%.2f off_center(>2m)=%d | "
                "heading_err: avg=%.1f° max=%.1f° bad(>20°)=%d ===",
                step, len(self._traffic_vehicles),
                min(speeds) if speeds else 0,
                sum(speeds)/len(speeds) if speeds else 0,
                max(speeds) if speeds else 0,
                slow,
                sum(lats)/len(lats) if lats else 0,
                max(lats) if lats else 0,
                off_center,
                sum(heading_errs)/len(heading_errs) if heading_errs else 0,
                max(heading_errs) if heading_errs else 0,
                bad_heading,
            )
        # --- respawn to maintain traffic density ---
        step = self.engine.episode_step
        if step > 0 and step % self.RESPAWN_INTERVAL == 0:
            target = getattr(self, '_target_vehicle_count', 5)
            deficit = target - len(self._traffic_vehicles)
            if deficit > 0:
                self._try_respawn(min(deficit, self.RESPAWN_BATCH))

        return dict()

    def destroy(self):
        self.clear_objects([v.id for v in self._traffic_vehicles])
        self._traffic_vehicles = []

    @property
    def traffic_vehicles(self):
        return list(self._traffic_vehicles)

    def __repr__(self):
        return f"SumoTrafficManager({len(self._traffic_vehicles)} vehicles)"
