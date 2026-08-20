"""
Traffic manager for SUMO-based environments.
Spawns TrajectoryIDM-controlled vehicles that follow PointLane trajectories
built from the navigation route — the same approach used in ScenarioNet.
"""
import logging
import math

# Suppress verbose NPC warnings (DRIFT, TRAJ_SHORT, SPAWN REPORT, RESPAWN, REMOVE).
# All logging.warning() calls in this module use the root logger directly;
# redirect them through a module-level logger so we can control the level.
_log = logging.getLogger(__name__)
_log.setLevel(logging.ERROR)

import numpy as np

from metadrive.component.lane.point_lane import PointLane
from metadrive.manager.base_manager import BaseManager
from pdd_bench.envs.sumo_idm_policy import SumoTrajectoryIDMPolicy


class SumoTrafficManager(BaseManager):
    EGO_SAFE_RADIUS = 50   # don't spawn near ego (metres)
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
                from pathlib import Path
                profiles = (
                    Path(__file__).resolve().parent.parent
                    / "bench"
                    / "core" / "profiles"
                )
                from pdd_bench.bench.core.profiles.nuplan_sampler import NuPlanSampler
                self._nuplan_sampler = NuPlanSampler(
                    stats_dir=str(profiles / "nuplan_statistics")
                )
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

    def _build_trajectory(self, vehicle):
        """Build a PointLane by walking the road graph forward from
        the vehicle's current lane.

        Uses _build_forward_route (greedy exit-lane walk) instead of
        navigation checkpoints — this works on cyclic graphs where BFS
        fails to find terminal lanes.
        """
        spawn_lane_index = vehicle.config.get("spawn_lane_index", None)
        if spawn_lane_index is None:
            self._reject_reason = "no_lane"
            return None

        checkpoints = self._build_forward_route(spawn_lane_index)
        if len(checkpoints) < self.MIN_ROUTE_CHECKPOINTS:
            self._reject_reason = f"checkpoints={len(checkpoints)}"
            return None

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

        # Filter lanes near ego and ego's own lane
        ego_position = None
        ego_lane_index = None
        agents = self.engine.agent_manager.active_agents
        if agents:
            ego = list(agents.values())[0]
            ego_position = ego.position
            if ego.lane is not None:
                ego_lane_index = ego.lane.index
        if ego_position is not None:
            safe_lanes = []
            for lane in spawnable_lanes:
                # Skip ego's own lane entirely
                if lane.index == ego_lane_index:
                    continue
                lng, lat = lane.local_coordinates(ego_position)
                if 0 <= lng <= lane.length and abs(lat) < self.EGO_SAFE_RADIUS:
                    continue
                safe_lanes.append(lane)
            spawnable_lanes = safe_lanes

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

            # Generate spawn positions spread along the lane
            positions = []
            for i in range(n_to_spawn):
                lng = 5 + i * self.VEHICLE_GAP_ON_LANE
                if lng > lane.length - 5:
                    break
                positions.append(lng)

            for lng in positions:
                _n_attempted += 1
                vehicle_type = self._random_vehicle_type()
                cfg = {
                    "spawn_lane_index": lane.index,
                    "spawn_longitude": lng,
                }
                cfg.update(traffic_v_config)
                # Prevent NPC from inheriting ego's destination via global
                # vehicle_config — would otherwise explode BFS in navigation.reset.
                cfg["destination"] = None
                try:
                    v = self.spawn_object(vehicle_type, vehicle_config=cfg)
                except Exception:
                    continue

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
                    v_init = self._sample_spawn_velocity()
                    v.set_velocity([v_init, 0.0], in_local_frame=True)
                except Exception:
                    pass

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
        ego_lane_index = None
        agents = self.engine.agent_manager.active_agents
        if agents:
            ego = list(agents.values())[0]
            ego_position = ego.position
            if ego.lane is not None:
                ego_lane_index = ego.lane.index
        if ego_position is not None:
            safe_lanes = []
            for lane in spawnable_lanes:
                if lane.index == ego_lane_index:
                    continue
                lng, lat = lane.local_coordinates(ego_position)
                if 0 <= lng <= lane.length and abs(lat) < self.EGO_SAFE_RADIUS:
                    continue
                safe_lanes.append(lane)
            spawnable_lanes = safe_lanes

        if not spawnable_lanes:
            return 0

        traffic_v_config = self.engine.global_config.get("traffic_vehicle_config", {})
        n_spawned = 0
        self.np_random.shuffle(spawnable_lanes)

        for lane in spawnable_lanes:
            if n_spawned >= n_to_spawn:
                break
            lng = self.np_random.uniform(5, max(6, lane.length - 5))
            vehicle_type = self._random_vehicle_type()
            cfg = {"spawn_lane_index": lane.index, "spawn_longitude": lng}
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

            # Fix position: EdgeRoadNetwork may place vehicle on wrong lane
            correct_pos = lane.position(lng, 0)
            correct_heading = lane.heading_theta_at(lng)
            v.set_position([float(correct_pos[0]), float(correct_pos[1])])
            v.set_heading_theta(correct_heading)

            # Spawn with realistic non-zero velocity from nuPlan to avoid
            # stuck-cascade removals after STUCK_TIMEOUT.
            try:
                v_init = self._sample_spawn_velocity()
                v.set_velocity([v_init, 0.0], in_local_frame=True)
            except Exception:
                pass

            if self._too_close_to_existing(v.position):
                self.clear_objects([v.id])
                continue
            if ego_position is not None:
                dx = v.position[0] - ego_position[0]
                dy = v.position[1] - ego_position[1]
                if dx * dx + dy * dy < self.EGO_SAFE_RADIUS ** 2:
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

    def after_step(self, *args, **kwargs):
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
