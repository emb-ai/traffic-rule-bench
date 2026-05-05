from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.manager.base_manager import BaseManager
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy
from metadrive.scenario.parse_object_state import parse_object_state
from metadrive.type import MetaDriveType


class ScenarioNetLikeReplayPolicy(ReplayTrafficParticipantPolicy):
    """
    Replay policy compatible with PG/SUMO envs where engine.data_manager is not available.
    Supports delay-based pausing for realistic yielding behavior.
    """

    def __init__(self, control_object, track, random_seed=None):
        super().__init__(control_object=control_object, track=track, random_seed=random_seed)
        self._delay_steps = 0

    @property
    def track_index(self) -> int:
        return max(0, int(self.episode_step) - int(self._delay_steps))

    @property
    def is_current_step_valid(self):
        idx = self.track_index
        return 0 <= idx < len(self.traj_info) and self.traj_info[idx] is not None

    def request_pause(self, steps: int = 1):
        self._delay_steps += int(max(0, steps))

    def get_trajectory_info(self, track):
        ret = []
        states = track.get("state", {})
        length = len(states.get("valid", []))
        for i in range(length):
            state = parse_object_state(track, i)
            ret.append(state if state["valid"] else None)
        return ret

    def act(self, *args, **kwargs):
        idx = self.track_index
        if idx >= len(self.traj_info):
            return None

        info = self.traj_info[idx]
        if info is None or not bool(info.get("valid", False)):
            return None

        if "throttle_brake" in info and hasattr(self.control_object, "set_throttle_brake"):
            self.control_object.set_throttle_brake(float(np.asarray(info["throttle_brake"]).item()))
        if "steering" in info and hasattr(self.control_object, "set_steering"):
            self.control_object.set_steering(float(np.asarray(info["steering"]).item()))
        self.control_object.set_position(info["position"])
        self.control_object.set_velocity(info["velocity"], in_local_frame=self._velocity_local_frame)
        self.control_object.set_heading_theta(info["heading"])
        self.control_object.set_angular_velocity(info.get("angular_velocity", 0))
        return None


@dataclass
class _CrosswalkSpec:
    crosswalk_id: str
    polygon: np.ndarray
    center: np.ndarray
    walk_start: np.ndarray
    walk_end: np.ndarray
    walk_dir: np.ndarray
    span_dir: np.ndarray
    half_span: float
    walk_length: float


class CrosswalkPedestrianManager(BaseManager):
    """
    ScenarioNet-like pedestrian manager:
    - generate synthetic track dictionaries (state.valid/position/velocity/heading)
    - spawn object only when current step is valid
    - move pedestrian via replay policy on each step
    """

    PRIORITY = 12

    def __init__(self):
        super().__init__()
        raw_cfg = self.engine.global_config.get("pedestrian_manager", {})
        self._cfg = raw_cfg.get_dict() if hasattr(raw_cfg, "get_dict") else dict(raw_cfg)

        self.enabled = bool(self.engine.global_config.get("use_pedestrian_manager", False)) and bool(
            self._cfg.get("enabled", True)
        )
        self.initial_pedestrians = int(self._cfg.get("initial_pedestrians", 2))
        self.max_pedestrians = int(self._cfg.get("max_pedestrians", 6))
        self.spawn_by_interval = bool(self._cfg.get("spawn_by_interval", True))
        self.spawn_probability = float(self._cfg.get("spawn_probability", 0.08))
        interval_range = self._cfg.get("crossing_interval_range", [6.0, 12.0])
        self.interval_min, self.interval_max = float(interval_range[0]), float(interval_range[1])
        self.max_active_per_crosswalk = int(self._cfg.get("max_active_per_crosswalk", 1))
        self.max_new_tracks_per_step = int(self._cfg.get("max_new_tracks_per_step", 1))
        self.min_spawn_gap = float(self._cfg.get("min_spawn_gap", 1.5))
        self.speed_mean = float(self._cfg.get("speed_mean", 1.2))
        self.speed_std = float(self._cfg.get("speed_std", 0.2))
        self.arrive_dist = float(self._cfg.get("arrive_dist", 0.35))

        wait_range = self._cfg.get("wait_time_range", [1.5, 4.0])
        self.wait_min, self.wait_max = float(wait_range[0]), float(wait_range[1])
        pause_range = self._cfg.get("pause_time_range", [0.8, 2.0])
        self.pause_min, self.pause_max = float(pause_range[0]), float(pause_range[1])
        self.yield_to_vehicles = bool(self._cfg.get("yield_to_vehicles", True))
        self.yield_on_crosswalk = bool(self._cfg.get("yield_on_crosswalk", False))
        self.yield_distance = float(self._cfg.get("yield_distance", 12.0))
        self.yield_speed_kmh = float(self._cfg.get("yield_speed_kmh", 8.0))
        self.spawn_jitter_steps = int(self._cfg.get("spawn_jitter_steps", 20))
        # Probability to allow spawning on a crosswalk whose adjacent TL is green for cars.
        # 1.0 disables suppression; 0.0 never spawns on green. Default 0.05 = rarely.
        self.green_tl_spawn_probability = float(self._cfg.get("green_tl_spawn_probability", 0.05))
        self.tl_match_radius = float(self._cfg.get("tl_match_radius", 40.0))

        self._crosswalks: Dict[str, _CrosswalkSpec] = {}
        self._tracks: Dict[str, dict] = {}
        self._scenario_id_to_obj_id: Dict[str, str] = {}
        self._scenario_id_to_crosswalk_id: Dict[str, str] = {}
        self._next_spawn_step: Dict[str, int] = {}
        self._pause_until_step: Dict[str, int] = {}
        self._crosswalk_to_tl: Dict[str, object] = {}
        self._tl_mapping_resolved: bool = False
        self._counter = 0

    def before_reset(self):
        super().before_reset()
        self._crosswalks = {}
        self._tracks = {}
        self._scenario_id_to_obj_id = {}
        self._scenario_id_to_crosswalk_id = {}
        self._next_spawn_step = {}
        self._pause_until_step = {}
        self._crosswalk_to_tl = {}
        self._tl_mapping_resolved = False
        self._counter = 0

    def reset(self):
        if not self.enabled:
            return
        self._crosswalks = self._collect_crosswalk_specs()
        if not self._crosswalks:
            return
        self._init_spawn_timers()
        target = max(0, min(self.initial_pedestrians, self.max_pedestrians))
        for _ in range(target):
            crosswalk_id = self._pick_crosswalk_for_spawn(due_only=False)
            if crosswalk_id is None:
                break
            if not self._schedule_track(crosswalk_id):
                continue
            self._set_next_spawn_step(crosswalk_id, int(self.episode_step))
        self._spawn_due_tracks()

    def after_step(self, *args, **kwargs):
        if not self.enabled:
            return {}

        to_cleanup = []
        for scenario_id, obj_id in list(self._scenario_id_to_obj_id.items()):
            obj = self.spawned_objects.get(obj_id, None)
            if obj is None:
                to_cleanup.append(scenario_id)
                continue
            policy = self.get_policy(obj_id)
            if policy is None or not policy.is_current_step_valid:
                to_cleanup.append(scenario_id)
                continue

            if self._should_pause_pedestrian(scenario_id, obj):
                policy.request_pause(1)

            policy.act()
            if self._has_arrived(scenario_id, obj, policy):
                to_cleanup.append(scenario_id)

        for scenario_id in to_cleanup:
            self._cleanup_scenario_track(scenario_id)

        self._spawn_due_tracks()

        self._schedule_new_tracks()
        self._spawn_due_tracks()

        return {"pedestrian_manager": {"pedestrians": len(self._scenario_id_to_obj_id)}}

    def _collect_crosswalk_specs(self) -> Dict[str, _CrosswalkSpec]:
        specs = {}
        crosswalks = getattr(self.engine.current_map, "crosswalks", {})
        for crosswalk_id, feat in crosswalks.items():
            polygon = np.asarray(feat.get("polygon", []), dtype=np.float64)
            if polygon.shape[0] < 4:
                continue
            if polygon.ndim != 2 or polygon.shape[1] < 2:
                continue
            polygon = polygon[:, :2]
            pts = polygon[:4]
            walk_start, walk_end, span_vec = self._extract_walk_and_span_axes(pts)
            span_len = float(np.linalg.norm(span_vec))
            walk_len = float(np.linalg.norm(walk_end - walk_start))
            if span_len < 0.15 or walk_len < 2.0:
                continue

            span_dir = span_vec / span_len
            walk_dir = (walk_end - walk_start) / walk_len
            specs[str(crosswalk_id)] = _CrosswalkSpec(
                crosswalk_id=str(crosswalk_id),
                polygon=polygon,
                center=np.mean(pts, axis=0),
                walk_start=walk_start,
                walk_end=walk_end,
                walk_dir=walk_dir,
                span_dir=span_dir,
                half_span=span_len / 2.0,
                walk_length=walk_len,
            )
        return specs

    @staticmethod
    def _extract_walk_and_span_axes(pts: np.ndarray):
        """
        Infer crossing direction and span direction from a quadrilateral.
        """
        p0, p1, p2, p3 = pts
        a_start = (p0 + p1) / 2.0
        a_end = (p3 + p2) / 2.0
        a_len = float(np.linalg.norm(a_end - a_start))

        b_start = (p1 + p2) / 2.0
        b_end = (p0 + p3) / 2.0
        b_len = float(np.linalg.norm(b_end - b_start))

        if a_len >= b_len:
            return a_start, a_end, (p1 - p0)
        return b_start, b_end, (p2 - p1)

    def _schedule_track(self, crosswalk_id: Optional[str] = None) -> bool:
        if not self._crosswalks:
            return False
        if crosswalk_id is None:
            crosswalk_id = self._pick_crosswalk_for_spawn(due_only=False)
        spec = self._crosswalks.get(str(crosswalk_id), None)
        if spec is None:
            return False

        side_offset = float(self.np_random.uniform(-0.35 * spec.half_span, 0.35 * spec.half_span))
        side_bias = spec.span_dir * side_offset

        start_from_left = bool(self.np_random.rand() < 0.5)
        start = spec.walk_start + side_bias if start_from_left else spec.walk_end + side_bias
        end = spec.walk_end + side_bias if start_from_left else spec.walk_start + side_bias

        if not self._is_position_free(start):
            return False

        dt = self._sim_dt()
        speed = max(0.5, float(self.np_random.normal(self.speed_mean, self.speed_std)))
        direction = end - start
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            return False
        direction = direction / dist
        heading = float(math.atan2(direction[1], direction[0]))

        wait_steps = int(self.np_random.uniform(self.wait_min, self.wait_max) / max(dt, 1e-3))
        travel_steps = max(2, int(math.ceil(dist / max(speed * dt, 1e-3))))
        start_step = int(self.episode_step) + int(self.np_random.randint(0, max(self.spawn_jitter_steps, 1)))
        track_len = start_step + wait_steps + travel_steps + 2

        position = np.zeros((track_len, 3), dtype=np.float32)
        velocity = np.zeros((track_len, 2), dtype=np.float32)
        heading_arr = np.zeros((track_len, ), dtype=np.float32)
        valid = np.zeros((track_len, ), dtype=bool)
        length = np.full((track_len, ), Pedestrian.RADIUS * 2, dtype=np.float32)
        width = np.full((track_len, ), Pedestrian.RADIUS * 2, dtype=np.float32)
        height = np.full((track_len, ), Pedestrian.HEIGHT, dtype=np.float32)

        # Waiting phase (on curb)
        for i in range(wait_steps):
            idx = start_step + i
            if idx >= track_len:
                break
            position[idx, :2] = start.astype(np.float32)
            heading_arr[idx] = np.float32(heading)
            valid[idx] = True

        # Crossing phase
        for i in range(travel_steps):
            idx = start_step + wait_steps + i
            if idx >= track_len:
                break
            t = i / max(travel_steps - 1, 1)
            p = start + (end - start) * t
            position[idx, :2] = p.astype(np.float32)
            velocity[idx, :] = (direction * speed).astype(np.float32)
            heading_arr[idx] = np.float32(heading)
            valid[idx] = True

        scenario_id = f"ped_{self._counter}"
        self._counter += 1
        track = {
            "type": MetaDriveType.PEDESTRIAN,
            "state": {
                "position": position,
                "velocity": velocity,
                "heading": heading_arr,
                "valid": valid,
                "length": length,
                "width": width,
                "height": height,
            },
            "metadata": {
                "type": MetaDriveType.PEDESTRIAN,
                "object_id": scenario_id,
            },
            "_spawn_crosswalk_id": spec.crosswalk_id,
            "_goal_position": end.astype(np.float32),
        }
        self._tracks[scenario_id] = track
        return True

    def _spawn_due_tracks(self):
        for scenario_id, track in list(self._tracks.items()):
            if scenario_id in self._scenario_id_to_obj_id:
                continue
            if len(self._scenario_id_to_obj_id) >= self.max_pedestrians:
                break
            state = parse_object_state(track, int(self.episode_step), include_z_position=False)
            if not state["valid"]:
                continue
            # Re-check TL state at spawn time: tracks scheduled during reset(), before
            # TrafficLightSigns were registered, could otherwise appear on green crosswalks.
            crosswalk_id = str(track.get("_spawn_crosswalk_id", ""))
            if crosswalk_id and self._is_tl_green_for_cars(crosswalk_id):
                if float(self.np_random.random()) > self.green_tl_spawn_probability:
                    self._tracks.pop(scenario_id, None)
                    self._set_next_spawn_step(crosswalk_id, int(self.episode_step))
                    continue
            self._spawn_pedestrian(scenario_id, track, state)

    def _spawn_pedestrian(self, scenario_id: str, track: dict, state: dict):
        obj = self.spawn_object(
            Pedestrian,
            name=scenario_id if self.engine.global_config["force_reuse_object_name"] else None,
            position=[float(state["position"][0]), float(state["position"][1])],
            heading_theta=float(state["heading"]),
            force_spawn=True,
        )
        self._scenario_id_to_obj_id[scenario_id] = obj.name
        self._scenario_id_to_crosswalk_id[scenario_id] = str(track.get("_spawn_crosswalk_id", ""))
        policy = self.add_policy(obj.name, ScenarioNetLikeReplayPolicy, obj, track)
        policy.act()

    def _cleanup_scenario_track(self, scenario_id: str):
        obj_id = self._scenario_id_to_obj_id.pop(scenario_id, None)
        crosswalk_id = self._scenario_id_to_crosswalk_id.pop(scenario_id, None)
        if obj_id is not None:
            if obj_id in self.spawned_objects:
                self.clear_objects([obj_id])
        self._tracks.pop(scenario_id, None)
        if crosswalk_id:
            self._set_next_spawn_step(crosswalk_id, int(self.episode_step))
        self._pause_until_step.pop(scenario_id, None)

    def _resolve_crosswalk_tl_mapping(self):
        """Match each crosswalk to the nearest TrafficLightSign within tl_match_radius."""
        self._crosswalk_to_tl = {}
        sign_mgr = getattr(self.engine, "traffic_sign_manager", None)
        if sign_mgr is None:
            return
        tl_signs = []
        for sign in getattr(sign_mgr, "signs", []):
            if type(sign).__name__ != "TrafficLightSign":
                continue
            pos = getattr(sign, "position", None)
            if pos is None:
                continue
            try:
                tl_signs.append((sign, np.asarray(pos[:2], dtype=np.float64)))
            except Exception:
                continue
        if not tl_signs:
            return
        radius_sq = float(self.tl_match_radius) ** 2
        for crosswalk_id, spec in self._crosswalks.items():
            center = np.asarray(spec.center, dtype=np.float64)
            best_sign = None
            best_d2 = radius_sq
            for sign, pos in tl_signs:
                d2 = float(np.sum((pos - center) ** 2))
                if d2 <= best_d2:
                    best_d2 = d2
                    best_sign = sign
            if best_sign is not None:
                self._crosswalk_to_tl[crosswalk_id] = best_sign

    def _is_tl_green_for_cars(self, crosswalk_id: str) -> bool:
        """Return True if the TL adjacent to this crosswalk is currently green for cars."""
        if not self._tl_mapping_resolved:
            self._resolve_crosswalk_tl_mapping()
            # Only mark as resolved once we actually found TLs, otherwise retry next call
            # (TL signs in SUMO env are added after the first reset tick).
            if self._crosswalk_to_tl:
                self._tl_mapping_resolved = True
        sign = self._crosswalk_to_tl.get(str(crosswalk_id), None)
        if sign is None:
            return False
        try:
            sign.update_state()
        except Exception:
            pass
        states = getattr(sign, "current_states", {}) or {}
        if not states:
            return False
        # Dominant state: red > yellow > green.
        priority = {"r": 3, "y": 2, "u": 2, "g": 1, "G": 1, "s": 1}
        top = max(priority.get(s, 0) for s in states.values())
        if top == 0:
            return False
        dominant = [s for s in states.values() if priority.get(s, 0) == top]
        return any(s in ("g", "G", "s") for s in dominant)

    def _pick_crosswalk_for_spawn(self, due_only: bool = False) -> Optional[str]:
        if not self._crosswalks:
            return None
        ids = list(self._crosswalks.keys())
        occupancy = self._crosswalk_track_load()
        current_step = int(self.episode_step)

        candidates = []
        for crosswalk_id in ids:
            if occupancy.get(crosswalk_id, 0) >= self.max_active_per_crosswalk:
                continue
            if due_only and self.spawn_by_interval:
                if current_step < int(self._next_spawn_step.get(crosswalk_id, 0)):
                    continue
            if self._is_tl_green_for_cars(crosswalk_id):
                if float(self.np_random.random()) > self.green_tl_spawn_probability:
                    continue
            candidates.append(crosswalk_id)

        if not candidates:
            return None

        weights = np.asarray([1.0 / (1.0 + occupancy.get(k, 0)) for k in candidates], dtype=np.float64)
        weights /= np.sum(weights)
        idx = int(self.np_random.choice(len(candidates), p=weights))
        return candidates[idx]

    def _is_position_free(self, pos: np.ndarray) -> bool:
        for obj in self.spawned_objects.values():
            d = np.asarray(obj.position[:2], dtype=np.float64) - pos
            if float(np.linalg.norm(d)) < self.min_spawn_gap:
                return False
        return True

    def _crosswalk_track_load(self) -> Dict[str, int]:
        load = {crosswalk_id: 0 for crosswalk_id in self._crosswalks.keys()}
        for track in self._tracks.values():
            crosswalk_id = str(track.get("_spawn_crosswalk_id", ""))
            if crosswalk_id in load:
                load[crosswalk_id] += 1
        return load

    def _init_spawn_timers(self):
        self._next_spawn_step = {}
        now = int(self.episode_step)
        max_jitter = max(1, int(self.spawn_jitter_steps))
        for crosswalk_id in self._crosswalks.keys():
            self._next_spawn_step[crosswalk_id] = now + int(self.np_random.randint(0, max_jitter))

    def _sample_interval_steps(self) -> int:
        if self.interval_max <= 0:
            return 1
        if self.interval_max <= self.interval_min:
            interval_sec = self.interval_min
        else:
            interval_sec = float(self.np_random.uniform(self.interval_min, self.interval_max))
        return max(1, int(round(interval_sec / max(self._sim_dt(), 1e-3))))

    def _set_next_spawn_step(self, crosswalk_id: str, base_step: int):
        if not self.spawn_by_interval:
            self._next_spawn_step[str(crosswalk_id)] = int(base_step)
            return
        next_step = int(base_step) + self._sample_interval_steps()
        self._next_spawn_step[str(crosswalk_id)] = next_step

    def _schedule_new_tracks(self):
        if len(self._tracks) >= self.max_pedestrians:
            return

        if not self.spawn_by_interval:
            if self.np_random.rand() < self.spawn_probability:
                self._schedule_track()
            return

        budget = min(
            max(0, self.max_pedestrians - len(self._tracks)),
            max(1, int(self.max_new_tracks_per_step)),
        )
        if budget <= 0:
            return

        now = int(self.episode_step)
        for _ in range(budget):
            crosswalk_id = self._pick_crosswalk_for_spawn(due_only=True)
            if crosswalk_id is None:
                break
            if self._schedule_track(crosswalk_id):
                self._set_next_spawn_step(crosswalk_id, now)
            else:
                # Retry this crosswalk later if spawn point was temporarily blocked.
                self._next_spawn_step[str(crosswalk_id)] = now + max(1, int(round(0.5 / max(self._sim_dt(), 1e-3))))

    def _has_arrived(self, scenario_id: str, obj: Pedestrian, policy: ScenarioNetLikeReplayPolicy) -> bool:
        goal = self._tracks.get(scenario_id, {}).get("_goal_position", None)
        if goal is None:
            return False
        dist = float(np.linalg.norm(np.asarray(goal[:2], dtype=np.float64) - np.asarray(obj.position[:2], dtype=np.float64)))
        return dist <= self.arrive_dist and policy.track_index >= len(policy.traj_info) - 2

    def _sim_dt(self) -> float:
        return float(self.engine.global_config["physics_world_step_size"]) * float(self.engine.global_config["decision_repeat"])

    def _sample_pause_steps(self) -> int:
        if self.pause_max <= 0:
            return 0
        pause_sec = float(self.np_random.uniform(self.pause_min, max(self.pause_min, self.pause_max)))
        return max(1, int(round(pause_sec / max(self._sim_dt(), 1e-3))))

    def _should_pause_pedestrian(self, scenario_id: str, ped_obj: Pedestrian) -> bool:
        if not self.yield_to_vehicles:
            return False
        crosswalk_id = self._scenario_id_to_crosswalk_id.get(scenario_id, "")
        spec = self._crosswalks.get(crosswalk_id, None)
        if spec is None:
            return False

        ped_pos = np.asarray(ped_obj.position[:2], dtype=np.float64)
        dist_to_polygon = self._distance_to_polygon(ped_pos, spec.polygon)
        in_crosswalk = self._point_in_polygon(ped_pos, spec.polygon) or dist_to_polygon <= 0.05
        if in_crosswalk and (not self.yield_on_crosswalk):
            self._pause_until_step.pop(scenario_id, None)
            return False

        if dist_to_polygon > max(self.arrive_dist, 0.6):
            self._pause_until_step.pop(scenario_id, None)
            return False

        current_step = int(self.episode_step)
        hazard = self._has_hazardous_vehicle(spec)
        if hazard:
            pause_steps = self._sample_pause_steps()
            until = max(current_step + pause_steps, self._pause_until_step.get(scenario_id, -1))
            self._pause_until_step[scenario_id] = until

        pause_until = self._pause_until_step.get(scenario_id, -1)
        should_pause = hazard or (current_step <= pause_until)
        if not should_pause:
            self._pause_until_step.pop(scenario_id, None)
        return should_pause

    def _has_hazardous_vehicle(self, spec: _CrosswalkSpec) -> bool:
        for obj in self.engine.get_objects(lambda o: isinstance(o, BaseVehicle)).values():
            veh_pos = np.asarray(obj.position[:2], dtype=np.float64)
            dist_to_crosswalk = self._distance_to_polygon(veh_pos, spec.polygon)
            if dist_to_crosswalk > self.yield_distance:
                continue
            speed_kmh = float(getattr(obj, "speed_km_h", 0.0) or 0.0)
            if speed_kmh < self.yield_speed_kmh:
                continue
            heading = float(getattr(obj, "heading_theta", 0.0) or 0.0)
            forward = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)
            to_center = spec.center - veh_pos
            if float(np.dot(forward, to_center)) < -1.0 and dist_to_crosswalk > 0.5:
                continue
            return True
        return False

    def get_active_crosswalk_state(self) -> Dict[str, dict]:
        """
        Returns current per-crosswalk pedestrian occupancy for external verifiers.
        """
        state = {
            cid: {
                "polygon": spec.polygon.copy(),
                "center": spec.center.copy(),
                "pedestrian_count": 0,
                "pedestrian_positions": [],
                "pedestrian_ids": [],
                "active": False,
            }
            for cid, spec in self._crosswalks.items()
        }

        for scenario_id, obj_id in self._scenario_id_to_obj_id.items():
            obj = self.spawned_objects.get(obj_id, None)
            if obj is None:
                continue
            crosswalk_id = self._scenario_id_to_crosswalk_id.get(scenario_id, "")
            if crosswalk_id not in state:
                continue
            pos = np.asarray(obj.position[:2], dtype=np.float64)
            entry = state[crosswalk_id]
            entry["pedestrian_count"] += 1
            entry["pedestrian_positions"].append(pos.copy())
            entry["pedestrian_ids"].append(obj_id)
            if self._point_in_polygon(pos, entry["polygon"]) or self._distance_to_polygon(pos, entry["polygon"]) <= 0.2:
                entry["active"] = True

        return state

    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        if polygon is None or len(polygon) < 3:
            return False
        x = float(point[0])
        y = float(point[1])
        inside = False
        n = len(polygon)
        j = n - 1
        for i in range(n):
            xi, yi = float(polygon[i][0]), float(polygon[i][1])
            xj, yj = float(polygon[j][0]), float(polygon[j][1])
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            if intersect:
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _distance_to_polygon(point: np.ndarray, polygon: np.ndarray) -> float:
        if polygon is None or len(polygon) == 0:
            return float("inf")
        if CrosswalkPedestrianManager._point_in_polygon(point, polygon):
            return 0.0
        p = np.asarray(point[:2], dtype=np.float64)
        n = len(polygon)
        return min(
            CrosswalkPedestrianManager._distance_point_to_segment(
                p,
                np.asarray(polygon[i][:2], dtype=np.float64),
                np.asarray(polygon[(i + 1) % n][:2], dtype=np.float64),
            )
            for i in range(n)
        )

    @staticmethod
    def _distance_point_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-9:
            return float(np.linalg.norm(point - a))
        t = float(np.dot(point - a, ab) / denom)
        t = min(1.0, max(0.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(point - proj))
