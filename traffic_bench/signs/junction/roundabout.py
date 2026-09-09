import numpy as np

from traffic_bench.signs.base import BaseTrafficSign
from traffic_bench.signs.junction.yield_sign import YieldSign


class RoundaboutSign(BaseTrafficSign):
    """Sign 4.3 — roundabout ahead (informational plate on the approach)."""

    def __init__(self, lane, intersection_name: str = None, **kwargs):
        super().__init__(lane, icon_path="roundabout.png", **kwargs)
        self.intersection_name = intersection_name
        self.is_priority_sign = True
        self.priority_type = "roundabout_ahead"

    def _is_violating(self, vehicle) -> bool:
        return False

    def get_rule_description(self) -> str:
        return "Roundabout ahead (4.3) — yield to traffic on the circle"

    @property
    def top_down_color(self):
        return [255, 204, 0]

    @property
    def top_down_color_name(self):
        return "yellow"


class RoundaboutYieldSign(YieldSign):
    """Invisible yield tracker for 4.3 — ego on spoke yields to ring traffic."""

    ENTRY_CONFLICT_BEFORE_M = 20.0
    ENTRY_CONFLICT_AFTER_M = 5.0
    # Straight-line path approximations used only for the sticky meet XY.
    EGO_RAY_AHEAD_M = 40.0
    FOE_RAY_AHEAD_M = 30.0
    MEET_LATERAL_TOL_M = 2.5

    def __init__(
        self,
        lane,
        intersection_name: str = None,
        ring_road_lanes: list = None,
        entry_incoming_lanes: list = None,
        entry_junction_xy: tuple[float, float] | list[float] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("show_model", False)
        kwargs["icon_path"] = None
        incoming = list(entry_incoming_lanes or [])
        conflict_lanes = incoming
        if not conflict_lanes:
            conflict_lanes = list(ring_road_lanes or [])
        super().__init__(
            lane,
            intersection_name=intersection_name,
            main_road_lanes=conflict_lanes,
            auto_detect_main_roads=False,
            **kwargs,
        )
        self.priority_type = "roundabout_yield"
        self.icon_path = None
        if entry_junction_xy is not None:
            self._entry_junction_xy = np.array(
                [float(entry_junction_xy[0]), float(entry_junction_xy[1])],
                dtype=np.float64,
            )
        else:
            self._entry_junction_xy = None
        # Locked once: all lane pieces on the nearest conflict edge to ego.
        self._active_main_zones: list[dict] | None = None
        self._main_zone_locked: bool = False
        # Legacy single-piece alias (first locked lane); prefer _active_main_zones.
        self._active_main_zone: dict | None = None
        # Foes that entered the locked main edge at least once (post-exit meet).
        self._main_seen_foes: set = set()

    def _road_key(self, lane) -> str | None:
        """SUMO/MetaDrive edge id shared by parallel lanes on one road.

        Must NOT use ``(index[0], index[1])`` on string lane keys: for
        ``\"lane_...\"`` that collapses to ``('l', 'a')`` and merges every
        conflict chunk into one fake road (all yellow ribbons + false pink).
        """
        return self._edge_id_from_lane_index(getattr(lane, "index", None))

    def _junction_at_lane_end(self, lane) -> bool:
        """True when the entry junction is at ``lane.length`` (SUMO to-node end)."""
        if self._entry_junction_xy is None:
            return True
        try:
            p0 = np.array(lane.position(0.0, 0.0), dtype=np.float64)
            p1 = np.array(lane.position(float(lane.length), 0.0), dtype=np.float64)
        except Exception:
            return True
        return float(np.linalg.norm(p1 - self._entry_junction_xy)) <= float(
            np.linalg.norm(p0 - self._entry_junction_xy)
        )

    def _conflict_longitudinal_range(self, lane) -> tuple[float, float]:
        """Arc around the ring-side entry, not always the raw lane end.

        When ``entry_junction_xy`` is set, center the window on the closest
        longitudinal sample to that point so long/split ring edges still count
        when aux is near the physical entry mid-lane.
        """
        before_m = self.ENTRY_CONFLICT_BEFORE_M
        after_m = self.ENTRY_CONFLICT_AFTER_M
        length = float(getattr(lane, "length", 0.0) or 0.0)
        anchor = length
        if self._entry_junction_xy is not None and length > 1e-3:
            try:
                from traffic_bench.eval.engine.map.roundabout_yield_zone import (
                    closest_long_on_lane,
                )

                closest = closest_long_on_lane(lane, self._entry_junction_xy)
                if closest is not None:
                    anchor = float(closest[0])
            except Exception:
                anchor = length
        return (
            max(0.0, anchor - before_m),
            anchor + after_m,
        )

    def _all_main_zone_pieces(self) -> list[dict]:
        """Every main-arc lane segment (may be several chunks on the ring)."""
        zones: list[dict] = []
        for lane in self.main_road_lanes or []:
            try:
                long_start, long_end = self._conflict_longitudinal_range(lane)
                zones.append(
                    {
                        "lane": lane,
                        "long_start": float(long_start),
                        "long_end": float(long_end),
                        "kind": "incoming",
                    }
                )
            except Exception:
                continue
        return zones

    def _zone_head_xy(
        self, lane, long_start: float, long_end: float
    ) -> np.ndarray | None:
        """Downstream tip of a zone piece (toward the ego entry)."""
        try:
            length = float(getattr(lane, "length", 0.0) or 0.0)
            ls = max(0.0, min(float(long_start), length))
            le = max(0.0, min(float(long_end), length))
            if le < ls:
                ls, le = le, ls
            if self._entry_junction_xy is not None:
                p0 = np.asarray(lane.position(ls, 0.0), dtype=float)[:2]
                p1 = np.asarray(lane.position(le, 0.0), dtype=float)[:2]
                entry = np.asarray(self._entry_junction_xy, dtype=float)[:2]
                if float(np.linalg.norm(p0 - entry)) <= float(np.linalg.norm(p1 - entry)):
                    return p0
                return p1
            long_at = le if self._junction_at_lane_end(lane) else ls
            return np.asarray(lane.position(long_at, 0.0), dtype=float)[:2]
        except Exception:
            return None

    def _nearest_main_road_key(self, ego_vehicle) -> str | None:
        """Edge id whose zone-head is closest to ego (pick once)."""
        ego_xy = self._xy(ego_vehicle)
        if ego_xy is None:
            return None
        best_key: str | None = None
        best_d = float("inf")
        for zone in self._all_main_zone_pieces():
            lane = zone.get("lane")
            key = self._road_key(lane)
            if not key:
                continue
            head = self._zone_head_xy(
                lane, zone["long_start"], zone["long_end"]
            )
            if head is None:
                continue
            dist = float(np.linalg.norm(np.asarray(head, dtype=float) - ego_xy))
            if dist < best_d:
                best_d = dist
                best_key = str(key)
        return best_key

    def _sibling_lanes_for_edge(self, edge_id: str) -> list:
        """All map lanes on ``edge_id`` (parallel lanes + seeds from main_road)."""
        wanted = str(edge_id)
        out: list = []
        seen: set = set()

        def _add(lane) -> None:
            if lane is None:
                return
            idx = getattr(lane, "index", None)
            token = str(idx) if idx is not None else f"id:{id(lane)}"
            if token in seen:
                return
            if str(self._road_key(lane) or "") != wanted:
                return
            seen.add(token)
            out.append(lane)

        for lane in self.main_road_lanes or []:
            _add(lane)

        try:
            rn = self.engine.current_map.road_network
            graph = rn.graph
            sample = next(iter(graph.values()), None) if graph else None
            if isinstance(sample, dict):
                # NodeRoadNetwork: graph[from][to] -> [lanes]
                for to_dict in graph.values():
                    if not isinstance(to_dict, dict):
                        continue
                    for lanes in to_dict.values():
                        for lane in lanes or []:
                            _add(lane)
            else:
                # SUMO-style flat graph: lane_key -> LaneInfo
                for lane_key in list(graph.keys()):
                    if str(self._edge_id_from_lane_index(lane_key) or "") != wanted:
                        continue
                    try:
                        _add(rn.get_lane(lane_key))
                    except Exception:
                        continue
        except Exception:
            pass
        return out

    def _zones_for_road_key(self, road_key: str) -> list[dict]:
        """All parallel-lane conflict pieces on one ring edge."""
        by_id: dict[str, dict] = {}
        for lane in self._sibling_lanes_for_edge(str(road_key)):
            try:
                long_start, long_end = self._conflict_longitudinal_range(lane)
                idx = getattr(lane, "index", None)
                token = str(idx) if idx is not None else f"id:{id(lane)}"
                by_id[token] = {
                    "lane": lane,
                    "long_start": float(long_start),
                    "long_end": float(long_end),
                    "kind": "incoming",
                }
            except Exception:
                continue
        return list(by_id.values())

    def _ensure_active_main_zones(self, ego_vehicle) -> list[dict]:
        """Lock nearest conflict edge (all its lanes) on the first successful pick."""
        if self._main_zone_locked and self._active_main_zones is not None:
            return self._active_main_zones
        key = self._nearest_main_road_key(ego_vehicle)
        if key is None:
            return list(self._active_main_zones or [])
        zones = self._zones_for_road_key(key)
        if not zones:
            return list(self._active_main_zones or [])
        self._active_main_zones = zones
        self._active_main_zone = zones[0]
        self._main_zone_locked = True
        return zones

    def _set_active_main_zone(self, ego_vehicle) -> dict | None:
        """Compat: ensure locked zones; return first lane piece."""
        zones = self._ensure_active_main_zones(ego_vehicle)
        self._active_main_zone = zones[0] if zones else None
        return self._active_main_zone

    def _vehicle_in_zone_piece(self, vehicle, zone: dict) -> bool:
        """True when vehicle is on this zone's road and inside the long window.

        Strict: must share the zone road ``(from, to)``. No off-road projection
        onto the zone lane (that falsely painted pink rays on distant NPCs).
        Parallel lanes on the same edge use their own lane coordinates with
        this piece's longitudinal window (windows match across the edge).
        """
        if not zone:
            return False
        zone_lane = zone.get("lane")
        if zone_lane is None:
            return False
        zone_key = self._road_key(zone_lane)
        if zone_key is None:
            return False
        try:
            vehicle_pos = vehicle.position
            vehicle_lane = getattr(vehicle, "lane", None)
            if vehicle_lane is None:
                return False
            v_key = self._road_key(vehicle_lane)
            if v_key != zone_key:
                return False
            zone_start = float(zone["long_start"])
            zone_end = float(zone["long_end"])
            long_pos, lat_pos = vehicle_lane.local_coordinates(vehicle_pos)
            half_w = float(getattr(vehicle_lane, "width", 3.5) or 3.5) * 1.5
            return zone_start <= float(long_pos) <= zone_end and abs(float(lat_pos)) <= half_w
        except Exception:
            return False

    def _is_vehicle_in_main_road_conflict_zone(self, vehicle) -> bool:
        """True inside any locked lane piece of the fixed nearest conflict edge."""
        zones = self._active_main_zones
        if not zones:
            return False
        return any(self._vehicle_in_zone_piece(vehicle, z) for z in zones)

    def get_top_down_aux_conflict_zones(self) -> list[dict]:
        """Draw every locked lane piece on the fixed nearest conflict edge."""
        zones = self._active_main_zones or []
        return [
            {
                "lane": z["lane"],
                "long_start": float(z["long_start"]),
                "long_end": float(z["long_end"]),
                "kind": "incoming",
            }
            for z in zones
            if z.get("lane") is not None
        ]

    def get_rule_description(self) -> str:
        return (
            "Roundabout (4.3) — must not leave the approach zone while traffic "
            "is present in the nearest main conflict edge "
            f"({self.ENTRY_CONFLICT_BEFORE_M:.0f} m upstream of the ego entry, "
            f"plus {self.ENTRY_CONFLICT_AFTER_M:.0f} m past), locked once at "
            "episode start for all parallel lanes on that edge. A foe is "
            "conflicting while inside that edge, or after exit while an "
            "approaching ego×aux heading-ray meet still exists."
        )

    def _entry_conflict_point(self) -> np.ndarray | None:
        """Geometric entry XY (debug marker only; not used to arm yield)."""
        if self._entry_junction_xy is not None:
            return np.asarray(self._entry_junction_xy, dtype=float)
        for lane in self.main_road_lanes or []:
            try:
                long_at = (
                    float(lane.length)
                    if self._junction_at_lane_end(lane)
                    else 0.0
                )
                p = lane.position(long_at, 0.0)
                return np.asarray([float(p[0]), float(p[1])], dtype=float)
            except Exception:
                continue
        return None

    def _heading_ray(self, vehicle, ahead_m: float) -> list[np.ndarray]:
        """Straight forward approximation of the vehicle path."""
        origin = self._xy(vehicle)
        if origin is None:
            return []
        try:
            heading = float(vehicle.heading_theta)
        except Exception:
            return [origin]
        ahead = max(0.0, float(ahead_m))
        tip = origin + ahead * np.asarray(
            [np.cos(heading), np.sin(heading)], dtype=float
        )
        return [origin, tip]

    @staticmethod
    def _segments_intersect(
        a0: np.ndarray,
        a1: np.ndarray,
        b0: np.ndarray,
        b1: np.ndarray,
    ) -> np.ndarray | None:
        """Exact 2D segment–segment intersection, or None if they miss."""
        a0 = np.asarray(a0, dtype=float)
        a1 = np.asarray(a1, dtype=float)
        b0 = np.asarray(b0, dtype=float)
        b1 = np.asarray(b1, dtype=float)
        r = a1 - a0
        s = b1 - b0
        rxs = float(r[0] * s[1] - r[1] * s[0])
        if abs(rxs) < 1e-12:
            return None
        q_p = b0 - a0
        t = float(q_p[0] * s[1] - q_p[1] * s[0]) / rxs
        u = float(q_p[0] * r[1] - q_p[1] * r[0]) / rxs
        if t < -1e-9 or t > 1.0 + 1e-9 or u < -1e-9 or u > 1.0 + 1e-9:
            return None
        t = max(0.0, min(1.0, t))
        return a0 + t * r

    @staticmethod
    def _closest_points_on_segments(
        a0: np.ndarray,
        a1: np.ndarray,
        b0: np.ndarray,
        b1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Closest points on two finite segments and distance between them."""
        a0 = np.asarray(a0, dtype=float)
        a1 = np.asarray(a1, dtype=float)
        b0 = np.asarray(b0, dtype=float)
        b1 = np.asarray(b1, dtype=float)
        u = a1 - a0
        v = b1 - b0
        w0 = a0 - b0
        a = float(np.dot(u, u))
        b = float(np.dot(u, v))
        c = float(np.dot(v, v))
        d = float(np.dot(u, w0))
        e = float(np.dot(v, w0))
        denom = a * c - b * b

        if a < 1e-12 and c < 1e-12:
            pa, pb = a0, b0
            return pa, pb, float(np.linalg.norm(pa - pb))
        if a < 1e-12:
            s = max(0.0, min(1.0, e / c if c > 1e-12 else 0.0))
            t = 0.0
        elif c < 1e-12:
            t = max(0.0, min(1.0, -d / a if a > 1e-12 else 0.0))
            s = 0.0
        elif abs(denom) < 1e-12:
            t = max(0.0, min(1.0, -d / a if a > 1e-12 else 0.0))
            s = max(0.0, min(1.0, (b * t + e) / c if c > 1e-12 else 0.0))
        else:
            t = (b * e - c * d) / denom
            s = (a * e - b * d) / denom
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, e / c if c > 1e-12 else 0.0))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b + e) / c if c > 1e-12 else 0.0))
            if s < 0.0:
                s = 0.0
                t = max(0.0, min(1.0, -d / a if a > 1e-12 else 0.0))
            elif s > 1.0:
                s = 1.0
                t = max(0.0, min(1.0, (b - d) / a if a > 1e-12 else 0.0))

        pa = a0 + t * u
        pb = b0 + s * v
        return pa, pb, float(np.linalg.norm(pa - pb))

    def _ray_meet_point(
        self,
        ego_vehicle,
        foe_vehicle,
        *,
        ego_ray: list[np.ndarray] | None = None,
        foe_ray: list[np.ndarray] | None = None,
    ) -> np.ndarray | None:
        """Meet XY from heading rays only if both still approach the cross.

        Exact segment cross, else nearest approach within lateral tol. Rejects
        meets behind either vehicle (aux already passed / diverging) so a
        geometric cross of remaining rays is not enough.
        """
        ray_e = ego_ray if ego_ray is not None else self._heading_ray(
            ego_vehicle, self.EGO_RAY_AHEAD_M
        )
        ray_f = foe_ray if foe_ray is not None else self._heading_ray(
            foe_vehicle, self.FOE_RAY_AHEAD_M
        )
        if len(ray_e) < 2 or len(ray_f) < 2:
            return None
        a0, a1 = ray_e[0], ray_e[-1]
        b0, b1 = ray_f[0], ray_f[-1]
        hit = self._segments_intersect(a0, a1, b0, b1)
        if hit is None:
            pa, pb, dist = self._closest_points_on_segments(a0, a1, b0, b1)
            if dist > float(self.MEET_LATERAL_TOL_M) + 1e-9:
                return None
            hit = 0.5 * (pa + pb)
        meet = np.asarray(hit, dtype=float)
        if not self._both_approaching_meet(ego_vehicle, foe_vehicle, meet):
            return None
        return meet

    @staticmethod
    def _along_heading_to_point(vehicle, point: np.ndarray) -> float | None:
        """Signed metres from vehicle XY to ``point`` along vehicle heading."""
        try:
            pos = vehicle.position
            origin = np.asarray([float(pos[0]), float(pos[1])], dtype=float)
            heading = float(vehicle.heading_theta)
        except Exception:
            return None
        forward = np.asarray([np.cos(heading), np.sin(heading)], dtype=float)
        return float(np.dot(np.asarray(point, dtype=float) - origin, forward))

    def _both_approaching_meet(
        self,
        ego_vehicle,
        foe_vehicle,
        meet_pt: np.ndarray,
        *,
        min_ahead_m: float = 0.5,
    ) -> bool:
        """True only when the meet still lies ahead of both vehicles.

        Filters cases where aux already passed the crossing but remaining
        forward rays still geometrically intersect.
        """
        along_ego = self._along_heading_to_point(ego_vehicle, meet_pt)
        along_foe = self._along_heading_to_point(foe_vehicle, meet_pt)
        if along_ego is None or along_foe is None:
            return False
        return float(along_ego) >= float(min_ahead_m) and float(along_foe) >= float(
            min_ahead_m
        )

    def _forget_foe(self, foe_id) -> None:
        self._path_sticky_foes.pop(foe_id, None)
        self._main_seen_foes.discard(foe_id)

    def _is_foe_blocking_ego(self, ego_vehicle, foe_vehicle) -> bool:
        """4.3 conflict: in locked main OR post-exit while ray-meet remains.

        1. Non-gated foe inside the locked nearest main edge → always blocking.
        2. After it leaves that edge: keep blocking while an approaching
           ego×aux heading-ray meet still exists; drop when the meet clears.
        3. Ray meet is also stored for the yellow GIF marker.
        """
        self._ensure_active_main_zones(ego_vehicle)

        if self._is_waiting_gated_aux(foe_vehicle):
            return False

        foe_id = getattr(foe_vehicle, "id", None)
        if foe_id is None:
            return False

        # Agents on the ego approach (yield zone) are not conflicting traffic.
        if self._is_vehicle_in_zone(foe_vehicle):
            self._forget_foe(foe_id)
            return False

        in_main = self._is_vehicle_in_main_road_conflict_zone(foe_vehicle)
        ego_ray = self._heading_ray(ego_vehicle, self.EGO_RAY_AHEAD_M)
        foe_ray = self._heading_ray(foe_vehicle, self.FOE_RAY_AHEAD_M)
        meet = self._ray_meet_point(
            ego_vehicle, foe_vehicle, ego_ray=ego_ray, foe_ray=foe_ray
        )

        if in_main:
            self._main_seen_foes.add(foe_id)
            if meet is not None:
                self._path_sticky_foes[foe_id] = {
                    "conflict_point": np.asarray(meet, dtype=float)
                }
            else:
                # Still conflicting by presence; clear stale X if rays miss.
                self._path_sticky_foes.pop(foe_id, None)
            return True

        # Left the main edge: only continue if we saw this foe in-main and
        # an approaching ray meet is still live.
        if foe_id not in self._main_seen_foes:
            self._forget_foe(foe_id)
            return False

        if meet is not None:
            self._path_sticky_foes[foe_id] = {
                "conflict_point": np.asarray(meet, dtype=float)
            }
            return True

        # Meet gone after exit → drop completely.
        self._forget_foe(foe_id)
        return False

    def get_top_down_path_conflict_overlay(self, ego_vehicle) -> dict:
        """GIF: yellow locked main; pink while in-main or post-exit meet."""
        if self._auto_detect and not self._pg_initialized:
            self._identify_main_roads()

        self._ensure_active_main_zones(ego_vehicle)
        ego_ray = self._heading_ray(ego_vehicle, self.EGO_RAY_AHEAD_M)
        foes: list[dict] = []
        for v in self._get_all_vehicles():
            if getattr(v, "id", None) == getattr(ego_vehicle, "id", None):
                continue
            if self._is_waiting_gated_aux(v):
                continue
            if self._is_vehicle_in_zone(v):
                continue

            foe_id = getattr(v, "id", None)
            in_main = bool(self._is_vehicle_in_main_road_conflict_zone(v))
            blocking = bool(self._is_foe_blocking_ego(ego_vehicle, v))
            # After blocking, in_main / sticky may have updated.
            in_main = bool(self._is_vehicle_in_main_road_conflict_zone(v))
            if not blocking:
                continue

            foe_ray = self._heading_ray(v, self.FOE_RAY_AHEAD_M)
            meet = self._ray_meet_point(
                ego_vehicle, v, ego_ray=ego_ray, foe_ray=foe_ray
            )
            sticky = self._path_sticky_foes.get(foe_id) or {}
            sticky_pt = sticky.get("conflict_point")
            draw_pt = meet if meet is not None else sticky_pt

            foes.append(
                {
                    "vehicle_id": foe_id,
                    "path": foe_ray,
                    "conflict_point": draw_pt,
                    "in_main_zone": in_main,
                    "blocking": True,
                    "waiting_gated": False,
                    "sticky": sticky_pt is not None,
                }
            )

        zones: list[dict] = []
        try:
            lane = getattr(self, "lane", None)
            if lane is not None:
                zones.append(
                    {
                        "lane": lane,
                        "long_start": float(getattr(self, "zone_start", 0.0)),
                        "long_end": float(
                            getattr(self, "zone_end", getattr(lane, "length", 0.0))
                        ),
                        "kind": "yield",
                    }
                )
        except Exception:
            pass
        try:
            zones.extend(list(self.get_top_down_aux_conflict_zones() or []))
        except Exception:
            pass

        return {
            "ego_path": ego_ray,
            "foes": foes,
            "zones": zones,
            "entry_point": None,
        }
