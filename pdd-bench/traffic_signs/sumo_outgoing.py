"""Shared SUMO outgoing-edge helpers for turn / one-way / direction signs.

NN policies often leave a junction on a different via-lane / lane index than
``lane.turns`` lists. Compare *edge* ids of real outgoing roads instead.
"""

from __future__ import annotations

import math


def normalize_turn_direction(raw_dir: str) -> str:
    d = str(raw_dir or "").strip().lower()
    if d in ("r", "right"):
        return "r"
    if d in ("l", "left"):
        return "l"
    if d in ("s", "straight"):
        return "s"
    if d in ("t", "u", "uturn", "u-turn"):
        return "t"
    return d


def is_internal_lane_id(lane_id) -> bool:
    if not isinstance(lane_id, str):
        return False
    raw = lane_id[5:] if lane_id.startswith("lane_") else lane_id
    return raw.startswith(":") or "junction_" in raw


def wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def heading_delta_to_dir(approach_heading: float, outgoing_heading: float) -> str:
    """Map heading change onto {l, r, s, t}. MetaDrive heading grows CCW."""
    delta = wrap_pi(outgoing_heading - approach_heading)
    ad = abs(delta)
    if ad < math.radians(35.0):
        return "s"
    if ad > math.radians(145.0):
        return "t"
    return "l" if delta > 0.0 else "r"


class SumoOutgoingMixin:
    """Requires BaseTrafficSign (``engine``, ``_sumo_edge_id_from_lane_index``, ``lane``)."""

    _is_internal_lane_id = staticmethod(is_internal_lane_id)
    _wrap_pi = staticmethod(wrap_pi)

    @staticmethod
    def _heading_delta_to_dir(approach_heading: float, outgoing_heading: float) -> str:
        return heading_delta_to_dir(approach_heading, outgoing_heading)

    def _real_outgoing_edges(self, lane_id, *, max_hops: int = 5) -> set:
        """Resolve a SUMO lane id to non-internal outgoing edge ids."""
        if not lane_id:
            return set()
        edge = self._sumo_edge_id_from_lane_index(lane_id)
        if edge and not str(edge).startswith(":"):
            return {str(edge)}
        try:
            road_network = self.engine.current_map.road_network
            graph = getattr(road_network, "graph", {}) or {}
        except Exception:
            return set()

        found = set()
        queue = [(lane_id, 0)]
        seen = {lane_id}
        while queue:
            cur, hops = queue.pop(0)
            cur_edge = self._sumo_edge_id_from_lane_index(cur)
            if cur_edge and not str(cur_edge).startswith(":"):
                found.add(str(cur_edge))
                continue
            if hops >= max_hops:
                continue
            info = graph.get(cur)
            lane_obj = getattr(info, "lane", None) if info is not None else None
            if lane_obj is None:
                try:
                    lane_obj = road_network.get_lane(cur)
                except Exception:
                    lane_obj = None
            exits = list(getattr(info, "exit_lanes", None) or []) if info is not None else []
            if not exits and lane_obj is not None:
                exits = list(getattr(lane_obj, "exit_lanes", None) or [])
            for nxt in exits:
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append((nxt, hops + 1))
        return found

    def _approach_heading(self):
        lane = getattr(self, "lane", None)
        if lane is None:
            return None
        try:
            s = max(0.1, float(lane.length) - 0.5)
            return float(lane.heading_theta_at(s))
        except Exception:
            return None

    def _edge_start_heading(self, edge_id: str):
        try:
            road_network = self.engine.current_map.road_network
            graph = getattr(road_network, "graph", {}) or {}
        except Exception:
            return None
        for key, info in graph.items():
            if self._sumo_edge_id_from_lane_index(key) != edge_id:
                continue
            if is_internal_lane_id(key):
                continue
            lane_obj = getattr(info, "lane", None)
            if lane_obj is None:
                continue
            try:
                return float(
                    lane_obj.heading_theta_at(
                        min(0.5, max(0.1, float(lane_obj.length) * 0.05))
                    )
                )
            except Exception:
                continue
        return None

    def _classify_outgoing_geometrically(self, edge_ids) -> dict:
        by_dir = {"l": set(), "r": set(), "s": set(), "t": set()}
        approach_h = self._approach_heading()
        if approach_h is None:
            return by_dir
        for edge_id in edge_ids:
            out_h = self._edge_start_heading(str(edge_id))
            if out_h is None:
                continue
            by_dir[heading_delta_to_dir(approach_h, out_h)].add(str(edge_id))
        return by_dir

    def _map_sumo_outgoing_from_lanes(self, approach_lanes) -> dict:
        """Build approach / per-dir / all outgoing SUMO edge sets from ``turns``."""
        approach_roads = set()
        by_dir = {"l": set(), "r": set(), "s": set(), "t": set()}
        unlabeled = set()
        for lane_obj in approach_lanes or []:
            lid = getattr(lane_obj, "index", None)
            road = self._sumo_edge_id_from_lane_index(lid)
            if road and not str(road).startswith(":"):
                approach_roads.add(str(road))
            for turn in getattr(lane_obj, "turns", None) or []:
                d = normalize_turn_direction(turn.get("direction"))
                edges = set()
                for key in ("to_lane", "via_lane"):
                    tgt = turn.get(key)
                    if tgt:
                        edges |= self._real_outgoing_edges(tgt)
                edges -= approach_roads
                if not edges:
                    continue
                if d in by_dir:
                    by_dir[d].update(edges)
                else:
                    unlabeled.update(edges)

        labeled = set().union(*by_dir.values()) if by_dir else set()
        unlabeled -= labeled
        if unlabeled:
            geo = self._classify_outgoing_geometrically(unlabeled)
            for d, edges in geo.items():
                by_dir[d].update(edges)

        all_outgoing = set().union(*by_dir.values()) if by_dir else set()
        return {
            "approach_roads": approach_roads,
            "by_dir": by_dir,
            "all_outgoing": all_outgoing,
        }

    def _forbidden_outgoing_edges(self, mapped: dict, prohibited_maneuver: str) -> set:
        by_dir = mapped.get("by_dir") or {}
        all_outgoing = set(mapped.get("all_outgoing") or ())
        forbidden = set(by_dir.get(prohibited_maneuver, ()))
        if not forbidden and all_outgoing:
            geo = self._classify_outgoing_geometrically(all_outgoing)
            forbidden = set(geo.get(prohibited_maneuver, ()))
        return forbidden

    def _allowed_outgoing_edges(self, mapped: dict, allowed_dirs) -> set:
        by_dir = mapped.get("by_dir") or {}
        allowed = set()
        for d in allowed_dirs or ():
            allowed |= set(by_dir.get(d, ()))
        if allowed:
            return allowed
        all_outgoing = set(mapped.get("all_outgoing") or ())
        if not all_outgoing:
            return set()
        geo = self._classify_outgoing_geometrically(all_outgoing)
        for d in allowed_dirs or ():
            allowed |= set(geo.get(d, ()))
        return allowed

    def _judge_sumo_outgoing(
        self,
        agent_id,
        current_lane,
        *,
        approach_roads,
        all_outgoing,
        violate_roads,
        states,
        cleared=None,
    ) -> bool:
        """Arm on approach; when ego lands on an outgoing road, violate if it is marked.

        Stays armed through internal junction hops and same-road lane changes.
        ``violate_roads`` is the forbidden (3.18 / 5.7) or disallowed (4.1) set.
        """
        if cleared is not None and agent_id in cleared:
            return False
        if is_internal_lane_id(current_lane):
            return False

        current_road = self._sumo_edge_id_from_lane_index(current_lane)
        state = states.setdefault(agent_id, {"armed": False, "last_road": None})

        if current_road is not None and current_road in set(approach_roads or ()):
            state["armed"] = True
            state["last_road"] = current_road
            return False

        if not state.get("armed"):
            return False

        if current_road is not None and current_road in set(all_outgoing or ()):
            state["armed"] = False
            state["last_road"] = current_road
            if cleared is not None:
                cleared.add(agent_id)
            return current_road in set(violate_roads or ())

        return False
