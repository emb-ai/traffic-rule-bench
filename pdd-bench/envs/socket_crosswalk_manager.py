from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from metadrive.manager.base_manager import BaseManager
from metadrive.type import MetaDriveType


class SocketCrosswalkManager(BaseManager):
    """
    Build synthetic crosswalks on socket ends for PG maps.
    """

    PRIORITY = 2
    CROSSWALK_LENGTH_M = 6.0
    CROSSWALK_HALF_THICKNESS_M = CROSSWALK_LENGTH_M / 2.0
    DEFAULT_SOCKET_MARGIN_M = 1.5
    MIN_LANE_LENGTH_M = 0.8
    MIN_SOCKET_SPAN_M = 1.0
    MIN_SAMPLE_MARGIN_M = 0.3
    DEFAULT_LANE_WIDTH_M = 3.5

    def __init__(self):
        super().__init__()
        raw_cfg = self.engine.global_config.get("crosswalk_manager", {})
        self._cfg = raw_cfg.get_dict() if hasattr(raw_cfg, "get_dict") else dict(raw_cfg)
        self.enabled = bool(self.engine.global_config.get("use_crosswalk_manager", False)) and bool(
            self._cfg.get("enabled", True)
        )
        self.num_crosswalks = int(self._cfg.get("num_crosswalks", -1))
        self.replace_existing = bool(self._cfg.get("replace_existing", True))
        self.random_select = bool(self._cfg.get("random_select", True))
        self.include_pre_block_socket = bool(self._cfg.get("include_pre_block_socket", True))
        self.include_all_sockets = bool(self._cfg.get("include_all_sockets", False))
        self.skip_first_block = bool(self._cfg.get("skip_first_block", True))
        self.socket_anchor = str(self._cfg.get("socket_anchor", "end")).lower()
        if self.socket_anchor not in {"start", "end"}:
            self.socket_anchor = "end"

        # Fixed crosswalk size along driving direction.
        self.half_thickness = float(self.CROSSWALK_HALF_THICKNESS_M)
        self.socket_margin = float(self._cfg.get("socket_margin", self.DEFAULT_SOCKET_MARGIN_M))
        self.debug_rejection_log = bool(self._cfg.get("debug_rejection_log", False))
        self.debug_max_rejections = int(self._cfg.get("debug_max_rejections", -1))

    def reset(self):
        if not self.enabled:
            return {}
        current_map = getattr(self.engine, "current_map", None)
        if current_map is None or not hasattr(current_map, "road_network"):
            return {}
        candidates, rejected = self._collect_candidates(current_map)
        if self.debug_rejection_log:
            self._print_rejection_log(candidates, rejected)
        selected = self._select_candidates(candidates)
        crosswalks = {}
        for idx, item in enumerate(selected):
            key = f"socket_cw_{idx}_{item['block_name']}"
            crosswalks[key] = {
                "type": MetaDriveType.CROSSWALK,
                "polygon": np.asarray(item["polygon"], dtype=np.float32),
                "meta": {
                    "block_name": item["block_name"],
                    "socket_index": item["socket_index"],
                    "socket_anchor": self.socket_anchor,
                },
            }
        current_map.crosswalks = crosswalks
        return {"crosswalk_manager": {"generated": len(selected), "total": len(crosswalks)}}

    def _collect_candidates(self, current_map):
        candidates = []
        rejected = []
        road_network = current_map.road_network
        for i, block in enumerate(getattr(current_map, "blocks", [])):
            if self.skip_first_block and i == 0:
                continue
            sockets = self._collect_block_sockets(block)
            for socket in sockets:
                block_name = str(getattr(block, "name", "blk"))
                socket_index = str(getattr(socket, "index", "sock"))
                polygon, reason = self._build_crosswalk_polygon(socket, road_network)
                if polygon is None:
                    rejected.append(
                        {
                            "block_name": block_name,
                            "socket_index": socket_index,
                            "reason": reason,
                        }
                    )
                    continue
                candidates.append(
                    {
                        "block_name": block_name,
                        "socket_index": socket_index,
                        "polygon": polygon,
                    }
                )
        return candidates, rejected

    def _collect_block_sockets(self, block) -> List:
        sockets = []
        if self.include_pre_block_socket:
            socket = getattr(block, "pre_block_socket", None)
            if socket is not None:
                sockets.append(socket)
        if self.include_all_sockets and hasattr(block, "get_socket_list"):
            sockets.extend(block.get_socket_list())
        return list({id(s): s for s in sockets}.values())

    def _select_candidates(self, candidates: List[dict]) -> List[dict]:
        if self.num_crosswalks == 0:
            return []
        if self.num_crosswalks < 0 or self.num_crosswalks >= len(candidates):
            return candidates
        n = int(self.num_crosswalks)
        if not self.random_select:
            return candidates[:n]
        ids = np.arange(len(candidates))
        chosen = self.np_random.choice(ids, size=n, replace=False)
        return [candidates[int(i)] for i in chosen]

    def _safe_lanes_for_road(self, road, road_network) -> List:
        if road is None:
            return []
        try:
            return list(road.get_lanes(road_network))
        except Exception:
            return []

    def _collect_lane_samples(self, pos_lanes: List, neg_lanes: List) -> List[Tuple[object, float]]:
        return (
            [(lane, self._sample_s(float(lane.length), is_positive=True)) for lane in pos_lanes]
            + [(lane, self._sample_s(float(lane.length), is_positive=False)) for lane in neg_lanes]
        )

    def _lane_width_at(self, lane, sample_s: float) -> float:
        try:
            return float(lane.width_at(sample_s))
        except Exception:
            return float(getattr(lane, "width", self.DEFAULT_LANE_WIDTH_M))

    @staticmethod
    def _sample_lane_edge_points(lane, sample_s: float, width: float) -> List[np.ndarray]:
        points = []
        for lat in (-width / 2.0, width / 2.0):
            try:
                pt = np.asarray(lane.position(sample_s, lat), dtype=np.float64)[:2]
            except Exception:
                continue
            points.append(pt)
        return points

    def _find_lateral_extremes(self, lane_samples: List[Tuple[object, float]], lateral: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        min_proj = float("inf")
        max_proj = float("-inf")
        min_pt = None
        max_pt = None
        for lane, sample_s in lane_samples:
            width = self._lane_width_at(lane, sample_s)
            for pt in self._sample_lane_edge_points(lane, sample_s, width):
                proj = float(np.dot(pt, lateral))
                if proj < min_proj:
                    min_proj = proj
                    min_pt = pt
                if proj > max_proj:
                    max_proj = proj
                    max_pt = pt
        return min_pt, max_pt

    def _get_socket_lanes(self, socket, road_network):
        pos_lanes = self._safe_lanes_for_road(getattr(socket, "positive_road", None), road_network)
        neg_lanes = self._safe_lanes_for_road(getattr(socket, "negative_road", None), road_network)
        lanes = pos_lanes + neg_lanes
        if not lanes:
            return None, None, "no_lanes"
        try:
            min_len = min(float(l.length) for l in lanes)
        except Exception:
            return None, None, "invalid_lane_length"
        if min_len < self.MIN_LANE_LENGTH_M:
            return None, None, "lane_too_short"
        return pos_lanes, neg_lanes, "ok"

    def _get_reference_heading(self, pos_lanes: List, neg_lanes: List) -> Optional[float]:
        try:
            if pos_lanes:
                ref_lane = pos_lanes[0]
                ref_s = self._sample_s(float(ref_lane.length), is_positive=True)
            else:
                ref_lane = neg_lanes[0]
                ref_s = self._sample_s(float(ref_lane.length), is_positive=False)
            return float(ref_lane.heading_theta_at(ref_s))
        except Exception:
            return None

    def _build_polygon_from_edges(self, min_pt: np.ndarray, max_pt: np.ndarray, forward: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                (min_pt - forward * self.half_thickness).tolist(),
                (min_pt + forward * self.half_thickness).tolist(),
                (max_pt + forward * self.half_thickness).tolist(),
                (max_pt - forward * self.half_thickness).tolist(),
            ],
            dtype=np.float64
        )

    def _build_crosswalk_polygon(self, socket, road_network) -> Tuple[Optional[np.ndarray], str]:
        pos_lanes, neg_lanes, lane_status = self._get_socket_lanes(socket, road_network)
        if lane_status != "ok":
            return None, lane_status

        heading = self._get_reference_heading(pos_lanes, neg_lanes)
        if heading is None:
            return None, "invalid_ref_lane"

        forward = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)
        lateral = np.asarray([-forward[1], forward[0]], dtype=np.float64)
        lane_samples = self._collect_lane_samples(pos_lanes, neg_lanes)
        min_pt, max_pt = self._find_lateral_extremes(lane_samples, lateral)

        if min_pt is None or max_pt is None:
            return None, "no_boundary_points"
        if float(np.linalg.norm(max_pt - min_pt)) < self.MIN_SOCKET_SPAN_M:
            return None, "too_narrow_socket"
        return self._build_polygon_from_edges(min_pt, max_pt, forward), "ok"

    def _sample_s(self, lane_length: float, is_positive: bool) -> float:
        margin = max(self.MIN_SAMPLE_MARGIN_M, float(self.socket_margin))
        max_s = max(self.MIN_SAMPLE_MARGIN_M, lane_length - self.MIN_SAMPLE_MARGIN_M)
        if self.socket_anchor == "end":
            target = lane_length - margin if is_positive else margin
        else:
            target = margin if is_positive else lane_length - margin
        return max(self.MIN_SAMPLE_MARGIN_M, min(max_s, target))

    def _print_rejection_log(self, accepted: List[dict], rejected: List[dict]) -> None:
        total = len(accepted) + len(rejected)
        print(f"[SocketCrosswalkManager] candidates={total} accepted={len(accepted)} rejected={len(rejected)}")
        if not rejected:
            return
        limit = len(rejected) if self.debug_max_rejections <= 0 else min(len(rejected), self.debug_max_rejections)
        for item in rejected[:limit]:
            print(f"[SocketCrosswalkReject] block={item.get('block_name', 'blk')} "
                  f"socket={item.get('socket_index', 'sock')} reason={item.get('reason', 'unknown')}")
        if limit < len(rejected):
            print(f"[SocketCrosswalkReject] ... {len(rejected) - limit} more")
