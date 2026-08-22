"""Crop-time dual-path geometry from ``meta.json``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from traffic_bench.eval.engine.spawn.scene_augmentation import SpawnScenario
from traffic_bench.eval.engine.map.lane_keys import lane_num_from_key, make_lane_key


@dataclass(frozen=True)
class DualPathScenario:
    """Ego approach + shared dest: short forbidden baseline, long compliant."""

    junction_id: str
    junction_center_xy: Tuple[float, float]
    ego_edge_id: str
    ego_lane_num: int
    dest_edge_id: str
    dest_lane_num: int
    turn_dir: str
    turn_first_exit: str
    straight_first_exit: str
    turn_path: Tuple[str, ...]
    straight_path: Tuple[str, ...]
    turn_length_m: float
    straight_length_m: float
    compliant_dir: str = "s"
    pdd_code: str = ""
    wrong_dir_edges: Tuple[str, ...] = ()

    @property
    def gain_m(self) -> float:
        return float(self.straight_length_m - self.turn_length_m)

    @property
    def baseline_dir(self) -> str:
        return self.turn_dir

    def to_meta_fields(self) -> dict:
        return {
            "road_id": self.ego_edge_id,
            "spawn_lane_num": self.ego_lane_num,
            "destination_edge_id": self.dest_edge_id,
            "destination_lane_id": make_lane_key(self.dest_edge_id, self.dest_lane_num),
            "dual_path": {
                "turn_dir": self.turn_dir,
                "baseline_dir": self.turn_dir,
                "compliant_dir": self.compliant_dir,
                "turn_first_exit": self.turn_first_exit,
                "straight_first_exit": self.straight_first_exit,
                "turn_path": list(self.turn_path),
                "straight_path": list(self.straight_path),
                "turn_length_m": self.turn_length_m,
                "straight_length_m": self.straight_length_m,
                "gain_m": self.gain_m,
                "wrong_dir_edges": list(self.wrong_dir_edges),
            },
        }


def dual_path_to_spawn_scenario(
    dp: DualPathScenario,
    *,
    ego_lane_num: Optional[int] = None,
) -> SpawnScenario:
    lane = int(dp.ego_lane_num if ego_lane_num is None else ego_lane_num)
    dest_key = make_lane_key(dp.dest_edge_id, dp.dest_lane_num)
    return SpawnScenario(
        ego_edge_id=dp.ego_edge_id,
        ego_lane_num=lane,
        ego_destination_edge_id=dp.dest_edge_id,
        ego_destination_lane_key=dest_key,
        aux_edge_id="",
        aux_lane_num=0,
        aux_destination_edge_id="",
        aux_destination_lane_key="",
        scenario_id=(
            f"dual_{dp.junction_id}_{dp.ego_edge_id}_{dp.dest_edge_id}"
            f"_{dp.turn_dir}_L{lane}"
        ),
    )


def ego_spawn_lane_nums_for_dual(
    dp: DualPathScenario,
    spawn_lanes: Sequence[Any],
    *,
    min_lane_length_m: float,
) -> List[int]:
    nums = sorted(
        {
            int(lane.lane_num)
            for lane in spawn_lanes
            if str(getattr(lane, "edge_id", "")) == str(dp.ego_edge_id)
            and float(getattr(lane, "length", 0.0) or 0.0) >= float(min_lane_length_m)
        }
    )
    if nums:
        return nums
    return [int(dp.ego_lane_num)]


def dual_path_scenario_from_meta(
    meta: dict,
    *,
    pdd_code: str,
) -> Optional[DualPathScenario]:
    """Rebuild the crop-time pick from moscow ``meta.json`` (no role filter)."""
    ego = meta.get("road_id")
    dest = meta.get("destination_edge_id")
    dp = meta.get("dual_path")
    if not ego or not dest or not isinstance(dp, dict):
        return None
    turn_path = tuple(str(e) for e in (dp.get("turn_path") or dp.get("baseline_path") or ()) if e)
    straight_path = tuple(
        str(e) for e in (dp.get("straight_path") or dp.get("compliant_path") or ()) if e
    )
    if not turn_path or not straight_path:
        return None
    center = meta.get("junction_center_xy") or (0.0, 0.0)
    try:
        center_xy = (float(center[0]), float(center[1]))
    except (TypeError, ValueError, IndexError):
        center_xy = (0.0, 0.0)
    dest_lane_id = str(meta.get("destination_lane_id") or "")
    dest_lane_num = lane_num_from_key(dest_lane_id) if dest_lane_id else 0
    baseline_dir = str(
        dp.get("baseline_dir") or dp.get("turn_dir") or meta.get("baseline_dir") or ""
    )
    compliant_dir = str(
        dp.get("compliant_dir") or meta.get("compliant_dir") or "s"
    )
    wrong = tuple(
        str(e)
        for e in (
            dp.get("wrong_dir_edges") or meta.get("background_excluded_edges") or ()
        )
        if e
    )
    return DualPathScenario(
        junction_id=str(meta.get("junction_id") or ""),
        junction_center_xy=center_xy,
        ego_edge_id=str(ego),
        ego_lane_num=int(meta.get("spawn_lane_num") or 0),
        dest_edge_id=str(dest),
        dest_lane_num=int(dest_lane_num),
        turn_dir=baseline_dir,
        turn_first_exit=str(
            dp.get("turn_first_exit") or dp.get("baseline_first_exit") or turn_path[0]
        ),
        straight_first_exit=str(
            dp.get("straight_first_exit")
            or dp.get("compliant_first_exit")
            or straight_path[0]
        ),
        turn_path=turn_path,
        straight_path=straight_path,
        turn_length_m=float(dp.get("turn_length_m") or dp.get("baseline_length_m") or 0.0),
        straight_length_m=float(
            dp.get("straight_length_m") or dp.get("compliant_length_m") or 0.0
        ),
        compliant_dir=compliant_dir,
        pdd_code=str(pdd_code),
        wrong_dir_edges=wrong,
    )


def roles_match(
    scenario: DualPathScenario,
    baseline_dirs: Sequence[str],
    compliant_dirs: Sequence[str],
) -> bool:
    return (
        str(scenario.turn_dir) in set(baseline_dirs)
        and str(scenario.compliant_dir) in set(compliant_dirs)
    )


def pick_meta_dual_path(
    scene_meta: Optional[dict],
    *,
    pdd_code: str,
    baseline_dirs: Sequence[str],
    compliant_dirs: Sequence[str],
    junction_ids: Optional[Sequence[str]] = None,
) -> List[DualPathScenario]:
    """Return the crop dual-path if roles match; otherwise empty."""
    if not scene_meta:
        return []
    scenario = dual_path_scenario_from_meta(scene_meta, pdd_code=pdd_code)
    if scenario is None:
        return []
    if not roles_match(scenario, baseline_dirs, compliant_dirs):
        return []
    if junction_ids:
        want = {str(j) for j in junction_ids if j}
        if want and scenario.junction_id and scenario.junction_id not in want:
            return []
    return [scenario]
