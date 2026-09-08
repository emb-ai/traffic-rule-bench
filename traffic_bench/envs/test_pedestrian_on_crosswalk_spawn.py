"""on_crosswalk spawn must keep a non-zero start→end when starting from either curb."""

from __future__ import annotations

import numpy as np

from traffic_bench.envs.pedestrians import CrosswalkPedestrianManager, _CrosswalkSpec


def _rect_spec(*, walk_len: float = 12.0, half_span: float = 2.0) -> _CrosswalkSpec:
    # Axis-aligned zebra: walk along +x, span along +y.
    poly = np.array(
        [
            [-walk_len / 2, -half_span],
            [walk_len / 2, -half_span],
            [walk_len / 2, half_span],
            [-walk_len / 2, half_span],
        ],
        dtype=np.float64,
    )
    walk_dir = np.array([1.0, 0.0], dtype=np.float64)
    span_dir = np.array([0.0, 1.0], dtype=np.float64)
    return _CrosswalkSpec(
        crosswalk_id="test_cw",
        polygon=poly,
        center=np.mean(poly, axis=0),
        walk_start=np.array([-walk_len / 2, 0.0], dtype=np.float64),
        walk_end=np.array([walk_len / 2, 0.0], dtype=np.float64),
        walk_dir=walk_dir,
        span_dir=span_dir,
        half_span=half_span,
        walk_length=walk_len,
    )


class _FakeMgr(CrosswalkPedestrianManager):
    """Skip BaseManager.__init__; only need schedule helpers + RNG."""

    def __init__(self, seed: int = 0):
        self.np_random = np.random.RandomState(seed)
        self._crosswalks = {"test_cw": _rect_spec()}
        self._tracks = {}
        self._counter = 0
        self.speed_mean = 1.0
        self.speed_std = 0.0
        self.wait_min = 0.0
        self.wait_max = 0.0
        self.spawn_jitter_steps = 1
        self.min_spawn_gap = 0.1
        self.spawned_objects = {}
        self._episode_step = 0

    @property
    def episode_step(self) -> int:
        return int(self._episode_step)

    def _sim_dt(self) -> float:
        return 0.1

    def _is_position_free(self, pos: np.ndarray) -> bool:
        return True


def test_on_crosswalk_schedule_both_sides_have_travel_distance():
    # Seeds that previously flipped start_from_left both ways.
    for seed in range(16):
        mgr = _FakeMgr(seed=seed)
        assert mgr._schedule_track("test_cw", on_crosswalk=True, immediate=True)
        track = next(iter(mgr._tracks.values()))
        valid = track["state"]["valid"]
        pos = track["state"]["position"]
        idxs = np.where(valid)[0]
        assert len(idxs) >= 2
        start = pos[idxs[0], :2]
        end = pos[idxs[-1], :2]
        dist = float(np.linalg.norm(end - start))
        assert dist > 1.0, f"seed={seed} collapsed track dist={dist}"
