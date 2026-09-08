"""Kinematic ring stream of lane users (buses / cyclists) on a reserved lane.

Pure Python, no MetaDrive: ``ReservedLaneAgentManager`` owns the bodies, this
class owns the longitudes. The users move along a ring of circumference
``circ = max(stretch + headway, n * headway)``; the first ``stretch`` metres of
the ring are the visible lane stretch (from the entry point towards the exit),
the rest is "parked" off-scene. So

* at most ``floor(stretch / headway) + 1`` users are on the lane at once,
* a user re-enters the lane only after a full lap, i.e. at least one headway
  behind the user ahead,
* a faster user never overtakes a slower one: it closes up to one headway
  behind it and then matches its speed (leader clamp),
* re-entry can be held by an ``entry_clear`` gate (a car or the ego on the entry
  spot) and the followers queue up behind the held user automatically.

``r`` is the ring position of each user in metres from the entry point along the
travel direction; the manager maps it to a lane longitude ``s = entry_s + dir*r``.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional

import numpy as np


def ego_travel_time_s(
    distance_m: float,
    v0_ms: float,
    *,
    accel_ms2: float = 1.5,
    cruise_ms: float = 10.0,
) -> float:
    """Time for an ego that starts at ``v0`` and accelerates to a cruise speed."""
    d = max(0.0, float(distance_m))
    if d <= 0.0:
        return 0.0
    v0 = max(0.0, float(v0_ms))
    vc = max(v0, float(cruise_ms))
    a = max(1e-3, float(accel_ms2))
    d_acc = (vc * vc - v0 * v0) / (2.0 * a)
    if d <= d_acc:
        return (-v0 + math.sqrt(v0 * v0 + 2.0 * a * d)) / a
    return (vc - v0) / a + (d - d_acc) / vc


class LaneStream:
    """Ring positions and speeds of ``n`` lane users."""

    def __init__(
        self,
        *,
        n: int,
        v_nom_ms: float,
        headway_m: float,
        min_gap_m: float,
        body_len_m: float,
        stretch_m: float,
        speed_jitter: float = 0.15,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        # n == 0 is the empty-lane counterfactual: the plate stands, nobody drives.
        self.n = max(0, int(n))
        rng = rng if rng is not None else np.random.RandomState(0)
        self.body_len_m = float(max(0.1, body_len_m))
        self.min_gap_m = float(max(0.0, min_gap_m))
        self.headway_m = float(max(headway_m, self.body_len_m + self.min_gap_m))
        # Followers keep a full headway behind their leader (a faster user closes
        # up to it and then matches speed), so the stream never compresses.
        self.follow_gap_m = self.headway_m - self.body_len_m
        self.stretch_m = float(max(1.0, stretch_m))
        self.circ_m = max(self.stretch_m + self.headway_m, self.n * self.headway_m)
        j = float(max(0.0, speed_jitter))
        self.v = np.asarray(
            [float(v_nom_ms) * float(rng.uniform(1.0 - j, 1.0 + j)) for _ in range(self.n)],
            dtype=float,
        )
        self.r = np.zeros(self.n, dtype=float)
        self.entries = 0        # how many times a user (re)entered the lane
        self.held_steps = 0     # how many user-steps were held at the entry gate

    # ------------------------------------------------------------- initial phases

    def init_spread(self, rng: np.random.RandomState) -> None:
        """Same flow: users evenly spread over the ring with one random phase."""
        if self.n == 0:
            return
        spacing = self.circ_m / float(self.n)
        phase = float(rng.uniform(0.0, spacing))
        self.r = np.asarray(
            [(phase + k * spacing) % self.circ_m for k in range(self.n)], dtype=float
        )

    def init_lead(self, lead_r0_m: float) -> None:
        """Counter flow: the first user at ``lead_r0`` (negative = still parked),
        the others one headway behind each other."""
        self.r = np.asarray(
            [float(lead_r0_m) - k * self.headway_m for k in range(self.n)], dtype=float
        )

    # --------------------------------------------------------------------- state

    def is_active(self, k: int) -> bool:
        return 0.0 <= float(self.r[k]) < self.stretch_m

    def active_indices(self) -> List[int]:
        return [k for k in range(self.n) if self.is_active(k)]

    def min_active_gap_m(self) -> Optional[float]:
        """Smallest bumper-to-bumper gap between users on the lane (None if < 2)."""
        rs = sorted(float(self.r[k]) for k in self.active_indices())
        if len(rs) < 2:
            return None
        return min(b - a - self.body_len_m for a, b in zip(rs, rs[1:]))

    # ---------------------------------------------------------------------- step

    def step(self, dt: float, entry_clear: Optional[Callable[[], bool]] = None) -> None:
        n = self.n
        r = self.r
        new_r = r.copy()
        order = sorted(range(n), key=lambda k: float(r[k]))  # ascending r: tail .. head
        gate: Optional[bool] = None

        def gate_ok() -> bool:
            nonlocal gate
            if gate is None:
                gate = True if entry_clear is None else bool(entry_clear())
            return gate

        for i in range(n - 1, -1, -1):  # head first, so followers see updated leaders
            k = order[i]
            if n == 1:
                lead_r = float(r[k]) + self.circ_m
            elif i == n - 1:
                lead_r = float(r[order[0]]) + self.circ_m  # wrap-around leader (old pos)
            else:
                lead_r = float(new_r[order[i + 1]])
            step = float(self.v[k]) * float(dt)
            gap = lead_r - float(r[k]) - self.body_len_m
            step = min(step, max(0.0, gap - self.follow_gap_m))

            crossing: Optional[float] = None
            if float(r[k]) < 0.0 <= float(r[k]) + step:
                crossing = 0.0
            elif float(r[k]) < self.circ_m <= float(r[k]) + step:
                crossing = self.circ_m
            if crossing is not None:
                if gate_ok():
                    self.entries += 1
                    gate = False  # one entry per step
                else:
                    step = min(step, max(0.0, crossing - 1e-3 - float(r[k])))
                    self.held_steps += 1

            nr = float(r[k]) + step
            if nr >= self.circ_m:
                nr -= self.circ_m
            new_r[k] = nr
        self.r = new_r
