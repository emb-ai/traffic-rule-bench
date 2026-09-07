"""LaneStream invariants: headway, no overlap, gated entry, counter-flow timing."""

import numpy as np

from traffic_bench.envs.lane_stream import LaneStream, ego_travel_time_s


def _stream(n=3, v=8.5, stretch=45.0, seed=1, **kw):
    rng = np.random.RandomState(seed)
    params = dict(n=n, v_nom_ms=v, headway_m=v * 6.0, min_gap_m=8.0, body_len_m=5.8,
                  stretch_m=stretch, speed_jitter=0.15, rng=rng)
    params.update(kw)
    return LaneStream(**params), rng


def test_gaps_never_below_min_gap_under_jitter():
    for seed in range(5):
        st, rng = _stream(n=3, seed=seed)
        st.init_spread(rng)
        for _ in range(5000):
            st.step(0.1)
            g = st.min_active_gap_m()
            assert g is None or g >= st.follow_gap_m - 1e-6, (seed, g)


def test_at_most_stretch_over_headway_plus_one_active():
    st, rng = _stream(n=3, stretch=45.0)  # headway 51 m > stretch: one at a time
    st.init_spread(rng)
    for _ in range(3000):
        st.step(0.1)
        assert len(st.active_indices()) <= 1


def test_reentry_is_at_least_one_headway_behind():
    st, rng = _stream(n=2, stretch=200.0, speed_jitter=0.0)
    st.init_spread(rng)
    seen = []
    for _ in range(4000):
        st.step(0.1)
        rs = sorted(float(x) for x in st.r)
        seen.append(rs[1] - rs[0])
    # ring distance between the two users stays >= headway (both ways round)
    for d in seen:
        assert min(d, st.circ_m - d) >= st.headway_m - 1e-6


def test_entry_gate_holds_and_releases():
    st, rng = _stream(n=1, stretch=40.0, speed_jitter=0.0)
    st.r = np.asarray([st.circ_m - 0.5])  # about to wrap onto the lane
    for _ in range(50):
        st.step(0.1, entry_clear=lambda: False)
        assert not st.is_active(0)
        assert st.r[0] < st.circ_m
    assert st.held_steps > 0 and st.entries == 0
    st.step(0.1, entry_clear=lambda: True)
    assert st.entries == 1
    for _ in range(5):
        st.step(0.1, entry_clear=lambda: True)
    assert st.is_active(0)


def test_followers_queue_behind_held_leader():
    st, rng = _stream(n=3, stretch=40.0, speed_jitter=0.0)
    st.init_lead(-1.0)  # all parked, lead about to enter
    for _ in range(400):
        st.step(0.1, entry_clear=lambda: False)
    rs = sorted(float(x) for x in st.r)
    assert all(x < 0.0 for x in rs)
    assert rs[2] - rs[1] >= st.headway_m - 1e-6
    assert rs[1] - rs[0] >= st.headway_m - 1e-6


def test_counter_flow_lead_reaches_meet_point_on_time():
    v_bus = 8.5
    meet_r = 100.0                     # ring distance entry -> meet point
    t_ego = ego_travel_time_s(90.0, 3.61)
    st, rng = _stream(n=2, v=v_bus, stretch=150.0, speed_jitter=0.0)
    st.init_lead(meet_r - v_bus * t_ego)
    steps = int(round(t_ego / 0.1))
    for _ in range(steps):
        st.step(0.1)
    assert abs(float(st.r[0]) - meet_r) < v_bus * 0.1 + 1e-6


def test_ego_travel_time_monotone_in_v0():
    ts = [ego_travel_time_s(90.0, v) for v in (3.61, 5.0, 7.75, 11.05)]
    assert all(a > b for a, b in zip(ts, ts[1:]))
    assert 7.0 < ts[0] < 12.0
