"""Capped destination arrive for same-edge no-split routes (crosswalk/detour)."""

from __future__ import annotations

from types import SimpleNamespace

from traffic_bench.eval.run.env import _ego_reached_capped_destination


def _vehicle(*, along: float, lane_len: float = 300.0, same_cp: bool = True):
    lane = SimpleNamespace(
        index="lane_edge_0",
        length=lane_len,
        local_coordinates=lambda pos: (along, 0.0),
    )
    final = SimpleNamespace(index="lane_edge_0", length=lane_len)
    checkpoints = (
        ["lane_edge_0", "lane_edge_0"]
        if same_cp
        else ["lane_edge_0", "lane_other_0"]
    )
    nav = SimpleNamespace(final_lane=final, checkpoints=checkpoints)
    return SimpleNamespace(navigation=nav, lane=lane, position=(0.0, 0.0))


def test_same_lane_cap_blocked_without_allow_same_lane():
    v = _vehicle(along=165.0)
    assert not _ego_reached_capped_destination(v, max_along_m=165.0, allow_same_lane=False)


def test_same_lane_cap_allowed_for_crosswalk_no_split():
    v = _vehicle(along=165.0)
    assert _ego_reached_capped_destination(v, max_along_m=165.0, allow_same_lane=True)
    # Still short of the cap.
    v_short = _vehicle(along=100.0)
    assert not _ego_reached_capped_destination(
        v_short, max_along_m=165.0, allow_same_lane=True
    )


def test_same_lane_cap_tol_and_past_cap():
    v = _vehicle(along=164.0)
    assert _ego_reached_capped_destination(
        v, max_along_m=165.0, allow_same_lane=True, arrive_tol_m=2.0
    )
    v_past = _vehicle(along=200.0)
    assert _ego_reached_capped_destination(
        v_past, max_along_m=165.0, allow_same_lane=True
    )
