"""Pure MetaDrive loop predicate — no simulator import."""

from traffic_bench.eval.signs.dual_path.route_probe import (
    metadrive_route_loops,
    resolve_route_probe_targets,
)


def test_metadrive_route_loops_matches_eval_predicate():
    spawn = "lane_a_0"
    dest = "lane_b_0"
    assert metadrive_route_loops([spawn, dest], spawn) is False
    assert metadrive_route_loops([spawn], spawn) is True
    assert metadrive_route_loops([spawn, spawn], spawn) is True
    assert metadrive_route_loops([dest, spawn], spawn) is True
    assert metadrive_route_loops([], spawn) is True
    assert metadrive_route_loops([spawn, dest], None) is True


def test_resolve_route_probe_targets_direction_family():
    one = resolve_route_probe_targets("direction/straight")
    assert len(one) == 1
    sign_id, scenes, pdd = one[0]
    assert sign_id == "direction/straight"
    assert scenes.name == "direction_straight"
    assert pdd == "4.1.1"

    family = resolve_route_probe_targets("direction")
    assert [t[0] for t in family] == [
        "direction/straight",
        "direction/right",
        "direction/left",
        "direction/straight_right",
        "direction/straight_left",
        "direction/left_right",
    ]
    assert [t[2] for t in family] == [
        "4.1.1",
        "4.1.2",
        "4.1.3",
        "4.1.4",
        "4.1.5",
        "4.1.6",
    ]
    assert [t[1].name for t in family] == [
        "direction_straight",
        "direction_right",
        "direction_left",
        "direction_straight_right",
        "direction_straight_left",
        "direction_left_right",
    ]
