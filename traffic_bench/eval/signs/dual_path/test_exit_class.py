"""Exit classification for dual-path episodes (no simulator import)."""

from traffic_bench.eval.signs.dual_path.nav import classify_dual_path_exit

# direction_right test row: the plate allows right, the default route turns left.
ROW = {
    "road_id": "115317673#5",
    "dual_path": {
        "turn_path": ["-115317612#0"],
        "straight_path": ["115317612#1"],
    },
}


def test_still_on_the_approach_or_inside_the_junction():
    assert classify_dual_path_exit("115317673#5", ROW) is None
    assert classify_dual_path_exit(":cluster_9_3", ROW) is None
    assert classify_dual_path_exit("", ROW) is None
    assert classify_dual_path_exit(None, ROW) is None


def test_allowed_forbidden_and_third_exit():
    assert classify_dual_path_exit("115317612#1", ROW) == "compliant"
    assert classify_dual_path_exit("-115317612#0", ROW) == "baseline"
    assert classify_dual_path_exit("-115317673#5", ROW) == "other"
