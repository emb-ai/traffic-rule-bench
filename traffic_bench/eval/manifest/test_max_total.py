"""Protocol-size max_total: n_train/n_test × max_scenarios, not hardcoded 200/800."""

from traffic_bench.eval.manifest.io import resolve_max_total, sign_split_quota


def test_quota_from_signs_yaml():
    assert sign_split_quota("3.18.2", "train") == 80
    assert sign_split_quota("3.18.2", "test") == 20
    assert sign_split_quota("3.18.2", "debug") is None


def test_resolve_derives_train_test_from_quota():
    assert resolve_max_total(None, max_scenarios=10, split="train", pdd_code="3.18.2") == 800
    assert resolve_max_total(None, max_scenarios=10, split="test", pdd_code="3.18.1") == 200
    assert resolve_max_total(None, max_scenarios=10, split="debug", pdd_code="3.18.2") is None


def test_explicit_max_total_wins():
    assert resolve_max_total(50, max_scenarios=10, split="train", pdd_code="3.18.2") == 50


def test_apply_scene_row_quota_keeps_unique_maps():
    from traffic_bench.eval.manifest.io import apply_scene_row_quota

    entries = [
        {"scene_name": f"m{s}", "i": r} for s in range(20) for r in range(12)
    ]
    out, used, pre = apply_scene_row_quota(
        entries,
        [f"m{s}" for s in range(20)],
        n_maps=20,
        max_scenarios=10,
        max_total=200,
        split="test",
        pdd_code="3.18.2",
    )
    assert pre == 240
    assert len(out) == 200
    assert len(used) == 20
    assert all(sum(1 for e in out if e["scene_name"] == sid) == 10 for sid in used)


def test_apply_scene_row_quota_does_not_drop_maps_when_short():
    from traffic_bench.eval.manifest.io import apply_scene_row_quota

    entries = [
        {"scene_name": f"m{s}", "i": r} for s in range(17) for r in range(10)
    ]
    out, used, pre = apply_scene_row_quota(
        entries,
        [f"m{s}" for s in range(17)],
        n_maps=20,
        max_scenarios=10,
        max_total=200,
        split="test",
        pdd_code="3.18.2",
    )
    assert pre == 170
    assert len(out) == 170
    assert len(used) == 17
