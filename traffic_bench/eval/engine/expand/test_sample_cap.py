"""Pre-build sampling helpers for manifest expansion."""

from traffic_bench.eval.engine.expand.manifest_expansion import (
    non_nominal_budget,
    sample_cap,
    shuffle_cap,
    shuffled_copy,
)


def test_shuffled_copy_is_deterministic():
    items = list(range(20))
    a = shuffled_copy(items, seed_key=("scene", "cap", 10))
    b = shuffled_copy(items, seed_key=("scene", "cap", 10))
    c = shuffled_copy(items, seed_key=("scene", "cap", 11))
    assert a == b
    assert a != c
    assert sorted(a) == items


def test_sample_cap_keeps_at_most_n():
    items = list(range(50))
    out = sample_cap(items, 10, seed_key=("s", 10))
    assert len(out) == 10
    assert set(out).issubset(set(items))


def test_non_nominal_budget_reserves_preserved():
    assert non_nominal_budget(10, 1) == 9
    assert non_nominal_budget(10, 0) == 10
    assert non_nominal_budget(None, 1) is None
    assert non_nominal_budget(3, 5) == 0


def test_shuffle_cap_preserves_nominal_after_early_sample():
    rows = [{"is_nominal": True, "id": "n"}] + [{"id": i} for i in range(20)]
    # Simulate early sample already under cap.
    out = shuffle_cap(rows, 10, seed_key=("s", 10))
    assert out[0]["is_nominal"] is True
    assert len(out) == 10
