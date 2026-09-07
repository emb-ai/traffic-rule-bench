"""World-grid expansion of the reserved-lane family: fit rule, nominal row, cap, determinism."""

from pathlib import Path
from types import SimpleNamespace

from traffic_bench.eval.engine.map.segment_length import SegmentCorridor
from traffic_bench.eval.signs.restricted_lane import expand as ex


def _corridor(length: float) -> SegmentCorridor:
    return SegmentCorridor(
        road_id="e1", net_length_m=length, window_length_m=length, corridor_s0=0.0, corridor_s1=length, from_window=False
    )


def _meta():
    return {"scene_name": "seg_t", "road_id": "e1", "length_m": 300.0, "vehicle_lane_indices": [0, 1], "lane_count": 2}


def _sim(**kw):
    base = dict(zone_levels_m=(60.0, 80.0, 100.0), approach_levels_m=(60.0, 80.0, 100.0), agents_n_levels=(1, 2))
    base.update(kw)
    return ex.RestrictedLaneSimParams(**base)


def test_fit_rule():
    assert ex.geometry_fits(150.0, approach_m=60, zone_m=60, tail_m=10, spawn_offset_from_start=10)
    assert not ex.geometry_fits(150.0, approach_m=80, zone_m=60, tail_m=10, spawn_offset_from_start=10)
    assert not ex.geometry_fits(150.0, approach_m=60, zone_m=100, tail_m=10, spawn_offset_from_start=10)
    assert ex.geometry_fits(240.0, approach_m=100, zone_m=100, tail_m=10, spawn_offset_from_start=10)


def test_candidates_respect_fit_and_lc_rule():
    sim = _sim()
    cands = ex._family_candidates(usable_length_m=150.0, sim=sim, route_levels=[220.0])
    assert cands and all(z == 60.0 and a == 60.0 for _, z, _, a, _, _, _ in cands)
    # 11.05 m/s needs >= 66.3 m of approach -> excluded at 60 m
    assert all(c[6].spawn_velocity_ms < 11.0 for c in cands)
    big = ex._family_candidates(usable_length_m=300.0, sim=sim, route_levels=[220.0])
    assert len(big) == 9 * 2 * 27 - 3 * 2 * 9  # all fits minus (approach 60, sv 11.05)


def test_nominal_row_first_and_cap(tmp_path=Path("/tmp")):
    sim = _sim()
    scene_dir = Path("/tmp/scenes/seg_t")
    meta = _meta()
    corridor = _corridor(300.0)
    row = ex.build_restricted_lane_entry(
        scene_dir=scene_dir, scenes_root=Path("/tmp/scenes"), meta=meta, sim=sim, pdd_code="5.14.1",
        corridor=corridor, default_variant=True, variant=0, zone_m=60.0, approach_m=60.0,
        max_path_length_m=220.0, spawn_velocity_ms=5.0, traffic_density=0.0,
    )
    assert row and row["sign_s"] == 230.0 and row["spawn_offset_from_start"] == 170.0
    assert row["zone_end_s"] == 290.0 and row["destination_max_along_m"] == 295.0
    assert row["reserved_agents_n"] == 1 and row["scene_id"] == "seg_t_v0"
    # unfit request -> {}
    assert ex.build_restricted_lane_entry(
        scene_dir=scene_dir, scenes_root=Path("/tmp/scenes"), meta=meta, sim=sim, pdd_code="5.14.1",
        corridor=_corridor(150.0), default_variant=False, variant=1, zone_m=100.0, approach_m=100.0,
        max_path_length_m=220.0,
    ) == {}
    # short budget -> {}
    assert ex.build_restricted_lane_entry(
        scene_dir=scene_dir, scenes_root=Path("/tmp/scenes"), meta=meta, sim=sim, pdd_code="5.14.1",
        corridor=corridor, default_variant=False, variant=1, zone_m=100.0, approach_m=100.0,
        max_path_length_m=150.0,
    ) == {}


def test_jitter_is_upstream_and_deterministic():
    sim = _sim()
    kw = dict(scene_dir=Path("/tmp/scenes/seg_t"), scenes_root=Path("/tmp/scenes"), meta=_meta(), sim=sim,
              pdd_code="5.11.2", corridor=_corridor(300.0), default_variant=False, variant=2, zone_m=80.0,
              approach_m=80.0, max_path_length_m=220.0, scene_id_suffix="z80a80n2_td50_sv1_v2")
    a = ex.build_restricted_lane_entry(**kw)
    b = ex.build_restricted_lane_entry(**kw)
    assert a == b
    assert a["sign_s"] <= a["sign_s_nominal"] and a["sign_s_nominal"] - a["sign_s"] <= 15.0
    assert a["spawn_offset_from_start"] >= 10.0
    assert a["reserved_lane_index"] == 1 and a["flow"] == "opposite" and a["lane_user"] == "bicycle"
