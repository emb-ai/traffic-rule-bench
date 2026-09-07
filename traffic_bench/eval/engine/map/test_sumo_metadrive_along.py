"""Unit tests for SUMO → MetaDrive along remapping."""

from traffic_bench.eval.engine.map.sumo_metadrive_along import (
    remap_sumo_along_to_metadrive,
    row_sumo_edge_length_m,
)


def test_remap_identity_when_lengths_match():
    assert remap_sumo_along_to_metadrive(
        245.0,
        sumo_edge_length_m=250.0,
        metadrive_lane_length_m=250.0,
    ) == 245.0


def test_remap_stitched_prefix_shifts_by_delta():
    # Detour bug case: SUMO edge 250m is the suffix of a 739.7m MD lane.
    remapped = remap_sumo_along_to_metadrive(
        245.0,
        sumo_edge_length_m=250.0,
        metadrive_lane_length_m=739.7,
    )
    assert abs(remapped - (739.7 - 250.0 + 245.0)) < 1e-6


def test_remap_keeps_spawn_past_dest_false():
    md_len = 739.7141108364118
    sumo_len = 250.0
    spawn_before_end = 92.05834192704245
    spawn_md = md_len - spawn_before_end
    dest_md = remap_sumo_along_to_metadrive(
        245.0,
        sumo_edge_length_m=sumo_len,
        metadrive_lane_length_m=md_len,
    )
    assert spawn_md < dest_md - 2.0


def test_row_sumo_edge_length_prefers_edge_length_m():
    assert row_sumo_edge_length_m({"edge_length_m": 250, "length_m": 100}) == 250.0
    assert row_sumo_edge_length_m({"length_m": 100}) == 100.0
    assert row_sumo_edge_length_m({"approach_lane_length_m": 80}) == 80.0
    assert row_sumo_edge_length_m({}) is None
