"""Tests for segment window vs net edge length / corridor frame."""

from pathlib import Path

from traffic_bench.eval.engine.map.segment_length import (
    SEGMENT_LENGTH_ABS_TOL_M,
    resolve_segment_corridor,
    resolve_segment_edge_length_m,
    segment_window_net_mismatch,
    window_length_from_meta,
)


def test_window_length_prefers_window_length_m():
    assert window_length_from_meta({"window_length_m": 250, "length_m": 100}) == 250.0
    assert window_length_from_meta({"length_m": 100}) == 100.0


def test_mismatch_ok_when_close(tmp_path: Path):
    net = tmp_path / "map.net.xml"
    net.write_text(
        """<?xml version="1.0"?>
<net>
  <edge id="e0">
    <lane id="e0_0" index="0" length="255.0" shape="0,0 255,0"/>
  </edge>
</net>
""",
        encoding="utf-8",
    )
    assert (
        segment_window_net_mismatch(
            net, {"road_id": "e0", "length_m": 250.0}
        )
        is None
    )


def test_mismatch_flags_long_edge(tmp_path: Path):
    net = tmp_path / "map.net.xml"
    net.write_text(
        """<?xml version="1.0"?>
<net>
  <edge id="1049581689#3">
    <lane id="1049581689#3_0" index="0" length="739.7" shape="0,0 739,0"/>
  </edge>
</net>
""",
        encoding="utf-8",
    )
    result = segment_window_net_mismatch(
        net, {"road_id": "1049581689#3", "length_m": 250.0}
    )
    assert result is not None
    assert result[0] == "segment_length_mismatch"


def test_corridor_projects_crop_window(tmp_path: Path):
    # 0──250──500──739 along a straight edge; window mid-edge.
    net = tmp_path / "map.net.xml"
    net.write_text(
        """<?xml version="1.0"?>
<net>
  <edge id="e0">
    <lane id="e0_0" index="0" length="739.0"
     shape="0.0,0.0 250.0,0.0 500.0,0.0 739.0,0.0"/>
  </edge>
</net>
""",
        encoding="utf-8",
    )
    meta = {
        "road_id": "e0",
        "length_m": 250.0,
        "crop_window": {
            "start_xy": [250.0, 0.0],
            "end_xy": [500.0, 0.0],
        },
    }
    c = resolve_segment_corridor(net, meta)
    assert c is not None
    assert c.from_window
    assert abs(c.corridor_s0 - 250.0) < 1.0
    assert abs(c.corridor_s1 - 500.0) < 1.0
    assert abs(c.to_absolute(100.0) - 350.0) < 1.0
    # Spawn near end of window, not near net end / junction.
    assert c.to_absolute(200.0) < 520.0


def test_resolve_prefers_net_over_meta_window(tmp_path: Path):
    net = tmp_path / "map.net.xml"
    net.write_text(
        """<?xml version="1.0"?>
<net>
  <edge id="e0">
    <lane id="e0_0" index="0" length="161.1" shape="0,0 161,0"/>
  </edge>
</net>
""",
        encoding="utf-8",
    )
    meta = {"road_id": "e0", "length_m": 161.03, "window_length_m": 161.03}
    assert abs(resolve_segment_edge_length_m(net, meta) - 161.1) < 0.01
    c = resolve_segment_corridor(net, meta)
    assert c is not None and not c.from_window
    assert SEGMENT_LENGTH_ABS_TOL_M == 20.0
