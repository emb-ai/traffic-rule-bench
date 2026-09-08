"""Per-map aggregation: map identity, std over maps, bootstrap CI, report."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import pytest

from traffic_bench.eval.metrics import aggregate as agg
from traffic_bench.eval.metrics import report
from traffic_bench.eval.metrics.csv import (
    CSV_COLUMNS,
    _build_row,
    _default_manifests_root,
    _episode_to_replay,
    _load_manifest_lookup,
)
from traffic_bench.eval.metrics.map_id import (
    base_scene_id,
    map_id_for,
    map_id_from_manifest_row,
)


# ---------------------------------------------------------------------------
# map identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scene_id, expected", [
    ("seg_1067603714_v0", "seg_1067603714"),
    ("seg_1067603714_v0_z60a60n2_td50_sv1_v0", "seg_1067603714"),
    ("seg_1067603714_v2_z100a100n1_td50_sv2_v2", "seg_1067603714"),
    ("seg_389266120_0_v1", "seg_389266120_0"),
    ("seg_123_v1_rl90_td50_sv1_v2", "seg_123"),
    # v6 speed families carry the spawn lane before the variant
    ("seg_100833537_0_l0_v0", "seg_100833537_0"),
    ("seg_100833537_0_l1_v3_rl120_td75_sv0_v1", "seg_100833537_0"),
    ("junc_cluster_331124312_331124313", "junc_cluster_331124312_331124313"),
    ("junc_8669788456", "junc_8669788456"),
    ("", ""),
])
def test_base_scene_id_strips_variant_and_world_cell(scene_id, expected):
    assert base_scene_id(scene_id) == expected


def test_map_id_prefers_manifest_net_path():
    assert map_id_from_manifest_row({"net_path": "seg_1067603714/map.net.xml"}) == "seg_1067603714"
    assert map_id_from_manifest_row(
        {"net_path": "data/scenes/speed_limit/seg_154326272_4/map.net.xml"}) == "seg_154326272_4"
    assert map_id_from_manifest_row({"net_path": "seg_y"}) == "seg_y"
    assert map_id_from_manifest_row({}) is None
    assert map_id_from_manifest_row(None) is None
    # manifest wins over the scene id, the scene id is the fallback
    assert map_id_for("seg_1_v0", {"net_path": "seg_other/map.net.xml"}) == "seg_other"
    assert map_id_for("seg_1_v0_z60a60n2_td50_sv1_v0", None) == "seg_1"


# ---------------------------------------------------------------------------
# synthetic episode rows in the shape load_episode_csv() produces
# ---------------------------------------------------------------------------
def _row(map_id: str, scene_id: str, *, success: bool = True, arrived: bool = True,
         pdd: str = "3.24", steps: int = 100, driving_score: float = 0.5,
         baseline: str = "idm_default") -> dict:
    return {
        "var_name": "var_0", "var_idx": 0, "baseline": baseline, "policy": "idm",
        "variant": "default", "display_policy": baseline, "backend": "sumo",
        "pdd_code": pdd, "sign_slug": pdd.replace(".", "_"),
        "target_sign_class": "SpeedLimitSign", "is_no_entry_sign": False,
        "scene_id": scene_id, "scene_uid": f"{scene_id}_lane0_seed1_v0", "map_id": map_id,
        "manifest_source": "", "is_paired_scene": False,
        "pdd_code_start": "", "pdd_code_end": "", "pdd_code_target": pdd,
        "sign_type_start": "", "sign_type_end": "", "zone_length_m": None,
        "valid": True, "arrived_dest": arrived, "crashed": False,
        "crashed_ego_fault": False, "crashed_npc_fault": False, "out_of_road": False,
        "success": success, "final_step": steps, "total_reward": 1.0,
        "route_completion": 1.0, "route_length_m": 100.0, "distance_travelled_m": 100.0,
        "driving_score": driving_score, "driving_efficiency": 50.0, "infraction_penalty": 0.0,
        "smoothness_ratio": 0.9, "frame_smooth_ratio": 0.9, "smooth_segments": 1,
        "total_segments": 1, "min_ttc_sec": 5.0, "mean_abs_lane_offset": 0.1,
        "mean_abs_steer_delta": 0.01, "hard_brake_count": 0, "hard_accel_count": 0,
        "total_violations": 0, "violations_event_count": 0, "in_zone_total_steps": 10,
        "viol_high_sign": 0, "viol_high_traffic_light": 0, "viol_high_crosswalk": 0,
        "violations_by_class_step": {}, "violations_by_class_event": {},
        "in_zone_by_class_step": {"SpeedLimitSign": 10},
        "target_violations_step": 0, "target_violations_event": 0,
        "target_in_zone_steps": 10, "target_in_zone": True,
        "target_compliant_event": True, "target_compliant_step": True,
        "sr_and_dest": arrived, "sign_compliant_high": True, "tl_compliant": True,
        "cw_compliant": True, "dest_recomputed": arrived, "passes_filter": arrived,
        "comfort": 0.9,
    }


def _variants(map_id: str, successes: list[bool]) -> list[dict]:
    """One map, len(successes) augmented variants (nominal + world cells)."""
    rows = []
    for i, ok in enumerate(successes):
        sid = f"{map_id}_v0" if i == 0 else f"{map_id}_v{i % 3}_rl90_td50_sv{i % 2}_v{i}"
        rows.append(_row(map_id, sid, success=ok, arrived=ok))
    return rows


FOUR_MAPS = {
    "seg_a": [True, True, True],
    "seg_b": [True, False, False],
    "seg_c": [False, False, False],
    "seg_d": [True, True, False],
}
FOUR_MAP_MEANS = [1.0, 1 / 3, 0.0, 2 / 3]


def _four_map_rows() -> list[dict]:
    return [r for mid, oks in FOUR_MAPS.items() for r in _variants(mid, oks)]


# ---------------------------------------------------------------------------
# aggregate_by_map
# ---------------------------------------------------------------------------
def test_aggregate_by_map_collapses_each_map_then_averages():
    out = agg.aggregate_by_map(_four_map_rows())
    assert out["n"] == 12
    assert out["n_maps"] == 4
    assert out["episodes_per_map_min"] == out["episodes_per_map_max"] == 3
    assert out["success_rate"] == pytest.approx(statistics.mean(FOUR_MAP_MEANS))
    assert out["sr_and_dest"] == pytest.approx(0.5)
    d = out["dispersion"]["success_rate"]
    assert d["n_maps"] == 4
    assert d["std"] == pytest.approx(statistics.stdev(FOUR_MAP_MEANS))
    assert 0.0 <= d["ci_lo"] <= 0.5 <= d["ci_hi"] <= 1.0
    # a constant metric has zero spread and a degenerate interval
    dd = out["dispersion"]["avg_driving_score"]
    assert dd["std"] == pytest.approx(0.0)
    assert dd["ci_lo"] == pytest.approx(0.5) and dd["ci_hi"] == pytest.approx(0.5)


def test_map_id_groups_augmented_scene_ids_of_one_net():
    """The old per-map aggregation keyed on scene_id, which carries the
    augmentation suffix, so every episode was its own map."""
    rows = _four_map_rows()
    assert len({r["scene_id"] for r in rows}) == 12
    assert agg.aggregate_by_map(rows)["n_maps"] == 4
    # a row without map_id falls back to the stripped scene id
    for r in rows:
        r.pop("map_id")
    assert agg.aggregate_by_map(rows)["n_maps"] == 4
    assert agg.map_key(rows[0]) == "3.24|seg_a"


def test_same_net_under_two_signs_is_two_maps():
    rows = _variants("seg_a", [True, True]) + _variants("seg_a", [False, False])
    for r in rows[2:]:
        r["pdd_code"] = "3.25"
    out = agg.aggregate_by_map(rows)
    assert out["n_maps"] == 2
    assert out["success_rate"] == pytest.approx(0.5)


def test_per_map_mean_differs_from_per_episode_when_unbalanced():
    rows = _variants("seg_a", [True]) + _variants("seg_b", [False, False, False])
    assert agg.aggregate(rows)["success_rate"] == pytest.approx(0.25)
    out = agg.aggregate_by_map(rows)
    assert out["success_rate"] == pytest.approx(0.5)
    assert (out["episodes_per_map_min"], out["episodes_per_map_max"]) == (1, 3)


def test_bootstrap_ci_is_seeded_and_optional():
    rows = _four_map_rows()
    a = agg.aggregate_by_map(rows, ci_seed=0)["dispersion"]["success_rate"]
    b = agg.aggregate_by_map(rows, ci_seed=0)["dispersion"]["success_rate"]
    assert (a["ci_lo"], a["ci_hi"]) == (b["ci_lo"], b["ci_hi"])
    c = agg.aggregate_by_map(rows, ci_seed=1)["dispersion"]["success_rate"]
    assert c["ci_lo"] <= 0.5 <= c["ci_hi"]
    no_ci = agg.aggregate_by_map(rows, n_boot=0)["dispersion"]["success_rate"]
    assert no_ci["ci_lo"] is None and no_ci["ci_hi"] is None
    assert no_ci["std"] == pytest.approx(a["std"])
    # one map: a mean but no spread
    one = agg.aggregate_by_map(_variants("seg_a", [True, False]))
    assert one["n_maps"] == 1
    assert one["dispersion"]["success_rate"] == {
        "n_maps": 1, "std": None, "ci_lo": None, "ci_hi": None}


def test_bootstrap_mean_ci_matches_normal_approximation_for_large_n():
    import numpy as np
    rng = np.random.default_rng(123)
    vals = list(rng.normal(0.7, 0.2, size=400))
    lo, hi = agg.bootstrap_mean_ci(vals, n_boot=5000, level=0.95, rng=rng)
    m = statistics.mean(vals)
    half = 1.96 * statistics.stdev(vals) / len(vals) ** 0.5
    assert lo == pytest.approx(m - half, abs=0.01)
    assert hi == pytest.approx(m + half, abs=0.01)
    assert agg.bootstrap_mean_ci([1.0], n_boot=100) is None
    assert agg.bootstrap_mean_ci([1.0, 2.0], n_boot=0) is None


# ---------------------------------------------------------------------------
# csv builder: map_id column and manifest sources
# ---------------------------------------------------------------------------
def _episode(scene_id: str, **extra) -> dict:
    ep = {
        "ok": True, "scene_id": scene_id, "scene_uid": f"{scene_id}_lane0_seed7_v0",
        "seed": 7, "sign_type": "3.24", "policy": "idm", "variant": "default",
        "reached_dest": True, "success": True, "steps": 120, "route_completion_pct": 100.0,
        "violations": 0, "violations_by_class_step": {}, "violations_by_class_event": {},
        "in_zone_by_class_step": {"SpeedLimitSign": 5}, "in_zone_total_steps": 5,
    }
    ep.update(extra)
    return ep


def test_build_row_stamps_map_id_from_manifest_or_scene_id():
    sid = "seg_1067603714_v0_z60a60n2_td50_sv1_v0"
    lookup = {(0, sid): {"scene_id": sid, "net_path": "seg_1067603714/map.net.xml"}}
    row = _build_row(_episode_to_replay(_episode(sid)), "var_0", 0, "idm_default", lookup)
    assert row["map_id"] == "seg_1067603714"
    row = _build_row(_episode_to_replay(_episode(sid)), "var_0", 0, "idm_default", None)
    assert row["map_id"] == "seg_1067603714"
    other = {(0, sid): {"scene_id": sid, "net_path": "seg_other/map.net.xml"}}
    row = _build_row(_episode_to_replay(_episode(sid)), "var_0", 0, "idm_default", other)
    assert row["map_id"] == "seg_other"
    assert "map_id" in CSV_COLUMNS


def test_manifest_lookup_accepts_flat_real_manifest(tmp_path: Path):
    man = tmp_path / "real_manifest.jsonl"
    rows = [
        {"scene_id": "seg_1_v0", "net_path": "seg_1/map.net.xml", "var_idx": 0},
        {"scene_id": "seg_1_v1_rl90_td50_sv1_v1", "net_path": "seg_1/map.net.xml", "var_idx": 1},
    ]
    man.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    # the run dir (holding real_manifest.jsonl), the file itself, and the
    # default resolution from the run dir all lead to the same lookup
    for root in (tmp_path, man, _default_manifests_root(tmp_path)):
        lookup = _load_manifest_lookup(root, wanted_var_idxs={0})
        assert set(lookup) == {(0, "seg_1_v0"), (0, "seg_1_v1_rl90_td50_sv1_v1")}
        assert lookup[(0, "seg_1_v0")]["net_path"] == "seg_1/map.net.xml"
    # chunks/ layout still wins when present
    chunks = tmp_path / "chunks" / "var_0"
    chunks.mkdir(parents=True)
    (chunks / "var_0.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    assert _default_manifests_root(tmp_path) == (tmp_path / "chunks").resolve()
    assert set(_load_manifest_lookup(tmp_path / "chunks", {0})) == {(0, "seg_1_v0")}
    # nothing found → empty lookup, no exception
    assert _load_manifest_lookup(tmp_path / "missing", {0}) == {}


# ---------------------------------------------------------------------------
# csv loader + end-to-end aggregate → report
# ---------------------------------------------------------------------------
def _write_episode_csv(path: Path, rows: list[dict], with_map_id: bool) -> None:
    cols = [c for c in CSV_COLUMNS if with_map_id or c != "map_id"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rec = dict(r)
            rec["violations_by_class_step_json"] = json.dumps(r["violations_by_class_step"])
            rec["violations_by_class_event_json"] = json.dumps(r["violations_by_class_event"])
            rec["in_zone_by_class_step_json"] = json.dumps(r["in_zone_by_class_step"])
            rec["zone_length_m"] = ""
            w.writerow(rec)


def test_load_episode_csv_reads_or_derives_map_id(tmp_path: Path):
    rows = _four_map_rows()
    for r in rows:
        r["map_id"] = "net_" + r["map_id"]     # a manifest-given id ≠ scene prefix
    p = tmp_path / "with.csv"
    _write_episode_csv(p, rows, with_map_id=True)
    loaded = agg.load_episode_csv(p)
    assert {r["map_id"] for r in loaded} == {"net_seg_a", "net_seg_b", "net_seg_c", "net_seg_d"}
    p = tmp_path / "without.csv"
    _write_episode_csv(p, rows, with_map_id=False)
    loaded = agg.load_episode_csv(p)
    assert {r["map_id"] for r in loaded} == {"seg_a", "seg_b", "seg_c", "seg_d"}
    assert agg.aggregate_by_map(loaded)["n_maps"] == 4


def test_aggregate_cli_writes_ci_tables_and_report_renders_them(tmp_path: Path, monkeypatch):
    rows = _four_map_rows() + [
        dict(r, baseline="carl_rule", display_policy="carl_rule")
        for r in _four_map_rows()
    ]
    csv_path = tmp_path / "metrics_per_episode.csv"
    _write_episode_csv(csv_path, rows, with_map_id=True)
    out_dir = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["aggregate", "--csv", str(csv_path),
                                     "--out-dir", str(out_dir), "--n-boot", "2000"])
    agg.main()

    # long CI table: one row per (baseline, metric)
    ci_csv = out_dir / "aggregations" / "agg_per_baseline_map_ci.csv"
    ci_rows = list(csv.DictReader(ci_csv.open(encoding="utf-8")))
    sr = {r["baseline"]: r for r in ci_rows if r["metric"] == "success_rate"}
    assert set(sr) == {"idm_default", "carl_rule"}
    assert float(sr["idm_default"]["mean"]) == pytest.approx(0.5)
    assert int(sr["idm_default"]["n_maps"]) == 4
    assert float(sr["idm_default"]["ci_lo"]) <= 0.5 <= float(sr["idm_default"]["ci_hi"])
    # the wide map table now counts physical maps and episodes per map
    map_rows = list(csv.DictReader(
        (out_dir / "aggregations" / "agg_per_baseline_map.csv").open(encoding="utf-8")))
    m = {r["baseline"]: r for r in map_rows}["idm_default"]
    assert (m["n"], m["n_maps"], m["episodes_per_map_min"], m["episodes_per_map_max"]) == (
        "12", "4", "3", "3")
    per_sign_ci = out_dir / "aggregations" / "agg_per_sign_baseline_map_ci.csv"
    assert any(r["pdd_code"] == "3.24" for r in csv.DictReader(per_sign_ci.open(encoding="utf-8")))

    # cumulative.json carries the CI blocks and how they were built
    cum = json.loads((out_dir / "reports" / "cumulative.json").read_text(encoding="utf-8"))
    assert cum["ci"]["n_boot"] == 2000 and cum["ci"]["level"] == 0.95 and cum["ci"]["unit"] == "map"
    blk = cum["per_baseline_map_ci"]["idm_default"]["sr_and_dest"]
    assert blk["mean"] == pytest.approx(0.5) and blk["n_maps"] == 4
    assert blk["ci_lo"] <= 0.5 <= blk["ci_hi"]
    assert cum["per_sign_map_ci"]["carl_rule"]["3.24"]["success_rate"]["n_maps"] == 4
    assert cum["per_baseline_map"]["idm_default"]["n_maps"] == 4
    assert "n_maps" not in cum["per_baseline"]["idm_default"]

    # markdown report: episode / map cells plus the dispersion tables
    monkeypatch.setattr("sys.argv", ["report", "--run-root", str(out_dir)])
    report.main()
    md = (out_dir / "reports" / "report_cumulative.md").read_text(encoding="utf-8")
    assert "### Overall — per-map mean ± std, 95% CI (2000 resamples)" in md
    assert "#### Sign `3.24` — per-map mean ± std" in md
    idm_line = next(l for l in md.splitlines()
                    if l.startswith("| `idm_default` | 4 | 3 |"))
    assert "0.500 ± 0.430 [" in idm_line
    # the classic table still renders episode / map with the map count
    assert "| `idm_default` | 12 | 12 | 4 | 0.500 / 0.500 |" in md
