"""Per-map metrics: map from the manifest, strict inputs, std / CI, report."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest

from traffic_bench.eval.engine.expand.world_axes import iter_world_axis_cells
from traffic_bench.eval.metrics import aggregate as agg
from traffic_bench.eval.metrics import combine
from traffic_bench.eval.metrics import csv as csv_mod
from traffic_bench.eval.metrics import plot_benchmark
from traffic_bench.eval.metrics import report
from traffic_bench.eval.metrics.csv import (
    CSV_COLUMNS,
    ManifestIndex,
    _build_row,
    _episode_to_replay,
    load_manifest,
)
from traffic_bench.eval.metrics.map_id import (
    MAP_ID_SOURCE,
    ManifestError,
    map_id_from_manifest_row,
)


def test_map_id_is_the_net_path_directory():
    assert map_id_from_manifest_row({"net_path": "seg_1067603714/map.net.xml"}) == "seg_1067603714"
    assert map_id_from_manifest_row(
        {"net_path": "data/scenes/speed_limit/seg_154326272_4/map.net.xml"}) == "seg_154326272_4"
    assert map_id_from_manifest_row({"net_path": "junc_8669788456/map.net.xml"}) == "junc_8669788456"


@pytest.mark.parametrize("net_path", [None, "", "   ", 5, "seg_y", "map.net.xml"])
def test_map_id_without_a_usable_net_path_raises(net_path):
    row = {"scene_id": "seg_1_v0"}
    if net_path is not None:
        row["net_path"] = net_path
    with pytest.raises(ManifestError):
        map_id_from_manifest_row(row)


def _row(map_id: str, scene_id: str, *, success: bool = True, arrived: bool = True,
         pdd: str = "3.24", steps: int = 100, driving_score: float = 0.5,
         baseline: str = "idm_default") -> dict:
    return {
        "var_name": "var_0", "var_idx": 0, "baseline": baseline, "policy": "idm",
        "variant": "default", "display_policy": baseline, "backend": "sumo",
        "pdd_code": pdd, "sign_slug": pdd.replace(".", "_"),
        "target_sign_class": "SpeedLimitSign", "is_no_entry_sign": False,
        "scene_id": scene_id, "scene_uid": f"{scene_id}_lane0_seed1_v0",
        "map_id": map_id, "net_path": f"{map_id}/map.net.xml",
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
    dd = out["dispersion"]["avg_driving_score"]
    assert dd["std"] == pytest.approx(0.0)
    assert dd["ci_lo"] == pytest.approx(0.5) and dd["ci_hi"] == pytest.approx(0.5)


def test_map_id_groups_augmented_scene_ids_of_one_net():
    rows = _four_map_rows()
    assert len({r["scene_id"] for r in rows}) == 12
    assert agg.aggregate_by_map(rows)["n_maps"] == 4
    assert agg.map_key(rows[0]) == "3.24|seg_a"


def test_row_without_map_id_raises_instead_of_parsing_the_scene_id():
    rows = _four_map_rows()
    rows[5]["map_id"] = ""
    with pytest.raises(ManifestError):
        agg.aggregate_by_map(rows)
    del rows[5]["map_id"]
    with pytest.raises(ManifestError):
        agg.map_key(rows[5])


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


def test_non_finite_values_raise_and_undefined_ones_are_left_out():
    with pytest.raises(ValueError):
        agg._mean([1.0, float("nan")])
    with pytest.raises(ValueError):
        agg.map_values([{"x": float("inf")}], "x")
    assert agg._mean([None, 1.0, 3.0]) == 2.0
    assert agg.map_values([{"x": None}, {"x": 0.5}], "x") == [0.5]


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


def _write_jsonl(path: Path, rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows),
                    encoding="utf-8")
    return path


def test_load_manifest_reads_a_manifest_file_or_a_chunks_dir(tmp_path: Path):
    rows = [
        {"scene_id": "seg_1_v0", "net_path": "seg_1/map.net.xml"},
        {"scene_id": "seg_1_v1_rl90_td50_sv1_v1", "net_path": "seg_1/map.net.xml"},
    ]
    idx = load_manifest(_write_jsonl(tmp_path / "real_manifest.jsonl", rows), {0})
    assert set(idx.rows) == {(0, "seg_1_v0"), (0, "seg_1_v1_rl90_td50_sv1_v1")}
    assert idx.row(0, "seg_1_v0")["net_path"] == "seg_1/map.net.xml"
    _write_jsonl(tmp_path / "chunks" / "var_0" / "var_0.jsonl", rows[:1])
    _write_jsonl(tmp_path / "chunks" / "var_1" / "var_1.jsonl", rows[1:])
    idx = load_manifest(tmp_path / "chunks", {0, 1})
    assert set(idx.rows) == {(0, "seg_1_v0"), (1, "seg_1_v1_rl90_td50_sv1_v1")}
    with pytest.raises(ManifestError, match="not in the manifest"):
        idx.row(0, "seg_1_v1_rl90_td50_sv1_v1")


def test_load_manifest_refuses_anything_it_would_have_to_guess(tmp_path: Path):
    good = {"scene_id": "seg_1_v0", "net_path": "seg_1/map.net.xml"}
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        load_manifest(tmp_path / "missing.jsonl", {0})
    _write_jsonl(tmp_path / "run" / "real_manifest.jsonl", [good])
    with pytest.raises(FileNotFoundError, match="chunks"):
        load_manifest(tmp_path / "run", {0})
    _write_jsonl(tmp_path / "chunks" / "var_0" / "var_0.jsonl", [good])
    with pytest.raises(FileNotFoundError, match="var_1"):
        load_manifest(tmp_path / "chunks", {0, 1})
    for name, rows in {
        "empty": [],
        "no_net_path": [{"scene_id": "seg_1_v0"}],
        "no_scene_id": [{"net_path": "seg_1/map.net.xml"}],
        "conflict": [good, {"scene_id": "seg_1_v0", "net_path": "seg_2/map.net.xml"}],
    }.items():
        with pytest.raises(ManifestError):
            load_manifest(_write_jsonl(tmp_path / f"{name}.jsonl", rows), {0})
    torn = _write_jsonl(tmp_path / "torn.jsonl", [good, '{"scene_id": "seg_1_v1", "net_pa'])
    with pytest.raises(ValueError, match=r"torn.jsonl:2: malformed JSON"):
        load_manifest(torn, {0})
    same = load_manifest(_write_jsonl(tmp_path / "dup.jsonl", [good, dict(good, seed=5)]), {0})
    assert same.row(0, "seg_1_v0")["net_path"] == "seg_1/map.net.xml"


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


def _index(rows: dict[str, str]) -> ManifestIndex:
    return ManifestIndex({(0, sid): {"scene_id": sid, "net_path": net}
                          for sid, net in rows.items()}, [Path("real_manifest.jsonl")])


def test_build_row_takes_the_map_from_the_manifest_row():
    sid = "seg_1067603714_v0_z60a60n2_td50_sv1_v0"
    row = _build_row(_episode_to_replay(_episode(sid)), "var_0", 0, "idm_default",
                     _index({sid: "seg_1067603714/map.net.xml"}))
    assert (row["map_id"], row["net_path"]) == ("seg_1067603714", "seg_1067603714/map.net.xml")
    row = _build_row(_episode_to_replay(_episode(sid)), "var_0", 0, "idm_default",
                     _index({sid: "seg_other/map.net.xml"}))
    assert row["map_id"] == "seg_other"
    assert {"map_id", "net_path"} <= set(CSV_COLUMNS)
    with pytest.raises(ManifestError, match="not in the manifest"):
        _build_row(_episode_to_replay(_episode("seg_9_v0")), "var_0", 0, "idm_default",
                   _index({sid: "seg_1067603714/map.net.xml"}))


def test_episode_records_must_be_well_formed():
    idx = _index({"seg_1_v0": "seg_1/map.net.xml"})
    no_uid = {k: v for k, v in _episode("seg_1_v0").items() if k != "scene_uid"}
    with pytest.raises(ValueError, match="scene_uid"):
        _episode_to_replay(no_uid)
    with pytest.raises(ValueError, match="bool"):
        _episode_to_replay(_episode("seg_1_v0", reached_dest="yes"))
    with pytest.raises(ValueError, match="integer"):
        _build_row(_episode_to_replay(_episode("seg_1_v0", steps=12.5)), "var_0", 0, "b", idx)
    with pytest.raises(ValueError, match="non-finite"):
        _build_row(_episode_to_replay(_episode("seg_1_v0", driving_score=float("nan"))),
                   "var_0", 0, "b", idx)
    with pytest.raises(ValueError, match="dict"):
        _episode_to_replay(_episode("seg_1_v0", violations_by_class_step=[1, 2]))
    row = _build_row(_episode_to_replay(_episode("seg_1_v0")), "var_0", 0, "b", idx)
    assert row["min_ttc_sec"] is None and row["final_step"] == 120


def _grid(task_conditioned: bool) -> list:
    return list(iter_world_axis_cells(route_levels=[90.0, 120.0],
                                      sim=SimpleNamespace(n_variations=3),
                                      task_conditioned_spawn=task_conditioned))


def _stop_rows(name: str, cells: list) -> list[dict]:
    """junction/expand.py: scene_id = name + world cell; row 10 repeats a cell."""
    picked = cells[::5][:9]
    spec = [(c, 0, 1000 + i) for i, c in enumerate(picked)] + [(picked[0], 1, 2000)]
    return [{"scene_id": f"{name}_{c.scene_suffix(route_augment=True)}",
             "net_path": f"{name}/map.net.xml", "seed": seed, "var_idx": c.npc_var,
             "spawn_lane_num": lane, "pdd_code": "2.5"} for c, lane, seed in spec]


def _speed_rows(name: str, cells: list) -> list[dict]:
    """speed/expand.py: scene_id = f"{name}_l{lane}_v{variant}" + world cell."""
    rows = []
    for k in range(10):
        c = cells[(k * 7) % len(cells)]
        rows.append({"scene_id": f"{name}_l{k % 2}_v{k // 2}_{c.scene_suffix(route_augment=True)}",
                     "net_path": f"{name}/map.net.xml", "seed": 5000 + k, "var_idx": k // 2,
                     "spawn_lane_num": k % 2, "pdd_code": "3.24"})
    return rows


def _speed_rows_aug27(name: str) -> list[dict]:
    """data/runs/*/test of 2026-08-27: seg_x_l0_td2_v0."""
    return [{"scene_id": f"{name}_l{k % 2}_td{k // 2 % 3}_v{k // 6}",
             "net_path": f"{name}/map.net.xml", "seed": 7000 + k, "var_idx": k // 6,
             "spawn_lane_num": k % 2, "pdd_code": "4.6"} for k in range(10)]


def _episode_for(m: dict, success: bool) -> dict:
    return {
        "ok": True, "backend": "sumo", "scene_id": m["scene_id"],
        "scene_uid": f"{m['scene_id']}_lane{m['spawn_lane_num']}_seed{m['seed']}_v{m['var_idx']}",
        "policy": "idm", "variant": "default", "sign_type": m["pdd_code"], "seed": m["seed"],
        "reached_dest": success, "success": success, "crashed": False, "out_of_road": False,
        "steps": 300, "route_completion_pct": 100.0 if success else 40.0,
        "violations": 0, "violations_event_count": 0,
        "violations_by_class_step": {}, "violations_by_class_event": {},
        "in_zone_total_steps": 10, "in_zone_by_class_step": {},
    }


def _layout(root: Path, manifest_rows: list[dict], share: dict[str, float]) -> tuple[Path, Path]:
    split = root / "test"
    man = _write_jsonl(split / "real_manifest.jsonl", manifest_rows)
    seen: dict[str, int] = {}
    eps = []
    for m in manifest_rows:
        mid = m["net_path"].split("/")[0]
        i = seen.get(mid, 0)
        seen[mid] = i + 1
        eps.append(_episode_for(m, i < round(share[mid] * 10)))
    _write_jsonl(split / "eval_out" / "idm" / "episodes_idm.jsonl", eps)
    return split, man


def _metrics_csv(monkeypatch, *args) -> None:
    monkeypatch.setattr("sys.argv", ["metrics csv", *map(str, args)])
    csv_mod.main()


def test_stop_and_speed_maps_come_from_the_manifest_whatever_the_id_format(
        tmp_path: Path, monkeypatch):
    jc, sc = _grid(False), _grid(True)
    rows = (_stop_rows("junc_10036627085", jc) + _stop_rows("junc_1056023309", jc)
            + _speed_rows("seg_100833537_0", sc) + _speed_rows("seg_1047867720_0", sc)
            + _speed_rows_aug27("seg_1004765070_0") + _speed_rows_aug27("seg_128075583_1"))
    share = {"junc_10036627085": 1.0, "junc_1056023309": 0.3, "seg_100833537_0": 0.9,
             "seg_1047867720_0": 0.2, "seg_1004765070_0": 0.5, "seg_128075583_1": 0.1}
    split, man = _layout(tmp_path, rows, share)
    out_csv = split / "eval_out" / "metrics_per_episode.csv"
    _metrics_csv(monkeypatch, "--episodes-root", split / "eval_out", "--manifest", man,
                 "--out", out_csv)
    by_sign: dict[str, list[dict]] = {}
    for r in agg.load_episode_csv(out_csv):
        by_sign.setdefault(r["pdd_code"], []).append(r)
    expected = {"2.5": (1.0 + 0.3) / 2, "3.24": (0.9 + 0.2) / 2, "4.6": (0.5 + 0.1) / 2}
    assert set(by_sign) == set(expected)
    for pdd, rs in by_sign.items():
        out = agg.aggregate_by_map(rs, n_boot=200)
        assert (out["n"], out["n_maps"], out["episodes_per_map_min"],
                out["episodes_per_map_max"]) == (20, 2, 10, 10), pdd
        assert out["success_rate"] == pytest.approx(expected[pdd]), pdd
    assert len({r["scene_id"] for r in by_sign["2.5"]}) == 18
    assert {r["map_id"] for r in by_sign["2.5"]} == {"junc_10036627085", "junc_1056023309"}


def test_metrics_csv_needs_the_manifest_the_episodes_were_run_from(
        tmp_path: Path, monkeypatch, capsys):
    sc = _grid(True)
    split, man = _layout(tmp_path, _speed_rows("seg_100833537_0", sc), {"seg_100833537_0": 0.5})
    root = split / "eval_out"
    out_csv = tmp_path / "m.csv"
    with pytest.raises(SystemExit) as e:
        _metrics_csv(monkeypatch, "--episodes-root", root, "--out", out_csv)
    assert e.value.code == 2
    assert "--manifest" in capsys.readouterr().err
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        _metrics_csv(monkeypatch, "--episodes-root", root,
                     "--manifest", tmp_path / "nope.jsonl", "--out", out_csv)
    with pytest.raises(FileNotFoundError, match="chunks"):
        _metrics_csv(monkeypatch, "--episodes-root", root, "--manifest", split, "--out", out_csv)
    other = _write_jsonl(tmp_path / "other.jsonl", _speed_rows("seg_1047867720_0", sc))
    with pytest.raises(ManifestError, match="not in the manifest"):
        _metrics_csv(monkeypatch, "--episodes-root", root, "--manifest", other, "--out", out_csv)
    assert not out_csv.exists()
    _metrics_csv(monkeypatch, "--episodes-root", root, "--manifest", man, "--out", out_csv)
    assert {r["map_id"] for r in agg.load_episode_csv(out_csv)} == {"seg_100833537_0"}


def test_failed_or_torn_episode_records_stop_the_build(tmp_path: Path, monkeypatch):
    split, man = _layout(tmp_path, _speed_rows("seg_100833537_0", _grid(True)),
                         {"seg_100833537_0": 0.5})
    ep_file = split / "eval_out" / "idm" / "episodes_idm.jsonl"
    eps = [json.loads(line) for line in ep_file.read_text(encoding="utf-8").splitlines()]
    failed = dict(eps[3], ok=False, error="Traceback …")
    out_csv = tmp_path / "m.csv"
    args = ("--episodes-root", split / "eval_out", "--manifest", man, "--out", out_csv)
    _write_jsonl(ep_file, eps[:3] + [failed] + eps[4:])
    with pytest.raises(ValueError, match="failed to run"):
        _metrics_csv(monkeypatch, *args)
    assert not out_csv.exists()
    _write_jsonl(ep_file, eps[:3] + [failed] + eps[4:] + [eps[3]])
    _metrics_csv(monkeypatch, *args)
    assert len(agg.load_episode_csv(out_csv)) == 10
    ep_file.write_text(ep_file.read_text(encoding="utf-8") + '{"ok": true, "scene_id": "seg_1',
                       encoding="utf-8")
    with pytest.raises(ValueError, match=r"episodes_idm.jsonl:12: malformed JSON"):
        _metrics_csv(monkeypatch, *args)
    _write_jsonl(ep_file, eps + [{k: v for k, v in eps[0].items() if k != "scene_uid"}])
    with pytest.raises(ValueError, match=r"episodes_idm.jsonl:11: record has no scene_uid"):
        _metrics_csv(monkeypatch, *args)


def _write_episode_csv(path: Path, rows: list[dict], drop: tuple[str, ...] = ()) -> None:
    cols = [c for c in CSV_COLUMNS if c not in drop]
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


def test_load_episode_csv_needs_manifest_columns_and_well_formed_cells(tmp_path: Path):
    rows = _four_map_rows()
    p = tmp_path / "ok.csv"
    _write_episode_csv(p, rows)
    loaded = agg.load_episode_csv(p)
    assert {r["map_id"] for r in loaded} == {"seg_a", "seg_b", "seg_c", "seg_d"}
    assert agg.aggregate_by_map(loaded)["n_maps"] == 4
    for col in ("map_id", "net_path"):
        old = tmp_path / f"no_{col}.csv"
        _write_episode_csv(old, rows, drop=(col,))
        with pytest.raises(ManifestError, match=col):
            agg.load_episode_csv(old)
    bad = [dict(r) for r in rows]
    bad[4]["map_id"] = "seg_x"
    _write_episode_csv(tmp_path / "mismatch.csv", bad)
    with pytest.raises(ManifestError, match=r"mismatch.csv:6"):
        agg.load_episode_csv(tmp_path / "mismatch.csv")
    for field, value in (("success", "yes"), ("final_step", ""), ("final_step", "12.5"),
                         ("driving_score", "nan"), ("violations_by_class_step", [])):
        broken = [dict(r) for r in rows]
        broken[0][field] = value
        _write_episode_csv(tmp_path / "broken.csv", broken)
        with pytest.raises(ValueError, match=r"broken.csv:2"):
            agg.load_episode_csv(tmp_path / "broken.csv")


def test_aggregate_cli_writes_ci_tables_and_report_renders_them(tmp_path: Path, monkeypatch):
    rows = _four_map_rows() + [
        dict(r, baseline="carl_rule", display_policy="carl_rule")
        for r in _four_map_rows()
    ]
    csv_path = tmp_path / "metrics_per_episode.csv"
    _write_episode_csv(csv_path, rows)
    out_dir = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["aggregate", "--csv", str(csv_path),
                                     "--out-dir", str(out_dir), "--n-boot", "2000"])
    agg.main()

    ci_csv = out_dir / "aggregations" / "agg_per_baseline_map_ci.csv"
    ci_rows = list(csv.DictReader(ci_csv.open(encoding="utf-8")))
    sr = {r["baseline"]: r for r in ci_rows if r["metric"] == "success_rate"}
    assert set(sr) == {"idm_default", "carl_rule"}
    assert float(sr["idm_default"]["mean"]) == pytest.approx(0.5)
    assert int(sr["idm_default"]["n_maps"]) == 4
    assert float(sr["idm_default"]["ci_lo"]) <= 0.5 <= float(sr["idm_default"]["ci_hi"])
    map_rows = list(csv.DictReader(
        (out_dir / "aggregations" / "agg_per_baseline_map.csv").open(encoding="utf-8")))
    m = {r["baseline"]: r for r in map_rows}["idm_default"]
    assert (m["n"], m["n_maps"], m["episodes_per_map_min"], m["episodes_per_map_max"]) == (
        "12", "4", "3", "3")

    cum_path = out_dir / "reports" / "cumulative.json"
    cum = json.loads(cum_path.read_text(encoding="utf-8"))
    assert cum["ci"]["map_id"] == MAP_ID_SOURCE
    assert cum["ci"]["n_boot"] == 2000 and cum["ci"]["level"] == 0.95 and cum["ci"]["unit"] == "map"
    blk = cum["per_baseline_map_ci"]["idm_default"]["sr_and_dest"]
    assert blk["mean"] == pytest.approx(0.5) and blk["n_maps"] == 4
    assert blk["ci_lo"] <= 0.5 <= blk["ci_hi"]
    assert cum["per_sign_map_ci"]["carl_rule"]["3.24"]["success_rate"]["n_maps"] == 4
    assert cum["per_baseline_map"]["idm_default"]["n_maps"] == 4
    assert "n_maps" not in cum["per_baseline"]["idm_default"]

    monkeypatch.setattr("sys.argv", ["report", "--run-root", str(out_dir)])
    report.main()
    md = (out_dir / "reports" / "report_cumulative.md").read_text(encoding="utf-8")
    assert "### Overall — per-map mean ± std, 95% CI (2000 resamples)" in md
    assert "#### Sign `3.24` — per-map mean ± std" in md
    idm_line = next(line for line in md.splitlines()
                    if line.startswith("| `idm_default` | 4 | 3 |"))
    assert "0.500 ± 0.430 [" in idm_line
    assert "| `idm_default` | 12 | 12 | 4 | 0.500 / 0.500 |" in md

    assert plot_benchmark._load_cumulative(cum_path, "map") == cum["per_sign_map"]
    assert plot_benchmark._load_cumulative(cum_path, "episode") == cum["per_sign"]

    no_marker = dict(cum, ci={k: v for k, v in cum["ci"].items() if k != "map_id"})
    no_blocks = {k: v for k, v in cum.items()
                 if not k.startswith("per_") or k in ("per_baseline", "per_sign")}
    for name, broken in (("no_marker", no_marker), ("no_blocks", no_blocks)):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(broken), encoding="utf-8")
        md_out = tmp_path / f"{name}.md"
        monkeypatch.setattr("sys.argv", ["report", "--run-root", str(out_dir),
                                         "--cumulative", str(p), "--out", str(md_out)])
        with pytest.raises(ValueError):
            report.main()
        assert not md_out.exists()
        with pytest.raises(ValueError):
            plot_benchmark._load_cumulative(p, "map")


def test_aggregate_cli_refuses_missing_csv_and_bad_ci_args(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["aggregate", "--csv", str(tmp_path / "nope.csv"),
                                     "--out-dir", str(tmp_path / "out")])
    with pytest.raises(FileNotFoundError):
        agg.main()
    csv_path = tmp_path / "m.csv"
    _write_episode_csv(csv_path, _four_map_rows())
    for bad in (["--n-boot", "-1"], ["--ci-level", "1.5"]):
        monkeypatch.setattr("sys.argv", ["aggregate", "--csv", str(csv_path),
                                         "--out-dir", str(tmp_path / "out"), *bad])
        with pytest.raises(SystemExit):
            agg.main()


def test_combine_refuses_csvs_of_another_schema(tmp_path: Path):
    rows = _four_map_rows()
    good = tmp_path / "a.csv"
    _write_episode_csv(good, rows)
    old = tmp_path / "b.csv"
    _write_episode_csv(old, rows, drop=("map_id", "net_path"))
    assert combine.concat_csvs([good, good], tmp_path / "merged.csv") == 24
    with pytest.raises(ValueError, match="map_id"):
        combine.concat_csvs([good, old], tmp_path / "merged2.csv")
    assert not (tmp_path / "merged2.csv").exists()
    assert not (tmp_path / "merged2.csv.tmp").exists()
