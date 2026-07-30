#!/usr/bin/env python3
"""Collect high-level dataset statistics for the reviewer-facing sign subset.

Outputs under ``output/raw/``:
  - scene_inventory.jsonl   one row per catalog / package scene
  - scenario_stats.jsonl    one row per final_metrics_v1 manifest entry
  - sign_summary.json       per-sign aggregates
  - overview.json           global headline numbers
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from sign_registry import (
    DT_SECONDS,
    META_DENSITY_SCALE,
    SCENES_ROOT,
    SPEED_MAPS_ROOT,
    TARGET_SIGNS,
    SignSpec,
    package_eval_agg,
    package_final_manifest,
    package_metrics_csv,
    package_scenes_dir,
)

SKIP_SCENE_NAMES = {"core", "_rejected", "_old", "_osm_temp"}


def _mean(vals: list[float]) -> Optional[float]:
    return float(statistics.mean(vals)) if vals else None


def _std(vals: list[float]) -> Optional[float]:
    return float(statistics.pstdev(vals)) if len(vals) >= 2 else (0.0 if vals else None)


def _percentile(vals: list[float], q: float) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(vals)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def _summarize_numeric(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": _mean(vals),
        "std": _std(vals),
        "min": float(min(vals)),
        "p25": _percentile(vals, 0.25),
        "median": _percentile(vals, 0.50),
        "p75": _percentile(vals, 0.75),
        "max": float(max(vals)),
    }


def parse_net_xml(path: Path) -> dict[str, Any]:
    """Lightweight SUMO net.xml counters (no full XML DOM)."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    return {
        "n_edges": len(re.findall(r"<edge\s", text)),
        "n_lanes": len(re.findall(r"<lane\s", text)),
        "n_junctions": len(re.findall(r"<junction\s", text)),
        "n_connections": len(re.findall(r"<connection\s", text)),
        "net_bytes": path.stat().st_size,
    }


def _iter_scene_dirs(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name in SKIP_SCENE_NAMES or d.name.startswith("."):
            continue
        yield d


def _load_scene_selection(scenes_dir: Path) -> dict[str, str]:
    path = scenes_dir / "scene_selection.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    scenes = data.get("scenes", data)
    return {str(k): str(v) for k, v in scenes.items()} if isinstance(scenes, dict) else {}


def collect_catalog_scenes(spec: SignSpec, parse_nets: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for catalog_name in spec.catalog_dirs:
        root = SCENES_ROOT / catalog_name
        if not root.is_dir():
            continue
        for scene_dir in _iter_scene_dirs(root):
            meta_path = scene_dir / "meta.json"
            meta: dict[str, Any] = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except json.JSONDecodeError:
                    meta = {}
            net_name = meta.get("net_file") or f"{scene_dir.name}.net.xml"
            net_path = scene_dir / net_name
            if not net_path.exists():
                nets = list(scene_dir.glob("*.net.xml"))
                net_path = nets[0] if nets else net_path
            row: dict[str, Any] = {
                "source": "catalog",
                "pdd_code": spec.pdd_code,
                "category": spec.category,
                "scene_id": scene_dir.name,
                "catalog_dir": catalog_name,
                "latitude": meta.get("latitude"),
                "longitude": meta.get("longitude"),
                "crop_radius_m": meta.get("crop_radius_m"),
                "junction_arm_count": meta.get("junction_arm_count")
                or meta.get("catalog_junction_arm_count"),
            }
            if parse_nets and net_path.exists():
                row.update(parse_net_xml(net_path))
            rows.append(row)
    return rows


def collect_package_scenes(spec: SignSpec, parse_nets: bool) -> list[dict[str, Any]]:
    scenes_dir = package_scenes_dir(spec)
    if scenes_dir is None or not scenes_dir.is_dir():
        return []
    selection = _load_scene_selection(scenes_dir)
    rows: list[dict[str, Any]] = []
    for scene_dir in _iter_scene_dirs(scenes_dir):
        meta_path = scene_dir / "meta.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                meta = {}
        pdd = (
            meta.get("pdd_code")
            or meta.get("sign_code")
            or meta.get("catalog_pdd_code")
            or spec.pdd_code
        )
        # Secondary package shares 2_3 across 2.3.1/2.3.2/2.3.3 — keep only matching.
        if spec.package == "secondary_sign" and str(pdd) != spec.pdd_code:
            continue
        status = selection.get(scene_dir.name, "keep" if not selection else "unknown")
        net_path = scene_dir / (meta.get("net_file") or "map.net.xml")
        if not net_path.exists():
            nets = list(scene_dir.glob("*.net.xml"))
            net_path = nets[0] if nets else net_path
        row: dict[str, Any] = {
            "source": "package",
            "pdd_code": str(pdd),
            "category": spec.category,
            "scene_id": scene_dir.name,
            "package": spec.package,
            "selection_status": status,
            "latitude": meta.get("latitude"),
            "longitude": meta.get("longitude"),
            "crop_radius_m": meta.get("crop_radius_m"),
            "junction_arm_count": meta.get("junction_arm_count")
            or meta.get("catalog_junction_arm_count"),
            "scene_kind": meta.get("scene_kind"),
        }
        if parse_nets and net_path.exists():
            row.update(parse_net_xml(net_path))
        rows.append(row)
    return rows


def estimate_agents(row: dict[str, Any], mode: str) -> Optional[float]:
    """Nominal agents at scenario start / target traffic level.

    - aux_convoy: 1 ego + convoy_size × lanes_occupied
    - density: 1 ego + nuPlan vehicles/frame (or density × 80)
    - density_ped: density agents + pedestrians
    - speed_ego: ego-centric speed / zone compliance (1 agent)
    - detour_ego: ego-centric detour / obstacle compliance (1 agent)
    """
    if mode == "catalog_only":
        return None
    if mode in {"speed_ego", "detour_ego"}:
        return 1.0

    if mode == "aux_convoy":
        if not row.get("auxiliary_agent"):
            return 1.0
        convoy = int(row.get("aux_convoy_size") or 0)
        lanes = int(row.get("aux_lanes_occupied") or 0)
        keys = row.get("aux_occupied_lane_keys")
        if isinstance(keys, list) and keys:
            lanes = max(lanes, len(keys))
        return float(1 + convoy * max(lanes, 1))

    n_bg: Optional[float] = None
    if row.get("nuplan_vehicles_per_frame") is not None:
        n_bg = float(row["nuplan_vehicles_per_frame"])
    elif row.get("traffic_density") is not None:
        n_bg = float(row["traffic_density"]) * META_DENSITY_SCALE

    n = 1.0 + (n_bg or 0.0)
    if mode == "density_ped":
        ped = row.get("pedestrian_count")
        if ped is None:
            pm = row.get("pedestrian_manager") or {}
            ped = pm.get("target_pedestrian_count") or pm.get("max_pedestrians") or 0
        n += float(ped or 0)
    return n


def _scenario_row_from_raw(spec: SignSpec, raw: dict[str, Any], pdd: str) -> dict[str, Any]:
    horizon = raw.get("horizon") or raw.get("profile_horizon_steps")
    if horizon is None and spec.default_horizon_steps is not None:
        horizon = spec.default_horizon_steps
    agents = estimate_agents(raw, spec.agent_mode)
    dens = raw.get("traffic_density")
    if dens is None:
        dens = raw.get("profile_traffic_density")
    return {
        "pdd_code": pdd,
        "category": spec.category,
        "agent_mode": spec.agent_mode,
        "scene_id": raw.get("scene_id") or raw.get("scene_name"),
        "horizon_steps": horizon,
        "horizon_seconds": (float(horizon) * DT_SECONDS) if horizon is not None else None,
        "traffic_density": dens,
        "traffic_density_level": raw.get("traffic_density_level_name"),
        "nuplan_vehicles_per_frame": raw.get("nuplan_vehicles_per_frame"),
        "aux_convoy_size": raw.get("aux_convoy_size"),
        "aux_lanes_occupied": raw.get("aux_lanes_occupied"),
        "pedestrian_count": raw.get("pedestrian_count"),
        "estimated_agents": agents,
        "latitude": raw.get("latitude") or raw.get("center_lat"),
        "longitude": raw.get("longitude") or raw.get("center_lon"),
        "crop_radius_m": raw.get("crop_radius_m"),
        "n_lanes": raw.get("n_lanes"),
        "v_target_kmh": raw.get("v_target_kmh"),
        "spawn_mode": raw.get("spawn_mode"),
        "manifest_source": "external_catalog"
        if spec.external_catalog
        else "final_metrics_v1",
    }


def collect_scenarios_from_external(spec: SignSpec) -> list[dict[str, Any]]:
    path = spec.external_catalog
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            pdd = str(raw.get("sign_code") or raw.get("pdd_code") or "")
            if pdd != spec.pdd_code:
                continue
            rows.append(_scenario_row_from_raw(spec, raw, pdd))
    return rows


def collect_external_unique_scenes(
    spec: SignSpec, scenario_rows: list[dict[str, Any]], parse_nets: bool
) -> list[dict[str, Any]]:
    """One inventory row per unique scene_id from the external speed catalog."""
    if not scenario_rows:
        return []
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    # Re-read catalog for net_path / sign_id (not kept on scenario rows).
    net_by_scene: dict[str, str] = {}
    sign_id_by_scene: dict[str, Any] = {}
    if spec.external_catalog and spec.external_catalog.exists():
        with spec.external_catalog.open() as f:
            for line in f:
                raw = json.loads(line)
                if str(raw.get("sign_code") or "") != spec.pdd_code:
                    continue
                sid = str(raw.get("scene_id") or "")
                if sid and sid not in net_by_scene:
                    net_by_scene[sid] = str(raw.get("net_path") or "")
                    sign_id_by_scene[sid] = raw.get("sign_id")
    for sc in scenario_rows:
        sid = str(sc.get("scene_id") or "")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        row: dict[str, Any] = {
            "source": "external_catalog",
            "pdd_code": spec.pdd_code,
            "category": spec.category,
            "scene_id": sid,
            "latitude": sc.get("latitude"),
            "longitude": sc.get("longitude"),
            "n_lanes": sc.get("n_lanes"),
        }
        if parse_nets:
            sign_id = sign_id_by_scene.get(sid)
            net_path = None
            if sign_id is not None:
                cand = SPEED_MAPS_ROOT / f"sign_{sign_id}.net.xml"
                if cand.exists():
                    net_path = cand
            if net_path is None and net_by_scene.get(sid):
                # Fall back to local scenes tree if mirrored.
                cand2 = SCENES_ROOT / net_by_scene[sid]
                if cand2.exists():
                    net_path = cand2
            if net_path is not None:
                row.update(parse_net_xml(net_path))
        rows.append(row)
    return rows


def collect_scenarios(spec: SignSpec) -> list[dict[str, Any]]:
    if spec.external_catalog is not None:
        return collect_scenarios_from_external(spec)

    path = package_final_manifest(spec)
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            pdd = str(raw.get("pdd_code") or raw.get("sign_code") or spec.pdd_code)
            if pdd != spec.pdd_code:
                # Shared 2_3 / similar packages: filter to this subtype.
                continue
            rows.append(_scenario_row_from_raw(spec, raw, pdd))
    return rows


# Display families for realized duration. Raw CSV names map via `_duration_family`.
DURATION_FAMILIES = ("idm", "idm_rule", "plant2", "plant_rule", "carl", "carl_rule")


def _duration_family(baseline: str) -> Optional[str]:
    """Map eval baseline name → reporting family."""
    bl = (baseline or "").lower()
    if bl.endswith("_old"):
        return None
    # idm_rule aliases used across packages / speed-detour evals
    if (
        bl.startswith("idm_rule")
        or bl.startswith("modified_idm")
        or bl.startswith("comprehensive_rule_expert")
    ):
        return "idm_rule"
    if bl.startswith("idm"):
        return "idm"
    if bl.startswith("plant2_rule") or bl.startswith("plant_rule"):
        return "plant_rule"
    if bl.startswith("plant2"):
        return "plant2"
    if bl.startswith("carl_rule"):
        return "carl_rule"
    if bl.startswith("carl"):
        return "carl"
    return None


def collect_realized_duration(spec: SignSpec) -> Optional[dict[str, Any]]:
    """Average realized episode length (steps→seconds) per policy family."""
    if spec.metrics_csv is not None and spec.metrics_csv.exists():
        out = _realized_from_metrics_csv(spec, spec.metrics_csv)
        if out is not None:
            return out

    path = package_eval_agg(spec)
    if path is not None and path.exists():
        out = _realized_from_agg_csv(spec, path)
        if out is not None:
            return out

    pkg_metrics = package_metrics_csv(spec)
    if pkg_metrics is not None and pkg_metrics.exists():
        return _realized_from_metrics_csv(spec, pkg_metrics)
    return None


def _pack_realized_families(
    buckets: dict[str, list[tuple[float, float]]],
    *,
    source: str,
    filtered_by_pdd: bool = False,
) -> Optional[dict[str, Any]]:
    """``buckets[family]`` = list of (avg_steps, weight) rows."""
    by_baseline: dict[str, dict[str, Any]] = {}
    for fam in DURATION_FAMILIES:
        rows = buckets.get(fam) or []
        if not rows:
            continue
        wsum = sum(s * n for s, n in rows)
        nsum = sum(n for _, n in rows)
        if nsum <= 0:
            continue
        avg_steps = wsum / nsum
        by_baseline[fam] = {
            "avg_steps": avg_steps,
            "avg_seconds": avg_steps * DT_SECONDS,
            "n_episodes_weighted": nsum,
        }
    if not by_baseline:
        return None
    primary = by_baseline.get("idm") or next(iter(by_baseline.values()))
    return {
        "avg_steps": primary["avg_steps"],
        "avg_seconds": primary["avg_seconds"],
        "n_episodes_weighted": primary["n_episodes_weighted"],
        "baselines_used": "idm" if "idm" in by_baseline else next(iter(by_baseline)),
        "by_baseline": by_baseline,
        "source": source,
        "filtered_by_pdd": filtered_by_pdd,
    }


def _realized_from_agg_csv(spec: SignSpec, path: Path) -> Optional[dict[str, Any]]:
    import csv

    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    has_pdd = False
    with path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_pdd = "pdd_code" in fieldnames
        for r in reader:
            if has_pdd and str(r.get("pdd_code") or "") != spec.pdd_code:
                continue
            fam = _duration_family(str(r.get("baseline") or ""))
            if fam is None:
                continue
            try:
                steps = float(r["avg_steps"])
                n = float(r.get("n") or 0)
            except (KeyError, ValueError, TypeError):
                continue
            buckets[fam].append((steps, n))
    return _pack_realized_families(
        buckets, source=str(path), filtered_by_pdd=has_pdd
    )


def _realized_from_metrics_csv(spec: SignSpec, path: Path) -> Optional[dict[str, Any]]:
    import csv

    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            code = str(r.get("pdd_code") or "")
            if code and code != spec.pdd_code:
                continue
            fam = _duration_family(str(r.get("baseline") or r.get("policy") or ""))
            if fam is None:
                continue
            try:
                steps = float(r["final_step"])
            except (KeyError, ValueError, TypeError):
                continue
            buckets[fam].append((steps, 1.0))
    return _pack_realized_families(buckets, source=str(path))


def aggregate_sign(
    spec: SignSpec,
    catalog_rows: list[dict[str, Any]],
    package_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    kept = [r for r in package_rows if r.get("selection_status") == "keep"]
    rejected = [r for r in package_rows if r.get("selection_status") == "reject"]

    agent_vals = [
        float(r["estimated_agents"])
        for r in scenario_rows
        if r.get("estimated_agents") is not None
    ]
    horizon_s = [
        float(r["horizon_seconds"])
        for r in scenario_rows
        if r.get("horizon_seconds") is not None
    ]
    dens_vals = [
        float(r["traffic_density"])
        for r in scenario_rows
        if r.get("traffic_density") is not None
    ]
    convoy = [
        float(r["aux_convoy_size"])
        for r in scenario_rows
        if r.get("aux_convoy_size") is not None
    ]
    ped = [
        float(r["pedestrian_count"])
        for r in scenario_rows
        if r.get("pedestrian_count") is not None
    ]
    lanes = [
        float(r["n_lanes"])
        for r in (package_rows or catalog_rows or scenario_rows)
        if r.get("n_lanes") is not None
    ]
    edges = [
        float(r["n_edges"])
        for r in (package_rows or catalog_rows)
        if r.get("n_edges") is not None
    ]
    arms = [
        float(r["junction_arm_count"])
        for r in (package_rows or catalog_rows)
        if r.get("junction_arm_count") is not None
    ]
    lats = [
        float(r["latitude"])
        for r in (scenario_rows or package_rows or catalog_rows)
        if r.get("latitude") is not None
    ]
    lons = [
        float(r["longitude"])
        for r in (scenario_rows or package_rows or catalog_rows)
        if r.get("longitude") is not None
    ]

    dens_levels = Counter(
        r.get("traffic_density_level")
        for r in scenario_rows
        if r.get("traffic_density_level")
    )
    v_targets = Counter(
        r.get("v_target_kmh") for r in scenario_rows if r.get("v_target_kmh") is not None
    )
    spawn_modes = Counter(
        r.get("spawn_mode") for r in scenario_rows if r.get("spawn_mode")
    )

    realized = collect_realized_duration(spec)

    local_catalog = [r for r in catalog_rows if r.get("source") == "catalog"]
    external_scenes = [r for r in catalog_rows if r.get("source") == "external_catalog"]
    # Prefer external unique scenes as the catalog count when present (balanced speed run).
    n_catalog = (
        len(external_scenes)
        if external_scenes
        else len(local_catalog) if local_catalog else len(catalog_rows)
    )

    manifest_ok = False
    if spec.external_catalog is not None:
        manifest_ok = spec.external_catalog.exists()
    else:
        m = package_final_manifest(spec)
        manifest_ok = bool(m is not None and m.exists())

    return {
        "pdd_code": spec.pdd_code,
        "name": spec.name,
        "category": spec.category,
        "agent_mode": spec.agent_mode,
        "n_catalog_scenes": n_catalog,
        "n_local_catalog_scenes": len(local_catalog),
        "n_external_unique_scenes": len(external_scenes),
        "n_package_scenes": len(package_rows),
        "n_package_kept": len(kept) if package_rows else None,
        "n_package_rejected": len(rejected) if package_rows else None,
        "n_scenarios": len(scenario_rows),
        "n_unique_scenes_in_manifest": len(
            {r["scene_id"] for r in scenario_rows if r.get("scene_id")}
        ),
        "agents": _summarize_numeric(agent_vals),
        "horizon_seconds": _summarize_numeric(horizon_s),
        "realized_duration": realized,
        "traffic_density": _summarize_numeric(dens_vals),
        "aux_convoy_size": _summarize_numeric(convoy),
        "pedestrian_count": _summarize_numeric(ped),
        "map_lanes": _summarize_numeric(lanes),
        "map_edges": _summarize_numeric(edges),
        "junction_arms": _summarize_numeric(arms),
        "density_level_counts": dict(dens_levels),
        "v_target_kmh_counts": {str(k): v for k, v in v_targets.items()},
        "spawn_mode_counts": dict(spawn_modes),
        "geo": {
            "n_with_coords": len(lats),
            "lat_min": min(lats) if lats else None,
            "lat_max": max(lats) if lats else None,
            "lon_min": min(lons) if lons else None,
            "lon_max": max(lons) if lons else None,
            "lat_mean": _mean(lats),
            "lon_mean": _mean(lons),
        },
        "manifest_available": manifest_ok,
        "notes": _notes_for(spec, catalog_rows, package_rows, scenario_rows),
    }


def _notes_for(
    spec: SignSpec,
    catalog_rows: list[dict[str, Any]],
    package_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
) -> list[str]:
    notes = []
    if spec.agent_mode == "catalog_only":
        notes.append(
            "Package / final_metrics_v1 manifest not present; catalog OSM crops only."
        )
    if spec.agent_mode == "speed_ego":
        notes.append(
            "Speed / zone signs from balanced run_v61_a6 catalog (map-trimmed 1.2k); "
            "ego-centric braking/accel scenarios (nominal 1 agent). "
            f"Configured horizon = {spec.default_horizon_steps} steps "
            f"({(spec.default_horizon_steps or 0) * DT_SECONDS:.0f} s)."
        )
        local_n = sum(1 for r in catalog_rows if r.get("source") == "catalog")
        ext_n = sum(1 for r in catalog_rows if r.get("source") == "external_catalog")
        if local_n and ext_n:
            notes.append(
                f"Local OSM crops={local_n}; unique scenes in balanced catalog={ext_n} "
                "(catalog count uses the balanced unique scenes)."
            )
    if spec.agent_mode == "detour_ego":
        notes.append(
            "Detour signs from detour_v1/catalog.jsonl; ego-centric scenarios "
            f"(nominal 1 agent). Horizon = {spec.default_horizon_steps} steps "
            f"({(spec.default_horizon_steps or 0) * DT_SECONDS:.0f} s)."
        )
    if not catalog_rows and spec.catalog_dirs:
        notes.append(f"Missing catalog dirs: {list(spec.catalog_dirs)}")
    if (
        not catalog_rows
        and not spec.catalog_dirs
        and spec.external_catalog is None
    ):
        notes.append("No top-level OSM catalog mirror for this sign.")
    if spec.package and not package_rows:
        notes.append("No package scene directory found.")
    if (
        spec.package
        and not scenario_rows
        and spec.agent_mode != "catalog_only"
        and spec.external_catalog is None
    ):
        notes.append("No final_metrics_v1 scenarios matched this pdd_code.")
    if spec.external_catalog is not None and not scenario_rows:
        notes.append(f"No rows for this sign in {spec.external_catalog}")
    return notes


def build_overview(sign_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    n_catalog = sum(s["n_catalog_scenes"] for s in sign_summaries)
    n_package = sum(s["n_package_scenes"] for s in sign_summaries)
    n_kept = sum(s["n_package_kept"] or 0 for s in sign_summaries)
    n_scenarios = sum(s["n_scenarios"] for s in sign_summaries)

    # Weighted agent / duration means across scenarios.
    agent_means = []
    agent_weights = []
    horizon_means = []
    horizon_weights = []
    realized_secs = []
    realized_w = []
    realized_by_family: dict[str, list[tuple[float, float]]] = defaultdict(list)
    by_mode: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for s in sign_summaries:
        a = s["agents"]
        if a.get("n", 0) > 0 and a.get("mean") is not None:
            agent_means.append(a["mean"])
            agent_weights.append(a["n"])
            by_mode[s["agent_mode"]].append((a["mean"], a["n"]))
        h = s["horizon_seconds"]
        if h.get("n", 0) > 0 and h.get("mean") is not None:
            horizon_means.append(h["mean"])
            horizon_weights.append(h["n"])
        rd = s.get("realized_duration") or {}
        if rd.get("avg_seconds") is not None:
            realized_secs.append(rd["avg_seconds"])
            realized_w.append(rd.get("n_episodes_weighted") or 1.0)
        for fam, payload in (rd.get("by_baseline") or {}).items():
            if payload.get("avg_seconds") is not None:
                realized_by_family[fam].append(
                    (payload["avg_seconds"], payload.get("n_episodes_weighted") or 1.0)
                )

    def wmean(vals, weights):
        if not vals:
            return None
        return sum(v * w for v, w in zip(vals, weights)) / sum(weights)

    agents_by_mode = {}
    for mode, pairs in by_mode.items():
        agents_by_mode[mode] = {
            "mean": wmean([m for m, _ in pairs], [w for _, w in pairs]),
            "n_scenarios": int(sum(w for _, w in pairs)),
        }

    dist = {
        s["pdd_code"]: {
            "n_scenarios": s["n_scenarios"],
            "pct_scenarios": (100.0 * s["n_scenarios"] / n_scenarios) if n_scenarios else 0.0,
            "n_catalog_scenes": s["n_catalog_scenes"],
            "n_package_scenes": s["n_package_scenes"],
            "category": s["category"],
        }
        for s in sign_summaries
    }

    by_cat: dict[str, dict[str, float]] = defaultdict(lambda: {"n_scenarios": 0, "n_catalog": 0})
    for s in sign_summaries:
        by_cat[s["category"]]["n_scenarios"] += s["n_scenarios"]
        by_cat[s["category"]]["n_catalog"] += s["n_catalog_scenes"]

    return {
        "n_signs": len(sign_summaries),
        "n_catalog_scenes": n_catalog,
        "n_package_scenes": n_package,
        "n_package_kept": n_kept,
        "n_scenarios": n_scenarios,
        "mean_agents_per_scenario": wmean(agent_means, agent_weights),
        "mean_agents_by_mode": agents_by_mode,
        "mean_configured_horizon_s": wmean(horizon_means, horizon_weights),
        "mean_realized_duration_s": wmean(realized_secs, realized_w),
        "mean_realized_duration_by_baseline_s": {
            fam: wmean([v for v, _ in pairs], [w for _, w in pairs])
            for fam, pairs in realized_by_family.items()
        },
        "realized_duration_coverage_note": (
            "Realized duration from eval aggregations / metrics_per_episode: "
            "idm (`idm_*`), idm_rule (`modified_idm_*` / `comprehensive_rule_expert_*`), "
            "plant2 (`plant2_default`), plant_rule (`plant2_rule_default`), "
            "carl (`carl_default`), carl_rule (`carl_rule_default`)."
        ),
        "dt_seconds": DT_SECONDS,
        "sign_distribution": dist,
        "category_distribution": dict(by_cat),
        "definition_notes": [
            "catalog_scenes = OSM crops under pdd-bench/scenes/<sign>/ "
            "(for speed signs: unique scene_ids in balanced run_v61_a6 catalog.jsonl)",
            "package_scenes = filtered junction/crosswalk crops under per_sign_bench/*/scenes/",
            "scenarios = final_metrics_v1 rows OR speed catalog.jsonl rows",
            "agents = nominal count: ego + aux convoy×lanes OR ego + nuPlan vehicles/frame "
            "(+ pedestrians for 5.19); speed_ego/detour_ego = 1",
            "configured horizon = manifest horizon_steps × 0.1 s (MetaDrive dt); "
            "speed signs default to 1500 steps (150 s); detour to 1200 steps (120 s)",
            "realized duration = weighted avg_steps/final_step × 0.1 s "
            "per family: idm, idm_rule, plant2, plant_rule, carl, carl_rule",
        ],
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    parser.add_argument(
        "--parse-nets",
        action="store_true",
        default=True,
        help="Parse SUMO net.xml for lane/edge/junction counts (default: on)",
    )
    parser.add_argument(
        "--no-parse-nets",
        action="store_true",
        help="Skip net.xml parsing (faster)",
    )
    args = parser.parse_args()
    parse_nets = not args.no_parse_nets

    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_scene_rows: list[dict[str, Any]] = []
    all_scenario_rows: list[dict[str, Any]] = []
    sign_summaries: list[dict[str, Any]] = []

    for spec in TARGET_SIGNS:
        print(f"[collect] {spec.pdd_code} …", flush=True)
        catalog_rows = collect_catalog_scenes(spec, parse_nets=parse_nets)
        package_rows = collect_package_scenes(spec, parse_nets=parse_nets)
        scenario_rows = collect_scenarios(spec)
        if spec.external_catalog is not None:
            # Unique scenes from the balanced speed catalog (authoritative for count).
            # Skip heavy net parsing by default for these; n_lanes comes from the catalog.
            ext_rows = collect_external_unique_scenes(
                spec, scenario_rows, parse_nets=False
            )
            catalog_rows = catalog_rows + ext_rows
        all_scene_rows.extend(catalog_rows)
        all_scene_rows.extend(package_rows)
        all_scenario_rows.extend(scenario_rows)
        summary = aggregate_sign(spec, catalog_rows, package_rows, scenario_rows)
        sign_summaries.append(summary)
        print(
            f"  catalog={summary['n_catalog_scenes']} "
            f"package={summary['n_package_scenes']} "
            f"scenarios={summary['n_scenarios']} "
            f"mean_agents={summary['agents'].get('mean')} "
            f"realized_s={(summary.get('realized_duration') or {}).get('avg_seconds')}",
            flush=True,
        )

    overview = build_overview(sign_summaries)

    n_scenes = write_jsonl(raw_dir / "scene_inventory.jsonl", all_scene_rows)
    n_scen = write_jsonl(raw_dir / "scenario_stats.jsonl", all_scenario_rows)
    (raw_dir / "sign_summary.json").write_text(
        json.dumps(sign_summaries, indent=2, ensure_ascii=False)
    )
    (raw_dir / "overview.json").write_text(
        json.dumps(overview, indent=2, ensure_ascii=False)
    )

    print(
        f"\nDone. scenes={n_scenes} scenarios={n_scen} "
        f"total_scenarios={overview['n_scenarios']} "
        f"mean_agents={overview['mean_agents_per_scenario']}",
        flush=True,
    )
    print(f"Wrote {raw_dir}", flush=True)


if __name__ == "__main__":
    main()
