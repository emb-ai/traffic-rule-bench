#!/usr/bin/env python3
"""Generate evaluation manifest from catalog scenes (no-entry signs 3.1 / 3.2).

Each scene under ``scenes/<slug>/`` places the active sign at the catalog
``distance_from_start`` on ``road_id``. Ego spawns before the sign and is
routed a few edges past it so baselines drive through the forbidden segment.
``run_benchmark.py`` places ``NoEntrySign`` (3.1) or ``NoTrafficSign`` (3.2).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from lib.no_entry_route import (
    DEFAULT_SPAWN_MARGIN_BEFORE_SIGN_M,
    MIN_LENGTH_PAST_SIGN_M,
    MIN_SIGN_DISTANCE_FROM_START_M,
    count_drivable_lanes,
    destination_lane_id,
    edge_length_m,
    scene_geometry_ok,
    spawn_longitude_before_sign,
)
from lib.no_entry_sign_spec import (
    DEFAULT_PDD_CODE,
    SIGN_FAMILY,
    NoEntrySignSpec,
    get_no_entry_sign_spec,
    local_scenes_root,
)
from lib.scene_selection import is_reserved_scene_dir, is_scene_rejected
from lib.sumo_utils import CORE_SCENES_SUBDIR


SCRIPT_DIR = Path(__file__).parent.resolve()
RUN_BENCH_SCRIPT = SCRIPT_DIR / "run_benchmark.py"
PDD_BENCH_DIR = SCRIPT_DIR.parents[2]

# Hydra: paths.output_base: benchmark_output/${pdd_slug:${sign.pdd_code}}
# so ``sign.pdd_code=3.1`` lands in ``benchmark_output/3_1/…`` without a
# manual paths.output_base override.
OmegaConf.register_new_resolver(
    "pdd_slug",
    lambda code: str(code).replace(".", "_"),
    replace=True,
)
DEFAULT_CARL_CKPT = (
    PDD_BENCH_DIR / "checkpoints" / "carl" / "nuplan_51479_1B" / "model_best.pth"
)
DEFAULT_NN_CHECKPOINTS = {
    "carl": DEFAULT_CARL_CKPT,
    "carl_rule": DEFAULT_CARL_CKPT,
    "plant2": PDD_BENCH_DIR / "checkpoints" / "plant2_finetuned" / "plant2_supervised_2nd_final.pt",
    "plant2_rule": PDD_BENCH_DIR
    / "checkpoints"
    / "plant2_finetuned"
    / "plant2_supervised_2nd_final.pt",
}

# Active sign for this process (set in ``main`` from Hydra).
_ACTIVE_SIGN: NoEntrySignSpec = get_no_entry_sign_spec(DEFAULT_PDD_CODE)
PDD_CODE = _ACTIVE_SIGN.pdd_code
SIGN_TYPE = SIGN_FAMILY


def _set_active_sign(pdd_code: str | None) -> NoEntrySignSpec:
    global _ACTIVE_SIGN, PDD_CODE, SIGN_TYPE
    _ACTIVE_SIGN = get_no_entry_sign_spec(pdd_code)
    PDD_CODE = _ACTIVE_SIGN.pdd_code
    SIGN_TYPE = SIGN_FAMILY
    return _ACTIVE_SIGN


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------
@dataclass
class PathsConfig:
    scenes_dir: Optional[str] = None
    output_base: Optional[str] = None
    experiment_name: Optional[str] = None


@dataclass
class ScenarioConfig:
    n_variants: int = 1
    respect_scene_selection: bool = True
    spawn_margin_before_sign_m: float = DEFAULT_SPAWN_MARGIN_BEFORE_SIGN_M
    min_sign_distance_from_start_m: float = MIN_SIGN_DISTANCE_FROM_START_M
    min_length_past_sign_m: float = MIN_LENGTH_PAST_SIGN_M
    destination_hops: int = 2
    validate_metadrive_routes: bool = True


@dataclass
class SimulationConfig:
    spawn_velocity_ms: float = 2.5
    traffic_density: float = 0.0
    horizon: int = 600
    sign_distance_before_end: float = 0.0
    spawn_distance_before_end: float = 0.0


@dataclass
class GifConfig:
    enabled: bool = False
    policy: str = "idm"
    max_scenes: Optional[int] = None
    dry_run: bool = False
    hide_signs: bool = True
    dir: Optional[str] = None
    run_name: Optional[str] = None
    model_path: Optional[str] = None  # Required for carl/plant2; default from checkpoints/


@dataclass
class ManifestConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    gif: GifConfig = field(default_factory=GifConfig)


# -----------------------------------------------------------------------------
# Seed / scene helpers
# -----------------------------------------------------------------------------
def _stable_seed(scene_name: str, variant: int = 0, scenario_id: str = "") -> int:
    """Generate deterministic 32-bit seed from scene name, variant, and scenario."""
    h = hashlib.sha256()
    h.update(scene_name.encode("utf-8"))
    h.update(b"|")
    h.update(str(variant).encode("utf-8"))
    if scenario_id:
        h.update(b"|")
        h.update(scenario_id.encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")


def discover_scenes(
    scenes_dir: Path,
    *,
    respect_scene_selection: bool = True,
) -> List[Path]:
    """Find catalog scene directories (skip core/ reserved dirs and rejected scenes)."""
    scenes: List[Path] = []
    if not scenes_dir.is_dir():
        return scenes
    for entry in sorted(scenes_dir.iterdir()):
        if not entry.is_dir():
            continue
        if is_reserved_scene_dir(entry.name) or entry.name == CORE_SCENES_SUBDIR:
            continue
        if respect_scene_selection and is_scene_rejected(scenes_dir, entry.name):
            continue
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        net_file = meta.get("net_file", "map.net.xml")
        if (entry / net_file).exists():
            scenes.append(entry)
    return scenes


def load_scene_metadata(scene_dir: Path) -> Dict[str, Any]:
    """Load scene metadata from meta.json (and optional center.json)."""
    meta_path = scene_dir / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    center_path = scene_dir / "center.json"
    if center_path.exists():
        with open(center_path, "r", encoding="utf-8") as f:
            center = json.load(f)
            meta["center_lat"] = center.get("lat")
            meta["center_lon"] = center.get("lon")
    return meta


def _scene_id_from_meta(meta: Dict[str, Any], scene_dir: Path) -> str:
    """Prefer ``sign_<id>`` when catalog sign_id is present; else folder name."""
    sign_id = meta.get("sign_id")
    if sign_id is not None and str(sign_id).strip() != "":
        try:
            return f"sign_{int(sign_id)}"
        except (TypeError, ValueError):
            return f"sign_{sign_id}"
    return str(meta.get("scene_name") or scene_dir.name)


def _optional_metadrive_route_ok(
    net_path: Path,
    *,
    road_id: str,
    dest_lane_id: str,
    pdd_code: str,
) -> Tuple[bool, str]:
    """Best-effort MetaDrive spawn→dest check; never hard-depends on dual-path APIs.

    Uses ``probe_road_network_for_net`` when available. On import/runtime failure,
    reports skip so the scene is kept (geometry already validated).
    """
    try:
        from lib.metadrive_route_check import (
            is_metadrive_path_ok,
            probe_road_network_for_net,
        )
        from lib.lane_keys import make_lane_key
    except Exception as exc:
        return True, f"metadrive check unavailable ({exc}); keeping scene"

    env = None
    try:
        env, road_network = probe_road_network_for_net(
            net_path,
            spawn_edge_id=road_id,
            spawn_lane_num=0,
            destination_lane_id=dest_lane_id,
            pdd_code=pdd_code,
        )
        start_lane = make_lane_key(road_id, 0)
        if start_lane not in road_network.graph or dest_lane_id not in road_network.graph:
            return False, f"lanes missing in MetaDrive graph ({start_lane} -> {dest_lane_id})"
        path = road_network.shortest_path(start_lane, dest_lane_id)
        if not is_metadrive_path_ok(path, spawn=start_lane, dest=dest_lane_id):
            return False, f"unroutable {start_lane} -> {dest_lane_id}"
        return True, "ok"
    except Exception as exc:
        return True, f"metadrive check failed ({exc}); keeping scene"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Manifest entry builder
# -----------------------------------------------------------------------------
def build_manifest_entry(
    scene_dir: Path,
    scenes_root: Path,
    meta: Dict[str, Any],
    variant: int,
    sim_cfg: SimulationConfig,
    scenario_cfg: ScenarioConfig,
) -> Optional[Dict[str, Any]]:
    """Build one manifest row for a catalog no-entry scene, or None if invalid."""
    scene_name = str(meta.get("scene_name") or scene_dir.name)
    net_file = meta.get("net_file", "map.net.xml")
    net_full_path = scene_dir / net_file
    net_rel = scene_dir.relative_to(scenes_root) / net_file

    road_id = str(meta.get("road_id") or "").strip()
    if not road_id:
        print(f"  [skip] {scene_name}: missing road_id")
        return None

    try:
        distance_from_start = float(meta["distance_from_start"])
    except (KeyError, TypeError, ValueError):
        print(f"  [skip] {scene_name}: missing/invalid distance_from_start")
        return None

    ok, reason = scene_geometry_ok(
        net_full_path,
        road_id,
        distance_from_start,
        min_sign_dist=scenario_cfg.min_sign_distance_from_start_m,
        min_past=scenario_cfg.min_length_past_sign_m,
    )
    if not ok:
        print(f"  [skip] {scene_name}: geometry {reason}")
        return None

    lane_length = edge_length_m(net_full_path, road_id)
    if lane_length is None:
        print(f"  [skip] {scene_name}: cannot read edge length for {road_id!r}")
        return None

    dest = destination_lane_id(
        net_full_path,
        road_id,
        hops=scenario_cfg.destination_hops,
    )
    if dest is None:
        print(f"  [skip] {scene_name}: no destination past signed edge")
        return None

    if scenario_cfg.validate_metadrive_routes:
        md_ok, md_reason = _optional_metadrive_route_ok(
            net_full_path,
            road_id=road_id,
            dest_lane_id=dest,
            pdd_code=PDD_CODE,
        )
        if not md_ok:
            print(f"  [skip] {scene_name}: MetaDrive route {md_reason}")
            return None
        if md_reason != "ok":
            print(f"  [metadrive] {scene_name}: {md_reason}")

    spawn_long = spawn_longitude_before_sign(
        distance_from_start,
        lane_length,
        margin_m=scenario_cfg.spawn_margin_before_sign_m,
    )
    scene_id = _scene_id_from_meta(meta, scene_dir)
    seed = _stable_seed(scene_name, variant, scenario_id=f"{road_id}|{dest}")
    n_lanes = count_drivable_lanes(net_full_path, road_id)

    entry: Dict[str, Any] = {
        "scene_id": scene_id,
        "scene_name": scene_name,
        "net_path": str(net_rel),
        "seed": seed,
        "var_idx": variant,
        "pdd_code": PDD_CODE,
        "sign_code": PDD_CODE,
        "sign_type": SIGN_TYPE,
        "sign_family": SIGN_FAMILY,
        "sign_title": _ACTIVE_SIGN.title,
        "sign_class": _ACTIVE_SIGN.class_name,
        "road_id": road_id,
        "spawn_lane_num": 0,
        # Exact catalog placement — no 30 m floor.
        "distance_from_start": distance_from_start,
        "sign_spawn_distance": distance_from_start,
        "spawn_longitude": spawn_long,
        "destination_lane_id": dest,
        "spawn_lane_length": lane_length,
        "n_lanes": n_lanes,
        "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
        "traffic_density": sim_cfg.traffic_density,
        "horizon": sim_cfg.horizon,
        "sign_distance_before_end": sim_cfg.sign_distance_before_end,
        "spawn_distance_before_end": sim_cfg.spawn_distance_before_end,
        "spawn_margin_before_sign_m": scenario_cfg.spawn_margin_before_sign_m,
        "destination_hops": scenario_cfg.destination_hops,
        "valid": True,
        "latitude": meta.get("latitude"),
        "longitude": meta.get("longitude"),
        "source_osm": meta.get("source_osm"),
        "osm_file": meta.get("osm_file"),
        "sign_id": meta.get("sign_id"),
        "net_file": net_file,
    }
    return {k: v for k, v in entry.items() if v is not None}


# -----------------------------------------------------------------------------
# Manifest generation
# -----------------------------------------------------------------------------
def generate_manifest(
    scenes_dir: Path,
    output_dir: Path,
    scenario_cfg: ScenarioConfig,
    sim_cfg: SimulationConfig,
) -> List[Dict[str, Any]]:
    """Generate real_manifest.jsonl from catalog no-entry scenes."""
    scenes = discover_scenes(
        scenes_dir,
        respect_scene_selection=scenario_cfg.respect_scene_selection,
    )
    entries: List[Dict[str, Any]] = []
    print(
        f"[no_entry_signs] Generating manifest for {PDD_CODE} "
        f"({_ACTIVE_SIGN.title}); scenes={len(scenes)}"
    )

    n_variants = max(1, int(scenario_cfg.n_variants))
    for scene_dir in scenes:
        meta = load_scene_metadata(scene_dir)
        scene_name = str(meta.get("scene_name") or scene_dir.name)
        print(f"\n=== {scene_name} ===")

        scene_pdd = str(meta.get("pdd_code") or meta.get("sign_type") or PDD_CODE)
        if scene_pdd != PDD_CODE:
            print(
                f"  [skip] scene pdd_code={scene_pdd!r} != active {PDD_CODE!r}"
            )
            continue

        scene_entries: List[Dict[str, Any]] = []
        for variant in range(n_variants):
            entry = build_manifest_entry(
                scene_dir=scene_dir,
                scenes_root=scenes_dir,
                meta=meta,
                variant=variant,
                sim_cfg=sim_cfg,
                scenario_cfg=scenario_cfg,
            )
            if entry is not None:
                scene_entries.append(entry)

        if not scene_entries:
            continue

        print(
            f"  ok: road={scene_entries[0]['road_id']} "
            f"dist={scene_entries[0]['distance_from_start']:.2f}m "
            f"spawn_long={scene_entries[0]['spawn_longitude']:.2f}m "
            f"dest={scene_entries[0]['destination_lane_id']} "
            f"rows={len(scene_entries)}"
        )
        entries.extend(scene_entries)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "real_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, default=str) + "\n")

    summary = {
        "pdd_code": PDD_CODE,
        "sign_type": SIGN_TYPE,
        "sign_family": SIGN_FAMILY,
        "sign_name": _ACTIVE_SIGN.title,
        "sign_class": _ACTIVE_SIGN.class_name,
        "sign_placement": "catalog distance_from_start on road_id (exact, no 30m floor)",
        "total_scenes": len(scenes),
        "total_entries": len(entries),
        "variants_per_scene": n_variants,
        "spawn_margin_before_sign_m": scenario_cfg.spawn_margin_before_sign_m,
        "min_sign_distance_from_start_m": scenario_cfg.min_sign_distance_from_start_m,
        "min_length_past_sign_m": scenario_cfg.min_length_past_sign_m,
        "destination_hops": scenario_cfg.destination_hops,
        "validate_metadrive_routes": scenario_cfg.validate_metadrive_routes,
        "spawn_velocity_ms": sim_cfg.spawn_velocity_ms,
        "traffic_density": sim_cfg.traffic_density,
        "horizon": sim_cfg.horizon,
        "generated_at": datetime.now().isoformat(),
        "scenes": [s.name for s in scenes],
    }

    summary_path = output_dir / "real_manifest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    manifest_meta_path = output_dir / "manifest.json"
    with open(manifest_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"entries_file": "real_manifest.jsonl", **summary},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[no_entry_signs] Wrote {len(entries)} entries -> {manifest_path}")
    return entries


# -----------------------------------------------------------------------------
# GIF rendering
# -----------------------------------------------------------------------------
def _iter_jsonl_rows(path: Path):
    """Iterate over JSONL file rows."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def render_gifs_from_manifest(
    manifest_path: Path,
    experiment_dir: Path,
    scenes_root: Path,
    gif_cfg: GifConfig,
) -> Tuple[int, int]:
    """Render GIFs for scenes from a manifest file via run_benchmark.py."""
    if not RUN_BENCH_SCRIPT.is_file():
        print(f"[GIF] run_benchmark.py not found at {RUN_BENCH_SCRIPT}", file=sys.stderr)
        return 0, 1

    if not manifest_path.is_file():
        print(f"[GIF] Manifest not found: {manifest_path}", file=sys.stderr)
        return 0, 1

    gif_dir = Path(gif_cfg.dir) if gif_cfg.dir else experiment_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    run_name = gif_cfg.run_name or experiment_dir.name

    rows: List[Dict[str, Any]] = []
    seen_keys: set[tuple] = set()

    for row in _iter_jsonl_rows(manifest_path):
        if not row.get("valid", True):
            continue
        if row.get("pdd_code") != PDD_CODE:
            continue

        scene_id = row.get("scene_id")
        seed = row.get("seed")
        if scene_id is None or seed is None:
            continue
        row_key = (scene_id, seed)
        if row_key in seen_keys:
            continue

        seen_keys.add(row_key)
        rows.append(row)

        if gif_cfg.max_scenes is not None and len(rows) >= gif_cfg.max_scenes:
            break

    if not rows:
        print(f"[GIF] No valid scenes found in manifest for {PDD_CODE}.")
        return 0, 0

    print(f"\n[GIF] Rendering {len(rows)} scene(s)...")

    model_path = gif_cfg.model_path
    if not model_path:
        default_ckpt = DEFAULT_NN_CHECKPOINTS.get(gif_cfg.policy)
        if default_ckpt is not None and Path(default_ckpt).is_file():
            model_path = str(default_ckpt)

    rendered = 0
    failed = 0
    for i, row in enumerate(rows, start=1):
        scene_uid = f"{row['scene_id']}:{row['pdd_code']}:{row['seed']}"
        cmd = [
            sys.executable,
            str(RUN_BENCH_SCRIPT),
            "--scene-uid", scene_uid,
            "--manifest", str(manifest_path),
            "--save-gifs",
            "--output-dir", str(experiment_dir),
            "--gif-dir", str(gif_dir),
            "--run-name", run_name,
            "--scenes-root", str(scenes_root),
            "--policy", gif_cfg.policy,
        ]
        if model_path:
            cmd.extend(["--model-path", model_path])
        if gif_cfg.hide_signs:
            cmd.append("--hide-signs")

        print(f"\n[GIF {i}/{len(rows)}] {scene_uid}")
        print("  " + " ".join(cmd))

        if gif_cfg.dry_run:
            rendered += 1
            continue

        seed_val = int(row.get("seed") or 0)
        var_idx = int(row.get("var_idx", 0) or 0)
        expected_gif = (
            gif_dir
            / f"{row['scene_id']}_v{var_idx}_s{seed_val}_{gif_cfg.policy}_default.gif"
        )
        if expected_gif.is_file():
            expected_gif.unlink()

        res = subprocess.run(cmd, cwd=str(RUN_BENCH_SCRIPT.parent))
        if res.returncode == 0 and expected_gif.is_file():
            rendered += 1
        else:
            failed += 1
            if res.returncode != 0:
                print(f"[GIF] Command failed with code {res.returncode}")
            else:
                print(
                    f"[GIF] Episode finished but GIF missing (likely bad route): "
                    f"{expected_gif.name}"
                )

    return rendered, failed


# -----------------------------------------------------------------------------
# Hydra entry point
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    sign_cfg = getattr(cfg, "sign", None)
    pdd_code = getattr(sign_cfg, "pdd_code", None) if sign_cfg is not None else None
    active = _set_active_sign(pdd_code)
    print(
        f"[no_entry_signs] Active sign {active.pdd_code} "
        f"({active.title}), class={active.class_name}"
    )

    scenes_dir_cfg = getattr(cfg.paths, "scenes_dir", None)
    scenes_base_cfg = getattr(cfg.paths, "scenes_base", "scenes") or "scenes"
    if scenes_dir_cfg in (None, "", "null"):
        scenes_dir = local_scenes_root(scenes_base_cfg, active.pdd_code)
    else:
        scenes_dir = Path(scenes_dir_cfg)
    if not scenes_dir.is_absolute():
        scenes_dir = (SCRIPT_DIR / scenes_dir).resolve()
    print(f"[no_entry_signs] Scenes dir: {scenes_dir}")

    experiment_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"[no_entry_signs] Output dir: {experiment_dir}")
    config_path = experiment_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    scenario_cfg = ScenarioConfig(
        n_variants=int(cfg.scenario.n_variants),
        respect_scene_selection=bool(
            getattr(cfg.scenario, "respect_scene_selection", True)
        ),
        spawn_margin_before_sign_m=float(
            getattr(
                cfg.scenario,
                "spawn_margin_before_sign_m",
                DEFAULT_SPAWN_MARGIN_BEFORE_SIGN_M,
            )
        ),
        min_sign_distance_from_start_m=float(
            getattr(
                cfg.scenario,
                "min_sign_distance_from_start_m",
                MIN_SIGN_DISTANCE_FROM_START_M,
            )
        ),
        min_length_past_sign_m=float(
            getattr(
                cfg.scenario,
                "min_length_past_sign_m",
                MIN_LENGTH_PAST_SIGN_M,
            )
        ),
        destination_hops=int(getattr(cfg.scenario, "destination_hops", 2)),
        validate_metadrive_routes=bool(
            getattr(cfg.scenario, "validate_metadrive_routes", True)
        ),
    )
    sim_cfg = SimulationConfig(
        spawn_velocity_ms=float(cfg.simulation.spawn_velocity_ms),
        traffic_density=float(cfg.simulation.traffic_density),
        horizon=int(cfg.simulation.horizon),
        sign_distance_before_end=float(
            getattr(cfg.simulation, "sign_distance_before_end", 0.0)
        ),
        spawn_distance_before_end=float(
            getattr(cfg.simulation, "spawn_distance_before_end", 0.0)
        ),
    )
    gif_cfg = GifConfig(
        enabled=bool(cfg.gif.enabled),
        policy=str(cfg.gif.policy),
        max_scenes=cfg.gif.max_scenes,
        dry_run=bool(cfg.gif.dry_run),
        hide_signs=bool(cfg.gif.hide_signs),
        dir=cfg.gif.dir,
        run_name=cfg.gif.run_name,
        model_path=getattr(cfg.gif, "model_path", None),
    )

    entries = generate_manifest(
        scenes_dir=scenes_dir,
        output_dir=experiment_dir,
        scenario_cfg=scenario_cfg,
        sim_cfg=sim_cfg,
    )

    if gif_cfg.enabled and entries:
        manifest_path = experiment_dir / "real_manifest.jsonl"
        gif_rendered, gif_failed = render_gifs_from_manifest(
            manifest_path=manifest_path,
            experiment_dir=experiment_dir,
            scenes_root=scenes_dir,
            gif_cfg=gif_cfg,
        )
        resolved_gif_dir = Path(gif_cfg.dir) if gif_cfg.dir else (experiment_dir / "gifs")
        print("\n[GIF RESULTS]")
        print(f"  - GIFs rendered: {gif_rendered}")
        print(f"  - GIF failures: {gif_failed}")
        print(f"  - Experiment directory: {experiment_dir}")
        print(f"  - GIF directory: {resolved_gif_dir}")

    print("\nOutput files:")
    print(f"  - Manifest: {experiment_dir / 'real_manifest.jsonl'}")
    print(f"  - Config: {config_path}")


if __name__ == "__main__":
    main()
