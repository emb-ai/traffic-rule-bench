#!/usr/bin/env python3
"""Generate real_manifest.jsonl for overtaking_sign (3.20).

Discovers cropped scenes under scenes/3_20/sign_*_s0 (or core/), validates
1+1 straight pairs, and writes one row per scene (+ optional density variants).

Usage:
  python generate_manifest.py
  python generate_manifest.py scenario.max_scenarios=20
  python generate_manifest.py gif.enabled=true gif.policy=idm gif.max_scenes=3
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from lib.manifest_config import (  # noqa: E402
    DEFAULT_AUX_FRAC,
    DEFAULT_MIN_EDGE_LENGTH_M,
    DEFAULT_SIGN_DISTANCE_FROM_START,
    DEFAULT_SPAWN_DISTANCE_FROM_START,
    DEFAULT_SPAWN_VELOCITY_MS,
    DEFAULT_WAIT_BEHIND_GAP_MAX_M,
    DEFAULT_WAIT_BEHIND_GAP_MIN_M,
    DEFAULT_WAIT_BEHIND_SPEED_MPS,
    DEFAULT_WAIT_BEHIND_SUCCESS_S,
)
from lib.straight_pair import analyze_road_pair  # noqa: E402

RUN_BENCH_SCRIPT = BENCH_DIR / "run_benchmark.py"
PDD_BENCH_DIR = BENCH_DIR.parents[2]
PDD_CODE = "3.20"
DEFAULT_CARL_CKPT = (
    PDD_BENCH_DIR / "checkpoints" / "carl" / "nuplan_51479_1B" / "model_best.pth"
)
DEFAULT_NN_CHECKPOINTS = {
    "carl": DEFAULT_CARL_CKPT,
    "carl_rule": DEFAULT_CARL_CKPT,
    "plant2": PDD_BENCH_DIR
    / "checkpoints"
    / "plant2_finetuned"
    / "plant2_supervised_2nd_final.pt",
    "plant2_rule": PDD_BENCH_DIR
    / "checkpoints"
    / "plant2_finetuned"
    / "plant2_supervised_2nd_final.pt",
}


def _pdd_slug(code: str) -> str:
    return str(code).replace(".", "_")


OmegaConf.register_new_resolver("pdd_slug", _pdd_slug, replace=True)


@dataclass
class ScenarioConfig:
    n_variants: int = 1
    n_variations: int = 1
    max_scenarios: Optional[int] = None
    max_scenarios_per_scene: int = 1
    min_ego_lane_m: float = DEFAULT_MIN_EDGE_LENGTH_M
    max_heading_std_deg: float = 12.0
    aux_frac: float = DEFAULT_AUX_FRAC


@dataclass
class SimulationConfig:
    spawn_velocity_ms: float = DEFAULT_SPAWN_VELOCITY_MS
    traffic_density: float = 0.0
    traffic_density_augment: bool = False
    horizon: int = 600
    sign_distance_from_start: float = DEFAULT_SIGN_DISTANCE_FROM_START
    spawn_distance_from_start: float = DEFAULT_SPAWN_DISTANCE_FROM_START
    wait_behind_success_seconds: float = DEFAULT_WAIT_BEHIND_SUCCESS_S
    wait_behind_speed_mps: float = DEFAULT_WAIT_BEHIND_SPEED_MPS
    wait_behind_gap_max_m: float = DEFAULT_WAIT_BEHIND_GAP_MAX_M
    wait_behind_gap_min_m: float = DEFAULT_WAIT_BEHIND_GAP_MIN_M


@dataclass
class GifConfig:
    enabled: bool = False
    policy: str = "idm"
    max_scenes: Optional[int] = None
    dry_run: bool = False
    hide_signs: bool = True
    dir: Optional[str] = None
    run_name: Optional[str] = None
    model_path: Optional[str] = None


def discover_crop_scenes(scenes_dir: Path) -> list[Path]:
    """Prefer sign_*_s0 crops; fall back to any scene with meta+net (not core/)."""
    if not scenes_dir.is_dir():
        return []
    crops = sorted(
        p for p in scenes_dir.iterdir()
        if p.is_dir()
        and p.name.endswith("_s0")
        and (p / "meta.json").is_file()
        and any(p.glob("*.net.xml"))
    )
    if crops:
        return crops
    return sorted(
        p for p in scenes_dir.iterdir()
        if p.is_dir()
        and p.name != "core"
        and (p / "meta.json").is_file()
        and any(p.glob("*.net.xml"))
    )


def _resolve_net(scene_dir: Path, meta: dict) -> Optional[Path]:
    nf = meta.get("net_file")
    if nf and (scene_dir / nf).is_file():
        return scene_dir / nf
    nets = sorted(scene_dir.glob("*.net.xml"))
    return nets[0] if nets else None


def generate_manifest(
    scenes_dir: Path,
    output_dir: Path,
    scenario: ScenarioConfig,
    sim: SimulationConfig,
    *,
    aux_frac: float,
) -> list[dict]:
    scenes = discover_crop_scenes(scenes_dir)
    rows: list[dict] = []
    for scene_dir in scenes:
        if scenario.max_scenarios is not None and len(rows) >= scenario.max_scenarios:
            break
        meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
        road_id = meta.get("road_id")
        net = _resolve_net(scene_dir, meta)
        if not road_id or net is None:
            continue
        pair = analyze_road_pair(
            net,
            str(road_id),
            min_length_m=scenario.min_ego_lane_m,
            max_heading_std_deg=scenario.max_heading_std_deg,
            aux_frac=aux_frac,
        )
        if pair is None:
            print(f"[skip] not 1+1/straight: {scene_dir.name}")
            continue
        seed_base = int(meta.get("sign_id") or hash(scene_dir.name) % 10_000_000)
        n_var = max(1, int(scenario.n_variations))
        for v in range(n_var):
            if scenario.max_scenarios is not None and len(rows) >= scenario.max_scenarios:
                break
            seed = seed_base + v * 100003
            entry = {
                "valid": True,
                "scene_id": f"{scene_dir.name}_v{v}",
                "sign_id": meta.get("sign_id"),
                "sign_type": "3.20",
                "sign_code": "3.20",
                "pdd_code": "3.20",
                "net_path": f"{scene_dir.name}/{net.name}",
                "road_id": pair.ego_edge,
                "opposite_edge_id": pair.opposite_edge,
                "spawn_lane_num": 0,
                "destination_edge_id": pair.destination_edge,
                "destination_lane_id": f"lane_{pair.destination_edge}_0",
                "aux_long_m": pair.aux_long_m,
                "aux_frac": float(aux_frac),
                "approach_length_m": pair.length_m,
                "heading_std_deg": pair.heading_std_deg,
                "sign_distance_from_start": float(sim.sign_distance_from_start),
                "spawn_distance_from_start": float(sim.spawn_distance_from_start),
                "wait_behind_success_seconds": float(sim.wait_behind_success_seconds),
                "wait_behind_speed_mps": float(sim.wait_behind_speed_mps),
                "wait_behind_gap_max_m": float(sim.wait_behind_gap_max_m),
                "wait_behind_gap_min_m": float(sim.wait_behind_gap_min_m),
                "spawn_velocity_ms": float(sim.spawn_velocity_ms),
                "traffic_density": float(sim.traffic_density),
                "horizon": int(sim.horizon),
                "seed": int(seed),
                "deterministic_seed": int(seed),
                "var_idx": int(v),
                "force_opposite_as_peer": True,
                "opposite_peer_side": "left",
                "probe_overtake_disable_wait_success": True,
            }
            rows.append(entry)
            print(
                f"[row] {entry['scene_id']}  {pair.ego_edge}  "
                f"L={pair.length_m:.1f}  aux@{pair.aux_long_m:.1f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "real_manifest.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    summary = {
        "pdd_code": "3.20",
        "n_rows": len(rows),
        "scenes_dir": str(scenes_dir),
        "sign_distance_from_start": sim.sign_distance_from_start,
        "spawn_distance_from_start": sim.spawn_distance_from_start,
        "wait_behind_success_seconds": sim.wait_behind_success_seconds,
        "min_ego_lane_m": scenario.min_ego_lane_m,
        "aux_frac": aux_frac,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {len(rows)} rows → {out_path}")
    return rows


def _iter_jsonl_rows(path: Path):
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
) -> tuple[int, int]:
    """Render GIFs for scenes from a manifest via run_benchmark.py."""
    if not RUN_BENCH_SCRIPT.is_file():
        print(f"[GIF] run_benchmark.py not found at {RUN_BENCH_SCRIPT}", file=sys.stderr)
        return 0, 1
    if not manifest_path.is_file():
        print(f"[GIF] Manifest not found: {manifest_path}", file=sys.stderr)
        return 0, 1

    gif_dir = Path(gif_cfg.dir) if gif_cfg.dir else experiment_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    run_name = gif_cfg.run_name or experiment_dir.name

    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple] = set()
    for row in _iter_jsonl_rows(manifest_path):
        if not row.get("valid", True):
            continue
        if str(row.get("pdd_code") or "") != PDD_CODE:
            continue
        scene_id = row.get("scene_id")
        seed = row.get("seed")
        if scene_id is None or seed is None:
            continue
        key = (scene_id, seed)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append(row)
        if gif_cfg.max_scenes is not None and len(rows) >= int(gif_cfg.max_scenes):
            break

    if not rows:
        print(f"[GIF] No valid scenes found in manifest for {PDD_CODE}.")
        return 0, 0

    print(f"\n[GIF] Rendering {len(rows)} scene(s) with policy={gif_cfg.policy}...")

    model_path = gif_cfg.model_path
    if not model_path:
        default_ckpt = DEFAULT_NN_CHECKPOINTS.get(gif_cfg.policy)
        if default_ckpt is not None and Path(default_ckpt).is_file():
            model_path = str(default_ckpt)
    if gif_cfg.policy in DEFAULT_NN_CHECKPOINTS and not model_path:
        print(
            f"[GIF] WARNING: policy={gif_cfg.policy} needs a checkpoint "
            f"(gif.model_path=... or default under checkpoints/)",
            file=sys.stderr,
        )

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

        res = subprocess.run(cmd, cwd=str(BENCH_DIR))
        if res.returncode == 0 and expected_gif.is_file():
            rendered += 1
        else:
            failed += 1
            if res.returncode != 0:
                print(f"[GIF] Command failed with code {res.returncode}")
            else:
                print(
                    f"[GIF] Episode finished but GIF missing: {expected_gif.name}"
                )

    return rendered, failed


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    pdd = str(cfg.sign.pdd_code)
    slug = _pdd_slug(pdd)
    scenes_dir = cfg.paths.scenes_dir
    if scenes_dir is None:
        scenes_dir = Path(cfg.paths.scenes_base) / slug
    else:
        scenes_dir = Path(scenes_dir)
    if not scenes_dir.is_absolute():
        scenes_dir = BENCH_DIR / scenes_dir

    out_dir = Path(cfg.paths.output_base) / str(cfg.paths.experiment_name)
    if not out_dir.is_absolute():
        out_dir = BENCH_DIR / out_dir

    scenario = ScenarioConfig(
        n_variants=int(cfg.scenario.n_variants),
        n_variations=int(getattr(cfg.scenario, "n_variations", 1) or 1),
        max_scenarios=cfg.scenario.max_scenarios,
        max_scenarios_per_scene=int(cfg.scenario.max_scenarios_per_scene or 1),
        min_ego_lane_m=float(cfg.scenario.min_ego_lane_m),
        max_heading_std_deg=float(getattr(cfg.scenario, "max_heading_std_deg", 12.0)),
        aux_frac=float(getattr(cfg.scenario, "aux_frac", DEFAULT_AUX_FRAC)),
    )
    sim = SimulationConfig(
        spawn_velocity_ms=float(cfg.simulation.spawn_velocity_ms),
        traffic_density=float(cfg.simulation.traffic_density),
        traffic_density_augment=bool(cfg.simulation.traffic_density_augment),
        horizon=int(cfg.simulation.horizon),
        sign_distance_from_start=float(cfg.simulation.sign_distance_from_start),
        spawn_distance_from_start=float(cfg.simulation.spawn_distance_from_start),
        wait_behind_success_seconds=float(cfg.simulation.wait_behind_success_seconds),
        wait_behind_speed_mps=float(cfg.simulation.wait_behind_speed_mps),
        wait_behind_gap_max_m=float(cfg.simulation.wait_behind_gap_max_m),
        wait_behind_gap_min_m=float(cfg.simulation.wait_behind_gap_min_m),
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
        scenes_dir,
        out_dir,
        scenario,
        sim,
        aux_frac=scenario.aux_frac,
    )

    if gif_cfg.enabled and entries:
        manifest_path = out_dir / "real_manifest.jsonl"
        gif_rendered, gif_failed = render_gifs_from_manifest(
            manifest_path=manifest_path,
            experiment_dir=out_dir,
            scenes_root=scenes_dir,
            gif_cfg=gif_cfg,
        )
        resolved_gif_dir = Path(gif_cfg.dir) if gif_cfg.dir else (out_dir / "gifs")
        print("\n[GIF RESULTS]")
        print(f"  - GIFs rendered: {gif_rendered}")
        print(f"  - GIF failures: {gif_failed}")
        print(f"  - Experiment directory: {out_dir}")
        print(f"  - GIF directory: {resolved_gif_dir}")

    print("\nOutput files:")
    print(f"  - Manifest: {out_dir / 'real_manifest.jsonl'}")


if __name__ == "__main__":
    main()
