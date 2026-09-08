#!/usr/bin/env python3
"""Render MetaDrive top-down first-frame PNGs for crosswalk scenes.

Same camera/style as eval GIFs (semantic top-down, heading-up, local film),
but only the spawn frame — ego at ``spawn_distance_before_end`` from the zebra
(from ``configs/shared/crosswalk.yaml``), 5.19 plate, zebra geom.

Examples::

    python tools/render_crosswalk_metadrive_previews.py
    python tools/render_crosswalk_metadrive_previews.py --max 8 --force
    python tools/render_crosswalk_metadrive_previews.py \\
        --scenes-dir data/scenes/crosswalk --out-name metadrive_first.png

Writes ``metadrive_first.png`` into each scene dir (and optional ``--gallery``).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from traffic_bench.eval.engine.sim.top_down_local_film_patch import (  # noqa: E402
    apply_top_down_local_film_patch,
    set_top_down_local_film,
)
from traffic_bench.eval.run.env import (  # noqa: E402
    _apply_manifest_ego_destination,
    _apply_manifest_ego_spawn_lane,
    _build_sumo_env,
    _reposition_ego_at_along,
    _reposition_ego_before_lane_end,
)
from traffic_bench.eval.run.gif import _topdown_gif_film_and_scaling  # noqa: E402
from traffic_bench.eval.run.place import place_signs_for_row  # noqa: E402
from traffic_bench.eval.run.score import _unwrap_base_env  # noqa: E402
from traffic_bench.eval.signs.crosswalk.expand import (  # noqa: E402
    CrosswalkExpansionConfig,
    CrosswalkSimParams,
    expand_crosswalk_scene_entries,
)
from traffic_bench.eval.signs.crosswalk.place import (  # noqa: E402
    install_segment_crosswalk_geometry,
)
from traffic_bench.eval.sign_registry import get_profile, scenes_dir as profile_scenes_dir  # noqa: E402
from traffic_bench.scene_collection.sign_scenes.filter.selection import (  # noqa: E402
    is_reserved_scene_dir,
)

apply_top_down_local_film_patch()

OUT_NAME_DEFAULT = "metadrive_first.png"
_CROSSWALK_YAML = REPO_ROOT / "traffic_bench" / "eval" / "configs" / "shared" / "crosswalk.yaml"


def _load_meta(scene_dir: Path) -> dict[str, Any]:
    return json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))


def _sim_params_from_config() -> tuple[CrosswalkSimParams, float]:
    """Match eval: spawn/sign distances + gif window from ``crosswalk.yaml``."""
    sim_cfg: dict[str, Any] = {}
    ped_cfg: dict[str, Any] = {}
    gif_cfg: dict[str, Any] = {}
    if _CROSSWALK_YAML.is_file():
        try:
            import yaml

            raw = yaml.safe_load(_CROSSWALK_YAML.read_text(encoding="utf-8")) or {}
            sim_cfg = dict(raw.get("simulation") or {})
            ped_cfg = dict(raw.get("pedestrian") or {})
            gif_cfg = dict(raw.get("gif") or {})
        except Exception:
            pass
    return CrosswalkSimParams(
        n_variations=1,
        max_ego_lanes=1,
        max_pedestrian_presets=1,
        traffic_density=0.0,
        spawn_distance_before_end=float(sim_cfg.get("spawn_distance_before_end", 50.0)),
        sign_distance_before_end=float(sim_cfg.get("sign_distance_before_end", 12.0)),
        spawn_velocity_ms=float(sim_cfg.get("spawn_velocity_ms", 2.5)),
        min_hops_after_depart=int(sim_cfg.get("min_hops_after_depart", 0) or 0),
        ped_ego_spawn_distance_m=float(
            ped_cfg.get("default_ego_spawn_distance_m", 50.0)
        ),
        ped_speed_mean=float(ped_cfg.get("default_speed_mean", 1.2)),
        ped_speed_std=float(ped_cfg.get("default_speed_std", 0.2)),
        ped_spawn_gap_s=float(ped_cfg.get("default_spawn_gap_s", 2.5)),
        ped_yield_distance=float(ped_cfg.get("yield_distance", 5.0)),
        ped_no_stop_before_crosswalk_m=float(
            ped_cfg.get("no_stop_before_crosswalk_m", 3.0)
        ),
    ), float(gif_cfg.get("window_m", 130.0))


def _one_row_for_scene(
    scene_dir: Path, scenes_root: Path, sim: CrosswalkSimParams
) -> Optional[dict[str, Any]]:
    """Build a single expand row (first approach × first preset) for preview."""
    meta = _load_meta(scene_dir)
    expansion = CrosswalkExpansionConfig(layout=True, max_scenarios=1)
    entries = expand_crosswalk_scene_entries(
        scene_dir=scene_dir,
        scenes_root=scenes_root,
        meta=meta,
        net_path=scene_dir / str(meta.get("net_file") or "map.net.xml"),
        sim=sim,
        expansion=expansion,
    )
    return entries[0] if entries else None


def _frame_to_png(frame: Any, out_png: Path) -> None:
    from PIL import Image

    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        # MetaDrive top-down returns BGR via to_cv2_image
        arr = arr[..., :3][:, :, ::-1]
    Image.fromarray(arr.astype(np.uint8)).save(out_png)


def _preview_film_center_xy(env, *, window_m: float) -> Optional[tuple[float, float]]:
    """Shift film center forward when the approach behind ego is shorter than the window.

    Ego-centered 130 m windows leave a white void behind short approach stubs.
    Bias the center along heading so the lane start sits on the bottom edge and
    the road continues through the frame (spawn/sign distances stay from config).
    """
    vehicle = getattr(env, "agent", None) or getattr(env, "vehicle", None)
    if vehicle is None:
        return None
    try:
        pos = np.asarray(vehicle.position, dtype=np.float64)[:2]
        heading = float(vehicle.heading_theta)
    except Exception:
        return None
    behind = 0.0
    lane = getattr(vehicle, "lane", None)
    if lane is not None:
        try:
            if hasattr(lane, "local_coordinates"):
                long, _lat = lane.local_coordinates(pos)
                behind = max(0.0, float(long))
            else:
                behind = max(0.0, float(getattr(lane, "length", 0.0) or 0.0) - 1.0)
        except Exception:
            behind = 0.0
    half = max(1.0, float(window_m) * 0.5)
    # Keep a few metres of margin past the lane start so the cut is off-screen.
    shift = max(0.0, half - behind - 2.0)
    if shift <= 0.5:
        return float(pos[0]), float(pos[1])
    return (
        float(pos[0] + shift * math.cos(heading)),
        float(pos[1] + shift * math.sin(heading)),
    )


def render_scene_first_frame(
    scene_dir: Path,
    *,
    scenes_root: Path,
    out_png: Path,
    window_m: float,
    sim: Optional[CrosswalkSimParams] = None,
) -> str:
    sim = sim or CrosswalkSimParams(
        n_variations=1,
        max_ego_lanes=1,
        max_pedestrian_presets=1,
        traffic_density=0.0,
        spawn_distance_before_end=50.0,
    )
    row = _one_row_for_scene(scene_dir, scenes_root, sim)
    if row is None:
        return "no_approach"

    # Keep expand's relative net_path (scenes_root / net_path). Absolute paths
    # get mis-resolved under MetaDrive's asset root on some setups.
    row = dict(row)
    row["traffic_density"] = 0.0
    row["auxiliary_agent"] = False

    env = None
    try:
        env = _build_sumo_env(row, scenes_root=scenes_root, max_steps=2)
        seed = int(row.get("seed") or row.get("deterministic_seed") or 0) % 100000
        env.reset(seed=seed)
        base_env = _unwrap_base_env(env)

        _apply_manifest_ego_spawn_lane(base_env, row)
        spawn_along = row.get("spawn_along_m")
        spawn_before = float(row.get("spawn_distance_before_end") or 0.0)
        if spawn_along is not None:
            # Match eval episode: SUMO-edge marks → MetaDrive lane along.
            from traffic_bench.eval.engine.map.sumo_metadrive_along import (
                remap_sumo_along_to_metadrive,
                row_sumo_edge_length_m,
            )

            lane = getattr(base_env.vehicle, "lane", None)
            along_m = float(spawn_along)
            if lane is not None:
                along_m = remap_sumo_along_to_metadrive(
                    along_m,
                    sumo_edge_length_m=row_sumo_edge_length_m(row),
                    metadrive_lane_length_m=float(lane.length),
                )
            _reposition_ego_at_along(base_env, along_m)
        elif spawn_before > 0:
            _reposition_ego_before_lane_end(base_env, spawn_before)
        _apply_manifest_ego_destination(base_env, row)
        install_segment_crosswalk_geometry(base_env, row)
        place_signs_for_row(
            base_env,
            row,
            scenes_root=scenes_root,
            distance_before_end=float(
                row.get("sign_distance_before_end") or sim.sign_distance_before_end
            ),
            show_model=True,
        )

        screen_size = (800, 800)
        film_size, scaling = _topdown_gif_film_and_scaling(
            base_env,
            screen_size=screen_size,
            window_m=float(window_m),
        )
        # Override ego-centered film when the approach stub is shorter than the window.
        center = _preview_film_center_xy(base_env, window_m=float(window_m))
        if center is not None:
            # Match gif helper half_m so MetaDrive keeps the requested zoom.
            film_px = float(film_size[0])
            max_len = film_px / (float(scaling) + 0.1) - 4.0
            half_m = max(float(window_m), 0.5 * max_len)
            set_top_down_local_film(center, half_m=half_m)

        frame = base_env.render(
            mode="top_down",
            film_size=film_size,
            scaling=float(scaling),
            screen_size=screen_size,
            semantic_map=True,
            semantic_broken_line=True,
            draw_target_vehicle_trajectory=False,
            target_agent_heading_up=True,
            screen_record=False,
            window=False,
            text={
                "scene": scene_dir.name,
                "spawn_m": f"{spawn_before:.0f}",
            },
        )
        if frame is None:
            renderer = getattr(base_env, "top_down_renderer", None)
            frames = getattr(renderer, "_screen_frames", None) if renderer else None
            if frames:
                frame = frames[-1]
        if frame is None:
            return "no_frame"

        out_png.parent.mkdir(parents=True, exist_ok=True)
        _frame_to_png(frame, out_png)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"fail:{type(exc).__name__}:{exc}"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scenes-dir",
        type=Path,
        default=None,
        help="Default: data/scenes/crosswalk",
    )
    ap.add_argument("--out-name", default=OUT_NAME_DEFAULT)
    ap.add_argument(
        "--gallery",
        type=Path,
        default=None,
        help="Also copy PNGs into this flat folder as <scene_id>.png",
    )
    ap.add_argument(
        "--window-m",
        type=float,
        default=None,
        help="Default: gif.window_m from crosswalk.yaml (130)",
    )
    ap.add_argument("--max", type=int, default=0, help="Limit number of scenes (0 = all)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing PNGs")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    args = ap.parse_args(argv)

    sim, yaml_window = _sim_params_from_config()
    window_m = float(args.window_m) if args.window_m is not None else yaml_window

    scenes_root = (
        args.scenes_dir.expanduser().resolve()
        if args.scenes_dir is not None
        else profile_scenes_dir(get_profile("crosswalk"))
    )
    if not scenes_root.is_dir():
        print(f"ERROR: scenes dir missing: {scenes_root}", file=sys.stderr)
        return 2

    scene_dirs = sorted(
        p
        for p in scenes_root.iterdir()
        if p.is_dir()
        and not is_reserved_scene_dir(p.name)
        and (p / "meta.json").is_file()
        and (p / "map.net.xml").is_file()
    )
    if args.max and args.max > 0:
        scene_dirs = scene_dirs[: int(args.max)]

    if args.gallery is not None:
        args.gallery.mkdir(parents=True, exist_ok=True)

    print(
        f"[metadrive-preview] scenes={scenes_root} n={len(scene_dirs)} "
        f"window_m={window_m} spawn_before={sim.spawn_distance_before_end}"
    )
    stats = {"ok": 0, "skip": 0, "fail": 0}
    for i, scene_dir in enumerate(scene_dirs, 1):
        out_png = scene_dir / str(args.out_name)
        if out_png.is_file() and args.skip_existing and not args.force:
            stats["skip"] += 1
            print(f"  [{i}/{len(scene_dirs)}] skip {scene_dir.name}")
            continue
        status = render_scene_first_frame(
            scene_dir,
            scenes_root=scenes_root,
            out_png=out_png,
            window_m=float(window_m),
            sim=sim,
        )
        if status == "ok":
            stats["ok"] += 1
            if args.gallery is not None:
                shutil.copy2(out_png, args.gallery / f"{scene_dir.name}.png")
            print(f"  [{i}/{len(scene_dirs)}] ok {scene_dir.name} → {out_png.name}")
        elif status == "no_approach":
            stats["fail"] += 1
            print(f"  [{i}/{len(scene_dirs)}] fail {scene_dir.name}: no viable approach")
        else:
            stats["fail"] += 1
            print(f"  [{i}/{len(scene_dirs)}] {status} {scene_dir.name}")

    print(f"[metadrive-preview] {stats}")
    return 0 if stats["fail"] == 0 or stats["ok"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
