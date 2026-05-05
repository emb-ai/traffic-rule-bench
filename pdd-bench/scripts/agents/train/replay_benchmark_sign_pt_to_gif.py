#!/usr/bin/env python3
"""
Replay saved benchmark-sign expert trajectories (.pt from
collect_benchmark_sign_trajectories.py) into top-down GIFs.

Resolves each episode's scene from ``sampled_dir/<slug>/materialized.jsonl``
(by ``scene_id``, then by ``seed``), rebuilds the env with
``prepare_benchmark_sign_env``, then steps with stored ``action_env``.

Usage (conda env with MetaDrive + traffic_signs, e.g. ``plant2``)::

    export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
    python replay_benchmark_sign_pt_to_gif.py \\
        --pt-dir /path/to/benchmark_sign_trajectories_v4 \\
        --glob '4_2_1_ep*.pt' \\
        --sampled-dir pdd-bench/scripts/per_sign_bench/benchmark_output/sampled_for_expert \\
        --output-dir /path/to/gifs_replay
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def _load_collect_module() -> Any:
    here = Path(__file__).resolve()
    collect_py = here.parent / "collect_benchmark_sign_trajectories.py"
    spec = importlib.util.spec_from_file_location(
        "collect_benchmark_sign_trajectories", str(collect_py)
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _index_materialized(
    jsonl_path: Path,
) -> Tuple[Dict[str, dict], Dict[int, dict]]:
    by_scene: Dict[str, dict] = {}
    by_seed: Dict[int, dict] = {}
    if not jsonl_path.is_file():
        return by_scene, by_seed
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("valid", True):
                continue
            sid = r.get("scene_id")
            if sid is not None:
                by_scene[str(sid)] = r
            seed = int(r.get("seed") or r.get("deterministic_seed") or -1)
            if seed >= 0:
                by_seed[seed] = r
    return by_scene, by_seed


def _find_row(
    ep: dict,
    by_scene: Dict[str, dict],
    by_seed: Dict[int, dict],
) -> Optional[dict]:
    sid = ep.get("scene_id")
    if sid is not None and str(sid) in by_scene:
        return by_scene[str(sid)]
    rs = int(ep.get("reset_seed", ep.get("base_seed", -1)))
    if rs >= 0 and rs in by_seed:
        return by_seed[rs]
    return None


def _replay_one(
    cb: Any,
    pt_path: Path,
    row: dict,
    pdd_code: str,
    sign_id: int,
    horizon: int,
    traffic_density: float,
    gif_path: Path,
) -> Dict[str, Any]:
    ep = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    steps: List[dict] = ep.get("steps") or []
    if not steps:
        return {"ok": False, "reason": "no steps in .pt"}

    prepared = cb.prepare_benchmark_sign_env(
        row, pdd_code, sign_id, horizon, traffic_density
    )
    if prepared is None:
        return {"ok": False, "reason": "prepare_benchmark_sign_env failed"}

    base_env = prepared["base_env"]
    backend = prepared["backend"]
    try:
        if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
            if hasattr(base_env.top_down_renderer, "_screen_frames"):
                base_env.top_down_renderer._screen_frames.clear()

        n_played = 0
        for st in steps:
            agent = getattr(base_env, "vehicle", None) or getattr(base_env, "agent", None)
            if agent is None:
                break
            a = np.asarray(st["action_env"], dtype=np.float64).reshape(-1)
            if a.size < 2:
                a = np.zeros(2, dtype=np.float64)
            action = np.asarray([float(a[0]), float(a[1])], dtype=np.float32)

            _, _r, terminated, truncated, _info = base_env.step(action)
            n_played += 1

            render_scaling = 24.0 if backend == "citymap" else 12.0
            try:
                base_env.render(
                    mode="top_down",
                    film_size=(2400, 2400),
                    scaling=render_scaling,
                    screen_size=(800, 800),
                    semantic_map=True,
                    semantic_broken_line=True,
                    draw_target_vehicle_trajectory=True,
                    target_agent_heading_up=True,
                    screen_record=True,
                    window=False,
                )
            except Exception:
                base_env.render(
                    mode="top_down",
                    screen_record=True,
                    window=False,
                    screen_size=(640, 640),
                )

            if terminated or truncated:
                break

        gif_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
            base_env.top_down_renderer.generate_gif(str(gif_path), duration=40)

        return {"ok": True, "steps_replayed": n_played, "gif": str(gif_path)}
    finally:
        try:
            base_env.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay benchmark_sign .pt trajectories to GIFs"
    )
    parser.add_argument(
        "--pt-dir",
        type=str,
        required=True,
        help="Directory containing *_ep*.pt files",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.pt",
        help="Glob under pt-dir (default all .pt)",
    )
    parser.add_argument(
        "--sampled-dir",
        type=str,
        default=None,
        help="Root with <slug>/materialized.jsonl (default: collector default)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to write GIF files",
    )
    parser.add_argument("--traffic-density", type=float, default=0.1)
    parser.add_argument(
        "--horizon",
        type=int,
        default=600,
        help="Env max steps (should exceed stored trajectory length)",
    )
    args = parser.parse_args()

    cb = _load_collect_module()
    sampled_root = Path(args.sampled_dir or cb.DEFAULT_SAMPLED_DIR)
    pt_dir = Path(args.pt_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(pt_dir.glob(args.glob))
    if not paths:
        print(f"No files match {pt_dir}/{args.glob}")
        sys.exit(1)

    print(f"Replay {len(paths)} file(s) → {out_dir}")

    ok_n = 0
    for pt_path in paths:
        name = pt_path.stem
        if "_ep" not in name:
            print(f"  skip (expected slug_epNNN): {pt_path.name}")
            continue
        slug, _, ep_part = name.partition("_ep")
        pdd_code = slug.replace("_", ".")
        jsonl = sampled_root / slug / "materialized.jsonl"
        by_scene, by_seed = _index_materialized(jsonl)

        ep = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        row = _find_row(ep, by_scene, by_seed)
        if row is None:
            print(f"  [{pt_path.name}] FAIL no materialized row (jsonl={jsonl})")
            continue

        sign_id = int(ep.get("sign_id", cb.SIGN_ID_MAP.get(pdd_code, 0)))
        gif_path = out_dir / f"{name}_replay.gif"
        print(f"  [{pt_path.name}] scene={ep.get('scene_id')} → {gif_path.name}")
        info = _replay_one(
            cb, pt_path, row, pdd_code, sign_id,
            max(args.horizon, len(ep.get("steps", [])) + 50),
            args.traffic_density,
            gif_path,
        )
        print(f"      {info}")
        if info.get("ok"):
            ok_n += 1

    print(f"Done — {ok_n}/{len(paths)} OK")


if __name__ == "__main__":
    main()
