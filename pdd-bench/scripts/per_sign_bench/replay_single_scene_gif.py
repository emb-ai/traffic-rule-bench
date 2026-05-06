#!/usr/bin/env python3
"""replay_single_scene_gif.py — replay ONE sign scene and record a GIF.

Usage:
  python3 replay_single_scene_gif.py \
    --sign 2.4 \
    --scene-id sumo_2.4_75912 \
    --backend sumo \
    --policy carl \
    --out /tmp/replay_2.4.gif \
    --src /home/jovyan/.../full_test_250_x10 \
    --scenes-root /home/jovyan/.../pdd-bench/scenes \
    --model-path /home/jovyan/.../carl_checkpoint.pth

  python3 replay_single_scene_gif.py \
    --sign 2.4 --backend pgmap --policy carl \
    --out /tmp/replay.gif \
    --src /home/jovyan/.../full_test_250_x10 \
    --scenes-root /home/jovyan/.../pdd-bench/scenes \
    --model-path /home/jovyan/.../carl_checkpoint.pth
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_benchmark_mini import (    # noqa: E402
    _build_pgmap_env, _build_sumo_env,
    _place_pgmap_sign, _unwrap_base_env,
    _load_jsonl_rows, _slug_to_code,
    _load_policy_models,
    IDMPolicy, ModifiedIDMPolicy, ExpertPolicy,
    ComprehensiveRuleExpertPolicy, RuleCompliantExpertPolicy,
    apply_ego_defaults,
)
import numpy as np
import torch


MANIFEST_NAMES = {
    "pgmap":   "pgmap_materialized.jsonl",
    "paired":  "paired_materialized.jsonl",
    "citymap": "citymap_materialized.jsonl",
}


def find_sumo_manifest(sign_dir: Path) -> Path | None:
    p1 = sign_dir / "sumo" / "sumo_manifest.jsonl"
    p2 = sign_dir / "real_manifest.jsonl"
    if p1.exists() and p1.stat().st_size > 0: return p1
    if p2.exists() and p2.stat().st_size > 0: return p2
    return None


def find_row(src: Path, sign_code: str, backend: str, scene_id: str | None) -> dict | None:
    sign_slug = sign_code.replace(".", "_")
    sign_dir = src / sign_slug
    if not sign_dir.is_dir():
        raise SystemExit(f"sign dir not found: {sign_dir}")

    if backend == "sumo":
        manifest = find_sumo_manifest(sign_dir)
    else:
        manifest = sign_dir / MANIFEST_NAMES.get(backend, "")
        if not manifest.exists() or manifest.stat().st_size == 0:
            manifest = None
    if manifest is None:
        raise SystemExit(f"manifest not found for sign={sign_code} backend={backend}")

    rows = _load_jsonl_rows(manifest)
    if not rows:
        raise SystemExit(f"manifest empty: {manifest}")

    if scene_id:
        for r in rows:
            if r.get("scene_id") == scene_id:
                if backend in ("pgmap", "paired", "citymap") and not r.get("sign_type"):
                    r["sign_type"] = sign_code
                return r
        raise SystemExit(f"scene_id {scene_id} not found in {manifest}")
    r = rows[0]
    if backend in ("pgmap", "paired", "citymap") and not r.get("sign_type"):
        r["sign_type"] = sign_code
    return r


def make_idm_policy(policy_type: str):
    return {
        "idm": IDMPolicy,
        "modified_idm": ModifiedIDMPolicy,
        "comprehensive_rule_expert": ComprehensiveRuleExpertPolicy,
        "rule_compliant": RuleCompliantExpertPolicy,
        "ppo_lidar": ExpertPolicy,
    }.get(policy_type)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sign", required=True, help="sign code, e.g. 2.4")
    p.add_argument("--backend", required=True, choices=["sumo", "pgmap", "paired", "citymap"])
    p.add_argument("--scene-id", default=None,
                   help="specific scene_id (if omitted — the first scene)")
    p.add_argument("--policy", required=True,
                   choices=["idm", "modified_idm", "comprehensive_rule_expert",
                            "rule_compliant", "ppo_5ch", "ppo_lidar",
                            "carl", "carl_rule", "plant2", "plant2_rule"])
    p.add_argument("--model-path", default=None,
                   help="required for carl/carl_rule/plant2/plant2_rule/ppo_5ch")
    p.add_argument("--out", required=True, help="output .gif path")
    p.add_argument("--src", required=True, help="full_test_250_x10")
    p.add_argument("--scenes-root", required=True, help="pdd-bench/scenes (for sumo)")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--gif-duration-ms", type=int, default=40,
                   help="gif frame duration (ms)")
    p.add_argument("--screen-size", type=int, default=800)
    p.add_argument("--scaling", type=float, default=12.0)
    args = p.parse_args()

    logging.getLogger().setLevel(logging.CRITICAL)

    src = Path(args.src).resolve()
    scenes_root = Path(args.scenes_root).resolve()
    out_gif = Path(args.out).resolve()
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    row = find_row(src, args.sign, args.backend, args.scene_id)
    print(f"Scene: {row.get('scene_id')}")
    print(f"  seed={row.get('seed')}  var_idx={row.get('var_idx')}")
    print(f"  sign_type={row.get('sign_type')}  backend={args.backend}")

    models = _load_policy_models(args.policy, args.model_path)

    seed = int(row.get("seed") or row.get("deterministic_seed") or 0)
    np.random.seed(seed); torch.manual_seed(seed)

    if args.backend in ("pgmap", "paired", "citymap"):
        env = _build_pgmap_env(row, max_steps=args.max_steps)
    else:
        env = _build_sumo_env(row, scenes_root=scenes_root, max_steps=args.max_steps)

    raw_env = env

    # 4. reset + place_sign
    if args.backend == "sumo":
        env_seed = (int(row.get("sign_id", 0)) + int(row.get("var_idx", 0))) % 100000
    elif args.backend == "citymap":
        env_seed = int(row.get("map_seed") or row.get("seed") or 0)
    else:
        env_seed = seed
    obs, _info = env.reset(seed=env_seed)
    base_env = _unwrap_base_env(env)

    if args.backend in ("pgmap", "paired", "citymap"):
        ok = _place_pgmap_sign(base_env, row, seed)
        if not ok:
            print("ERROR: failed to place pgmap sign")
            sys.exit(2)

    idm_cls = make_idm_policy(args.policy)
    policy_obj = None
    if idm_cls is not None:
        policy_obj = idm_cls(base_env.vehicle, seed)
        if args.policy in ("idm", "modified_idm", "comprehensive_rule_expert"):
            apply_ego_defaults(policy_obj)
    elif args.policy == "carl_rule":
        policy_obj = models["carl_rule_class"](base_env.vehicle, seed)
    elif args.policy == "plant2_rule":
        policy_obj = models["plant2_rule_class"](base_env.vehicle, seed)

    if args.policy == "carl" and models["carl_model"] is not None:
        models["carl_model"].reset()

    print(f"Recording {args.max_steps} steps...")
    total_reward = 0.0
    steps_done = 0
    crashed = False
    arrived = False
    for step in range(args.max_steps):
        if args.policy == "ppo_5ch":
            action, _ = models["ppo_model"].predict(obs, deterministic=True)
        elif args.policy == "carl":
            action = models["carl_model"].get_action(env.agent, env.engine)
        elif policy_obj is not None:
            action = policy_obj.act(base_env.vehicle.name)
        else:
            action = [0.0, 0.0]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps_done = step + 1

        try:
            env.render(
                mode="top_down",
                film_size=(2400, 2400),
                scaling=args.scaling,
                screen_size=(args.screen_size, args.screen_size),
                semantic_map=True,
                semantic_broken_line=True,
                draw_target_vehicle_trajectory=True,
                target_agent_heading_up=True,
                screen_record=True,
                window=False,
            )
        except Exception as exc:
            print(f"render warn (step {step}): {exc}")

        if terminated or truncated:
            arrived = bool(info.get("arrive_dest", False))
            crashed = bool(info.get("crash", False) or info.get("out_of_road", False))
            break

    print(f"Done: steps={steps_done}  arrived={arrived}  crashed={crashed}  reward={total_reward:.2f}")

    try:
        if hasattr(env, "top_down_renderer") and env.top_down_renderer is not None:
            env.top_down_renderer.generate_gif(str(out_gif), duration=args.gif_duration_ms)
            print(f"GIF: {out_gif} ({out_gif.stat().st_size / 1024:.0f} KB)")
        elif hasattr(base_env, "top_down_renderer") and base_env.top_down_renderer is not None:
            base_env.top_down_renderer.generate_gif(str(out_gif), duration=args.gif_duration_ms)
            print(f"GIF: {out_gif} ({out_gif.stat().st_size / 1024:.0f} KB)")
        else:
            print("WARN: no top_down_renderer attached — gif not saved", file=sys.stderr)
    except Exception as exc:
        print(f"ERROR generate_gif: {exc}", file=sys.stderr)
    finally:
        try: env.close()
        except: pass
        if raw_env is not env:
            try: raw_env.close()
            except: pass


if __name__ == "__main__":
    main()
