"""CLI for one policy × one manifest. Episode loop lives in ``bench.episode``."""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

from traffic_bench.eval.core.runtime.checkpoints import (
    DEFAULT_MODEL_PATHS,
    NN_NEED_CHECKPOINT,
)
from traffic_bench.eval.core.scenarios.auxiliary_agent import (
    DEFAULT_CONVOY_GAP_M,
    DEFAULT_CONVOY_SIZE,
    DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END,
    DEFAULT_SPAWN_VELOCITY_MS,
)
from traffic_bench.eval.core.manifest.manifest_config import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_AUX_LANES_OCCUPIED_MAX,
    DEFAULT_STOP_WAIT_STEPS,
)
from traffic_bench.eval.core.profiles.ego_defaults import sample_ego_params
from traffic_bench.eval.bench.episode import (
    _episode_key_from_result,
    _episode_key_from_row,
    _load_enriched_manifest_rows,
    _load_existing_results,
    _load_policy_models,
    aggregate_results,
    collect_rows,
    resolve_model_path,
    run_one_episode,
)
from traffic_bench.eval.bench.place import place_signs_for_row
from traffic_bench.eval.signs.dual_path.place import resolve_row_for_policy

# Public aliases (oracle / older callers import these from run_benchmark).
_place_junction_priority_signs = place_signs_for_row
_resolve_dual_path_row_for_policy = resolve_row_for_policy
_resolve_one_way_row_for_policy = resolve_row_for_policy

BENCH_DIR = Path(__file__).resolve().parent
PDD_BENCH_DIR = BENCH_DIR.parent
SDC_ROOT = PDD_BENCH_DIR.parent
REPO_ROOT = SDC_ROOT


def main():
    parser = argparse.ArgumentParser(description="Run policies on real SUMO maps (main sign / 2.1 benchmark)")
    parser.add_argument("--policy", required=True,
                        choices=["idm", "modified_idm", "comprehensive_rule_expert",
                                 "rule_compliant", "ppo_lidar",
                                 "carl", "carl_rule",
                                 "plant2", "plant2_rule", "plant2_ft"])
    parser.add_argument("--model-path", type=str, default=None,
                        help="Checkpoint for carl/plant2 (defaults under traffic-bench/checkpoints/; "
                             "plant2_ft → checkpoints/plant2_finetuned)")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--preset", type=str, default="full", choices=["full", "full_last"])
    parser.add_argument("--benchmark-output", type=str, default="benchmark_output",
                        help="Base dir that contains <preset>/")
    parser.add_argument("--scenes-root", type=str, default=str(SDC_ROOT / "scenes"))
    parser.add_argument("--sign-type", type=str, default=None,
                        help="Single sign code, e.g. 2.1")
    parser.add_argument("--sign-types", type=str, default="",
                        help="Comma-separated sign codes")
    parser.add_argument("--max-scenes-per-sign", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--ego-variant", type=str, default="default",
                        help="Ego IDM variant label: default or s1/s2/s3/s4")
    parser.add_argument("--ego-sample-seed-base", type=int, default=42,
                        help="Base seed for sampled IDM ego variants")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="Recompute scenes with existing failed records (ok=false)")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Ignore existing results and rerun all scenes")
    parser.add_argument("--skip-error-episodes", action="store_true",
                        help="When used with --rerun-failed, keep previously errored episodes skipped")
    parser.add_argument("--debug-one-way-sign-selection", action="store_true",
                        help="Enable verbose lane-selection debug logs")
    parser.add_argument("--emit-replay-sidecar", action="store_true",
                        help="Also emit per-(scene_uid, variant) replay.json sidecar")
    parser.add_argument("--replay-root", type=str, default=None,
                        help="Output dir for sidecar files")
    parser.add_argument("--unique-scene-id", action="store_true",
                        help="Dedup manifest rows by scene_id")
    parser.add_argument("--scene-id", type=str, default=None,
                        help="Run only the scene with this scene_id")
    parser.add_argument("--scene-uid", type=str, default=None,
                        help="Run only the scene matching this exact UID <scene_id>:<sign_type>:<seed>")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to a custom *.jsonl manifest")
    parser.add_argument("--save-gifs", action="store_true",
                        help="Record top-down GIF per episode")
    parser.add_argument("--gif-dir", type=str, default=None,
                        help="Directory for GIFs")
    parser.add_argument(
        "--gif-window-m",
        type=float,
        default=80.0,
        help="Visible top-down GIF window in meters (same across signs; "
             "film_size auto-grows so MetaDrive does not clamp zoom).",
    )
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Write episodes/summary/gifs here directly "
                             "(skips <benchmark-output>/<preset>/policy_eval/<run-name>)")
    parser.add_argument("--plant2-action-mode", type=str, default="pid",
                        choices=["pid", "wps_pure_pursuit"],
                        help="How PlanT2 converts pred_plan -> action")
    parser.add_argument("--hide-signs", action="store_true",
                        help="Hide traffic sign visual models (signs still affect behavior)")
    parser.add_argument(
        "--draw-path-conflict",
        action="store_true",
        help="Overlay ego/aux route polylines + conflict point on top-down GIFs",
    )

    # Auxiliary agent options
    parser.add_argument("--auxiliary-agent", action="store_true", default=True,
                        help="Spawn an auxiliary agent on an incoming lane near intersection")
    parser.add_argument(
        "--aux-distance-from-intersection",
        type=float,
        default=DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
        help=f"Fallback aux spawn distance from intersection (meters); "
             f"manifest row aux_distance_from_intersection takes precedence "
             f"(default: {DEFAULT_AUX_DISTANCE_FROM_INTERSECTION})",
    )
    parser.add_argument("--aux-policy", type=str, default="idm", choices=["idm", "stationary"],
                        help="Auxiliary agent behavior: idm drives to outgoing lane, stationary stays put")
    parser.add_argument(
        "--aux-spawn-velocity-ms",
        type=float,
        default=DEFAULT_SPAWN_VELOCITY_MS,
        help=f"Aux IDM cruise/release speed in m/s (default: {DEFAULT_SPAWN_VELOCITY_MS})",
    )
    parser.add_argument(
        "--aux-release-when-ego-within-m",
        type=float,
        default=DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END,
        help=(
            "Release gated IDM aux when ego is within this distance of spawn lane end (m); "
            f"0 = immediate (default: {DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END}). "
            "Clamped up to spawn_distance_before_end so aux is not held while ego yields."
        ),
    )
    parser.add_argument(
        "--aux-convoy-size",
        type=int,
        default=DEFAULT_CONVOY_SIZE,
        help=f"Max convoy size at manifest generation; spawns rows for sizes 1..N "
             f"(default: {DEFAULT_CONVOY_SIZE}). Per-row size is stored as aux_convoy_size.",
    )
    parser.add_argument(
        "--aux-convoy-gap-m",
        type=float,
        default=DEFAULT_CONVOY_GAP_M,
        help=f"Longitudinal spacing between convoy vehicles in meters (default: {DEFAULT_CONVOY_GAP_M})",
    )
    parser.add_argument(
        "--aux-lanes-occupied",
        type=int,
        default=DEFAULT_AUX_LANES_OCCUPIED_MAX,
        help=f"Fallback max main-road lanes to occupy when manifest row omits aux_lanes_occupied "
             f"(default: {DEFAULT_AUX_LANES_OCCUPIED_MAX})",
    )
    parser.add_argument(
        "--stop-wait-steps",
        type=int,
        default=None,
        help=f"Override expert stop-line dwell in sim steps (default from manifest / "
             f"{DEFAULT_STOP_WAIT_STEPS} ≈ 1.5 s at 0.1 s/step)",
    )

    args = parser.parse_args()

    if args.scene_id and args.scene_uid:
        raise ValueError("--scene-id and --scene-uid are mutually exclusive")

    assert args.ego_variant in ("default", "s1", "s2", "s3", "s4"), \
        f"--ego-variant must be one of default/s1/s2/s3/s4, got {args.ego_variant!r}"

    if args.ego_variant != "default":
        _t = args.ego_sample_seed_base + 12345
        _p1 = sample_ego_params(_t)
        _p2 = sample_ego_params(_t)
        for _k in _p1:
            assert math.isclose(float(_p1[_k]), float(_p2[_k]), abs_tol=1e-9), (
                f"sample_ego_params nondeterministic on key {_k!r}: "
                f"{_p1[_k]} vs {_p2[_k]}")
        print(f"[determinism check OK] sample_ego_params({_t}) reproducible.")

    logging.getLogger().setLevel(getattr(logging, "CRITICAL"))

    benchmark_output_dir = (BENCH_DIR / args.benchmark_output / args.preset).resolve()
    if not args.manifest and not benchmark_output_dir.exists():
        raise ValueError(f"Benchmark output not found: {benchmark_output_dir}")

    scenes_root = Path(args.scenes_root).resolve()

    only_codes: set[str] = set()
    if args.sign_type:
        only_codes.add(args.sign_type)
    if args.sign_types.strip():
        only_codes.update([c.strip() for c in args.sign_types.split(",") if c.strip()])

    print(f"Policy: {args.policy}")
    print(f"Preset: {args.preset}")
    print(f"Backend: sumo (real maps only)")
    print(f"Input: {benchmark_output_dir}")
    if args.auxiliary_agent:
        print(f"Auxiliary agent: ENABLED ({args.aux_policy}, near intersection)")
        print(f"  - Distance from intersection: {args.aux_distance_from_intersection}m")
        if args.aux_policy == "idm":
            print(f"  - Release when ego within: {args.aux_release_when_ego_within_m}m of spawn lane end")
            print(f"  - Speed after release: {args.aux_spawn_velocity_ms} m/s")
            print(f"  - Convoy size: from manifest row aux_convoy_size (CLI default {args.aux_convoy_size})")
            print(f"  - Lanes occupied: from manifest row aux_lanes_occupied (CLI default {args.aux_lanes_occupied})")
            print(f"  - Convoy gap: {args.aux_convoy_gap_m}m")
            print(f"  - Route: incoming lane -> reachable outgoing lane")

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"--manifest not found: {manifest_path}")
        rows: list[dict] = []
        for row in _load_enriched_manifest_rows(manifest_path):
            if "valid" in row and not row["valid"]:
                continue
            row["_backend"] = "sumo"
            if not row.get("_sign_code"):
                row["_sign_code"] = (row.get("sign_code") or row.get("pdd_code")
                                      or row.get("sign_type") or "")
            rows.append(row)
    else:
        rows = collect_rows(
            benchmark_output_dir=benchmark_output_dir,
            only_codes=only_codes,
            max_scenes_per_sign=args.max_scenes_per_sign,
            unique_scene_id=args.unique_scene_id,
        )

    if args.scene_id:
        rows = [r for r in rows if str(r.get("scene_id")) == args.scene_id]
    if args.scene_uid:
        print(f"[DEBUG] Looking for scene_uid: {args.scene_uid}")
        print(f"[DEBUG] Available scene keys (first 5):")
        for i, r in enumerate(rows[:5]):
            key = ":".join(str(x) for x in _episode_key_from_row(r))
            print(f"  [{i}] {key}")
        rows = [r for r in rows
                if ":".join(str(x) for x in _episode_key_from_row(r)) == args.scene_uid]
        print(f"[DEBUG] Matched {len(rows)} rows")

    if not rows:
        raise RuntimeError(
            "No scenes selected. Check --preset/--sign-type/"
            "--scene-id/--scene-uid/--manifest")

    print(f"Selected scenes: {len(rows)}")
    model_path = resolve_model_path(args.policy, args.model_path)
    if args.policy in NN_NEED_CHECKPOINT and not model_path:
        default = DEFAULT_MODEL_PATHS.get(args.policy)
        raise ValueError(
            f"--model-path is required for --policy {args.policy}"
            + (f" (default missing: {default})" if default else "")
        )
    models = _load_policy_models(
        args.policy, model_path, plant2_action_mode=args.plant2_action_mode,
    )

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = benchmark_output_dir / "policy_eval" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / f"episodes_{args.policy}.jsonl"

    replay_root: Path | None = None
    if args.emit_replay_sidecar:
        replay_root = Path(args.replay_root) if args.replay_root else (out_dir / "replays")
        replay_root.mkdir(parents=True, exist_ok=True)
        print(f"Sidecars: {replay_root}")

    gifs_dir: Path | None = None
    if args.save_gifs:
        gifs_dir = Path(args.gif_dir) if args.gif_dir else (out_dir / "gifs")
        gifs_dir.mkdir(parents=True, exist_ok=True)
        print(f"GIFs: {gifs_dir}")

    existing_results = _load_existing_results(episodes_path)
    existing_by_key: dict[tuple[str, str, int], dict] = {}
    for r in existing_results:
        existing_by_key[_episode_key_from_result(r)] = r

    rows_to_run: list[dict] = []
    skipped = 0
    for row in rows:
        key = _episode_key_from_row(row)
        old = existing_by_key.get(key)
        if args.force_rerun:
            rows_to_run.append(row)
            continue
        if old is None:
            rows_to_run.append(row)
            continue
        if args.skip_error_episodes and not bool(old.get("ok", False)):
            skipped += 1
            continue
        if args.rerun_failed and not bool(old.get("ok", False)):
            rows_to_run.append(row)
            continue
        skipped += 1

    print(f"Resume: loaded {len(existing_results)} existing episodes, skip {skipped}, run {len(rows_to_run)}"
          + (" (--force-rerun: ignoring existing)" if args.force_rerun else ""))

    results_by_key: dict[tuple[str, str, int], dict] = dict(existing_by_key)
    write_mode = "a" if episodes_path.exists() else "w"
    with open(episodes_path, write_mode, encoding="utf-8") as f:
        for idx, row in enumerate(rows_to_run, start=1):
            scene_id = row.get("scene_id")
            sign_code = row.get("_sign_code")
            print(f"[{idx}/{len(rows_to_run)}] sign={sign_code} scene={scene_id}")
            
            if args.debug_one_way_sign_selection:
                row["debug_one_way_sign_selection"] = True
            if args.stop_wait_steps is not None:
                row["stop_wait_steps"] = int(args.stop_wait_steps)
            gif_path = None
            if gifs_dir is not None:
                seed_val = int(row.get("seed") or row.get("deterministic_seed") or 0)
                var_idx = int(row.get("var_idx", 0) or 0)
                uid = f"{scene_id or 'scene'}_v{var_idx}_s{seed_val}"
                gif_path = gifs_dir / f"{uid}_{args.policy}_{args.ego_variant}.gif"
            episode_t0 = time.time()

            r = run_one_episode(
                row=row,
                policy_type=args.policy,
                models=models,
                scenes_root=scenes_root,
                max_steps=args.max_steps,
                ego_variant=args.ego_variant,
                ego_sample_seed_base=args.ego_sample_seed_base,
                replay_root=replay_root,
                save_gif=gif_path,
                gif_window_m=args.gif_window_m,
                hide_signs=args.hide_signs,
                draw_path_conflict=bool(args.draw_path_conflict),
                auxiliary_agent=args.auxiliary_agent,
                aux_distance_from_intersection=args.aux_distance_from_intersection,
                aux_policy=args.aux_policy,
                aux_spawn_velocity_ms=args.aux_spawn_velocity_ms,
                aux_release_when_ego_within_m=args.aux_release_when_ego_within_m,
                aux_convoy_size=args.aux_convoy_size,
                aux_convoy_gap_m=args.aux_convoy_gap_m,
                aux_lanes_occupied=args.aux_lanes_occupied,
            )
            episode_dt = time.time() - episode_t0
            print(f"{args.policy}  elapsed_s={episode_dt:.3f}")

            key = _episode_key_from_row(row)
            results_by_key[key] = r
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()

    results: list[dict] = list(results_by_key.values())
    summary = aggregate_results(results)
    summary_path = out_dir / f"summary_{args.policy}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    ok_runs = sum(1 for r in results if r.get("ok"))
    print("\n=== Done ===")
    print(f"Episodes OK: {ok_runs}/{len(results)}")
    print(f"Episodes: {episodes_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
