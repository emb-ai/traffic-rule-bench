#!/usr/bin/env python3
"""Crosswalk-sign (5.19) trajectory collector (pedestrian-aware).

Mirrors the role of per_sign_bench/expert_replay.py, but drives episodes through
crosswalk_sign/run_benchmark.run_one_episode. Auxiliary agents stay OFF by
default; pedestrians / PedestrianYieldRule come from the manifest.

Writes, matching colleague expert_replay layout (output-dir = <OUT>/<policy>/5_19):
  <output-dir>/all_runs.jsonl
  <output-dir>/by_sign/5_19/by_scene/<uid>/<policy>_<variant>/replay.json
  <output-dir>/by_sign/5_19/by_scene/<uid>/<policy>_<variant>/replay.pkl
  <output-dir>/gifs/*.gif              optional (--save-gifs)

Sidecar path is produced by crosswalk run_benchmark as
  <replay_root>/<sign_slug>/by_sign/<sign_slug>/by_scene/...
so we set replay_root = output-dir.parent (= <OUT>/<policy>) to avoid a nested
extra 5_19/ and land on the same tree as per_sign_bench/expert_replay.py.

Recording uses the shared RecordManager tolerance patch so mid-episode
pedestrian spawns do not crash the recorder. Oracle selection uses
all_runs.jsonl; pkl is for PlanT2 training / expert_replay_inenv.

Usage:
  python expert_replay_crosswalk.py \\
      --manifest ../benchmark_output/5_19/<ts>/real_manifest.jsonl \\
      --policy comprehensive_rule_expert --ego-extra-samples 4 \\
      --count 3 --save-gifs --output-dir ./out/cre
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
CROSSWALK_DIR = HERE.parent
PER_SIGN_DIR = CROSSWALK_DIR.parent
PDD_BENCH = PER_SIGN_DIR.parent.parent
# Put crosswalk_sign FIRST so `import run_benchmark` resolves to
# crosswalk_sign/run_benchmark.py, NOT per_sign_bench/run_benchmark.py.
for p in (str(PDD_BENCH), str(PER_SIGN_DIR)):
    if p not in sys.path:
        sys.path.append(p)
if str(CROSSWALK_DIR) in sys.path:
    sys.path.remove(str(CROSSWALK_DIR))
sys.path.insert(0, str(CROSSWALK_DIR))

from lib.manifest_config import (  # noqa: E402
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_AUX_LANES_OCCUPIED_MAX,
    enrich_manifest_row,
    load_manifest_config,
)
from lib.auxiliary_agent import (  # noqa: E402
    DEFAULT_CONVOY_GAP_M,
    DEFAULT_CONVOY_SIZE,
    DEFAULT_SPAWN_VELOCITY_MS,
)

import run_benchmark as crb  # noqa: E402

# Sanity: must be the crosswalk runner (has hide_signs + auxiliary_agent + record).
_sig = getattr(crb.run_one_episode, "__code__", None)
if _sig is not None and "auxiliary_agent" not in _sig.co_varnames:
    raise ImportError(
        f"Wrong run_benchmark imported from {getattr(crb, '__file__', '?')}; "
        "expected crosswalk_sign/run_benchmark.py with auxiliary_agent"
    )
if _sig is not None and "record_episode" not in _sig.co_varnames:
    raise ImportError(
        f"Wrong run_benchmark imported from {getattr(crb, '__file__', '?')}; "
        "expected crosswalk_sign/run_benchmark.py with record_episode"
    )

SIGN_CODE = "5.19"
SIGN_SLUG = "5_19"
IDM_VARIANT_POLICIES = {"idm", "modified_idm", "comprehensive_rule_expert"}
POLICY_CHOICES = [
    "idm", "modified_idm", "comprehensive_rule_expert",
    "rule_compliant", "ppo_lidar",
    "carl", "carl_rule", "plant2", "plant2_rule",
]


def _scene_uid(row: dict) -> str:
    seed = int(row.get("seed") or row.get("deterministic_seed") or 0)
    sid = row.get("scene_id") or f"scene_{seed}"
    return (f"{sid}_lane{int(row.get('spawn_lane_num', 0) or 0)}"
            f"_seed{seed}_v{int(row.get('var_idx', 0) or 0)}")


def _load_manifest(path: Path, count: Optional[int], start: int) -> list[dict]:
    cfg = load_manifest_config(path)
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("valid") is False:
                continue
            row = enrich_manifest_row(row, cfg)
            # Force 5.19 so crosswalk run_benchmark sidecar path uses slug 5_19
            # (not "crosswalk" from sign_type).
            row["sign_code"] = SIGN_CODE
            row["_sign_code"] = SIGN_CODE
            rows.append(row)
    if start:
        rows = rows[start:]
    if count is not None:
        rows = rows[:count]
    return rows


def _ego_variants(policy: str, ego_variant: str, extra_samples: int) -> list[str]:
    if policy not in IDM_VARIANT_POLICIES:
        return [ego_variant or "default"]
    if extra_samples <= 0:
        return [ego_variant or "default"]
    # default + s1..sN  (same convention as collect_trajectories.sh)
    variants = ["default"]
    variants.extend(f"s{i}" for i in range(1, extra_samples + 1))
    return variants


def _flat_all_runs_row(
    *,
    row: dict,
    policy: str,
    variant: str,
    episode: dict,
    sidecar_path: Optional[Path],
    gif_path: Optional[Path],
    pkl_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Flatten crosswalk episode result into select_experts all_runs schema."""
    scene_uid = _scene_uid(row)
    ok = bool(episode.get("ok", False))
    # Prefer nested sidecar metrics if present on disk; else episode return.
    metrics: dict[str, Any] = {}
    if sidecar_path is not None and sidecar_path.is_file():
        try:
            side = json.loads(sidecar_path.read_text(encoding="utf-8"))
            metrics = dict(side.get("metrics") or {})
        except (json.JSONDecodeError, OSError):
            metrics = {}

    if not metrics:
        metrics = {
            "arrived_dest": bool(episode.get("reached_dest")),
            "crashed": bool(episode.get("crashed")),
            "out_of_road": bool(episode.get("out_of_road")),
            "final_step": int(episode.get("steps") or 0),
            "route_completion": float(episode.get("route_completion_pct") or 0.0) / 100.0,
            "frame_smooth_ratio": float(
                episode.get("smoothness_frame_ratio")
                or episode.get("smoothness")
                or 0.0
            ),
            "smoothness_ratio": float(episode.get("smoothness") or 0.0),
            "violations_event_count": int(episode.get("violations_event_count") or 0),
            "violations_by_class_event": dict(
                episode.get("violations_by_class_event") or {}
            ),
            "total_violations": int(episode.get("violations") or 0),
            "driving_efficiency": float(episode.get("driving_efficiency") or 0.0),
            "driving_score": float(episode.get("driving_score") or 0.0),
            "success": bool(episode.get("success")),
        }

    # Ensure comfort field name used by select_experts.comfort()
    if "frame_smooth_ratio" not in metrics:
        metrics["frame_smooth_ratio"] = float(
            metrics.get("smoothness_ratio") or 0.0
        )

    flat = {
        "valid": ok and not episode.get("error"),
        "policy": policy,
        "variant": variant or "default",
        "sign_code": SIGN_CODE,
        "sign_slug": SIGN_SLUG,
        "sign_type": row.get("sign_type") or "crosswalk",
        "scene_id": row.get("scene_id"),
        "scene_uid": scene_uid,
        "seed": int(row.get("seed") or row.get("deterministic_seed") or 0),
        "spawn_lane_num": int(row.get("spawn_lane_num", 0) or 0),
        "var_idx": int(row.get("var_idx", 0) or 0),
        "net_path": row.get("net_path"),
        "backend": "sumo",
        "pkl_path": (
            episode.get("pkl_path")
            or (str(pkl_path) if pkl_path else None)
        ),
        "sidecar_path": str(sidecar_path) if sidecar_path else None,
        "gif_path": str(gif_path) if gif_path else None,
        "failure_reason": episode.get("error"),
        "initial_speed_mps": float(row.get("spawn_velocity_ms") or 0.0),
        # Colleague all_runs also stores km/h (same source as m/s).
        "initial_speed_kmh": float(row.get("spawn_velocity_ms") or 0.0) * 3.6,
        "ego_idm_params": episode.get("ego_params") or "DEFAULT_EGO_PARAMS",
        **{k: v for k, v in metrics.items() if k != "violations_timeline"},
    }
    # Normalize destination flag name
    if "arrived_dest" not in flat and "reached_dest" in episode:
        flat["arrived_dest"] = bool(episode.get("reached_dest"))
    return flat


def _expected_sidecar_path(
    replay_root: Path, row: dict, policy: str, variant: str
) -> Path:
    """Colleague layout under <policy>/: 5_19/by_sign/5_19/by_scene/.../replay.json."""
    scene_uid = _scene_uid(row)
    expert_subdir = f"{policy}_{variant}" if variant else policy
    # replay_root is <OUT>/<policy>; crosswalk run_benchmark prefixes sign_slug.
    return (
        replay_root / SIGN_SLUG / "by_sign" / SIGN_SLUG
        / "by_scene" / scene_uid / expert_subdir / "replay.json"
    )


def _expected_pkl_path(
    replay_root: Path, row: dict, policy: str, variant: str
) -> Path:
    return _expected_sidecar_path(replay_root, row, policy, variant).with_name(
        "replay.pkl"
    )


def _episode_already_recorded(
    pkl_path: Path, sidecar_path: Path, gif_path: Optional[Path], require_gif: bool
) -> bool:
    if not (pkl_path.exists() and pkl_path.stat().st_size > 0):
        return False
    if not (sidecar_path.exists() and sidecar_path.stat().st_size > 0):
        return False
    if require_gif and gif_path is not None:
        if not (gif_path.exists() and gif_path.stat().st_size > 0):
            return False
    return True


def write_catalog(manifest_rows: list[dict], out_path: Path) -> None:
    """Catalog for select_experts_coverage join (scene_uid → net_path)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps({
                "sign_code": SIGN_CODE,
                "scene_id": row.get("scene_id"),
                "scene_uid": _scene_uid(row),
                "net_path": row.get("net_path"),
                "spawn_lane_num": int(row.get("spawn_lane_num", 0) or 0),
                "var_idx": int(row.get("var_idx", 0) or 0),
                "seed": int(row.get("seed") or row.get("deterministic_seed") or 0),
                "valid": True,
            }, default=str) + "\n")


def run_collection(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest).resolve()
    if not manifest.is_file():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 2

    scenes_root = Path(args.scenes_root).resolve()
    # Auto-fix: manifests use "sign_XXX/map.net.xml" under scenes/5_19/.
    if not scenes_root.is_dir():
        print(f"ERROR: scenes-root not a directory: {scenes_root}", file=sys.stderr)
        return 2
    probe_rows = _load_manifest(manifest, count=1, start=0)
    if probe_rows:
        net = probe_rows[0].get("net_path") or ""
        cand = scenes_root / net
        if not cand.exists():
            alt = scenes_root / "5_19"
            if (alt / net).exists():
                print(f"[auto] scenes-root {scenes_root} → {alt} "
                      f"(found {net} there)")
                scenes_root = alt
            else:
                print(
                    f"ERROR: map not found at {cand}\n"
                    f"  Also tried: {alt / net}\n"
                    f"  Pass --scenes-root pointing at the folder that contains "
                    f"sign_*/map.net.xml (usually crosswalk_sign/scenes/5_19).",
                    file=sys.stderr,
                )
                return 2
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Colleague: output-dir = <OUT>/<policy>/<sign>, sidecars at
    #   <output-dir>/by_sign/<sign>/by_scene/...
    # Crosswalk run_benchmark writes <replay_root>/<sign>/by_sign/<sign>/...
    # so replay_root must be the policy dir (parent of output-dir).
    replay_root = out_dir.parent
    gifs_dir = (out_dir / "gifs") if args.save_gifs else None
    if gifs_dir is not None:
        gifs_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest(manifest, args.count, args.start)
    if not rows:
        print("ERROR: no valid rows in manifest (after start/count)", file=sys.stderr)
        return 2

    catalog_path = out_dir / "catalog.jsonl"
    # Parallel workers (--worker-id): only worker 0 writes the full catalog.
    worker_id: Optional[int] = getattr(args, "worker_id", None)
    if worker_id is None or worker_id == 0:
        catalog_rows = (
            _load_manifest(manifest, None, 0) if worker_id is not None else rows
        )
        write_catalog(catalog_rows, catalog_path)
        print(f"Catalog: {catalog_path} ({len(catalog_rows)} uids)")
    else:
        print(f"Catalog: skip (worker_id={worker_id})")

    variants = _ego_variants(args.policy, args.ego_variant, args.ego_extra_samples)
    models = crb._load_policy_models(
        args.policy, args.model_path, args.plant2_action_mode
    )

    if worker_id is None:
        all_runs_path = out_dir / "all_runs.jsonl"
    else:
        all_runs_path = out_dir / f"all_runs.w{worker_id:02d}.jsonl"

    done_keys: set[tuple] = set()
    if args.resume:
        for path in [out_dir / "all_runs.jsonl", *sorted(out_dir.glob("all_runs.w*.jsonl"))]:
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done_keys.add(
                    (r.get("scene_uid"), r.get("policy"), r.get("variant"))
                )

    mode = "a" if args.resume and all_runs_path.exists() else "w"

    any_aux = any(bool(r.get("auxiliary_agent")) for r in rows)
    print(
        f"Policy={args.policy}  variants={variants}  scenes={len(rows)}  "
        f"aux={'ON' if any_aux else 'OFF'}  record=ON  "
        f"gifs={'yes' if args.save_gifs else 'no'}"
    )
    wtag = f" worker={worker_id}" if worker_id is not None else ""
    print(
        f"Output: {out_dir}  (start={args.start}, count={args.count}{wtag})"
    )

    n_ok = n_fail = n_skip = 0
    total_eps = len(rows) * len(variants)
    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=total_eps,
            desc=args.policy,
            unit="ep",
            dynamic_ncols=True,
            mininterval=2.0,
            file=sys.stderr,
        )

    def _pbar_update(*, status: str, variant: str, scene: str) -> None:
        if pbar is None:
            return
        pbar.set_postfix_str(f"{status} {variant} {scene}", refresh=False)
        pbar.update(1)

    try:
        with open(all_runs_path, mode, encoding="utf-8") as ao:
            for i, row in enumerate(rows, start=1):
                for variant in variants:
                    uid = _scene_uid(row)
                    key = (uid, args.policy, variant)
                    sidecar = _expected_sidecar_path(
                        replay_root, row, args.policy, variant
                    )
                    pkl_path = _expected_pkl_path(
                        replay_root, row, args.policy, variant
                    )
                    gif_path = None
                    if gifs_dir is not None:
                        gif_path = gifs_dir / f"{uid}_{args.policy}_{variant}.gif"

                    if args.resume and (
                        key in done_keys
                        or _episode_already_recorded(
                            pkl_path, sidecar, gif_path, args.save_gifs
                        )
                    ):
                        n_skip += 1
                        _pbar_update(
                            status="skip",
                            variant=variant,
                            scene=str(row.get("scene_id") or ""),
                        )
                        continue

                    print(
                        f"[{i}/{len(rows)}] {args.policy}/{variant}  "
                        f"scene={row.get('scene_id')} uid={uid}",
                        flush=True,
                    )
                    t0 = time.time()
                    # Aux OFF by default; honor rare rows that set auxiliary_agent=True.
                    # Pedestrian flags come from enrich_manifest_row / the row itself —
                    # do not force aux kwargs that would override ped setup.
                    use_aux = bool(row.get("auxiliary_agent", False))
                    episode_kwargs = dict(
                        row=row,
                        policy_type=args.policy,
                        models=models,
                        scenes_root=scenes_root,
                        max_steps=args.max_steps,
                        ego_variant=variant,
                        ego_sample_seed_base=args.ego_sample_seed_base,
                        replay_root=replay_root,
                        save_gif=gif_path,
                        hide_signs=args.hide_signs,
                        auxiliary_agent=use_aux,
                        record_episode=True,
                    )
                    if use_aux:
                        episode_kwargs.update(
                            aux_distance_from_intersection=float(
                                row.get("aux_distance_from_intersection")
                                or args.aux_distance_from_intersection
                            ),
                            aux_policy=args.aux_policy,
                            aux_spawn_velocity_ms=float(
                                row.get("aux_spawn_velocity_ms")
                                or args.aux_spawn_velocity_ms
                            ),
                            aux_release_when_ego_within_m=(
                                args.aux_release_when_ego_within_m
                            ),
                            aux_convoy_size=int(
                                row.get("aux_convoy_size") or args.aux_convoy_size
                            ),
                            aux_convoy_gap_m=float(
                                row.get("aux_convoy_gap_m") or args.aux_convoy_gap_m
                            ),
                            aux_lanes_occupied=int(
                                row.get("aux_lanes_occupied") or args.aux_lanes_occupied
                            ),
                        )
                    episode = crb.run_one_episode(**episode_kwargs)
                    dt = time.time() - t0
                    # Refresh paths after write
                    if not sidecar.is_file():
                        sidecar = None  # type: ignore[assignment]
                    ep_pkl = episode.get("pkl_path")
                    if ep_pkl and Path(ep_pkl).is_file():
                        pkl_resolved: Optional[Path] = Path(ep_pkl)
                    elif pkl_path.is_file():
                        pkl_resolved = pkl_path
                    else:
                        pkl_resolved = None
                    flat = _flat_all_runs_row(
                        row=row,
                        policy=args.policy,
                        variant=variant,
                        episode=episode,
                        sidecar_path=sidecar if isinstance(sidecar, Path) else None,
                        gif_path=gif_path if gif_path and gif_path.is_file() else None,
                        pkl_path=pkl_resolved,
                    )
                    ao.write(json.dumps(flat, default=str) + "\n")
                    ao.flush()

                    if flat.get("valid"):
                        n_ok += 1
                        status = "ok"
                    else:
                        n_fail += 1
                        status = f"FAIL({episode.get('error') or 'invalid'})"
                    pkl_note = ""
                    if flat.get("pkl_path"):
                        pkl_note = "  pkl=yes"
                    elif episode.get("dump_error"):
                        pkl_note = f"  pkl=NO({episode.get('dump_error')})"
                    print(f"  → {status}  steps={flat.get('final_step')}  {dt:.1f}s"
                          + pkl_note
                          + (f"  gif={gif_path.name}" if flat.get("gif_path") else ""),
                          flush=True)
                    _pbar_update(
                        status=status,
                        variant=variant,
                        scene=str(row.get("scene_id") or ""),
                    )
    finally:
        if pbar is not None:
            pbar.close()

    print(
        f"\nDone. ok={n_ok} fail={n_fail} skip={n_skip}  "
        f"({n_ok + n_fail + n_skip}/{total_eps} episodes)  "
        f"all_runs={all_runs_path}"
    )
    if gifs_dir is not None:
        n_gif = len(list(gifs_dir.glob("*.gif")))
        print(f"GIFs: {n_gif} under {gifs_dir}")
    return 0 if n_fail == 0 or n_ok > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect crosswalk-sign (5.19) expert trajectories "
                    "(pedestrians; aux off)"
    )
    p.add_argument("--manifest", required=True,
                   help="crosswalk real_manifest.jsonl")
    p.add_argument("--scenes-root", default=str(CROSSWALK_DIR / "scenes" / "5_19"),
                   help="crosswalk scenes root (default: crosswalk_sign/scenes/5_19)")
    p.add_argument("--policy", required=True, choices=POLICY_CHOICES)
    p.add_argument("--model-path", default=None,
                   help="Required for carl/plant2 and *_rule variants")
    p.add_argument("--plant2-action-mode", default="pid",
                   choices=["pid", "wps_pure_pursuit"])
    p.add_argument("--output-dir", required=True,
                   help="Per-policy output root (writes all_runs.jsonl here)")
    p.add_argument("--count", type=int, default=None,
                   help="Only first N manifest rows (smoke / visual check)")
    p.add_argument("--start", type=int, default=0,
                   help="Skip first N rows of the manifest")
    p.add_argument(
        "--worker-id",
        type=int,
        default=None,
        help="Parallel shard id: write all_runs.wXX.jsonl (safe concurrent "
             "collection). Use with --start/--count.",
    )
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--ego-variant", default="default",
                   help="Single ego variant when --ego-extra-samples=0")
    p.add_argument("--ego-extra-samples", type=int, default=0,
                   help="For IDM-family: also run s1..sN (plus default)")
    p.add_argument("--ego-sample-seed-base", type=int, default=42)
    p.add_argument("--save-gifs", action="store_true",
                   help="Save top-down GIF per episode under <output-dir>/gifs")
    p.add_argument("--hide-signs", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Skip episodes that already have replay.pkl + replay.json "
                        "(and gif if --save-gifs); also skip keys in all_runs")
    # Aux CLI kept for parity; unused unless a row sets auxiliary_agent=True
    p.add_argument("--aux-policy", default="idm", choices=["idm", "stationary"])
    p.add_argument("--aux-distance-from-intersection", type=float,
                   default=DEFAULT_AUX_DISTANCE_FROM_INTERSECTION)
    p.add_argument("--aux-spawn-velocity-ms", type=float,
                   default=DEFAULT_SPAWN_VELOCITY_MS)
    p.add_argument("--aux-release-when-ego-within-m", type=float, default=5.0,
                   help="Release gated aux when ego is within this many meters of "
                        "its spawn-lane end (must match run_benchmark CLI default=5)")
    p.add_argument("--aux-convoy-size", type=int, default=DEFAULT_CONVOY_SIZE)
    p.add_argument("--aux-convoy-gap-m", type=float, default=DEFAULT_CONVOY_GAP_M)
    p.add_argument("--aux-lanes-occupied", type=int,
                   default=DEFAULT_AUX_LANES_OCCUPIED_MAX)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.policy in {"carl", "carl_rule", "plant2", "plant2_rule"} and not args.model_path:
        print(f"ERROR: --model-path required for {args.policy}", file=sys.stderr)
        sys.exit(2)
    sys.exit(run_collection(args))


if __name__ == "__main__":
    main()
