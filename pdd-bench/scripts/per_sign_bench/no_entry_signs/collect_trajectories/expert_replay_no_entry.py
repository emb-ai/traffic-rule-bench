#!/usr/bin/env python3
"""No-entry (3.1 + 3.2) multi-sign trajectory collector.

Drives episodes through no_entry_signs/run_benchmark.run_one_episode.
No aux agents, no pedestrians; SUMO density comes from the manifest.

Output layout (preferred: one process, --output-dir = <OUT>/<policy>):
  <OUT>/<policy>/3_1/all_runs.jsonl
  <OUT>/<policy>/3_1/by_sign/3_1/by_scene/<uid>/<policy>_<variant>/replay.{json,pkl}
  <OUT>/<policy>/3_2/all_runs.jsonl
  <OUT>/<policy>/3_2/by_sign/3_2/by_scene/<uid>/<policy>_<variant>/replay.{json,pkl}
  <OUT>/<policy>/catalog.jsonl

sign_code is read per row (3.1 or 3.2). Scenes resolve under
  --scenes-root/<slug>/   when --scenes-root is the parent .../scenes
  or under --scenes-root itself when it already points at scenes/3_1
  (sibling 3_2 is preferred when the row's sign differs).

Legacy single-sign mode still works if --output-dir ends with 3_1 or 3_2
(replay_root = parent, same as yield/stop).

Recording uses the shared RecordManager patch. Resume requires pkl+json.

Usage:
  python expert_replay_no_entry.py \\
      --manifest ../benchmark_output/combined/catalog_train80.jsonl \\
      --scenes-root ../scenes \\
      --policy comprehensive_rule_expert --ego-extra-samples 4 \\
      --count 2 --output-dir ./out/cre
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
SIGN_DIR = HERE.parent
PER_SIGN_DIR = SIGN_DIR.parent
PDD_BENCH = PER_SIGN_DIR.parent.parent
# Put no_entry_signs FIRST so `import run_benchmark` resolves correctly.
for p in (str(PDD_BENCH), str(PER_SIGN_DIR)):
    if p not in sys.path:
        sys.path.append(p)
if str(SIGN_DIR) in sys.path:
    sys.path.remove(str(SIGN_DIR))
sys.path.insert(0, str(SIGN_DIR))

from lib.manifest_config import (  # noqa: E402
    enrich_manifest_row,
    load_manifest_config,
)

import run_benchmark as rb  # noqa: E402

_sig = getattr(rb.run_one_episode, "__code__", None)
if _sig is not None and "record_episode" not in _sig.co_varnames:
    raise ImportError(
        f"Wrong run_benchmark imported from {getattr(rb, '__file__', '?')}; "
        "expected no_entry_signs/run_benchmark.py with record_episode"
    )

KNOWN_SLUGS = {"3_1", "3_2"}
KNOWN_CODES = {"3.1", "3.2"}
IDM_VARIANT_POLICIES = {"idm", "modified_idm", "comprehensive_rule_expert"}
POLICY_CHOICES = [
    "idm", "modified_idm", "comprehensive_rule_expert",
    "rule_compliant", "ppo_lidar",
    "carl", "carl_rule", "plant2", "plant2_rule",
]


def _sign_code(row: dict) -> str:
    code = str(row.get("sign_code") or row.get("pdd_code") or "").strip()
    if code in KNOWN_CODES:
        return code
    # Accept slug form if present
    slug = str(row.get("sign_slug") or "").strip()
    if slug in KNOWN_SLUGS:
        return slug.replace("_", ".")
    return code or "3.1"


def _sign_slug(code: str) -> str:
    return str(code).replace(".", "_")


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
            code = _sign_code(row)
            row["sign_code"] = code
            row.setdefault("pdd_code", code)
            # Sidecar path uses _sign_code / sign_code → slug 3_1 / 3_2.
            row["_sign_code"] = code
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
    variants = ["default"]
    variants.extend(f"s{i}" for i in range(1, extra_samples + 1))
    return variants


def _resolve_scenes_root_for_row(scenes_root: Path, row: dict) -> Path:
    """Resolve per-row scenes dir under parent .../scenes or sibling slug."""
    slug = _sign_slug(_sign_code(row))
    net = row.get("net_path") or ""
    if net and (scenes_root / net).exists():
        return scenes_root
    # Parent scenes/: try scenes/<slug>/
    cand = scenes_root / slug
    if net and (cand / net).exists():
        return cand
    if cand.is_dir() and not net:
        return cand
    # If scenes_root is already scenes/3_1 but row is 3.2, prefer sibling.
    if scenes_root.name in KNOWN_SLUGS:
        sibling = scenes_root.parent / slug
        if net and (sibling / net).exists():
            return sibling
        # Prefer parent/scenes layout when current slug dir misses the map.
        parent = scenes_root.parent
        if parent.name == "scenes" or (parent / slug).is_dir():
            alt = parent / slug
            if net and (alt / net).exists():
                return alt
    return scenes_root


def _resolve_output_layout(out_dir: Path) -> tuple[Path, bool]:
    """Return (policy_root, multi_sign).

    multi_sign=True  → --output-dir is <OUT>/<policy>; per-row slug dirs under it.
    multi_sign=False → --output-dir ends with 3_1/3_2; replay_root = parent.
    """
    out_dir = out_dir.resolve()
    if out_dir.name in KNOWN_SLUGS:
        return out_dir.parent, False
    return out_dir, True


def _policy_sign_dir(policy_root: Path, slug: str, multi_sign: bool,
                     out_dir: Path) -> Path:
    if multi_sign:
        return policy_root / slug
    # Single-sign: keep writing under the given out_dir when slug matches;
    # otherwise spill to sibling under policy_root.
    if out_dir.name == slug:
        return out_dir
    return policy_root / slug


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
    scene_uid = _scene_uid(row)
    ok = bool(episode.get("ok", False))
    code = _sign_code(row)
    slug = _sign_slug(code)
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

    if "frame_smooth_ratio" not in metrics:
        metrics["frame_smooth_ratio"] = float(
            metrics.get("smoothness_ratio") or 0.0
        )

    flat = {
        "valid": ok and not episode.get("error"),
        "policy": policy,
        "variant": variant or "default",
        "sign_code": code,
        "sign_slug": slug,
        "sign_type": row.get("sign_type") or "no_entry",
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
        "initial_speed_kmh": float(row.get("spawn_velocity_ms") or 0.0) * 3.6,
        "ego_idm_params": episode.get("ego_params") or "DEFAULT_EGO_PARAMS",
        **{k: v for k, v in metrics.items() if k != "violations_timeline"},
    }
    if "arrived_dest" not in flat and "reached_dest" in episode:
        flat["arrived_dest"] = bool(episode.get("reached_dest"))
    return flat


def _expected_sidecar_path(
    replay_root: Path, row: dict, policy: str, variant: str
) -> Path:
    slug = _sign_slug(_sign_code(row))
    scene_uid = _scene_uid(row)
    expert_subdir = f"{policy}_{variant}" if variant else policy
    return (
        replay_root / slug / "by_sign" / slug
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            code = _sign_code(row)
            f.write(json.dumps({
                "sign_code": code,
                "sign_slug": _sign_slug(code),
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
    if not scenes_root.is_dir():
        print(f"ERROR: scenes-root not a directory: {scenes_root}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_root, multi_sign = _resolve_output_layout(out_dir)
    # run_benchmark writes <replay_root>/<slug>/by_sign/<slug>/...
    # multi: replay_root = OUT/policy; single: OUT/policy (parent of slug dir)
    replay_root = policy_root

    rows = _load_manifest(manifest, args.count, args.start)
    if not rows:
        print("ERROR: no valid rows in manifest (after start/count)", file=sys.stderr)
        return 2

    # Probe scenes for first row (and warn if maps missing).
    probe = rows[0]
    probe_root = _resolve_scenes_root_for_row(scenes_root, probe)
    net = probe.get("net_path") or ""
    if net and not (probe_root / net).exists():
        print(
            f"ERROR: map not found at {probe_root / net}\n"
            f"  scenes-root={scenes_root}  sign={_sign_code(probe)}\n"
            f"  Pass --scenes-root pointing at no_entry_signs/scenes "
            f"(parent of 3_1 and 3_2).",
            file=sys.stderr,
        )
        return 2

    catalog_path = (policy_root if multi_sign else out_dir) / "catalog.jsonl"
    write_catalog(rows, catalog_path)
    print(f"Catalog: {catalog_path} ({len(rows)} uids)")

    variants = _ego_variants(args.policy, args.ego_variant, args.ego_extra_samples)
    models = rb._load_policy_models(
        args.policy, args.model_path, args.plant2_action_mode
    )

    # Per-slug all_runs handles
    all_runs_files: dict[str, Any] = {}
    done_keys: set[tuple] = set()
    mode_base = "a" if args.resume else "w"

    def _ao_for(slug: str):
        if slug in all_runs_files:
            return all_runs_files[slug]
        sign_dir = _policy_sign_dir(policy_root, slug, multi_sign, out_dir)
        sign_dir.mkdir(parents=True, exist_ok=True)
        path = sign_dir / "all_runs.jsonl"
        mode = mode_base
        if mode == "a" and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done_keys.add((r.get("scene_uid"), r.get("policy"), r.get("variant")))
        elif mode == "a" and not path.exists():
            mode = "w"
        fh = open(path, mode, encoding="utf-8")
        all_runs_files[slug] = (fh, path)
        return all_runs_files[slug]

    gifs_dir = None
    if args.save_gifs:
        gifs_dir = (policy_root if multi_sign else out_dir) / "gifs"
        gifs_dir.mkdir(parents=True, exist_ok=True)

    sign_counts = Counter(_sign_code(r) for r in rows)
    print(
        f"Policy={args.policy}  variants={variants}  scenes={len(rows)}  "
        f"signs={dict(sign_counts)}  aux=OFF  record=ON  "
        f"gifs={'yes' if args.save_gifs else 'no'}"
    )
    print(f"Output: {out_dir}  (multi_sign={multi_sign}, replay_root={replay_root})")

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
        for i, row in enumerate(rows, start=1):
            code = _sign_code(row)
            slug = _sign_slug(code)
            row_scenes = _resolve_scenes_root_for_row(scenes_root, row)
            ao, all_runs_path = _ao_for(slug)
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
                    f"sign={code} scene={row.get('scene_id')} uid={uid}",
                    flush=True,
                )
                t0 = time.time()
                episode = rb.run_one_episode(
                    row=row,
                    policy_type=args.policy,
                    models=models,
                    scenes_root=row_scenes,
                    max_steps=args.max_steps,
                    ego_variant=variant,
                    ego_sample_seed_base=args.ego_sample_seed_base,
                    replay_root=replay_root,
                    save_gif=gif_path,
                    hide_signs=args.hide_signs,
                    record_episode=True,
                )
                dt = time.time() - t0
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
        for fh, path in all_runs_files.values():
            fh.close()
            print(f"all_runs: {path}")

    print(
        f"\nDone. ok={n_ok} fail={n_fail} skip={n_skip}  "
        f"({n_ok + n_fail + n_skip}/{total_eps} episodes)"
    )
    if gifs_dir is not None:
        n_gif = len(list(gifs_dir.glob("*.gif")))
        print(f"GIFs: {n_gif} under {gifs_dir}")
    return 0 if n_fail == 0 or n_ok > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect no-entry (3.1+3.2) expert trajectories "
                    "(equal-map combined catalog; aux off)"
    )
    p.add_argument("--manifest", required=True,
                   help="combined/balanced catalog or real_manifest.jsonl")
    p.add_argument("--scenes-root", default=str(SIGN_DIR / "scenes"),
                   help="Parent of 3_1 and 3_2 scene dirs "
                        "(default: no_entry_signs/scenes)")
    p.add_argument("--policy", required=True, choices=POLICY_CHOICES)
    p.add_argument("--model-path", default=None,
                   help="Required for carl/plant2 and *_rule variants")
    p.add_argument("--plant2-action-mode", default="pid",
                   choices=["pid", "wps_pure_pursuit"])
    p.add_argument(
        "--output-dir", required=True,
        help="Prefer <OUT>/<policy> (multi-sign: writes <policy>/3_1 and "
             "<policy>/3_2). Legacy: <OUT>/<policy>/3_1 also works.",
    )
    p.add_argument("--count", type=int, default=None,
                   help="Only first N manifest rows (smoke / visual check)")
    p.add_argument("--start", type=int, default=0,
                   help="Skip first N rows of the manifest")
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
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.policy in {"carl", "carl_rule", "plant2", "plant2_rule"} and not args.model_path:
        print(f"ERROR: --model-path required for {args.policy}", file=sys.stderr)
        sys.exit(2)
    sys.exit(run_collection(args))


if __name__ == "__main__":
    main()
