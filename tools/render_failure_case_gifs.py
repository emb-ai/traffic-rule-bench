#!/usr/bin/env python3
"""Render top-down GIFs for failure_cases replay sidecars.

Re-runs each episode with the same policy and manifest row, saving
``replay.gif`` next to ``replay.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "third_party" / "metadrive") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "metadrive"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SIGN_GROUP_TO_SCENES = {
    "main_road": REPO_ROOT / "data/scenes/main_road",
    "secondary_road": REPO_ROOT / "data/scenes/secondary_road",
    "yield": REPO_ROOT / "data/scenes/yield",
    "stop": REPO_ROOT / "data/scenes/stop",
}

MODEL_CACHE: dict[str, dict] = {}


def _load_models(policy: str) -> dict:
    if policy not in MODEL_CACHE:
        from traffic_bench.eval.run.policy import _load_policy_models

        MODEL_CACHE[policy] = _load_policy_models(
            policy, None, plant2_action_mode="discrete"
        )
    return MODEL_CACHE[policy]


def _infer_sign_group(case_dir: Path) -> str | None:
    parts = case_dir.parts
    for key in SIGN_GROUP_TO_SCENES:
        if key in parts:
            return key
    return None


def _find_replays(root: Path, *, category: str | None, sign: str | None) -> list[Path]:
    replays: list[Path] = []
    search_roots: list[Path]
    if category and sign:
        search_roots = [root / category / sign]
    elif category:
        search_roots = [root / category]
    elif sign:
        search_roots = [
            root / "rule_expert_sign_compliance_0" / sign,
            root / "baseline_sign_compliance_1" / sign,
        ]
    else:
        search_roots = [root]

    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for replay in sorted(search_root.rglob("replay.json")):
            replays.append(replay)
    return replays


def render_one(replay_path: Path, *, force: bool, gif_window_m: float) -> str:
    gif_path = replay_path.parent / "replay.gif"
    if gif_path.exists() and not force:
        return "skip"

    data = json.loads(replay_path.read_text(encoding="utf-8"))
    row = data.get("source_row")
    policy = data.get("policy")
    variant = data.get("variant", "default")
    if not row or not policy:
        return "bad_sidecar"

    sign_group = _infer_sign_group(replay_path)
    if sign_group is None:
        return "no_sign_group"
    scenes_root = SIGN_GROUP_TO_SCENES[sign_group]
    if not scenes_root.is_dir():
        return "missing_scenes_root"

    from traffic_bench.eval.run.episode import run_one_episode

    models = _load_models(policy)
    result = run_one_episode(
        row=row,
        policy_type=policy,
        models=models,
        scenes_root=scenes_root,
        max_steps=int(row.get("horizon", 600)),
        ego_variant=variant,
        ego_sample_seed_base=0,
        replay_root=None,
        save_gif=gif_path,
        gif_window_m=gif_window_m,
        hide_signs=False,
        draw_path_conflict=False,
        auxiliary_agent=bool(row.get("auxiliary_agent")),
        aux_distance_from_intersection=float(
            row.get("aux_distance_from_intersection", 20.0)
        ),
        aux_convoy_size=int(row.get("aux_convoy_size", 1) or 1),
        aux_lanes_occupied=int(row.get("aux_lanes_occupied", 1) or 1),
    )
    if not gif_path.exists():
        return f"no_gif:{result.get('error')}"
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "data/runs/failure_cases",
    )
    parser.add_argument("--category", choices=[
        "rule_expert_sign_compliance_0",
        "baseline_sign_compliance_1",
    ])
    parser.add_argument("--sign", choices=list(SIGN_GROUP_TO_SCENES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gif-window-m", type=float, default=60.0)
    args = parser.parse_args()

    replays = _find_replays(
        args.root.resolve(),
        category=args.category,
        sign=args.sign,
    )
    if args.limit is not None:
        replays = replays[: args.limit]

    stats: dict[str, int] = {}
    for i, replay_path in enumerate(replays, start=1):
        rel = replay_path.relative_to(args.root.resolve())
        try:
            status = render_one(
                replay_path,
                force=args.force,
                gif_window_m=args.gif_window_m,
            )
        except Exception:
            status = "error"
            traceback.print_exc()
        stats[status] = stats.get(status, 0) + 1
        print(f"[{i}/{len(replays)}] {status:8s} {rel}")

    print("\nSummary:", stats)


if __name__ == "__main__":
    main()
