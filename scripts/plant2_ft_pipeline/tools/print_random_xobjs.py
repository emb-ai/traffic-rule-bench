#!/usr/bin/env python3
"""Open raw boxes/NNNN.json.gz and print all objects.

No PlanTDataset / generate_batch — only what's on disk in the dump.

Examples::

  source traffic-rule-bench/scripts/plant2_ft_pipeline/_env.sh

  # random route + random frame
  $PY scripts/plant2_ft_pipeline/print_random_xobjs.py

  # random route, all frames (compact)
  $PY scripts/plant2_ft_pipeline/print_random_xobjs.py --all-frames --stride 10

  # fixed route + frame
  $PY scripts/plant2_ft_pipeline/print_random_xobjs.py \\
    --route $SHEPELEV/plant2_l1_fv_experts_split_signs_2.5/train/data/sign_100062_j0_lane0_seed1974118946_v0_default \\
    --frame 105

  # raw JSON
  $PY scripts/plant2_ft_pipeline/print_random_xobjs.py --json --seed 42
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

from lib.env import shepelev

DEFAULT_SPLIT = shepelev() / "plant2_l1_fv_experts_split_signs_2.5" / "train"


def _data_root(split: Path) -> Path:
    return split / "data" if (split / "data").is_dir() else split


def _list_routes(data_root: Path) -> list[Path]:
    return sorted(
        p for p in data_root.iterdir()
        if p.is_dir() and (p / "boxes").is_dir()
    )


def _list_frames(route: Path) -> list[Path]:
    return sorted(route.glob("boxes/*.json.gz"))


def _frame_num(path: Path) -> int:
    return int(path.name.split(".")[0])


def load_boxes(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path}: expected list, got {type(data).__name__}")
    return data


def _fmt_pos(pos: list[float]) -> str:
    return f"({pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:5.2f})"


def _fmt_extent(ext: list[float]) -> str:
    return f"[{ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f}]"


def _dist_xy(obj: dict) -> float:
    x, y = obj["position"][:2]
    return (x * x + y * y) ** 0.5


def _print_object(idx: int, obj: dict, *, verbose: bool) -> None:
    cls = obj.get("class", "?")
    pos = obj.get("position", [0.0, 0.0, 0.0])
    ext = obj.get("extent", [0.0, 0.0, 0.0])
    known = {"class", "position", "yaw", "speed", "extent", "id", "type_id"}
    extra_keys = sorted(k for k in obj if k not in known)
    print(
        f"[{idx:2d}] class={cls!r:14s} id={obj.get('id', '?'):3} "
        f"pos={_fmt_pos(pos)} dist={_dist_xy(obj):5.1f}m "
        f"yaw={obj.get('yaw', 0.0):7.3f} rad speed={obj.get('speed', 0.0):6.2f} "
        f"extent={_fmt_extent(ext)}"
    )
    if obj.get("type_id") is not None:
        print(f"      type_id={obj['type_id']!r}")
    for key in extra_keys:
        print(f"      {key}={obj[key]!r}")
    if verbose:
        print(f"      raw={json.dumps(obj, ensure_ascii=False)}")


def _print_boxes_file(
    path: Path,
    *,
    as_json: bool,
    verbose: bool,
    class_filter: str | None,
) -> None:
    boxes = load_boxes(path)
    if as_json:
        if class_filter is not None:
            boxes = [o for o in boxes if o.get("class") == class_filter]
        print(json.dumps(boxes, indent=2, ensure_ascii=False))
        return

    print(f"=== {path} ({len(boxes)} objects) ===")
    shown = 0
    for idx, obj in enumerate(boxes):
        if class_filter is not None and obj.get("class") != class_filter:
            continue
        _print_object(idx, obj, verbose=verbose)
        shown += 1
    if class_filter is not None:
        print(f"--- matched {shown} / {len(boxes)} (class={class_filter!r}) ---")


def _print_frame_compact(path: Path, boxes: list[dict]) -> None:
    frame = _frame_num(path)
    if not boxes:
        print(f"frame {frame:04d}  {path.name}  (empty)")
        return
    parts = []
    for obj in boxes:
        cls = obj.get("class", "?")
        pos = obj.get("position", [0, 0, 0])
        spd = obj.get("speed", 0.0)
        parts.append(f"{cls}@({pos[0]:.1f},{pos[1]:.1f},spd={spd:.1f})")
    print(f"frame {frame:04d}  {path.name}  [{len(boxes)} objs]  {', '.join(parts)}")


def _resolve_route(data_root: Path, route: Path | None, rng: random.Random) -> Path:
    if route is not None:
        route = route.resolve()
        if not route.is_dir():
            raise FileNotFoundError(f"route not found: {route}")
        return route
    routes = _list_routes(data_root)
    if not routes:
        raise FileNotFoundError(f"no routes under {data_root}")
    return rng.choice(routes)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=Path, default=DEFAULT_SPLIT, help="split root to pick random route")
    p.add_argument("--route", type=Path, default=None, help="fixed route dir (default: random from split)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument("--frame", type=int, default=None, help="frame index N → boxes/NNNN.json.gz")
    p.add_argument(
        "--all-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="loop all boxes/*.json.gz in the route",
    )
    p.add_argument("--stride", type=int, default=1, help="with --all-frames: every N-th file")
    p.add_argument("--class", dest="class_filter", default=None, help="filter by boxes[i]['class']")
    p.add_argument("--json", action="store_true", help="print raw JSON array")
    p.add_argument("--verbose", action="store_true", help="print full raw JSON per object")
    args = p.parse_args(argv)

    data_root = _data_root(args.split)
    if not data_root.is_dir():
        print(f"split not found: {args.split}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    try:
        route = _resolve_route(data_root, args.route, rng)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    frames = _list_frames(route)
    if not frames:
        print(f"no boxes/*.json.gz under {route}", file=sys.stderr)
        return 1

    if args.all_frames:
        picked = frames[:: max(1, args.stride)]
        print(f"route: {route}")
        print(f"files: {len(picked)} / {len(frames)} (stride={args.stride})")
        print("-" * 72)
        for path in picked:
            boxes = load_boxes(path)
            if args.json:
                print(json.dumps({"file": str(path), "boxes": boxes}, ensure_ascii=False))
            elif args.verbose:
                print()
                _print_boxes_file(
                    path,
                    as_json=False,
                    verbose=True,
                    class_filter=args.class_filter,
                )
                print("-" * 72)
            else:
                if args.class_filter is not None:
                    boxes = [o for o in boxes if o.get("class") == args.class_filter]
                _print_frame_compact(path, boxes)
        return 0

    if args.frame is not None:
        path = route / "boxes" / f"{int(args.frame):04d}.json.gz"
        if not path.is_file():
            print(f"frame not found: {path}", file=sys.stderr)
            return 1
    else:
        path = rng.choice(frames)

    print(f"route: {route}")
    print(f"file:  {path}")
    print("-" * 72)
    _print_boxes_file(
        path,
        as_json=args.json,
        verbose=args.verbose,
        class_filter=args.class_filter,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
