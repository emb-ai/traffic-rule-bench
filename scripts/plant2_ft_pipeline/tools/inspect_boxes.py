#!/usr/bin/env python3
"""Pretty-print objects from boxes/NNNN.json.gz dumps.

No PlanTDataset / generate_batch — only what's on disk in the dump.

Examples (from any cwd; source _env.sh first):
  source traffic-rule-bench/scripts/plant2_ft_pipeline/_env.sh

  ROUTE="$SHEPELEV/plant2_l1_fv_experts_split_signs_2.5/train/data/sign_100062_j2_lane0_seed1413785215_v0_default"

  # one frame, explicit route
  $PY $INSPECT_BOXES --route "$ROUTE" --frame 21

  # only sign 2.5
  $PY $INSPECT_BOXES --route "$ROUTE" --frame 21 --class 2.5 --verbose

  # summary over all frames
  $PY $INSPECT_BOXES --route "$ROUTE" --summary --class 2.5

  # direct file path
  $PY $INSPECT_BOXES --file "$ROUTE/boxes/0021.json.gz" --class 2.5 --json

  # random route + random frame from a split
  $PY $INSPECT_BOXES --random

  # random route, all frames (compact)
  $PY $INSPECT_BOXES --random --all-frames --stride 10

  # random route from a fixed split, raw JSON
  $PY $INSPECT_BOXES --random --split "$SHEPELEV/plant2_l1_fv_experts_split_signs_2.5/train" --json --seed 42
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
from collections import Counter

from lib.env import shepelev

DEFAULT_ROUTE = (
    Path(__file__).resolve().parents[3]
    / "plant2_l1_fv_experts_split_signs_2.5/train/data/"
    "sign_100062_j2_lane0_seed1413785215_v0_default"
)
DEFAULT_SPLIT = shepelev() / "plant2_l1_fv_experts_split_signs_2.5" / "train"


# --------------------------------------------------------------------------
# route / frame discovery
# --------------------------------------------------------------------------


def data_root_of(split: Path) -> Path:
    return split / "data" if (split / "data").is_dir() else split


def list_routes(data_root: Path) -> list[Path]:
    return sorted(
        p for p in data_root.iterdir()
        if p.is_dir() and (p / "boxes").is_dir()
    )


def list_frames(route: Path) -> list[Path]:
    return sorted(route.glob("boxes/*.json.gz"))


def frame_num(path: Path) -> int:
    return int(path.name.split(".")[0])


def frame_path(route: Path, frame: int | str) -> Path:
    stem = f"{int(frame):04d}"
    path = route / "boxes" / f"{stem}.json.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_route(*, route: Path | None, random_: bool, split: Path, rng: random.Random) -> Path:
    """Explicit --route always wins; --random picks from --split; else DEFAULT_ROUTE."""
    if route is not None:
        route = route.resolve()
        if not route.is_dir():
            raise FileNotFoundError(f"route not found: {route}")
        return route
    if random_:
        data_root = data_root_of(split)
        if not data_root.is_dir():
            raise FileNotFoundError(f"split not found: {split}")
        routes = list_routes(data_root)
        if not routes:
            raise FileNotFoundError(f"no routes under {data_root}")
        return rng.choice(routes)
    return DEFAULT_ROUTE


# --------------------------------------------------------------------------
# box loading / formatting
# --------------------------------------------------------------------------


def load_boxes(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path}: expected list, got {type(data).__name__}")
    return data


def fmt_pos(pos: list[float]) -> str:
    return f"({pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:5.2f})"


def fmt_extent(ext: list[float]) -> str:
    return f"[{ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f}]"


def dist_xy(obj: dict) -> float:
    x, y = obj["position"][:2]
    return (x * x + y * y) ** 0.5


def print_object(idx: int, obj: dict, *, verbose: bool) -> None:
    cls = obj.get("class", "?")
    pos = obj.get("position", [0.0, 0.0, 0.0])
    ext = obj.get("extent", [0.0, 0.0, 0.0])
    known = {"class", "position", "yaw", "speed", "extent", "id", "type_id"}
    extra_keys = sorted(k for k in obj if k not in known)
    line = (
        f"[{idx:2d}] class={cls!r:14s} id={obj.get('id', '?'):3} "
        f"pos={fmt_pos(pos)} dist={dist_xy(obj):5.1f}m "
        f"yaw={obj.get('yaw', 0.0):7.3f} rad speed={obj.get('speed', 0.0):6.2f} "
        f"extent={fmt_extent(ext)}"
    )
    print(line)
    if obj.get("type_id") is not None:
        print(f"      type_id={obj['type_id']!r}")
    for key in extra_keys:
        print(f"      {key}={obj[key]!r}")
    if verbose:
        print(f"      raw={json.dumps(obj, ensure_ascii=False)}")


def print_boxes_file(
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
        print_object(idx, obj, verbose=verbose)
        shown += 1
    if class_filter is not None:
        print(f"--- matched {shown} / {len(boxes)} objects (class={class_filter!r}) ---")


def print_frame_compact(path: Path, boxes: list[dict]) -> None:
    frame = frame_num(path)
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


def print_summary(route: Path, *, class_filter: str | None) -> None:
    files = sorted(route.glob("boxes/*.json.gz"))
    if not files:
        raise FileNotFoundError(f"no boxes/*.json.gz under {route}")

    per_frame_total: list[tuple[str, int]] = []
    class_totals: Counter[str] = Counter()
    frames_with_class: Counter[str] = Counter()

    for path in files:
        boxes = load_boxes(path)
        frame_classes = Counter(obj.get("class", "?") for obj in boxes)
        per_frame_total.append((path.stem, len(boxes)))
        for cls, n in frame_classes.items():
            class_totals[cls] += n
            frames_with_class[cls] += 1

    print(f"=== summary: {route} ===")
    print(f"frames: {len(files)}")
    print(f"objects total: {sum(class_totals.values())}")
    print("\nclass totals (object-instances across all frames):")
    for cls, n in class_totals.most_common():
        print(f"  {cls!r:16s}  n={n:5d}  frames_with={frames_with_class[cls]:4d}")

    if class_filter is not None:
        hits = [
            (path.stem, obj)
            for path in files
            for obj in load_boxes(path)
            if obj.get("class") == class_filter
        ]
        print(f"\nframes with class={class_filter!r}: {frames_with_class.get(class_filter, 0)}")
        if hits:
            print("first / last appearance:")
            for label, obj in (hits[0], hits[-1]):
                pos = obj["position"]
                print(
                    f"  frame {label}: pos=({pos[0]:.1f}, {pos[1]:.1f}) "
                    f"dist={dist_xy(obj):.1f}m affects_ego={obj.get('affects_ego')}"
                )
        return

    busiest = max(per_frame_total, key=lambda x: x[1])
    print(f"\nbusiest frame: {busiest[0]} ({busiest[1]} objects)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--route",
        type=Path,
        default=None,
        help=f"route directory with boxes/ (default: DEFAULT_ROUTE={DEFAULT_ROUTE.name}, "
        "or a random route under --split if --random is given)",
    )
    p.add_argument(
        "--random",
        action="store_true",
        help="pick a random route from --split (ignored if --route is given)",
    )
    p.add_argument("--split", type=Path, default=DEFAULT_SPLIT, help="split root to pick a random route from")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for --random / random-frame fallback")
    p.add_argument(
        "--frame",
        type=int,
        default=None,
        help="frame index N -> boxes/NNNN.json.gz (omit with --summary/--all-frames for a random frame)",
    )
    p.add_argument(
        "--all-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="loop all boxes/*.json.gz in the route",
    )
    p.add_argument("--stride", type=int, default=1, help="with --all-frames: every N-th file")
    p.add_argument(
        "--class",
        dest="class_filter",
        default=None,
        metavar="CODE",
        help='filter by boxes[i]["class"], e.g. 2.5, car, static',
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="aggregate stats over all frames in --route",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="print raw JSON (respects --class filter)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="also print full raw JSON per object",
    )
    p.add_argument(
        "--file",
        type=Path,
        default=None,
        help="direct path to boxes/NNNN.json.gz (overrides --route/--random/--frame)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.file is not None:
        print_boxes_file(args.file, as_json=args.json, verbose=args.verbose, class_filter=args.class_filter)
        return 0

    rng = random.Random(args.seed)
    route = resolve_route(route=args.route, random_=args.random, split=args.split, rng=rng)

    if args.summary:
        print_summary(route, class_filter=args.class_filter)
        return 0

    frames = list_frames(route)
    if not frames:
        raise FileNotFoundError(f"no boxes/*.json.gz under {route}")

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
                print_boxes_file(path, as_json=False, verbose=True, class_filter=args.class_filter)
                print("-" * 72)
            else:
                if args.class_filter is not None:
                    boxes = [o for o in boxes if o.get("class") == args.class_filter]
                print_frame_compact(path, boxes)
        return 0

    if args.frame is not None:
        path = frame_path(route, args.frame)
    else:
        # No --frame / --all-frames / --summary: show one random frame from the route.
        path = rng.choice(frames)

    print(f"route: {route}")
    print(f"file:  {path}")
    print("-" * 72)
    print_boxes_file(path, as_json=args.json, verbose=args.verbose, class_filter=args.class_filter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
