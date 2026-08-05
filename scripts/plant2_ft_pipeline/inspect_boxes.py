#!/usr/bin/env python3
"""Pretty-print objects from boxes/NNNN.json.gz dumps.

Examples (from any cwd; source _env.sh first):
  source traffic-rule-bench/scripts/plant2_ft_pipeline/_env.sh

  ROUTE="$SHEPELEV/plant2_l1_fv_experts_split_signs_2.5/train/data/sign_100062_j2_lane0_seed1413785215_v0_default"

  # one frame
  $PY $INSPECT_BOXES --route "$ROUTE" --frame 21

  # only sign 2.5
  $PY $INSPECT_BOXES --route "$ROUTE" --frame 21 --class 2.5 --verbose

  # summary over all frames
  $PY $INSPECT_BOXES --route "$ROUTE" --summary --class 2.5

  # direct file path
  $PY $INSPECT_BOXES --file "$ROUTE/boxes/0021.json.gz" --class 2.5 --json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_ROUTE = (
    Path(__file__).resolve().parents[3]
    / "plant2_l1_fv_experts_split_signs_2.5/train/data/"
    "sign_100062_j2_lane0_seed1413785215_v0_default"
)


def load_boxes(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path}: expected list, got {type(data).__name__}")
    return data


def frame_path(route: Path, frame: int | str) -> Path:
    stem = f"{int(frame):04d}"
    path = route / "boxes" / f"{stem}.json.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


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
    extra_keys = sorted(
        k for k in obj
        if k not in {"class", "position", "yaw", "speed", "extent", "id", "type_id"}
    )
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


def print_frame(path: Path, *, class_filter: str | None, verbose: bool, as_json: bool) -> None:
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


def print_summary(route: Path, *, class_filter: str | None) -> None:
    files = sorted(route.glob("boxes/*.json.gz"))
    if not files:
        raise FileNotFoundError(f"no boxes/*.json.gz under {route}")

    per_frame_counts: Counter[str] = Counter()
    per_frame_total: list[tuple[str, int]] = []
    class_totals: Counter[str] = Counter()
    frames_with_class: Counter[str] = Counter()

    for path in files:
        boxes = load_boxes(path)
        frame_classes = Counter(obj.get("class", "?") for obj in boxes)
        per_frame_total.append((path.stem, len(boxes)))
        for cls, n in frame_classes.items():
            class_totals[cls] += n
            per_frame_counts[cls] = max(per_frame_counts[cls], n)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--route",
        type=Path,
        default=DEFAULT_ROUTE,
        help=f"route directory with boxes/ (default: {DEFAULT_ROUTE.name})",
    )
    p.add_argument(
        "--frame",
        type=int,
        default=None,
        help="frame index N → boxes/NNNN.json.gz (omit with --summary)",
    )
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
        help="direct path to boxes/NNNN.json.gz (overrides --route/--frame)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.file is not None:
        print_frame(args.file, class_filter=args.class_filter, verbose=args.verbose, as_json=args.json)
        return 0

    route = args.route.resolve()
    if not route.is_dir():
        print(f"route not found: {route}", file=sys.stderr)
        return 1

    if args.summary:
        print_summary(route, class_filter=args.class_filter)
        return 0

    if args.frame is None:
        print("provide --frame N or --summary (or --file PATH)", file=sys.stderr)
        return 2

    path = frame_path(route, args.frame)
    print_frame(path, class_filter=args.class_filter, verbose=args.verbose, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
