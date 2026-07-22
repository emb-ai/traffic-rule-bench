#!/usr/bin/env python3
"""Scene-pool helper for no-entry signs (3.1 / 3.2).

Catalog scenes under ``pdd-bench/scenes/3.1`` / ``3.2`` are already local OSM
extracts, so junction cropping is not part of this bench. Use
``import_catalog_scenes.py`` to grow the pool, then review:

    python tools/filter_scenes/import_catalog_scenes.py --pdd-code 3.1 --limit 40
    python tools/filter_scenes/review_junction_scenes.py --pdd-code 3.1
    python tools/filter_scenes/build_scene_pool.py status --pdd-code 3.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

FILTER_SCENES_DIR = Path(__file__).resolve().parent
NO_ENTRY_SIGNS_DIR = FILTER_SCENES_DIR.parent.parent

sys.path.insert(0, str(NO_ENTRY_SIGNS_DIR))

from lib.no_entry_sign_spec import (  # noqa: E402
    DEFAULT_PDD_CODE,
    NO_ENTRY_SIGN_CODES,
    local_scenes_root,
)
from lib.scene_selection import (  # noqa: E402
    VERDICT_KEEP,
    VERDICT_PENDING,
    VERDICT_REJECT,
    load_scene_selection,
    is_reserved_scene_dir,
)
from lib.sumo_utils import CORE_SCENES_SUBDIR  # noqa: E402


def discover_scenes(scenes_root: Path) -> list[Path]:
    if not scenes_root.is_dir():
        return []
    out: list[Path] = []
    for entry in sorted(scenes_root.iterdir()):
        if not entry.is_dir():
            continue
        if is_reserved_scene_dir(entry.name) or entry.name == CORE_SCENES_SUBDIR:
            continue
        if (entry / "meta.json").is_file():
            out.append(entry)
    return out


def print_status(scenes_root: Path, *, target: int) -> int:
    scenes = discover_scenes(scenes_root)
    selection = load_scene_selection(scenes_root)
    kept = pending = rejected = unmarked = 0
    for scene in scenes:
        verdict = (selection.get("scenes") or {}).get(scene.name, {}).get("verdict")
        if verdict == VERDICT_KEEP:
            kept += 1
        elif verdict == VERDICT_REJECT:
            rejected += 1
        elif verdict == VERDICT_PENDING:
            pending += 1
        else:
            unmarked += 1
    print(f"scenes_root: {scenes_root}")
    print(f"total={len(scenes)} kept={kept} rejected={rejected} pending={pending} unmarked={unmarked} target={target}")
    print("Grow the pool with: python tools/filter_scenes/import_catalog_scenes.py --pdd-code … --limit N")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="status", choices=["status", "crop", "fill"])
    parser.add_argument("--pdd-code", default=DEFAULT_PDD_CODE, choices=list(NO_ENTRY_SIGN_CODES))
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--scenes-base", type=Path, default=NO_ENTRY_SIGNS_DIR / "scenes")
    args = parser.parse_args(argv)

    scenes_root = local_scenes_root(args.scenes_base, args.pdd_code)
    if args.command in ("crop", "fill"):
        print(
            f"[{args.command}] Not needed for no-entry signs. "
            "Catalog scenes are already local — use import_catalog_scenes.py."
        )
        return print_status(scenes_root, target=args.target)
    return print_status(scenes_root, target=args.target)


if __name__ == "__main__":
    raise SystemExit(main())
