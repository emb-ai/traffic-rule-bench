#!/usr/bin/env python
"""Validate every data path run_signfix_pipeline.sh needs, before any stage runs.

The failure this guards against: on 2026-08-20 the zinkovich tree was
restructured underneath the pipeline, data/stop/scenes/junc_* became dangling
symlinks, and stage_eval spawned 50 scene subprocesses that each died on a
missing map. eval_pipeline.py reports that only as a per-scene "exit 1", the
driver aborted without a FAIL line, and the run looked healthy in STATUS.txt.

So this refuses to be quiet: every required path is checked for existence, for
being a dangling symlink, and for being readable, and the first failure names
the offending path.

Exit 0 = all good, 1 = at least one problem (all problems are printed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROBLEMS: list[str] = []
# Vendoring exists to get off this tree; a vendored path pointing back into it
# means the copy is incomplete.
ZINK_PREFIX = "/home/jovyan/shares/SR006.nfs2/zinkovich/"


def bad(what: str, path: object, why: str) -> None:
    PROBLEMS.append(f"{what}: {path} -- {why}")


def check_path(what: str, path: Path, *, kind: str = "file", allow_symlink: bool = False) -> bool:
    """True if path is present, resolvable and readable."""
    if path.is_symlink() and not os.path.exists(path):
        bad(what, path, f"dangling symlink -> {os.readlink(path)}")
        return False
    if path.is_symlink() and not allow_symlink:
        bad(what, path, f"is a symlink -> {os.readlink(path)} (vendored data must be a real file)")
        return False
    if not path.exists():
        bad(what, path, "missing")
        return False
    if kind == "dir":
        if not path.is_dir():
            bad(what, path, "not a directory")
            return False
        if not os.access(path, os.R_OK | os.X_OK):
            bad(what, path, "not readable/traversable")
            return False
        return True
    if not path.is_file():
        bad(what, path, "not a regular file")
        return False
    if not os.access(path, os.R_OK):
        bad(what, path, "not readable")
        return False
    if path.stat().st_size == 0:
        bad(what, path, "empty")
        return False
    return True


def read_jsonl(what: str, path: Path) -> list[dict]:
    if not check_path(what, path):
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                bad(what, f"{path}:{i}", f"not valid JSON ({exc})")
    if not rows:
        bad(what, path, "no rows")
    return rows


def check_scenes(rows: list[dict], scenes: Path, what: str) -> None:
    """Every scene the manifest names must be a real, readable SUMO map."""
    for scene_id in sorted({str(r.get("scene_id")) for r in rows if r.get("scene_id")}):
        d = scenes / scene_id
        if not check_path(f"{what} scene dir", d, kind="dir"):
            continue
        check_path(f"{what} scene map", d / "map.net.xml")
    # net_path is relative to scenes-root and is what the env actually opens.
    for net in sorted({str(r.get("net_path")) for r in rows if r.get("net_path")}):
        check_path(f"{what} net_path", scenes / net)


def check_no_zink(what: str, path: Path) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    if ZINK_PREFIX in text:
        bad(what, path, "still references the zinkovich tree (vendoring incomplete)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", type=Path, required=True)
    ap.add_argument("--test-manifest", type=Path, required=True)
    ap.add_argument("--train-experts", type=Path, default=None,
                    help="skip to validate eval inputs only")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--skip-replays", action="store_true",
                    help="do not stat the per-expert replay pkl/json pairs")
    args = ap.parse_args()

    check_path("scenes root", args.scenes, kind="dir")

    test_rows = read_jsonl("test manifest", args.test_manifest)
    if test_rows:
        check_scenes(test_rows, args.scenes, "test")
    # load_manifest_config() reads these from the manifest's own directory; when
    # manifest.json is absent spawn_distance_before_end silently falls back to
    # 12.0 and the eval is no longer the scenario the manifest describes.
    for sidecar in ("manifest.json", "real_manifest_summary.json"):
        check_path("manifest sidecar", args.test_manifest.parent / sidecar)

    if args.train_experts is not None:
        train_rows = read_jsonl("train experts", args.train_experts)
        if train_rows:
            check_scenes(train_rows, args.scenes, "train")
            check_no_zink("train experts", args.train_experts)
            if not args.skip_replays:
                for row in train_rows:
                    for key in ("pkl_path", "sidecar_path"):
                        v = row.get(key)
                        if not v:
                            bad("train experts", args.train_experts,
                                f"row {row.get('scene_uid')} has no {key}")
                            continue
                        check_path(f"train {key}", Path(v))

    if args.ckpt is not None:
        check_path("checkpoint", args.ckpt)

    if PROBLEMS:
        print(f"stop-data check FAILED with {len(PROBLEMS)} problem(s):", file=sys.stderr)
        for p in PROBLEMS:
            print(f"  {p}", file=sys.stderr)
        return 1
    n_test_scenes = len({r.get("scene_id") for r in test_rows})
    print(f"stop-data check OK: {len(test_rows)} test rows over {n_test_scenes} scenes, "
          f"scenes-root={args.scenes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
