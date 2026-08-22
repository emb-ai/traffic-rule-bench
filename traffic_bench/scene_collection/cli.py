"""Scene collection CLI: collect → assign → materialize → prepare → filter → pack."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional


def _run_module_main(module, argv: List[str]) -> int:
    old = sys.argv
    try:
        sys.argv = [module.__name__, *argv]
        module.main()
    except SystemExit as exc:
        code = exc.code
        return 0 if code in (None, 0) else int(code)
    finally:
        sys.argv = old
    return 0


def cmd_collect(argv: List[str]) -> int:
    from traffic_bench.scene_collection.collect import build_net
    from traffic_bench.scene_collection.collect.enumerate import junctions as enum_j
    from traffic_bench.scene_collection.collect.enumerate import segments as enum_s
    from traffic_bench.scene_collection.collect import make_split
    from traffic_bench.scene_collection.collect.junctions import crop as crop_j
    from traffic_bench.scene_collection.collect.dual_path import crop as crop_dp
    from traffic_bench.scene_collection.collect.segments import crop as crop_seg

    ap = argparse.ArgumentParser(
        prog="python -m traffic_bench.scene_collection collect",
        description="OSM → net → enumerate → make_split → crop junction/dual_path/segment",
    )
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-netconvert", action="store_true")
    ap.add_argument("--skip-enumerate", action="store_true")
    ap.add_argument("--skip-split", action="store_true")
    ap.add_argument("--skip-crop", action="store_true")
    ap.add_argument("--skip-dual-path", action="store_true")
    ap.add_argument("--skip-segment", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--shapes", default="T,X,O")
    ap.add_argument("--max-per-shape", type=int, default=None)
    ap.add_argument("--radius-m", type=float, default=80.0)
    ap.add_argument("--min-lane-m", type=float, default=10.0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dual-path-max-per-slot", type=int, default=0)
    args = ap.parse_args(argv)

    build_argv: List[str] = []
    if args.skip_download:
        build_argv.append("--skip-download")
    if args.force_download:
        build_argv.append("--force-download")
    if args.skip_netconvert:
        build_argv.append("--skip-netconvert")
    rc = _run_module_main(build_net, build_argv)
    if rc:
        return rc

    if not args.skip_enumerate:
        rc = _run_module_main(
            enum_j,
            ["--shapes", args.shapes, "--min-lane-m", str(args.min_lane_m)],
        )
        if rc:
            return rc

    if not args.skip_split:
        rc = _run_module_main(make_split, [])
        if rc:
            return rc

    if not args.skip_crop:
        crop_argv = [
            "--shapes",
            args.shapes,
            "--radius-m",
            str(args.radius_m),
            "--workers",
            str(args.workers),
        ]
        if args.max_per_shape is not None:
            crop_argv += ["--max-per-shape", str(args.max_per_shape)]
        if args.skip_existing:
            crop_argv.append("--skip-existing")
        rc = _run_module_main(crop_j, crop_argv)
        if rc:
            return rc

    if not args.skip_dual_path:
        dual_argv = ["--max-per-slot", str(args.dual_path_max_per_slot)]
        if args.skip_existing:
            dual_argv.append("--skip-existing")
        rc = _run_module_main(crop_dp, dual_argv)
        if rc:
            return rc

    if not args.skip_segment:
        if not args.skip_enumerate:
            rc = _run_module_main(enum_s, [])
            if rc:
                return rc
        seg_argv = [
            "--max-scenes",
            "0",
            "--workers",
            str(args.workers),
        ]
        if args.skip_existing:
            seg_argv.append("--skip-existing")
        rc = _run_module_main(crop_seg, seg_argv)
        if rc:
            return rc

    print("[collect] Finished.")
    return 0


def cmd_assign(argv: List[str]) -> int:
    from traffic_bench.scene_collection.assign import assign as assign_mod

    return _run_module_main(assign_mod, argv)


def cmd_materialize(argv: List[str]) -> int:
    from traffic_bench.scene_collection.sign_scenes.materialize import run as mat

    return _run_module_main(mat, argv)


def cmd_prepare(argv: List[str]) -> int:
    from traffic_bench.scene_collection.sign_scenes.prepare import run as prep

    return prep.main(argv)


def cmd_reject(argv: List[str]) -> int:
    from traffic_bench.scene_collection.sign_scenes.filter import reject as rej

    return _run_module_main(rej, argv)


def cmd_review(argv: List[str]) -> int:
    from traffic_bench.scene_collection.sign_scenes.filter import review as rev

    return _run_module_main(rev, argv)


def _copy_scene_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    if src.is_symlink():
        shutil.copytree(src.resolve(), dst, symlinks=False)
    else:
        shutil.copytree(src, dst, symlinks=False)


def cmd_analysis(argv: List[str]) -> int:
    from traffic_bench.scene_collection.analysis import run as analysis_run

    return analysis_run.main(argv)


def cmd_pack(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m traffic_bench.scene_collection pack",
        description=(
            "Dereference symlinks and copy one sign's scenes into a standalone folder "
            "(so you can share/download just that sign without maps/)."
        ),
    )
    ap.add_argument("--sign", required=True, help="Eval profile id, e.g. yield")
    ap.add_argument("--out", type=Path, required=True, help="Destination directory")
    ap.add_argument("--scenes-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    from traffic_bench.eval.sign_registry import get_profile, scenes_dir as profile_scenes_dir
    from traffic_bench.scene_collection.sign_scenes.filter.selection import is_reserved_scene_dir

    profile = get_profile(args.sign)
    src = args.scenes_dir or profile_scenes_dir(profile)
    if not src.is_dir():
        print(f"ERROR: scenes dir not found: {src}", file=sys.stderr)
        return 1
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for child in sorted(src.iterdir()):
        name = child.name
        if child.is_dir() and is_reserved_scene_dir(name):
            continue
        dest = out / name
        if child.is_dir() or child.is_symlink():
            if name in {"moscow_pool.json", "scene_selection.json"}:
                continue
            if child.is_file() or (child.is_symlink() and child.resolve().is_file()):
                shutil.copy2(child, dest)
            else:
                _copy_scene_dir(child, dest)
                n += 1
        elif child.is_file():
            shutil.copy2(child, dest)
    print(f"[pack] {profile.id}: {n} scenes → {out}")
    print(
        "This folder is self-contained (symlinks followed). "
        "You can copy it without maps/."
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = (
        "collect",
        "assign",
        "materialize",
        "prepare",
        "reject",
        "review",
        "pack",
        "analysis",
    )
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "usage: python -m traffic_bench.scene_collection "
            f"{{{','.join(commands)}}} ...\n\n"
            "Collect city maps, assign them to signs, materialize and filter scenes.\n\n"
            "Commands:\n"
            "  collect       OSM → maps/ (net, index, split, crops)\n"
            "  assign        signs.yaml → sign_allocations.json\n"
            "  materialize   allocated maps → data/scenes/<sign>/\n"
            "  prepare       sign-specific surgery (crosswalk zebra)\n"
            "  reject        drop unusable scenes, optional refill\n"
            "  review        browser keep/reject UI\n"
            "  pack          dereference symlinks into a standalone folder\n"
            "  analysis      harvest counts + diversity figures\n"
        )
        return 0
    command = argv[0]
    if command not in commands:
        print(f"ERROR: unknown command {command!r}. Expected one of: {', '.join(commands)}", file=sys.stderr)
        return 2
    dispatch = {
        "collect": cmd_collect,
        "assign": cmd_assign,
        "materialize": cmd_materialize,
        "prepare": cmd_prepare,
        "reject": cmd_reject,
        "review": cmd_review,
        "pack": cmd_pack,
        "analysis": cmd_analysis,
    }
    return dispatch[command](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
