#!/usr/bin/env python
"""Vendor the STOP (2.5) pipeline's data inputs out of the zinkovich tree.

The zinkovich layout moved (pdd-bench/scripts/per_sign_bench/priority_bench ->
pdd-bench/sign_bench) and data/stop/scenes/junc_* became dangling symlinks into
the deleted path, which breaks every stage of run_signfix_pipeline.sh. This
copies the four things the pipeline reads into $TRB_ROOT/stop_data/ and rewrites
the absolute paths embedded in the expert index so the vendored tree is
self-consistent.

Read-only with respect to /home/jovyan/shares/SR006.nfs2/zinkovich/.
Idempotent: re-running skips files that already match in size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

TRB_ROOT = Path("/home/jovyan/shares/SR006.nfs2/belyaev/traffic-rule-bench")
ZINK = Path("/home/jovyan/shares/SR006.nfs2/zinkovich/zinkovich/traffic-rule-bench")
ZPDD = ZINK / "pdd-bench"
SRC_STOP = ZPDD / "sign_bench" / "data" / "stop"
DEST = TRB_ROOT / "stop_data"

# scenes/junc_* symlinks still point here; the tree was deleted in the move.
DEAD_SCENE_ROOTS = (
    ZPDD / "scripts" / "per_sign_bench" / "moscow_scenes" / "scenes",
    ZPDD / "scripts" / "per_sign_bench" / "moscow_junctions" / "scenes",
)
LIVE_SCENE_ROOT = ZPDD / "moscow_scenes" / "scenes"
SCENE_SHAPES = ("T", "X", "O")

# pkl_path / sidecar_path inside the expert index and the replay sidecars are
# absolute and point at the deleted tree.
OLD_TRAJ_PREFIX = (
    "/home/jovyan/shares/SR006.nfs2/zinkovich/zinkovich/traffic-rule-bench/pdd-bench"
    "/scripts/per_sign_bench/priority_bench/data/stop/trajectories/debug_train_400"
)
LIVE_TRAJ = SRC_STOP / "trajectories" / "debug_train_400"
NEW_TRAJ = DEST / "trajectories" / "debug_train_400"

TS_TEST_FILES = (
    "real_manifest.jsonl",
    "manifest.json",
    "real_manifest_summary.json",
    "config.yaml",
)
CKPT0_REL = "checkpoints/plant2_pretrain/epoch=029_final_3.ckpt"


def die(msg: str) -> None:
    sys.exit(f"FAIL: {msg}")


def guard_dest(p: Path) -> Path:
    """Refuse to write anywhere outside the belyaev tree."""
    rp = p.resolve() if p.exists() else Path(os.path.abspath(p))
    if not str(rp).startswith(str(TRB_ROOT)):
        die(f"refusing to write outside {TRB_ROOT}: {p}")
    return p


def copy_file(src: Path, dst: Path, *, dry: bool) -> int:
    """Copy src->dst unless dst already has the same size. Returns bytes copied."""
    guard_dest(dst)
    size = src.stat().st_size
    if dst.is_file() and dst.stat().st_size == size:
        return 0
    if dry:
        return size
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return size


def resolve_scene(scene_id: str) -> Path:
    """Find the real directory for a scene, following or repairing the symlink."""
    link = SRC_STOP / "scenes" / scene_id
    if link.is_symlink():
        target = Path(os.readlink(link))
        if target.is_dir():
            return target
        # Dangling: the same scene name exists under the surviving map root.
        for dead in DEAD_SCENE_ROOTS:
            try:
                rel = target.relative_to(dead)
            except ValueError:
                continue
            cand = LIVE_SCENE_ROOT / rel
            if cand.is_dir():
                return cand
    elif link.is_dir():
        return link
    for shape in SCENE_SHAPES:
        cand = LIVE_SCENE_ROOT / shape / scene_id
        if cand.is_dir():
            return cand
    die(f"cannot resolve scene {scene_id} (link {link})")
    raise AssertionError  # unreachable, keeps type checkers happy


def needed_scene_ids() -> tuple[list[str], list[str]]:
    test_manifest = SRC_STOP / "output" / "ts_test" / "real_manifest.jsonl"
    experts = LIVE_TRAJ / "experts" / "experts_scene_uid_top1.jsonl"
    for p in (test_manifest, experts):
        if not p.is_file():
            die(f"source missing: {p}")
    test = sorted({json.loads(l)["scene_id"] for l in test_manifest.open() if l.strip()})
    train = sorted({json.loads(l)["scene_id"] for l in experts.open() if l.strip()})
    return test, train


def vendor_scenes(scene_ids: list[str], *, dry: bool) -> int:
    total = 0
    for sid in scene_ids:
        src = resolve_scene(sid)
        dst = DEST / "scenes" / sid
        for f in sorted(src.rglob("*")):
            if f.is_file():
                total += copy_file(f, dst / f.relative_to(src), dry=dry)
    return total


def vendor_ts_test(*, dry: bool) -> int:
    src = SRC_STOP / "output" / "ts_test"
    total = 0
    for name in TS_TEST_FILES:
        f = src / name
        if not f.is_file():
            die(f"ts_test sidecar missing: {f}")
        total += copy_file(f, DEST / "output" / "ts_test" / name, dry=dry)
    return total


def _relocate(path_str: str) -> str:
    if not path_str.startswith(OLD_TRAJ_PREFIX):
        die(f"unexpected trajectory path, cannot relocate: {path_str}")
    return str(NEW_TRAJ) + path_str[len(OLD_TRAJ_PREFIX):]


def _live(path_str: str) -> Path:
    """Map an old absolute trajectory path to where the file lives today."""
    return LIVE_TRAJ / Path(path_str[len(OLD_TRAJ_PREFIX):].lstrip("/"))


def vendor_train_trajectories(*, dry: bool) -> tuple[int, int]:
    experts_src = LIVE_TRAJ / "experts" / "experts_scene_uid_top1.jsonl"
    rows = [json.loads(l) for l in experts_src.open() if l.strip()]
    total = 0
    for row in rows:
        for key in ("pkl_path", "sidecar_path"):
            src = _live(row[key])
            if not src.is_file():
                die(f"train replay missing at live location: {src}")
            total += copy_file(src, Path(_relocate(row[key])), dry=dry)

    if not dry:
        # Rewrite the two absolute paths the sidecars carry so the vendored tree
        # never points back at the deleted zinkovich path.
        for row in rows:
            dst = Path(_relocate(row["sidecar_path"]))
            text = dst.read_text(encoding="utf-8")
            if OLD_TRAJ_PREFIX in text:
                dst.write_text(text.replace(OLD_TRAJ_PREFIX, str(NEW_TRAJ)), encoding="utf-8")

        out_dir = guard_dest(NEW_TRAJ / "experts")
        out_dir.mkdir(parents=True, exist_ok=True)
        # Keep the untouched original next to the rewritten index so the
        # relocation stays auditable.
        copy_file(experts_src, out_dir / "experts_scene_uid_top1.jsonl.zink_orig", dry=False)
        lines = []
        for row in rows:
            row["pkl_path"] = _relocate(row["pkl_path"])
            row["sidecar_path"] = _relocate(row["sidecar_path"])
            lines.append(json.dumps(row, ensure_ascii=False))
        (out_dir / "experts_scene_uid_top1.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        cov = LIVE_TRAJ / "experts" / "coverage_report.json"
        if cov.is_file():
            copy_file(cov, out_dir / "coverage_report.json", dry=False)
    return total, len(rows)


def vendor_ckpt0(*, dry: bool) -> int:
    src = ZPDD / CKPT0_REL
    if not src.is_file():
        die(f"pretrain checkpoint missing: {src}")
    return copy_file(src, DEST / CKPT0_REL, dry=dry)


def gb(n: int) -> str:
    return f"{n / 1024 ** 3:.3f} GB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report sizes, copy nothing")
    args = ap.parse_args()
    dry = args.dry_run

    test_ids, train_ids = needed_scene_ids()
    scene_ids = sorted(set(test_ids) | set(train_ids))
    print(f"scenes needed: {len(scene_ids)} ({len(test_ids)} test + {len(train_ids)} train)")

    n_scenes = vendor_scenes(scene_ids, dry=dry)
    print(f"scenes:               {gb(n_scenes)}")
    n_ts = vendor_ts_test(dry=dry)
    print(f"ts_test sidecars:     {gb(n_ts)}")
    n_traj, n_rows = vendor_train_trajectories(dry=dry)
    print(f"train replays:        {gb(n_traj)} ({n_rows} experts)")
    n_ckpt = vendor_ckpt0(dry=dry)
    print(f"pretrain checkpoint:  {gb(n_ckpt)}")
    print(f"TOTAL {'would copy' if dry else 'copied'}: "
          f"{gb(n_scenes + n_ts + n_traj + n_ckpt)} -> {DEST}")


if __name__ == "__main__":
    main()
