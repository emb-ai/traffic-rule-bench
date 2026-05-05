#!/usr/bin/env python3
"""
Generate GIFs for sign 3.20 on pgmap:
pick 10 scenes with distinct scene_id (and their seeds) from the manifest
and call run_benchmark_mini.py with --save-gifs.

python generate_gifs.py --num-scenes 10 --dry-run
"""

import json
import subprocess
import sys
from pathlib import Path

SIGN = "3.20"

PDD_ROOT = Path("/home/jovyan/shares/SR006.nfs2/smirnova/sdc")
MANIFEST = PDD_ROOT / f"full_test_250_x10/{SIGN.replace('.', '_')}/pgmap_materialized.jsonl"
RUN_BENCH = PDD_ROOT / "pdd-bench/scripts/per_sign_bench/run_benchmark_mini.py"
GIF_DIR = PDD_ROOT / f"pdd-bench/gifs/{SIGN}"


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main(num_scenes: int = 10, dry_run: bool = False) -> int:
    if not RUN_BENCH.is_file():
        print(f"run_benchmark_mini.py not found at {RUN_BENCH}", file=sys.stderr)
        return 1

    GIF_DIR.mkdir(parents=True, exist_ok=True)

    picked = []
    seen_scene_ids = set()

    for row in iter_rows(MANIFEST):
        if row.get("source") != "pgmap":
            continue
        if row.get("pdd_code") != SIGN:
            continue

        sid = row.get("scene_id")
        seed = row.get("seed")
        if sid is None or seed is None:
            continue

        # ensure distinct scene_id (seed is already different per row)
        if sid in seen_scene_ids:
            continue

        seen_scene_ids.add(sid)
        picked.append(row)
        if len(picked) >= num_scenes:
            break

    if len(picked) < num_scenes:
        print(
            f"Only found {len(picked)} distinct scene_ids for pgmap 2.4 in {MANIFEST}",
            file=sys.stderr,
        )
        return 1

    for i, row in enumerate(picked, start=1):
        scene_uid = f"pgmap:{row['scene_id']}:{row['pdd_code']}:{row['seed']}"
        cmd = [
            sys.executable,
            str(RUN_BENCH),
            "--scene-uid",
            scene_uid,
            "--manifest",
            str(MANIFEST),
            "--save-gifs",
            "--gif-dir",
            str(GIF_DIR),
            "--run-name",
            SIGN,
            "--backends",
            "pgmap",
            "--policy",
            "idm",
        ]

        print(f"[{i}/{len(picked)}] {scene_uid}")
        print(" ", " ".join(cmd))

        if dry_run:
            continue

        res = subprocess.run(cmd, cwd=str(RUN_BENCH.parent))
        if res.returncode != 0:
            print(f"Command failed with code {res.returncode}", file=sys.stderr)
            return res.returncode

    return 0


if __name__ == "__main__":
    # change dry_run=True first if you want to only print commands
    raise SystemExit(main(num_scenes=10, dry_run=False))