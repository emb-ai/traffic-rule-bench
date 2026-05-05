#!/usr/bin/env python3
"""
prepare_shard_splits.py  --nodes N  [--pt-dir DIR]  [--output-dir DIR]

Splits the .pt files in pt-dir into N non-overlapping lists.
Writes one file-list per node:
    shard_splits/node_00.txt
    shard_splits/node_01.txt
    ...

Each node then runs:
    python shard_plant2_pt.py \\
        --file-list shard_splits/node_<K>.txt \\
        --output-dir plant2_shards_expert \\
        --steps-per-shard 1024 --shuffle
"""
import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

# Default paths: under pdd-bench/outputs/ (same repo layout for any checkout).
_PDD_BENCH = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument(
        "--pt-dir",
        default=str(_PDD_BENCH / "outputs" / "plant2_pt_episodes"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_PDD_BENCH / "outputs" / "shard_splits"),
    )
    args = parser.parse_args()

    pt_dir = Path(args.pt_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only new-named files (contain __)
    files = sorted(f for f in pt_dir.glob("*.pt") if "__" in f.stem)
    print(f"Total .pt files: {len(files)}")

    # Shuffle globally so each node gets a random mix of episodes,
    # giving good cross-dataset coverage within every node's shards.
    random.seed(42)
    random.shuffle(files)
    print("Files shuffled (seed=42)")

    N = args.nodes
    slice_size = math.ceil(len(files) / N)

    print(f"Split into {N} nodes  (slice_size ≈ {slice_size}):")
    for k in range(N):
        chunk = files[k * slice_size : (k + 1) * slice_size]
        out_file = out_dir / f"node_{k:02d}.txt"
        with out_file.open("w") as f:
            for p in chunk:
                f.write(str(p) + "\n")
        print(f"  node_{k:02d}.txt   {len(chunk):5d} files")

    info = {
        "created_at": datetime.now().isoformat(),
        "n_nodes": N,
        "total_files": len(files),
        "slice_size": slice_size,
    }
    with (out_dir / "split_info.json").open("w") as f:
        json.dump(info, f, indent=2)

    print("\nLaunch on each node K:")
    print("  python shard_plant2_pt.py \\")
    print(f"      --file-list {out_dir}/node_<K>.txt \\")
    print("      --output-dir plant2_shards_expert \\")
    print("      --steps-per-shard 1024 --shuffle")


if __name__ == "__main__":
    main()
