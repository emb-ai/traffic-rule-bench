#!/usr/bin/env python3
"""Which expert did the oracle actually pick, and did the ego-style samples earn
their keep?

The IDM expert is recorded five times per scene -- `default` plus `s1..s4`,
each a different sampled driving style -- and selection keeps one. If the
samples were redundant, `default` would win nearly everything and recording
them would be four fifths wasted. This prints the split, per sign and overall,
next to the score that decided it.

    python tools/oracle_stats.py <collection_root> [--experts experts_idm]

<collection_root> holds one directory per sign family, each with an
<experts>/experts_scene_uid_top1.jsonl written by
`traffic_bench.oracle.select.coverage`.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
from pathlib import Path

VARIANTS = ["default", "s1", "s2", "s3", "s4"]


def load(root: Path, experts_dir: str):
    """Yield (family, record) for every picked expert under `root`."""
    for fam in sorted(os.listdir(root)):
        f = root / fam / experts_dir / "experts_scene_uid_top1.jsonl"
        if not f.is_file():
            continue
        for line in f.open():
            line = line.strip()
            if line:
                yield fam, json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--experts", default="experts_idm",
                    help="per-family selection directory (default: experts_idm)")
    ap.add_argument("--pool", default=None, metavar="DIR",
                    help="also summarise every candidate run from "
                         "<family>/DIR/all_runs_dedup.jsonl (e.g. --pool experts), "
                         "each IDM ego-style sample counted as its own expert")