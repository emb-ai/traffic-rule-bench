"""One-chunk worker entry for NN eval (collect.sh-style GPU pin via env).

Launched as::

    CUDA_VISIBLE_DEVICES=<index> python -m traffic_bench.eval.run.shard_worker JOB.json

Leave cluster ``NVIDIA_VISIBLE_DEVICES`` intact — do not remount to a single UUID.
"""
from __future__ import annotations

import json
import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m traffic_bench.eval.run.shard_worker JOB.json", file=sys.stderr)
        return 2
    with open(args[0], encoding="utf-8") as handle:
        job = json.load(handle)
    from traffic_bench.eval.run.policies import _run_scene_shard

    _run_scene_shard(job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
