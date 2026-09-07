#!/usr/bin/env python3
"""Print the detour metrics for one or more historical eval directories.

    ./summarize_detour.py joint_k3_augFIX_lr3e4_ep24_last

The metric itself lives in `plant2_pipeline/metrics.py` — there is one
definition of compliance, not two. This script only resolves the historical
`eval/<tag>/fv_fast` layout to a path and prints a row per tag.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plant2_pipeline import config, metrics  # noqa: E402

EVAL_ROOT = Path(__file__).resolve().parent / "eval"
# Any detour family will do: family_summary only looks at `.group` to decide
# that compliance must additionally require `reached_zone`.
DETOUR = config.FAMILIES["detour_either"]


def main(tags: list[str]) -> None:
    print(metrics.Summary.header())
    for tag in tags:
        summary = metrics.family_summary(EVAL_ROOT / tag / "fv_fast", DETOUR)
        summary.benchmark = tag[:28]
        print(summary.as_row())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
