"""Thin shim — PlanT2 batch collection lives in ``expert_replay_inenv.py``.

Prefer:

    python expert_replay_inenv.py \\
        --experts /path/to/experts_scene_uid_top1.jsonl \\
        --scenes-root /path/to/scenes_balanced \\
        --save-plant2-dir /path/to/plant2_out \\
        --count 5 --save-gifs

This module keeps the old entry point working by forwarding argv.
"""
from __future__ import annotations

import sys
import warnings

# Preserve relative imports when invoked as a script from this directory.
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def main() -> None:
    warnings.warn(
        "export_plant2_from_experts.py is deprecated; use expert_replay_inenv.py "
        "with --experts --scenes-root --save-plant2-dir instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Re-dispatch: if caller used this script's argv shape, expert_replay_inenv
    # accepts the same --experts / --scenes-root / --save-plant2-dir flags.
    from expert_replay_inenv import main as inenv_main
    inenv_main()


if __name__ == "__main__":
    main()
