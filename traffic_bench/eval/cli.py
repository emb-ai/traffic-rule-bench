"""Eval CLI: manifest → run → metrics."""

from __future__ import annotations

import sys
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


def cmd_manifest(argv: List[str]) -> int:
    from traffic_bench.eval.manifest import run as generate_manifest

    return _run_module_main(generate_manifest, argv)


def cmd_run(argv: List[str]) -> int:
    from traffic_bench.eval.run import main as run_main

    return _run_module_main(run_main, argv)


def cmd_metrics(argv: List[str]) -> int:
    commands = ("csv", "aggregate", "report")
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "usage: python -m traffic_bench.eval metrics "
            f"{{{','.join(commands)}}} ...\n\n"
            "  csv         episodes / replays → metrics_per_episode.csv\n"
            "  aggregate   CSV → aggregations + reports/cumulative.json\n"
            "  report      cumulative JSON → markdown table\n"
        )
        return 0
    command = argv[0]
    if command not in commands:
        print(
            f"ERROR: unknown metrics command {command!r}. "
            f"Expected one of: {', '.join(commands)}",
            file=sys.stderr,
        )
        return 2
    if command == "csv":
        from traffic_bench.eval.metrics import csv as mod
    elif command == "aggregate":
        from traffic_bench.eval.metrics import aggregate as mod
    else:
        from traffic_bench.eval.metrics import report as mod
    return _run_module_main(mod, argv[1:])


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = ("manifest", "run", "metrics")
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "usage: python -m traffic_bench.eval "
            f"{{{','.join(commands)}}} ...\n\n"
            "Build a sign manifest, run one or many policies, or score episodes.\n\n"
            "Commands:\n"
            "  manifest    scenes + Hydra config → real_manifest.jsonl\n"
            "  run         one policy, or policies=[…] then metrics\n"
            "  metrics     episode JSONL → CSV / markdown\n"
        )
        return 0
    command = argv[0]
    if command not in commands:
        print(
            f"ERROR: unknown command {command!r}. "
            f"Expected one of: {', '.join(commands)}",
            file=sys.stderr,
        )
        return 2
    dispatch = {
        "manifest": cmd_manifest,
        "run": cmd_run,
        "metrics": cmd_metrics,
    }
    return dispatch[command](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
