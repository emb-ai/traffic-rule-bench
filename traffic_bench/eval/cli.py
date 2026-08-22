"""Eval CLI: manifest → run / pipeline → metrics.

Facades over the current entry points. Hydra / CLI / pipeline shells
still live in ``generate_manifest.py``, ``run_benchmark.py``, and
``eval_pipeline.py``; the episode loop is ``bench.episode``, family
expand/place is under ``signs/``.
"""

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
    from traffic_bench.eval import generate_manifest

    return _run_module_main(generate_manifest, argv)


def cmd_run(argv: List[str]) -> int:
    from traffic_bench.eval import run_benchmark

    if "--run-name" not in argv and "--policy" in argv:
        try:
            policy = argv[argv.index("--policy") + 1]
        except (ValueError, IndexError):
            policy = None
        if policy and not str(policy).startswith("-"):
            argv = [*argv, "--run-name", policy]
    return _run_module_main(run_benchmark, argv)


def cmd_pipeline(argv: List[str]) -> int:
    from traffic_bench.eval import eval_pipeline

    return _run_module_main(eval_pipeline, argv)


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
        from traffic_bench.eval.metrics import build_episode_metrics_csv as mod
    elif command == "aggregate":
        from traffic_bench.eval.metrics import aggregate_episode_metrics as mod
    else:
        from traffic_bench.eval.metrics import generate_cumulative_markdown_report as mod
    return _run_module_main(mod, argv[1:])


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = ("manifest", "run", "pipeline", "metrics")
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "usage: python -m traffic_bench.eval "
            f"{{{','.join(commands)}}} ...\n\n"
            "Build a sign manifest, run one policy, or score many policies.\n\n"
            "Commands:\n"
            "  manifest    scenes + Hydra config → real_manifest.jsonl\n"
            "  run         one policy on one manifest (closed-loop episodes)\n"
            "  pipeline    many policies + metrics report\n"
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
        "pipeline": cmd_pipeline,
        "metrics": cmd_metrics,
    }
    return dispatch[command](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
