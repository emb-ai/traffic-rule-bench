"""Eval CLI: manifest → run → metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from traffic_bench.eval.run_layout import (
    default_run_manifest_dir,
    latest_debug_dir,
    register_path_resolvers,
)
from traffic_bench.eval.sign_registry import (
    SignProfile,
    hydra_sign_override,
    resolve_sign_token,
    runs_dir,
    profiles_from_sign_value,
)

register_path_resolvers()


def _take_override(argv: List[str], key: str) -> Tuple[Optional[str], List[str]]:
    prefix = f"{key}="
    value: Optional[str] = None
    kept: List[str] = []
    for arg in argv:
        if value is None and arg.startswith(prefix):
            value = arg[len(prefix) :]
            continue
        kept.append(arg)
    return value, kept


def _run_module_main(module, argv: List[str]) -> int:
    old = sys.argv
    try:
        sys.argv = [module.__name__, *argv]
        module.main()
    except SystemExit as exc:
        code = exc.code
        if code in (None, 0):
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    finally:
        sys.argv = old
    return 0


def _jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def _summary_n_scenes(run_dir: Path) -> int:
    summary_path = run_dir / "real_manifest_summary.json"
    if summary_path.is_file():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        scenes = data.get("scenes")
        if isinstance(scenes, list):
            return len(scenes)
        for key in ("n_scenes", "num_scenes"):
            if key in data:
                return int(data[key])
    manifest = run_dir / "real_manifest.jsonl"
    if not manifest.is_file():
        return 0
    ids: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = row.get("scene_id") or row.get("scene_name")
            if sid:
                ids.add(str(sid))
    return len(ids)


def _manifest_folder(argv: Sequence[str]) -> str:
    raw, _ = _take_override(list(argv), "paths.split")
    return (raw or "debug").strip() or "debug"


def print_manifest_sign_summary(
    profiles: Sequence[SignProfile],
    *,
    folder: str,
) -> None:
    rows: list[tuple[str, int, int, str]] = []
    for profile in profiles:
        run_dir = runs_dir(profile) / folder
        if folder == "debug":
            latest = latest_debug_dir(run_dir)
            if latest is not None:
                run_dir = latest
                rel = f"data/runs/{profile.data_subdir}/debug/{latest.name}"
            else:
                rel = f"data/runs/{profile.data_subdir}/debug"
        else:
            rel = f"data/runs/{profile.data_subdir}/{folder}"
        n_scenes = _summary_n_scenes(run_dir)
        n_rows = _jsonl_rows(run_dir / "real_manifest.jsonl")
        rows.append((hydra_sign_override(profile), n_scenes, n_rows, rel))
    name_w = max(4, max(len(r[0]) for r in rows))
    print("\n======== manifest summary ========")
    print(f"folder: {folder}   signs: {len(rows)}")
    print(f"{'sign':<{name_w}}  {'scenes':>6}  {'rows':>6}  path")
    for name, n_scenes, n_rows, rel in rows:
        print(f"{name:<{name_w}}  {n_scenes:6d}  {n_rows:6d}  {rel}")
    tot_scenes = sum(r[1] for r in rows)
    tot_rows = sum(r[2] for r in rows)
    print(f"{'total':<{name_w}}  {tot_scenes:6d}  {tot_rows:6d}")
    print("==================================")


def cmd_manifest(argv: List[str]) -> int:
    from traffic_bench.eval.manifest import run as generate_manifest

    raw_sign, rest = _take_override(argv, "sign")
    if raw_sign is None:
        return _run_module_main(generate_manifest, argv)
    profiles = profiles_from_sign_value(raw_sign)
    if profiles is None:
        return _run_module_main(generate_manifest, [f"sign={raw_sign}", *rest])
    folder = _manifest_folder(rest)
    for idx, profile in enumerate(profiles, start=1):
        override = hydra_sign_override(profile)
        print(f"\n======== manifest [{idx}/{len(profiles)}] sign={override} ========")
        code = _run_module_main(generate_manifest, [f"sign={override}", *rest])
        if code != 0:
            return code
    print_manifest_sign_summary(profiles, folder=folder)
    return 0


def _reject_policy_as_list(argv: Sequence[str]) -> int:
    raw, _ = _take_override(list(argv), "policy")
    if raw is None:
        return 0
    text = raw.strip()
    if text.startswith("[") or "," in text:
        hint = text if text.startswith("[") else f"[{text}]"
        print(
            f"ERROR: policy= takes one name (got {text!r}). "
            f"For several policies use policies={hint}",
            file=sys.stderr,
        )
        return 2
    return 0


def cmd_run(argv: List[str]) -> int:
    if _reject_policy_as_list(argv):
        return 2
    from traffic_bench.eval.run import main as run_main

    raw_sign, rest = _take_override(argv, "sign")
    raw_manifest, rest = _take_override(rest, "manifest")
    if raw_manifest:
        return _run_module_main(run_main, [f"manifest={raw_manifest}", *rest])
    if raw_sign is None:
        print(
            "ERROR: run needs manifest=… or sign=… "
            "(example: sign=yield → data/runs/yield/test/)",
            file=sys.stderr,
        )
        return 2
    profiles = profiles_from_sign_value(raw_sign)
    if profiles is None:
        profiles = [resolve_sign_token(raw_sign)]
    chosen: list[tuple[SignProfile, Path]] = []
    missing: list[SignProfile] = []
    for profile in profiles:
        found = default_run_manifest_dir(profile)
        if found is None:
            missing.append(profile)
        else:
            chosen.append((profile, found))
    if missing:
        first = missing[0]
        print(
            f"ERROR: no test/ or debug/ manifest for {hydra_sign_override(first)} "
            f"(tried data/runs/{first.data_subdir}/test and "
            f"data/runs/{first.data_subdir}/debug).\n"
            f"Build one with: python -m traffic_bench.eval manifest "
            f"sign={hydra_sign_override(first)}",
            file=sys.stderr,
        )
        return 2
    if len(chosen) > 1:
        print(f"\n======== run {len(chosen)} sign(s) (test/ if present, else debug/latest) ========")
    for idx, (profile, man_dir) in enumerate(chosen, start=1):
        rel = Path("data") / "runs" / profile.data_subdir / man_dir.name
        if len(chosen) > 1:
            print(
                f"\n======== run [{idx}/{len(chosen)}] "
                f"sign={hydra_sign_override(profile)} manifest={rel} ========"
            )
        code = _run_module_main(run_main, [f"manifest={rel}", *rest])
        if code != 0:
            return code
    if len(chosen) > 1:
        from traffic_bench.eval.metrics.combine import (
            combine_from_eval_outs,
            eval_out_from_manifest_dir,
        )

        outs: list[Path] = []
        for _, man_dir in chosen:
            try:
                out = eval_out_from_manifest_dir(man_dir)
            except FileNotFoundError:
                continue
            if (out / "metrics_per_episode.csv").is_file():
                outs.append(out)
        if outs:
            print(f"\n======== combined report ({len(outs)} signs) ========")
            print(combine_from_eval_outs(outs))
    return 0


def cmd_metrics(argv: List[str]) -> int:
    commands = ("csv", "aggregate", "report", "combine")
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "usage: python -m traffic_bench.eval metrics "
            f"{{{','.join(commands)}}} ...\n\n"
            "  csv         episodes / replays → metrics_per_episode.csv\n"
            "  aggregate   CSV → aggregations + reports/cumulative.json\n"
            "  report      cumulative JSON → markdown table\n"
            "  combine     per-sign CSVs → one overall report (sign=all)\n"
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
    elif command == "combine":
        from traffic_bench.eval.metrics import combine as mod
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
            "  run         one policy, or policies=all / policies=[…] then metrics\n"
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
