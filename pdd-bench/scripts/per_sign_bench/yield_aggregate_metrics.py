#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from consolidate_replays import consolidate as _consolidate_replays  # type: ignore
from aggregate_from_consolidated_replays import (  # type: ignore
    _aggregate_baseline as _aggregate_replays,
)


_COMPLIANCE_KEYS = ("__speed_compliance__", "__detour_compliance__")


def _discover_runs(runs_dir: Path, include: set[str] | None) -> list[Path]:
    out: list[Path] = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("chunk_"):
            continue
        if include is not None and d.name not in include:
            continue
        has_replays = (d / "replays").is_dir()
        has_episodes = bool(list(d.glob("episodes_*.jsonl")))
        if has_replays or has_episodes:
            out.append(d)
    return out


def _read_summary_compliance(run_dir: Path) -> dict | None:
    for sj in sorted(run_dir.glob("summary_*.json")):
        try:
            data = json.loads(sj.read_text(encoding="utf-8"))
        except Exception:
            continue
        for k in _COMPLIANCE_KEYS:
            if isinstance(data.get(k), dict):
                return {"source_file": sj.name, "kind": k.strip("_"), **data[k]}
    return None


def _consolidate_one(runs_dir: Path, run_name: str) -> tuple[Path, int]:
    out_path = runs_dir / f"{run_name}_replays.jsonl"
    n_written, n_failed, n_dupes = _consolidate_replays(
        runs_dir, run_name, out_path, dedup=True,
    )
    if n_failed or n_dupes:
        print(f"  [warn] {run_name}: {n_failed} failed, {n_dupes} dupes")
    return out_path, n_written


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _fmt_pct(x) -> str:
    if x is None:
        return "  n/a"
    return f"{100.0*float(x):5.2f}%"


def _fmt_num(x, fmt: str = "7.2f") -> str:
    if x is None:
        return "—".rjust(int("".join(filter(str.isdigit, fmt))))
    return format(float(x), fmt)


def _print_main_table(per_run: dict[str, dict]) -> list[str]:
    cols = [
        ("n",        "n_episodes_ok",         "5d"),
        ("succ%",    "success_rate",          "_pct"),
        ("arr%",     "arrival_rate",          "_pct"),
        ("crash%",   "crash_rate",            "_pct"),
        ("oor%",     "out_of_road_rate",      "_pct"),
        ("DS",       "avg_driving_score",     "6.2f"),
        ("eff",      "avg_driving_efficiency", "5.2f"),
        ("smth",     "avg_smoothness",        "5.3f"),
        ("ttc",      "avg_min_ttc_sec",       "5.2f"),
        ("vstep",    "total_violation_steps",  "7d"),
        ("vevt",     "total_violation_events", "5d"),
        ("zone",     "total_in_zone_steps",    "7d"),
    ]
    width_run = max([len(r) for r in per_run] + [10])
    header = f"  {'run':<{width_run}}  " + "  ".join(
        f"{c[0]:>{int(''.join(filter(str.isdigit, c[2]))) if c[2] != '_pct' else 6}}"
        for c in cols
    )
    out_lines = [header, "  " + "-" * (len(header) - 2)]
    for name in sorted(per_run):
        m = per_run[name]
        cells = []
        for label, key, fmt in cols:
            v = m.get(key)
            if fmt == "_pct":
                cells.append(_fmt_pct(v))
            elif "d" in fmt:
                if v is None:
                    cells.append("—".rjust(int("".join(filter(str.isdigit, fmt)))))
                else:
                    cells.append(format(int(v), fmt))
            else:
                cells.append(_fmt_num(v, fmt))
        out_lines.append(f"  {name:<{width_run}}  " + "  ".join(cells))
    print("\n".join(out_lines))
    print()
    return out_lines


def _print_compliance_table(name: str, comp: dict, kind: str) -> list[str]:
    out_lines: list[str] = []
    head = f"=== {name}  ({kind}) ==="
    out_lines.append(head)

    def _row(label: str, b: dict) -> str:
        n = b.get("n_episodes", 0)
        if n == 0:
            return f"  {label:<22}  no episodes"
        c = b.get("COMPLIANT", 0); v = b.get("VIOLATED", 0); z = b.get("NO_ZONE", 0)
        ep = b.get("episode_compliance_rate")
        sp = b.get("step_compliance_rate")
        return (f"  {label:<22}  n={n:>3}  compliant={c:>3}  violated={v:>3}  "
                f"no_zone={z:>3}  ep_rate={_fmt_pct(ep)}  step_rate={_fmt_pct(sp)}")

    out_lines.append(_row("OVERALL", comp.get("overall", {})))
    by_sign = comp.get("by_sign_type", {})
    if by_sign:
        out_lines.append("  by sign_type:")
        for k in sorted(by_sign):
            out_lines.append(_row(f"    {k}", by_sign[k]))
    by_block = comp.get("by_block_id", {})
    if by_block:
        out_lines.append("  by block_id:")
        for k in sorted(by_block):
            out_lines.append(_row(f"    block={k}", by_block[k]))
    out_lines.append("")
    print("\n".join(out_lines))
    return out_lines


def _markdown_table(per_run: dict[str, dict]) -> str:
    cols = [
        ("run",        None,                     "{}"),
        ("n",          "n_episodes_ok",          "{}"),
        ("succ",       "success_rate",           "{:.2%}"),
        ("arr",        "arrival_rate",           "{:.2%}"),
        ("crash",      "crash_rate",             "{:.2%}"),
        ("oor",        "out_of_road_rate",       "{:.2%}"),
        ("DS",         "avg_driving_score",      "{:.2f}"),
        ("eff",        "avg_driving_efficiency", "{:.2f}"),
        ("smth",       "avg_smoothness",         "{:.3f}"),
        ("ttc",        "avg_min_ttc_sec",        "{:.2f}"),
        ("v_step",     "total_violation_steps",  "{}"),
        ("v_event",    "total_violation_events", "{}"),
        ("zone_step",  "total_in_zone_steps",    "{}"),
    ]
    lines = []
    lines.append("| " + " | ".join(c[0] for c in cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for name in sorted(per_run):
        m = per_run[name]
        cells = [name]
        for label, key, fmt in cols[1:]:
            v = m.get(key)
            if v is None:
                cells.append("—")
            else:
                try:
                    cells.append(fmt.format(v))
                except Exception:
                    cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _markdown_compliance(name: str, comp: dict, kind: str) -> str:
    rows = []
    rows.append(f"### {name} — compliance ({kind})")
    rows.append("")
    rows.append("| scope | n | compliant | violated | no_zone | ep_rate | step_rate |")
    rows.append("|---|---|---|---|---|---|---|")

    def _md_row(scope: str, b: dict) -> str:
        n = b.get("n_episodes", 0)
        if n == 0:
            return f"| {scope} | 0 | — | — | — | — | — |"
        ep = b.get("episode_compliance_rate")
        sp = b.get("step_compliance_rate")
        ep_s = f"{ep:.2%}" if ep is not None else "n/a"
        sp_s = f"{sp:.2%}" if sp is not None else "n/a"
        return (f"| {scope} | {n} | {b.get('COMPLIANT',0)} | {b.get('VIOLATED',0)} "
                f"| {b.get('NO_ZONE',0)} | {ep_s} | {sp_s} |")

    rows.append(_md_row("**OVERALL**", comp.get("overall", {})))
    for k, b in sorted(comp.get("by_sign_type", {}).items()):
        rows.append(_md_row(f"sign={k}", b))
    for k, b in sorted(comp.get("by_block_id", {}).items()):
        rows.append(_md_row(f"block={k}", b))
    rows.append("")
    return "\n".join(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs-dir", required=True,
                   help="Path to <bench>/benchmark_output/<preset>/policy_eval")
    p.add_argument("--include", default=None,
                   help="Comma-separated list of run_name'ов (default: все)")
    p.add_argument("--out-md", default=None, help=" out markdown report ")
    p.add_argument("--out-json", default=None,
                   help="(default: <runs-dir>/cumulative_speed_detour.json)")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.is_dir():
        print(f"--runs-dir not a directory: {runs_dir}", file=sys.stderr)
        return 2

    include_set: set[str] | None = None
    if args.include:
        include_set = {x.strip() for x in args.include.split(",") if x.strip()}

    runs = _discover_runs(runs_dir, include_set)
    if not runs:
        print(f"no runs found under {runs_dir}", file=sys.stderr)
        return 2

    print(f"[scan] {len(runs)} run(s) under {runs_dir}")
    for r in runs:
        print(f"   - {r.name}")

    per_run_metrics: dict[str, dict] = {}
    per_run_compliance: dict[str, dict] = {}

    for run_dir in runs:
        name = run_dir.name
        print(f"\n[consolidate] {name}")
        consolidated, n_written = _consolidate_one(runs_dir, name)
        if n_written == 0:
            print(f"  [skip] {name}: no replay.json sidecars (нужен --emit-replay-sidecar)")
            continue
        rows = _load_jsonl(consolidated)
        per_run_metrics[name] = _aggregate_replays(rows)
        comp = _read_summary_compliance(run_dir)
        if comp is not None:
            per_run_compliance[name] = comp
        print(f"  [ok] {name}: {n_written} scenes → {consolidated.name}; "
              f"compliance={'present' if comp else 'absent'}")

    if not per_run_metrics:
        print("no usable runs (все без replays sidecars)")
        return 1

    # ----- stdout -----
    print()
    print("=" * 80)
    print("AGGREGATE METRICS (from replay.json sidecars)")
    print("=" * 80)
    _print_main_table(per_run_metrics)

    if per_run_compliance:
        print()
        print("=" * 80)
        print("COMPLIANCE BREAKDOWN  (per run, from summary_<policy>.json)")
        print("=" * 80)
        for name in sorted(per_run_compliance):
            comp = per_run_compliance[name]
            _print_compliance_table(name, comp, comp.get("kind", "compliance"))

    # ----- JSON -----
    out_json_path = (Path(args.out_json).resolve()
                     if args.out_json
                     else runs_dir / "cumulative_speed_detour.json")
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(
        json.dumps({"runs_dir": str(runs_dir),
                    "per_run_metrics": per_run_metrics,
                    "per_run_compliance": per_run_compliance},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[write] cumulative json → {out_json_path}")

    # ----- markdown -----
    if args.out_md:
        out_md = Path(args.out_md).resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        md = []
        md.append(f"# Speed/Detour benchmark report — `{runs_dir}`\n")
        md.append("## Overall metrics (per run)\n")
        md.append(_markdown_table(per_run_metrics))
        md.append("")
        if per_run_compliance:
            md.append("\n## Compliance breakdown\n")
            for name in sorted(per_run_compliance):
                comp = per_run_compliance[name]
                md.append(_markdown_compliance(name, comp, comp.get("kind", "compliance")))
        out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"[write] markdown → {out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
