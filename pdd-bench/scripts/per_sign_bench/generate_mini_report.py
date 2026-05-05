#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_policy_summaries(results_dir: Path):
    pattern = "mini_all_policies_*/summary_*.json"
    files = sorted(results_dir.glob(pattern))
    out = {}
    for fp in files:
        policy = fp.stem.replace("summary_", "")
        with fp.open("r", encoding="utf-8") as f:
            out[policy] = json.load(f)
    return out


def _wavg(rows, key):
    total = sum(int(r.get("total_runs", 0)) for r in rows)
    if total <= 0:
        return 0.0
    return sum(float(r.get(key, 0.0)) * int(r.get("total_runs", 0)) for r in rows) / total


def _fmt(x, nd=3):
    return f"{float(x):.{nd}f}"


def _overall_table(policy_data):
    rows = []
    for policy, summary in sorted(policy_data.items()):
        items = list(summary.values())
        runs = sum(int(r.get("total_runs", 0)) for r in items)
        rows.append({
            "policy": policy,
            "runs": runs,
            "success_rate": _wavg(items, "success_rate"),
            "crash_rate": _wavg(items, "crash_rate"),
            "avg_violations": _wavg(items, "average_violations"),
            "avg_driving_score": _wavg(items, "average_driving_score"),
            "avg_efficiency": _wavg(items, "average_driving_efficiency"),
            "avg_smoothness": _wavg(items, "average_smoothness"),
        })
    return rows


def _by_backend_table(policy_data):
    rows = []
    for policy, summary in sorted(policy_data.items()):
        by_backend = defaultdict(list)
        for rec in summary.values():
            by_backend[str(rec.get("backend", "unknown"))].append(rec)
        for backend, items in sorted(by_backend.items()):
            runs = sum(int(r.get("total_runs", 0)) for r in items)
            rows.append({
                "policy": policy,
                "backend": backend,
                "runs": runs,
                "success_rate": _wavg(items, "success_rate"),
                "crash_rate": _wavg(items, "crash_rate"),
                "avg_violations": _wavg(items, "average_violations"),
                "avg_driving_score": _wavg(items, "average_driving_score"),
            })
    return rows


def _per_sign_tables(policy_data):
    all_signs = set()
    for summary in policy_data.values():
        for rec in summary.values():
            all_signs.add(str(rec.get("sign_type", "unknown")))

    out = {}
    for sign in sorted(all_signs):
        rows = []
        for policy, summary in sorted(policy_data.items()):
            items = [r for r in summary.values() if str(r.get("sign_type", "unknown")) == sign]
            if not items:
                continue
            runs = sum(int(r.get("total_runs", 0)) for r in items)
            rows.append({
                "policy": policy,
                "runs": runs,
                "success_rate": _wavg(items, "success_rate"),
                "crash_rate": _wavg(items, "crash_rate"),
                "avg_violations": _wavg(items, "average_violations"),
                "avg_driving_score": _wavg(items, "average_driving_score"),
                "avg_efficiency": _wavg(items, "average_driving_efficiency"),
            })
        if rows:
            out[sign] = rows
    return out


def _render_markdown(results_dir: Path, policy_data):
    overall = _overall_table(policy_data)
    by_backend = _by_backend_table(policy_data)
    per_sign = _per_sign_tables(policy_data)

    lines = []
    lines.append("# Mini Benchmark Results")
    lines.append("")
    lines.append("Source: `benchmark_output/mini/policy_eval/mini_all_policies_*/summary_*.json`")
    lines.append("")
    lines.append("## Overall (weighted by total_runs)")
    lines.append("")
    lines.append("| Policy | Runs | Success rate | Crash rate | Avg violations | Avg driving score | Avg efficiency | Avg smoothness |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in overall:
        lines.append(
            f"| `{r['policy']}` | {r['runs']} | {_fmt(r['success_rate'])} | {_fmt(r['crash_rate'])} | "
            f"{_fmt(r['avg_violations'])} | {_fmt(r['avg_driving_score'])} | {_fmt(r['avg_efficiency'])} | {_fmt(r['avg_smoothness'])} |"
        )

    lines.append("")
    lines.append("## By Backend (weighted by total_runs)")
    lines.append("")
    lines.append("| Policy | Backend | Runs | Success rate | Crash rate | Avg violations | Avg driving score |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in by_backend:
        lines.append(
            f"| `{r['policy']}` | `{r['backend']}` | {r['runs']} | {_fmt(r['success_rate'])} | "
            f"{_fmt(r['crash_rate'])} | {_fmt(r['avg_violations'])} | {_fmt(r['avg_driving_score'])} |"
        )

    lines.append("")
    lines.append("## Per Sign (aggregated over available backends)")
    lines.append("")
    for sign, rows in per_sign.items():
        lines.append(f"### Sign `{sign}`")
        lines.append("")
        lines.append("| Policy | Runs | Success rate | Crash rate | Avg violations | Avg driving score | Avg efficiency |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                f"| `{r['policy']}` | {r['runs']} | {_fmt(r['success_rate'])} | {_fmt(r['crash_rate'])} | "
                f"{_fmt(r['avg_violations'])} | {_fmt(r['avg_driving_score'])} | {_fmt(r['avg_efficiency'])} |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate markdown report for mini policy_eval summaries")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/home/gbuhtuev/sdc/pdd-bench/scripts/per_sign_bench/benchmark_output/mini/policy_eval"),
        help="Directory containing mini_all_policies_*/summary_*.json",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write markdown report")
    args = parser.parse_args()

    policy_data = _load_policy_summaries(args.results_dir)
    if not policy_data:
        raise SystemExit(f"No summary files found under: {args.results_dir}")

    md = _render_markdown(args.results_dir, policy_data)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
        print(f"Wrote report: {args.output}")
    else:
        print(md)


if __name__ == "__main__":
    main()
