#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

NO_ENTRY_SIGNS = {"3.1", "3.2", "3.18.1", "3.18.2", "3.19"}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _compute_success(r: dict, horizon: int) -> bool:
    reached_dest = bool(r.get("reached_dest", False))
    crashed = bool(r.get("crashed", False))
    out_of_road = bool(r.get("out_of_road", False))
    steps = int(r.get("steps") or 0)
    sign = str(r.get("sign_type") or "")
    sign_viol = int(r.get("sign_violations", r.get("violations", 0)) or 0)
    if reached_dest and not crashed and not out_of_road:
        return True
    if steps >= horizon and not crashed and not out_of_road:
        return True
    if sign in NO_ENTRY_SIGNS and (not reached_dest) and sign_viol == 0 and not crashed and not out_of_road:
        return True
    return False


def _agg(rows: list[dict], horizon: int) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "runs": 0,
            "success_rate": 0.0,
            "crash_rate": 0.0,
            "avg_violations": 0.0,
            "avg_driving_score": 0.0,
            "avg_efficiency": 0.0,
            "avg_smoothness": 0.0,
            "avg_route_length_m": 0.0,
            "traffic_light_sr": 0.0,
            "crosswalk_sr": 0.0,
        }
    tl_clean = sum(1 for r in rows if int(r.get("traffic_light_violations", 0) or 0) == 0)
    cw_clean = sum(1 for r in rows if int(r.get("crosswalk_violations", 0) or 0) == 0)
    return {
        "runs": n,
        "success_rate": sum(1 for r in rows if _compute_success(r, horizon=horizon)) / n,
        "crash_rate": sum(1 for r in rows if bool(r.get("crashed", False))) / n,
        "avg_violations": sum(float(r.get("violations", 0) or 0) for r in rows) / n,
        "avg_driving_score": sum(float(r.get("driving_score", 0.0) or 0.0) for r in rows) / n,
        "avg_efficiency": sum(float(r.get("driving_efficiency", 0.0) or 0.0) for r in rows) / n,
        "avg_smoothness": sum(float(r.get("smoothness", 0.0) or 0.0) for r in rows) / n,
        "avg_route_length_m": sum(float(r.get("route_length_m", 0.0) or 0.0) for r in rows) / n,
        "traffic_light_sr": tl_clean / n,
        "crosswalk_sr": cw_clean / n,
    }


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate markdown report for chunk/cumulative baseline runs")
    ap.add_argument("--run-root", required=True,
                    help="Root with runs/<chunk_name>/<baseline>/episodes_*.jsonl")
    ap.add_argument("--chunk-name", required=True,
                    help="Chunk folder name, e.g. chunk_0_200")
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--out", default=None,
                    help="Output markdown path (default: <run-root>/reports/report_<chunk>.md)")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    chunk_root = run_root / "runs" / args.chunk_name
    if not chunk_root.exists():
        raise FileNotFoundError(f"Chunk path not found: {chunk_root}")

    by_policy_rows: dict[str, list[dict]] = defaultdict(list)
    by_policy_backend_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_sign_policy_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for policy_dir in sorted([d for d in chunk_root.iterdir() if d.is_dir()]):
        policy = policy_dir.name
        for ep_file in policy_dir.glob("episodes_*.jsonl"):
            rows = [r for r in _read_jsonl(ep_file) if r.get("ok", False)]
            by_policy_rows[policy].extend(rows)
            for r in rows:
                backend = str(r.get("backend") or "")
                sign = str(r.get("sign_type") or "")
                by_policy_backend_rows[(policy, backend)].append(r)
                by_sign_policy_rows[(sign, policy)].append(r)

    overall = {p: _agg(rows, args.horizon) for p, rows in sorted(by_policy_rows.items())}
    by_backend = {(p, b): _agg(rows, args.horizon) for (p, b), rows in sorted(by_policy_backend_rows.items())}

    sign_to_policies: dict[str, dict[str, dict]] = defaultdict(dict)
    for (sign, policy), rows in sorted(by_sign_policy_rows.items()):
        sign_to_policies[sign][policy] = _agg(rows, args.horizon)

    lines: list[str] = []
    lines.append("# Chunk Benchmark Results")
    lines.append("")
    lines.append(f"Source: `{chunk_root}`")
    lines.append("")
    lines.append("## Overall (weighted by total_runs)")
    lines.append("")
    lines.append("| Policy | Runs | Success rate | Crash rate | TL SR | CW SR | Avg violations | Avg driving score | Avg efficiency | Avg smoothness | Avg route len (m) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for policy, m in overall.items():
        lines.append(
            f"| `{policy}` | {m['runs']} | {_fmt(m['success_rate'])} | {_fmt(m['crash_rate'])} | {_fmt(m['traffic_light_sr'])} | {_fmt(m['crosswalk_sr'])} | "
            f"{_fmt(m['avg_violations'])} | {_fmt(m['avg_driving_score'])} | {_fmt(m['avg_efficiency'])} | {_fmt(m['avg_smoothness'])} | {_fmt(m['avg_route_length_m'])} |"
        )

    lines.append("")
    lines.append("## By Backend (weighted by total_runs)")
    lines.append("")
    lines.append("| Policy | Backend | Runs | Success rate | Crash rate | TL SR | CW SR | Avg violations | Avg driving score | Avg route len (m) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (policy, backend), m in by_backend.items():
        lines.append(
            f"| `{policy}` | `{backend}` | {m['runs']} | {_fmt(m['success_rate'])} | {_fmt(m['crash_rate'])} | {_fmt(m['traffic_light_sr'])} | {_fmt(m['crosswalk_sr'])} | "
            f"{_fmt(m['avg_violations'])} | {_fmt(m['avg_driving_score'])} | {_fmt(m['avg_route_length_m'])} |"
        )

    lines.append("")
    lines.append("## Per Sign (aggregated over available backends)")
    lines.append("")
    for sign in sorted(sign_to_policies.keys()):
        lines.append(f"### Sign `{sign}`")
        lines.append("")
        lines.append("| Policy | Runs | Success rate | Crash rate | TL SR | CW SR | Avg violations | Avg driving score | Avg efficiency | Avg route len (m) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for policy, m in sorted(sign_to_policies[sign].items()):
            lines.append(
                f"| `{policy}` | {m['runs']} | {_fmt(m['success_rate'])} | {_fmt(m['crash_rate'])} | {_fmt(m['traffic_light_sr'])} | {_fmt(m['crosswalk_sr'])} | "
                f"{_fmt(m['avg_violations'])} | {_fmt(m['avg_driving_score'])} | {_fmt(m['avg_efficiency'])} | {_fmt(m['avg_route_length_m'])} |"
            )
        lines.append("")

    out_path = Path(args.out) if args.out else (run_root / "reports" / f"report_{args.chunk_name}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
