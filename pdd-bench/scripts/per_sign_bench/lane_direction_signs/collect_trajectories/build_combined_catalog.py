#!/usr/bin/env python3
"""Build a catalog for lane_direction (5.15.1) trajectory collection.

Single-sign: pick the newest ``benchmark_output/5_15_1/*/real_manifest.jsonl``
(or an explicit ``--manifest``), stamp ``sign_code=5.15.1``, and write under
``lane_direction_signs/benchmark_output/combined/``:

  real_manifest_balanced.jsonl   (same as source; name kept for collector parity)
  real_manifest_unbalanced.jsonl
  balance_report.json

Then split maps 80/20:

  python make_map_split.py \\
      --catalog ../benchmark_output/combined/real_manifest_balanced.jsonl

Usage:
  python build_combined_catalog.py
  python build_combined_catalog.py --manifest ../benchmark_output/5_15_1/<ts>/real_manifest.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIGN_DIR = HERE.parent
DEFAULT_OUT = SIGN_DIR / "benchmark_output" / "combined"
CODE = "5.15.1"
SLUG = "5_15_1"


def _newest_manifest() -> Path | None:
    root = SIGN_DIR / "benchmark_output" / SLUG
    cands = sorted(root.glob("*/real_manifest.jsonl"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("valid") is False:
                continue
            r = dict(r)
            r["sign_code"] = CODE
            r.setdefault("pdd_code", CODE)
            r.setdefault("sign_slug", SLUG)
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Source real_manifest.jsonl (default: newest under "
                         f"benchmark_output/{SLUG}/)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    src = args.manifest or _newest_manifest()
    if src is None or not Path(src).is_file():
        sys.exit(
            f"ERROR: no manifest found. Pass --manifest or generate one under "
            f"benchmark_output/{SLUG}/"
        )
    src = Path(src)

    rows = _load_rows(src)
    if not rows:
        sys.exit(f"ERROR: empty/invalid manifest: {src}")

    by_map: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        net = r.get("net_path")
        if not net:
            continue
        by_map[str(net)].append(r)

    # Stable order for reproducibility.
    ordered: list[dict] = []
    for net in sorted(by_map):
        group = sorted(
            by_map[net],
            key=lambda r: (
                str(r.get("scene_id") or ""),
                int(r.get("seed") or r.get("deterministic_seed") or 0),
                int(r.get("var_idx", 0) or 0),
            ),
        )
        ordered.extend(group)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bal_path = out_dir / "real_manifest_balanced.jsonl"
    unbal_path = out_dir / "real_manifest_unbalanced.jsonl"
    report_path = out_dir / "balance_report.json"

    with open(bal_path, "w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, default=str) + "\n")
    with open(unbal_path, "w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, default=str) + "\n")

    report = {
        "sign": CODE,
        "source": str(src.resolve()),
        "rule": "single-sign 5.15.1 catalog (all maps/rows from source manifest)",
        "maps": len(by_map),
        "rows": len(ordered),
        "outputs": {
            "balanced": str(bal_path.resolve()),
            "unbalanced": str(unbal_path.resolve()),
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"[{CODE}] source  {src}")
    print(f"[{CODE}] maps={len(by_map)}  rows={len(ordered)}")
    print(f"balanced:   {bal_path}")
    print(f"unbalanced: {unbal_path}")
    print(f"report:     {report_path}")
    print()
    print("Next:")
    print(f"  python make_map_split.py --catalog {bal_path}")


if __name__ == "__main__":
    main()
