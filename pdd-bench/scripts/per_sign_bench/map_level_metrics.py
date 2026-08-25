#!/usr/bin/env python3
"""Map-level metrics: collapse a map's variations first, then average over maps.

Episode-level averages let a map with many variations outweigh a map with few,
so a policy can win by doing well where the catalog happens to be dense. Here
every map contributes exactly one number, and the reported figure is the mean
over maps -- of the requested split (train / test) when a split source is given.

Adds `comp_dest_inzone`: sign compliance among episodes that BOTH reached the
destination and entered the zone of effect. Unlike compliance over all
episodes, a run that crashes before the sign cannot score on it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

TRUE = {"1", "true", "yes", "t", "y"}


def tb(v) -> bool:
    return str(v).strip().lower() in TRUE


def fnum(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_split(paths: list[str]) -> dict[str, str]:
    """scene_id -> 'train' / 'test', read from catalog_fv_*.jsonl files."""
    split: dict[str, str] = {}
    for p in paths:
        name = Path(p).name.lower()
        label = "train" if "train" in name else ("test" if "test" in name else None)
        if label is None:
            print(f"[split] cannot tell train from test in {p}", file=sys.stderr)
            continue
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sid = row.get("scene_id")
                if sid:
                    split[str(sid)] = label
    return split


def map_key(row: dict) -> str:
    """A map is one net; scene_id names it, variations differ by var_idx/seed."""
    return str(row.get("scene_id") or row.get("scene_uid") or "?")


def collapse(rows: list[dict]) -> dict:
    """One map's episodes -> one record of rates, or None where undefined."""
    n = len(rows)
    in_zone = [r for r in rows if tb(r.get("target_in_zone"))]
    arrived = [r for r in rows if tb(r.get("arrived_dest"))]
    both = [r for r in in_zone if tb(r.get("arrived_dest"))]
    comp = lambda rs: (sum(1 for r in rs if tb(r.get("sign_compliant_high"))) / len(rs)
                       if rs else None)
    lane = [fnum(r.get("mean_abs_lane_offset")) for r in rows]
    lane = [v for v in lane if v is not None]
    return {
        "n_ep": n,
        "n_in_zone": len(in_zone),
        "n_dest_in_zone": len(both),
        "dest_rate": len(arrived) / n if n else None,
        "crash_rate": sum(1 for r in rows if tb(r.get("crashed"))) / n if n else None,
        "comp_sr": comp(rows),
        "comp_x": comp(in_zone),
        "comp_dest_inzone": comp(both),
        "dest_x_comp_sr": (sum(1 for r in rows
                               if tb(r.get("arrived_dest")) and tb(r.get("sign_compliant_high")))
                           / n) if n else None,
        "lane_off": (sum(lane) / len(lane)) if lane else None,
    }


def mean_over_maps(records: list[dict], field: str):
    vals = [r[field] for r in records if r.get(field) is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, nargs="+",
                    help="metrics_per_episode*.csv (several allowed)")
    ap.add_argument("--split", nargs="*", default=[],
                    help="catalog_fv_train80.jsonl / catalog_fv_test20.jsonl")
    ap.add_argument("--only-split", choices=["train", "test"], default=None)
    ap.add_argument("--codes", nargs="*", default=None, help="keep only these pdd_codes")
    ap.add_argument("--baselines", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="write the table as CSV here too")
    args = ap.parse_args()

    split = load_split(args.split) if args.split else {}

    rows: list[dict] = []
    for p in args.csv:
        with open(p, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    if args.codes:
        keep = set(args.codes)
        rows = [r for r in rows if str(r.get("pdd_code")) in keep]
    if args.baselines:
        keep = set(args.baselines)
        rows = [r for r in rows if str(r.get("baseline")) in keep]
    if args.only_split:
        rows = [r for r in rows if split.get(map_key(r)) == args.only_split]

    if not rows:
        print("no rows after filtering", file=sys.stderr)
        return 1

    # (code, baseline) -> map -> episodes
    grouped: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[(str(r.get("pdd_code")), str(r.get("baseline")))][map_key(r)].append(r)

    FIELDS = ["dest_rate", "comp_x", "comp_dest_inzone", "dest_x_comp_sr",
              "crash_rate", "lane_off"]
    header = ["pdd_code", "baseline", "n_maps", "n_ep"] + FIELDS + ["maps_with_dest_in_zone"]
    table = []
    for (code, baseline), maps in sorted(grouped.items()):
        per_map = [collapse(eps) for eps in maps.values()]
        rec = {"pdd_code": code, "baseline": baseline, "n_maps": len(per_map),
               "n_ep": sum(m["n_ep"] for m in per_map)}
        for f in FIELDS:
            val, _cnt = mean_over_maps(per_map, f)
            rec[f] = val
        _v, cnt = mean_over_maps(per_map, "comp_dest_inzone")
        rec["maps_with_dest_in_zone"] = cnt
        table.append(rec)

    widths = {"pdd_code": 8, "baseline": 34}
    print("  ".join(h.ljust(widths.get(h, 10)) for h in header))
    for rec in table:
        cells = []
        for h in header:
            v = rec[h]
            s = "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
            cells.append(s.ljust(widths.get(h, 10)))
        print("  ".join(cells))

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            w.writerows(table)
        print(f"\n[write] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
