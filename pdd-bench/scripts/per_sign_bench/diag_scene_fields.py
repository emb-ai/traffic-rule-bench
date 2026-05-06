#!/usr/bin/env python3
"""diag_scene_fields.py — shows which fields each backend uses in its manifests
and which fields can be used to deduplicate scenes.

Usage:
  python3 diag_scene_fields.py --root /path/to/benchmark_root
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict


BACKEND_FILES = {
    "pgmap":   "pgmap_materialized.jsonl",
    "paired":  "paired_materialized.jsonl",
    "citymap": "citymap_materialized.jsonl",
}


def is_sumo_manifest(name: str) -> bool:
    return name == "sumo_manifest.jsonl" or name == "real_manifest.jsonl"


def detect_backend(path: Path) -> str:
    n = path.name
    if "pgmap_materialized" in n:   return "pgmap"
    if "paired_materialized" in n:  return "paired"
    if "citymap_materialized" in n: return "citymap"
    if is_sumo_manifest(n):         return "sumo"
    return "?"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--sample", type=int, default=3,
                   help="How many rows to print as a sample per backend")
    args = p.parse_args()

    root = Path(args.root)
    by_be = defaultdict(lambda: {
        "rows": 0,
        "field_counts": Counter(),
        "samples": [],
        "scene_id_formats": Counter(),
        "scene_id_unique": set(),
        "key_fields_unique": defaultdict(set),
    })

    for f in root.glob("chunks/chunk_*/mini/*/[!_]*.jsonl"):
        be = detect_backend(f)
        if be == "?": continue
        b = by_be[be]
        for line in f.read_text().splitlines():
            if not line.strip(): continue
            try: r = json.loads(line)
            except: continue
            b["rows"] += 1
            for k in r.keys():
                b["field_counts"][k] += 1
            if len(b["samples"]) < args.sample:
                b["samples"].append(r)

            sid = r.get("scene_id")
            if sid:
                b["scene_id_unique"].add(str(sid))
                fmt = "<empty>"
                s = str(sid)
                if "_" in s:
                    fmt = s.split("_", 1)[0] + "_*"
                else:
                    fmt = "<numeric>" if s.isdigit() else "<other>"
                b["scene_id_formats"][fmt] += 1

            for key_name, key_fn in [
                ("scene_id",                lambda r: r.get("scene_id")),
                ("seed",                    lambda r: r.get("seed") or r.get("deterministic_seed")),
                ("(scene_id, seed)",        lambda r: (r.get("scene_id"), r.get("seed") or r.get("deterministic_seed"))),
                ("(scene_id, seed, var)",   lambda r: (r.get("scene_id"), r.get("seed") or r.get("deterministic_seed"), r.get("var_idx"))),
                ("(net_path, sign_type, seed)", lambda r: (r.get("net_path"), r.get("sign_type") or r.get("sign_code"), r.get("seed") or r.get("deterministic_seed"))),
                ("(map_seed, sign_type)",   lambda r: (r.get("map_seed"), r.get("sign_type") or r.get("sign_code"))),
                ("(sign_id, var_idx)",      lambda r: (r.get("sign_id"), r.get("var_idx"))),
                ("(road_id, sign_id, var)", lambda r: (r.get("road_id"), r.get("sign_id"), r.get("var_idx"))),
            ]:
                k = key_fn(r)
                if k is not None and not (isinstance(k, tuple) and all(v is None for v in k)):
                    b["key_fields_unique"][key_name].add(str(k))

    for be in ("pgmap", "paired", "citymap", "sumo"):
        b = by_be.get(be)
        if not b or b["rows"] == 0:
            print(f"\n=== {be}: NO DATA ===")
            continue

        print(f"\n=== {be}: {b['rows']} rows ===")
        print(f"  scene_id format:")
        for fmt, n in b["scene_id_formats"].most_common(5):
            print(f"    {fmt:20s} {n}")

        print(f"  Unique by various keys:")
        for key_name, vals in b["key_fields_unique"].items():
            if vals:
                print(f"    {key_name:35s} {len(vals)}")

        print(f"\n  ALL fields (count = number of occurrences):")
        for k, n in sorted(b["field_counts"].items()):
            pct = 100 * n / b["rows"]
            mark = "*" if pct < 100 else " "
            print(f"   {mark}{k:30s} {n} ({pct:.0f}%)")
        print(f"  (* = field missing in some rows)")

        print(f"\n  Samples (first {args.sample} rows):")
        for s in b["samples"]:
            keys_short = {k: v for k, v in s.items()
                          if k in ("scene_id", "seed", "deterministic_seed", "map_seed",
                                   "var_idx", "sign_type", "sign_code", "sign_id",
                                   "net_path", "road_id", "spawn_lane_num", "block_id",
                                   "block_sequence", "lane_num")}
            print(f"    {keys_short}")


if __name__ == "__main__":
    main()
