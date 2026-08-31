#!/usr/bin/env python3
"""Which expert did the oracle actually pick, and did the ego-style samples earn
their keep?

The IDM expert is recorded five times per scene -- `default` plus `s1..s4`,
each a different sampled driving style -- and selection keeps one. If the
samples were redundant, `default` would win nearly everything and recording
them would be four fifths wasted. This prints the split, per sign and overall,
next to the score that decided it.

    python tools/oracle_stats.py <collection_root> [--experts experts_idm]

<collection_root> holds one directory per sign family, each with an
<experts>/experts_scene_uid_top1.jsonl written by
`traffic_bench.oracle.select.coverage`.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
from pathlib import Path

VARIANTS = ["default", "s1", "s2", "s3", "s4"]


def load(root: Path, experts_dir: str):
    """Yield (family, record) for every picked expert under `root`."""
    for fam in sorted(os.listdir(root)):
        f = root / fam / experts_dir / "experts_scene_uid_top1.jsonl"
        if not f.is_file():
            continue
        for line in f.open():
            line = line.strip()
            if line:
                yield fam, json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--experts", default="experts_idm",
                    help="per-family selection directory (default: experts_idm)")
    ap.add_argument("--pool", default=None, metavar="DIR",
                    help="also summarise every candidate run from "
                         "<family>/DIR/all_runs_dedup.jsonl (e.g. --pool experts), "
                         "counting each IDM ego-style sample as its own expert")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 2

    per_fam: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    f1_by_fam: dict[str, list] = collections.defaultdict(list)
    f1_by_var: dict[str, list] = collections.defaultdict(list)
    sign_of: dict[str, str] = {}
    policies = collections.Counter()
    n = 0

    for fam, r in load(root, args.experts):
        n += 1
        var = str(r.get("winner_variant") or "?")
        per_fam[fam][var] += 1
        sign_of[fam] = str(r.get("sign") or "?")
        policies[str(r.get("winner_policy") or "?")] += 1
        f1 = float(r.get("f1_score") or 0.0)
        f1_by_fam[fam].append(f1)
        f1_by_var[var].append(f1)

    if not n:
        print(f"no picks found under {root}/*/{args.experts}/")
        return 2

    head = "%-24s %-7s %6s " % ("family", "sign", "n") + \
        " ".join("%7s" % v for v in VARIANTS) + "  %7s %7s" % ("sample%", "avgF1")
    print(head)
    print("-" * len(head))
    for fam in sorted(per_fam, key=lambda k: -sum(per_fam[k].values())):
        c = per_fam[fam]
        tot = sum(c.values())
        sample = tot - c["default"]
        print("%-24s %-7s %6d " % (fam, sign_of[fam], tot)
              + " ".join("%7d" % c[v] for v in VARIANTS)
              + "  %6.0f%% %7.3f" % (100 * sample / tot,
                                     statistics.fmean(f1_by_fam[fam])))

    total = collections.Counter()
    for c in per_fam.values():
        total.update(c)
    tot = sum(total.values())
    print("-" * len(head))
    print("%-24s %-7s %6d " % ("TOTAL", "", tot)
          + " ".join("%7d" % total[v] for v in VARIANTS)
          + "  %6.0f%%" % (100 * (tot - total["default"]) / tot))
    print()
    print("Average F1 of the winner, by variant -- a sample winning more often "
          "than\ndefault is expected; winning with a better score is what makes "
          "it worth recording:")
    for v in VARIANTS:
        if f1_by_var[v]:
            print("  %-8s n=%-6d avgF1=%.3f" % (v, len(f1_by_var[v]),
                                                statistics.fmean(f1_by_var[v])))
    if len(policies) > 1 or "idm_rule" not in policies:
        print()
        print("winner_policy mix:", dict(policies.most_common()))

    if args.pool:
        _print_pool(root, args.pool)
    return 0


def _print_pool(root: Path, pool_dir: str) -> None:
    """Summarise the candidate pool the pick chose from.

    Every recorded run, not just the winner, so the four sampled ego styles
    are visible as their own experts rather than folded into `idm_rule`.
    """
    agg: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "arrived": 0, "clean": 0, "ok": 0, "steps": []})
    for fam in sorted(os.listdir(root)):
        f = root / fam / pool_dir / "all_runs_dedup.jsonl"
        if not f.is_file():
            continue
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = "%s/%s" % (r.get("policy") or "?", r.get("variant") or "?")
            a = agg[key]
            a["n"] += 1
            a["arrived"] += bool(r.get("arrived_dest"))
            a["clean"] += int(r.get("total_violations") or 0) == 0
            a["ok"] += bool(r.get("success"))
            if r.get("final_step"):
                a["steps"].append(int(r["final_step"]))
    if not agg:
        print()
        print("no candidate pool under %s/*/%s/" % (root, pool_dir))
        return
    print()
    print("Candidate pool (every recorded run, before the pick):")
    hdr = "  %-22s %7s %8s %8s %8s %9s" % (
        "expert", "n", "arrived", "no_viol", "success", "med_steps")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key in sorted(agg, key=lambda k: -agg[k]["n"]):
        a = agg[key]
        n = a["n"]
        print("  %-22s %7d %8.2f %8.2f %8.2f %9s" % (
            key, n, a["arrived"] / n, a["clean"] / n, a["ok"] / n,
            int(statistics.median(a["steps"])) if a["steps"] else "-"))


if __name__ == "__main__":
    raise SystemExit(main())
