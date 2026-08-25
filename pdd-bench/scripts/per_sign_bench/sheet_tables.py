#!/usr/bin/env python3
"""Emit the benchmark sheet layout for two map-level metrics."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

TRUE = {"1", "true", "yes", "t", "y"}
tb = lambda v: str(v).strip().lower() in TRUE

BASE = ["idm_default", "idm_s1", "idm_s2", "idm_s3", "idm_s4",
        "ppo_lidar_default", "carl_default", "plant2_default"]
RULE = ["comprehensive_rule_expert_default", "comprehensive_rule_expert_s1",
        "comprehensive_rule_expert_s2", "comprehensive_rule_expert_s3",
        "comprehensive_rule_expert_s4", "rule_compliant_default",
        "carl_rule_default", "plant2_rule_default"]
COLS = BASE + RULE + ["oracle_rule"]
LABELS = (["IDM(def)", "IDM s1", "IDM s2", "IDM s3", "IDM s4", "PPO", "CaRL", "PlanT-2"] * 2
          + ["Oracle"])

GROUPS = [
    ("2.1", ["2.1"]), ("2.5", ["2.5"]), ("4.3", ["4.3"]), ("5.19", ["5.19"]),
    ("3.24", ["3.24"]), ("4.6", ["4.6"]), ("5.21", ["5.21"]), ("5.31", ["5.31"]),
    ("4.2.1-4.2.3", ["4.2.1", "4.2.2", "4.2.3"]),
    ("5.15.1-5.15.2", ["5.15.1", "5.15.2"]),
]


def compliance(rows, need_dest):
    sel = [r for r in rows if tb(r.get("target_in_zone"))]
    if need_dest:
        sel = [r for r in sel if tb(r.get("arrived_dest"))]
    if not sel:
        return None
    return sum(1 for r in sel if tb(r.get("sign_compliant_high"))) / len(sel)


def main(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            rows.extend(csv.DictReader(fh))

    # code -> baseline -> map -> episodes
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        data[str(r.get("pdd_code"))][str(r.get("baseline"))][str(r.get("scene_id"))].append(r)

    for need_dest, title in ((False, "TABLE 1 - Sign Compliance (in zone), mean over maps"),
                             (True, "TABLE 2 - Sign Compliance (arrived AND in zone), mean over maps")):
        print("\n" + title)
        print("%-14s %6s %6s  %s" % ("sign", "maps", "eps",
                                     " ".join(f"{l:>10s}" for l in LABELS)))
        for name, codes in GROUPS:
            present = [c for c in codes if c in data]
            if not present:
                print("%-14s %6s %6s  %s" % (name, "-", "-",
                                             " ".join(f"{chr(45):>10s}" for _ in LABELS)))
                continue
            maps_total = sum(len(data[c][BASE[0]]) for c in present if BASE[0] in data[c])
            eps_total = sum(len(e) for c in present for e in data[c].get(BASE[0], {}).values())
            cells = []
            for b in COLS:
                per_map, flat = [], []
                for c in present:
                    for eps in data[c].get(b, {}).values():
                        flat.extend(eps)
                        v = compliance(eps, need_dest)
                        if v is not None:
                            per_map.append(v)
                # old: every episode weighs the same; new: every map weighs the same
                o = compliance(flat, need_dest)
                n = sum(per_map) / len(per_map) if per_map else None
                fo = "-" if o is None else f"{o:.2f}"
                fn = "-" if n is None else f"{n:.2f}"
                cells.append(f"{fo}/{fn}".rjust(10))
            print("%-14s %6d %6d  %s" % (name, maps_total, eps_total, " ".join(cells)))


if __name__ == "__main__":
    main(sys.argv[1:])
