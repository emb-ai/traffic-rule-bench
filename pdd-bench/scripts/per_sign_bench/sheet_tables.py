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

# Row order follows the benchmark sheet, so a paste lands on the right lines.
# Groups with no run yet stay in place and come out blank.
GROUPS = [
    ("2.1", ["2.1"]),
    ("2.3.1-2.3.3", ["2.3.1", "2.3.2", "2.3.3"]),
    ("2.4", ["2.4"]),
    ("2.5", ["2.5"]),
    ("4.3", ["4.3"]),
    ("5.19", ["5.19"]),
    ("3.24", ["3.24"]),
    ("4.6", ["4.6"]),
    ("5.21", ["5.21"]),
    ("5.31", ["5.31"]),
    ("3.20", ["3.20"]),
    ("4.2.1-4.2.3", ["4.2.1", "4.2.2", "4.2.3"]),
    ("5.15.1-5.15.2", ["5.15.1", "5.15.2"]),
    ("3.1-3.2", ["3.1", "3.2"]),
    ("3.18.1-3.18.2", ["3.18.1", "3.18.2"]),
    ("4.1.1-4.1.6", ["4.1.1", "4.1.2", "4.1.3", "4.1.4", "4.1.5", "4.1.6"]),
    ("5.7.1-5.7.2", ["5.7.1", "5.7.2"]),
]


def compliance(rows, need_dest):
    """Share of runs that obeyed the sign.

    Without need_dest the denominator is the in-zone episodes -- the sheet's
    established SC (in zone). With need_dest the denominator is EVERY episode,
    and an episode counts only when it reached the destination, entered the
    zone and stayed compliant: a run that crashes before the sign then scores
    zero instead of leaving the denominator and inflating the rate.
    """
    if not rows:
        return None
    if need_dest:
        good = sum(1 for r in rows
                   if tb(r.get("target_in_zone"))
                   and tb(r.get("arrived_dest"))
                   and tb(r.get("sign_compliant_high")))
        return good / len(rows)
    sel = [r for r in rows if tb(r.get("target_in_zone"))]
    if not sel:
        return None
    return sum(1 for r in sel if tb(r.get("sign_compliant_high"))) / len(sel)


def main(paths, tsv=False, md=False):
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            rows.extend(csv.DictReader(fh))

    # code -> baseline -> map -> episodes
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        data[str(r.get("pdd_code"))][str(r.get("baseline"))][str(r.get("scene_id"))].append(r)

    for need_dest, title in ((False, "Sign Compliance (in zone)"),
                             (True, "Sign Compliance (arrived AND in zone), over all episodes")):
        if md:
            print("\n**" + title + "** — old/new in each cell\n")
        else:
            print("\n" + title + ("  [old/new per cell]" if not tsv else ""))
        if tsv:
            # Two header rows so the group split survives a paste into a sheet.
            groups = ([""] * 3 + ["Base planners"] * 8
                      + ["Rule-compliant experts"] * 8 + ["Oracle"])
            print("\t".join(groups))
            print("\t".join(["sign", "maps", "scenes"] + LABELS))
        elif md:
            head = ["sign", "maps", "scenes"] + [
                l + (" (base)" if i < 8 else " (rule)" if i < 16 else "")
                for i, l in enumerate(LABELS)]
            print("| " + " | ".join(head) + " |")
            print("|" + "|".join(["---"] * len(head)) + "|")
        else:
            print("%-14s %6s %6s  %s" % ("sign", "maps", "eps",
                                         " ".join(f"{l:>10s}" for l in LABELS)))
        for name, codes in GROUPS:
            present = [c for c in codes if c in data]
            if not present:
                if tsv:
                    print("\t".join([name, "", ""] + [""] * len(LABELS)))
                elif md:
                    print("| " + " | ".join([name, "—", "—"]
                                            + ["—"] * len(LABELS)) + " |")
                else:
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
                fo = "" if o is None else f"{o:.2f}"
                fn = "" if n is None else f"{n:.2f}"
                if tsv:
                    cells.append(fn)
                elif md:
                    cells.append(f"{fo}/{fn}" if fo or fn else "—")
                else:
                    cells.append(f"{fo or chr(45)}/{fn or chr(45)}".rjust(10))
            if tsv:
                print("\t".join([name, str(maps_total), str(eps_total)] + cells))
            elif md:
                print("| " + " | ".join([name, str(maps_total), str(eps_total)]
                                        + cells) + " |")
            else:
                print("%-14s %6d %6d  %s" % (name, maps_total, eps_total, " ".join(cells)))


if __name__ == "__main__":
    argv = sys.argv[1:]
    as_tsv = "--tsv" in argv
    as_md = "--md" in argv
    main([a for a in argv if a not in ("--tsv", "--md")], tsv=as_tsv, md=as_md)
