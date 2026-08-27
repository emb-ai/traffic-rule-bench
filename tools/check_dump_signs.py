#!/usr/bin/env python3
"""Is the sign actually in the recorded frames, and does it carry its plate?

Run this on the first few routes of any new dump, before recording the rest.
Route count is not evidence: a dump can report every route present and still
carry no sign at all, and nothing downstream complains -- the sign simply never
reaches the model and no later fix recovers it, because the box was never
written.

    python tools/check_dump_signs.py <plant2_dir>/data [--routes 5]

Reports, per route: the share of frames holding a box whose class is a PDD code,
which codes appear, whether the plate value is present for the codes that carry
one, and whether results.json.gz records why a route failed.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import re
from pathlib import Path

# A class string that looks like a PDD code: digits and dots, e.g. "3.24", "4.2.1".
_CODE_RE = re.compile(r"^\d+(?:\.\d+)*$")
# Codes whose plate carries a number; without it 20 and 40 km/h are one token.
_VALUED = {"3.24", "5.31", "4.6"}


def frames_of(route: Path):
    boxes = route / "boxes"
    if not boxes.is_dir():
        return []
    return sorted(f for f in os.listdir(boxes) if f.endswith(".json.gz"))


def scan(route: Path, sample: int):
    names = frames_of(route)
    if not names:
        return None
    step = max(1, len(names) // sample)
    seen_codes = collections.Counter()
    valued = collections.Counter()
    hit = 0
    total = 0
    for name in names[::step]:
        with gzip.open(route / "boxes" / name, "rt") as fh:
            boxes = json.load(fh)
        total += 1
        codes = [b for b in boxes if _CODE_RE.match(str(b.get("class", "")))]
        if codes:
            hit += 1
        for b in codes:
            code = str(b["class"])
            seen_codes[code] += 1
            if code in _VALUED and b.get("sign_value_kmh") is not None:
                valued[code] += 1
    return {
        "frames": len(names),
        "sampled": total,
        "with_sign": hit,
        "codes": dict(seen_codes),
        "valued": dict(valued),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", help="<plant2_dir>/data")
    ap.add_argument("--routes", type=int, default=5, help="how many routes to scan")
    ap.add_argument("--sample", type=int, default=50, help="frames sampled per route")
    args = ap.parse_args()

    data = Path(args.data_dir)
    if not data.is_dir():
        print(f"not a directory: {data}")
        return 2

    routes = [data / n for n in sorted(os.listdir(data))][: args.routes]
    if not routes:
        print(f"no routes under {data}")
        return 2

    bad = 0
    print("%-46s %8s %10s %-22s %s" % ("route", "frames", "sign share", "codes", "plate value"))
    for r in routes:
        info = scan(r, args.sample)
        if info is None:
            print("%-46s  no frames" % r.name[:46])
            bad += 1
            continue
        share = info["with_sign"] / max(info["sampled"], 1)
        need_value = {c for c in info["codes"] if c in _VALUED}
        got_value = set(info["valued"])
        value_note = ("-" if not need_value
                      else "ok" if need_value <= got_value
                      else "MISSING for " + ",".join(sorted(need_value - got_value)))
        print("%-46s %8d %9.2f  %-22s %s"
              % (r.name[:46], info["frames"], share,
                 ",".join(sorted(info["codes"])) or "NONE", value_note))
        if share == 0.0 or not info["codes"]:
            bad += 1
        elif need_value and not need_value <= got_value:
            bad += 1

        res = r / "results.json.gz"
        if res.is_file():
            with gzip.open(res, "rt") as fh:
                d = json.load(fh)
            if d.get("status") != "Completed" and "failure_reason" not in d:
                print("%-46s   status=%s but no failure_reason recorded"
                      % ("", d.get("status")))

    print()
    if bad:
        print(f"{bad} of {len(routes)} routes look unusable for sign training.")
        print("A share of 0.00 means the sign was never written into boxes; "
              "re-recording is the only fix.")
        return 1
    print(f"all {len(routes)} routes carry the sign. Detour scenes should read "
          f"~0.98, speed scenes ~0.3 or more at a 120 m radius.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
