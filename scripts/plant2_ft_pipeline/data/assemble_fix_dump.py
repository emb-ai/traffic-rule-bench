#!/usr/bin/env python3
"""Fill the re-dumped expert tree with the routes we did NOT re-dump.

The finetune must differ from the best_024 baseline in one thing only: the
signs we re-dumped now carry auxiliary traffic. Everything else -- the other
priority signs, the detour routes -- has to stay bit-identical, otherwise a
metric change could just as well come from the altered data mixture.

So: hardlink every route of the old dump into the new tree, except the routes
whose sign we re-dumped (those already sit there, with traffic). Hardlinks cost
no space and no copy time; the fallback is a real copy on cross-device trees.

  python3 assemble_fix_dump.py --old <old_dump> --new <new_dump> --labels "2.5 4.3"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_train_val_split_fv_experts_signs import (  # noqa: E402
    load_priority_uid_sign,
    route_sign,
)


def link_route(src: Path, dst: Path) -> str:
    if dst.exists():
        return "exists"
    r = subprocess.run(["cp", "-al", str(src), str(dst)], capture_output=True, text=True)
    if r.returncode == 0:
        return "linked"
    r = subprocess.run(["cp", "-a", str(src), str(dst)], capture_output=True, text=True)
    if r.returncode != 0:
        return f"failed: {r.stderr.strip()[:120]}"
    return "copied"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="dump the baseline was trained on")
    ap.add_argument("--new", required=True, help="dump built with --aux-agents")
    ap.add_argument("--labels", required=True,
                    help="space-separated sign codes that were re-dumped")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    old_data = Path(args.old) / "data"
    new_data = Path(args.new) / "data"
    if not old_data.is_dir():
        raise SystemExit(f"missing {old_data}")
    if not new_data.is_dir():
        raise SystemExit(f"missing {new_data} — run the dump stage first")

    fresh = {p.name for p in new_data.iterdir() if p.is_dir()}
    redumped = set(args.labels.split())
    uid2sign = load_priority_uid_sign()

    stats: Counter = Counter()
    per_sign: Counter = Counter()
    for p in sorted(old_data.iterdir()):
        if not p.is_dir():
            continue
        sign = route_sign(p.name, uid2sign)
        if sign in redumped:
            stats["skipped_redumped"] += 1
            continue
        if p.name in fresh:
            stats["skipped_present"] += 1
            continue
        if args.dry_run:
            stats["would_link"] += 1
            per_sign[sign or "?"] += 1
            continue
        st = link_route(p, new_data / p.name)
        stats[st.split(":")[0]] += 1
        if st.startswith("failed"):
            print(f"  !! {p.name}: {st}", flush=True)
        else:
            per_sign[sign or "?"] += 1

    print(f"old routes: {sum(stats.values())}  ->  {dict(stats)}")
    print(f"carried over per sign: {dict(sorted(per_sign.items(), key=lambda kv: str(kv[0])))}")
    total = len([p for p in new_data.iterdir() if p.is_dir()])
    print(f"new tree now holds {total} route(s); re-dumped signs: {sorted(redumped)}")


if __name__ == "__main__":
    main()
