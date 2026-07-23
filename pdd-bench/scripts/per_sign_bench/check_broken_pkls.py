#!/usr/bin/env python3
"""Verify replay.pkl integrity after a hard kill.

Reads pkl paths from stdin, tries to unpickle each, and reports:
  BROKEN(EOFError) / BROKEN(UnpicklingError)  -> genuinely truncated file
                                                 (delete pkl+json, resume rewrites)
  anything else                               -> the file is fine

The script bootstraps the repo sys.path itself, so MetaDrive classes inside
the pkl resolve regardless of the active conda env — a plain `python3 -c`
check would report ModuleNotFoundError for every healthy file.

Usage:
  find <OUT_BASE> -name replay.pkl -newermt '-15 minutes' | \\
      python3 check_broken_pkls.py
"""
import pickle
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PDD_BENCH_DIR = SCRIPT_PATH.parent.parent.parent
METADRIVE_DIR = PDD_BENCH_DIR.parent / "metadrive"
for _p in (PDD_BENCH_DIR, METADRIVE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TRUNCATION_ERRORS = ("EOFError", "UnpicklingError")


def main() -> None:
    ok = truncated = env_or_other = 0
    for p in sys.stdin.read().split():
        try:
            pickle.load(open(p, "rb"))
            ok += 1
        except Exception as exc:
            kind = type(exc).__name__
            if kind in TRUNCATION_ERRORS:
                truncated += 1
                print(f"TRUNCATED({kind}) {p}")
            else:
                env_or_other += 1
                print(f"UNREADABLE({kind}) {p}  <- likely NOT truncation")
    print(f"\nOK: {ok}  truncated: {truncated}  other-errors: {env_or_other}")
    if truncated:
        print("delete each truncated pkl together with its sibling replay.json;"
              " RESUME=1 re-records them")


if __name__ == "__main__":
    main()
